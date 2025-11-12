# ---
# Filename: ../ComfyUI_INSTARAW/modules/detection_bypass/utils/non_semantic_unmarker.py
# ---

"""
A direct, self-contained implementation of the UnMarker "non-semantic" attack,
adapted for use within the ComfyUI INSTARAW pipeline.
This module performs an LPIPS-constrained adversarial optimization in the spectral
domain to remove statistical fingerprints from AI-generated images.
Ref: https://github.com/andrekassis/ai-watermark
"""

import torch
import torch.optim as optim
import lpips
import numpy as np
import threading

def _optimization_worker(img_np: np.ndarray, config: dict, device: torch.device) -> np.ndarray:
    """
    Runs the core UnMarker optimization in a separate thread.
    This is CRITICAL to escape the global torch.no_grad() context of ComfyUI,
    as this optimization requires gradient calculation.
    """
    # Re-initialize the image tensor in this new thread, which is not in no_grad() mode.
    img_tensor = torch.from_numpy(img_np).to(device)
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

    # CRITICAL FIX: The delta is the perturbation we are optimizing.
    # Make it SINGLE CHANNEL (grayscale) to prevent RGB channels from being
    # optimized independently, which causes blue color shifts.
    # We'll broadcast this single channel to all RGB channels.
    batch_size, channels, height, width = img_tensor.shape
    # Initialize larger (0.01 not 1e-5) so optimizer has signal
    delta = torch.randn(batch_size, 1, height, width, device=device, dtype=img_tensor.dtype) * 0.01
    delta.requires_grad = True

    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam([delta], lr=config['learning_rate'])

    # CRITICAL FIX: Convert to grayscale for FFT to prevent blue hue color shift
    # RGB to grayscale (ITU-R BT.601)
    if img_tensor.shape[1] == 3:
        img_gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    else:
        img_gray = img_tensor

    with torch.no_grad():
        img_fft = torch.fft.fft2(img_gray)

    print(f"  [UnMarker] ✓ Thread started. Optimizing for {config['iterations']} iterations...")
    print(f"  [UnMarker] ✓ Using achromatic (grayscale) perturbation to prevent color shift")

    for i in range(config['iterations']):
        optimizer.zero_grad()

        # Broadcast single-channel delta to all RGB channels
        # This ensures the perturbation is achromatic (same for R, G, B)
        delta_rgb = delta.expand_as(img_tensor)
        x_nw = torch.clamp(img_tensor + delta_rgb, -1, 1)

        # Spectral Loss (DFL - Deep Feature Loss in Frequency Domain)
        # Convert perturbed image to grayscale for FFT comparison
        if x_nw.shape[1] == 3:
            x_nw_gray = 0.299 * x_nw[:, 0:1] + 0.587 * x_nw[:, 1:2] + 0.114 * x_nw[:, 2:3]
        else:
            x_nw_gray = x_nw

        x_nw_fft = torch.fft.fft2(x_nw_gray)
        fft_diff = x_nw_fft - img_fft
        magnitude = torch.sqrt(fft_diff.real**2 + fft_diff.imag**2 + 1e-8)
        loss_dfl = -magnitude.sum()

        # Perceptual Loss (LPIPS) to ensure visual similarity
        loss_lpips = lpips_model(x_nw, img_tensor).mean()

        # Geometric Loss (L2 Norm of the perturbation, per-pixel normalized)
        loss_l2 = delta.pow(2).mean().sqrt()

        # Combine losses with thresholds as per the UnMarker paper
        lpips_penalty = config['c_lpips'] * torch.relu(loss_lpips - config['t_lpips'])
        l2_penalty = config['c_l2'] * torch.relu(loss_l2 - config['t_l2'])
        total_loss = loss_dfl + lpips_penalty + l2_penalty

        total_loss.backward()

        if delta.grad is not None:
            delta.grad.data.clamp_(-config['grad_clip_value'], config['grad_clip_value'])

        optimizer.step()

    print(f"  [UnMarker] ✓ Optimization complete.")

    # Apply final achromatic perturbation (broadcast to RGB)
    delta_rgb = delta.expand_as(img_tensor)
    final_x_nw = torch.clamp(img_tensor + delta_rgb, -1, 1).squeeze(0).cpu().detach()
    final_x_nw = (final_x_nw + 1) / 2  # Denormalize to [0, 1]
    final_x_nw = final_x_nw.permute(1, 2, 0)  # (C, H, W) to (H, W, C)
    return (final_x_nw.clamp(0, 1) * 255).numpy().astype(np.uint8)


@torch.enable_grad()  # Decorator to ensure gradients are enabled within this function
def attack_non_semantic(img_arr: np.ndarray,
                        iterations: int = 500,
                        learning_rate: float = 3e-4,
                        t_lpips: float = 4e-2,
                        t_l2: float = 3e-5,
                        c_lpips: float = 1e-2,
                        c_l2: float = 0.6,
                        grad_clip_value: float = 0.05
                        ) -> np.ndarray:
    """
    Main entry point for the UnMarker non-semantic attack.
    This function prepares the data and spawns a worker thread to perform the optimization.
    """
    config = {
        'iterations': iterations, 'learning_rate': learning_rate,
        't_lpips': t_lpips, 't_l2': t_l2, 'c_lpips': c_lpips,
        'c_l2': c_l2, 'grad_clip_value': grad_clip_value
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess numpy array: (H, W, C) uint8 [0, 255] -> (1, C, H, W) float32 [0, 1]
    img_np = img_arr.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    result_container = []
    exception_container = []

    def worker_fn():
        try:
            result = _optimization_worker(img_np, config, device)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)

    # The research is correct: inference_mode is thread-local.
    # By running the optimization in a new thread, we escape the main no_grad() context.
    thread = threading.Thread(target=worker_fn)
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]

    if not result_container:
        raise RuntimeError("UnMarker optimization thread failed to return a result.")

    return result_container[0]
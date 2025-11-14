# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/texture_engine.py
# (Definitive Version with Stable Optimization)
import torch
import torch.optim as optim
import numpy as np
import threading
import lpips
import os
import kornia.filters as kfilters
import math

from ...modules.detection_bypass.utils.texture_utils import TextureMatcher

TEXTURE_MATCHER_CACHE = {}

def _optimization_worker(
    img_tensor_main: torch.Tensor,
    profile_path: str,
    iterations: int,
    learning_rate: float,
    strength: float,
    smoothness: float, # This now controls blur sigma
    device: torch.device,
    result_container: list,
    exception_container: list
):
    try:
        if profile_path in TEXTURE_MATCHER_CACHE:
            texture_matcher = TEXTURE_MATCHER_CACHE[profile_path]
        else:
            texture_matcher = TextureMatcher(profile_path, device)
            TEXTURE_MATCHER_CACHE[profile_path] = texture_matcher

        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad = False

        delta = torch.zeros_like(img_tensor_main, requires_grad=True)
        optimizer = optim.Adam([delta], lr=learning_rate)
        
        print(f"🧠 INSTARAW Texture Engine: Starting stable optimization for {iterations} steps...")
        with torch.no_grad():
            initial_loss = texture_matcher(img_tensor_main, log_stats=True)
            print(f"    - Initial Texture Loss: {initial_loss.item():.4f}")

        for i in range(iterations):
            optimizer.zero_grad()

            # --- THE FIX: Apply blur directly to the delta during the loop ---
            # This forces the optimizer to only learn smooth patterns.
            if smoothness > 0:
                kernel_size = 2 * math.ceil(2.0 * smoothness) + 1
                delta_smooth = kfilters.gaussian_blur2d(delta, (kernel_size, kernel_size), (smoothness, smoothness))
            else:
                delta_smooth = delta
            # --- END FIX ---
            
            perturbed_image = torch.clamp(img_tensor_main + delta_smooth, 0.0, 1.0)
            log_this_step = (i == 0) or ((i + 1) % 50 == 0) or (i == iterations - 1)
            
            loss_texture = texture_matcher(perturbed_image, log_stats=log_this_step)
            loss_lpips = lpips_model(perturbed_image, img_tensor_main).mean()
            
            # Final, balanced loss function
            total_loss = (loss_texture * 10.0) + (loss_lpips * 1.0)

            total_loss.backward()
            optimizer.step()
            
            if log_this_step:
                print(f"\n  [Step {i+1}/{iterations}]")
                print(f"    - LPIPS: {loss_lpips.item():.4f} | Total Loss: {total_loss.item():.4f}")

        with torch.no_grad():
            # Apply the final blur one last time for consistency
            if smoothness > 0:
                kernel_size = 2 * math.ceil(2.0 * smoothness) + 1
                final_delta = kfilters.gaussian_blur2d(delta, (kernel_size, kernel_size), (smoothness, smoothness))
            else:
                final_delta = delta
            
            final_image = torch.clamp(img_tensor_main + (final_delta * strength), 0.0, 1.0)
        
        result_container.append(final_image)
        print("✅ INSTARAW Texture Engine: Optimization complete.")

    except Exception as e:
        exception_container.append(e)

class INSTARAW_Texture_Engine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "profile_base_path": ("STRING", {"forceInput": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "smoothness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Controls the smoothness/scale of the added texture. 0.5 is fine grain, 2.0 is coarse."
                }),
                "iterations": ("INT", {"default": 200, "min": 50, "max": 1000, "step": 10}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Authenticity"

    def execute(self, image, profile_base_path, strength, smoothness, iterations, learning_rate):
        if strength == 0: return (image,)
        if not profile_base_path or not profile_base_path.strip():
            raise ValueError("An Authenticity Profile must be connected.")
        npz_path = f"{profile_base_path}.npz"
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Texture profile not found at: {npz_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processed_batches = []
        for img_tensor in image:
            img_tensor_batch = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            result_container, exception_container = [], []
            thread = threading.Thread(
                target=_optimization_worker,
                args=(img_tensor_batch, npz_path, iterations, learning_rate, strength, smoothness, device, result_container, exception_container)
            )
            thread.start(); thread.join()
            if exception_container: raise exception_container[0]
            if not result_container: raise RuntimeError("Texture Engine optimization failed to return a result.")
            processed_tensor_bchw = result_container[0]
            processed_tensor_bhwc = processed_tensor_bchw.permute(0, 2, 3, 1).to(image.device)
            processed_batches.append(processed_tensor_bhwc)
        return (torch.cat(processed_batches, dim=0),)

NODE_CLASS_MAPPINGS = { "INSTARAW_Texture_Engine": INSTARAW_Texture_Engine, }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_Texture_Engine": "🧠 INSTARAW Texture Engine", }
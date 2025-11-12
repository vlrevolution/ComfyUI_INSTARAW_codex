# Filename: ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_full.py
# ---
"""
Full two-stage spectral normalization implementation. FINAL VERSION.
Accepts external overrides for fine-tuning.
"""

import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import threading
from typing import Optional, Tuple, Dict, Any
import kornia

from .unmarker_losses import FFTLoss, LpipsVGG, LpipsAlex, NormLoss, DeeplossVGG
from .adaptive_filter import AdaptiveFilter
from .stats_utils import StatsMatcher

class SpectralNormalizer:
    # ... (The class and its methods __init__, _create_fft_loss, _apply_preprocess, _optimize_stage are all IDENTICAL to the last version. No changes needed here.) ...
    def __init__(
        self,
        stage1_iterations: int = 500, stage1_learning_rate: float = 3e-4, stage1_binary_steps: int = 5,
        stage2_iterations: int = 300, stage2_learning_rate: float = 1e-4, stage2_binary_steps: int = 3,
        lpips_type: str = "alex", t_lpips: float = 0.04, c_lpips: float = 0.01,
        t_l2: float = 3e-5, c_l2: float = 0.6, grad_clip_value: float = 0.005,
        use_adaptive_filter: bool = False, filter_kernels: Optional[list] = None,
        binary_const_growth: float = 10.0, crop_ratio: Optional[Tuple[float, float]] = None,
        attack_long_side_limit: Optional[int] = None, fft_scale_target: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True, stats_path: Optional[str] = None,
    ):
        self.stage1_iterations = stage1_iterations
        self.stage1_lr = stage1_learning_rate
        self.stage1_binary_steps = stage1_binary_steps
        self.stage2_iterations = stage2_iterations
        self.stage2_lr = stage2_learning_rate
        self.stage2_binary_steps = stage2_binary_steps
        self.t_lpips = t_lpips
        self.c_lpips = c_lpips
        self.t_l2 = t_l2
        self.c_l2 = c_l2
        self.grad_clip_value = grad_clip_value
        self.binary_const_growth = binary_const_growth
        self.use_adaptive_filter = use_adaptive_filter
        self.filter_kernels = filter_kernels if filter_kernels else [(7, 7)]
        self.device = torch.device(device)
        self.verbose = verbose
        self.crop_ratio = crop_ratio
        self.attack_long_side_limit = attack_long_side_limit
        self.fft_scale_target = fft_scale_target
        stats_path_to_load = (
            stats_path or globals().get("_stats_path_override") or os.environ.get("IPHONE_STATS_PATH")
            or Path(__file__).resolve().parents[3] / "pretrained" / "iphone_stats.npz"
        )
        self.stats_matcher = None
        if stats_path_to_load and Path(stats_path_to_load).is_file():
            try:
                self.stats_matcher = StatsMatcher(stats_path_to_load, self.device)
                if self.verbose:
                    print(f"✅ Stats-based guardrail enabled using: {stats_path_to_load}")
            except Exception as exc:
                print(f"⚠️ Failed to load stats from {stats_path_to_load}: {exc}")
        else:
            if self.verbose:
                print("ℹ️ iPhone stats file not found. Attack will be UNGUIDED.")
        if lpips_type == "deeploss":
            self.lpips_model = DeeplossVGG(device=self.device).eval()
        elif lpips_type == "vgg":
            self.lpips_model = LpipsVGG().to(self.device).eval()
        else:
            self.lpips_model = LpipsAlex().to(self.device).eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False
    def _create_fft_loss(self, stage: str):
        if stage == "high_freq": return FFTLoss(norm=1, power=1, use_tanh=False, use_grayscale=True).to(self.device)
        else: return FFTLoss(norm=2, power=1, use_tanh=False, use_grayscale=True).to(self.device)
    def _apply_preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.crop_ratio: return tensor
        ratio_h, ratio_w = self.crop_ratio
        _, _, height, width = tensor.shape
        crop_h, crop_w = max(1, int(round(height * ratio_h))), max(1, int(round(width * ratio_w)))
        start_h, start_w = max(0, (height - crop_h) // 2), max(0, (width - crop_w) // 2)
        cropped = tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        if cropped.shape[-2:] != (height, width):
            cropped = F.interpolate(cropped, size=(height, width), mode="bicubic", align_corners=False)
        return cropped
    def _optimize_stage(
        self, img_tensor: torch.Tensor, stage_name: str, iterations: int, learning_rate: float,
        binary_steps: int, stage_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        fft_loss_fn = self._create_fft_loss(stage_name)
        stage_params = stage_params.copy() if stage_params else {}
        const = stage_params.get("initial_const", 1.0)
        c_lpips = stage_params.get("c_lpips", 1.0)
        stats_weight = stage_params.get("stats_weight", 0.0)
        fft_scale_target = stage_params.get("fft_scale_target", self.fft_scale_target)
        delta_blur_sigma = stage_params.get("delta_blur_sigma", 0.0)
        if self.verbose:
            print(f"  [{stage_name}] Starting unified optimization ({iterations} steps)...")
            print(f"  [{stage_name}] Trade-off params: const={const:.2f}, c_lpips={c_lpips:.2f}, stats_weight={stats_weight:.2f}, delta_blur={delta_blur_sigma:.2f}")
        delta = torch.randn(img_tensor.shape[0], 1, img_tensor.shape[2], img_tensor.shape[3], device=self.device, dtype=img_tensor.dtype) * 0.0001
        delta.requires_grad = True
        filt = AdaptiveFilter(kernels=self.filter_kernels, shape=img_tensor.shape).to(self.device) if self.use_adaptive_filter else None
        optimizer = optim.Adam([delta] + (list(filt.parameters()) if filt else []), lr=learning_rate)
        dynamic_fft_scale = None
        early_stop_patience = stage_params.get("early_stop_patience", 400)
        no_improve_steps, best_iter_loss = 0, float("inf")
        for i in range(iterations):
            optimizer.zero_grad()
            delta_rgb = delta.expand_as(img_tensor)
            x_adv = torch.clamp(img_tensor + (filt(delta_rgb, img_tensor) if filt else delta_rgb), -1, 1)
            loss_fft_raw = fft_loss_fn(x_adv, img_tensor).mean()
            if dynamic_fft_scale is None:
                init_fft = abs(loss_fft_raw.detach()).item()
                dynamic_fft_scale = fft_scale_target / max(init_fft, 1e-9)
            loss_fft = -loss_fft_raw * dynamic_fft_scale
            loss_lpips = self.lpips_model(x_adv, img_tensor).mean()
            stats_penalty = self.stats_matcher(x_adv)[0] if self.stats_matcher else 0
            loss_filter = filt.compute_loss(x_adv).mean() if filt else 0
            total_loss = (const * loss_fft) + (c_lpips * loss_lpips) + (stats_weight * stats_penalty) + loss_filter
            total_loss.backward()
            if delta.grad is not None:
                delta.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)
            optimizer.step()
            if delta_blur_sigma > 0:
                with torch.no_grad():
                    kernel_size = int(delta_blur_sigma * 3) * 2 + 1
                    delta.data = kornia.filters.gaussian_blur2d(delta.data, (kernel_size, kernel_size), (delta_blur_sigma, delta_blur_sigma))
            current_loss = total_loss.item()
            if current_loss < best_iter_loss - 1e-5:
                best_iter_loss, no_improve_steps = current_loss, 0
            else:
                no_improve_steps += 1
            if no_improve_steps > early_stop_patience:
                if self.verbose: print(f"    DEBUG: Early stopping due to stagnation.")
                break
            if self.verbose and (i % 100 == 0 or i == iterations - 1):
                stats_str = f", Stats={stats_penalty.item():.4f}" if self.stats_matcher else ""
                print(f"    Iter {i}/{iterations}: TotalLoss={total_loss.item():.4f}, FFT={loss_fft.item():.4f}, LPIPS={loss_lpips.item():.4f}{stats_str}")
        with torch.no_grad():
            delta_rgb = delta.expand_as(img_tensor)
            final_adv = torch.clamp(img_tensor + (filt(delta_rgb, img_tensor) if filt else delta_rgb), -1, 1)
            final_lpips = self.lpips_model(final_adv, img_tensor).mean().item()
            stats_final_val = self.stats_matcher(final_adv)[0].item() if self.stats_matcher else float('inf')
        if self.verbose:
            print(f"  [{stage_name}] Optimization finished. Final LPIPS={final_lpips:.4f}, Final Stats={stats_final_val:.4f}")
        return final_adv
    def process(self, img_np: np.ndarray, stage_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> np.ndarray:
        stage_overrides = stage_overrides or {}
        img_np_float = img_np.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np_float, (2, 0, 1))[None, ...]).to(self.device)
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = self._apply_preprocess(img_tensor)
        original_shape = img_tensor.shape[-2:]
        resize_back_shape = None
        if self.attack_long_side_limit and max(original_shape) > self.attack_long_side_limit:
            scale = self.attack_long_side_limit / max(original_shape)
            new_h, new_w = int(round(original_shape[0] * scale)), int(round(original_shape[1] * scale))
            img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode="bicubic", align_corners=False)
            resize_back_shape = original_shape
            if self.verbose: print(f"   ↳ Downscaling canvas to {new_h}x{new_w}")
        if self.verbose:
            print(f"🖼️ Input resolution: {img_np.shape[0]}x{img_np.shape[1]}")
            print("🚀 INSTARAW Spectral Engine: Starting GUIDED normalization...")
        stage1_params = stage_overrides.get("high_freq", {})
        if self.verbose: print("\n=== STAGE 1: High Frequency Normalization ===")
        stage1_out = self._optimize_stage(
            img_tensor, "high_freq", self.stage1_iterations, self.stage1_lr,
            self.stage1_binary_steps, stage1_params
        )
        stage2_params = stage_overrides.get("low_freq", {})
        if self.verbose: print("\n=== STAGE 2: Low Frequency Refinement ===")
        stage2_out = self._optimize_stage(
            stage1_out, "low_freq", self.stage2_iterations, self.stage2_lr,
            self.stage2_binary_steps, stage2_params
        )
        if resize_back_shape is not None:
            stage2_out = F.interpolate(stage2_out, size=resize_back_shape, mode="bicubic", align_corners=False)
        final_tensor = (stage2_out.squeeze(0).cpu().detach() + 1) / 2
        final_np = (final_tensor.permute(1, 2, 0).clamp(0, 1) * 255).numpy().astype(np.uint8)
        if self.verbose: print("\n✅ INSTARAW Spectral Engine: Normalization complete!")
        return final_np

def _size_profile_overrides(
    height: int, width: int, preset_name: str, base_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    max_dim = max(height, width)
    size_ratio = max(1.0, max_dim / 512.0)
    overrides: Dict[str, Any] = {}
    stage_params: Dict[str, Dict[str, Any]] = {"high_freq": {}, "low_freq": {}}
    overrides["crop_ratio"] = (1.0, 1.0)
    overrides["lpips_type"] = "vgg" if max_dim > 768 else "alex"
    s1_iters = int(min(1500, 800 * size_ratio))
    s2_iters = int(min(800, 400 * size_ratio))
    overrides["stage1_iterations"] = max(base_config.get("stage1_iterations", 500), s1_iters)
    overrides["stage2_iterations"] = max(base_config.get("stage2_iterations", 300), s2_iters)
    stage_params["high_freq"].update({
        "initial_const": 30.0, "c_lpips": 40.0, "stats_weight": 4.0,
        "fft_scale_target": 0.05, "delta_blur_sigma": 0.5,
    })
    stage_params["low_freq"].update({
        "initial_const": 20.0, "c_lpips": 50.0, "stats_weight": 3.0,
        "fft_scale_target": 0.03, "delta_blur_sigma": 1.0,
    })
    meta = {}
    return overrides, stage_params, meta

def normalize_spectrum_twostage(img_arr: np.ndarray, preset: str = "balanced", **kwargs) -> np.ndarray: # Rebranded function
    presets = {
        "fast": {"stage1_binary_steps": 3, "stage2_binary_steps": 2, "use_adaptive_filter": False, "attack_long_side_limit": 900},
        "balanced": {"stage1_binary_steps": 4, "stage2_binary_steps": 3, "use_adaptive_filter": False, "attack_long_side_limit": 1152, "lpips_type": "deeploss"},
        "quality": {"stage1_binary_steps": 6, "stage2_binary_steps": 4, "use_adaptive_filter": True, "filter_kernels": [(7, 7), (15, 15)], "attack_long_side_limit": 1536, "lpips_type": "deeploss"},
    }
    base_config = {**presets.get(preset, presets["balanced"])}
    
    # --- THIS IS THE FIX ---
    # Enforce the analysis resolution for maximum effectiveness.
    # We will override the preset's attack_long_side_limit with our known best value.
    base_config['attack_long_side_limit'] = 1920
    # --- END FIX ---

    user_stage_overrides = kwargs.pop("stage_overrides", {})
    seed_value = kwargs.pop("seed", None)
    
    config = {**base_config, **kwargs}
    height, width = img_arr.shape[:2]
    attack_limit = config.get("attack_long_side_limit")
    effective_h, effective_w = height, width
    if attack_limit and max(height, width) > attack_limit:
        scale = attack_limit / max(height, width)
        effective_h, effective_w = int(round(height * scale)), int(round(width * scale))

    # ... (the rest of the function is identical) ...
    size_overrides, stage_params, meta = _size_profile_overrides(effective_h, effective_w, preset, config)
    for key, value in size_overrides.items():
        if key not in kwargs: config[key] = value
    final_stage_overrides = {
        "high_freq": {**stage_params.get("high_freq", {}), **user_stage_overrides.get("high_freq", {})},
        "low_freq": {**stage_params.get("low_freq", {}), **user_stage_overrides.get("low_freq", {})},
        "_meta": meta
    }
    result_container, exception_container = [], []
    def worker_fn():
        try:
            if seed_value is not None:
                torch.manual_seed(seed_value)
            engine = SpectralNormalizer(**config)
            result = engine.process(img_arr, stage_overrides=final_stage_overrides)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)
    thread = threading.Thread(target=worker_fn)
    thread.start()
    thread.join()
    if exception_container: raise exception_container[0]
    if not result_container: raise RuntimeError("INSTARAW Spectral Engine failed to return a result.")
    return result_container[0]
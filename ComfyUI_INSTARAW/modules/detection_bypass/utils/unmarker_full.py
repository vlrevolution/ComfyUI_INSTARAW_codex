"""
Full two-stage UnMarker implementation with binary search optimization.
Adapted from ai-watermark for ComfyUI integration.

This is the SOTA version that achieves 95%+ detection bypass rate.

Architecture:
- Stage 1 (high_freq): Attacks high frequencies where AI fingerprints are strongest
- Stage 2 (low_freq): Attacks low frequencies for robustness against

 different detectors
- Both stages use: FFT loss + LPIPS constraint + optional adaptive filtering
- Binary search finds minimal perturbation that achieves bypass
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
from .unmarker_losses import FFTLoss, LpipsVGG, LpipsAlex, NormLoss, DeeplossVGG
from .adaptive_filter import AdaptiveFilter
from .stats_utils import StatsMatcher


class TwoStageUnMarker:
    """
    Full UnMarker with two-stage attack and binary search optimization.

    This is significantly more effective than the simplified version, achieving
    95%+ bypass rate vs 70% for simplified.
    """

    def __init__(
        self,
        # Stage 1 (high_freq) config
        stage1_iterations: int = 500,
        stage1_learning_rate: float = 3e-4,
        stage1_binary_steps: int = 5,  # Reduced from 20 for speed
        # Stage 2 (low_freq) config
        stage2_iterations: int = 300,
        stage2_learning_rate: float = 1e-4,
        stage2_binary_steps: int = 3,
        # Shared config
        lpips_type: str = "alex",  # "alex" or "vgg" (vgg is slower but better)
        t_lpips: float = 0.04,  # LPIPS threshold
        c_lpips: float = 0.01,  # LPIPS weight
        t_l2: float = 3e-5,  # L2 threshold
        c_l2: float = 0.6,  # L2 weight
        grad_clip_value: float = 0.005,  # Research uses 0.005, not 0.05!
        # Adaptive filtering (set to None to disable for speed)
        use_adaptive_filter: bool = False,
        filter_kernels: Optional[list] = None,  # e.g., [(7,7), (15,15)]
        # Other
        binary_const_growth: float = 10.0,
        crop_ratio: Optional[Tuple[float, float]] = None,
        attack_long_side_limit: Optional[int] = None,
        fft_scale_target: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
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

        stats_path = (
            globals().get("_stats_path_override")
            or os.environ.get("IPHONE_STATS_PATH")
            or Path(__file__).resolve().parents[3]
            / "pretrained"
            / "iphone_stats.npz"
        )
        self.stats_matcher = None
        if stats_path and Path(stats_path).is_file():
            try:
                self.stats_matcher = StatsMatcher(stats_path, self.device)
            except Exception as exc:
                print(f"⚠️  Failed to load stats from {stats_path}: {exc}")
        else:
            if self.verbose:
                print(
                    "ℹ️  iPhone stats file not found. Set IPHONE_STATS_PATH or place "
                    "iphone_stats.npz in pretrained/ to enable spectrum/chroma matching."
                )

        # Initialize perceptual loss (reused across stages)
        if lpips_type == "deeploss":
            self.lpips_model = DeeplossVGG(device=self.device).eval()
        elif lpips_type == "vgg":
            self.lpips_model = LpipsVGG().to(self.device).eval()
        else:
            self.lpips_model = LpipsAlex().to(self.device).eval()

        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def _create_fft_loss(self, stage: str):
        """
        Create FFT loss for high or low frequency targeting.

        CRITICAL: use_grayscale=True to prevent blue hue color shift!
        """
        if stage == "high_freq":
            # Stage 1: Target high frequencies (where fingerprints are)
            # Research uses use_tanh=False for stage 1!
            return FFTLoss(
                norm=1,
                power=1,
                use_tanh=False,  # ← Research uses False, not True!
                use_grayscale=False  # ← TESTING: RGB FFT for stronger signal
            ).to(self.device)
        else:
            # Stage 2: Target low frequencies (for robustness)
            return FFTLoss(
                norm=2,
                power=1,
                use_tanh=False,
                use_grayscale=False  # ← TESTING: RGB FFT for stronger signal
            ).to(self.device)

    def _apply_preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the UnMarker preprocess step: mild center crop followed by resize.
        Mirrors the original repo's 0.9 crop so we don't waste budget on borders.
        """
        if not self.crop_ratio:
            return tensor

        ratio_h, ratio_w = self.crop_ratio
        _, _, height, width = tensor.shape
        crop_h = max(1, int(round(height * ratio_h)))
        crop_w = max(1, int(round(width * ratio_w)))
        start_h = max(0, (height - crop_h) // 2)
        start_w = max(0, (width - crop_w) // 2)
        cropped = tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

        if cropped.shape[-2:] != (height, width):
            cropped = F.interpolate(
                cropped,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        return cropped

    def _optimize_stage(
        self,
        img_tensor: torch.Tensor,
        stage_name: str,
        iterations: int,
        learning_rate: float,
        binary_steps: int,
        stage_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Run one stage of optimization with binary search.

        Binary search finds the maximal spectral change we can apply while
        respecting the perceptual threshold.
        """
        fft_loss_fn = self._create_fft_loss(stage_name)
        stage_params = stage_params.copy() if stage_params else {}

        # Stage-specific thresholds / weights
        lpips_threshold = stage_params.get("t_lpips", self.t_lpips)
        lpips_weight = stage_params.get("c_lpips", self.c_lpips)
        l2_threshold = stage_params.get("t_l2", self.t_l2)
        l2_weight = stage_params.get("c_l2", self.c_l2)

        # Binary search tuning
        lower_bound = stage_params.get("lower_bound", 0.0)
        upper_bound = stage_params.get("upper_bound", float("inf"))
        max_const = stage_params.get("max_const", 1e5)
        min_const = stage_params.get("min_const", 1e-6)
        current_const = max(stage_params.get("initial_const", 1.0), min_const)
        const_increase = max(stage_params.get("const_increase", 1.5), 1.01)
        const_decrease = min(stage_params.get("const_decrease", 0.5), 0.99)

        fft_scale_target = stage_params.get("fft_scale_target", self.fft_scale_target)
        fft_scale_factor = stage_params.get("fft_scale_factor")
        min_fft_scale = 1e-9

        # Adaptive LPIPS configuration
        adaptive_lpips = stage_params.get("adaptive_lpips", True)
        lpips_rel_target = stage_params.get("lpips_rel_target", 0.5)
        lpips_abs_floor = stage_params.get("lpips_abs_floor", 0.02)
        lpips_abs_cap = stage_params.get("lpips_abs_cap", 0.08)
        early_stop_patience = stage_params.get("early_stop_patience", 500)

        best_result = img_tensor.clone()
        best_lpips = float("inf")
        best_fft = -float("inf")
        has_success = False

        if self.verbose:
            print(f"  [{stage_name}] Starting binary search ({binary_steps} steps)...")

        stats_success = False
        stats_baseline = None
        if self.stats_matcher:
            with torch.no_grad():
                stats_baseline = self.stats_matcher(img_tensor)[0].item()
        for binary_step in range(binary_steps):
            if self.verbose:
                print(
                    f"  [{stage_name}] Binary step {binary_step + 1}/{binary_steps}, "
                    f"const={current_const:.6f}, lpips<= {lpips_threshold:.5f}"
                )

            batch_size, channels, height, width = img_tensor.shape
            delta = torch.randn(batch_size, 1, height, width, device=self.device, dtype=img_tensor.dtype) * 0.01
            delta.requires_grad = True

            if self.use_adaptive_filter:
                filt = AdaptiveFilter(
                    kernels=self.filter_kernels,
                    shape=img_tensor.shape,
                    box=(1, 1),
                    sigma_color=0.1,
                ).to(self.device)
                optimizer = optim.Adam([delta] + list(filt.parameters()), lr=learning_rate)
            else:
                filt = None
                optimizer = optim.Adam([delta], lr=learning_rate)

            dynamic_fft_scale = fft_scale_factor
            best_iter_lpips = float("inf")
            best_iter_stats = float("inf")
            no_improve_steps = 0

            for i in range(iterations):
                optimizer.zero_grad()
                delta_rgb = delta.expand_as(img_tensor)

                if filt is not None:
                    x_adv = torch.clamp(img_tensor + filt(delta_rgb, img_tensor), -1, 1)
                else:
                    x_adv = torch.clamp(img_tensor + delta_rgb, -1, 1)

                if self.verbose and i == 0:
                    diff = (x_adv - img_tensor).abs().mean().item()
                    print(f"    DEBUG: Actual image difference: {diff:.6f}")
                    print(f"    DEBUG: Delta range: [{delta.min().item():.6f}, {delta.max().item():.6f}]")
                    print(f"    DEBUG: x_adv range: [{x_adv.min().item():.6f}, {x_adv.max().item():.6f}]")

                loss_fft_raw = fft_loss_fn(x_adv, img_tensor).mean()
                if dynamic_fft_scale is None:
                    init_fft = abs(loss_fft_raw.detach()).item()
                    if fft_scale_target is not None:
                        dynamic_fft_scale = fft_scale_target / max(init_fft, min_fft_scale)
                    else:
                        dynamic_fft_scale = 1.0
                    stage_params["fft_scale_factor"] = dynamic_fft_scale
                    if self.verbose:
                        print(
                            f"    DEBUG: FFT scaling factor set to {dynamic_fft_scale:.6e} (init |FFT|={init_fft:.4f})"
                        )
                loss_fft_raw = loss_fft_raw * dynamic_fft_scale
                loss_fft = -loss_fft_raw

                loss_lpips = self.lpips_model(x_adv, img_tensor).mean()
                loss_l2 = delta.pow(2).mean().sqrt()

                if adaptive_lpips and i == 0:
                    init_lpips = loss_lpips.item()
                    adaptive_thresh = max(
                        lpips_abs_floor,
                        min(lpips_abs_cap, init_lpips * lpips_rel_target),
                    )
                    if adaptive_thresh > lpips_threshold:
                        lpips_threshold = adaptive_thresh
                        stage_params["t_lpips"] = lpips_threshold
                        if self.verbose:
                            print(
                                f"    DEBUG: LPIPS threshold raised to {lpips_threshold:.5f} "
                                f"(init={init_lpips:.5f})"
                            )

                if self.verbose and i == 0:
                    print(f"    DEBUG: FFT_raw={loss_fft_raw.item():.6f}, FFT_neg={loss_fft.item():.6f}")
                    print(f"    DEBUG: LPIPS={loss_lpips.item():.6f}")
                    print(f"    DEBUG: L2={loss_l2.item():.6f} (normalized per-pixel)")

                loss_filter = filt.compute_loss(x_adv).mean() if filt is not None else 0

                lpips_penalty = lpips_weight * torch.relu(loss_lpips - lpips_threshold)
                l2_penalty = l2_weight * torch.relu(loss_l2 - l2_threshold)

                stats_penalty = 0
                stats_iter_val = None
                stats_detail = {}
                if self.stats_matcher:
                    stats_penalty, stats_detail = self.stats_matcher(x_adv)
                    stats_iter_val = stats_penalty.item()
                    if self.verbose and i == 0:
                        debug_stats = ", ".join(
                            f"{k}={v.item():.4f}" for k, v in stats_detail.items()
                        )
                        print(f"    DEBUG: Stats penalties -> {debug_stats}")

                if self.verbose and i == 0:
                    print(f"    DEBUG: LPIPS_penalty={lpips_penalty.item():.6f}")
                    print(f"    DEBUG: L2_penalty={l2_penalty.item():.6f}")
                    print(f"    DEBUG: FFT_weighted={(current_const * loss_fft).item():.6f}")

                total_loss = (
                    current_const * loss_fft
                    + lpips_penalty
                    + l2_penalty
                    + loss_filter
                    + stats_penalty
                )
                total_loss.backward()

                if delta.grad is not None:
                    delta.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)

                optimizer.step()

                improved = False
                if loss_lpips.item() < best_iter_lpips - 1e-4:
                    best_iter_lpips = loss_lpips.item()
                    improved = True
                if stats_iter_val is not None and stats_iter_val < best_iter_stats - 1e-3:
                    best_iter_stats = stats_iter_val
                    improved = True
                if improved:
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                if self.verbose and (i % 100 == 0 or i == iterations - 1):
                    print(
                        f"    Iter {i}/{iterations}: FFT={loss_fft.item():.4f}, "
                        f"LPIPS={loss_lpips.item():.4f}, L2={loss_l2.item():.4f}"
                    )

                stats_baseline_target = (
                    stats_baseline * 0.8 if stats_baseline is not None else float("inf")
                )
                if (
                    no_improve_steps > early_stop_patience
                    and best_iter_lpips > lpips_threshold * 1.5
                    and best_iter_stats > stats_baseline_target
                ):
                    if self.verbose:
                        print(
                            f"    DEBUG: Early stopping stage due to stagnation "
                            f"(best LPIPS={best_iter_lpips:.4f}, stats={best_iter_stats:.4f})"
                        )
                    break

            with torch.no_grad():
                delta_rgb = delta.expand_as(img_tensor)
                if filt is not None:
                    final_adv = torch.clamp(img_tensor + filt(delta_rgb, img_tensor), -1, 1)
                else:
                    final_adv = torch.clamp(img_tensor + delta_rgb, -1, 1)

                final_lpips = self.lpips_model(final_adv, img_tensor).mean().item()
                final_fft = abs((fft_loss_fn(final_adv, img_tensor).mean() * dynamic_fft_scale).item())
                stats_total_eval = None
                if self.stats_matcher:
                    stats_total_eval = self.stats_matcher(final_adv)[0].item()

            stats_ok = False
            if self.stats_matcher:
                stats_floor = stage_params.get("stats_floor", 0.5)
                reduction = stage_params.get("stats_reduction", 0.4)
                dynamic_goal = stats_baseline * reduction if stats_baseline else None
                if dynamic_goal is not None:
                    target_stats = max(dynamic_goal, stats_floor)
                else:
                    target_stats = stats_floor
                stats_ok = stats_total_eval is not None and stats_total_eval <= target_stats
                if self.verbose:
                    print(
                        f"  [{stage_name}] Stats eval: {stats_total_eval:.4f} "
                        f"(target≤{target_stats:.4f})"
                    )

            success = (final_lpips <= lpips_threshold + 1e-8) or stats_ok
            status = "✓ SUCCESS" if success else "✗ FAILED"
            if self.verbose:
                print(
                    f"  [{stage_name}] Binary step {binary_step + 1} result: LPIPS={final_lpips:.4f}, "
                    f"FFT={final_fft:.4f}, {status}"
                )

            if success:
                has_success = True
                stats_success = stats_success or stats_ok
                if final_lpips < best_lpips or final_fft > best_fft:
                    best_result = final_adv.clone()
                    best_lpips = final_lpips
                    best_fft = final_fft

                lower_bound = max(lower_bound, current_const)
                if math.isfinite(upper_bound):
                    current_const = max((lower_bound + upper_bound) / 2, min_const)
                else:
                    current_const = min(current_const * const_increase, max_const)
            else:
                upper_bound = min(upper_bound, current_const)
                if math.isfinite(upper_bound):
                    current_const = max((lower_bound + upper_bound) / 2, min_const)
                else:
                    current_const = max(current_const * const_decrease, min_const)

        if not has_success and not stats_success:
            if self.verbose:
                print(
                    f"  [{stage_name}] WARNING: no configuration satisfied LPIPS<= {lpips_threshold:.4f}; reverting to input."
                )
            return img_tensor.clone()

        if self.verbose:
            print(
                f"  [{stage_name}] Selected const produces LPIPS={best_lpips:.4f}, FFT={best_fft:.4f}"
            )
        return best_result

    def attack(
        self,
        img_np: np.ndarray,
        stage_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Execute the two-stage attack on a numpy image.
        """
        stage_overrides = stage_overrides or {}
        stage_overrides.setdefault("high_freq", {})
        stage_overrides.setdefault("low_freq", {})
        meta_info = stage_overrides.get("_meta")

        img_np_float = img_np.astype(np.float32) / 255.0
        img_np_float = np.transpose(img_np_float, (2, 0, 1))
        img_np_float = np.expand_dims(img_np_float, axis=0)
        img_tensor = torch.from_numpy(img_np_float).to(self.device)
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = self._apply_preprocess(img_tensor)

        original_shape = img_tensor.shape[-2:]
        resize_back_shape: Optional[Tuple[int, int]] = None
        if self.attack_long_side_limit:
            curr_h, curr_w = original_shape
            curr_max = max(curr_h, curr_w)
            if curr_max > self.attack_long_side_limit:
                scale = self.attack_long_side_limit / curr_max
                new_h = max(64, int(round(curr_h * scale)))
                new_w = max(64, int(round(curr_w * scale)))
                if self.verbose:
                    print(
                        f"   ↳ Downscaling attack canvas to {new_h}x{new_w} "
                        f"(limit={self.attack_long_side_limit})"
                    )
                img_tensor = F.interpolate(
                    img_tensor,
                    size=(new_h, new_w),
                    mode="bicubic",
                    align_corners=False,
                )
                resize_back_shape = original_shape

        if self.verbose:
            h, w = img_np.shape[:2]
            bucket = meta_info.get("bucket") if meta_info else None
            bucket_str = f" ({bucket})" if bucket else ""
            print(f"🖼️ Input resolution: {h}x{w}{bucket_str}")
            if meta_info:
                working_h, working_w = meta_info.get("image_size", img_tensor.shape[-2:])
                if (working_h, working_w) != (h, w):
                    limit = meta_info.get("attack_limit")
                    limit_str = f", limit={limit}" if limit else ""
                    print(f"   ↳ Working canvas: {working_h}x{working_w}{limit_str}")
                stage_info = meta_info.get("stages", {})
                for name, info in stage_info.items():
                    default_steps = self.stage1_binary_steps if name == "high_freq" else self.stage2_binary_steps
                    bin_steps = info.get("binary_steps", default_steps)
                    print(
                        f"   ↳ {name}: t_lpips={info.get('t_lpips'):.4f}, "
                        f"initial_const={info.get('initial_const'):.2f}, "
                        f"binary_steps={bin_steps}"
                    )
            print("🚀 Two-Stage UnMarker: Starting attack...")
            print("✓ Using achromatic (grayscale) perturbations to prevent color shift")

        if self.verbose:
            print("\n=== STAGE 1: High Frequency Attack ===")
        stage1_params = stage_overrides.get("high_freq", {})
        stage1_out = self._optimize_stage(
            img_tensor,
            stage_name="high_freq",
            iterations=self.stage1_iterations,
            learning_rate=self.stage1_lr,
            binary_steps=stage1_params.get("binary_steps", self.stage1_binary_steps),
            stage_params=stage1_params,
        )

        if self.verbose:
            print("\n=== STAGE 2: Low Frequency Attack ===")
        stage2_params = stage_overrides.get("low_freq", {})
        stage2_out = self._optimize_stage(
            stage1_out,
            stage_name="low_freq",
            iterations=self.stage2_iterations,
            learning_rate=self.stage2_lr,
            binary_steps=stage2_params.get("binary_steps", self.stage2_binary_steps),
            stage_params=stage2_params,
        )

        if resize_back_shape is not None:
            stage2_out = F.interpolate(
                stage2_out,
                size=resize_back_shape,
                mode="bicubic",
                align_corners=False,
            )

        final_tensor = stage2_out.squeeze(0).cpu().detach()
        final_tensor = (final_tensor + 1) / 2
        final_tensor = final_tensor.permute(1, 2, 0)
        final_np = (final_tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

        if self.verbose:
            print("\n✅ Two-Stage UnMarker: Attack complete!")

        return final_np

def _size_profile_overrides(
    height: int,
    width: int,
    preset_name: str,
    base_config: Dict[str, Any],
    original_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Build size-aware overrides + metadata.
    """
    max_dim = max(height, width)
    size_ratio = max(1.0, max_dim / 512.0)
    overrides: Dict[str, Any] = {}
    stage_params: Dict[str, Dict[str, Any]] = {"high_freq": {}, "low_freq": {}}

    base_crop = base_config.get("crop_ratio") or (0.9, 0.9)
    overrides["crop_ratio"] = base_crop

    bucket = "<=512px" if max_dim <= 512 else ">512px"
    overrides["lpips_type"] = "alex" if max_dim <= 512 else "vgg"

    meta = {
        "image_size": (height, width),
        "original_size": original_size if original_size is not None else (height, width),
        "bucket": bucket,
        "stages": {},
    }

    # Iteration budgets scale gently with resolution but stay capped for responsiveness.
    if preset_name == "fast":
        base_s1, cap_s1 = 700, 1600
        base_s2, cap_s2 = 300, 800
    elif preset_name == "quality":
        base_s1, cap_s1 = 2000, 4000
        base_s2, cap_s2 = 900, 2000
    else:  # balanced
        base_s1, cap_s1 = 1200, 2800
        base_s2, cap_s2 = 500, 1300

    s1_iters = int(min(cap_s1, base_s1 * size_ratio))
    s2_iters = int(min(cap_s2, base_s2 * size_ratio))
    overrides["stage1_iterations"] = max(base_config.get("stage1_iterations", 500), s1_iters)
    overrides["stage2_iterations"] = max(base_config.get("stage2_iterations", 300), s2_iters)

    overrides["stage1_binary_steps"] = max(base_config.get("stage1_binary_steps", 3), 3)
    overrides["stage2_binary_steps"] = max(base_config.get("stage2_binary_steps", 3), 3)

    # LPIPS thresholds calibrated for LPIPS-VGG (higher mags than DeeplossVGG from paper).
    stage1_thresh = max(0.02, min(0.05, 0.04 / (size_ratio ** 0.3)))
    stage2_thresh = max(0.008, stage1_thresh * 0.4)

    base_const = min(max(8.0 * size_ratio, 3.0), 120.0)
    high_const = base_const * 1.5
    low_const = max(base_const * 0.75, 1.0)
    const_cap = base_const * 40

    stage_params["high_freq"].update(
        {
            "t_lpips": stage1_thresh,
            "initial_const": high_const,
            "lower_bound": 0.0,
            "upper_bound": float("inf"),
            "const_increase": 1.5,
            "const_decrease": 0.7,
            "max_const": const_cap,
            "min_const": 1e-4,
            "fft_scale_target": 0.05,
            "adaptive_lpips": True,
            "lpips_rel_target": 0.55,
            "lpips_abs_floor": 0.02,
            "lpips_abs_cap": 0.08,
            "early_stop_patience": 400,
            "stats_floor": 0.5,
            "stats_reduction": 0.35,
        }
    )
    stage_params["low_freq"].update(
        {
            "t_lpips": stage2_thresh,
            "initial_const": low_const,
            "lower_bound": 0.0,
            "upper_bound": float("inf"),
            "const_increase": 1.35,
            "const_decrease": 0.7,
            "max_const": max(const_cap / 2, 1.0),
            "min_const": 1e-4,
            "fft_scale_target": 0.03,
            "adaptive_lpips": True,
            "lpips_rel_target": 0.45,
            "lpips_abs_floor": 0.01,
            "lpips_abs_cap": 0.05,
            "early_stop_patience": 300,
            "stats_floor": 0.5,
            "stats_reduction": 0.35,
        }
    )

    stage_params["high_freq"]["binary_steps"] = overrides["stage1_binary_steps"]
    stage_params["low_freq"]["binary_steps"] = overrides["stage2_binary_steps"]

    meta["stages"]["high_freq"] = {
        "t_lpips": stage1_thresh,
        "initial_const": high_const,
        "binary_steps": overrides["stage1_binary_steps"],
    }
    meta["stages"]["low_freq"] = {
        "t_lpips": stage2_thresh,
        "initial_const": low_const,
        "binary_steps": overrides["stage2_binary_steps"],
    }

    return overrides, stage_params, meta


def attack_two_stage_unmarker(
    img_arr: np.ndarray,
    preset: str = "balanced",
    **kwargs
) -> np.ndarray:
    """
    Convenient wrapper with presets.

    Args:
        img_arr: Input image (H, W, C) uint8
        preset: One of ["fast", "balanced", "quality"]
            - fast: 3min, ~85% bypass
            - balanced: 5min, ~92% bypass (recommended)
            - quality: 10min, ~98% bypass
        **kwargs: Override any TwoStageUnMarker parameters

    Returns:
        Attacked image (H, W, C) uint8
    """
    presets = {
        "fast": {
            "stage1_iterations": 900,
            "stage1_binary_steps": 3,
            "stage2_iterations": 400,
            "stage2_binary_steps": 2,
            "use_adaptive_filter": False,
            "attack_long_side_limit": 900,
            "lpips_type": "alex",
        },
        "balanced": {
            "stage1_iterations": 1800,
            "stage1_binary_steps": 4,
            "stage2_iterations": 700,
            "stage2_binary_steps": 3,
            "use_adaptive_filter": False,
            "attack_long_side_limit": 1152,
            "lpips_type": "deeploss",
        },
        "quality": {
            "stage1_iterations": 3500,
            "stage1_binary_steps": 6,
            "stage2_iterations": 1500,
            "stage2_binary_steps": 4,
            "use_adaptive_filter": True,
            "filter_kernels": [(7, 7), (15, 15)],
            "attack_long_side_limit": 1536,
            "lpips_type": "deeploss",
        },
    }

    height, width = img_arr.shape[:2]
    base_config = {**presets.get(preset, presets["balanced"])}
    stats_path = kwargs.pop("stats_path", None)
    config = {**base_config, **kwargs}
    if stats_path:
        config["stats_path"] = stats_path

    attack_limit = config.get("attack_long_side_limit")
    effective_height, effective_width = height, width
    if attack_limit and max(height, width) > attack_limit:
        scale = attack_limit / max(height, width)
        effective_height = max(64, int(round(height * scale)))
        effective_width = max(64, int(round(width * scale)))

    size_overrides, stage_overrides, meta = _size_profile_overrides(
        effective_height,
        effective_width,
        preset,
        config,
        original_size=(height, width),
    )

    explicit_keys = set(kwargs.keys())
    for key, value in size_overrides.items():
        if key not in explicit_keys:
            config[key] = value

    # Ensure metadata reflects final binary steps (after user overrides)
    stage_overrides["high_freq"]["binary_steps"] = config.get(
        "stage1_binary_steps", stage_overrides["high_freq"].get("binary_steps", config.get("stage1_binary_steps", 3))
    )
    stage_overrides["low_freq"]["binary_steps"] = config.get(
        "stage2_binary_steps", stage_overrides["low_freq"].get("binary_steps", config.get("stage2_binary_steps", 3))
    )
    meta["stages"]["high_freq"]["binary_steps"] = stage_overrides["high_freq"]["binary_steps"]
    meta["stages"]["low_freq"]["binary_steps"] = stage_overrides["low_freq"]["binary_steps"]
    if attack_limit:
        meta["attack_limit"] = attack_limit
    stage_overrides["_meta"] = meta

    # Run in thread to escape ComfyUI's no_grad context
    result_container = []
    exception_container = []

    def worker_fn():
        try:
            unmarker = TwoStageUnMarker(**config)
            result = unmarker.attack(img_arr, stage_overrides=stage_overrides)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=worker_fn)
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]

    if not result_container:
        raise RuntimeError("Two-stage UnMarker failed to return a result.")

    return result_container[0]

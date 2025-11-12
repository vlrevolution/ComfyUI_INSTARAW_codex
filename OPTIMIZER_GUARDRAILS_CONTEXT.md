# Optimizer Guardrails Plan

This document captures the pending changes for `ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_full.py` so we can resume later (even in another environment) without re-deriving the design.

## Goal
The current two-stage UnMarker optimizer overdrives LPIPS (0.5+), creating dark spots and reverting aggressively. We want to:

1. Keep LPIPS within ~0.12 so perturbations stay subtle.
2. Add a smoothness (TV) penalty to avoid blotchy artifacts.
3. Weight spectral/stats losses by a quality mask so we only push low-detail regions.
4. Continue using stats-baseline logic (already added) to accept perturbations once stats improve significantly.

## Implementation Plan

### 1. Quality Mask Helper
Add a method to `TwoStageUnMarker`:
```python
def _quality_mask(self, tensor: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    gray = (0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3])
    blur = F.avg_pool2d(gray, ksize, 1, ksize // 2)
    var = F.avg_pool2d(gray ** 2, ksize, 1, ksize // 2) - blur ** 2
    norm_var = var / (var.max().clamp_min(1e-6))
    mask = 1.0 - norm_var
    return mask.clamp(0.2, 1.0)
```

Call this once per stage (after we compute `img_tensor`) and store `quality_mask`.

### 2. Total Variation Penalty
Add helper:
```python
def _tv_penalty(self, delta_rgb):
    dh = (delta_rgb[:, :, 1:, :] - delta_rgb[:, :, :-1, :]).abs()
    dw = (delta_rgb[:, :, :, 1:] - delta_rgb[:, :, :, :-1]).abs()
    return dh.mean() + dw.mean()
```

Add `tv_penalty = 1e-3 * self._tv_penalty(delta_rgb)` inside the loop and include it in `total_loss`.

### 3. LPIPS Cap and Early Stop
Before continuing the loop, insert:
```python
lpips_cap = stage_params.get("lpips_cap", 0.12)
if loss_lpips.item() > lpips_cap:
    if self.verbose:
        print(f"    DEBUG: LPIPS cap {lpips_cap} reached; stopping iteration.")
    break
```

### 4. Stats Penalty w/ Mask
When calling the stats matcher:
```python
if self.stats_matcher:
    stats_penalty_raw, stats_detail = self.stats_matcher(x_adv * quality_mask)
    stats_penalty = stats_penalty_raw
```

### 5. Baseline Logic (already present)
We already capture `stats_baseline`, compute `target_stats = max(stats_baseline * reduction, stats_floor)`, and accept successes once stats fall below this target. Keep that logic; just ensure the mask/TV/lpips-cap modifications don’t conflict.

### 6. Recompile/Test
After patching:
```bash
python3 -m compileall ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_full.py
```
Then rerun the Balanced preset and verify LPIPS stays under 0.12 while stats converge.

## Notes
- The mask ensures high-variance regions (faces, highlights) get lighter penalties, so the optimizer mostly tweaks flat/shadow areas.
- LPIPS cap prevents runaway contrast; TV penalty smooths the perturbation.
- Once the guardrails are in, we can proceed to add the ISP LUT/noise stages to get the full “iPhone alibi.”

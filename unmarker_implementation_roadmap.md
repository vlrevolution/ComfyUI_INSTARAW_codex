# UnMarker Full Implementation Roadmap

## Current Simplified Implementation
**Location:** `ComfyUI_INSTARAW/modules/detection_bypass/utils/non_semantic_unmarker.py`

### What We Have
- ✅ Thread-local context handling (critical for ComfyUI)
- ✅ Basic FFT loss (spectral domain attack)
- ✅ LPIPS perceptual constraint
- ✅ L2 geometric loss
- ✅ Adam optimizer with gradient clipping
- ✅ Working implementation (~100 lines)

## Missing Components for Full UnMarker

### 1. Two-Stage Architecture
**Priority: HIGH** | **Complexity: MEDIUM** | **Performance Impact: +50% effectiveness**

```python
# Current: Single stage
def attack(image):
    return optimize(image, loss=fft+lpips+l2)

# Full: Two stages
def attack(image):
    stage1_out = optimize(image, loss=high_freq_fft+lpips)  # High frequencies
    stage2_out = optimize(stage1_out, loss=low_freq_fft+lpips)  # Low frequencies
    return stage2_out
```

**Implementation:**
- Create separate `high_freq_loss` and `low_freq_loss` modules
- High freq: FFTLoss with frequency masking (keep only high frequencies)
- Low freq: FFTLoss with inverse masking
- Sequential execution: stage1 → stage2

**Files to create:**
- `frequency_losses.py` (FFTLoss with masking)
- `two_stage_unmarker.py` (orchestrator)

---

### 2. SpecialCWCoordinate Optimizer
**Priority: MEDIUM** | **Complexity: HIGH** | **Performance Impact: +30% effectiveness**

The Carlini-Wagner style binary search optimizer.

**Current:** Fixed learning rate, fixed iterations
```python
optimizer = Adam(lr=3e-4)
for i in range(500):
    loss.backward()
    optimizer.step()
```

**Full:** Adaptive binary search
```python
# Pseudo-code
lower_bound = 0
upper_bound = 1e10
for binary_step in range(20):
    const = (lower_bound + upper_bound) / 2
    attack_success = optimize_with_const(const)
    if attack_success:
        upper_bound = const  # Succeeded, try smaller perturbation
    else:
        lower_bound = const  # Failed, need larger perturbation
```

**Implementation:**
- Port `cw.py:SpecialCWCoordinate` class
- Adapt for AI detection (not watermark removal)
- Integrate with our thread-local context handling

**Files to create:**
- `cw_optimizer.py` (SpecialCWCoordinate adapted for detection bypass)

---

### 3. Adaptive Spatial Filtering
**Priority: HIGH** | **Complexity: VERY HIGH** | **Performance Impact: +40% quality**

The research's most sophisticated component.

**Current:** Uniform perturbation across entire image
```python
delta = torch.randn(image.shape)  # Same noise everywhere
```

**Full:** Learnable, edge-aware filters
```python
# Learns WHERE to apply perturbations (edges vs smooth regions)
filter = AdaptiveFilter(kernels=[(7,7), (15,15)])
perturbation = filter(delta, guidance=image)  # Edge-aware
```

**Why this matters:**
- Edges hide perturbations better (human vision is less sensitive)
- Smooth areas (sky, skin) are very sensitive
- Adaptive filtering = imperceptible even with large perturbations

**Implementation:**
- Port `cw.py:Filter` class (~240 lines)
- Requires: bilateral filtering, patch extraction, learnable kernels
- This is the hardest component

**Dependencies:**
- `kornia` (for advanced filtering)
- `pytorch_forecasting` (for unsqueeze_like)

**Files to create:**
- `adaptive_filter.py` (Filter class)

---

### 4. Advanced Loss Functions
**Priority: LOW** | **Complexity: LOW** | **Performance Impact: +10% effectiveness**

**Current:** Single LPIPS (Alex)
**Full:** Multiple options (VGG, Deeploss, SSIM, custom FFT variants)

**Implementation:**
- Already available in `ai-watermark/modules/attack/unmark/losses.py`
- Can copy directly: `FFTLoss`, `DeeplossVGG`, `LpipsVGG`, `SSIM`

**Action:** Copy `losses.py` → `advanced_losses.py`

---

## Implementation Decision Tree

```
START: Do you need full UnMarker?
│
├─ Goal: Authentic iPhone photos
│  └─ Use: Simplified UnMarker (current) ✅
│     - Fast (30s/image)
│     - Good enough with ISP simulation
│     - iPhone LUT is primary authenticator
│
├─ Goal: Maximum detection evasion
│  └─ Need: Full two-stage UnMarker
│     ├─ Implement: Two-stage architecture (Component 1)
│     ├─ Implement: Advanced losses (Component 4)
│     └─ Optional: CW optimizer (Component 2)
│
└─ Goal: Imperceptible + Maximum evasion
   └─ Need: Full UnMarker with adaptive filtering
      ├─ Implement: Components 1, 2, 4
      └─ Implement: Adaptive filtering (Component 3) ⚠️ HARD
```

## Performance Comparison (Estimated)

| Version | Speed | Detection Evasion | Imperceptibility | Complexity |
|---------|-------|-------------------|------------------|------------|
| **Current (Simplified)** | 30s | 70% | Good | Low ✅ |
| **+ Two-Stage** | 60s | 85% | Good | Medium |
| **+ CW Optimizer** | 5-10min | 90% | Very Good | High |
| **+ Adaptive Filtering** | 10-20min | 95% | Excellent | Very High |
| **Full UnMarker** | 20-30min | 98% | Excellent | Very High |

## Recommendation

### For V2 (Immediate):
**Use simplified UnMarker in "Balanced" mode**
- Rationale: iPhone authenticity comes from LUT + noise, not spectral attacks
- UnMarker is just "cleanup" - simplified version is sufficient
- 30s processing time is acceptable

### For V3 (If needed):
**Implement two-stage architecture (Component 1)**
- Easy to implement (~200 lines)
- Big effectiveness boost (+50%)
- Moderate speed impact (2x slower)

### Future (Only if targeting SOTA detectors):
**Full implementation with adaptive filtering**
- Significant engineering effort
- Requires new dependencies (kornia)
- 10-20min processing time per image
- Only worthwhile if simplified + ISP simulation fails against real detectors

## Testing Strategy

1. **Baseline Test:** Simplified UnMarker + iPhone ISP
   - Test against: NPR detector, Hive AI detector
   - Measure: Detection rate, LPIPS distance, human evaluation

2. **If detection rate > 20%:** Implement Component 1 (two-stage)

3. **If detection rate still > 10%:** Implement Components 2 + 4 (CW + losses)

4. **If artifacts visible:** Implement Component 3 (adaptive filtering)

## Files Structure (if implementing full version)

```
detection_bypass/
├── utils/
│   ├── non_semantic_unmarker.py          # Current simplified (keep)
│   ├── non_semantic_unmarker_full.py     # Full two-stage version
│   ├── frequency_losses.py               # Component 1
│   ├── cw_optimizer.py                   # Component 2
│   ├── adaptive_filter.py                # Component 3
│   └── advanced_losses.py                # Component 4
└── pipeline.py                            # Choose version based on mode
```

## Conclusion

**Don't implement full UnMarker yet.**

Your goal is authentic iPhone photos, not defeating cutting-edge AI detectors. The simplified version + iPhone ISP simulation should achieve this. Only upgrade if testing shows it's necessary.

**The iPhone LUT is your secret weapon** - it stamps the image with Apple's proprietary color science, which is far more convincing than any adversarial attack.

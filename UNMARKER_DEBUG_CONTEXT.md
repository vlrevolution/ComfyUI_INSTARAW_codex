# UnMarker Attack Debug Context

## 🎯 Project Goal
**Building a better AI image detector** by understanding attack methods (defensive security research). We need UnMarker to actually work so we can study what spectral patterns it modifies, then build detectors that look for those attack signatures.

## 📚 Background: What is UnMarker?

UnMarker is a **two-stage adversarial attack** from the research paper that removes AI fingerprints from generated images by attacking spectral (frequency domain) patterns.

**How it works:**
1. **Stage 1 (high_freq)**: Attacks high-frequency components where AI fingerprints hide
2. **Stage 2 (low_freq)**: Attacks low-frequency components for robustness
3. Uses **LPIPS constraint** to stay perceptually imperceptible
4. Uses **binary search** to find minimal perturbation that achieves bypass
5. Uses **achromatic perturbation** (grayscale delta broadcast to RGB) to prevent color shift

**Original research:** `ai-watermark` repository

## 🔴 THE PROBLEM

UnMarker is **NOT working** - it's failing to bypass AI detection.

### Symptoms:

**Initial Issue (BEFORE our fixes):**
```
Iter 0:   FFT=-0.0089, LPIPS=0.3642, L2=16.6272
Iter 100: FFT=-0.0001, LPIPS=0.0000, L2=0.1055  ← Collapsed to zero
Iter 499: FFT=-0.0000, LPIPS=0.0000, L2=0.0182

Colors: 145.5 → 145.0 (0.5 pixel change = nothing!)
Detector: 99% AI confidence (FAILED)
```

**Latest Issue (AFTER our fixes):**
```
Stage 1: All 5 binary search attempts FAILED (LPIPS 0.0400-0.0469, needs <0.04)
  - Binary step 2 went to NaN (numerical explosion!)

Stage 2: "Succeeded" but useless
  - LPIPS=0.0020 ✓ (below threshold)
  - FFT=-0.0003 ✗ (barely attacking)
  - Colors: 145.5 → 145.1 (0.4 pixel change)

Detector: 99% AI confidence (STILL FAILED)
```

## 🔧 What We've Fixed So Far

Based on deep research of the original `ai-watermark` codebase, we found the implementation was using **completely wrong hyperparameters**:

### Applied Fixes:

1. ✅ **L2 Normalization Bug**
   - Changed from `torch.linalg.norm(delta)` (total norm = 16.6)
   - To `delta.pow(2).mean().sqrt()` (RMS per-pixel)
   - Fixed loss balancing issue

2. ✅ **Gradient Clipping**
   - Changed from `0.05` to `0.005` (research value)
   - Prevents NaN explosions

3. ✅ **FFT Normalization**
   - Changed Stage 1 `use_tanh=True` to `False`
   - Matches research config

4. ✅ **Delta Initialization**
   - Changed from `0.01` to `0.001`
   - More stable, matches research

5. ✅ **Iterations Increased**
   - Balanced: 500→2000 (stage1), 300→800 (stage2)
   - Quality: 800→5000 (stage1), 500→2000 (stage2)
   - Research uses 2000-5000 iterations minimum

6. ✅ **Binary Search Bounds**
   - Changed upper_bound from `1.0` to `100.0`
   - Research uses `1e10`, this is safer middle ground

7. ✅ **Adaptive Filter**
   - Enabled for quality preset
   - Added all 9 kernel configs from research

8. ✅ **Achromatic Delta**
   - Already implemented (single-channel broadcast to RGB)
   - Prevents blue color shift issue

### Files Modified:
- `/home/dreamer/code/combined_test/ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_full.py`
- `/home/dreamer/code/combined_test/ComfyUI_INSTARAW/modules/detection_bypass/utils/non_semantic_unmarker.py`
- `/home/dreamer/code/combined_test/ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_losses.py`
- `/home/dreamer/code/combined_test/ComfyUI_INSTARAW/modules/detection_bypass/pipeline.py`

## ❌ What's STILL Not Working

After applying all the above fixes, user reports it might still be failing (waiting for test results).

### Possible Remaining Issues:

#### 1. **Image Size Mismatch** (MOST LIKELY!)
The research uses **completely different thresholds** for different image sizes:

```yaml
# For 256×256 images:
t_lpips: 4e-2  (0.04)
initial_const: 1e-2
max_iterations: 2000

# For >256px images (512, 768, 1024, etc):
t_lpips: 1e-4  (0.0001) ← 400x MORE STRICT!
initial_const: 1e6  ← 100,000,000x DIFFERENT!
max_iterations: 5000
```

**Current implementation uses ONE-SIZE-FITS-ALL** (0.04 threshold), which is WRONG for images >256px!

#### 2. **Wrong Loss Function**
- We're using: `LPIPS (alex/vgg)` from lpips library
- Research uses: `DeeplossVGG` from custom loss_provider
- These have **different gradient characteristics** and thresholds

#### 3. **Binary Search Strategy**
- Current: Starts at const=0.5 (midpoint of 0.0-1.0 range)
- Research: Starts at const=1e6 or 1e-2 depending on image size
- This massively affects optimization trajectory

#### 4. **Missing Preprocessing**
- Research crops images to 90% before attack
- We're attacking full image
- Might affect boundary effects

## 🔍 Key Research Findings

### From `ai-watermark/modules/attack/config.yaml`:

**For image_size == 256:**
```yaml
loss_fn: DeeplossVGG
loss_thresh: 4.0e-2
binary_search_steps: 2
max_iterations: 2000
initial_const: 1.0e-2
regularization:
  thresh: 3.0e-5
  factor: 0.6
max_grad_l_inf: 0.005
learning_rate: 0.0002
```

**For image_size > 256:**
```yaml
loss_fn: DeeplossVGG
loss_thresh: 1.0e-4  ← CRITICAL DIFFERENCE
binary_search_steps: 4
max_iterations: 5000
initial_const: 1.0e6  ← CRITICAL DIFFERENCE
regularization:
  thresh: 1.0e-4
max_grad_l_inf: 0.005
```

### From `ai-watermark/modules/attack/unmark/cw.py`:

Binary search initialization:
```python
self.lower_bound = torch.zeros((len(x))).float().to(x.device)
self.upper_bound = torch.ones((len(x))).float().to(x.device) * 1e10  # Not 1.0!
self.const = x.new_ones(len(x)) * initial_const  # 1e6 or 1e-2
```

## 🚨 Debug Test Results Needed

User is running test now. Looking for:

1. **Console output** - Full debug log with:
   - Image dimensions (H×W)
   - Binary search results per stage
   - Final color changes
   - Detector confidence

2. **Specific checks:**
   - Are losses still collapsing to zero?
   - Is Stage 1 passing any binary steps?
   - Is image actually changing? (not 145→145.1, need 145→135 range)
   - Any NaN explosions?

3. **Test configuration:**
   - Mode: Balanced
   - UnMarker Version: full_balanced
   - Debug Mode: TRUE

## 📝 Next Steps to Try

### If Still Failing:

#### Priority 1: Implement Image-Size-Dependent Config
```python
def get_config_for_image_size(height, width):
    size = max(height, width)

    if size <= 256:
        return {
            't_lpips': 5e-4,  # Very loose
            'initial_const': 1e-2,
            'max_iterations': 2000,
        }
    elif size == 256:
        return {
            't_lpips': 4e-2,
            'initial_const': 1e-2,
            'max_iterations': 2000,
        }
    else:  # > 256
        return {
            't_lpips': 1e-4,  # Very strict
            'initial_const': 1e6,  # HUGE!
            'max_iterations': 5000,
        }
```

#### Priority 2: Check Loss Function
- Verify LPIPS values are sensible (0.04 threshold might be too strict for our LPIPS vs DeeplossVGG)
- Try relaxing threshold to 0.08 or 0.1 temporarily to see if attack works at all

#### Priority 3: Debug Binary Search
- Add logging for const values at each step
- Check if const is growing properly (should go 0.5 → 0.75 → 0.875 → ...)
- Verify optimizer is actually updating delta

#### Priority 4: Try Disabling LPIPS Constraint Temporarily
```python
# For debugging only - remove perceptual constraint to see if FFT attack works
lpips_penalty = 0  # Instead of self.c_lpips * torch.relu(...)
```

This will make it work but look terrible - but confirms if the issue is LPIPS threshold or FFT loss itself.

## 📊 Expected Behavior When Working

**Stage 1 should show:**
```
Binary step 1: LPIPS=0.038, ✓ SUCCESS
Binary step 2: LPIPS=0.032, ✓ SUCCESS
...
Colors: 145.5 → 138.2 (visible change!)
```

**Final result:**
```
Detector confidence: 45% AI (down from 99%)
```

## 🔗 Key File Paths

- **Main implementation:** `ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_full.py`
- **Simplified version:** `ComfyUI_INSTARAW/modules/detection_bypass/utils/non_semantic_unmarker.py`
- **Loss functions:** `ComfyUI_INSTARAW/modules/detection_bypass/utils/unmarker_losses.py`
- **Pipeline integration:** `ComfyUI_INSTARAW/modules/detection_bypass/pipeline.py`
- **Research reference:** `ai-watermark/modules/attack/config.yaml`
- **Binary search reference:** `ai-watermark/modules/attack/unmark/cw.py`

## 💡 Key Insights

1. **UnMarker is HIGHLY sensitive to image size** - one-size-fits-all won't work
2. **The threshold t_lpips=0.04 is ONLY for 256px images** - larger images need 0.0001!
3. **Binary search initial_const varies by 100 MILLION times** (1e-2 vs 1e6) depending on size
4. **LPIPS vs DeeplossVGG are different metrics** with different sensitivities
5. **Gradient clipping must be 0.005** not 0.05 (10x difference)
6. **Need 2000-5000 iterations** not 500

## 🎯 Goal for Next Debug Session

1. Get full console output from latest test
2. Identify exactly where it's failing:
   - Is LPIPS constraint too strict?
   - Is FFT loss not strong enough?
   - Is binary search not exploring properly?
3. Implement image-size-dependent config if needed
4. Get UnMarker actually bypassing detection so we can study the attack signatures

---

**Status:** Waiting for test results after applying all 7 fixes above. If still failing, likely need image-size-dependent config system.

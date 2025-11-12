# 🔵 Debug Mode for Blue Shift Issue

## ✅ What Changed

### 1. Removed awb_ref_image
- No longer needed (wasn't being used)
- Cleaner UI

### 2. Added Debug Mode
- New checkbox: `debug_mode`
- Shows detailed color analysis at each stage

## 🎮 How to Use Debug Mode

### In ComfyUI:

1. **Add the node** as usual: `INSTARAW Detection Bypass V2`

2. **Enable debug mode:**
   - Check the `debug_mode` checkbox ✓

3. **Configure normally:**
   ```
   Mode: Balanced
   UnMarker Version: simplified (for faster testing)
   Debug Mode: TRUE ← Enable this!
   ```

4. **Run the workflow**

5. **Check the CONSOLE** - you'll see detailed output:

```
🔍 DEBUG MODE ENABLED - Will show detailed color analysis

  🔍 [INPUT IMAGE] Color Analysis:
     R: mean=145.3, std=62.1
     G: mean=138.7, std=59.4
     B: mean=132.2, std=61.8
     R/G ratio: 1.048 (balanced ≈ 1.0)
     B/G ratio: 0.953 (balanced ≈ 1.0)

  🔍 [BEFORE UnMarker] Color Analysis:
     R: mean=145.3, std=62.1
     G: mean=138.7, std=59.4
     B: mean=132.2, std=61.8
     R/G ratio: 1.048 (balanced ≈ 1.0)
     B/G ratio: 0.953 (balanced ≈ 1.0)

  [UnMarker processing...]

  🔍 [AFTER UnMarker] Color Analysis:
     R: mean=143.1, std=61.8
     G: mean=137.2, std=58.9
     B: mean=187.4, std=63.2  ← ⚠️ BLUE IS HIGH!
     R/G ratio: 1.043 (balanced ≈ 1.0)
     B/G ratio: 1.366 (balanced ≈ 1.0)  ← ⚠️ BLUE SHIFT!
     ⚠️ BLUE SHIFT DETECTED! B/G ratio too high: 1.366

  🔍 [FINAL OUTPUT] Color Analysis:
     R: mean=143.1, std=61.8
     G: mean=137.2, std=58.9
     B: mean=187.4, std=63.2
     R/G ratio: 1.043 (balanced ≈ 1.0)
     B/G ratio: 1.366 (balanced ≈ 1.0)
     ⚠️ BLUE SHIFT DETECTED! B/G ratio too high: 1.366
```

## 📊 What the Numbers Mean

### Channel Means
- **R, G, B means**: Average pixel value for each channel (0-255)
- **Should be roughly equal** for natural-looking images
- If B > R and B > G by a lot → blue shift!

### Ratios
- **R/G ratio**: Red vs Green balance
  - `≈ 1.0` = balanced
  - `> 1.1` = reddish
  - `< 0.9` = greenish/cyan

- **B/G ratio**: Blue vs Green balance
  - `≈ 1.0` = balanced
  - `> 1.15` = **BLUE SHIFT** ⚠️
  - `< 0.85` = yellow shift

## 🔍 What to Look For

### If Blue Shift Happens at "BEFORE UnMarker"
**Problem:** Input image itself is blue OR tensor conversion is broken
**Action:** Check your input image / check tensor_to_pil function

### If Blue Shift Happens at "AFTER UnMarker"
**Problem:** UnMarker is causing it
**Possible causes:**
1. Achromatic delta not working (code not reloaded?)
2. LPIPS model doing something weird
3. Tensor → numpy → PIL → tensor conversions breaking colors
4. Some other bug in the unmarker code

### If Blue Shift is GRADUAL (small increase each stage)
**Problem:** Optimization is slowly drifting blue
**Action:** Check LPIPS loss, check delta constraints

## 🛠️ Troubleshooting Steps

### Step 1: Test with unmarker_version="none"
```
Mode: Balanced
UnMarker Version: none  ← SKIP UNMARKER
Debug Mode: TRUE
```

**Expected:** All color stats should be IDENTICAL (no blue shift)
**If blue shift still happens:** Problem is NOT unmarker, it's the pipeline/conversions

### Step 2: Test simplified vs full
```
# First test
UnMarker Version: simplified

# Second test
UnMarker Version: full_balanced
```

**Compare:** Does one cause more blue shift than the other?

### Step 3: Check if achromatic delta is actually being used
Look for this in console:
```
✓ Using achromatic (grayscale) perturbations to prevent color shift
```

**If NOT showing:** Code changes not loaded, restart ComfyUI!

### Step 4: Check actual delta values
We can add more debug output to print delta statistics during optimization.

## 🎯 Next Steps After Testing

### Scenario A: Blue shift at "INPUT IMAGE"
→ Problem is BEFORE unmarker (tensor conversions, input image)

### Scenario B: Blue shift appears at "AFTER UnMarker"
→ Problem is IN unmarker code

### Scenario C: No blue shift in debug, but output looks blue
→ Problem is display/color space (unlikely but possible)

## 📝 Report Back

After running with debug mode, send me:
1. The full console output (especially color analysis sections)
2. Which stage shows blue shift first
3. The exact B/G ratios at each stage

This will tell us EXACTLY where the blue is coming from!

## 🔧 Quick Reference

| B/G Ratio | Meaning |
|-----------|---------|
| 0.9 - 1.1 | ✅ Balanced (good) |
| 1.15 - 1.3 | ⚠️ Slight blue shift |
| 1.3 - 1.5 | 🔵 Moderate blue shift (dabadee) |
| > 1.5 | 🔵🔵 Heavy blue shift (full smurf mode) |
| < 0.85 | 🟡 Yellow/warm shift |

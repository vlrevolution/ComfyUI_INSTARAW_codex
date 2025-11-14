# Filename: ComfyUI_INSTARAW/analyze_photos.py
# ---

import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import argparse

FFT_BINS = 256

def radial_profile(mag: np.ndarray, bins: int):
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    radial_mean = tbin / (nr + 1e-9)
    if len(radial_mean) < bins:
        radial_mean = np.pad(radial_mean, (0, bins - len(radial_mean)), 'edge')
    return radial_mean[:bins]

def analyze_image(filepath: str) -> dict | None:
    try:
        with Image.open(filepath) as img:
            img_arr = np.array(img.convert("RGB"))

        img_float = img_arr.astype(np.float32)
        
        # --- NEW: Analyze each channel separately ---
        spectra_channels = []
        for i in range(3): # R, G, B
            channel = img_float[:, :, i]
            fft = np.fft.fftshift(np.fft.fft2(channel))
            mag = np.log1p(np.abs(fft))
            spectrum = radial_profile(mag, bins=FFT_BINS)
            spectrum_smooth = gaussian_filter1d(spectrum, sigma=2)
            spectra_channels.append(spectrum_smooth)
        
        return {
            "spectra_r": spectra_channels[0],
            "spectra_g": spectra_channels[1],
            "spectra_b": spectra_channels[2],
        }
    except Exception as e:
        print(f"Skipping {os.path.basename(filepath)} due to error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compute reference statistics from real photos.")
    parser.add_argument("--image-dir", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()
    
    # ... (rest of main function is mostly the same, just handles new keys) ...
    image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_stats = []
    for f in tqdm(image_files, desc="Analyzing Photos"):
        stats = analyze_image(f)
        if stats: all_stats.append(stats)
    
    if not all_stats: exit("Error: Analysis failed for all images.")

    # Aggregate stats for each channel
    aggregated_spectra_r = np.array([s["spectra_r"] for s in all_stats])
    aggregated_spectra_g = np.array([s["spectra_g"] for s in all_stats])
    aggregated_spectra_b = np.array([s["spectra_b"] for s in all_stats])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez_compressed(
        args.output_path,
        spectra_r=aggregated_spectra_r,
        spectra_g=aggregated_spectra_g,
        spectra_b=aggregated_spectra_b,
    )
    print(f"\n✅ Success! New multi-channel fingerprint saved to: {args.output_path}")

if __name__ == "__main__":
    main()
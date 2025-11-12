# Filename: ComfyUI_INSTARAW/analyze_photos.py
# ---

import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from skimage.feature import graycomatrix, graycoprops
import argparse

# --- CONFIGURATION ---
# These parameters are now defaults but can be overridden by command-line args.
# They MUST match the expectations of the StatsMatcher in stats_utils.py
FFT_BINS = 256
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# --- HELPER FUNCTIONS ---

def radial_profile(mag: np.ndarray, bins: int):
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    rmax = np.sqrt(cx**2 + cy**2)
    if rmax == 0: return np.zeros(bins, dtype=np.float64)
    r /= rmax

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    sums, _ = np.histogram(r, bins=bin_edges, weights=mag)
    counts, _ = np.histogram(r, bins=bin_edges)
    
    radial_mean = np.zeros_like(sums, dtype=np.float64)
    nonzero = counts > 0
    radial_mean[nonzero] = sums[nonzero] / counts[nonzero]
    
    return radial_mean

def analyze_image(filepath: str) -> dict | None:
    """Analyzes a single image and returns its statistical fingerprint."""
    try:
        with Image.open(filepath) as img:
            img_arr = np.array(img.convert("RGB"))

        # 1. Chroma (Color) Analysis
        pixels = img_arr.astype(np.float32).reshape(-1, 3)
        chroma_mean = pixels.mean(axis=0)
        centered = pixels - chroma_mean
        chroma_cov = np.cov(centered, rowvar=False)

        # 2. Spectral (FFT) Analysis
        gray_arr = np.mean(img_arr.astype(np.float32), axis=2)
        fft = np.fft.fftshift(np.fft.fft2(gray_arr))
        mag = np.log1p(np.abs(fft))
        spectrum = radial_profile(mag, bins=FFT_BINS)
        spectrum_smooth = gaussian_filter1d(spectrum, sigma=5)

        # 3. Texture (GLCM) Analysis
        gray_quantized = (gray_arr / 256 * 32).astype(np.uint8)
        glcm = graycomatrix(gray_quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=32, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # The order here must match what StatsMatcher expects if we expand it later.
        # For now, we only use contrast and homogeneity, but calculating all is good practice.
        glcm_features = np.array([contrast, homogeneity]) 

        return {
            "spectrum": spectrum_smooth,
            "chroma_mean": chroma_mean,
            "chroma_cov": chroma_cov.flatten(),
            "glcm": glcm_features
        }

    except Exception as e:
        print(f"Skipping {os.path.basename(filepath)} due to error: {e}")
        return None

# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="Compute reference statistics from real photos.")
    parser.add_argument("--image-dir", required=True, type=str, help="Path to the directory of images to analyze.")
    parser.add_argument("--output-path", required=True, type=str, help="Path to save the final .npz stats file.")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: Provided image directory does not exist: {args.image_dir}")
        exit(1)

    image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Error: No images found in '{args.image_dir}'.")
        exit(1)

    print(f"Found {len(image_files)} images. Starting analysis...")

    all_stats = []
    for f in tqdm(image_files, desc="Analyzing Photos"):
        stats = analyze_image(f)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("Error: Analysis failed for all images.")
        exit(1)
        
    print(f"\nSuccessfully analyzed {len(all_stats)} images.")

    aggregated_spectra = np.array([s["spectrum"] for s in all_stats])
    aggregated_chroma_means = np.array([s["chroma_mean"] for s in all_stats])
    aggregated_chroma_covs = np.array([s["chroma_cov"] for s in all_stats])
    aggregated_glcms = np.array([s["glcm"] for s in all_stats])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.savez_compressed(
        args.output_path,
        spectra=aggregated_spectra,
        chroma_mean=aggregated_chroma_means,
        chroma_cov=aggregated_chroma_covs,
        glcm=aggregated_glcms
    )

    print(f"\n✅ Success! New statistical fingerprint saved to:")
    print(f"   {args.output_path}")
    print("\nRestart ComfyUI for the Spectral Engine to use the new data.")

if __name__ == "__main__":
    main()
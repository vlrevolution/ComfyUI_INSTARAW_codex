# Filename: ComfyUI_INSTARAW/scripts/create_authenticity_profile.py
# (Definitive Version 7.0 - Correct ExifTool Output Handling)

import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import argparse
import json
import io
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import subprocess
import shutil

try:
    import exiftool
    PYEXIFTOOL_AVAILABLE = True
except ImportError:
    PYEXIFTOOL_AVAILABLE = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

FFT_BINS = 256

METADATA_BLACKLIST = [
    'SourceFile', 'File:FileName', 'File:Directory', 'File:FileModifyDate',
    'File:FileAccessDate', 'File:FileInodeChangeDate', 'File:FileSize',
    'File:FilePermissions', 'ExifTool:ExifToolVersion',
]

def get_metadata_with_exiftool(filepath: str) -> dict:
    """
    Extracts metadata, preserving group prefixes, and filters out
    blacklisted and ICC profile tags (which are handled separately).
    """
    clean_metadata = {}
    try:
        with exiftool.ExifToolHelper() as et:
            raw_meta_list = et.get_metadata(filepath)
            if not raw_meta_list:
                return {}
            
            raw_metadata = raw_meta_list[0]
            for key, value in raw_metadata.items():
                if key not in METADATA_BLACKLIST and not key.startswith('ICC_Profile:'):
                    clean_metadata[key] = value
    except Exception as e:
        tqdm.write(f"  - WARNING: pyexiftool failed for '{os.path.basename(filepath)}'. Error: {e}")
    return clean_metadata

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

def analyze_image(filepath: str, analysis_resolution: int) -> dict | None:
    try:
        metadata = get_metadata_with_exiftool(filepath)
        
        with Image.open(filepath) as img:
            img_for_spectral = img.copy()
            if analysis_resolution > 0:
                img_for_spectral.thumbnail((analysis_resolution, analysis_resolution), Image.Resampling.LANCZOS)
            
            img_arr_resized = np.array(img_for_spectral.convert("RGB"))
            img_float_resized = img_arr_resized.astype(np.float32)
            
            spectra_channels = []
            for i in range(3):
                channel = img_float_resized[:, :, i]
                fft = np.fft.fftshift(np.fft.fft2(channel))
                mag = np.log1p(np.abs(fft))
                spectrum = radial_profile(mag, bins=FFT_BINS)
                spectra_channels.append(gaussian_filter1d(spectrum, sigma=2))

            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            
            with Image.open(buffer) as jpg_img:
                gray_u8 = np.array(jpg_img.convert("L"))
                glcm = graycomatrix(gray_u8, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                
                n_points, radius = 24, 3
                lbp = local_binary_pattern(gray_u8, n_points, radius, method='uniform')
                n_bins = int(lbp.max() + 1)
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

            return {
                "stats": {
                    "spectra_r": spectra_channels[0],
                    "spectra_g": spectra_channels[1],
                    "spectra_b": spectra_channels[2],
                    "glcm_props": np.array([contrast, homogeneity]),
                    "lbp_hist": lbp_hist,
                },
                "metadata": metadata
            }
    except Exception as e:
        tqdm.write(f"Skipping {os.path.basename(filepath)} due to error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create a unified authenticity profile from a directory of real photos.")
    parser.add_argument("--image-dir", required=True, type=str, help="Directory containing photos to analyze.")
    parser.add_argument("--output-path", required=True, type=str, help="Base path for profile files (e.g., 'profiles/iPhone15_Pro').")
    parser.add_argument("--analysis-resolution", type=int, default=1920, help="Resize images to this max dimension for spectral analysis. 0 for full-res (slow).")
    args = parser.parse_args()

    if not PYEXIFTOOL_AVAILABLE:
        exit("ERROR: The 'pyexiftool' library is required. Please run 'pip install pyexiftool'.")
    
    if args.analysis_resolution > 0: print(f"⚡ Performance Mode: Analyzing spectra at max {args.analysis_resolution}px resolution.")
    else: print("⚠️ Quality Mode: Analyzing spectra at full resolution. This will be very slow.")
    
    supported_exts = ['.jpg', '.jpeg', '.png', '.webp']
    if HEIF_SUPPORT: 
        supported_exts.extend(['.heic', '.heif'])
        print("✅ HEIC/HEIF support is enabled.")
    else:
        print("⚠️ HEIC/HEIF support is disabled. Install 'pillow-heif' to enable.")

    image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if not f.startswith('.') and os.path.splitext(f)[1].lower() in supported_exts]
    
    if not image_files:
        exit(f"Error: No supported images found in {args.image_dir}.")

    # --- DEFINITIVE ICC PROFILE EXTRACTION LOGIC ---
    icc_filename_to_embed = None
    icc_output_path = f"{args.output_path}.icc"
    exiftool_path = shutil.which("exiftool") or shutil.which("exiftool.exe")
    
    if exiftool_path:
        print("\nSearching for an ICC Color Profile in the image set...")
        
        for image_to_check in image_files:
            try:
                # Command to print the binary ICC profile to standard output
                command = [exiftool_path, "-icc_profile", "-b", image_to_check]
                
                # Run the command and capture the output
                result = subprocess.run(command, capture_output=True, check=True)
                
                # If stdout has content, we found a profile
                if result.stdout:
                    os.makedirs(os.path.dirname(icc_output_path), exist_ok=True)
                    # Write the captured binary data to our target file
                    with open(icc_output_path, 'wb') as f:
                        f.write(result.stdout)
                    
                    icc_filename_to_embed = os.path.basename(icc_output_path)
                    print(f"✅ Successfully extracted ICC Profile from '{os.path.basename(image_to_check)}'.")
                    print(f"   Saved to: {icc_output_path}")
                    break # Stop searching once we find one
            except (subprocess.CalledProcessError, Exception):
                # This file didn't have a profile or failed, just try the next one silently
                continue
        
        if not icc_filename_to_embed:
            print("   - No ICC Profile found in any of the analyzed images. Skipping.")
            
    else:
        print("\n⚠️ WARNING: 'exiftool' command not found. Cannot extract ICC Profile.")
    # --- END DEFINITIVE LOGIC ---

    all_results = [res for f in tqdm(image_files, desc=f"Analyzing Photos for '{os.path.basename(args.output_path)}'") if (res := analyze_image(f, args.analysis_resolution))]
    if not all_results: exit(f"Error: Analysis failed for all supported images in {args.image_dir}.")

    all_stats = [r['stats'] for r in all_results]
    aggregated_stats = {
        "spectra_r": np.array([s["spectra_r"] for s in all_stats]),
        "spectra_g": np.array([s["spectra_g"] for s in all_stats]),
        "spectra_b": np.array([s["spectra_b"] for s in all_stats]),
        "glcm_props": np.array([s["glcm_props"] for s in all_stats]),
    }
    max_lbp_len = max(len(s["lbp_hist"]) for s in all_stats)
    padded_lbp_hists = [np.pad(s["lbp_hist"], (0, max_lbp_len - len(s["lbp_hist"])), 'constant') for s in all_stats]
    aggregated_stats["lbp_hist"] = np.array(padded_lbp_hists)
    
    npz_path = f"{args.output_path}.npz"
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **aggregated_stats)

    metadata_library = []
    for r in all_results:
        if r['metadata']:
            meta_dict = r['metadata']
            if icc_filename_to_embed:
                meta_dict["_instaraw_icc_profile_file"] = icc_filename_to_embed
            metadata_library.append(meta_dict)

    json_path = f"{args.output_path}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_library, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"✅ Success! Profile created from {len(all_results)} images.")
    print(f"   - Statistical Data (.npz): {npz_path}")
    print(f"   - Metadata Library (.json): {json_path}")
    if icc_filename_to_embed:
        print(f"   - ICC Color Profile (.icc): {icc_output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
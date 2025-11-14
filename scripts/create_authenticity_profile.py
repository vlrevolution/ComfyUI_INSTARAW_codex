# Filename: ComfyUI_INSTARAW/scripts/create_authenticity_profile.py
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import argparse
import json
import io
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import piexif

FFT_BINS = 512

def clean_and_translate_exif(raw_exif_bytes):
    """
    Loads raw EXIF bytes, translates integer tags to human-readable names,
    and cleans values for JSON serialization.
    """
    if not raw_exif_bytes:
        return {}

    try:
        exif_data = piexif.load(raw_exif_bytes)
    except Exception as e:
        print(f"  - Warning: piexif failed to load EXIF data: {e}")
        return {}
        
    translated_data = {}

    for ifd_name in exif_data:
        if ifd_name == "thumbnail":
            continue
        
        translated_ifd = {}
        # piexif.TAGS is the master dictionary for all IFD sections
        tag_map = piexif.TAGS.get(ifd_name, {})

        for tag_id, value in exif_data[ifd_name].items():
            # Get the human-readable tag name
            tag_info = tag_map.get(tag_id)
            tag_name = tag_info['name'] if tag_info else f"UnknownTag_{tag_id}"

            # Clean the value for JSON
            if isinstance(value, bytes):
                try:
                    cleaned_value = value.decode('utf-8', errors='ignore').strip('\x00').strip()
                except:
                    cleaned_value = repr(value)
            elif isinstance(value, tuple) and len(value) > 0 and all(isinstance(v, int) for v in value):
                 # Handle rational numbers like (1, 125) -> "1/125"
                 if len(value) == 2 and value[1] != 0:
                     cleaned_value = f"{value[0]}/{value[1]}"
                 else:
                     cleaned_value = list(value)
            elif isinstance(value, (int, float, str, bool, list)) or value is None:
                cleaned_value = value
            else:
                cleaned_value = repr(value)

            translated_ifd[tag_name] = cleaned_value
        
        translated_data[ifd_name] = translated_ifd

    return translated_data

def radial_profile(mag: np.ndarray, bins: int):
    """Computes a robust radial profile of a 2D array."""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    max_r = min(r.max(), bins - 1)
    valid_mask = r.ravel() <= max_r
    tbin = np.bincount(r.ravel()[valid_mask], mag.ravel()[valid_mask], minlength=bins)
    nr = np.bincount(r.ravel()[valid_mask], minlength=bins)
    radial_mean = np.zeros(bins, dtype=np.float64)
    valid_bins = nr > 0
    radial_mean[valid_bins] = tbin[valid_bins] / nr[valid_bins]
    last_valid_val = 0
    for i in range(bins):
        if radial_mean[i] > 0:
            last_valid_val = radial_mean[i]
        elif last_valid_val > 0:
            radial_mean[i] = last_valid_val
    return radial_mean

def analyze_image(filepath: str) -> dict | None:
    """Analyzes a single image (JPG, PNG) and extracts stats and metadata."""
    try:
        with Image.open(filepath) as img:
            # --- 1. Metadata Extraction ---
            # Get the raw EXIF data block before any conversions
            raw_exif_bytes = img.info.get('exif')
            exif_data = clean_and_translate_exif(raw_exif_bytes)
            
            # --- 2. Statistical Analysis (on a JPG version) ---
            # Convert to RGB and save to an in-memory JPG buffer to simulate the final format
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=95, exif=raw_exif_bytes or b"")
            buffer.seek(0)
            
            with Image.open(buffer) as jpg_img:
                jpg_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                img_arr = np.array(jpg_img)
                img_float = img_arr.astype(np.float32)
                
                spectra_channels = []
                for i in range(3):
                    channel = img_float[:, :, i]
                    fft = np.fft.fftshift(np.fft.fft2(channel))
                    mag = np.log1p(np.abs(fft))
                    spectrum = radial_profile(mag, bins=FFT_BINS)
                    spectra_channels.append(gaussian_filter1d(spectrum, sigma=2))
                
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
                "metadata": exif_data
            }
    except Exception as e:
        print(f"Skipping {os.path.basename(filepath)} due to error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create a comprehensive authenticity profile from a directory of real photos.")
    parser.add_argument("--image-dir", required=True, type=str, help="Directory containing photos to analyze (JPG, PNG, HEIC, etc.).")
    parser.add_argument("--output-path", required=True, type=str, help="Base path for profile files (e.g., 'profiles/iPhone14').")
    args = parser.parse_args()
    
    # We no longer need the pre-conversion step, so we read from the source directory
    image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if not f.startswith('.')]
    
    all_results = [res for f in tqdm(image_files, desc=f"Analyzing Photos for '{os.path.basename(args.output_path)}'") if (res := analyze_image(f))]
    
    if not all_results:
        exit(f"Error: Analysis failed for all images in {args.image_dir}.")

    # Aggregate and Save Statistical Data (.npz)
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

    # Aggregate and Save Metadata Library (.json)
    metadata_library = [r['metadata'] for r in all_results]
    json_path = f"{args.output_path}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_library, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"✅ Success! Profile created with {len(all_results)} images.")
    print(f"   - Statistical data saved to: {npz_path}")
    print(f"   - Metadata library saved to: {json_path}")
    print("="*50)

if __name__ == "__main__":
    main()
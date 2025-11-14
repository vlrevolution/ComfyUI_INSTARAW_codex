# Filename: ComfyUI_INSTARAW/modules/detection_bypass/utils/direct_spectral_matching.py
# ---

import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image

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

def direct_spectral_match(
    img_arr: np.ndarray,
    target_spectrum_luminance: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Performs a color-safe spectral match.
    
    This algorithm works by:
    1. Calculating a spectral correction map based on the luminance of the source
       and target spectra.
    2. Applying this SAME correction map to the magnitude of each R, G, and B
       channel individually, while preserving each channel's original phase.
    
    This preserves the inter-channel relationships and prevents color shifts.
    """
    img_float = img_arr.astype(np.float32)
    h, w, c = img_float.shape
    
    if c != 3:
        raise ValueError("Input image must be RGB.")
        
    nbins = len(target_spectrum_luminance)

    # --- 1. Calculate Luminance-based Correction Map ---
    source_gray = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
    
    fft_gray = np.fft.fftshift(np.fft.fft2(source_gray))
    mag_gray = np.abs(fft_gray)
    
    source_profile = radial_profile(np.log1p(mag_gray), bins=nbins)
    source_profile_smooth = gaussian_filter1d(source_profile, sigma=2)

    eps = 1e-8
    log_multiplier_1d = (target_spectrum_luminance + eps) / (source_profile_smooth + eps)
    
    y_coords, x_coords = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    r = np.sqrt((x_coords - center[1])**2 + (y_coords - center[0])**2)
    rmax = np.sqrt(center[1]**2 + center[0]**2)
    if rmax > 0: r /= rmax
    
    bin_centers = np.linspace(0.0, 1.0, nbins)
    log_filter_2d = np.interp(r, bin_centers, log_multiplier_1d)
    
    # --- 2. Apply Correction to Each Channel ---
    output_img = np.zeros_like(img_float)
    for i in range(3): # R, G, B
        channel = img_float[:, :, i]
        
        fft_channel = np.fft.fftshift(np.fft.fft2(channel))
        mag_channel = np.abs(fft_channel)
        phase_channel = np.angle(fft_channel)

        # Apply the luminance-derived filter in the log domain
        filtered_log_mag = np.log1p(mag_channel) * log_filter_2d
        
        # Convert back from log domain
        filtered_mag = np.expm1(filtered_log_mag)
        
        # Reconstruct with original phase
        new_fft_channel = filtered_mag * np.exp(1j * phase_channel)
        
        # Inverse FFT to get the modified channel
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(new_fft_channel)))
        
        # Blend with original channel based on strength
        output_img[:, :, i] = channel * (1 - strength) + img_back * strength

    return np.clip(output_img, 0, 255).astype(np.uint8)
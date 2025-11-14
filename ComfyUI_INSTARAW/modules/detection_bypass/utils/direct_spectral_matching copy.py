# Filename: ComfyUI_INSTARAW/modules/detection_bypass/utils/direct_spectral_matching.py
# ---

import numpy as np
from scipy.ndimage import gaussian_filter1d

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

def direct_spectral_match(
    img_arr: np.ndarray,
    target_spectra: dict, # Expects a dict with 'r', 'g', 'b' keys
    strength: float = 1.0,
) -> np.ndarray:
    img_float = img_arr.astype(np.float32)
    h, w, c = img_float.shape
    
    if c != 3:
        raise ValueError("Input image must be RGB.")
        
    output_img = np.zeros_like(img_float)

    for i, key in enumerate(['r', 'g', 'b']):
        channel = img_float[:, :, i]
        target_spectrum = target_spectra[key]
        nbins = len(target_spectrum)
        
        # --- Analysis per channel ---
        fft_channel = np.fft.fftshift(np.fft.fft2(channel))
        mag_channel = np.abs(fft_channel)
        phase_channel = np.angle(fft_channel) # Preserve original phase

        source_profile = radial_profile(np.log1p(mag_channel), bins=nbins)
        source_profile_smooth = gaussian_filter1d(source_profile, sigma=2)

        # --- Transformation per channel ---
        eps = 1e-8
        # We match log spectrums now, which is more stable
        log_multiplier_1d = (target_spectrum + eps) / (source_profile_smooth + eps)
        
        y, x = np.indices((h, w))
        center = np.array([h // 2, w // 2])
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        rmax = np.sqrt(center[1]**2 + center[0]**2)
        if rmax > 0: r /= rmax
        
        bin_centers = np.linspace(0.0, 1.0, nbins)
        log_filter_2d = np.interp(r, bin_centers, log_multiplier_1d)
        
        # Apply the filter in the log domain
        filtered_log_mag = np.log1p(mag_channel) * log_filter_2d
        
        # Convert back from log and blend
        filtered_mag = np.expm1(filtered_log_mag)
        blended_mag = mag_channel * (1 - strength) + filtered_mag * strength
        
        # --- Reconstruction per channel with ORIGINAL phase ---
        new_fft_channel = blended_mag * np.exp(1j * phase_channel)
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(new_fft_channel)))
        output_img[:, :, i] = img_back

    return np.clip(output_img, 0, 255).astype(np.uint8)
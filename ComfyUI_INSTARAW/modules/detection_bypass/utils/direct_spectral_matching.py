# Filename: ComfyUI_INSTARAW/modules/detection_bypass/utils/direct_spectral_matching.py
# ---

import numpy as np
from scipy.ndimage import gaussian_filter1d

# The radial_profile function remains the same.
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

def _direct_spectral_match_per_channel(img_arr: np.ndarray, target_spectra: dict, strength: float) -> np.ndarray:
    """Internal function for the original per-channel matching algorithm."""
    img_float = img_arr.astype(np.float32)
    h, w, _ = img_float.shape
    output_img = np.zeros_like(img_float)

    for i, key in enumerate(['r', 'g', 'b']):
        channel = img_float[:, :, i]
        target_spectrum = target_spectra[key]
        nbins = len(target_spectrum)
        
        fft_channel = np.fft.fftshift(np.fft.fft2(channel))
        mag_channel = np.abs(fft_channel)
        phase_channel = np.angle(fft_channel)

        source_profile = radial_profile(np.log1p(mag_channel), bins=nbins)
        source_profile_smooth = gaussian_filter1d(source_profile, sigma=2)

        eps = 1e-8
        log_multiplier_1d = (target_spectrum + eps) / (source_profile_smooth + eps)
        
        y, x = np.indices((h, w))
        center = np.array([(h-1)/2.0, (w-1)/2.0])
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        rmax = np.sqrt(center[1]**2 + center[0]**2)
        if rmax > 0: r /= rmax
        
        bin_centers = np.linspace(0.0, 1.0, nbins)
        log_filter_2d = np.interp(r, bin_centers, log_multiplier_1d)
        
        filtered_log_mag = np.log1p(mag_channel) * log_filter_2d
        filtered_mag = np.expm1(filtered_log_mag)
        
        blended_mag = mag_channel * (1 - strength) + filtered_mag * strength
        
        new_fft_channel = blended_mag * np.exp(1j * phase_channel)
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(new_fft_channel)))
        output_img[:, :, i] = img_back

    return np.clip(output_img, 0, 255).astype(np.uint8)

def _direct_spectral_match_color_safe(img_arr: np.ndarray, target_spectrum_luminance: np.ndarray, strength: float) -> np.ndarray:
    """Internal function for the new, color-safe algorithm."""
    img_float = img_arr.astype(np.float32)
    h, w, _ = img_float.shape
    nbins = len(target_spectrum_luminance)

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
    
    output_img = np.zeros_like(img_float)
    for i in range(3):
        channel = img_float[:, :, i]
        fft_channel = np.fft.fftshift(np.fft.fft2(channel))
        mag_channel = np.abs(fft_channel)
        phase_channel = np.angle(fft_channel)

        filtered_log_mag = np.log1p(mag_channel) * log_filter_2d
        filtered_mag = np.expm1(filtered_log_mag)
        
        new_fft_channel = filtered_mag * np.exp(1j * phase_channel)
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(new_fft_channel)))
        
        output_img[:, :, i] = channel * (1 - strength) + img_back * strength

    return np.clip(output_img, 0, 255).astype(np.uint8)

def direct_spectral_match(img_arr, target_spectra, strength, mode="Per-Channel (Legacy)"):
    """
    Smart dispatcher for spectral matching.
    - Defaults to "Per-Channel (Legacy)" to ensure backward compatibility with
      the INSTARAW_SpectralEngine node, which does not pass a 'mode'.
    - Can be explicitly called with "Color-Safe (Luminance)" by the new
      INSTARAW_FFT_Match node.
    """
    if mode == "Color-Safe (Luminance)":
        # The color-safe algorithm needs a single luminance spectrum.
        # If it receives a dict (from a legacy call or profile), average it.
        if isinstance(target_spectra, dict):
            lum_spectrum = (target_spectra['r'] + target_spectra['g'] + target_spectra['b']) / 3.0
        else:
            lum_spectrum = target_spectra
        return _direct_spectral_match_color_safe(img_arr, lum_spectrum, strength)
    
    elif mode == "Per-Channel (Legacy)":
        # The per-channel algorithm requires a dictionary of spectra.
        if not isinstance(target_spectra, dict):
             raise TypeError("Per-Channel mode requires a dictionary of target spectra for 'r', 'g', and 'b'.")
        return _direct_spectral_match_per_channel(img_arr, target_spectra, strength)
        
    else:
        raise ValueError(f"Unknown spectral match mode: {mode}")
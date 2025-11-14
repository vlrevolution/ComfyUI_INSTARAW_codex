# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/fft_match.py
import torch
import numpy as np
from PIL import Image

from ...modules.detection_bypass.utils.direct_spectral_matching import direct_spectral_match, radial_profile

class INSTARAW_FFT_Match:
    """
    Performs a spectral match using either a reference image or an
    Authenticity Profile. Offers two algorithms for comparison.
    """
    
    MODES = ["Color-Safe (Luminance)", "Per-Channel (Legacy)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "profile_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image, mode, strength, ref_image=None, profile_path=None):
        if strength == 0:
            return (image,)

        target_spectra = None
        
        if ref_image is not None:
            print(f"🛡️ INSTARAW FFT Match: Using live reference image with '{mode}' mode.")
            ref_numpy = self._tensor_to_numpy(ref_image[0:1])
            
            if mode == "Color-Safe (Luminance)":
                ref_gray = 0.299 * ref_numpy[:,:,0] + 0.587 * ref_numpy[:,:,1] + 0.114 * ref_numpy[:,:,2]
                fft = np.fft.fftshift(np.fft.fft2(ref_gray))
                mag = np.log1p(np.abs(fft))
                target_spectra = radial_profile(mag, bins=512)
            else: # Per-Channel
                target_spectra = {}
                for i, key in enumerate(['r', 'g', 'b']):
                    fft = np.fft.fftshift(np.fft.fft2(ref_numpy[:,:,i]))
                    mag = np.log1p(np.abs(fft))
                    target_spectra[key] = radial_profile(mag, bins=512)

        elif profile_path and profile_path.strip():
            print(f"🛡️ INSTARAW FFT Match: Using Authenticity Profile with '{mode}' mode.")
            npz_path = f"{profile_path}.npz"
            try:
                stats = np.load(npz_path)
                target_spectra = {
                    'r': stats['spectra_r'].mean(axis=0),
                    'g': stats['spectra_g'].mean(axis=0),
                    'b': stats['spectra_b'].mean(axis=0)
                }
            except Exception as e:
                raise ValueError(f"Failed to load spectral data from profile '{npz_path}': {e}")

        if target_spectra is None:
            print("⚠️ INSTARAW FFT Match: No reference image or profile provided. Passing through.")
            return (image,)

        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            processed_numpy = direct_spectral_match(
                img_arr=img_numpy,
                target_spectra=target_spectra,
                strength=strength,
                mode=mode # Explicitly pass the mode
            )
            processed_images.append(self._numpy_to_tensor(processed_numpy))

        return (torch.cat(processed_images, dim=0),)
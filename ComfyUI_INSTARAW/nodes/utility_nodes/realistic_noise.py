# ---
# Filename: ../ComfyUI_INSTARAW/nodes/utility_nodes/realistic_noise.py
# ---

# ---
# ComfyUI INSTARAW - Realistic Noise Node
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
A node to inject natural-looking camera sensor noise, mimicking modern smartphones.
This is different from cinematic grain; it's softer, color-dependent, and more prominent in shadows.
"""

import torch
import math

try:
    import kornia.filters as kfilters
    import kornia.color as kcolor  # --- THIS IS THE FIX #1: Import the correct color module ---
    KORNIA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    KORNIA_AVAILABLE = False
    print("⚠️ Kornia not available. Realistic Noise node requires Kornia for blurring effects.")

class INSTARAW_RealisticNoise:
    """
    Adds realistic, adjustable camera sensor noise to an image.
    Features separate controls for luma (brightness) and chroma (color) noise,
    blurring for a softer look, and highlights protection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        if not KORNIA_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": "Kornia library not found. Please install with 'pip install kornia' to use this node.",
                        "multiline": True
                    })
                }
            }
        
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "luma_intensity": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Intensity of brightness noise."}),
                "chroma_intensity": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Intensity of color noise."}),
                "luma_blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Softness of the brightness noise. Higher values are more 'blotchy'."}),
                "chroma_blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Softness of the color noise. Usually higher than luma blur."}),
                "highlights_protection": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Protects brighter areas from noise. 1.0 = full protection, 0.0 = no protection."}),
            },
        }

    CATEGORY = "INSTARAW/Utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    DESCRIPTION = """
# INSTARAW Realistic Noise
Adds natural-looking camera sensor noise, perfect for mimicking smartphone photos.
- **Luma/Chroma Intensity**: Control brightness and color noise separately.
- **Blur Sigma**: Soften the noise to avoid a sharp, 'cinematic' look.
- **Highlights Protection**: Keeps bright areas clean, as they are in real photos.
"""

    def add_noise(self, image: torch.Tensor, seed: int, luma_intensity: float, chroma_intensity: float,
                  luma_blur_sigma: float, chroma_blur_sigma: float, highlights_protection: float):
        
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia library is not available. Please install it to use the Realistic Noise node.")

        torch.manual_seed(seed)

        batch_size, height, width, _ = image.shape
        image_bchw = image.permute(0, 3, 1, 2).to(DEVICE)

        total_noise = torch.zeros_like(image_bchw)

        if luma_intensity > 0:
            luma_noise_chw = torch.randn(batch_size, 1, height, width, device=DEVICE).repeat(1, 3, 1, 1)
            
            if luma_blur_sigma > 0:
                kernel_size = 2 * math.ceil(3.0 * luma_blur_sigma) + 1
                luma_noise_chw = kfilters.gaussian_blur2d(
                    luma_noise_chw, (kernel_size, kernel_size), (luma_blur_sigma, luma_blur_sigma)
                )
            
            total_noise += luma_noise_chw * luma_intensity

        if chroma_intensity > 0:
            chroma_noise_chw = torch.randn(batch_size, 3, height, width, device=DEVICE)

            if chroma_blur_sigma > 0:
                kernel_size = 2 * math.ceil(3.0 * chroma_blur_sigma) + 1
                chroma_noise_chw = kfilters.gaussian_blur2d(
                    chroma_noise_chw, (kernel_size, kernel_size), (chroma_blur_sigma, chroma_blur_sigma)
                )

            total_noise += chroma_noise_chw * chroma_intensity

        if highlights_protection > 0:
            # --- THIS IS THE FIX #2: Use the correct module for the function call ---
            luminance = kcolor.rgb_to_grayscale(image_bchw)
            
            protection_mask = 1.0 - torch.clamp((luminance - highlights_protection) / (1.0 - highlights_protection + 1e-6), 0.0, 1.0)
            
            total_noise *= protection_mask

        noisy_image_bchw = torch.clamp(image_bchw + total_noise, 0.0, 1.0)
        
        final_image = noisy_image_bchw.permute(0, 2, 3, 1).to(image.device)

        return (final_image,)

# =================================================================================
# NODE REGISTRATION
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_RealisticNoise": INSTARAW_RealisticNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_RealisticNoise": "✨ INSTARAW Realistic Noise",
}
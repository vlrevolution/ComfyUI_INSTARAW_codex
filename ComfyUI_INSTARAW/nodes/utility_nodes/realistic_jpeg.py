# ---
# Filename: ../ComfyUI_INSTARAW/nodes/utility_nodes/realistic_jpeg.py
# ---

# ---
# ComfyUI INSTARAW - JPEG Degradation Node (V3 - Corrected Scaling)
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
An advanced node for simulating image degradation from compression.
Offers two modes: 'True JPEG' for authentic artifacts and 'Downscale/Upscale' for detail loss.
"""

import torch
import numpy as np
from PIL import Image
import io
import math
import comfy.utils

try:
    import kornia.filters as kfilters
    KORNIA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    KORNIA_AVAILABLE = False

class INSTARAW_JPEG_Degradation:
    """
    Applies realistic image degradation through either true JPEG compression
    or a downscale/upscale cycle, with optional artifact softening.
    """

    MODES = ["True JPEG", "Downscale/Upscale"]
    CHROMA_SUBSAMPLING_MODES = ["Standard (4:2:0 - Blotchy Color)", "High Quality (4:4:4)", "Aggressive (4:1:1)"]
    UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "True JPEG"}),
                
                "quality": ("INT", {"default": 60, "min": 1, "max": 100, "step": 1, "tooltip": "Overall JPEG quality (1=worst, 100=best)."}),
                "chroma_subsampling": (cls.CHROMA_SUBSAMPLING_MODES, {"default": "Standard (4:2:0 - Blotchy Color)"}),
                
                "downscale_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Factor to shrink the image by before upscaling."}),
                "upscale_method": (cls.UPSCALE_METHODS, {"default": "bicubic"}),

                "soften_artifacts": ("BOOLEAN", {"default": False, "label_on": "Soften (Rounder)", "label_off": "Sharp (Blockier)"}),
                "soften_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Amount of blur to apply to soften artifact edges."}),
            },
        }

    CATEGORY = "INSTARAW/Utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "degrade"
    DESCRIPTION = """
# INSTARAW JPEG Degradation
Simulates image quality loss.
- **True JPEG**: Authentic compression artifacts.
- **Downscale/Upscale**: Simulates resolution loss.
"""

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def degrade(self, image: torch.Tensor, mode: str, quality: int, chroma_subsampling: str, 
                downscale_factor: float, upscale_method: str, soften_artifacts: bool, soften_sigma: float):
        
        original_device = image.device
        
        if mode == "True JPEG":
            processed_images = []
            
            subsampling_map = {
                "High Quality (4:4:4)": 0,
                "Standard (4:2:0 - Blotchy Color)": 2,
                "Aggressive (4:1:1)": 1,
            }
            subsampling_val = subsampling_map.get(chroma_subsampling, 2)

            for i in range(image.shape[0]):
                img_pil = self.tensor_to_pil(image[i:i+1])
                buffer = io.BytesIO()
                
                img_pil.save(buffer, format='JPEG', quality=quality, subsampling=subsampling_val)
                
                buffer.seek(0)
                reloaded_pil = Image.open(buffer).convert('RGB')
                processed_images.append(self.pil_to_tensor(reloaded_pil))
            
            final_batch = torch.cat(processed_images, dim=0)

        elif mode == "Downscale/Upscale":
            _b, original_height, original_width, _c = image.shape
            
            target_w = int(original_width * downscale_factor)
            target_h = int(original_height * downscale_factor)
            
            img_bchw = image.permute(0, 3, 1, 2)

            # --- THIS IS THE FIX ---
            # We use `crop="disabled"` to prevent any edge trimming. This guarantees the aspect ratio
            # is perfectly maintained, so the final upscale will match the original size exactly.
            downscaled_bchw = comfy.utils.common_upscale(img_bchw, target_w, target_h, "area", "disabled")
            upscaled_bchw = comfy.utils.common_upscale(downscaled_bchw, original_width, original_height, upscale_method, "disabled")
            # --- END FIX ---
            
            final_batch = upscaled_bchw.permute(0, 2, 3, 1)

        else:
            final_batch = image

        if soften_artifacts and soften_sigma > 0 and KORNIA_AVAILABLE:
            final_batch_bchw = final_batch.permute(0, 3, 1, 2).to(DEVICE)
            kernel_size = 2 * math.ceil(3.0 * soften_sigma) + 1
            blurred_batch = kfilters.gaussian_blur2d(
                final_batch_bchw, (kernel_size, kernel_size), (soften_sigma, soften_sigma)
            )
            final_batch = blurred_batch.permute(0, 2, 3, 1)
        elif soften_artifacts and not KORNIA_AVAILABLE:
            print("⚠️ INSTARAW JPEG Degradation: Soften artifacts is enabled, but Kornia is not installed. Skipping blur.")

        return (final_batch.to(original_device),)

NODE_CLASS_MAPPINGS = {
    "INSTARAW_JPEG_Degradation": INSTARAW_JPEG_Degradation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_JPEG_Degradation": "🗜️ INSTARAW JPEG Degradation",
}
# ---
# ComfyUI INSTARAW - Ensure Image Resolution Node
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
An advanced utility node to ensure an image meets both minimum total pixel and maximum
side length requirements. It will upscale or downscale as needed while preserving aspect ratio.
"""

import torch
import comfy.utils
import math

class INSTARAW_EnsureImageResolution:
    """
    A resolution guard that ensures an image is within specified boundaries.
    - If total pixels are below a minimum, it scales the image up.
    - If the longest side is above a maximum, it scales the image down.
    This replaces complex logic chains with a single, efficient utility.
    """

    INTERPOLATION_METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "area", "nearest-exact"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "min_total_pixels": ("INT", {
                    "default": 921600, "min": 0, "max": 16777216, "step": 1,
                    "tooltip": "Minimum required total pixels (width * height). 921600 is the minimum for SeeDream."
                }),
                "max_side_length": ("INT", {
                    "default": 4096, "min": 64, "max": 16384, "step": 64,
                    "tooltip": "Maximum length for the longest side of the image."
                }),
                "interpolation": (cls.INTERPOLATION_METHODS, {"default": "lanczos"}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }
        }

    CATEGORY = "INSTARAW/Utils"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING",)
    RETURN_NAMES = ("image", "was_changed", "info_text",)
    FUNCTION = "ensure_resolution"

    def ensure_resolution(self, image, enabled, min_total_pixels, max_side_length, interpolation, multiple_of):
        if not enabled:
            _b, h, w, _c = image.shape
            return (image, False, f"Bypassed. Original size: {w}x{h}",)

        _batch, original_height, original_width, _channels = image.shape
        original_pixels = original_width * original_height
        
        current_image = image
        was_upscaled = False
        was_downscaled = False
        info_lines = [f"Original: {original_width}x{original_height} ({original_pixels:,} pixels)"]

        # --- Step 1: Upscale if total pixels are too low ---
        if original_pixels > 0 and original_pixels < min_total_pixels:
            was_upscaled = True
            scale_ratio = math.sqrt(min_total_pixels / original_pixels)
            
            new_width = int(round(original_width * scale_ratio))
            new_height = int(round(original_height * scale_ratio))

            # Ensure new dimensions are a multiple of the specified value
            new_width = (new_width + multiple_of - 1) // multiple_of * multiple_of
            new_height = (new_height + multiple_of - 1) // multiple_of * multiple_of

            info_lines.append(f"Action: Upscaled to meet min pixels. New size: {new_width}x{new_height}.")
            
            image_bchw = current_image.permute(0, 3, 1, 2)
            scaled_image_bchw = comfy.utils.common_upscale(image_bchw, new_width, new_height, interpolation, "center")
            current_image = scaled_image_bchw.permute(0, 2, 3, 1)

        # --- Step 2: Downscale if longest side is too large (check the potentially upscaled image) ---
        _b, current_height, current_width, _c = current_image.shape
        longest_side = max(current_width, current_height)

        if longest_side > max_side_length:
            was_downscaled = True
            scale_ratio = max_side_length / longest_side
            
            new_width = int(round(current_width * scale_ratio))
            new_height = int(round(current_height * scale_ratio))

            # Ensure new dimensions are a multiple of the specified value
            new_width = (new_width // multiple_of) * multiple_of
            new_height = (new_height // multiple_of) * multiple_of

            info_lines.append(f"Action: Downscaled to fit max side length. Final size: {new_width}x{new_height}.")
            
            image_bchw = current_image.permute(0, 3, 1, 2)
            scaled_image_bchw = comfy.utils.common_upscale(image_bchw, new_width, new_height, interpolation, "center")
            current_image = scaled_image_bchw.permute(0, 2, 3, 1)

        was_changed = was_upscaled or was_downscaled
        if not was_changed:
            info_lines.append("Action: No scaling needed.")

        final_info = "\n".join(info_lines)
        return (current_image, was_changed, final_info,)

# =================================================================================
# EXPORT NODE MAPPINGS
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_EnsureImageResolution": INSTARAW_EnsureImageResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_EnsureImageResolution": "📏 INSTARAW Ensure Image Resolution",
}
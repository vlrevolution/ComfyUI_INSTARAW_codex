# ---
# ComfyUI INSTARAW - Image Resize & Fill Node (Definitive Version)
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
#
# This node is inspired by and extends the functionality of rgthree's Image Resize node.
# ---

import torch
import comfy.utils

from nodes import ImageScale, MAX_RESOLUTION

class INSTARAW_ImageResizeFill:
    """
    An advanced image resize node with crop, pad, contain, and a simple, reliable 
    'pad_with_background' fit method to fill empty space with another image.
    """

    TITLE = "📐 INSTARAW Image Resize & Fill"

    FIT_METHODS = ["crop", "pad", "contain", "pad_with_background"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "fit": (cls.FIT_METHODS,),
                "method": (ImageScale.upscale_methods, {"default": "lanczos"}),
            },
            "optional": {
                "background_image": ("IMAGE", {"tooltip": "Image to use for the background when fit is 'pad_with_background'"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT",)
    FUNCTION = "main"
    CATEGORY = "INSTARAW/Utils"

    def main(self, image, width, height, method, fit, background_image=None):
        if not image.numel(): return (image, 0, 0)
        _, H, W, _ = image.shape

        if (width == 0 and height == 0) or (width == W and height == H):
            return (image, W, H)

        if width == 0 or height == 0:
            width = round(height / H * W) if width == 0 else width
            height = round(width / W * H) if height == 0 else height
            if fit not in ["pad_with_background", "contain"]:
                fit = "contain"

        # --- DEFINITIVE 'PAD WITH BACKGROUND' LOGIC ---
        if fit == "pad_with_background":
            if background_image is None:
                raise ValueError("A 'background_image' is required for the 'pad_with_background' fit method.")

            print("✨ Performing 'Pad with Background' resize.")
            fg, bg = image, background_image
            
            # 1. Create the canvas: Resize the background to the final target size.
            canvas = torch.nn.functional.interpolate(bg.permute(0, 3, 1, 2), size=(height, width), mode='bilinear', antialias=True).permute(0, 2, 3, 1)

            # 2. 'Contain' resize the foreground image to fit inside the canvas.
            fg_resized = self.resize_contain(fg, width, height, method)
            fg_h, fg_w = fg_resized.shape[1], fg_resized.shape[2]

            # 3. Calculate the position to paste the foreground.
            y_start = (height - fg_h) // 2
            x_start = (width - fg_w) // 2
            
            # 4. Paste the resized foreground directly onto the canvas.
            canvas[:, y_start:y_start + fg_h, x_start:x_start + fg_w, :] = fg_resized
            
            return (canvas, width, height)

        # --- Standard Resize Logic (rgthree-based) ---
        resized_width, resized_height = self.calculate_resize_dims(W, H, width, height, fit)
        out_image = comfy.utils.common_upscale(image.clone().permute(0, 3, 1, 2), resized_width, resized_height, method, "disabled").permute(0, 2, 3, 1)
        
        if fit != "contain":
            out_image = self.crop_and_pad(out_image, width, height, image.dtype, image.device)

        return (out_image, out_image.shape[2], out_image.shape[1])

    def calculate_resize_dims(self, src_w, src_h, dst_w, dst_h, fit):
        src_aspect = src_w / src_h
        dst_aspect = dst_w / dst_h
        if fit == "crop": return (round(dst_h * src_aspect), dst_h) if src_aspect > dst_aspect else (dst_w, round(dst_w / src_aspect))
        elif fit in ["contain", "pad"]: return (dst_w, round(dst_w / src_aspect)) if src_aspect > dst_aspect else (round(dst_h * src_aspect), dst_h)
        return dst_w, dst_h

    def resize_contain(self, image, width, height, method):
        _, H, W, _ = image.shape
        resized_width, resized_height = self.calculate_resize_dims(W, H, width, height, "contain")
        return comfy.utils.common_upscale(image.clone().permute(0, 3, 1, 2), resized_width, resized_height, method, "disabled").permute(0, 2, 3, 1)

    def crop_and_pad(self, image, width, height, dtype, device):
        # This handles both 'crop' and 'pad' (with black)
        # Crop if oversized
        if image.shape[2] > width:
            crop_x = (image.shape[2] - width) // 2
            image = image[:, :, crop_x:crop_x + width, :]
        if image.shape[1] > height:
            crop_y = (image.shape[1] - height) // 2
            image = image[:, crop_y:crop_y + height, :, :]
        # Pad if undersized
        if image.shape[2] < width or image.shape[1] < height:
            padded_image = torch.zeros((image.shape[0], height, width, image.shape[3]), dtype=dtype, device=device)
            pad_x = (width - image.shape[2]) // 2
            pad_y = (height - image.shape[1]) // 2
            padded_image[:, pad_y:pad_y + image.shape[1], pad_x:pad_x + image.shape[2], :] = image
            image = padded_image
        return image

# =================================================================================
# NODE REGISTRATION
# =================================================================================

NODE_CLASS_MAPPINGS = {"INSTARAW_ImageResizeFill": INSTARAW_ImageResizeFill}
NODE_DISPLAY_NAME_MAPPINGS = {"INSTARAW_ImageResizeFill": "📐 INSTARAW Image Resize & Fill"}
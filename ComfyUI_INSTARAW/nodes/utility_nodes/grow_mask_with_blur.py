# ---
# ComfyUI INSTARAW - Grow Mask With Blur Node
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
Hybrid pixel and percentage-based mask growing/blurring utility with a dedicated text output for clarity.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

# _INSTARA_INJECT_

# Tensor conversion utilities
def tensor2pil(image):
    """Convert tensor (mask) to PIL Image in grayscale mode"""
    np_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(np_image, mode='L')]

def pil2tensor(image):
    """Convert PIL Image to tensor (mask)"""
    # Ensure we're working with grayscale
    if image.mode != 'L':
        image = image.convert('L')
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class INSTARAWGrowMask:
    """
    Hybrid mask utility for growing/shrinking with pixel or percentage-based control.
    Provides precise, predictable results with a dedicated string output for calculated values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_method": (["Pixels", "Percentage"],),
                "expand_pixels": ("INT", {
                    "default": 10, "min": -8192, "max": 8192, "step": 1
                }),
                "expand_percent": ("FLOAT", {
                    "default": 5.0, "min": -100.0, "max": 100.0, "step": 0.01
                }),
                "incremental_expand_rate_px": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001
                }),
                 "incremental_expand_rate_percent": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.001
                }),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius_multiplier": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001
                }),
                "lerp_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "decay_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "INSTARAW/Utils"
    RETURN_TYPES = ("MASK", "MASK", "STRING",)
    RETURN_NAMES = ("mask", "mask_inverted", "info_text",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
# INSTARAW Grow Mask (Hybrid)
- **expand_method**: Choose 'Pixels' for absolute values or 'Percentage' for resolution-independent scaling.
- **expand**: Set the expansion amount in pixels or percent.
- **incremental_expand_rate**: Increase expansion per frame (in pixels or percent).
- **blur_radius_multiplier**: Blur radius as a multiplier of the final expand pixel value.
- **info_text**: A string output showing the calculated pixel values for clarity.
"""

    def expand_mask(self, mask, expand_method, expand_pixels, expand_percent,
                   incremental_expand_rate_px, incremental_expand_rate_percent,
                   tapered_corners, flip_input, blur_radius_multiplier,
                   lerp_alpha, decay_factor, fill_holes=False):

        mask_height, mask_width = mask.shape[-2:]
        reference_dimension = min(mask_height, mask_width)

        if expand_method == "Percentage":
            initial_expand_px = (expand_percent / 100.0) * reference_dimension
            incremental_rate_px = (incremental_expand_rate_percent / 100.0) * reference_dimension
        else:
            initial_expand_px = float(expand_pixels)
            incremental_rate_px = incremental_expand_rate_px

        try:
            import kornia.morphology as morph
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            print("⚠️ Kornia not available for morphology, expansion/shrinking will be skipped.")
            device = torch.device("cpu")
            morph = None

        if flip_input:
            mask = 1.0 - mask

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        previous_output = None
        current_expand = initial_expand_px

        for m in tqdm(growmask, desc="Processing Mask"):
            output = m.unsqueeze(0).unsqueeze(0).to(device)

            if abs(round(current_expand)) > 0 and morph is not None:
                kernel_shape = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=device) if tapered_corners else torch.ones(3, 3, device=device)
                op = morph.dilation if current_expand > 0 else morph.erosion
                for _ in range(abs(round(current_expand))):
                    output = op(output, kernel_shape)
            output = output.squeeze(0).squeeze(0)

            if fill_holes:
                try:
                    import scipy.ndimage
                    output = torch.from_numpy(scipy.ndimage.binary_fill_holes(output.cpu().numpy() > 0).astype(np.float32)).to(device)
                except ImportError:
                    print("⚠️ scipy not available, skipping fill_holes")

            if (lerp_alpha < 1.0 or decay_factor < 1.0) and previous_output is not None:
                if lerp_alpha < 1.0:
                    output = lerp_alpha * output + (1 - lerp_alpha) * previous_output
                if decay_factor < 1.0:
                    output += decay_factor * previous_output
                    max_val = output.max()
                    if max_val > 0:
                        output = output / max_val

            previous_output = output
            out.append(output.cpu())
            current_expand += incremental_rate_px

        final_masks = torch.stack(out, dim=0)
        if blur_radius_multiplier > 0:
            blurred_out = []
            blur_tracker_px = initial_expand_px
            for tensor in final_masks:
                frame_blur_radius = abs(blur_tracker_px) * blur_radius_multiplier
                if frame_blur_radius > 0.01:
                    pil_image = tensor2pil(tensor.detach())[0].filter(ImageFilter.GaussianBlur(frame_blur_radius))
                    blurred_tensor = pil2tensor(pil_image)
                    max_val = blurred_tensor.max()
                    if max_val > 0:
                        blurred_tensor = blurred_tensor / max_val
                    blurred_out.append(blurred_tensor)
                else:
                    blurred_out.append(tensor.unsqueeze(0))
                blur_tracker_px += incremental_rate_px
            final_masks = torch.cat(blurred_out, dim=0)

        inverted_mask = 1.0 - final_masks

        # Prepare the string output for clarity
        start_blur = abs(initial_expand_px) * blur_radius_multiplier
        percent_str = f" ({expand_percent}%)" if expand_method == "Percentage" else ""
        info_text_lines = [
            f"Mode: {expand_method}",
            f"Expand (start): {initial_expand_px:.2f}px{percent_str}",
            f"Blur (start): {start_blur:.2f}px"
        ]
        if incremental_rate_px != 0.0 and len(growmask) > 1:
            final_expand_px = initial_expand_px + incremental_rate_px * (len(growmask) - 1)
            final_blur = abs(final_expand_px) * blur_radius_multiplier
            percent_str_end = f" ({expand_percent + incremental_expand_rate_percent * (len(growmask) - 1):.2f}%)" if expand_method == "Percentage" else ""
            info_text_lines.extend([
                f"Expand (end): {final_expand_px:.2f}px{percent_str_end}",
                f"Blur (end): {final_blur:.2f}px"
            ])
        
        info_text_string = "\n".join(info_text_lines)

        return (final_masks, inverted_mask, info_text_string,)


# =================================================================================
# EXPORT NODE MAPPINGS
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAWGrowMask": INSTARAWGrowMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAWGrowMask": "🎭 INSTARAW Grow Mask",
}
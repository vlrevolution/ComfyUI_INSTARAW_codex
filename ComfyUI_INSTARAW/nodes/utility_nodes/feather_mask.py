# ---
# ComfyUI INSTARAW - Feather Mask Node
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
The definitive, Photoshop-style feathering node. This version uses the correct compositing
method of a solid core over a blurred outer shell to produce a perfect, artifact-free result.
"""
import torch
from tqdm import tqdm
import math

try:
    import kornia.filters as kfilters
    import kornia.morphology as morph
    KORNIA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    KORNIA_AVAILABLE = False
    print("⚠️ Kornia not available. Feather Mask node requires Kornia.")

class INSTARAWFeatherMask:
    """
    Applies a true feathering effect by compositing a solid core with a blurred outer boundary.
    This ensures a 100% solid core with a perfectly seamless gradient falloff.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 75, "min": -8192, "max": 8192, "step": 1}),
                "feather": ("INT", {"default": 60, "min": 0, "max": 8192, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "INSTARAW/Utils"
    RETURN_TYPES = ("MASK", "MASK", "STRING",)
    RETURN_NAMES = ("mask", "mask_inverted", "info_text",)
    FUNCTION = "feather_mask"
    DESCRIPTION = """
# INSTARAW Feather Mask (Corrected)
- **expand**: Expands the solid core of the mask by this many pixels.
- **feather**: Creates a smooth gradient over this many pixels, starting from the new solid edge.
- This provides a perfect, artifact-free feather just like in Photoshop.
"""

    def feather_mask(self, mask, expand, feather, tapered_corners):
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia library is required for this node.")

        expand = float(expand)
        feather = float(max(0, feather))

        # If feather is zero, just do a simple, hard expansion for performance.
        if feather < 0.01:
            final_masks = []
            for m in mask:
                output = m.unsqueeze(0).unsqueeze(0).to(DEVICE)
                if round(expand) != 0:
                    op = morph.dilation if expand > 0 else morph.erosion
                    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE) if tapered_corners else torch.ones(3, 3, device=DEVICE)
                    for _ in range(abs(round(expand))):
                        output = op(output, kernel)
                final_masks.append(output.squeeze(0).squeeze(0).cpu())
            final_masks_stack = torch.stack(final_masks, dim=0)
            info_text = f"Solid Expand: {expand:.2f}px\nFeather: 0px (disabled)"
            return (final_masks_stack, 1.0 - final_masks_stack, info_text)

        # --- True Feathering Algorithm ---
        final_masks = []
        for m in tqdm(mask, desc="Feathering Mask"):
            source_tensor = m.unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # 1. Create the solid core mask by expanding by the 'expand' amount.
            core_mask = source_tensor
            expand_rounded = round(expand)
            if expand_rounded != 0:
                op = morph.dilation if expand_rounded > 0 else morph.erosion
                kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE) if tapered_corners else torch.ones(3, 3, device=DEVICE)
                for _ in range(abs(expand_rounded)):
                    core_mask = op(core_mask, kernel)

            # 2. Create the larger outer boundary for the full effect.
            outer_mask = source_tensor
            total_expand_size = round(expand + feather)
            if total_expand_size != 0:
                op = morph.dilation if total_expand_size > 0 else morph.erosion
                kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE) if tapered_corners else torch.ones(3, 3, device=DEVICE)
                for _ in range(abs(total_expand_size)):
                    outer_mask = op(outer_mask, kernel)
            
            # 3. Blur the outer boundary mask to create the gradient.
            # Sigma should be proportional to the feather size. feather / 2 is a good standard.
            sigma = feather / 2.0
            # Kernel size must be an odd number.
            kernel_size = 2 * math.ceil(3.0 * sigma) + 1
            blurred_outer_mask = kfilters.gaussian_blur2d(
                outer_mask, (kernel_size, kernel_size), (sigma, sigma)
            )

            # 4. Composite the solid core on top of the blurred outer mask.
            # This is the key step that guarantees a solid core and seamless gradient.
            final_mask = torch.max(core_mask, blurred_outer_mask)
            final_masks.append(final_mask.squeeze(0).squeeze(0).cpu())

        final_masks_stack = torch.stack(final_masks, dim=0)
        
        total_effect = expand + feather
        info_text = f"Solid Expand: {expand:.2f}px\nFeather Size: {feather:.2f}px\nTotal Effect Size: {total_effect:.2f}px"

        return (final_masks_stack, 1.0 - final_masks_stack, info_text)


NODE_CLASS_MAPPINGS = {
    "INSTARAWFeatherMask": INSTARAWFeatherMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAWFeatherMask": "🎭 INSTARAW Feather Mask",
}
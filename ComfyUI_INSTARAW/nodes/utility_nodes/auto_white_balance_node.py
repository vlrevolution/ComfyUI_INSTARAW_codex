# Filename: ComyUI_INSTARAW/nodes/utility_nodes/auto_white_balance_node.py
# ---

import torch
import numpy as np
from PIL import Image
import os

# Import the AWB utility function from our modules
from ...modules.detection_bypass.utils import auto_white_balance_ref

class INSTARAW_AutoWhiteBalance:
    """
    Corrects the color cast of an image using either the "Grey World" assumption
    or a reference image to achieve a natural white balance. This should typically
    be the first step in an authenticity pipeline.
    """

    MODES = ["Grey World", "Reference Image"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "Grey World"}),
            },
            "optional": {
                "awb_ref_image": ("IMAGE", {
                    "tooltip": "A real, well-balanced photo to use as a color temperature reference."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Converts a torch tensor to a PIL image."""
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL image to a torch tensor."""
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def execute(self, image: torch.Tensor, mode: str, awb_ref_image: torch.Tensor = None):
        if mode == "Reference Image" and awb_ref_image is None:
            print("⚠️ INSTARAW Auto White Balance: 'Reference Image' mode selected but no reference connected. Passing through original image.")
            return (image,)
            
        print(f"🎨 INSTARAW Auto White Balance: Applying '{mode}' correction.")

        ref_numpy_image = None
        if mode == "Reference Image":
            # Use the first image from the reference batch
            ref_pil_image = self.tensor_to_pil(awb_ref_image[0:1])
            ref_numpy_image = np.array(ref_pil_image)

        processed_images = []
        for i in range(image.shape[0]):
            single_image_tensor = image[i:i+1]
            pil_image = self.tensor_to_pil(single_image_tensor)
            numpy_image = np.array(pil_image)

            # Call our utility function
            processed_numpy_image = auto_white_balance_ref(
                img_arr=numpy_image,
                ref_img_arr=ref_numpy_image # This will be None for "Grey World" mode, which is the desired behavior
            )

            processed_pil_image = Image.fromarray(processed_numpy_image)
            processed_tensor = self.pil_to_tensor(processed_pil_image)
            processed_images.append(processed_tensor)

        if not processed_images:
            return (image,)
            
        final_batch = torch.cat(processed_images, dim=0)
        
        print("✅ INSTARAW Auto White Balance: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "INSTARAW_AutoWhiteBalance": INSTARAW_AutoWhiteBalance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_AutoWhiteBalance": "🎨 INSTARAW Auto White Balance"
}
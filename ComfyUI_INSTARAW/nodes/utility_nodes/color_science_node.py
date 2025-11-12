# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/color_science_node.py
# ---

import torch
import numpy as np
from PIL import Image
import os

# Import the LUT application logic from our modules
from ...modules.detection_bypass.utils import apply_lut, load_lut

class INSTARAW_ColorScience:
    """
    Applies a 3D LUT (.cube file) to an image to simulate the unique color science
    of a specific camera profile (e.g., iPhone, Sony, Fuji).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"forceInput": True, "tooltip": "Connect an INSTARAW LUT Selector node here."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend strength of the LUT. 0.0 is no effect, 1.0 is full effect."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_science"
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

    def apply_color_science(self, image: torch.Tensor, lut_path: str, strength: float):
        print(f"🎨 INSTARAW Color Science: Applying LUT '{os.path.basename(lut_path)}' with strength {strength:.2f}.")

        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"INSTARAW Color Science: LUT file not found at path: {lut_path}")

        # Load the LUT from the provided path
        try:
            lut_data = load_lut(lut_path)
        except Exception as e:
            raise ValueError(f"Failed to load or parse LUT file at {lut_path}: {e}")

        processed_images = []
        for i in range(image.shape[0]):
            single_image_tensor = image[i:i+1]
            pil_image = self.tensor_to_pil(single_image_tensor)
            numpy_image = np.array(pil_image)

            # Apply the LUT using our utility function
            processed_numpy_image = apply_lut(numpy_image, lut_data, strength)

            processed_pil_image = Image.fromarray(processed_numpy_image)
            processed_tensor = self.pil_to_tensor(processed_pil_image)
            processed_images.append(processed_tensor)

        if not processed_images:
            print("⚠️ INSTARAW Color Science: No images were processed. Returning original image.")
            return (image,)
            
        final_batch = torch.cat(processed_images, dim=0)
        
        print("✅ INSTARAW Color Science: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "INSTARAW_ColorScience": INSTARAW_ColorScience
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_ColorScience": "🎨 INSTARAW Color Science (LUT)"
}
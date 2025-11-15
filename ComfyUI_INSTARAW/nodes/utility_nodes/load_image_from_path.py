import torch
import numpy as np
from PIL import Image
import os

class INSTARAW_LoadImageFromPath:
    """
    A simple utility node to load an image from a given file path string.
    This is essential for previewing the final output of nodes like the 
    Synthesize Authentic Metadata node, allowing you to see the image *after*
    it has been saved, compressed, and had its color profile potentially converted.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The 'forceInput' is crucial. It creates an input socket 
                # instead of a text box, allowing you to connect the filepath string.
                "filepath": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "INSTARAW/Utilities"

    def load_image(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"INSTARAW Load Image: File not found at {filepath}")

        # Open the image file from the provided path
        img_pil = Image.open(filepath).convert('RGB')

        # Convert the PIL Image to a NumPy array and normalize to 0.0-1.0
        img_np = np.array(img_pil).astype(np.float32) / 255.0

        # Convert the NumPy array to a PyTorch tensor
        img_tensor = torch.from_numpy(img_np)

        # Add the batch dimension (ComfyUI expects Batch, Height, Width, Channels)
        img_tensor = img_tensor.unsqueeze(0)

        # For the mask, we can just use the alpha channel if it exists, or create one.
        # Since we convert to RGB, we'll just use the luminance as a simple mask.
        mask = img_tensor.mean(dim=-1, keepdim=True)

        return (img_tensor, mask)

NODE_CLASS_MAPPINGS = {
    "INSTARAW_LoadImageFromPath": INSTARAW_LoadImageFromPath
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_LoadImageFromPath": "➡️ INSTARAW Load Image From Path"
}
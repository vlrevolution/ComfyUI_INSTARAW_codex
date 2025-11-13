# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/compression_node.py
# ---

import torch
import numpy as np
from PIL import Image
import io

class INSTARAW_Compression:
    """
    Simulates the final, lossy compression stage of a camera's image pipeline.
    This adds authentic JPEG artifacts, a critical signal for AI detection systems.
    This should be the very last step before saving the final image.
    """

    CHROMA_SUBSAMPLING_MODES = ["Standard (4:2:0 - Most Common)", "High Quality (4:4:4)", "Aggressive (4:1:1)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "quality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": "JPEG quality level. 90-98 is typical for high-quality phone photos."
                }),
                "chroma_subsampling": (cls.CHROMA_SUBSAMPLING_MODES, {
                    "default": "Standard (4:2:0 - Most Common)",
                    "tooltip": "Controls how color information is compressed. 4:2:0 is the standard for most JPEGs."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_compression"
    CATEGORY = "INSTARAW/Post-Processing"

    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def apply_compression(self, image: torch.Tensor, quality: int, chroma_subsampling: str):
        print(f"🗜️ INSTARAW Compression: Applying JPEG compression with quality {quality}.")

        subsampling_map = {
            "High Quality (4:4:4)": 0,
            "Standard (4:2:0 - Most Common)": 2,
            "Aggressive (4:1:1)": 1,
        }
        subsampling_val = subsampling_map.get(chroma_subsampling, 2)

        processed_images = []
        for i in range(image.shape[0]):
            img_pil = self.tensor_to_pil(image[i:i+1])
            
            # Use an in-memory buffer to save and reload the image as a JPEG
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality, subsampling=subsampling_val)
            
            # Rewind the buffer and reload the compressed image data
            buffer.seek(0)
            reloaded_pil = Image.open(buffer).convert('RGB')
            
            processed_images.append(self.pil_to_tensor(reloaded_pil))
        
        final_batch = torch.cat(processed_images, dim=0)

        print("✅ INSTARAW Compression: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "INSTARAW_Compression": INSTARAW_Compression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_Compression": "🗜️ INSTARAW Compression",
}
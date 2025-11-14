# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/compression_node.py
import torch
import numpy as np
from ...modules.detection_bypass.camera_pipeline import _jpeg_recompress
import random

class INSTARAW_Multi_Compression:
    """
    Simulates the generational loss from repeated JPEG compression, a key
    artifact of images shared online. This should be a final step.
    """

    CHROMA_SUBSAMPLING_MODES = ["Standard (4:2:0 - Most Common)", "High Quality (4:4:4)", "Aggressive (4:1:1)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "cycles": ("INT", {
                    "default": 4, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of times to re-compress the image."
                }),
                "min_quality": ("INT", {
                    "default": 43, "min": 1, "max": 100, "step": 1,
                    "tooltip": "The minimum possible quality for each random compression cycle."
                }),
                "max_quality": ("INT", {
                    "default": 75, "min": 1, "max": 100, "step": 1,
                    "tooltip": "The maximum possible quality for each random compression cycle."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4: tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image: torch.Tensor, enabled: bool, cycles: int, min_quality: int, max_quality: int, seed: int):
        if not enabled: return (image,)

        # Ensure min is not greater than max
        min_q = min(min_quality, max_quality)
        max_q = max(min_quality, max_quality)

        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            # Seed the random number generator for this specific image
            rng = random.Random(seed + i)
            
            # The multi-cycle compression loop
            for cycle in range(cycles):
                # Pick a random quality within the specified range for each cycle
                quality = rng.randint(min_q, max_q)
                img_numpy = _jpeg_recompress(img_numpy, quality=quality)
            
            processed_images.append(self._numpy_to_tensor(img_numpy))

        return (torch.cat(processed_images, dim=0),) if processed_images else (image,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {"INSTARAW_Multi_Compression": INSTARAW_Multi_Compression}
NODE_DISPLAY_NAME_MAPPINGS = {"INSTARAW_Multi_Compression": "🛡️ INSTARAW Multi-Compression"}
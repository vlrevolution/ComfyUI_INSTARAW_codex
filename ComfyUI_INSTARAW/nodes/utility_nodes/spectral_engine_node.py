# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/spectral_engine_node.py
# ---

import torch
import numpy as np
import os

from ...modules.detection_bypass.utils import direct_spectral_match

IPHONE_STATS = None

class INSTARAW_SpectralEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Seed is no longer used but good to keep
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def load_stats(self):
        global IPHONE_STATS
        if IPHONE_STATS is None:
            node_file_path = os.path.dirname(os.path.realpath(__file__))
            instaraw_root_path = os.path.abspath(os.path.join(node_file_path, "..", ".."))
            stats_path = os.path.join(instaraw_root_path, "pretrained", "iphone_stats.npz")
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"iPhone stats file not found! Expected at: {stats_path}")
            print("🧠 INSTARAW Spectral Engine: Loading iPhone stats profile...")
            IPHONE_STATS = np.load(stats_path)
            print("✅ INSTARAW Spectral Engine: Stats loaded.")
        return IPHONE_STATS

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    def numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image: torch.Tensor, strength: float, seed: int):
        print(f"🚀 INSTARAW Spectral Engine (Direct Match v2): Starting.")
        stats = self.load_stats()
        
        # Create the dictionary of target spectra for each channel
        target_spectra = {
            'r': stats['spectra_r'].mean(axis=0),
            'g': stats['spectra_g'].mean(axis=0),
            'b': stats['spectra_b'].mean(axis=0),
        }

        processed_images = []
        for i in range(image.shape[0]):
            numpy_image = self.tensor_to_numpy(image[i:i+1])
            processed_numpy_image = direct_spectral_match(
                img_arr=numpy_image,
                target_spectra=target_spectra,
                strength=strength
            )
            processed_tensor = self.numpy_to_tensor(processed_numpy_image)
            processed_images.append(processed_tensor)

        final_batch = torch.cat(processed_images, dim=0)
        print("✅ INSTARAW Spectral Engine: Processing complete.")
        return (final_batch,)

NODE_CLASS_MAPPINGS = { "INSTARAW_SpectralEngine": INSTARAW_SpectralEngine }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_SpectralEngine": "🛡️ INSTARAW Spectral Engine" }
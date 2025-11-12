# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/spectral_engine_node.py
# ---

import torch
import numpy as np
from PIL import Image
import os

from ...modules.detection_bypass.utils import normalize_spectrum_twostage

class INSTARAW_SpectralEngine:
    """
    A dedicated node for applying the two-stage, stats-guided spectral normalization attack.
    Exposes advanced parameters for fine-tuning the trade-off between bypass and visual quality.
    """

    PRESETS = ["fast", "balanced", "quality", "custom"] # Added custom preset

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (cls.PRESETS, { "default": "balanced" }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
            },
            "optional": {
                # --- NEW POWER-USER OVERRIDES ---
                "s1_attack_strength": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "s1_visual_cost": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "s1_stats_guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "s1_smoothness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                
                "s2_attack_strength": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "s2_visual_cost": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 0.5}),
                "s2_stats_guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "s2_smoothness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                
                "verbose_logging": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 4 and tensor.shape[0] == 1: tensor = tensor.squeeze(0)
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def execute(self, image: torch.Tensor, preset: str, seed: int, **kwargs):
        print(f"🚀 INSTARAW Spectral Engine: Starting '{preset}' preset.")
        
        # --- NEW LOGIC TO HANDLE CUSTOM PARAMS ---
        custom_params = {}
        if preset == "custom":
            print("  - Using CUSTOM preset with user-defined parameters.")
            custom_params = {
                "stage_overrides": {
                    "high_freq": {
                        "initial_const": kwargs.get("s1_attack_strength", 30.0),
                        "c_lpips": kwargs.get("s1_visual_cost", 40.0),
                        "stats_weight": kwargs.get("s1_stats_guidance", 4.0),
                        "delta_blur_sigma": kwargs.get("s1_smoothness", 0.5),
                    },
                    "low_freq": {
                        "initial_const": kwargs.get("s2_attack_strength", 20.0),
                        "c_lpips": kwargs.get("s2_visual_cost", 50.0),
                        "stats_weight": kwargs.get("s2_stats_guidance", 3.0),
                        "delta_blur_sigma": kwargs.get("s2_smoothness", 1.0),
                    }
                }
            }
        
        processed_images = []
        for i in range(image.shape[0]):
            single_image_tensor = image[i:i+1]
            pil_image = self.tensor_to_pil(single_image_tensor)
            numpy_image = np.array(pil_image)

            # Pass the custom params dict to the core function
            processed_numpy_image = normalize_spectrum_twostage(
                img_arr=numpy_image,
                preset=preset if preset != "custom" else "balanced", # Use balanced as base for custom
                seed=seed + i,
                verbose=kwargs.get("verbose_logging", False),
                **custom_params
            )

            processed_pil_image = Image.fromarray(processed_numpy_image)
            processed_tensor = self.pil_to_tensor(processed_pil_image)
            processed_images.append(processed_tensor)

        if not processed_images:
            print("⚠️ INSTARAW Spectral Engine: No images were processed. Returning original image.")
            return (image,)
            
        final_batch = torch.cat(processed_images, dim=0)
        
        print("✅ INSTARAW Spectral Engine: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = { "INSTARAW_SpectralEngine": INSTARAW_SpectralEngine, }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_SpectralEngine": "🛡️ INSTARAW Spectral Engine", }
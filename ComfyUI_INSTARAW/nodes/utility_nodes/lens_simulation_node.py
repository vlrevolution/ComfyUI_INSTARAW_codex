# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/lens_simulation_node.py
# ---

import torch
import numpy as np
from PIL import Image

# We will use the camera simulation utility, but only for specific effects
from ...modules.detection_bypass.camera_pipeline import simulate_camera_pipeline

class INSTARAW_LensSimulation:
    """
    Simulates the subtle, physical imperfections of a real camera lens, such as
    vignetting and chromatic aberration. This should be applied late in the
    post-processing pipeline, before final compression.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vignette_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Strength of the corner darkening effect."
                }),
                "chromatic_aberration": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Strength of the color fringing on high-contrast edges."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effects"
    CATEGORY = "INSTARAW/Post-Processing"

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def apply_effects(self, image: torch.Tensor, vignette_strength: float, chromatic_aberration: float, seed: int):
        if vignette_strength == 0 and chromatic_aberration == 0:
            return (image,)

        print(f"렌즈 INSTARAW Lens Simulation: Applying effects.")

        processed_images = []
        for i in range(image.shape[0]):
            numpy_image = self.tensor_to_numpy(image[i:i+1])

            # Call the camera pipeline simulator, but only enable the effects we want.
            # All other effects (noise, jpeg, etc.) are disabled by default or set to zero.
            processed_numpy_image = simulate_camera_pipeline(
                img_arr=numpy_image,
                vignette_strength=vignette_strength,
                chroma_aberr_strength=chromatic_aberration,
                seed=seed + i,
                # Explicitly disable other effects to be safe
                bayer=False,
                jpeg_cycles=0,
                iso_scale=0,
                read_noise_std=0,
                hot_pixel_prob=0,
            )

            processed_tensor = self.numpy_to_tensor(processed_numpy_image)
            processed_images.append(processed_tensor)

        if not processed_images:
            return (image,)
            
        final_batch = torch.cat(processed_images, dim=0)
        
        print("✅ INSTARAW Lens Simulation: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "INSTARAW_LensSimulation": INSTARAW_LensSimulation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_LensSimulation": "렌즈 INSTARAW Lens Simulation",
}
# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/camera_simulator.py
import torch
import numpy as np
from ...modules.detection_bypass.camera_pipeline import simulate_camera_pipeline

class INSTARAW_Camera_Simulator:
    """
    A comprehensive node for simulating a full, realistic camera pipeline.
    This includes sensor effects (Bayer, noise, hot pixels), lens effects
    (vignette, chromatic aberration), and processing artifacts (motion blur).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # Lens Effects
                "vignette_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_aberr_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                
                # Sensor Effects
                "bayer_demosaic": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "iso_noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "sensor_read_noise": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "hot_pixel_prob": ("FLOAT", {"default": 1e-6, "min": 0.0, "max": 1e-3, "step": 1e-7, "display": "slider"}),
                
                # Processing Artifacts
                "banding_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_blur_kernel": ("INT", {"default": 1, "min": 1, "max": 51, "step": 2, "tooltip": "Kernel size for motion blur. 1 means no blur."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4: tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image: torch.Tensor, enabled: bool, seed: int, vignette_strength: float,
                chroma_aberr_strength: float, bayer_demosaic: bool, iso_noise_scale: float,
                sensor_read_noise: float, hot_pixel_prob: float, banding_strength: float,
                motion_blur_kernel: int):
        
        if not enabled:
            return (image,)

        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            # Call the full pipeline with all parameters
            processed_numpy = simulate_camera_pipeline(
                img_arr=img_numpy,
                bayer=bayer_demosaic,
                vignette_strength=vignette_strength,
                chroma_aberr_strength=chroma_aberr_strength,
                iso_scale=iso_noise_scale,
                read_noise_std=sensor_read_noise,
                hot_pixel_prob=hot_pixel_prob,
                banding_strength=banding_strength,
                motion_blur_kernel=motion_blur_kernel,
                seed=seed + i,
                # We explicitly disable JPEG cycles here, as that belongs in its own dedicated node.
                jpeg_cycles=0 
            )
            
            processed_images.append(self._numpy_to_tensor(processed_numpy))

        if not processed_images:
            return (image,)

        return (torch.cat(processed_images, dim=0),)
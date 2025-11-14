# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/camera_simulator.py
import torch
import numpy as np
from ...modules.detection_bypass.camera_pipeline import simulate_camera_pipeline

class INSTARAW_Camera_Simulator:
    """
    A comprehensive node for simulating a full, realistic camera pipeline,
    now with improved defaults and corrected UI widgets for precision control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypassed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # In-Camera JPEG Save
                "initial_jpeg_quality": ("INT", {
                    "default": 98, "min": 85, "max": 100, "step": 1,
                    "tooltip": "The quality of the first JPEG save, simulating the camera's output."
                }),

                # Lens Effects
                "vignette_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_aberr_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                
                # Sensor Effects
                "bayer_demosaic": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "iso_noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "sensor_read_noise": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "hot_pixel_prob": ("FLOAT", {"default": 1e-6, "min": 0.0, "max": 1e-3, "step": 1e-7, "display": "number"}),
                
                # Processing Artifacts
                "banding_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number"}),
                "motion_blur_kernel": ("INT", {
                    "default": 3, "min": 1, "max": 51, 
                    "step": 1, # --- THIS IS THE FIX ---
                    "tooltip": "Kernel size for motion blur. Even numbers will be rounded up to the next odd number. 1 = no blur."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4: tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image: torch.Tensor, enabled: bool, seed: int, initial_jpeg_quality: int, **kwargs):
        if not enabled: return (image,)
        
        motion_blur_kernel = kwargs.get('motion_blur_kernel', 3)
        # The backend safety check: automatically make the kernel size odd
        if motion_blur_kernel > 1 and motion_blur_kernel % 2 == 0:
            motion_blur_kernel += 1
            
        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            processed_numpy = simulate_camera_pipeline(
                img_arr=img_numpy,
                bayer=kwargs.get('bayer_demosaic', True),
                vignette_strength=kwargs.get('vignette_strength', 0.1),
                chroma_aberr_strength=kwargs.get('chroma_aberr_strength', 1.0),
                iso_scale=kwargs.get('iso_noise_scale', 1.0),
                read_noise_std=kwargs.get('sensor_read_noise', 2.0),
                hot_pixel_prob=kwargs.get('hot_pixel_prob', 1e-6),
                banding_strength=kwargs.get('banding_strength', 0.0),
                motion_blur_kernel=motion_blur_kernel,
                jpeg_cycles=1,
                jpeg_quality_range=(initial_jpeg_quality, initial_jpeg_quality),
                seed=seed + i,
            )
            processed_images.append(self._numpy_to_tensor(processed_numpy))

        return (torch.cat(processed_images, dim=0),) if processed_images else (image,)
# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/spectral_normalizer_node.py
import torch
import numpy as np
from ...modules.detection_bypass.utils.non_semantic_attack import non_semantic_attack

class INSTARAW_Spectral_Normalizer:
    """
    Applies a non-semantic adversarial attack to remove AI fingerprints.
    This advanced version exposes all key hyperparameters for fine-tuning
    and provides detailed logging of the optimization process.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "iterations": ("INT", {"default": 500, "min": 10, "max": 10000}),
                "learning_rate": ("FLOAT", {"default": 3e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "t_lpips": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "LPIPS threshold. Loss is penalized above this value."}),
                "c_lpips": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "LPIPS loss weight."}),
                "t_l2": ("FLOAT", {"default": 3e-5, "min": 0.0, "max": 1.0, "step": 1e-6, "tooltip": "L2 norm threshold for the perturbation."}),
                "c_l2": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "L2 loss weight."}),
                "grad_clip_value": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "print_log_every_n": ("INT", {"default": 100, "min": 0, "max": 1000, "tooltip": "How often to print log updates. 0 to disable."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def execute(self, image: torch.Tensor, iterations: int, learning_rate: float,
                t_lpips: float, c_lpips: float, t_l2: float, c_l2: float,
                grad_clip_value: float, seed: int, print_log_every_n: int):
        
        processed_images = []
        for i in range(image.shape[0]):
            img_numpy = self._tensor_to_numpy(image[i:i+1])
            
            # Call the threaded attack function with all parameters
            processed_numpy = non_semantic_attack(
                img_arr=img_numpy,
                iterations=iterations,
                learning_rate=learning_rate,
                t_lpips=t_lpips,
                c_lpips=c_lpips,
                t_l2=t_l2,
                c_l2=c_l2,
                grad_clip_value=grad_clip_value,
                seed=seed + i,
                print_log_every_n=print_log_every_n # Pass the logging frequency
            )
            processed_images.append(self._numpy_to_tensor(processed_numpy))

        return (torch.cat(processed_images, dim=0),)
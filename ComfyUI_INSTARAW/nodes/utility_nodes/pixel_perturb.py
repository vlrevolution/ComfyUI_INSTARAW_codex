# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/pixel_perturb.py
import torch
import numpy as np

class INSTARAW_Pixel_Perturb:
    """
    Applies a randomized, low-magnitude perturbation to each pixel.
    This is a 1:1 implementation of the 'randomized_perturbation' function
    from the Image-Detection-Bypass-Utility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "magnitude": ("FLOAT", {
                    "default": 0.008, "min": 0.0, "max": 0.05, "step": 0.001,
                    "tooltip": "Fractional magnitude of the random noise. 0.008 is the default."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        # Convert tensor [0, 1] float to numpy [0, 255] uint8
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        # Convert numpy [0, 255] uint8 back to tensor [0, 1] float
        return torch.from_numpy(np_array.astype(np.float32) / 255.0)

    def execute(self, image: torch.Tensor, magnitude: float, seed: int):
        if magnitude == 0:
            return (image,)

        # Create a NumPy Random Generator for thread-safe randomness
        rng = np.random.default_rng(seed)

        processed_images = []
        for i in range(image.shape[0]):
            # 1. Convert single image from batch to numpy [0, 255] uint8
            img_numpy_uint8 = self._tensor_to_numpy(image[i])

            # --- This is the exact logic from the reference repository ---
            # 2. Calculate magnitude in the [0, 255] range
            mag_255 = magnitude * 255.0
            
            # 3. Generate uniform random noise between -mag and +mag
            perturb = rng.uniform(low=-mag_255, high=mag_255, size=img_numpy_uint8.shape)
            
            # 4. Add noise to a float32 version of the image
            out = img_numpy_uint8.astype(np.float32) + perturb
            
            # 5. Clip the result and convert back to uint8
            processed_numpy_uint8 = np.clip(out, 0, 255).astype(np.uint8)
            # --- End of reference logic ---

            # 6. Convert back to a tensor for ComfyUI
            processed_tensor = self._numpy_to_tensor(processed_numpy_uint8)
            processed_images.append(processed_tensor)

        # Stack the processed images back into a single batch tensor
        final_batch = torch.stack(processed_images, dim=0)

        return (final_batch,)
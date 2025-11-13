# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/auto_white_balance_node.py
# ---

import torch
import numpy as np
from PIL import Image
import os

NODE_DIR = os.path.dirname(os.path.realpath(__file__))
IPHONE13_REF_PATH = os.path.join(NODE_DIR, "_refs", "iphone13.jpg")

class INSTARAW_AutoWhiteBalance:
    """
    Corrects the color cast and exposure. Features a built-in iPhone 13 reference,
    multiple target grey levels, or can use an external reference image. Includes a
    strength slider to control the intensity of the correction.
    """

    MODES = [
        "iPhone 13 (Internal Ref)", 
        "Reference Image",
        "Balanced Scene (Mid Grey)", 
        "Bright Scene (Light Grey)", 
        "Dark Scene (Dark Grey)", 
    ]
    
    GREY_TARGETS = {
        "Balanced Scene (Mid Grey)": 128.0,
        "Bright Scene (Light Grey)": 160.0,
        "Dark Scene (Dark Grey)": 96.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "iPhone 13 (Internal Ref)"}),
                # --- NEW STRENGTH SLIDER ---
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Intensity of the white balance correction. 0.0 is no effect, 1.0 is full correction."
                }),
            },
            "optional": {
                "awb_ref_image": ("IMAGE", {
                    "tooltip": "External reference image for the 'Reference Image' mode."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def _auto_white_balance_ref(self, img_arr: np.ndarray, ref_img_arr: np.ndarray = None, grey_target: float = 128.0) -> np.ndarray:
        img = img_arr.astype(np.float32)
        if ref_img_arr is not None:
            target_mean = ref_img_arr.astype(np.float32).reshape(-1, 3).mean(axis=0)
        else:
            target_mean = np.array([grey_target, grey_target, grey_target], dtype=np.float32)
        img_mean = img.reshape(-1, 3).mean(axis=0)
        if np.all(img_mean < 1e-6):
            return img_arr
        scale = target_mean / (img_mean + 1e-6)
        corrected = np.clip(img * scale, 0, 255) # Keep as float for blending
        return corrected

    def execute(self, image: torch.Tensor, mode: str, strength: float, awb_ref_image: torch.Tensor = None):
        if strength == 0:
            return (image,) # Pass through if strength is zero

        if mode == "Reference Image" and awb_ref_image is None:
            print("⚠️ INSTARAW Auto White Balance: 'Reference Image' mode selected but no reference connected. Passing through.")
            return (image,)
            
        print(f"🎨 INSTARAW Auto White Balance: Applying '{mode}' correction with strength {strength:.2f}.")

        ref_numpy_image = None
        grey_target_value = 128.0
        if mode == "iPhone 13 (Internal Ref)":
            if not os.path.exists(IPHONE13_REF_PATH):
                raise FileNotFoundError(f"Internal reference image not found! Expected at: {IPHONE13_REF_PATH}")
            ref_pil = Image.open(IPHONE13_REF_PATH)
            ref_numpy_image = np.array(ref_pil.convert("RGB"))
        elif mode == "Reference Image":
            ref_numpy_image = self._tensor_to_numpy(awb_ref_image[0:1])
        else:
            grey_target_value = self.GREY_TARGETS.get(mode, 128.0)

        processed_images = []
        for i in range(image.shape[0]):
            original_numpy = self._tensor_to_numpy(image[i:i+1])
            
            # Get the fully corrected image as a float
            corrected_numpy_float = self._auto_white_balance_ref(
                img_arr=original_numpy,
                ref_img_arr=ref_numpy_image,
                grey_target=grey_target_value
            )

            # --- NEW BLENDING LOGIC ---
            # Blend the original and the corrected image based on strength
            blended_numpy_float = (original_numpy.astype(np.float32) * (1 - strength)) + (corrected_numpy_float * strength)
            
            # Final clip and conversion to uint8
            blended_numpy_uint8 = np.clip(blended_numpy_float, 0, 255).astype(np.uint8)
            # --- END BLENDING LOGIC ---

            processed_tensor = self._numpy_to_tensor(blended_numpy_uint8)
            processed_images.append(processed_tensor)

        if not processed_images:
            return (image,)
            
        final_batch = torch.cat(processed_images, dim=0)
        
        print("✅ INSTARAW Auto White Balance: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = { "INSTARAW_AutoWhiteBalance": INSTARAW_AutoWhiteBalance }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_AutoWhiteBalance": "🎨 INSTARAW Auto White Balance" }
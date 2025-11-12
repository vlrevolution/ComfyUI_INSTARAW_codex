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
    multiple target grey levels, or can use an external reference image.
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

    # --- REVERTED TO THE CORRECT, SIMPLER LOGIC ---
    def _auto_white_balance_ref(self, img_arr: np.ndarray, ref_img_arr: np.ndarray = None, grey_target: float = 128.0) -> np.ndarray:
        """
        Corrects color and luminance using a single, robust linear scaling method.
        """
        img = img_arr.astype(np.float32)

        if ref_img_arr is not None:
            # Reference mode: target the average RGB value of the reference image
            target_mean = ref_img_arr.astype(np.float32).reshape(-1, 3).mean(axis=0)
        else:
            # Target Grey mode: target a specific shade of neutral grey
            target_mean = np.array([grey_target, grey_target, grey_target], dtype=np.float32)

        img_mean = img.reshape(-1, 3).mean(axis=0)
        
        # Avoid division by zero for black images
        if np.all(img_mean < 1e-6):
            return img_arr

        # Calculate the scaling factor for each channel
        scale = target_mean / (img_mean + 1e-6)

        # Apply the scaling and clip the result to the valid 0-255 range
        corrected = np.clip(img * scale, 0, 255).astype(np.uint8)

        return corrected
    # --- END REPLACEMENT ---

    def execute(self, image: torch.Tensor, mode: str, awb_ref_image: torch.Tensor = None):
        # ... (This function's logic is correct and remains the same) ...
        print(f"🎨 INSTARAW Auto White Balance: Applying '{mode}' correction.")
        ref_numpy_image = None
        grey_target_value = 128.0
        if mode == "iPhone 13 (Internal Ref)":
            if not os.path.exists(IPHONE13_REF_PATH):
                raise FileNotFoundError(f"Internal reference image not found! Expected at: {IPHONE13_REF_PATH}")
            print(f"  - Using internal reference: {os.path.basename(IPHONE13_REF_PATH)}")
            ref_pil = Image.open(IPHONE13_REF_PATH)
            ref_numpy_image = np.array(ref_pil.convert("RGB"))
        elif mode == "Reference Image":
            if awb_ref_image is None:
                print("⚠️ 'Reference Image' mode selected but no reference connected. Passing through.")
                return (image,)
            ref_numpy_image = self._tensor_to_numpy(awb_ref_image[0:1])
        else:
            grey_target_value = self.GREY_TARGETS.get(mode, 128.0)
        processed_images = []
        for i in range(image.shape[0]):
            numpy_image = self._tensor_to_numpy(image[i:i+1])
            processed_numpy_image = self._auto_white_balance_ref(
                img_arr=numpy_image,
                ref_img_arr=ref_numpy_image,
                grey_target=grey_target_value
            )
            processed_tensor = self._numpy_to_tensor(processed_numpy_image)
            processed_images.append(processed_tensor)
        if not processed_images:
            return (image,)
        final_batch = torch.cat(processed_images, dim=0)
        print("✅ INSTARAW Auto White Balance: Processing complete.")
        return (final_batch,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = { "INSTARAW_AutoWhiteBalance": INSTARAW_AutoWhiteBalance }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_AutoWhiteBalance": "🎨 INSTARAW Auto White Balance" }
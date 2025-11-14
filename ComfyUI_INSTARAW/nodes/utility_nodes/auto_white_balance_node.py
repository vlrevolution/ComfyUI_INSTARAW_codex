# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/auto_white_balance_node.py
import torch
import numpy as np
from PIL import Image
import os

# Define the path to the internal reference image
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
# We need to go up two levels from `utility_nodes` to the `ComfyUI_INSTARAW` root
INSTARAW_ROOT = os.path.abspath(os.path.join(NODE_DIR, "..", ".."))
IPHONE13_REF_PATH = os.path.join(INSTARAW_ROOT, "modules", "_refs", "iphone13.jpg")


class INSTARAW_AutoWhiteBalance:
    """
    Corrects color cast. Can use an Authenticity Profile, a live reference image,
    a built-in iPhone 13 sample, or a simple grey-world assumption.
    """
    
    # --- UPDATED MODES ---
    MODES = [
        "Authenticity Profile",
        "Internal Sample Image (iPhone)", # Renamed for clarity
        "Reference Image",
        "Grey World (Balanced)", 
        "Grey World (Bright)", 
        "Grey World (Dark)", 
    ]
    
    GREY_TARGETS = {
        "Grey World (Balanced)": 128.0,
        "Grey World (Bright)": 160.0,
        "Grey World (Dark)": 96.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "Authenticity Profile"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "awb_ref_image": ("IMAGE",),
                "profile_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",); FUNCTION = "execute"; CATEGORY = "INSTARAW/Authenticity"

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor is None: return None
        if tensor.ndim == 4: tensor = tensor[0]
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def _numpy_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array.astype(np.float32) / 255.0).unsqueeze(0)

    def _auto_white_balance(self, img_arr: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
        img = img_arr.astype(np.float32)
        img_mean = img.reshape(-1, 3).mean(axis=0)
        if np.all(img_mean < 1e-6):
            return img_arr # Avoid division by zero for black images
        scale = target_mean / (img_mean + 1e-6)
        return np.clip(img * scale, 0, 255)

    def execute(self, image: torch.Tensor, mode: str, strength: float, awb_ref_image: torch.Tensor = None, profile_path: str = None):
        if strength == 0:
            return (image,)
            
        target_mean = None

        # --- NEW, HIERARCHICAL LOGIC ---
        # The node intelligently finds the best available reference based on the selected mode.
        
        # 1. Profile Mode (Highest Priority if selected)
        if mode == "Authenticity Profile":
            if profile_path and profile_path.strip():
                npz_path = f"{profile_path}.npz"
                try:
                    stats = np.load(npz_path)
                    if 'chroma_mean' in stats:
                        print(f"🎨 AWB: Using Authenticity Profile '{os.path.basename(npz_path)}'.")
                        target_mean = stats['chroma_mean'].mean(axis=0)
                    else:
                        print(f"⚠️ AWB Warning: 'chroma_mean' not found in profile. Check profile or select another mode.")
                except Exception as e:
                    print(f"⚠️ AWB Warning: Could not load profile '{npz_path}'. Error: {e}")
            else:
                print("⚠️ AWB Warning: 'Authenticity Profile' mode selected but no profile connected.")

        # 2. Internal Sample Mode
        elif mode == "Internal Sample Image (iPhone 13)":
            if not os.path.exists(IPHONE13_REF_PATH):
                raise FileNotFoundError(f"Internal reference image not found! Expected at: {IPHONE13_REF_PATH}")
            print("🎨 AWB: Using Internal Sample Image (iPhone 13).")
            ref_pil = Image.open(IPHONE13_REF_PATH)
            ref_numpy = np.array(ref_pil.convert("RGB"))
            target_mean = ref_numpy.astype(np.float32).reshape(-1, 3).mean(axis=0)

        # 3. Live Reference Image Mode
        elif mode == "Reference Image":
            if awb_ref_image is not None:
                print("🎨 AWB: Using live reference image.")
                ref_numpy = self._tensor_to_numpy(awb_ref_image)
                target_mean = ref_numpy.astype(np.float32).reshape(-1, 3).mean(axis=0)
            else:
                print("⚠️ AWB Warning: 'Reference Image' mode selected but no reference connected.")
        
        # 4. Fallback to Grey World if no other target was found, or if explicitly selected
        if target_mean is None:
            grey_target_value = self.GREY_TARGETS.get(mode, 128.0)
            print(f"🎨 AWB: Falling back to Grey World assumption (target: {grey_target_value}).")
            target_mean = np.array([grey_target_value, grey_target_value, grey_target_value], dtype=np.float32)

        # --- BATCH PROCESSING ---
        processed_images = []
        for i in range(image.shape[0]):
            original_numpy = self._tensor_to_numpy(image[i:i+1])
            corrected_float = self._auto_white_balance(original_numpy, target_mean)
            blended_float = (original_numpy.astype(np.float32) * (1 - strength)) + (corrected_float * strength)
            blended_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)
            processed_images.append(self._numpy_to_tensor(blended_uint8))

        return (torch.cat(processed_images, dim=0),)
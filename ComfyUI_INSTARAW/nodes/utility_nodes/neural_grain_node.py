# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/neural_grain_node.py
# ---

import torch
import numpy as np
import os
from PIL import Image

# Import the GrainNet architecture from our modules
from ...modules.neural_grain.net import GrainNet

# Get the absolute path to the root of the ComfyUI_INSTARAW custom node
NODE_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
INSTARAW_ROOT_PATH = os.path.abspath(os.path.join(NODE_FILE_PATH, "..", ".."))
MODEL_PATH = os.path.join(INSTARAW_ROOT_PATH, "pretrained", "neural_grain", "grainnet.pt")

# Global variable to hold the loaded model, so we don't reload it every time
GRAIN_NET_MODEL = None

class INSTARAW_NeuralGrain:
    """
    Applies a sophisticated, learned grain pattern using the GrainNet model from the
    "Neural Film Grain Rendering" paper. This version is adapted to apply achromatic
    (non-colored) grain to simulate digital sensor noise.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "grain_size": ("FLOAT", {
                    "default": 0.2, "min": 0.01, "max": 0.8, "step": 0.01,
                    "tooltip": "Controls the scale of the grain texture. Smaller values = finer grain."
                }),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Blend strength of the grain effect. 0.0 is no effect, 1.0 is normal."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "INSTARAW/Post-Processing"

    def load_model(self):
        """Loads the GrainNet model into a global variable for reuse."""
        global GRAIN_NET_MODEL
        if GRAIN_NET_MODEL is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Neural Grain model not found! Expected at: {MODEL_PATH}")
            
            print("🧠 INSTARAW Neural Grain: Loading GrainNet model...")
            # The pretrained model uses block_nb=2 and activation='tanh'
            state_dict = torch.load(MODEL_PATH, map_location="cpu")
            GRAIN_NET_MODEL = GrainNet(block_nb=2, activation='tanh')
            GRAIN_NET_MODEL.load_state_dict(state_dict)
            GRAIN_NET_MODEL.eval()
            print("✅ INSTARAW Neural Grain: Model loaded.")
        return GRAIN_NET_MODEL

    def apply_grain(self, image: torch.Tensor, seed: int, grain_size: float, strength: float):
        if strength == 0:
            return (image,)

        model = self.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        original_device = image.device
        batch_size, height, width, _ = image.shape
        
        # Convert to BCHW format for the model
        image_bchw = image.permute(0, 3, 1, 2).to(device)

        # --- THE ACHROMATIC FIX ---
        # 1. Convert the color image to grayscale for processing
        # Using standard ITU-R BT.601 luma weights
        image_gray = (image_bchw[:, 0:1, :, :] * 0.299 + 
                      image_bchw[:, 1:2, :, :] * 0.587 + 
                      image_bchw[:, 2:3, :, :] * 0.114)

        # 2. Run the GrainNet model on the grayscale image
        grain_size_tensor = torch.tensor([grain_size] * batch_size, device=device).float().unsqueeze(1)
        
        # The model expects a seed for its internal random noise generator
        torch.manual_seed(seed)
        grainy_gray_output = model(image_gray, grain_size_tensor)
        
        # 3. Isolate the noise pattern
        # The model outputs a full image; we extract the noise by subtracting the input
        noise_pattern = grainy_gray_output - image_gray
        
        # 4. Add the grayscale noise pattern back to the original color image
        noisy_image_bchw = image_bchw + noise_pattern
        # --- END FIX ---
        
        # Blend the result with the original image based on strength
        blended_image_bchw = torch.clamp((1 - strength) * image_bchw + strength * noisy_image_bchw, 0.0, 1.0)

        final_image = blended_image_bchw.permute(0, 2, 3, 1).to(original_device)

        print("✅ INSTARAW Neural Grain: Processing complete.")
        return (final_image,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "INSTARAW_NeuralGrain": INSTARAW_NeuralGrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_NeuralGrain": "✨ INSTARAW Neural Grain",
}
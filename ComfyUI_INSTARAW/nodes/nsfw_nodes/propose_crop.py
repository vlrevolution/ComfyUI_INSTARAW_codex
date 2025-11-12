# ---
# ComfyUI INSTARAW - Propose Crop from Detections Node
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import json
import torch

class INSTARAW_CropBox_From_BoundingBoxes:
    """
    Calculates a "safe" crop area that excludes detected NSFW bounding boxes.
    It takes JSON-formatted bounding box data and an image (for dimensions)
    and outputs a JSON-formatted crop proposal.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bounding_boxes_json": ("STRING", {"forceInput": True, "multiline": True}),
                "crop_strategy": (["Keep Upper Body", "Keep Top Half"], {"default": "Keep Upper Body"}),
                "padding": ("INT", {"default": 20, "min": 0, "max": 1024, "step": 1, "tooltip": "Safety margin in pixels to add above the highest detected box."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("proposed_crop_json",)
    FUNCTION = "propose_crop"
    CATEGORY = "INSTARAW/NSFW"

    def propose_crop(self, image, bounding_boxes_json, crop_strategy, padding):
        _batch, img_height, img_width, _channels = image.shape
        
        try:
            detections = json.loads(bounding_boxes_json)
        except (json.JSONDecodeError, TypeError):
            detections = []

        # If no NSFW regions are detected, propose a full-image crop.
        if not detections:
            full_crop = {"x": 0, "y": 0, "width": img_width, "height": img_height, "status": "no_detections"}
            return (json.dumps(full_crop),)
            
        # Find the highest point (minimum y-coordinate) of all detected boxes.
        # The box format is [x, y, width, height].
        min_y = min(d['box'][1] for d in detections if 'box' in d and len(d['box']) == 4)
        
        # Calculate the final crop height based on the strategy.
        crop_height = 0
        if crop_strategy == "Keep Upper Body":
            # The crop ends 'padding' pixels above the highest detection.
            crop_height = max(0, min_y - padding)
        elif crop_strategy == "Keep Top Half":
            crop_height = img_height // 2
            
        # Ensure the crop height is not zero or negative.
        if crop_height <= 0:
            print("⚠️ INSTARAW Propose Crop: Calculated crop height is zero or less. Defaulting to top half.")
            crop_height = img_height // 2
        
        # The crop always starts from the top-left corner.
        proposed_crop = {
            "x": 0,
            "y": 0,
            "width": img_width,
            "height": int(crop_height),
            "status": "proposal_generated"
        }
        
        return (json.dumps(proposed_crop),)

# --- ADD THIS CODE TO THE END OF THE FILE ---

# =================================================================================
# NODE REGISTRATION
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_CropBox_From_BoundingBoxes": INSTARAW_CropBox_From_BoundingBoxes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_CropBox_From_BoundingBoxes": "✂️ INSTARAW Propose Crop from Detections",
}
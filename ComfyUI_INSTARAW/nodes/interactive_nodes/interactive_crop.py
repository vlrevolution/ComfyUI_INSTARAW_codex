# ---
# ComfyUI INSTARAW - Interactive Crop Node (Final Corrected Version)
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import torch
import json
from comfy.model_management import InterruptProcessingException
from nodes import PreviewImage
from .image_filter_messaging import send_and_wait, Response, TimeoutResponse

class INSTARAW_Interactive_Crop(PreviewImage):
    """
    An interactive node that displays an image and allows a user to define a crop region
    by drawing, moving, and resizing a rectangle.
    """
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_data_json",)
    FUNCTION = "crop_interactively"
    CATEGORY = "INSTARAW/Interactive"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "timeout": ("INT", {"default": 600, "min": 1, "max": 999999}),
            },
            "optional": {
                "mask": ("MASK",),
                "proposed_crop_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "uid": "UNIQUE_ID",
                "node_identifier": "NID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def crop_interactively(self, **kwargs):
        image = kwargs.get('image')
        timeout = kwargs.get('timeout')
        uid = kwargs.get('uid')
        node_identifier = kwargs.get('node_identifier')
        mask = kwargs.get('mask')
        proposed_crop_json = kwargs.get('proposed_crop_json')

        if image is None or timeout is None or uid is None or node_identifier is None:
            raise ValueError("INSTARAW_Interactive_Crop is missing required inputs. Check connections.")

        save_kwargs = {}
        if "prompt" in kwargs: save_kwargs["prompt"] = kwargs.get("prompt")
        if "extra_pnginfo" in kwargs: save_kwargs["extra_pnginfo"] = kwargs.get("extra_pnginfo")

        urls = self.save_images(images=image, **save_kwargs)['ui']['images']
        
        payload = {
            "uid": uid,
            "node_identifier": node_identifier,
            "urls": urls,
            "interactive_crop": True,
        }
        
        if proposed_crop_json and proposed_crop_json.strip():
            try:
                payload["proposed_crop"] = json.loads(proposed_crop_json)
            except json.JSONDecodeError:
                print("⚠️ INSTARAW Interactive Crop: Invalid proposed_crop_json. Ignoring.")

        # --- ADD THIS LINE FOR DEBUGGING ---
        print(f"DEBUG: Payload being sent to frontend: {json.dumps(payload, indent=2)}")
        # --- END DEBUG LINE ---

        print("💨 INSTARAW Interactive Crop: Waiting for user to define crop area...")
        response = send_and_wait(payload, timeout, uid, node_identifier)

        if isinstance(response, TimeoutResponse):
            print("⏰ INSTARAW Interactive Crop: Timed out. Passing through original image.")
            full_mask = torch.ones_like(image[:, :, :, 0]) if mask is None else mask
            crop_data = {"x": 0, "y": 0, "width": image.shape[2], "height": image.shape[1], "status": "timeout"}
            return (image, full_mask, json.dumps(crop_data))

        # --- THIS IS THE FIX ---
        # The crop data is now directly on the response object, not nested.
        crop_data = response.crop if hasattr(response, 'crop') else None
        # --- END FIX ---
        
        if not crop_data:
            raise InterruptProcessingException("User cancelled the crop operation.")

        x = int(crop_data.get("x", 0))
        y = int(crop_data.get("y", 0))
        width = int(crop_data.get("width", image.shape[2]))
        height = int(crop_data.get("height", image.shape[1]))

        print(f"✅ INSTARAW Interactive Crop: Received crop data: {crop_data}")

        cropped_image = image[:, y:y+height, x:x+width, :]
        
        cropped_mask = None
        if mask is not None:
            mask_to_crop = mask
            if mask.dim() == 4:
                mask_to_crop = mask.squeeze(1)
            
            cropped_mask = mask_to_crop[:, y:y+height, x:x+width]
        else:
            cropped_mask = torch.ones_like(cropped_image[:, :, :, 0])

        return (cropped_image, cropped_mask, json.dumps(crop_data))
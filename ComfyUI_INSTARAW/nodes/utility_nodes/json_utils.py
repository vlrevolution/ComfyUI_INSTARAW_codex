# ---
# ComfyUI INSTARAW - JSON Utility Nodes
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---
import json

class INSTARAW_JSON_Extract_Values:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("x", "y", "width", "height",)
    FUNCTION = "extract"
    CATEGORY = "INSTARAW/Utils"

    def extract(self, json_string):
        try:
            data = json.loads(json_string)
            if not isinstance(data, dict):
                return (0, 0, 0, 0)
        except (json.JSONDecodeError, TypeError):
            return (0, 0, 0, 0)
        
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        
        return (x, y, width, height,)

NODE_CLASS_MAPPINGS = {
    "INSTARAW_JSON_Extract_Values": INSTARAW_JSON_Extract_Values,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_JSON_Extract_Values": "... INSTARAW JSON Extract Values",
}
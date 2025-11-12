# ---
# ComfyUI INSTARAW - Branding Node (Final Stable Version)
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
A simple, stable, purely visual node for placing a brand logo on the workflow.
Configuration is done via the right-click Properties Panel.
"""

class INSTARAW_BrandingNode:
    @classmethod
    def INPUT_TYPES(cls):
        # This node has no inputs/widgets on its body.
        # All configuration is handled via the properties panel on the frontend.
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "INSTARAW/Branding"
    OUTPUT_NODE = True

    def do_nothing(self, **kwargs):
        # This node is purely visual.
        return (None,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# =================================================================================
# NODE REGISTRATION
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_BrandingNode": INSTARAW_BrandingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_BrandingNode": "🖼️ INSTARAW Branding",
}
# ---
# ComfyUI INSTARAW - API Model Selector Node (Corrected)
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
A utility node to select a generative API model, simplifying workflows that use the unified API nodes.
"""

from ..api_nodes.generative_api_nodes import MODEL_CONFIG

class INSTARAW_API_ModelSelector:
    """
    Selects a generative model from a dropdown. The list of models is dynamically
    loaded from the generative_api_nodes configuration, ensuring it's always up to date.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_CONFIG.keys()), {"default": "SeeDream v4"}),
            }
        }

    # --- THE FIX: This node only has ONE output ---
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "select_model"
    CATEGORY = "INSTARAW/Utils"

    def select_model(self, model):
        # --- THE FIX: Return a tuple with ONE item ---
        return (model,)

# =================================================================================
# EXPORT NODE MAPPINGS
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_API_ModelSelector": INSTARAW_API_ModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_API_ModelSelector": "🤖 INSTARAW API Model Selector",
}
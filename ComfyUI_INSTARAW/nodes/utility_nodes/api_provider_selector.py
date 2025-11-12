# ---
# ComfyUI INSTARAW - API Provider Selector Nodes
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
Utility nodes to select API providers and their corresponding keys,
simplifying workflows that use multiple API services.
"""

class INSTARAW_API_ProviderSelector:
    """
    Selects a generic API provider (for SeeDream, Nano Banana, etc.)
    and outputs the provider's name and the corresponding API key.
    """
    PROVIDERS = ["wavespeed.ai", "fal.ai"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (cls.PROVIDERS, {"default": "wavespeed.ai"}),
                "wavespeed_api_key": ("STRING", {"default": "", "multiline": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("provider", "api_key",)
    FUNCTION = "select_provider"
    CATEGORY = "INSTARAW/Utils"

    def select_provider(self, provider, fal_api_key, wavespeed_api_key):
        if provider == "fal.ai":
            api_key = fal_api_key
        elif provider == "wavespeed.ai":
            api_key = wavespeed_api_key
        else:
            print(f"⚠️ Warning: Unknown provider '{provider}' selected. Returning empty API key.")
            api_key = ""
        return (provider, api_key,)

# --- NEW NODE FOR IDEOGRAM ---
class INSTARAW_IdeogramProviderSelector:
    """
    Selects an Ideogram API provider (Official or fal.ai) and outputs
    the provider's name and the corresponding API key.
    """
    PROVIDERS = ["Official Ideogram", "fal.ai"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (cls.PROVIDERS, {"default": "Official Ideogram"}),
                "official_ideogram_api_key": ("STRING", {"default": "", "multiline": False}),
                "fal_api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("provider", "api_key",)
    FUNCTION = "select_provider"
    CATEGORY = "INSTARAW/Utils"

    def select_provider(self, provider, official_ideogram_api_key, fal_api_key):
        if provider == "Official Ideogram":
            api_key = official_ideogram_api_key
        elif provider == "fal.ai":
            api_key = fal_api_key
        else:
            print(f"⚠️ Warning: Unknown Ideogram provider '{provider}' selected. Returning empty API key.")
            api_key = ""
        return (provider, api_key,)


# =================================================================================
# EXPORT NODE MAPPINGS
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_API_ProviderSelector": INSTARAW_API_ProviderSelector,
    "INSTARAW_IdeogramProviderSelector": INSTARAW_IdeogramProviderSelector, # New node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_API_ProviderSelector": "🔑 INSTARAW API Provider Selector",
    "INSTARAW_IdeogramProviderSelector": "🔑 INSTARAW Ideogram Provider Selector", # New node
}
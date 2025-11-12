"""
INSTARAW Interactive Nodes
Nodes that pause the workflow for user input.
"""

from .image_filter import (
    INSTARAW_ImageFilter,
    INSTARAW_MaskImageFilter,
    INSTARAW_TextImageFilter,
)
from .interactive_crop import INSTARAW_Interactive_Crop

NODE_CLASS_MAPPINGS = {
    "INSTARAW_ImageFilter": INSTARAW_ImageFilter,
    "INSTARAW_TextImageFilter": INSTARAW_TextImageFilter,
    "INSTARAW_MaskImageFilter": INSTARAW_MaskImageFilter,
    "INSTARAW_Interactive_Crop": INSTARAW_Interactive_Crop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_ImageFilter": "🎭 INSTARAW Image Filter",
    "INSTARAW_TextImageFilter": "📝 INSTARAW Text/Image Filter",
    "INSTARAW_MaskImageFilter": "✂️ INSTARAW Mask Filter",
    "INSTARAW_Interactive_Crop": "🖼️ INSTARAW Interactive Crop",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
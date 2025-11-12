"""
ComfyUI INSTARAW
A general-purpose custom nodes package by Instara

INSTARAW is a collection of powerful custom nodes for ComfyUI that brings together
hard-to-install dependencies and integrations in one convenient package.

Features:
- API Integrations: SeeDream, Ideogram, and more
- Easy Installation: Pre-packaged dependencies
- Modular Design: Easy to extend and customize
- INSTARAW Brand: Where we push the boundaries

Created by Instara
"""

import os
import nodes

# This line is critical for loading the JavaScript and CSS files.
# It tells ComfyUI where to find the web assets for this extension.
if "ComfyUI_INSTARAW" not in nodes.EXTENSION_WEB_DIRS:
    nodes.EXTENSION_WEB_DIRS["ComfyUI_INSTARAW"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Required exports for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Package metadata
__version__ = "1.2.0" # Version bumped to reflect the major addition
__author__ = "Instara"
__description__ = "INSTARAW - General purpose custom nodes for ComfyUI"
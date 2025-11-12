"""
INSTARAW Output Nodes
Nodes that are final endpoints in a workflow, like saving files.
"""

from .save_with_metadata import (
    NODE_CLASS_MAPPINGS as SAVE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SAVE_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **SAVE_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SAVE_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
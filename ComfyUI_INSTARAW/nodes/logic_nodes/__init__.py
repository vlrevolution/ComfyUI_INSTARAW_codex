"""
INSTARAW Logic Nodes
Nodes for controlling workflow execution and logic.
"""

from .logic_nodes import (
    NODE_CLASS_MAPPINGS as LOGIC_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LOGIC_DISPLAY_MAPPINGS,
)

# Import from our new virtual nodes file
from .virtual_nodes import (
    NODE_CLASS_MAPPINGS as VIRTUAL_LOGIC_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIRTUAL_LOGIC_DISPLAY_MAPPINGS,
)

# Combine all logic node mappings
NODE_CLASS_MAPPINGS = {
    **LOGIC_MAPPINGS,
    **VIRTUAL_LOGIC_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LOGIC_DISPLAY_MAPPINGS,
    **VIRTUAL_LOGIC_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
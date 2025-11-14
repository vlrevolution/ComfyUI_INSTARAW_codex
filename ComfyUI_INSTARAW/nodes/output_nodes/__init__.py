"""
INSTARAW Output Nodes
Nodes that are final endpoints in a workflow, like saving files.
"""

# Import mappings from the first node file, using aliases to avoid name conflicts
from .save_with_metadata import NODE_CLASS_MAPPINGS as SAVE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SAVE_DISPLAY_MAPPINGS

# Import mappings from the new node file, also using aliases
from .synthesize_with_metadata import NODE_CLASS_MAPPINGS as SYNTH_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SYNTH_DISPLAY_MAPPINGS

# Create the final dictionaries that will be exported
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge the mappings from all imported nodes
NODE_CLASS_MAPPINGS.update(SAVE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SYNTH_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(SAVE_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SYNTH_DISPLAY_MAPPINGS)


# The __all__ export is not strictly necessary for ComfyUI's loader but is good practice.
# It ensures that only these specific dictionaries are exposed when this package is imported.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
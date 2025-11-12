# --- Filename: ../ComfyUI_INSTARAW/nodes/logic_nodes/virtual_nodes.py (UPDATED) ---
import json
from comfy.comfy_types.node_typing import IO
import torch


class INSTARAW_BooleanBypass:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {}),
                "invert_input": (
                    "BOOLEAN",
                    {"default": False, "label_on": "INVERTED", "label_off": "NORMAL"},
                ),
            },
            "optional": {
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
            },
        }

    OUTPUT_NODE = False
    RETURN_TYPES = (
        "*",
        "*",
        "*",
        "*",
    )
    RETURN_NAMES = (
        "output_1",
        "output_2",
        "output_3",
        "output_4",
    )
    FUNCTION = "passthrough"
    CATEGORY = "INSTARAW/Logic"

    def passthrough(
        self,
        boolean,
        invert_input,
        input_1=None,
        input_2=None,
        input_3=None,
        input_4=None,
    ):
        return (
            input_1,
            input_2,
            input_3,
            input_4,
        )

# --- NEW NODE CLASS ---
class INSTARAW_GroupBypassToBoolean:
    """
    Outputs a boolean reflecting its own bypass state.
    Place this node inside a group to use the group's bypass as a boolean controller.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # This widget will be controlled by the JS. It's 'required'
                # to ensure it's always present for the JS to find.
                "is_active": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "get_status"
    CATEGORY = "INSTARAW/Logic"

    def get_status(self, is_active):
        # This function simply returns the value of the widget,
        # which is dynamically set by the frontend JavaScript.
        return (is_active,)
# --- END NEW NODE CLASS ---


class INSTARAW_PreviewAnyAndPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        # We only define the input socket. The UI will be generated automatically by the return message.
        return {
            "required": {"source": (IO.ANY, {})},
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("source_out",)
    FUNCTION = "preview_and_pass"
    CATEGORY = "INSTARAW/Logic"
    OUTPUT_NODE = True  # CRITICAL for passthrough functionality

    def preview_and_pass(self, source=None):
        value = "None"
        if isinstance(source, torch.Tensor):
            value = f"Tensor with shape: {source.shape}\n(dtype: {source.dtype}, device: {source.device})"
        elif isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source, indent=4)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = "source exists, but could not be serialized."

        return {"ui": {"text": [value]}, "result": (source,)}


NODE_CLASS_MAPPINGS = {
    "INSTARAW_BooleanBypass": INSTARAW_BooleanBypass,
    "INSTARAW_PreviewAnyAndPassthrough": INSTARAW_PreviewAnyAndPassthrough,
    "INSTARAW_GroupBypassToBoolean": INSTARAW_GroupBypassToBoolean, # Add new node here
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_BooleanBypass": "🔀 INSTARAW Boolean Bypass",
    "INSTARAW_PreviewAnyAndPassthrough": "👁️ INSTARAW Preview Any & Passthrough",
    "INSTARAW_GroupBypassToBoolean": "🔀 INSTARAW Group Bypass Detector", # And here
}
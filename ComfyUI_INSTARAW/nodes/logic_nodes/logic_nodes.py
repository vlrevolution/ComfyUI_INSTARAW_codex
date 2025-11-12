# ---
# Filename: ../ComfyUI_INSTARAW/nodes/logic_nodes/logic_nodes.py
# ---

# ---
# ComfyUI INSTARAW - Logic Nodes
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
Type-specific logic nodes for creating reliable, conditional workflows.
This approach avoids client-side JavaScript for maximum stability.
"""

# _INSTARA_INJECT_

# --- Boolean Logic Node ---
class INSTARAW_BooleanLogic:
    """
    Performs logical operations (AND, OR, XOR, NOT) on boolean inputs.
    Perfect for creating complex conditions to control switches.
    """

    OPERATIONS = ["AND", "OR", "XOR", "NOT A"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean_a": ("BOOLEAN", {"forceInput": True}),
                "operation": (cls.OPERATIONS, {"default": "AND"}),
            },
            "optional": {
                "boolean_b": ("BOOLEAN", {"forceInput": True, "default": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    FUNCTION = "operate"
    CATEGORY = "INSTARAW/Logic"

    def operate(self, boolean_a, operation, boolean_b=True):
        result = False
        if operation == "AND":
            result = boolean_a and boolean_b
        elif operation == "OR":
            result = boolean_a or boolean_b
        elif operation == "XOR":
            result = boolean_a != boolean_b
        elif operation == "NOT A":
            result = not boolean_a

        return (result,)


# --- Image To Boolean Node ---
class INSTARAW_ImageToBoolean:
    """
    Outputs a boolean value based on the presence of an image input.
    True if an image is connected and not None, otherwise False.
    Useful for controlling switches based on whether an image exists in the workflow path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "check_image"
    CATEGORY = "INSTARAW/Logic"

    def check_image(self, image=None):
        is_present = image is not None
        return (is_present,)


# A simple base class to avoid repeating the switch logic.
class INSTARAW_SwitchBase:
    FUNCTION = "switch"
    CATEGORY = "INSTARAW/Logic"

    def switch(self, boolean=False, input_true=None, input_false=None):
        """
        Selects which input to pass through. If boolean is False or None (not connected/bypassed),
        it defaults to the input_false path.
        """
        if boolean:
            return (input_true,)
        else:
            return (input_false,)


# --- Type-Specific Switches (UPDATED) ---


class INSTARAW_ImageSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("IMAGE",),
                "input_false": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("output",)


class INSTARAW_MaskSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("MASK",),
                "input_false": ("MASK",),
            },
        }
    RETURN_TYPES = ("MASK",); RETURN_NAMES = ("output",)


class INSTARAW_LatentSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("LATENT",),
                "input_false": ("LATENT",),
            },
        }
    RETURN_TYPES = ("LATENT",); RETURN_NAMES = ("output",)


class INSTARAW_ConditioningSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("CONDITIONING",),
                "input_false": ("CONDITIONING",),
            },
        }
    RETURN_TYPES = ("CONDITIONING",); RETURN_NAMES = ("output",)


class INSTARAW_IntSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("INT", {}),
                "input_false": ("INT", {}),
            },
        }
    RETURN_TYPES = ("INT",); RETURN_NAMES = ("output",)


class INSTARAW_FloatSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("FLOAT", {}),
                "input_false": ("FLOAT", {}),
            },
        }
    RETURN_TYPES = ("FLOAT",); RETURN_NAMES = ("output",)


class INSTARAW_StringSwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("STRING", {}),
                "input_false": ("STRING", {}),
            },
        }
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("output",)

class INSTARAW_AnySwitch(INSTARAW_SwitchBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boolean": ("BOOLEAN", {}),
                "input_true": ("*",),
                "input_false": ("*",),
            },
        }
    RETURN_TYPES = ("*",); RETURN_NAMES = ("output",)

class INSTARAW_InvertBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"boolean": ("BOOLEAN", {"forceInput": True})}}
    RETURN_TYPES = ("BOOLEAN",); FUNCTION = "invert"; CATEGORY = "INSTARAW/Logic"
    def invert(self, boolean): return (not boolean,)


# =================================================================================
# EXPORT NODE MAPPINGS
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_BooleanLogic": INSTARAW_BooleanLogic,
    "INSTARAW_ImageToBoolean": INSTARAW_ImageToBoolean,
    "INSTARAW_ImageSwitch": INSTARAW_ImageSwitch,
    "INSTARAW_MaskSwitch": INSTARAW_MaskSwitch,
    "INSTARAW_LatentSwitch": INSTARAW_LatentSwitch,
    "INSTARAW_ConditioningSwitch": INSTARAW_ConditioningSwitch,
    "INSTARAW_IntSwitch": INSTARAW_IntSwitch,
    "INSTARAW_FloatSwitch": INSTARAW_FloatSwitch,
    "INSTARAW_StringSwitch": INSTARAW_StringSwitch,
    "INSTARAW_AnySwitch": INSTARAW_AnySwitch,
    "INSTARAW_InvertBoolean": INSTARAW_InvertBoolean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_BooleanLogic": "🔀 INSTARAW Boolean Logic",
    "INSTARAW_ImageToBoolean": "🔀 INSTARAW Image to Boolean",
    "INSTARAW_ImageSwitch": "🔀 INSTARAW Switch (Image)",
    "INSTARAW_MaskSwitch": "🔀 INSTARAW Switch (Mask)",
    "INSTARAW_LatentSwitch": "🔀 INSTARAW Switch (Latent)",
    "INSTARAW_ConditioningSwitch": "🔀 INSTARAW Switch (Conditioning)",
    "INSTARAW_IntSwitch": "🔀 INSTARAW Switch (Int)",
    "INSTARAW_FloatSwitch": "🔀 INSTARAW Switch (Float)",
    "INSTARAW_StringSwitch": "🔀 INSTARAW Switch (String)",
    "INSTARAW_AnySwitch": "🔀 INSTARAW Switch (Any)",
    "INSTARAW_InvertBoolean": "🔁 INSTARAW Invert Boolean",
}
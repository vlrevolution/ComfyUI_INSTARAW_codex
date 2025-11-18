"""
WAN Aspect Ratio Node
Provides preset aspect ratios optimized for WAN workflows
Resolution is 2x what will be generated (due to 2x upscaling in WAN workflow)
"""


class INSTARAW_WANAspectRatio:
    """
    Aspect ratio presets for WAN (Weighted Attention Network) workflows.
    These resolutions are optimized for the WAN 2x upscaling workflow.
    """

    # Aspect ratio to resolution mapping
    ASPECT_RATIOS = {
        "3:4 (Portrait)": {"width": 720, "height": 960},
        "9:16 (Tall Portrait)": {"width": 540, "height": 960},
        "1:1 (Square)": {"width": 960, "height": 960},
        "16:9 (Landscape)": {"width": 960, "height": 540},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()), {
                    "default": "1:1 (Square)"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_label")
    FUNCTION = "get_resolution"
    CATEGORY = "INSTARAW/Input"
    DESCRIPTION = "WAN aspect ratio presets. Outputs resolution optimized for WAN 2x upscaling workflow."

    def get_resolution(self, aspect_ratio):
        """
        Returns width, height, and aspect label for the selected aspect ratio.

        Args:
            aspect_ratio: Selected aspect ratio string

        Returns:
            Tuple of (width, height, aspect_label)
        """
        resolution = self.ASPECT_RATIOS.get(aspect_ratio, self.ASPECT_RATIOS["1:1 (Square)"])
        # Extract clean aspect ratio (e.g., "3:4 (Portrait)" -> "3:4")
        aspect_label = aspect_ratio.split(" ")[0]
        return (resolution["width"], resolution["height"], aspect_label)


# Node registration
NODE_CLASS_MAPPINGS = {
    "INSTARAW_WANAspectRatio": INSTARAW_WANAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_WANAspectRatio": "📐 WAN Aspect Ratio"
}

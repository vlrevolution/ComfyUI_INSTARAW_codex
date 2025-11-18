"""
SDXL Aspect Ratio Node
Provides preset aspect ratios optimized for SDXL (Stable Diffusion XL) workflows
"""


class INSTARAW_SDXLAspectRatio:
    """
    Aspect ratio presets for SDXL (Stable Diffusion XL) workflows.
    These resolutions are optimized for SDXL's native training resolution.
    """

    # Aspect ratio to resolution mapping
    ASPECT_RATIOS = {
        "3:4 (Portrait)": {"width": 896, "height": 1152},
        "9:16 (Tall Portrait)": {"width": 768, "height": 1344},
        "1:1 (Square)": {"width": 1024, "height": 1024},
        "16:9 (Landscape)": {"width": 1344, "height": 768},
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
    DESCRIPTION = "SDXL aspect ratio presets. Outputs resolution optimized for Stable Diffusion XL."

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
    "INSTARAW_SDXLAspectRatio": INSTARAW_SDXLAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_SDXLAspectRatio": "📐 SDXL Aspect Ratio"
}

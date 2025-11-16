# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/line_splitter.py
# ---

class INSTARAW_LineSplitter:
    """
    Splits a multiline string into a list of clean lines.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiline_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "girl in red dress\n"
                                   "girl in blue dress\n"
                                   "girl in green dress",
                        "tooltip": "Each non-empty line becomes a separate entry in the output list.",
                    },
                ),
                "keep_empty_lines": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True, empty lines are kept as '' entries instead of being removed.",
                    },
                ),
            }
        }

    # 👇 This is the important line
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = [True]   # list_out is a STRING LIST

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list_out",)
    FUNCTION = "split"
    CATEGORY = "INSTARAW/Utility"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def split(self, multiline_text, keep_empty_lines):
        import re

        raw_lines = multiline_text.splitlines()
        cleaned = [re.sub(r"\s+", " ", ln).strip() for ln in raw_lines]

        if not keep_empty_lines:
            cleaned = [ln for ln in cleaned if ln]

        # cleaned is a Python list[str]
        return (cleaned,)
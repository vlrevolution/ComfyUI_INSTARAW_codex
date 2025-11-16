# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/debug_prompt_overlay.py
# ---

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class INSTARAW_DebugPromptOverlay:
    """
    Debug node: overlay prompt text on the image.
    Designed for LIST mode: receives a single image and a single prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "overlay"
    CATEGORY = "INSTARAW/Debug"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def overlay(self, image, prompt):
        pil = tensor_to_pil(image)
        draw = ImageDraw.Draw(pil)

        # Try to load a nicer font, otherwise fallback to default
        font = None
        fonts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "fonts")
        try:
            font_path = os.path.join(fonts_dir, "BricolageGrotesque.ttf")
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size=int(pil.height * 0.03))
        except Exception:
            pass

        if font is None:
            font = ImageFont.load_default()

        # Prepare prompt text
        if prompt is None:
            prompt = ""
        prompt = str(prompt)[:200]   # Avoid overlong prompts

        # Measure text with Pillow >= 8
        bbox = draw.textbbox((0, 0), prompt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad = 8
        x = pad
        y = pil.height - th - pad

        # Background rectangle
        draw.rectangle(
            [x - pad, y - pad, x + tw + pad, y + th + pad],
            fill=(0, 0, 0, 180)
        )

        # Draw text
        draw.text((x, y), prompt, font=font, fill=(255, 255, 255))

        return (pil_to_tensor(pil),)


# Register in-module
NODE_CLASS_MAPPINGS = {
    "INSTARAW_DebugPromptOverlay": INSTARAW_DebugPromptOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_DebugPromptOverlay": "🐛 INSTARAW Debug Prompt Overlay",
}
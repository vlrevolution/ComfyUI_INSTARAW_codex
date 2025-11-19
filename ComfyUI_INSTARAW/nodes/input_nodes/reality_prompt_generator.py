# ---
# Filename: ../ComfyUI_INSTARAW/nodes/input_nodes/reality_prompt_generator.py
# Reality Prompt Generator (RPG) - Full Implementation
# ---

import json
import hashlib


class INSTARAW_RealityPromptGenerator:
    """
    Reality Prompt Generator (RPG) - A comprehensive prompt batch manager
    that integrates with a 22MB prompts database and supports creative AI generation.

    Outputs STRING LISTS compatible with batch tensor workflows.
    Follows GPT spec architecture with auto/img2img/txt2img modes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["auto", "img2img", "txt2img"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "'img2img': tie prompts to AIL images.\n"
                            "'txt2img': drive EmptyLatentImage by prompt batch length.\n"
                            "'auto': img2img if AIL linked; else txt2img."
                        ),
                    },
                ),
                "global_negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Fallback negative prompt if a batch entry has no specific negative prompt.",
                    },
                ),
                "generation_mode": (
                    ["sum_repeat_counts", "one_per_entry"],
                    {
                        "default": "sum_repeat_counts",
                        "tooltip": (
                            "sum_repeat_counts: total generation_count = sum(repeat_count).\n"
                            "one_per_entry: generation_count = number of entries (ignores repeat_count)."
                        ),
                    },
                ),
                "creative_model": (
                    [
                        "gemini-2.5-pro",
                        "gemini-flash-latest",
                        "grok-4-fast-reasoning",
                        "grok-4-fast-non-reasoning",
                        "grok-4-0709",
                        "grok-3-mini"
                    ],
                    {
                        "default": "gemini-2.5-pro",
                        "tooltip": "AI model to use for creative prompt generation.",
                    },
                ),
                "gemini_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Google Gemini API key. Leave empty to use GEMINI_API_KEY environment variable.",
                    },
                ),
                "grok_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "xAI Grok API key. Leave empty to use XAI_API_KEY environment variable.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional: Connect images from Advanced Image Loader (Batch Tensor mode).\n"
                            "Used for img2img workflows to validate prompt count matches image count."
                        ),
                    },
                ),
                "character_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional: Single character reference image for Creative/Character modes.\n"
                            "Will be sent to Gemini/Grok for visual character consistency."
                        ),
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "Target output width. Connect from INSTARAW Aspect Ratio Selector.",
                    },
                ),
                "output_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "Target output height. Connect from INSTARAW Aspect Ratio Selector.",
                    },
                ),
                "aspect_label": (
                    "STRING",
                    {
                        "default": "1:1",
                        "tooltip": "Aspect ratio label (e.g., '16:9'). Connect from INSTARAW Aspect Ratio Selector.",
                    },
                ),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "prompt_batch_data": (
                    "STRING",
                    {
                        "default": "[]",
                    },
                ),
                "resolved_mode": (
                    "STRING",
                    {
                        "default": "txt2img",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "prompt_list_positive",
        "prompt_list_negative",
        "generation_count",
        "resolved_mode",
    )

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = [True, True, False, False]
    # prompt_list_positive => list
    # prompt_list_negative => list
    # generation_count, resolved_mode => scalars

    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Prompts"

    def execute(
        self,
        mode,
        global_negative,
        generation_mode,
        creative_model,
        gemini_api_key,
        grok_api_key,
        images=None,
        character_image=None,
        node_id=None,
        prompt_batch_data="[]",
        resolved_mode="txt2img",
    ):
        """
        Execute the RPG node - parse the prompt batch, expand prompts, return STRING LISTS.

        This method is deterministic and has no side effects. All creative generation
        happens in the frontend via backend API calls.

        Args:
            images: Optional IMAGE batch from AIL for img2img mode
            character_image: Optional single IMAGE for character reference
        """
        # 1) Parse prompt batch
        try:
            prompt_batch = json.loads(prompt_batch_data or "[]")
        except Exception as e:
            print(f"[RPG] Error parsing prompt_batch_data: {e}")
            prompt_batch = []

        # 2) Get image count from actual connected images
        image_count = 0
        if images is not None:
            import torch
            if isinstance(images, torch.Tensor):
                image_count = images.shape[0]  # Batch dimension

        # 3) Compute effective prompt list (positive + negative) with repeat_count
        positives = []
        negatives = []

        for entry in prompt_batch:
            pos = (entry.get("positive_prompt") or "").strip()
            neg = (entry.get("negative_prompt") or "").strip()
            rc = max(1, int(entry.get("repeat_count", 1)))

            if generation_mode == "one_per_entry":
                # Treat every entry as 1 slot
                rc_effective = 1
            else:
                # sum_repeat_counts mode
                rc_effective = rc

            for _ in range(rc_effective):
                positives.append(pos)
                negatives.append(neg if neg else global_negative)

        # 4) Determine effective mode (img2img / txt2img)
        if mode == "auto":
            if image_count > 0:
                resolved = "img2img"
            else:
                resolved = "txt2img"
        else:
            resolved = mode

        # 5) Compute generation_count
        generation_count = len(positives)

        # 6) Log validation info
        if image_count > 0 and generation_count > 0:
            if image_count == generation_count:
                print(f"[RPG] ✅ {generation_count} prompts ↔ {image_count} images (perfect match)")
            else:
                print(f"[RPG] ⚠️ {generation_count} prompts vs {image_count} images (mismatch)")

        # 7) If prompt_batch is empty, return valid empty lists (no exception)
        if generation_count == 0:
            print("[RPG] Warning: Prompt batch is empty. Returning empty prompt lists.")
            return ([""], [""], 0, resolved)

        # 8) Return STRING LISTS
        return (positives, negatives, generation_count, resolved)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Let ComfyUI use default change detection based on inputs.
        We compute a deterministic hash from prompt_batch_data for caching.
        """
        prompt_batch_data = kwargs.get("prompt_batch_data", "[]")
        mode = kwargs.get("mode", "auto")
        global_negative = kwargs.get("global_negative", "")
        generation_mode = kwargs.get("generation_mode", "sum_repeat_counts")
        expected_image_count = kwargs.get("expected_image_count", -1)

        # Create a hash of all inputs that affect output
        hasher = hashlib.sha256()
        hasher.update(prompt_batch_data.encode("utf-8"))
        hasher.update(mode.encode("utf-8"))
        hasher.update(global_negative.encode("utf-8"))
        hasher.update(generation_mode.encode("utf-8"))
        hasher.update(str(expected_image_count).encode("utf-8"))

        return hasher.hexdigest()


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "INSTARAW_RealityPromptGenerator": INSTARAW_RealityPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_RealityPromptGenerator": "🎲 INSTARAW Reality Prompt Generator",
}

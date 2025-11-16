class INSTARAW_ImagePromptIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt_list": ("STRING", {"multiline": False, "default": ""}),
                "extend_mode": (
                    ["truncate", "repeat_last", "cycle"],
                    {"default": "repeat_last"},
                ),
            }
        }

    # First output (images) is NOT a list, second (prompts) IS a list
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = [False, True]

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "prompts",)
    FUNCTION = "map"
    CATEGORY = "INSTARAW/Workflow Logic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def map(self, images, prompt_list, extend_mode):
        B = images.shape[0]

        if isinstance(prompt_list, (list, tuple)):
            prompts = [str(p) for p in prompt_list]
        else:
            prompts = [str(prompt_list)] if str(prompt_list) != "" else []

        if len(prompts) == 0:
            aligned = [""] * B
        else:
            aligned = []
            if extend_mode == "truncate":
                aligned = prompts[:B] if len(prompts) >= B else prompts + [""] * (B - len(prompts))
            elif extend_mode == "repeat_last":
                last = prompts[-1]
                for i in range(B):
                    aligned.append(prompts[i] if i < len(prompts) else last)
            elif extend_mode == "cycle":
                for i in range(B):
                    aligned.append(prompts[i % len(prompts)])

        return (images, aligned)
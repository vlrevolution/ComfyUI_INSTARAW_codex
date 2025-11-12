# ---
# Filename: ../ComfyUI_INSTARAW/nodes/utility_nodes/authenticity_v2.py
# ---
import os
import torch
from ...modules.detection_bypass.pipeline_v2 import BypassPipelineV2, BypassConfigV2

class INSTARAW_Authenticity_V2:
    """
    Applies a SOTA multi-stage pipeline to imbue AI images with the statistical
    fingerprint of a real camera, specifically targeting an authentic iPhone look.
    """
    MODES = ["Full Pipeline (iPhone)", "Ultra-Minimal (Stealth)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"forceInput": True, "tooltip": "Connect an INSTARAW LUT Selector node here."}),
                "mode": (cls.MODES, {"default": "Full Pipeline (iPhone)"}),
                "strength": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1, "tooltip": "Overall effect strength (0-100). Adjusts all stages."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
            },
            "optional": {
                "awb_ref_image": ("IMAGE", {"tooltip": "Optional reference for Auto White Balance to prevent color shifts."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def execute(self, image, lut_path, mode, strength, seed, awb_ref_image=None):
        print(f"🚀 INSTARAW Authenticity V2: Starting '{mode}' mode.")

        if not os.path.exists(lut_path) and mode == "Full Pipeline (iPhone)":
            raise FileNotFoundError(f"iPhone LUT file not found at path: {lut_path}. This is required for 'Full Pipeline' mode.")

        config = BypassConfigV2(
            mode=mode,
            strength=strength / 100.0,
            lut_path=lut_path,
            seed=seed
        )

        pipeline = BypassPipelineV2(config)
        
        final_result_tensor = pipeline.run(
            input_image_tensor=image,
            awb_ref_tensor=awb_ref_image
        )

        print("✅ INSTARAW Authenticity V2: Processing complete.")
        return (final_result_tensor,)
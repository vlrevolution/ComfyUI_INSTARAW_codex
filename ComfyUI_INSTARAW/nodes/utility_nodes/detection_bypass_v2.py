import os
import torch
from ...modules.detection_bypass.pipeline import BypassPipeline, BypassConfig

class INSTARAW_DetectionBypass_V2:
    """
    Applies an intelligent, multi-stage pipeline to make AI images statistically
    indistinguishable from real photos, bypassing detection systems while preserving quality.
    """
    # --- Dynamic Profile Loading ---
    # This automatically finds all our fingerprint profiles and adds them to the dropdown.
    PROFILE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "modules", "detection_bypass", "profiles")
    try:
        FINGERPRINT_PROFILES = sorted([f.replace('.json', '') for f in os.listdir(PROFILE_DIR) if f.endswith('.json')])
    except FileNotFoundError:
        print("⚠️ INSTARAW Detection Bypass: 'profiles' directory not found. Using default list.")
        FINGERPRINT_PROFILES = ["Sony_A7IV_Natural"]

    MODES = ["Ultra-Minimal", "Balanced", "Aggressive"]
    UNMARKER_VERSIONS = ["none", "simplified", "full_fast", "full_balanced", "full_quality"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "Balanced"}),
                "unmarker_version": (cls.UNMARKER_VERSIONS, {
                    "default": "full_balanced",
                    "tooltip": "UnMarker attack version:\n"
                               "- none: No spectral attack (fastest)\n"
                               "- simplified: Basic attack (~30s, 70% bypass)\n"
                               "- full_fast: Two-stage attack (~3min, 85% bypass)\n"
                               "- full_balanced: Two-stage attack (~5min, 92% bypass, RECOMMENDED)\n"
                               "- full_quality: Two-stage with adaptive filtering (~10min, 98% bypass)"
                }),
                "strength": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1, "tooltip": "Overall effect strength (0-100). Primarily used in 'Balanced' mode."}),
                "fingerprint_profile": (cls.FINGERPRINT_PROFILES, {"default": "Sony_A7IV_Natural" if "Sony_A7IV_Natural" in cls.FINGERPRINT_PROFILES else cls.FINGERPRINT_PROFILES[0]}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "debug_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable debug mode to diagnose color shift issues"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def execute(self, image, mode, unmarker_version, strength, fingerprint_profile, seed, debug_mode):
        print(f"🚀 INSTARAW Detection Bypass V2: Starting '{mode}' mode with '{unmarker_version}' UnMarker.")
        if debug_mode:
            print("🔍 DEBUG MODE ENABLED - Will show detailed color analysis")

        config = BypassConfig(
            mode=mode,
            unmarker_version=unmarker_version,
            strength=strength / 100.0,  # Normalize strength to 0-1
            profile_name=fingerprint_profile,
            seed=seed,
            debug_mode=debug_mode
        )

        pipeline = BypassPipeline(config)

        final_result_tensor = pipeline.run(input_image_tensor=image)

        print("✅ INSTARAW Detection Bypass V2: Processing complete.")
        return (final_result_tensor,)
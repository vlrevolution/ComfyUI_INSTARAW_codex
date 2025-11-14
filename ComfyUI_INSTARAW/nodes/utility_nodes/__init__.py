# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/__init__.py
# ---

"""
INSTARAW Utility Nodes
Helper nodes for workflow convenience
"""

# Keep all the working imports
from .seed_generator import (
    NODE_CLASS_MAPPINGS as SEED_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SEED_DISPLAY_MAPPINGS,
)
from .grow_mask_with_blur import (
    NODE_CLASS_MAPPINGS as MASK_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_DISPLAY_MAPPINGS,
)
from .feather_mask import (
    NODE_CLASS_MAPPINGS as FEATHER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FEATHER_DISPLAY_MAPPINGS,
)
from .image_resolution_clamp import (
    NODE_CLASS_MAPPINGS as RESOLUTION_GUARD_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RESOLUTION_GUARD_DISPLAY_MAPPINGS,
)
from .api_provider_selector import (
    NODE_CLASS_MAPPINGS as PROVIDER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PROVIDER_DISPLAY_MAPPINGS,
)
from .realistic_noise import (
    NODE_CLASS_MAPPINGS as NOISE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as NOISE_DISPLAY_MAPPINGS,
)
from .realistic_jpeg import (
    NODE_CLASS_MAPPINGS as JPEG_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as JPEG_DISPLAY_MAPPINGS,
)
from .workflow_logic_nodes import (
    NODE_CLASS_MAPPINGS as WORKFLOW_LOGIC_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as WORKFLOW_LOGIC_DISPLAY_MAPPINGS,
)
from .list_utility_nodes import (
    INSTARAW_BatchFromImageList,
    INSTARAW_ImageListFromBatch,
    INSTARAW_PickFromList,
    INSTARAW_StringListFromStrings,
)
from .string_utility_nodes import (
    INSTARAW_SplitByCommas,
    INSTARAW_StringToFloat,
    INSTARAW_StringToInt,
    INSTARAW_AnyListToString,
    INSTARAW_StringCombine,
)
from .mask_utility_nodes import INSTARAW_MaskedSection, INSTARAW_MaskCombine
from .branding_node import (
    NODE_CLASS_MAPPINGS as BRANDING_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BRANDING_DISPLAY_MAPPINGS,
)
from .image_resize_advanced import (
    NODE_CLASS_MAPPINGS as RESIZE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RESIZE_DISPLAY_MAPPINGS,
)
from .api_model_selector import (
    NODE_CLASS_MAPPINGS as MODEL_SELECTOR_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MODEL_SELECTOR_DISPLAY_MAPPINGS,
)
from .json_utils import (
    NODE_CLASS_MAPPINGS as JSON_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as JSON_DISPLAY_MAPPINGS,
)
from .lut_selector import INSTARAW_LUT_Selector
from .spectral_engine_node import (
    NODE_CLASS_MAPPINGS as SPECTRAL_ENGINE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SPECTRAL_ENGINE_DISPLAY_MAPPINGS,
)
from .color_science_node import (
    NODE_CLASS_MAPPINGS as COLOR_SCIENCE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COLOR_SCIENCE_DISPLAY_MAPPINGS,
)
from .auto_white_balance_node import INSTARAW_AutoWhiteBalance
from .neural_grain_node import (
    NODE_CLASS_MAPPINGS as NEURAL_GRAIN_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as NEURAL_GRAIN_DISPLAY_MAPPINGS,
)
from .lens_simulation_node import (
    NODE_CLASS_MAPPINGS as LENS_EFFECTS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LENS_EFFECTS_DISPLAY_MAPPINGS,
)
# ADD THIS NEW IMPORT
from .compression_node import (
    NODE_CLASS_MAPPINGS as COMPRESSION_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COMPRESSION_DISPLAY_MAPPINGS,
)
from .authenticity_profile_selector import INSTARAW_AuthenticityProfile_Selector
from .fft_match import INSTARAW_FFT_Match
from .texture_normalize import INSTARAW_GLCM_Normalize, INSTARAW_LBP_Normalize
from .metadata_inspector import INSTARAW_Metadata_Inspector
from .texture_engine import NODE_CLASS_MAPPINGS as TEXTURE_ENGINE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TEXTURE_ENGINE_DISPLAY_MAPPINGS
from .spectral_normalizer_node import INSTARAW_Spectral_Normalizer
from .pixel_perturb import INSTARAW_Pixel_Perturb
from .blend_colors import INSTARAW_BlendColors
from .camera_simulator import INSTARAW_Camera_Simulator


# --- CLEANED UP MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    **SEED_MAPPINGS,
    **MASK_MAPPINGS,
    **FEATHER_MAPPINGS,
    **RESOLUTION_GUARD_MAPPINGS,
    **PROVIDER_MAPPINGS,
    **NOISE_MAPPINGS,
    **JPEG_MAPPINGS,
    **WORKFLOW_LOGIC_MAPPINGS,
    "INSTARAW_BatchFromImageList": INSTARAW_BatchFromImageList,
    "INSTARAW_ImageListFromBatch": INSTARAW_ImageListFromBatch,
    "INSTARAW_PickFromList": INSTARAW_PickFromList,
    "INSTARAW_StringListFromStrings": INSTARAW_StringListFromStrings,
    "INSTARAW_SplitByCommas": INSTARAW_SplitByCommas,
    "INSTARAW_StringToFloat": INSTARAW_StringToFloat,
    "INSTARAW_StringToInt": INSTARAW_StringToInt,
    "INSTARAW_AnyListToString": INSTARAW_AnyListToString,
    "INSTARAW_MaskedSection": INSTARAW_MaskedSection,
    "INSTARAW_MaskCombine": INSTARAW_MaskCombine,
    "INSTARAW_StringCombine": INSTARAW_StringCombine,
    **BRANDING_MAPPINGS,
    **RESIZE_MAPPINGS,
    **MODEL_SELECTOR_MAPPINGS,
    **JSON_MAPPINGS,
    "INSTARAW_LUT_Selector": INSTARAW_LUT_Selector,
    **SPECTRAL_ENGINE_MAPPINGS,
    **COLOR_SCIENCE_MAPPINGS,
    "INSTARAW_AutoWhiteBalance": INSTARAW_AutoWhiteBalance,
    **NEURAL_GRAIN_MAPPINGS,
    **LENS_EFFECTS_MAPPINGS,
    **COMPRESSION_MAPPINGS,
    "INSTARAW_AuthenticityProfile_Selector": INSTARAW_AuthenticityProfile_Selector,
    "INSTARAW_FFT_Match": INSTARAW_FFT_Match,
    "INSTARAW_GLCM_Normalize": INSTARAW_GLCM_Normalize,
    "INSTARAW_LBP_Normalize": INSTARAW_LBP_Normalize,
    "INSTARAW_Metadata_Inspector": INSTARAW_Metadata_Inspector,
    **TEXTURE_ENGINE_MAPPINGS,
    "INSTARAW_Spectral_Normalizer": INSTARAW_Spectral_Normalizer,
    "INSTARAW_Pixel_Perturb": INSTARAW_Pixel_Perturb,
    "INSTARAW_BlendColors": INSTARAW_BlendColors,
    "INSTARAW_Camera_Simulator": INSTARAW_Camera_Simulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SEED_DISPLAY_MAPPINGS,
    **MASK_DISPLAY_MAPPINGS,
    **FEATHER_DISPLAY_MAPPINGS,
    **RESOLUTION_GUARD_DISPLAY_MAPPINGS,
    **PROVIDER_DISPLAY_MAPPINGS,
    **NOISE_DISPLAY_MAPPINGS,
    **JPEG_DISPLAY_MAPPINGS,
    **WORKFLOW_LOGIC_DISPLAY_MAPPINGS,
    "INSTARAW_BatchFromImageList": "🗃️ INSTARAW Batch From Image List",
    "INSTARAW_ImageListFromBatch": "📑 INSTARAW Image List From Batch",
    "INSTARAW_PickFromList": "👉 INSTARAW Pick From List",
    "INSTARAW_StringListFromStrings": "📑 INSTARAW String List",
    "INSTARAW_SplitByCommas": "🔪 INSTARAW Split String",
    "INSTARAW_StringToFloat": "→FLOAT INSTARAW String To Float",
    "INSTARAW_StringToInt": "→INT INSTARAW String To Int",
    "INSTARAW_AnyListToString": "→STRING INSTARAW List To String",
    "INSTARAW_MaskedSection": "🖼️ INSTARAW Masked Section",
    "INSTARAW_MaskCombine": "➕ INSTARAW Mask Combine",
    "INSTARAW_StringCombine": "✍️ INSTARAW String Combine (Safe)",
    **BRANDING_DISPLAY_MAPPINGS,
    **RESIZE_DISPLAY_MAPPINGS,
    **MODEL_SELECTOR_DISPLAY_MAPPINGS,
    **JSON_DISPLAY_MAPPINGS,
    "INSTARAW_LUT_Selector": "🎨 INSTARAW LUT Selector",
    **SPECTRAL_ENGINE_DISPLAY_MAPPINGS,
    **COLOR_SCIENCE_DISPLAY_MAPPINGS,
    "INSTARAW_AutoWhiteBalance": "🎨 INSTARAW Auto White Balance",
    **NEURAL_GRAIN_DISPLAY_MAPPINGS,
    **LENS_EFFECTS_DISPLAY_MAPPINGS,
    **COMPRESSION_DISPLAY_MAPPINGS,
    "INSTARAW_AuthenticityProfile_Selector": "👑 INSTARAW Authenticity Profile",
    "INSTARAW_FFT_Match": "🛡️ INSTARAW FFT Match",
    "INSTARAW_GLCM_Normalize": "🛡️ INSTARAW GLCM Normalize",
    "INSTARAW_LBP_Normalize": "🛡️ INSTARAW LBP Normalize",
    "INSTARAW_Metadata_Inspector": "📊 INSTARAW Metadata Inspector",
    **TEXTURE_ENGINE_DISPLAY_MAPPINGS,
    "INSTARAW_Spectral_Normalizer": "🛡️ INSTARAW Spectral Normalizer",
    "INSTARAW_Pixel_Perturb": "🛡️ INSTARAW Pixel Perturb",
    "INSTARAW_BlendColors": "🛡️ INSTARAW Blend Colors",
    "INSTARAW_Camera_Simulator": "🛡️ INSTARAW Camera Simulator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
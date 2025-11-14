# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/authenticity_profile_selector.py
import os

class INSTARAW_AuthenticityProfile_Selector:
    """
    Dynamically finds and lists all .npz authenticity profiles from the internal
    'modules/authenticity_profiles' directory. Outputs the base path to the selected profile.
    """
    
    NODE_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    INSTARAW_ROOT_PATH = os.path.abspath(os.path.join(NODE_FILE_PATH, "..", ".."))
    PROFILES_DIR = os.path.join(INSTARAW_ROOT_PATH, "modules", "authenticity_profiles")

    PROFILE_FILES = []
    if os.path.isdir(PROFILES_DIR):
        for f in os.listdir(PROFILES_DIR):
            if f.lower().endswith('.npz'):
                # FIX: Store only the base name (without extension) for the dropdown
                PROFILE_FILES.append(os.path.splitext(f)[0])
    PROFILE_FILES.sort()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.PROFILE_FILES:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": "ERROR: No .npz profiles found in modules/authenticity_profiles/",
                        "multiline": True
                    })
                }
            }
            
        return {
            "required": {
                "profile_name": (cls.PROFILE_FILES, {"default": cls.PROFILE_FILES[0] if cls.PROFILE_FILES else ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("profile_path",)
    FUNCTION = "get_path"
    CATEGORY = "INSTARAW/Authenticity"

    def get_path(self, profile_name):
        # FIX: Return the base path without the extension.
        # The consuming nodes (FFT_Match, SaveWithMetadata, etc.) will append the correct extension.
        base_path = os.path.join(self.PROFILES_DIR, profile_name)
        
        # We check for the .npz file's existence here to be safe.
        if not os.path.exists(f"{base_path}.npz"):
            raise FileNotFoundError(f"Selected authenticity profile .npz file could not be found: {base_path}.npz")
            
        print(f"👑 INSTARAW Profile Selector: Selected profile base '{profile_name}'")
        return (base_path,)
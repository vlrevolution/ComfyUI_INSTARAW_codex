# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/authenticity_profile_selector.py
import os

# Locate the root of the custom node package to find the profiles directory
NODE_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
INSTARAW_ROOT_PATH = os.path.abspath(os.path.join(NODE_FILE_PATH, "..", ".."))
PROFILES_DIR = os.path.join(INSTARAW_ROOT_PATH, "modules", "authenticity_profiles")

class INSTARAW_AuthenticityProfile_Selector:
    """
    Dynamically finds and lists all .npz authenticity profiles from the internal
    'modules/authenticity_profiles' directory. Outputs the full path to the selected profile.
    """
    
    PROFILE_FILES = []
    if os.path.isdir(PROFILES_DIR):
        for f in os.listdir(PROFILES_DIR):
            if f.lower().endswith('.npz'):
                PROFILE_FILES.append(f)
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
    CATEGORY = "INSTARAW/Utils"

    def get_path(self, profile_name):
        full_path = os.path.join(self.PROFILES_DIR, profile_name)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Selected authenticity profile could not be found: {full_path}")
            
        print(f"✅ INSTARAW Profile Selector: Providing path for '{profile_name}'")
        return (full_path,)
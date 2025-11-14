# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/metadata_inspector.py
import os
import json
import shutil

try:
    import exiftool
    PYEXIFTOOL_AVAILABLE = True
except ImportError:
    PYEXIFTOOL_AVAILABLE = False

# --- CORRECTED BLACKLIST ---
# We only blacklist tags that are truly irrelevant to the image's authenticity,
# such as file system timestamps or the version of ExifTool itself.
METADATA_BLACKLIST = [
    # File-system specific tags that change every time the file is saved
    'SourceFile',
    'File:FileName',
    'File:Directory',
    'File:FileModifyDate',
    'File:FileAccessDate',
    'File:FileInodeChangeDate',
    'File:FileSize',
    'File:FilePermissions',
    # Tool-specific tag
    'ExifTool:ExifToolVersion',
]

class INSTARAW_Metadata_Inspector:
    """
    Inspects an image file and outputs all of its metadata as a formatted
    JSON string, along with a count of the total tags found. This is the 
    definitive tool for verifying the 'Save With Authentic Metadata' node.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("metadata_json", "tag_count",)
    FUNCTION = "inspect_metadata"
    CATEGORY = "INSTARAW/Authenticity"

    def inspect_metadata(self, filepath):
        if not PYEXIFTOOL_AVAILABLE:
            raise ImportError("pyexiftool is required. Please run 'pip install pyexiftool'.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Inspector could not find file: {filepath}")

        print(f"📊 INSTARAW Metadata Inspector: Analyzing file '{os.path.basename(filepath)}'...")
        
        clean_metadata = {}
        try:
            with exiftool.ExifToolHelper() as et:
                # Use '-g' to get group names, which is what we've been using implicitly.
                # Use '-b' to get binary data as "(Binary data ...)" instead of base64.
                raw_meta_list = et.get_metadata(filepath)
                if raw_meta_list:
                    raw_meta = raw_meta_list[0]
                    for key, value in raw_meta.items():
                        if key not in METADATA_BLACKLIST:
                            clean_metadata[key] = value
        except Exception as e:
            error_message = f"ERROR: pyexiftool failed for {filepath}: {e}"
            print(error_message)
            return (error_message, 0)

        total_count = len(clean_metadata)
        icc_count = 0
        for key in clean_metadata.keys():
            if key.startswith("ICC_Profile:"):
                icc_count += 1
        other_count = total_count - icc_count
        
        log_summary = f"({total_count} tags total: {icc_count} ICC, {other_count} other)"

        formatted_json = json.dumps(clean_metadata, indent=2, ensure_ascii=False)

        print("\n" + "="*25 + " INSTARAW Inspector Log " + "="*24)
        print(f"--- METADATA FOUND IN {os.path.basename(filepath)} {log_summary} ---")
        print(formatted_json)
        print("="*70 + "\n")
        
        return (formatted_json, total_count)

NODE_CLASS_MAPPINGS = { "INSTARAW_Metadata_Inspector": INSTARAW_Metadata_Inspector }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_Metadata_Inspector": "📊 INSTARAW Metadata Inspector" }
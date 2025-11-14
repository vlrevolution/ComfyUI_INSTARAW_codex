# Filename: ComfyUI_INSTARAW/nodes/output_nodes/save_with_metadata.py
import torch
import numpy as np
from PIL import Image
import os
import json
import random
import folder_paths
import tempfile
import shutil
import uuid
import subprocess
import sys

def is_tool(name):
    return shutil.which(name) is not None

EXIFTOOL_AVAILABLE = is_tool("exiftool") or is_tool("exiftool.exe")

class INSTARAW_SaveWithAuthenticMetadata:
    OUTPUT_NODE = False 
    CATEGORY = "INSTARAW/Authenticity"
    FUNCTION = "save_image"

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "image": ("IMAGE",), "filename_prefix": ("STRING", {"default": "INSTARAW_Authentic"}), }, "optional": { "profile_path": ("STRING", {"forceInput": True}), } }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)

    def save_image(self, image: torch.Tensor, filename_prefix: str, profile_path=None):
        if not EXIFTOOL_AVAILABLE:
            raise Exception("ExifTool is not installed or not in your system's PATH. Please install it from https://exiftool.org/.")

        if image.shape[0] > 1:
            print("⚠️ INSTARAW Save: Input batch contains more than one image. Only the first will be processed.")

        img_tensor = image[0:1]
        img_np = img_tensor.squeeze(0).cpu().numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, img_pil.width, img_pil.height)
        
        selected_profile = None
        json_path = None
        if profile_path and profile_path.strip():
            json_path = f"{profile_path}.json" if not profile_path.endswith('.json') else profile_path
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata_library = json.load(f)
                if metadata_library:
                    selected_profile = random.choice(metadata_library)
        
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, f"instaraw_img_{uuid.uuid4()}.jpg")
        img_pil.save(temp_filepath, quality=95, format='JPEG')
        
        arg_filepath = None
        final_filepath = ""
        try:
            if selected_profile:
                print("\n" + "="*25 + " INSTARAW Save Log " + "="*25)

                icc_profile_key = "_instaraw_icc_profile_file"
                has_icc_profile = icc_profile_key in selected_profile
                
                # --- DEFINITIVE LOGGING LOGIC ---
                tags_to_write = {k: v for k, v in selected_profile.items() if not k.startswith(('File:', 'Composite:', 'JFIF:', '_instaraw'))}
                total_profile_keys = len(selected_profile)
                writable_count = len(tags_to_write)
                icc_count = 1 if has_icc_profile else 0
                skipped_count = total_profile_keys - writable_count - icc_count
                
                log_message = f"Injecting metadata: {writable_count} writable tags, {skipped_count} skipped (read-only/derived), {icc_count} ICC Profile..."
                print(log_message)
                # --- END LOGGING LOGIC ---

                print("--- PROFILE TO INJECT ---")
                print(json.dumps(selected_profile, indent=2, ensure_ascii=False))
                print("="*70)

                exiftool_cmd = shutil.which("exiftool") or shutil.which("exiftool.exe")
                command = [exiftool_cmd]
                
                icc_profile_path = None
                if has_icc_profile:
                    icc_filename = selected_profile.pop(icc_profile_key) # Use pop on the original dict
                    if json_path:
                        profile_dir = os.path.dirname(json_path)
                        full_icc_path = os.path.join(profile_dir, icc_filename)
                        if os.path.exists(full_icc_path):
                            icc_profile_path = full_icc_path
                            print(f"   - Found ICC Profile to inject: {full_icc_path}")
                        else:
                            print(f"   - ⚠️ WARNING: ICC Profile file not found at {full_icc_path}")
                
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix=".txt", delete=False) as arg_file:
                    arg_filepath = arg_file.name
                    for key, value in tags_to_write.items():
                        arg_file.write(f"-{key}={value}\n")
                
                command.extend(["-@", arg_filepath])

                if icc_profile_path:
                    command.append(f"-ICC_Profile<={icc_profile_path}")

                command.extend([
                    "-overwrite_original",
                    "-ignoreMinorErrors",
                    temp_filepath
                ])
                
                print(f"   - Executing ExifTool command...")
                result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')

                if result.returncode != 0:
                    print(f"❌ INSTARAW Save: ExifTool process failed. Stderr: {result.stderr.strip()}")
                elif "1 image files updated" not in result.stdout:
                     print(f"⚠️ INSTARAW Save: ExifTool ran but may not have updated the file. Full output:")
                     print(f"   - Stdout: {result.stdout.strip()}")
                     print(f"   - Stderr: {result.stderr.strip()}")
                else:
                    print(f"   - ExifTool Success: {result.stdout.strip()}")

            file = f"{filename}_{counter:05}_.jpg"
            final_filepath = os.path.join(full_output_folder, file)
            shutil.move(temp_filepath, final_filepath)
            
        finally:
            if os.path.exists(temp_filepath): os.remove(temp_filepath)
            if arg_filepath and os.path.exists(arg_filepath): os.remove(arg_filepath)

        results = [{"filename": os.path.basename(final_filepath), "subfolder": subfolder, "type": "output"}]
        return {"ui": {"images": results}, "result": (final_filepath,)}

NODE_CLASS_MAPPINGS = { "INSTARAW_SaveWithAuthenticMetadata": INSTARAW_SaveWithAuthenticMetadata }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_SaveWithAuthenticMetadata": "💾 INSTARAW Save With Authentic Metadata" }
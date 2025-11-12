# ---
# ComfyUI INSTARAW - NSFW Detector Node (v11.0 - UI Polish)
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import torch
import numpy as np
import json
import onnxruntime
import os
import requests
from tqdm import tqdm
from .nsfw_core import preprocess_image, postprocess_detections
import folder_paths
from typing_extensions import TypedDict

# (Helper functions setup_nsfw_models_folder and download_model_if_missing remain the same)
def setup_nsfw_models_folder():
    model_folder = "nsfw_models"
    if model_folder not in folder_paths.folder_names_and_paths:
        nsfw_model_path = os.path.join(folder_paths.models_dir, model_folder)
        os.makedirs(nsfw_model_path, exist_ok=True)
        folder_paths.add_model_folder_path(model_folder, nsfw_model_path)
setup_nsfw_models_folder()
def download_model_if_missing(model_name="nudenet.onnx"):
    model_folder = "nsfw_models"
    model_path = folder_paths.get_full_path(model_folder, model_name)
    if model_path and os.path.exists(model_path): return
    print(f"💨 INSTARAW NSFW: Model '{model_name}' not found. Downloading...")
    target_dir = os.path.join(folder_paths.models_dir, model_folder)
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, model_name)
    url = "https://d2xl8ijk56kv4u.cloudfront.net/models/nudenet.onnx"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f, tqdm(desc=model_name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024): f.write(data); bar.update(len(data))
        print(f"✅ INSTARAW NSFW: Model downloaded to {model_path}")
    except Exception as e:
        print(f"❌ INSTARAW NSFW: Failed to download model. Error: {e}")
        if os.path.exists(model_path): os.remove(model_path)
        raise e

class INSTARAW_NSFW_ModelLoader:
    class ModelLoader(TypedDict):
        session: onnxruntime.InferenceSession; input_width: int; input_name: str
    @classmethod
    def INPUT_TYPES(cls):
        try: download_model_if_missing()
        except Exception: pass
        return {"required": {"model_name": (folder_paths.get_filename_list("nsfw_models"),)}}
    RETURN_TYPES = ("NSFW_MODEL",); FUNCTION = "load_model"; CATEGORY = "INSTARAW/NSFW"
    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("nsfw_models", model_name)
        if not model_path or not os.path.exists(model_path):
            download_model_if_missing(model_name)
            model_path = folder_paths.get_full_path("nsfw_models", model_name)
            if not model_path: raise FileNotFoundError(f"NSFW model '{model_name}' not found.")
        try: session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception: session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        model_inputs = session.get_inputs()
        return (self.ModelLoader(session=session, input_width=model_inputs[0].shape[2], input_name=model_inputs[0].name),)

class INSTARAW_CensorLabels:
    # --- THIS IS THE REBRANDING LOGIC ---
    # Technical names required by the model
    ID_MAP = {0: "FEMALE_GENITALIA_COVERED", 1: "FACE_FEMALE", 2: "BUTTOCKS_EXPOSED", 3: "FEMALE_BREAST_EXPOSED", 4: "FEMALE_GENITALIA_EXPOSED", 5: "MALE_BREAST_EXPOSED", 6: "ANUS_EXPOSED", 7: "FEET_EXPOSED", 8: "BELLY_COVERED", 9: "FEET_COVERED", 10: "ARMPITS_COVERED", 11: "ARMPITS_EXPOSED", 12: "FACE_MALE", 13: "BELLY_EXPOSED", 14: "MALE_GENITALIA_EXPOSED", 15: "ANUS_COVERED", 16: "FEMALE_BREAST_COVERED", 17: "BUTTOCKS_COVERED"}
    LABEL_TO_ID = {v: k for k, v in ID_MAP.items()}

    # New, UI-friendly display names
    DISPLAY_NAMES = {
        "ANUS_COVERED": "Anus (Covered)", "ANUS_EXPOSED": "Anus (Exposed)",
        "ARMPITS_COVERED": "Armpits (Covered)", "ARMPITS_EXPOSED": "Armpits (Exposed)",
        "BELLY_COVERED": "Belly (Covered)", "BELLY_EXPOSED": "Belly (Exposed)",
        "BUTTOCKS_COVERED": "Buttocks (Covered)", "BUTTOCKS_EXPOSED": "Buttocks (Exposed)",
        "FACE_FEMALE": "Face (Female)", "FACE_MALE": "Face (Male)",
        "FEET_COVERED": "Feet (Covered)", "FEET_EXPOSED": "Feet (Exposed)",
        "FEMALE_BREAST_COVERED": "Female Breast (Covered)", "FEMALE_BREAST_EXPOSED": "Female Breast (Exposed)",
        "FEMALE_GENITALIA_COVERED": "Female Genitalia (Covered)", "FEMALE_GENITALIA_EXPOSED": "Female Genitalia (Exposed)",
        "MALE_BREAST_EXPOSED": "Male Breast (Exposed)", "MALE_GENITALIA_EXPOSED": "Male Genitalia (Exposed)",
    }
    # Reverse mapping to get the technical name from the display name
    NAME_TO_LABEL = {v: k for k, v in DISPLAY_NAMES.items()}
    
    @classmethod
    def INPUT_TYPES(cls):
        # Use the sorted display names for a clean UI
        sorted_display_names = sorted(cls.DISPLAY_NAMES.values())
        return {"required": { name: ("BOOLEAN", {"default": False}) for name in sorted_display_names}}

    RETURN_TYPES = ("CENSOR_LABELS",); FUNCTION = "get_censor_labels"; CATEGORY = "INSTARAW/NSFW"
    
    def get_censor_labels(self, **kwargs):
        # Logic is now: Censor if the toggle is ON (True)
        censor_ids = []
        for display_name, should_censor in kwargs.items():
            if should_censor:
                technical_label = self.NAME_TO_LABEL[display_name]
                censor_ids.append(self.LABEL_TO_ID[technical_label])
        return (censor_ids,)

class INSTARAW_NSFW_Detector:
    CLASS_MAP = INSTARAW_CensorLabels.ID_MAP
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "image": ("IMAGE",), "nsfw_model": ("NSFW_MODEL",), "censor_labels": ("CENSOR_LABELS",), "min_score": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}), } }
    RETURN_TYPES = ("STRING", "MASK", "STRING",); RETURN_NAMES = ("censored_detections_json", "detection_mask", "all_detections_json",); FUNCTION = "detect"; CATEGORY = "INSTARAW/NSFW"
    
    def detect(self, image, nsfw_model, censor_labels, min_score):
        image_np_float = image[0].cpu().numpy()
        preprocessed, factor, pad_l, pad_t = preprocess_image(image_np_float, nsfw_model["input_width"])
        outputs = nsfw_model["session"].run(None, {nsfw_model["input_name"]: preprocessed})
        all_detections = postprocess_detections(outputs, factor, pad_l, pad_t, min_score)
        
        all_detections_labeled = [{'label': self.CLASS_MAP.get(d.get("id"), "UNKNOWN"), **d} for d in all_detections]
        
        # Censor a detection if its ID IS in the censor_labels list.
        censored_detections = [d for d in all_detections_labeled if d.get('id') in censor_labels]
        
        _b, h, w, _c = image.shape
        mask_tensor = torch.zeros((h, w), dtype=torch.float32, device=image.device)
        for d in censored_detections:
            box = d["box"]; x, y, w_box, h_box = box[0], box[1], box[2], box[3]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(int(w), x + w_box), min(int(h), y + h_box)
            mask_tensor[y1:y2, x1:x2] = 1.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return (json.dumps(censored_detections, indent=2), mask_tensor, json.dumps(all_detections_labeled, indent=2),)

NODE_CLASS_MAPPINGS = { "INSTARAW_NSFW_ModelLoader": INSTARAW_NSFW_ModelLoader, "INSTARAW_CensorLabels": INSTARAW_CensorLabels, "INSTARAW_NSFW_Detector": INSTARAW_NSFW_Detector, }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_NSFW_ModelLoader": "💿 INSTARAW NSFW Model Loader", "INSTARAW_CensorLabels": "✅ INSTARAW Censor Labels", "INSTARAW_NSFW_Detector": "⚠️ INSTARAW NSFW Detector", }
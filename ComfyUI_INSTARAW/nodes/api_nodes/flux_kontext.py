# ---
# ComfyUI INSTARAW - Flux Kontext LoRA API Node
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import requests
import base64
import io
import json
import hashlib
import os
from PIL import Image
import numpy as np
import torch

# =================================================================================
# BASE CLASS FOR FAL.AI SYNC REQUESTS
# =================================================================================

class FalAIBase:
    def __init__(self):
        self.api_key = None

    def set_api_key(self, api_key):
        if not api_key or not api_key.strip():
            raise ValueError("fal.ai API key is missing or empty.")
        self.api_key = api_key

    def _log_and_update_hash(self, hasher, key, value):
        if value is None: return
        byte_value = b""
        if isinstance(value, str): byte_value = value.encode("utf-8")
        elif isinstance(value, (int, float, bool)): byte_value = str(value).encode("utf-8")
        elif isinstance(value, torch.Tensor): byte_value = value.cpu().numpy().tobytes()
        else: return
        hasher.update(byte_value)

    def image_to_base64(self, image_tensor, max_size_mb=10):
        """Converts a tensor to a base64 data URI and compresses it to stay under API limits."""
        if image_tensor is None: return None
        
        image_np = image_tensor.cpu().numpy()[0]
        if image_np.max() <= 1.0: image_np = (image_np * 255).astype(np.uint8)
        
        image_pil = Image.fromarray(image_np).convert("RGB")
        
        quality_levels = [95, 90, 85, 80, 75]
        max_bytes = max_size_mb * 1024 * 1024

        buffer = io.BytesIO()
        for quality in quality_levels:
            buffer.seek(0)
            buffer.truncate(0)
            image_pil.save(buffer, format="JPEG", quality=quality)
            
            if buffer.tell() <= max_bytes:
                print(f"✅ Image compressed to JPEG (quality={quality}) to fit API limits. Size: {buffer.tell() / 1024:.2f} KB")
                base64_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{base64_str}"

        final_size_kb = buffer.tell() / 1024
        raise Exception(f"Image is too large for API. After compressing to lowest quality JPEG, size is {final_size_kb:.2f} KB (limit is {max_size_mb * 1024:.2f} KB).")

    def _submit_fal_sync(self, endpoint, payload):
        """Submits a synchronous request to a fal.ai endpoint."""
        url = f"https://fal.run/{endpoint}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        print(f"🚀 Submitting SYNC request to fal.ai: {url}")
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=600) # 10 minute timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            print("📦 fal.ai sync response received.")
            
            if "images" in result and len(result["images"]) > 0 and "url" in result["images"][0]:
                return result["images"][0]["url"]
            else:
                raise Exception(f"API response did not contain a valid image URL. Full response: {json.dumps(result, indent=2)}")

        except requests.exceptions.HTTPError as e:
            error_message = f"API request failed: {e.response.status_code} - {e.response.text}"
            print(f"❌ {error_message}")
            raise Exception(error_message) from e
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            raise

# =================================================================================
# FLUX KONTEXT LORA API NODE
# =================================================================================

class INSTARAW_FluxKontextLoraAPI(FalAIBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "make the woman's breasts larger and her hips wider"}),
                "lora_url": ("STRING", {"default": "https://huggingface.co/BAZILEVS-BASED/kontext_big_breasts_and_butts/resolve/main/kontext_big_breasts_and_butts.safetensors"}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/API"

    def execute(self, api_key, image, prompt, lora_url, lora_scale, steps, guidance_scale, seed, enable_safety_checker):
        # --- Caching Logic ---
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        hasher = hashlib.sha256()
        all_args = locals()
        for key in sorted(all_args.keys()):
            if key not in ['self', 'api_key']: # Don't include self or api_key in hash
                self._log_and_update_hash(hasher, key, all_args[key])
        
        cache_filepath = os.path.join(cache_dir, f"{hasher.hexdigest()}_flux_kontext.png")

        if os.path.exists(cache_filepath):
            print(f"✅ Flux Kontext API Cache Hit! Loading image from {cache_filepath}")
            pil_image = Image.open(cache_filepath).convert("RGB")
            img_np = np.array(pil_image).astype(np.float32) / 255.0
            return (torch.from_numpy(img_np).unsqueeze(0),)

        print("💨 Flux Kontext API Cache Miss. Proceeding with API call...")

        # --- API Call Logic ---
        self.set_api_key(api_key)
        
        # Convert image to base64 data URI
        base64_image = self.image_to_base64(image)

        # Build the payload
        payload = {
            "image_url": base64_image,
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "enable_safety_checker": enable_safety_checker,
            "loras": [
                {
                    "path": lora_url,
                    "scale": lora_scale
                }
            ],
            # These are fixed for this specific use case but could be exposed as inputs
            "num_images": 1,
            "output_format": "png",
            "resolution_mode": "match_input", 
        }
        if seed >= 0:
            payload["seed"] = seed

        # Submit the request and get the result image URL
        image_url = self._submit_fal_sync("fal-ai/flux-kontext-lora", payload)

        if not image_url:
            raise Exception("API did not return a valid image URL.")

        # Download the resulting image
        print(f"✅ Image generated. Downloading from: {image_url}")
        image_response = requests.get(image_url, timeout=300)
        image_response.raise_for_status()

        # Convert back to tensor and save to cache
        pil_image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        pil_image.save(cache_filepath, "PNG")
        
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return (torch.from_numpy(img_np).unsqueeze(0),)

# =================================================================================
# NODE REGISTRATION
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "INSTARAW_FluxKontextLoraAPI": INSTARAW_FluxKontextLoraAPI,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_FluxKontextLoraAPI": "🍑 INSTARAW Flux Kontext LoRA API",
}
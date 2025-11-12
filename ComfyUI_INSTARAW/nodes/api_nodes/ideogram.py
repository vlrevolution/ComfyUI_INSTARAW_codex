# ---
# ComfyUI INSTARAW - Ideogram Nodes
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

"""
ComfyUI nodes for Ideogram API integration.
Supports image generation, editing, and description using Ideogram v3 API.
"""
import requests
import io
from PIL import Image
import numpy as np
import torch
import hashlib
import os
import base64
import json
from datetime import datetime, timezone, timedelta


# --- List of supported resolutions and aspect ratios for Ideogram V3 ---
IDEOGRAM_V3_RESOLUTIONS = [
    "1024x1024",
    "864x1152",
    "768x1024",
    "512x1536",
    "576x1408",
    "576x1472",
    "576x1536",
    "640x1344",
    "640x1408",
    "640x1472",
    "640x1536",
    "704x1152",
    "704x1216",
    "704x1280",
    "704x1344",
    "704x1408",
    "704x1472",
    "736x1312",
    "768x1088",
    "768x1216",
    "768x1280",
    "768x1344",
    "800x1280",
    "832x960",
    "832x1024",
    "832x1088",
    "832x1152",
    "832x1216",
    "832x1248",
    "896x960",
    "896x1024",
    "896x1088",
    "896x1120",
    "896x1152",
    "960x832",
    "960x896",
    "960x1024",
    "960x1088",
    "1024x832",
    "1024x896",
    "1024x960",
    "1088x768",
    "1088x832",
    "1088x896",
    "1088x960",
    "1120x896",
    "1152x704",
    "1152x832",
    "1152x864",
    "1152x896",
    "1216x704",
    "1216x768",
    "1216x832",
    "1248x832",
    "1280x704",
    "1280x768",
    "1280x800",
    "1312x736",
    "1344x640",
    "1344x704",
    "1344x768",
    "1408x576",
    "1408x640",
    "1408x704",
    "1472x576",
    "1472x640",
    "1472x704",
    "1536x512",
    "1536x576",
    "1536x640",
]
IDEOGRAM_V3_ASPECT_RATIOS = [
    "1x1",
    "9x16",
    "10x16",
    "3x4",
    "4x5",
    "2x3",
    "1x2",
    "1x3",
    "16x9",
    "16x10",
    "4x3",
    "5x4",
    "3x2",
    "2x1",
    "3x1",
]


class IdeogramBase:
    def __init__(self):
        self.api_key = None
        self.base_url = "https://api.ideogram.ai"
        self.fal_cdn_url = "https://v3.fal.media/files/upload"
        self.fal_cdn_auth_url = (
            "https://rest.alpha.fal.ai/storage/auth/token?storage_type=fal-cdn-v3"
        )
        self._fal_cdn_token_cache = {}

    def set_api_key(self, api_key):
        self.api_key = api_key

    def _get_fal_cdn_auth_header(self):
        if "token" in self._fal_cdn_token_cache and self._fal_cdn_token_cache.get(
            "expires_at", datetime.min.replace(tzinfo=timezone.utc)
        ) > datetime.now(timezone.utc) + timedelta(seconds=60):
            token_data = self._fal_cdn_token_cache
        else:
            print("🔐 Fetching new fal.ai CDN token...")
            auth_headers = {
                "Authorization": f"Key {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            response = requests.post(
                self.fal_cdn_auth_url, headers=auth_headers, data=b"{}", timeout=30
            )
            if not response.ok:
                raise Exception(
                    f"Failed to get fal.ai CDN auth token: {response.status_code} - {response.text}"
                )
            token_data = response.json()
            expires_at_str = token_data["expires_at"].replace("Z", "+00:00")
            token_data["expires_at"] = datetime.fromisoformat(expires_at_str)
            self._fal_cdn_token_cache = token_data
            print("✅ New CDN token acquired.")
        return {"Authorization": f"{token_data['token_type']} {token_data['token']}"}

    def tensor_to_bytesio(self, tensor, is_mask=False):
        if tensor is None:
            return None
        image_np = tensor.cpu().numpy()
        if image_np.shape[0] == 1:
            image_np = image_np[0]
        if not is_mask:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np, "RGB")
        else:
            image_np = (image_np > 0.5).astype(np.float32)
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np, "L").convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def image_to_base64(self, image_tensor, is_mask=False):
        if image_tensor is None:
            return None
        bytes_io = self.tensor_to_bytesio(image_tensor, is_mask)
        if bytes_io is None:
            return None
        base64_str = base64.b64encode(bytes_io.getvalue()).decode()
        return f"data:image/png;base64,{base64_str}"

    def _upload_tensor_to_fal_cdn(self, tensor, is_mask=False):
        if tensor is None:
            return None
        print(f"🚀 Uploading {'mask' if is_mask else 'image'} to fal.ai CDN...")
        auth_header = self._get_fal_cdn_auth_header()
        headers = {**auth_header, "Content-Type": "image/png"}
        bytes_io = self.tensor_to_bytesio(tensor, is_mask=is_mask)
        if bytes_io is None:
            return None
        image_bytes = bytes_io.getvalue()
        response = requests.post(
            self.fal_cdn_url, data=image_bytes, headers=headers, timeout=60
        )
        if not response.ok:
            raise Exception(
                f"fal.ai CDN upload failed: {response.status_code} - {response.text}"
            )
        url = response.json()["access_url"]
        print(f"✅ CDN Upload successful. URL: {url}")
        return url

    # --- THIS IS THE FIX: UPGRADED ERROR HANDLING ---
    def _submit_fal(self, endpoint, payload):
        url = f"https://fal.run/{endpoint}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        print(f"🚀 Submitting SYNC request to fal.ai: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=180)

        if not response.ok:
            if response.status_code == 422:
                try:
                    error_data = response.json()
                    error_obj = error_data.get("detail", [{}])[0]

                    if error_obj.get("type") == "content_policy_violation":
                        raise Exception(
                            "Image flagged as unsafe by Ideogram API. Try enabling the 🔞 FULLY CLOTHE FOR NSFW FIX or using a different SWAP TO image."
                        )
                    else:
                        error_msg = error_obj.get("msg", "Unknown validation error.")
                        raise Exception(
                            f"API Error (422): Unprocessable Entity. Details: {error_msg}"
                        )
                except (json.JSONDecodeError, IndexError, KeyError):
                    raise Exception(
                        f"API request failed with 422 (Unprocessable Entity), but the error response was not in the expected format. Raw response: {response.text}"
                    )
            else:
                raise Exception(
                    f"API request failed: {response.status_code} - {response.text}"
                )

        result = response.json()
        print("📦 fal.ai sync response received.")
        if "images" in result and len(result["images"]) > 0:
            return result["images"][0]["url"]
        raise Exception(
            f"API response did not contain an image URL. Full response: {result}"
        )

    # --- END FIX ---

    def submit_multipart_request(self, endpoint, data_payload, files_payload):
        if not self.api_key:
            raise ValueError("API key not set.")
        url = f"{self.base_url}{endpoint}"
        headers = {"Api-Key": self.api_key}
        print(f"🚀 Submitting multipart request to {url}")
        response = requests.post(
            url, headers=headers, data=data_payload, files=files_payload, timeout=180
        )
        if not response.ok:
            raise Exception(
                f"API request failed: {response.status_code} - {response.text}"
            )
        return response.json()


class IdeogramDescribeImage(IdeogramBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "describe_model_version": (["V_3"], {"default": "V_3"}),
            }
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "describe_image", "INSTARAW/API"

    def describe_image(self, api_key, image, describe_model_version):
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        hasher = hashlib.sha256()
        hasher.update(describe_model_version.encode("utf-8"))
        hasher.update(image.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_describe.txt")
        if os.path.exists(cache_filepath):
            with open(cache_filepath, "r", encoding="utf-8") as f:
                return (f.read(),)
        self.set_api_key(api_key)
        image_file = self.tensor_to_bytesio(image)
        data_payload = {"describe_model_version": describe_model_version}
        files_payload = {"image_file": ("image.png", image_file, "image/png")}
        result = self.submit_multipart_request(
            "/describe", data_payload=data_payload, files_payload=files_payload
        )
        if (
            result.get("descriptions")
            and len(result["descriptions"]) > 0
            and "text" in result["descriptions"][0]
        ):
            description = result["descriptions"][0]["text"]
            with open(cache_filepath, "w", encoding="utf-8") as f:
                f.write(description)
            return (description,)
        else:
            raise Exception("API did not return a valid description.")


class IdeogramGenerateImage(IdeogramBase):
    @classmethod
    def INPUT_TYPES(cls):
        magic_prompt_options = ["AUTO", "ON", "OFF"]
        rendering_speed_options = ["QUALITY", "FLASH", "DEFAULT", "TURBO"]
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "A photo of a cat."},
                ),
                "sizing_method": (["By Aspect Ratio", "By Resolution"],),
                "aspect_ratio": (IDEOGRAM_V3_ASPECT_RATIOS, {"default": "1x1"}),
                "resolution": (IDEOGRAM_V3_RESOLUTIONS, {"default": "1024x1024"}),
                "rendering_speed": (rendering_speed_options, {"default": "DEFAULT"}),
                "magic_prompt": (magic_prompt_options, {"default": "AUTO"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "character_reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "generate_image", "INSTARAW/API"

    def generate_image(
        self,
        api_key,
        prompt,
        sizing_method,
        aspect_ratio,
        resolution,
        rendering_speed,
        magic_prompt,
        negative_prompt="",
        seed=-1,
        character_reference_image=None,
    ):
        if magic_prompt in ["AUTO", "OFF"] and not prompt.strip():
            raise ValueError("A non-empty 'prompt' is required.")
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        hasher = hashlib.sha256()
        hasher.update(prompt.encode("utf-8"))
        hasher.update(sizing_method.encode("utf-8"))
        hasher.update(aspect_ratio.encode("utf-8"))
        hasher.update(resolution.encode("utf-8"))
        hasher.update(rendering_speed.encode("utf-8"))
        hasher.update(magic_prompt.encode("utf-8"))
        hasher.update(negative_prompt.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        if character_reference_image is not None:
            hasher.update(character_reference_image.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_ideogram_gen.png")
        if os.path.exists(cache_filepath):
            pil_image = Image.open(cache_filepath).convert("RGB")
            return (
                torch.from_numpy(
                    np.array(pil_image).astype(np.float32) / 255.0
                ).unsqueeze(0),
            )
        self.set_api_key(api_key)
        char_ref_file = self.tensor_to_bytesio(character_reference_image)
        data_payload = {
            "prompt": prompt,
            "rendering_speed": rendering_speed,
            "magic_prompt": magic_prompt,
        }
        if sizing_method == "By Resolution":
            data_payload["resolution"] = resolution
        else:
            data_payload["aspect_ratio"] = aspect_ratio
        if seed >= 0:
            data_payload["seed"] = seed
        if negative_prompt.strip():
            data_payload["negative_prompt"] = negative_prompt
        files_payload = {}
        if char_ref_file:
            files_payload["character_reference_images"] = (
                "char_ref.png",
                char_ref_file,
                "image/png",
            )
            data_payload["style_type"] = "REALISTIC"
        result = self.submit_multipart_request(
            "/v1/ideogram-v3/generate", data_payload, files_payload
        )
        if result.get("data") and len(result["data"]) > 0:
            image_data = result["data"][0]
            if image_data.get("is_image_safe") is False:
                raise Exception("Image generation failed: unsafe result.")
            image_url = image_data.get("url")
            if not image_url:
                raise Exception("API did not provide a valid image URL.")
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            pil_image.save(cache_filepath, "PNG")
            return (
                torch.from_numpy(
                    np.array(pil_image).astype(np.float32) / 255.0
                ).unsqueeze(0),
            )
        else:
            raise Exception("API did not return any images.")


class IdeogramEditImage(IdeogramBase):
    @classmethod
    def INPUT_TYPES(cls):
        magic_prompt_options = ["AUTO", "ON", "OFF"]
        rendering_speed_options = ["QUALITY", "FLASH", "DEFAULT", "TURBO", "BALANCED"]
        return {
            "required": {
                "api_key": ("STRING", {"forceInput": True}),
                "provider": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "rendering_speed": (rendering_speed_options,),
                "magic_prompt": (magic_prompt_options, {"default": "AUTO"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "character_reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "edit_image", "INSTARAW/API"

    def edit_image(
        self,
        api_key,
        provider,
        image,
        mask,
        prompt,
        rendering_speed,
        magic_prompt,
        seed=-1,
        character_reference_image=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError(
                "A non-empty 'prompt' is required for the Ideogram Edit Image node to function correctly."
            )

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        hasher = hashlib.sha256()
        hasher.update(provider.encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        hasher.update(rendering_speed.encode("utf-8"))
        hasher.update(magic_prompt.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        hasher.update(image.cpu().numpy().tobytes())
        hasher.update(mask.cpu().numpy().tobytes())
        if character_reference_image is not None:
            hasher.update(character_reference_image.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_ideogram_edit.png")
        if os.path.exists(cache_filepath):
            print(f"✅ Ideogram Cache Hit! Loading image from {cache_filepath}")
            pil_image = Image.open(cache_filepath).convert("RGB")
            return (
                torch.from_numpy(
                    np.array(pil_image).astype(np.float32) / 255.0
                ).unsqueeze(0),
            )
        print(f"💨 Ideogram Cache Miss. Proceeding with {provider} API call...")
        self.set_api_key(api_key)
        image_url = ""
        if provider == "Official Ideogram":
            print("🎭 Inverting mask for Official Ideogram provider.")
            mask_to_send = 1.0 - mask
            speed = "DEFAULT" if rendering_speed == "BALANCED" else rendering_speed
            data_payload = {
                "prompt": prompt,
                "rendering_speed": speed,
                "magic_prompt": magic_prompt,
            }
            if seed >= 0:
                data_payload["seed"] = seed
            files_payload = {
                "image": ("image.png", self.tensor_to_bytesio(image), "image/png"),
                "mask": (
                    "mask.png",
                    self.tensor_to_bytesio(mask_to_send, is_mask=True),
                    "image/png",
                ),
            }
            if character_reference_image is not None:
                files_payload["character_reference_images"] = (
                    "char_ref.png",
                    self.tensor_to_bytesio(character_reference_image),
                    "image/png",
                )
            result = self.submit_multipart_request(
                "/v1/ideogram-v3/edit", data_payload, files_payload
            )
            if result.get("data") and result["data"]:
                image_data = result["data"][0]
                if image_data.get("is_image_safe") is False:
                    raise Exception(
                        "Image flagged as unsafe by Ideogram API. Try enabling the 🔞 FULLY CLOTHE FOR NSFW FIX or using a different SWAP TO image."
                    )
                image_url = image_data.get("url")
            else:
                raise Exception("Official Ideogram API did not return any images.")

        elif provider == "fal.ai":
            print(
                "🎭 Using fal.ai provider. Uploading all images to CDN to avoid size limits."
            )
            payload = {
                "prompt": prompt,
                "image_url": self._upload_tensor_to_fal_cdn(image),
                "mask_url": self._upload_tensor_to_fal_cdn(mask, is_mask=True),
            }
            if character_reference_image is not None:
                payload["reference_image_urls"] = [
                    self._upload_tensor_to_fal_cdn(character_reference_image)
                ]
                payload["style"] = "REALISTIC"
            speed_map = {
                "QUALITY": "QUALITY",
                "TURBO": "TURBO",
                "FLASH": "TURBO",
                "DEFAULT": "BALANCED",
                "BALANCED": "BALANCED",
            }
            payload["rendering_speed"] = speed_map.get(rendering_speed, "BALANCED")
            payload["expand_prompt"] = magic_prompt in ["ON", "AUTO"]
            if seed >= 0:
                payload["seed"] = seed
            image_url = self._submit_fal("fal-ai/ideogram/character/edit", payload)

        if not image_url:
            raise Exception("API response did not provide a valid image URL.")
        print(f"✅ Image edited. Downloading from: {image_url}")
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        pil_image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        pil_image.save(cache_filepath, "PNG")
        return (
            torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(
                0
            ),
        )


NODE_CLASS_MAPPINGS = {
    "IdeogramDescribeImage": IdeogramDescribeImage,
    "IdeogramGenerateImage": IdeogramGenerateImage,
    "IdeogramEditImage": IdeogramEditImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramDescribeImage": "💭 Ideogram Describe Image",
    "IdeogramGenerateImage": "🖼️ Ideogram Generate Image",
    "IdeogramEditImage": "✏️ Ideogram Edit Image",
}

# ---
# Filename: ../ComfyUI_INSTARAW/nodes/api_nodes/grok_native.py
# Native Grok (xAI) text helper that mirrors the Gemini node UX
# ---

import base64
import hashlib
import io
import json
import os
from typing import List

import requests
from PIL import Image
import numpy as np


class INSTARAW_GrokNative:
	"""
	Text-first Grok helper that talks to the public xAI REST API.
	We keep the surface identical to the Gemini node: simple prompt entry,
	optional temperature + seed, and up to four reference images.
	"""

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"api_key": ("STRING", {"multiline": False, "default": ""}),
				"prompt": ("STRING", {"multiline": True, "default": "Describe the concept or provide instructions."}),
				"model": (
					[
						"grok-4-fast-reasoning",
						"grok-4-fast-non-reasoning",
						"grok-4-0709",
					],
					{"default": "grok-4-fast-reasoning"},
				),
				"system_prompt": (
					"STRING",
					{"multiline": True, "default": "You are a helpful creative assistant."},
				),
				"temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
				"top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
				"seed": (
					"INT",
					{
						"default": 111111,
						"min": 0,
						"max": 0xFFFFFFFFFFFFFFFF,
						"control_after_generate": "randomize",
					},
				),
			},
			"optional": {
				"image_1": ("IMAGE",),
				"image_2": ("IMAGE",),
				"image_3": ("IMAGE",),
				"image_4": ("IMAGE",),
			},
		}

	RETURN_TYPES = ("STRING",)
	FUNCTION = "generate_content"
	CATEGORY = "INSTARAW/API"

	def generate_content(
		self,
		api_key,
		prompt,
		model,
		system_prompt,
		temperature,
		top_p,
		seed,
		image_1=None,
		image_2=None,
		image_3=None,
		image_4=None,
	):
		api_key = (api_key or "").strip() or os.environ.get("XAI_API_KEY", "").strip()
		if not api_key:
			raise ValueError("Grok API Key is missing. Provide it in the node or set XAI_API_KEY env var.")

		images = [img for img in (image_1, image_2, image_3, image_4) if img is not None]
		payload = self._build_payload(prompt, system_prompt, model, temperature, top_p, seed, images)

		cache_hit = self._load_from_cache(payload)
		if cache_hit is not None:
			return (cache_hit,)

		base_url = os.environ.get("XAI_API_BASE", "https://api.x.ai").rstrip("/")
		url = f"{base_url}/v1/chat/completions"

		headers = {
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json",
		}

		response = requests.post(url, headers=headers, json=payload, timeout=300)
		body_text = response.text
		if response.status_code >= 400:
			raise RuntimeError(f"Grok API error {response.status_code}: {body_text}")

		try:
			body = response.json()
		except ValueError as err:
			raise RuntimeError(f"Grok API returned invalid JSON: {err}") from err

		content = self._extract_text(body)
		self._save_to_cache(payload, content)
		return (content,)

	# --- Internals --------------------------------------------------------

	def _build_payload(self, prompt, system_prompt, model, temperature, top_p, seed, images) -> dict:
		messages: List[dict] = []
		if system_prompt and system_prompt.strip():
			messages.append({"role": "system", "content": system_prompt.strip()})

		user_content: List[dict] = []
		if prompt and prompt.strip():
			user_content.append({"type": "text", "text": prompt.strip()})

		for image_tensor in images:
			data_url = self._tensor_to_data_url(image_tensor)
			user_content.append(
				{
					"type": "image_url",
					"image_url": {"url": f"data:image/png;base64,{data_url}", "detail": "high"},
				}
			)

		if not user_content:
			raise ValueError("Provide a prompt or an image for Grok to process.")

		messages.append({"role": "user", "content": user_content})

		seed_value = None
		int32_min, int32_max = -2147483648, 2147483647
		if seed is not None:
			seed_value = int(seed)
			if seed_value < int32_min or seed_value > int32_max:
				seed_value = ((seed_value + 2 ** 31) % (2 ** 32)) - 2 ** 31
			if seed_value <= 0:
				seed_value = (abs(seed_value) % int32_max) + 1
		if seed_value is None:
			seed_value = self._derive_seed_from_payload(prompt, system_prompt, model)

		payload = {
			"model": model,
			"messages": messages,
			"temperature": float(temperature),
			"top_p": float(top_p),
		}
		payload["seed"] = seed_value
		return payload

	def _derive_seed_from_payload(self, prompt: str, system_prompt: str, model: str) -> int:
		int32_max = 2147483647
		hasher = hashlib.sha256()
		hasher.update((prompt or "").encode("utf-8"))
		hasher.update((system_prompt or "").encode("utf-8"))
		hasher.update((model or "").encode("utf-8"))
		digest = hasher.digest()
		value = int.from_bytes(digest[:4], byteorder="big", signed=True)
		if value <= 0:
			value = (abs(value) % int32_max) + 1
		return min(value, int32_max)

	def _tensor_to_data_url(self, tensor) -> str:
		if hasattr(tensor, "detach"):
			array = tensor.detach().cpu().numpy()
		else:
			array = np.asarray(tensor)

		if array.ndim == 4:
			array = array[0]
		if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4):
			array = np.transpose(array, (1, 2, 0))

		array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
		image = Image.fromarray(array)
		buffer = io.BytesIO()
		image.save(buffer, format="PNG")
		encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
		return encoded

	def _extract_text(self, body: dict) -> str:
		choices = body.get("choices") or []
		if not choices:
			raise RuntimeError("Grok API returned no choices.")

		first = choices[0]
		message = first.get("message") or {}
		content = message.get("content")
		if isinstance(content, list):
			text = "".join(
				part.get("text", "") if isinstance(part, dict) else str(part)
				for part in content
			).strip()
			if text:
				return text
		if isinstance(content, str) and content.strip():
			return content.strip()

		legacy = first.get("text")
		if legacy:
			return legacy.strip()

		raise RuntimeError("Unable to extract text from Grok response.")

	def _cache_path(self) -> str:
		cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
		os.makedirs(cache_dir, exist_ok=True)
		return cache_dir

	def _payload_hash(self, payload: dict) -> str:
		data = json.dumps(payload, sort_keys=True).encode("utf-8")
		return hashlib.sha256(data).hexdigest()

	def _cache_file(self, payload: dict) -> str:
		return os.path.join(self._cache_path(), f"{self._payload_hash(payload)}_grok_native.txt")

	def _load_from_cache(self, payload: dict):
		path = self._cache_file(payload)
		if os.path.exists(path):
			try:
				with open(path, "r", encoding="utf-8") as handle:
					return handle.read()
			except OSError:
				return None
		return None

	def _save_to_cache(self, payload: dict, text: str):
		path = self._cache_file(payload)
		try:
			with open(path, "w", encoding="utf-8") as handle:
				handle.write(text)
		except OSError:
			pass


NODE_CLASS_MAPPINGS = {
	"INSTARAW_GrokNative": INSTARAW_GrokNative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"INSTARAW_GrokNative": "🧠 INSTARAW Grok (Native)",
}

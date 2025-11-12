# --- Filename: ../ComfyUI_INSTARAW/nodes/api_nodes/gemini_native.py (FINAL CORRECTED VERSION) ---

import os
import hashlib
import torch
import numpy as np
from PIL import Image
import io

# We will no longer rely on a global flag. The check will be done at runtime.

class INSTARAW_GeminiNative:
    """
    A native, stable, and fully-featured Google Gemini node for INSTARAW.
    This version supports up to 4 separate image inputs for a user-friendly workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # We now attempt the import here to define inputs. If it fails, we show an error.
        try:
            import google.genai
            import google.genai.types as types
            return {
                "required": {
                    "api_key": ("STRING", {"multiline": False, "default": ""}),
                    "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                    "model": (["gemini-2.5-pro", "gemini-flash-latest"], {"default": "gemini-2.5-pro"}),
                    "seed": ("INT", {"default": 1111111, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                    "temperature": ("FLOAT", {"default": 0.111, "min": 0.0, "max": 2.0, "step": 0.001}),
                    "enable_thinking": ("BOOLEAN", {"default": True, "label_on": "Thinking Enabled", "label_off": "Thinking Disabled"}),
                    "safety_level": (["Block None", "Block Few (High Only)", "Block Some (Medium+)", "Block Most (Low+)"], {"default": "Block None"}),
                },
                "optional": { "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",) }
            }
        except ImportError:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": "google-genai library failed to import. Please run 'pip install -U google-genai' and restart.",
                        "multiline": True
                    })
                }
            }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_content"
    CATEGORY = "INSTARAW/API"

    def generate_content(self, api_key, prompt, model, seed, temperature, enable_thinking, safety_level, 
                         image_1=None, image_2=None, image_3=None, image_4=None):
        
        # --- THE DEFINITIVE FIX: Perform the import check HERE, at runtime. ---
        try:
            from google import genai
            from google.genai import types
            from google.genai import errors
            from google.api_core import exceptions as core_exceptions
        except ImportError:
            # This is now the only place the error is raised, ensuring it's a real-time failure.
            raise ImportError("The 'google-genai' library is required. Please run 'pip install -U google-genai' to ensure you have the latest version.")
        
        if not api_key or api_key.strip() == "":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key or api_key.strip() == "":
                raise ValueError("Gemini API Key is missing. Provide it in the node or set GEMINI_API_KEY env var.")

        provided_images = [img for img in [image_1, image_2, image_3, image_4] if img is not None]

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        hasher = hashlib.sha256()
        hasher.update(str(seed).encode('utf-8')); hasher.update(prompt.encode('utf-8')); hasher.update(model.encode('utf-8'))
        hasher.update(f"{temperature:.3f}".encode('utf-8')); hasher.update(str(enable_thinking).encode('utf-8')); hasher.update(safety_level.encode('utf-8'))
        for image_tensor in provided_images: hasher.update(image_tensor.cpu().numpy().tobytes())
        cache_key = hasher.hexdigest()
        cache_filepath = os.path.join(cache_dir, f"{cache_key}_gemini_native.txt")

        if os.path.exists(cache_filepath):
            with open(cache_filepath, 'r', encoding='utf-8') as f: return (f.read(),)

        try:
            client = genai.Client(api_key=api_key)
            parts = [types.Part.from_text(text=prompt)]
            for image_tensor in provided_images:
                pil_image = Image.fromarray((image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
                buffer = io.BytesIO(); pil_image.save(buffer, format="PNG")
                parts.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type='image/png'))

            contents = [types.Content(role="user", parts=parts)]
            thinking_budget = -1 if model == "gemini-2.5-pro" or enable_thinking else 0
            
            safety_map = { "Block None": "BLOCK_NONE", "Block Few (High Only)": "BLOCK_ONLY_HIGH", "Block Some (Medium+)": "BLOCK_MEDIUM_AND_ABOVE", "Block Most (Low+)": "BLOCK_LOW_AND_ABOVE" }
            safety_threshold = safety_map.get(safety_level, "BLOCK_MEDIUM_AND_ABOVE")
            safety_settings = [types.SafetySetting(category=cat, threshold=safety_threshold) for cat in ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT"]]

            config = types.GenerateContentConfig(temperature=temperature, candidate_count=1, thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget), safety_settings=safety_settings)
            
            response = client.models.generate_content(model=f"models/{model}", contents=contents, config=config)
            
            if not response.candidates:
                raise Exception(f"Prompt blocked by safety filters. Reason: {response.prompt_feedback.block_reason.name}" if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason else "API returned no candidates (likely blocked by safety filters).")

            result_text = response.text
            with open(cache_filepath, 'w', encoding='utf-8') as f: f.write(result_text)
            return (result_text,)

        except core_exceptions.ResourceExhausted as e:
            raise Exception("INSTARAW Gemini Error: Rate limit exceeded (429). Enable billing on your Google AI Studio account or wait and try again.") from e
        except errors.ServerError as e:
            raise Exception("INSTARAW Gemini Error: Model overloaded or unavailable (503). Enable billing for a higher quota or try again later.") from e
        except Exception as e:
            raise e

NODE_CLASS_MAPPINGS = { "INSTARAW_GeminiNative": INSTARAW_GeminiNative }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_GeminiNative": "🧠 INSTARAW Gemini (Native)" }
# ---
# Filename: ../combined_test_codex/ComfyUI_INSTARAW/nodes/api_nodes/creative_api.py
# ---

# ---
# Filename: ../ComfyUI_INSTARAW/creative_api.py
# Creative Prompt Generation API - Gemini & Grok Integration
# ---

"""
Backend API endpoint for creative prompt generation using Gemini and Grok APIs.
Supports both inspiration-based generation and character-consistent generation.
"""

import os
import json
import hashlib
import aiohttp
from aiohttp import web
from server import PromptServer

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


# === Gemini Integration ===
async def generate_with_gemini(system_prompt, user_prompt, model="gemini-2.5-pro", api_key=None, temperature=0.9, top_p=0.9):
    """
    Generate creative prompts using Google Gemini API.
    Returns a list of {positive, negative, tags} dictionaries.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("The 'google-genai' library is required. Run: pip install -U google-genai")

    # Use provided API key or fall back to environment variable
    if not api_key or api_key.strip() == "":
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("Gemini API Key is missing. Provide it in the node or set GEMINI_API_KEY env var.")

    try:
        # Use the async client for non-blocking calls
        genai.configure(api_key=api_key)
        async_client = genai.GenerativeModel(model_name=model)

        # Configure generation
        generation_config = types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            candidate_count=1,
        )
        safety_settings=[
            types.SafetySetting(category=cat, threshold="BLOCK_NONE")
            for cat in ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT",
                       "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT"]
        ]

        response = await async_client.generate_content_async(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if not response.candidates:
            raise Exception("Gemini returned no candidates (likely blocked by safety filters)")

        # Parse JSON response
        result_text = response.text
        return parse_prompt_json(result_text)

    except Exception as e:
        print(f"[RPG Creative API] Gemini error: {e}")
        raise


# === Grok Integration ===
async def generate_with_grok(system_prompt, user_prompt, model="grok-4", api_key=None, temperature=0.9, top_p=0.9):
	"""
	Generate creative prompts using xAI Grok API.
	Returns a list of {positive, negative, tags} dictionaries.
	"""
	# Use provided API key or fall back to environment variable
	if not api_key or api_key.strip() == "":
		api_key = os.environ.get("XAI_API_KEY")
	if not api_key or api_key.strip() == "":
		raise ValueError("Grok API Key is missing. Provide it in the node or set XAI_API_KEY env var.")

	try:
		base_url = os.environ.get("XAI_API_BASE", "https://api.x.ai")
		url = f"{base_url.rstrip('/')}/v1/chat/completions"

		payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "response_format": {"type": "json_object"},
        }

		headers = {
			"Authorization": f"Bearer {api_key.strip()}",
			"Content-Type": "application/json",
		}

		timeout = aiohttp.ClientTimeout(total=300)
		async with aiohttp.ClientSession(timeout=timeout) as session:
			async with session.post(url, json=payload, headers=headers) as resp:
				body = await resp.text()
				if resp.status >= 400:
					raise RuntimeError(f"Grok API error {resp.status}: {body}")
				try:
					data = json.loads(body)
				except json.JSONDecodeError as e:
					raise RuntimeError(f"Grok API returned invalid JSON: {e}")

		choices = data.get("choices") or []
		if not choices:
			raise RuntimeError("Grok API returned no choices")

		first_choice = choices[0]
		message = first_choice.get("message") or {}
		content = message.get("content") or first_choice.get("text", "")

		if not content:
			raise RuntimeError("Grok API returned empty content")

		return parse_prompt_json(content)
	except Exception as e:
		print(f"[RPG Creative API] Grok error: {e}")
		raise


# === Helper Functions ===
def parse_prompt_json(text):
    """
    Parse JSON array of prompts from API response.
    Handles markdown code blocks and extracts JSON.
    """
    # Try to find JSON in markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
        else:
            return [data]
    except json.JSONDecodeError:
        # If JSON parsing fails, create a single entry from the text
        return [{
            "positive": text.strip(),
            "negative": "",
            "tags": []
        }]


def build_system_prompt(is_sdxl=False, character_reference=""):
    """
    Build system prompt for creative generation.
    """
    base_prompt = """You are an expert AI prompt engineer specializing in creating high-quality, detailed prompts for image generation models.

Your task is to generate creative, diverse, and professional prompts that can be used for image generation."""

    if is_sdxl:
        base_prompt += "\n\nIMPORTANT: Generate prompts optimized for SDXL (Stable Diffusion XL). Use SDXL-specific quality tags and styling."

    if character_reference and character_reference.strip():
        base_prompt += f"\n\nCHARACTER REFERENCE: All prompts must include and maintain consistency with this character reference: {character_reference}"

    base_prompt += """\n\nOUTPUT FORMAT: You MUST return a valid JSON array of prompt objects. Do not include any other text or markdown. Each object must have these keys:
- "positive": A string containing the detailed positive prompt.
- "negative": A string containing the negative prompt (can be empty).
- "tags": An array of strings representing relevant tags.

Example:
[
  {
    "positive": "masterpiece, best quality, 1girl, blonde hair, blue eyes, smiling, park background, natural lighting, photorealistic",
    "negative": "deformed, bad anatomy, blurry, low quality",
    "tags": ["portrait", "photorealistic", "outdoor"]
  }
]
"""

    return base_prompt


def build_user_prompt(source_prompts, generation_count, inspiration_count):
    """
    Build user prompt for creative generation.
    """
    if len(source_prompts) > 0 and inspiration_count > 0:
        # Inspiration-based generation
        inspiration_text = "\n\n".join([
            f"Inspiration {i+1}:\nPositive: {p['prompt']['positive']}\nNegative: {p['prompt'].get('negative', '')}"
            for i, p in enumerate(source_prompts[:inspiration_count])
        ])

        return f"""Generate {generation_count} creative variations inspired by these prompts:

{inspiration_text}

Create diverse variations that:
1. Maintain the core style and quality level
2. Explore different subjects, compositions, or moods
3. Use varied descriptive language
4. Are suitable for professional image generation

Generate {generation_count} unique prompts."""

    else:
        # From-scratch generation
        return f"""Generate {generation_count} high-quality, creative prompts for image generation.

Create diverse prompts covering different:
1. Subjects (people, landscapes, objects, abstract, etc.)
2. Styles (photorealistic, artistic, cinematic, etc.)
3. Moods and atmospheres
4. Compositions and perspectives

Each prompt should be detailed, professional, and ready for image generation.

Generate {generation_count} unique prompts."""


# === API Endpoint ===
async def _generate_creative_prompts(request):
    """
    POST /instaraw/generate_creative_prompts

    Body:
    {
        "source_prompts": [{id, prompt: {positive, negative}}, ...],
        "generation_count": 5,
        "inspiration_count": 3,
        "is_sdxl": false,
        "character_reference": "",
        "model": "gemini-2.5-pro",
        "gemini_api_key": "",
        "grok_api_key": "",
        "force_regenerate": false
    }

    Returns:
    {
        "success": true,
        "prompts": [{positive, negative, tags}, ...]
    }
    """
    try:
        data = await request.json()

        source_prompts = data.get("source_prompts", [])
        generation_count = int(data.get("generation_count", 5))
        inspiration_count = int(data.get("inspiration_count", 0))
        is_sdxl = bool(data.get("is_sdxl", False))
        character_reference = data.get("character_reference", "")
        model = data.get("model", "gemini-2.5-pro")
        gemini_api_key = data.get("gemini_api_key", "")
        grok_api_key = data.get("grok_api_key", "")
        custom_system_prompt = (data.get("system_prompt") or "").strip()
        temperature = float(data.get("temperature", 0.9))
        top_p = float(data.get("top_p", 0.9))
        temperature = max(0.0, min(2.0, temperature))
        top_p = max(0.0, min(1.0, top_p))
        force_regenerate = bool(data.get("force_regenerate", False))

        # Build prompts
        system_prompt = custom_system_prompt or build_system_prompt(is_sdxl, character_reference)
        user_prompt = build_user_prompt(source_prompts, generation_count, inspiration_count)

        # Check cache (skip if force_regenerate is True)
        cache_key = hashlib.sha256(
            f"{system_prompt}_{user_prompt}_{model}_{temperature}_{top_p}".encode("utf-8")
        ).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}_creative.json")

        if not force_regenerate and os.path.exists(cache_file):
            print(f"[RPG Creative API] Using cached result: {cache_key[:8]}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                return web.json_response({"success": True, "prompts": prompts})

        if force_regenerate:
            print(f"[RPG Creative API] Force regenerate enabled - bypassing cache for {cache_key[:8]}")

        # Generate with appropriate API
        if model.startswith("gemini"):
            prompts = await generate_with_gemini(system_prompt, user_prompt, model, gemini_api_key, temperature, top_p)
        elif model.startswith("grok"):
            prompts = await generate_with_grok(system_prompt, user_prompt, model, grok_api_key, temperature, top_p)
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Cache result
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2)

        print(f"[RPG Creative API] Generated {len(prompts)} prompts with {model}")
        return web.json_response({"success": True, "prompts": prompts}, headers=CORS_HEADERS)

    except Exception as e:
        print(f"[RPG Creative API] Error: {e}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500, headers=CORS_HEADERS)


@PromptServer.instance.routes.post("/instaraw/generate_creative_prompts")
async def generate_creative_prompts_endpoint(request):
    return await _generate_creative_prompts(request)


@PromptServer.instance.routes.post("/instaraw/generate_creative_prompts/")
async def generate_creative_prompts_endpoint_slash(request):
    return await _generate_creative_prompts(request)


@PromptServer.instance.routes.options("/instaraw/generate_creative_prompts")
@PromptServer.instance.routes.options("/instaraw/generate_creative_prompts/")
async def generate_creative_prompts_options(request):
    return web.Response(headers=CORS_HEADERS)


print("[RPG Creative API] Endpoint registered: POST /instaraw/generate_creative_prompts")

# Add these lines to the end of the file
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
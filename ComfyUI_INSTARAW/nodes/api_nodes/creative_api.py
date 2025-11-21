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


def build_system_prompt(is_sdxl=False, character_description="", generation_mode="img2img", affect_elements=None, user_text_input="", random_inspiration_prompts=None, generation_style="reality"):
    """
    Build system prompt for creative generation.

    Args:
        is_sdxl: Whether to optimize for SDXL
        character_description: Detailed character description for consistency
        generation_mode: "img2img" or "txt2img"
        affect_elements: List of elements to modify (img2img only): ["background", "outfit", "pose", "lighting"]
        user_text_input: User's custom input (txt2img only)
        random_inspiration_prompts: List of random prompts for inspiration (txt2img only)
        generation_style: "reality" (strict adherence) or "creative" (flexible inspiration)
    """
    base_prompt = """You are an expert AI prompt engineer specializing in creating high-quality, detailed prompts for REALISTIC photographic image generation models.

IMPORTANT: All images are REALISTIC photography. Do not include artistic styles, illustrations, or non-photographic elements."""

    if is_sdxl:
        base_prompt += "\n\nIMPORTANT: Generate prompts optimized for SDXL (Stable Diffusion XL). Use SDXL-specific quality tags like 'masterpiece', 'best quality', 'photorealistic', 'high resolution'."

    # Character consistency
    if character_description and character_description.strip():
        base_prompt += f"\n\nCHARACTER CONSISTENCY: All prompts must accurately describe this character with exact physical features:\n{character_description}"

    # Mode-specific instructions
    if generation_mode == "img2img":
        base_prompt += "\n\nMODE: Image-to-Image Generation"

        if affect_elements and len(affect_elements) > 0:
            # Inclusion mode - only modify checked elements
            elements_str = ", ".join(affect_elements)
            base_prompt += f"\n\nIMPORTANT INSTRUCTION: You are modifying ONLY these elements: {elements_str}"
            base_prompt += "\n\nFor all OTHER elements NOT in this list, describe them EXACTLY as they appear in the input image. Do not change or modify them."
            base_prompt += f"\n\nElements to MODIFY creatively: {elements_str}"
            base_prompt += "\nElements to KEEP as-is: Everything else (describe exactly as shown)"
        else:
            # No checkboxes - describe as-is
            base_prompt += "\n\nIMPORTANT INSTRUCTION: Describe the input images EXACTLY as they appear. Create detailed, accurate descriptions that capture all visual elements faithfully."

    elif generation_mode == "txt2img":
        base_prompt += "\n\nMODE: Text-to-Image Generation (creating new prompts from scratch)"

        if random_inspiration_prompts and len(random_inspiration_prompts) > 0:
            if generation_style == "reality":
                # Reality Mode: Strict adherence to library prompts
                base_prompt += "\n\n🎯 REALITY MODE: You must ONLY use elements, words, and concepts from these reference prompts. Stay precise and constrained to what's provided:"
                for i, prompt in enumerate(random_inspiration_prompts[:5]):  # Limit to 5 for context
                    pos = prompt.get("prompt", {}).get("positive", "")
                    tags = ", ".join(prompt.get("tags", [])[:5])
                    base_prompt += f"\n\nReference {i+1}:"
                    base_prompt += f"\nPrompt: {pos[:200]}..."  # Truncate long prompts
                    base_prompt += f"\nTags: {tags}"
                base_prompt += "\n\nIMPORTANT: Your generated prompts should ONLY combine and rearrange elements from these references. Do not introduce new concepts or elements not present in these prompts."
            else:
                # Creative Mode: Flexible inspiration
                base_prompt += "\n\n✨ CREATIVE MODE: Use these prompts as creative inspiration for generating diverse, high-quality variations. Feel free to be flexible and creative:"
                for i, prompt in enumerate(random_inspiration_prompts[:5]):  # Limit to 5 for context
                    pos = prompt.get("prompt", {}).get("positive", "")
                    tags = ", ".join(prompt.get("tags", [])[:5])
                    base_prompt += f"\n\nInspiration {i+1}:"
                    base_prompt += f"\nPrompt: {pos[:200]}..."  # Truncate long prompts
                    base_prompt += f"\nTags: {tags}"
                base_prompt += "\n\nYou can be creative and add new elements while maintaining the overall style and quality of the inspiration prompts."

        if user_text_input and user_text_input.strip():
            base_prompt += f"\n\nUSER INPUT: Incorporate this user guidance into your prompts:\n{user_text_input}"

    base_prompt += """\n\nOUTPUT FORMAT: You MUST return a valid JSON array of prompt objects. Do not include any other text or markdown. Each object must have these keys:
- "positive": A string containing the detailed positive prompt for realistic photography
- "negative": A string containing the negative prompt (avoid: unrealistic, illustration, painting, drawing, art, artistic, low quality, deformed, etc.)
- "tags": An array of strings representing relevant tags

Example:
[
  {
    "positive": "masterpiece, best quality, photorealistic, high resolution, professional photography, 1girl, blonde hair, blue eyes, smiling, park background, natural sunlight, bokeh, sharp focus",
    "negative": "unrealistic, illustration, painting, drawing, art, artistic, low quality, deformed, bad anatomy, blurry, amateur",
    "tags": ["portrait", "photorealistic", "outdoor", "natural_lighting"]
  }
]
"""

    return base_prompt


def build_user_prompt(generation_count, generation_mode="img2img", images_data=None):
    """
    Build user prompt for creative generation.

    Args:
        generation_count: Number of prompts to generate
        generation_mode: "img2img" or "txt2img"
        images_data: For img2img, information about the images (optional)
    """
    if generation_mode == "img2img":
        if images_data and len(images_data) > 0:
            return f"""Generate {generation_count} detailed, accurate prompts for the {len(images_data)} input image(s) provided.

Follow the instructions in the system prompt regarding which elements to modify and which to keep as-is.

Each prompt should:
1. Accurately describe all visual elements as specified
2. Maintain photorealistic quality
3. Be detailed and professional
4. Follow the modification rules (if any elements are specified to be changed)

Generate {generation_count} unique prompt(s)."""
        else:
            return f"""Generate {generation_count} high-quality prompts for realistic photographic image generation based on the input images.

Each prompt should be detailed, professional, and capture all visual elements accurately.

Generate {generation_count} unique prompt(s)."""

    else:  # txt2img
        return f"""Generate {generation_count} high-quality, creative prompts for realistic photographic image generation.

Using the inspiration prompts and user input provided (if any), create diverse, professional prompts that:
1. Are detailed and vivid
2. Focus on realistic photographic elements
3. Include appropriate quality tags
4. Specify lighting, composition, and technical details
5. Are varied and creative while maintaining high quality

Generate {generation_count} unique prompts."""


# === API Endpoint ===
async def _generate_creative_prompts(request):
    """
    POST /instaraw/generate_creative_prompts

    Body:
    {
        // Legacy parameters (still supported)
        "source_prompts": [{id, prompt: {positive, negative}}, ...],
        "inspiration_count": 3,
        "character_reference": "",

        // New unified parameters
        "generation_count": 5,
        "is_sdxl": false,
        "model": "gemini-2.5-pro",
        "gemini_api_key": "",
        "grok_api_key": "",
        "temperature": 0.9,
        "top_p": 0.9,
        "force_regenerate": false,

        // Character likeness (NEW)
        "character_description": "Detailed character description...",
        "use_character_likeness": true,

        // Generation mode (NEW)
        "generation_mode": "img2img",  // or "txt2img"

        // img2img parameters (NEW)
        "images": ["base64_image_data..."],  // Images for vision models
        "affect_elements": ["background", "outfit"],  // Which elements to modify

        // txt2img parameters (NEW)
        "random_inspiration_prompts": [{prompt, tags}, ...],  // Random library prompts
        "user_text_input": "Custom user guidance..."
    }

    Returns:
    {
        "success": true,
        "prompts": [{positive, negative, tags}, ...]
    }
    """
    try:
        data = await request.json()

        # Core parameters
        generation_count = int(data.get("generation_count", 5))
        is_sdxl = bool(data.get("is_sdxl", False))
        model = data.get("model", "gemini-2.5-pro")
        gemini_api_key = data.get("gemini_api_key", "")
        grok_api_key = data.get("grok_api_key", "")
        temperature = float(data.get("temperature", 0.9))
        top_p = float(data.get("top_p", 0.9))
        temperature = max(0.0, min(2.0, temperature))
        top_p = max(0.0, min(1.0, top_p))
        force_regenerate = bool(data.get("force_regenerate", False))

        # NEW: Character likeness
        character_description = data.get("character_description", "")
        use_character_likeness = bool(data.get("use_character_likeness", False))
        if not use_character_likeness:
            character_description = ""

        # NEW: Generation mode
        generation_mode = data.get("generation_mode", "img2img")

        # NEW: img2img parameters
        images = data.get("images", [])
        affect_elements = data.get("affect_elements", [])

        # NEW: txt2img parameters
        random_inspiration_prompts = data.get("random_inspiration_prompts", [])
        user_text_input = data.get("user_text_input", "")

        # NEW: Generation style (Reality vs Creative)
        generation_style = data.get("generation_style", "reality")

        # Legacy support: convert old parameters to new format
        source_prompts = data.get("source_prompts", [])
        inspiration_count = int(data.get("inspiration_count", 0))
        character_reference = data.get("character_reference", "")

        if character_reference and not character_description:
            character_description = character_reference
            use_character_likeness = True

        if source_prompts and len(source_prompts) > 0 and not random_inspiration_prompts:
            random_inspiration_prompts = source_prompts[:inspiration_count] if inspiration_count > 0 else []

        # Custom system prompt override
        custom_system_prompt = (data.get("system_prompt") or "").strip()

        # Build prompts
        if custom_system_prompt:
            system_prompt = custom_system_prompt
        else:
            system_prompt = build_system_prompt(
                is_sdxl=is_sdxl,
                character_description=character_description,
                generation_mode=generation_mode,
                affect_elements=affect_elements,
                user_text_input=user_text_input,
                random_inspiration_prompts=random_inspiration_prompts,
                generation_style=generation_style
            )

        user_prompt = build_user_prompt(
            generation_count=generation_count,
            generation_mode=generation_mode,
            images_data=images
        )

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
                return web.json_response({"success": True, "prompts": prompts}, headers=CORS_HEADERS)

        if force_regenerate:
            print(f"[RPG Creative API] Force regenerate enabled - bypassing cache for {cache_key[:8]}")

        # Generate with appropriate API
        print(f"[RPG Creative API] Generating {generation_count} prompts - Mode: {generation_mode}, Model: {model}")
        if use_character_likeness:
            print(f"[RPG Creative API] Using character likeness (description length: {len(character_description)} chars)")
        if generation_mode == "img2img" and affect_elements:
            print(f"[RPG Creative API] Affecting elements: {affect_elements}")
        if generation_mode == "txt2img":
            print(f"[RPG Creative API] Inspiration prompts: {len(random_inspiration_prompts)}, User input: {len(user_text_input)} chars")

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
        import traceback
        traceback.print_exc()
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


# === Character Description Generation ===
# In-memory cache for character descriptions
CHARACTER_DESCRIPTION_CACHE = {}

def get_character_system_prompt(complexity="balanced"):
    """
    Generate system prompt based on complexity level.
    Complexity levels: concise (50-75 words), balanced (100-150 words), detailed (200-250 words)
    """
    base_instruction = """You are an expert at analyzing images and generating character descriptions for image generation prompts.

Generate a character description focusing on PERMANENT physical features:
- Facial features (face shape, eyes, nose, lips, skin tone)
- Hair (color, length, style, texture)
- Body type and build
- Age and ethnicity
- Distinctive features (scars, tattoos, piercings, etc.)

DO NOT include clothing, background, pose, or temporary features.
DO NOT use tags like "1girl, solo" or similar categorization prefixes."""

    if complexity == "concise":
        length_instruction = "\nOUTPUT: A concise description (50-75 words) focusing only on the most essential and distinctive physical features."
    elif complexity == "detailed":
        length_instruction = "\nOUTPUT: A comprehensive, detailed description (200-250 words) covering all physical aspects with nuanced detail and specific characteristics."
    else:  # balanced
        length_instruction = "\nOUTPUT: A balanced description (100-150 words) covering key physical features in natural language."

    return base_instruction + length_instruction


async def generate_character_description_with_gemini(user_prompt, model="gemini-2.5-pro", api_key=None, character_image=None, complexity="balanced", custom_system_prompt=None, temperature=0.7, top_p=0.9):
    """
    Generate plain text character description using Gemini.
    Returns raw text, not JSON.
    Uses the NEW Google Genai SDK pattern (matching gemini_native.py)
    """
    try:
        from google import genai
        from google.genai import types
        import base64
    except ImportError:
        raise ImportError("The 'google-genai' library is required. Run: pip install -U google-genai")

    if not api_key or api_key.strip() == "":
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("Gemini API Key is missing")

    try:
        # NEW SDK pattern: use Client instead of configure()
        client = genai.Client(api_key=api_key)

        # Use custom system prompt if provided, otherwise generate based on complexity
        if custom_system_prompt:
            system_instruction = custom_system_prompt
        else:
            system_instruction = get_character_system_prompt(complexity)

        # Build parts list (text + optional image)
        parts = [types.Part.from_text(text=f"{system_instruction}\n\n{user_prompt}")]

        # Add image if provided (matching gemini_native.py pattern)
        if character_image:
            # Extract base64 data (remove data URL prefix if present)
            if character_image.startswith("data:"):
                base64_data = character_image.split(",", 1)[1]
            else:
                base64_data = character_image

            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type='image/png'))

        contents = [types.Content(role="user", parts=parts)]

        # Safety settings
        safety_settings = [
            types.SafetySetting(category=cat, threshold="BLOCK_NONE")
            for cat in ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT",
                       "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT"]
        ]

        # Generation config with thinking support
        # Gemini 3.0 uses thinking_level (HIGH/LOW), Gemini 2.5 uses thinking_budget
        if model == "gemini-3-pro-preview":
            # Gemini 3.0 format
            thinking_config = types.ThinkingConfig(thinking_level="HIGH")
        elif model == "gemini-2.5-pro":
            # Gemini 2.5 format (unlimited thinking)
            thinking_config = types.ThinkingConfig(thinking_budget=-1)
        else:
            # No thinking for other models
            thinking_config = None

        config_params = {
            "temperature": temperature,
            "top_p": top_p,
            "candidate_count": 1,
            "safety_settings": safety_settings
        }

        # Only add thinking_config if it's not None
        if thinking_config:
            config_params["thinking_config"] = thinking_config

        config = types.GenerateContentConfig(**config_params)

        # NEW SDK pattern: use client.models.generate_content()
        response = client.models.generate_content(
            model=f"models/{model}",
            contents=contents,
            config=config
        )

        if not response.candidates:
            raise Exception("Gemini returned no candidates (likely blocked by safety filters)")

        # Return raw text, not JSON
        return response.text.strip()

    except Exception as e:
        print(f"[RPG Character API] Gemini error: {e}")
        raise


async def generate_character_description_with_grok(user_prompt, model="grok-4-fast-reasoning", api_key=None, character_image=None, complexity="balanced", custom_system_prompt=None, temperature=0.7, top_p=0.9):
    """
    Generate plain text character description using Grok.
    Supports vision like grok_native.py
    Returns raw text, not JSON.
    """
    if not api_key or api_key.strip() == "":
        api_key = os.environ.get("XAI_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("Grok API Key is missing")

    try:
        base_url = os.environ.get("XAI_API_BASE", "https://api.x.ai").rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        # Use custom system prompt if provided, otherwise generate based on complexity
        if custom_system_prompt:
            system_instruction = custom_system_prompt
        else:
            system_instruction = get_character_system_prompt(complexity)

        # Build messages array (like grok_native.py)
        messages = [
            {"role": "system", "content": system_instruction}
        ]

        # Build user content (text + image if provided)
        user_content = [
            {"type": "text", "text": user_prompt}
        ]

        # Add image if provided (same format as grok_native.py line 127-132)
        if character_image:
            # character_image is already base64 from JavaScript
            # Extract just the base64 part (remove "data:image/png;base64," prefix if present)
            if character_image.startswith("data:"):
                base64_data = character_image.split(",", 1)[1]
            else:
                base64_data = character_image

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_data}",
                    "detail": "high"
                }
            })

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
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
        content = message.get("content")

        # Handle different response formats
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
            if text:
                return text
        elif isinstance(content, str) and content.strip():
            return content.strip()

        # Fallback
        legacy = first_choice.get("text")
        if legacy:
            return legacy.strip()

        raise RuntimeError("Unable to extract text from Grok response")

    except Exception as e:
        print(f"[RPG Character API] Grok error: {e}")
        raise


def build_character_description_prompt():
    """Build system prompt for character description generation."""
    return """You are an expert at analyzing images and generating detailed character descriptions for image generation prompts.

Your task is to generate a comprehensive, detailed character description that can be used consistently across multiple image generation prompts.

IMPORTANT: Focus on PERMANENT, CONSISTENT physical features that define the character:
- Facial features (face shape, eyes, nose, lips, skin tone)
- Hair (color, length, style, texture)
- Body type and build
- Age and ethnicity
- Distinctive features (scars, tattoos, piercings, etc.)

DO NOT include:
- Clothing or outfit (this varies per image)
- Background or setting
- Pose or action
- Temporary features (makeup, accessories)

OUTPUT FORMAT: Return a single detailed paragraph (100-150 words) describing the character's permanent physical features. This description will be inserted into other prompts to ensure character consistency.

Example output:
"A young woman in her mid-20s with an athletic build and olive skin tone. She has striking almond-shaped green eyes, high cheekbones, and full lips. Her dark brown hair falls in natural waves to shoulder length with subtle copper highlights. She has a defined jawline, straight nose, and arched eyebrows. Her features suggest Mediterranean heritage. She stands approximately 5'7" with a toned, balanced physique and confident posture."
"""

async def _generate_character_description(request):
    """
    POST /instaraw/generate_character_description

    Body:
    {
        "character_image": "base64_string_or_null",
        "character_text": "manual_description_or_null",
        "model": "gemini-2.5-pro",
        "temperature": 0.7,
        "top_p": 0.9,
        "gemini_api_key": "",
        "grok_api_key": "",
        "force_regenerate": false
    }

    Returns:
    {
        "success": true,
        "description": "Detailed character description...",
        "cached": false,
        "cache_key": "hash_of_inputs"
    }
    """
    try:
        data = await request.json()

        character_image = data.get("character_image")
        character_text = data.get("character_text")
        model = data.get("model", "gemini-2.5-pro")
        temperature = float(data.get("temperature", 0.7))
        top_p = float(data.get("top_p", 0.9))
        gemini_api_key = data.get("gemini_api_key", "")
        grok_api_key = data.get("grok_api_key", "")
        force_regenerate = bool(data.get("force_regenerate", False))
        complexity = data.get("complexity", "balanced")
        custom_system_prompt = data.get("custom_system_prompt", "")

        # Validation
        if not character_image and not character_text:
            return web.json_response({
                "success": False,
                "error": "Either character_image or character_text must be provided"
            }, status=400, headers=CORS_HEADERS)

        # Generate cache key (include complexity and custom prompt in key)
        cache_input = character_image if character_image else character_text
        cache_key = hashlib.sha256(f"{cache_input}_{model}_{complexity}_{custom_system_prompt}".encode("utf-8")).hexdigest()

        # Check cache
        if not force_regenerate and cache_key in CHARACTER_DESCRIPTION_CACHE:
            print(f"[RPG Character API] Using cached character description: {cache_key[:8]}")
            return web.json_response({
                "success": True,
                "description": CHARACTER_DESCRIPTION_CACHE[cache_key],
                "cached": True,
                "cache_key": cache_key
            }, headers=CORS_HEADERS)

        # Build user prompt
        if character_image:
            user_prompt = "Analyze this image and generate a detailed character description following the instructions."
        else:
            user_prompt = f"Enhance this character description into a detailed, structured format suitable for image generation:\n\n{character_text}"

        # Generate description using dedicated function
        print(f"[RPG Character API] Generating character description with {model} (complexity: {complexity})...")

        if model.startswith("gemini"):
            description = await generate_character_description_with_gemini(
                user_prompt,
                model=model,
                api_key=gemini_api_key,
                character_image=character_image,
                complexity=complexity,
                custom_system_prompt=custom_system_prompt if custom_system_prompt else None,
                temperature=temperature,
                top_p=top_p
            )
        elif model.startswith("grok"):
            description = await generate_character_description_with_grok(
                user_prompt,
                model=model,
                api_key=grok_api_key,
                character_image=character_image,
                complexity=complexity,
                custom_system_prompt=custom_system_prompt if custom_system_prompt else None,
                temperature=temperature,
                top_p=top_p
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        if not description or description.strip() == "":
            raise ValueError("Generated description is empty. API may have failed.")

        # Cache the description
        CHARACTER_DESCRIPTION_CACHE[cache_key] = description

        print(f"[RPG Character API] ✅ Generated character description ({len(description)} chars)")
        return web.json_response({
            "success": True,
            "description": description,
            "cached": False,
            "cache_key": cache_key
        }, headers=CORS_HEADERS)

    except Exception as e:
        print(f"[RPG Character API] Error: {e}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500, headers=CORS_HEADERS)


# === Random Prompt Selection ===
PROMPTS_DB_CACHE = None
PROMPTS_DB_URL = "https://instara.s3.us-east-1.amazonaws.com/prompts.db.json"

async def load_prompts_database():
    """Load and cache the prompts database from remote URL."""
    global PROMPTS_DB_CACHE

    if PROMPTS_DB_CACHE is not None:
        return PROMPTS_DB_CACHE

    print("[RPG Prompts API] Loading prompts database from remote URL...")

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(PROMPTS_DB_URL) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Failed to load prompts database: HTTP {resp.status}")

                data = await resp.json()
                PROMPTS_DB_CACHE = data
                print(f"[RPG Prompts API] Loaded {len(data)} prompts from database")
                return data

    except Exception as e:
        print(f"[RPG Prompts API] Error loading database: {e}")
        raise


async def _get_random_prompts(request):
    """
    POST /instaraw/get_random_prompts

    Body:
    {
        "count": 5,
        "filters": {
            "content_type": "person",  // optional
            "safety_level": "sfw",     // optional
            "shot_type": "portrait"    // optional
        }
    }

    Returns:
    {
        "success": true,
        "prompts": [
            {
                "id": "...",
                "tags": ["..."],
                "prompt": {
                    "positive": "...",
                    "negative": "..."
                },
                "classification": {...}
            }
        ]
    }
    """
    try:
        import random

        data = await request.json()
        count = int(data.get("count", 5))
        filters = data.get("filters", {})

        # Load database
        prompts_db = await load_prompts_database()

        # Apply filters
        filtered_prompts = prompts_db

        if "content_type" in filters and filters["content_type"]:
            content_type = filters["content_type"]
            filtered_prompts = [
                p for p in filtered_prompts
                if p.get("classification", {}).get("content_type") == content_type
            ]

        if "safety_level" in filters and filters["safety_level"]:
            safety_level = filters["safety_level"]
            filtered_prompts = [
                p for p in filtered_prompts
                if p.get("classification", {}).get("safety_level") == safety_level
            ]

        if "shot_type" in filters and filters["shot_type"]:
            shot_type = filters["shot_type"]
            filtered_prompts = [
                p for p in filtered_prompts
                if p.get("classification", {}).get("shot_type") == shot_type
            ]

        # Random selection
        if len(filtered_prompts) == 0:
            return web.json_response({
                "success": False,
                "error": "No prompts match the specified filters"
            }, status=400, headers=CORS_HEADERS)

        selected_count = min(count, len(filtered_prompts))
        selected_prompts = random.sample(filtered_prompts, selected_count)

        print(f"[RPG Prompts API] Selected {selected_count} random prompts from {len(filtered_prompts)} filtered")

        return web.json_response({
            "success": True,
            "prompts": selected_prompts
        }, headers=CORS_HEADERS)

    except Exception as e:
        print(f"[RPG Prompts API] Error: {e}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500, headers=CORS_HEADERS)


# === Register New Endpoints ===
@PromptServer.instance.routes.post("/instaraw/generate_character_description")
async def generate_character_description_endpoint(request):
    return await _generate_character_description(request)


@PromptServer.instance.routes.post("/instaraw/generate_character_description/")
async def generate_character_description_endpoint_slash(request):
    return await _generate_character_description(request)


@PromptServer.instance.routes.options("/instaraw/generate_character_description")
@PromptServer.instance.routes.options("/instaraw/generate_character_description/")
async def generate_character_description_options(request):
    return web.Response(headers=CORS_HEADERS)


@PromptServer.instance.routes.post("/instaraw/get_random_prompts")
async def get_random_prompts_endpoint(request):
    return await _get_random_prompts(request)


@PromptServer.instance.routes.post("/instaraw/get_random_prompts/")
async def get_random_prompts_endpoint_slash(request):
    return await _get_random_prompts(request)


@PromptServer.instance.routes.options("/instaraw/get_random_prompts")
@PromptServer.instance.routes.options("/instaraw/get_random_prompts/")
async def get_random_prompts_options(request):
    return web.Response(headers=CORS_HEADERS)


print("[RPG Creative API] Endpoint registered: POST /instaraw/generate_creative_prompts")
print("[RPG Character API] Endpoint registered: POST /instaraw/generate_character_description")
print("[RPG Prompts API] Endpoint registered: POST /instaraw/get_random_prompts")

# Add these lines to the end of the file
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
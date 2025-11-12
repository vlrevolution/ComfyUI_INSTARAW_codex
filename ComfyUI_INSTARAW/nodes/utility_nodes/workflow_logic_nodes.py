# ---
# ComfyUI INSTARAW - Workflow Logic Nodes (Final Corrected Version)
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import threading
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from ..api_nodes.gemini_native import INSTARAW_GeminiNative

DEFAULT_PROMPT_TEMPLATE = """{llm_base_prompt}

Based on the image and the rules above, generate a description.
---
USER REQUEST: "{user_prompt}"
---
IMPORTANT: You MUST modify your description to perfectly match the USER REQUEST. The user's request has the highest priority and should be treated as a final instruction that overrides the image's content if there is a conflict. Provide only the final, clean description without commentary."""

def tensor_to_pil(image_tensor):
    return Image.fromarray(np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil_to_tensor(pil_image):
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

class INSTARAW_FeatureDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        gemini_inputs = INSTARAW_GeminiNative.INPUT_TYPES()
        # --- FIX V3: Use full, identical fallback definitions ---
        required = gemini_inputs.get("required", {})
        api_key_input = required.get("api_key", ("STRING", {"default": "GEMINI_LOAD_FAILED"}))
        model_input = required.get("model", (["gemini-2.5-pro", "gemini-flash-latest"], {"default": "gemini-2.5-pro"}))
        seed_input = required.get("seed", ("INT", {"default": 1111111, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}))
        temp_input = required.get("temperature", ("FLOAT", {"default": 0.111, "min": 0.0, "max": 2.0, "step": 0.001}))
        thinking_input = required.get("enable_thinking", ("BOOLEAN", {"default": True, "label_on": "Thinking Enabled", "label_off": "Thinking Disabled"}))
        safety_input = required.get("safety_level", (["Block None", "Block Few (High Only)", "Block Some (Medium+)", "Block Most (Low+)"], {"default": "Block None"}))

        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "feature_image": ("IMAGE",),
                "use_llm_description": ("BOOLEAN", {"default": True, "label_on": "Use LLM", "label_off": "User Prompt Only"}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llm_base_prompt": ("STRING", {"multiline": True, "default": "Describe this feature accurately..."}),
                "prompt_template": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_TEMPLATE}),
                "api_key": api_key_input, "model": model_input, "seed": seed_input, "temperature": temp_input,
                "enable_thinking": thinking_input, "safety_level": safety_input,
            }
        }
    RETURN_TYPES = ("STRING", "IMAGE",); FUNCTION = "describe"; CATEGORY = "INSTARAW/Workflow Logic"
    def describe(self, **kwargs):
        if not kwargs.get('enabled'): return (None, None)
        final_desc = kwargs.get('user_prompt', '').strip()
        if kwargs.get('use_llm_description'):
            prompt = kwargs.get('prompt_template').format(llm_base_prompt=kwargs.get('llm_base_prompt'), user_prompt=final_desc) if final_desc else kwargs.get('llm_base_prompt')
            final_desc = INSTARAW_GeminiNative().generate_content(**{k: v for k, v in kwargs.items() if k not in ['enabled', 'feature_image', 'use_llm_description', 'user_prompt', 'llm_base_prompt', 'prompt_template']}, image_1=kwargs.get('feature_image'))[0].strip()
        return (final_desc, kwargs.get('feature_image'))

class INSTARAW_UniversalDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        gemini_inputs = INSTARAW_GeminiNative.INPUT_TYPES()
        # --- FIX V3: Use full, identical fallback definitions ---
        required = gemini_inputs.get("required", {})
        api_key_input = required.get("api_key", ("STRING", {"default": "GEMINI_LOAD_FAILED"}))
        model_input = required.get("model", (["gemini-2.5-pro", "gemini-flash-latest"], {"default": "gemini-2.5-pro"}))
        seed_input = required.get("seed", ("INT", {"default": 1111111, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}))
        temp_input = required.get("temperature", ("FLOAT", {"default": 0.111, "min": 0.0, "max": 2.0, "step": 0.001}))
        thinking_input = required.get("enable_thinking", ("BOOLEAN", {"default": True, "label_on": "Thinking Enabled", "label_off": "Thinking Disabled"}))
        safety_input = required.get("safety_level", (["Block None", "Block Few (High Only)", "Block Some (Medium+)", "Block Most (Low+)"], {"default": "Block None"}))

        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}), "use_llm_description": ("BOOLEAN", {"default": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}), "llm_base_prompt": ("STRING", {"multiline": True, "default": "Describe..."}),
                "prompt_template": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_TEMPLATE}),
                "api_key": api_key_input, "model": model_input, "seed": seed_input, "temperature": temp_input,
                "enable_thinking": thinking_input, "safety_level": safety_input,
            }, "optional": { "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",) }
        }
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE",); FUNCTION = "describe"; CATEGORY = "INSTARAW/Workflow Logic"
    def describe(self, **kwargs):
        images = (kwargs.get('image_1'), kwargs.get('image_2'), kwargs.get('image_3'), kwargs.get('image_4'))
        if not kwargs.get('enabled'): return (None, *images)
        final_desc = kwargs.get('user_prompt', '').strip()
        if kwargs.get('use_llm_description'):
            prompt = kwargs.get('prompt_template').format(llm_base_prompt=kwargs.get('llm_base_prompt'), user_prompt=final_desc) if final_desc else kwargs.get('llm_base_prompt')
            final_desc = INSTARAW_GeminiNative().generate_content(**{k: v for k, v in kwargs.items() if k.startswith('api_') or k in ['model', 'seed', 'temperature', 'enable_thinking', 'safety_level']}, prompt=prompt, image_1=kwargs.get('image_1'), image_2=kwargs.get('image_2'), image_3=kwargs.get('image_3'), image_4=kwargs.get('image_4'))[0].strip()
        return (final_desc, *images)

class INSTARAW_SwapPromptAssembler:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "hair_prefix": ("STRING", {}), "outfit_prefix": ("STRING", {}), "separator": ("STRING", {"default": " + "}), }, "optional": { "hair_description": ("STRING", {"forceInput": True}), "outfit_description": ("STRING", {"forceInput": True}), } }
    RETURN_TYPES = ("STRING", "BOOLEAN",); FUNCTION = "assemble"; CATEGORY = "INSTARAW/Workflow Logic"
    def assemble(self, hair_prefix, outfit_prefix, separator, hair_description=None, outfit_description=None):
        parts = []
        if hair_description and hair_description.strip(): parts.append(f"{hair_prefix.strip()} {hair_description.strip()}")
        if outfit_description and outfit_description.strip(): parts.append(f"{outfit_prefix.strip()} {outfit_description.strip()}")
        return (separator.join(parts), bool(parts))

class INSTARAW_ParallelFeatureDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        gemini_inputs = INSTARAW_GeminiNative.INPUT_TYPES()
        # --- FIX V3: Use full, identical fallback definitions ---
        required = gemini_inputs.get("required", {})
        api_key_input = required.get("api_key", ("STRING", {"default": "GEMINI_LOAD_FAILED"}))
        model_input = required.get("model", (["gemini-2.5-pro", "gemini-flash-latest"], {"default": "gemini-2.5-pro"}))
        seed_input = required.get("seed", ("INT", {"default": 1111111, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}))
        temp_input = required.get("temperature", ("FLOAT", {"default": 0.111, "min": 0.0, "max": 2.0, "step": 0.001}))
        thinking_input = required.get("enable_thinking", ("BOOLEAN", {"default": True, "label_on": "Thinking Enabled", "label_off": "Thinking Disabled"}))
        safety_input = required.get("safety_level", (["Block None", "Block Few (High Only)", "Block Some (Medium+)", "Block Most (Low+)"], {"default": "Block None"}))

        return {
            "required": {
                "hair_enabled": ("BOOLEAN", {"default": True}), "hair_use_llm": ("BOOLEAN", {"default": True}),
                "hair_user_prompt": ("STRING", {"multiline": True}), "hair_llm_base_prompt": ("STRING", {"multiline": True}),
                "hair_prompt_template": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_TEMPLATE}),
                "outfit_enabled": ("BOOLEAN", {"default": True}), "outfit_use_llm": ("BOOLEAN", {"default": True}),
                "outfit_user_prompt": ("STRING", {"multiline": True}), "outfit_llm_base_prompt": ("STRING", {"multiline": True}),
                "outfit_prompt_template": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_TEMPLATE}),
                "api_key": api_key_input, "model": model_input, "seed": seed_input, "temperature": temp_input,
                "enable_thinking": thinking_input, "safety_level": safety_input,
            }, "optional": { "hair_feature_image": ("IMAGE",), "outfit_feature_image": ("IMAGE",), }
        }
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "IMAGE",); FUNCTION = "describe_parallel"; CATEGORY = "INSTARAW/Workflow Logic"
    def describe_parallel(self, **kwargs):
        results = {}
        def worker(feature_name, enabled, use_llm, user_prompt, llm_base_prompt, prompt_template, image):
            try:
                if not enabled: results[f"{feature_name}_description"], results[f"{feature_name}_image_out"] = None, None; return
                desc = user_prompt.strip()
                if use_llm:
                    # --- FIX V3: Use 'is None' to check for Tensor existence ---
                    if image is None:
                        print(f"⚠️ Warning: LLM for '{feature_name}' enabled but no image provided. Skipping."); results[f"{feature_name}_description"], results[f"{feature_name}_image_out"] = None, None; return
                    prompt_for_llm = prompt_template.format(llm_base_prompt=llm_base_prompt, user_prompt=user_prompt) if user_prompt.strip() else llm_base_prompt
                    gemini_args = {k: v for k, v in kwargs.items() if k.startswith('api_') or k in ['model', 'seed', 'temperature', 'enable_thinking', 'safety_level']}
                    gemini_args['prompt'] = prompt_for_llm
                    gemini_args['image_1'] = image
                    desc = INSTARAW_GeminiNative().generate_content(**gemini_args)[0].strip()
                results[f"{feature_name}_description"] = desc
                results[f"{feature_name}_image_out"] = image if enabled else None
            except Exception as e: results[f"{feature_name}_exception"] = e
        threads = [threading.Thread(target=worker, args=(name, kwargs[f'{name}_enabled'], kwargs[f'{name}_use_llm'], kwargs[f'{name}_user_prompt'], kwargs[f'{name}_llm_base_prompt'], kwargs[f'{name}_prompt_template'], kwargs.get(f'{name}_feature_image'))) for name in ["hair", "outfit"]]
        for t in threads: t.start()
        for t in threads: t.join()
        for name in ["hair", "outfit"]:
            if (e := results.get(f"{name}_exception")): raise e
        return (results.get("hair_description"), results.get("hair_image_out"), results.get("outfit_description"), results.get("outfit_image_out"),)

# The rest of the file remains the same...
class INSTARAW_SeeDreamPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "hair_prefix_single": ("STRING", {}), "outfit_prefix_single": ("STRING", {}), "hair_prefix_multi_template": ("STRING", {"default": "perfectly change hair of image {} to"}), "outfit_prefix_multi_template": ("STRING", {"default": "perfectly change outfit of image {} to"}), "separator": ("STRING", {"default": " + "}), }, "optional": { "hair_description": ("STRING", {"forceInput": True}), "outfit_description": ("STRING", {"forceInput": True}), "hair_image_ref": ("IMAGE",), "outfit_image_ref": ("IMAGE",), } }
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE",); FUNCTION = "build_prompt"; CATEGORY = "INSTARAW/Workflow Logic"
    def build_prompt(self, hair_prefix_single, outfit_prefix_single, hair_prefix_multi_template, outfit_prefix_multi_template, separator, hair_description=None, outfit_description=None, hair_image_ref=None, outfit_image_ref=None):
        num_refs = (1 if hair_image_ref is not None else 0) + (1 if outfit_image_ref is not None else 0)
        target_idx = num_refs + 1
        hair_prefix = hair_prefix_multi_template.format(target_idx) if num_refs > 0 else hair_prefix_single
        outfit_prefix = outfit_prefix_multi_template.format(target_idx) if num_refs > 0 else outfit_prefix_single
        parts = []
        if hair_description and hair_description.strip(): parts.append(f"{hair_prefix.strip()} {hair_description.strip()}")
        if outfit_description and outfit_description.strip(): parts.append(f"{outfit_prefix.strip()} {outfit_description.strip()}")
        return (separator.join(parts), hair_image_ref, outfit_image_ref)

class INSTARAW_PreviewAssembler:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "main_image": ("IMAGE",), "layout": (["Horizontal", "Vertical"],), "spacing": ("INT", {"default": 10}), "add_labels": ("BOOLEAN", {"default": True}), "label_scale": ("FLOAT", {"default": 1.0}), "font_color": ("STRING", {"default": "white"}), "label_position": (["Top Left", "Top Center", "Bottom Left", "Bottom Center"],), }, "optional": { "hair_image_ref": ("IMAGE",), "outfit_image_ref": ("IMAGE",), } }
    RETURN_TYPES = ("IMAGE",); FUNCTION = "assemble_preview"; CATEGORY = "INSTARAW/Workflow Logic"
    def assemble_preview(self, main_image, layout, spacing, add_labels, label_scale, font_color, label_position, hair_image_ref=None, outfit_image_ref=None):
        images = [("Target", main_image)]
        if hair_image_ref is not None: images.append(("Hair Ref", hair_image_ref))
        if outfit_image_ref is not None: images.append(("Outfit Ref", outfit_image_ref))
        if len(images) == 1: return (main_image,)
        pils = [tensor_to_pil(img) for _, img in images]; labels = [lbl for lbl, _ in images]
        main_p = pils[0]; mw, mh = main_p.size; resized = []
        for img in pils:
            w, h = img.size
            if layout == "Horizontal": resized.append(img.resize((int(w * (mh / h)), mh), Image.LANCZOS) if h != mh else img)
            else: resized.append(img.resize((mw, int(h * (mw / w))), Image.LANCZOS) if w != mw else img)
        tw = sum(i.width for i in resized) + spacing * (len(resized) - 1); th = sum(i.height for i in resized) + spacing * (len(resized) - 1)
        canvas = Image.new('RGB', (tw, mh) if layout == "Horizontal" else (mw, th), 'black')
        font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'fonts', 'BricolageGrotesque.ttf')
        draw = ImageDraw.Draw(canvas); offset = 0
        for i, img in enumerate(resized):
            px, py = (offset, 0) if layout == "Horizontal" else (0, offset)
            canvas.paste(img, (px, py))
            if add_labels and os.path.exists(font_path):
                fs = int(img.height * 0.05 * label_scale); pad = int(img.height * 0.01 * label_scale)
                font = ImageFont.truetype(font_path, fs if fs > 0 else 1)
                bbox = font.getbbox(labels[i]); t_w, t_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                bx = px + pad if "Left" in label_position else px + (img.width - t_w) // 2
                by = py + pad if "Top" in label_position else py + img.height - t_h - (pad * 2)
                draw.rectangle((bx - pad, by - pad, bx + t_w + pad, by + t_h + pad), fill=(0,0,0,128))
                draw.text((bx, by - bbox[1]), labels[i], font=font, fill=font_color)
            offset += (img.width if layout == "Horizontal" else img.height) + spacing
        return (pil_to_tensor(canvas),)

class INSTARAW_TeleportInputAssembler:
    @classmethod
    def INPUT_TYPES(cls): return { "required": { "character_image": ("IMAGE",), "teleport_to_image": ("IMAGE",), "layout": (["Horizontal", "Vertical"], {}), "spacing": ("INT", {"default": 10}), "add_labels": ("BOOLEAN", {"default": True}), "label_scale": ("FLOAT", {"default": 1.0}), "font_color": ("STRING", {"default": "white"}), "label_position": (["Top Left", "Top Center", "Bottom Left", "Bottom Center"],), } }
    RETURN_TYPES = ("IMAGE",); FUNCTION = "assemble_inputs"; CATEGORY = "INSTARAW/Workflow Logic"
    def assemble_inputs(self, **kwargs): return INSTARAW_PreviewAssembler().assemble_preview(main_image=kwargs['character_image'], hair_image_ref=kwargs['teleport_to_image'], **{k:v for k,v in kwargs.items() if k not in ['character_image', 'teleport_to_image']})

NODE_CLASS_MAPPINGS = { "INSTARAW_FeatureDescriber": INSTARAW_FeatureDescriber, "INSTARAW_UniversalDescriber": INSTARAW_UniversalDescriber, "INSTARAW_SwapPromptAssembler": INSTARAW_SwapPromptAssembler, "INSTARAW_ParallelFeatureDescriber": INSTARAW_ParallelFeatureDescriber, "INSTARAW_SeeDreamPromptBuilder": INSTARAW_SeeDreamPromptBuilder, "INSTARAW_PreviewAssembler": INSTARAW_PreviewAssembler, "INSTARAW_TeleportInputAssembler": INSTARAW_TeleportInputAssembler, }
NODE_DISPLAY_NAME_MAPPINGS = { "INSTARAW_FeatureDescriber": "✨ INSTARAW Feature Describer (Single)", "INSTARAW_UniversalDescriber": "✨ INSTARAW Universal Describer", "INSTARAW_SwapPromptAssembler": "✍️ INSTARAW Swap Prompt Assembler", "INSTARAW_ParallelFeatureDescriber": "🚀 INSTARAW Feature Describer (Parallel)", "INSTARAW_SeeDreamPromptBuilder": "🗣️ INSTARAW SeeDream Prompt Builder", "INSTARAW_PreviewAssembler": "🖼️ INSTARAW Preview Assembler", "INSTARAW_TeleportInputAssembler": "🖼️ INSTARAW Teleport Input Assembler", }
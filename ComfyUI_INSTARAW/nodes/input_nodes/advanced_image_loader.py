# ---
# Filename: ../ComfyUI_INSTARAW/nodes/input_nodes/advanced_image_loader.py
# ---

"""
ComfyUI INSTARAW - Advanced Image Loader
Modern batch image loading with instant preview, reordering, and per-image repeat control.
"""

import torch
import numpy as np
from PIL import Image
import os
import json
import uuid
import hashlib
from aiohttp import web
import folder_paths
from server import PromptServer
import comfy.utils

class INSTARAW_AdvancedImageLoader:
    """
    An advanced, interactive multi-image batch loader that receives its state
    from a custom frontend widget.
    """

    def __init__(self):
        self.node_states = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Batch Tensor", "Sequential"], {"default": "Batch Tensor"}),
            },
            "optional": {
                "resize_mode": (["Center Crop", "Letterbox", "Stretch", "Fit to Largest"], {"default": "Center Crop"}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "batch_data": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "index", "total", "info")
    FUNCTION = "load_batch"
    CATEGORY = "INSTARAW/Input"

    def load_batch(self, mode="Batch Tensor", resize_mode="Center Crop", batch_index=0, node_id=None, batch_data="{}"):
        print(f"\n{'='*60}")
        print(f"[INSTARAW Adv Loader] Mode: {mode} | Resize: {resize_mode} | Index: {batch_index}")
        print(f"[INSTARAW Adv Loader] Loading batch for node: {node_id}")
        
        try:
            data = json.loads(batch_data) if batch_data else {}
        except json.JSONDecodeError:
            print("[INSTARAW Adv Loader] Invalid batch_data JSON, using empty.")
            data = {}

        images_meta = data.get('images', [])
        order = data.get('order', [])

        if not images_meta or not order:
            print("[INSTARAW Adv Loader] No images in batch, returning empty.")
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0, 0, "No images loaded")

        upload_dir = self._get_upload_dir(node_id)
        if not os.path.exists(upload_dir):
            print(f"[INSTARAW Adv Loader] Upload directory not found: {upload_dir}")
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0, 0, f"Directory not found: {upload_dir}")

        total_count = sum(img.get('repeat_count', 1) for img in images_meta)

        if mode == "Sequential":
            state_key = f"{node_id}_{hashlib.md5(batch_data.encode()).hexdigest()[:8]}"
            if state_key in self.node_states and self.node_states[state_key]['last_widget_index'] != batch_index:
                self.node_states[state_key] = {'current_index': batch_index, 'last_widget_index': batch_index}
            elif state_key not in self.node_states:
                self.node_states[state_key] = {'current_index': batch_index, 'last_widget_index': batch_index}

            current_index = self.node_states[state_key]['current_index']
            img_tensor, _, total, info = self._load_sequential(upload_dir, images_meta, order, current_index, total_count)
            next_index = (current_index + 1) % total_count
            self.node_states[state_key]['current_index'] = next_index
            self.node_states[state_key]['last_widget_index'] = batch_index
            
            try:
                PromptServer.instance.send_sync("instaraw_adv_loader_update", {"node_id": str(node_id), "current_index": current_index, "next_index": next_index, "total_count": total_count})
            except Exception as e:
                print(f"[INSTARAW Adv Loader] Error sending WebSocket update: {e}")
            
            return (img_tensor, current_index, total, info)
        else:
            return self._load_batch_tensor(upload_dir, images_meta, order, resize_mode, total_count)

    def _load_sequential(self, upload_dir, images_meta, order, batch_index, total_count):
        flat_list = []
        for img_id in order:
            img_meta = next((img for img in images_meta if img['id'] == img_id), None)
            if not img_meta: continue
            for i in range(img_meta.get('repeat_count', 1)):
                flat_list.append((img_meta, i + 1, img_meta.get('repeat_count', 1)))

        if batch_index >= len(flat_list):
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, batch_index, total_count, "Index out of range")

        img_meta, copy_num, repeat_count = flat_list[batch_index]
        filename, original_name = img_meta['filename'], img_meta.get('original_name', img_meta['filename'])
        img_path = os.path.join(upload_dir, filename)

        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            info = f"[{batch_index}/{total_count}] {original_name} (copy {copy_num}/{repeat_count})"
            return (img_tensor, batch_index, total_count, info)
        except Exception as e:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, batch_index, total_count, f"Error: {e}")

    def _load_batch_tensor(self, upload_dir, images_meta, order, resize_mode, total_count):
        loaded_images, info_lines, target_height, target_width = [], [], None, None
        for img_id in order:
            img_meta = next((img for img in images_meta if img['id'] == img_id), None)
            if not img_meta: continue
            
            filename, repeat_count, original_name = img_meta['filename'], img_meta.get('repeat_count', 1), img_meta.get('original_name', img_meta['filename'])
            img_path = os.path.join(upload_dir, filename)
            if not os.path.exists(img_path): continue

            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                if target_height is None:
                    target_height, target_width = img_tensor.shape[0], img_tensor.shape[1]
                if img_tensor.shape[0] != target_height or img_tensor.shape[1] != target_width:
                    img_tensor = self._resize_image(img_tensor, target_width, target_height, resize_mode)
                for i in range(repeat_count):
                    loaded_images.append(img_tensor)
                    info_lines.append(f"{original_name} (copy {i+1}/{repeat_count})")
            except Exception as e:
                print(f"[INSTARAW Adv Loader] Error loading {filename}: {e}")
                continue

        if not loaded_images:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0, total_count, "Failed to load images")

        return (torch.stack(loaded_images, dim=0), 0, total_count, "\n".join(info_lines))

    def _resize_image(self, img_tensor, target_width, target_height, resize_mode):
        if resize_mode == "Stretch":
            return comfy.utils.common_upscale(img_tensor.movedim(-1, 0).unsqueeze(0), target_width, target_height, "bilinear", "disabled").squeeze(0).movedim(0, -1)
        if resize_mode == "Center Crop":
            return comfy.utils.common_upscale(img_tensor.movedim(-1, 0).unsqueeze(0), target_width, target_height, "bilinear", "center").squeeze(0).movedim(0, -1)
        if resize_mode == "Letterbox":
            scale = min(target_width / img_tensor.shape[1], target_height / img_tensor.shape[0])
            new_w, new_h = int(img_tensor.shape[1] * scale), int(img_tensor.shape[0] * scale)
            resized = comfy.utils.common_upscale(img_tensor.movedim(-1, 0).unsqueeze(0), new_w, new_h, "bilinear", "disabled").squeeze(0).movedim(0, -1)
            canvas = torch.zeros((target_height, target_width, 3), dtype=torch.float32)
            pad_top, pad_left = (target_height - new_h) // 2, (target_width - new_w) // 2
            canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized
            return canvas
        return img_tensor

    def _get_upload_dir(self, node_id):
        input_dir = folder_paths.get_input_directory()
        return os.path.join(input_dir, "INSTARAW_BatchUploads", str(node_id))

    @classmethod
    def IS_CHANGED(cls, mode="Batch Tensor", batch_data="{}", **kwargs):
        if mode == "Sequential":
            import time
            return f"seq_{time.time()}_{hashlib.md5(batch_data.encode()).hexdigest()[:8]}"
        else:
            return hashlib.md5(batch_data.encode()).hexdigest()

# --- API Endpoints (Branded) ---
@PromptServer.instance.routes.post("/instaraw/batch_upload")
async def batch_upload_endpoint(request):
    try:
        reader = await request.multipart()
        node_id, uploaded_files = None, []
        async for field in reader:
            if field.name == 'node_id':
                node_id = await field.text()
            elif field.name == 'files' and node_id:
                filename = field.filename
                file_data = await field.read()
                input_dir = folder_paths.get_input_directory()
                upload_dir = os.path.join(input_dir, "INSTARAW_BatchUploads", str(node_id))
                os.makedirs(upload_dir, exist_ok=True)
                
                file_id = str(uuid.uuid4())
                ext = os.path.splitext(filename)[1].lower()
                safe_filename = f"{file_id}{ext}"
                full_path = os.path.join(upload_dir, safe_filename)
                with open(full_path, 'wb') as f: f.write(file_data)

                try:
                    img = Image.open(full_path)
                    width, height = img.size
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    thumb_filename = f"thumb_{safe_filename}"
                    img.save(os.path.join(upload_dir, thumb_filename))
                    uploaded_files.append({"id": file_id, "filename": safe_filename, "original_name": filename, "width": width, "height": height, "thumbnail": thumb_filename, "repeat_count": 1})
                except Exception as e:
                    print(f"[INSTARAW Adv Loader] Error processing image: {e}")
                    if os.path.exists(full_path): os.remove(full_path)
        return web.json_response({"success": True, "images": uploaded_files})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

@PromptServer.instance.routes.delete("/instaraw/batch_delete/{node_id}/{image_id}")
async def batch_delete_endpoint(request):
    node_id, image_id = request.match_info['node_id'], request.match_info['image_id']
    try:
        input_dir = folder_paths.get_input_directory()
        upload_dir = os.path.join(input_dir, "INSTARAW_BatchUploads", str(node_id))
        deleted = []
        for filename in os.listdir(upload_dir):
            if image_id in filename:
                os.remove(os.path.join(upload_dir, filename))
                deleted.append(filename)
        return web.json_response({"success": True, "deleted": deleted})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

NODE_CLASS_MAPPINGS = {"INSTARAW_AdvancedImageLoader": INSTARAW_AdvancedImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"INSTARAW_AdvancedImageLoader": "🖼️ INSTARAW Advanced Image Loader"}
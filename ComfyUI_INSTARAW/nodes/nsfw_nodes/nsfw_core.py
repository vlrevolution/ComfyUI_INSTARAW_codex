# ---
# ComfyUI INSTARAW - NSFW Core Logic (v4.0 - Correct, Working Version)
# This version perfectly mimics the original Nudenet.py's preprocessing to ensure identical results.
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import cv2
import numpy as np
import math

def preprocess_image(image_np_float: np.ndarray, target_size: int = 320):
    if image_np_float.dtype != np.float32:
        image_np_float = image_np_float.astype(np.float32)

    img_channel_swapped = cv2.cvtColor(image_np_float, cv2.COLOR_RGB2BGR)

    img_height, img_width = img_channel_swapped.shape[:2]
    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt((img_width**2 + img_height**2) / (new_width**2 + new_height**2))
    
    img_resized = cv2.resize(img_channel_swapped, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height
    pad_top, pad_bottom = pad_y // 2, pad_y - (pad_y // 2)
    pad_left, pad_right = pad_x // 2, pad_x - (pad_x // 2)

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    img_final = cv2.resize(img_padded, (target_size, target_size))

    image_data = img_final.astype(np.float32)
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    
    return image_data, resize_factor, pad_left, pad_top

def postprocess_detections(output, resize_factor, pad_left, pad_top, min_score):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= min_score:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            
            left = (x - w * 0.5 - pad_left) * resize_factor
            top = (y - h * 0.5 - pad_top) * resize_factor
            width = w * resize_factor
            height = h * resize_factor
            
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, min_score, 0.45)
    
    if len(indices) == 0: return []

    return [{"id": int(class_ids[i]), "score": round(float(scores[i]), 2), "box": [int(round(coord)) for coord in boxes[i]],} for i in indices]
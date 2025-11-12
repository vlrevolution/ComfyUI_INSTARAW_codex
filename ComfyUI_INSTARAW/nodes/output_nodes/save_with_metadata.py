# ---
# Filename: ../ComfyUI_INSTARAW/nodes/utility_nodes/save_with_metadata.py
# ---

# ---
# ComfyUI INSTARAW - Save With Authentic Metadata Node
# Part of the INSTARAW custom nodes collection by Instara
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import torch
import numpy as np
from PIL import Image
import piexif
import os
import re
import json
import random
from datetime import datetime, timedelta
import folder_paths

class INSTARAW_SaveWithMetadata:
    """
    A node that saves an image while stripping all ComfyUI metadata and injecting
    a realistic EXIF profile mimicking a modern iPhone, including GPS data and authentic filenames.
    """
    
    OUTPUT_NODE = True
    CATEGORY = "INSTARAW/Output"
    FUNCTION = "save_image"

    LOCATIONS = {
        "None": None,
        "New York, NY": (40.712776, -74.005974),
        "Los Angeles, CA": (34.052235, -118.243683),
        "Miami, FL": (25.761681, -80.191788),
        "Chicago, IL": (41.878113, -87.629799),
        "Las Vegas, NV": (36.169941, -115.139832),
        "Austin, TX": (30.267153, -97.743061),
        "San Francisco, CA": (37.774929, -122.419418),
        "Nashville, TN": (36.162663, -86.781601),
        "Atlanta, GA": (33.748997, -84.387985),
        "Honolulu, HI": (21.306944, -157.858337),
        "London, UK": (51.507351, -0.127758),
        "Tokyo, JP": (35.689487, 139.691711),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_iphone_naming": ("BOOLEAN", {"default": True, "label_on": "Authentic iPhone Filename", "label_off": "Use Prefix"}),
                "filename_prefix": ("STRING", {"default": "INSTARAW_Image"}),
                "format": (["jpg", "png"], {"default": "jpg"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "iphone_model": (["iPhone 16 Pro", "iPhone 16 Pro Max", "iPhone 15 Pro"], {"default": "iPhone 16 Pro"}),
                "ios_version": ("STRING", {"default": "19.1"}),
                "location": (list(cls.LOCATIONS.keys()), {"default": "New York, NY"}),
                "randomize_details": ("BOOLEAN", {"default": True, "label_on": "Randomize Time/ISO/Shutter", "label_off": "Use Fixed Values"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()

    def _get_next_iphone_filenumber(self, output_dir):
        """Scans the output directory to find the next available IMG_XXXX number."""
        pattern = re.compile(r"IMG_(\d{4})\.(jpg|jpeg|png|heic)", re.IGNORECASE)
        highest_num = 0
        for f in os.listdir(output_dir):
            match = pattern.match(f)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num
        return highest_num + 1

    def _gps_to_exif(self, latitude, longitude):
        # ... (this function remains unchanged)
        def to_deg(value, loc):
            if value < 0: loc_value = loc[1]
            elif value > 0: loc_value = loc[0]
            else: loc_value = ""
            abs_value = abs(value)
            deg = int(abs_value)
            t1 = (abs_value - deg) * 60
            min = int(t1)
            sec = round((t1 - min) * 60, 5)
            return (deg, min, sec, loc_value)
        lat_deg = to_deg(latitude, ("N", "S"))
        lng_deg = to_deg(longitude, ("E", "W"))
        return {
            piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
            piexif.GPSIFD.GPSLatitude: ((lat_deg[0], 1), (lat_deg[1], 1), (int(lat_deg[2] * 100000), 100000)),
            piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
            piexif.GPSIFD.GPSLongitude: ((lng_deg[0], 1), (lng_deg[1], 1), (int(lng_deg[2] * 100000), 100000)),
        }

    def save_image(self, image: torch.Tensor, use_iphone_naming: bool, filename_prefix: str, format: str, quality: int,
                   iphone_model: str, ios_version: str, location: str, randomize_details: bool,
                   prompt=None, extra_pnginfo=None):

        output_dir = folder_paths.get_output_directory()
        
        # --- THIS IS THE NEW LOGIC ---
        start_number = 0
        if use_iphone_naming:
            start_number = self._get_next_iphone_filenumber(output_dir)
        
        saved_files = []
        # --- END NEW LOGIC ---

        for i in range(image.shape[0]):
            img_tensor = image[i:i+1]
            img_np = img_tensor.squeeze(0).cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            width, height = img_pil.size

            # --- Filename Generation ---
            if use_iphone_naming:
                current_num = start_number + i
                filename = f"IMG_{current_num:04d}" # :04d pads with leading zeros, e.g., 9 -> 0009
            else:
                filename = f"{filename_prefix}_{i:03d}"
            
            filepath = os.path.join(output_dir, f"{filename}.{format}")
            # --- End Filename Generation ---

            now = datetime.now() - timedelta(days=random.randint(1, 30))
            if randomize_details:
                now -= timedelta(seconds=random.randint(0, 3600))
                iso = random.randint(50, 200)
                shutter_speed_denom = random.choice([60, 100, 120, 250, 500])
                f_stop = random.choice([1.8, 2.0, 2.2, 2.8])
            else:
                iso, shutter_speed_denom, f_stop = 80, 120, 1.8
            date_str = now.strftime("%Y:%m:%d %H:%M:%S")

            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_dict["0th"][piexif.ImageIFD.Make] = b"Apple"
            exif_dict["0th"][piexif.ImageIFD.Model] = iphone_model.encode('utf-8')
            exif_dict["0th"][piexif.ImageIFD.Software] = ios_version.encode('utf-8')
            exif_dict["0th"][piexif.ImageIFD.DateTime] = date_str.encode('utf-8')
            exif_dict["0th"][piexif.ImageIFD.XResolution] = (72, 1)
            exif_dict["0th"][piexif.ImageIFD.YResolution] = (72, 1)
            exif_dict["0th"][piexif.ImageIFD.ResolutionUnit] = 2
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_str.encode('utf-8')
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = date_str.encode('utf-8')
            exif_dict["Exif"][piexif.ExifIFD.PixelXDimension] = width
            exif_dict["Exif"][piexif.ExifIFD.PixelYDimension] = height
            exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = iso
            exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (1, shutter_speed_denom)
            exif_dict["Exif"][piexif.ExifIFD.FNumber] = (int(f_stop * 100), 100)
            exif_dict["Exif"][piexif.ExifIFD.LensModel] = b"iPhone 16 Pro Camera"
            exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (240, 10)
            exif_dict["Exif"][piexif.ExifIFD.ColorSpace] = 1
            if self.LOCATIONS[location] is not None:
                lat, lon = self.LOCATIONS[location]
                exif_dict["GPS"] = self._gps_to_exif(lat, lon)
            exif_bytes = piexif.dump(exif_dict)
            
            if format == 'jpg':
                img_pil.save(filepath, quality=quality, exif=exif_bytes)
            elif format == 'png':
                img_pil.save(filepath)
            
            saved_files.append({"filename": f"{filename}.{format}", "subfolder": "", "type": "output"})

        return {"ui": {"images": saved_files}}

NODE_CLASS_MAPPINGS = {
    "INSTARAW_SaveWithMetadata": INSTARAW_SaveWithMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_SaveWithMetadata": "💾 INSTARAW Save w/ Metadata",
}
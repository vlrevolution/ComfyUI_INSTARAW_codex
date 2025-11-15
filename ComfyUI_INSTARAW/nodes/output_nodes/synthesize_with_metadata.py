import torch
import numpy as np
from PIL import Image
import os
import json
import random
import folder_paths
import tempfile
import shutil
import uuid
import subprocess
import datetime
from pathlib import Path

# ---------------- CONFIG / PATHS ----------------

NODE_DIR = Path(__file__).parent.parent.parent
LOCATIONS_FILE = NODE_DIR / "modules" / "location_data" / "locations.json"

# Global sRGB ICC profile – used ONLY for tagging, no color transform
SRGB_PROFILE_FILE = NODE_DIR / "modules" / "color_profiles" / "sRGB_IEC61966-2-1_no_black_scaling.icc"

def is_tool(name: str) -> bool:
    return shutil.which(name) is not None

def load_locations_data():
    if not LOCATIONS_FILE.exists():
        print(f"⚠️ INSTARAW: Locations file not found. Creating a default at '{LOCATIONS_FILE}'")
        LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        default_data = {
          "North America": {
              "United States": [
                  {
                      "city": "New York",
                      "lat": [40.4774, 40.9176],
                      "lon": [-74.2591, -73.7002]
                  }
              ]
          },
          "Europe": {
              "France": [
                  {
                      "city": "Paris",
                      "lat": [48.8156, 48.9021],
                      "lon": [2.2241, 2.4699]
                  }
              ]
          }
        }
        with open(LOCATIONS_FILE, 'w', encoding="utf-8") as f:
            json.dump(default_data, f, indent=2)
        return default_data
    with open(LOCATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

EXIFTOOL_AVAILABLE = is_tool("exiftool") or is_tool("exiftool.exe")


class INSTARAW_SynthesizeAuthenticMetadata:
    """
    Final design goals:

    - Do NOT change pixel colors here.
      The tensor coming from Comfy is assumed to already be "final look" in sRGB.
    - Only write EXIF/XMP metadata and (optionally) embed an sRGB ICC profile.
    - Orientation and ColorSpace are written as numeric EXIF codes using exiftool -n
      so they are honored correctly.
    """
    OUTPUT_NODE = False
    CATEGORY = "INSTARAW/Authenticity"
    FUNCTION = "synthesize_and_save"

    _locations_data = None
    _location_choices = None

    # ------------- Comfy INPUT / RETURN TYPES -------------

    @classmethod
    def get_location_choices(cls):
        if cls._locations_data is None:
            cls._locations_data = load_locations_data()

        if cls._location_choices is None:
            special_options = [
                "From Profile",
                "Synthesize Random (Any City)",
                "Synthesize Random (US City)",
            ]
            random_country_options = []
            all_countries = set()

            for region in cls._locations_data.values():
                for country in region.keys():
                    all_countries.add(country)

            for country in sorted(list(all_countries)):
                if country != "United States":
                    random_country_options.append(f"Synthesize Random ({country})")

            specific_city_options = []
            for region, countries in cls._locations_data.items():
                for country, cities in countries.items():
                    for city_data in cities:
                        specific_city_options.append(f"{city_data['city']}, {country}")

            cls._location_choices = (
                special_options + random_country_options + sorted(specific_city_options)
            )
        return cls._location_choices

    @classmethod
    def INPUT_TYPES(cls):
        now = datetime.datetime.now()
        seven_days_ago = now - datetime.timedelta(days=7)
        default_end_date = now.strftime("%Y-%m-%dT%H:%M:%S")
        default_start_date = seven_days_ago.strftime("%Y-%m-%dT%H:%M:%S")
        scene_types = [
            "From Profile",
            "Synthesize Random",
            "Daylight (Bright)",
            "Daylight (Cloudy)",
            "Golden Hour",
            "Indoors",
            "Night",
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "profile_path": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "inject_color_profile": ("BOOLEAN", {"default": False}),
                "location_synthesis": (
                    cls.get_location_choices(),
                    {"default": "Synthesize Random (US City)"}
                ),
                "datetime_synthesis": (
                    ["From Profile", "Synthesize Random"],
                    {"default": "Synthesize Random"}
                ),
                "start_date": ("STRING", {"default": default_start_date}),
                "end_date": ("STRING", {"default": default_end_date}),
                "scene_type": (scene_types, {"default": "Synthesize Random"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)

    # ------------- LOCATION HELPERS -------------

    def get_random_city_from_country(self, country_name, rng: random.Random):
        cities_in_country = []
        for region, countries in self._locations_data.items():
            if country_name in countries:
                cities_in_country.extend(countries[country_name])
        if cities_in_country:
            return rng.choice(cities_in_country)
        return None

    # ------------- MAIN FUNCTION -------------

    def synthesize_and_save(
        self,
        image: torch.Tensor,
        profile_path: str,
        seed: int,
        inject_color_profile=False,
        location_synthesis="Synthesize Random (US City)",
        datetime_synthesis="Synthesize Random",
        start_date="",
        end_date="",
        scene_type="Synthesize Random",
    ):

        if not EXIFTOOL_AVAILABLE:
            raise Exception("ExifTool is not installed. Please install exiftool to use this node.")

        output_dir = folder_paths.get_output_directory()

        final_filepath_for_return = ""
        results_for_ui = []

        for i in range(image.shape[0]):
            img_tensor = image[i:i+1]

            # 1. Build target filename (force JPEG)
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                "IMG", output_dir
            )
            file = f"{filename}_{counter:05}_.jpg"
            final_filepath = os.path.join(full_output_folder, file)

            image_seed = seed + counter
            rng = random.Random(image_seed)

            # 2. Convert tensor -> PIL, without any color transform
            img_np = img_tensor.squeeze(0).cpu().numpy()
            img_pil = Image.fromarray(
                (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8),
                'RGB'
            )

            # 3. Load authenticity metadata profile
            json_path = f"{profile_path}.json" if not profile_path.endswith('.json') else profile_path
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata_library = json.load(f)
            source_profile = rng.choice(metadata_library).copy()

            # 4. Start from profile metadata and clean it up
            final_tags = source_profile.copy()

            # Remove known undesired/raw-only keys
            for tag in ["EXIF:ThumbnailImage", "_instaraw_icc_profile_file"]:
                if tag in final_tags:
                    del final_tags[tag]

            # Remove any existing Orientation / ColorSpace – we'll set our own
            for key in list(final_tags.keys()):
                base = key.split(":", 1)[-1]
                if base in ("Orientation", "ColorSpace"):
                    del final_tags[key]

            # 5. Synthesize GPS based on requested mode
            bounds = None
            if location_synthesis == "From Profile":
                pass
            elif location_synthesis == "Synthesize Random (Any City)":
                all_cities = self.get_location_choices()[3:]
                if all_cities:
                    random_city_str = rng.choice(all_cities)
                    city_name, country_name = [x.strip() for x in random_city_str.split(',')]
                    bounds = self.get_random_city_from_country(country_name, rng)
            elif location_synthesis.startswith("Synthesize Random ("):
                country_name = location_synthesis[19:-1]  # between '(' and ')'
                bounds = self.get_random_city_from_country(country_name, rng)
            else:
                # Specific "City, Country"
                city_name, country_name = [x.strip() for x in location_synthesis.split(',')]
                for region, countries in self._locations_data.items():
                    if country_name in countries:
                        for city_data in countries[country_name]:
                            if city_data['city'] == city_name:
                                bounds = city_data
                                break

            if bounds:
                lat = rng.uniform(bounds['lat'][0], bounds['lat'][1])
                lon = rng.uniform(bounds['lon'][0], bounds['lon'][1])
                final_tags.update({
                    "EXIF:GPSLatitude": abs(lat),
                    "EXIF:GPSLatitudeRef": "N" if lat >= 0 else "S",
                    "EXIF:GPSLongitude": abs(lon),
                    "EXIF:GPSLongitudeRef": "E" if lon >= 0 else "W",
                })

            # 6. Synthesize datetime if requested
            if datetime_synthesis == "Synthesize Random":
                try:
                    start = datetime.datetime.fromisoformat(start_date)
                    end = datetime.datetime.fromisoformat(end_date)
                    random_date = start + (end - start) * rng.random()
                    exif_date_str = random_date.strftime("%Y:%m:%d %H:%M:%S")
                    sub_sec = rng.randint(100, 999)
                    final_tags.update({
                        "EXIF:DateTimeOriginal": exif_date_str,
                        "EXIF:CreateDate": exif_date_str,
                        "EXIF:ModifyDate": exif_date_str,
                        "EXIF:SubSecTimeOriginal": sub_sec,
                        "EXIF:SubSecTimeDigitized": sub_sec,
                    })
                except Exception as e:
                    print(f"⚠️ INSTARAW Synthesize: Invalid date format. Error: {e}")

            # 7. Scene-based exposure tweaks
            if scene_type == "Synthesize Random":
                actual_scene_type = rng.choice(
                    self.INPUT_TYPES()["optional"]["scene_type"][0][2:]
                )
            else:
                actual_scene_type = scene_type

            if actual_scene_type != "From Profile":
                settings = {}
                if actual_scene_type == "Daylight (Bright)":
                    settings = {
                        "EXIF:ISO": rng.choice([25, 32, 50]),
                        "EXIF:ExposureTime": 1 / rng.randint(1000, 4000),
                        "EXIF:FNumber": 2.2,
                    }
                elif actual_scene_type == "Night":
                    settings = {
                        "EXIF:ISO": rng.choice([800, 1600, 2500]),
                        "EXIF:ExposureTime": 1 / rng.randint(15, 60),
                        "EXIF:FNumber": 1.8,
                    }
                final_tags.update(settings)

            # 8. Give the image a fresh UUID for PhotoIdentifier if present
            if "MakerNotes:PhotoIdentifier" in final_tags:
                final_tags["MakerNotes:PhotoIdentifier"] = str(uuid.uuid4()).upper()

            # 9. Ensure dimensions tags are correct (bare tag names, numeric)
            width, height = img_pil.size
            final_tags.update({
                "ImageWidth": width,
                "ImageHeight": height,
                "ExifImageWidth": width,
                "ExifImageHeight": height,
                "XMP:RegionAppliedToDimensionsW": width,
                "XMP:RegionAppliedToDimensionsH": height,
            })

            # 10. Save a clean JPEG (no EXIF) to a temp file
            temp_dir = tempfile.gettempdir()
            temp_filepath = os.path.join(temp_dir, f"instaraw_img_{uuid.uuid4()}.jpg")
            img_pil.save(temp_filepath, quality=95, format='JPEG', exif=b'')

            # 11. Build ExifTool args
            arg_filepath = None
            try:
                print(f"\n--- Processing Image {i+1}/{image.shape[0]} (Seed: {image_seed}) ---")
                print(f"   - Target Filename: {os.path.basename(final_filepath)}")

                exiftool_cmd = shutil.which("exiftool") or shutil.which("exiftool.exe")
                if not exiftool_cmd:
                    raise RuntimeError("ExifTool not found in PATH")

                # -n   → numeric values (critical for enums like ColorSpace / Orientation)
                # -all= → nuke existing metadata from the temp JPEG
                command = [exiftool_cmd, "-n", "-all="]

                # Filter out non-writable / internal tags by prefix
                tags_for_argfile = {
                    k: v for k, v in final_tags.items()
                    if not k.startswith((
                        "File:",       # file system info
                        "Composite:",  # derived tags
                        "JFIF:",       # JFIF container
                        "_",           # internal keys like _instaraw...
                    ))
                }

                # Force Orientation & ColorSpace to known values (numeric, sRGB)
                # Orientation: 1 = Horizontal (normal)
                tags_for_argfile["Orientation"] = 1

                # REMOVE ExifTool-generated XMPToolkit to keep metadata authentic
                tags_for_argfile["XMP:XMPToolkit"] = ""

                # ColorSpace: 1 = sRGB (numeric code)
                if not inject_color_profile:
                    tags_for_argfile["ColorSpace"] = 1
                else:
                    # With ICC embedding we can either still force 1
                    # or let viewers rely on ICC tag. For safety we still set 1.
                    tags_for_argfile["ColorSpace"] = 1

                # Write all tags via an argfile
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding='utf-8',
                    suffix=".txt",
                    delete=False
                ) as arg_file:
                    arg_filepath = arg_file.name
                    for key, value in tags_for_argfile.items():
                        arg_file.write(f"-{key}={value}\n")

                command.extend(["-@", arg_filepath])

                # Optionally embed an sRGB ICC profile (no color transform)
                if inject_color_profile and SRGB_PROFILE_FILE.exists():
                    command.append(f"-ICC_Profile<={SRGB_PROFILE_FILE}")

                command.extend([
                    "-overwrite_original",
                    "-ignoreMinorErrors",
                    temp_filepath
                ])

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    encoding='utf-8',
                    errors='ignore'
                )
                if result.stderr:
                    # Log but don't fail hard – some Apple MakerNotes / MPF tags may warn
                    print("[ExifTool STDERR]")
                    print(result.stderr.strip())
                shutil.move(temp_filepath, final_filepath)

            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                if arg_filepath and os.path.exists(arg_filepath):
                    os.remove(arg_filepath)

            if i == 0:
                final_filepath_for_return = final_filepath

            results_for_ui.append({
                "filename": os.path.basename(final_filepath),
                "subfolder": subfolder,
                "type": "output"
            })

        return {
            "ui": {"images": results_for_ui},
            "result": (final_filepath_for_return,)
        }


NODE_CLASS_MAPPINGS = {
    "INSTARAW_SynthesizeAuthenticMetadata": INSTARAW_SynthesizeAuthenticMetadata
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_SynthesizeAuthenticMetadata": "💾 INSTARAW Synthesize Authentic Metadata"
}
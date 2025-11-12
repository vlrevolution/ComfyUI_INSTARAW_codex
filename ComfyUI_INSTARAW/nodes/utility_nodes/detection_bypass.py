# ---
# ComfyUI INSTARAW - AI Detection Bypass Node (Corrected & Improved)
# Part of the INSTARAW custom nodes collection by Instara
#
# This node encapsulates the multi-pass techniques from the Image-Detection-Bypass-Utility
# to help make AI-generated images statistically closer to real photographs.
#
# Copyright © 2025 Instara. All rights reserved.
# PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED
# ---

import torch
import numpy as np
from PIL import Image
import os
import tempfile
import time
import gc
from types import SimpleNamespace

# Import the processing logic from our new 'modules' folder
from ...modules.detection_bypass.processor import process_image

# Helper functions to convert between torch tensors and PIL images
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


class INSTARAW_DetectionBypass:
    """
    Applies a multi-pass post-processing pipeline to reduce AI-generated artifacts
    and bypass detection systems.
    """

    METHODS = ["Subtle (2-Pass)", "Aggressive (2-Pass)", "High Similarity (2-Pass)", "Ultra-Minimal (1-Pass)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (cls.METHODS, {
                    "default": "Ultra-Minimal (1-Pass)",
                    "tooltip": "Ultra-Minimal: Just AI Normalizer (least visible) | High Similarity: Light effects | Subtle: Moderate | Aggressive: Heavy (most effective)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
            },
            "optional": {
                "fft_ref_image": ("IMAGE", {"tooltip": "Real photo reference for FFT matching. HIGHLY RECOMMENDED for Ultra-Minimal and High Similarity modes."}),
                "awb_ref_image": ("IMAGE", {"tooltip": "Reference for Auto White Balance. Used in Ultra-Minimal and High Similarity modes to prevent color casts."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "INSTARAW/Post-Processing"

    def _run_pass(self, input_image_tensor, args, awb_ref_tensor=None, fft_ref_tensor=None):
        """Helper to run a single processing pass with robust temp file handling."""
        pil_img = tensor_to_pil(input_image_tensor)
        
        # Use NamedTemporaryFile correctly within `with` statements to ensure they are created
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input_file, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_output_file:
            tmp_input_path = tmp_input_file.name
            tmp_output_path = tmp_output_file.name

        tmp_files_to_clean = [tmp_input_path, tmp_output_path]

        try:
            pil_img.save(tmp_input_path)

            args.input = tmp_input_path
            args.output = tmp_output_path

            awb_ref_path, fft_ref_path = None, None
            
            if awb_ref_tensor is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_awb_file:
                    awb_ref_path = tmp_awb_file.name
                tensor_to_pil(awb_ref_tensor).save(awb_ref_path)
                tmp_files_to_clean.append(awb_ref_path)
            
            if fft_ref_tensor is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fft_file:
                    fft_ref_path = tmp_fft_file.name
                tensor_to_pil(fft_ref_tensor).save(fft_ref_path)
                tmp_files_to_clean.append(fft_ref_path)

            args.ref = awb_ref_path
            args.fft_ref = fft_ref_path

            process_image(args.input, args.output, args)
            
            # Open result, convert, and immediately close to release the file handle
            with Image.open(tmp_output_path) as result_pil:
                result_tensor = pil_to_tensor(result_pil.convert("RGB"))

            return result_tensor
        finally:
            # Force garbage collection and wait briefly for the OS (especially Windows)
            del pil_img
            gc.collect()
            time.sleep(0.05)

            # Clean up all temporary files robustly
            for f_path in tmp_files_to_clean:
                for _ in range(3): # Retry loop for stubborn files
                    try:
                        if os.path.exists(f_path):
                            os.unlink(f_path)
                        break
                    except PermissionError:
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Warning: Could not delete temp file {f_path}: {e}")
                        break

    def execute(self, image, method, seed, fft_ref_image=None, awb_ref_image=None):
        print(f"🚀 INSTARAW Detection Bypass: Starting '{method}' method.")
        
        # FIX: Create a complete base_args object with ALL possible parameters.
        # This prevents AttributeError when a parameter is accessed but wasn't set for a pass.
        base_args = SimpleNamespace(
            seed=seed, awb=False, ref=None, fft_ref=None, lut=None, blend=False,
            noise=False, clahe=False, fft=False, perturb=False, glcm=False, lbp=False,
            sim_camera=False, non_semantic=False,
            noise_std=0.02, clahe_clip=2.0, tile=8, cutoff=0.25, fstrength=0.9,
            randomness=0.05, phase_perturb=0.08, radial_smooth=5,
            fft_mode="auto", fft_alpha=1.0, perturb_magnitude=0.008,
            glcm_distances=[1], glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            glcm_levels=256, glcm_strength=0.9,
            lbp_radius=3, lbp_n_points=24, lbp_method='uniform', lbp_strength=0.9,
            no_no_bayer=True, jpeg_cycles=1, jpeg_qmin=88, jpeg_qmax=96,
            vignette_strength=0.35, chroma_strength=1.2, iso_scale=1.0,
            read_noise=2.0, hot_pixel_prob=1e-6, banding_strength=0.0, motion_blur_kernel=1,
            ns_iterations=500, ns_learning_rate=3e-4, ns_t_lpips=0.04, ns_t_l2=3e-5,
            ns_c_lpips=1e-2, ns_c_l2=0.6, ns_grad_clip=0.05,
            # Blending options to prevent errors
            blend_tolerance=10.0, blend_min_region=50, blend_max_samples=100000, blend_n_jobs=None
        )

        if method == "Subtle (2-Pass)":
            # Pass 1: AI Normalizer + Camera Sim
            args1 = SimpleNamespace(**base_args.__dict__)
            args1.non_semantic = True
            args1.sim_camera = True
            print("  - Pass 1/2: Applying AI Normalizer and Camera Simulation...")
            pass1_result = self._run_pass(image, args1)
            # Pass 2: Noise + Perturbation
            args2 = SimpleNamespace(**base_args.__dict__)
            args2.noise = True
            args2.perturb = True
            print("  - Pass 2/2: Applying Noise and Pixel Perturbation...")
            final_result = self._run_pass(pass1_result, args2)

        elif method == "Aggressive (2-Pass)":
            # Pass 1: Deep Restructuring
            args1 = SimpleNamespace(**base_args.__dict__)
            args1.fft = True
            args1.fstrength = 0.9
            args1.fft_mode = "ref" if fft_ref_image is not None else "auto"
            args1.glcm = True
            args1.glcm_distances = [2]
            args1.glcm_strength = 0.5
            args1.sim_camera = True
            args1.perturb = True
            if awb_ref_image is not None:
                args1.awb = True
            print("  - Pass 1/2: Applying FFT, GLCM, Camera Sim, and Perturbation...")
            pass1_result = self._run_pass(image, args1, awb_ref_tensor=awb_ref_image, fft_ref_tensor=fft_ref_image)
            # Pass 2: AI Normalizer + Noise
            args2 = SimpleNamespace(**base_args.__dict__)
            args2.non_semantic = True
            args2.noise = True
            print("  - Pass 2/2: Applying AI Normalizer and Noise...")
            final_result = self._run_pass(pass1_result, args2)

        elif method == "High Similarity (2-Pass)":
            if fft_ref_image is None:
                print("  ⚠️  WARNING: High Similarity mode works best with an FFT reference image!")
            args1 = SimpleNamespace(**base_args.__dict__)
            args1.fft = True
            args1.fstrength = 0.5
            args1.fft_mode = "ref" if fft_ref_image is not None else "auto"
            args1.awb = True
            args1.sim_camera = True
            args1.vignette_strength = 0.15
            args1.chroma_strength = 0.8
            args1.perturb = True
            args1.perturb_magnitude = 0.005
            print("  - Pass 1/2: Applying Light FFT, AWB, and Subtle Camera Effects...")
            pass1_result = self._run_pass(image, args1, awb_ref_tensor=awb_ref_image, fft_ref_tensor=fft_ref_image)
            # Pass 2: Conservative AI Normalizer
            args2 = SimpleNamespace(**base_args.__dict__)
            args2.non_semantic = True
            args2.ns_t_lpips = 0.02
            args2.ns_iterations = 300
            args2.noise = True
            args2.noise_std = 0.01
            print("  - Pass 2/2: Applying Conservative AI Normalizer and Minimal Noise...")
            final_result = self._run_pass(pass1_result, args2)

        elif method == "Ultra-Minimal (1-Pass)":
            # IMPROVED based on research: Use non_semantic for spectral work, not FFT matching.
            if fft_ref_image is None and awb_ref_image is None:
                print("  ⚠️  WARNING: Ultra-Minimal works best with a real photo connected to both 'fft_ref_image' and 'awb_ref_image' for color accuracy!")
            
            args1 = SimpleNamespace(**base_args.__dict__)
            
            # Use the SOTA spectral attack instead of the flawed FFT matching
            args1.non_semantic = True
            args1.ns_t_lpips = 0.025
            args1.ns_iterations = 222
            args1.ns_learning_rate = 2.5e-4
            
            # AWB is critical to prevent color cast
            args1.awb = True
            
            # Light camera simulation for iPhone-like authenticity
            args1.sim_camera = True
            args1.vignette_strength = 0.08
            args1.chroma_strength = 0.6
            args1.jpeg_cycles = 1
            args1.jpeg_qmin = 92
            args1.jpeg_qmax = 96

            # Very subtle perturbation
            args1.perturb = True
            args1.perturb_magnitude = 0.003
            
            # Disable redundant or less effective steps for this mode
            args1.fft = False # Replaced by non_semantic
            args1.noise = False # Allow external control for better quality noise

            print("  - Single pass: AI Normalizer (222 iter) + AWB + Light Camera Sim + Subtle Perturbation (No Noise)...")
            final_result = self._run_pass(image, args1, awb_ref_tensor=awb_ref_image, fft_ref_tensor=fft_ref_image)

        else:
            raise ValueError(f"Unknown method: {method}")

        print("✅ INSTARAW Detection Bypass: Processing complete.")
        return (final_result,)

NODE_CLASS_MAPPINGS = {
    "INSTARAW_DetectionBypass": INSTARAW_DetectionBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "INSTARAW_DetectionBypass": "🛡️ INSTARAW Detection Bypass",
}
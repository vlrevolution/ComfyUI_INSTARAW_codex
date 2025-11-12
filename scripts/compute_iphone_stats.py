#!/usr/bin/env python3
"""
Compute reference statistics from real iPhone photos (e.g., DPED dataset).

Usage:
    python scripts/compute_iphone_stats.py \
        --image-root /workspace/images/iphone_dped \
        --output /workspace/images/iphone_stats.npz

This aggregates spectral, chroma, and texture descriptors for every image
under --image-root and writes the summary to an .npz file that the ComfyUI
pipeline can load as its authenticity target.
"""

import argparse
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    from skimage.feature import greycomatrix, greycoprops  # older versions
except ImportError:
    from skimage.feature import graycomatrix as greycomatrix  # new API spelling
    from skimage.feature import graycoprops as greycoprops
import cv2


def radial_profile(magnitude: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute radial FFT magnitude profile."""
    h, w = magnitude.shape
    y, x = np.indices((h, w))
    center = np.array([h // 2, w // 2])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)
    max_r = min(r.max(), bins - 1)
    profile = np.bincount(r.ravel(), weights=magnitude.ravel(), minlength=bins)
    counts = np.bincount(r.ravel(), minlength=bins)
    counts[counts == 0] = 1
    return profile[: max_r + 1] / counts[: max_r + 1]


def process_image(path: pathlib.Path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    spectrum = radial_profile(mag, bins=512)

    pixels = arr.reshape(-1, 3)
    chroma_mean = pixels.mean(axis=0)
    chroma_cov = np.cov(pixels, rowvar=False)

    gray_u8 = (gray * 255).astype(np.uint8)
    glcm = greycomatrix(
        gray_u8,
        distances=[5],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = greycoprops(glcm, "contrast")[0, 0]
    homogeneity = greycoprops(glcm, "homogeneity")[0, 0]

    return spectrum, chroma_mean, chroma_cov, np.array([contrast, homogeneity])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Randomly sample this many images (default: use all)")
    args = parser.parse_args()

    spectra = []
    chroma_means = []
    chroma_covs = []
    glcm_stats = []

    image_paths = sorted(
        list(args.image_root.rglob("*.jpg")) + list(args.image_root.rglob("*.png"))
    )
    if not image_paths:
        raise SystemExit(f"No images found under {args.image_root}")

    if args.max_images and len(image_paths) > args.max_images:
        rng = np.random.default_rng(seed=42)
        image_paths = list(rng.choice(image_paths, size=args.max_images, replace=False))

    for path in tqdm(image_paths, desc="Processing images"):
        try:
            spectrum, c_mean, c_cov, glcm = process_image(path)
        except Exception as exc:
            print(f"Warning: skipping {path} ({exc})")
            continue

        spectra.append(spectrum)
        chroma_means.append(c_mean)
        chroma_covs.append(c_cov)
        glcm_stats.append(glcm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        spectra=np.stack(spectra, axis=0),
        chroma_mean=np.stack(chroma_means, axis=0),
        chroma_cov=np.stack(chroma_covs, axis=0),
        glcm=np.stack(glcm_stats, axis=0),
    )
    print(f"Saved stats for {len(spectra)} images to {args.output}")


if __name__ == "__main__":
    main()

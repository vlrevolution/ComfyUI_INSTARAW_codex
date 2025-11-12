# iPhone Reference Stats Setup

These instructions explain how to build the `iphone_stats.npz` reference file used by the ComfyUI INSTARAW pipeline to match real iPhone spectra, color science, and texture statistics.

## 1. Download the DPED iPhone Dataset

```bash
cd /workspace
mkdir images && cd images
wget https://download.ai-benchmark.com/s/rC6PwBK8exRomy8/download/original_images.gz
tar -xzvf original_images.gz
mv original_images/iphone ./iphone_dped
```

The `iphone_dped` folder now contains thousands of real iPhone captures.

## 2. Create a Python Environment (optional but recommended)

```bash
cd /workspace
python3 -m venv stats-env
source stats-env/bin/activate
pip install numpy scipy pillow scikit-image tqdm opencv-python
```

## 3. Compute the Statistics

From the repository root (or adjust the path), run:

```bash
python scripts/compute_iphone_stats.py \
    --image-root /workspace/images/iphone_dped \
    --output /workspace/images/iphone_stats.npz
```

This iterates over every JPEG/PNG under `iphone_dped`, computing:
- Radial FFT magnitude profiles (spectral statistics)
- Chroma mean/covariance
- GLCM texture metrics (contrast, homogeneity)

The aggregated results are written to `/workspace/images/iphone_stats.npz`.

## 4. Install the Stats for ComfyUI INSTARAW

Copy the output NPZ into the repo so the pipeline can load it:

```bash
cp /workspace/images/iphone_stats.npz \
   /home/dreamer/code/combined_test_codex/ComfyUI_INSTARAW/pretrained/iphone_stats.npz
```

Alternatively set `IPHONE_STATS_PATH=/path/to/iphone_stats.npz` before running ComfyUI so the code can find it.

## Notes

- You can substitute any other iPhone dataset as long as the images are real captures. The more diverse the scenes, the better the reference statistics.
- If the dataset structure differs, adjust `--image-root` accordingly; the script recursively scans all subfolders.
- Re-run the script whenever you update the dataset; simply overwrite the NPZ.

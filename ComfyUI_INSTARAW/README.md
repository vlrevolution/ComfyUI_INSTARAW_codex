# ComfyUI INSTARAW

## Deeploss Weights

The full UnMarker pipeline uses the original Deeploss-VGG perceptual model from the ai-watermark project. To comply with their distribution, run the helper script to download the required weight file directly from the authors' vault:

```bash
cd ComfyUI_INSTARAW
# Option A: I already have the weight locally
DEELOSS_LOCAL_FILE=/path/to/rgb_pnet_lin_vgg_trial0.pth bash scripts/download_deeploss_weights.sh

# Option B: I re-hosted the file somewhere else
DEEPLLSS_DIRECT_URL=https://my.mirror/rgb_pnet_lin_vgg_trial0.pth bash scripts/download_deeploss_weights.sh

# Option C: Fallback to the original multi-part archive (downloads ~30 GB once)
bash scripts/download_deeploss_weights.sh
```

The script installs `rgb_pnet_lin_vgg_trial0.pth` under `pretrained/deeploss/`. The runtime auto-detects that path or you can set `DEEPLOSS_WEIGHTS_DIR=/path/to/pretrained/deeploss`.

If you already cloned the full ai-watermark repository and ran `download_data_and_models.sh`, point `DEEPL‍OSS_LOCAL_FILE` at `ai-watermark/pretrained_models/loss_provider/weights/rgb_pnet_lin_vgg_trial0.pth` or set `DEEPLOSS_WEIGHTS_DIR` to that directory.

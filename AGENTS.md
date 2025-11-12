# Repository Guidelines

## Project Structure & Module Organization
`ai-watermark/` hosts the PyTorch attack stack: use `attack.py` as the entry point, keep scheme knobs in `attack_configs/<scheme>.yaml`, add implementations under `modules/attack/{unmark,diffusion,vae}`, and reflect watermark-specific edits in `systems/{configs,watermarkers}`. `ComfyUI_INSTARAW/` is the ComfyUI plug-in; its `nodes/` tree is grouped by node role (input/output/logic/api) and pulls helpers from the sibling `modules/` plus UI assets in `js/` and `fonts/`. Update the root docs (`COMFYUI_TESTING_GUIDE.md`, `DEBUG_BLUE_SHIFT.md`, `unmarker_implementation_roadmap.md`) whenever presets, troubleshooting steps, or milestones move.

## Build, Test, and Development Commands
- `cd ai-watermark && ./install.sh` installs CUDA, Torch, and all Python deps inside the active Conda env; rerun after touching `requirements.txt`.
- `cd ai-watermark && ./download_data_and_models.sh` grabs the pretrained watermark models and sample datasets into the git-ignored `datasets/` and `systems/weights/` trees.
- `python ai-watermark/attack.py -a UnMarker -e TreeRing -o outputs/treering_smoke --total_imgs 2` performs a quick CLI regression and writes `aggregated_results.yaml` for diffing.
- `pip install -r ComfyUI_INSTARAW/requirements.txt` keeps the plug-in in sync with the ComfyUI host; then link the folder into ComfyUI’s `custom_nodes/` and load the Balanced/full_balanced preset described in `COMFYUI_TESTING_GUIDE.md`.

## Coding Style & Naming Conventions
Stick to 4-space indentation, snake_case modules (`base_attack.py`, `tree_ring.py`), CamelCase classes, and f-strings with the shared `modules.logs.get_logger()` instead of prints. Configuration belongs in YAML next to the scheme it influences, while ComfyUI assets follow PascalCase component names and kebab-case CSS classes so presets match console output labels.

## Testing Guidelines
Exercise at least one semantic scheme (`StegaStamp` or `TreeRing`) and one non-semantic scheme (`StableSignature` or `SynthID`) before opening a PR. Capture the LPIPS/L2 values emitted into `aggregated_results.yaml`, plus console snippets from ComfyUI runs showing Stage 1/2 summaries and (when relevant) detector screenshots from Hive or Illuminarty. No automated test suite exists yet, so spell out the manual steps and expected metrics in the PR description.

## Commit & Pull Request Guidelines
Without Git history here, stick to short imperative commits (`add vine config`, `tighten instaraw balance`). PRs should describe intent, list touched configs/scripts, attach command output or screenshots, and mention any extra downloads reviewers need.

## Security & Configuration Tips
Never commit downloaded checkpoints or datasets; if you edit `download_data_and_models.sh`, record expected SHA output. Document tested image sizes and `stage_selector` values when proposing heavier defaults, and load detector or cloud credentials from env vars instead of hard-coding them.

# Nucleus Image FP8 Standalone

Minimal standalone workspace for testing `D-Squarius-Green-Jr/Nucleus-Image-FP8` outside ComfyUI.

This repo is set up for a transformer-only FP8 finetune:
- local folder `models/Nucleus-Image-FP8/` holds `config.json`, `moe_fp8_patch.py`, and `Nucleus-Image-FP8.safetensors`
- the script loads the base Nucleus pipeline separately from Hugging Face or a local clone
- the Python environment is standalone and does not depend on the ComfyUI venv

## Files
- `setup.sh`: create `.venv` and install dependencies
- `download_model.sh`: optional helper to download the transformer-only FP8 repo locally
- `generate.py`: load the local FP8 transformer, inject it into the base pipeline, and run inference
- `generate_image.sh`: shortest wrapper for everyday image generation

## Quick start
```bash
cd ~/nucleus-image-fp8
./generate_image.sh "A cinematic portrait of an astronaut in snowfall"
```

This uses the local FP8 transformer, the merged local base model, and writes an auto-named PNG into `outputs/`.

## Everyday usage
Simplest form:
```bash
./generate_image.sh "A cinematic portrait of an astronaut in snowfall"
```

Pick a custom basename:
```bash
./generate_image.sh "A cinematic portrait of an astronaut in snowfall" --name astronaut
```

Override size or steps:
```bash
./generate_image.sh "A chrome robot violinist under stage lights" \
  --height 1024 \
  --width 1024 \
  --steps 20
```

If you prefer calling Python directly, only `--prompt` is required now:
```bash
source .venv/bin/activate
python generate.py --local-files-only --prompt "A cinematic portrait of an astronaut in snowfall"
```

## Default memory strategy
The runner uses a conservative default layout:
- FP8 transformer sharded with `accelerate` across `cuda:0`, `cuda:1`, and CPU
- prompt embeddings generated on `cpu`
- VAE decode placed on the pipeline execution device

Default transformer memory budgets:
- `cuda:0`: `14GiB`
- `cuda:1`: `15GiB`
- `cpu`: `28GiB`

## Example two-GPU run
```bash
./generate_image.sh "A studio photo of a chrome robot violinist" \
  --text-encoder-device cpu \
  --vae-device execution \
  --max-gpu0 14GiB \
  --max-gpu1 15GiB \
  --max-cpu 28GiB
```

If you want to try using GPU 1 for prompt encoding too:
```bash
./generate_image.sh "A chrome robot violinist under stage lights" \
  --text-encoder-device cuda:1 \
  --name robot-gpu-text
```

## Notes
- This is a standalone inference harness, not a ComfyUI integration.
- The local FP8 repo is not a full pipeline repo; it only replaces the transformer.
- If the first run fails on memory, lower `--max-gpu0` or `--max-gpu1`, or keep the text encoder on CPU.

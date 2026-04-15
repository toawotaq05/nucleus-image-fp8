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
- `interactive_generate.sh`: keep the model loaded and generate many prompts in one session

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

Interactive form:
```bash
./interactive_generate.sh
```

On a two-GPU machine, the wrapper now automatically prefers a more balanced split by default:
- `--max-gpu0 10GiB`
- `--max-gpu1 15GiB`

That keeps more headroom on `cuda:0` for the rest of the pipeline and pushes more transformer blocks onto `cuda:1`.

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

Generate several variations from one loaded model and one prompt-encoding pass:
```bash
./generate_image.sh "White Owl in Prague" \
  --count 4
```

That reuses the already-loaded pipeline and prompt embeddings, then runs seeds `42`, `43`, `44`, and `45`.

If you want a fresh random set each time:
```bash
./generate_image.sh "White Owl in Prague" \
  --count 4 \
  --randomize-seeds
```

If you want to keep the model hot in memory and iterate on prompts:
```bash
./interactive_generate.sh --count 4 --randomize-seeds
```

Inside interactive mode:
```text
prompt> White Owl in Prague at dawn
prompt> /set steps 20
prompt> /set count 6
prompt> /random on
prompt> White Owl in Prague in snowfall
prompt> /negative blurry, low quality
prompt> White Owl in Prague, cinematic lighting
prompt> /quit
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

Base `generate.py` defaults:
- `cuda:0`: `14GiB`
- `cuda:1`: `15GiB`
- `cpu`: `28GiB`

The `generate_image.sh` wrapper overrides this on two-GPU machines to prefer:
- `cuda:0`: `10GiB`
- `cuda:1`: `15GiB`

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

This can speed up prompt encoding, but it is not the default because the Qwen3-VL text encoder is large and can compete with the transformer for VRAM on `cuda:1`.

## Performance notes
- CPU is only used for prompt embedding generation by default. The denoising pass still runs on the GPUs.
- Full single-GPU placement of the transformer on one 16 GB card is not currently viable in this setup. A direct `--transformer-device-map none` test failed during transformer placement on `cuda:0`.
- If you want the fastest stable setup on these two 16 GB cards, keep prompt encoding on CPU and let the transformer stay split across both GPUs.
- `--count N` avoids paying the model load and prompt-encoding cost N times. Only the denoising pass repeats for each image.
- `./interactive_generate.sh` goes one step further and avoids reloads across multiple prompts too. The model loads once per session.

## Notes
- This is a standalone inference harness, not a ComfyUI integration.
- The local FP8 repo is not a full pipeline repo; it only replaces the transformer.
- If the first run fails on memory, lower `--max-gpu0` or `--max-gpu1`, or keep the text encoder on CPU.

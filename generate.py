#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "Nucleus-Image-FP8"
DEFAULT_BASE_MODEL_DIR = PROJECT_DIR / "models" / "Nucleus-Image-base-merged"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
LOCAL_VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"


def ensure_local_venv() -> None:
    if os.environ.get("NUCLEUS_IMAGE_SKIP_VENV_REEXEC") == "1":
        return

    expected_prefix = (PROJECT_DIR / ".venv").resolve()
    current_prefix = Path(sys.prefix).resolve()

    if current_prefix == expected_prefix:
        return

    if not LOCAL_VENV_PYTHON.exists():
        raise SystemExit(
            f"Local venv python not found at {LOCAL_VENV_PYTHON}. Run ./setup.sh first."
        )

    os.execv(str(LOCAL_VENV_PYTHON), [str(LOCAL_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_local_venv()

import torch
from accelerate import dispatch_model, infer_auto_device_map
from diffusers import NucleusMoEImagePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Nucleus Image FP8 inference runner")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing config.json, moe_fp8_patch.py, and the FP8 .safetensors transformer file",
    )
    parser.add_argument(
        "--base-model",
        default=str(DEFAULT_BASE_MODEL_DIR if DEFAULT_BASE_MODEL_DIR.exists() else "NucleusAI/Nucleus-Image"),
        help="Base pipeline repo id or local directory with VAE, scheduler, text encoder, and processor",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--negative", default=None, help="Negative prompt")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to an auto-named PNG in outputs/")
    parser.add_argument("--name", default=None, help="Optional output basename without extension")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--return-index", type=int, default=-8)
    parser.add_argument(
        "--transformer-device-map",
        choices=["auto", "none"],
        default="auto",
        help="Use accelerate to shard the FP8 transformer across devices",
    )
    parser.add_argument("--max-gpu0", default="14GiB", help="Max memory budget for GPU 0 when sharding transformer")
    parser.add_argument("--max-gpu1", default="15GiB", help="Max memory budget for GPU 1 when sharding transformer")
    parser.add_argument("--max-cpu", default="28GiB", help="Max CPU memory budget when sharding transformer")
    parser.add_argument(
        "--text-encoder-device",
        default="cpu",
        help="Device used only for prompt embedding generation, e.g. cpu or cuda:1",
    )
    parser.add_argument(
        "--vae-device",
        default="execution",
        help="Device for VAE decode: execution, cpu, cuda:0, cuda:1",
    )
    parser.add_argument(
        "--keep-text-encoder-on-device",
        action="store_true",
        help="Keep the text encoder on its device after prompt encoding instead of moving it back to CPU",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Avoid network access during base model load")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def resolve_path_or_repo(path_or_repo: str) -> str:
    path = Path(path_or_repo)
    return str(path.resolve()) if path.exists() else path_or_repo


def slugify_prompt(text: str, max_len: int = 48) -> str:
    chars = []
    prev_dash = False
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
            prev_dash = False
        elif not prev_dash:
            chars.append("-")
            prev_dash = True
    slug = "".join(chars).strip("-")
    if not slug:
        slug = "image"
    return slug[:max_len].rstrip("-")


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = args.name.strip() if args.name else slugify_prompt(args.prompt)
    return (DEFAULT_OUTPUT_DIR / f"{stamp}-{base}.png").resolve()


def load_patch_module(model_dir: Path):
    patch_path = model_dir / "moe_fp8_patch.py"
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing patch module: {patch_path}")

    spec = importlib.util.spec_from_file_location("nucleus_local_fp8_patch", patch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import local patch module from {patch_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_transformer_weights(model_dir: Path) -> Path:
    preferred = model_dir / "Nucleus-Image-FP8.safetensors"
    if preferred.exists():
        return preferred

    candidates = sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors file found in {model_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple .safetensors files found in {model_dir}: {[p.name for p in candidates]}")
    return candidates[0]


def build_transformer_max_memory(args: argparse.Namespace) -> dict:
    max_memory = {"cpu": args.max_cpu}
    if torch.cuda.device_count() > 0:
        max_memory[0] = args.max_gpu0
    if torch.cuda.device_count() > 1:
        max_memory[1] = args.max_gpu1
    return max_memory


def summarize_device_map(device_map: dict) -> str:
    grouped: dict[str, list[str]] = {}
    for module_name, device in device_map.items():
        grouped.setdefault(str(device), []).append(module_name)
    parts = []
    for device, names in grouped.items():
        names_preview = ", ".join(names[:4])
        suffix = "" if len(names) <= 4 else f", ... ({len(names)} entries)"
        parts.append(f"{device}: {names_preview}{suffix}")
    return " | ".join(parts)


def primary_module_device(module: torch.nn.Module) -> str:
    if hasattr(module, "hf_device_map") and getattr(module, "hf_device_map"):
        values = list(module.hf_device_map.values())
        if values:
            return str(values[0])

    try:
        first_param = next(module.parameters())
        return str(first_param.device)
    except StopIteration:
        return "cpu"


def maybe_dispatch_transformer(transformer: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.transformer_device_map == "none" or torch.cuda.device_count() == 0:
        target = "cuda:0" if torch.cuda.is_available() else "cpu"
        transformer = transformer.to(target)
        print(f"Transformer placed on {target}")
        return transformer

    max_memory = build_transformer_max_memory(args)
    no_split = getattr(transformer, "_no_split_modules", None)
    device_map = infer_auto_device_map(
        transformer,
        max_memory=max_memory,
        no_split_module_classes=no_split,
    )
    print("Transformer auto device map:")
    print(f"  {summarize_device_map(device_map)}")
    transformer = dispatch_model(transformer, device_map=device_map, offload_buffers=True)
    setattr(transformer, "hf_device_map", device_map)
    return transformer


def base_model_dtypes(args: argparse.Namespace) -> torch.dtype | dict:
    default_dtype = torch_dtype(args.dtype)
    if args.text_encoder_device == "cpu":
        return {"default": default_dtype, "text_encoder": torch.float32}
    return default_dtype


def configure_text_encoder_for_prompt_encoding(pipe: NucleusMoEImagePipeline) -> None:
    text_cfg = getattr(pipe.text_encoder.config, "text_config", None)
    if text_cfg is None:
        return
    if hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = False


def encode_prompts(pipe: NucleusMoEImagePipeline, args: argparse.Namespace):
    text_device = torch.device(args.text_encoder_device)
    pipe.text_encoder.to(text_device)
    configure_text_encoder_for_prompt_encoding(pipe)

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=args.prompt,
            device=text_device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
            return_index=args.return_index,
        )

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if args.guidance > 1:
            negative_text = args.negative if args.negative is not None else ""
            negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
                prompt=negative_text,
                device=text_device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
                return_index=args.return_index,
            )

    if not args.keep_text_encoder_on_device and text_device.type == "cuda":
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    return prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask


def cast_prompt_tensors(
    args: argparse.Namespace,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    negative_prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds_mask: torch.Tensor | None,
):
    target_dtype = torch_dtype(args.dtype)
    prompt_embeds = prompt_embeds.to(dtype=target_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=target_dtype)
    return prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask


def place_vae(pipe: NucleusMoEImagePipeline, args: argparse.Namespace) -> str:
    target = args.vae_device
    if target == "execution":
        target = str(pipe._execution_device)
    pipe.vae.to(target)
    return target


def build_generator(device_str: str, seed: int) -> torch.Generator:
    device = torch.device(device_str)
    gen_device = device.type if device.type == "cpu" else device_str
    return torch.Generator(gen_device).manual_seed(seed)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model).resolve()
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"cuda:{i}={props.name} vram={props.total_memory / 1024**3:.2f} GiB")

    patch_module = load_patch_module(model_dir)
    transformer_weights = find_transformer_weights(model_dir)
    print(f"Using local FP8 transformer: {transformer_weights.name}")

    started = time.time()
    transformer = patch_module.load_fp8_safetensors_transformer(
        str(transformer_weights),
        config_dir=str(model_dir),
    )
    print(f"Transformer loaded in {time.time() - started:.1f}s")

    started = time.time()
    transformer = maybe_dispatch_transformer(transformer, args)
    print(f"Transformer placement finished in {time.time() - started:.1f}s")

    base_model = resolve_path_or_repo(args.base_model)
    print(f"Loading base pipeline from {base_model}")
    started = time.time()
    pipe = NucleusMoEImagePipeline.from_pretrained(
        base_model,
        transformer=transformer,
        torch_dtype=base_model_dtypes(args),
        local_files_only=args.local_files_only,
    )
    print(f"Base pipeline loaded in {time.time() - started:.1f}s")

    execution_device = str(pipe._execution_device)
    vae_device = place_vae(pipe, args)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    print(f"Pipeline execution device: {execution_device}")
    print(f"Transformer primary device: {primary_module_device(pipe.transformer)}")
    print(f"Text encoder device for embedding pass: {args.text_encoder_device}")
    print(f"VAE decode device: {vae_device}")

    started = time.time()
    prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask = encode_prompts(pipe, args)
    prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask = cast_prompt_tensors(
        args,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
    )
    print(f"Prompt encoding finished in {time.time() - started:.1f}s")

    generator = build_generator(execution_device, args.seed)

    call_kwargs = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance,
        "num_inference_steps": args.steps,
        "max_sequence_length": args.max_sequence_length,
        "return_index": args.return_index,
        "generator": generator,
        "output_type": "pil",
    }

    print("Running inference...")
    started = time.time()
    result = pipe(**call_kwargs)
    image = result.images[0]
    image.save(output_path)
    print(f"Saved {output_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()

"""
SwiGLUExperts FP8 monkey-patch.

Imported by both 03_quantize_fp8.py (before loading BF16 model) and 04_test_inference.py
(before loading FP8 model). Idempotent: importing twice is a no-op.

What it does:
- Adds two persistent buffers (gate_up_proj_scale, down_proj_scale) to every SwiGLUExperts.
- Replaces _run_experts_for_loop to dequantize per-expert weights on-the-fly when stored as fp8_e4m3fn.
- Forces use_grouped_mm=False (the grouped_mm kernel doesn't accept fp8 e4m3 inputs as of torch 2.11).
- If weights are still bf16 (un-quantized model), behavior is identical to the original SwiGLU forward.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.transformers import transformer_nucleusmoe_image as _moe_mod

FP8_E4M3_MAX = 448.0  # float8_e4m3fn dynamic range
SCALE_DTYPE = torch.bfloat16

_PATCH_FLAG = "_nucleus_fp8_patched_v1"


def apply_patch():
    if getattr(_moe_mod.SwiGLUExperts, _PATCH_FLAG, False):
        return  # already patched

    cls = _moe_mod.SwiGLUExperts
    orig_init = cls.__init__

    def patched_init(self, hidden_size, moe_intermediate_dim, num_experts, use_grouped_mm: bool = False):
        # Force use_grouped_mm off — fp8 inputs aren't supported by F.grouped_mm on Blackwell yet,
        # and the for-loop path is the patched/quantized path.
        orig_init(self, hidden_size, moe_intermediate_dim, num_experts, use_grouped_mm=False)
        # Persistent buffers; default to ones so a non-quantized BF16 checkpoint still produces
        # mathematically identical output through patched_for_loop.
        self.register_buffer(
            "gate_up_proj_scale",
            torch.ones(num_experts, 1, 2 * moe_intermediate_dim, dtype=SCALE_DTYPE),
            persistent=True,
        )
        self.register_buffer(
            "down_proj_scale",
            torch.ones(num_experts, 1, hidden_size, dtype=SCALE_DTYPE),
            persistent=True,
        )

    def patched_for_loop(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        n_list = num_tokens_per_expert.tolist()
        n_real = sum(n_list)
        n_pad = x.shape[0] - n_real
        x_per_expert = torch.split(x[:n_real], split_size_or_sections=n_list, dim=0)

        is_fp8 = self.gate_up_proj.dtype == torch.float8_e4m3fn
        compute_dtype = x.dtype  # bf16 in normal use

        outs = []
        for i, xe in enumerate(x_per_expert):
            if is_fp8:
                w_gu = self.gate_up_proj[i].to(compute_dtype) * self.gate_up_proj_scale[i].to(compute_dtype)
                w_dn = self.down_proj[i].to(compute_dtype) * self.down_proj_scale[i].to(compute_dtype)
            else:
                w_gu = self.gate_up_proj[i]
                w_dn = self.down_proj[i]
            gate_up = torch.matmul(xe, w_gu)
            gate, up = gate_up.chunk(2, dim=-1)
            outs.append(torch.matmul(F.silu(gate) * up, w_dn))

        out = torch.cat(outs, dim=0)
        return torch.vstack((out, out.new_zeros((n_pad, out.shape[-1]))))

    def patched_forward(self, x, num_tokens_per_expert):
        return patched_for_loop(self, x, num_tokens_per_expert)

    cls.__init__ = patched_init
    cls._run_experts_for_loop = patched_for_loop
    cls.forward = patched_forward
    setattr(cls, _PATCH_FLAG, True)


def _patched_fp8_linear_forward(self, x):
    # Dequantize qdata fp8 + per-output-channel scale on the fly, then F.linear.
    w = (self.weight.to(torch.float32) * self._fp8_scale.to(torch.float32)).to(x.dtype)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def install_fp8_linear(linear: nn.Linear, fp8_qdata: torch.Tensor, scale: torch.Tensor) -> None:
    """Convert one Linear instance to fp8-stored, per-instance forward override."""
    assert isinstance(linear, nn.Linear), f"expected nn.Linear, got {type(linear).__name__}"
    linear.weight = nn.Parameter(fp8_qdata, requires_grad=False)
    linear.register_buffer("_fp8_scale", scale.to(SCALE_DTYPE))
    # Bind forward at the instance level — does NOT affect other nn.Linear instances.
    linear.forward = _patched_fp8_linear_forward.__get__(linear, type(linear))


def load_fp8_safetensors_transformer(safetensors_path: str, config_dir: str | None = None):
    """
    Load a Nucleus-Image FP8 transformer from a single safetensors file. The safetensors
    holds raw fp8 + per-channel bf16 scale tensors (no torchao wrapper objects).

    SwiGLUExperts: weights `gate_up_proj` / `down_proj` (fp8) + buffers `*_scale` (bf16).
    Standard nn.Linear: weight (fp8) + buffer `_fp8_scale` (bf16) — forward override
    installed by `install_fp8_linear` so the dequantize+matmul happens inline.
    """
    from pathlib import Path
    import json
    from safetensors.torch import safe_open
    from diffusers import AutoModel
    from accelerate import init_empty_weights

    apply_patch()

    cfg_dir = Path(config_dir) if config_dir else Path(safetensors_path).parent
    cfg = json.loads((cfg_dir / "config.json").read_text(encoding="utf-8"))

    # Locate the actual transformer class via the AutoModel mapping.
    from diffusers import NucleusMoEImageTransformer2DModel
    with init_empty_weights():
        model = NucleusMoEImageTransformer2DModel.from_config(cfg)

    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)

    # Two passes over the file:
    # Pass 1: identify Linears that have an `_fp8_scale` sibling — install fp8 weight + scale + forward.
    # Pass 2: load everything else via standard assign.
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        # Linear keys with fp8 representation: those with a `_fp8_scale` buffer.
        linear_paths = set()
        for k in all_keys:
            if k.endswith("._fp8_scale"):
                base = k[: -len("._fp8_scale")]
                if (base + ".weight") in all_keys:
                    linear_paths.add(base)

        # Pass 1: install fp8 Linears.
        for base in linear_paths:
            module = model.get_submodule(base)
            qdata = f.get_tensor(base + ".weight").to("cpu")
            scale = f.get_tensor(base + "._fp8_scale").to("cpu")
            bias = None
            if (base + ".bias") in all_keys:
                bias = f.get_tensor(base + ".bias").to("cpu").to(torch.bfloat16)
            # The init_empty_weights model has meta-device weights; replace cleanly.
            if module.bias is not None and bias is not None:
                module.bias = nn.Parameter(bias)
            install_fp8_linear(module, qdata, scale)

        # Pass 2: rest of the state dict (SwiGLUExperts fp8 + scales, norms, embeddings, etc.).
        for k in all_keys:
            # Skip keys we already handled in pass 1.
            if any(k == p + ".weight" or k == p + "._fp8_scale" or k == p + ".bias" for p in linear_paths):
                continue
            t = f.get_tensor(k).to("cpu")
            module_path, _, attr = k.rpartition(".")
            module = model.get_submodule(module_path) if module_path else model
            cur = getattr(module, attr, None)
            if isinstance(cur, nn.Parameter) or cur is None:
                # cur may be a meta-device Parameter — replace whole thing
                if t.dtype.is_floating_point and t.dtype not in fp8_dtypes and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                setattr(module, attr, nn.Parameter(t, requires_grad=False) if attr in dict(module.named_parameters(recurse=False)) or cur is None or isinstance(cur, nn.Parameter) else t)
            else:
                # Buffer
                if t.dtype.is_floating_point and t.dtype not in fp8_dtypes and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                module.register_buffer(attr, t)

    # Final cleanup: any param/buffer still on meta should never happen, but assert.
    meta_left = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_left:
        raise RuntimeError(f"Some parameters still on meta device after load: {meta_left[:5]}... ({len(meta_left)} total)")

    print(f"  load_fp8_safetensors_transformer: installed {len(linear_paths)} fp8 Linears")
    return model


def load_fp8_transformer(model_dir: str):
    """
    Load an FP8-quantized Nucleus transformer from `model_dir` while preserving on-disk
    fp8_e4m3fn dtypes for SwiGLUExperts weights. Other floating params are normalized to bf16.

    Why: AutoModel.from_pretrained(torch_dtype=torch.bfloat16) force-casts ALL floating
    weights including fp8. We do the standard load (which casts to bf16), then re-stream
    the on-disk shards and reassign every fp8_e4m3fn tensor as a fresh nn.Parameter so
    the dtype is preserved. TorchAO Float8Tensor wrappers for nn.Linear are restored
    correctly by the standard loader (its DiffusersAutoQuantizer hook intercepts them).
    """
    import json
    from pathlib import Path
    from diffusers import AutoModel

    apply_patch()  # ensure SwiGLUExperts is patched before construction

    model = AutoModel.from_pretrained(
        model_dir,
        use_safetensors=False,
        low_cpu_mem_usage=True,
    )

    # Re-stream disk to recover fp8 dtypes (auto-cast lost them).
    md = Path(model_dir)
    idx_path = md / "diffusion_pytorch_model.bin.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        files = sorted(set(idx["weight_map"].values()))
    else:
        files = ["diffusion_pytorch_model.bin"]

    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    fp8_reassigned = 0
    for fname in files:
        shard = torch.load(md / fname, map_location="cpu", weights_only=False)
        for key, tensor in shard.items():
            if not hasattr(tensor, "dtype") or tensor.dtype not in fp8_dtypes:
                continue
            module_path, _, attr = key.rpartition(".")
            module = model.get_submodule(module_path)
            cur = getattr(module, attr, None)
            if isinstance(cur, nn.Parameter):
                setattr(module, attr, nn.Parameter(tensor, requires_grad=False))
            else:
                # buffer
                module.register_buffer(attr, tensor)
            fp8_reassigned += 1
        del shard

    # Final pass: any non-fp8 floating param in fp32/fp16 → bf16 (uniformity).
    for _, p in model.named_parameters():
        if p.dtype in fp8_dtypes:
            continue
        if p.dtype.is_floating_point and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    for _, b in model.named_buffers():
        if b.dtype in fp8_dtypes:
            continue
        if b.dtype.is_floating_point and b.dtype != torch.bfloat16:
            b.data = b.data.to(torch.bfloat16)

    print(f"  load_fp8_transformer: re-assigned {fp8_reassigned} fp8 tensors from shards")
    return model


@torch.no_grad()
def quantize_swiglu_experts_(module: nn.Module) -> dict:
    """Quantize a single SwiGLUExperts module in-place. Returns a small report dict."""
    assert type(module).__name__ == "SwiGLUExperts", f"expected SwiGLUExperts, got {type(module).__name__}"
    device = module.gate_up_proj.device

    def _quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # w shape: (num_experts, in_dim, out_dim). Per-expert per-output-channel scale on dim=-2.
        w32 = w.detach().to(torch.float32)
        scale = w32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-12) / FP8_E4M3_MAX
        q = (w32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return q, scale.to(SCALE_DTYPE)

    q_gu, s_gu = _quant(module.gate_up_proj.data)
    q_dn, s_dn = _quant(module.down_proj.data)

    # Reconstruction error gate. Use rel L2 (||W_hat - W||_2 / ||W||_2) per tensor — the right
    # metric for forward fidelity (matmul amplifies L2-style error, not max-abs).
    w_gu_orig = module.gate_up_proj.data.to(torch.float32)
    w_dn_orig = module.down_proj.data.to(torch.float32)
    rec_gu = q_gu.to(torch.float32) * s_gu.to(torch.float32)
    rec_dn = q_dn.to(torch.float32) * s_dn.to(torch.float32)

    rep = {
        "gu_rel_l2": ((rec_gu - w_gu_orig).norm() / w_gu_orig.norm().clamp(min=1e-12)).item(),
        "dn_rel_l2": ((rec_dn - w_dn_orig).norm() / w_dn_orig.norm().clamp(min=1e-12)).item(),
        # Keep the loose rel-max-err for visibility but no longer used as a gate.
        "gu_rel_max": ((rec_gu - w_gu_orig).abs().amax() / w_gu_orig.abs().amax().clamp(min=1e-12)).item(),
        "dn_rel_max": ((rec_dn - w_dn_orig).abs().amax() / w_dn_orig.abs().amax().clamp(min=1e-12)).item(),
    }

    # In-place replacement
    module.gate_up_proj = nn.Parameter(q_gu.to(device), requires_grad=False)
    module.down_proj = nn.Parameter(q_dn.to(device), requires_grad=False)
    module.gate_up_proj_scale.data = s_gu.to(device)
    module.down_proj_scale.data = s_dn.to(device)
    return rep

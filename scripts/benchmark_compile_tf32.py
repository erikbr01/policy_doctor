"""Benchmark torch.compile and TF32 speedup on diffusion policy backbones.

Run with:
    conda run -n robocasa python scripts/benchmark_compile_tf32.py

Reports forward and backward pass throughput for ConditionalUnet1D and
TransformerForDiffusion under four flag combinations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "cupid"))

DEVICE = torch.device("cuda:0")
WARMUP = 30
REPS = 200


# ---------------------------------------------------------------------------
# Model + input factories
# Each factory returns (model, make_inputs) where make_inputs() -> tuple of tensors.
# ---------------------------------------------------------------------------

def make_unet(device: torch.device):
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    model = ConditionalUnet1D(
        input_dim=10,
        global_cond_dim=20,            # obs_dim=10 * n_obs_steps=2
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
    ).to(device)

    B, T, Da, Do = 64, 16, 10, 20

    def make_inputs():
        return (
            torch.randn(B, T, Da, device=device),
            torch.randint(0, 100, (B,), device=device),
        ), {"global_cond": torch.randn(B, Do, device=device)}

    return model, make_inputs


def make_transformer(device: torch.device):
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

    model = TransformerForDiffusion(
        input_dim=10,
        output_dim=10,
        horizon=16,
        n_obs_steps=2,
        cond_dim=10,
        n_layer=8,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    ).to(device)

    B, T, Da, No, Do = 64, 16, 10, 2, 10

    def make_inputs():
        return (
            torch.randn(B, T, Da, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
            torch.randn(B, No, Do, device=device),
        ), {}

    return model, make_inputs


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_forward(model: nn.Module, make_inputs: Callable, warmup: int, reps: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            args, kwargs = make_inputs()
            model(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            args, kwargs = make_inputs()
            model(*args, **kwargs)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def _time_fwd_bwd(model: nn.Module, make_inputs: Callable, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        args, kwargs = make_inputs()
        out = model(*args, **kwargs)
        out.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        args, kwargs = make_inputs()
        out = model(*args, **kwargs)
        out.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_backbone(backbone_name: str, factory: Callable) -> list[dict]:
    results = []
    for compile_ in (False, True):
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32

            model, make_inputs = factory(DEVICE)
            if compile_:
                model = torch.compile(model, fullgraph=True, dynamic=False)

            fwd_ms    = _time_forward(model, make_inputs, WARMUP, REPS)
            fwdbwd_ms = _time_fwd_bwd(model, make_inputs, WARMUP, REPS)

            tag = f"compile={'Y' if compile_ else 'N'} tf32={'Y' if tf32 else 'N'}"
            print(f"  {backbone_name:12s}  {tag}  fwd={fwd_ms:6.2f}ms  fwd+bwd={fwdbwd_ms:6.2f}ms")
            results.append({
                "label": backbone_name,
                "compile": compile_,
                "tf32": tf32,
                "fwd_ms": fwd_ms,
                "fwdbwd_ms": fwdbwd_ms,
            })
    return results


def print_speedup_table(results: list[dict]) -> None:
    base = next(r for r in results if not r["compile"] and not r["tf32"])

    print(f"\n  {'config':30s}  {'fwd':>8s}  {'speedup':>8s}  {'fwd+bwd':>8s}  {'speedup':>8s}")
    print("  " + "-" * 72)
    for r in results:
        tag = f"compile={'Y' if r['compile'] else 'N'}  tf32={'Y' if r['tf32'] else 'N'}"
        print(
            f"  {tag:30s}  "
            f"{r['fwd_ms']:7.2f}ms  "
            f"{base['fwd_ms'] / r['fwd_ms']:>7.2f}x  "
            f"{r['fwdbwd_ms']:7.2f}ms  "
            f"{base['fwdbwd_ms'] / r['fwdbwd_ms']:>7.2f}x"
        )


def main():
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup : {WARMUP}  Reps: {REPS}\n")

    for name, factory in [("UNet", make_unet), ("Transformer", make_transformer)]:
        print(f"=== {name} ===")
        results = bench_backbone(name, factory)
        print_speedup_table(results)
        print()


if __name__ == "__main__":
    main()

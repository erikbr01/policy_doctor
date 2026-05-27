"""Test whether torch 2.5+ fixes obs_encoder torch.compile backward (TensorAlias bug).

Run with:
    conda activate torch25_bench
    python scripts/bench_compile_torch25.py
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "third_party/cupid")

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

SHAPE_META = {
    "action": {"shape": [8]},
    "obs": {
        "hand_camera_image":     {"shape": [3, 256, 256], "type": "rgb"},
        "exterior_image_1_left": {"shape": [3, 256, 256], "type": "rgb"},
        "joint_positions":       {"shape": [7]},
        "gripper_position":      {"shape": [1]},
    },
}


def build_policy():
    sched = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon",
    )
    p = DiffusionUnetHybridImagePolicy(
        shape_meta=SHAPE_META, noise_scheduler=sched,
        horizon=16, n_action_steps=8, n_obs_steps=2,
        num_inference_steps=100, obs_as_global_cond=True,
        down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        cond_predict_scale=True, obs_encoder_group_norm=True,
        eval_fixed_crop=True, crop_shape=[224, 224],
        pretrained_backbone=False, diffusion_step_embed_dim=128,
    ).cuda()
    norm = LinearNormalizer()
    identity = SingleFieldLinearNormalizer.create_identity()
    norm["action"] = identity
    for k in SHAPE_META["obs"]:
        norm[k] = identity
    p.set_normalizer(norm)
    p.normalizer.cuda()
    return p


def make_batch(B=16):
    return {
        "obs": {
            "hand_camera_image":     torch.rand(B, 2, 3, 256, 256, device="cuda"),
            "exterior_image_1_left": torch.rand(B, 2, 3, 256, 256, device="cuda"),
            "joint_positions":       torch.randn(B, 2, 7, device="cuda"),
            "gripper_position":      torch.randn(B, 2, 1, device="cuda"),
        },
        "action":        torch.randn(B, 16, 8, device="cuda"),
        "action_is_pad": torch.zeros(B, 16, dtype=torch.bool, device="cuda"),
    }


def bench(fn, n_warmup, n_runs):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000


def main():
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)

    # ── Eager baseline ──────────────────────────────────────────────────────
    print("\n[1/3] Building eager policy...", flush=True)
    p_eager = build_policy()
    batch = make_batch(B=64)

    def fwdbwd_eager():
        loss = p_eager.compute_loss(batch)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        loss.backward()
        p_eager.zero_grad()

    eager_ms = bench(fwdbwd_eager, n_warmup=3, n_runs=10)
    print(f"  eager fwd+bwd: {eager_ms:.1f}ms", flush=True)

    # ── UNet-only compile (current code path, baseline for comparison) ──────
    print("\n[2/3] Building UNet-only compiled policy...", flush=True)
    p_unet = build_policy()
    if hasattr(p_unet, "model"):
        p_unet.model = torch.compile(p_unet.model, fullgraph=True, dynamic=False)

    def fwdbwd_unet():
        loss = p_unet.compute_loss(batch)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        loss.backward()
        p_unet.zero_grad()

    # warmup for inductor
    print("  warming up UNet compile...", flush=True)
    for _ in range(8):
        fwdbwd_unet()
    unet_ms = bench(fwdbwd_unet, n_warmup=0, n_runs=10)
    print(f"  unet-only compile fwd+bwd: {unet_ms:.1f}ms  ({eager_ms/unet_ms:.2f}×)", flush=True)

    # ── Full compile: obs_encoder + UNet ────────────────────────────────────
    print("\n[3/3] Building FULL compiled policy (obs_encoder + UNet)...", flush=True)
    print("  This is the path that crashed with TensorAlias on torch 2.4.x", flush=True)
    p_full = build_policy()
    if hasattr(p_full, "obs_encoder"):
        p_full.obs_encoder = torch.compile(
            p_full.obs_encoder, fullgraph=False, dynamic=False
        )
    if hasattr(p_full, "model"):
        p_full.model = torch.compile(p_full.model, fullgraph=True, dynamic=False)

    def fwdbwd_full():
        loss = p_full.compute_loss(batch)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        loss.backward()
        p_full.zero_grad()

    print("  warming up full compile...", flush=True)
    try:
        for _ in range(8):
            fwdbwd_full()
        full_ms = bench(fwdbwd_full, n_warmup=0, n_runs=10)
        print(f"  full compile fwd+bwd: {full_ms:.1f}ms  ({eager_ms/full_ms:.2f}×)", flush=True)
        print("\n=== RESULT: obs_encoder compile + backward SUCCEEDED ===", flush=True)
        print(f"  TensorAlias bug appears FIXED in torch {torch.__version__}", flush=True)
        speedup_tag = f"speedup={eager_ms/full_ms:.2f}x"
        print(f"batch=64 {speedup_tag}", flush=True)
    except Exception as e:
        print(f"\n=== RESULT: obs_encoder compile + backward FAILED ===", flush=True)
        print(f"  Error: {e}", flush=True)
        print(f"  Bug NOT fixed in torch {torch.__version__}", flush=True)
        print(f"batch=64 speedup=FAILED", flush=True)


if __name__ == "__main__":
    main()

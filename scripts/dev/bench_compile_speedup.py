"""Benchmark torch.compile forward and backward speedup for the droid image policy.

Run with:
    conda activate cupid_torch2
    python scripts/bench_compile_speedup.py
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "third_party/cupid")

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.ddp_util import compile_policy

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


def make_batch(B):
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


def sync():
    torch.cuda.synchronize()


def bench(fn, n_warmup, n_runs):
    for _ in range(n_warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    sync()
    return (time.perf_counter() - t0) / n_runs * 1000  # ms


def main():
    B = 64
    N_WARMUP, N_RUNS = 5, 20
    N_WARMUP_COMPILED = 10  # extra warmup for inductor

    batch = make_batch(B)

    # ── Eager ──────────────────────────────────────────────────────────────
    print("Building eager policy...", flush=True)
    p_eager = build_policy()

    def eager_fwd():
        with torch.no_grad():
            p_eager.compute_loss(batch)

    def eager_fwdbwd():
        loss = p_eager.compute_loss(batch)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        loss.backward()
        p_eager.zero_grad()

    eager_fwd_ms    = bench(eager_fwd,    N_WARMUP, N_RUNS)
    eager_fwdbwd_ms = bench(eager_fwdbwd, N_WARMUP, N_RUNS)
    print(f"Eager: fwd={eager_fwd_ms:.1f}ms  fwd+bwd={eager_fwdbwd_ms:.1f}ms", flush=True)

    # ── Compiled ───────────────────────────────────────────────────────────
    print("Building compiled policy (inductor JIT on first call)...", flush=True)
    p_comp = build_policy()
    compile_policy(p_comp)

    def comp_fwd():
        with torch.no_grad():
            p_comp.compute_loss(batch)

    def comp_fwdbwd():
        loss = p_comp.compute_loss(batch)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        loss.backward()
        p_comp.zero_grad()

    comp_fwd_ms    = bench(comp_fwd,    N_WARMUP_COMPILED, N_RUNS)
    comp_fwdbwd_ms = bench(comp_fwdbwd, N_WARMUP_COMPILED, N_RUNS)
    print(f"Compiled: fwd={comp_fwd_ms:.1f}ms  fwd+bwd={comp_fwdbwd_ms:.1f}ms", flush=True)

    # ── Results ────────────────────────────────────────────────────────────
    bwd_eager = eager_fwdbwd_ms - eager_fwd_ms
    bwd_comp  = comp_fwdbwd_ms  - comp_fwd_ms

    print(f"\n{'':22s} {'eager':>9s} {'compiled':>9s} {'speedup':>9s}")
    print(f"{'─'*52}")
    print(f"{'fwd only':22s} {eager_fwd_ms:>8.1f}ms {comp_fwd_ms:>8.1f}ms {eager_fwd_ms/comp_fwd_ms:>8.2f}×")
    print(f"{'fwd+bwd':22s} {eager_fwdbwd_ms:>8.1f}ms {comp_fwdbwd_ms:>8.1f}ms {eager_fwdbwd_ms/comp_fwdbwd_ms:>8.2f}×")
    if bwd_comp > 0.5:
        print(f"{'bwd only (derived)':22s} {bwd_eager:>8.1f}ms {bwd_comp:>8.1f}ms {bwd_eager/bwd_comp:>8.2f}×")
    print(f"\nbatch={B}, RTX 4090, torch={torch.__version__}")


if __name__ == "__main__":
    main()

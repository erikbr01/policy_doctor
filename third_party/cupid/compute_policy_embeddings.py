"""Extract per-timestep policy embeddings from rollout episodes.

Saves a (N_total_timesteps, D) array to
    <eval_dir>/policy_embeddings/<layer>.npz   key: "rollout_embeddings"

This file is read by PolicyEmbeddingRepresentation in policy_doctor.

Supported layers
----------------
bottleneck
    Output of policy.model.mid_modules[-1], global-avg-pooled over the
    spatial dimension, averaged over N_noise_levels uniformly-spaced
    diffusion timesteps.  Shape: (N_timesteps, 512).

Usage (cupid_torch2 env, GPU):
    python compute_policy_embeddings.py \\
        --train_dir /path/to/train/run \\
        --eval_dir  /path/to/eval/latest \\
        --layer bottleneck \\
        --n_noise_levels 64 \\
        --device cuda:0
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import torch
import tqdm
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_index_checkpoint,
    get_policy_from_checkpoint,
)
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

_SUPPORTED_POLICIES = (DiffusionUnetLowdimPolicy,)
_SUPPORTED_LAYERS = ("bottleneck",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_episode_meta(eval_dir: pathlib.Path):
    meta_path = eval_dir / "episodes" / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return meta["episode_lengths"], meta.get("episode_successes", [None] * len(meta["episode_lengths"]))


def _list_episode_pkls(eval_dir: pathlib.Path):
    pkls = sorted((eval_dir / "episodes").glob("ep*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No ep*.pkl under {eval_dir / 'episodes'}")
    return pkls


def _build_noise_schedule(n_infer: int, device: torch.device):
    """DDPM linear beta schedule — same as diffusion_policy default."""
    betas = torch.linspace(1e-4, 0.02, n_infer, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return alphas_cumprod.sqrt(), (1.0 - alphas_cumprod).sqrt()


def _extract_bottleneck_batch(
    policy: DiffusionUnetLowdimPolicy,
    obs_batch: np.ndarray,       # (B, n_obs_steps, obs_dim) float32
    n_noise_levels: int,
    noise_step_indices: torch.Tensor,  # (N,) long
    sqrt_alpha: torch.Tensor,          # (n_infer,)
    sqrt_one_minus_alpha: torch.Tensor,
    device: torch.device,
    batch_seed: int,
) -> np.ndarray:
    """Return (B, D) bottleneck embeddings for a batch of rollout timesteps.

    Stacks B*N noise-level variants into a single UNet forward pass, hooks
    the last mid-module, global-avg-pools the spatial dimension, and averages
    over the N noise levels per timestep.
    """
    model = policy.model
    B = len(obs_batch)
    N = n_noise_levels
    horizon = policy.horizon
    action_dim = policy.action_dim

    # obs conditioning: (B, n_obs * obs_dim) repeated N times → (B*N, cond_dim)
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    global_cond_single = obs_t.reshape(B, -1)          # (B, cond_dim)
    global_cond = global_cond_single.repeat_interleave(N, dim=0)  # (B*N, cond_dim)

    # fixed per-timestep noise repeated across noise levels
    rng = torch.Generator(device=device)
    rng.manual_seed(batch_seed)
    base_noise = torch.randn(B, horizon, action_dim, generator=rng, device=device)
    base_noise_rep = base_noise.repeat_interleave(N, dim=0)  # (B*N, horizon, action_dim)

    # build noisy actions for each (timestep, noise_level) pair
    t_indices = noise_step_indices.repeat(B)             # (B*N,)
    a = sqrt_alpha[t_indices].view(B * N, 1, 1)
    s = sqrt_one_minus_alpha[t_indices].view(B * N, 1, 1)
    noisy_actions = s * base_noise_rep  # pure noise (zero-mean action prior)

    captured = []

    def _hook(module, inp, out):
        # out: (B*N, C, T_spatial) → pool spatial → (B*N, C)
        captured.append(out.detach().float().mean(dim=-1).cpu())

    handle = model.mid_modules[-1].register_forward_hook(_hook)
    with torch.no_grad():
        model(noisy_actions, t_indices, global_cond=global_cond)
    handle.remove()

    # captured[0]: (B*N, C) → reshape → (B, N, C) → mean over N → (B, C)
    emb_bn = captured[0].reshape(B, N, -1).mean(dim=1)
    return emb_bn.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_dir", required=True, help="Training run directory (contains checkpoints/).")
    ap.add_argument("--eval_dir", required=True, help="Eval episodes directory (contains episodes/).")
    ap.add_argument("--train_ckpt", default="best", help="'best', an epoch index, or a .ckpt filename stem.")
    ap.add_argument("--layer", default="bottleneck", choices=_SUPPORTED_LAYERS)
    ap.add_argument("--n_noise_levels", type=int, default=None,
                    help="Number of uniformly-spaced diffusion timesteps to average over. "
                         "Defaults to policy.num_inference_steps (read from the checkpoint), "
                         "so the full denoising range is always covered regardless of policy config.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Rollout timesteps per GPU forward pass. B*N_noise_levels images "
                         "are processed together — increase for better GPU utilisation.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory. Defaults to <eval_dir>/policy_embeddings/.")
    args = ap.parse_args()

    train_dir = pathlib.Path(args.train_dir)
    eval_dir = pathlib.Path(args.eval_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else eval_dir / "policy_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.layer}.npz"
    device = torch.device(args.device)

    # Load checkpoint
    checkpoint_dir = train_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if args.train_ckpt == "best":
        checkpoint = get_best_checkpoint(checkpoints)
    elif args.train_ckpt.isdigit():
        checkpoint = get_index_checkpoint(checkpoints, int(args.train_ckpt))
    else:
        checkpoint = checkpoint_dir / f"{args.train_ckpt}.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading policy from {checkpoint} …")
    policy, _ = get_policy_from_checkpoint(checkpoint, device=device)
    if not isinstance(policy, _SUPPORTED_POLICIES):
        raise TypeError(f"Unsupported policy type: {type(policy)}")
    policy.eval()

    # Resolve n_noise_levels from policy config if not set explicitly.
    n_noise_levels = args.n_noise_levels
    if n_noise_levels is None:
        n_noise_levels = policy.num_inference_steps
        print(f"n_noise_levels not set — using policy.num_inference_steps = {n_noise_levels}")

    # Episode metadata
    ep_lens, _ = _load_episode_meta(eval_dir)
    pkls = _list_episode_pkls(eval_dir)
    assert len(pkls) == len(ep_lens), f"Episode count mismatch: {len(pkls)} pkl vs {len(ep_lens)} in metadata"
    N_total = sum(ep_lens)
    print(f"Episodes: {len(pkls)}  |  Total timesteps: {N_total}")
    print(f"Layer: {args.layer}  |  Noise levels: {n_noise_levels}  |  "
          f"Batch size: {args.batch_size}  |  Device: {device}")

    # Pre-load all obs into a flat array for easy batching.
    print("Loading episodes …")
    all_obs = []  # list of (n_obs_steps, obs_dim) arrays
    for ep_i, (pkl_path, ep_len) in enumerate(zip(pkls, ep_lens)):
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        assert len(df) == ep_len, f"Episode {ep_i} length mismatch"
        for t in range(ep_len):
            obs = np.asarray(df.iloc[t]["obs"], dtype=np.float32)
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]
            all_obs.append(obs)
    all_obs_arr = np.stack(all_obs, axis=0)  # (N_total, n_obs_steps, obs_dim)
    print(f"  obs array: {all_obs_arr.shape}")

    # Build noise schedule and noise-level indices once.
    sqrt_alpha, sqrt_one_minus_alpha = _build_noise_schedule(
        policy.num_inference_steps, device
    )
    noise_step_indices = torch.linspace(
        0, policy.num_inference_steps - 1, n_noise_levels, dtype=torch.long, device=device
    )

    # Process in batches.
    all_embs = np.empty((N_total, 0), dtype=np.float32)  # placeholder; filled below
    results = []
    B = args.batch_size
    for batch_start in tqdm.tqdm(range(0, N_total, B), desc="Batches"):
        batch_obs = all_obs_arr[batch_start: batch_start + B]
        if args.layer == "bottleneck":
            emb = _extract_bottleneck_batch(
                policy, batch_obs, n_noise_levels,
                noise_step_indices, sqrt_alpha, sqrt_one_minus_alpha,
                device, batch_seed=batch_start,
            )
        else:
            raise NotImplementedError(args.layer)
        results.append(emb)

    rollout_embeddings = np.concatenate(results, axis=0)  # (N_total, D)
    print(f"Saving {rollout_embeddings.shape} to {out_path}")
    np.savez_compressed(out_path, rollout_embeddings=rollout_embeddings)
    print("Done.")


if __name__ == "__main__":
    main()

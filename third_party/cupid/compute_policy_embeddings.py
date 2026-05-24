#!/usr/bin/env python3
"""Compute and save policy embeddings for saved rollout episodes.

Reads episode pkl files from <eval_dir>/episodes/, computes per-timestep
embeddings, and writes <eval_dir>/policy_embeddings/<layer>.npz.

Layers
------
obs_encoder
    Normalized observation history flattened: shape (D,) where
    D = obs_dim * n_obs_steps. No UNet call needed — very fast.
plan_bottleneck
    UNet mid-block (last ConditionalResidualBlock1D) output averaged over
    the time axis at denoising step t=0: shape (D,) where D = down_dims[-1].

Usage
-----
python compute_policy_embeddings.py \\
    --train_dir data/outputs/train/<date>/<run_name> \\
    --train_ckpt latest \\
    --eval_dir data/outputs/eval_save_episodes/<date>/<run_name>/latest \\
    --layer plan_bottleneck \\
    --device cuda:0
"""

from __future__ import annotations

import pathlib
import pickle
import sys
from typing import List

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import click
import numpy as np
import pandas as pd
import torch

from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_index_checkpoint,
    get_policy_from_checkpoint,
)


def _resolve_checkpoint(checkpoint_dir: pathlib.Path, train_ckpt: str) -> pathlib.Path:
    checkpoints = list(checkpoint_dir.iterdir())
    if train_ckpt == "latest":
        # Highest epoch checkpoint (not the "latest.ckpt" symlink)
        epoch_ckpts = sorted(
            [c for c in checkpoints if c.suffix == ".ckpt" and "latest" not in c.name],
            key=lambda p: int(p.stem.split("=")[1].split("-")[0]),
        )
        return epoch_ckpts[-1]
    if train_ckpt == "best":
        return get_best_checkpoint(checkpoints)
    if train_ckpt.isdigit():
        return get_index_checkpoint(checkpoints, int(train_ckpt))
    return checkpoint_dir / f"{train_ckpt}.ckpt"


def _load_episode_obs(pkl_path: pathlib.Path) -> np.ndarray:
    """Return obs array of shape (T, n_obs_steps, obs_dim) from episode pkl."""
    with open(pkl_path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)
    obs_list = df["obs"].tolist()
    return np.stack(obs_list, axis=0).astype(np.float32)  # (T, n_obs_steps, obs_dim)


@torch.no_grad()
def _compute_obs_encoder(
    policy,
    obs_windows: np.ndarray,
    device: torch.device,
    chunk_size: int = 256,
) -> np.ndarray:
    """obs_windows: (T, n_obs_steps, obs_dim) → (T, global_cond_dim)."""
    T = obs_windows.shape[0]
    action_dim = policy.action_dim
    results = []
    for start in range(0, T, chunk_size):
        obs_chunk = torch.from_numpy(obs_windows[start : start + chunk_size]).to(device)
        B = obs_chunk.shape[0]
        batch = {
            "obs": obs_chunk,
            "action": torch.zeros(B, obs_chunk.shape[1], action_dim, device=device),
        }
        emb = policy.compute_obs_embedding(batch)  # (B, global_cond_dim)
        results.append(emb.cpu().numpy())
    return np.vstack(results)


@torch.no_grad()
def _compute_plan_bottleneck(
    policy,
    obs_windows: np.ndarray,
    device: torch.device,
    chunk_size: int = 128,
) -> np.ndarray:
    """obs_windows: (T, n_obs_steps, obs_dim) → (T, mid_dim).

    Hooks the last mid_module of the ConditionalUnet1D and runs a forward
    pass at denoising step t=0 with a zero-initialized noisy trajectory.
    """
    T = obs_windows.shape[0]
    action_dim = policy.action_dim
    horizon = policy.horizon
    n_obs_steps = policy.n_obs_steps

    captured: List[np.ndarray] = []

    def _hook(_module, _input, output):
        # output: (B, mid_dim, horizon) → pool over time → (B, mid_dim)
        captured.append(output.mean(dim=-1).cpu().numpy())

    handle = policy.model.mid_modules[-1].register_forward_hook(_hook)

    results = []
    try:
        for start in range(0, T, chunk_size):
            obs_chunk = torch.from_numpy(obs_windows[start : start + chunk_size]).to(device)
            B = obs_chunk.shape[0]

            batch = {
                "obs": obs_chunk,
                "action": torch.zeros(B, obs_chunk.shape[1], action_dim, device=device),
            }
            nbatch = policy.normalizer.normalize(batch)
            nobs = nbatch["obs"]
            global_cond = nobs[:, :n_obs_steps, :].reshape(B, -1)

            captured.clear()
            noisy = torch.zeros(B, horizon, action_dim, device=device)
            timestep = torch.zeros(B, dtype=torch.long, device=device)
            policy.model(noisy, timestep, global_cond=global_cond)
            results.append(captured[0])
    finally:
        handle.remove()

    return np.vstack(results)


@click.command()
@click.option("--train_dir", required=True, help="Training output directory")
@click.option("--train_ckpt", default="latest", help="Checkpoint: 'latest', 'best', epoch index, or filename")
@click.option("--eval_dir", required=True, help="Eval output directory (contains episodes/)")
@click.option("--layer", default="plan_bottleneck", help="obs_encoder or plan_bottleneck")
@click.option("--device", default="cuda:0")
@click.option("--chunk_size", default=256, help="Batch size for embedding computation")
@click.option("--overwrite", is_flag=True)
def main(train_dir, train_ckpt, eval_dir, layer, device, chunk_size, overwrite):
    train_dir = pathlib.Path(train_dir)
    eval_dir = pathlib.Path(eval_dir)
    out_dir = eval_dir / "policy_embeddings"
    out_path = out_dir / f"{layer}.npz"

    if out_path.exists() and not overwrite:
        print(f"[compute_policy_embeddings] Already exists, skipping: {out_path}")
        print("  Pass --overwrite to recompute.")
        return

    device = torch.device(device)

    # Load policy
    ckpt = _resolve_checkpoint(train_dir / "checkpoints", train_ckpt)
    print(f"[compute_policy_embeddings] Loading policy from {ckpt.name}")
    policy, _cfg = get_policy_from_checkpoint(ckpt, device=device)
    policy.eval()

    # Gather episode pkl files
    ep_dir = eval_dir / "episodes"
    pkl_files = sorted(ep_dir.glob("ep*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No episode pkl files found in {ep_dir}. "
            "Run eval_baseline with save_episodes=True first."
        )
    print(f"[compute_policy_embeddings] {len(pkl_files)} episodes | layer={layer} | device={device}")

    # Compute embeddings per episode
    all_embeddings = []
    for i, pkl_path in enumerate(pkl_files):
        obs_windows = _load_episode_obs(pkl_path)  # (T, n_obs_steps, obs_dim)
        if layer == "obs_encoder":
            emb = _compute_obs_encoder(policy, obs_windows, device, chunk_size=chunk_size)
        elif layer == "plan_bottleneck":
            emb = _compute_plan_bottleneck(policy, obs_windows, device, chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown layer: {layer!r}. Choose 'obs_encoder' or 'plan_bottleneck'.")
        all_embeddings.append(emb)
        if (i + 1) % 50 == 0 or (i + 1) == len(pkl_files):
            print(f"  [{i + 1}/{len(pkl_files)}] episode shape={emb.shape}")

    rollout_embeddings = np.vstack(all_embeddings)
    print(f"[compute_policy_embeddings] Total embeddings: {rollout_embeddings.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, rollout_embeddings=rollout_embeddings)
    print(f"[compute_policy_embeddings] Saved: {out_path}")


if __name__ == "__main__":
    main()

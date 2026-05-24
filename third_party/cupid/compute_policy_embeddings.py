"""Extract per-timestep policy embeddings from rollout episodes.

Saves a (N_total_timesteps, D) array to
    <eval_dir>/policy_embeddings/<layer>.npz   key: "rollout_embeddings"

Layer name convention:  {hook}[_{action}][_t{T}]

  hook        Module to hook
  --------    ------------------------------------------------------------
  bottleneck  policy.model.mid_modules[-1]  (512D after global avg pool)
  decoder     policy.model.up_modules[-1][1]  (up-block final resnet)
  encoder     policy.model.down_modules[-1][1]  (down-block final resnet)

  action      Action input to the UNet
  --------    ------------------------------------------------------------
  (omit)      Random noise — action washes out when averaged over t
  plan        Full rollout action plan  (horizon × action_dim)
  exec        action[0] tiled across the full horizon
  plan8       First 8 executed steps, zero-padded to full horizon

  t           Diffusion timestep
  --------    ------------------------------------------------------------
  (omit)      Average over n_noise_levels uniformly-spaced timesteps
  _t{T}       Evaluate at the single timestep T  (0 = final denoising step)

Examples
--------
  bottleneck               random noise, avg over 100t, bottleneck hook
  bottleneck_plan_t0       actual plan,  t=0,            bottleneck hook  ← best known
  decoder_plan_t0          actual plan,  t=0,            decoder hook
  encoder_plan_t0          actual plan,  t=0,            encoder hook
  bottleneck_exec_t0       exec action,  t=0,            bottleneck hook
  bottleneck_plan8_t0      plan (8 steps + zero), t=0,   bottleneck hook
  bottleneck_plan_t5       actual plan,  t=5,            bottleneck hook

Usage (cupid_torch2 env, GPU):
    python compute_policy_embeddings.py \\
        --train_dir /path/to/train/run \\
        --eval_dir  /path/to/eval/latest \\
        --layer bottleneck_plan_t0 \\
        --batch_size 128 --device cuda:0
"""
from __future__ import annotations

import pathlib
import pickle
import re
import sys
from typing import Optional

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


# ---------------------------------------------------------------------------
# Layer name parser
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(
    r"^(?P<hook>bottleneck|decoder|encoder)"
    r"(?:_(?P<action>plan8|plan|exec))?"
    r"(?:_t(?P<t>\d+))?$"
)


def _parse_layer(layer: str) -> dict:
    """Parse a layer name into (hook, action, t_single).

    Returns dict with keys: hook, action (None/'plan'/'plan8'/'exec'), t_single (None or int).
    """
    m = _LAYER_RE.match(layer)
    if m is None:
        raise ValueError(
            f"Unknown layer {layer!r}. Format: {{bottleneck|decoder|encoder}}"
            f"[_{{plan|plan8|exec}}][_t{{0-99}}]"
        )
    return {
        "hook":     m.group("hook"),
        "action":   m.group("action"),   # None → random noise
        "t_single": None if m.group("t") is None else int(m.group("t")),
    }


def _get_hook_module(model, hook: str):
    if hook == "bottleneck":
        return model.mid_modules[-1]
    if hook == "decoder":
        # last ResNet block in the last up_module (resnet, resnet2, upsample)
        return model.up_modules[-1][1]
    if hook == "encoder":
        # last ResNet block in the last down_module (resnet, resnet2, downsample)
        return model.down_modules[-1][1]
    raise ValueError(f"Unknown hook: {hook}")


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
    betas = torch.linspace(1e-4, 0.02, n_infer, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return alphas_cumprod.sqrt(), (1.0 - alphas_cumprod).sqrt()


def _build_action_batch(
    policy,
    action_arr: Optional[np.ndarray],  # (B, horizon, action_dim) or None
    action_type: Optional[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return clean action tensor (B, horizon, action_dim) or None for random noise."""
    if action_type is None:
        return None
    assert action_arr is not None
    B, horizon, action_dim = action_arr.shape
    if action_type == "plan":
        clean = action_arr
    elif action_type == "exec":
        # tile action[0] across the full horizon
        clean = np.tile(action_arr[:, 0:1, :], (1, horizon, 1))
    elif action_type == "plan8":
        # first 8 steps + zero-pad
        clean = np.concatenate(
            [action_arr[:, :8, :], np.zeros((B, horizon - 8, action_dim), dtype=np.float32)],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown action_type: {action_type}")
    return torch.as_tensor(clean, dtype=torch.float32, device=device)


def _extract_batch(
    policy: DiffusionUnetLowdimPolicy,
    obs_batch: np.ndarray,          # (B, n_obs_steps, obs_dim)
    action_batch: Optional[np.ndarray],  # (B, horizon, action_dim) or None
    action_type: Optional[str],
    hook_name: str,
    n_noise_levels: int,
    noise_step_indices: torch.Tensor,   # (N,) long
    sqrt_alpha: torch.Tensor,
    sqrt_one_minus_alpha: torch.Tensor,
    device: torch.device,
    batch_seed: int,
) -> np.ndarray:
    """Return (B, D) embedding for a batch of rollout timesteps."""
    model = policy.model
    B = len(obs_batch)
    N = n_noise_levels
    horizon = policy.horizon
    action_dim = policy.action_dim

    # obs conditioning: (B, cond_dim) repeated N times → (B*N, cond_dim)
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    global_cond = obs_t.reshape(B, -1).repeat_interleave(N, dim=0)  # (B*N, cond_dim)

    # fixed random noise (used as base noise regardless of action type)
    rng = torch.Generator(device=device)
    rng.manual_seed(batch_seed)
    base_noise = torch.randn(B, horizon, action_dim, generator=rng, device=device)
    base_noise_rep = base_noise.repeat_interleave(N, dim=0)  # (B*N, horizon, action_dim)

    t_indices = noise_step_indices.repeat(B)    # (B*N,)
    a = sqrt_alpha[t_indices].view(B * N, 1, 1)
    s = sqrt_one_minus_alpha[t_indices].view(B * N, 1, 1)

    clean_actions = _build_action_batch(policy, action_batch, action_type, device)
    if clean_actions is not None:
        clean_rep = clean_actions.repeat_interleave(N, dim=0)
        noisy_actions = a * clean_rep + s * base_noise_rep
    else:
        noisy_actions = s * base_noise_rep

    captured = []
    hook_module = _get_hook_module(model, hook_name)

    def _hook(module, inp, out):
        # out may be a tuple (resnet blocks return tensor directly; but just in case)
        feat = out[0] if isinstance(out, tuple) else out
        captured.append(feat.detach().float().mean(dim=-1).cpu())  # global avg pool

    handle = hook_module.register_forward_hook(_hook)
    with torch.no_grad():
        model(noisy_actions, t_indices, global_cond=global_cond)
    handle.remove()

    # (B*N, D) → (B, N, D) → mean over N → (B, D)
    emb = captured[0].reshape(B, N, -1).mean(dim=1)
    return emb.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--train_ckpt", default="best")
    ap.add_argument("--layer", required=True,
                    help="Layer name encoding hook, action type, and timestep.")
    ap.add_argument("--n_noise_levels", type=int, default=None,
                    help="Noise levels to average over. Defaults to policy.num_inference_steps. "
                         "Ignored when layer contains _t{N} (single-step evaluation).")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    parsed = _parse_layer(args.layer)
    hook_name  = parsed["hook"]
    action_type = parsed["action"]
    t_single   = parsed["t_single"]
    needs_actions = action_type is not None

    train_dir = pathlib.Path(args.train_dir)
    eval_dir  = pathlib.Path(args.eval_dir)
    out_dir   = pathlib.Path(args.out_dir) if args.out_dir else eval_dir / "policy_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"{args.layer}.npz"
    device    = torch.device(args.device)

    if out_path.exists():
        print(f"Output already exists, skipping: {out_path}")
        return

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

    print(f"Loading policy from {checkpoint.name} …")
    policy, _ = get_policy_from_checkpoint(checkpoint, device=device)
    if not isinstance(policy, _SUPPORTED_POLICIES):
        raise TypeError(f"Unsupported policy: {type(policy)}")
    policy.eval()

    # Resolve noise levels and timestep indices
    n_infer = policy.num_inference_steps
    if t_single is not None:
        noise_step_indices = torch.tensor([t_single], dtype=torch.long, device=device)
        n_noise_levels = 1
    else:
        n_noise_levels = args.n_noise_levels or n_infer
        noise_step_indices = torch.linspace(0, n_infer - 1, n_noise_levels,
                                            dtype=torch.long, device=device)

    print(f"Layer: {args.layer}  |  hook={hook_name}  action={action_type}  "
          f"t={'single:'+str(t_single) if t_single is not None else f'avg:{n_noise_levels}'}  "
          f"batch={args.batch_size}")

    # Load episodes
    ep_lens, _ = _load_episode_meta(eval_dir)
    pkls = _list_episode_pkls(eval_dir)
    assert len(pkls) == len(ep_lens)
    N_total = sum(ep_lens)
    print(f"Episodes: {len(pkls)}  |  Timesteps: {N_total}")

    all_obs, all_actions = [], []
    for ep_i, (pkl_path, ep_len) in enumerate(zip(pkls, ep_lens)):
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        assert len(df) == ep_len
        for t in range(ep_len):
            row = df.iloc[t]
            obs = np.asarray(row["obs"], dtype=np.float32)
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]
            all_obs.append(obs)
            if needs_actions:
                all_actions.append(np.asarray(row["action"], dtype=np.float32))

    all_obs_arr = np.stack(all_obs, axis=0)
    all_actions_arr = np.stack(all_actions, axis=0) if needs_actions else None

    sqrt_alpha, sqrt_one_minus_alpha = _build_noise_schedule(n_infer, device)

    results = []
    B = args.batch_size
    for batch_start in tqdm.tqdm(range(0, N_total, B), desc="Batches"):
        emb = _extract_batch(
            policy,
            all_obs_arr[batch_start: batch_start + B],
            all_actions_arr[batch_start: batch_start + B] if needs_actions else None,
            action_type,
            hook_name,
            n_noise_levels,
            noise_step_indices,
            sqrt_alpha,
            sqrt_one_minus_alpha,
            device,
            batch_seed=batch_start,
        )
        results.append(emb)

    rollout_embeddings = np.concatenate(results, axis=0)
    print(f"Saving {rollout_embeddings.shape} → {out_path}")
    np.savez_compressed(out_path, rollout_embeddings=rollout_embeddings)
    print("Done.")


if __name__ == "__main__":
    main()

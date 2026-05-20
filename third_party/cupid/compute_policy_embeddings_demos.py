"""Extract per-timestep policy embeddings for training demonstrations.

Mirror of ``compute_policy_embeddings.py`` but operates on the policy's
training dataset (loaded via ``hydra.utils.instantiate(cfg.task.dataset)``)
instead of rollout episode pickles.  The output format mirrors the rollout
file so that downstream code can reuse the same windowing helper.

Saves to
    <train_dir>/policy_embeddings_demos/<layer>.npz
        "demo_embeddings": (N_total_demo_timesteps, D) float32
        "episode_lengths": (N_demos,) int64 — samples per demo (matches the
                           policy dataset's sliding-window count, not the
                           raw HDF5 timestep count).

Usage (mimicgen_torch2 env, GPU):
    python compute_policy_embeddings_demos.py \\
        --train_dir /path/to/train/run \\
        --layer bottleneck_plan_t0 \\
        --batch_size 128 --device cuda:0
"""
from __future__ import annotations

import pathlib
import sys
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_index_checkpoint,
    get_policy_from_checkpoint,
)
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from torch.utils.data import DataLoader

# Reuse the layer parser + hook resolver + batch extractor from the rollout
# script so demo and rollout embeddings stay bit-equivalent for the same layer.
from compute_policy_embeddings import (  # noqa: E402
    _build_noise_schedule,
    _extract_batch,
    _parse_layer,
)

_SUPPORTED_POLICIES = (DiffusionUnetLowdimPolicy,)


def _resolve_checkpoint(train_dir: pathlib.Path, train_ckpt: str) -> pathlib.Path:
    checkpoint_dir = train_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if train_ckpt == "best":
        return get_best_checkpoint(checkpoints)
    if train_ckpt.isdigit():
        return get_index_checkpoint(checkpoints, int(train_ckpt))
    return checkpoint_dir / f"{train_ckpt}.ckpt"


def _episode_lengths_from_dataset(dataset) -> np.ndarray:
    """Return per-training-demo sample counts in DataLoader-iteration order.

    Critical: the replay buffer holds *every* demo in the HDF5 (training +
    val + holdout), but the diffusion_policy training mask only emits
    sampler indices for the training subset (e.g. 192 of 300 demos when
    ``dataset_mask_kwargs.train_ratio=0.64`` + ``uniform_quality=True``).
    If we attribute the (N_samples, D) embedding stack uniformly across
    *all* demos in the replay buffer, every per-demo split is wrong by the
    holdout ratio and demo boundaries leak across.

    Correct attribution: each ``sampler.indices`` row is
    ``(buffer_start, buffer_end, sample_start, sample_end)`` where
    ``buffer_start`` is the position in the concatenated replay buffer.
    Digitising those into the ``episode_ends`` bins gives the demo idx per
    sample; ``np.bincount`` over demos then yields the actual training
    sample count per demo.  Output preserves DataLoader order (samples
    are emitted demo-by-demo, ascending), and demos with zero training
    contribution are dropped — so the returned array has one entry per
    *contributing* demo.
    """
    rb = getattr(dataset, "replay_buffer", None)
    sampler = getattr(dataset, "sampler", None)
    if rb is None or sampler is None or not hasattr(sampler, "indices"):
        raise RuntimeError(
            "Cannot determine episode lengths from dataset: missing "
            "replay_buffer.episode_ends or sampler.indices."
        )
    ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
    indices = np.asarray(sampler.indices)
    if indices.ndim != 2 or indices.shape[1] < 1:
        raise RuntimeError(
            f"sampler.indices has unexpected shape {indices.shape}; "
            "expected (N, ≥1) with buffer_start in column 0."
        )
    buf_start = indices[:, 0].astype(np.int64)
    # side='right' so a buf_start equal to an end goes to the *next* demo —
    # episode_ends marks the past-the-end position of each demo.
    demo_idx = np.searchsorted(ends, buf_start, side="right")
    counts = np.bincount(demo_idx, minlength=len(ends))
    # Drop demos that didn't contribute (i.e. masked-out by the train mask).
    return counts[counts > 0].astype(np.int64)


def _patch_dataset_path(cfg, dataset_path_override: Optional[str]) -> None:
    """Best-effort dataset-path repair.

    The checkpoint config stores absolute training paths that may be stale on
    a different machine / after rename.  policy_doctor ships a small adapter
    that does the same fixup compute_infembed_embeddings.py uses.
    """
    try:
        from policy_doctor.data.adapters import patch_attribution_dataset_path
        patch_attribution_dataset_path(
            cfg,
            repo_root=pathlib.Path(__file__).resolve().parent,
            dataset_path_override=dataset_path_override,
        )
    except ImportError:
        # Running standalone; respect the checkpoint's stored path.
        pass


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--train_ckpt", default="best")
    ap.add_argument("--layer", required=True,
                    help="Same layer name convention as compute_policy_embeddings.py.")
    ap.add_argument("--n_noise_levels", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_path", default=None,
                    help="Override training dataset path (passed to patch_attribution_dataset_path).")
    ap.add_argument("--include_holdout", action="store_true",
                    help="Also embed the holdout (validation) demos and concatenate.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    parsed = _parse_layer(args.layer)
    hook_name = parsed["hook"]
    action_type = parsed["action"]
    t_single = parsed["t_single"]
    needs_actions = action_type is not None

    train_dir = pathlib.Path(args.train_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else train_dir / "policy_embeddings_demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.layer}.npz"
    device = torch.device(args.device)

    if out_path.exists() and not args.overwrite:
        print(f"Output already exists, skipping: {out_path}  (use --overwrite to replace)")
        return

    checkpoint = _resolve_checkpoint(train_dir, args.train_ckpt)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading policy from {checkpoint.name} …")
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)
    if not isinstance(policy, _SUPPORTED_POLICIES):
        raise TypeError(f"Unsupported policy: {type(policy)}")
    policy.eval()
    n_obs_steps = int(getattr(policy, "n_obs_steps", cfg.policy.n_obs_steps))

    _patch_dataset_path(cfg, args.dataset_path)

    # Diffusion timestep schedule (identical to rollout path).
    n_infer = policy.num_inference_steps
    if t_single is not None:
        noise_step_indices = torch.tensor([t_single], dtype=torch.long, device=device)
        n_noise_levels = 1
    else:
        n_noise_levels = args.n_noise_levels or n_infer
        noise_step_indices = torch.linspace(
            0, n_infer - 1, n_noise_levels, dtype=torch.long, device=device
        )

    print(
        f"Layer: {args.layer}  |  hook={hook_name}  action={action_type}  "
        f"t={'single:'+str(t_single) if t_single is not None else f'avg:{n_noise_levels}'}  "
        f"batch={args.batch_size}  n_obs_steps={n_obs_steps}"
    )

    # Load training dataset (and optional holdout) via Hydra — identical to
    # compute_infembed_embeddings.py.
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    holdout_set = None
    if args.include_holdout:
        holdout_set = train_set.get_holdout_dataset()
        if holdout_set is None or len(holdout_set) == 0:
            holdout_set = None

    parts = [("train", train_set)]
    if holdout_set is not None:
        parts.append(("holdout", holdout_set))

    sqrt_alpha, sqrt_one_minus_alpha = _build_noise_schedule(n_infer, device)

    all_embs = []
    all_ep_lens_parts = []
    for split_name, ds in parts:
        ep_lens = _episode_lengths_from_dataset(ds)
        N = sum(int(x) for x in ep_lens)
        print(f"Split {split_name}: {len(ds)} samples across {len(ep_lens)} demos (sum ep_lens = {N})")
        if N != len(ds):
            # Should match exactly after the sampler-indices fix; if it
            # doesn't, the dataset/sampler shape has changed and we should
            # know rather than silently distribute.
            raise RuntimeError(
                f"sum(ep_lens)={N} != len(ds)={len(ds)} for split {split_name!r}. "
                "Refusing to persist a mismatched embedding/length pair; "
                "investigate the dataset's sampler.indices structure."
            )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        split_embs = []
        for batch in tqdm.tqdm(loader, desc=f"{split_name} batches"):
            obs = batch["obs"]  # (B, T, obs_dim) or (B, n_obs_steps, obs_dim)
            if obs.ndim == 3 and obs.shape[1] >= n_obs_steps:
                obs_input = obs[:, :n_obs_steps, :]
            else:
                obs_input = obs
            obs_np = obs_input.detach().cpu().numpy().astype(np.float32)

            if needs_actions:
                action_np = batch["action"].detach().cpu().numpy().astype(np.float32)
            else:
                action_np = None

            emb = _extract_batch(
                policy,
                obs_np,
                action_np,
                action_type,
                hook_name,
                n_noise_levels,
                noise_step_indices,
                sqrt_alpha,
                sqrt_one_minus_alpha,
                device,
                batch_seed=len(all_embs) * 1009 + len(split_embs),
            )
            split_embs.append(emb)

        split_arr = np.concatenate(split_embs, axis=0) if split_embs else np.zeros((0, 0), dtype=np.float32)
        if int(ep_lens.sum()) != split_arr.shape[0]:
            raise RuntimeError(
                f"emb count {split_arr.shape[0]} != sum(ep_lens) {int(ep_lens.sum())} "
                f"for split {split_name!r}; refusing to persist."
            )
        all_embs.append(split_arr)
        all_ep_lens_parts.append(ep_lens)

    demo_embeddings = np.concatenate(all_embs, axis=0)
    episode_lengths = np.concatenate(all_ep_lens_parts, axis=0)

    print(f"Saving {demo_embeddings.shape} demo embeddings ({len(episode_lengths)} demos) → {out_path}")
    np.savez_compressed(
        out_path,
        demo_embeddings=demo_embeddings,
        episode_lengths=episode_lengths.astype(np.int64),
    )
    print("Done.")


if __name__ == "__main__":
    main()

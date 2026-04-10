"""
Load the train dataset (same as IV/training) and return replay_buffer.episode_ends.

Used to compute dataset_fingerprint and total_raw_samples for curation configs
without depending on IV's in-memory buffer, so the config is valid across the
full lifecycle (creation in policy_doctor, verification in training, etc.).
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional, Tuple

import numpy as np


def _get_repo_root(train_dir: pathlib.Path) -> pathlib.Path:
    """Find repo root (directory named 'cupid' or similar) by walking up from train_dir."""
    root = pathlib.Path(train_dir).resolve()
    while root.parent != root:
        if root.name == "cupid":
            return root
        root = root.parent
    return pathlib.Path(train_dir).resolve().parent  # fallback


def get_checkpoint_path(train_dir: pathlib.Path, train_ckpt: str = "latest") -> pathlib.Path:
    """Path to the checkpoint file. Matches IV/training convention."""
    train_dir = pathlib.Path(train_dir).resolve()
    checkpoint_dir = train_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    if train_ckpt == "latest":
        return checkpoint_dir / "latest.ckpt"
    if train_ckpt == "best":
        try:
            from diffusion_policy.common.trak_util import get_best_checkpoint

            checkpoints = list(checkpoint_dir.iterdir())
            return get_best_checkpoint(checkpoints)
        except ImportError:
            return checkpoint_dir / "latest.ckpt"
        except Exception:
            return checkpoint_dir / "latest.ckpt"
    if train_ckpt.isdigit():
        try:
            from diffusion_policy.common.trak_util import get_index_checkpoint

            checkpoints = list(checkpoint_dir.iterdir())
            return get_index_checkpoint(checkpoints, int(train_ckpt))
        except ImportError:
            return checkpoint_dir / "latest.ckpt"
        except Exception:
            return checkpoint_dir / "latest.ckpt"
    return checkpoint_dir / f"{train_ckpt}.ckpt"


def load_checkpoint_config(checkpoint_path: pathlib.Path) -> Tuple[Any, dict]:
    """Load config and payload from checkpoint. Returns (cfg, payload)."""
    import torch
    try:
        import dill
        pickle_module = dill
    except ImportError:
        import pickle
        pickle_module = pickle
    payload = torch.load(open(str(checkpoint_path), "rb"), pickle_module=pickle_module)
    cfg = payload["cfg"]
    return cfg, payload


def _fix_dataset_path(
    cfg,
    train_dir: pathlib.Path,
    repo_root: pathlib.Path,
    dataset_path_override: Optional[str] = None,
) -> None:
    """Resolve and patch ``cfg.task.dataset.dataset_path`` for all data sources.

    Uses the unified adapter so MimicGen and RoboCasa checkpoints resolve
    correctly in addition to standard Robomimic layouts.  The path is stored
    relative to ``repo_root`` so ``get_dataset_masks`` (which inspects
    ``Path.parts``) correctly identifies the source name.

    Caller must ``os.chdir(repo_root)`` before ``hydra.utils.instantiate``.
    """
    try:
        from policy_doctor.data.adapters import patch_attribution_dataset_path
    except ImportError:
        # Graceful fallback: adapter module not available; leave cfg unchanged.
        return
    patch_attribution_dataset_path(cfg, repo_root=repo_root, dataset_path_override=dataset_path_override)


# Keep old name as a thin alias so any callers that imported it directly still work.
def _fix_robomimic_dataset_path(cfg, train_dir: pathlib.Path, repo_root: pathlib.Path) -> None:
    _fix_dataset_path(cfg, train_dir, repo_root)


def load_dataset_episode_ends(
    train_dir: pathlib.Path,
    train_ckpt: str = "latest",
    repo_root: Optional[pathlib.Path] = None,
    dataset_path_override: Optional[str] = None,
) -> np.ndarray:
    """Load the train dataset from checkpoint config and return replay_buffer.episode_ends.

    Uses the same code path as IV and training: checkpoint -> cfg -> fix paths ->
    hydra.utils.instantiate(cfg.task.dataset) -> replay_buffer.episode_ends.

    Args:
        train_dir: Training output directory (contains checkpoints/).
        train_ckpt: Checkpoint name ("latest", "best", or epoch number).
        repo_root: Repo root for resolving relative dataset paths. Default: inferred from train_dir.
        dataset_path_override: Optional explicit HDF5 path; overrides whatever is in the checkpoint.

    Returns:
        episode_ends as int64 array (cumulative timestep counts per episode).
    """
    train_dir = pathlib.Path(train_dir).resolve()
    if repo_root is None:
        repo_root = _get_repo_root(train_dir)
    else:
        repo_root = pathlib.Path(repo_root).resolve()

    checkpoint_path = get_checkpoint_path(train_dir, train_ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    cfg, _ = load_checkpoint_config(checkpoint_path)
    _fix_dataset_path(cfg, train_dir, repo_root, dataset_path_override=dataset_path_override)

    import os
    import hydra
    cwd = os.getcwd()
    try:
        os.chdir(str(repo_root))
        dataset = hydra.utils.instantiate(cfg.task.dataset)
    finally:
        os.chdir(cwd)
    if not hasattr(dataset, "replay_buffer") or dataset.replay_buffer is None:
        raise ValueError("Dataset has no replay_buffer")
    rb = dataset.replay_buffer
    if not hasattr(rb, "episode_ends"):
        raise ValueError("Replay buffer has no episode_ends")
    episode_ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
    return episode_ends

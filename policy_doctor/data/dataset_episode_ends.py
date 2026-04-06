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
        except Exception:
            return checkpoint_dir / "latest.ckpt"
    if train_ckpt.isdigit():
        try:
            from diffusion_policy.common.trak_util import get_index_checkpoint
            checkpoints = list(checkpoint_dir.iterdir())
            return get_index_checkpoint(checkpoints, int(train_ckpt))
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


def _fix_robomimic_dataset_path(cfg, train_dir: pathlib.Path, repo_root: pathlib.Path) -> None:
    """Set dataset_path in cfg to a path relative to repo_root. Must use relative path:
    get_dataset_masks() uses path.parts[1] as dataset name (e.g. 'robomimic'); absolute
    paths give wrong part (e.g. 'Users'). Caller must chdir to repo_root before instantiate."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return
    try:
        if hasattr(OmegaConf, "set_struct"):
            OmegaConf.set_struct(cfg, False)
    except Exception:
        pass
    dataset_path_value = ""
    if hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
        dataset_path_value = cfg.task.dataset.get("dataset_path", "") or ""
    if not dataset_path_value and hasattr(cfg, "task"):
        dataset_path_value = cfg.task.get("dataset_path", "") or ""
    needs_fix = not dataset_path_value or not str(dataset_path_value).strip()
    if not needs_fix:
        p = pathlib.Path(dataset_path_value)
        if p.is_absolute():
            # Convert to relative so sampler sees parts[1] = 'robomimic' not 'Users'
            try:
                rel = p.relative_to(repo_root)
                if (repo_root / rel).exists():
                    dataset_path_value = str(rel)
                    needs_fix = False
            except ValueError:
                pass
        if not needs_fix and (repo_root / dataset_path_value).exists():
            if hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
                cfg.task.dataset.dataset_path = dataset_path_value
            if hasattr(cfg, "task"):
                cfg.task.dataset_path = dataset_path_value
            return
        if not p.is_absolute() and (repo_root / p).exists():
            return
        needs_fix = True
    if not needs_fix:
        return
    task_name = (getattr(cfg, "task_name", "") or "").lower()
    robomimic_tasks = ["transport", "lift", "can", "square", "tool_hang"]
    matched_task = next((t for t in robomimic_tasks if t in task_name), None)
    if not matched_task:
        return
    data_type = "ph" if "ph" in task_name else "mh"
    filename = "image_abs.hdf5" if "image" in task_name else "low_dim_abs.hdf5"
    relative_path = f"data/robomimic/datasets/{matched_task}/{data_type}/{filename}"
    if (repo_root / relative_path).exists():
        if hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
            cfg.task.dataset.dataset_path = relative_path
        if hasattr(cfg, "task"):
            cfg.task.dataset_path = relative_path


def load_dataset_episode_ends(
    train_dir: pathlib.Path,
    train_ckpt: str = "latest",
    repo_root: Optional[pathlib.Path] = None,
) -> np.ndarray:
    """Load the train dataset from checkpoint config and return replay_buffer.episode_ends.

    Uses the same code path as IV and training: checkpoint -> cfg -> fix paths ->
    hydra.utils.instantiate(cfg.task.dataset) -> replay_buffer.episode_ends.

    Args:
        train_dir: Training output directory (contains checkpoints/).
        train_ckpt: Checkpoint name ("latest", "best", or epoch number).
        repo_root: Repo root for resolving relative dataset paths. Default: inferred from train_dir.

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
    _fix_robomimic_dataset_path(cfg, train_dir, repo_root)

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

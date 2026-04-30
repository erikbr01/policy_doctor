"""Build per-slice embedding matrices for clustering from InfEmbed or TRAK.

Shared by the pipeline ``run_clustering`` step and the Streamlit clustering tab.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from policy_doctor.data.influence_loader import InfluenceDataContainer


def aggregate_window(window: np.ndarray, method: str, axis: int) -> np.ndarray:
    if method == "sum":
        return np.sum(window, axis=axis)
    if method == "mean":
        return np.mean(window, axis=axis)
    if method == "max":
        return np.max(window, axis=axis)
    if method == "min":
        return np.min(window, axis=axis)
    if method == "std":
        return np.std(window, axis=axis)
    if method == "median":
        return np.median(window, axis=axis)
    return np.sum(window, axis=axis)


def get_split_data(data: Any, split: str) -> tuple:
    num_train_samples = len(data.demo_sample_infos)
    if split == "train":
        influence_matrix = data.influence_matrix[:, :num_train_samples]
        demo_episodes = data.demo_episodes
        ep_lens = np.array([ep.num_samples for ep in demo_episodes], dtype=np.int64)
        ep_ends = ep_lens.cumsum()
        ep_idxs = [np.arange(0 if i == 0 else ep_ends[i - 1], ep_end) for i, ep_end in enumerate(ep_ends)]
    elif split == "holdout":
        influence_matrix = data.influence_matrix[:, num_train_samples:]
        demo_episodes = data.holdout_episodes
        ep_lens = np.array([ep.num_samples for ep in demo_episodes], dtype=np.int64)
        ep_ends = ep_lens.cumsum()
        ep_idxs = [np.arange(0 if i == 0 else ep_ends[i - 1], ep_end) for i, ep_end in enumerate(ep_ends)]
    elif split == "both":
        influence_matrix = data.influence_matrix
        demo_episodes = list(data.demo_episodes) + list(data.holdout_episodes)
        ep_idxs = []
        ep_lens_list = []
        train_ep_lens = np.array([ep.num_samples for ep in data.demo_episodes], dtype=np.int64)
        train_ep_ends = train_ep_lens.cumsum()
        for i, ep_end in enumerate(train_ep_ends):
            ep_idxs.append(np.arange(0 if i == 0 else train_ep_ends[i - 1], ep_end))
        ep_lens_list.extend(train_ep_lens.tolist())
        holdout_ep_lens = np.array([ep.num_samples for ep in data.holdout_episodes], dtype=np.int64)
        holdout_ep_ends = holdout_ep_lens.cumsum() + num_train_samples
        for i, ep_end in enumerate(holdout_ep_ends):
            ep_idxs.append(np.arange(num_train_samples if i == 0 else holdout_ep_ends[i - 1], ep_end))
        ep_lens_list.extend(holdout_ep_lens.tolist())
        ep_lens = np.array(ep_lens_list, dtype=np.int64)
    else:
        raise ValueError(f"Unknown split: {split}")
    return influence_matrix, demo_episodes, ep_idxs, ep_lens


def build_windows_from_rollout_timestep_embeddings(
    rollout_emb: np.ndarray,
    episode_lengths: list,
    episode_successes: list,
    window_width: int,
    stride: int,
    aggregation: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    all_embeddings: List[np.ndarray] = []
    all_metadata: List[Dict[str, Any]] = []
    offset = 0
    for ep_idx, ep_len in enumerate(episode_lengths):
        ep_emb = rollout_emb[offset : offset + ep_len]
        offset += ep_len
        if ep_len < window_width:
            continue
        for start in range(0, ep_len - window_width + 1, stride):
            end = start + window_width
            window = ep_emb[start:end]
            emb = aggregate_window(window, aggregation, axis=0)
            all_embeddings.append(emb)
            all_metadata.append(
                {
                    "rollout_idx": ep_idx,
                    "window_start": start,
                    "window_end": end,
                    "window_width": window_width,
                    "success": episode_successes[ep_idx],
                }
            )
    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    return embeddings_arr, all_metadata


def extract_infembed_slice_windows(
    eval_dir_abs: pathlib.Path,
    window_width: int,
    stride: int,
    aggregation: str,
    infembed_dir: pathlib.Path | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract sliding-window embeddings from InfEmbed output.

    Args:
        eval_dir_abs:  Absolute path to the eval output directory.
        window_width:  Sliding-window width in timesteps.
        stride:        Sliding-window stride.
        aggregation:   How to aggregate within each window (sum/mean/…).
        infembed_dir:  Explicit path to the directory containing
                       ``infembed_embeddings.npz``.  When provided, the
                       legacy ``default_trak_results-*`` glob is skipped.
                       Required for flywheel runs that have no TRAK dir.
    """
    if infembed_dir is not None:
        trak_dir = pathlib.Path(infembed_dir)
    else:
        trak_dirs = sorted(eval_dir_abs.glob("default_trak_results-*"))
        if not trak_dirs:
            raise FileNotFoundError(
                f"No InfEmbed/TRAK results found in {eval_dir_abs}. "
                "Pass infembed_dir explicitly for flywheel eval dirs."
            )
        trak_dir = trak_dirs[-1]

    emb_path = trak_dir / "infembed_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"InfEmbed embeddings not found: {emb_path}")

    print(f"  Loading infembed embeddings from {emb_path}")
    with np.load(emb_path) as f:
        rollout_emb = np.asarray(f["rollout_embeddings"])

    episodes_meta_path = eval_dir_abs / "episodes" / "metadata.yaml"
    with open(episodes_meta_path) as f:
        episodes_meta = yaml.safe_load(f)
    episode_lengths = episodes_meta["episode_lengths"]
    episode_successes = episodes_meta.get("episode_successes", [None] * len(episode_lengths))

    print(f"  Rollout embeddings: {rollout_emb.shape}")
    print(f"  Sliding windows: width={window_width}, stride={stride}, agg={aggregation}")

    return build_windows_from_rollout_timestep_embeddings(
        rollout_emb,
        episode_lengths,
        episode_successes,
        window_width,
        stride,
        aggregation,
    )


def extract_trak_slice_windows(
    eval_dir_abs: pathlib.Path,
    train_dir_base: str | None,
    repo_root: pathlib.Path,
    seed: str,
    reference_seed: str,
    window_width: int,
    stride: int,
    aggregation: str,
    demo_split: str = "both",
    level: str = "rollout",
    train_ckpt: str = "latest",
    exp_date: str = "default",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    from policy_doctor.data.influence_loader import load_influence_data
    from influence_visualizer.data_loader import get_train_dir_for_seed

    eval_dir_str = str(eval_dir_abs)
    if train_dir_base:
        train_dir_seed = get_train_dir_for_seed(train_dir_base, seed, reference_seed)
        train_dir_str = str(repo_root / train_dir_seed)
    else:
        raise ValueError("train_dir required in task config for TRAK clustering")

    print(f"  Loading TRAK influence data for seed {seed}")
    data = load_influence_data(
        eval_dir=eval_dir_str,
        train_dir=train_dir_str,
        train_ckpt=train_ckpt,
        exp_date=exp_date,
        include_holdout=True,
        image_dataset_path=None,
        lazy_load_images=True,
    )
    inf_mat, _, _, _ = get_split_data(data, demo_split)
    print(f"  Influence matrix ({demo_split} split): {inf_mat.shape}")
    print(
        f"  Level: {level}, sliding windows: width={window_width}, stride={stride}, agg={aggregation}"
    )
    return extract_trak_slice_windows_from_container(
        data,
        window_width=window_width,
        stride=stride,
        aggregation=aggregation,
        demo_split=demo_split,
        level=level,
    )


def extract_trak_slice_windows_from_container(
    data: InfluenceDataContainer,
    window_width: int,
    stride: int,
    aggregation: str,
    demo_split: str = "both",
    level: str = "rollout",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, demo_split)

    all_embeddings: List[np.ndarray] = []
    all_metadata: List[Dict[str, Any]] = []

    if level == "rollout":
        for rollout_ep in data.rollout_episodes:
            rollout_indices = np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx)
            rollout_data = influence_matrix[rollout_indices, :]
            ep_len = rollout_data.shape[0]
            if ep_len < window_width:
                continue
            for start in range(0, ep_len - window_width + 1, stride):
                end = start + window_width
                window = rollout_data[start:end, :]
                emb = aggregate_window(window, aggregation, axis=0)
                all_embeddings.append(emb.astype(np.float32))
                all_metadata.append(
                    {
                        "rollout_idx": rollout_ep.index,
                        "start": start,
                        "end": end,
                        "window_start": start,
                        "window_end": end,
                        "window_width": window_width,
                        "success": getattr(rollout_ep, "success", None),
                    }
                )

    elif level == "demo":
        for demo_ep_idx, demo_ep in enumerate(demo_episodes):
            col_indices = ep_idxs[demo_ep_idx]
            demo_data = influence_matrix[:, col_indices]
            ep_len = demo_data.shape[1]
            if ep_len < window_width:
                continue
            for start in range(0, ep_len - window_width + 1, stride):
                end = start + window_width
                window = demo_data[:, start:end]
                emb = aggregate_window(window, aggregation, axis=1)
                all_embeddings.append(emb.astype(np.float32))
                all_metadata.append(
                    {
                        "demo_idx": demo_ep.index,
                        "start": start,
                        "end": end,
                        "window_start": start,
                        "window_end": end,
                        "window_width": window_width,
                    }
                )
    else:
        raise ValueError(f"Unknown clustering level: {level}. Expected 'rollout' or 'demo'.")

    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    return embeddings_arr, all_metadata

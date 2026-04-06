"""Load influence matrix from TRAK memmap or provide in-memory store for tests."""

import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from policy_doctor.data.backing import InMemoryBackingStore, MemmapBackingStore
from policy_doctor.data.structures import EpisodeInfo, GlobalInfluenceMatrix


def _build_episode_infos_from_lengths(
    episode_lengths: List[int],
    success_mask: List[bool],
) -> List[EpisodeInfo]:
    """Build EpisodeInfo list from per-episode sample counts and optional success."""
    episodes = []
    start = 0
    for i, length in enumerate(episode_lengths):
        success = success_mask[i] if (success_mask and i < len(success_mask)) else None
        episodes.append(
            EpisodeInfo(
                index=i,
                num_samples=length,
                sample_start_idx=start,
                sample_end_idx=start + length,
                success=success,
                raw_length=length,
            )
        )
        start += length
    return episodes


def load_influence_matrix_from_memmap(
    eval_dir: Path,
    exp_name: str,
    rollout_episode_lengths: List[int],
    rollout_success: List[bool],
    demo_episode_lengths: List[int],
) -> Tuple[GlobalInfluenceMatrix, List[EpisodeInfo], List[EpisodeInfo]]:
    """Load influence matrix from TRAK memmap and build episode infos.

    Expects eval_dir/exp_name/metadata.json and experiments.json, and
    eval_dir/exp_name/scores/all_episodes.mmap. Matrix on disk is (train, test);
    we expose (test, train) = (rollout, demo).

    rollout_episode_lengths and demo_episode_lengths must match the actual sizes
    used by TRAK (sum(rollout_episode_lengths) = test_set_size, etc.).
    """
    eval_dir = Path(eval_dir)
    exp_dir = eval_dir / exp_name
    metadata_path = exp_dir / "metadata.json"
    experiments_path = exp_dir / "experiments.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"TRAK metadata not found at {metadata_path}")
    if not experiments_path.exists():
        raise FileNotFoundError(f"TRAK experiments not found at {experiments_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(experiments_path) as f:
        experiments = json.load(f)
    train_set_size = metadata["train set size"]
    test_set_size = experiments["all_episodes"]["num_targets"]
    scores_path = exp_dir / "scores" / "all_episodes.mmap"
    if not scores_path.exists():
        raise FileNotFoundError(f"TRAK scores not found at {scores_path}")

    # On disk: (train_set_size, test_set_size). We want (test_set_size, train_set_size).
    store = MemmapBackingStore(
        scores_path,
        shape=(train_set_size, test_set_size),
        dtype=np.float32,
    )
    # Transpose: we'll read (r,d) as (test_idx, train_idx) so we need to index store as (train_idx, test_idx)
    # So when we want M_rollout_demo[r,d] we read from disk at [d, r]. So we need a wrapper that transposes.
    # Easiest: load the transpose into an in-memory store for now to avoid reimplementing transpose in backing.
    # For true zero-copy we'd need a transposed view; numpy doesn't support that for memmap. So we use a
    # TransposedMemmapStore that implements read_slice(r_lo,r_hi,d_lo,d_hi) as store.read_slice(d_lo,d_hi,r_lo,r_hi).T
    # or store[d_lo:d_hi, r_lo:r_hi].T
    class TransposedStore:
        def __init__(self, mmap: np.memmap) -> None:
            self._mmap = mmap
            self._shape = (mmap.shape[1], mmap.shape[0])

        @property
        def shape(self) -> tuple:
            return self._shape

        def read_slice(self, r_lo: int, r_hi: int, d_lo: int, d_hi: int) -> np.ndarray:
            block = np.array(self._mmap[d_lo:d_hi, r_lo:r_hi], dtype=np.float32)
            return block.T

        def read_cell(self, r: int, d: int) -> float:
            return float(self._mmap[d, r])

    transposed = TransposedStore(store._mmap)
    rollout_episodes = _build_episode_infos_from_lengths(
        rollout_episode_lengths, rollout_success
    )
    demo_episodes = _build_episode_infos_from_lengths(
        demo_episode_lengths, [False] * len(demo_episode_lengths)
    )
    global_matrix = GlobalInfluenceMatrix(
        transposed, rollout_episodes, demo_episodes
    )
    return global_matrix, rollout_episodes, demo_episodes


def create_global_influence_from_array(
    matrix: np.ndarray,
    rollout_episode_lengths: List[int],
    demo_episode_lengths: List[int],
    rollout_success: Union[List[bool], None] = None,
) -> GlobalInfluenceMatrix:
    """Create GlobalInfluenceMatrix from an in-memory array (e.g. for tests).

    matrix shape must be (sum(rollout_episode_lengths), sum(demo_episode_lengths)).
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] != sum(rollout_episode_lengths) or matrix.shape[1] != sum(
        demo_episode_lengths
    ):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match "
            f"rollout {sum(rollout_episode_lengths)} x demo {sum(demo_episode_lengths)}"
        )
    rollout_episodes = _build_episode_infos_from_lengths(
        rollout_episode_lengths,
        rollout_success or [None] * len(rollout_episode_lengths),
    )
    demo_episodes = _build_episode_infos_from_lengths(
        demo_episode_lengths,
        [False] * len(demo_episode_lengths),
    )
    return GlobalInfluenceMatrix(matrix, rollout_episodes, demo_episodes)

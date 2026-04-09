import hashlib
import pathlib
from collections import defaultdict
from typing import Optional, Tuple, Union

import h5py
import numba
import numpy as np
import yaml

from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    assert episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    if len(indices) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    indices = np.array(indices)
    return indices


@numba.jit(nopython=True)
def create_indices_with_sample_mask(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    sample_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
    sample_mask_drop_threshold: float = 1.0,
) -> np.ndarray:
    """Create sequence indices, skipping sequences that exceed excluded fraction.

    Like create_indices(), but additionally checks sample_mask. A sequence is
    dropped when the fraction of samples in its window that are excluded
    (sample_mask[s] == False) is >= sample_mask_drop_threshold.

    Args:
        episode_ends: Array of episode end indices in the replay buffer.
        sequence_length: Length of each sampled sequence.
        episode_mask: Boolean mask for which episodes to include.
        sample_mask: Boolean mask of shape (total_num_samples,). True = keep.
        pad_before: Number of steps to pad before each episode.
        pad_after: Number of steps to pad after each episode.
        debug: Whether to run debug assertions.
        sample_mask_drop_threshold: Drop sequence if fraction of window excluded
            is >= this value (0.0 = drop if any excluded, 1.0 = drop only if
            entire window excluded). Default 1.0. In [0.0, 1.0].

    Returns:
        Array of shape (N, 4) with columns:
        [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    """
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )

            # Count excluded samples in this window.
            n_excluded = 0
            window_len = buffer_end_idx - buffer_start_idx
            for s in range(buffer_start_idx, buffer_end_idx):
                if not sample_mask[s]:
                    n_excluded += 1
            # Drop sequence if excluded fraction >= threshold.
            # threshold 0.0: drop if any excluded. threshold 1.0: drop only if all excluded.
            drop = False
            if sample_mask_drop_threshold <= 0.0:
                drop = n_excluded > 0
            else:
                drop = (window_len > 0 and
                        (n_excluded / float(window_len)) >= sample_mask_drop_threshold)
            if drop:
                continue

            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    if len(indices) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    indices = np.array(indices)
    return indices


def load_sample_mask_from_curation_config(
    curation_config_path: Union[str, pathlib.Path],
    episode_ends: np.ndarray,
    episode_mask: np.ndarray,
    *,
    verify_dataset_identity: bool = True,
) -> np.ndarray:
    """Load a curation config YAML and convert it to a sample mask.

    The curation config format (from influence_visualizer) is:
        slices:
          - episode_idx: 3
            start: 10
            end: 25
          ...
        episode_lengths:
            3: 45
            7: 60

    Index semantics: (start, end) are RAW timestep indices within the episode
    (0 to raw_length-1), i.e. replay buffer / HDF5 sample indices. We map to
    global buffer: abs_start = ep_start + start, abs_end = ep_start + end + 1.

    If episode lengths in the config differ from the dataset (e.g. config from
    a different HDF5 or visualizer run), a warning is emitted and only in-bounds
    slice ranges are applied.

    Args:
        curation_config_path: Path to curation config YAML file.
        episode_ends: Array of episode end indices from the replay buffer.
        episode_mask: Boolean mask of which episodes are in the training set.
        verify_dataset_identity: If False, skip fingerprint / total_raw_samples / episode_length
            checks (unsafe for training; for offline tooling e.g. approximate dataset size).

    Returns:
        Boolean array of shape (total_num_samples,) where True = keep.
    """
    curation_config_path = pathlib.Path(curation_config_path)
    if not curation_config_path.exists():
        raise FileNotFoundError(f"Curation config not found: {curation_config_path}")

    with open(curation_config_path) as f:
        config = yaml.safe_load(f)

    if config is None or "slices" not in config:
        # Empty or invalid config — return all-True mask
        total_samples = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
        return np.ones(total_samples, dtype=bool)

    total_samples = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
    metadata = config.get("metadata") or {}

    # Verify config was created from this dataset (refuse to apply if wrong dataset).
    # Configs from policy_doctor are built from train split only, so compare fingerprint of
    # train episodes' cumulative ends to metadata["dataset_fingerprint"].
    if verify_dataset_identity:
        if metadata.get("dataset_fingerprint"):
            ep_lengths_arr = np.diff(np.concatenate([[0], np.asarray(episode_ends, dtype=np.int64)]))
            train_lengths = ep_lengths_arr[episode_mask]
            train_ends = np.cumsum(train_lengths).astype(np.int64)
            current_fingerprint = hashlib.sha256(train_ends.tobytes()).hexdigest()
            if current_fingerprint != metadata["dataset_fingerprint"]:
                raise ValueError(
                    "Curation config was created from a different dataset (fingerprint mismatch). "
                    "The mask would exclude the wrong samples. Use the same dataset in the visualizer "
                    "and for training, or re-create the curation config from the current dataset."
                )
        if metadata.get("total_raw_samples") is not None:
            train_raw = int(np.sum(np.diff(np.concatenate([[0], np.asarray(episode_ends, dtype=np.int64)]))[episode_mask]))
            if train_raw != int(metadata["total_raw_samples"]):
                raise ValueError(
                    f"Curation config total_raw_samples ({metadata['total_raw_samples']}) does not "
                    f"match dataset train split ({train_raw}). Config was created from a different dataset."
                )

    sample_mask = np.ones(total_samples, dtype=bool)

    # Build episode start/end lookup
    ep_starts = np.zeros(len(episode_ends), dtype=np.int64)
    ep_lengths = np.zeros(len(episode_ends), dtype=np.int64)
    for i in range(len(episode_ends)):
        ep_starts[i] = 0 if i == 0 else episode_ends[i - 1]
        ep_lengths[i] = episode_ends[i] - ep_starts[i]

    # Refuse to apply if episode lengths in config don't match dataset (wrong HDF5/version)
    config_ep_lengths = config.get("episode_lengths", {})
    length_mismatches = []
    for ep_idx_str, expected_len in config_ep_lengths.items():
        ep_idx = int(ep_idx_str)
        if ep_idx >= len(episode_ends):
            continue
        actual_len = int(ep_lengths[ep_idx])
        if actual_len != int(expected_len):
            length_mismatches.append((ep_idx, int(expected_len), actual_len))
    if verify_dataset_identity and length_mismatches:
        ex = length_mismatches[0]
        raise ValueError(
            "[Sample Curation] Episode length mismatch(es) between curation config and dataset. "
            "The config was created from a different HDF5 or dataset version. "
            "Refusing to apply the mask so the wrong samples are never excluded. "
            f"Example: episode {ex[0]} config length={ex[1]} dataset length={ex[2]}. "
            f"Total mismatches: {len(length_mismatches)}. "
            "Use the same dataset in the visualizer and for training, or re-create the curation config from the current dataset."
        )

    # Apply slices — mark samples as excluded (False); clamp to actual episode bounds
    for slice_info in config["slices"]:
        ep_idx = int(slice_info["episode_idx"])
        start = int(slice_info["start"])
        end = int(slice_info["end"])

        if ep_idx >= len(episode_ends):
            continue

        if not episode_mask[ep_idx]:
            # This episode isn't in the training set, skip
            continue

        ep_start = int(ep_starts[ep_idx])
        ep_len = int(ep_lengths[ep_idx])

        # Defense-in-depth: clamp to episode bounds in case of manual config edits
        if start < 0:
            start = 0
        if end >= ep_len:
            end = ep_len - 1
        if start > end:
            continue

        # Mark samples as excluded (end is inclusive in our config format)
        abs_start = ep_start + start
        abs_end = ep_start + end + 1
        sample_mask[abs_start:abs_end] = False

    # Filter mode: only train episodes (minus excluded slices) should be included.
    # Zero out holdout episodes so they are never used when only sample_curation_config is set.
    for ep_idx in range(len(episode_ends)):
        if not episode_mask[ep_idx]:
            start_s = int(ep_starts[ep_idx])
            end_s = int(episode_ends[ep_idx])
            sample_mask[start_s:end_s] = False

    return sample_mask


def load_holdout_selection_mask(
    curation_config_path: Union[str, pathlib.Path],
    episode_ends: np.ndarray,
    holdout_mask: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    *,
    verify_dataset_identity: bool = True,
) -> np.ndarray:
    """Load a selection-mode curation config and build a sample mask for holdout slices.

    The config must have metadata.curation_mode == "selection". Slices in the config
    define ranges to **include** from holdout; the returned mask is True only in those
    ranges (and only for episodes where holdout_mask[episode_idx] is True).

    Same index semantics and fingerprint/episode_length validation as
    load_sample_mask_from_curation_config.

    Args:
        curation_config_path: Path to curation config YAML.
        episode_ends: Array of episode end indices from the replay buffer.
        holdout_mask: Boolean mask of which episodes are in the holdout set.
        train_mask: Boolean mask of training episodes; used for fingerprint verification
            (must match how run_curation_config computed the stored fingerprint).

    Returns:
        Boolean array of shape (total_num_samples,) where True = include (selected).
        All False if config is not selection mode or has no slices.
    """
    curation_config_path = pathlib.Path(curation_config_path)
    if not curation_config_path.exists():
        raise FileNotFoundError(f"Curation config not found: {curation_config_path}")

    with open(curation_config_path) as f:
        config = yaml.safe_load(f)

    total_samples = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
    sample_mask = np.zeros(total_samples, dtype=bool)

    if config is None or "slices" not in config or not config["slices"]:
        return sample_mask

    metadata = config.get("metadata") or {}
    if metadata.get("curation_mode") != "selection":
        return sample_mask

    # Same fingerprint and episode length validation as load_sample_mask_from_curation_config.
    # Fingerprint is computed from cumulative training episode lengths (not full episode_ends).
    if verify_dataset_identity:
        if metadata.get("dataset_fingerprint"):
            if train_mask is not None:
                ep_lengths_arr = np.diff(
                    np.concatenate([[0], np.asarray(episode_ends, dtype=np.int64)])
                )
                train_lengths = ep_lengths_arr[train_mask]
                arr = np.cumsum(train_lengths).astype(np.int64)
            else:
                arr = np.asarray(episode_ends, dtype=np.int64)
            current_fingerprint = hashlib.sha256(arr.tobytes()).hexdigest()
            if current_fingerprint != metadata["dataset_fingerprint"]:
                raise ValueError(
                    "Holdout selection config was created from a different dataset (fingerprint mismatch)."
                )
        if metadata.get("total_raw_samples") is not None:
            if train_mask is not None:
                ep_lengths_arr = np.diff(
                    np.concatenate([[0], np.asarray(episode_ends, dtype=np.int64)])
                )
                train_total = int(ep_lengths_arr[train_mask].sum())
            else:
                train_total = total_samples
            if train_total != int(metadata["total_raw_samples"]):
                raise ValueError(
                    f"Holdout selection config total_raw_samples ({metadata['total_raw_samples']}) "
                    f"does not match dataset train split ({train_total})."
                )

    ep_starts = np.zeros(len(episode_ends), dtype=np.int64)
    ep_lengths = np.zeros(len(episode_ends), dtype=np.int64)
    for i in range(len(episode_ends)):
        ep_starts[i] = 0 if i == 0 else episode_ends[i - 1]
        ep_lengths[i] = episode_ends[i] - ep_starts[i]

    config_ep_lengths = config.get("episode_lengths", {})
    if verify_dataset_identity:
        for ep_idx_str, expected_len in config_ep_lengths.items():
            ep_idx = int(ep_idx_str)
            if ep_idx >= len(episode_ends):
                continue
            if int(ep_lengths[ep_idx]) != int(expected_len):
                raise ValueError(
                    "[Holdout Selection] Episode length mismatch between config and dataset "
                    f"for episode {ep_idx}. Re-create the config from the current dataset."
                )

    for slice_info in config["slices"]:
        ep_idx = int(slice_info["episode_idx"])
        start = int(slice_info["start"])
        end = int(slice_info["end"])

        if ep_idx >= len(episode_ends):
            continue
        if not holdout_mask[ep_idx]:
            continue

        ep_start = int(ep_starts[ep_idx])
        ep_len = int(ep_lengths[ep_idx])

        if start < 0:
            start = 0
        if end >= ep_len:
            end = ep_len - 1
        if start > end:
            continue

        abs_start = ep_start + start
        abs_end = ep_start + end + 1
        sample_mask[abs_start:abs_end] = True

    return sample_mask


def build_combined_curation_masks(
    episode_ends: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    sample_curation_config: Optional[Union[str, pathlib.Path]] = None,
    holdout_selection_config: Optional[Union[str, pathlib.Path]] = None,
    sample_mask_drop_threshold: float = 1.0,
    *,
    verify_curation_dataset_identity: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build effective episode mask and sample mask for train + optional holdout selection.

    When only sample_curation_config (filter) is set: train episodes with excluded slices.
    When only holdout_selection_config is set: train episodes all included, plus holdout
    episodes/samples from the selection config.
    When both are set: train with filter applied, plus selected holdout slices.

    Args:
        episode_ends: Episode end indices from the replay buffer.
        train_mask: Boolean mask of training episodes.
        holdout_mask: Boolean mask of holdout episodes.
        sample_curation_config: Path to filter config (slices to exclude from train).
        holdout_selection_config: Path to selection config (slices to include from holdout).
        sample_mask_drop_threshold: Passed through when building train sample mask.
        verify_curation_dataset_identity: Passed to YAML loaders; set False only for offline
            tooling (e.g. approximate sequence counts when fingerprints disagree).

    Returns:
        (effective_episode_mask, effective_sample_mask). effective_sample_mask is None
        if neither config is set (no sample-level masking).
    """
    total_samples = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
    n_episodes = len(episode_ends)

    use_filter = sample_curation_config is not None
    use_selection = (
        holdout_selection_config is not None
        and pathlib.Path(holdout_selection_config).exists()
    )

    if not use_filter and not use_selection:
        return train_mask.copy(), None

    effective_episode_mask = train_mask.copy()

    # Precompute episode start positions (used by both filter and selection)
    ep_starts = np.zeros(n_episodes + 1, dtype=np.int64)
    ep_starts[1:] = episode_ends

    # Train part of sample mask: True only for train episodes, then apply filter
    if use_filter:
        train_sample_mask = load_sample_mask_from_curation_config(
            curation_config_path=sample_curation_config,
            episode_ends=episode_ends,
            episode_mask=train_mask,
            verify_dataset_identity=verify_curation_dataset_identity,
        )
    else:
        # Only mark train episode samples as True; holdout regions stay False
        # so that holdout selection controls exactly which holdout samples are used.
        train_sample_mask = np.zeros(total_samples, dtype=bool)
        for ep_idx in range(n_episodes):
            if train_mask[ep_idx]:
                train_sample_mask[int(ep_starts[ep_idx]):int(episode_ends[ep_idx])] = True

    # When using selection, also zero out holdout regions in train_sample_mask
    # (load_sample_mask_from_curation_config starts all-True and only sets filtered
    # slices to False, so holdout regions would remain True without this step).
    if use_selection and use_filter:
        for ep_idx in range(n_episodes):
            if holdout_mask[ep_idx]:
                train_sample_mask[int(ep_starts[ep_idx]):int(episode_ends[ep_idx])] = False

    # Holdout selection: which holdout episodes have any selected slice
    if use_selection:
        holdout_sample_mask = load_holdout_selection_mask(
            curation_config_path=holdout_selection_config,
            episode_ends=episode_ends,
            train_mask=train_mask,
            holdout_mask=holdout_mask,
            verify_dataset_identity=verify_curation_dataset_identity,
        )
        holdout_episodes_with_slices = set()
        for ep_idx in range(n_episodes):
            if not holdout_mask[ep_idx]:
                continue
            start = int(ep_starts[ep_idx])
            end = int(episode_ends[ep_idx])
            if np.any(holdout_sample_mask[start:end]):
                holdout_episodes_with_slices.add(ep_idx)
        for ep_idx in holdout_episodes_with_slices:
            effective_episode_mask[ep_idx] = True
        # Combine: train regions from train_sample_mask, holdout from holdout_sample_mask
        effective_sample_mask = train_sample_mask.copy()
        effective_sample_mask |= holdout_sample_mask
    else:
        effective_sample_mask = train_sample_mask

    return effective_episode_mask, effective_sample_mask


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


def filter_training_episodes(
    train_mask: np.ndarray,
    filter_ratio: float,
    curation_config: pathlib.Path,
    curation_method: str,
    seed: int,
) -> np.ndarray:
    """Filter training data by curation method."""
    if filter_ratio <= 0.0:
        return train_mask

    # Load curation config.
    with open(curation_config, "r") as f:
        config = yaml.safe_load(f)

    # Filter training episodes.
    num_filter = int(train_mask.sum() * filter_ratio)
    filter_idxs = np.array(config[curation_method][seed])
    assert np.all(train_mask[filter_idxs]), "Indexing non-training data."
    train_mask[filter_idxs[:num_filter]] = False

    return train_mask


def select_holdout_episodes(
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    select_ratio: float,
    curation_config: pathlib.Path,
    curation_method: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select training data by curation method."""
    if select_ratio <= 0.0:
        return train_mask, holdout_mask

    # Load curation config.
    with open(curation_config, "r") as f:
        config = yaml.safe_load(f)

    # Select holdout episodes.
    num_select = int(holdout_mask.sum() * select_ratio)
    select_idxs = np.array(config[curation_method][seed])
    assert np.all(holdout_mask[select_idxs]), "Indexing non-holdout data."
    assert not np.any(train_mask[select_idxs]), "Indexing training data."
    holdout_mask[select_idxs[:num_select]] = False
    train_mask[select_idxs[:num_select]] = True

    return train_mask, holdout_mask


# Note: On Line 269, we sample training, then validation, and then holdout demos.
# Optionally for future experiments, we should swap the order in which validation
# demos and holdout demos are sampled. That is, we want a sliding window between
# sampled training and holdout demos, without validation demos being sampled in
# between. Doing so would result in greater coherence across the filtering and
# selection experiments because they would a) be curating from the same pool of
# demos and b) they would share the same validation set.
def get_dataset_masks(
    dataset_path: Union[str, pathlib.Path],
    num_episodes: int,
    val_ratio: float,
    max_train_episodes: Optional[int] = None,
    train_ratio: Optional[float] = None,
    uniform_quality: bool = False,
    curate_dataset: bool = False,
    curation_config_dir: Optional[str] = None,
    curation_method: Optional[str] = None,
    filter_ratio: Optional[float] = None,
    select_ratio: Optional[float] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training, validation, and holdout masks."""
    assert not (max_train_episodes is not None and train_ratio is not None), (
        "One or neither of max_train_episodes or train_ratio should be specified."
    )

    # Dataset splits.
    if max_train_episodes is not None:
        num_train = max_train_episodes
        num_val = int(num_episodes * val_ratio)
        num_holdout = num_episodes - num_train - num_val
    else:
        train_ratio = 1.0 - val_ratio if train_ratio is None else train_ratio
        num_train = int(num_episodes * train_ratio)
        num_val = int(num_episodes * val_ratio)
        num_holdout = num_episodes - num_train - num_val

    assert_str = f"num_train ({num_train}) + num_val ({num_val}) + num_holdout ({num_holdout}) != num_episodes ({num_episodes})"
    assert num_train + num_val + num_holdout == num_episodes, assert_str

    # Dataset info.
    dataset_path = pathlib.Path(dataset_path)
    path_str = str(dataset_path)
    parts = dataset_path.parts
    parts_lower_set = {p.lower() for p in parts}
    path_lower = path_str.lower()
    dataset_name = ""
    # Path segment match (case-insensitive on segments).
    for known in ("robomimic", "hardware", "pusht", "libero", "mimicgen", "robocasa"):
        if known in parts_lower_set:
            dataset_name = known
            break
    # Resolved symlinks may point at e.g. .../robocasa_data/... or .../mimicgen_data/...
    # with no literal ``robocasa`` / ``mimicgen`` path component.
    if not dataset_name:
        if "libero" in path_lower:
            dataset_name = "libero"
        elif "mimicgen" in path_lower:
            dataset_name = "mimicgen"
        elif "robocasa" in path_lower:
            dataset_name = "robocasa"
        elif "robomimic" in path_lower:
            dataset_name = "robomimic"
        elif "hardware" in path_lower:
            dataset_name = "hardware"
        elif "pusht" in path_lower:
            dataset_name = "pusht"
    if dataset_name in ["robomimic", "hardware"]:
        task_name = dataset_path.parts[-3]
        task_type = dataset_path.parts[-2]
    elif dataset_name in ("mimicgen", "robocasa"):
        # Local / symlinked layouts: same i.i.d. train/val/holdout split as ``ph``.
        task_name = ""
        task_type = "ph"
    elif dataset_name == "pusht":
        task_name = dataset_name
        task_type = "ph"
    elif len(dataset_path.parts) > 2 and dataset_path.parts[2] == "eval_save_episodes":
        task_name = ""
        task_type = "ph"
    elif len(dataset_path.parts) > 2 and dataset_path.parts[2] == "eval_save_episodes_real":
        task_name = ""
        task_type = "ph_real"
    elif "libero" in path_str.lower() or dataset_name == "libero":
        # LIBERO benchmarks: same i.i.d. train/val/holdout as ph.
        task_name = ""
        task_type = "ph"
    elif not dataset_name:
        raise ValueError(f"Unsupported dataset (unrecognized path): {dataset_path}.")
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}.")

    if task_type == "ph_real":
        # Samples in consecutive order (from evaluation).
        train_mask = np.zeros(num_episodes, dtype=bool)
        train_mask[:num_train] = True
        val_mask = np.zeros(num_episodes, dtype=bool)
        val_mask[num_train : num_train + num_val] = True
        holdout_mask = ~np.logical_or(train_mask, val_mask)

    elif task_type == "ph" or not uniform_quality:
        # i.i.d. sampling across all quality tiers.
        val_mask = get_val_mask(num_episodes, val_ratio, seed=seed)
        train_mask = ~val_mask
        if max_train_episodes is not None:
            train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
        else:
            train_mask = downsample_mask(train_mask, num_train, seed=seed)
        holdout_mask = ~np.logical_or(train_mask, val_mask)

    elif task_type == "mh" and uniform_quality:
        # i.i.d. sampling within quality tiers.
        assert max_train_episodes is None, "Does not support max_train_episodes."
        if dataset_name == "robomimic":
            with h5py.File(dataset_path) as file:
                if any(x in task_name for x in ["lift", "can", "square"]):
                    demo_quality_sets = ["worse", "okay", "better"]
                elif "transport" in task_name:
                    demo_quality_sets = [
                        "worse",
                        "worse_okay",
                        "okay",
                        "okay_better",
                        "better",
                        "worse_better",
                    ]
                else:
                    raise ValueError(f"Task {task_name} is not of type {task_type}.")
                decode_fn = lambda x: np.array(
                    [int(name.decode().split("_")[-1]) for name in x]
                )
                demo_quality_idxs = [
                    decode_fn(file["mask"][s][:]) for s in demo_quality_sets
                ]

        elif dataset_name == "hardware":
            if task_name == "figure8_v3":
                demo_quality_sets = ["0", "1", "2"]
            elif task_name in ["figure8_v4", "bookshelf_v2", "bookshelf_v3"]:
                demo_quality_sets = ["0", "1", "2", "3"]
            elif any(x in task_name for x in ["figure8", "tuckbox", "bookshelf"]):
                demo_quality_sets = ["0", "1"]
            else:
                raise ValueError(f"Task {task_name} is not of type {task_type}.")

            quality_to_episode_idx = defaultdict(list)
            for episode_path in sorted((dataset_path.parent / "episodes").iterdir()):
                if episode_path.is_dir():
                    with open(episode_path / "quality.txt", "r") as file:
                        quality_label = file.read().strip()
                        assert quality_label in demo_quality_sets, (
                            f"Unexpected quality label: {quality_label}"
                        )
                    quality_to_episode_idx[quality_label].append(int(episode_path.stem))
            demo_quality_idxs = [
                np.array(quality_to_episode_idx[s]) for s in demo_quality_sets
            ]

        # Dataset masks.
        train_mask = np.zeros(num_episodes, dtype=bool)
        val_mask = np.zeros(num_episodes, dtype=bool)
        holdout_mask = np.zeros(num_episodes, dtype=bool)

        # Samples per quality tier (accounts for quality sets of varying sizes).
        demo_quality_counts = np.array(
            [len(idxs) for idxs in demo_quality_idxs], dtype=float
        )
        demo_quality_ratios = demo_quality_counts / demo_quality_counts.sum()
        num_samples_per_set = defaultdict(list)
        for i, quality_label in enumerate(demo_quality_sets):
            for k in [num_train, num_val, num_holdout]:
                num_samples_per_set[quality_label].append(
                    round(k * demo_quality_ratios[i])
                )

        rng = np.random.default_rng(seed=seed)
        for idxs, quality_label in zip(demo_quality_idxs, demo_quality_sets):
            shuffle_idxs = idxs.copy()
            rng.shuffle(shuffle_idxs)
            start_idx = 0
            for split_mask, split_size in zip(
                [train_mask, val_mask, holdout_mask],
                num_samples_per_set[quality_label],
            ):
                end_idx = start_idx + split_size
                split_mask[shuffle_idxs[start_idx:end_idx]] = True
                start_idx = end_idx
    else:
        raise ValueError(f"Unsupport task type {task_type}.")

    # Assert no remainder demos.
    assert (
        train_mask.sum() == num_train
        and val_mask.sum() == num_val
        and holdout_mask.sum() == num_holdout
        and not np.logical_and(train_mask, val_mask).any()
        and not np.logical_and(train_mask, holdout_mask).any()
        and not np.logical_and(val_mask, holdout_mask).any()
    ), "Remainder demos!"

    # Dataset curation.
    if curate_dataset:
        assert (
            (curation_config_dir is not None)
            and (curation_method is not None)
            and (filter_ratio is not None and 0.0 <= filter_ratio <= 1.0)
            and (select_ratio is not None and 0.0 <= select_ratio <= 1.0)
        ), "Curation arguments must be set together"

        # Filter episodes from training data.
        train_mask = filter_training_episodes(
            train_mask=train_mask,
            filter_ratio=filter_ratio,
            curation_config=pathlib.Path(curation_config_dir) / "train_config.yaml",
            curation_method=curation_method,
            seed=seed,
        )

        # Select episodes from holdout data.
        train_mask, holdout_mask = select_holdout_episodes(
            train_mask=train_mask,
            holdout_mask=holdout_mask,
            select_ratio=select_ratio,
            curation_config=pathlib.Path(curation_config_dir) / "holdout_config.yaml",
            curation_method=curation_method,
            seed=seed,
        )

    return train_mask, val_mask, holdout_mask


class SequenceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        sample_mask: Optional[np.ndarray] = None,
        sample_mask_drop_threshold: float = 1.0,
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        sample_mask: Optional boolean array of shape (total_num_samples,).
            If provided, sequences are dropped when the fraction of excluded
            (False) samples in the window is >= sample_mask_drop_threshold.
        sample_mask_drop_threshold: In [0.0, 1.0]. Default 1.0 = drop only if
            entire window excluded; 0.0 = drop if any excluded.
        """

        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            if sample_mask is not None:
                total_samples = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
                assert sample_mask.shape == (total_samples,), (
                    f"sample_mask shape {sample_mask.shape} does not match "
                    f"total samples {total_samples}"
                )
                baseline_indices = create_indices(
                    episode_ends,
                    sequence_length=sequence_length,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    episode_mask=episode_mask,
                )
                indices = create_indices_with_sample_mask(
                    episode_ends,
                    sequence_length=sequence_length,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    episode_mask=episode_mask,
                    sample_mask=sample_mask,
                    sample_mask_drop_threshold=sample_mask_drop_threshold,
                )
                n_baseline = len(baseline_indices)
                n_curated = len(indices)
                n_dropped = n_baseline - n_curated
                pct_dropped = 100.0 * n_dropped / n_baseline if n_baseline else 0.0
                n_included = int(sample_mask.sum())
                n_total_samples = len(sample_mask)
                n_episodes_active = int(episode_mask.sum())
                print(
                    f"\n{'='*70}\n"
                    f"  [Sample Curation] SEQUENCE-LEVEL IMPACT\n"
                    f"  Episodes active:     {n_episodes_active}\n"
                    f"  Samples included:    {n_included}/{n_total_samples}"
                    f"  ({100.0 * n_included / n_total_samples:.1f}%)\n"
                    f"  Baseline sequences:  {n_baseline}\n"
                    f"  After curation:      {n_curated}\n"
                    f"  Sequences dropped:   {n_dropped}  ({pct_dropped:.1f}%)\n"
                    f"  Drop threshold:      {sample_mask_drop_threshold}\n"
                    f"{'='*70}\n"
                )
            else:
                indices = create_indices(
                    episode_ends,
                    sequence_length=sequence_length,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    episode_mask=episode_mask,
                )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full(
                    (n_data,) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                try:
                    sample[:k_data] = input_arr[
                        buffer_start_idx : buffer_start_idx + k_data
                    ]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load key '{key}' for buffer range "
                        f"[{buffer_start_idx}:{buffer_start_idx + k_data}]: {e}"
                    ) from e
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

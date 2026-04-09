"""Curation configuration management for sample-wise data curation.

This module provides functions to load, save, and validate curation configs.
Configs are stored as YAML files in influence_visualizer/configs/<task_config>/curation/

CRITICAL - Index semantics (must match training):
  - slice (start, end): RAW timestep indices within the episode (0 to raw_length-1).
    Same as replay buffer / HDF5 sample indices within that episode.
  - episode_lengths[episode_idx]: RAW episode length (number of timesteps in HDF5).
  - Training (load_sample_mask_from_curation_config in diffusion_policy/common/sampler.py)
    applies these by: global_index = ep_start + start (and end), then marks
    sample_mask[global_start:global_end+1] = False. So indices are correct end-to-end
    when both visualizer and training use the same dataset.

A curation config stores:
  - Training demo slices to include/exclude for data curation
  - Episode length metadata for safety validation (raw lengths)
  - Provenance metadata (when/how slices were added)

Format:
    slices:
      - episode_idx: 3
        start: 10
        end: 25
        label: "reaching"
        source: "behavior_search"
      - ...
    metadata: {...}
    episode_lengths: {...}
    # Optional: selections (one per "add to curation config" click); slices = merge(selections)
    selections:
      - id: 0
        label: "reaching"
        demo_split: "train"
        source: "behavior_search_train"
        slices: [...]
        created: "2026-02-09T12:00:00"
      - ...
"""

import hashlib
import pathlib
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml


def compute_dataset_fingerprint(episode_ends: np.ndarray) -> str:
    """Compute a deterministic hash of the dataset layout (episode boundaries).

    Used to verify at training time that the curation config was created from
    the same dataset; if the fingerprint differs, we refuse to apply the mask
    so the wrong samples are never excluded.
    """
    arr = np.asarray(episode_ends, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass
class CurationSlice:
    """A single curation slice within a training demo episode."""

    episode_idx: int
    start: int
    end: int
    label: str = ""
    source: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_idx": int(self.episode_idx),
            "start": int(self.start),
            "end": int(self.end),
            "label": self.label,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CurationSlice":
        return CurationSlice(
            episode_idx=d["episode_idx"],
            start=d["start"],
            end=d["end"],
            label=d.get("label", ""),
            source=d.get("source", "manual"),
        )


# ---------------------------------------------------------------------------
# Selection config: one "selection" = one "add to curation config" click
# ---------------------------------------------------------------------------


@dataclass
class SelectionSlice:
    """A single demo slice in a selection, with optional link to the rollout slice that produced it."""

    episode_idx: int
    start: int
    end: int
    label: str = ""
    source: str = "manual"
    # Rollout slice this demo slice was linked to (for source distribution)
    rollout_episode_idx: Optional[int] = None
    rollout_start: Optional[int] = None
    rollout_end: Optional[int] = None
    # Position in the split's influence matrix (avoids timestep-to-position offset issues)
    local_sample_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "episode_idx": int(self.episode_idx),
            "start": int(self.start),
            "end": int(self.end),
            "label": self.label,
            "source": self.source,
        }
        if self.rollout_episode_idx is not None:
            d["rollout_episode_idx"] = self.rollout_episode_idx
        if self.rollout_start is not None:
            d["rollout_start"] = self.rollout_start
        if self.rollout_end is not None:
            d["rollout_end"] = self.rollout_end
        if self.local_sample_idx is not None:
            d["local_sample_idx"] = int(self.local_sample_idx)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SelectionSlice":
        return SelectionSlice(
            episode_idx=d["episode_idx"],
            start=d["start"],
            end=d["end"],
            label=d.get("label", ""),
            source=d.get("source", "manual"),
            rollout_episode_idx=d.get("rollout_episode_idx"),
            rollout_start=d.get("rollout_start"),
            rollout_end=d.get("rollout_end"),
            local_sample_idx=d.get("local_sample_idx"),
        )

    def to_curation_slice(self) -> CurationSlice:
        return CurationSlice(
            episode_idx=self.episode_idx,
            start=self.start,
            end=self.end,
            label=self.label,
            source=self.source,
        )


@dataclass
class Selection:
    """One batch added by a single 'add to curation config' click."""

    id: int
    label: str
    demo_split: str
    source: str
    slices: List[SelectionSlice] = field(default_factory=list)
    created: Optional[str] = None
    # Parameters used when this selection was made (for replaying score distribution)
    selection_method_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "demo_split": self.demo_split,
            "source": self.source,
            "slices": [s.to_dict() for s in self.slices],
        }
        if self.created:
            d["created"] = self.created
        if self.selection_method_metadata:
            d["selection_method_metadata"] = self.selection_method_metadata
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Selection":
        return Selection(
            id=int(d["id"]),
            label=d.get("label", ""),
            demo_split=d.get("demo_split", "train"),
            source=d.get("source", "manual"),
            slices=[SelectionSlice.from_dict(s) for s in d.get("slices", [])],
            created=d.get("created"),
            selection_method_metadata=d.get("selection_method_metadata"),
        )


@dataclass
class SelectionConfig:
    """Config storing a list of selections (each = one add-to-config click)."""

    selections: List[Selection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selections": [s.to_dict() for s in self.selections],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SelectionConfig":
        return SelectionConfig(
            selections=[Selection.from_dict(s) for s in d.get("selections", [])],
            metadata=d.get("metadata", {}),
        )

    def get_all_slices(self) -> List[SelectionSlice]:
        """Return all slices across all selections (order preserved)."""
        out: List[SelectionSlice] = []
        for sel in self.selections:
            out.extend(sel.slices)
        return out

    def to_curation_config(
        self,
        episode_lengths: Optional[Dict[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CurationConfig":
        """Merge all selection slices into a single CurationConfig (merge overlapping)."""
        all_slices = [
            s.to_curation_slice() for s in self.get_all_slices()
        ]
        merged = _merge_overlapping_slices(all_slices)
        ep_lengths: Dict[int, int] = dict(episode_lengths or {})
        meta = dict(metadata or {})
        return CurationConfig(slices=merged, metadata=meta, episode_lengths=ep_lengths)


def _merge_overlapping_slices(slices: List[CurationSlice]) -> List[CurationSlice]:
    """Merge overlapping or adjacent slices per episode into contiguous slices.

    Slices in the same episode that overlap or are adjacent (end+1 >= next start)
    are merged into one slice. Label and source are taken from the first slice
    in each merged group.
    """
    if not slices:
        return []
    # Group by episode_idx, sort by start within group
    by_episode: Dict[int, List[CurationSlice]] = {}
    for s in slices:
        by_episode.setdefault(s.episode_idx, []).append(s)
    for ep in by_episode:
        by_episode[ep].sort(key=lambda x: (x.start, x.end))

    merged: List[CurationSlice] = []
    for episode_idx in sorted(by_episode.keys()):
        group = by_episode[episode_idx]
        cur_start = group[0].start
        cur_end = group[0].end
        cur_label = group[0].label
        cur_source = group[0].source
        for s in group[1:]:
            if s.start <= cur_end + 1:
                # Overlap or adjacent: extend current range
                cur_end = max(cur_end, s.end)
            else:
                merged.append(
                    CurationSlice(
                        episode_idx=episode_idx,
                        start=cur_start,
                        end=cur_end,
                        label=cur_label,
                        source=cur_source,
                    )
                )
                cur_start, cur_end = s.start, s.end
                cur_label, cur_source = s.label, s.source
        merged.append(
            CurationSlice(
                episode_idx=episode_idx,
                start=cur_start,
                end=cur_end,
                label=cur_label,
                source=cur_source,
            )
        )
    return merged


@dataclass
class CurationConfig:
    """A curation configuration containing demo slices to curate.

    Optionally stores selections (each = one 'add to curation config' click) in the
    same file so you can see how the merged slices were composed.
    """

    slices: List[CurationSlice] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_lengths: Dict[int, int] = field(default_factory=dict)
    # Selections (one per add click); when present, slices = merge(all selection slices)
    selections: Optional[List[Selection]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "slices": [s.to_dict() for s in self.slices],
            "metadata": self.metadata,
            "episode_lengths": {int(k): int(v) for k, v in self.episode_lengths.items()},
        }
        if self.selections is not None:
            out["selections"] = [s.to_dict() for s in self.selections]
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CurationConfig":
        slices = [CurationSlice.from_dict(s) for s in d.get("slices", [])]
        slices = _merge_overlapping_slices(slices)
        metadata = d.get("metadata", {})
        # YAML may load int keys as int already, but ensure consistency
        raw_ep_lens = d.get("episode_lengths", {})
        episode_lengths = {int(k): int(v) for k, v in raw_ep_lens.items()}
        selections = None
        if "selections" in d and d["selections"]:
            selections = [Selection.from_dict(s) for s in d["selections"]]
        return CurationConfig(
            slices=slices,
            metadata=metadata,
            episode_lengths=episode_lengths,
            selections=selections,
        )

    def add_slice(
        self,
        episode_idx: int,
        start: int,
        end: int,
        episode_length: int,
        label: str = "",
        source: str = "manual",
    ) -> bool:
        """Add a slice. Returns True if added, False if duplicate.

        Also stores/validates episode_length for safety checks.
        """
        episode_idx = int(episode_idx)
        start = int(start)
        end = int(end)
        episode_length = int(episode_length)

        # Validate against stored episode length
        if episode_idx in self.episode_lengths:
            stored_len = self.episode_lengths[episode_idx]
            if stored_len != episode_length:
                raise ValueError(
                    f"Episode {episode_idx} length mismatch: "
                    f"stored={stored_len}, provided={episode_length}. "
                    f"This likely means the dataset has changed since the config was created."
                )
        else:
            self.episode_lengths[episode_idx] = episode_length

        # Validate bounds
        if start < 0 or end < start or end >= episode_length:
            raise ValueError(
                f"Invalid slice bounds: start={start}, end={end}, "
                f"episode_length={episode_length}"
            )

        # Check for exact duplicate
        for existing in self.slices:
            if (
                existing.episode_idx == episode_idx
                and existing.start == start
                and existing.end == end
            ):
                return False

        self.slices.append(
            CurationSlice(
                episode_idx=episode_idx,
                start=start,
                end=end,
                label=label,
                source=source,
            )
        )
        self.slices = _merge_overlapping_slices(self.slices)
        return True

    def get_slices_for_episode(self, episode_idx: int) -> List[CurationSlice]:
        """Get all slices for a specific episode."""
        return [s for s in self.slices if s.episode_idx == episode_idx]

    def get_unique_episode_indices(self) -> List[int]:
        """Get sorted list of unique episode indices that have slices."""
        return sorted(set(s.episode_idx for s in self.slices))

    def to_sample_mask(
        self,
        demo_episodes: list,
        total_num_samples: int,
    ) -> "np.ndarray":
        """Convert curation slices to a boolean sample mask (RAW buffer space).

        WARNING: This method produces a mask in RAW sample space (same semantics as
        training). total_num_samples must be the total number of RAW samples (sum of
        raw episode lengths), and the returned mask has one entry per raw buffer index.
        Slice (start, end) are raw timesteps within the episode; we map to global
        raw indices using episode boundaries from demo_episodes.

        Training uses load_sample_mask_from_curation_config() in
        diffusion_policy/common/sampler.py, which works in raw buffer space. This
        method is for visualizer use only when a raw-space mask is needed.

        Args:
            demo_episodes: List of EpisodeInfo (must have .raw_length or .num_samples)
            total_num_samples: Total number of RAW demo samples (sum of episode raw lengths)

        Returns:
            Boolean array of shape (total_num_samples,) where True = keep, False = exclude.
        """
        import numpy as np

        mask = np.ones(total_num_samples, dtype=bool)
        ep_lookup = {ep.index: ep for ep in demo_episodes}

        # Build episode start indices in raw buffer space (same as training)
        cum = 0
        ep_starts = {}
        for ep in demo_episodes:
            ep_starts[ep.index] = cum
            cum += getattr(ep, "raw_length", None) or ep.num_samples

        for s in self.slices:
            if s.episode_idx not in ep_lookup:
                raise ValueError(
                    f"Curation config references episode {s.episode_idx} "
                    f"which is not in the dataset"
                )
            ep = ep_lookup[s.episode_idx]
            ep_len = getattr(ep, "raw_length", None) or ep.num_samples

            if s.episode_idx in self.episode_lengths:
                expected_len = self.episode_lengths[s.episode_idx]
                if ep_len != expected_len:
                    raise ValueError(
                        f"Episode {s.episode_idx} length mismatch: "
                        f"curation config expects {expected_len} samples (raw length), "
                        f"but dataset has {ep_len}."
                    )

            # Map raw (start, end) within episode to global raw indices
            ep_start = ep_starts[s.episode_idx]
            abs_start = ep_start + max(0, s.start)
            abs_end = ep_start + min(s.end + 1, ep_len)
            if abs_start < abs_end:
                mask[abs_start:abs_end] = False

        return mask

    def validate_against_data(self, demo_episodes: list) -> List[str]:
        """Validate this config against actual data. Returns list of warnings/errors."""
        errors = []
        ep_lookup = {ep.index: ep for ep in demo_episodes}

        for s in self.slices:
            if s.episode_idx not in ep_lookup:
                errors.append(
                    f"Slice references episode {s.episode_idx} not in dataset"
                )
                continue

            ep = ep_lookup[s.episode_idx]
            # Slices are in raw timestep space; episode_lengths store raw length
            ep_len = getattr(ep, "raw_length", None) or ep.num_samples

            # Check episode length
            if s.episode_idx in self.episode_lengths:
                if ep_len != self.episode_lengths[s.episode_idx]:
                    errors.append(
                        f"Episode {s.episode_idx}: expected {self.episode_lengths[s.episode_idx]} "
                        f"samples (raw length), got {ep_len}"
                    )

            # Check bounds (slice start/end are raw timestep indices)
            if s.start < 0:
                errors.append(
                    f"Episode {s.episode_idx}: slice start {s.start} is negative"
                )
            if s.end >= ep_len:
                errors.append(
                    f"Episode {s.episode_idx}: slice end {s.end} >= episode length {ep_len}"
                )
            if s.end < s.start:
                errors.append(
                    f"Episode {s.episode_idx}: slice end {s.end} < start {s.start}"
                )

        return errors


def get_curation_dir(task_config_name: str) -> pathlib.Path:
    """Get the curation config directory for a task config."""
    configs_dir = (
        pathlib.Path(__file__).parent / "configs" / task_config_name / "curation"
    )
    return configs_dir


def list_curation_configs(task_config_name: str) -> List[str]:
    """List available curation config files for a task config.

    Returns:
        List of config names (without .yaml extension), sorted alphabetically.
    """
    curation_dir = get_curation_dir(task_config_name)
    if not curation_dir.exists():
        return []
    return sorted([f.stem for f in curation_dir.glob("*.yaml")])


def load_curation_config(task_config_name: str, config_name: str) -> CurationConfig:
    """Load a curation config from YAML file.

    Args:
        task_config_name: Name of the task config (e.g., 'lift_mh')
        config_name: Name of the curation config (without .yaml)

    Returns:
        CurationConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = get_curation_dir(task_config_name) / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Curation config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return CurationConfig()

    return CurationConfig.from_dict(data)


def save_curation_config(
    task_config_name: str,
    config_name: str,
    config: CurationConfig,
    episode_ends: Optional[np.ndarray] = None,
) -> pathlib.Path:
    """Save a curation config to YAML file.

    Args:
        task_config_name: Name of the task config
        config_name: Name of the curation config (without .yaml)
        config: CurationConfig to save
        episode_ends: Episode end indices from the replay buffer / dataset.
            When provided, dataset_fingerprint and total_raw_samples are
            computed and stored in metadata so training can verify the config
            matches the dataset before applying the mask.

    Returns:
        Path to the saved file
    """
    curation_dir = get_curation_dir(task_config_name)
    curation_dir.mkdir(parents=True, exist_ok=True)

    config_path = curation_dir / f"{config_name}.yaml"

    # Update metadata
    config.metadata["last_modified"] = datetime.now().isoformat()
    config.metadata["task_config"] = task_config_name
    config.metadata["num_slices"] = len(config.slices)
    if episode_ends is not None:
        config.metadata["dataset_fingerprint"] = compute_dataset_fingerprint(
            episode_ends
        )
        config.metadata["total_raw_samples"] = int(episode_ends[-1]) if len(episode_ends) > 0 else 0

    with open(config_path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    return config_path


CurationMode = str  # "filter" | "selection"


def create_curation_config(
    task_config_name: str,
    config_name: str,
    split: str = "train",
    mode: str = "filter",
) -> CurationConfig:
    """Create a new empty curation config and save it.

    Args:
        task_config_name: Name of the task config
        config_name: Name for the new curation config
        split: Demo split this config applies to (used when mode="filter").
        mode: "filter" (exclude slices from train) or "selection" (include
            slices from holdout). When mode="selection", split is forced to "holdout".

    Returns:
        The newly created CurationConfig
    """
    if mode == "selection":
        split = "holdout"
    config = CurationConfig(
        metadata={
            "created": datetime.now().isoformat(),
            "task_config": task_config_name,
            "split": split,
            "curation_mode": mode,
            "num_slices": 0,
        },
        selections=[],
    )
    save_curation_config(task_config_name, config_name, config)
    return config


def validate_selection_config_holdout(
    config: CurationConfig,
    holdout_episode_indices: Set[int],
) -> List[str]:
    """Validate that all slices in a selection config reference holdout episodes.

    When curation_mode is "selection", every slice's episode_idx must be in the
    holdout set. Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []
    mode = (config.metadata or {}).get("curation_mode", "filter")
    if mode != "selection":
        return errors
    for s in config.slices:
        if s.episode_idx not in holdout_episode_indices:
            errors.append(
                f"Selection config slice references episode {s.episode_idx} "
                "which is not in the holdout set."
            )
    return errors


# ---------------------------------------------------------------------------
# Selection helpers (selections live in the same file as curation config)
# ---------------------------------------------------------------------------


def list_selection_configs(task_config_name: str) -> List[str]:
    """List config names that can have selections (same as curation configs)."""
    return list_curation_configs(task_config_name)


def load_selection_config(
    task_config_name: str,
    config_name: str,
) -> SelectionConfig:
    """Load selections from the curation config file (same file as slices)."""
    try:
        config = load_curation_config(task_config_name, config_name)
    except FileNotFoundError:
        return SelectionConfig(
            metadata={
                "task_config": task_config_name,
                "config_name": config_name,
                "num_selections": 0,
            }
        )
    return SelectionConfig(
        selections=config.selections if config.selections is not None else [],
        metadata=config.metadata,
    )


def add_selection_to_config(
    task_config_name: str,
    config_name: str,
    slices_with_rollout: List[Tuple],
    label: str,
    demo_split: str,
    source: str = "behavior_search",
    episode_lengths: Optional[Dict[int, int]] = None,
    episode_ends: Optional[np.ndarray] = None,
    selection_method_metadata: Optional[Dict[str, Any]] = None,
) -> CurationConfig:
    """Append one selection to the curation config and re-merge slices. Saves to same file.

    slices_with_rollout: list of tuples. Each tuple is either:
        (CurationSlice, rollout_episode_idx, rollout_start, rollout_end) or
        (CurationSlice, rollout_episode_idx, rollout_start, rollout_end, local_sample_idx).
    episode_lengths: optional dict to merge into config.episode_lengths for new episodes.
    episode_ends: episode end indices for dataset fingerprint (passed to save_curation_config).
    selection_method_metadata: optional dict with keys such as selection_method, window_width,
        aggregation_method (used when replaying score distribution for a segment).
    """
    config = load_curation_config(task_config_name, config_name)
    if config.selections is None:
        config.selections = []
    next_id = max((s.id for s in config.selections), default=-1) + 1
    selection_slices = []
    for t in slices_with_rollout:
        cs, ro_ep, ro_start, ro_end = t[0], t[1], t[2], t[3]
        local_idx = t[4] if len(t) > 4 else None
        selection_slices.append(
            SelectionSlice(
                episode_idx=cs.episode_idx,
                start=cs.start,
                end=cs.end,
                label=cs.label,
                source=cs.source,
                rollout_episode_idx=ro_ep,
                rollout_start=ro_start,
                rollout_end=ro_end,
                local_sample_idx=local_idx,
            )
        )
    config.selections.append(
        Selection(
            id=next_id,
            label=label,
            demo_split=demo_split,
            source=source,
            slices=selection_slices,
            created=datetime.now().isoformat(),
            selection_method_metadata=selection_method_metadata,
        )
    )
    # Recompute slices from all selections and keep in sync
    all_cs = [ss.to_curation_slice() for s in config.selections for ss in s.slices]
    config.slices = _merge_overlapping_slices(all_cs)
    if episode_lengths:
        for k, v in episode_lengths.items():
            config.episode_lengths.setdefault(k, v)
    save_curation_config(task_config_name, config_name, config, episode_ends=episode_ends)
    return config


def remove_selection_from_config(
    task_config_name: str,
    config_name: str,
    selection_id: int,
    episode_ends: Optional[np.ndarray] = None,
) -> CurationConfig:
    """Remove a selection by id and re-merge slices. Saves to same file."""
    config = load_curation_config(task_config_name, config_name)
    if config.selections is None:
        config.selections = []
    config.selections = [s for s in config.selections if s.id != selection_id]
    # Recompute slices from remaining selections
    all_cs = [ss.to_curation_slice() for s in config.selections for ss in s.slices]
    config.slices = _merge_overlapping_slices(all_cs)
    save_curation_config(task_config_name, config_name, config, episode_ends=episode_ends)
    return config


def generate_curation_config_from_selections(
    task_config_name: str,
    config_name: str,
    episode_lengths: Dict[int, int],
    metadata: Optional[Dict[str, Any]] = None,
    episode_ends: Optional[np.ndarray] = None,
) -> CurationConfig:
    """Recompute slices from selections in the same file and save (keeps selections unchanged)."""
    config = load_curation_config(task_config_name, config_name)
    if not config.selections:
        return config
    all_cs = [ss.to_curation_slice() for s in config.selections for ss in s.slices]
    config.slices = _merge_overlapping_slices(all_cs)
    if episode_lengths:
        for k, v in episode_lengths.items():
            config.episode_lengths.setdefault(k, v)
    if metadata:
        config.metadata.update(metadata)
    save_curation_config(task_config_name, config_name, config, episode_ends=episode_ends)
    return config


def selection_source_distribution(
    sel_config: SelectionConfig,
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[int, int]]:
    """Compute source (rollout) distribution from a selection config.

    Returns:
        - per_slice: (rollout_ep_idx, rollout_start, rollout_end) -> count of demo slices linked
        - per_rollout_ep: rollout_episode_idx -> total count of demo slices linked (aggregated)
    """
    per_slice: Dict[Tuple[int, int, int], int] = {}
    per_rollout_ep: Dict[int, int] = {}
    for sel in sel_config.selections:
        for s in sel.slices:
            if s.rollout_episode_idx is None:
                continue
            key = (
                s.rollout_episode_idx,
                s.rollout_start or 0,
                s.rollout_end or 0,
            )
            per_slice[key] = per_slice.get(key, 0) + 1
            per_rollout_ep[s.rollout_episode_idx] = (
                per_rollout_ep.get(s.rollout_episode_idx, 0) + 1
            )
    return per_slice, per_rollout_ep


def selection_target_distribution(
    sel_config: SelectionConfig,
) -> Dict[int, int]:
    """Compute target (demo) distribution: demo_episode_idx -> number of selected samples."""
    per_ep: Dict[int, int] = {}
    for s in sel_config.get_all_slices():
        n = s.end - s.start + 1
        per_ep[s.episode_idx] = per_ep.get(s.episode_idx, 0) + n
    return per_ep


# ---------------------------------------------------------------------------
# Attribution breakdown: per-rollout-episode and per-selection analysis
# ---------------------------------------------------------------------------

def selection_attribution_breakdown(
    sel_config: SelectionConfig,
) -> Dict[str, Any]:
    """Full breakdown of how selected demo slices attribute to rollout episodes.

    Use this to diagnose bias (e.g. one rollout episode dominating selections).

    Returns:
        dict with:
        - total_demo_slices: total slice entries across all selections
        - total_with_rollout_link: slices that have rollout_episode_idx set
        - per_rollout_ep: list of {rollout_ep, count, pct, distinct_rollout_segments}
          (distinct_rollout_segments = number of distinct (start,end) ranges from that ep)
        - per_selection_rollout: list of {selection_id, label, rollout_ep_counts}
          (rollout_ep_counts = dict rollout_ep -> count of slices from that selection)
    """
    all_slices = sel_config.get_all_slices()
    total_demo_slices = len(all_slices)
    per_slice, per_rollout_ep = selection_source_distribution(sel_config)
    total_with_rollout_link = sum(per_rollout_ep.values())

    # Distinct rollout segments per episode: count unique (start, end) per rollout_ep
    segments_per_ep: Dict[int, Set[Tuple[int, int]]] = {}
    for (ro_ep, ro_start, ro_end), _ in per_slice.items():
        segments_per_ep.setdefault(ro_ep, set()).add((ro_start or 0, ro_end or 0))
    distinct_per_ep = {ep: len(segs) for ep, segs in segments_per_ep.items()}

    total_linked = total_with_rollout_link or 1
    per_rollout_list = []
    for ro_ep in sorted(per_rollout_ep.keys()):
        count = per_rollout_ep[ro_ep]
        pct = 100.0 * count / total_linked
        per_rollout_list.append({
            "rollout_ep": ro_ep,
            "count": count,
            "pct": round(pct, 1),
            "distinct_rollout_segments": distinct_per_ep.get(ro_ep, 0),
        })

    # Per selection: how many slices from each rollout_ep did this selection add?
    # Also compute primary source per selection: the (rollout_ep, start, end) with most slices
    per_selection_rollout: List[Dict[str, Any]] = []
    per_selection_primary: List[Dict[str, Any]] = []
    for sel in sel_config.selections:
        ro_counts: Dict[int, int] = {}
        segment_counts: Dict[Tuple[int, int, int], int] = {}
        for s in sel.slices:
            if s.rollout_episode_idx is None:
                continue
            ro_ep = s.rollout_episode_idx
            ro_counts[ro_ep] = ro_counts.get(ro_ep, 0) + 1
            key = (ro_ep, s.rollout_start or 0, s.rollout_end or 0)
            segment_counts[key] = segment_counts.get(key, 0) + 1
        per_selection_rollout.append({
            "selection_id": sel.id,
            "label": sel.label,
            "rollout_ep_counts": ro_counts,
            "total_slices_with_link": sum(ro_counts.values()),
        })
        if segment_counts:
            (prim_ep, prim_start, prim_end) = max(segment_counts.keys(), key=lambda k: segment_counts[k])
            per_selection_primary.append({
                "selection_id": sel.id,
                "label": sel.label,
                "primary_rollout_ep": prim_ep,
                "primary_start": prim_start,
                "primary_end": prim_end,
                "primary_count": segment_counts[(prim_ep, prim_start, prim_end)],
                "total_slices_with_link": sum(ro_counts.values()),
            })
        else:
            per_selection_primary.append({
                "selection_id": sel.id,
                "label": sel.label,
                "primary_rollout_ep": None,
                "primary_start": None,
                "primary_end": None,
                "primary_count": 0,
                "total_slices_with_link": 0,
            })

    # Per rollout segment (every distinct rollout window): for "which points" analysis
    per_rollout_segment: List[Dict[str, Any]] = []
    for (ro_ep, ro_start, ro_end), count in per_slice.items():
        per_rollout_segment.append({
            "rollout_ep": ro_ep,
            "rollout_start": ro_start or 0,
            "rollout_end": ro_end or 0,
            "count": count,
        })
    per_rollout_segment.sort(key=lambda x: (-x["count"], x["rollout_ep"], x["rollout_start"]))

    return {
        "total_demo_slices": total_demo_slices,
        "total_with_rollout_link": total_with_rollout_link,
        "per_rollout_ep": per_rollout_list,
        "per_rollout_segment": per_rollout_segment,
        "per_selection_rollout": per_selection_rollout,
        "per_selection_primary": per_selection_primary,
    }

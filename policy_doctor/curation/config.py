"""Curation config: slices, save/load, fingerprint."""

import hashlib
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml


def compute_dataset_fingerprint(episode_ends: np.ndarray) -> str:
    arr = np.asarray(episode_ends, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass
class CurationSlice:
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


def merge_overlapping_slices(slices: List[CurationSlice]) -> List[CurationSlice]:
    """Merge overlapping or adjacent slices per episode into contiguous slices.

    Slices in the same episode that overlap or are adjacent (end+1 >= next start)
    are merged into one slice. Label and source are taken from the first slice
    in each merged group.
    """
    if not slices:
        return []
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
                cur_end = max(cur_end, s.end)
            else:
                merged.append(CurationSlice(
                    episode_idx=episode_idx,
                    start=cur_start,
                    end=cur_end,
                    label=cur_label,
                    source=cur_source,
                ))
                cur_start, cur_end = s.start, s.end
                cur_label, cur_source = s.label, s.source
        merged.append(CurationSlice(
            episode_idx=episode_idx,
            start=cur_start,
            end=cur_end,
            label=cur_label,
            source=cur_source,
        ))
    return merged


@dataclass
class CurationConfig:
    slices: List[CurationSlice] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_lengths: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slices": [s.to_dict() for s in self.slices],
            "metadata": self.metadata,
            "episode_lengths": {int(k): int(v) for k, v in self.episode_lengths.items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CurationConfig":
        slices = [CurationSlice.from_dict(s) for s in d.get("slices", [])]
        raw_ep = d.get("episode_lengths", {})
        episode_lengths = {int(k): int(v) for k, v in raw_ep.items()}
        return CurationConfig(
            slices=slices,
            metadata=d.get("metadata", {}),
            episode_lengths=episode_lengths,
        )


def get_curation_dir(task_config_name: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "configs" / task_config_name / "curation"


def list_curation_configs(task_config_name: str) -> List[str]:
    curation_dir = get_curation_dir(task_config_name)
    if not curation_dir.exists():
        return []
    return sorted(
        p.stem for p in curation_dir.iterdir()
        if p.suffix == ".yaml"
    )


def load_curation_config(
    task_config_name: str,
    config_name: str,
) -> CurationConfig:
    curation_dir = get_curation_dir(task_config_name)
    config_path = curation_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Curation config not found: {config_path}")
    return load_curation_config_from_path(config_path)


def load_curation_config_from_path(path: Union[str, pathlib.Path]) -> CurationConfig:
    """Load a curation config from an arbitrary YAML path."""
    path = pathlib.Path(path)
    with open(path) as f:
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
    curation_dir = get_curation_dir(task_config_name)
    curation_dir.mkdir(parents=True, exist_ok=True)
    config_path = curation_dir / f"{config_name}.yaml"
    config.metadata["last_modified"] = datetime.now().isoformat()
    config.metadata["task_config"] = task_config_name
    config.metadata["num_slices"] = len(config.slices)
    if episode_ends is not None:
        config.metadata["dataset_fingerprint"] = compute_dataset_fingerprint(episode_ends)
        # episode_ends: cumulative end indices (replay buffer convention); total raw samples = last element
        config.metadata["total_raw_samples"] = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
    with open(config_path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    return config_path

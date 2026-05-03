"""Rollout pool: indexed view over a set of base-policy rollouts on disk.

Reads the existing ``eval_save_episodes`` output layout — no new on-disk format
introduced. The pool is a frozen view of a chosen ``episodes_dir`` plus optional
storyboard / video sidecars. The :class:`RolloutPool` provides:

  - opaque ``rollout_id`` (deterministic from ``episode_idx``) for VLM use
  - per-rollout sim states & success outcomes for ``init_state`` reset
  - per-rollout cluster path (when a clustering directory is supplied)
  - storyboard composite image paths (if ``build_rollout_pool`` step ran)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# Number of digits in the rollout_id zero-pad. 4 is plenty for pools ≤ 9999.
_ID_PAD = 4


def episode_idx_to_rollout_id(idx: int) -> str:
    """Deterministic stable id used in DemonstrationRequest.

    Format: ``r{idx:04d}``. Opaque from the operator's perspective (no condition).
    """
    return f"r{int(idx):0{_ID_PAD}d}"


def rollout_id_to_episode_idx(rollout_id: str) -> int:
    if not rollout_id.startswith("r"):
        raise ValueError(f"rollout_id must start with 'r', got {rollout_id!r}")
    return int(rollout_id[1:])


@dataclass
class RolloutEntry:
    """One pool rollout."""

    rollout_id: str
    episode_idx: int
    episode_pkl: Path
    length: int
    success: Optional[bool]
    storyboard_path: Optional[Path] = None
    video_path: Optional[Path] = None
    cluster_path: Optional[List[int]] = None     # collapsed graph path; None until set


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class RolloutPool:
    """Frozen index over a set of eval-save-episodes rollouts."""

    episodes_dir: Path
    entries: List[RolloutEntry]

    # Optional sidecars produced by ``build_rollout_pool`` step
    storyboard_dir: Optional[Path] = None
    video_dir: Optional[Path] = None
    sim_states_dir: Optional[Path] = None

    # ----------------- access ----------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def rollout_ids(self) -> List[str]:
        return [e.rollout_id for e in self.entries]

    def by_id(self, rollout_id: str) -> RolloutEntry:
        for e in self.entries:
            if e.rollout_id == rollout_id:
                return e
        raise KeyError(f"rollout_id {rollout_id!r} not in pool")

    def successes(self) -> List[RolloutEntry]:
        return [e for e in self.entries if e.success is True]

    def failures(self) -> List[RolloutEntry]:
        return [e for e in self.entries if e.success is False]

    # ----------------- factory ---------------------------------------------

    @classmethod
    def from_episodes_dir(
        cls,
        episodes_dir: Path,
        *,
        storyboard_dir: Optional[Path] = None,
        video_dir: Optional[Path] = None,
        sim_states_dir: Optional[Path] = None,
        cluster_paths: Optional[Dict[int, List[int]]] = None,
    ) -> "RolloutPool":
        """Build a pool from a directory of ``ep{NNNN}.pkl`` files + ``metadata.yaml``.

        ``cluster_paths`` (optional): map ``episode_idx -> collapsed cluster path``.
        Compute via :class:`policy_doctor.monitoring.TrajectoryClassifier`.
        """
        episodes_dir = Path(episodes_dir)
        if not episodes_dir.is_dir():
            raise FileNotFoundError(f"episodes_dir not found: {episodes_dir}")

        # Match the format eval_save_episodes writes
        pkls = sorted(episodes_dir.glob("ep*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No ep*.pkl files in {episodes_dir}")

        meta_path = episodes_dir / "metadata.yaml"
        successes: Sequence[bool] = []
        lengths: Sequence[int] = []
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            successes = meta.get("episode_successes", []) or []
            lengths = meta.get("episode_lengths", []) or []

        entries: List[RolloutEntry] = []
        for i, pkl in enumerate(pkls):
            rid = episode_idx_to_rollout_id(i)
            entry = RolloutEntry(
                rollout_id=rid,
                episode_idx=i,
                episode_pkl=pkl,
                length=int(lengths[i]) if i < len(lengths) else -1,
                success=bool(successes[i]) if i < len(successes) else None,
                storyboard_path=(storyboard_dir / f"{rid}.png") if storyboard_dir else None,
                video_path=(video_dir / f"{rid}.mp4") if video_dir else None,
                cluster_path=(cluster_paths or {}).get(i),
            )
            entries.append(entry)

        return cls(
            episodes_dir=episodes_dir,
            entries=entries,
            storyboard_dir=storyboard_dir,
            video_dir=video_dir,
            sim_states_dir=sim_states_dir,
        )

    # ----------------- serialization (for the proposal_server's pool index)

    def to_index_dict(self) -> Dict[str, Any]:
        """Compact JSON-shaped index, suitable for :class:`flask.jsonify`."""
        return {
            "episodes_dir": str(self.episodes_dir),
            "n_rollouts": len(self.entries),
            "rollouts": [
                {
                    "rollout_id": e.rollout_id,
                    "episode_idx": e.episode_idx,
                    "length": e.length,
                    "success": e.success,
                    "has_storyboard": e.storyboard_path is not None and e.storyboard_path.exists(),
                    "has_video": e.video_path is not None and e.video_path.exists(),
                    "cluster_path": e.cluster_path,
                }
                for e in self.entries
            ],
        }

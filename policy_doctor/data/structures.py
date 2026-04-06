"""Hierarchical data structures: Trajectory, Segment, Sample; influence matrix wrappers."""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Hierarchy: Trajectory -> Segment -> Sample
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One sample: one row or column index in the influence matrix (one policy call or demo chunk)."""

    global_idx: int  # Index in the full influence matrix (row for rollout, col for demo)
    timestep: int = 0  # Timestep within the segment/trajectory
    horizon: int = 1  # Number of env timesteps this sample represents (optional)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            self.horizon = 1


@dataclass
class Segment:
    """A segment of a trajectory with a label (e.g. cluster id or behavior name). Contains samples."""

    label: str  # Cluster id as string or behavior name
    samples: List[Sample] = field(default_factory=list)
    start_global_idx: Optional[int] = None  # First sample global index in this segment
    end_global_idx: Optional[int] = None  # Last sample global index + 1

    @property
    def num_samples(self) -> int:
        return len(self.samples)


@dataclass
class Trajectory:
    """Entire trajectory (rollout or demo). Composed of segments."""

    index: int  # Episode/trajectory index
    segments: List[Segment] = field(default_factory=list)
    success: Optional[bool] = None  # For rollouts: whether the episode succeeded
    raw_length: Optional[int] = None  # Raw episode length in buffer

    @property
    def num_samples(self) -> int:
        return sum(s.num_samples for s in self.segments)


# ---------------------------------------------------------------------------
# Compatibility: SampleInfo, EpisodeInfo (for alignment with existing code)
# ---------------------------------------------------------------------------


@dataclass
class SampleInfo:
    """Information about a single sample in the influence matrix (compatibility)."""

    global_idx: int
    episode_idx: int
    timestep: int
    buffer_start_idx: int = 0
    buffer_end_idx: int = 0
    sample_start_idx: int = 0
    sample_end_idx: int = 0


@dataclass
class EpisodeInfo:
    """Information about a single episode (rollout or demonstration) (compatibility)."""

    index: int
    num_samples: int
    sample_start_idx: int
    sample_end_idx: int
    success: Optional[bool] = None
    raw_length: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "num_samples": self.num_samples,
            "sample_start_idx": self.sample_start_idx,
            "sample_end_idx": self.sample_end_idx,
            "success": self.success,
            "raw_length": self.raw_length,
        }


# ---------------------------------------------------------------------------
# Influence matrix wrappers (use a backing store interface)
# ---------------------------------------------------------------------------


class ActionInfluence:
    """Scalar influence for one (rollout sample, demo sample) pair."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __float__(self) -> float:
        return self.value


class LocalInfluenceMatrix:
    """Influence submatrix for one rollout trajectory x one demo trajectory."""

    def __init__(
        self,
        data: Union[np.ndarray, "MemmapBackingStore"],
        rollout_trajectory_idx: int,
        demo_trajectory_idx: int,
        r_start: int,
        r_end: int,
        d_start: int,
        d_end: int,
    ) -> None:
        self._data = data
        self.rollout_trajectory_idx = rollout_trajectory_idx
        self.demo_trajectory_idx = demo_trajectory_idx
        self.r_start = r_start
        self.r_end = r_end
        self.d_start = d_start
        self.d_end = d_end

    @property
    def shape(self) -> tuple:
        return (self.r_end - self.r_start, self.d_end - self.d_start)

    def _read_block(self) -> np.ndarray:
        if isinstance(self._data, np.ndarray):
            return np.asarray(
                self._data[self.r_start : self.r_end, self.d_start : self.d_end],
                dtype=np.float32,
            )
        if hasattr(self._data, "read_slice"):
            return self._data.read_slice(
                self.r_start, self.r_end, self.d_start, self.d_end
            )
        raise TypeError("Backing store must be ndarray or have read_slice")

    def get_slice(
        self,
        rollout_sample_lo: int,
        rollout_sample_hi: int,
        demo_sample_lo: int,
        demo_sample_hi: int,
    ) -> np.ndarray:
        """Return sub-block [rollout_sample_lo:rollout_sample_hi, demo_sample_lo:demo_sample_hi] in local coords."""
        r0 = self.r_start + rollout_sample_lo
        r1 = self.r_start + rollout_sample_hi
        d0 = self.d_start + demo_sample_lo
        d1 = self.d_start + demo_sample_hi
        if isinstance(self._data, np.ndarray):
            return np.asarray(self._data[r0:r1, d0:d1], dtype=np.float32)
        if hasattr(self._data, "read_slice"):
            return self._data.read_slice(r0, r1, d0, d1)
        raise TypeError("Backing store must be ndarray or have read_slice")

    def get_action_influence(
        self, rollout_sample_idx: int, demo_sample_idx: int
    ) -> ActionInfluence:
        """Scalar influence for (rollout_sample_idx, demo_sample_idx) in local coords."""
        r = self.r_start + rollout_sample_idx
        d = self.d_start + demo_sample_idx
        if isinstance(self._data, np.ndarray):
            return ActionInfluence(float(self._data[r, d]))
        if hasattr(self._data, "read_cell"):
            return ActionInfluence(float(self._data.read_cell(r, d)))
        raise TypeError("Backing store must be ndarray or have read_cell")

    def aggregate(
        self,
        axis: Optional[int] = None,
        agg_fn: str = "sum",
    ) -> np.ndarray:
        """Aggregate over axis (0=rollout, 1=demo). agg_fn in ('sum', 'mean')."""
        block = self._read_block()
        if agg_fn == "sum":
            fn = np.nansum
        elif agg_fn == "mean":
            fn = np.nanmean
        else:
            raise ValueError(f"agg_fn must be 'sum' or 'mean', got {agg_fn!r}")
        if axis is None:
            return np.array(fn(block), dtype=np.float32)
        return np.asarray(fn(block, axis=axis), dtype=np.float32)


class GlobalInfluenceMatrix:
    """Wrapper around the full influence matrix (on-disk or in-memory). Shape (num_rollout_samples, num_demo_samples)."""

    def __init__(
        self,
        store: Union[np.ndarray, "MemmapBackingStore"],
        rollout_episodes: List[EpisodeInfo],
        demo_episodes: List[EpisodeInfo],
    ) -> None:
        self._store = store
        self.rollout_episodes = rollout_episodes
        self.demo_episodes = demo_episodes
        if isinstance(store, np.ndarray):
            self._shape = store.shape
        else:
            self._shape = getattr(store, "shape", (0, 0))

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def num_rollout_samples(self) -> int:
        return self._shape[0]

    @property
    def num_demo_samples(self) -> int:
        return self._shape[1]

    def get_slice(
        self,
        r_lo: int,
        r_hi: int,
        d_lo: int,
        d_hi: int,
    ) -> np.ndarray:
        """Return block [r_lo:r_hi, d_lo:d_hi]."""
        if isinstance(self._store, np.ndarray):
            return np.asarray(
                self._store[r_lo:r_hi, d_lo:d_hi], dtype=np.float32
            )
        if hasattr(self._store, "read_slice"):
            return self._store.read_slice(r_lo, r_hi, d_lo, d_hi)
        raise TypeError("Store must be ndarray or have read_slice")

    def get_local_matrix(
        self, rollout_trajectory_idx: int, demo_trajectory_idx: int
    ) -> LocalInfluenceMatrix:
        """Return LocalInfluenceMatrix for the given trajectory pair."""
        r_ep = self.rollout_episodes[rollout_trajectory_idx]
        d_ep = self.demo_episodes[demo_trajectory_idx]
        return LocalInfluenceMatrix(
            self._store,
            rollout_trajectory_idx,
            demo_trajectory_idx,
            r_ep.sample_start_idx,
            r_ep.sample_end_idx,
            d_ep.sample_start_idx,
            d_ep.sample_end_idx,
        )

    def get_action_influence(
        self, rollout_sample_idx: int, demo_sample_idx: int
    ) -> ActionInfluence:
        """Scalar influence for global indices."""
        if isinstance(self._store, np.ndarray):
            return ActionInfluence(
                float(self._store[rollout_sample_idx, demo_sample_idx])
            )
        if hasattr(self._store, "read_cell"):
            return ActionInfluence(
                float(self._store.read_cell(rollout_sample_idx, demo_sample_idx))
            )
        raise TypeError("Store must be ndarray or have read_cell")

    def aggregate(
        self,
        axis: Optional[int] = None,
        agg_fn: str = "sum",
        window_rollout: Optional[tuple] = None,
        window_demo: Optional[tuple] = None,
    ) -> np.ndarray:
        """Aggregate over axis (0=rollout, 1=demo), optionally over a window. Reads in chunks to avoid full load."""
        r_lo, r_hi = window_rollout or (0, self._shape[0])
        d_lo, d_hi = window_demo or (0, self._shape[1])
        block = self.get_slice(r_lo, r_hi, d_lo, d_hi)
        if agg_fn == "sum":
            fn = np.nansum
        elif agg_fn == "mean":
            fn = np.nanmean
        else:
            raise ValueError(f"agg_fn must be 'sum' or 'mean', got {agg_fn!r}")
        if axis is None:
            return np.array(fn(block), dtype=np.float32)
        return np.asarray(fn(block, axis=axis), dtype=np.float32)

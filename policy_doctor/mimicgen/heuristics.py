"""Trajectory selection heuristics for MimicGen seed selection.

The :class:`TrajectorySelectionHeuristic` abstract base defines the interface
for selecting a seed rollout trajectory to use as MimicGen input.  Concrete
implementations differ in *how* they pick among available rollouts:

* :class:`BehaviorGraphPathHeuristic` — ranks paths to SUCCESS in the behavior
  graph by probability and returns the first rollout whose collapsed cluster
  sequence exactly matches the highest-probability path.  This is the
  *proposed method*: informed seed selection should yield higher-quality
  generated data and thereby better retrained policies.

* :class:`RandomSelectionHeuristic` — picks a rollout uniformly at random
  from the eligible pool (successful by default).  Used as the *baseline* to
  isolate the effect of informed seed selection from other pipeline factors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from policy_doctor.mimicgen.seed_trajectory import MimicGenSeedTrajectory


@dataclass
class SeedSelectionResult:
    """Result returned by any :class:`TrajectorySelectionHeuristic`.

    Attributes:
        trajectory:  The selected seed trajectory ready for MimicGen.
        rollout_idx: Episode index in the source rollout HDF5.
        info:        Heuristic-specific metadata (path info, probability, etc.).
    """

    trajectory: MimicGenSeedTrajectory
    rollout_idx: int
    info: dict[str, Any] = field(default_factory=dict)


class TrajectorySelectionHeuristic(ABC):
    """Abstract base for MimicGen seed-trajectory selection strategies."""

    @abstractmethod
    def select_multiple(
        self,
        n: int,
        cluster_labels: np.ndarray,
        metadata: list[dict[str, Any]],
        rollout_hdf5_path: str,
        level: str = "rollout",
    ) -> list[SeedSelectionResult]:
        """Select up to *n* distinct seed trajectories from the available rollouts.

        Implementations should return as many results as possible (up to *n*),
        and warn if fewer than *n* distinct candidates are available rather than
        raising an error.

        Args:
            n:                Number of seed trajectories to select.
            cluster_labels:   ``(N,)`` int array from clustering.
            metadata:         Per-sample metadata dicts (same order as labels).
            rollout_hdf5_path: Path to the HDF5 file containing rollout data.
            level:            ``"rollout"`` or ``"demo"``.

        Returns:
            List of :class:`SeedSelectionResult` (length ≤ *n*).
        """

    def select(
        self,
        cluster_labels: np.ndarray,
        metadata: list[dict[str, Any]],
        rollout_hdf5_path: str,
        level: str = "rollout",
    ) -> SeedSelectionResult:
        """Convenience wrapper: select exactly one seed trajectory."""
        results = self.select_multiple(1, cluster_labels, metadata, rollout_hdf5_path, level)
        return results[0]


class BehaviorGraphPathHeuristic(TrajectorySelectionHeuristic):
    """Select the rollout that best follows the highest-probability success path.

    Builds a :class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph`
    from the clustering result, ranks paths to the SUCCESS node by probability,
    then returns the first rollout whose collapsed cluster-label sequence exactly
    matches the top-ranked path.  If no rollout matches the top path, the next
    path is tried, down to *top_k_paths*.

    This is the *proposed method* for informed MimicGen seed selection.

    Args:
        top_k_paths:         Number of candidate paths to try (default 5).
        min_path_probability: Minimum path probability to consider (default 0.0).
        success_only:        Only consider successful rollouts (default True).
    """

    def __init__(
        self,
        top_k_paths: int = 5,
        min_path_probability: float = 0.0,
        success_only: bool = True,
    ) -> None:
        self.top_k_paths = top_k_paths
        self.min_path_probability = min_path_probability
        self.success_only = success_only

    def select_multiple(
        self,
        n: int,
        cluster_labels: np.ndarray,
        metadata: list[dict[str, Any]],
        rollout_hdf5_path: str,
        level: str = "rollout",
    ) -> list[SeedSelectionResult]:
        from policy_doctor.behaviors.behavior_graph import BehaviorGraph, SUCCESS_NODE_ID
        from policy_doctor.mimicgen.graph_seed import top_paths_with_rollouts

        graph = BehaviorGraph.from_cluster_assignments(cluster_labels, metadata, level=level)

        if SUCCESS_NODE_ID not in graph.nodes:
            raise RuntimeError(
                "Behavior graph has no SUCCESS node — all rollouts may have failed.  "
                "Cannot select a success path for MimicGen generation."
            )

        ranked = top_paths_with_rollouts(
            graph,
            cluster_labels,
            metadata,
            top_k=self.top_k_paths,
            min_path_probability=self.min_path_probability,
            success_only=self.success_only,
            level=level,
        )

        if not ranked:
            raise RuntimeError(
                f"BehaviorGraphPathHeuristic: no paths to SUCCESS found in graph "
                f"with {graph.num_episodes} episodes and {len(graph.nodes)} nodes."
            )

        top_paths_info = [
            {
                "path": e["path"],
                "path_prob": e["path_prob"],
                "cluster_seq": e["cluster_seq"],
                "rollout_idxs": e["rollout_idxs"],
                "has_match": e["has_match"],
            }
            for e in ranked
        ]

        results: list[SeedSelectionResult] = []
        seen: set[int] = set()

        # Drain paths in probability order; within each path take rollouts in order.
        for entry in ranked:
            if not entry["has_match"]:
                continue
            for rollout_idx in entry["rollout_idxs"]:
                if rollout_idx in seen:
                    continue
                traj = MimicGenSeedTrajectory.from_rollout_hdf5(
                    rollout_hdf5_path, demo_key=f"demo_{rollout_idx}"
                )
                results.append(
                    SeedSelectionResult(
                        trajectory=traj,
                        rollout_idx=rollout_idx,
                        info={
                            "heuristic": "behavior_graph_path",
                            "selected_path": entry["path"],
                            "selected_path_prob": entry["path_prob"],
                            "selected_cluster_seq": entry["cluster_seq"],
                            "all_eligible_rollouts": entry["rollout_idxs"],
                            "top_paths": top_paths_info,
                        },
                    )
                )
                seen.add(rollout_idx)
                if len(results) == n:
                    return results

        if not results:
            raise RuntimeError(
                f"BehaviorGraphPathHeuristic: no rollout matched any of the top-{self.top_k_paths} "
                f"paths to SUCCESS.\n"
                f"  success_only={self.success_only}  top_k_paths={self.top_k_paths}\n"
                "  Possible causes: too few rollouts, too many clusters, or success_only=True "
                "with no success rollout matching a top path.  Try increasing top_k_paths."
            )

        if len(results) < n:
            print(
                f"  [BehaviorGraphPathHeuristic] WARNING: requested {n} seeds but only "
                f"{len(results)} distinct rollouts matched top-{self.top_k_paths} paths."
            )
        return results


class RandomSelectionHeuristic(TrajectorySelectionHeuristic):
    """Randomly select a successful rollout as the MimicGen seed.

    This is the *baseline* method.  By comparing policies trained on MimicGen
    data generated from random vs. graph-informed seeds, we can isolate the
    benefit of informed seed selection.

    Args:
        random_seed:  RNG seed for reproducibility (default None = non-deterministic).
        success_only: Only draw from successful rollouts (default True).
    """

    def __init__(
        self,
        random_seed: int | None = None,
        success_only: bool = True,
    ) -> None:
        self.rng = np.random.default_rng(random_seed)
        self.success_only = success_only

    def select_multiple(
        self,
        n: int,
        cluster_labels: np.ndarray,
        metadata: list[dict[str, Any]],
        rollout_hdf5_path: str,
        level: str = "rollout",
    ) -> list[SeedSelectionResult]:
        from policy_doctor.mimicgen.graph_seed import episode_success_map

        success_map = episode_success_map(metadata, level=level)

        if self.success_only:
            eligible = [idx for idx, success in success_map.items() if success]
        else:
            eligible = list(success_map.keys())

        if not eligible:
            raise RuntimeError(
                f"RandomSelectionHeuristic: no eligible rollouts found "
                f"(success_only={self.success_only}, total episodes={len(success_map)})."
            )

        actual_n = min(n, len(eligible))
        if actual_n < n:
            print(
                f"  [RandomSelectionHeuristic] WARNING: requested {n} seeds but only "
                f"{actual_n} eligible rollouts available."
            )

        chosen = self.rng.choice(eligible, size=actual_n, replace=False)
        results: list[SeedSelectionResult] = []
        for rollout_idx in chosen:
            rollout_idx = int(rollout_idx)
            traj = MimicGenSeedTrajectory.from_rollout_hdf5(
                rollout_hdf5_path, demo_key=f"demo_{rollout_idx}"
            )
            results.append(
                SeedSelectionResult(
                    trajectory=traj,
                    rollout_idx=rollout_idx,
                    info={
                        "heuristic": "random",
                        "eligible_count": len(eligible),
                        "success_only": self.success_only,
                    },
                )
            )
        return results


def build_heuristic(
    heuristic_name: str,
    top_k_paths: int = 5,
    min_path_probability: float = 0.0,
    success_only: bool = True,
    random_seed: int | None = None,
) -> TrajectorySelectionHeuristic:
    """Factory: build a heuristic by name.

    Args:
        heuristic_name: ``"behavior_graph"`` or ``"random"``.
        top_k_paths:    For :class:`BehaviorGraphPathHeuristic`.
        min_path_probability: For :class:`BehaviorGraphPathHeuristic`.
        success_only:   For both heuristics.
        random_seed:    For :class:`RandomSelectionHeuristic`.

    Returns:
        A concrete :class:`TrajectorySelectionHeuristic` instance.

    Raises:
        ValueError: If *heuristic_name* is not recognised.
    """
    if heuristic_name == "behavior_graph":
        return BehaviorGraphPathHeuristic(
            top_k_paths=top_k_paths,
            min_path_probability=min_path_probability,
            success_only=success_only,
        )
    if heuristic_name == "random":
        return RandomSelectionHeuristic(
            random_seed=random_seed,
            success_only=success_only,
        )
    raise ValueError(
        f"Unknown heuristic: {heuristic_name!r}.  "
        "Valid options: 'behavior_graph', 'random'."
    )

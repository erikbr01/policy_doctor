"""Trajectory selection heuristics for MimicGen seed selection.

The :class:`TrajectorySelectionHeuristic` abstract base defines the interface
for selecting a seed rollout trajectory to use as MimicGen input.  Concrete
implementations differ in *how* they pick among available rollouts:

* :class:`BehaviorGraphPathHeuristic` — ranks paths to SUCCESS in the behavior
  graph by probability and returns the first rollout whose collapsed cluster
  sequence exactly matches the highest-probability path.  This is the
  *proposed method*: informed seed selection should yield higher-quality
  generated data and thereby better retrained policies.

* :class:`DiversitySelectionHeuristic` — ranks paths to SUCCESS by probability
  but takes exactly **one** rollout per path before moving to the next.  This
  maximises behavioral diversity: every seed comes from a different execution
  strategy.  Contrast with :class:`BehaviorGraphPathHeuristic` which exhausts
  the top path before moving to lower-probability ones.

* :class:`NearFailurePathHeuristic` — ranks success paths by the maximum
  probability of eventually reaching FAILURE from any node on the path (solved
  via Bellman equations).  Selects seeds that succeeded but traversed risky
  territory.

* :class:`PathLikelihoodHeuristic` — samples rollouts with probability
  proportional to their path's likelihood, distributing seeds across paths
  according to how often each path occurs in the graph.

* :class:`ReversePathLikelihoodHeuristic` — prefers the rarest successful
  paths, exhausting the least-likely paths before moving to more common ones.

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
        random_seed:         If set, shuffle eligible rollouts within each path before
                             selecting, so that different seeds pick different rollouts
                             from the same path.  Enables reproducible variance trials.
    """

    def __init__(
        self,
        top_k_paths: int = 5,
        min_path_probability: float = 0.0,
        success_only: bool = True,
        random_seed: int | None = None,
    ) -> None:
        self.top_k_paths = top_k_paths
        self.min_path_probability = min_path_probability
        self.success_only = success_only
        self.rng = np.random.default_rng(random_seed) if random_seed is not None else None

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

        # Read rollouts.hdf5 success flags — more reliable than metadata for
        # state consistency (the states we load come from rollouts.hdf5).
        import h5py as _h5py
        hdf5_success: dict[int, bool] = {}
        try:
            with _h5py.File(rollout_hdf5_path, "r") as _f:
                for _k in _f["data"].keys():
                    _idx = int(_k.split("_")[1])
                    hdf5_success[_idx] = bool(_f["data"][_k].attrs.get("success", False))
        except Exception:
            pass

        results: list[SeedSelectionResult] = []
        seen: set[int] = set()

        # Drain paths in probability order; within each path take rollouts in order
        # (or shuffled order if random_seed was set).
        for entry in ranked:
            if not entry["has_match"]:
                continue
            idxs = list(entry["rollout_idxs"])
            if self.rng is not None:
                self.rng.shuffle(idxs)
            for rollout_idx in idxs:
                # Skip if rollouts.hdf5 says this episode is a failure
                if hdf5_success and not hdf5_success.get(rollout_idx, False):
                    continue
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


class DiversitySelectionHeuristic(TrajectorySelectionHeuristic):
    """Select seeds to maximise behavioral diversity — one rollout per distinct path.

    Ranks paths to SUCCESS by probability (same as
    :class:`BehaviorGraphPathHeuristic`) but takes exactly **one** rollout per
    path before advancing to the next.  With *n* seeds and *n* or more distinct
    paths this guarantees every seed represents a different execution strategy.

    This directly tests the hypothesis that what matters is coverage of distinct
    behavioral modes rather than concentration on the highest-probability mode.

    Args:
        top_k_paths:          Maximum number of candidate paths to consider.
        min_path_probability: Minimum path probability to include (default 0.0).
        success_only:         Only consider successful rollouts (default True).
        random_seed:          If set, shuffle eligible rollouts within each path
                              before picking, so replicates draw different
                              individuals while preserving the per-path constraint.
    """

    def __init__(
        self,
        top_k_paths: int = 20,
        min_path_probability: float = 0.0,
        success_only: bool = True,
        random_seed: int | None = None,
    ) -> None:
        self.top_k_paths = top_k_paths
        self.min_path_probability = min_path_probability
        self.success_only = success_only
        self.rng = np.random.default_rng(random_seed) if random_seed is not None else None

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
                "Behavior graph has no SUCCESS node — all rollouts may have failed."
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
                "DiversitySelectionHeuristic: no paths to SUCCESS found in graph."
            )

        import h5py as _h5py
        hdf5_success: dict[int, bool] = {}
        try:
            with _h5py.File(rollout_hdf5_path, "r") as _f:
                for _k in _f["data"].keys():
                    _idx = int(_k.split("_")[1])
                    hdf5_success[_idx] = bool(_f["data"][_k].attrs.get("success", False))
        except Exception:
            pass

        top_paths_info = [
            {
                "path": e["path"],
                "path_prob": e["path_prob"],
                "cluster_seq": e["cluster_seq"],
                "rollout_idxs": e["rollout_idxs"],
            }
            for e in ranked
        ]

        results: list[SeedSelectionResult] = []
        seen: set[int] = set()

        # One pass: take exactly one rollout per path (in probability order).
        # If we haven't reached n after exhausting all paths, do a second pass
        # to backfill from paths that had more than one eligible rollout.
        for pass_num in range(2):
            for entry in ranked:
                if not entry["has_match"]:
                    continue
                idxs = list(entry["rollout_idxs"])
                if self.rng is not None:
                    self.rng.shuffle(idxs)
                # First pass: take only 1 per path; second pass: take remaining.
                limit = 1 if pass_num == 0 else len(idxs)
                taken_this_path = 0
                for rollout_idx in idxs:
                    if hdf5_success and not hdf5_success.get(rollout_idx, False):
                        continue
                    if rollout_idx in seen:
                        continue
                    if taken_this_path >= limit:
                        break
                    traj = MimicGenSeedTrajectory.from_rollout_hdf5(
                        rollout_hdf5_path, demo_key=f"demo_{rollout_idx}"
                    )
                    results.append(
                        SeedSelectionResult(
                            trajectory=traj,
                            rollout_idx=rollout_idx,
                            info={
                                "heuristic": "diversity",
                                "selected_path": entry["path"],
                                "selected_path_prob": entry["path_prob"],
                                "selected_cluster_seq": entry["cluster_seq"],
                                "pass": pass_num,
                                "top_paths": top_paths_info,
                            },
                        )
                    )
                    seen.add(rollout_idx)
                    taken_this_path += 1
                    if len(results) == n:
                        return results

        if not results:
            raise RuntimeError(
                "DiversitySelectionHeuristic: no rollout matched any path to SUCCESS."
            )
        if len(results) < n:
            print(
                f"  [DiversitySelectionHeuristic] WARNING: requested {n} seeds but only "
                f"{len(results)} distinct rollouts available across top-{self.top_k_paths} paths."
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
        import h5py as _h5py
        from policy_doctor.mimicgen.graph_seed import episode_success_map

        # Prefer rollouts.hdf5 success flags over metadata: the states loaded
        # come from rollouts.hdf5, so success must be consistent with it.
        hdf5_success: dict[int, bool] = {}
        try:
            with _h5py.File(rollout_hdf5_path, "r") as _f:
                for _k in _f["data"].keys():
                    _idx = int(_k.split("_")[1])
                    hdf5_success[_idx] = bool(_f["data"][_k].attrs.get("success", False))
        except Exception:
            pass

        if hdf5_success:
            if self.success_only:
                eligible = [idx for idx, success in hdf5_success.items() if success]
            else:
                eligible = list(hdf5_success.keys())
        else:
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


class NearFailurePathHeuristic(TrajectorySelectionHeuristic):
    """Select seeds from successful paths that traverse nodes with high failure probability.

    Scores each success path by the maximum probability of eventually reaching
    FAILURE from any node on the path, computed by solving the Bellman equations
    on the behavior graph (gamma=1.0, reward_failure=1, all others 0).  Paths
    scoring highest succeeded but passed through risky territory.

    Args:
        top_k_paths:          Number of success paths to enumerate.
        failure_weight:       ``"max"`` (default) — score a path by the highest
                              per-node failure probability on it.  ``"sum"``
                              accumulates risk across all nodes on the path.
        min_path_probability: Minimum path probability to consider (default 0.0).
        success_only:         Only consider successful rollouts (default True).
        random_seed:          If set, shuffle eligible rollouts within each path
                              before selecting, enabling reproducible variance trials.
    """

    def __init__(
        self,
        top_k_paths: int = 30,
        failure_weight: str = "max",
        min_path_probability: float = 0.0,
        success_only: bool = True,
        random_seed: int | None = None,
    ) -> None:
        if failure_weight not in ("max", "sum"):
            raise ValueError(f"failure_weight must be 'max' or 'sum', got {failure_weight!r}")
        self.top_k_paths = top_k_paths
        self.failure_weight = failure_weight
        self.min_path_probability = min_path_probability
        self.success_only = success_only
        self.rng = np.random.default_rng(random_seed) if random_seed is not None else None

    def _score_path(
        self,
        path: list[int],
        failure_probs: dict[int, float],
        special_ids: frozenset[int],
    ) -> float:
        body = [n for n in path if n not in special_ids]
        risks = [failure_probs.get(n, 0.0) for n in body]
        if not risks:
            return 0.0
        return max(risks) if self.failure_weight == "max" else sum(risks)

    def select_multiple(
        self,
        n: int,
        cluster_labels: np.ndarray,
        metadata: list[dict[str, Any]],
        rollout_hdf5_path: str,
        level: str = "rollout",
    ) -> list[SeedSelectionResult]:
        from policy_doctor.behaviors.behavior_graph import (
            BehaviorGraph,
            FAILURE_NODE_ID,
            SUCCESS_NODE_ID,
        )
        from policy_doctor.mimicgen.graph_seed import top_paths_with_rollouts

        graph = BehaviorGraph.from_cluster_assignments(cluster_labels, metadata, level=level)

        if SUCCESS_NODE_ID not in graph.nodes:
            raise RuntimeError(
                "Behavior graph has no SUCCESS node — all rollouts may have failed."
            )

        # P(eventually reach FAILURE | start at node n), solved via Bellman.
        failure_probs = graph.compute_values(
            gamma=1.0,
            reward_success=0.0,
            reward_failure=1.0,
            reward_end=0.0,
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
                "NearFailurePathHeuristic: no paths to SUCCESS found in graph."
            )

        from policy_doctor.behaviors.behavior_graph import START_NODE_ID, TERMINAL_NODE_IDS
        _special = frozenset({START_NODE_ID}) | TERMINAL_NODE_IDS

        ranked_by_risk = sorted(
            ranked,
            key=lambda e: self._score_path(e["path"], failure_probs, _special),
            reverse=True,
        )

        top_paths_info = [
            {
                "path": e["path"],
                "path_prob": e["path_prob"],
                "cluster_seq": e["cluster_seq"],
                "rollout_idxs": e["rollout_idxs"],
                "failure_score": self._score_path(e["path"], failure_probs, _special),
            }
            for e in ranked_by_risk
        ]

        import h5py as _h5py
        hdf5_success: dict[int, bool] = {}
        try:
            with _h5py.File(rollout_hdf5_path, "r") as _f:
                for _k in _f["data"].keys():
                    _idx = int(_k.split("_")[1])
                    hdf5_success[_idx] = bool(_f["data"][_k].attrs.get("success", False))
        except Exception:
            pass

        results: list[SeedSelectionResult] = []
        seen: set[int] = set()

        for entry in ranked_by_risk:
            if not entry["has_match"]:
                continue
            failure_score = self._score_path(entry["path"], failure_probs, _special)
            idxs = list(entry["rollout_idxs"])
            if self.rng is not None:
                self.rng.shuffle(idxs)
            for rollout_idx in idxs:
                if hdf5_success and not hdf5_success.get(rollout_idx, False):
                    continue
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
                            "heuristic": "near_failure",
                            "selected_path": entry["path"],
                            "selected_path_prob": entry["path_prob"],
                            "selected_cluster_seq": entry["cluster_seq"],
                            "selected_failure_score": failure_score,
                            "failure_weight": self.failure_weight,
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
                f"NearFailurePathHeuristic: no rollout matched any of the top-{self.top_k_paths} "
                "paths to SUCCESS.  Try increasing top_k_paths."
            )
        if len(results) < n:
            print(
                f"  [NearFailurePathHeuristic] WARNING: requested {n} seeds but only "
                f"{len(results)} distinct rollouts matched top-{self.top_k_paths} paths."
            )
        return results


class PathLikelihoodHeuristic(TrajectorySelectionHeuristic):
    """Sample seed rollouts with probability proportional to their path's likelihood.

    Each eligible rollout is weighted by the probability of its success path
    (product of transition probabilities along the path).  Rollouts are sampled
    without replacement from this weighted distribution, so common paths yield
    more seeds proportionally while rare paths still receive some representation.

    Unlike :class:`BehaviorGraphPathHeuristic`, which concentrates all seeds on
    the highest-probability path, this distributes seeds across paths in
    proportion to how often each behavioral strategy is observed.

    Args:
        top_k_paths:          Number of success paths to enumerate (default 50).
                              Larger values give a better spread of weights.
        min_path_probability: Minimum path probability to include (default 0.0).
        success_only:         Only consider successful rollouts (default True).
        random_seed:          Required — controls the weighted sampling.  Raises
                              ``ValueError`` if not set.
    """

    def __init__(
        self,
        top_k_paths: int = 50,
        min_path_probability: float = 0.0,
        success_only: bool = True,
        random_seed: int | None = None,
    ) -> None:
        if random_seed is None:
            raise ValueError(
                "PathLikelihoodHeuristic requires random_seed to be set — "
                "weighted sampling is non-deterministic without it."
            )
        self.top_k_paths = top_k_paths
        self.min_path_probability = min_path_probability
        self.success_only = success_only
        self.rng = np.random.default_rng(random_seed)

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
                "Behavior graph has no SUCCESS node — all rollouts may have failed."
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
                "PathLikelihoodHeuristic: no paths to SUCCESS found in graph."
            )

        import h5py as _h5py
        hdf5_success: dict[int, bool] = {}
        try:
            with _h5py.File(rollout_hdf5_path, "r") as _f:
                for _k in _f["data"].keys():
                    _idx = int(_k.split("_")[1])
                    hdf5_success[_idx] = bool(_f["data"][_k].attrs.get("success", False))
        except Exception:
            pass

        # Build flat pool: (rollout_idx, path_prob, path_entry)
        pool_idxs: list[int] = []
        pool_weights: list[float] = []
        pool_entries: list[dict[str, Any]] = []
        seen_rollout: set[int] = set()

        for entry in ranked:
            if not entry["has_match"]:
                continue
            for rollout_idx in entry["rollout_idxs"]:
                if hdf5_success and not hdf5_success.get(rollout_idx, False):
                    continue
                if rollout_idx in seen_rollout:
                    continue
                pool_idxs.append(rollout_idx)
                pool_weights.append(entry["path_prob"])
                pool_entries.append(entry)
                seen_rollout.add(rollout_idx)

        if not pool_idxs:
            raise RuntimeError(
                f"PathLikelihoodHeuristic: no eligible rollouts found across "
                f"top-{self.top_k_paths} paths to SUCCESS."
            )

        weights = np.array(pool_weights, dtype=float)
        weights /= weights.sum()

        actual_n = min(n, len(pool_idxs))
        if actual_n < n:
            print(
                f"  [PathLikelihoodHeuristic] WARNING: requested {n} seeds but only "
                f"{actual_n} eligible rollouts available across top-{self.top_k_paths} paths."
            )

        chosen_positions = self.rng.choice(len(pool_idxs), size=actual_n, replace=False, p=weights)

        top_paths_info = [
            {
                "path": e["path"],
                "path_prob": e["path_prob"],
                "cluster_seq": e["cluster_seq"],
                "rollout_idxs": e["rollout_idxs"],
            }
            for e in ranked
        ]

        results: list[SeedSelectionResult] = []
        for pos in chosen_positions:
            rollout_idx = pool_idxs[pos]
            entry = pool_entries[pos]
            traj = MimicGenSeedTrajectory.from_rollout_hdf5(
                rollout_hdf5_path, demo_key=f"demo_{rollout_idx}"
            )
            results.append(
                SeedSelectionResult(
                    trajectory=traj,
                    rollout_idx=rollout_idx,
                    info={
                        "heuristic": "path_likelihood",
                        "selected_path": entry["path"],
                        "selected_path_prob": entry["path_prob"],
                        "selected_cluster_seq": entry["cluster_seq"],
                        "sampling_weight": pool_weights[pos],
                        "top_paths": top_paths_info,
                    },
                )
            )
        return results


class ReversePathLikelihoodHeuristic(TrajectorySelectionHeuristic):
    """Select seeds from the rarest successful paths first.

    The inverse of :class:`BehaviorGraphPathHeuristic`: enumerates success
    paths and exhausts the least-likely ones before moving to more common ones.
    Seeds drawn from rare paths may encode edge-case behaviors not well
    represented by the dominant execution strategy.

    Args:
        top_k_paths:          Number of success paths to enumerate (default 50).
                              Larger values are needed so that rare paths are
                              actually fetched by the enumerator.
        min_path_probability: Minimum path probability to include (default 0.0).
        success_only:         Only consider successful rollouts (default True).
        random_seed:          If set, shuffle eligible rollouts within each path
                              before selecting, enabling reproducible variance trials.
    """

    def __init__(
        self,
        top_k_paths: int = 50,
        min_path_probability: float = 0.0,
        success_only: bool = True,
        random_seed: int | None = None,
    ) -> None:
        self.top_k_paths = top_k_paths
        self.min_path_probability = min_path_probability
        self.success_only = success_only
        self.rng = np.random.default_rng(random_seed) if random_seed is not None else None

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
                "Behavior graph has no SUCCESS node — all rollouts may have failed."
            )

        # enumerate_paths returns paths in descending probability order; reverse for rarest-first.
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
                "ReversePathLikelihoodHeuristic: no paths to SUCCESS found in graph."
            )

        ranked_asc = list(reversed(ranked))

        top_paths_info = [
            {
                "path": e["path"],
                "path_prob": e["path_prob"],
                "cluster_seq": e["cluster_seq"],
                "rollout_idxs": e["rollout_idxs"],
            }
            for e in ranked_asc
        ]

        import h5py as _h5py
        hdf5_success: dict[int, bool] = {}
        try:
            with _h5py.File(rollout_hdf5_path, "r") as _f:
                for _k in _f["data"].keys():
                    _idx = int(_k.split("_")[1])
                    hdf5_success[_idx] = bool(_f["data"][_k].attrs.get("success", False))
        except Exception:
            pass

        results: list[SeedSelectionResult] = []
        seen: set[int] = set()

        for entry in ranked_asc:
            if not entry["has_match"]:
                continue
            idxs = list(entry["rollout_idxs"])
            if self.rng is not None:
                self.rng.shuffle(idxs)
            for rollout_idx in idxs:
                if hdf5_success and not hdf5_success.get(rollout_idx, False):
                    continue
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
                            "heuristic": "reverse_path_likelihood",
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
                f"ReversePathLikelihoodHeuristic: no rollout matched any of the top-{self.top_k_paths} "
                "paths to SUCCESS.  Try increasing top_k_paths."
            )
        if len(results) < n:
            print(
                f"  [ReversePathLikelihoodHeuristic] WARNING: requested {n} seeds but only "
                f"{len(results)} distinct rollouts matched top-{self.top_k_paths} paths."
            )
        return results


def build_heuristic(
    heuristic_name: str,
    top_k_paths: int = 5,
    min_path_probability: float = 0.0,
    success_only: bool = True,
    random_seed: int | None = None,
    failure_weight: str = "max",
) -> TrajectorySelectionHeuristic:
    """Factory: build a heuristic by name.

    Args:
        heuristic_name: One of ``"behavior_graph"``, ``"diversity"``,
                        ``"near_failure"``, ``"path_likelihood"``,
                        ``"reverse_path_likelihood"``, or ``"random"``.
        top_k_paths:    For all graph-based heuristics.
        min_path_probability: Minimum path probability to consider.
        success_only:   Only consider successful rollouts.
        random_seed:    RNG seed.  Required for ``"path_likelihood"``.  For
                        other graph-based heuristics, shuffles eligible rollout
                        order within each path so replicates pick different
                        individuals.
        failure_weight: ``"max"`` or ``"sum"`` — only used by ``"near_failure"``.

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
            random_seed=random_seed,
        )
    if heuristic_name == "diversity":
        return DiversitySelectionHeuristic(
            top_k_paths=top_k_paths,
            min_path_probability=min_path_probability,
            success_only=success_only,
            random_seed=random_seed,
        )
    if heuristic_name == "near_failure":
        return NearFailurePathHeuristic(
            top_k_paths=top_k_paths,
            failure_weight=failure_weight,
            min_path_probability=min_path_probability,
            success_only=success_only,
            random_seed=random_seed,
        )
    if heuristic_name == "path_likelihood":
        return PathLikelihoodHeuristic(
            top_k_paths=top_k_paths,
            min_path_probability=min_path_probability,
            success_only=success_only,
            random_seed=random_seed,
        )
    if heuristic_name == "reverse_path_likelihood":
        return ReversePathLikelihoodHeuristic(
            top_k_paths=top_k_paths,
            min_path_probability=min_path_probability,
            success_only=success_only,
            random_seed=random_seed,
        )
    if heuristic_name == "random":
        return RandomSelectionHeuristic(
            random_seed=random_seed,
            success_only=success_only,
        )
    raise ValueError(
        f"Unknown heuristic: {heuristic_name!r}.  "
        "Valid options: 'behavior_graph', 'diversity', 'near_failure', "
        "'path_likelihood', 'reverse_path_likelihood', 'random'."
    )

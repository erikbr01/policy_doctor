"""Pure analysis functions for graph-guided failure state targeting.

Identifies states that lead to policy failure via behavior graph analysis,
clusters them in object-pose space, and provides utilities for seed selection
and IC/subtask constraint derivation.

All functions are pure numpy/h5py — no pipeline or Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Default state schema for robosuite NutAssemblySquare
# Maps object names → qpos indices in the raw state vector stored in HDF5.
# State layout (Square task, 45-dim): [robot_qpos(0-9) | nut_free_joint(10-16) |
#   robot_qvel(17-25) | nut_free_joint_vel(26-31) | gripper_contact(32-44)]
# qpos indices: nut_x=10, nut_y=11, nut_z=12, nut_qw=13, nut_qx=14, nut_qy=15, nut_qz=16
# ---------------------------------------------------------------------------
DEFAULT_SQUARE_STATE_SCHEMA: dict[str, dict[str, int]] = {
    "nut": {
        "x_idx": 10,
        "y_idx": 11,
        "z_idx": 12,
        "qw_idx": 13,
        "qx_idx": 14,
        "qy_idx": 15,
        "qz_idx": 16,
    }
}


# Number of feature dims per object in the cluster-feature encoding:
#   [x, y, z, qw, qx, qy, qz]  — full SE(3), with the quaternion stored in
#   canonical (qw ≥ 0) form. Mean-of-quaternions inside a tight cluster
#   approximates the true mean rotation; we renormalise on the way out.
FEATURE_DIM_PER_OBJECT: int = 7


# ---------------------------------------------------------------------------
# State / pose extraction
# ---------------------------------------------------------------------------

def _canonical_quat(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float, float]:
    """Force the quaternion to the qw ≥ 0 hemisphere.

    The quaternions q and -q represent the same rotation; KMeans on raw
    quaternion components would treat them as far apart. Canonicalising
    to qw ≥ 0 collapses that ambiguity so Euclidean means make sense
    inside a tight cluster.
    """
    if qw < 0:
        return -qw, -qx, -qy, -qz
    return qw, qx, qy, qz


def _state_to_object_poses(
    state: np.ndarray,
    state_schema: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Extract world-frame SE(3) pose for each object from a raw state vector.

    Args:
        state:        1-D state array (qpos slice from HDF5 states).
        state_schema: Mapping from object name to qpos index dict. Must include
                      ``x_idx``, ``y_idx``, ``z_idx``, ``qw_idx``, ``qx_idx``,
                      ``qy_idx``, ``qz_idx`` per object.

    Returns:
        ``{obj_name: {"x", "y", "z", "qw", "qx", "qy", "qz"}}``  — quaternion
        in canonical (qw ≥ 0) hemisphere.
    """
    result: dict[str, dict[str, float]] = {}
    for obj_name, schema in state_schema.items():
        x = float(state[schema["x_idx"]])
        y = float(state[schema["y_idx"]])
        z = float(state[schema["z_idx"]])
        qw, qx, qy, qz = _canonical_quat(
            float(state[schema["qw_idx"]]),
            float(state[schema["qx_idx"]]),
            float(state[schema["qy_idx"]]),
            float(state[schema["qz_idx"]]),
        )
        result[obj_name] = {
            "x": x, "y": y, "z": z,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
        }
    return result


def _state_to_cluster_features(
    state: np.ndarray,
    state_schema: dict[str, dict[str, int]],
) -> np.ndarray:
    """Return a (7*n_objects,) feature vector suitable for KMeans clustering.

    Encodes each object as [x, y, z, qw, qx, qy, qz] with the quaternion
    canonicalised to the qw ≥ 0 hemisphere — see :func:`_canonical_quat`.
    """
    features: list[float] = []
    for obj_name in sorted(state_schema):
        poses = _state_to_object_poses(state, {obj_name: state_schema[obj_name]})
        p = poses[obj_name]
        features += [p["x"], p["y"], p["z"], p["qw"], p["qx"], p["qy"], p["qz"]]
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Graph analysis
# ---------------------------------------------------------------------------

def find_pre_failure_nodes(
    graph: Any,
    node_values: dict[int, float],
    value_threshold: float = 0.0,
    min_transition_prob: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Return edges (prior_node, low_value_node, transition_prob) from graph.

    A "pre-failure edge" is any edge src → dst where:
      - V(dst) < value_threshold  (dst is failure-prone)
      - transition_prob >= min_transition_prob

    Args:
        graph:               BehaviorGraph instance.
        node_values:         Output of ``graph.compute_values()``.
        value_threshold:     Nodes with V < threshold are considered failure-prone.
        min_transition_prob: Ignore edges with probability below this.

    Returns:
        List of (prior_node_id, low_value_node_id, transition_prob), sorted
        descending by transition probability.
    """
    from policy_doctor.behaviors.behavior_graph import TERMINAL_NODE_IDS

    edges: list[tuple[int, int, float]] = []
    for node_id in list(graph.nodes):
        if node_id in TERMINAL_NODE_IDS:
            continue
        v = node_values.get(node_id, 0.0)
        if v >= value_threshold:
            # Only consider SOURCES that transition INTO low-value nodes
            for tgt_id, tgt_prob in graph.transition_probs.get(node_id, {}).items():
                if tgt_prob < min_transition_prob:
                    continue
                tgt_v = node_values.get(tgt_id, 0.0)
                if tgt_v < value_threshold:
                    edges.append((node_id, tgt_id, float(tgt_prob)))

    edges.sort(key=lambda e: e[2], reverse=True)
    return edges


# ---------------------------------------------------------------------------
# State collection from rollout HDF5
# ---------------------------------------------------------------------------

def _episode_to_demo_key(episode_idx: int) -> str:
    return f"demo_{episode_idx}"


def collect_failed_initial_states(
    rollouts_hdf5: str,
    metadata: list[dict[str, Any]],
    node_values: dict[int, float],
    value_threshold: float = 0.0,
    state_schema: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Return feature vectors and episode indices for failed episodes.

    Reads t=0 state from each failed episode and converts to object-pose
    features using *state_schema*.

    Args:
        rollouts_hdf5: Path to the rollout HDF5 file.
        metadata:      Per-sample metadata list (from clustering loader).
        node_values:   Node V-values (used only for documentation; success/failure
                       is read directly from HDF5 attrs for reliability).
        value_threshold: Unused; kept for API consistency.
        state_schema:  Object → qpos-index mapping.  Defaults to
                       :data:`DEFAULT_SQUARE_STATE_SCHEMA`.

    Returns:
        Tuple of:
          - ``features``: ``(N, D)`` float32 feature array (one row per failed ep)
          - ``episode_indices``: list of episode indices (same order as rows)
    """
    schema = state_schema or DEFAULT_SQUARE_STATE_SCHEMA
    features: list[np.ndarray] = []
    episode_indices: list[int] = []

    with h5py.File(rollouts_hdf5, "r") as f:
        data_grp = f["data"]
        for demo_key in sorted(data_grp.keys()):
            if not demo_key.startswith("demo_"):
                continue
            ep = data_grp[demo_key]
            success = bool(ep.attrs.get("success", False))
            if success:
                continue
            ep_idx = int(demo_key.split("_")[1])
            states = np.array(ep["states"], dtype=np.float64)
            if states.ndim != 2 or states.shape[0] == 0:
                continue
            feat = _state_to_cluster_features(states[0], schema)
            features.append(feat)
            episode_indices.append(ep_idx)

    if not features:
        return np.zeros((0, 4 * len(schema)), dtype=np.float32), []
    return np.stack(features).astype(np.float32), episode_indices


def collect_states_at_node(
    rollouts_hdf5: str,
    cluster_labels: np.ndarray,
    metadata: list[dict[str, Any]],
    node_id: int,
    state_schema: dict[str, dict[str, int]] | None = None,
    level: str = "rollout",
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return feature vectors and (episode_idx, timestep) pairs for samples at node_id.

    Args:
        rollouts_hdf5:  Path to rollout HDF5 file.
        cluster_labels: ``(N,)`` int array from clustering.
        metadata:       Per-sample metadata (same order as labels).
        node_id:        Cluster ID to collect states for.
        state_schema:   Object → qpos-index mapping.
        level:          ``"rollout"`` or ``"demo"``.

    Returns:
        Tuple of:
          - ``features``: ``(N, D)`` float32 array (one row per sample at node)
          - ``tags``: list of (episode_idx, timestep) for each row
    """
    schema = state_schema or DEFAULT_SQUARE_STATE_SCHEMA
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    ts_key = "timestep"

    # Find which samples are at node_id
    sample_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == node_id]
    if not sample_indices:
        return np.zeros((0, 4 * len(schema)), dtype=np.float32), []

    tags: list[tuple[int, int]] = []
    features: list[np.ndarray] = []

    with h5py.File(rollouts_hdf5, "r") as f:
        data_grp = f["data"]
        for si in sample_indices:
            meta = metadata[si]
            ep_idx = int(meta[ep_key])
            ts = int(meta.get(ts_key, meta.get("window_start", 0)))
            demo_key = _episode_to_demo_key(ep_idx)
            if demo_key not in data_grp:
                continue
            states = np.array(data_grp[demo_key]["states"], dtype=np.float64)
            ts_clamped = min(ts, len(states) - 1)
            feat = _state_to_cluster_features(states[ts_clamped], schema)
            features.append(feat)
            tags.append((ep_idx, ts))

    if not features:
        return np.zeros((0, 4 * len(schema)), dtype=np.float32), []
    return np.stack(features).astype(np.float32), tags


# ---------------------------------------------------------------------------
# State clustering
# ---------------------------------------------------------------------------

@dataclass
class StateClusterResult:
    """Result of clustering target states."""

    centers: np.ndarray          # (K, D) cluster centres
    labels: np.ndarray           # (N,) cluster assignments for input states
    n_clusters: int
    # Per-cluster eligible episode indices (episodes whose representative state
    # is in that cluster — used to filter seed candidates).
    cluster_episode_indices: list[list[int]] = field(default_factory=list)


def cluster_target_states(
    features: np.ndarray,
    episode_tags: list[int | tuple[int, int]],
    n_clusters: int = 5,
    method: str = "kmeans",
    random_seed: int = 0,
) -> StateClusterResult:
    """Cluster object-pose feature vectors with KMeans (default) or minibatch KMeans.

    Args:
        features:      ``(N, D)`` float32 array from
                       :func:`collect_failed_initial_states` or
                       :func:`collect_states_at_node`.
        episode_tags:  Episode indices (int) or (episode_idx, timestep) tuples
                       aligned with *features* rows.
        n_clusters:    Number of clusters.
        method:        ``"kmeans"`` (default) or ``"minibatch_kmeans"``.
        random_seed:   RNG seed for reproducibility.

    Returns:
        :class:`StateClusterResult`.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    n = len(features)
    k = min(n_clusters, n)
    if k < n_clusters:
        print(
            f"  [cluster_target_states] WARNING: only {n} states available; "
            f"reducing n_clusters from {n_clusters} to {k}"
        )

    cls = MiniBatchKMeans if method == "minibatch_kmeans" else KMeans
    km = cls(n_clusters=k, random_state=random_seed, n_init="auto")
    labels = km.fit_predict(features)

    # Group episodes by cluster
    cluster_eps: list[list[int]] = [[] for _ in range(k)]
    for i, tag in enumerate(episode_tags):
        ep_idx = tag[0] if isinstance(tag, tuple) else int(tag)
        ci = int(labels[i])
        if ep_idx not in cluster_eps[ci]:
            cluster_eps[ci].append(ep_idx)

    return StateClusterResult(
        centers=km.cluster_centers_.astype(np.float32),
        labels=labels.astype(np.int32),
        n_clusters=k,
        cluster_episode_indices=cluster_eps,
    )


# ---------------------------------------------------------------------------
# Constraint derivation
# ---------------------------------------------------------------------------

def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw (rotation about world Z axis) from a wxyz quaternion."""
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def cluster_center_to_object_poses(
    center: np.ndarray,
    state_schema: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[str, float]]:
    """Reconstruct world-frame SE(3) poses from a cluster-center feature vector.

    The feature vector is ``[x, y, z, qw, qx, qy, qz]`` per object (from
    :func:`_state_to_cluster_features`). The quaternion part of the centroid
    is renormalised so the result is a valid unit quaternion (a KMeans mean
    of canonical quaternions usually has ``||q||`` slightly under 1 inside
    a tight cluster).

    For convenience the result also exposes ``z_rot`` — yaw recovered from
    the quaternion — so downstream code that only cares about planar
    rotation (e.g. ``object_poses_to_pose_ranges`` building the IC range
    schema) can keep working unchanged.

    Returns:
        ``{obj_name: {"x", "y", "z", "qw", "qx", "qy", "qz", "z_rot"}}``.
    """
    schema = state_schema or DEFAULT_SQUARE_STATE_SCHEMA
    result: dict[str, dict[str, float]] = {}
    for i, obj_name in enumerate(sorted(schema)):
        base = FEATURE_DIM_PER_OBJECT * i
        x = float(center[base + 0])
        y = float(center[base + 1])
        z = float(center[base + 2])
        qw = float(center[base + 3])
        qx = float(center[base + 4])
        qy = float(center[base + 5])
        qz = float(center[base + 6])
        norm = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
        if norm > 1e-9:
            qw /= norm; qx /= norm; qy /= norm; qz /= norm
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        z_rot = _yaw_from_quat(qw, qx, qy, qz)
        result[obj_name] = {
            "x": x, "y": y, "z": z,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "z_rot": z_rot,
        }
    return result


def object_poses_to_pose_ranges(
    poses: dict[str, dict[str, float]],
    slack_x: float = 0.03,
    slack_y: float = 0.03,
    slack_z_rot: float = 0.5,
) -> dict[str, dict[str, list[float]]]:
    """Build an ``object_pose_ranges`` dict centred on *poses* with given slack.

    The returned dict uses the same format as the existing IC-constraint config:
    ``{obj_name: {x: [lo, hi], y: [lo, hi], z_rot: [lo, hi]}}``.

    Args:
        poses:      World-frame {x, y, z_rot} per object.
        slack_x:    ± offset in world-frame x (metres).
        slack_y:    ± offset in world-frame y (metres).
        slack_z_rot: ± offset in z_rot (radians).

    Returns:
        ``object_pose_ranges`` dict ready for ``GenerateMimicgenDemosStep``.

    Note:
        The returned x/y are *offsets from the seed pose*, not absolute world
        values.  This is consistent with how :func:`_constrained_bounds` in
        ``run_mimicgen_generate.py`` interprets ``object_pose_ranges``:
        it subtracts the environment reference and seed value, then applies
        [lo_offset, hi_offset] on top.  Since we select seeds whose initial
        pose ≈ cluster center, setting offset = [-slack, +slack] effectively
        constrains generation near the cluster center.
    """
    return {
        obj_name: {
            "x": [-slack_x, slack_x],
            "y": [-slack_y, slack_y],
            "z_rot": [-slack_z_rot, slack_z_rot],
        }
        for obj_name in poses
    }


def derive_subtask_constraints(
    poses: dict[str, dict[str, float]],
    subtask_idx: int,
    slack_multiplier: float = 1.5,
    slack_x: float = 0.05,
    slack_y: float = 0.05,
    slack_z_rot: float = 0.8,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Build a ``subtask_constraints`` dict for the given subtask boundary.

    Uses *slack_multiplier* × base IC slack to give the subtask constraint a
    slightly wider range than the initial condition constraint.

    Returns:
        ``{str(subtask_idx): {obj_name: {x:[lo,hi], y:[lo,hi], z_rot:[lo,hi]}}}``
    """
    constraint = {
        obj_name: {
            "x": [-slack_x * slack_multiplier, slack_x * slack_multiplier],
            "y": [-slack_y * slack_multiplier, slack_y * slack_multiplier],
            "z_rot": [-slack_z_rot * slack_multiplier, slack_z_rot * slack_multiplier],
        }
        for obj_name in poses
    }
    return {str(subtask_idx): constraint}


# ===========================================================================
# Phase 1 — path-based failure trajectory selection.
#
# The pre-failure-node code above pools states from all trajectories (success
# and failure) that visited risky-looking graph nodes. That over-includes
# successful traversals — the policy clearly didn't need extra coverage at
# states it navigated successfully.
#
# The path-based approach instead:
#   1. enumerates the top-K most-probable START→FAILURE paths in the graph,
#   2. matches each path to the failure episodes whose run-length-collapsed
#      cluster sequence equals the path interior (these are "the failure
#      trajectories that explain how the path is traversed"),
#   3. for each node on the path, collects the (object, robot) sim states
#      from those failure trajectories at the timesteps they spent in that
#      node — that's the target pool.
# ===========================================================================

from collections import defaultdict


@dataclass
class FailurePath:
    """A high-probability START→FAILURE path plus the failure episodes that
    actually follow it.

    Attributes:
        path:               Node-ID sequence from START to FAILURE inclusive,
                            e.g. ``[-2, 7, 11, -5]``.
        probability:        Cumulative edge-probability product (same as the
                            BehaviorGraph path ranking).
        matched_episodes:   Failure-episode indices whose run-length-collapsed
                            cluster sequence equals ``path[1:-1]`` (path
                            interior, START / FAILURE removed).
    """

    path: list[int]
    probability: float
    matched_episodes: list[int] = field(default_factory=list)


def enumerate_failure_paths(
    graph: Any,
    top_k: int = 5,
    min_edge_probability: float = 0.0,
    min_path_probability: float = 0.0,
    drop_paths_with_loops: bool = True,
) -> list[tuple[list[int], float]]:
    """Top-K most-probable START→FAILURE simple paths.

    Thin wrapper around :meth:`BehaviorGraph.enumerate_paths_to_terminal`.

    Args:
        graph:                 The behavior graph.
        top_k:                 Number of paths to keep (best-first).
        min_edge_probability:  Skip edges below this probability.
        min_path_probability:  Skip paths whose cumulative probability is below this.
        drop_paths_with_loops: Exclude paths that the underlying enumerator
                               reports as having loop edges; keeps the matched
                               trajectory selection unambiguous.

    Returns:
        List of ``(path, probability)`` tuples, sorted by probability descending.
        ``path`` includes both endpoints (START and FAILURE).
    """
    from policy_doctor.behaviors.behavior_graph import FAILURE_NODE_ID

    raw = graph.enumerate_paths_to_terminal(
        terminal_id=FAILURE_NODE_ID,
        max_paths=top_k * 4 if drop_paths_with_loops else top_k,
        min_probability=min_path_probability,
        min_edge_probability=min_edge_probability,
    )
    out: list[tuple[list[int], float]] = []
    for path, prob, loops in raw:
        if drop_paths_with_loops and loops:
            continue
        out.append((list(path), float(prob)))
        if len(out) >= top_k:
            break
    return out


def _collapsed_sequences_for_episodes(
    cluster_labels: np.ndarray,
    metadata: list[dict[str, Any]],
    level: str = "rollout",
) -> tuple[dict[int, list[int]], dict[int, "bool | None"]]:
    """Per-episode run-length-collapsed cluster sequence, plus success flags.

    Re-uses the same convention as ``behavior_graph._extract_collapsed_sequences``
    but exposed here so we don't have to import a private helper.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    by_ep: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outcomes: dict[int, "bool | None"] = {}
    for i, meta in enumerate(metadata):
        label = int(cluster_labels[i])
        if label == -1:
            continue
        ep_idx = int(meta[ep_key])
        sort_key = int(meta.get("timestep", meta.get("window_start", 0)))
        by_ep[ep_idx].append((sort_key, label))
        if "success" in meta and ep_idx not in outcomes:
            outcomes[ep_idx] = bool(meta["success"])

    collapsed: dict[int, list[int]] = {}
    for ep_idx, seq in by_ep.items():
        seq.sort(key=lambda x: x[0])
        run: list[int] = [seq[0][1]]
        for _, label in seq[1:]:
            if label != run[-1]:
                run.append(label)
        collapsed[ep_idx] = run
    return collapsed, outcomes


def match_failure_trajectories_to_paths(
    cluster_labels: np.ndarray,
    metadata: list[dict[str, Any]],
    paths: list[list[int]],
    level: str = "rollout",
) -> list[list[int]]:
    """For each path, return the failure-episode indices that traverse it.

    A failure episode is *matched* to a path if its run-length-collapsed
    cluster sequence equals the path interior (path with START / FAILURE removed).

    Args:
        cluster_labels: ``(N,)`` int array of per-sample cluster IDs.
        metadata:       Per-sample metadata (same length as labels).
        paths:          List of ``[START, ..., FAILURE]`` node sequences from
                        :func:`enumerate_failure_paths`.
        level:          ``"rollout"`` or ``"demo"``.

    Returns:
        ``[[ep_idx, ...], ...]`` parallel to ``paths``.  Episodes are reported
        in ascending order; an episode appears in at most one path's list
        (the first matching one, by paths-list order — usually the
        most-probable path the episode is consistent with).
    """
    from policy_doctor.behaviors.behavior_graph import (
        START_NODE_ID,
        FAILURE_NODE_ID,
    )

    collapsed, outcomes = _collapsed_sequences_for_episodes(
        cluster_labels, metadata, level=level
    )
    interiors = [
        tuple(p for p in path if p not in (START_NODE_ID, FAILURE_NODE_ID))
        for path in paths
    ]
    matched: list[list[int]] = [[] for _ in paths]
    claimed: set[int] = set()
    for ep_idx in sorted(collapsed):
        if outcomes.get(ep_idx) is not False:
            continue  # only count failures
        if ep_idx in claimed:
            continue
        seq = tuple(collapsed[ep_idx])
        for i, interior in enumerate(interiors):
            if seq == interior:
                matched[i].append(ep_idx)
                claimed.add(ep_idx)
                break
    return matched


def collect_failure_trajectory_states_by_node(
    rollouts_hdf5: str,
    cluster_labels: np.ndarray,
    metadata: list[dict[str, Any]],
    matched_episodes: list[int],
    state_schema: "dict[str, dict[str, int]] | None" = None,
    level: str = "rollout",
) -> dict[int, tuple[np.ndarray, list[tuple[int, int]]]]:
    """Per-node sim-state features from matched failure trajectories.

    For each cluster-node visited by any matched failure episode, returns the
    (object, robot) state features at every timestep the episode spent in that
    node. Each episode also contributes its t=0 state under the
    :data:`START_NODE_ID` key — that becomes the IC pool downstream.

    Args:
        rollouts_hdf5:    Path to the rollout HDF5 produced by eval_save_episodes.
        cluster_labels:   ``(N,)`` per-sample cluster IDs.
        metadata:         Per-sample metadata.
        matched_episodes: Failure-episode indices to include.
        state_schema:     Object → qpos-index mapping. Defaults to Square.
        level:            ``"rollout"`` or ``"demo"``.

    Returns:
        ``{node_id: (features (N, D), [(ep_idx, timestep), ...])}``.
        ``node_id == START_NODE_ID`` holds t=0 states.
    """
    from policy_doctor.behaviors.behavior_graph import START_NODE_ID

    schema = state_schema or DEFAULT_SQUARE_STATE_SCHEMA
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    matched_set = set(int(e) for e in matched_episodes)

    # Group sample indices by (episode, label).
    by_ep_node: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        ep_idx = int(meta[ep_key])
        if ep_idx not in matched_set:
            continue
        label = int(cluster_labels[i])
        if label == -1:
            continue
        ts = int(meta.get("timestep", meta.get("window_start", 0)))
        by_ep_node[(ep_idx, label)].append((ts, i))

    # node_id → (features list, tags list)
    out: dict[int, tuple[list[np.ndarray], list[tuple[int, int]]]] = defaultdict(
        lambda: ([], [])
    )
    with h5py.File(rollouts_hdf5, "r") as f:
        data_grp = f["data"]
        # IC pool: t=0 of each matched episode.
        for ep_idx in sorted(matched_set):
            demo_key = _episode_to_demo_key(ep_idx)
            if demo_key not in data_grp:
                continue
            states = np.array(data_grp[demo_key]["states"], dtype=np.float64)
            if states.ndim != 2 or states.shape[0] == 0:
                continue
            feat = _state_to_cluster_features(states[0], schema)
            out[START_NODE_ID][0].append(feat)
            out[START_NODE_ID][1].append((ep_idx, 0))

        # Intermediate pool: states at every timestep each episode was at each node.
        for (ep_idx, node_id), samples in by_ep_node.items():
            demo_key = _episode_to_demo_key(ep_idx)
            if demo_key not in data_grp:
                continue
            states = np.array(data_grp[demo_key]["states"], dtype=np.float64)
            n_ts = states.shape[0]
            for ts, _sample_i in samples:
                ts_clamped = min(ts, n_ts - 1)
                feat = _state_to_cluster_features(states[ts_clamped], schema)
                out[node_id][0].append(feat)
                out[node_id][1].append((ep_idx, ts))

    return {
        node_id: (
            np.stack(feats).astype(np.float32) if feats
            else np.zeros((0, 4 * len(schema)), dtype=np.float32),
            list(tags),
        )
        for node_id, (feats, tags) in out.items()
    }


@dataclass
class NodeClusterResult:
    """KMeans result for one node's state pool, with silhouette-picked k."""

    node_id: int
    n_states: int
    k: int                           # selected number of clusters
    silhouette: float                # silhouette score at the selected k (or NaN)
    centers: np.ndarray              # (k, D)
    stddevs: np.ndarray              # (k, D) per-cluster per-dim stddev
    labels: np.ndarray               # (n_states,) cluster assignment for each row
    tags: list[tuple[int, int]] = field(default_factory=list)
    # Eligible episodes per cluster (in cluster member order).
    cluster_episode_indices: list[list[int]] = field(default_factory=list)


def silhouette_kmeans(
    features: np.ndarray,
    tags: list[tuple[int, int]],
    node_id: int,
    k_min: int = 2,
    k_max: int = 10,
    random_seed: int = 0,
) -> NodeClusterResult:
    """KMeans where ``k`` is chosen by silhouette score over ``[k_min, k_max]``.

    Degenerate cases:
      - ``n_states == 0``  → returns an empty result (k=0).
      - ``n_states == 1``  → single-cluster result, k=1, silhouette = NaN.
      - ``n_states < k_min`` → forces ``k = max(1, n_states - 1)``.

    Always reports per-cluster per-dim stddev (used downstream to size the
    constraint slack box: ``slack = α × stddev``).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(features)
    D = features.shape[1] if n > 0 else 0
    if n == 0:
        return NodeClusterResult(
            node_id=node_id, n_states=0, k=0, silhouette=float("nan"),
            centers=np.zeros((0, D), dtype=np.float32),
            stddevs=np.zeros((0, D), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int32),
            tags=[],
            cluster_episode_indices=[],
        )
    if n == 1:
        return NodeClusterResult(
            node_id=node_id, n_states=1, k=1, silhouette=float("nan"),
            centers=features.astype(np.float32),
            stddevs=np.zeros((1, D), dtype=np.float32),
            labels=np.zeros(1, dtype=np.int32),
            tags=list(tags),
            cluster_episode_indices=[[int(tags[0][0])]] if tags else [[]],
        )

    # Constrain the sweep range to what the data can support.
    k_upper = min(k_max, n - 1)
    k_lower = min(k_min, k_upper)
    if k_lower < 2:
        # Fall back to single cluster.
        center = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        ep_idxs: list[int] = []
        for ep, _ in tags:
            if ep not in ep_idxs:
                ep_idxs.append(int(ep))
        return NodeClusterResult(
            node_id=node_id, n_states=n, k=1, silhouette=float("nan"),
            centers=center.astype(np.float32),
            stddevs=std.astype(np.float32),
            labels=np.zeros(n, dtype=np.int32),
            tags=list(tags),
            cluster_episode_indices=[ep_idxs],
        )

    best_k, best_score, best_labels, best_km = 0, -2.0, None, None
    for k in range(k_lower, k_upper + 1):
        km = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        labels = km.fit_predict(features)
        if len(set(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(features, labels))
        except ValueError:
            continue
        if score > best_score:
            best_k, best_score, best_labels, best_km = k, score, labels, km

    # If silhouette never succeeded, fall back to k=k_lower.
    if best_km is None:
        from sklearn.cluster import KMeans as _KM
        km = _KM(n_clusters=k_lower, random_state=random_seed, n_init="auto")
        best_labels = km.fit_predict(features)
        best_km = km
        best_k = k_lower
        best_score = float("nan")

    centers = best_km.cluster_centers_.astype(np.float32)
    stds = np.zeros_like(centers)
    cluster_eps: list[list[int]] = [[] for _ in range(best_k)]
    for ci in range(best_k):
        mask = best_labels == ci
        if mask.any():
            stds[ci] = features[mask].std(axis=0).astype(np.float32)
        seen: set[int] = set()
        for i, in_cluster in enumerate(mask):
            if not in_cluster:
                continue
            ep = int(tags[i][0])
            if ep not in seen:
                cluster_eps[ci].append(ep)
                seen.add(ep)

    return NodeClusterResult(
        node_id=node_id, n_states=n, k=best_k, silhouette=best_score,
        centers=centers, stddevs=stds,
        labels=best_labels.astype(np.int32),
        tags=list(tags),
        cluster_episode_indices=cluster_eps,
    )


def pick_intermediate_target_node(
    path: list[int],
    heuristic: str = "closest_to_failure",
) -> "int | None":
    """Pick the single intermediate node to constrain on for this path.

    Heuristics:
        ``"closest_to_failure"`` — the last interior node (just before FAILURE).
        ``"first"``              — the first interior node (just after START).
        ``"middle"``             — interior[len(interior)//2].

    Use :func:`intermediate_nodes_for_path` to get the full list if you want
    to constrain multiple nodes per generated demo (slower / higher rejection).

    Returns ``None`` if the path has no intermediate nodes.
    """
    from policy_doctor.behaviors.behavior_graph import (
        START_NODE_ID,
        FAILURE_NODE_ID,
    )

    interior = [n for n in path if n not in (START_NODE_ID, FAILURE_NODE_ID)]
    if not interior:
        return None
    if heuristic == "closest_to_failure":
        return interior[-1]
    if heuristic == "first":
        return interior[0]
    if heuristic == "middle":
        return interior[len(interior) // 2]
    raise ValueError(f"unknown heuristic: {heuristic!r}")


def intermediate_nodes_for_path(path: list[int]) -> list[int]:
    """Return all interior nodes (START/FAILURE removed) in path order."""
    from policy_doctor.behaviors.behavior_graph import (
        START_NODE_ID,
        FAILURE_NODE_ID,
    )

    return [n for n in path if n not in (START_NODE_ID, FAILURE_NODE_ID)]

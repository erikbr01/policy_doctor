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
        "qw_idx": 13,
        "qx_idx": 14,
        "qy_idx": 15,
        "qz_idx": 16,
    }
}


# ---------------------------------------------------------------------------
# State / pose extraction
# ---------------------------------------------------------------------------

def _state_to_object_poses(
    state: np.ndarray,
    state_schema: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Extract world-frame {x, y, z_rot} for each object from a raw state vector.

    Args:
        state:        1-D state array (qpos slice from HDF5 states).
        state_schema: Mapping from object name to qpos index dict.

    Returns:
        ``{obj_name: {"x": float, "y": float, "z_rot": float}}``
    """
    result: dict[str, dict[str, float]] = {}
    for obj_name, schema in state_schema.items():
        x = float(state[schema["x_idx"]])
        y = float(state[schema["y_idx"]])
        qw = float(state[schema["qw_idx"]])
        qx = float(state.flat[schema.get("qx_idx", schema["qw_idx"] + 1)])
        qy = float(state.flat[schema.get("qy_idx", schema["qw_idx"] + 2)])
        qz = float(state.flat[schema.get("qz_idx", schema["qw_idx"] + 3)])
        # Yaw (rotation about world Z axis) from quaternion (MuJoCo wxyz convention).
        z_rot = float(np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        ))
        result[obj_name] = {"x": x, "y": y, "z_rot": z_rot}
    return result


def _state_to_cluster_features(
    state: np.ndarray,
    state_schema: dict[str, dict[str, int]],
) -> np.ndarray:
    """Return a (4*n_objects,) feature vector suitable for KMeans clustering.

    Encodes each object as [x, y, sin(z_rot), cos(z_rot)] to avoid angle
    wrap-around artefacts in Euclidean distance.
    """
    features: list[float] = []
    for obj_name in sorted(state_schema):
        poses = _state_to_object_poses(state, {obj_name: state_schema[obj_name]})
        p = poses[obj_name]
        z = p["z_rot"]
        features += [p["x"], p["y"], np.sin(z), np.cos(z)]
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

def cluster_center_to_object_poses(
    center: np.ndarray,
    state_schema: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[str, float]]:
    """Reconstruct world-frame {x, y, z_rot} from a cluster-center feature vector.

    The feature vector is [x, y, sin(z_rot), cos(z_rot)] per object (from
    :func:`_state_to_cluster_features`).  This inverts the encoding.

    Args:
        center:       1-D feature vector (length 4 * n_objects).
        state_schema: Used only for object names / ordering.

    Returns:
        ``{obj_name: {"x": float, "y": float, "z_rot": float}}``
    """
    schema = state_schema or DEFAULT_SQUARE_STATE_SCHEMA
    result: dict[str, dict[str, float]] = {}
    for i, obj_name in enumerate(sorted(schema)):
        x = float(center[4 * i + 0])
        y = float(center[4 * i + 1])
        sin_z = float(center[4 * i + 2])
        cos_z = float(center[4 * i + 3])
        z_rot = float(np.arctan2(sin_z, cos_z))
        result[obj_name] = {"x": x, "y": y, "z_rot": z_rot}
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

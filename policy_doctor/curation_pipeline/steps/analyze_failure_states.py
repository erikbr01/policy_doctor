"""Analyze behavior graph to identify and cluster failure-prone states — pipeline step.

Reads the clustering result from :class:`RunClusteringStep`, builds the behavior
graph, finds pre-failure transitions, collects the corresponding initial/intermediate
states from the rollout HDF5, clusters them, and persists per-cluster seed candidates
and suggested IC/subtask constraints.

Config keys (all under ``mimicgen_datagen.failure_analysis``):

    enabled               If false, step is skipped (writes empty result).  Default false.
    value_threshold       V-value below which a node is considered failure-prone (default 0.0).
    min_transition_prob   Minimum edge probability to treat as pre-failure (default 0.05).
    n_clusters            Number of target-state clusters (default 5).
    budget_per_cluster    Seed budget allocated per cluster (default 4).
    state_schema          Per-object → qpos-index mapping for state parsing.  Null uses
                          the Square-task default.
    slack_x               IC constraint ± offset in x (metres, default 0.03).
    slack_y               IC constraint ± offset in y (metres, default 0.03).
    slack_z_rot           IC constraint ± offset in z_rot (radians, default 0.5).
    cluster_target_mode   ``"centroid"`` or ``"sample"`` — how to derive the suggested
                          target pose from a cluster (default ``"centroid"``).
    targeting_mode        ``"initial_state"``, ``"intermediate_state"``, or ``"both"``
                          (default ``"both"``).
    subtask_constraint_idx    Integer subtask boundary to enforce constraints at.  Null
                              disables subtask constraints (default null).
    subtask_constraint_slack  Extra slack multiplier for subtask constraints (default 1.5).

Result JSON (``step_dir/result.json``):
    enabled               bool — whether failure analysis ran.
    n_clusters            int
    pre_failure_nodes     list of [prior_id, target_id, prob]
    clusters              list of cluster dicts (see below)

Each cluster dict:
    cluster_idx               int
    n_states                  int — number of states in this cluster
    eligible_rollout_idxs     list[int]
    suggested_object_pose_ranges  dict (same format as ``mimicgen_datagen.object_pose_ranges``)
    suggested_subtask_constraints dict (null when subtask_constraint_idx is null)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.data.clustering_loader import load_clustering_result_from_path
from policy_doctor.mimicgen.failure_targeting import (
    DEFAULT_SQUARE_STATE_SCHEMA,
    StateClusterResult,
    cluster_center_to_object_poses,
    cluster_target_states,
    collect_failed_initial_states,
    collect_states_at_node,
    derive_subtask_constraints,
    find_pre_failure_nodes,
    object_poses_to_pose_ranges,
)

# Re-use the rollout HDF5 resolver from the existing graph step.
from policy_doctor.curation_pipeline.steps.select_mimicgen_seed_from_graph import (
    _resolve_rollouts_hdf5,
)


class AnalyzeFailureStatesStep(PipelineStep[dict]):
    """Identify and cluster failure-prone states for targeted MimicGen generation.

    When ``mimicgen_datagen.failure_analysis.enabled`` is false (the default)
    the step writes an ``{"enabled": false}`` result and exits immediately —
    all downstream steps behave as if failure analysis never ran.
    """

    name = "analyze_failure_states"

    def compute(self) -> dict[str, Any]:
        from policy_doctor.behaviors.behavior_graph import BehaviorGraph
        from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

        fa_cfg = OmegaConf.select(self.cfg, "mimicgen_datagen.failure_analysis") or {}

        enabled: bool = bool(OmegaConf.select(fa_cfg, "enabled") or False)
        if not enabled:
            print("  [analyze_failure_states] failure_analysis.enabled=false — skipping.")
            return {"enabled": False, "n_clusters": 0, "pre_failure_nodes": [], "clusters": []}

        # --- Config ---
        value_threshold: float = float(OmegaConf.select(fa_cfg, "value_threshold") or 0.0)
        min_transition_prob: float = float(OmegaConf.select(fa_cfg, "min_transition_prob") or 0.05)
        n_clusters: int = int(OmegaConf.select(fa_cfg, "n_clusters") or 5)
        budget_per_cluster: int = int(OmegaConf.select(fa_cfg, "budget_per_cluster") or 4)
        slack_x: float = float(OmegaConf.select(fa_cfg, "slack_x") or 0.03)
        slack_y: float = float(OmegaConf.select(fa_cfg, "slack_y") or 0.03)
        slack_z_rot: float = float(OmegaConf.select(fa_cfg, "slack_z_rot") or 0.5)
        cluster_target_mode: str = str(OmegaConf.select(fa_cfg, "cluster_target_mode") or "centroid")
        targeting_mode: str = str(OmegaConf.select(fa_cfg, "targeting_mode") or "both")
        subtask_constraint_idx_raw = OmegaConf.select(fa_cfg, "subtask_constraint_idx")
        subtask_constraint_idx: int | None = (
            int(subtask_constraint_idx_raw) if subtask_constraint_idx_raw is not None else None
        )
        subtask_constraint_slack: float = float(
            OmegaConf.select(fa_cfg, "subtask_constraint_slack") or 1.5
        )

        # State schema (configurable, defaults to Square task layout).
        state_schema_cfg = OmegaConf.select(fa_cfg, "state_schema")
        if state_schema_cfg is not None:
            state_schema = OmegaConf.to_container(state_schema_cfg, resolve=True)
        else:
            state_schema = DEFAULT_SQUARE_STATE_SCHEMA

        # --- Load clustering result ---
        prior = RunClusteringStep(self.cfg, self.parent_run_dir).load()
        clustering_dirs: dict[str, str] = {}
        if prior and isinstance(prior.get("clustering_dirs"), dict):
            clustering_dirs = {str(k): str(v) for k, v in prior["clustering_dirs"].items()}

        if not clustering_dirs:
            raise ValueError(
                "analyze_failure_states: no clustering directories found. "
                "Run run_clustering first."
            )

        policy_seed_cfg = OmegaConf.select(self.cfg, "mimicgen_datagen.policy_seed")
        seed = (
            str(policy_seed_cfg)
            if policy_seed_cfg is not None
            else sorted(clustering_dirs.keys())[0]
        )
        cdir = Path(clustering_dirs[seed])
        if not cdir.is_absolute():
            cdir = (self.repo_root / cdir).resolve()

        print(f"  [analyze_failure_states] seed={seed}  clustering_dir={cdir}")
        labels, metadata, manifest = load_clustering_result_from_path(cdir)
        level: str = manifest.get("level") or "rollout"

        # --- Rollout HDF5 ---
        rollouts_hdf5 = _resolve_rollouts_hdf5(self.cfg, self.repo_root, seed)
        print(f"  [analyze_failure_states] rollouts_hdf5={rollouts_hdf5}")

        # --- Build graph + compute values ---
        graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
        node_values = graph.compute_values()
        print(
            f"  [analyze_failure_states] graph: {len(graph.nodes)} nodes, "
            f"{graph.num_episodes} episodes"
        )

        # --- Find pre-failure edges ---
        pre_failure_edges = find_pre_failure_nodes(
            graph, node_values,
            value_threshold=value_threshold,
            min_transition_prob=min_transition_prob,
        )
        print(
            f"  [analyze_failure_states] pre-failure edges: {len(pre_failure_edges)} "
            f"(threshold={value_threshold}, min_prob={min_transition_prob})"
        )
        for src, tgt, prob in pre_failure_edges[:5]:
            v_src = node_values.get(src, 0.0)
            v_tgt = node_values.get(tgt, 0.0)
            print(f"    node {src} (V={v_src:.3f}) → {tgt} (V={v_tgt:.3f}), p={prob:.3f}")

        # --- Collect target states ---
        all_features: list[np.ndarray] = []
        all_tags: list[int | tuple[int, int]] = []

        if targeting_mode in ("initial_state", "both"):
            feats, ep_idxs = collect_failed_initial_states(
                str(rollouts_hdf5), metadata, node_values,
                value_threshold=value_threshold,
                state_schema=state_schema,
            )
            print(f"  [analyze_failure_states] initial states: {len(feats)} failed episodes")
            all_features.extend(feats)
            all_tags.extend(ep_idxs)

        if targeting_mode in ("intermediate_state", "both"):
            # Collect states at each PRIOR node (source of pre-failure edges).
            prior_node_ids = sorted({src for src, _, _ in pre_failure_edges})
            for node_id in prior_node_ids:
                feats, tags = collect_states_at_node(
                    str(rollouts_hdf5), labels, metadata, node_id,
                    state_schema=state_schema, level=level,
                )
                print(
                    f"    node {node_id}: {len(feats)} intermediate states "
                    f"(V={node_values.get(node_id, 0.0):.3f})"
                )
                all_features.extend(feats)
                all_tags.extend(tags)

        if not all_features:
            print(
                "  [analyze_failure_states] WARNING: no target states found. "
                "Check value_threshold, min_transition_prob, and targeting_mode."
            )
            return {
                "enabled": True,
                "n_clusters": 0,
                "pre_failure_nodes": [[src, tgt, p] for src, tgt, p in pre_failure_edges],
                "clusters": [],
            }

        combined_features = np.stack(all_features).astype(np.float32)
        print(f"  [analyze_failure_states] clustering {len(combined_features)} states → {n_clusters} clusters")

        # --- Cluster ---
        cluster_result: StateClusterResult = cluster_target_states(
            combined_features, all_tags, n_clusters=n_clusters, random_seed=0
        )
        print(f"  [analyze_failure_states] cluster sizes: "
              f"{[len(idxs) for idxs in cluster_result.cluster_episode_indices]}")

        # --- Build per-cluster output ---
        clusters_out: list[dict[str, Any]] = []
        for ci in range(cluster_result.n_clusters):
            center = cluster_result.centers[ci]

            # Derive target pose from cluster center or sample.
            if cluster_target_mode == "sample":
                ep_idxs_for_cluster = cluster_result.cluster_episode_indices[ci]
                member_mask = cluster_result.labels == ci
                member_feats = combined_features[member_mask]
                if len(member_feats) > 0:
                    rng = np.random.default_rng(ci)
                    sampled_center = member_feats[rng.integers(len(member_feats))]
                else:
                    sampled_center = center
                pose_center = cluster_center_to_object_poses(sampled_center, state_schema)
            else:
                pose_center = cluster_center_to_object_poses(center, state_schema)

            suggested_pose_ranges = object_poses_to_pose_ranges(
                pose_center, slack_x=slack_x, slack_y=slack_y, slack_z_rot=slack_z_rot
            )

            suggested_subtask: dict | None = None
            if subtask_constraint_idx is not None:
                suggested_subtask = derive_subtask_constraints(
                    pose_center,
                    subtask_idx=subtask_constraint_idx,
                    slack_multiplier=subtask_constraint_slack,
                    slack_x=slack_x,
                    slack_y=slack_y,
                    slack_z_rot=slack_z_rot,
                )

            eligible_idxs = cluster_result.cluster_episode_indices[ci]
            n_states = int((cluster_result.labels == ci).sum())

            print(
                f"    cluster {ci}: {n_states} states, "
                f"{len(eligible_idxs)} episodes, "
                f"center_x={pose_center.get(list(pose_center)[0], {}).get('x', 0):.3f}"
                if pose_center else f"    cluster {ci}: {n_states} states"
            )

            clusters_out.append({
                "cluster_idx": ci,
                "n_states": n_states,
                "eligible_rollout_idxs": eligible_idxs,
                "budget_per_cluster": budget_per_cluster,
                "suggested_object_pose_ranges": suggested_pose_ranges,
                "suggested_subtask_constraints": suggested_subtask,
            })

        return {
            "enabled": True,
            "n_clusters": cluster_result.n_clusters,
            "pre_failure_nodes": [[src, tgt, float(p)] for src, tgt, p in pre_failure_edges],
            "clusters": clusters_out,
        }

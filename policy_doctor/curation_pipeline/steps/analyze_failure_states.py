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
    NodeClusterResult,
    StateClusterResult,
    cluster_center_to_object_poses,
    cluster_target_states,
    collect_failed_initial_states,
    collect_failure_trajectory_states_by_node,
    collect_states_at_node,
    derive_subtask_constraints,
    enumerate_failure_paths,
    find_pre_failure_nodes,
    intermediate_nodes_for_path,
    match_failure_trajectories_to_paths,
    object_poses_to_pose_ranges,
    pick_intermediate_target_node,
    silhouette_kmeans,
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
        # mode selects which pipeline to run:
        #   "prefailure_node" (default, legacy): pool states from pre-failure graph nodes
        #     across all trajectories that visited them.
        #   "path_based": pick the top-K most-probable START→FAILURE paths, match each to
        #     the failure trajectories that follow it, collect states from those
        #     trajectories per node along the path.
        mode: str = str(OmegaConf.select(fa_cfg, "mode") or "prefailure_node")
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
        # Path-based-only knobs.
        top_k_paths: int = int(OmegaConf.select(fa_cfg, "top_k_paths") or 5)
        path_min_edge_probability: float = float(
            OmegaConf.select(fa_cfg, "path_min_edge_probability") or 0.0
        )
        intermediate_heuristic: str = str(
            OmegaConf.select(fa_cfg, "intermediate_heuristic") or "closest_to_failure"
        )
        kmeans_k_min: int = int(OmegaConf.select(fa_cfg, "kmeans_k_min") or 2)
        kmeans_k_max: int = int(OmegaConf.select(fa_cfg, "kmeans_k_max") or 10)

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

        # --- Dispatch on mode ---
        if mode == "path_based":
            return self._compute_path_based(
                graph=graph,
                node_values=node_values,
                labels=labels,
                metadata=metadata,
                level=level,
                rollouts_hdf5=rollouts_hdf5,
                state_schema=state_schema,
                top_k_paths=top_k_paths,
                path_min_edge_probability=path_min_edge_probability,
                intermediate_heuristic=intermediate_heuristic,
                kmeans_k_min=kmeans_k_min,
                kmeans_k_max=kmeans_k_max,
                slack_x=slack_x,
                slack_y=slack_y,
                slack_z_rot=slack_z_rot,
                subtask_constraint_idx=subtask_constraint_idx,
                subtask_constraint_slack=subtask_constraint_slack,
                budget_per_cluster=budget_per_cluster,
                cluster_target_mode=cluster_target_mode,
            )
        if mode != "prefailure_node":
            raise ValueError(
                f"analyze_failure_states: unknown mode={mode!r}. "
                "Valid modes: 'prefailure_node' (default), 'path_based'."
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
            "mode": "prefailure_node",
            "n_clusters": cluster_result.n_clusters,
            "pre_failure_nodes": [[src, tgt, float(p)] for src, tgt, p in pre_failure_edges],
            "clusters": clusters_out,
        }

    # ------------------------------------------------------------------
    # Path-based mode (Phase 1 rewrite)
    # ------------------------------------------------------------------

    def _compute_path_based(
        self,
        *,
        graph: Any,
        node_values: dict[int, float],
        labels: np.ndarray,
        metadata: list[dict[str, Any]],
        level: str,
        rollouts_hdf5: str,
        state_schema: dict[str, dict[str, int]],
        top_k_paths: int,
        path_min_edge_probability: float,
        intermediate_heuristic: str,
        kmeans_k_min: int,
        kmeans_k_max: int,
        slack_x: float,
        slack_y: float,
        slack_z_rot: float,
        subtask_constraint_idx: int | None,
        subtask_constraint_slack: float,
        budget_per_cluster: int,
        cluster_target_mode: str,
    ) -> dict[str, Any]:
        """Path-based failure targeting (see module docstring for design)."""
        # 1. Top-K failure paths.
        paths_with_prob = enumerate_failure_paths(
            graph,
            top_k=top_k_paths,
            min_edge_probability=path_min_edge_probability,
        )
        print(
            f"  [analyze_failure_states/path_based] {len(paths_with_prob)} failure paths "
            f"(top_k={top_k_paths})"
        )
        for i, (path, prob) in enumerate(paths_with_prob):
            print(f"    path {i}: p={prob:.3f}  nodes={path}")

        if not paths_with_prob:
            return {
                "enabled": True,
                "mode": "path_based",
                "paths": [],
            }

        # 2. Match each path to failure trajectories.
        path_only = [p for p, _ in paths_with_prob]
        matched = match_failure_trajectories_to_paths(
            labels, metadata, path_only, level=level
        )
        print(
            "  [analyze_failure_states/path_based] matched episodes/path: "
            f"{[len(m) for m in matched]}"
        )

        paths_out: list[dict[str, Any]] = []
        for path_idx, ((path, prob), matched_eps) in enumerate(zip(paths_with_prob, matched)):
            if not matched_eps:
                print(f"    path {path_idx}: no matched failure trajectories — skipped")
                continue

            # 3. Per-node state collection from failure trajectories.
            per_node = collect_failure_trajectory_states_by_node(
                str(rollouts_hdf5),
                labels,
                metadata,
                matched_episodes=matched_eps,
                state_schema=state_schema,
                level=level,
            )

            # 4. Pick the IC node + the intermediate node to constrain on.
            from policy_doctor.behaviors.behavior_graph import START_NODE_ID

            ic_node = START_NODE_ID
            intermediate_node = pick_intermediate_target_node(
                path, intermediate_heuristic
            )

            path_entry: dict[str, Any] = {
                "path_idx": path_idx,
                "path": [int(n) for n in path],
                "probability": float(prob),
                "matched_episodes": [int(e) for e in matched_eps],
                "intermediate_node_id": (
                    int(intermediate_node) if intermediate_node is not None else None
                ),
                "intermediate_heuristic": intermediate_heuristic,
                "ic_pool": None,
                "intermediate_pool": None,
            }

            # 5. Cluster the IC pool with silhouette-k.
            ic_feats, ic_tags = per_node.get(
                ic_node, (np.zeros((0, 4 * len(state_schema)), dtype=np.float32), [])
            )
            ic_result = silhouette_kmeans(
                ic_feats, ic_tags, node_id=ic_node,
                k_min=kmeans_k_min, k_max=kmeans_k_max,
            )
            path_entry["ic_pool"] = self._serialize_node_cluster(
                ic_result, state_schema=state_schema,
                slack_x=slack_x, slack_y=slack_y, slack_z_rot=slack_z_rot,
                subtask_constraint_idx=None,
                subtask_constraint_slack=subtask_constraint_slack,
                budget_per_cluster=budget_per_cluster,
                cluster_target_mode=cluster_target_mode,
            )

            # 6. Cluster the intermediate pool (only if a node was picked).
            if intermediate_node is not None and intermediate_node in per_node:
                mid_feats, mid_tags = per_node[intermediate_node]
                mid_result = silhouette_kmeans(
                    mid_feats, mid_tags, node_id=intermediate_node,
                    k_min=kmeans_k_min, k_max=kmeans_k_max,
                )
                path_entry["intermediate_pool"] = self._serialize_node_cluster(
                    mid_result, state_schema=state_schema,
                    slack_x=slack_x, slack_y=slack_y, slack_z_rot=slack_z_rot,
                    subtask_constraint_idx=subtask_constraint_idx,
                    subtask_constraint_slack=subtask_constraint_slack,
                    budget_per_cluster=budget_per_cluster,
                    cluster_target_mode=cluster_target_mode,
                )

            print(
                f"    path {path_idx}: matched={len(matched_eps)}  "
                f"IC k={path_entry['ic_pool']['k'] if path_entry['ic_pool'] else 0}  "
                f"intermediate node={intermediate_node} "
                f"k={path_entry['intermediate_pool']['k'] if path_entry['intermediate_pool'] else 0}"
            )

            paths_out.append(path_entry)

        return {
            "enabled": True,
            "mode": "path_based",
            "paths": paths_out,
        }

    @staticmethod
    def _serialize_node_cluster(
        result: NodeClusterResult,
        *,
        state_schema: dict[str, dict[str, int]],
        slack_x: float,
        slack_y: float,
        slack_z_rot: float,
        subtask_constraint_idx: int | None,
        subtask_constraint_slack: float,
        budget_per_cluster: int,
        cluster_target_mode: str,
    ) -> dict[str, Any]:
        """Render a :class:`NodeClusterResult` as JSON-serializable per-cluster constraints."""
        clusters_out: list[dict[str, Any]] = []
        for ci in range(result.k):
            center_feat = result.centers[ci]
            stddev_feat = result.stddevs[ci]

            # cluster_target_mode is currently always treated as "centroid" in path-based mode.
            # Sampling a member as the target would require carrying raw features through
            # NodeClusterResult; revisit if needed.
            _ = cluster_target_mode
            pose_center = cluster_center_to_object_poses(center_feat, state_schema)
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
            clusters_out.append({
                "cluster_idx": ci,
                "n_states": int((result.labels == ci).sum()),
                "eligible_rollout_idxs": result.cluster_episode_indices[ci],
                "budget_per_cluster": budget_per_cluster,
                "center_feature": [float(v) for v in center_feat],
                "stddev_feature": [float(v) for v in stddev_feat],
                "suggested_object_pose_ranges": suggested_pose_ranges,
                "suggested_subtask_constraints": suggested_subtask,
            })

        return {
            "node_id": int(result.node_id),
            "n_states": int(result.n_states),
            "k": int(result.k),
            "silhouette": float(result.silhouette) if result.silhouette == result.silhouette else None,
            "clusters": clusters_out,
        }

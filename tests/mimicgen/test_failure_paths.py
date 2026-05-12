"""Phase 1 failure-targeting unit tests.

Covers path enumeration, trajectory→path matching, per-node state collection,
silhouette-k clustering, and the intermediate-target heuristic — the building
blocks the rewritten arm uses to pick generation targets.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    START_NODE_ID,
    FAILURE_NODE_ID,
)
from policy_doctor.mimicgen.failure_targeting import (
    DEFAULT_SQUARE_STATE_SCHEMA,
    collect_failure_trajectory_states_by_node,
    enumerate_failure_paths,
    intermediate_nodes_for_path,
    match_failure_trajectories_to_paths,
    pick_intermediate_target_node,
    silhouette_kmeans,
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/mimicgen/test_heuristics.py's style)
# ---------------------------------------------------------------------------

def _flatten(
    ep_sequences: dict[int, list[int]],
    success_map: dict[int, bool],
) -> tuple[np.ndarray, list[dict]]:
    """Build (labels, metadata) from per-episode label sequences."""
    labels: list[int] = []
    metadata: list[dict] = []
    for ep_idx in sorted(ep_sequences):
        success = success_map[ep_idx]
        for t, label in enumerate(ep_sequences[ep_idx]):
            labels.append(label)
            metadata.append(
                {"rollout_idx": ep_idx, "timestep": t * 10, "success": success}
            )
    return np.array(labels, dtype=np.int64), metadata


def _build_graph(
    ep_sequences: dict[int, list[int]],
    success_map: dict[int, bool],
):
    labels, metadata = _flatten(ep_sequences, success_map)
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout"), labels, metadata


def _write_states_hdf5(
    path: Path,
    states_per_ep: dict[int, np.ndarray],
    success_map: dict[int, bool],
) -> Path:
    """Write a tiny robomimic-shaped rollout HDF5 with raw `states` arrays."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = np.int64(sum(s.shape[0] for s in states_per_ep.values()))
        data.attrs["env_args"] = json.dumps({"env_name": "Dummy", "type": 1, "env_kwargs": {}})
        for ep_idx, states in states_per_ep.items():
            ep = data.create_group(f"demo_{ep_idx}")
            ep.create_dataset("states", data=states.astype(np.float32))
            ep.attrs["num_samples"] = np.int64(states.shape[0])
            ep.attrs["success"] = np.int64(int(success_map.get(ep_idx, True)))
    return path


# ---------------------------------------------------------------------------
# enumerate_failure_paths
# ---------------------------------------------------------------------------

class TestEnumerateFailurePaths(unittest.TestCase):
    def test_two_paths_ranked_by_probability(self):
        # 2 episodes follow 0 → 1, 1 follows 0 → 2. Both end in FAILURE.
        graph, *_ = _build_graph(
            ep_sequences={0: [0, 1], 1: [0, 1], 2: [0, 2]},
            success_map={0: False, 1: False, 2: False},
        )
        paths = enumerate_failure_paths(graph, top_k=5)
        # Both should be returned, with the more frequent one first.
        self.assertEqual(len(paths), 2)
        path0, prob0 = paths[0]
        path1, prob1 = paths[1]
        self.assertEqual(path0, [START_NODE_ID, 0, 1, FAILURE_NODE_ID])
        self.assertEqual(path1, [START_NODE_ID, 0, 2, FAILURE_NODE_ID])
        self.assertGreater(prob0, prob1)
        self.assertAlmostEqual(prob0, 2 / 3, places=5)
        self.assertAlmostEqual(prob1, 1 / 3, places=5)

    def test_top_k_caps(self):
        # 3 different failure paths; top_k=2 keeps the best 2.
        graph, *_ = _build_graph(
            ep_sequences={
                0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2],   # path A ×3
                3: [0, 4], 4: [0, 4],                       # path B ×2
                5: [0, 5],                                  # path C ×1
            },
            success_map={i: False for i in range(6)},
        )
        paths = enumerate_failure_paths(graph, top_k=2)
        self.assertEqual(len(paths), 2)
        interiors = [tuple(p for p in path if p not in (START_NODE_ID, FAILURE_NODE_ID))
                     for path, _ in paths]
        self.assertEqual(interiors[0], (0, 1, 2))
        self.assertEqual(interiors[1], (0, 4))

    def test_no_failure_episodes_returns_empty(self):
        graph, *_ = _build_graph(
            ep_sequences={0: [0, 1], 1: [0, 1]},
            success_map={0: True, 1: True},
        )
        self.assertEqual(enumerate_failure_paths(graph, top_k=5), [])


# ---------------------------------------------------------------------------
# match_failure_trajectories_to_paths
# ---------------------------------------------------------------------------

class TestMatchFailureTrajectoriesToPaths(unittest.TestCase):
    def test_each_failure_matches_its_path(self):
        ep_sequences = {0: [0, 1], 1: [0, 1], 2: [0, 2]}
        labels, metadata = _flatten(ep_sequences, {0: False, 1: False, 2: False})
        graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")
        paths = [p for p, _ in enumerate_failure_paths(graph, top_k=5)]
        matched = match_failure_trajectories_to_paths(labels, metadata, paths, level="rollout")
        self.assertEqual(matched, [[0, 1], [2]])

    def test_successful_episodes_are_excluded(self):
        # Episode 2 has the same collapsed sequence as the failure path but is a SUCCESS.
        ep_sequences = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
        labels, metadata = _flatten(ep_sequences, {0: False, 1: False, 2: True})
        graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")
        paths = [p for p, _ in enumerate_failure_paths(graph, top_k=5)]
        matched = match_failure_trajectories_to_paths(labels, metadata, paths, level="rollout")
        self.assertEqual(matched, [[0, 1]])

    def test_episode_claimed_by_first_matching_path(self):
        # When two paths have the same interior, claim is first-come-first-served.
        # (This shouldn't normally happen — paths are distinct — but we want determinism.)
        ep_sequences = {0: [0, 1]}
        labels, metadata = _flatten(ep_sequences, {0: False})
        path_a = [START_NODE_ID, 0, 1, FAILURE_NODE_ID]
        path_b = [START_NODE_ID, 0, 1, FAILURE_NODE_ID]
        matched = match_failure_trajectories_to_paths(
            labels, metadata, [path_a, path_b], level="rollout"
        )
        self.assertEqual(matched, [[0], []])

    def test_run_length_collapsing(self):
        # Episode visits 0 for many timesteps, then 1, then 0 again, then fails.
        # Collapsed: [0, 1, 0].
        labels, metadata = _flatten({0: [0, 0, 0, 1, 1, 0]}, {0: False})
        path = [START_NODE_ID, 0, 1, 0, FAILURE_NODE_ID]
        matched = match_failure_trajectories_to_paths(labels, metadata, [path], level="rollout")
        self.assertEqual(matched, [[0]])


# ---------------------------------------------------------------------------
# intermediate_nodes / pick_intermediate_target_node
# ---------------------------------------------------------------------------

class TestIntermediateTargetPicker(unittest.TestCase):
    def test_closest_to_failure(self):
        path = [START_NODE_ID, 7, 11, 4, FAILURE_NODE_ID]
        self.assertEqual(pick_intermediate_target_node(path, "closest_to_failure"), 4)

    def test_first(self):
        path = [START_NODE_ID, 7, 11, 4, FAILURE_NODE_ID]
        self.assertEqual(pick_intermediate_target_node(path, "first"), 7)

    def test_middle(self):
        path = [START_NODE_ID, 7, 11, 4, FAILURE_NODE_ID]
        # len(interior)=3 → index 1 → 11.
        self.assertEqual(pick_intermediate_target_node(path, "middle"), 11)

    def test_no_interior(self):
        path = [START_NODE_ID, FAILURE_NODE_ID]
        self.assertIsNone(pick_intermediate_target_node(path, "closest_to_failure"))

    def test_intermediate_nodes_for_path(self):
        path = [START_NODE_ID, 7, 11, 4, FAILURE_NODE_ID]
        self.assertEqual(intermediate_nodes_for_path(path), [7, 11, 4])

    def test_unknown_heuristic_raises(self):
        with self.assertRaises(ValueError):
            pick_intermediate_target_node([START_NODE_ID, 1, FAILURE_NODE_ID], "??")


# ---------------------------------------------------------------------------
# silhouette_kmeans
# ---------------------------------------------------------------------------

class TestSilhouetteKMeans(unittest.TestCase):
    def test_picks_two_for_obvious_two_cluster_data(self):
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=[0.0, 0.0, 0.0, 0.0], scale=0.05, size=(20, 4))
        cluster_b = rng.normal(loc=[5.0, 5.0, 1.0, 1.0], scale=0.05, size=(20, 4))
        features = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        tags = [(i, 0) for i in range(40)]
        result = silhouette_kmeans(features, tags, node_id=7, k_min=2, k_max=6)
        self.assertEqual(result.k, 2)
        self.assertGreater(result.silhouette, 0.8)   # very well separated
        self.assertEqual(result.centers.shape, (2, 4))
        self.assertEqual(result.stddevs.shape, (2, 4))

    def test_picks_three_for_three_cluster_data(self):
        rng = np.random.default_rng(1)
        clusters = [
            rng.normal(loc=[c, 0.0, 0.0, 0.0], scale=0.05, size=(15, 4))
            for c in (0.0, 5.0, 10.0)
        ]
        features = np.vstack(clusters).astype(np.float32)
        tags = [(i, 0) for i in range(45)]
        result = silhouette_kmeans(features, tags, node_id=7, k_min=2, k_max=8)
        self.assertEqual(result.k, 3)

    def test_empty(self):
        result = silhouette_kmeans(
            np.zeros((0, 4), dtype=np.float32), [], node_id=7, k_min=2, k_max=10
        )
        self.assertEqual(result.k, 0)
        self.assertEqual(result.n_states, 0)

    def test_single_point(self):
        result = silhouette_kmeans(
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            [(0, 5)], node_id=7, k_min=2, k_max=10,
        )
        self.assertEqual(result.k, 1)
        self.assertEqual(result.cluster_episode_indices, [[0]])

    def test_few_points_falls_back(self):
        # Only 2 points, k_min=2 forces k=k_upper=1 → single-cluster fallback.
        result = silhouette_kmeans(
            np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            [(0, 0), (1, 0)], node_id=7, k_min=2, k_max=10,
        )
        # With n=2 and k_min=2, k_upper = n-1 = 1 < k_min → single-cluster fallback.
        self.assertEqual(result.k, 1)
        self.assertEqual(result.n_states, 2)
        self.assertEqual(set(int(e) for e in result.cluster_episode_indices[0]), {0, 1})


# ---------------------------------------------------------------------------
# collect_failure_trajectory_states_by_node
# ---------------------------------------------------------------------------

class TestCollectFailureTrajectoryStates(unittest.TestCase):
    """End-to-end node-state collection from a tiny rollout HDF5.

    Builds two failure episodes whose collapsed cluster sequence is [0, 1] —
    both should match the single failure path, contribute t=0 states to the
    START pool, and contribute timestep-by-timestep states to nodes 0 and 1.
    """

    def setUp(self):
        # 2 failure episodes with 4 timesteps each.
        # All states are 17-dim qpos slices to match DEFAULT_SQUARE_STATE_SCHEMA
        # (which reads indices 10/11/13/14/15/16).
        self.state_dim = 20
        rng = np.random.default_rng(0)
        ep0_states = rng.normal(size=(4, self.state_dim)).astype(np.float32)
        ep1_states = rng.normal(size=(4, self.state_dim)).astype(np.float32)
        # Make t=0 of each episode distinctive at idx 10/11 so we can verify identity.
        ep0_states[0, 10:12] = [0.10, 0.20]
        ep1_states[0, 10:12] = [0.30, 0.40]
        self.states = {0: ep0_states, 1: ep1_states}

        # Cluster sequence per episode: [0, 0, 1, 1].
        # Collapsed → [0, 1].
        self.labels, self.metadata = _flatten(
            {0: [0, 0, 1, 1], 1: [0, 0, 1, 1]},
            success_map={0: False, 1: False},
        )
        # However: metadata uses "timestep" key (= t*10 by _flatten). For the
        # HDF5 lookup we need the timestep index to fall within states.shape[0]=4.
        # _flatten writes t*10 which would clamp; rewrite to identity so the
        # tests against actual timesteps are unambiguous.
        for i, m in enumerate(self.metadata):
            m["timestep"] = i % 4
        self.tmpdir = tempfile.TemporaryDirectory()
        self.h5_path = Path(self.tmpdir.name) / "rollouts.hdf5"
        _write_states_hdf5(self.h5_path, self.states, {0: False, 1: False})

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_start_pool_has_t0_states_for_each_matched_episode(self):
        out = collect_failure_trajectory_states_by_node(
            str(self.h5_path), self.labels, self.metadata,
            matched_episodes=[0, 1],
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
            level="rollout",
        )
        self.assertIn(START_NODE_ID, out)
        feats, tags = out[START_NODE_ID]
        self.assertEqual(feats.shape, (2, 4))  # 2 episodes × 4-dim Square features
        self.assertEqual([ep for ep, _ in tags], [0, 1])
        # Verify the x/y values from t=0 propagated through the encoding.
        self.assertAlmostEqual(float(feats[0, 0]), 0.10, places=4)
        self.assertAlmostEqual(float(feats[0, 1]), 0.20, places=4)
        self.assertAlmostEqual(float(feats[1, 0]), 0.30, places=4)
        self.assertAlmostEqual(float(feats[1, 1]), 0.40, places=4)

    def test_intermediate_nodes_get_all_matching_timesteps(self):
        out = collect_failure_trajectory_states_by_node(
            str(self.h5_path), self.labels, self.metadata,
            matched_episodes=[0, 1],
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
            level="rollout",
        )
        # Each episode contributes 2 timesteps to node 0 (t=0,1) and 2 to node 1 (t=2,3).
        self.assertIn(0, out)
        self.assertIn(1, out)
        feats0, tags0 = out[0]
        feats1, tags1 = out[1]
        self.assertEqual(feats0.shape[0], 4)
        self.assertEqual(feats1.shape[0], 4)
        # Sanity check episode coverage on each node.
        self.assertEqual(sorted({ep for ep, _ in tags0}), [0, 1])
        self.assertEqual(sorted({ep for ep, _ in tags1}), [0, 1])

    def test_unmatched_episode_excluded(self):
        # Only include episode 0; episode 1's states should not appear anywhere.
        out = collect_failure_trajectory_states_by_node(
            str(self.h5_path), self.labels, self.metadata,
            matched_episodes=[0],
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
            level="rollout",
        )
        for node_id, (_feats, tags) in out.items():
            self.assertNotIn(1, {ep for ep, _ in tags},
                             f"node {node_id} accidentally collected ep=1")


# ---------------------------------------------------------------------------
# AnalyzeFailureStatesStep._compute_path_based — integration
# ---------------------------------------------------------------------------

class TestAnalyzeFailureStatesPathBased(unittest.TestCase):
    """End-to-end: build a graph + tiny HDF5, call the path-based compute
    helper directly, verify the per-path/per-cluster output schema and
    contents."""

    def setUp(self):
        # Topology: 4 failure episodes follow [0, 1] (the dominant path),
        # 2 follow [0, 2] (rarer). Success episodes are excluded from matching.
        self.ep_sequences = {
            0: [0, 0, 1, 1],
            1: [0, 0, 1, 1],
            2: [0, 0, 1, 1],
            3: [0, 0, 1, 1],
            4: [0, 0, 2, 2],
            5: [0, 0, 2, 2],
            6: [0, 0, 1, 1],  # success: should NOT match the failure path
        }
        self.success_map = {0: False, 1: False, 2: False, 3: False,
                            4: False, 5: False, 6: True}
        self.labels, self.metadata = _flatten(self.ep_sequences, self.success_map)
        # Make timesteps identity for direct lookup into HDF5.
        for i, m in enumerate(self.metadata):
            m["timestep"] = i % 4

        self.graph = BehaviorGraph.from_cluster_assignments(
            self.labels, self.metadata, level="rollout"
        )
        self.node_values = self.graph.compute_values()

        # Distinct t=0 / intermediate poses per episode so we can see clusters separate.
        rng = np.random.default_rng(0)
        self.states = {}
        for ep, seq in self.ep_sequences.items():
            arr = rng.normal(scale=0.02, size=(len(seq), 20)).astype(np.float32)
            # x at idx 10 differs by episode (group eps by collapsed seq).
            arr[:, 10] = 0.10 + 0.001 * ep
            arr[:, 11] = 0.20 + 0.001 * ep
            self.states[ep] = arr

        self.tmpdir = tempfile.TemporaryDirectory()
        self.h5_path = Path(self.tmpdir.name) / "rollouts.hdf5"
        _write_states_hdf5(self.h5_path, self.states, self.success_map)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_path_based_end_to_end(self):
        # Construct the step manually (bypass parent_run_dir; we only call the helper).
        from policy_doctor.curation_pipeline.steps.analyze_failure_states import (
            AnalyzeFailureStatesStep,
        )

        # Create a minimal-cfg instance just to call the helper. The helper is a
        # bound method, but it doesn't read self.cfg / self.parent_run_dir — it
        # takes everything via kwargs.
        step = AnalyzeFailureStatesStep.__new__(AnalyzeFailureStatesStep)

        result = step._compute_path_based(
            graph=self.graph,
            node_values=self.node_values,
            labels=self.labels,
            metadata=self.metadata,
            level="rollout",
            rollouts_hdf5=str(self.h5_path),
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
            top_k_paths=5,
            path_min_edge_probability=0.0,
            intermediate_heuristic="closest_to_failure",
            kmeans_k_min=2,
            kmeans_k_max=4,
            slack_x=0.03, slack_y=0.03, slack_z_rot=0.5,
            subtask_constraint_idx=0,
            subtask_constraint_slack=1.5,
            budget_per_cluster=4,
            cluster_target_mode="centroid",
        )

        self.assertTrue(result["enabled"])
        self.assertEqual(result["mode"], "path_based")
        self.assertEqual(len(result["paths"]), 2)

        # Path 0 should be the more probable one (interior [0, 1]).
        p0 = result["paths"][0]
        self.assertEqual(p0["path_idx"], 0)
        self.assertGreater(p0["probability"], result["paths"][1]["probability"])
        self.assertEqual(p0["intermediate_node_id"], 1)
        self.assertEqual(p0["intermediate_heuristic"], "closest_to_failure")
        # 4 failure episodes follow this path; success ep 6 must NOT be claimed.
        self.assertEqual(set(p0["matched_episodes"]), {0, 1, 2, 3})
        self.assertNotIn(6, p0["matched_episodes"])

        # IC pool: 4 episodes, each contributing a t=0 state.
        ic = p0["ic_pool"]
        self.assertEqual(ic["n_states"], 4)
        self.assertGreaterEqual(ic["k"], 1)
        # Each cluster entry should have the constraint structure.
        for c in ic["clusters"]:
            self.assertIn("suggested_object_pose_ranges", c)
            self.assertIn("center_feature", c)
            self.assertIn("stddev_feature", c)
            self.assertGreater(c["n_states"], 0)

        # Intermediate pool: cluster 1 timesteps from each matched episode.
        mid = p0["intermediate_pool"]
        self.assertIsNotNone(mid)
        self.assertEqual(mid["node_id"], 1)
        self.assertGreater(mid["n_states"], 0)
        # Subtask constraints should be set (subtask_constraint_idx=0).
        self.assertIsNotNone(mid["clusters"][0]["suggested_subtask_constraints"])
        self.assertIn("0", mid["clusters"][0]["suggested_subtask_constraints"])

        # Path 1 is the rarer one (interior [0, 2]); should have 2 matched eps.
        p1 = result["paths"][1]
        self.assertEqual(set(p1["matched_episodes"]), {4, 5})
        self.assertEqual(p1["intermediate_node_id"], 2)


if __name__ == "__main__":
    unittest.main()

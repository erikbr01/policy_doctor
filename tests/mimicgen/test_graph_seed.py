"""Unit tests for policy_doctor.mimicgen.graph_seed."""

from __future__ import annotations

import unittest

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    START_NODE_ID,
    SUCCESS_NODE_ID,
    FAILURE_NODE_ID,
    BehaviorGraph,
)
from policy_doctor.mimicgen.graph_seed import (
    episode_success_map,
    find_rollouts_for_path,
    path_cluster_sequence,
    reconstruct_episode_sequences,
    top_paths_with_rollouts,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic cluster data
# ---------------------------------------------------------------------------

def _make_flat_data(
    ep_sequences: dict[int, list[int]],
    success_map: dict[int, bool] | None = None,
    level: str = "rollout",
    timestep_step: int = 10,
) -> tuple[np.ndarray, list[dict]]:
    """Build flat (cluster_labels, metadata) from per-episode sequences.

    Each episode produces one sample per cluster label in its sequence.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    labels: list[int] = []
    metadata: list[dict] = []
    for ep_idx, seq in ep_sequences.items():
        success = (success_map or {}).get(ep_idx, True)
        for t, label in enumerate(seq):
            labels.append(label)
            metadata.append({
                ep_key: ep_idx,
                "window_start": t * timestep_step,
                "success": success,
            })
    return np.array(labels, dtype=np.int64), metadata


def _make_graph_with_data(
    ep_sequences: dict[int, list[int]],
    success_map: dict[int, bool] | None = None,
    level: str = "rollout",
) -> tuple[BehaviorGraph, np.ndarray, list[dict]]:
    labels, metadata = _make_flat_data(ep_sequences, success_map, level)
    graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
    return graph, labels, metadata


# ---------------------------------------------------------------------------
# reconstruct_episode_sequences
# ---------------------------------------------------------------------------

class TestReconstructEpisodeSequences(unittest.TestCase):
    def test_basic_single_episode(self):
        labels = np.array([0, 1, 2])
        metadata = [
            {"rollout_idx": 0, "window_start": 0},
            {"rollout_idx": 0, "window_start": 10},
            {"rollout_idx": 0, "window_start": 20},
        ]
        seqs = reconstruct_episode_sequences(labels, metadata)
        self.assertEqual(seqs[0], [0, 1, 2])

    def test_run_length_encoding(self):
        """Consecutive identical labels should be collapsed."""
        labels = np.array([3, 3, 3, 1, 1, 2])
        metadata = [
            {"rollout_idx": 0, "window_start": i * 5}
            for i in range(6)
        ]
        seqs = reconstruct_episode_sequences(labels, metadata)
        self.assertEqual(seqs[0], [3, 1, 2])

    def test_multiple_episodes(self):
        ep_seqs = {0: [1, 2], 1: [1, 3, 2]}
        labels, metadata = _make_flat_data(ep_seqs)
        seqs = reconstruct_episode_sequences(labels, metadata)
        self.assertEqual(seqs[0], [1, 2])
        self.assertEqual(seqs[1], [1, 3, 2])

    def test_noise_samples_excluded(self):
        labels = np.array([-1, 0, -1, 1])
        metadata = [
            {"rollout_idx": 0, "window_start": i * 10}
            for i in range(4)
        ]
        seqs = reconstruct_episode_sequences(labels, metadata)
        self.assertEqual(seqs[0], [0, 1])

    def test_demo_level_uses_demo_idx(self):
        labels = np.array([5, 6])
        metadata = [
            {"demo_idx": 7, "window_start": 0},
            {"demo_idx": 7, "window_start": 10},
        ]
        seqs = reconstruct_episode_sequences(labels, metadata, level="demo")
        self.assertIn(7, seqs)
        self.assertEqual(seqs[7], [5, 6])

    def test_timestep_fallback(self):
        """Should order by 'timestep' when 'window_start' is absent."""
        labels = np.array([2, 1])
        metadata = [
            {"rollout_idx": 0, "timestep": 5},
            {"rollout_idx": 0, "timestep": 1},
        ]
        seqs = reconstruct_episode_sequences(labels, metadata)
        # Sorted by timestep: label@t=1 is 1, label@t=5 is 2 → [1, 2]
        self.assertEqual(seqs[0], [1, 2])


# ---------------------------------------------------------------------------
# episode_success_map
# ---------------------------------------------------------------------------

class TestEpisodeSuccessMap(unittest.TestCase):
    def test_basic(self):
        metadata = [
            {"rollout_idx": 0, "success": True},
            {"rollout_idx": 0, "success": True},  # duplicate — ignored
            {"rollout_idx": 1, "success": False},
        ]
        sm = episode_success_map(metadata)
        self.assertIs(sm[0], True)
        self.assertIs(sm[1], False)

    def test_missing_success_key(self):
        metadata = [{"rollout_idx": 0}]
        sm = episode_success_map(metadata)
        self.assertIsNone(sm[0])

    def test_first_occurrence_wins(self):
        metadata = [
            {"rollout_idx": 0, "success": True},
            {"rollout_idx": 0, "success": False},
        ]
        sm = episode_success_map(metadata)
        self.assertIs(sm[0], True)


# ---------------------------------------------------------------------------
# path_cluster_sequence
# ---------------------------------------------------------------------------

class TestPathClusterSequence(unittest.TestCase):
    def test_strips_start_and_terminal(self):
        path = [START_NODE_ID, 3, 1, 5, SUCCESS_NODE_ID]
        self.assertEqual(path_cluster_sequence(path), [3, 1, 5])

    def test_failure_node_stripped(self):
        path = [START_NODE_ID, 2, 4, FAILURE_NODE_ID]
        self.assertEqual(path_cluster_sequence(path), [2, 4])

    def test_empty_path(self):
        self.assertEqual(path_cluster_sequence([]), [])

    def test_all_special_stripped(self):
        path = [START_NODE_ID, SUCCESS_NODE_ID]
        self.assertEqual(path_cluster_sequence(path), [])


# ---------------------------------------------------------------------------
# find_rollouts_for_path
# ---------------------------------------------------------------------------

class TestFindRolloutsForPath(unittest.TestCase):
    def setUp(self):
        self.ep_seqs = {
            0: [1, 2, 3],
            1: [1, 3, 3],
            2: [1, 2, 3],   # matches same as ep 0
            3: [4, 5],
        }
        self.success_map = {0: True, 1: True, 2: False, 3: True}

    def test_exact_match(self):
        path = [START_NODE_ID, 1, 2, 3, SUCCESS_NODE_ID]
        idxs = find_rollouts_for_path(path, self.ep_seqs, success_only=False)
        self.assertIn(0, idxs)
        self.assertIn(2, idxs)
        self.assertNotIn(1, idxs)

    def test_success_only_filters_failures(self):
        path = [START_NODE_ID, 1, 2, 3, SUCCESS_NODE_ID]
        idxs = find_rollouts_for_path(
            path, self.ep_seqs,
            success_only=True,
            success_map=self.success_map,
        )
        self.assertIn(0, idxs)
        self.assertNotIn(2, idxs)   # ep 2 has success=False

    def test_no_match_returns_empty(self):
        path = [START_NODE_ID, 99, 99, SUCCESS_NODE_ID]
        idxs = find_rollouts_for_path(path, self.ep_seqs, success_only=False)
        self.assertEqual(idxs, [])

    def test_returns_sorted(self):
        path = [START_NODE_ID, 1, 2, 3, SUCCESS_NODE_ID]
        idxs = find_rollouts_for_path(path, self.ep_seqs, success_only=False)
        self.assertEqual(idxs, sorted(idxs))


# ---------------------------------------------------------------------------
# top_paths_with_rollouts
# ---------------------------------------------------------------------------

class TestTopPathsWithRollouts(unittest.TestCase):
    def _build(self):
        ep_seqs = {
            0: [1, 2],   # successful
            1: [1, 2],   # successful
            2: [1, 3],   # failed
            3: [3, 2],   # successful
        }
        success_map = {0: True, 1: True, 2: False, 3: True}
        graph, labels, metadata = _make_graph_with_data(ep_seqs, success_map)
        return graph, labels, metadata

    def test_returns_list_of_dicts(self):
        graph, labels, metadata = self._build()
        results = top_paths_with_rollouts(graph, labels, metadata, top_k=5)
        self.assertIsInstance(results, list)
        for entry in results:
            self.assertIn("path", entry)
            self.assertIn("path_prob", entry)
            self.assertIn("cluster_seq", entry)
            self.assertIn("rollout_idxs", entry)
            self.assertIn("has_match", entry)

    def test_top_k_limits_results(self):
        graph, labels, metadata = self._build()
        results = top_paths_with_rollouts(graph, labels, metadata, top_k=1)
        self.assertLessEqual(len(results), 1)

    def test_has_match_set_correctly(self):
        graph, labels, metadata = self._build()
        results = top_paths_with_rollouts(
            graph, labels, metadata, top_k=5, success_only=True
        )
        for entry in results:
            if entry["rollout_idxs"]:
                self.assertTrue(entry["has_match"])
            else:
                self.assertFalse(entry["has_match"])

    def test_paths_reach_success(self):
        """All returned paths should end at SUCCESS_NODE_ID."""
        graph, labels, metadata = self._build()
        results = top_paths_with_rollouts(graph, labels, metadata, top_k=5)
        for entry in results:
            self.assertEqual(entry["path"][-1], SUCCESS_NODE_ID)

    def test_success_only_false_includes_failed_rollouts(self):
        # ep 0 succeeds (creates path to SUCCESS), ep 1 fails but same cluster seq.
        # With success_only=False, both eps should match the path.
        ep_seqs = {0: [1, 2], 1: [1, 2]}
        success_map = {0: True, 1: False}
        graph, labels, metadata = _make_graph_with_data(ep_seqs, success_map)
        # success_only=True should only return ep 0
        results_strict = top_paths_with_rollouts(
            graph, labels, metadata, top_k=5, success_only=True
        )
        matching_strict = [e for e in results_strict if e["has_match"]]
        self.assertTrue(len(matching_strict) > 0)
        rollout_idxs_strict = matching_strict[0]["rollout_idxs"]
        self.assertIn(0, rollout_idxs_strict)
        self.assertNotIn(1, rollout_idxs_strict)

        # success_only=False should include ep 1 as well
        results_any = top_paths_with_rollouts(
            graph, labels, metadata, top_k=5, success_only=False
        )
        matching_any = [e for e in results_any if e["has_match"]]
        self.assertTrue(len(matching_any) > 0)
        rollout_idxs_any = matching_any[0]["rollout_idxs"]
        self.assertIn(0, rollout_idxs_any)
        self.assertIn(1, rollout_idxs_any)

    def test_cluster_seq_matches_path_without_special_nodes(self):
        graph, labels, metadata = self._build()
        results = top_paths_with_rollouts(graph, labels, metadata, top_k=5)
        for entry in results:
            expected = path_cluster_sequence(entry["path"])
            self.assertEqual(entry["cluster_seq"], expected)


if __name__ == "__main__":
    unittest.main()

"""Tests for behavior graph node matching across flywheel iterations."""

import unittest

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.behaviors.graph_matching import (
    GraphMatchResult,
    NodeMatch,
    build_tracking_chains,
    match_graph_sequence,
    match_graphs,
    match_graphs_ensemble,
)


def _make_window_metadata(n_rollouts: int, episode_length: int, window_width: int, stride: int):
    """Generate synthetic sliding-window metadata."""
    metadata = []
    labels = []
    for ep_idx in range(n_rollouts):
        for start in range(0, episode_length - window_width + 1, stride):
            metadata.append({
                "rollout_idx": ep_idx,
                "window_start": start,
                "window_end": start + window_width,
                "window_width": window_width,
                "success": ep_idx % 2 == 0,
            })
            # Assign cluster based on temporal position
            labels.append(0 if start < episode_length // 2 else 1)
    return np.array(labels, dtype=np.int32), metadata


def _make_graph(cluster_seq_per_ep, success_per_ep, n_timesteps_per_cluster=10):
    """Build a BehaviorGraph from explicit per-episode cluster sequences."""
    metadata = []
    labels = []
    ts = 0
    for ep_idx, (seq, success) in enumerate(zip(cluster_seq_per_ep, success_per_ep)):
        for cid in seq:
            for t in range(n_timesteps_per_cluster):
                metadata.append({
                    "rollout_idx": ep_idx,
                    "window_start": ts,
                    "window_end": ts + 1,
                    "window_width": 1,
                    "success": success,
                })
                labels.append(cid)
                ts += 1
    labels_arr = np.array(labels, dtype=np.int32)
    return BehaviorGraph.from_cluster_assignments(labels_arr, metadata), labels_arr, metadata


class TestMatchGraphsIdentity(unittest.TestCase):
    """Matching a graph against itself should give a perfect all-zero-distance result."""

    def setUp(self):
        self.graph, self.labels, self.meta = _make_graph(
            cluster_seq_per_ep=[[0, 1, 2], [0, 1, 2], [0, 2]],
            success_per_ep=[True, True, False],
        )

    def _assert_perfect_match(self, result):
        self.assertEqual(len(result.unmatched_a), 0)
        self.assertEqual(len(result.unmatched_b), 0)
        matched = {m.node_id_a: m.node_id_b for m in result.matches}
        for nid in self.graph.cluster_nodes:
            self.assertEqual(matched[nid], nid)
        for m in result.matches:
            self.assertAlmostEqual(m.distance, 0.0, places=5)

    def test_structural_identity(self):
        result = match_graphs(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            method="structural", ratio=1.0,
        )
        self._assert_perfect_match(result)

    def test_temporal_identity(self):
        result = match_graphs(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            method="temporal", ratio=1.0,
        )
        self._assert_perfect_match(result)

    def test_combined_identity(self):
        result = match_graphs(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            method="combined", ratio=1.0,
        )
        self._assert_perfect_match(result)

    def test_state_action_identity(self):
        n = len(self.labels)
        obs = {ep_idx: np.random.randn(30, 5).astype(np.float32) for ep_idx in range(3)}
        result = match_graphs(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            method="state_action", ratio=1.0, obs_a=obs, obs_b=obs,
        )
        self._assert_perfect_match(result)

    def test_state_structural_identity(self):
        obs = {ep_idx: np.random.randn(30, 5).astype(np.float32) for ep_idx in range(3)}
        result = match_graphs(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            method="state_structural", ratio=1.0, obs_a=obs, obs_b=obs,
        )
        self._assert_perfect_match(result)


class TestMatchGraphsPermutation(unittest.TestCase):
    """Matching a graph against a cluster-ID-permuted copy should recover the permutation."""

    def setUp(self):
        graph_a, labels_a, meta_a = _make_graph(
            cluster_seq_per_ep=[[0, 1, 2], [0, 1, 2]],
            success_per_ep=[True, False],
        )
        # Permute: 0→5, 1→3, 2→7
        perm = {0: 5, 1: 3, 2: 7}
        labels_b = np.array([perm[l] for l in labels_a], dtype=np.int32)
        meta_b = meta_a  # same episodes
        graph_b = BehaviorGraph.from_cluster_assignments(labels_b, meta_b)
        self.graph_a, self.labels_a, self.meta_a = graph_a, labels_a, meta_a
        self.graph_b, self.labels_b, self.meta_b = graph_b, labels_b, meta_b
        self.perm = perm

    def _assert_correct_permutation(self, result):
        for m in result.matches:
            self.assertEqual(self.perm.get(m.node_id_a), m.node_id_b,
                             f"Expected {m.node_id_a}→{self.perm[m.node_id_a]}, got {m.node_id_b}")

    def test_structural_permutation(self):
        result = match_graphs(
            self.graph_a, self.meta_a, self.labels_a,
            self.graph_b, self.meta_b, self.labels_b,
            method="structural", ratio=1.0,
        )
        self.assertGreater(len(result.matches), 0)
        self._assert_correct_permutation(result)

    def test_temporal_permutation(self):
        result = match_graphs(
            self.graph_a, self.meta_a, self.labels_a,
            self.graph_b, self.meta_b, self.labels_b,
            method="temporal", ratio=1.0,
        )
        self.assertGreater(len(result.matches), 0)
        self._assert_correct_permutation(result)

    def test_combined_permutation(self):
        result = match_graphs(
            self.graph_a, self.meta_a, self.labels_a,
            self.graph_b, self.meta_b, self.labels_b,
            method="combined", ratio=1.0,
        )
        self.assertGreater(len(result.matches), 0)
        self._assert_correct_permutation(result)


class TestUnmatchedNodes(unittest.TestCase):
    """Nodes present in one graph but absent in the other should appear as unmatched."""

    def test_extra_node_in_b(self):
        graph_a, labels_a, meta_a = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        # graph_b has an extra cluster 2 that never appears in graph_a
        graph_b, labels_b, meta_b = _make_graph(
            cluster_seq_per_ep=[[0, 1, 2], [0, 1, 2]],
            success_per_ep=[True, False],
        )
        result = match_graphs(
            graph_a, meta_a, labels_a,
            graph_b, meta_b, labels_b,
            method="structural", ratio=1.0, max_distance=None,
        )
        # graph_a has 2 nodes, graph_b has 3: exactly one node in b must be unmatched
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(len(result.unmatched_b), 1)
        self.assertEqual(len(result.unmatched_a), 0)

    def test_node_disappears(self):
        graph_a, labels_a, meta_a = _make_graph(
            cluster_seq_per_ep=[[0, 1, 2], [0, 1, 2]],
            success_per_ep=[True, False],
        )
        # graph_b has only 2 clusters
        graph_b, labels_b, meta_b = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        result = match_graphs(
            graph_a, meta_a, labels_a,
            graph_b, meta_b, labels_b,
            method="structural", ratio=1.0, max_distance=None,
        )
        # graph_a has 3 nodes, graph_b has 2: exactly one node in a must be unmatched
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(len(result.unmatched_a), 1)
        self.assertEqual(len(result.unmatched_b), 0)


class TestRatioTestFiltering(unittest.TestCase):
    """When two nodes are equidistant, the ratio test should reject the match."""

    def test_ratio_rejects_ambiguous_match(self):
        # Build two identical graphs — structural descriptors for nodes 0 and 1
        # will be identical (same structure), so ratio ≈ 1.0 → rejected with ratio=0.8
        graph_a, labels_a, meta_a = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1], [0, 1], [0, 1]],
            success_per_ep=[True, True, True, True],
        )
        # Permuted: 0→0, 1→1 but both have identical structural descriptors
        # (same graph structure, so matching is ambiguous)
        # Use strict ratio to force rejection
        result_strict = match_graphs(
            graph_a, meta_a, labels_a,
            graph_a, meta_a, labels_a,
            method="structural", ratio=0.0,  # reject everything
        )
        # With ratio=0.0, nothing should pass (d/d2 is always > 0)
        # In degenerate case (only 1 node in B), ratio test is not applied
        if len(graph_a.cluster_nodes) >= 2:
            self.assertEqual(len(result_strict.matches), 0)

    def test_ratio_1_accepts_all(self):
        graph_a, labels_a, meta_a = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        result = match_graphs(
            graph_a, meta_a, labels_a,
            graph_a, meta_a, labels_a,
            method="combined", ratio=1.0,
        )
        self.assertEqual(len(result.unmatched_a), 0)
        self.assertEqual(len(result.unmatched_b), 0)


class TestMaxDistanceFiltering(unittest.TestCase):
    """max_distance should reject matches whose distance exceeds the threshold."""

    def test_max_distance_rejects_nonzero(self):
        # Match graph_a against a permuted copy; distances > 0. max_distance just
        # below the minimum non-zero distance should reject all matches.
        graph_a, labels_a, meta_a = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        graph_b, labels_b, meta_b = _make_graph(
            cluster_seq_per_ep=[[10, 11], [10, 11]],
            success_per_ep=[True, False],
        )
        # max_distance=1e-9 should reject matches whose structural distance > 0
        result_tight = match_graphs(
            graph_a, meta_a, labels_a,
            graph_b, meta_b, labels_b,
            method="structural", ratio=1.0, max_distance=1e-9,
        )
        result_loose = match_graphs(
            graph_a, meta_a, labels_a,
            graph_b, meta_b, labels_b,
            method="structural", ratio=1.0, max_distance=None,
        )
        # Permuted graph has identical structure — distances should be ≈ 0
        # (same graph structure, different cluster IDs). With tight threshold,
        # only exact matches (distance ≤ 1e-9) survive.
        # The key property: loose finds at least as many matches as tight.
        self.assertGreaterEqual(len(result_loose.matches), len(result_tight.matches))


class TestMatchGraphSequence(unittest.TestCase):
    """match_graph_sequence should produce N-1 results for N graphs."""

    def test_sequence_length(self):
        graph, labels, meta = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        graphs = [graph, graph, graph]
        results = match_graph_sequence(
            graphs, [meta, meta, meta], [labels, labels, labels], method="structural"
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, GraphMatchResult)

    def test_single_graph_returns_empty(self):
        graph, labels, meta = _make_graph([[0]], [True])
        results = match_graph_sequence([graph], [meta], [labels])
        self.assertEqual(results, [])


class TestBuildTrackingChains(unittest.TestCase):
    """build_tracking_chains should correctly propagate node identities."""

    def _simple_result(self, matches, unmatched_a, unmatched_b):
        return GraphMatchResult(
            matches=[NodeMatch(a, b, 0.0, "test") for a, b in matches],
            unmatched_a=unmatched_a,
            unmatched_b=unmatched_b,
            method="test",
        )

    def test_three_iter_no_gaps(self):
        # iter_0: {0,1,2}, iter_1: {3,4,5}, iter_2: {6,7,8}
        # 0→3→6, 1→4→7, 2→5→8
        r01 = self._simple_result([(0,3),(1,4),(2,5)], [], [])
        r12 = self._simple_result([(3,6),(4,7),(5,8)], [], [])
        chains = build_tracking_chains([r01, r12], ["iter_0", "iter_1", "iter_2"])
        self.assertEqual(len(chains), 3)
        expected = sorted([[0,3,6],[1,4,7],[2,5,8]])
        self.assertEqual(sorted(chains), expected)

    def test_node_disappears(self):
        # iter_0: {0,1}, iter_1: {2,3}, but node 1 is unmatched
        r01 = self._simple_result([(0,2)], unmatched_a=[1], unmatched_b=[3])
        chains = build_tracking_chains([r01], ["iter_0", "iter_1"])
        # Should have: [0,2], [1,None], [None,3]
        self.assertEqual(len(chains), 3)
        chain_map = {c[0]: c for c in chains if c[0] is not None}
        self.assertEqual(chain_map[0], [0, 2])
        self.assertEqual(chain_map[1], [1, None])
        none_start = [c for c in chains if c[0] is None]
        self.assertEqual(len(none_start), 1)
        self.assertEqual(none_start[0][1], 3)

    def test_empty_match_results(self):
        chains = build_tracking_chains([], [])
        self.assertEqual(chains, [])

    def test_length_mismatch_raises(self):
        r = self._simple_result([], [], [])
        with self.assertRaises(ValueError):
            build_tracking_chains([r, r], ["iter_0", "iter_1"])  # 2 results, 2 ids → needs 3 ids

    def test_three_iters_with_gap(self):
        # 0→2, 1 disappears at iter_1, 3 appears at iter_1, 2→4, 3 disappears at iter_2
        r01 = self._simple_result([(0,2)], unmatched_a=[1], unmatched_b=[3])
        r12 = self._simple_result([(2,4)], unmatched_a=[3], unmatched_b=[])
        chains = build_tracking_chains([r01, r12], ["i0", "i1", "i2"])
        self.assertEqual(len(chains), 3)
        # identity of node 0: [0, 2, 4]
        self.assertIn([0, 2, 4], chains)
        # identity of node 1: [1, None, None]
        self.assertIn([1, None, None], chains)
        # identity of new node 3: [None, 3, None]
        self.assertIn([None, 3, None], chains)


class TestStateActionDescriptor(unittest.TestCase):
    """State-action matching should use obs arrays correctly."""

    def test_obs_agg_variants(self):
        graph, labels, meta = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        obs = {0: np.ones((20, 3), dtype=np.float32), 1: np.ones((20, 3), dtype=np.float32) * 2}
        for agg in ("mean", "mean_std", "temporal_profile"):
            with self.subTest(obs_agg=agg):
                result = match_graphs(
                    graph, meta, labels,
                    graph, meta, labels,
                    method="state_action", ratio=1.0, obs_a=obs, obs_b=obs, obs_agg=agg,
                )
                self.assertIsInstance(result, GraphMatchResult)

    def test_missing_obs_raises(self):
        graph, labels, meta = _make_graph([[0, 1]], [True])
        with self.assertRaises(ValueError):
            match_graphs(graph, meta, labels, graph, meta, labels, method="state_action")

    def test_state_structural_obs_agg_mean(self):
        graph, labels, meta = _make_graph(
            cluster_seq_per_ep=[[0, 1], [0, 1]],
            success_per_ep=[True, False],
        )
        obs = {0: np.random.randn(20, 4).astype(np.float32),
               1: np.random.randn(20, 4).astype(np.float32)}
        result = match_graphs(
            graph, meta, labels,
            graph, meta, labels,
            method="state_structural", ratio=1.0, obs_a=obs, obs_b=obs,
        )
        self.assertEqual(len(result.unmatched_a), 0)
        self.assertEqual(len(result.unmatched_b), 0)


class TestEnsembleMatching(unittest.TestCase):
    """match_graphs_ensemble should populate NodeMatch.confidence and resolve conflicts."""

    def setUp(self):
        self.graph, self.labels, self.meta = _make_graph(
            cluster_seq_per_ep=[[0, 1, 2], [0, 1, 2]],
            success_per_ep=[True, False],
        )

    def test_single_method_confidence_is_1(self):
        result = match_graphs_ensemble(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            methods=["structural"],
        )
        self.assertEqual(result.method, "ensemble")
        for m in result.matches:
            self.assertAlmostEqual(m.confidence, 1.0)

    def test_two_agreeing_methods_confidence_is_1(self):
        result = match_graphs_ensemble(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            methods=["structural", "temporal"],
            ratio=1.0,
        )
        self.assertGreater(len(result.matches), 0)
        for m in result.matches:
            self.assertAlmostEqual(m.confidence, 1.0)
            self.assertIsNotNone(m.confidence)

    def test_single_method_match_graphs_confidence_is_none(self):
        # Existing single-method API should leave confidence=None
        result = match_graphs(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            method="structural", ratio=1.0,
        )
        for m in result.matches:
            self.assertIsNone(m.confidence)

    def test_min_agreement_filters_low_confidence(self):
        # Run ensemble with ["structural", "temporal"] on a permuted graph — both should agree.
        perm = {0: 5, 1: 3, 2: 7}
        labels_b = np.array([perm[l] for l in self.labels], dtype=np.int32)
        graph_b = BehaviorGraph.from_cluster_assignments(labels_b, self.meta)

        # min_agreement=1.0 requires all methods to agree
        result_strict = match_graphs_ensemble(
            self.graph, self.meta, self.labels,
            graph_b, self.meta, labels_b,
            methods=["structural", "temporal"],
            min_agreement=1.0, ratio=1.0,
        )
        # min_agreement=0.0 accepts any single vote
        result_loose = match_graphs_ensemble(
            self.graph, self.meta, self.labels,
            graph_b, self.meta, labels_b,
            methods=["structural", "temporal"],
            min_agreement=0.0, ratio=1.0,
        )
        # strict can only have fewer or equal matches
        self.assertLessEqual(len(result_strict.matches), len(result_loose.matches))
        # loose must have some matches
        self.assertGreater(len(result_loose.matches), 0)

    def test_empty_methods_raises(self):
        with self.assertRaises(ValueError):
            match_graphs_ensemble(
                self.graph, self.meta, self.labels,
                self.graph, self.meta, self.labels,
                methods=[],
            )

    def test_confidence_in_valid_range(self):
        result = match_graphs_ensemble(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            methods=["structural", "temporal", "combined"],
            ratio=1.0,
        )
        for m in result.matches:
            self.assertGreater(m.confidence, 0.0)
            self.assertLessEqual(m.confidence, 1.0)

    def test_no_conflict_in_output(self):
        # No node should appear on both sides of two different matches.
        result = match_graphs_ensemble(
            self.graph, self.meta, self.labels,
            self.graph, self.meta, self.labels,
            methods=["structural", "temporal"],
            ratio=1.0,
        )
        seen_a = [m.node_id_a for m in result.matches]
        seen_b = [m.node_id_b for m in result.matches]
        self.assertEqual(len(seen_a), len(set(seen_a)), "duplicate node_id_a in matches")
        self.assertEqual(len(seen_b), len(set(seen_b)), "duplicate node_id_b in matches")


class TestInvalidMethod(unittest.TestCase):
    def test_unknown_method_raises(self):
        graph, labels, meta = _make_graph([[0]], [True])
        with self.assertRaises(ValueError):
            match_graphs(graph, meta, labels, graph, meta, labels, method="bogus")


if __name__ == "__main__":
    unittest.main()

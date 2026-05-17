"""Tests for behavior graph simplification methods.

Construction strategy: synthesize cluster_labels + metadata for "fake
episodes" with known structure, then verify each method recovers what we
expect.
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Tuple

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.behaviors.simplification.api import METHODS, run_method
from policy_doctor.behaviors.simplification.frontier import (
    LEVER_GRIDS,
    sweep_method,
)
from policy_doctor.behaviors.simplification.metrics import (
    bootstrap_metric,
    compute_metrics,
    hoeffding_compatible,
    js_distance_bits,
    kl_bits,
    markov_violation_bits,
    smoothed_probs,
    trajectory_nll_bits,
)


def make_episode(
    sequence: List[int], rollout_idx: int, success: bool = True,
) -> Tuple[List[int], List[Dict]]:
    labels = list(sequence)
    metadata = [
        {"rollout_idx": rollout_idx, "timestep": t, "success": success}
        for t, _ in enumerate(sequence)
    ]
    return labels, metadata


def stack_episodes(
    episodes: List[List[int]], successes: List[bool],
) -> Tuple[np.ndarray, List[Dict]]:
    all_labels: List[int] = []
    all_meta: List[Dict] = []
    for ep_idx, (seq, succ) in enumerate(zip(episodes, successes)):
        labs, meta = make_episode(seq, rollout_idx=ep_idx, success=succ)
        all_labels.extend(labs)
        all_meta.extend(meta)
    return np.array(all_labels, dtype=np.int64), all_meta


class TestMetrics(unittest.TestCase):
    def test_smoothed_probs_sums_to_one(self):
        counts = {1: 3, 2: 7}
        support = [1, 2, 3]
        p = smoothed_probs(counts, support, alpha=1.0)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        # Unseen symbol gets nonzero mass
        self.assertGreater(p[3], 0.0)

    def test_kl_zero_for_identical(self):
        p = {1: 0.5, 2: 0.5}
        self.assertAlmostEqual(kl_bits(p, p), 0.0, places=6)

    def test_kl_positive_for_different(self):
        self.assertGreater(kl_bits({1: 0.9, 2: 0.1}, {1: 0.1, 2: 0.9}), 0.5)

    def test_hoeffding_compatible_identical(self):
        c = {1: 50, 2: 50}
        self.assertTrue(hoeffding_compatible(c, c, delta=0.05))

    def test_hoeffding_not_compatible_with_enough_data(self):
        c1 = {1: 90, 2: 10}
        c2 = {1: 10, 2: 90}
        self.assertFalse(hoeffding_compatible(c1, c2, delta=0.05))

    def test_hoeffding_compatible_with_low_data(self):
        # Same proportions but few samples → should be compatible due to wide bound
        c1 = {1: 4, 2: 1}
        c2 = {1: 1, 2: 4}
        self.assertTrue(hoeffding_compatible(c1, c2, delta=0.05))

    def test_markov_violation_zero_for_markov_chain(self):
        # Independent transitions: P(next | curr) doesn't depend on prev.
        # Use long repeated cycles A→B→C→A so I(prev;next|curr) = 0 for B and C.
        episodes = [[0, 1, 2] * 30 for _ in range(40)]
        labels, meta = stack_episodes(episodes, [True] * 40)
        total, per_node = markov_violation_bits(labels, meta)
        # I should be zero since each interior state has only one predecessor and one successor
        self.assertLess(total, 0.05)

    def test_markov_violation_positive_for_non_markov(self):
        # Each state's successor depends on the predecessor.
        # Cluster 1 has two paths through it:
        #   0 -> 1 -> 2 (deterministic if prev=0)
        #   3 -> 1 -> 4 (deterministic if prev=3)
        # So I(prev;next|1) is high.
        eps_a = [[0, 1, 2] * 10 for _ in range(20)]
        eps_b = [[3, 1, 4] * 10 for _ in range(20)]
        labels, meta = stack_episodes(eps_a + eps_b, [True] * 40)
        total, per_node = markov_violation_bits(labels, meta)
        self.assertGreater(per_node.get(1, 0.0), 0.5)

    def test_markov_violation_2nd_order_detects_length_2_memory(self):
        # 2nd-order Markov chain: memory of depth 2.
        # Two paths share cluster 5 with the SAME immediate predecessor (4):
        #   ep_a: [0, 4, 5, 6] — when prev_2=0, next is 6
        #   ep_b: [1, 4, 5, 7] — when prev_2=1, next is 7
        # 1st-order: knowing only prev=4 gives uncertain next (could be 6 or 7).
        #           But knowing only curr=5 also gives uncertain next.
        #           I(prev; next | curr) ≈ low because prev is constant (=4).
        # 2nd-order: knowing (prev_1=4, prev_2 ∈ {0, 1}) pins down next.
        #           I((prev_1, prev_2); next | curr) is HIGH.
        from policy_doctor.behaviors.simplification.metrics import (
            markov_violation_against_original_bits,
        )
        eps_a = [[0, 4, 5, 6] * 10 for _ in range(30)]
        eps_b = [[1, 4, 5, 7] * 10 for _ in range(30)]
        labels, meta = stack_episodes(eps_a + eps_b, [True] * 60)

        # 1st-order (identity mapping, so merged == original)
        mv1, per1 = markov_violation_against_original_bits(
            labels, meta, node_mapping={}, level="rollout", order=1,
        )
        # 2nd-order on the same data
        mv2, per2 = markov_violation_against_original_bits(
            labels, meta, node_mapping={}, level="rollout", order=2,
        )
        # 2nd-order should reveal additional structure: mv2 > mv1 at the
        # informative cluster (5)
        self.assertGreater(per2.get(5, 0.0), per1.get(5, 0.0))


class TestMergingMethods(unittest.TestCase):
    def _graph_with_redundant_nodes(self) -> Tuple[BehaviorGraph, np.ndarray, List[Dict]]:
        # Build two cluster nodes (3 and 4) that have IDENTICAL outgoing distributions
        # to a single target node (5). They should be merged by all merge methods.
        eps = []
        for _ in range(30):
            eps.append([0, 3, 5, 1])
            eps.append([0, 4, 5, 1])
        labels, meta = stack_episodes(eps, [True] * len(eps))
        graph = BehaviorGraph.from_cluster_assignments(labels, meta, level="rollout")
        return graph, labels, meta

    def test_js_merge_collapses_redundant(self):
        graph, labels, meta = self._graph_with_redundant_nodes()
        # n_cluster_nodes before = 5 (ids 0,1,3,4,5)
        n_before = len(graph.cluster_nodes)
        result = run_method("js_merge", graph, labels, meta, lever=0.2)
        self.assertLess(len(result.graph.cluster_nodes), n_before)
        # 3 and 4 should be merged (one of them disappears or both map to the same)
        new_unique = set(int(x) for x in result.new_labels)
        # We expect either 3 or 4 to be gone
        self.assertFalse({3, 4}.issubset(new_unique))

    def test_hoeffding_merge_collapses_redundant(self):
        graph, labels, meta = self._graph_with_redundant_nodes()
        n_before = len(graph.cluster_nodes)
        result = run_method("hoeffding_merge", graph, labels, meta, lever=0.05)
        self.assertLess(len(result.graph.cluster_nodes), n_before)

    def test_chi2_merge_collapses_redundant(self):
        graph, labels, meta = self._graph_with_redundant_nodes()
        n_before = len(graph.cluster_nodes)
        result = run_method("chi2_merge", graph, labels, meta, lever=0.05)
        self.assertLess(len(result.graph.cluster_nodes), n_before)


class TestSimplificationMonotone(unittest.TestCase):
    """As the lever moves toward 'fewer nodes', n_nodes should decrease."""

    def _data(self) -> Tuple[BehaviorGraph, np.ndarray, List[Dict]]:
        # 6 cluster nodes, mixed redundancy
        eps = []
        for _ in range(20):
            eps.append([0, 1, 2, 3, 4, 5, 1])  # success
            eps.append([0, 1, 3, 5, 1])         # success
        labels, meta = stack_episodes(eps, [True] * len(eps))
        graph = BehaviorGraph.from_cluster_assignments(labels, meta, level="rollout")
        return graph, labels, meta

    def test_js_merge_more_merging_at_higher_tau(self):
        graph, labels, meta = self._data()
        r_low = run_method("js_merge", graph, labels, meta, lever=0.0)
        r_high = run_method("js_merge", graph, labels, meta, lever=1.0)
        self.assertGreaterEqual(
            len(r_low.graph.cluster_nodes),
            len(r_high.graph.cluster_nodes),
        )

    def test_hoeffding_merge_more_merging_at_lower_delta(self):
        graph, labels, meta = self._data()
        r_strict = run_method("hoeffding_merge", graph, labels, meta, lever=0.5)
        r_loose = run_method("hoeffding_merge", graph, labels, meta, lever=1e-4)
        self.assertGreaterEqual(
            len(r_strict.graph.cluster_nodes),
            len(r_loose.graph.cluster_nodes),
        )

    def test_pcca_plus_runs(self):
        graph, labels, meta = self._data()
        n = len(graph.cluster_nodes)
        result = run_method("pcca_plus", graph, labels, meta, lever=max(2, n // 2))
        self.assertLessEqual(len(result.graph.cluster_nodes), n)

    def test_markov_stability_runs(self):
        graph, labels, meta = self._data()
        result = run_method("markov_stability", graph, labels, meta, lever=5)
        self.assertGreater(len(result.graph.cluster_nodes), 0)

    def test_stationary_skeleton_runs(self):
        graph, labels, meta = self._data()
        result = run_method("stationary_skeleton", graph, labels, meta, lever=0.01)
        self.assertGreater(len(result.graph.cluster_nodes), 0)

    def test_mdl_greedy_runs(self):
        graph, labels, meta = self._data()
        n0 = len(graph.cluster_nodes)
        r_low = run_method("mdl_greedy", graph, labels, meta, lever=0.01)
        r_high = run_method("mdl_greedy", graph, labels, meta, lever=20.0)
        # Higher penalty → fewer nodes (or equal)
        self.assertGreaterEqual(
            len(r_low.graph.cluster_nodes),
            len(r_high.graph.cluster_nodes),
        )

    def test_vomm_split_merge_runs(self):
        graph, labels, meta = self._data()
        result = run_method("vomm_split_merge", graph, labels, meta, lever=0.1)
        self.assertGreater(len(result.graph.cluster_nodes), 0)


class TestSweepAndHeldout(unittest.TestCase):
    def test_sweep_produces_pareto_curve(self):
        # Larger synthetic data so K-fold has something to chew on
        eps = []
        for i in range(50):
            eps.append([0, 1, 2, 3, 4, 5, 1])
            eps.append([0, 1, 3, 4, 1])
        labels, meta = stack_episodes(eps, [True] * len(eps))
        graph = BehaviorGraph.from_cluster_assignments(labels, meta, level="rollout")
        points = sweep_method(
            "hoeffding_merge", graph, labels, meta,
            lever_grid=np.geomspace(1e-4, 0.5, 5),
            with_heldout=True, n_folds=3,
        )
        self.assertEqual(len(points), 5)
        # At least one point should report held-out NLL
        self.assertTrue(any(p.heldout_nll_per_trans_bits is not None for p in points))

    def test_passthrough_returns_original(self):
        eps = [[0, 1, 2, 1] for _ in range(20)]
        labels, meta = stack_episodes(eps, [True] * len(eps))
        graph = BehaviorGraph.from_cluster_assignments(labels, meta, level="rollout")
        result = run_method("passthrough", graph, labels, meta)
        self.assertEqual(len(result.graph.cluster_nodes), len(graph.cluster_nodes))
        np.testing.assert_array_equal(result.new_labels, labels)


class TestAllMethodsRegistered(unittest.TestCase):
    def test_expected_methods_present(self):
        expected = {
            "passthrough", "degree_one_prune",
            "js_merge", "hoeffding_merge", "chi2_merge",
            "bayesian_merge", "vomm_split_merge", "mdl_greedy",
            "pcca_plus", "markov_stability", "stationary_skeleton",
        }
        self.assertTrue(expected.issubset(set(METHODS)))


if __name__ == "__main__":
    unittest.main()

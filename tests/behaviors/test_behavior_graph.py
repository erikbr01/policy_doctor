"""Tests for behavior graph (from_cluster_assignments, values, paths, Markov property)."""

import unittest

import numpy as np

from policy_doctor.behaviors.behavior_graph import (
    BehaviorGraph,
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    get_rollout_slices_for_paths,
    test_markov_property,
    test_markov_property_pooled,
)


class TestBehaviorGraph(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 0, 1, 0, 1])
        self.metadata = [
            {"rollout_idx": 0, "timestep": 0, "success": True},
            {"rollout_idx": 0, "timestep": 1, "success": True},
            {"rollout_idx": 0, "timestep": 2, "success": True},
            {"rollout_idx": 1, "timestep": 0, "success": False},
            {"rollout_idx": 1, "timestep": 1, "success": False},
            {"rollout_idx": 1, "timestep": 2, "success": False},
        ]

    def test_behavior_graph_from_cluster_assignments(self):
        graph = BehaviorGraph.from_cluster_assignments(
            self.labels, self.metadata, level="rollout"
        )
        self.assertEqual(graph.num_episodes, 2)
        self.assertIn(0, graph.nodes)
        self.assertIn(1, graph.nodes)
        self.assertTrue(SUCCESS_NODE_ID in graph.nodes or FAILURE_NODE_ID in graph.nodes)

    def test_behavior_graph_compute_values(self):
        graph = BehaviorGraph.from_cluster_assignments(
            self.labels, self.metadata, level="rollout"
        )
        values = graph.compute_values(gamma=0.99)
        self.assertIsInstance(values, dict)
        self.assertTrue(all(isinstance(v, (int, float)) for v in values.values()))

    def test_behavior_graph_compute_slice_values(self):
        graph = BehaviorGraph.from_cluster_assignments(
            self.labels, self.metadata, level="rollout"
        )
        node_values = graph.compute_values()
        q_values, advantages, next_cluster = graph.compute_slice_values(
            self.labels, self.metadata, node_values
        )
        self.assertEqual(q_values.shape, (6,))
        self.assertEqual(advantages.shape, (6,))

    def test_enumerate_paths_to_terminal(self):
        graph = BehaviorGraph.from_cluster_assignments(
            self.labels, self.metadata, level="rollout"
        )
        paths = graph.enumerate_paths_to_terminal(SUCCESS_NODE_ID, max_paths=10)
        self.assertIsInstance(paths, list)

    def test_get_rollout_slices_for_paths(self):
        graph = BehaviorGraph.from_cluster_assignments(
            self.labels, self.metadata, level="rollout"
        )
        paths = graph.enumerate_paths_to_terminal(SUCCESS_NODE_ID, max_paths=5)
        slice_tuples = get_rollout_slices_for_paths(
            self.labels, self.metadata, "rollout", [p[0] for p in paths]
        )
        self.assertIsInstance(slice_tuples, list)


class TestSimplifyDegreeOnePruning(unittest.TestCase):
    def test_single_outgoing_merges_into_successor(self):
        """Node 0 only leads to 1; slices labeled 0 become 1."""
        # Ep0: 0,0,1,1 -> 0->1->SUCCESS; Ep1: 1,1 -> 1->FAILURE
        labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
        metadata = [
            {"rollout_idx": 0, "timestep": 0, "success": True},
            {"rollout_idx": 0, "timestep": 1, "success": True},
            {"rollout_idx": 0, "timestep": 2, "success": True},
            {"rollout_idx": 0, "timestep": 3, "success": True},
            {"rollout_idx": 1, "timestep": 0, "success": False},
            {"rollout_idx": 1, "timestep": 1, "success": False},
        ]
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level="rollout",
        )
        new_graph, new_labels, n_rounds, n_merged = graph.simplify_by_degree_one_pruning(
            labels, metadata,
        )
        self.assertTrue(np.all(new_labels == 1))
        self.assertNotIn(0, new_graph.nodes)
        self.assertGreater(n_rounds, 0)
        self.assertGreater(n_merged, 0)

    def test_chain_prefers_outgoing_merge_then_iterates(self):
        """Middle of a chain merges forward first; outer may merge in a second pass."""
        labels = np.array([0, 1, 2], dtype=np.int64)
        metadata = [
            {"rollout_idx": 0, "timestep": t, "success": True}
            for t in range(3)
        ]
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level="rollout",
        )
        new_graph, new_labels, n_rounds, n_merged = graph.simplify_by_degree_one_pruning(
            labels, metadata,
        )
        self.assertTrue(np.all(new_labels == 2))
        self.assertNotIn(0, new_graph.nodes)
        self.assertNotIn(1, new_graph.nodes)
        self.assertGreaterEqual(n_rounds, 1)
        self.assertGreaterEqual(n_merged, 2)

    def test_single_incoming_merges_into_predecessor(self):
        """Hub B (id 0) has unique predecessor A (1) but two successors; B -> A.

        A also reaches E on a third episode so A is not pruned by the
        single-outgoing rule first.
        """
        # Ep0: A,B,C -> 1,0,2; Ep1: A,B,D -> 1,0,3; Ep2: A,E,F -> 1,4,5
        labels = np.array([1, 0, 2, 1, 0, 3, 1, 4, 5], dtype=np.int64)
        metadata = [
            {"rollout_idx": 0, "timestep": t, "success": True}
            for t in range(3)
        ] + [
            {"rollout_idx": 1, "timestep": t, "success": True}
            for t in range(3)
        ] + [
            {"rollout_idx": 2, "timestep": t, "success": True}
            for t in range(3)
        ]
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level="rollout",
        )
        new_graph, new_labels, n_rounds, n_merged = graph.simplify_by_degree_one_pruning(
            labels, metadata,
        )
        self.assertFalse(np.any(new_labels == 0))
        self.assertEqual(int(new_labels[1]), 1)
        self.assertEqual(int(new_labels[4]), 1)
        self.assertNotIn(0, new_graph.nodes)
        self.assertGreater(n_rounds, 0)
        self.assertGreater(n_merged, 0)


def _generate_markov_episodes(
    transition_matrix: np.ndarray,
    num_episodes: int,
    min_len: int,
    max_len: int,
    rng: np.random.RandomState,
    success_prob: float = 0.5,
) -> tuple:
    """Simulate episodes from a first-order Markov chain.

    Returns (labels, metadata) suitable for test_markov_property.
    """
    n_states = transition_matrix.shape[0]
    labels_list = []
    metadata_list = []
    for ep in range(num_episodes):
        length = rng.randint(min_len, max_len + 1)
        state = rng.choice(n_states)
        success = rng.random() < success_prob
        for t in range(length):
            labels_list.append(state)
            metadata_list.append(
                {"rollout_idx": ep, "timestep": t, "success": success}
            )
            state = rng.choice(n_states, p=transition_matrix[state])
    return np.array(labels_list), metadata_list


def _generate_non_markov_episodes(
    num_episodes: int,
    rng: np.random.RandomState,
) -> tuple:
    """Generate episodes where the next state depends on the *previous* state.

    Uses 3 states (0, 1, 2).  State 1 always transitions to state 2 if the
    preceding state was 0, and always to state 0 if the preceding state was 2.
    This deterministic second-order dependency violates the Markov property.
    """
    labels_list = []
    metadata_list = []
    for ep in range(num_episodes):
        success = rng.random() < 0.5
        # Fixed sequence structure: 0 -> 1 -> 2 -> 1 -> 0 -> 1 -> 2 -> ...
        # After state 1, next depends on what came before 1:
        #   if prev=0 then next=2; if prev=2 then next=0
        seq = [0, 1]
        for _ in range(18):
            current = seq[-1]
            prev = seq[-2]
            if current == 1:
                seq.append(2 if prev == 0 else 0)
            elif current == 0:
                seq.append(1)
            else:  # current == 2
                seq.append(1)

        for t, state in enumerate(seq):
            labels_list.append(state)
            metadata_list.append(
                {"rollout_idx": ep, "timestep": t, "success": success}
            )
    return np.array(labels_list), metadata_list


class TestMarkovProperty(unittest.TestCase):
    """Verify that test_markov_property correctly identifies Markov and
    non-Markov transition dynamics."""

    def test_true_markov_chain_passes(self):
        """Data generated from a genuine Markov chain should not be rejected."""
        rng = np.random.RandomState(42)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=500, min_len=15, max_len=25, rng=rng
        )
        result = test_markov_property(
            labels, metadata, significance_level=0.01, exclude_terminals=True,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertTrue(
            result["markov_holds"],
            f"Markov test incorrectly rejected true Markov data. "
            f"Per-state p-values: "
            + str(
                {
                    k: v.p_value
                    for k, v in result["per_state"].items()
                    if v.testable
                }
            ),
        )
        self.assertGreater(result["num_states_tested"], 0)

    def test_non_markov_chain_rejected(self):
        """Data with second-order dependencies should be rejected."""
        rng = np.random.RandomState(123)
        labels, metadata = _generate_non_markov_episodes(
            num_episodes=300, rng=rng
        )
        result = test_markov_property(
            labels, metadata, significance_level=0.05, exclude_terminals=True,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertFalse(
            result["markov_holds"],
            "Markov test failed to detect non-Markov second-order dependency.",
        )

    def test_non_markov_state1_detected(self):
        """The specific state with the second-order dependency (state 1)
        should be flagged."""
        rng = np.random.RandomState(123)
        labels, metadata = _generate_non_markov_episodes(
            num_episodes=300, rng=rng
        )
        result = test_markov_property(
            labels, metadata, significance_level=0.05, exclude_terminals=True,
        )
        state1_result = result["per_state"].get(1)
        self.assertIsNotNone(state1_result)
        self.assertTrue(state1_result.testable)
        self.assertFalse(
            state1_result.markov_holds,
            f"State 1 should violate Markov property (p={state1_result.p_value})",
        )

    def test_non_markov_detected_with_terminals(self):
        """Non-Markov data should also be rejected when terminals are included."""
        rng = np.random.RandomState(123)
        labels, metadata = _generate_non_markov_episodes(
            num_episodes=300, rng=rng
        )
        result = test_markov_property(
            labels, metadata, significance_level=0.05, exclude_terminals=False,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertFalse(result["markov_holds"])

    def test_single_state_untestable(self):
        """With only one cluster the test should report untestable."""
        labels = np.zeros(20, dtype=int)
        metadata = [
            {"rollout_idx": i // 5, "timestep": i % 5, "success": True}
            for i in range(20)
        ]
        result = test_markov_property(labels, metadata)
        self.assertEqual(result["num_states_tested"], 0)
        self.assertIsNone(result["markov_holds"])

    def test_insufficient_data_untestable(self):
        """Very short episodes with few samples should be marked untestable."""
        labels = np.array([0, 1])
        metadata = [
            {"rollout_idx": 0, "timestep": 0},
            {"rollout_idx": 0, "timestep": 1},
        ]
        result = test_markov_property(labels, metadata)
        self.assertEqual(result["num_states_tested"], 0)

    def test_result_structure(self):
        """Verify the returned dict has the expected keys and types."""
        rng = np.random.RandomState(0)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=200, min_len=10, max_len=20, rng=rng
        )
        result = test_markov_property(labels, metadata, exclude_terminals=True)
        self.assertIn("markov_holds", result)
        self.assertIn("significance_level", result)
        self.assertIn("num_states_tested", result)
        self.assertIn("num_states_untestable", result)
        self.assertIn("per_state", result)
        for state_result in result["per_state"].values():
            self.assertIsInstance(state_result.state, int)
            self.assertIsInstance(state_result.testable, bool)
            if state_result.testable:
                self.assertIsNotNone(state_result.chi2)
                self.assertIsNotNone(state_result.p_value)
                self.assertIsNotNone(state_result.dof)
                self.assertIsNotNone(state_result.markov_holds)

    def test_no_outcome_info(self):
        """When metadata lacks 'success' key, all episodes end at END node."""
        rng = np.random.RandomState(7)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=500, min_len=15, max_len=25, rng=rng
        )
        for m in metadata:
            del m["success"]
        result = test_markov_property(
            labels, metadata, significance_level=0.01, exclude_terminals=True,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertTrue(
            result["markov_holds"],
            f"Markov test incorrectly rejected true Markov data (no outcome). "
            f"Per-state p-values: "
            + str(
                {
                    k: v.p_value
                    for k, v in result["per_state"].items()
                    if v.testable
                }
            ),
        )

    def test_exclude_terminals_reduces_false_positives(self):
        """With terminals included, episode boundaries can introduce spurious
        violations.  exclude_terminals=True should eliminate them."""
        rng = np.random.RandomState(42)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=500, min_len=15, max_len=25, rng=rng
        )
        result_with = test_markov_property(
            labels, metadata, exclude_terminals=False,
        )
        result_without = test_markov_property(
            labels, metadata, exclude_terminals=True,
        )
        for state_id in result_without["per_state"]:
            r_ex = result_without["per_state"][state_id]
            r_in = result_with["per_state"].get(state_id)
            if r_ex.testable and r_in is not None and r_in.testable:
                self.assertGreaterEqual(
                    r_ex.p_value,
                    r_in.p_value * 0.5,
                    f"State {state_id}: exclude_terminals p-value unexpectedly "
                    f"lower ({r_ex.p_value} vs {r_in.p_value})",
                )


class TestMarkovExactMethod(unittest.TestCase):
    """Tests for method='exact' (permutation-based independence test)."""

    def test_exact_markov_passes(self):
        rng = np.random.RandomState(42)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=500, min_len=15, max_len=25, rng=rng,
        )
        result = test_markov_property(
            labels, metadata, method="exact", n_permutations=2000,
            exclude_terminals=True, significance_level=0.01, random_state=0,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertTrue(
            result["markov_holds"],
            "Exact test incorrectly rejected true Markov data.",
        )

    def test_exact_non_markov_rejected(self):
        rng = np.random.RandomState(123)
        labels, metadata = _generate_non_markov_episodes(300, rng)
        result = test_markov_property(
            labels, metadata, method="exact", n_permutations=2000,
            exclude_terminals=True, significance_level=0.05, random_state=0,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertFalse(
            result["markov_holds"],
            "Exact test failed to detect non-Markov dependency.",
        )

    def test_exact_tests_more_states_than_chi2(self):
        """The exact test has a lower data threshold (3 vs 5), so it should
        be able to test at least as many states as chi2."""
        rng = np.random.RandomState(42)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=100, min_len=8, max_len=12, rng=rng,
        )
        r_chi2 = test_markov_property(labels, metadata, method="chi2", exclude_terminals=True)
        r_exact = test_markov_property(
            labels, metadata, method="exact", n_permutations=500,
            exclude_terminals=True, random_state=0,
        )
        self.assertGreaterEqual(r_exact["num_states_tested"], r_chi2["num_states_tested"])


class TestMarkovModalMethod(unittest.TestCase):
    """Tests for method='modal' (permutation test on successor mode)."""

    def test_modal_markov_passes(self):
        rng = np.random.RandomState(42)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=500, min_len=15, max_len=25, rng=rng,
        )
        result = test_markov_property(
            labels, metadata, method="modal", n_permutations=2000,
            exclude_terminals=True, significance_level=0.01, random_state=0,
        )
        self.assertIsNotNone(result["markov_holds"])
        self.assertTrue(
            result["markov_holds"],
            "Modal test incorrectly rejected true Markov data.",
        )

    def test_modal_non_markov_rejected(self):
        """The deterministic non-Markov data has different modes per
        predecessor at state 1, so the modal test should catch it."""
        rng = np.random.RandomState(123)
        labels, metadata = _generate_non_markov_episodes(300, rng)
        result = test_markov_property(
            labels, metadata, method="modal", n_permutations=2000,
            exclude_terminals=True, significance_level=0.05, random_state=0,
        )
        state1 = result["per_state"].get(1)
        self.assertIsNotNone(state1)
        if state1.testable:
            self.assertFalse(
                state1.markov_holds,
                f"Modal test should flag state 1 (p={state1.p_value})",
            )


class TestMarkovPooled(unittest.TestCase):
    """Tests for test_markov_property_pooled."""

    def _make_split_datasets(self):
        """Generate data from one Markov chain and split into two halves."""
        rng = np.random.RandomState(99)
        P = np.array([
            [0.1, 0.5, 0.4],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
        ])
        labels, metadata = _generate_markov_episodes(
            P, num_episodes=400, min_len=15, max_len=25, rng=rng,
        )
        ep_indices = np.array([m["rollout_idx"] for m in metadata])
        half = 200
        mask_a = ep_indices < half
        mask_b = ep_indices >= half
        labels_a, meta_a = labels[mask_a], [m for m, f in zip(metadata, mask_a) if f]
        labels_b, meta_b = labels[mask_b], [m for m, f in zip(metadata, mask_b) if f]
        return (labels_a, meta_a), (labels_b, meta_b), (labels, metadata)

    def test_pooled_more_power_than_halves(self):
        """Pooling two halves should test at least as many states as either half alone."""
        ds_a, ds_b, _ = self._make_split_datasets()
        r_a = test_markov_property(ds_a[0], ds_a[1], exclude_terminals=True)
        r_b = test_markov_property(ds_b[0], ds_b[1], exclude_terminals=True)
        r_pooled = test_markov_property_pooled(
            [ds_a, ds_b], exclude_terminals=True,
        )
        max_half = max(r_a["num_states_tested"], r_b["num_states_tested"])
        self.assertGreaterEqual(
            r_pooled["num_states_tested"], max_half,
            "Pooling should not reduce the number of testable states.",
        )

    def test_pooled_agrees_with_full(self):
        """Pooling both halves should give similar results to using all data."""
        ds_a, ds_b, (full_labels, full_meta) = self._make_split_datasets()
        r_full = test_markov_property(
            full_labels, full_meta, exclude_terminals=True, significance_level=0.01,
        )
        r_pooled = test_markov_property_pooled(
            [ds_a, ds_b], exclude_terminals=True, significance_level=0.01,
        )
        self.assertEqual(r_full["markov_holds"], r_pooled["markov_holds"])

    def test_pooled_with_exact_method(self):
        """Pooled + exact method should work without error."""
        ds_a, ds_b, _ = self._make_split_datasets()
        result = test_markov_property_pooled(
            [ds_a, ds_b], method="exact", n_permutations=500,
            exclude_terminals=True, random_state=0,
        )
        self.assertIn("markov_holds", result)
        self.assertGreater(result["num_states_tested"], 0)

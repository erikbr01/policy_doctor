"""Unit tests for NearestCentroidAssigner (no diffusion_policy dependency)."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.monitoring.base import AssignmentResult
from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner


def _make_graph(cluster_ids):
    """Build a minimal BehaviorGraph with given cluster IDs from rollout metadata."""
    n = len(cluster_ids)
    labels = np.array(cluster_ids, dtype=np.int32)
    metadata = [{"rollout_idx": i, "timestep": 0} for i in range(n)]
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")


class TestNearestCentroidAssigner(unittest.TestCase):

    def setUp(self):
        # 3 clusters with clear separation in 4-d embedding space
        rng = np.random.RandomState(0)
        self.proj_dim = 4

        # Cluster 0: centered at [1, 0, 0, 0], 10 samples
        c0 = rng.randn(10, 4) * 0.1 + np.array([1, 0, 0, 0])
        # Cluster 1: centered at [0, 1, 0, 0], 10 samples
        c1 = rng.randn(10, 4) * 0.1 + np.array([0, 1, 0, 0])
        # Cluster 2: centered at [0, 0, 1, 0], 10 samples
        c2 = rng.randn(10, 4) * 0.1 + np.array([0, 0, 1, 0])

        self.embeddings = np.vstack([c0, c1, c2]).astype(np.float32)
        self.labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int32)
        self.graph = _make_graph(self.labels)

        self.assigner = NearestCentroidAssigner(
            rollout_embeddings=self.embeddings,
            cluster_labels=self.labels,
            graph=self.graph,
        )

    def test_correct_cluster_assignment(self):
        # Point near cluster 0 centroid
        result = self.assigner.assign(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result.cluster_id, 0)
        self.assertEqual(result.node_id, 0)
        self.assertIsInstance(result.distance, float)
        self.assertIsInstance(result.node_name, str)

    def test_cluster_1_assignment(self):
        result = self.assigner.assign(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result.cluster_id, 1)

    def test_cluster_2_assignment(self):
        result = self.assigner.assign(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        self.assertEqual(result.cluster_id, 2)

    def test_noise_labels_excluded(self):
        # Noise label -1 should not appear as a centroid
        labels_with_noise = np.array([-1] * 5 + [0] * 10 + [1] * 10, dtype=np.int32)
        rng = np.random.RandomState(1)
        embeddings = np.vstack([
            rng.randn(5, 4),                              # noise
            rng.randn(10, 4) * 0.1 + np.array([1, 0, 0, 0]),
            rng.randn(10, 4) * 0.1 + np.array([0, 1, 0, 0]),
        ]).astype(np.float32)
        graph = _make_graph(np.array([0] * 10 + [1] * 10, dtype=np.int32))
        assigner = NearestCentroidAssigner(
            rollout_embeddings=embeddings,
            cluster_labels=labels_with_noise,
            graph=graph,
        )
        self.assertEqual(len(assigner._cluster_ids), 2)
        self.assertNotIn(-1, assigner._cluster_ids.tolist())

    def test_explicit_cluster_to_node_mapping(self):
        # After pruning cluster 0 is merged into cluster 1 → node_id should be 1
        cluster_id_to_node_id = {0: 1}
        assigner = NearestCentroidAssigner(
            rollout_embeddings=self.embeddings,
            cluster_labels=self.labels,
            graph=self.graph,
            cluster_id_to_node_id=cluster_id_to_node_id,
        )
        result = assigner.assign(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result.cluster_id, 0)  # raw cluster
        self.assertEqual(result.node_id, 1)      # after mapping

    def test_assignment_result_type(self):
        result = self.assigner.assign(np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))
        self.assertIsInstance(result, AssignmentResult)
        self.assertIn(result.cluster_id, [0, 1, 2])

    def test_numpy_input(self):
        # Should work with float64 input (auto-cast)
        result = self.assigner.assign(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(result.cluster_id, 0)


class TestNearestCentroidAssigerFromPaths(unittest.TestCase):

    def test_from_paths_raises_if_missing(self):
        import tempfile, pathlib
        graph = _make_graph(np.array([0, 1], dtype=np.int32))
        with self.assertRaises(FileNotFoundError):
            NearestCentroidAssigner.from_paths(
                rollout_embeddings=np.zeros((2, 4), dtype=np.float32),
                clustering_dir="/nonexistent/path",
                graph=graph,
            )


if __name__ == "__main__":
    unittest.main()

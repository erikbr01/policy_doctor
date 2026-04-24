"""Unit tests for FittedModelAssigner — exact pipeline assignment via saved models."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.data.clustering_loader import ClusteringModels, load_clustering_models, save_clustering_models
from policy_doctor.monitoring.graph_assigner import FittedModelAssigner


def _make_graph(n_clusters: int) -> BehaviorGraph:
    labels = np.arange(n_clusters, dtype=np.int32)
    metadata = [{"rollout_idx": i, "timestep": 0} for i in range(n_clusters)]
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")


def _fit_pipeline(n_clusters: int = 3, proj_dim: int = 10, n_samples: int = 60):
    """Fit a small prescaler + KMeans pipeline and return models + data."""
    rng = np.random.RandomState(0)
    embeddings = np.vstack([
        rng.randn(n_samples // n_clusters, proj_dim) * 0.1 + rng.randn(proj_dim) * (k + 1) * 5
        for k in range(n_clusters)
    ]).astype(np.float32)

    prescaler = StandardScaler()
    scaled = prescaler.fit_transform(embeddings).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(scaled)

    models = ClusteringModels(
        normalizer=None,
        normalizer_method="none",
        prescaler=prescaler,
        prescaler_method="standard",
        reducer=None,
        reducer_method="umap",
        kmeans=kmeans,
    )
    return models, embeddings


class TestFittedModelAssigner(unittest.TestCase):

    def setUp(self):
        self.n_clusters = 3
        self.proj_dim = 10
        self.models, self.embeddings = _fit_pipeline(self.n_clusters, self.proj_dim)
        self.graph = _make_graph(self.n_clusters)
        self.assigner = FittedModelAssigner(models=self.models, graph=self.graph)

    def test_returns_assignment_result(self):
        from policy_doctor.monitoring.base import AssignmentResult
        result = self.assigner.assign(self.embeddings[0])
        self.assertIsInstance(result, AssignmentResult)

    def test_cluster_id_in_range(self):
        for emb in self.embeddings[:10]:
            result = self.assigner.assign(emb)
            self.assertIn(result.cluster_id, list(range(self.n_clusters)))

    def test_distance_is_positive(self):
        result = self.assigner.assign(self.embeddings[0])
        self.assertGreater(result.distance, 0.0)

    def test_consistent_with_kmeans_predict(self):
        prescaler = self.models.prescaler
        kmeans = self.models.kmeans
        for emb in self.embeddings[:5]:
            expected = int(kmeans.predict(prescaler.transform(emb.reshape(1, -1)))[0])
            result = self.assigner.assign(emb)
            self.assertEqual(result.cluster_id, expected)

    def test_node_name_is_string(self):
        result = self.assigner.assign(self.embeddings[0])
        self.assertIsInstance(result.node_name, str)

    def test_no_reducer_path(self):
        # reducer=None means skip UMAP step
        result = self.assigner.assign(self.embeddings[0])
        self.assertIsNotNone(result)

    def test_no_normalizer_path(self):
        self.assertIsNone(self.models.normalizer)
        result = self.assigner.assign(self.embeddings[0])
        self.assertIsNotNone(result)

    def test_float64_input_accepted(self):
        result = self.assigner.assign(self.embeddings[0].astype(np.float64))
        self.assertIsNotNone(result)


class TestFittedModelAssigerFromPaths(unittest.TestCase):

    def setUp(self):
        self.n_clusters = 3
        self.proj_dim = 10
        self.models, self.embeddings = _fit_pipeline(self.n_clusters, self.proj_dim)
        self.graph = _make_graph(self.n_clusters)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_clustering_models(
                result_dir=tmp_path,
                normalizer=self.models.normalizer,
                normalizer_method=self.models.normalizer_method,
                prescaler=self.models.prescaler,
                prescaler_method=self.models.prescaler_method,
                reducer=self.models.reducer,
                reducer_method=self.models.reducer_method,
                kmeans=self.models.kmeans,
            )
            loaded = load_clustering_models(tmp_path)
            self.assertEqual(loaded.prescaler_method, "standard")
            self.assertIsNotNone(loaded.prescaler)
            self.assertIsNotNone(loaded.kmeans)

    def test_from_paths_assigns_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_clustering_models(
                result_dir=tmp_path,
                normalizer=self.models.normalizer,
                normalizer_method=self.models.normalizer_method,
                prescaler=self.models.prescaler,
                prescaler_method=self.models.prescaler_method,
                reducer=self.models.reducer,
                reducer_method=self.models.reducer_method,
                kmeans=self.models.kmeans,
            )
            assigner = FittedModelAssigner.from_paths(tmp_path, self.graph)
            result = assigner.assign(self.embeddings[0])
            self.assertIn(result.cluster_id, list(range(self.n_clusters)))

    def test_from_paths_raises_if_missing(self):
        with self.assertRaises(FileNotFoundError):
            FittedModelAssigner.from_paths("/nonexistent/path", self.graph)

    def test_models_file_not_found_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_clustering_models(Path(tmp))


if __name__ == "__main__":
    unittest.main()

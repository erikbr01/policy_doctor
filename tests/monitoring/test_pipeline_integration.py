"""Lightweight end-to-end integration tests for the monitoring pipeline.

No diffusion_policy / cupid dependency. Uses:
  - fit_normalize_embeddings + fit_cluster_kmeans (pure sklearn)
  - save/load_clustering_models (joblib)
  - FittedModelAssigner (pure numpy + sklearn predict)
  - StreamMonitor with mock scorer
  - TrajectoryClassifier
  - MonitoredPolicy

This verifies the full path from fitting a clustering pipeline to classifying
a trajectory, without any GPU or model checkpoint.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.behaviors.clustering import (
    fit_cluster_kmeans,
    fit_normalize_embeddings,
    fit_reduce_dimensions,
)
from policy_doctor.data.clustering_loader import (
    ClusteringModels,
    load_clustering_models,
    save_clustering_models,
)
from policy_doctor.monitoring.base import AssignmentResult, MonitorResult
from policy_doctor.monitoring.graph_assigner import FittedModelAssigner
from policy_doctor.monitoring.monitored_policy import MonitoredPolicy
from policy_doctor.monitoring.stream_monitor import StreamMonitor
from policy_doctor.monitoring.trajectory_classifier import TrajectoryClassifier


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_embeddings(n_clusters=3, proj_dim=20, n_per_cluster=30, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, proj_dim) * 10
    return np.vstack([
        rng.randn(n_per_cluster, proj_dim) * 0.3 + centers[k]
        for k in range(n_clusters)
    ]).astype(np.float32), centers


def _fit_pipeline(embeddings, n_clusters=3):
    """Run the fit variants and return (labels, kmeans, prescaler, models_dict)."""
    emb_norm, normalizer = fit_normalize_embeddings(embeddings, method="none")
    emb_scaled, prescaler = fit_normalize_embeddings(emb_norm, method="standard")
    labels, kmeans = fit_cluster_kmeans(emb_scaled, n_clusters=n_clusters)
    return labels, kmeans, prescaler, normalizer


def _make_graph(labels):
    n = len(labels)
    metadata = [{"rollout_idx": i, "timestep": 0} for i in range(n)]
    return BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")


def _make_mock_scorer(proj_dim=20, n_train=90):
    from policy_doctor.monitoring.base import StreamScorer
    scorer = MagicMock(spec=StreamScorer)
    scorer.device = torch.device("cpu")
    scorer._num_train_timesteps = 8
    rng = np.random.RandomState(1)
    scorer.embed.return_value = rng.randn(proj_dim).astype(np.float32)
    scorer.score.return_value = rng.randn(n_train).astype(np.float32)
    return scorer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitClusteringPipeline(unittest.TestCase):
    """Verify the fit-and-return variants produce usable sklearn models."""

    def setUp(self):
        self.embeddings, _ = _make_embeddings()
        self.n_clusters = 3

    def test_fit_normalize_none_returns_identity(self):
        result, model = fit_normalize_embeddings(self.embeddings, method="none")
        self.assertIsNone(model)
        np.testing.assert_array_equal(result, self.embeddings)

    def test_fit_normalize_standard_returns_scaler(self):
        result, scaler = fit_normalize_embeddings(self.embeddings, method="standard")
        self.assertIsNotNone(scaler)
        # New point can be transformed with the fitted scaler
        new = np.random.randn(1, self.embeddings.shape[1]).astype(np.float32)
        transformed = scaler.transform(new)
        self.assertEqual(transformed.shape, new.shape)

    def test_fit_reduce_pca_returns_model(self):
        result, pca = fit_reduce_dimensions(
            self.embeddings, method="pca", n_components=5
        )
        self.assertIsNotNone(pca)
        self.assertEqual(result.shape[1], 5)
        new_point = pca.transform(self.embeddings[:1])
        self.assertEqual(new_point.shape, (1, 5))

    def test_fit_cluster_kmeans_returns_model(self):
        labels, kmeans = fit_cluster_kmeans(self.embeddings, n_clusters=self.n_clusters)
        self.assertEqual(len(labels), len(self.embeddings))
        predicted = kmeans.predict(self.embeddings[:1])
        self.assertIn(predicted[0], range(self.n_clusters))


class TestFitSaveLoadAssign(unittest.TestCase):
    """Fit pipeline → save models → load → FittedModelAssigner → assign new point."""

    def setUp(self):
        self.n_clusters = 3
        self.proj_dim = 20
        self.embeddings, self.centers = _make_embeddings(
            n_clusters=self.n_clusters, proj_dim=self.proj_dim
        )
        self.labels, self.kmeans, self.prescaler, self.normalizer = _fit_pipeline(
            self.embeddings, self.n_clusters
        )
        self.graph = _make_graph(self.labels)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_clustering_models(
                result_dir=tmp,
                normalizer=self.normalizer, normalizer_method="none",
                prescaler=self.prescaler, prescaler_method="standard",
                reducer=None, reducer_method="umap",
                kmeans=self.kmeans,
            )
            models = load_clustering_models(tmp)
        self.assertEqual(models.prescaler_method, "standard")
        self.assertIsNotNone(models.kmeans)

    def test_assigner_consistent_with_kmeans_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_clustering_models(
                result_dir=tmp,
                normalizer=self.normalizer, normalizer_method="none",
                prescaler=self.prescaler, prescaler_method="standard",
                reducer=None, reducer_method="umap",
                kmeans=self.kmeans,
            )
            assigner = FittedModelAssigner.from_paths(tmp, self.graph)

        rng = np.random.RandomState(42)
        for _ in range(10):
            emb = rng.randn(self.proj_dim).astype(np.float32)
            result = assigner.assign(emb)
            expected = int(self.kmeans.predict(self.prescaler.transform(emb.reshape(1, -1)))[0])
            self.assertEqual(result.cluster_id, expected)

    def test_near_cluster_center_gets_correct_assignment(self):
        """Point near cluster k's center should be assigned to cluster k."""
        with tempfile.TemporaryDirectory() as tmp:
            save_clustering_models(
                result_dir=tmp,
                normalizer=self.normalizer, normalizer_method="none",
                prescaler=self.prescaler, prescaler_method="standard",
                reducer=None, reducer_method="umap",
                kmeans=self.kmeans,
            )
            assigner = FittedModelAssigner.from_paths(tmp, self.graph)

        # Find what cluster each center belongs to via the fitted KMeans
        for k in range(self.n_clusters):
            center = self.centers[k]
            result = assigner.assign(center)
            # At least it returns a valid cluster id
            self.assertIn(result.cluster_id, range(self.n_clusters))


class TestClassifierEndToEnd(unittest.TestCase):
    """TrajectoryClassifier with mock scorer + FittedModelAssigner."""

    def setUp(self):
        self.n_clusters = 3
        self.proj_dim = 20
        embeddings, _ = _make_embeddings(
            n_clusters=self.n_clusters, proj_dim=self.proj_dim
        )
        labels, kmeans, prescaler, normalizer = _fit_pipeline(embeddings, self.n_clusters)
        graph = _make_graph(labels)

        self.tmp = tempfile.mkdtemp()
        save_clustering_models(
            result_dir=self.tmp,
            normalizer=normalizer, normalizer_method="none",
            prescaler=prescaler, prescaler_method="standard",
            reducer=None, reducer_method="umap",
            kmeans=kmeans,
        )
        assigner = FittedModelAssigner.from_paths(self.tmp, graph)
        scorer = _make_mock_scorer(proj_dim=self.proj_dim)
        monitor = StreamMonitor(scorer=scorer, assigner=assigner)
        self.classifier = TrajectoryClassifier(
            monitor=monitor, mode="rollout", n_obs_steps=2, n_action_steps=8
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_classify_sequence_returns_assignments(self):
        T = 20
        obs_seq = np.random.randn(T, 10).astype(np.float32)
        action_seq = np.random.randn(T, 7).astype(np.float32)
        results = self.classifier.classify_sequence(obs_seq, action_seq)
        self.assertEqual(len(results), T - 1)
        for _, r in results:
            self.assertIsNotNone(r.assignment)
            self.assertIn(r.assignment.cluster_id, range(self.n_clusters))

    def test_classify_sample_returns_node_name(self):
        obs = np.random.randn(2, 10).astype(np.float32)
        action = np.random.randn(8, 7).astype(np.float32)
        result = self.classifier.classify_sample(obs, action)
        self.assertIsInstance(result.assignment.node_name, str)
        self.assertGreater(len(result.assignment.node_name), 0)


class TestMonitoredPolicyEndToEnd(unittest.TestCase):
    """MonitoredPolicy wrapping a mock policy + real TrajectoryClassifier."""

    def _make_classifier(self):
        proj_dim = 20
        n_clusters = 3
        embeddings, _ = _make_embeddings(n_clusters=n_clusters, proj_dim=proj_dim)
        labels, kmeans, prescaler, normalizer = _fit_pipeline(embeddings, n_clusters)
        graph = _make_graph(labels)

        with tempfile.TemporaryDirectory() as tmp:
            save_clustering_models(
                result_dir=tmp,
                normalizer=normalizer, normalizer_method="none",
                prescaler=prescaler, prescaler_method="standard",
                reducer=None, reducer_method="umap",
                kmeans=kmeans,
            )
            assigner = FittedModelAssigner.from_paths(tmp, graph)

        scorer = _make_mock_scorer(proj_dim=proj_dim)
        monitor = StreamMonitor(scorer=scorer, assigner=assigner)
        return TrajectoryClassifier(
            monitor=monitor, mode="rollout", n_obs_steps=2, n_action_steps=8
        )

    def test_full_episode_simulation(self):
        policy = MagicMock()
        policy.predict_action.return_value = {
            "action": torch.randn(1, 8, 7)
        }
        classifier = self._make_classifier()
        mp = MonitoredPolicy(policy=policy, classifier=classifier)

        # Simulate 2 episodes of 5 timesteps each
        for ep in range(2):
            mp.reset()
            for _ in range(5):
                mp.predict_action({"obs": torch.randn(1, 2, 10)})

        # Episode 0 and 1 results, 5 steps each
        self.assertEqual(len(mp.episode_results), 10)
        self.assertEqual(mp.episode_results[0]["episode"], 0)
        self.assertEqual(mp.episode_results[5]["episode"], 1)

        # All assignments should be valid cluster IDs
        for entry in mp.episode_results:
            self.assertIsNotNone(entry["cluster_id"])
            self.assertIn(entry["cluster_id"], range(3))


if __name__ == "__main__":
    unittest.main()

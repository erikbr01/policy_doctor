"""Unit tests for StreamMonitor with a mock scorer (no diffusion_policy dependency)."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.monitoring.base import MonitorResult, StreamScorer
from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner
from policy_doctor.monitoring.stream_monitor import StreamMonitor


def _make_mock_scorer(proj_dim=50, n_train=200, device="cpu", num_train_timesteps=100):
    """Return a StreamScorer mock that returns deterministic arrays."""
    scorer = MagicMock(spec=StreamScorer)
    scorer.device = torch.device(device)
    scorer._num_train_timesteps = num_train_timesteps
    rng = np.random.RandomState(42)
    embedding = rng.randn(proj_dim).astype(np.float32)
    scores = rng.randn(n_train).astype(np.float32)
    scorer.embed.return_value = embedding
    scorer.score.return_value = scores
    return scorer, embedding, scores


def _make_graph_and_assigner(proj_dim=50):
    rng = np.random.RandomState(0)
    # 3 clusters, 15 samples each
    c0 = rng.randn(15, proj_dim) * 0.1 + rng.randn(proj_dim)
    c1 = rng.randn(15, proj_dim) * 0.1 + rng.randn(proj_dim) * 5
    c2 = rng.randn(15, proj_dim) * 0.1 + rng.randn(proj_dim) * 10
    embeddings = np.vstack([c0, c1, c2]).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15 + [2] * 15, dtype=np.int32)
    metadata = [{"rollout_idx": i, "timestep": 0} for i in range(len(labels))]
    graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level="rollout")
    assigner = NearestCentroidAssigner(embeddings, labels, graph)
    return graph, assigner, embeddings


class TestStreamMonitorWithMock(unittest.TestCase):

    def setUp(self):
        self.proj_dim = 50
        self.n_train = 200
        self.scorer, self.embedding, self.scores = _make_mock_scorer(
            proj_dim=self.proj_dim, n_train=self.n_train
        )
        self.graph, self.assigner, self.rollout_embs = _make_graph_and_assigner(self.proj_dim)

    def test_process_sample_returns_monitor_result(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=self.assigner)
        obs = np.random.randn(2, 20).astype(np.float32)
        action = np.random.randn(8, 14).astype(np.float32)
        result = monitor.process_sample(obs, action)
        self.assertIsInstance(result, MonitorResult)

    def test_embedding_shape(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=self.assigner)
        obs = np.random.randn(2, 20).astype(np.float32)
        action = np.random.randn(8, 14).astype(np.float32)
        result = monitor.process_sample(obs, action)
        self.assertEqual(result.embedding.shape, (self.proj_dim,))

    def test_influence_scores_shape(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=self.assigner)
        result = monitor.process_sample(
            np.random.randn(2, 20).astype(np.float32),
            np.random.randn(8, 14).astype(np.float32),
        )
        self.assertEqual(result.influence_scores.shape, (self.n_train,))

    def test_assignment_present_when_assigner_set(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=self.assigner)
        result = monitor.process_sample(
            np.random.randn(2, 20).astype(np.float32),
            np.random.randn(8, 14).astype(np.float32),
        )
        self.assertIsNotNone(result.assignment)
        self.assertIn(result.assignment.cluster_id, [0, 1, 2])

    def test_no_assigner_gives_none_assignment(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=None)
        result = monitor.process_sample(
            np.random.randn(2, 20).astype(np.float32),
            np.random.randn(8, 14).astype(np.float32),
        )
        self.assertIsNone(result.assignment)

    def test_timing_keys_present(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=self.assigner)
        result = monitor.process_sample(
            np.random.randn(2, 20).astype(np.float32),
            np.random.randn(8, 14).astype(np.float32),
        )
        self.assertIn("total_ms", result.timing_ms)
        self.assertIn("gradient_project_ms", result.timing_ms)
        self.assertIn("score_ms", result.timing_ms)
        self.assertIn("assign_ms", result.timing_ms)
        self.assertGreater(result.timing_ms["total_ms"], 0)

    def test_embed_only_skips_scoring(self):
        monitor = StreamMonitor(scorer=self.scorer, assigner=self.assigner)
        result = monitor.process_sample_embed_only(
            np.random.randn(2, 20).astype(np.float32),
            np.random.randn(8, 14).astype(np.float32),
        )
        self.assertEqual(result.embedding.shape, (self.proj_dim,))
        self.assertIsNotNone(result.assignment)
        self.assertIsNone(result.influence_scores)
        self.scorer.score.assert_not_called()

    def test_torch_tensor_input(self):
        monitor = StreamMonitor(scorer=self.scorer)
        obs = torch.randn(2, 20)
        action = torch.randn(8, 14)
        result = monitor.process_sample(obs, action)
        self.assertIsNotNone(result.embedding)

    def test_scorer_called_with_correct_batch_keys(self):
        monitor = StreamMonitor(scorer=self.scorer)
        monitor.process_sample(
            np.random.randn(2, 20).astype(np.float32),
            np.random.randn(8, 14).astype(np.float32),
        )
        call_kwargs = self.scorer.embed.call_args
        batch = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]["batch"]
        self.assertIn("obs", batch)
        self.assertIn("action", batch)
        self.assertIn("timesteps", batch)
        self.assertEqual(batch["obs"].shape[0], 1)    # batch dim = 1
        self.assertEqual(batch["action"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for PretrainRNN, PrioritizedReplayBuffer, and train_pretrain_rnn."""

import os
import tempfile
import unittest

import numpy as np
import torch

from policy_doctor.enap.rnn_encoder import (
    PrioritizedReplayBuffer,
    PretrainRNN,
    phase_aware_contrastive_loss_pretrain,
    train_pretrain_rnn,
)


def _make_episodes(n=6, a_dim=3, s_dim=4, min_T=4, max_T=8, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n):
        T = rng.integers(min_T, max_T + 1)
        A = rng.standard_normal((T, a_dim)).astype(np.float32)
        idx = rng.integers(0, s_dim, T)
        S = np.zeros((T, s_dim), dtype=np.float32)
        S[np.arange(T), idx] = 1.0
        eps.append({"S": S, "A": A})
    return eps


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def test_add_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=20)
        eps = _make_episodes(10)
        buf.fill(eps)
        self.assertEqual(len(buf), 10)

        batch, weights, indices = buf.sample(4)
        self.assertEqual(len(batch), 4)
        self.assertEqual(len(weights), 4)
        self.assertEqual(len(indices), 4)
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(weights <= 1.0 + 1e-6))

    def test_capacity_overflow(self):
        buf = PrioritizedReplayBuffer(capacity=5)
        buf.fill(_make_episodes(8))
        self.assertEqual(len(buf), 5)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        buf.fill(_make_episodes(6))
        _, _, indices = buf.sample(3)
        losses = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        buf.update_priorities(indices, losses)

    def test_anneal_beta(self):
        buf = PrioritizedReplayBuffer(beta_start=0.4, beta_end=1.0, beta_steps=10)
        start = buf.beta
        for _ in range(10):
            buf.anneal_beta()
        self.assertGreater(buf.beta, start)
        self.assertLessEqual(buf.beta, 1.0)


class TestPretrainRNN(unittest.TestCase):
    def setUp(self):
        self.a_dim, self.s_dim, self.e_dim, self.h_dim = 3, 4, 8, 16
        self.model = PretrainRNN(self.a_dim, self.s_dim, self.e_dim, self.h_dim)

    def test_forward_shape(self):
        B, T = 2, 5
        A = torch.randn(B, T, self.a_dim)
        S = torch.randint(0, self.s_dim, (B, T))
        h_seq, a_pred, s_pred = self.model(A, S)
        self.assertEqual(h_seq.shape, (B, T, self.h_dim))
        self.assertEqual(a_pred.shape, (B, T, self.a_dim))
        self.assertEqual(s_pred.shape, (B, T, self.s_dim))

    def test_extract_hidden_states_episode(self):
        T = 7
        A = np.random.randn(T, self.a_dim).astype(np.float32)
        S = np.random.randint(0, self.s_dim, T).astype(np.int64)
        h = self.model.extract_hidden_states_episode(A, S)
        self.assertEqual(h.shape, (T, self.h_dim))

    def test_checkpoint_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self.model.save_checkpoint(path)
            loaded = PretrainRNN.load_checkpoint(path)
            self.assertEqual(loaded.a_dim, self.a_dim)
            self.assertEqual(loaded.s_dim, self.s_dim)
            self.assertEqual(loaded.h_dim, self.h_dim)
            # Weights should match
            for (k, v), (k2, v2) in zip(
                self.model.state_dict().items(),
                loaded.state_dict().items(),
            ):
                self.assertTrue(torch.allclose(v, v2), f"Mismatch in {k}")
        finally:
            os.unlink(path)

    def test_dims_in_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self.model.save_checkpoint(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["dims"]["a"], self.a_dim)
            self.assertEqual(ckpt["dims"]["s"], self.s_dim)
            self.assertEqual(ckpt["dims"]["e"], self.e_dim)
            self.assertEqual(ckpt["dims"]["h"], self.h_dim)
        finally:
            os.unlink(path)


class TestPhaseAwareContrastiveLoss(unittest.TestCase):
    def test_same_phase_zero_when_identical(self):
        h = torch.randn(4, 8)
        s = torch.zeros(4, dtype=torch.long)
        loss = phase_aware_contrastive_loss_pretrain(h, h, s, s, margin=0.5)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_transition_zero_when_far_apart(self):
        # orthogonal → cos_sim = 0 → dist = 1 > margin → relu(0.5 - 1) = 0
        h_t = torch.tensor([[1.0, 0.0]])
        h_t1 = torch.tensor([[0.0, 1.0]])
        s_t = torch.tensor([0])
        s_t1 = torch.tensor([1])
        loss = phase_aware_contrastive_loss_pretrain(h_t, h_t1, s_t, s_t1, margin=0.5)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_output_scalar(self):
        h = torch.randn(8, 16)
        s = torch.randint(0, 3, (8,))
        loss = phase_aware_contrastive_loss_pretrain(h, h.roll(1, 0), s, s.roll(1), margin=0.5)
        self.assertEqual(loss.shape, ())


class TestTrainPretrainRNN(unittest.TestCase):
    def test_trains_without_error(self):
        model = PretrainRNN(a_dim=3, s_dim=4, e_dim=8, h_dim=16)
        eps = _make_episodes(8)
        losses = train_pretrain_rnn(
            model, eps, num_epochs=3, lr=1e-3, batch_size=4, verbose=False
        )
        self.assertEqual(len(losses), 3)
        self.assertIn("total", losses[0])
        self.assertIn("act", losses[0])
        self.assertIn("contrast", losses[0])

    def test_trains_without_per(self):
        model = PretrainRNN(a_dim=3, s_dim=4, e_dim=8, h_dim=16)
        eps = _make_episodes(6)
        losses = train_pretrain_rnn(
            model, eps, num_epochs=2, use_per=False, verbose=False
        )
        self.assertEqual(len(losses), 2)

    def test_loss_is_finite(self):
        model = PretrainRNN(a_dim=3, s_dim=4, e_dim=8, h_dim=16)
        eps = _make_episodes(8)
        losses = train_pretrain_rnn(
            model, eps, num_epochs=5, verbose=False
        )
        for entry in losses:
            self.assertTrue(np.isfinite(entry["total"]))

"""Tests for ResidualMLP and train_residual_mlp."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from policy_doctor.enap.residual_policy import ResidualMLP, PMMAgent, train_residual_mlp


def _make_data(N=64, feat_dim=8, a_dim=3, s_dim=4, n_nodes=3, seed=7):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((N, feat_dim)).astype(np.float32)
    actions = rng.standard_normal((N, a_dim)).astype(np.float32)
    symbols = rng.integers(0, s_dim, N).astype(np.int64)
    nodes = rng.integers(0, n_nodes, N).astype(np.int64)
    cluster_centers = rng.standard_normal((s_dim, feat_dim)).astype(np.float32)
    return features, actions, symbols, nodes, cluster_centers


def _make_mock_pmm(a_dim=3, s_dim=4, n_nodes=3):
    pmm = MagicMock()
    pmm.a_dim = a_dim
    pmm.predict.return_value = np.zeros(a_dim, dtype=np.float32)
    pmm.step.return_value = 0
    pmm.pmm = {"Q": list(range(n_nodes)), "delta": {}}
    return pmm


class TestResidualMLP(unittest.TestCase):
    def setUp(self):
        self.feat_dim, self.a_dim = 8, 3
        self.model = ResidualMLP(feat_dim=self.feat_dim, a_dim=self.a_dim, hidden=32)

    def test_forward_shape(self):
        B = 4
        f = torch.randn(B, self.feat_dim)
        ab = torch.randn(B, self.a_dim)
        cc = torch.randn(B, self.feat_dim)
        out = self.model(ab, cc, f)
        self.assertEqual(out.shape, (B, self.a_dim))

    def test_residual_connection(self):
        # With zero net weights, output ≈ action_base
        for p in self.model.net.parameters():
            p.data.zero_()
        ab = torch.ones(2, self.a_dim) * 3.0
        out = self.model(ab, torch.zeros(2, self.feat_dim), torch.zeros(2, self.feat_dim))
        self.assertTrue(torch.allclose(out, ab, atol=1e-5))

    def test_get_log_prob_shape(self):
        B = 6
        f = torch.randn(B, self.feat_dim)
        ab = torch.randn(B, self.a_dim)
        cc = torch.randn(B, self.feat_dim)
        a_target = torch.randn(B, self.a_dim)
        lp = self.model.get_log_prob(ab, cc, f, a_target)
        self.assertEqual(lp.shape, (B,))

    def test_log_prob_finite(self):
        B = 8
        f = torch.randn(B, self.feat_dim)
        ab = torch.randn(B, self.a_dim)
        cc = torch.randn(B, self.feat_dim)
        lp = self.model.get_log_prob(ab, cc, f, torch.randn(B, self.a_dim))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_checkpoint_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self.model.save_checkpoint(path)
            loaded = ResidualMLP.load_checkpoint(path)
            self.assertEqual(loaded.feat_dim, self.feat_dim)
            self.assertEqual(loaded.a_dim, self.a_dim)
        finally:
            os.unlink(path)


class TestTrainResidualMLP(unittest.TestCase):
    def test_trains_without_error(self):
        feat_dim, a_dim, s_dim = 8, 3, 4
        features, actions, symbols, nodes, cc = _make_data(
            N=80, feat_dim=feat_dim, a_dim=a_dim, s_dim=s_dim
        )
        pmm = _make_mock_pmm(a_dim=a_dim, s_dim=s_dim)
        model = ResidualMLP(feat_dim=feat_dim, a_dim=a_dim, hidden=32)
        stats = train_residual_mlp(
            model=model,
            features=features,
            actions=actions,
            symbols=symbols,
            node_assignments=nodes,
            cluster_centers=cc,
            pmm=pmm,
            num_epochs=5,
            batch_size=32,
            verbose=False,
        )
        self.assertIn("best_epoch", stats)
        self.assertIn("best_val_loss", stats)
        self.assertEqual(len(stats["train_losses"]), 5)

    def test_val_loss_finite(self):
        feat_dim, a_dim, s_dim = 8, 3, 4
        features, actions, symbols, nodes, cc = _make_data(N=80)
        pmm = _make_mock_pmm()
        model = ResidualMLP(feat_dim=feat_dim, a_dim=a_dim, hidden=32)
        stats = train_residual_mlp(
            model=model,
            features=features,
            actions=actions,
            symbols=symbols,
            node_assignments=nodes,
            cluster_centers=cc,
            pmm=pmm,
            num_epochs=3,
            verbose=False,
        )
        self.assertTrue(np.isfinite(stats["best_val_loss"]))


class TestPMMAgent(unittest.TestCase):
    def test_act_returns_correct_shape(self):
        feat_dim, a_dim, s_dim = 8, 3, 4
        pmm = _make_mock_pmm(a_dim=a_dim, s_dim=s_dim)
        mlp = ResidualMLP(feat_dim=feat_dim, a_dim=a_dim, hidden=32)
        cc = np.random.randn(s_dim, feat_dim).astype(np.float32)
        agent = PMMAgent(pmm=pmm, residual_mlp=mlp, cluster_centers=cc)

        z_t = np.random.randn(feat_dim).astype(np.float32)
        action = agent.act(z_t, c_t=0)
        self.assertEqual(action.shape, (a_dim,))

    def test_reset_returns_to_initial_state(self):
        feat_dim, a_dim, s_dim = 8, 3, 4
        pmm = _make_mock_pmm(a_dim=a_dim, s_dim=s_dim)
        mlp = ResidualMLP(feat_dim=feat_dim, a_dim=a_dim, hidden=32)
        cc = np.random.randn(s_dim, feat_dim).astype(np.float32)
        agent = PMMAgent(pmm=pmm, residual_mlp=mlp, cluster_centers=cc)
        agent._q = 2
        agent.reset()
        self.assertEqual(agent._q, 0)

    def test_identify_symbol_returns_nearest_centroid(self):
        feat_dim, s_dim = 8, 4
        pmm = _make_mock_pmm(s_dim=s_dim)
        mlp = ResidualMLP(feat_dim=feat_dim, a_dim=3, hidden=32)
        cc = np.eye(s_dim, feat_dim, dtype=np.float32)
        agent = PMMAgent(pmm=pmm, residual_mlp=mlp, cluster_centers=cc)

        # z_t is unit vector along axis 2 → should pick symbol 2
        z_t = cc[2].copy()
        sym = agent.identify_symbol(z_t)
        self.assertEqual(sym, 2)

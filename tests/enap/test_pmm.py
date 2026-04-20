"""Tests for PMM (faithful Extended L* learning engine).

Uses minimal synthetic data: 2 symbols, 2 task phases, short episodes.
"""

import os
import tempfile
import unittest

import numpy as np

from policy_doctor.enap.pmm import PMM, TrainableRNN
from policy_doctor.enap.rnn_encoder import PretrainRNN


def _make_pretrain_checkpoint(a_dim=2, s_dim=2, e_dim=4, h_dim=8, path=None):
    """Save a random PretrainRNN checkpoint in PMM-compatible format."""
    model = PretrainRNN(a_dim=a_dim, s_dim=s_dim, e_dim=e_dim, h_dim=h_dim)
    if path is None:
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        path = f.name
        f.close()
    model.save_checkpoint(path)
    return path


def _make_trajectory_batch(
    n_episodes=8,
    a_dim=2,
    s_dim=2,
    min_T=4,
    max_T=8,
    seed=42,
):
    """Synthetic 2-symbol trajectory batch."""
    rng = np.random.default_rng(seed)
    batch = []
    for ep_idx in range(n_episodes):
        T = rng.integers(min_T, max_T + 1)
        steps = []
        # Phase 0 for first half, phase 1 for second half
        for t in range(T):
            sym = 0 if t < T // 2 else 1
            action = rng.standard_normal(a_dim).astype(np.float32)
            state = np.zeros(s_dim, dtype=np.float32)
            state[sym] = 1.0
            steps.append({"action": action, "state": state})
        batch.append(steps)
    return batch


class TestTrainableRNN(unittest.TestCase):
    def test_loads_from_pretrain_checkpoint(self):
        path = _make_pretrain_checkpoint(a_dim=2, s_dim=2, e_dim=4, h_dim=8)
        try:
            rnn = TrainableRNN(action_dim=2, state_dim=2, embed_dim=4, h_dim=8)
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            rnn.load_weights(ckpt["model_state"])
        finally:
            os.unlink(path)

    def test_encode_returns_correct_shape(self):
        path = _make_pretrain_checkpoint(a_dim=2, s_dim=2, e_dim=4, h_dim=8)
        try:
            rnn = TrainableRNN(action_dim=2, state_dim=2, embed_dim=4, h_dim=8)
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            rnn.load_weights(ckpt["model_state"])
            A = np.random.randn(5, 2).astype(np.float32)
            S = np.array([0, 0, 1, 1, 1], dtype=np.int64)
            h = rnn.encode(A, S)
            self.assertEqual(h.shape, (8,))
        finally:
            os.unlink(path)

    def test_forward_trajectory_shape(self):
        path = _make_pretrain_checkpoint(a_dim=2, s_dim=2, e_dim=4, h_dim=8)
        try:
            rnn = TrainableRNN(action_dim=2, state_dim=2, embed_dim=4, h_dim=8)
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            rnn.load_weights(ckpt["model_state"])
            A = np.random.randn(6, 2).astype(np.float32)
            S = np.random.randint(0, 2, 6).astype(np.int64)
            h_seq = rnn.forward_trajectory(A, S)
            self.assertEqual(h_seq.shape, (6, 8))
        finally:
            os.unlink(path)


class TestPMMLearning(unittest.TestCase):
    def setUp(self):
        self.a_dim = 2
        self.s_dim = 2
        self.ckpt_path = _make_pretrain_checkpoint(
            a_dim=self.a_dim, s_dim=self.s_dim, e_dim=4, h_dim=8
        )
        self.trajectory_batch = _make_trajectory_batch(
            n_episodes=10, a_dim=self.a_dim, s_dim=self.s_dim
        )
        self.cluster_centers = np.random.randn(self.s_dim, 8).astype(np.float32)

    def tearDown(self):
        os.unlink(self.ckpt_path)

    def test_learn_pmm_runs_and_returns_dict(self):
        pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
        result = pmm.learn_pmm(
            self.trajectory_batch,
            rnn_weights_path=self.ckpt_path,
            cluster_centers=self.cluster_centers,
        )
        self.assertIn("Q", result)
        self.assertIn("delta", result)
        self.assertGreater(len(result["Q"]), 0)

    def test_pmm_has_at_least_one_edge(self):
        pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
        pmm.learn_pmm(
            self.trajectory_batch,
            rnn_weights_path=self.ckpt_path,
        )
        n_edges = sum(len(d) for d in pmm.pmm["delta"].values())
        self.assertGreater(n_edges, 0)

    def test_predict_returns_action(self):
        pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
        pmm.learn_pmm(
            self.trajectory_batch,
            rnn_weights_path=self.ckpt_path,
        )
        q0 = pmm.pmm["Q"][0]
        # Find any valid (q, x) key
        for (q, x) in pmm._qx_actions:
            a = pmm.predict(q, x)
            self.assertEqual(a.shape, (self.a_dim,))
            break

    def test_step_returns_valid_node(self):
        pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
        pmm.learn_pmm(
            self.trajectory_batch,
            rnn_weights_path=self.ckpt_path,
        )
        valid_nodes = set(pmm.pmm["Q"])
        q_next = pmm.step(0, 0)
        self.assertIn(q_next, valid_nodes)

    def test_node_assignments_length_matches_data(self):
        pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
        pmm.learn_pmm(
            self.trajectory_batch,
            rnn_weights_path=self.ckpt_path,
        )
        total_timesteps = sum(len(ep) for ep in self.trajectory_batch)
        n_assigned = sum(
            len(indices) for indices in pmm._edge_cache.values()
        )
        self.assertLessEqual(n_assigned, total_timesteps)
        self.assertGreater(n_assigned, 0)

    def test_to_json_serializable(self):
        pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
        pmm.learn_pmm(
            self.trajectory_batch,
            rnn_weights_path=self.ckpt_path,
        )
        d = pmm.to_json_serializable()
        import json
        json.dumps(d)  # should not raise
        self.assertIn("Q", d)
        self.assertIn("num_nodes", d)


class TestPMMPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        ckpt_path = _make_pretrain_checkpoint(a_dim=2, s_dim=2, e_dim=4, h_dim=8)
        traj = _make_trajectory_batch(n_episodes=6, a_dim=2, s_dim=2)
        try:
            pmm = PMM(max_inner_iters=3, stabil_required=1, use_tqdm=False)
            pmm.learn_pmm(traj, rnn_weights_path=ckpt_path)
            n_nodes_before = len(pmm.pmm["Q"])

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pkl_path = f.name
            try:
                pmm.save_pmm(pkl_path)
                pmm2 = PMM()
                pmm2.load_pmm(pkl_path)
                self.assertEqual(len(pmm2.pmm["Q"]), n_nodes_before)
                self.assertEqual(pmm2.sigma, pmm.sigma)
            finally:
                os.unlink(pkl_path)
        finally:
            os.unlink(ckpt_path)

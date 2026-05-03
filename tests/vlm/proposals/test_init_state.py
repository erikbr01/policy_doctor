"""Unit tests for policy_doctor.vlm.proposals.init_state."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from policy_doctor.vlm.proposals.init_state import (
    extract_object_pose_at_frame,
    extract_sim_state_at_frame,
    verify_sim_state_replays,
)


def _make_episode_pkl(path: Path, T: int = 5, sim_state_dim: int = 6) -> None:
    sim_states = [
        np.arange(sim_state_dim, dtype=np.float64) + t
        for t in range(T)
    ]
    obs_list = [
        {
            "object": np.array([t * 0.1, t * 0.2, t * 0.3], dtype=np.float32),
            "robot0_eef_pos": np.array([0.0, 0.0, t * 0.05], dtype=np.float32),
        }
        for t in range(T)
    ]
    df = pd.DataFrame({"sim_state": sim_states, "obs": obs_list})
    df.to_pickle(str(path))


class TestExtractSimStateAtFrame(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_init_state_"))
        self.pkl = self.tmp / "ep0000.pkl"
        _make_episode_pkl(self.pkl, T=5, sim_state_dim=6)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extracts_expected_vector(self):
        for k in range(5):
            v = extract_sim_state_at_frame(self.pkl, k)
            expected = np.arange(6, dtype=np.float64) + k
            np.testing.assert_array_equal(v, expected)

    def test_out_of_range_high_raises(self):
        with self.assertRaises(IndexError):
            extract_sim_state_at_frame(self.pkl, 5)

    def test_out_of_range_negative_raises(self):
        with self.assertRaises(IndexError):
            extract_sim_state_at_frame(self.pkl, -1)


class TestVerifySimStateReplays(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_init_state_replay_"))
        self.pkl = self.tmp / "ep0000.pkl"
        _make_episode_pkl(self.pkl, T=4, sim_state_dim=6)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_recorded_state_replays_true(self):
        recorded = extract_sim_state_at_frame(self.pkl, 2)
        self.assertTrue(verify_sim_state_replays(self.pkl, recorded, 2))

    def test_perturbed_state_replays_false(self):
        recorded = extract_sim_state_at_frame(self.pkl, 2)
        perturbed = recorded + 1.0
        self.assertFalse(verify_sim_state_replays(self.pkl, perturbed, 2))

    def test_wrong_shape_returns_false(self):
        bad = np.zeros(3, dtype=np.float64)
        self.assertFalse(verify_sim_state_replays(self.pkl, bad, 0))


class TestExtractObjectPoseAtFrame(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_init_state_obj_"))
        self.pkl = self.tmp / "ep0000.pkl"
        _make_episode_pkl(self.pkl, T=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_default_object_key(self):
        v = extract_object_pose_at_frame(self.pkl, 1, obs_key="object")
        np.testing.assert_allclose(v, np.array([0.1, 0.2, 0.3], dtype=np.float32), rtol=0, atol=1e-6)

    def test_reads_alternative_key(self):
        v = extract_object_pose_at_frame(self.pkl, 2, obs_key="robot0_eef_pos")
        np.testing.assert_allclose(v, np.array([0.0, 0.0, 0.10], dtype=np.float32), rtol=0, atol=1e-6)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            extract_object_pose_at_frame(self.pkl, 0, obs_key="not_a_real_key")

    def test_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            extract_object_pose_at_frame(self.pkl, 99, obs_key="object")


if __name__ == "__main__":
    unittest.main()

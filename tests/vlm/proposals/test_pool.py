"""Unit tests for policy_doctor.vlm.proposals.pool."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from policy_doctor.vlm.proposals.pool import (
    RolloutPool,
    episode_idx_to_rollout_id,
    rollout_id_to_episode_idx,
)


def _write_fake_episodes(episodes_dir: Path, successes, lengths):
    episodes_dir.mkdir(parents=True, exist_ok=True)
    n = len(successes)
    for i in range(n):
        # Each pkl is a small DataFrame with a sim_state column
        T = lengths[i]
        df = pd.DataFrame({
            "sim_state": [np.zeros(4, dtype=np.float64) + i + t * 0.01 for t in range(T)],
            "obs": [{"object": np.array([0.0, 0.0, 0.0], dtype=np.float32)} for _ in range(T)],
            "success": [bool(successes[i])] * T,
        })
        df.to_pickle(str(episodes_dir / f"ep{i:04d}.pkl"))
    meta = {
        "episode_successes": [bool(s) for s in successes],
        "episode_lengths": [int(t) for t in lengths],
    }
    with open(episodes_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)


class TestEpisodeIdHelpers(unittest.TestCase):
    def test_episode_idx_to_rollout_id(self):
        self.assertEqual(episode_idx_to_rollout_id(0), "r0000")
        self.assertEqual(episode_idx_to_rollout_id(7), "r0007")
        self.assertEqual(episode_idx_to_rollout_id(123), "r0123")

    def test_round_trip(self):
        for i in (0, 1, 9, 12, 9999):
            self.assertEqual(rollout_id_to_episode_idx(episode_idx_to_rollout_id(i)), i)


class TestRolloutPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="test_pool_"))
        cls.episodes_dir = cls.tmp / "eps"
        cls.successes = [True, False, True, True, False]
        cls.lengths = [10, 12, 8, 11, 9]
        _write_fake_episodes(cls.episodes_dir, cls.successes, cls.lengths)
        cls.pool = RolloutPool.from_episodes_dir(cls.episodes_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_count_and_ids(self):
        self.assertEqual(len(self.pool), len(self.successes))
        ids = [e.rollout_id for e in self.pool.entries]
        self.assertEqual(ids, [f"r{i:04d}" for i in range(len(self.successes))])

    def test_per_entry_fields(self):
        for i, e in enumerate(self.pool.entries):
            self.assertEqual(e.episode_idx, i)
            self.assertEqual(e.success, self.successes[i])
            self.assertEqual(e.length, self.lengths[i])
            self.assertTrue(e.episode_pkl.exists())

    def test_successes_failures(self):
        n_pass = sum(1 for s in self.successes if s)
        n_fail = sum(1 for s in self.successes if not s)
        self.assertEqual(len(self.pool.successes()), n_pass)
        self.assertEqual(len(self.pool.failures()), n_fail)

    def test_by_id_known(self):
        self.assertEqual(self.pool.by_id("r0002").episode_idx, 2)

    def test_by_id_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.pool.by_id("rZZZZ")

    def test_to_index_dict_round_trip(self):
        idx = self.pool.to_index_dict()
        self.assertEqual(idx["n_rollouts"], len(self.successes))
        self.assertEqual(len(idx["rollouts"]), len(self.successes))
        for i, r in enumerate(idx["rollouts"]):
            self.assertEqual(r["rollout_id"], f"r{i:04d}")
            self.assertEqual(r["episode_idx"], i)
            self.assertEqual(r["length"], self.lengths[i])
            self.assertEqual(r["success"], self.successes[i])

    def test_episode_idx_helpers_round_trip(self):
        self.assertEqual(episode_idx_to_rollout_id(0), "r0000")
        self.assertEqual(rollout_id_to_episode_idx("r0000"), 0)


class TestRolloutPoolMissingDir(unittest.TestCase):
    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            RolloutPool.from_episodes_dir(Path("/nonexistent/path/does/not/exist"))

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                RolloutPool.from_episodes_dir(Path(td))


if __name__ == "__main__":
    unittest.main()

"""Tests for ``policy_doctor.data.slice_representations``."""

from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
import pandas as pd
import yaml

from policy_doctor.data.slice_representations import (
    InfEmbedRepresentation,
    SliceRepresentation,
    SliceWindowParams,
    StateActionRepresentation,
    StateRepresentation,
    _flatten_action_row,
    _flatten_obs_row,
    get_slice_representation,
    list_slice_representations,
)


def _make_eval_dir(
    tmp: Path,
    *,
    n_episodes: int = 4,
    ep_len: int = 12,
    obs_history: int = 2,
    obs_dim: int = 7,
    action_horizon: int = 3,
    action_dim: int = 5,
) -> Path:
    """Materialize a synthetic eval_dir mimicking eval_save_episodes layout."""
    eval_dir = tmp / "eval"
    ep_dir = eval_dir / "episodes"
    ep_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for ep_i in range(n_episodes):
        rows = []
        for t in range(ep_len):
            rows.append({
                "obs": rng.standard_normal((obs_history, obs_dim)).astype(np.float32),
                "action": rng.standard_normal((action_horizon, action_dim)).astype(np.float32),
                "img": np.zeros((4, 4, 3), dtype=np.uint8),
                "success": ep_i % 2 == 0,
            })
        df = pd.DataFrame(rows)
        suffix = "succ" if ep_i % 2 == 0 else "fail"
        with open(ep_dir / f"ep{ep_i:04d}_{suffix}.pkl", "wb") as f:
            pickle.dump(df, f)
    with open(ep_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({
            "episode_lengths": [ep_len] * n_episodes,
            "episode_successes": [bool(i % 2 == 0) for i in range(n_episodes)],
        }, f)
    return eval_dir


class FlattenHelpers(unittest.TestCase):
    def test_obs_current_picks_last_frame(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])  # T=2, D=2
        out = _flatten_obs_row(v, obs_strategy="current")
        np.testing.assert_array_equal(out, np.array([3.0, 4.0], dtype=np.float32))

    def test_obs_full_history_flattens_all(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _flatten_obs_row(v, obs_strategy="full_history")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype=np.float32))

    def test_obs_passes_through_1d(self):
        v = np.array([1.0, 2.0, 3.0])
        out = _flatten_obs_row(v, obs_strategy="current")
        np.testing.assert_array_equal(out, v.astype(np.float32))

    def test_obs_invalid_strategy_raises(self):
        v = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            _flatten_obs_row(v, obs_strategy="bogus")

    def test_action_executed_picks_first(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])  # H=2, A=2
        out = _flatten_action_row(v, action_strategy="executed")
        np.testing.assert_array_equal(out, np.array([1.0, 2.0], dtype=np.float32))

    def test_action_full_plan_flattens(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _flatten_action_row(v, action_strategy="full_plan")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype=np.float32))


class Registry(unittest.TestCase):
    def test_three_reps_registered(self):
        names = list_slice_representations()
        self.assertEqual(set(names), {"infembed", "state", "state_action"})

    def test_get_returns_correct_class(self):
        self.assertIsInstance(get_slice_representation("infembed"), InfEmbedRepresentation)
        self.assertIsInstance(get_slice_representation("state"), StateRepresentation)
        self.assertIsInstance(get_slice_representation("state_action"), StateActionRepresentation)

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            get_slice_representation("not_a_rep")


class StateRepShape(unittest.TestCase):
    def test_extract_shapes_and_metadata(self):
        with TemporaryDirectory() as td:
            eval_dir = _make_eval_dir(Path(td), ep_len=12)
            rep = StateRepresentation()
            params = SliceWindowParams(window_width=5, stride=2, aggregation="mean")
            features, metadata = rep.extract(eval_dir, params, obs_strategy="current")

            # 4 episodes × ((12 - 5) // 2 + 1) = 4 × 4 = 16 slices
            self.assertEqual(features.shape, (16, 7))   # obs_dim = 7 with current strategy
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(len(metadata), 16)
            for m in metadata:
                self.assertIn("rollout_idx", m)
                self.assertIn("window_start", m)
                self.assertIn("window_end", m)
                self.assertIn("window_width", m)
                self.assertIn("success", m)
                self.assertEqual(m["window_width"], 5)
            # Episodes are 0..3
            self.assertEqual(set(m["rollout_idx"] for m in metadata), {0, 1, 2, 3})

    def test_full_history_doubles_dim(self):
        with TemporaryDirectory() as td:
            eval_dir = _make_eval_dir(Path(td), ep_len=10, obs_history=2, obs_dim=7)
            rep = StateRepresentation()
            params = SliceWindowParams(window_width=5, stride=5, aggregation="sum")
            features, _ = rep.extract(eval_dir, params, obs_strategy="full_history")
            self.assertEqual(features.shape[1], 14)  # 2 × 7


class StateActionRepShape(unittest.TestCase):
    def test_extract_concatenates_obs_action(self):
        with TemporaryDirectory() as td:
            eval_dir = _make_eval_dir(
                Path(td), ep_len=10, obs_history=2, obs_dim=7,
                action_horizon=3, action_dim=5,
            )
            rep = StateActionRepresentation()
            params = SliceWindowParams(window_width=5, stride=5, aggregation="mean")
            features, _ = rep.extract(
                eval_dir, params,
                obs_strategy="current", action_strategy="executed",
            )
            # features dim = obs_dim (7) + action_dim (5) = 12
            self.assertEqual(features.shape[1], 12)


class FingerprintAndDescribe(unittest.TestCase):
    def test_describe_includes_kwargs(self):
        rep = StateActionRepresentation()
        params = SliceWindowParams(window_width=5, stride=2, aggregation="mean")
        d = rep.describe(params, obs_strategy="current", action_strategy="full_plan")
        self.assertEqual(d["representation"], "state_action")
        self.assertEqual(d["window_width"], 5)
        self.assertEqual(d["aggregation"], "mean")
        self.assertEqual(d["obs_strategy"], "current")
        self.assertEqual(d["action_strategy"], "full_plan")


if __name__ == "__main__":
    unittest.main()

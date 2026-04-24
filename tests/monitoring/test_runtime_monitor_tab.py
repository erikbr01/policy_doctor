"""Unit tests for policy_doctor.streamlit_app.tabs.runtime_monitor helpers.

These tests cover the pure helper functions (CSV parsing, intervention
computation, index mapping, path suggestion) without starting Streamlit.
"""

import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from policy_doctor.streamlit_app.tabs.runtime_monitor import (
    _compute_interventions,
    _demo_abs_sample_idx,
    _ep_abs_sample_idx,
    _parse_monitor_csv,
    _suggested_csv_path,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _make_csv_bytes(**extra_cols) -> bytes:
    """Return a minimal valid monitor CSV as bytes."""
    data = {
        "episode": [0, 0, 0, 1, 1],
        "timestep": [0, 1, 2, 0, 1],
        "node_name": ["A", "B", "A", "C", "A"],
        "node_id":   [0,   1,   0,   2,   0],
        "cluster_id": [0,  1,   0,   2,   0],
        "distance":  [0.1, 0.2, 0.3, 0.4, 0.5],
        "total_ms":  [10.0, 11.0, 12.0, 13.0, 14.0],
    }
    data.update(extra_cols)
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _make_episode_info(index, start, end, success=True):
    ep = MagicMock()
    ep.index = index
    ep.sample_start_idx = start
    ep.sample_end_idx = end
    ep.success = success
    return ep


# ──────────────────────────────────────────────────────────────
# _parse_monitor_csv
# ──────────────────────────────────────────────────────────────

class TestParseMonitorCsv(unittest.TestCase):

    def test_parses_valid_csv(self):
        df = _parse_monitor_csv(_make_csv_bytes())
        self.assertEqual(len(df), 5)
        self.assertIn("episode", df.columns)
        self.assertIn("timestep", df.columns)
        self.assertIn("node_name", df.columns)

    def test_episode_col_added_when_missing(self):
        data = {"timestep": [0, 1], "node_name": ["A", "B"]}
        buf = io.StringIO()
        pd.DataFrame(data).to_csv(buf, index=False)
        df = _parse_monitor_csv(buf.getvalue().encode())
        self.assertIn("episode", df.columns)
        self.assertTrue((df["episode"] == 0).all())

    def test_env_idx_col_added_when_missing(self):
        data = {"timestep": [0], "node_name": ["A"]}
        buf = io.StringIO()
        pd.DataFrame(data).to_csv(buf, index=False)
        df = _parse_monitor_csv(buf.getvalue().encode())
        self.assertIn("env_idx", df.columns)
        self.assertTrue((df["env_idx"] == 0).all())

    def test_missing_required_col_raises(self):
        data = {"episode": [0], "node_name": ["A"]}   # missing timestep
        buf = io.StringIO()
        pd.DataFrame(data).to_csv(buf, index=False)
        with self.assertRaises(ValueError):
            _parse_monitor_csv(buf.getvalue().encode())

    def test_node_name_nan_filled(self):
        data = {"timestep": [0, 1], "node_name": ["A", float("nan")]}
        buf = io.StringIO()
        pd.DataFrame(data).to_csv(buf, index=False)
        df = _parse_monitor_csv(buf.getvalue().encode())
        self.assertEqual(df["node_name"].iloc[1], "N/A")

    def test_episode_dtype_is_int(self):
        df = _parse_monitor_csv(_make_csv_bytes())
        self.assertEqual(df["episode"].dtype, np.dtype("int64"))

    def test_timestep_dtype_is_int(self):
        df = _parse_monitor_csv(_make_csv_bytes())
        self.assertEqual(df["timestep"].dtype, np.dtype("int64"))

    def test_distance_parsed_as_float(self):
        df = _parse_monitor_csv(_make_csv_bytes())
        self.assertTrue(pd.api.types.is_float_dtype(df["distance"]))

    def test_intervention_col_preserved(self):
        extra = {"intervention_triggered": [True, False, True, False, True]}
        df = _parse_monitor_csv(_make_csv_bytes(**extra))
        self.assertIn("intervention_triggered", df.columns)


# ──────────────────────────────────────────────────────────────
# _compute_interventions
# ──────────────────────────────────────────────────────────────

class TestComputeInterventions(unittest.TestCase):

    def setUp(self):
        data = {
            "timestep": [0, 1, 2, 3],
            "node_id": [0, 1, 2, 0],
            "node_name": ["A", "B", "C", "A"],
        }
        self.df = pd.DataFrame(data)
        # V(0)=0.5, V(1)=-0.3, V(2)=0.8
        self.node_values = {0: 0.5, 1: -0.3, 2: 0.8}

    def test_correct_timesteps_flagged(self):
        mask = _compute_interventions(self.df, self.node_values, threshold=0.0)
        # node 1 has V=-0.3 < 0.0 → index 1 triggered
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])

    def test_threshold_above_all_values(self):
        mask = _compute_interventions(self.df, self.node_values, threshold=1.0)
        self.assertTrue(mask.all())

    def test_threshold_below_all_values(self):
        mask = _compute_interventions(self.df, self.node_values, threshold=-1.0)
        self.assertFalse(mask.any())

    def test_empty_node_values(self):
        mask = _compute_interventions(self.df, {}, threshold=0.0)
        self.assertFalse(mask.any())

    def test_missing_node_id_col(self):
        df_no_nid = self.df.drop(columns=["node_id"])
        mask = _compute_interventions(df_no_nid, self.node_values, threshold=0.0)
        self.assertFalse(mask.any())

    def test_output_length_matches_dataframe(self):
        mask = _compute_interventions(self.df, self.node_values, threshold=0.0)
        self.assertEqual(len(mask), len(self.df))

    def test_output_dtype_is_bool(self):
        mask = _compute_interventions(self.df, self.node_values, threshold=0.0)
        self.assertEqual(mask.dtype, bool)


# ──────────────────────────────────────────────────────────────
# _ep_abs_sample_idx / _demo_abs_sample_idx
# ──────────────────────────────────────────────────────────────

class TestAbsSampleIdx(unittest.TestCase):

    def _make_episodes(self):
        # ep0: samples 0..9, ep1: samples 10..19
        return [
            _make_episode_info(0, start=0, end=10),
            _make_episode_info(1, start=10, end=20),
        ]

    def test_rollout_basic_mapping(self):
        eps = self._make_episodes()
        self.assertEqual(_ep_abs_sample_idx(eps, ep_idx=0, local_t=3), 3)
        self.assertEqual(_ep_abs_sample_idx(eps, ep_idx=1, local_t=0), 10)
        self.assertEqual(_ep_abs_sample_idx(eps, ep_idx=1, local_t=5), 15)

    def test_rollout_out_of_range_returns_none(self):
        eps = self._make_episodes()
        # local_t=10 → abs=10, but ep0 ends at 10 (exclusive)
        self.assertIsNone(_ep_abs_sample_idx(eps, ep_idx=0, local_t=10))

    def test_rollout_episode_out_of_range_returns_none(self):
        eps = self._make_episodes()
        self.assertIsNone(_ep_abs_sample_idx(eps, ep_idx=5, local_t=0))

    def test_rollout_none_episodes_returns_none(self):
        self.assertIsNone(_ep_abs_sample_idx(None, ep_idx=0, local_t=0))

    def test_demo_basic_mapping(self):
        eps = self._make_episodes()
        self.assertEqual(_demo_abs_sample_idx(eps, ep_idx=0, local_t=2), 2)
        self.assertEqual(_demo_abs_sample_idx(eps, ep_idx=1, local_t=4), 14)

    def test_demo_out_of_range_returns_none(self):
        eps = self._make_episodes()
        self.assertIsNone(_demo_abs_sample_idx(eps, ep_idx=0, local_t=10))


# ──────────────────────────────────────────────────────────────
# _suggested_csv_path
# ──────────────────────────────────────────────────────────────

class TestSuggestedCsvPath(unittest.TestCase):

    def _make_config(self, eval_dir):
        cfg = MagicMock()
        cfg.eval_dir = eval_dir
        return cfg

    def test_returns_none_when_no_eval_dir(self):
        cfg = self._make_config(None)
        self.assertIsNone(_suggested_csv_path(cfg))

    def test_returns_path_with_filename(self):
        cfg = self._make_config("/some/eval/dir")
        result = _suggested_csv_path(cfg)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "monitor_assignments.csv")

    def test_absolute_eval_dir_used_directly(self):
        cfg = self._make_config("/abs/eval")
        result = _suggested_csv_path(cfg)
        self.assertEqual(result, Path("/abs/eval/monitor_assignments.csv"))

    def test_relative_eval_dir_joined_with_repo_root(self):
        from policy_doctor.paths import REPO_ROOT
        cfg = self._make_config("data/outputs/eval/my_run")
        result = _suggested_csv_path(cfg)
        expected = REPO_ROOT / "data/outputs/eval/my_run" / "monitor_assignments.csv"
        self.assertEqual(result, expected)

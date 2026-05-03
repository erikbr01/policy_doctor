"""Unit tests for policy_doctor.vlm.proposals.adherence.

Strategy: monkey-patch the in-module ``classify_demo_pkl`` reference so we
control the cluster path returned for any demo pkl. A FakeTrajectoryClassifier
is provided as a sentinel object only — it is never actually invoked.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd

from policy_doctor.vlm.proposals import adherence as adherence_mod
from policy_doctor.vlm.proposals.adherence import (
    DEFAULT_FILTER_THRESHOLD,
    DEFAULT_WEIGHTS,
    score_batch_to_jsonl,
    score_request_adherence,
)
from policy_doctor.vlm.proposals.request import (
    DemonstrationRequest,
    InitialConditions,
)


class FakeTrajectoryClassifier:
    """Sentinel — adherence code only ever passes this through to
    ``classify_demo_pkl``, which we monkey-patch in each test."""


def _make_demo_pkl(path: Path, success: bool, T: int = 4) -> None:
    df = pd.DataFrame({
        "sim_state": [np.zeros(4, dtype=np.float64) + t for t in range(T)],
        "obs": [{} for _ in range(T)],
        "success": [success] * T,
    })
    df.to_pickle(str(path))


def _make_request(
    *,
    request_type: str,
    target_cluster: int = None,
    reference_rollout_id: str = "r0000",
    reference_frame: int = 0,
    source_condition: str = "graph",
    request_id: str = "req-1",
) -> DemonstrationRequest:
    return DemonstrationRequest(
        request_id=request_id,
        request_type=request_type,
        initial_conditions=InitialConditions(
            reference_rollout_id=reference_rollout_id,
            reference_frame=reference_frame,
        ),
        target_behavior="grasp the object and lift it",
        prohibitions=[],
        success_criterion="task_success",
        target_cluster=target_cluster,
        source_condition=source_condition,
    )


class _AdherenceCommon(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_adh_"))
        self.demo_pkl = self.tmp / "demo.pkl"
        _make_demo_pkl(self.demo_pkl, success=True)
        self.classifier = FakeTrajectoryClassifier()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_path(self, path: List[int]):
        return mock.patch.object(
            adherence_mod, "classify_demo_pkl", return_value=list(path)
        )


class TestFullTrajectoryCluster(_AdherenceCommon):
    def test_target_in_demo_path_score_one(self):
        req = _make_request(request_type="full_trajectory", target_cluster=2)
        with self._patch_path([0, 2, 3]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=True,
            )
        self.assertEqual(score.axes["cluster"].score, 1.0)

    def test_target_not_in_demo_path_score_zero(self):
        req = _make_request(request_type="full_trajectory", target_cluster=99)
        with self._patch_path([0, 1, 2]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=True,
            )
        self.assertEqual(score.axes["cluster"].score, 0.0)


class TestRecoveryCluster(_AdherenceCommon):
    def test_starts_in_target_and_success(self):
        req = _make_request(request_type="recovery", target_cluster=5)
        with self._patch_path([5, 6, 7]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=True,
            )
        self.assertEqual(score.axes["cluster"].score, 1.0)

    def test_only_starts_in_target(self):
        req = _make_request(request_type="recovery", target_cluster=5)
        with self._patch_path([5, 6, 7]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=False,
            )
        self.assertEqual(score.axes["cluster"].score, 0.5)

    def test_only_success(self):
        req = _make_request(request_type="recovery", target_cluster=5)
        with self._patch_path([1, 2]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=True,
            )
        self.assertEqual(score.axes["cluster"].score, 0.5)

    def test_neither(self):
        req = _make_request(request_type="recovery", target_cluster=5)
        with self._patch_path([1, 2]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=False,
            )
        self.assertEqual(score.axes["cluster"].score, 0.0)


class TestAlternativeStrategy(_AdherenceCommon):
    def test_disjoint_paths_score_one(self):
        req = _make_request(request_type="alternative_strategy", target_cluster=None)
        with self._patch_path([3, 4, 5]):
            score = score_request_adherence(
                req,
                self.demo_pkl,
                self.classifier,
                success=True,
                reference_cluster_path=[0, 1, 2],
            )
        # Jaccard of disjoint sets = 0; cluster axis = 1 - 0 = 1
        self.assertAlmostEqual(score.axes["cluster"].score, 1.0, places=6)

    def test_identical_paths_score_zero(self):
        req = _make_request(request_type="alternative_strategy", target_cluster=None)
        with self._patch_path([0, 1, 2]):
            score = score_request_adherence(
                req,
                self.demo_pkl,
                self.classifier,
                success=True,
                reference_cluster_path=[0, 1, 2],
            )
        self.assertAlmostEqual(score.axes["cluster"].score, 0.0, places=6)

    def test_overlapping_paths(self):
        req = _make_request(request_type="alternative_strategy", target_cluster=None)
        # demo {0,1,2}, ref {1,2,3} → intersect=2 union=4 jaccard=0.5
        with self._patch_path([0, 1, 2]):
            score = score_request_adherence(
                req,
                self.demo_pkl,
                self.classifier,
                success=True,
                reference_cluster_path=[1, 2, 3],
            )
        self.assertAlmostEqual(score.axes["cluster"].score, 0.5, places=6)


class TestOverallAndFilter(_AdherenceCommon):
    def test_overall_is_weighted_sum(self):
        # full_trajectory; demo_path contains target → cluster=1.0
        req = _make_request(request_type="full_trajectory", target_cluster=1)
        with self._patch_path([0, 1, 2]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=True,
            )
        # With default weights: init=0.25*1 + cluster=0.5*1 + success=0.25*1 = 1.0
        # init=1.0 because reference_pkl_resolver=None (the bypass)
        self.assertAlmostEqual(score.overall, 1.0, places=6)
        self.assertTrue(score.passed_filter)

    def test_passed_filter_threshold_honored(self):
        req = _make_request(request_type="full_trajectory", target_cluster=99)
        with self._patch_path([0, 1, 2]):
            # cluster=0, init=1.0, success=1.0 → overall = 0.25+0+0.25 = 0.5
            score = score_request_adherence(
                req,
                self.demo_pkl,
                self.classifier,
                success=True,
                filter_threshold=0.6,
            )
        self.assertAlmostEqual(score.overall, 0.5, places=6)
        self.assertFalse(score.passed_filter)

        # Lower threshold — same score now passes
        with self._patch_path([0, 1, 2]):
            score2 = score_request_adherence(
                req,
                self.demo_pkl,
                self.classifier,
                success=True,
                filter_threshold=0.4,
            )
        self.assertTrue(score2.passed_filter)


class TestInitialConditionAxis(_AdherenceCommon):
    def test_no_resolver_returns_one(self):
        req = _make_request(request_type="full_trajectory", target_cluster=0)
        with self._patch_path([0]):
            score = score_request_adherence(
                req,
                self.demo_pkl,
                self.classifier,
                success=True,
                reference_pkl_resolver=None,
            )
        ic = score.axes["initial_condition"]
        self.assertEqual(ic.score, 1.0)
        self.assertIn("no resolver", ic.description)


class TestSuccessResolution(_AdherenceCommon):
    def test_caller_arg_takes_precedence(self):
        req = _make_request(request_type="full_trajectory", target_cluster=0)
        # demo pkl has success=True for all rows; caller forces False
        with self._patch_path([0]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=False,
            )
        self.assertEqual(score.axes["success"].score, 0.0)
        self.assertIn("from caller", score.axes["success"].description)

    def test_falls_back_to_pkl_success_column(self):
        req = _make_request(request_type="full_trajectory", target_cluster=0)
        with self._patch_path([0]):
            score = score_request_adherence(
                req, self.demo_pkl, self.classifier, success=None,
            )
        self.assertEqual(score.axes["success"].score, 1.0)
        self.assertIn("'success' column", score.axes["success"].description)


class TestScoreBatchToJsonl(_AdherenceCommon):
    def test_writes_expected_files_and_buckets(self):
        out_dir = self.tmp / "scoring_out"

        # Two demo pkls: one will pass, one will fail.
        demo_pass = self.tmp / "demo_pass.pkl"
        demo_fail = self.tmp / "demo_fail.pkl"
        _make_demo_pkl(demo_pass, success=True)
        _make_demo_pkl(demo_fail, success=False)

        # Pair 1: graph cond, full_trajectory, target IN path → cluster=1, init=1, success=1 → pass
        req1 = _make_request(
            request_type="full_trajectory",
            target_cluster=0,
            source_condition="graph",
            request_id="req-pass",
        )
        # Pair 2: outcome_only cond, full_trajectory, target NOT in path, success=False
        # → cluster=0, init=1, success=0 → overall=0.25 → fail
        req2 = _make_request(
            request_type="full_trajectory",
            target_cluster=99,
            source_condition="outcome_only",
            request_id="req-fail",
        )

        # Custom side_effect: we want different paths per demo pkl.
        path_for: Dict[str, List[int]] = {
            str(demo_pass): [0, 1, 2],
            str(demo_fail): [7, 8, 9],
        }

        def fake_classify(_classifier, pkl_path):
            return path_for[str(Path(pkl_path))]

        with mock.patch.object(adherence_mod, "classify_demo_pkl", side_effect=fake_classify):
            summary = score_batch_to_jsonl(
                [(req1, demo_pass), (req2, demo_fail)],
                self.classifier,
                out_dir,
            )

        per_demo = out_dir / "per_demo_scores.jsonl"
        filtered = out_dir / "filtered_demos.jsonl"
        summary_json = out_dir / "filter_summary.json"
        self.assertTrue(per_demo.exists())
        self.assertTrue(filtered.exists())
        self.assertTrue(summary_json.exists())

        per_demo_lines = per_demo.read_text().strip().splitlines()
        filtered_lines = filtered.read_text().strip().splitlines()
        self.assertEqual(len(per_demo_lines), 2)
        # Only req1 passes
        self.assertEqual(len(filtered_lines), 1)
        passed_rec = json.loads(filtered_lines[0])
        self.assertEqual(passed_rec["request_id"], "req-pass")
        self.assertTrue(passed_rec["passed_filter"])

        # Summary structure
        with open(summary_json) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["n_total"], 2)
        self.assertEqual(on_disk["n_passed"], 1)
        self.assertEqual(on_disk["n_failed"], 1)
        self.assertEqual(on_disk["weights"], dict(DEFAULT_WEIGHTS))
        self.assertAlmostEqual(on_disk["filter_threshold"], DEFAULT_FILTER_THRESHOLD, places=6)

        # Per-condition bucketing
        by_cond = on_disk["by_condition"]
        self.assertIn("graph", by_cond)
        self.assertIn("outcome_only", by_cond)
        self.assertEqual(by_cond["graph"]["n_passed"], 1)
        self.assertEqual(by_cond["graph"]["n_total"], 1)
        self.assertEqual(by_cond["outcome_only"]["n_passed"], 0)
        self.assertEqual(by_cond["outcome_only"]["n_total"], 1)

        # Per-request_type bucketing
        by_rt = on_disk["by_request_type"]
        self.assertIn("full_trajectory", by_rt)
        self.assertEqual(by_rt["full_trajectory"]["n_total"], 2)
        self.assertEqual(by_rt["full_trajectory"]["n_passed"], 1)

        # Function return mirrors disk
        self.assertEqual(summary["n_total"], on_disk["n_total"])
        self.assertEqual(summary["n_passed"], on_disk["n_passed"])


if __name__ == "__main__":
    unittest.main()

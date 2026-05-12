"""Unit tests for the constraint-aware MimicGen generator.

These tests target the pure-Python parts (constraint check, slack widening,
slack-from-stddev derivation, helpers). The end-to-end integration test
that actually runs MimicGen lives in tests/mimicgen/test_chained_warp_e2e.py
and runs under the mimicgen_torch2 env (skipped here when robosuite isn't
importable).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from policy_doctor.mimicgen.chained_warp_generator import (
    GenerationOutcome,
    IntermediateConstraint,
    _datagen_info_to_xy_yaw,
    _wrap_angle,
    derive_slack_from_stddev,
)
from policy_doctor.mimicgen.failure_targeting import DEFAULT_SQUARE_STATE_SCHEMA


# ---------------------------------------------------------------------------
# IntermediateConstraint.is_satisfied
# ---------------------------------------------------------------------------

def _t(x=0.0, y=0.0, z_rot=0.0):
    return {"x": x, "y": y, "z_rot": z_rot}


class TestIntermediateConstraintIsSatisfied(unittest.TestCase):
    def setUp(self):
        self.c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _t(x=0.10, y=0.20, z_rot=0.0)},
            slack={"nut": _t(x=0.03, y=0.03, z_rot=0.5)},
        )

    def test_inside_box_satisfied(self):
        sat, dists = self.c.is_satisfied({"nut": _t(x=0.12, y=0.21, z_rot=0.1)})
        self.assertTrue(sat)
        # worst axis ratio: max(0.02/0.03, 0.01/0.03, 0.1/0.5) = 0.667
        self.assertLess(dists["nut"], 1.0)

    def test_on_boundary_satisfied(self):
        # Exactly on the boundary should still satisfy (≤, not <).
        sat, dists = self.c.is_satisfied({"nut": _t(x=0.13, y=0.20, z_rot=0.0)})
        self.assertTrue(sat)
        self.assertAlmostEqual(dists["nut"], 1.0, places=4)

    def test_outside_x_axis_violates(self):
        sat, dists = self.c.is_satisfied({"nut": _t(x=0.15, y=0.20, z_rot=0.0)})
        self.assertFalse(sat)
        self.assertGreater(dists["nut"], 1.0)

    def test_outside_z_rot_violates(self):
        sat, _ = self.c.is_satisfied({"nut": _t(x=0.10, y=0.20, z_rot=1.0)})  # |dθ|=1.0 > 0.5
        self.assertFalse(sat)

    def test_angle_wraparound(self):
        # Target is z_rot=0; achieved is z_rot ≈ 2π (= 0 modulo wrap). x/y on target.
        sat, _ = self.c.is_satisfied({"nut": _t(x=0.10, y=0.20, z_rot=math.pi * 2 - 1e-3)})
        self.assertTrue(sat)

    def test_missing_object_fails_conservatively(self):
        sat, dists = self.c.is_satisfied({})
        self.assertFalse(sat)
        self.assertEqual(dists, {"nut": float("inf")})

    def test_objects_filter(self):
        # Constraint configured to only check "nut"; if we pass another obj,
        # it should be ignored.
        c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _t(x=0.10), "other": _t(x=99.0)},
            slack={"nut": _t(x=0.03), "other": _t(x=0.01)},
            objects=["nut"],
        )
        sat, _ = c.is_satisfied({"nut": _t(x=0.10), "other": _t(x=0.0)})
        self.assertTrue(sat)


# ---------------------------------------------------------------------------
# IntermediateConstraint.widen
# ---------------------------------------------------------------------------

class TestIntermediateConstraintWiden(unittest.TestCase):
    def test_widens_all_axes(self):
        c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _t(x=0.10)},
            slack={"nut": {"x": 0.03, "y": 0.03, "z_rot": 0.5}},
        )
        c2 = c.widen(2.0)
        self.assertEqual(c2.slack["nut"]["x"], 0.06)
        self.assertEqual(c2.slack["nut"]["y"], 0.06)
        self.assertEqual(c2.slack["nut"]["z_rot"], 1.0)
        # Original unchanged.
        self.assertEqual(c.slack["nut"]["x"], 0.03)

    def test_widen_preserves_other_fields(self):
        c = IntermediateConstraint(
            subtask_idx=3,
            target_pose={"nut": _t()},
            slack={"nut": {"x": 0.01, "y": 0.01, "z_rot": 0.1}},
            objects=["nut"],
        )
        c2 = c.widen(1.5)
        self.assertEqual(c2.subtask_idx, 3)
        self.assertEqual(c2.objects, ["nut"])


# ---------------------------------------------------------------------------
# derive_slack_from_stddev
# ---------------------------------------------------------------------------

class TestDeriveSlackFromStddev(unittest.TestCase):
    def test_typical_cluster(self):
        # Square schema (1 object "nut"). Feature is [x, y, sin, cos].
        stddev = [0.02, 0.015, 0.1, 0.1]  # moderately tight
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA, alpha=1.5,
        )
        # 1.5 × 0.02 = 0.030
        self.assertAlmostEqual(slack["nut"]["x"], 0.03, places=4)
        self.assertAlmostEqual(slack["nut"]["y"], 0.0225, places=4)
        # angular slack: 1.5 × sqrt(0.01 + 0.01) ≈ 1.5 × 0.1414 ≈ 0.2121
        self.assertAlmostEqual(slack["nut"]["z_rot"], 1.5 * math.sqrt(0.02), places=3)

    def test_clamps_lower_bound(self):
        # Cluster is a single point → stddev = 0. Slack should clamp to min.
        stddev = [0.0, 0.0, 0.0, 0.0]
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA, alpha=1.5,
            min_slack_xy=0.005, min_slack_z_rot=0.05,
        )
        self.assertEqual(slack["nut"]["x"], 0.005)
        self.assertEqual(slack["nut"]["y"], 0.005)
        self.assertEqual(slack["nut"]["z_rot"], 0.05)

    def test_clamps_upper_bound(self):
        # Cluster spans the whole workspace → huge stddev. Slack clamps to max.
        stddev = [10.0, 10.0, 1.0, 1.0]
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA, alpha=2.0,
            max_slack_xy=0.05, max_slack_z_rot=1.0,
        )
        self.assertEqual(slack["nut"]["x"], 0.05)
        self.assertEqual(slack["nut"]["y"], 0.05)
        self.assertEqual(slack["nut"]["z_rot"], 1.0)

    def test_default_clamps_are_realistic_for_manipulation(self):
        # Default upper bound for xy should be O(cm), not O(workspace).
        stddev = [10.0, 10.0, 1.0, 1.0]
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
        )
        # 3 cm is the upper bound — anything larger is a no-op constraint on
        # a typical 30-cm tabletop workspace.
        self.assertLessEqual(slack["nut"]["x"], 0.03 + 1e-9)
        self.assertLessEqual(slack["nut"]["y"], 0.03 + 1e-9)


# ---------------------------------------------------------------------------
# _wrap_angle and _datagen_info_to_xy_yaw
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(_wrap_angle(0.0), 0.0, places=6)
        self.assertAlmostEqual(_wrap_angle(2 * math.pi - 0.1), -0.1, places=6)
        self.assertAlmostEqual(_wrap_angle(-3 * math.pi), -math.pi, places=4)

    def test_datagen_info_4x4_pose(self):
        # Build a fake datagen_info with a 4x4 pose matrix.
        # Yaw of pi/4 around z: R[0,0]=cos, R[1,0]=sin.
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        pose = np.array([
            [ c, -s, 0.0, 0.123],
            [ s,  c, 0.0, 0.456],
            [0,   0, 1.0, 0.789],
            [0,   0, 0.0, 1.000],
        ])

        class FakeInfo:
            object_poses = {"nut": pose}
        out = _datagen_info_to_xy_yaw(FakeInfo())
        self.assertIn("nut", out)
        self.assertAlmostEqual(out["nut"]["x"], 0.123, places=5)
        self.assertAlmostEqual(out["nut"]["y"], 0.456, places=5)
        self.assertAlmostEqual(out["nut"]["z_rot"], math.pi / 4, places=5)

    def test_datagen_info_empty(self):
        class FakeInfo:
            object_poses = {}
        self.assertEqual(_datagen_info_to_xy_yaw(FakeInfo()), {})


# ---------------------------------------------------------------------------
# GenerationOutcome
# ---------------------------------------------------------------------------

class TestGenerationOutcome(unittest.TestCase):
    def test_defaults_fillable(self):
        o = GenerationOutcome(task_success=True, constraint_met=True, failure_reason=None)
        self.assertEqual(o.subtasks_executed, 0)
        self.assertEqual(o.distances, {})


if __name__ == "__main__":
    unittest.main()

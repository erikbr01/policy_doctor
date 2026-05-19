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
    cluster_to_chained_warp_constraint,
    derive_slack_from_stddev,
)
from policy_doctor.mimicgen.failure_targeting import DEFAULT_SQUARE_STATE_SCHEMA


# ---------------------------------------------------------------------------
# IntermediateConstraint.is_satisfied
# ---------------------------------------------------------------------------

def _yaw_quat(theta: float) -> dict[str, float]:
    """Quaternion (wxyz) representing a rotation of theta about world Z."""
    return {
        "qw": math.cos(theta / 2.0),
        "qx": 0.0,
        "qy": 0.0,
        "qz": math.sin(theta / 2.0),
    }


def _pose(x=0.0, y=0.0, z=0.0, z_rot=0.0):
    """Build a {x, y, z, qw, qx, qy, qz} dict for a yaw-only rotation."""
    return {"x": x, "y": y, "z": z, **_yaw_quat(z_rot)}


def _slack(x=0.03, y=0.03, z=0.03, rotation=0.5):
    return {"x": x, "y": y, "z": z, "rotation": rotation}


class TestIntermediateConstraintIsSatisfied(unittest.TestCase):
    def setUp(self):
        self.c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _pose(x=0.10, y=0.20, z=0.0, z_rot=0.0)},
            slack={"nut": _slack(x=0.03, y=0.03, z=0.03, rotation=0.5)},
        )

    def test_inside_box_satisfied(self):
        sat, dists = self.c.is_satisfied(
            {"nut": _pose(x=0.12, y=0.21, z=0.005, z_rot=0.1)}
        )
        self.assertTrue(sat)
        self.assertLess(dists["nut"], 1.0)

    def test_on_boundary_satisfied(self):
        # Exactly at the x-axis boundary.
        sat, dists = self.c.is_satisfied(
            {"nut": _pose(x=0.13, y=0.20, z=0.0, z_rot=0.0)}
        )
        self.assertTrue(sat)
        self.assertAlmostEqual(dists["nut"], 1.0, places=4)

    def test_outside_x_axis_violates(self):
        sat, dists = self.c.is_satisfied(
            {"nut": _pose(x=0.15, y=0.20, z=0.0, z_rot=0.0)}
        )
        self.assertFalse(sat)
        self.assertGreater(dists["nut"], 1.0)

    def test_outside_z_translation_violates(self):
        # 5 cm above target with 3 cm slack on z.
        sat, _ = self.c.is_satisfied(
            {"nut": _pose(x=0.10, y=0.20, z=0.05, z_rot=0.0)}
        )
        self.assertFalse(sat)

    def test_outside_rotation_violates(self):
        # 1.0 rad of yaw, slack is 0.5 rad.
        sat, _ = self.c.is_satisfied(
            {"nut": _pose(x=0.10, y=0.20, z=0.0, z_rot=1.0)}
        )
        self.assertFalse(sat)

    def test_quaternion_double_cover_satisfied(self):
        # q and -q are the same rotation; the constraint must use |dot|.
        target_q = self.c.target_pose["nut"]
        flipped = {
            "x": 0.10, "y": 0.20, "z": 0.0,
            "qw": -target_q["qw"], "qx": -target_q["qx"],
            "qy": -target_q["qy"], "qz": -target_q["qz"],
        }
        sat, _ = self.c.is_satisfied({"nut": flipped})
        self.assertTrue(sat)

    def test_angle_wraparound(self):
        # 2π yaw = 0 yaw (full revolution). Quaternion encoding handles this
        # naturally; achieved is *literally* the target quaternion.
        sat, _ = self.c.is_satisfied(
            {"nut": _pose(x=0.10, y=0.20, z=0.0, z_rot=2 * math.pi)}
        )
        self.assertTrue(sat)

    def test_missing_object_fails_conservatively(self):
        sat, dists = self.c.is_satisfied({})
        self.assertFalse(sat)
        self.assertEqual(dists, {"nut": float("inf")})

    def test_objects_filter(self):
        c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _pose(x=0.10), "other": _pose(x=99.0)},
            slack={"nut": _slack(), "other": _slack(x=0.01)},
            objects=["nut"],
        )
        sat, _ = c.is_satisfied({"nut": _pose(x=0.10), "other": _pose(x=0.0)})
        self.assertTrue(sat)

    def test_none_slack_skips_axis(self):
        # Setting slack.z to None should skip the z check, allowing arbitrary z.
        c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _pose(x=0.10, y=0.20, z=0.0)},
            slack={"nut": {"x": 0.03, "y": 0.03, "z": None, "rotation": 0.5}},
        )
        sat, _ = c.is_satisfied({"nut": _pose(x=0.10, y=0.20, z=99.0)})
        self.assertTrue(sat)


# ---------------------------------------------------------------------------
# IntermediateConstraint.widen
# ---------------------------------------------------------------------------

class TestIntermediateConstraintWiden(unittest.TestCase):
    def test_widens_all_axes(self):
        c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _pose(x=0.10)},
            slack={"nut": {"x": 0.03, "y": 0.03, "z": 0.03, "rotation": 0.5}},
        )
        c2 = c.widen(2.0)
        self.assertEqual(c2.slack["nut"]["x"], 0.06)
        self.assertEqual(c2.slack["nut"]["y"], 0.06)
        self.assertEqual(c2.slack["nut"]["z"], 0.06)
        self.assertEqual(c2.slack["nut"]["rotation"], 1.0)
        # Original unchanged.
        self.assertEqual(c.slack["nut"]["x"], 0.03)

    def test_widen_preserves_other_fields(self):
        c = IntermediateConstraint(
            subtask_idx=3,
            target_pose={"nut": _pose()},
            slack={"nut": {"x": 0.01, "y": 0.01, "z": 0.01, "rotation": 0.1}},
            objects=["nut"],
        )
        c2 = c.widen(1.5)
        self.assertEqual(c2.subtask_idx, 3)
        self.assertEqual(c2.objects, ["nut"])

    def test_widen_preserves_none(self):
        c = IntermediateConstraint(
            subtask_idx=0,
            target_pose={"nut": _pose()},
            slack={"nut": {"x": 0.01, "y": 0.01, "z": None, "rotation": 0.1}},
        )
        c2 = c.widen(2.0)
        self.assertIsNone(c2.slack["nut"]["z"])
        self.assertEqual(c2.slack["nut"]["x"], 0.02)


# ---------------------------------------------------------------------------
# derive_slack_from_stddev
# ---------------------------------------------------------------------------

class TestDeriveSlackFromStddev(unittest.TestCase):
    def test_typical_cluster(self):
        # 7-dim per-object feature: [x, y, z, qw, qx, qy, qz].
        # Translation stddev modest; quaternion stddev concentrated in qz
        # (yaw-only rotation has all its variance in qz with qw close to 1).
        stddev = [0.02, 0.015, 0.005, 0.0, 0.0, 0.0, 0.05]
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA, alpha=1.5,
        )
        self.assertAlmostEqual(slack["nut"]["x"], 0.03, places=4)
        self.assertAlmostEqual(slack["nut"]["y"], 0.0225, places=4)
        self.assertAlmostEqual(slack["nut"]["z"], 0.0075, places=4)
        # angular slack: alpha × 2 × ||q_stddev||  →  1.5 × 2 × 0.05 = 0.15
        self.assertAlmostEqual(slack["nut"]["rotation"], 0.15, places=4)

    def test_clamps_lower_bound(self):
        stddev = [0.0] * 7
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA, alpha=1.5,
            min_slack_xyz=0.005, min_slack_rotation=0.05,
        )
        self.assertEqual(slack["nut"]["x"], 0.005)
        self.assertEqual(slack["nut"]["y"], 0.005)
        self.assertEqual(slack["nut"]["z"], 0.005)
        self.assertEqual(slack["nut"]["rotation"], 0.05)

    def test_clamps_upper_bound(self):
        stddev = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA, alpha=2.0,
            max_slack_xyz=0.05, max_slack_rotation=1.0,
        )
        self.assertEqual(slack["nut"]["x"], 0.05)
        self.assertEqual(slack["nut"]["y"], 0.05)
        self.assertEqual(slack["nut"]["z"], 0.05)
        self.assertEqual(slack["nut"]["rotation"], 1.0)

    def test_default_clamps_are_realistic_for_manipulation(self):
        # Default upper bound for translation should be O(cm), not O(workspace).
        stddev = [10.0] * 7
        slack = derive_slack_from_stddev(
            stddev, state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
        )
        for axis in ("x", "y", "z"):
            self.assertLessEqual(slack["nut"][axis], 0.03 + 1e-9)
        self.assertLessEqual(slack["nut"]["rotation"], 0.5 + 1e-9)


# ---------------------------------------------------------------------------
# _wrap_angle and _datagen_info_to_xy_yaw
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(_wrap_angle(0.0), 0.0, places=6)
        self.assertAlmostEqual(_wrap_angle(2 * math.pi - 0.1), -0.1, places=6)
        self.assertAlmostEqual(_wrap_angle(-3 * math.pi), -math.pi, places=4)

    def test_datagen_info_4x4_pose(self):
        # 4x4 pose with yaw=pi/4 around z. Expect SE(3) dict back.
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
        self.assertAlmostEqual(out["nut"]["z"], 0.789, places=5)
        # Yaw of pi/4 → quaternion (cos(π/8), 0, 0, sin(π/8)).
        self.assertAlmostEqual(out["nut"]["qw"], math.cos(math.pi / 8), places=5)
        self.assertAlmostEqual(out["nut"]["qz"], math.sin(math.pi / 8), places=5)
        self.assertAlmostEqual(out["nut"]["qx"], 0.0, places=5)
        self.assertAlmostEqual(out["nut"]["qy"], 0.0, places=5)

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


# ---------------------------------------------------------------------------
# cluster_to_chained_warp_constraint
# ---------------------------------------------------------------------------

class TestClusterToChainedWarpConstraint(unittest.TestCase):
    def test_builds_constraint_from_cluster(self):
        # SE(3) cluster centre: nut at (0.10, 0.20, 0.05), identity rotation
        # (qw=1, qx=qy=qz=0).
        center = [0.10, 0.20, 0.05,  1.0, 0.0, 0.0, 0.0]
        # Stddev: 1 cm xy, 5 mm z, small quaternion spread in qz only.
        stddev = [0.01, 0.01, 0.005,  0.0, 0.0, 0.0, 0.05]
        cw = cluster_to_chained_warp_constraint(
            center_feature=center,
            stddev_feature=stddev,
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
            subtask_idx=0,
            slack_alpha=1.5,
        )
        self.assertEqual(cw["subtask_idx"], 0)
        self.assertEqual(cw["slack_widen_factor"], 2.0)
        self.assertAlmostEqual(cw["target_pose"]["nut"]["x"], 0.10, places=4)
        self.assertAlmostEqual(cw["target_pose"]["nut"]["y"], 0.20, places=4)
        self.assertAlmostEqual(cw["target_pose"]["nut"]["z"], 0.05, places=4)
        self.assertAlmostEqual(cw["target_pose"]["nut"]["qw"], 1.0, places=4)
        # No `z_rot` in the constraint payload — only the SE(3) keys.
        self.assertNotIn("z_rot", cw["target_pose"]["nut"])
        # Slack: alpha × stddev clamped to [3 mm, 3 cm] for xyz.
        self.assertAlmostEqual(cw["slack"]["nut"]["x"], 0.015, places=4)
        self.assertAlmostEqual(cw["slack"]["nut"]["y"], 0.015, places=4)
        self.assertAlmostEqual(cw["slack"]["nut"]["z"], 0.0075, places=4)
        # Rotation slack: 1.5 × 2 × ||q_stddev|| = 1.5 × 2 × 0.05 = 0.15.
        self.assertAlmostEqual(cw["slack"]["nut"]["rotation"], 0.15, places=4)

    def test_objects_filter_passthrough(self):
        center = [0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0]
        stddev = [0.001] * 3 + [0.0] * 4
        cw = cluster_to_chained_warp_constraint(
            center_feature=center,
            stddev_feature=stddev,
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA,
            subtask_idx=1,
            objects=["nut"],
        )
        self.assertEqual(cw["objects"], ["nut"])

    def test_satisfiable_by_perfect_match(self):
        center = [0.10, 0.20, 0.05,  1.0, 0.0, 0.0, 0.0]
        stddev = [0.005, 0.005, 0.003,  0.0, 0.0, 0.0, 0.02]
        cw = cluster_to_chained_warp_constraint(
            center_feature=center, stddev_feature=stddev,
            state_schema=DEFAULT_SQUARE_STATE_SCHEMA, subtask_idx=0,
        )
        c = IntermediateConstraint(
            subtask_idx=cw["subtask_idx"],
            target_pose=cw["target_pose"],
            slack=cw["slack"],
        )
        sat, _ = c.is_satisfied(cw["target_pose"])
        self.assertTrue(sat)


if __name__ == "__main__":
    unittest.main()

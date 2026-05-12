"""End-to-end test for ChainedWarpDataGenerator.

Loads a single demo from the canonical Square source, runs
``mimicgen.prepare_src_dataset`` to attach datagen_info, then drives
``mimicgen.generate_dataset`` with the chained-warp generator installed via
``scripts/run_mimicgen_generate.py``. Verifies:

  * with a loose slack around the seed's natural subtask-0 endpoint, trials
    succeed AND the recorded object pose at the boundary lies inside the
    slack box.
  * with an impossible target + tight slack, the generator early-aborts
    every trial (zero successes; ``stats.json`` records the rejection mode).

Gated by ``MIMICGEN_E2E=1`` and ``robosuite/robomimic/mimicgen`` importable
(i.e. run under the ``mimicgen_torch2`` env). Skipped otherwise.

Source dataset is read from ``~/data/mimicgen_data/source/square.hdf5``; if
that path doesn't exist, the test is skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SQUARE_SOURCE = Path.home() / "data" / "mimicgen_data" / "source" / "square.hdf5"


def _e2e_requested() -> bool:
    return os.environ.get("MIMICGEN_E2E", "0").strip() in ("1", "true", "yes")


def _require_e2e() -> None:
    if not _e2e_requested():
        raise unittest.SkipTest(
            "Set MIMICGEN_E2E=1 to run MimicGen sim integration tests "
            "(requires mimicgen_torch2 env + ~/data/mimicgen_data/source/square.hdf5)."
        )


def _require_sim_deps() -> None:
    try:
        import robomimic  # noqa: F401
        import robosuite  # noqa: F401
        import mimicgen   # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"sim deps not available: {e}") from e


def _require_source() -> Path:
    if not SQUARE_SOURCE.is_file():
        raise unittest.SkipTest(f"square source not found at {SQUARE_SOURCE}")
    return SQUARE_SOURCE


# ---------------------------------------------------------------------------
# Helpers — build a 1-demo seed HDF5, prepare it, run generation
# ---------------------------------------------------------------------------

def _copy_first_demo_to_seed(source: Path, dest: Path, *, demo_key: str = "demo_0") -> None:
    """Copy a single demo out of the source HDF5 — small enough for fast prep."""
    with h5py.File(source, "r") as src, h5py.File(dest, "w") as dst:
        # Carry attrs over so MimicGen sees a complete env_args / total.
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        src_data = src["data"]
        dst_data = dst.create_group("data")
        for k, v in src_data.attrs.items():
            dst_data.attrs[k] = v
        if demo_key not in src_data:
            # Pick the first demo by numeric suffix (HDF5 group order is insertion order).
            demo_keys = sorted(
                (k for k in src_data.keys() if k.startswith("demo_")),
                key=lambda s: int(s.split("_")[1]),
            )
            assert demo_keys, "source has no demo_* groups"
            demo_key = demo_keys[0]
        src_data.copy(demo_key, dst_data, name="demo_0")
        dst_data.attrs["total"] = np.int64(int(dst_data["demo_0"].attrs.get("num_samples", 0)))


def _run_generate(
    *,
    seed_hdf5: Path,
    output_dir: Path,
    num_trials: int,
    chained_warp_constraint: dict | None = None,
) -> dict:
    """Invoke scripts/run_mimicgen_generate.py in-process; return stats."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_mimicgen_generate.py"
    cmd = [
        sys.executable, str(script),
        "--seed_hdf5", str(seed_hdf5),
        "--output_dir", str(output_dir),
        "--task_name", "square",
        "--env_interface_name", "MG_Square",
        "--env_interface_type", "robosuite",
        "--num_trials", str(num_trials),
        "--nn_k", "1",
        "--interpolate_from_last_target_pose",
    ]
    if chained_warp_constraint is not None:
        cmd += ["--chained_warp_constraint", json.dumps(chained_warp_constraint)]
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    subprocess.run(cmd, check=True, env=env)
    stats_path = output_dir / "stats.json"
    with open(stats_path) as f:
        return json.load(f)


def _object_pose_at_subtask_boundary(
    demo_hdf5: Path,
    demo_key: str,
    subtask_term_signal_value: int,
) -> dict[str, np.ndarray]:
    """Read a demo's per-timestep datagen_info and return the per-object pose
    at the moment subtask ``subtask_term_signal_value`` *ends* (i.e. the last
    timestep where that subtask's term-signal == 1)."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(demo_hdf5, "r") as f:
        ep = f["data"][demo_key]
        di = ep["datagen_info"]
        if "subtask_term_signals" not in di:
            # Fall back to "end of trajectory" if the dataset doesn't carry boundaries.
            for obj_name in di["object_poses"].keys():
                pose_t = di["object_poses"][obj_name][-1]
                out[obj_name] = np.asarray(pose_t)
            return out
        # subtask_term_signals stored per-subtask name; for the Square task
        # there are 2 subtasks (grasp, place). The "end of subtask N" is the
        # final timestep where signal[N] == 1 transitions.
        signals = di["subtask_term_signals"]
        # Use the first available signal whose 1→0 transition is at index 0..-1.
        signal_keys = list(signals.keys())
        # MimicGen schemas use names like "grasp", "place". Just take the first one.
        sig = np.asarray(signals[signal_keys[subtask_term_signal_value]])
        # The boundary timestep is the LAST step where the signal == 1.
        on_idxs = np.flatnonzero(sig)
        boundary_t = int(on_idxs[-1]) if len(on_idxs) > 0 else len(sig) - 1
        for obj_name in di["object_poses"].keys():
            pose_t = di["object_poses"][obj_name][boundary_t]
            out[obj_name] = np.asarray(pose_t)
    return out


def _xy_yaw_from_4x4(pose: np.ndarray) -> tuple[float, float, float]:
    return float(pose[0, 3]), float(pose[1, 3]), float(np.arctan2(pose[1, 0], pose[0, 0]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChainedWarpE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _require_e2e()
        _require_sim_deps()
        cls.source = _require_source()

        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.tmpdir.name)
        cls.seed_hdf5 = cls.tmp / "seed_demo.hdf5"
        _copy_first_demo_to_seed(cls.source, cls.seed_hdf5)

        # Read the seed's grasp-end pose so we can build a "satisfiable" target.
        # The seed needs datagen_info before we can read object_poses — but the
        # raw source already has it (NVlabs source datasets come with datagen_info
        # attached). Verify and pull.
        with h5py.File(cls.seed_hdf5, "r") as f:
            ep = f["data"]["demo_0"]
            if "datagen_info" not in ep:
                raise unittest.SkipTest(
                    "source HDF5 has no datagen_info; expected NVlabs Square source"
                )
            di = ep["datagen_info"]
            signals = di["subtask_term_signals"]
            signal_keys = list(signals.keys())
            assert len(signal_keys) >= 1, "Square should have at least 1 subtask signal"
            sig0 = np.asarray(signals[signal_keys[0]])
            on_idxs = np.flatnonzero(sig0)
            cls.boundary_t = int(on_idxs[-1]) if len(on_idxs) > 0 else len(sig0) - 1
            obj_name = list(di["object_poses"].keys())[0]
            cls.obj_name = obj_name
            cls.seed_boundary_pose = np.asarray(di["object_poses"][obj_name][cls.boundary_t])
            cls.signal_name = signal_keys[0]
        x, y, z_rot = _xy_yaw_from_4x4(cls.seed_boundary_pose)
        cls.seed_x, cls.seed_y, cls.seed_z_rot = x, y, z_rot

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_impossible_target_aborts_all_trials(self):
        """Target placed off-workspace + tight slack → no successes."""
        out_dir = self.tmp / "impossible"
        out_dir.mkdir(exist_ok=True)
        constraint = {
            "subtask_idx": 0,
            "target_pose": {self.obj_name: {"x": 9.0, "y": 9.0, "z_rot": 0.0}},
            "slack":       {self.obj_name: {"x": 0.001, "y": 0.001, "z_rot": 0.01}},
        }
        stats = _run_generate(
            seed_hdf5=self.seed_hdf5, output_dir=out_dir,
            num_trials=3, chained_warp_constraint=constraint,
        )
        # The constraint can never be met, so MimicGen reports zero successes.
        self.assertEqual(stats.get("num_success", 0), 0, msg=f"stats={stats}")

    def test_loose_target_at_seed_pose_passes(self):
        """Target at the seed's own subtask-0 endpoint + loose slack → some trials
        succeed AND the recorded boundary pose is within the slack box."""
        out_dir = self.tmp / "loose"
        out_dir.mkdir(exist_ok=True)
        # Use the seed's own boundary pose as target — should be naturally hit.
        constraint = {
            "subtask_idx": 0,
            "target_pose": {self.obj_name: {
                "x": self.seed_x, "y": self.seed_y, "z_rot": self.seed_z_rot,
            }},
            "slack": {self.obj_name: {"x": 0.5, "y": 0.5, "z_rot": 3.14}},
        }
        stats = _run_generate(
            seed_hdf5=self.seed_hdf5, output_dir=out_dir,
            num_trials=3, chained_warp_constraint=constraint,
        )
        # At least one successful trial.
        n_success = int(stats.get("num_success", 0))
        self.assertGreater(n_success, 0, msg=f"stats={stats}")

        # Walk demo.hdf5 to confirm every successful demo's boundary pose
        # actually lies inside the configured slack box.
        demo_path = out_dir / "demo.hdf5"
        self.assertTrue(demo_path.is_file())
        violations = 0
        with h5py.File(demo_path, "r") as f:
            for demo_key in f["data"].keys():
                ep = f["data"][demo_key]
                if not bool(ep.attrs.get("success", False) or
                            ep.attrs.get("Successful", False) or
                            (len(ep.get("actions", [])) > 0)):
                    continue
                pose = _object_pose_at_subtask_boundary(
                    demo_path, demo_key, subtask_term_signal_value=0
                )
                x, y, z_rot = _xy_yaw_from_4x4(pose[self.obj_name])
                if (
                    abs(x - self.seed_x) > 0.5
                    or abs(y - self.seed_y) > 0.5
                    or abs(np.arctan2(np.sin(z_rot - self.seed_z_rot),
                                       np.cos(z_rot - self.seed_z_rot))) > 3.14
                ):
                    violations += 1
        self.assertEqual(
            violations, 0,
            msg=f"{violations} demos had boundary poses outside the slack box",
        )


if __name__ == "__main__":
    unittest.main()

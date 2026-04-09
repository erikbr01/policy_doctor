"""Unit tests for MimicGen seed trajectory types and robomimic HDF5 materialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy_doctor.paths import MIMICGEN_ROOT, PROJECT_ROOT
from tests.support.mimicgen_seed.pipeline import (
    ensure_mimicgen_importable,
    materialize_robomimic_seed_hdf5,
)
from tests.support.mimicgen_seed.robomimic_source import (
    LiberoRobomimicSeedMaterializer,
    RobocasaRobomimicSeedMaterializer,
    RobomimicSeedMaterializer,
    SeedDatasetMaterializer,
)
from tests.support.mimicgen_seed.schema import (
    MimicGenBinding,
    PolicyRolloutTrajectory,
    SimulationBackend,
    validate_policy_rollout_trajectory,
)


def _minimal_env_meta() -> dict:
    return {
        "env_name": "DummyEnv",
        "type": 1,
        "env_kwargs": {},
    }


def _valid_traj(model_file: str | None = None) -> PolicyRolloutTrajectory:
    return PolicyRolloutTrajectory(
        states=np.zeros((3, 5), dtype=np.float32),
        actions=np.zeros((3, 7), dtype=np.float32),
        env_meta=_minimal_env_meta(),
        model_file=model_file,
    )


class TestPolicyRolloutTrajectoryValidation(unittest.TestCase):
    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            PolicyRolloutTrajectory(
                states=np.zeros((2, 3)),
                actions=np.zeros((4, 3)),
                env_meta=_minimal_env_meta(),
            )

    def test_rejects_non_2d_states(self):
        with self.assertRaises(ValueError):
            PolicyRolloutTrajectory(
                states=np.zeros(3),
                actions=np.zeros((3, 1)),
                env_meta=_minimal_env_meta(),
            )

    def test_rejects_empty_env_meta(self):
        with self.assertRaises(ValueError):
            PolicyRolloutTrajectory(
                states=np.zeros((1, 2)),
                actions=np.zeros((1, 2)),
                env_meta={},
            )

    def test_validate_policy_rollout_trajectory_explicit(self):
        traj = _valid_traj()
        validate_policy_rollout_trajectory(traj)


class TestMimicGenBinding(unittest.TestCase):
    def test_as_prepare_kwargs_keys(self):
        b = MimicGenBinding(env_interface_name="MG_Stack", env_interface_type="robosuite")
        kw = b.as_prepare_kwargs()
        self.assertEqual(
            kw,
            {"env_interface_name": "MG_Stack", "env_interface_type": "robosuite"},
        )


class TestRobomimicSeedMaterializer(unittest.TestCase):
    def test_writes_expected_hdf5_structure(self):
        traj = _valid_traj(model_file="<mujoco model='dummy'/>")
        mat = RobomimicSeedMaterializer()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seed.hdf5"
            mat.write_source_dataset(traj, path, demo_key="demo_0")
            with h5py.File(path, "r") as f:
                self.assertIn("data", f)
                self.assertEqual(f["data"].attrs["total"], 3)
                env_args = json.loads(f["data"].attrs["env_args"])
                self.assertEqual(env_args["env_name"], "DummyEnv")
                demo = f["data/demo_0"]
                self.assertEqual(demo["actions"].shape, (3, 7))
                self.assertEqual(demo["states"].shape, (3, 5))
                self.assertEqual(demo.attrs["num_samples"], 3)
                self.assertEqual(
                    demo.attrs["model_file"],
                    "<mujoco model='dummy'/>",
                )

    def test_materialize_robomimic_seed_hdf5_alias(self):
        traj = _valid_traj()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.hdf5"
            out = materialize_robomimic_seed_hdf5(traj, path)
            self.assertTrue(out.is_file())

    def test_libero_and_robocasa_materializers_are_protocol_compatible(self):
        self.assertIsInstance(RobomimicSeedMaterializer(), SeedDatasetMaterializer)
        self.assertIsInstance(LiberoRobomimicSeedMaterializer(), SeedDatasetMaterializer)
        self.assertIsInstance(RobocasaRobomimicSeedMaterializer(), SeedDatasetMaterializer)
        self.assertEqual(LiberoRobomimicSeedMaterializer().backend, SimulationBackend.LIBERO_ROBO_MIMIC)
        self.assertEqual(
            RobocasaRobomimicSeedMaterializer().backend,
            SimulationBackend.ROBOCASA_ROBO_MIMIC,
        )


class TestEnsureMimicgenImportable(unittest.TestCase):
    def test_inserts_vendored_repo_on_path(self):
        self.assertTrue(MIMICGEN_ROOT.is_dir(), f"expected submodule at {MIMICGEN_ROOT}")
        root = ensure_mimicgen_importable()
        self.assertEqual(root, MIMICGEN_ROOT.resolve())
        import sys

        self.assertIn(str(MIMICGEN_ROOT.resolve()), sys.path)

    def test_import_mimicgen_after_ensure(self):
        ensure_mimicgen_importable()
        import mimicgen  # noqa: F401

        mf = Path(mimicgen.__file__).resolve()
        root = PROJECT_ROOT.resolve()
        try:
            mf.relative_to(root)
        except ValueError:
            self.fail(f"expected mimicgen under project root {root}, got {mf}")


class TestRunMimicgenPrepareOptional(unittest.TestCase):
    """Smoke import for full stack; skipped when robomimic is not installed."""

    def test_prepare_src_dataset_import(self):
        ensure_mimicgen_importable()
        try:
            import robomimic  # noqa: F401
        except ImportError:
            self.skipTest("robomimic not installed")
        from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

        self.assertTrue(callable(prepare_src_dataset))

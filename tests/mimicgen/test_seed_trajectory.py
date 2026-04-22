"""Unit tests for policy_doctor.mimicgen.seed_trajectory."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy_doctor.mimicgen.seed_trajectory import (
    MimicGenSeedTrajectory,
    SeedSource,
)


def _minimal_env_meta() -> dict:
    return {"env_name": "DummyEnv", "type": 1, "env_kwargs": {}}


def _write_minimal_hdf5(
    path: Path,
    *,
    n_timesteps: int = 5,
    state_dim: int = 4,
    action_dim: int = 3,
    model_file: str | None = "<mujoco/>",
    demo_key: str = "demo_0",
) -> None:
    states = np.zeros((n_timesteps, state_dim), dtype=np.float32)
    actions = np.zeros((n_timesteps, action_dim), dtype=np.float32)
    env_meta = _minimal_env_meta()
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = np.int64(n_timesteps)
        data.attrs["env_args"] = json.dumps(env_meta)
        ep = data.create_group(demo_key)
        ep.create_dataset("actions", data=actions, compression="gzip")
        ep.create_dataset("states", data=states, compression="gzip")
        ep.attrs["num_samples"] = np.int64(n_timesteps)
        if model_file is not None:
            ep.attrs["model_file"] = model_file


class TestFromRobomimicHdf5Demo(unittest.TestCase):
    def test_loads_states_and_actions(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "src.hdf5"
            _write_minimal_hdf5(p, n_timesteps=7, state_dim=6, action_dim=4)
            traj = MimicGenSeedTrajectory.from_robomimic_hdf5_demo(p)
        self.assertEqual(traj.states.shape, (7, 6))
        self.assertEqual(traj.actions.shape, (7, 4))
        self.assertEqual(traj.source, SeedSource.DEMONSTRATION)
        self.assertEqual(traj.model_file, "<mujoco/>")
        self.assertEqual(traj.env_meta["env_name"], "DummyEnv")

    def test_custom_demo_key(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "src.hdf5"
            _write_minimal_hdf5(p, demo_key="demo_3")
            traj = MimicGenSeedTrajectory.from_robomimic_hdf5_demo(p, demo_key="demo_3")
        self.assertEqual(traj.states.shape[0], 5)

    def test_no_model_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "src.hdf5"
            _write_minimal_hdf5(p, model_file=None)
            traj = MimicGenSeedTrajectory.from_robomimic_hdf5_demo(p)
        self.assertIsNone(traj.model_file)


class TestFromRolloutHdf5(unittest.TestCase):
    def test_source_tag_is_rollout(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rollout.hdf5"
            _write_minimal_hdf5(p)
            traj = MimicGenSeedTrajectory.from_rollout_hdf5(p)
        self.assertEqual(traj.source, SeedSource.ROLLOUT)

    def test_same_data_as_demo_factory(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rollout.hdf5"
            _write_minimal_hdf5(p, n_timesteps=3, state_dim=2, action_dim=2)
            traj_r = MimicGenSeedTrajectory.from_rollout_hdf5(p)
            traj_d = MimicGenSeedTrajectory.from_robomimic_hdf5_demo(p)
        np.testing.assert_array_equal(traj_r.states, traj_d.states)
        np.testing.assert_array_equal(traj_r.actions, traj_d.actions)


class TestFromPolicyRolloutTrajectory(unittest.TestCase):
    def _make_prt(self, n: int = 4) -> object:
        """Create a duck-typed PolicyRolloutTrajectory-like object."""
        from tests.support.mimicgen_seed.schema import PolicyRolloutTrajectory
        return PolicyRolloutTrajectory(
            states=np.zeros((n, 3), dtype=np.float32),
            actions=np.zeros((n, 2), dtype=np.float32),
            env_meta=_minimal_env_meta(),
            model_file="<xml/>",
        )

    def test_bridge_from_prt(self):
        prt = self._make_prt(4)
        traj = MimicGenSeedTrajectory.from_policy_rollout_trajectory(prt)
        self.assertEqual(traj.states.shape, (4, 3))
        self.assertEqual(traj.source, SeedSource.ROLLOUT)
        self.assertEqual(traj.model_file, "<xml/>")

    def test_bridge_with_demo_source(self):
        prt = self._make_prt(2)
        traj = MimicGenSeedTrajectory.from_policy_rollout_trajectory(
            prt, source=SeedSource.DEMONSTRATION
        )
        self.assertEqual(traj.source, SeedSource.DEMONSTRATION)

    def test_to_policy_rollout_trajectory_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "src.hdf5"
            _write_minimal_hdf5(p, n_timesteps=3)
            traj = MimicGenSeedTrajectory.from_robomimic_hdf5_demo(p)
        prt = traj.to_policy_rollout_trajectory()
        np.testing.assert_array_equal(traj.states, prt.states)
        np.testing.assert_array_equal(traj.actions, prt.actions)
        self.assertEqual(traj.model_file, prt.model_file)


class TestValidation(unittest.TestCase):
    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            MimicGenSeedTrajectory(
                states=np.zeros((3, 2)),
                actions=np.zeros((5, 2)),
                env_meta=_minimal_env_meta(),
                model_file=None,
                source=SeedSource.DEMONSTRATION,
            )

    def test_rejects_1d_states(self):
        with self.assertRaises(ValueError):
            MimicGenSeedTrajectory(
                states=np.zeros(3),
                actions=np.zeros((3, 2)),
                env_meta=_minimal_env_meta(),
                model_file=None,
                source=SeedSource.DEMONSTRATION,
            )

    def test_rejects_empty_env_meta(self):
        with self.assertRaises(ValueError):
            MimicGenSeedTrajectory(
                states=np.zeros((2, 2)),
                actions=np.zeros((2, 2)),
                env_meta={},
                model_file=None,
                source=SeedSource.DEMONSTRATION,
            )

    def test_rejects_empty_trajectory(self):
        with self.assertRaises(ValueError):
            MimicGenSeedTrajectory(
                states=np.zeros((0, 2)),
                actions=np.zeros((0, 2)),
                env_meta=_minimal_env_meta(),
                model_file=None,
                source=SeedSource.DEMONSTRATION,
            )

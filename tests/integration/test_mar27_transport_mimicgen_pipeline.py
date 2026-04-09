"""End-to-end checks: mar27 transport eval rollout → robomimic HDF5 (MimicGen-ready source).

Classes are split so failures localize to: layout, eval logs, pickle schema, replay,
HDF5 materialization, robomimic reload, MimicGen registry, and ``generate_dataset`` import.

Upstream MimicGen has no ``MG_*`` interface for ``TwoArmTransport``; this file still
builds a valid source HDF5 from a successful policy rollout for when you add one.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy_doctor.paths import CUPID_ROOT, REPO_ROOT
from tests.support.mimicgen_seed.pipeline import (
    ensure_mimicgen_importable,
    materialize_robomimic_seed_hdf5,
)
from tests.support.mimicgen_seed.schema import PolicyRolloutTrajectory


def _mar27_replay_script_path() -> Path | None:
    for base in (CUPID_ROOT, REPO_ROOT):
        p = base / "policy_doctor_mar27_mimicgen_replay.py"
        if p.is_file():
            return p
    return None


def _load_mar27_replay():
    path = _mar27_replay_script_path()
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("_pd_mar27_mimicgen_replay", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m27 = _load_mar27_replay()
if _m27 is not None:
    default_cupid_root = _m27.default_cupid_root
    ensure_cupid_repo_on_path = _m27.ensure_cupid_repo_on_path
    first_successful_episode_pkl = _m27.first_successful_episode_pkl
    mar27_eval_dir = _m27.mar27_eval_dir
    replay_transport_eval_pickle = _m27.replay_transport_eval_pickle
    _cupid_for_path = default_cupid_root()
    if _cupid_for_path is not None:
        try:
            ensure_cupid_repo_on_path(_cupid_for_path)
        except FileNotFoundError:
            pass
else:
    default_cupid_root = lambda: None  # type: ignore[assignment]
    mar27_eval_dir = None  # type: ignore[assignment]
    first_successful_episode_pkl = None  # type: ignore[assignment]
    replay_transport_eval_pickle = None  # type: ignore[assignment]


def _require_cupid_mar27_eval() -> Path:
    if _m27 is None:
        raise unittest.SkipTest(
            "policy_doctor_mar27_mimicgen_replay.py not found under third_party/cupid or REPO_ROOT."
        )
    root = default_cupid_root()
    if root is None:
        raise unittest.SkipTest(
            "Cupid checkout not found. Set CUPID_REPO_ROOT or vendor third_party/cupid."
        )
    ev = mar27_eval_dir(root)
    if not (ev / "eval_log.json").is_file():
        raise unittest.SkipTest(f"mar27 eval bundle missing (no eval_log.json): {ev}")
    ep_dir = ev / "episodes"
    if not ep_dir.is_dir() or not list(ep_dir.glob("ep*_succ.pkl")):
        raise unittest.SkipTest(f"No successful episode pickles under {ep_dir}")
    return root


def _require_diffusion_policy() -> None:
    try:
        import diffusion_policy  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"diffusion_policy not importable (use cupid conda env): {e}") from e


def _require_pytorch3d() -> None:
    try:
        import pytorch3d  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(
            "pytorch3d required for mar27 replay (diffusion_policy rotation_transformer); "
            f"use cupid conda env: {e}"
        ) from e


def _require_robomimic() -> None:
    try:
        import robomimic  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(
            f"robomimic required for mar27 HDF5 reload / MimicGen scripts; use cupid or mimicgen env: {e}"
        ) from e


class TestCupidMar27Layout(unittest.TestCase):
    """Filesystem layout only (no heavy imports)."""

    def test_eval_dir_path_matches_task_config(self):
        root = _require_cupid_mar27_eval()
        ev = mar27_eval_dir(root)
        self.assertTrue(ev.is_dir())
        self.assertEqual(ev.name, "latest")
        self.assertIn("mar27_train_diffusion_unet_lowdim_transport_mh_0", str(ev))


class TestMar27EvalLog(unittest.TestCase):
    """eval_log.json presence and basic structure."""

    def test_eval_log_has_aggregate_and_per_seed_keys(self):
        root = _require_cupid_mar27_eval()
        log_path = mar27_eval_dir(root) / "eval_log.json"
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        self.assertIn("test/mean_score", log)
        self.assertIn("test/sim_max_reward_100000", log)


class TestMar27SuccessfulEpisodePickle(unittest.TestCase):
    """Pickle schema for one successful rollout."""

    def test_first_succ_episode_schema(self):
        root = _require_cupid_mar27_eval()
        pkl = first_successful_episode_pkl(mar27_eval_dir(root))
        with open(pkl, "rb") as f:
            df = pickle.load(f)
        cols = set(df.columns)
        for c in ("action", "obs", "success", "timestep"):
            self.assertIn(c, cols, msg=f"missing column {c} in {pkl.name}")
        self.assertTrue(bool(df["success"].iloc[-1]), msg="expected successful terminal flag")
        a0 = np.asarray(df.iloc[0]["action"])
        self.assertEqual(a0.shape, (16, 20), msg="expected horizon 16 × 20 (dual-arm 6D)")


class TestMar27ReplayToStatesActions(unittest.TestCase):
    """Diffusion eval actions → flat mujoco states (replay)."""

    @classmethod
    def setUpClass(cls):
        _require_cupid_mar27_eval()
        _require_diffusion_policy()
        _require_pytorch3d()
        _require_robomimic()

    def test_replay_shapes_align_for_mimicgen(self):
        root = default_cupid_root()
        ensure_cupid_repo_on_path(root)
        pkl = first_successful_episode_pkl(mar27_eval_dir(root))
        states, actions, env_meta, xml = replay_transport_eval_pickle(root, pkl)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(actions.ndim, 2)
        self.assertEqual(states.shape[0], actions.shape[0])
        self.assertGreater(states.shape[0], 10)
        self.assertIsInstance(env_meta, dict)
        self.assertGreater(len(xml), 1000)


class TestMar27MaterializeSeedHdf5(unittest.TestCase):
    """Test-support writer → temp robomimic source HDF5."""

    @classmethod
    def setUpClass(cls):
        _require_cupid_mar27_eval()
        _require_diffusion_policy()
        _require_pytorch3d()
        _require_robomimic()

    def test_roundtrip_hdf5_keys_and_lengths(self):
        import tempfile

        root = default_cupid_root()
        pkl = first_successful_episode_pkl(mar27_eval_dir(root))
        states, actions, env_meta, xml = replay_transport_eval_pickle(root, pkl)
        traj = PolicyRolloutTrajectory(
            states=states.astype(np.float32),
            actions=actions.astype(np.float32),
            env_meta=env_meta,
            model_file=xml,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "seed_from_mar27.hdf5"
            materialize_robomimic_seed_hdf5(traj, out)
            with h5py.File(out, "r") as f:
                self.assertIn("data/demo_0/states", f)
                self.assertIn("data/demo_0/actions", f)
                ds_s = f["data/demo_0/states"][()]
                ds_a = f["data/demo_0/actions"][()]
                self.assertEqual(ds_s.shape[0], ds_a.shape[0])


class TestMar27RobomimicReloadSeedHdf5(unittest.TestCase):
    """robomimic can build an env from the written file and replay a few steps."""

    @classmethod
    def setUpClass(cls):
        _require_cupid_mar27_eval()
        _require_diffusion_policy()
        _require_pytorch3d()
        _require_robomimic()

    def test_reset_to_first_states(self):
        import tempfile

        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils

        root = default_cupid_root()
        pkl = first_successful_episode_pkl(mar27_eval_dir(root))
        states, actions, env_meta, xml = replay_transport_eval_pickle(root, pkl)
        traj = PolicyRolloutTrajectory(
            states=states.astype(np.float32),
            actions=actions.astype(np.float32),
            env_meta=env_meta,
            model_file=xml,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "seed.hdf5"
            materialize_robomimic_seed_hdf5(traj, out)
            meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(out))
            env = EnvUtils.create_env_for_data_processing(
                env_meta=meta,
                camera_names=[],
                camera_height=84,
                camera_width=84,
                reward_shaping=False,
            )
            with h5py.File(out, "r") as f:
                st = f["data/demo_0/states"][()]
                act = f["data/demo_0/actions"][()]
                model = f["data/demo_0"].attrs["model_file"]
            env.reset()
            env.reset_to({"states": st[0], "model": model})
            for t in range(min(5, len(act))):
                env.reset_to({"states": st[t]})
                env.step(act[t])


class TestMar27MimicGenPrepare(unittest.TestCase):
    """MimicGen ``prepare_src_dataset`` requires a task-specific ``MG_*`` interface."""

    @classmethod
    def setUpClass(cls):
        _require_cupid_mar27_eval()
        ensure_mimicgen_importable()

    def test_two_arm_transport_has_no_builtin_mg_interface(self):
        from mimicgen.env_interfaces.base import REGISTERED_ENV_INTERFACES

        rob = REGISTERED_ENV_INTERFACES.get("robosuite", {})
        names = " ".join(sorted(rob.keys()))
        self.assertNotIn("Transport", names)
        self.assertNotIn("TwoArm", names)


class TestMar27MimicGenGenerateDatasetEntrypoint(unittest.TestCase):
    """Sanity: MimicGen generate_dataset script is importable (actual run is task-specific)."""

    def test_generate_dataset_import(self):
        _require_cupid_mar27_eval()
        _require_robomimic()
        ensure_mimicgen_importable()
        from mimicgen.scripts.generate_dataset import generate_dataset  # noqa: F401

        self.assertTrue(callable(generate_dataset))

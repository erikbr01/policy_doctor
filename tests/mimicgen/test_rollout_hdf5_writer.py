"""Tests for the eval-runner rollouts.hdf5 writer.

Verifies that:

1. ``RobomimicLowdimWrapper`` exposes ``_get_simulator_state`` and
   ``_get_episode_model_file`` without a real MuJoCo sim.
2. ``VideoRecordingWrapper`` forwards those private methods to its inner env
   (gym.Wrapper.__getattr__ blocks ``_``-prefixed names, so explicit passthroughs
   are required — these tests confirm the passthroughs exist and work).
3. ``MultiStepWrapper`` buffers (state, action) pairs during ``step()`` and
   exposes them via ``_get_episode_sim_data()``.  The layout must satisfy
   MimicGen's ``prepare_src_dataset`` convention: ``states[t]`` = sim state
   *before* ``actions[t]`` is applied.
4. ``MimicgenLowdimRunner._write_rollouts_hdf5`` produces an HDF5 file
   compatible with ``MimicGenSeedTrajectory.from_rollout_hdf5`` (and therefore
   with ``prepare_src_dataset``).

Run with::

    conda run -n cupid python -m pytest tests/mimicgen/test_rollout_hdf5_writer.py -v
    # or via the run_tests.py suite runner:
    conda run -n cupid python run_tests.py --suite cupid
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

# Ensure diffusion_policy is importable when run from the repo root.
_CUPID_ROOT = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "cupid"
if str(_CUPID_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUPID_ROOT))


def _require_gym():
    try:
        import gym  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"gym not importable (use cupid conda env): {e}") from e


def _require_diffusion_policy():
    try:
        import diffusion_policy  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(
            f"diffusion_policy not importable (use cupid conda env): {e}"
        ) from e


# ---------------------------------------------------------------------------
# Minimal fake environments
# ---------------------------------------------------------------------------

def _make_fake_robomimic_wrapper():
    """Return a RobomimicLowdimWrapper instance backed by a fully mocked EnvRobosuite.

    The mock exposes:
    - ``get_state()`` returning ``{'states': np.array([1.0, 2.0, 3.0])}``
    - ``env.model.get_xml()`` returning a dummy XML string
    - ``get_observation()`` returning a zeros vector so reset() completes
    - ``action_dimension`` / ``is_success()`` for the wrapper's __init__
    """
    _require_gym()
    _require_diffusion_policy()
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import (
        RobomimicLowdimWrapper,
    )

    # Minimal robomimic EnvRobosuite mock.
    fake_sim_state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    fake_xml = "<mujoco><worldbody/></mujoco>"

    mock_env = MagicMock()
    mock_env.action_dimension = 7
    mock_env.get_state.return_value = {"states": fake_sim_state}
    mock_env.get_observation.return_value = {
        "object": np.zeros(10),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
    }
    mock_env.step.return_value = (
        {
            "object": np.zeros(10),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.zeros(2),
        },
        0.0,
        False,
        {},
    )
    mock_env.reset.return_value = None
    mock_env.is_success.return_value = {"task": False}
    # raw robosuite env: mock_env.env.model.get_xml()
    mock_env.env.model.get_xml.return_value = fake_xml

    wrapper = RobomimicLowdimWrapper(
        env=mock_env,
        obs_keys=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    )
    return wrapper, fake_sim_state, fake_xml


def _make_fake_video_recording_wrapper(inner_env):
    """Wrap inner_env in a VideoRecordingWrapper with a no-op VideoRecorder."""
    _require_diffusion_policy()
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder  # may not exist
    # VideoRecorder is actually in real_world; use a minimal mock instead.
    mock_recorder = MagicMock()
    mock_recorder.is_ready.return_value = False
    mock_recorder.stop.return_value = None
    return VideoRecordingWrapper(
        env=inner_env,
        video_recoder=mock_recorder,
        file_path=None,
    )


def _make_multistep_wrapper(inner_env, n_obs_steps=2, n_action_steps=4):
    _require_diffusion_policy()
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    return MultiStepWrapper(
        env=inner_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=50,
    )


# ---------------------------------------------------------------------------
# Test: RobomimicLowdimWrapper new methods
# ---------------------------------------------------------------------------

class TestRobomimicLowdimWrapperSimMethods(unittest.TestCase):
    """_get_simulator_state and _get_episode_model_file exist and return correct values."""

    def setUp(self):
        self.wrapper, self.fake_state, self.fake_xml = _make_fake_robomimic_wrapper()

    def test_get_simulator_state_returns_float64_array(self):
        state = self.wrapper._get_simulator_state()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.dtype, np.float64)
        np.testing.assert_array_equal(state, self.fake_state)

    def test_get_episode_model_file_returns_xml_string(self):
        xml = self.wrapper._get_episode_model_file()
        self.assertEqual(xml, self.fake_xml)

    def test_get_simulator_state_calls_get_state(self):
        _ = self.wrapper._get_simulator_state()
        self.wrapper.env.get_state.assert_called()

    def test_get_episode_model_file_calls_inner_env_model(self):
        _ = self.wrapper._get_episode_model_file()
        self.wrapper.env.env.model.get_xml.assert_called()


# ---------------------------------------------------------------------------
# Test: VideoRecordingWrapper passes through private methods
# ---------------------------------------------------------------------------

class TestVideoRecordingWrapperDelegation(unittest.TestCase):
    """VideoRecordingWrapper explicitly exposes _get_simulator_state and
    _get_episode_model_file — necessary because gym.Wrapper.__getattr__ rejects
    names starting with '_'."""

    def setUp(self):
        _require_gym()
        import gym
        self._gym = gym
        self.inner, self.fake_state, self.fake_xml = _make_fake_robomimic_wrapper()
        self.vrw = _make_fake_video_recording_wrapper(self.inner)

    def test_gym_wrapper_getattr_blocks_underscore_names(self):
        """Confirm the gym limitation that motivates the explicit passthroughs."""
        with self.assertRaises(AttributeError):
            # gym.Wrapper.__getattr__ raises for private attributes
            self._gym.Wrapper.__getattr__(self.vrw, "_nonexistent_private")

    def test_get_simulator_state_is_callable_on_video_wrapper(self):
        self.assertTrue(callable(getattr(self.vrw, "_get_simulator_state", None)))

    def test_get_episode_model_file_is_callable_on_video_wrapper(self):
        self.assertTrue(callable(getattr(self.vrw, "_get_episode_model_file", None)))

    def test_get_simulator_state_delegates_to_inner_env(self):
        state = self.vrw._get_simulator_state()
        np.testing.assert_array_equal(state, self.fake_state)

    def test_get_episode_model_file_delegates_to_inner_env(self):
        xml = self.vrw._get_episode_model_file()
        self.assertEqual(xml, self.fake_xml)


# ---------------------------------------------------------------------------
# Test: MultiStepWrapper state buffering
# ---------------------------------------------------------------------------

class TestMultiStepWrapperStateBuf(unittest.TestCase):
    """MultiStepWrapper buffers (state, action) pairs in MimicGen layout."""

    def _make_wrapper(self, n_action_steps=4):
        inner, fake_state, fake_xml = _make_fake_robomimic_wrapper()
        vrw = _make_fake_video_recording_wrapper(inner)
        msw = _make_multistep_wrapper(vrw, n_obs_steps=2, n_action_steps=n_action_steps)
        return msw, fake_state, fake_xml

    def test_buffers_empty_before_reset(self):
        msw, _, _ = self._make_wrapper()
        # Before any reset, buffers should be empty lists
        self.assertEqual(msw._sim_states_buf, [])
        self.assertEqual(msw._sim_actions_buf, [])

    def test_reset_clears_buffers_and_captures_model_file(self):
        msw, _, fake_xml = self._make_wrapper()
        msw.reset()
        self.assertEqual(msw._sim_states_buf, [])
        self.assertEqual(msw._sim_actions_buf, [])
        self.assertEqual(msw._episode_model_file, fake_xml)

    def test_step_fills_state_buffer(self):
        n_action_steps = 4
        msw, fake_state, _ = self._make_wrapper(n_action_steps=n_action_steps)
        msw.reset()
        action = np.zeros((n_action_steps, 7), dtype=np.float32)
        msw.step(action)
        self.assertEqual(len(msw._sim_states_buf), n_action_steps)
        self.assertEqual(len(msw._sim_actions_buf), n_action_steps)
        # Each captured state should match what _get_simulator_state returns
        for s in msw._sim_states_buf:
            np.testing.assert_array_equal(s, fake_state.astype(np.float64))

    def test_step_buffers_accumulate_across_calls(self):
        n_action_steps = 3
        msw, _, _ = self._make_wrapper(n_action_steps=n_action_steps)
        msw.reset()
        action = np.zeros((n_action_steps, 7), dtype=np.float32)
        msw.step(action)
        msw.step(action)
        # Two step calls → 2 * n_action_steps buffered states
        self.assertEqual(len(msw._sim_states_buf), 2 * n_action_steps)

    def test_state_captured_before_action(self):
        """State at index t in the buffer must equal the pre-step state at that position.

        We simulate this by making _get_simulator_state return a monotonically
        increasing counter so we can check ordering.  The captured state must be
        the state *before* the corresponding action's super().step() is applied.
        """
        _require_diffusion_policy()
        _require_gym()
        import gym
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper

        call_counter = [0]

        class CountingEnv(gym.Env):
            """Obs=1-D, action space=1-D.  State returned = call count at that moment."""
            def __init__(self):
                super().__init__()
                import gym.spaces as spaces
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self):
                return np.zeros(1, dtype=np.float32)

            def step(self, action):
                call_counter[0] += 1
                return np.zeros(1, dtype=np.float32), 0.0, False, {}

            def _get_simulator_state(self):
                return np.array([float(call_counter[0])], dtype=np.float64)

            def _get_episode_model_file(self):
                return "<mujoco/>"

        # Wrap CountingEnv in VideoRecordingWrapper (needs passthrough).
        mock_recorder = MagicMock()
        mock_recorder.is_ready.return_value = False
        mock_recorder.stop.return_value = None
        vrw = VideoRecordingWrapper(
            env=CountingEnv(),
            video_recoder=mock_recorder,
            file_path=None,
        )
        n_action_steps = 4
        msw = MultiStepWrapper(
            env=vrw,
            n_obs_steps=1,
            n_action_steps=n_action_steps,
            max_episode_steps=100,
        )

        msw.reset()
        call_counter[0] = 0  # reset counter after reset calls
        action = np.zeros((n_action_steps, 1), dtype=np.float32)
        msw.step(action)

        # State captured before action[k] should equal call_counter value *before*
        # step k's super().step() increments it.  Call counter starts at 0 before
        # any step, so:
        #   captured[0] should be 0 (before first super().step → counter becomes 1)
        #   captured[1] should be 1 (before second super().step → counter becomes 2)
        #   ...
        states = msw._sim_states_buf
        self.assertEqual(len(states), n_action_steps)
        for k, state in enumerate(states):
            self.assertAlmostEqual(
                float(state[0]), float(k),
                msg=f"State at position {k} should be captured before step {k} "
                    f"(expected {k}, got {float(state[0])})",
            )

    def test_get_episode_sim_data_shape(self):
        n_action_steps = 3
        msw, _, fake_xml = self._make_wrapper(n_action_steps=n_action_steps)
        msw.reset()
        action = np.zeros((n_action_steps, 7), dtype=np.float32)
        msw.step(action)
        msw.step(action)
        data = msw._get_episode_sim_data()
        T = 2 * n_action_steps
        self.assertEqual(data["states"].shape[0], T)
        self.assertEqual(data["actions"].shape[0], T)
        self.assertEqual(data["model_file"], fake_xml)
        self.assertEqual(data["states"].dtype, np.float64)
        self.assertEqual(data["actions"].dtype, np.float64)

    def test_reset_clears_buf_between_episodes(self):
        n_action_steps = 2
        msw, _, _ = self._make_wrapper(n_action_steps=n_action_steps)
        msw.reset()
        action = np.zeros((n_action_steps, 7), dtype=np.float32)
        msw.step(action)
        # Second episode
        msw.reset()
        self.assertEqual(len(msw._sim_states_buf), 0)
        self.assertEqual(len(msw._sim_actions_buf), 0)


# ---------------------------------------------------------------------------
# Test: MimicgenLowdimRunner._write_rollouts_hdf5
# ---------------------------------------------------------------------------

class TestWriteRolloutsHdf5(unittest.TestCase):
    """_write_rollouts_hdf5 writes a MimicGen/from_rollout_hdf5-compatible HDF5."""

    def _make_runner_stub(self, output_dir: pathlib.Path):
        """Return a minimal object with just _write_rollouts_hdf5 and the attrs it reads."""
        _require_diffusion_policy()
        from diffusion_policy.env_runner.mimicgen_lowdim_runner import MimicgenLowdimRunner

        # Instantiate without actually building envs/runner by patching __init__.
        with patch.object(MimicgenLowdimRunner, "__init__", lambda self, *a, **k: None):
            runner = MimicgenLowdimRunner.__new__(MimicgenLowdimRunner)
        runner.env_meta = {"env_name": "Square_D1", "type": 1, "env_kwargs": {}}
        runner.episode_dir = output_dir / "episodes"
        runner.episode_dir.mkdir(parents=True, exist_ok=True)
        runner.output_dir = output_dir
        return runner

    def _dummy_episodes(self, n_eps=3, T=20, state_dim=45, action_dim=7):
        episodes = []
        for i in range(n_eps):
            episodes.append({
                "states": np.random.randn(T, state_dim).astype(np.float64),
                "actions": np.random.randn(T, action_dim).astype(np.float64),
                "model_file": f"<mujoco><worldbody/></mujoco>",
                "success": i % 2 == 0,
            })
        return episodes

    def test_creates_rollouts_hdf5_in_episodes_dir(self):
        with tempfile.TemporaryDirectory() as td:
            runner = self._make_runner_stub(pathlib.Path(td))
            eps = self._dummy_episodes(n_eps=2)
            out = runner._write_rollouts_hdf5(eps)
            self.assertTrue(out.exists())
            self.assertEqual(out.name, "rollouts.hdf5")
            self.assertEqual(out.parent, runner.episode_dir)

    def test_hdf5_structure_matches_from_rollout_hdf5_schema(self):
        """The written file must satisfy MimicGenSeedTrajectory.from_rollout_hdf5."""
        with tempfile.TemporaryDirectory() as td:
            runner = self._make_runner_stub(pathlib.Path(td))
            n_eps, T, state_dim, action_dim = 3, 15, 45, 7
            eps = self._dummy_episodes(n_eps=n_eps, T=T,
                                       state_dim=state_dim, action_dim=action_dim)
            out = runner._write_rollouts_hdf5(eps)

            with h5py.File(out, "r") as f:
                # top-level env_args
                env_args = json.loads(f["data"].attrs["env_args"])
                self.assertEqual(env_args["env_name"], "Square_D1")
                self.assertEqual(f["data"].attrs["total"], n_eps)

                for i, ep in enumerate(eps):
                    demo_key = f"demo_{i}"
                    self.assertIn(demo_key, f["data"])
                    grp = f[f"data/{demo_key}"]

                    # states and actions shape
                    self.assertEqual(grp["states"].shape, (T, state_dim))
                    self.assertEqual(grp["actions"].shape, (T, action_dim))

                    # model_file attribute
                    self.assertIn("model_file", grp.attrs)
                    self.assertIsInstance(grp.attrs["model_file"], str)

                    # success attribute
                    self.assertIn("success", grp.attrs)

    def test_env_args_is_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            runner = self._make_runner_stub(pathlib.Path(td))
            eps = self._dummy_episodes(n_eps=1)
            out = runner._write_rollouts_hdf5(eps)
            with h5py.File(out, "r") as f:
                env_args_str = f["data"].attrs["env_args"]
                parsed = json.loads(env_args_str)
                self.assertIsInstance(parsed, dict)

    def test_states_actions_aligned(self):
        """states[t] and actions[t] must have the same leading dimension."""
        with tempfile.TemporaryDirectory() as td:
            runner = self._make_runner_stub(pathlib.Path(td))
            T = 12
            eps = self._dummy_episodes(n_eps=1, T=T)
            out = runner._write_rollouts_hdf5(eps)
            with h5py.File(out, "r") as f:
                states = f["data/demo_0/states"][()]
                actions = f["data/demo_0/actions"][()]
                self.assertEqual(states.shape[0], actions.shape[0])
                self.assertEqual(states.shape[0], T)

    def test_from_rollout_hdf5_can_load_written_file(self):
        """Round-trip: MimicGenSeedTrajectory.from_rollout_hdf5 can read our output."""
        try:
            from policy_doctor.mimicgen.seed_trajectory import MimicGenSeedTrajectory
        except ImportError as e:
            self.skipTest(f"policy_doctor.mimicgen not importable: {e}")

        with tempfile.TemporaryDirectory() as td:
            runner = self._make_runner_stub(pathlib.Path(td))
            T, state_dim, action_dim = 10, 45, 7
            eps = self._dummy_episodes(n_eps=2, T=T,
                                       state_dim=state_dim, action_dim=action_dim)
            out = runner._write_rollouts_hdf5(eps)

            for i in range(len(eps)):
                traj = MimicGenSeedTrajectory.from_rollout_hdf5(out, demo_key=f"demo_{i}")
                self.assertEqual(traj.states.shape, (T, state_dim))
                self.assertEqual(traj.actions.shape, (T, action_dim))
                self.assertIsNotNone(traj.model_file)


if __name__ == "__main__":
    unittest.main()

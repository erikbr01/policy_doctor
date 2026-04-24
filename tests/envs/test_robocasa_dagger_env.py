"""Unit tests for RobocasaDAggerEnv."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import gym
import numpy as np
import pandas as pd
import pytest

from policy_doctor.envs.robocasa_dagger_env import RobocasaDAggerEnv


class MockRobomimicEnv:
    """Mock robomimic environment."""

    def __init__(self, obs_dim=23, action_dim=10):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._obs_idx = 0

    def get_observation(self):
        """Return per-key obs dict."""
        return {
            "object": np.random.randn(8).astype(np.float32),
            "robot0_eef_pos": np.random.randn(3).astype(np.float32),
            "robot0_eef_quat": np.random.randn(4).astype(np.float32),
            "robot0_gripper_qpos": np.random.randn(2).astype(np.float32),
        }

    def get_state(self):
        """Return MuJoCo state."""
        return {"states": np.random.randn(30).astype(np.float32)}


class MockRobomimicWrapper(gym.Env):
    """Mock RobomimicLowdimWrapper."""

    def __init__(self):
        self.env = MockRobomimicEnv()
        self.action_space = gym.spaces.Box(-1, 1, (10,))
        self.observation_space = gym.spaces.Box(-1, 1, (23,))

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate(
            [raw_obs["object"], raw_obs["robot0_eef_pos"], raw_obs["robot0_eef_quat"], raw_obs["robot0_gripper_qpos"]]
        )
        return obs

    def render(self, mode="rgb_array", height=256, width=256):
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class MockMultiStepWrapper(gym.Env):
    """Mock MultiStepWrapper with obs stacking."""

    def __init__(self, n_obs_steps=2):
        self.env = MockRobomimicWrapper()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.n_obs_steps = n_obs_steps
        self._obs_history = []
        self._done = False

    def reset(self):
        obs = self.env.get_observation()
        self._obs_history = [obs] * self.n_obs_steps
        self._done = False
        return np.stack(self._obs_history)

    def step(self, action):
        obs = self.env.get_observation()
        self._obs_history.append(obs)
        self._obs_history = self._obs_history[-self.n_obs_steps:]
        reward = float(np.random.randn())
        self._done = np.random.rand() < 0.05
        info = {}
        return np.stack(self._obs_history), reward, self._done, info

    def close(self):
        pass


def test_dagger_env_init():
    """Test DAggerEnv initialization."""
    inner_env = MockMultiStepWrapper()
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys)

    assert env.obs_keys == obs_keys
    assert env._acting_agent == "robot"
    assert len(env.episode_data) == 0


def test_dagger_env_set_acting_agent():
    """Test setting acting agent label."""
    inner_env = MockMultiStepWrapper()
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys)

    env.set_acting_agent("human")
    assert env._acting_agent == "human"

    env.set_acting_agent("robot")
    assert env._acting_agent == "robot"

    with pytest.raises(AssertionError):
        env.set_acting_agent("invalid")


def test_dagger_env_step_records_data():
    """Test that step() records per-key obs and sim state."""
    inner_env = MockMultiStepWrapper()
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys)

    env.reset()
    action = np.random.randn(10).astype(np.float32)
    obs, reward, done, info = env.step(action)

    # Check that data was recorded
    assert len(env.episode_data) == 1
    record = env.episode_data[0]

    assert "timestep" in record
    assert "obs" in record
    assert "stacked_obs" in record
    assert "action" in record
    assert "reward" in record
    assert "done" in record
    assert "acting_agent" in record
    assert "sim_state" in record

    assert record["acting_agent"] == "robot"
    assert record["action"].shape == (10,)
    assert isinstance(record["reward"], float)
    assert isinstance(record["done"], bool)


def test_dagger_env_reset_clears_data():
    """Test that reset() clears episode data."""
    inner_env = MockMultiStepWrapper()
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys)

    env.reset()
    action = np.zeros(10)
    env.step(action)
    env.step(action)

    assert len(env.episode_data) == 2

    env.reset()
    assert len(env.episode_data) == 0
    assert env._acting_agent == "robot"


def test_dagger_env_save_episode():
    """Test saving episode to pkl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inner_env = MockMultiStepWrapper()
        obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        output_dir = Path(tmpdir)
        env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys, output_dir=output_dir)

        env.reset()
        for _ in range(5):
            action = np.random.randn(10)
            obs, reward, done, info = env.step(action)
            if done:
                break

        path = env.save_episode()

        # Check file was created
        assert path.exists()
        assert path.name == "ep0000.pkl"

        # Reload and check
        df = pd.read_pickle(str(path))
        assert len(df) == 5
        assert "acting_agent" in df.columns
        assert all(df["acting_agent"] == "robot")


def test_dagger_env_acting_agent_labels():
    """Test that acting_agent labels are correctly recorded."""
    inner_env = MockMultiStepWrapper()
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys)

    env.reset()

    # Robot steps
    for _ in range(3):
        env.step(np.zeros(10))

    # Human steps
    env.set_acting_agent("human")
    for _ in range(2):
        env.step(np.zeros(10))

    data = env.episode_data
    assert len(data) == 5
    assert all(d["acting_agent"] == "robot" for d in data[:3])
    assert all(d["acting_agent"] == "human" for d in data[3:])


def test_dagger_env_render_camera():
    """Test camera rendering."""
    inner_env = MockMultiStepWrapper()
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    env = RobocasaDAggerEnv(inner_env, obs_keys=obs_keys)

    env.reset()

    img = env.render_camera()
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8

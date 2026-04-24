"""DAgger data collection wrapper around robocasa gym environment.

Records per-step observations (per-key + stacked), actions, rewards, done flags,
acting agent labels ("robot" or "human"), and MuJoCo simulation states for
deterministic replay and HDF5 conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gym
import numpy as np
import pandas as pd


class RobocasaDAggerEnv(gym.Env):
    """Wraps a MultiStepWrapper(RobomimicLowdimWrapper(...)) for DAgger data recording.

    Captures per-key obs, sim states, and acting_agent labels at each step for
    later conversion to robomimic HDF5 format.

    Parameters
    ----------
    inner_env : gym.Env
        A MultiStepWrapper-wrapped environment chain. Assumed to have:
        - inner_env.env: RobomimicLowdimWrapper
        - inner_env.env.env: robomimic.envs.env_robosuite.EnvRobosuite
    obs_keys : list[str]
        Observation keys to record (e.g., ["object", "robot0_eef_pos", ...])
    output_dir : Path or str, optional
        Where to save episode pkl files. If None, save_episode() requires explicit path.
    """

    def __init__(
        self,
        inner_env: gym.Env,
        obs_keys: list[str],
        output_dir: Optional[Path | str] = None,
    ) -> None:
        self.inner_env = inner_env
        self.obs_keys = obs_keys
        self.output_dir = Path(output_dir) if output_dir else None

        self.action_space = inner_env.action_space
        self.observation_space = inner_env.observation_space

        self._acting_agent = "robot"
        self._episode_data: list[dict] = []
        self._episode_idx: int = 0

    def _get_raw_obs_dict(self) -> dict[str, np.ndarray]:
        """Get per-key obs from underlying robomimic env."""
        # Navigate wrapper hierarchy: MultiStepWrapper -> RobomimicLowdimWrapper -> robomimic_env
        lowdim_wrapper = self.inner_env.env
        robomimic_env = lowdim_wrapper.env
        return robomimic_env.get_observation()

    def _get_sim_state(self) -> np.ndarray:
        """Get current MuJoCo state for deterministic replay."""
        lowdim_wrapper = self.inner_env.env
        robomimic_env = lowdim_wrapper.env
        return robomimic_env.get_state()["states"]

    def set_acting_agent(self, agent: str) -> None:
        """Set label for who is controlling the robot ('robot' or 'human')."""
        assert agent in ("robot", "human"), f"Invalid agent: {agent}"
        self._acting_agent = agent

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and record step data.

        Returns
        -------
        obs : np.ndarray
            Stacked observation (n_obs_steps, obs_dim) from MultiStepWrapper.
        reward : float
        done : bool
        info : dict
        """
        stacked_obs, reward, done, info = self.inner_env.step(action)

        # Capture per-key obs and sim state
        raw_obs = self._get_raw_obs_dict()
        sim_state = self._get_sim_state()

        self._episode_data.append(
            {
                "timestep": len(self._episode_data),
                "obs": {k: raw_obs[k].copy() for k in self.obs_keys},
                "stacked_obs": stacked_obs.copy(),
                "action": action.copy(),
                "reward": float(reward),
                "done": bool(done),
                "acting_agent": self._acting_agent,
                "sim_state": sim_state.copy(),
            }
        )

        return stacked_obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment and clear episode data."""
        self._episode_data = []
        self._acting_agent = "robot"
        return self.inner_env.reset()

    def render_camera(
        self,
        camera_name: str = "agentview",
        hw: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Get camera image for live visualization.

        Parameters
        ----------
        camera_name : str
            Camera name (e.g., "agentview", "robot0_eye_in_hand")
        hw : Tuple[int, int]
            (height, width) for the rendered image.

        Returns
        -------
        img : np.ndarray
            RGB image of shape (height, width, 3).
        """
        lowdim_wrapper = self.inner_env.env
        return lowdim_wrapper.render(mode="rgb_array", height=hw[0], width=hw[1])

    def save_episode(self, path: Optional[Path | str] = None) -> Path:
        """Save current episode data to pickle file in robomimic format.

        The resulting DataFrame has columns: timestep, obs (dict), stacked_obs,
        action, reward, done, acting_agent, sim_state.

        Parameters
        ----------
        path : Path or str, optional
            Explicit path to save to. If None, uses output_dir/ep{idx:04d}.pkl

        Returns
        -------
        path : Path
            Path where the file was saved.
        """
        if not self._episode_data:
            return None

        if path is None:
            assert (
                self.output_dir is not None
            ), "Must provide path or set output_dir in __init__"
            path = self.output_dir / f"ep{self._episode_idx:04d}.pkl"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._episode_data)
        df.to_pickle(str(path))
        self._episode_idx += 1

        return path

    @property
    def episode_data(self) -> list[dict]:
        """Return current episode data (list of step dicts)."""
        return list(self._episode_data)

    def close(self) -> None:
        """Close the underlying environment."""
        if hasattr(self.inner_env, "close"):
            self.inner_env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        if hasattr(self.inner_env, "seed"):
            return self.inner_env.seed(seed)

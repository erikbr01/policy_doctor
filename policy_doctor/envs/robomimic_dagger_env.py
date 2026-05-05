"""DAgger data collection wrapper around robomimic-style gym environments.

Records per-step observations (per-key + stacked), actions, rewards, done flags,
acting agent labels ("robot" or "human"), and MuJoCo simulation states for
deterministic replay and HDF5 conversion.

Works with any environment using RobomimicLowdimWrapper (robomimic, kitchen,
robocasa, libero, blockpush, mimicgen, etc.).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import gym
import numpy as np
import pandas as pd


class RobomimicDAggerEnv(gym.Env):
    """Wraps a MultiStepWrapper(RobomimicLowdimWrapper(...)) for DAgger data recording.

    Captures per-key obs, sim states, and acting_agent labels at each step for
    later conversion to robomimic HDF5 format. Works with any robomimic-compatible
    environment (robomimic, kitchen, robocasa, libero, blockpush, mimicgen, etc.).

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
        *,
        env_meta: Optional[dict[str, Any]] = None,
        save_format: str = "hdf5",
        record_data: bool = True,
    ) -> None:
        self.inner_env = inner_env
        self.obs_keys = obs_keys
        self.output_dir = Path(output_dir) if output_dir else None
        self.env_meta = env_meta or {}
        self.save_format = save_format
        self.record_data = record_data

        self.action_space = inner_env.action_space
        self.observation_space = inner_env.observation_space

        self._acting_agent = "robot"
        self._episode_data: list[dict] = []
        self._episode_idx: int = 0
        self._last_save_path: Optional[Path] = None

    def _get_raw_obs_dict(self) -> dict[str, np.ndarray]:
        """Get per-key obs from underlying robomimic env."""
        # Navigate wrapper hierarchy until we find a raw per-key observation dict.
        inner = self.inner_env
        seen = set()
        while inner is not None and id(inner) not in seen:
            seen.add(id(inner))
            if hasattr(inner, "get_observation"):
                obs = inner.get_observation()
                if isinstance(obs, dict):
                    return obs
            if hasattr(inner, "_get_observations"):
                obs = inner._get_observations()
                if isinstance(obs, dict):
                    return obs
            inner = getattr(inner, "env", None)
        raise AttributeError("Could not find raw per-key observation dict in env wrapper stack")

    def _get_sim_state(self) -> np.ndarray:
        """Get current MuJoCo state for deterministic replay."""
        inner = self.inner_env
        seen = set()
        while inner is not None and id(inner) not in seen:
            seen.add(id(inner))
            if hasattr(inner, "get_state"):
                return inner.get_state()["states"]
            inner = getattr(inner, "env", None)
        raise AttributeError("Could not find get_state() in env wrapper stack")

    def _get_model_file(self) -> Optional[str]:
        """Get per-episode MJCF XML for robosuite-backed HDF5 datasets."""
        if hasattr(self.inner_env, "_get_episode_model_file"):
            try:
                return self.inner_env._get_episode_model_file()
            except Exception:
                return None
        return None

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
        if self.record_data:
            pre_obs = self._get_raw_obs_dict()
            pre_state = self._get_sim_state()

        step_out = self.inner_env.step(action)
        obs = step_out[0]
        reward = step_out[1]
        done = step_out[2] if len(step_out) == 4 else (step_out[2] or step_out[3])
        info = step_out[-1]

        if isinstance(action, np.ndarray):
            action_np = action
        else:
            import torch
            action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)

        if not self.record_data:
            return obs, reward, done, info

        post_obs = self._get_raw_obs_dict()
        post_state = self._get_sim_state()

        self._episode_data.append(
            {
                "timestep": len(self._episode_data),
                "obs": {k: pre_obs[k].copy() for k in self.obs_keys},
                "next_obs": {k: post_obs[k].copy() for k in self.obs_keys},
                "stacked_obs": obs.copy() if hasattr(obs, "copy") else obs,
                "action": action_np.copy(),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(info.get("success", False)),
                "acting_agent": self._acting_agent,
                "sim_state": pre_state.copy(),
                "next_sim_state": post_state.copy(),
            }
        )

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment and clear episode data."""
        self._episode_data = []
        self._acting_agent = "robot"
        return self.inner_env.reset()

    @staticmethod
    def _render_supports_camera_name(obj: Any) -> bool:
        fn = getattr(obj, "render", None)
        if fn is None or not callable(fn):
            return False
        try:
            return "camera_name" in inspect.signature(fn).parameters
        except (ValueError, TypeError):
            return False

    def _robomimic_base_env_for_render(self) -> Any:
        """Walk ``inner_env.env -> ...`` to the robomimic sim env ``render(..., camera_name=...)``."""
        cur: Any = self.inner_env
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            child = getattr(cur, "env", None)
            if child is not None and self._render_supports_camera_name(child):
                return child
            cur = child
        raise AttributeError(
            "Could not find a nested env with render(..., camera_name=...) in inner_env chain"
        )

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
        h, w = hw
        base = self._robomimic_base_env_for_render()
        return base.render(mode="rgb_array", height=h, width=w, camera_name=camera_name)

    def save_episode(self, path: Optional[Path | str] = None) -> Path:
        """Save current episode data in pkl, HDF5, or both formats.

        HDF5 output follows robomimic's training layout:
        ``data/demo_N/{states, actions, rewards, dones, obs, next_obs}``.
        """
        if not self._episode_data:
            return None

        fmt = str(self.save_format or "hdf5").lower()
        if fmt not in {"pkl", "hdf5", "both"}:
            raise ValueError(f"Unknown save_format={self.save_format!r}; use pkl, hdf5, or both")

        if fmt in {"pkl", "both"}:
            pkl_path = self._default_path(path, suffix=".pkl")
            self._save_episode_pkl(pkl_path)
            self._last_save_path = pkl_path

        if fmt in {"hdf5", "both"}:
            hdf5_path = self._default_path(path, suffix=".hdf5")
            self._save_episode_hdf5(hdf5_path)
            self._last_save_path = hdf5_path

        self._episode_idx += 1
        return self._last_save_path

    def _default_path(self, path: Optional[Path | str], *, suffix: str) -> Path:
        if path is not None:
            path = Path(path)
            if suffix == ".hdf5" and path.suffix in {".pkl", ".pickle"}:
                return path.with_name("demo.hdf5")
            if suffix == ".pkl" and path.suffix in {".hdf5", ".h5"}:
                return path.with_suffix(".pkl")
            return path
        assert self.output_dir is not None, "Must provide path or set output_dir in __init__"
        if suffix == ".hdf5":
            return self.output_dir / "demo.hdf5"
        return self.output_dir / f"ep{self._episode_idx:04d}.pkl"

    def _save_episode_pkl(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._episode_data)
        df.to_pickle(str(path))
        return path

    def _save_episode_hdf5(self, path: Path) -> Path:
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        demo_key = f"demo_{self._episode_idx}"

        actions = np.asarray([r["action"] for r in self._episode_data], dtype=np.float32)
        states = np.asarray([r["sim_state"] for r in self._episode_data], dtype=np.float32)
        rewards = np.asarray([r["reward"] for r in self._episode_data], dtype=np.float32)
        dones = np.asarray([r["done"] for r in self._episode_data], dtype=np.uint8)
        successes = np.asarray([r["success"] for r in self._episode_data], dtype=np.uint8)
        acting_agents = np.asarray(
            [str(r["acting_agent"]).encode("utf-8") for r in self._episode_data]
        )

        with h5py.File(path, "a") as f:
            data = f.require_group("data")
            if self.env_meta and "env_args" not in data.attrs:
                data.attrs["env_args"] = json.dumps(dict(self.env_meta), indent=4)
            if demo_key in data:
                del data[demo_key]
            grp = data.create_group(demo_key)
            grp.create_dataset("actions", data=actions, compression="gzip")
            grp.create_dataset("states", data=states, compression="gzip")
            grp.create_dataset("rewards", data=rewards, compression="gzip")
            grp.create_dataset("dones", data=dones, compression="gzip")
            grp.create_dataset("success", data=successes, compression="gzip")
            grp.create_dataset("acting_agent", data=acting_agents, compression="gzip")
            obs_grp = grp.create_group("obs")
            next_obs_grp = grp.create_group("next_obs")
            for key in self.obs_keys:
                obs = np.asarray([r["obs"][key] for r in self._episode_data])
                next_obs = np.asarray([r["next_obs"][key] for r in self._episode_data])
                obs_grp.create_dataset(key, data=obs, compression="gzip")
                next_obs_grp.create_dataset(key, data=next_obs, compression="gzip")

            grp.attrs["num_samples"] = np.int64(len(actions))
            grp.attrs["success"] = bool(successes[-1]) if len(successes) else False
            model_file = self._get_model_file()
            if model_file is not None:
                grp.attrs["model_file"] = model_file

            total = 0
            for key in data:
                if key.startswith("demo_"):
                    total += int(data[key].attrs.get("num_samples", 0))
            data.attrs["total"] = np.int64(total)
        return path

    @property
    def last_save_path(self) -> Optional[Path]:
        """Path returned by the most recent save_episode call."""
        return self._last_save_path

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

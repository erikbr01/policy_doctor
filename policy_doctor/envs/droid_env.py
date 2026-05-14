"""DROID real-robot inference environment.

Wraps droid.robot_env.RobotEnv with:
  - observation preprocessing that matches the training format (resize, BGR→RGB, state assembly)
  - rate-limited step() (rate control is the runner's responsibility; we expose raw step)
  - episode recording via droid TrajectoryWriter
  - dry_run mode for smoke-testing without hardware

droid imports are lazy (inside methods) so this module can be imported in the
policy_doctor conda env even when the droid ZED/Franka drivers are absent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _ensure_droid_on_path():
    src = Path("~/src_droid").expanduser()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


class DROIDInferenceEnv:
    """Inference wrapper around droid.robot_env.RobotEnv.

    Parameters
    ----------
    action_space : str
        One of 'joint_velocity', 'cartesian_velocity'.
    wrist_serial : str
        ZED serial number for the wrist camera.
    ext1_serial : str
        ZED serial number for exterior camera 1 (left when standing behind robot).
    ext2_serial : str
        ZED serial number for exterior camera 2 (right when standing behind robot).
    dry_run : bool
        When True, skip hardware init entirely. get_obs() returns random tensors;
        step() is a no-op. Useful for smoke-testing the runner without a real robot.
    record_data : bool
        Whether to accumulate per-step data for save_episode().
    output_dir : Path, optional
        Where to save episode HDF5 files. Required if record_data=True and you
        intend to call save_episode() without an explicit path.
    """

    # Camera serial numbers matching our rig (from droid/misc/parameters.py).
    DEFAULT_WRIST_SERIAL = "14313307"
    DEFAULT_EXT1_SERIAL = "36716034"
    DEFAULT_EXT2_SERIAL = "37617599"

    def __init__(
        self,
        action_space: str = "joint_velocity",
        wrist_serial: str = DEFAULT_WRIST_SERIAL,
        ext1_serial: str = DEFAULT_EXT1_SERIAL,
        ext2_serial: str = DEFAULT_EXT2_SERIAL,
        dry_run: bool = False,
        record_data: bool = True,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        assert action_space in ("joint_velocity", "cartesian_velocity"), action_space

        self._action_space_name = action_space
        self.wrist_serial = wrist_serial
        self.ext1_serial = ext1_serial
        self.ext2_serial = ext2_serial
        self.dry_run = dry_run
        self.record_data = record_data
        self.output_dir = Path(output_dir) if output_dir else None

        # 7-DOF Franka: cartesian = 6+1=7, joint = 7+1=8.  Matches RobotEnv.DoF.
        self._action_dim = 7 if "cartesian" in action_space else 8

        self._robot_env = None
        self._episode_data: list[dict] = []
        self._episode_idx: int = 0
        self._last_obs: Optional[dict] = None

        if not dry_run:
            self._init_robot()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_robot(self) -> None:
        _ensure_droid_on_path()
        from droid.robot_env import RobotEnv

        camera_kwargs = {
            self.wrist_serial: {"image": True},
            self.ext1_serial: {"image": True},
            self.ext2_serial: {"image": True},
        }
        self._robot_env = RobotEnv(
            action_space=self._action_space_name,
            camera_kwargs=camera_kwargs,
            do_reset=True,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self._episode_data = []
        if self.dry_run:
            obs = self._dummy_obs()
        else:
            self._robot_env.reset()
            obs = self.get_obs()
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray):
        if self.dry_run:
            obs = self._dummy_obs()
        else:
            self._robot_env.update_robot(
                action,
                action_space=self._action_space_name,
                gripper_action_space="position",
                blocking=False,
            )
            obs = self.get_obs()

        if self.record_data:
            self._episode_data.append(
                {
                    "timestep": len(self._episode_data),
                    "obs": {k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()},
                    "action": np.asarray(action, dtype=np.float32).copy(),
                }
            )
        self._last_obs = obs
        return obs, 0.0, False, {}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_obs(self) -> dict:
        """Return preprocessed observation dict ready for policy consumption."""
        if self.dry_run:
            return self._dummy_obs()

        raw = self._robot_env.get_observation()
        return self._extract_observation(raw)

    def _extract_observation(self, raw: dict) -> dict:
        image_obs = raw.get("image", {})
        robot_state = raw.get("robot_state", {})

        obs: dict[str, np.ndarray] = {
            "joint_position": np.array(robot_state.get("joint_positions", np.zeros(7)), dtype=np.float32),
            "gripper_position": np.atleast_1d(
                np.array(robot_state.get("gripper_position", 0.0), dtype=np.float32)
            ),
            "cartesian_position": np.array(
                robot_state.get("cartesian_position", np.zeros(6)), dtype=np.float32
            ),
        }

        for serial, key in (
            (self.wrist_serial, "wrist_image"),
            (self.ext1_serial, "exterior_image_1_left"),
            (self.ext2_serial, "exterior_image_2_left"),
        ):
            img = image_obs.get(f"{serial}_left")
            if img is not None:
                img = self._preprocess_image(img)
                obs[key] = img

        return obs

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # Only convert colour space — do NOT resize here.  Policy backends
        # (WebSocketPolicy) are responsible for resizing to their model's
        # expected resolution (e.g. resize_with_pad to 224×224 for openpi).
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1].copy()  # BGR → RGB
        return img

    def _dummy_obs(self) -> dict:
        rng = np.random.default_rng()
        return {
            "joint_position": rng.random(7).astype(np.float32),
            "gripper_position": rng.random(1).astype(np.float32),
            "cartesian_position": rng.random(6).astype(np.float32),
            "wrist_image": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "exterior_image_1_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "exterior_image_2_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        }

    # ------------------------------------------------------------------
    # Episode saving
    # ------------------------------------------------------------------

    def save_episode(self, path: Optional[Path | str] = None, success: bool = False) -> Optional[Path]:
        """Save episode as DROID-format HDF5 via TrajectoryWriter.

        Produces a trajectory.h5 compatible with convert_droid_to_robomimic.py
        for later re-ingestion into the attribution pipeline.
        """
        if not self._episode_data:
            return None

        if path is None:
            assert self.output_dir is not None, "Provide path or set output_dir"
            path = self.output_dir / f"ep{self._episode_idx:04d}" / "trajectory.h5"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        _ensure_droid_on_path()
        from droid.trajectory_utils.trajectory_writer import TrajectoryWriter

        writer = TrajectoryWriter(str(path), metadata={"success": success}, save_images=False)
        for step in self._episode_data:
            obs = step["obs"]
            action = step["action"]
            # Reconstruct timestep dict in DROID format so convert script can reload it.
            joint_vel = action[:7]
            gripper_pos = action[7:8]
            timestep = {
                "observation": {
                    "robot_state": {
                        "joint_positions": obs.get("joint_position", np.zeros(7)),
                        "cartesian_position": obs.get("cartesian_position", np.zeros(6)),
                        "gripper_position": float(obs.get("gripper_position", np.zeros(1))[0]),
                    },
                    "controller_info": {"movement_enabled": True, "success": False, "failure": False},
                },
                "action": {
                    "joint_velocity": joint_vel,
                    "gripper_position": float(gripper_pos[0]),
                    "cartesian_velocity": np.zeros(6),
                },
            }
            writer.write_timestep(timestep)
        writer.close(metadata={"success": success})
        self._episode_idx += 1
        return path

    # ------------------------------------------------------------------

    def close(self) -> None:
        pass  # ZED cameras / RobotEnv have no explicit close in current droid version

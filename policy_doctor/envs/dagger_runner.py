"""DAgger episode runner: orchestrates policy rollout with monitored intervention triggering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from policy_doctor.envs.intervention_device import InterventionDevice
from policy_doctor.envs.robomimic_dagger_env import RobomimicDAggerEnv
from policy_doctor.envs.visualization import DAggerVisualizer
from policy_doctor.monitoring.monitored_policy import MonitoredPolicy


@dataclass
class EpisodeRecord:
    """Record of one completed DAgger episode."""

    episode_idx: int
    success: bool
    n_steps: int
    n_robot_steps: int
    n_human_steps: int
    n_auto_interventions: int
    manual_overrides: int


class RobomimicDAggerRunner:
    """Main loop for DAgger rollouts with threaded policy inference.

    Policy inference runs in a background thread so the sim step loop and
    visualization run continuously.  The design is a rolling action buffer:

      1. At the start of each new chunk, submit inference for the *next* chunk
         immediately — policy thinks while the sim executes the current chunk.
      2. When the chunk is exhausted, wait for the pending future (updating the
         visualizer on every poll iteration so the display stays live).
      3. Human-mode steps execute at full speed; inference is re-submitted
         when control returns to the robot.

    Parameters
    ----------
    monitored_policy : MonitoredPolicy
    env : RobomimicDAggerEnv
    intervention_device : InterventionDevice
    n_obs_steps : int
    n_action_steps : int
    max_steps : int
    output_dir : Path or str, optional
    visualizer : DAggerVisualizer, optional
    action_transform : callable, optional
        Per-action transform applied before stepping the env (e.g. rotation_6d
        → axis_angle conversion).
    """

    def __init__(
        self,
        monitored_policy: MonitoredPolicy,
        env: RobomimicDAggerEnv,
        intervention_device: InterventionDevice,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_steps: int = 500,
        output_dir: Optional[Path | str] = None,
        visualizer: Optional[DAggerVisualizer] = None,
        action_transform=None,
    ) -> None:
        self.monitored_policy = monitored_policy
        self.env = env
        self.intervention_device = intervention_device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.output_dir = Path(output_dir) if output_dir else None
        self.visualizer = visualizer
        self._action_transform = action_transform

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs_dict(self, obs: np.ndarray, obs_queue: deque) -> dict:
        """Build (1, n_obs_steps, obs_dim) obs dict for the policy."""
        if obs.ndim >= 2:
            return {"obs": obs[None]}
        return {"obs": np.stack(list(obs_queue), axis=0)[None]}

    def _unpack_action_chunk(self, result: dict) -> np.ndarray:
        """Extract numpy action chunk from predict_action result."""
        chunk = result["action"][0]
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().cpu().numpy()
        return chunk  # (n_action_steps, raw_action_dim)

    def _step_action(self, act: np.ndarray):
        """Apply optional transform and step the env with a single action."""
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        if self._action_transform is not None:
            act = self._action_transform(act)
        step_out = self.env.step(act)
        obs = step_out[0]
        reward = step_out[1]
        done = step_out[2] if len(step_out) == 4 else (step_out[2] or step_out[3])
        return obs, reward, done

    def _update_viz(self, step: int) -> None:
        """Render camera + push frame to the cv2 window."""
        if self.visualizer is None:
            return
        try:
            camera_imgs = {
                name: self.env.render_camera(camera_name=name)
                for name in self.visualizer.camera_names
            }
            node_name, node_value, intervention_reason = "unknown", None, ""
            if self.monitored_policy.episode_results:
                latest = self.monitored_policy.episode_results[-1]
                node_name = latest.get("node_name") or "unknown"
                node_value = latest.get("distance")
                iv = latest.get("intervention")
                if iv and iv.triggered:
                    intervention_reason = iv.reason
            self.visualizer.update(
                camera_imgs=camera_imgs,
                node_name=node_name,
                node_value=node_value,
                acting_agent=self.env._acting_agent,
                step=step,
                intervention_reason=intervention_reason,
            )
        except Exception as e:
            print(f"Visualization update failed: {e}")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run_episode(self, episode_idx: int = 0) -> EpisodeRecord:
        # Two inference modes:
        #   async (PolicyClient): submit HTTP request, execute chunk, get() at boundary
        #   sync  (MonitoredPolicy / _BareMonitor): blocking predict_action on main thread
        from policy_doctor.envs.policy_server import PolicyClient
        _async = isinstance(self.monitored_policy, PolicyClient)

        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.monitored_policy.reset()
        self.intervention_device.reset()

        obs_queue = deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)
        acting_agent = "robot"
        self.env.set_acting_agent(acting_agent)

        n_robot_steps = n_human_steps = n_auto_interventions = manual_overrides = 0
        step = 0
        done = False
        info = {}

        chunk: Optional[np.ndarray] = None
        chunk_idx = 0

        if _async:
            # Kick off first request immediately so it runs while we set up
            self.monitored_policy.submit(self._make_obs_dict(obs, obs_queue))

        try:
            while not done and step < self.max_steps:

                # ── ROBOT MODE ────────────────────────────────────────────
                if acting_agent == "robot":

                    # Fetch a new action chunk when the current one is exhausted.
                    if chunk is None or chunk_idx >= len(chunk):
                        if _async:
                            # Block for the HTTP result (usually already done)
                            chunk = self.monitored_policy.get()
                        else:
                            obs_dict = self._make_obs_dict(obs, obs_queue)
                            result = self.monitored_policy.predict_action(obs_dict)
                            chunk = self._unpack_action_chunk(result)
                        chunk_idx = 0
                        if _async:
                            # Pre-submit next request immediately so it overlaps
                            # with executing the current chunk
                            self.monitored_policy.submit(self._make_obs_dict(obs, obs_queue))

                    # Execute one action from the chunk, update viz each step.
                    act = chunk[chunk_idx]
                    chunk_idx += 1
                    obs, reward, done = self._step_action(act)
                    obs_queue.append(obs)
                    n_robot_steps += 1
                    step += 1

                    # Check auto-intervention
                    if self.monitored_policy.episode_results:
                        iv = self.monitored_policy.episode_results[-1].get("intervention")
                        if iv and iv.triggered:
                            acting_agent = "human"
                            self.env.set_acting_agent(acting_agent)
                            n_auto_interventions += 1

                    # Check manual override
                    if self.intervention_device.is_intervening:
                        acting_agent = "human"
                        self.env.set_acting_agent(acting_agent)

                    self._update_viz(step)

                # ── HUMAN MODE ────────────────────────────────────────────
                else:
                    human_action = self.intervention_device.get_action()
                    if human_action is not None:
                        obs, reward, done = self._step_action(human_action)
                        obs_queue.append(obs)
                        n_human_steps += 1
                        step += 1
                        self._update_viz(step)

                    if not self.intervention_device.is_intervening:
                        acting_agent = "robot"
                        self.env.set_acting_agent(acting_agent)
                        manual_overrides += 1
                        chunk, chunk_idx = None, 0  # fetch fresh chunk on re-entry

        finally:
            if _async and hasattr(self.monitored_policy, "stop"):
                self.monitored_policy.stop()
            if self.visualizer:
                self.visualizer.close()

        if self.output_dir:
            self.env.save_episode()

        success = bool(info.get("success", False)) if done else False
        record = EpisodeRecord(
            episode_idx=episode_idx,
            success=success,
            n_steps=step,
            n_robot_steps=n_robot_steps,
            n_human_steps=n_human_steps,
            n_auto_interventions=n_auto_interventions,
            manual_overrides=manual_overrides,
        )
        return record

    def run(self, n_episodes: int) -> list[EpisodeRecord]:
        """Run multiple episodes."""
        import traceback as _tb
        records = []
        for ep_idx in range(n_episodes):
            print(f"\n=== Episode {ep_idx + 1}/{n_episodes} ===", flush=True)
            try:
                record = self.run_episode(ep_idx)
            except Exception:
                print("run_episode raised an exception:", flush=True)
                _tb.print_exc()
                break
            records.append(record)
            print(
                f"  Steps: {record.n_steps} "
                f"(robot={record.n_robot_steps}, human={record.n_human_steps}) "
                f"| Auto-triggers: {record.n_auto_interventions} "
                f"| Manual overrides: {record.manual_overrides} "
                f"| Success: {record.success}",
                flush=True,
            )
        return records

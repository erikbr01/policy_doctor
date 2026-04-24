"""DAgger episode runner: orchestrates policy rollout with monitored intervention triggering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

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
    """Main loop for DAgger rollouts with automatic intervention triggering.

    Combines MonitoredPolicy (which classifies timesteps and detects when to intervene)
    with RobocasaDAggerEnv (which records data) and an InterventionDevice (which handles
    human input during interventions).

    Parameters
    ----------
    monitored_policy : MonitoredPolicy
        Wrapped policy with behavior graph monitoring.
    env : RobomimicDAggerEnv
        Data-recording environment wrapper.
    intervention_device : InterventionDevice
        Human input handler (keyboard, SpaceMouse, etc.)
    n_obs_steps : int, default 2
        Observation history length (from policy config).
    n_action_steps : int, default 8
        Action prediction horizon (from policy config).
    max_steps : int, default 500
        Maximum steps per episode before auto-termination.
    output_dir : Path or str, optional
        Directory to save episode pkl files.
    visualizer : DAggerVisualizer, optional
        Live display of camera feed + node assignment + intervention status.
    """

    def __init__(
        self,
        monitored_policy: MonitoredPolicy,
        env: RobocasaDAggerEnv,
        intervention_device: InterventionDevice,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_steps: int = 500,
        output_dir: Optional[Path | str] = None,
        visualizer: Optional[DAggerVisualizer] = None,
    ) -> None:
        self.monitored_policy = monitored_policy
        self.env = env
        self.intervention_device = intervention_device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.output_dir = Path(output_dir) if output_dir else None
        self.visualizer = visualizer

    def run_episode(self, episode_idx: int = 0) -> EpisodeRecord:
        """Execute one DAgger episode with automatic intervention triggering.

        Main control loop:
        1. Roll out policy in chunks
        2. After each chunk, check if MonitoredPolicy triggered intervention
           (based on behavior graph V-value crossing threshold)
        3. If triggered or manual override active: hand off to human
        4. Human provides corrective actions; when done, return to robot
        5. Save episode with per-step acting_agent labels

        Parameters
        ----------
        episode_idx : int
            Episode index for logging/saving.

        Returns
        -------
        record : EpisodeRecord
            Summary of the episode.
        """
        # Initialize
        obs = self.env.reset()
        self.monitored_policy.reset()
        self.intervention_device.reset()

        # Prime observation buffer (deque of n_obs_steps)
        from collections import deque

        obs_queue = deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)

        acting_agent = "robot"
        self.env.set_acting_agent(acting_agent)

        n_robot_steps = 0
        n_human_steps = 0
        n_auto_interventions = 0
        manual_overrides = 0
        step = 0
        done = False

        while not done and step < self.max_steps:
            # --- ROBOT MODE ---
            if acting_agent == "robot":
                # Build obs dict for policy
                stacked_obs = np.stack(list(obs_queue), axis=0)  # (n_obs_steps, obs_dim)
                obs_dict = {"obs": stacked_obs[None]}  # (1, n_obs_steps, obs_dim)

                # Get policy action and classify
                action_chunk = self.monitored_policy.predict_action(obs_dict)["action"][
                    0
                ]  # (n_action_steps, action_dim)

                # Check if intervention was triggered by behavior graph
                intervention_decision = self.monitored_policy.episode_results[-1][
                    "intervention"
                ]
                if intervention_decision is not None and intervention_decision.triggered:
                    # Auto-trigger: switch to human mode
                    acting_agent = "human"
                    self.env.set_acting_agent(acting_agent)
                    self.intervention_device.notify(intervention_decision.reason)
                    n_auto_interventions += 1
                    continue

                # Also check for manual override
                if self.intervention_device.is_intervening:
                    acting_agent = "human"
                    self.env.set_acting_agent(acting_agent)
                    manual_overrides += 1
                    continue

                # Execute action chunk
                for t in range(min(self.n_action_steps, len(action_chunk))):
                    obs, reward, done, info = self.env.step(action_chunk[t])
                    obs_queue.append(obs)
                    n_robot_steps += 1
                    step += 1

                    if done:
                        break

                    # Mid-chunk: check for manual override
                    if self.intervention_device.is_intervening:
                        acting_agent = "human"
                        self.env.set_acting_agent(acting_agent)
                        break

                    # Update visualization
                    if self.visualizer is not None:
                        self._update_viz(step, intervention_decision)

            # --- HUMAN MODE ---
            else:
                # Get human action from keyboard/device
                human_action = self.intervention_device.get_action()
                if human_action is not None:
                    obs, reward, done, info = self.env.step(human_action)
                    obs_queue.append(obs)
                    n_human_steps += 1
                    step += 1

                    if self.visualizer is not None:
                        self._update_viz(step, None)

                # Check if human returned control to robot
                if not self.intervention_device.is_intervening:
                    acting_agent = "robot"
                    self.env.set_acting_agent(acting_agent)
                    self.monitored_policy.reset()  # clear history for new phase

        # Save episode
        if self.output_dir:
            self.env.save_episode()

        # Return summary
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
        """Run multiple episodes and return summaries."""
        records = []
        for ep_idx in range(n_episodes):
            print(f"\n=== Episode {ep_idx + 1}/{n_episodes} ===")
            record = self.run_episode(ep_idx)
            records.append(record)
            print(
                f"  Steps: {record.n_steps} "
                f"(robot={record.n_robot_steps}, human={record.n_human_steps}) "
                f"| Auto-triggers: {record.n_auto_interventions} "
                f"| Manual overrides: {record.manual_overrides} "
                f"| Success: {record.success}"
            )

        return records

    def _update_viz(self, step: int, intervention_decision) -> None:
        """Update live visualization."""
        if self.visualizer is None:
            return

        try:
            camera_img = self.env.render_camera()
            node_name = "unknown"
            node_value = None

            if self.monitored_policy.episode_results:
                latest_result = self.monitored_policy.episode_results[-1]
                if latest_result.get("result", {}).get("assignment"):
                    assignment = latest_result["result"]["assignment"]
                    node_name = assignment.node_name
                    node_value = assignment.distance  # or V-value if stored

            acting_agent = self.env._acting_agent
            intervention_reason = ""
            if intervention_decision and intervention_decision.triggered:
                intervention_reason = intervention_decision.reason

            self.visualizer.update(
                camera_imgs={"agentview": camera_img},
                node_name=node_name,
                node_value=node_value,
                acting_agent=acting_agent,
                step=step,
                intervention_reason=intervention_reason,
            )
        except Exception as e:
            print(f"Visualization update failed: {e}")

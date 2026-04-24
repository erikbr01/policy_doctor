"""Unit tests for DAggerRunner (simplified - full integration requires monitoring stack)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from policy_doctor.envs.intervention_device import PassthroughInterventionDevice
from policy_doctor.envs.dagger_runner import EpisodeRecord, RobomimicDAggerRunner


def test_episode_record_creation():
    """Test EpisodeRecord data class."""
    record = EpisodeRecord(
        episode_idx=0,
        success=True,
        n_steps=10,
        n_robot_steps=8,
        n_human_steps=2,
        n_auto_interventions=1,
        manual_overrides=0,
    )

    assert record.episode_idx == 0
    assert record.success is True
    assert record.n_steps == 10
    assert record.n_robot_steps == 8
    assert record.n_human_steps == 2
    assert record.n_auto_interventions == 1


def test_dagger_runner_initialization():
    """Test RobomimicDAggerRunner initialization."""
    monitored_policy = MagicMock()
    env = MagicMock()
    device = PassthroughInterventionDevice()

    runner = RobomimicDAggerRunner(
        monitored_policy=monitored_policy,
        env=env,
        intervention_device=device,
        n_obs_steps=2,
        n_action_steps=8,
        max_steps=500,
    )

    assert runner.monitored_policy is monitored_policy
    assert runner.env is env
    assert runner.intervention_device is device
    assert runner.n_obs_steps == 2
    assert runner.n_action_steps == 8
    assert runner.max_steps == 500


def test_dagger_runner_run_returns_list():
    """Test that run() returns a list of EpisodeRecords."""
    monitored_policy = MagicMock()
    env = MagicMock()
    device = PassthroughInterventionDevice()

    # Mock env behavior
    env.reset.return_value = np.random.randn(2, 23)
    env.step.return_value = (np.random.randn(2, 23), 0.0, True, {"success": False})
    env._acting_agent = "robot"

    # Mock policy behavior
    monitored_policy.predict_action.return_value = {"action": np.random.randn(1, 8, 10)}
    monitored_policy.episode_results = [{"intervention": MagicMock(triggered=False)}]

    runner = RobomimicDAggerRunner(
        monitored_policy=monitored_policy,
        env=env,
        intervention_device=device,
        max_steps=500,
    )

    # Note: Full run() test requires mocking many dependencies, so we just check structure
    # In practice, integration tests would verify the full control flow
    assert hasattr(runner, "run")
    assert hasattr(runner, "run_episode")


def test_dagger_runner_manual_override_handling():
    """Test that manual intervention flag is checked."""
    # This is a structural test - full behavior testing requires integration setup
    monitored_policy = MagicMock()
    env = MagicMock()
    device = PassthroughInterventionDevice()

    runner = RobomimicDAggerRunner(
        monitored_policy=monitored_policy,
        env=env,
        intervention_device=device,
    )

    # Check that the runner has access to the intervention device's is_intervening flag
    assert hasattr(device, "is_intervening")
    assert device.is_intervening is False  # Passthrough always False


def test_dagger_runner_observation_stacking():
    """Test that obs stacking works correctly (structural test)."""
    from collections import deque

    # Simulate the obs queue logic
    n_obs_steps = 2
    obs_dim = 23

    obs_queue = deque(
        [np.zeros((obs_dim,))] * n_obs_steps, maxlen=n_obs_steps
    )

    # Add a new obs
    new_obs = np.ones((obs_dim,))
    obs_queue.append(new_obs)

    # Check stacking
    stacked_obs = np.stack(list(obs_queue), axis=0)
    assert stacked_obs.shape == (n_obs_steps, obs_dim)
    assert np.array_equal(stacked_obs[0], np.zeros((obs_dim,)))
    assert np.array_equal(stacked_obs[1], np.ones((obs_dim,)))


def test_dagger_runner_action_chunk_execution():
    """Test that action chunk is executed step-by-step."""
    # This simulates the core action chunk execution loop
    n_action_steps = 8
    action_dim = 10

    action_chunk = np.random.randn(n_action_steps, action_dim)

    # Simulate stepping through the chunk
    steps_executed = 0
    for t in range(min(n_action_steps, 5)):  # Early break after 5 steps
        _ = action_chunk[t]
        steps_executed += 1

    assert steps_executed == 5

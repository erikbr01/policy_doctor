"""Unit tests for intervention devices."""

import numpy as np
import pytest

from policy_doctor.envs.intervention_device import (
    KeyboardInterventionDevice,
    PassthroughInterventionDevice,
)


def test_passthrough_device_never_intervenes():
    """Test PassthroughInterventionDevice never triggers intervention."""
    device = PassthroughInterventionDevice()

    assert device.is_intervening is False
    assert device.get_action() is None

    # Call notify (should not raise)
    device.notify("test message")

    # Reset (should not raise)
    device.reset()


def test_keyboard_device_init():
    """Test KeyboardInterventionDevice initialization."""
    device = KeyboardInterventionDevice(action_dim=10, step_size=0.05)

    assert device.action_dim == 10
    assert device.is_intervening is False
    assert device.get_action() is None

    device.close()


def test_keyboard_device_action_keys():
    """Test that action keys exist and have correct dimensions."""
    device = KeyboardInterventionDevice(action_dim=10)

    # Check that all action keys are in bindings
    assert "w" in device.KEY_BINDINGS
    assert "s" in device.KEY_BINDINGS
    assert "a" in device.KEY_BINDINGS
    assert "d" in device.KEY_BINDINGS

    # Check that all bindings have correct shape
    for key, action in device.KEY_BINDINGS.items():
        assert action.shape == (10,), f"Key {key} has wrong shape"
        assert -1 <= action.min() and action.max() <= 1, f"Key {key} out of bounds"

    device.close()


def test_keyboard_device_no_action_when_idle():
    """Test that get_action returns None when no keys are pressed."""
    device = KeyboardInterventionDevice(action_dim=10)

    assert device.get_action() is None

    device.close()


def test_keyboard_device_intervention_toggle():
    """Test that is_intervening can be toggled (simulated)."""
    device = KeyboardInterventionDevice(action_dim=10)

    assert device.is_intervening is False

    # Simulate space key press by toggling directly
    device._is_intervening = True
    assert device.is_intervening is True

    device._is_intervening = False
    assert device.is_intervening is False

    device.close()


def test_keyboard_device_action_composition():
    """Test that actions are correctly composed from key bindings."""
    device = KeyboardInterventionDevice(action_dim=10)

    # Simulate holding 'w' (arm +z)
    device._keys_pressed.add("w")
    action = device.get_action()

    assert action is not None
    assert action.shape == (10,)
    expected = device.KEY_BINDINGS["w"]
    np.testing.assert_array_almost_equal(action, expected)

    # Clear and test 'a' (arm -x)
    device._keys_pressed.clear()
    device._keys_pressed.add("a")
    action = device.get_action()

    expected = device.KEY_BINDINGS["a"]
    np.testing.assert_array_almost_equal(action, expected)

    device.close()


def test_keyboard_device_multi_key_composition():
    """Test that multiple keys are summed correctly."""
    device = KeyboardInterventionDevice(action_dim=10)

    # Simulate holding 'w' and 'a'
    device._keys_pressed.add("w")
    device._keys_pressed.add("a")
    action = device.get_action()

    assert action is not None
    expected = device.KEY_BINDINGS["w"] + device.KEY_BINDINGS["a"]
    np.testing.assert_array_almost_equal(action, expected)

    device.close()


def test_keyboard_device_action_clipping():
    """Test that actions are clipped to [-1, 1]."""
    device = KeyboardInterventionDevice(action_dim=10)

    # Add too many keys to cause overflow
    device._keys_pressed.add("w")
    device._keys_pressed.add("s")
    device._keys_pressed.add("q")
    device._keys_pressed.add("e")
    action = device.get_action()

    assert action is not None
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    device.close()


def test_keyboard_device_reset():
    """Test that reset() clears state."""
    device = KeyboardInterventionDevice(action_dim=10)

    device._keys_pressed.add("w")
    device._is_intervening = True

    device.reset()

    assert len(device._keys_pressed) == 0
    assert device.is_intervening is False

    device.close()


def test_keyboard_device_notify():
    """Test that notify does not raise."""
    device = KeyboardInterventionDevice(action_dim=10)

    device.notify("test message")
    device.notify("another test")

    device.close()

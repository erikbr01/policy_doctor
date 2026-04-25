"""Unit tests for intervention devices."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from policy_doctor.envs.intervention_device import (
    KeyboardInterventionDevice,
    PassthroughInterventionDevice,
    XboxControllerInterventionDevice,
)


# ---------------------------------------------------------------------------
# Fixture: mock the `inputs` library so tests run without a physical gamepad
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_inputs(monkeypatch):
    """Stub out the `inputs` library with a silent no-op gamepad."""
    mock_gamepad = MagicMock()
    mock_gamepad.name = "Xbox Controller (mock)"
    mock_gamepad.read.return_value = []  # no events

    mock_devices = MagicMock()
    mock_devices.gamepads = [mock_gamepad]

    mock_mod = MagicMock()
    mock_mod.devices = mock_devices
    mock_mod.get_gamepad.return_value = []

    monkeypatch.setitem(sys.modules, "inputs", mock_mod)
    return mock_mod


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


# ---------------------------------------------------------------------------
# XboxControllerInterventionDevice tests
# ---------------------------------------------------------------------------

def test_xbox_device_init(mock_inputs):
    device = XboxControllerInterventionDevice()
    assert device.is_intervening is False
    assert device.get_action() is None
    device.close()


def test_xbox_device_no_action_when_idle(mock_inputs):
    device = XboxControllerInterventionDevice()
    assert device.get_action() is None
    device.close()


def test_xbox_device_action_shape(mock_inputs):
    """Action vector is 10D when inputs are active."""
    device = XboxControllerInterventionDevice()
    device._left_x = 1.0
    action = device.get_action()
    assert action is not None
    assert action.shape == (10,)
    device.close()


def test_xbox_device_left_stick_maps_to_arm_xy(mock_inputs):
    """Left stick X/Y maps to arm x/y (index 0 and 1)."""
    device = XboxControllerInterventionDevice(deadzone=0.0, scale_position=1.0)

    device._left_x = 1.0
    device._left_y = 0.0
    action = device.get_action()
    assert action is not None
    assert action[0] == pytest.approx(1.0)
    assert action[1] == pytest.approx(0.0)

    # Push forward on left stick (raw Y negative after negation → positive)
    device._left_x = 0.0
    device._left_y = -1.0   # raw: negative = pushed forward
    action = device.get_action()
    assert action[1] == pytest.approx(1.0)

    device.close()


def test_xbox_device_right_stick_maps_to_arm_z_yaw(mock_inputs):
    """Right stick X maps to yaw [5], right stick Y maps to arm z [2]."""
    device = XboxControllerInterventionDevice(deadzone=0.0, scale_rotation=1.0)

    device._right_x = 0.5
    device._right_y = -0.5  # raw negative = pushed up → positive z
    action = device.get_action()
    assert action[5] == pytest.approx(0.5)
    assert action[2] == pytest.approx(0.5)

    device.close()


def test_xbox_device_triggers_map_to_pitch(mock_inputs):
    """RT positive pitch, LT negative pitch (RT-LT at index 4)."""
    device = XboxControllerInterventionDevice(deadzone=0.0, scale_rotation=1.0)

    device._rt = 1.0
    device._lt = 0.0
    action = device.get_action()
    assert action[4] == pytest.approx(1.0)

    device._rt = 0.0
    device._lt = 1.0
    action = device.get_action()
    assert action[4] == pytest.approx(-1.0)

    device.close()


def test_xbox_device_lb_closes_gripper(mock_inputs):
    """Holding LB sets gripper to -1."""
    device = XboxControllerInterventionDevice()
    device._lb_held = True
    action = device.get_action()
    assert action is not None
    assert action[6] == pytest.approx(-1.0)
    device.close()


def test_xbox_device_rb_opens_gripper(mock_inputs):
    """Holding RB sets gripper to +1."""
    device = XboxControllerInterventionDevice()
    device._rb_held = True
    action = device.get_action()
    assert action is not None
    assert action[6] == pytest.approx(1.0)
    device.close()


def test_xbox_device_lb_takes_priority_over_rb(mock_inputs):
    """LB takes priority over RB when both held."""
    device = XboxControllerInterventionDevice()
    device._lb_held = True
    device._rb_held = True
    action = device.get_action()
    assert action[6] == pytest.approx(-1.0)
    device.close()


def test_xbox_device_toggle_intervention(mock_inputs):
    """Start/menu button (or any _TOGGLE_CODES key) toggles is_intervening."""
    device = XboxControllerInterventionDevice()
    assert device.is_intervening is False

    device._is_intervening = True
    assert device.is_intervening is True

    device._is_intervening = False
    assert device.is_intervening is False

    device.close()


def test_xbox_device_deadzone_filters_noise(mock_inputs):
    """Axis values below deadzone produce no action."""
    device = XboxControllerInterventionDevice(deadzone=0.15)
    device._left_x = 0.10   # below deadzone
    device._left_y = 0.05   # below deadzone
    assert device.get_action() is None
    device.close()


def test_xbox_device_deadzone_rescales_above_threshold(mock_inputs):
    """Axis values just above deadzone start at 0 (no jump)."""
    device = XboxControllerInterventionDevice(deadzone=0.15, scale_position=1.0)
    device._left_x = 0.15   # exactly at deadzone boundary → 0
    action = device.get_action()
    # At exactly deadzone, _apply_deadzone returns 0.0 → None
    assert action is None

    device._left_x = 0.16   # just above → small positive value
    action = device.get_action()
    assert action is not None
    assert action[0] > 0.0
    assert action[0] < 0.05   # should be small

    device.close()


def test_xbox_device_action_clipped_to_unit_range(mock_inputs):
    """Actions are clipped to [-1, 1] regardless of scale."""
    device = XboxControllerInterventionDevice(deadzone=0.0, scale_position=10.0)
    device._left_x = 1.0
    action = device.get_action()
    assert action is not None
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    device.close()


def test_xbox_device_reset_clears_state(mock_inputs):
    """reset() zeros all axes and clears intervention flag."""
    device = XboxControllerInterventionDevice()
    device._left_x = 0.5
    device._right_y = -0.3
    device._lb_held = True
    device._is_intervening = True

    device.reset()

    assert device._left_x == 0.0
    assert device._right_y == 0.0
    assert device._lb_held is False
    assert device.is_intervening is False
    assert device.get_action() is None

    device.close()


def test_xbox_device_notify_does_not_raise(mock_inputs):
    device = XboxControllerInterventionDevice()
    device.notify("test notification")
    device.close()


def test_xbox_device_no_hardware_raises(monkeypatch):
    """RuntimeError when no gamepads are connected."""
    mock_mod = MagicMock()
    mock_mod.devices.gamepads = []
    monkeypatch.setitem(sys.modules, "inputs", mock_mod)

    with pytest.raises(RuntimeError, match="No gamepads found"):
        XboxControllerInterventionDevice()


def test_xbox_device_bad_index_raises(monkeypatch):
    """RuntimeError when controller_index exceeds available gamepads."""
    mock_gamepad = MagicMock()
    mock_mod = MagicMock()
    mock_mod.devices.gamepads = [mock_gamepad]
    monkeypatch.setitem(sys.modules, "inputs", mock_mod)

    with pytest.raises(RuntimeError, match="out of range"):
        XboxControllerInterventionDevice(controller_index=5)

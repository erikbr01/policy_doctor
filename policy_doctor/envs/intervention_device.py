"""Intervention devices for DAgger: human input handlers (keyboard, SpaceMouse, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy as np


class InterventionDevice(ABC):
    """Abstract base class for human intervention input devices."""

    @property
    @abstractmethod
    def is_intervening(self) -> bool:
        """Whether the human is currently intervening (True = human control active)."""

    @abstractmethod
    def get_action(self) -> Optional[np.ndarray]:
        """Get the human's action for the current step.

        Returns
        -------
        action : np.ndarray or None
            Action vector of shape (action_dim,), or None if no action ready.
        """

    @abstractmethod
    def notify(self, message: str) -> None:
        """Display a notification (e.g., intervention trigger reason)."""

    def reset(self) -> None:
        """Reset internal state (called at episode start)."""


class PassthroughInterventionDevice(InterventionDevice):
    """Stub device: never intervenes, always returns None."""

    @property
    def is_intervening(self) -> bool:
        return False

    def get_action(self) -> Optional[np.ndarray]:
        return None

    def notify(self, message: str) -> None:
        pass


class KeyboardInterventionDevice(InterventionDevice):
    """pynput-based keyboard controller for arm teleoperation.

    Maps keys to 10D OSC_POSE actions for PandaMobile robot:
      - W/S: arm +/-z (up/down)
      - A/D: arm +/-x (left/right)
      - Q/E: arm +/-y (back/forward)
      - G/H: gripper close/open
      - I/K: base +/-x (forward/back)
      - J/L: base turn left/right
      - Space: toggle is_intervening flag

    Parameters
    ----------
    action_dim : int, default 10
        Expected action dimension (9 for arm pose only, 10 for arm+gripper,
        13 for arm+gripper+base, etc.)
    step_size : float, default 0.05
        Distance/rotation per key press (in meters or radians).
    """

    ACTION_DIM = 10

    KEY_BINDINGS = {
        "w": np.array([0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0]),  # arm +z
        "s": np.array([0, 0, -0.05, 0, 0, 0, 0, 0, 0, 0]),  # arm -z
        "a": np.array([-0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # arm -x
        "d": np.array([0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # arm +x
        "q": np.array([0, -0.05, 0, 0, 0, 0, 0, 0, 0, 0]),  # arm -y
        "e": np.array([0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0]),  # arm +y
        "g": np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0]),  # gripper close
        "h": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # gripper open
        "i": np.array([0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0]),  # base +x
        "k": np.array([0, 0, 0, 0, 0, 0, 0, -0.1, 0, 0]),  # base -x
        "j": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5]),  # base turn left
        "l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]),  # base turn right
    }

    def __init__(self, action_dim: int = 10, step_size: float = 0.05) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise ImportError(
                "pynput not found. Install with: pip install pynput"
            ) from e

        self.action_dim = action_dim
        self.step_size = step_size
        self._is_intervening = False
        self._keys_pressed = set()
        self._listener = None

        self._keyboard = keyboard
        self._start_listener()

    def _start_listener(self) -> None:
        """Start the keyboard listener thread."""

        def on_press(key):
            try:
                char = key.char
                if char and char.lower() in self.KEY_BINDINGS:
                    self._keys_pressed.add(char.lower())
                if char == " ":  # space key
                    self._is_intervening = not self._is_intervening
            except AttributeError:
                pass

        def on_release(key):
            try:
                char = key.char
                if char and char.lower() in self.KEY_BINDINGS:
                    self._keys_pressed.discard(char.lower())
            except AttributeError:
                pass

        self._listener = self._keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    @property
    def is_intervening(self) -> bool:
        return self._is_intervening

    def get_action(self) -> Optional[np.ndarray]:
        """Compose action from currently pressed keys.

        If no keys are pressed, returns None. Otherwise, sums the keybinding
        vectors for all pressed keys and returns the result (clipped to [-1, 1]).
        """
        if not self._keys_pressed:
            return None

        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        for key in self._keys_pressed:
            if key in self.KEY_BINDINGS:
                action += self.KEY_BINDINGS[key]

        action = np.clip(action, -1, 1).astype(np.float32)
        return action

    def notify(self, message: str) -> None:
        """Print a notification message."""
        print(f"[INTERVENTION] {message}")

    def reset(self) -> None:
        """Reset internal state."""
        self._keys_pressed.clear()
        self._is_intervening = False

    def close(self) -> None:
        """Stop the listener thread."""
        if self._listener is not None:
            self._listener.stop()

    def __del__(self):
        self.close()

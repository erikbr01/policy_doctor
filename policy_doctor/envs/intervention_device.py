"""Intervention devices for DAgger: human input handlers (keyboard, SpaceMouse, etc.)."""

from __future__ import annotations

import threading
import time
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
            except AttributeError:
                # Special key (arrows, F-keys, space, etc.)
                if key == self._keyboard.Key.space:
                    self._is_intervening = not self._is_intervening

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


class SpaceMouseInterventionDevice(InterventionDevice):
    """3DConnexion SpaceMouse controller for arm teleoperation.

    Maps 6-DOF SpaceMouse input to 10D OSC_POSE actions for PandaMobile:
      - X/Y/Z translation → arm x/y/z
      - Roll/Pitch/Yaw rotation → arm roll/pitch/yaw
      - Left button (hold) → gripper close
      - Right button (toggle) → switch is_intervening
      - Optional keyboard controls for base movement

    Requires hidapi (pip install hidapi) and SpaceMouse drivers installed.

    Parameters
    ----------
    vendor_id : int, default 9583
        USB vendor ID for 3DConnexion SpaceMouse Compact (Wired).
    product_id : int, default 50741
        USB product ID for 3DConnexion SpaceMouse Compact (Wired).
    deadzone : float, default 0.1
        Ignore sensor values below this magnitude to filter noise.
    scale_position : float, default 125.0
        Scale factor for translational movements.
    scale_rotation : float, default 50.0
        Scale factor for rotational movements.
    """

    ACTION_DIM = 10

    def __init__(
        self,
        vendor_id: int = 9583,
        product_id: int = 50741,
        deadzone: float = 0.1,
        scale_position: float = 125.0,
        scale_rotation: float = 50.0,
    ) -> None:
        try:
            import hid
        except ImportError as e:
            raise ImportError(
                "hidapi not found. Install with: pip install hidapi"
            ) from e

        self.hid = hid
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.deadzone = deadzone
        self.scale_position = scale_position
        self.scale_rotation = scale_rotation

        self._is_intervening = False
        self._device = None
        self._reading_thread = None
        self._stop_reading = False

        # Current state
        self._pose = np.zeros(6, dtype=np.float32)  # x, y, z, roll, pitch, yaw
        self._gripper_active = False
        self._last_left_button_time = 0.0

        self._open_device()
        self._start_reading_thread()

    def _open_device(self) -> None:
        """Open SpaceMouse HID device."""
        try:
            self._device = self.hid.device()
            self._device.open(self.vendor_id, self.product_id)
            self._device.set_nonblocking(True)
            print("[SpaceMouse] Device opened successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open SpaceMouse (vendor={self.vendor_id}, "
                f"product={self.product_id}): {e}. "
                "Check that 3DConnexion drivers are installed and SpaceMouse is connected."
            ) from e

    def _to_int16(self, byte1: int, byte2: int) -> int:
        """Convert two bytes to signed 16-bit integer."""
        val = (byte2 << 8) | byte1
        if val >= 32768:
            val -= 65536
        return val

    def _convert(self, byte1: int, byte2: int) -> float:
        """Convert two bytes to normalized float in [-1, 1]."""
        val = self._to_int16(byte1, byte2)
        return float(val) / 350.0  # 350 is empirical max from reference

    def _parse_hid_packet(self, data: list[int]) -> None:
        """Parse a 13-byte HID packet from SpaceMouse."""
        if not data or len(data) < 13:
            return

        if data[0] == 1:  # 6-DOF sensor reading
            # Extract raw values (data[1:13])
            y = self._convert(data[1], data[2])
            x = self._convert(data[3], data[4])
            z = -self._convert(data[5], data[6])  # negate Z per reference

            roll = self._convert(data[7], data[8])
            pitch = self._convert(data[9], data[10])
            yaw = self._convert(data[11], data[12])

            # Apply deadzone
            if abs(x) < self.deadzone:
                x = 0.0
            if abs(y) < self.deadzone:
                y = 0.0
            if abs(z) < self.deadzone:
                z = 0.0
            if abs(roll) < self.deadzone:
                roll = 0.0
            if abs(pitch) < self.deadzone:
                pitch = 0.0
            if abs(yaw) < self.deadzone:
                yaw = 0.0

            self._pose = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

        elif data[0] == 3:  # Button press
            # Left button (data[1] == 1): gripper control
            if data[1] == 1:
                current_time = time.time()
                if current_time - self._last_left_button_time > 0.2:
                    self._gripper_active = not self._gripper_active
                    self._last_left_button_time = current_time

            # Right button (data[1] == 2): toggle is_intervening
            if data[1] == 2:
                self._is_intervening = not self._is_intervening

    def _read_loop(self) -> None:
        """Background thread: continuously read HID packets."""
        while not self._stop_reading:
            try:
                if self._device is not None:
                    data = self._device.read(13)
                    if data:
                        self._parse_hid_packet(data)
            except Exception as e:
                print(f"[SpaceMouse] Read error: {e}")

            time.sleep(0.001)  # 1ms polling interval

    def _start_reading_thread(self) -> None:
        """Start the HID reading thread."""
        self._stop_reading = False
        self._reading_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reading_thread.start()

    @property
    def is_intervening(self) -> bool:
        return self._is_intervening

    def get_action(self) -> Optional[np.ndarray]:
        """Get 10D action from current SpaceMouse state.

        Maps to 10D OSC_POSE action:
          [0:3]: arm position (x, y, z) from SpaceMouse translation
          [3:6]: arm rotation (roll, pitch, yaw) from SpaceMouse rotation
          [6]: gripper command (1.0 if active, 0.0 otherwise)
          [7:10]: base movement (currently zeros - extend if needed)

        Returns
        -------
        action : np.ndarray or None
            10D action vector scaled to [-1, 1], or None if no input detected.
        """
        # Check if any non-zero input
        if np.allclose(self._pose, 0.0) and not self._gripper_active:
            return None

        # Build 10D action
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        # Pose: apply scaling and clipping
        pose_scaled = self._pose.copy()
        pose_scaled[:3] *= self.scale_position
        pose_scaled[3:6] *= self.scale_rotation

        action[0:6] = np.clip(pose_scaled, -1, 1)

        # Gripper
        action[6] = 1.0 if self._gripper_active else 0.0

        # Base (leave zeros for now; can extend if keyboard controls added)
        # action[7:10] = ...

        return action

    def notify(self, message: str) -> None:
        """Print a notification message."""
        print(f"[SpaceMouse INTERVENTION] {message}")

    def reset(self) -> None:
        """Reset internal state."""
        self._pose.fill(0.0)
        self._gripper_active = False
        self._is_intervening = False

    def close(self) -> None:
        """Stop reading thread and close device."""
        self._stop_reading = True
        if self._reading_thread is not None:
            self._reading_thread.join(timeout=1.0)
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


class XboxControllerInterventionDevice(InterventionDevice):
    """Xbox controller for arm teleoperation using the `inputs` library.

    Axis mapping to 10D OSC_POSE:
      Left stick  X (ABS_X):    arm x       [0]
      Left stick  Y (ABS_Y):    arm y       [1]  (negated: push forward = +y)
      Right stick Y (ABS_RY):   arm z       [2]  (negated: push up = +z)
      Right stick X (ABS_RX):   arm yaw     [5]
      RT - LT (ABS_RZ - ABS_Z): arm pitch   [4]

    Button mapping:
      LB (BTN_TL):     gripper close (action[6] = -1)
      RB (BTN_TR):     gripper open  (action[6] = +1)
      Start/Menu:      toggle is_intervening

    Parameters
    ----------
    controller_index : int
        Index of the gamepad to use (0 = first connected).
    deadzone : float
        Ignore axis values below this magnitude (in normalized [-1, 1] range).
    scale_position : float
        Scale factor applied to translational axes before clipping to [-1, 1].
    scale_rotation : float
        Scale factor applied to rotational axes before clipping to [-1, 1].
    """

    ACTION_DIM = 10
    _STICK_MAX = 32768.0   # inputs reports sticks as -32768..32767
    _TRIGGER_MAX = 255.0   # inputs reports triggers as 0..255

    # Button codes that toggle is_intervening (controller-specific names)
    _TOGGLE_CODES = frozenset({"BTN_START", "BTN_SELECT", "BTN_MODE"})

    def __init__(
        self,
        controller_index: int = 0,
        deadzone: float = 0.15,
        scale_position: float = 1.0,
        scale_rotation: float = 1.0,
    ) -> None:
        try:
            from inputs import devices
        except ImportError as e:
            raise ImportError("inputs not found. Install with: pip install inputs") from e

        # Initialize cleanup handles first so __del__ is always safe
        self._stop = threading.Event()
        self._reading_thread = None

        gamepads = devices.gamepads
        if not gamepads:
            raise RuntimeError(
                "No gamepads found. Connect your Xbox controller and try again."
            )
        if controller_index >= len(gamepads):
            raise RuntimeError(
                f"Controller index {controller_index} out of range; "
                f"{len(gamepads)} gamepad(s) connected."
            )

        self.controller_index = controller_index
        self.deadzone = deadzone
        self.scale_position = scale_position
        self.scale_rotation = scale_rotation

        self._is_intervening = False
        self._lock = threading.Lock()

        # Normalized state: sticks in [-1, 1], triggers in [0, 1]
        self._left_x = 0.0
        self._left_y = 0.0
        self._right_x = 0.0
        self._right_y = 0.0
        self._lt = 0.0
        self._rt = 0.0
        self._lb_held = False
        self._rb_held = False

        self._reading_thread = threading.Thread(
            target=self._read_loop, name="xbox-controller", daemon=True
        )
        self._reading_thread.start()

    def _apply_deadzone(self, value: float) -> float:
        """Linear rescale from [deadzone, 1] → [0, 1], zero below deadzone."""
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _read_loop(self) -> None:
        try:
            from inputs import devices
        except ImportError:
            return

        gamepad = devices.gamepads[self.controller_index]
        while not self._stop.is_set():
            try:
                events = gamepad.read()
            except Exception:
                if self._stop.is_set():
                    return
                time.sleep(0.05)
                continue

            for event in events:
                if self._stop.is_set():
                    return
                self._process_event(event)

    def _process_event(self, event) -> None:
        ev_type = event.ev_type
        code = event.code
        state = event.state

        with self._lock:
            if ev_type == "Absolute":
                if code == "ABS_X":
                    self._left_x = state / self._STICK_MAX
                elif code == "ABS_Y":
                    self._left_y = state / self._STICK_MAX
                elif code == "ABS_RX":
                    self._right_x = state / self._STICK_MAX
                elif code == "ABS_RY":
                    self._right_y = state / self._STICK_MAX
                elif code == "ABS_Z":
                    self._lt = state / self._TRIGGER_MAX
                elif code == "ABS_RZ":
                    self._rt = state / self._TRIGGER_MAX

            elif ev_type == "Key":
                if code == "BTN_TL":
                    self._lb_held = bool(state)
                elif code == "BTN_TR":
                    self._rb_held = bool(state)
                elif code in self._TOGGLE_CODES and state == 1:
                    self._is_intervening = not self._is_intervening

    @property
    def is_intervening(self) -> bool:
        return self._is_intervening

    def get_action(self) -> Optional[np.ndarray]:
        """Get 10D action from current controller state.

        Returns None when all inputs are at rest (no action to apply).
        """
        with self._lock:
            lx = self._apply_deadzone(self._left_x)
            ly = self._apply_deadzone(-self._left_y)   # negate: push fwd = +y
            rx = self._apply_deadzone(self._right_x)
            ry = self._apply_deadzone(-self._right_y)  # negate: push up = +z
            lt = self._lt
            rt = self._rt
            lb = self._lb_held
            rb = self._rb_held

        pitch = rt - lt
        gripper = -1.0 if lb else (1.0 if rb else 0.0)

        if not any([lx, ly, rx, ry, abs(pitch) > 0.01, lb, rb]):
            return None

        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[0] = np.clip(lx * self.scale_position, -1, 1)   # arm x
        action[1] = np.clip(ly * self.scale_position, -1, 1)   # arm y
        action[2] = np.clip(ry * self.scale_position, -1, 1)   # arm z
        action[4] = np.clip(pitch * self.scale_rotation, -1, 1)  # arm pitch
        action[5] = np.clip(rx * self.scale_rotation, -1, 1)   # arm yaw
        action[6] = gripper
        return action

    def notify(self, message: str) -> None:
        print(f"[Xbox INTERVENTION] {message}")

    def reset(self) -> None:
        with self._lock:
            self._left_x = 0.0
            self._left_y = 0.0
            self._right_x = 0.0
            self._right_y = 0.0
            self._lt = 0.0
            self._rt = 0.0
            self._lb_held = False
            self._rb_held = False
        self._is_intervening = False

    def close(self) -> None:
        self._stop.set()
        if self._reading_thread is not None:
            self._reading_thread.join(timeout=1.0)

    def __del__(self):
        self.close()


class HTTPInterventionDevice(InterventionDevice):
    """Polls intervention state from the viz server's GET /intervention endpoint.

    The viz server captures cv2 key presses in its own main thread and exposes
    them here — no pynput, no accessibility permissions required.

    Key bindings (same as KeyboardInterventionDevice):
      Space    toggle human/robot control
      W/S/A/D  arm +z/-z/-x/+x
      Q/E      arm -y/+y
      G/H      gripper close/open

    Parameters
    ----------
    server_url : str
        Base URL of the viz server (e.g. "http://localhost:5002").
    poll_interval : float
        Seconds between polls (default 0.02 = 50 Hz).
    """

    def __init__(self, server_url: str, poll_interval: float = 0.02) -> None:
        import requests
        self._url = server_url.rstrip("/") + "/intervention"
        self._session = requests.Session()
        self._poll_interval = poll_interval
        self._state: dict = {"is_intervening": False, "action": None}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop, name="http-intervention", daemon=True
        )
        self._thread.start()

    @property
    def is_intervening(self) -> bool:
        with self._lock:
            return self._state["is_intervening"]

    def get_action(self) -> Optional[np.ndarray]:
        with self._lock:
            a = self._state.get("action")
        return np.array(a, dtype=np.float32) if a is not None else None

    def notify(self, message: str) -> None:
        pass  # viz server shows it in the frame overlay

    def reset(self) -> None:
        with self._lock:
            self._state = {"is_intervening": False, "action": None}

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def __del__(self) -> None:
        self.close()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                resp = self._session.get(self._url, timeout=0.5)
                if resp.ok:
                    with self._lock:
                        self._state = resp.json()
            except Exception:
                pass
            time.sleep(self._poll_interval)

"""Visualization server: receives camera frames + metadata over HTTP, displays with cv2.

Run in a separate terminal:
    python -m policy_doctor.envs.viz_server [--port 5002] [--fps 30]

The DAgger runner sends frames via HTTP POST; this process owns the cv2
window on its own main thread — no macOS main-thread constraint from the
simulation process.

Wire format (POST /frame):
    First 4 bytes : uint32 big-endian — length of JSON metadata header
    Next N bytes  : UTF-8 JSON  {"node_name": ..., "node_value": ...,
                                 "acting_agent": ..., "step": ...,
                                 "reason": ..., "cameras": ["agentview"]}
    Remaining     : concatenated raw RGB uint8 frames, each (H, W, 3)
                    in the order listed in "cameras"
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, request


# ---------------------------------------------------------------------------
# Shared frame slot
# ---------------------------------------------------------------------------

class _FrameSlot:
    """Thread-safe latest-frame store (drops old frames if consumer is slow)."""

    def __init__(self) -> None:
        self._canvas: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, canvas: np.ndarray) -> None:
        with self._lock:
            self._canvas = canvas
        self._event.set()

    def get_nowait(self) -> Optional[np.ndarray]:
        with self._lock:
            c = self._canvas
            self._canvas = None
        return c

    def wait(self, timeout: float = 0.1) -> bool:
        triggered = self._event.wait(timeout=timeout)
        self._event.clear()
        return triggered


_slot = _FrameSlot()


# ---------------------------------------------------------------------------
# Key state (set by cv2 loop on main thread, read by /intervention endpoint)
# ---------------------------------------------------------------------------

# 10-dim action matching KeyboardInterventionDevice.KEY_BINDINGS
_KEY_ACTIONS = {
    ord("w"): [0, 0,  0.05, 0, 0, 0, 0, 0, 0, 0],
    ord("s"): [0, 0, -0.05, 0, 0, 0, 0, 0, 0, 0],
    ord("a"): [-0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ord("d"): [ 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ord("q"): [0, -0.05, 0, 0, 0, 0, 0, 0, 0, 0],
    ord("e"): [0,  0.05, 0, 0, 0, 0, 0, 0, 0, 0],
    ord("g"): [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    ord("h"): [0, 0, 0, 0, 0, 0,  1, 0, 0, 0],
}

_key_lock = threading.Lock()
_key_state = {"is_intervening": False, "action": None}


def _handle_key(key: int) -> None:
    """Called from the cv2 main-thread loop on each waitKey result."""
    with _key_lock:
        if key == ord(" "):
            _key_state["is_intervening"] = not _key_state["is_intervening"]
        elif key in _KEY_ACTIONS and _key_state["is_intervening"]:
            _key_state["action"] = _KEY_ACTIONS[key]
        else:
            _key_state["action"] = None


# ---------------------------------------------------------------------------
# SpaceMouse reader (optional — started only when device is found)
# ---------------------------------------------------------------------------

def _start_spacemouse(vendor_id: int = 9583, product_id: int = 50741,
                      deadzone: float = 0.1,
                      scale_pos: float = 0.05,
                      scale_rot: float = 0.3) -> bool:
    """Try to open the SpaceMouse and start a reader thread.

    Returns True if the device was found, False otherwise.
    SpaceMouse state is merged into _key_state so /intervention covers both.
    """
    try:
        import hid
    except ImportError:
        return False

    try:
        dev = hid.device()
        dev.open(vendor_id, product_id)
        dev.set_nonblocking(True)
    except Exception:
        return False

    print("[viz server] SpaceMouse connected", flush=True)

    _sm_pose = [0.0] * 6          # x y z roll pitch yaw
    _sm_gripper_close = [False]
    _sm_last_btn_time = [0.0]

    def _convert(b1: int, b2: int) -> float:
        v = (b2 << 8) | b1
        if v >= 32768:
            v -= 65536
        return float(v) / 350.0

    def _reader():
        while True:
            try:
                data = dev.read(13)
            except Exception:
                break
            if not data or len(data) < 13:
                time.sleep(0.001)
                continue

            if data[0] == 1:
                y  = _convert(data[1], data[2])
                x  = _convert(data[3], data[4])
                z  = -_convert(data[5], data[6])
                ro = _convert(data[7], data[8])
                pi = _convert(data[9], data[10])
                ya = _convert(data[11], data[12])
                _sm_pose[:] = [
                    x if abs(x) >= deadzone else 0.0,
                    y if abs(y) >= deadzone else 0.0,
                    z if abs(z) >= deadzone else 0.0,
                    ro if abs(ro) >= deadzone else 0.0,
                    pi if abs(pi) >= deadzone else 0.0,
                    ya if abs(ya) >= deadzone else 0.0,
                ]

            elif data[0] == 3:
                now = time.time()
                if data[1] == 1 and now - _sm_last_btn_time[0] > 0.2:
                    _sm_gripper_close[0] = not _sm_gripper_close[0]
                    _sm_last_btn_time[0] = now
                if data[1] == 2:
                    with _key_lock:
                        _key_state["is_intervening"] = not _key_state["is_intervening"]

            # Build 10-dim action from SpaceMouse pose
            x, y, z, ro, pi, ya = _sm_pose
            gripper = -1.0 if _sm_gripper_close[0] else 1.0
            action = [
                x * scale_pos, y * scale_pos, z * scale_pos,
                ro * scale_rot, pi * scale_rot, ya * scale_rot,
                gripper, 0, 0, 0,
            ]
            with _key_lock:
                if _key_state["is_intervening"]:
                    _key_state["action"] = action

            time.sleep(0.01)

    t = threading.Thread(target=_reader, name="spacemouse", daemon=True)
    t.start()
    return True


# ---------------------------------------------------------------------------
# Flask app (runs in worker threads)
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.logger.disabled = True


@app.get("/intervention")
def get_intervention():
    with _key_lock:
        return dict(_key_state)


@app.post("/frame")
def receive_frame():
    data = request.data
    if len(data) < 4:
        return "bad request", 400

    # Parse header
    meta_len = struct.unpack(">I", data[:4])[0]
    meta = json.loads(data[4 : 4 + meta_len])
    raw = data[4 + meta_len :]

    cameras = meta.get("cameras", ["agentview"])
    h = meta.get("h", 256)
    w = meta.get("w", 256)
    frame_bytes = h * w * 3

    frames = []
    for i, name in enumerate(cameras):
        chunk = raw[i * frame_bytes : (i + 1) * frame_bytes]
        if len(chunk) < frame_bytes:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img)

    canvas = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    canvas = _overlay(canvas, meta)
    _slot.put(canvas)
    return "ok", 200


@app.get("/health")
def health():
    return {"status": "ok"}


def _overlay(canvas: np.ndarray, meta: dict) -> np.ndarray:
    node_value = meta.get("node_value")
    val_str = f"{node_value:.3f}" if node_value is not None else "?"
    acting_agent = meta.get("acting_agent", "robot")
    agent_color = (0, 200, 0) if acting_agent == "robot" else (0, 80, 255)
    lines = [
        (f"Step {meta.get('step', 0)}  |  Node: {meta.get('node_name','?')}  (d={val_str})",
         (255, 255, 255)),
        (f"Agent: {acting_agent.upper()}", agent_color),
    ]
    reason = meta.get("reason", "")
    if reason:
        lines.append((f"[INTERVENTION] {reason}", (0, 80, 255)))
    y = 24
    for text, color in lines:
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
        y += 26
    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(port: int = 5002, fps: int = 30, device: str = "auto") -> None:
    # Optional SpaceMouse
    if device in ("spacemouse", "auto"):
        found = _start_spacemouse()
        if not found and device == "spacemouse":
            print("[viz server] WARNING: SpaceMouse not found", flush=True)
        elif found:
            print("[viz server] Input: SpaceMouse + keyboard", flush=True)
        else:
            print("[viz server] Input: keyboard only", flush=True)
    else:
        print("[viz server] Input: keyboard only", flush=True)

    # Flask in a daemon thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port,
                               threaded=True, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    print(f"[viz server] listening on http://127.0.0.1:{port}", flush=True)

    # cv2 display loop on the main thread
    window = "DAgger"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)
    delay_ms = max(1, 1000 // fps)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for i, line in enumerate([
        "Waiting for frames...",
        "Keys: Space=toggle  W/S/A/D/Q/E=arm  G/H=gripper  Q=quit",
    ]):
        cv2.putText(blank, line, (20, 220 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.imshow(window, blank)

    while True:
        _slot.wait(timeout=0.1)
        canvas = _slot.get_nowait()
        if canvas is not None:
            cv2.imshow(window, canvas)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key != 255:  # 255 = no key pressed
            _handle_key(key)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    _cupid = _root / "third_party" / "cupid"
    for p in [str(_root), str(_cupid)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    parser = argparse.ArgumentParser(description="DAgger visualization server")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "keyboard", "spacemouse"],
                        help="auto: try SpaceMouse, fall back to keyboard")
    args = parser.parse_args()
    serve(args.port, args.fps, args.device)

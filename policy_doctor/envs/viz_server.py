"""Visualization server: receives camera frames + metadata over HTTP, displays with cv2.

Run in a separate terminal:
    python -m policy_doctor.envs.viz_server [--port 5002] [--fps 30] [--task square_mh]

The DAgger runner sends frames via HTTP POST; this process owns the cv2
window on its own main thread — no macOS main-thread constraint from the
simulation process.

On macOS, opencv-python and pygame may each load ``libSDL2``; ``serve()`` installs
a stderr line filter for the noisy ObjC duplicate-class messages, and suppresses
pygame's ``pkg_resources`` warning. OpenCV is imported after the pygame controller
when ``--device pygame``; the pygame device uses ``SDL_VIDEODRIVER=dummy``
(joystick-only).

Wire format (POST /frame):
    First 4 bytes : uint32 big-endian — length of JSON metadata header
    Next N bytes  : UTF-8 JSON  {"node_name": ..., "node_value": ...,
                                 "acting_agent": ..., "step": ...,
                                 "reason": ..., "cameras": ["agentview", "robot0_eye_in_hand"]}
    Remaining     : concatenated raw RGB uint8 frames, each (H, W, 3)
                    in the order listed in "cameras"
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from flask import Flask, request

from policy_doctor.envs.macos_quiet import install_macos_sdl_noise_suppression
from policy_doctor.envs.dagger_config import (
    build_spacemouse_spatial_matrices,
    load_merged_dagger_config,
)
from policy_doctor.spacemouse_hid import (
    SPACEMOUSE_USB_PAIRS_DEFAULT,
    apply_spacemouse_spatial_mapping,
    dedupe_usb_pairs,
    decode_spacemouse_motion_report,
    spacemouse_usb_pairs_with_override,
    try_open_first_spacemouse,
)

_OPENCV: Any = None


def _opencv():
    """Lazily import cv2 after optional pygame/SDL init to reduce duplicate-libSDL2 issues on macOS."""
    global _OPENCV
    if _OPENCV is None:
        import cv2

        _OPENCV = cv2
    return _OPENCV


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
_key_state = {"is_intervening": False, "action": None, "reset_requested": False}
_pygame_device = None


def _handle_key(key: int) -> None:
    """Called from the cv2 main-thread loop on each waitKey result."""
    with _key_lock:
        if key == ord(" "):
            _key_state["is_intervening"] = not _key_state["is_intervening"]
        if key == ord("r"):
            _key_state["reset_requested"] = True
        if key in _KEY_ACTIONS:
            _key_state["action"] = _KEY_ACTIONS[key]
        else:
            _key_state["action"] = None


def _start_pygame_controller(dagger_config_name: str, task: Optional[str] = None) -> bool:
    """Open a pygame controller in the viz process (same ``pygame:`` YAML as ``run_dagger``).

    When using ``viz_url``, input runs here — button maps and sensitivity must load from YAML,
    not from the runner process.
    """
    global _pygame_device
    try:
        from policy_doctor.envs.data_collection_config import load_data_collection_task_config
        from policy_doctor.envs.dagger_config import (
            build_pygame_controller_kwargs,
            load_dagger_config,
            merge_data_collection_task_into_dagger_cfg,
        )
        from policy_doctor.envs.intervention_device import PygameControllerInterventionDevice

        dagger_cfg = load_dagger_config(dagger_config_name)
        if task:
            merge_data_collection_task_into_dagger_cfg(
                dagger_cfg, load_data_collection_task_config(task)
            )
        kw = build_pygame_controller_kwargs(dagger_cfg)
        _pygame_device = PygameControllerInterventionDevice(**kw)
    except Exception as e:
        print(f"[viz server] pygame controller unavailable: {e}", flush=True)
        _pygame_device = None
        return False
    msg = (
        f"[viz server] Input: pygame controller + keyboard "
        f"(dagger_config={dagger_config_name}"
    )
    if task:
        msg += f", task={task}"
    msg += ")"
    print(msg, flush=True)
    return True


def _poll_pygame_controller() -> None:
    if _pygame_device is None:
        return
    try:
        action = _pygame_device.get_action()
        intervening = bool(_pygame_device.is_intervening)
        reset_requested = bool(_pygame_device.consume_reset_request())
    except Exception as e:
        print(f"[viz server] pygame controller poll failed: {e}", flush=True)
        return
    with _key_lock:
        _key_state["is_intervening"] = intervening
        _key_state["action"] = action.tolist() if action is not None else None
        _key_state["reset_requested"] = _key_state["reset_requested"] or reset_requested


# ---------------------------------------------------------------------------
# SpaceMouse reader (optional — started only when device is found)
# ---------------------------------------------------------------------------

def _log_spacemouse_hid_probe() -> None:
    """Print 3Dconnexion HID devices seen by hidapi (helps debug wireless PIDs)."""
    try:
        import hid
    except ImportError:
        return
    hits: list[str] = []
    for d in hid.enumerate():
        vid = int(d.get("vendor_id", 0))
        pid = int(d.get("product_id", 0))
        man = (d.get("manufacturer_string") or "") or ""
        prod = (d.get("product_string") or "") or ""
        if vid == 0x256F or "3dconnexion" in man.lower() or "space" in prod.lower():
            hits.append(
                f"    vid=0x{vid:04x} pid=0x{pid:04x}  ({vid},{pid})  {prod!r}"
            )
    if hits:
        print("[viz server] HID devices matching 3Dconnexion / SpaceMouse:", flush=True)
        print("\n".join(hits), flush=True)
        print(
            "[viz server] Pass matching pair: "
            "--spacemouse-vid 0x256f --spacemouse-pid 0x....",
            flush=True,
        )
    else:
        print(
            "[viz server] hidapi enumerated no 3Dconnexion devices — "
            "plug the receiver/USB cable and check System Settings → Privacy.",
            flush=True,
        )


def _start_spacemouse(
    usb_pairs: Optional[list[tuple[int, int]]] = None,
    deadzone: float = 0.1,
    scale_pos: float = 0.15,
    scale_rot: float = 0.1,
    translation_mix: Optional[np.ndarray] = None,
    rotation_mix: Optional[np.ndarray] = None,
) -> bool:
    """Try to open a SpaceMouse on one of ``usb_pairs`` and start a reader thread.

    Returns True if the device was found, False otherwise.
    SpaceMouse state is merged into _key_state so /intervention covers both.

    ``translation_mix`` / ``rotation_mix`` are 3×3 from merged dagger YAML
    (``spacemouse.spatial_mapping``); default identity if omitted.
    """
    trans_m = np.asarray(
        translation_mix if translation_mix is not None else np.eye(3),
        dtype=np.float64,
    )
    rot_m = np.asarray(
        rotation_mix if rotation_mix is not None else np.eye(3),
        dtype=np.float64,
    )
    try:
        opened = try_open_first_spacemouse(usb_pairs)
    except ImportError:
        print(
            "[viz server] SpaceMouse requires hidapi:  pip install hidapi",
            flush=True,
        )
        return False

    pairs = dedupe_usb_pairs(list(usb_pairs or list(SPACEMOUSE_USB_PAIRS_DEFAULT)))
    if opened is None:
        tried = ", ".join(f"0x{v:04x}:0x{p:04x}" for v, p in pairs)
        print(f"[viz server] SpaceMouse not found (tried {tried})", flush=True)
        _log_spacemouse_hid_probe()
        return False

    dev, opened_vid, opened_pid = opened
    print(
        f"[viz server] SpaceMouse connected  usb 0x{opened_vid:04x}:0x{opened_pid:04x}",
        flush=True,
    )

    _sm_pose = [0.0] * 6          # x y z roll pitch yaw
    _sm_gripper_close = [False]
    _sm_last_left_toggle = [0.0]
    _sm_prev_btn_bits = [0]  # previous data[1] for report id 3 (bitmask)

    def _reader():
        while True:
            try:
                # Motion reports are 13 bytes; button reports are often shorter — read a full buffer.
                data = dev.read(64)
            except Exception:
                break
            if not data:
                time.sleep(0.001)
                continue

            rid = data[0]
            if rid == 1 and len(data) >= 13:
                try:
                    tx, ty, tz, r0, r1, r2 = decode_spacemouse_motion_report(data)
                    x, y, z, ro, pi, ya = apply_spacemouse_spatial_mapping(
                        tx, ty, tz, r0, r1, r2, trans_m, rot_m
                    )
                except ValueError:
                    continue
                _sm_pose[:] = [
                    x if abs(x) >= deadzone else 0.0,
                    y if abs(y) >= deadzone else 0.0,
                    z if abs(z) >= deadzone else 0.0,
                    ro if abs(ro) >= deadzone else 0.0,
                    pi if abs(pi) >= deadzone else 0.0,
                    ya if abs(ya) >= deadzone else 0.0,
                ]

            elif rid == 3 and len(data) >= 2:
                # data[1] is a bitmask on most firmware (1=left, 2=right, 3=both).
                b = int(data[1])
                prev = int(_sm_prev_btn_bits[0])
                now = time.time()
                if (b & 1) and not (prev & 1):
                    if now - _sm_last_left_toggle[0] > 0.2:
                        _sm_gripper_close[0] = not _sm_gripper_close[0]
                        _sm_last_left_toggle[0] = now
                if (b & 2) and not (prev & 2):
                    with _key_lock:
                        _key_state["is_intervening"] = not _key_state["is_intervening"]
                _sm_prev_btn_bits[0] = b

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
        state = dict(_key_state)
        _key_state["reset_requested"] = False
        return state


@app.post("/frame")
def receive_frame():
    cv2 = _opencv()
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
    cv2 = _opencv()
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

    cameras = meta.get("cameras") or ["agentview"]
    ch, cw = canvas.shape[:2]
    ncam = len(cameras)
    if ncam > 1:
        col_w = cw // ncam
        for i, cam_name in enumerate(cameras):
            x0 = i * col_w + 8
            label = str(cam_name)
            cv2.putText(canvas, label, (x0, ch - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (x0, ch - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(
    port: int = 5002,
    fps: int = 30,
    device: str = "pygame",
    dagger_config_name: str = "spacemouse_default",
    task: Optional[str] = None,
    spacemouse_vendor_id: Optional[int] = None,
    spacemouse_product_id: Optional[int] = None,
) -> None:
    install_macos_sdl_noise_suppression()
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    # Optional input devices. The viz process owns all user input whenever the
    # runner is launched with viz_url.
    if device == "pygame":
        _start_pygame_controller(dagger_config_name, task=task)
    elif device in ("spacemouse", "auto"):
        merged = load_merged_dagger_config(dagger_config_name, task)
        trans_mix, rot_mix = build_spacemouse_spatial_matrices(merged)
        sm_pairs = spacemouse_usb_pairs_with_override(spacemouse_vendor_id, spacemouse_product_id)
        found = _start_spacemouse(
            usb_pairs=sm_pairs,
            translation_mix=trans_mix,
            rotation_mix=rot_mix,
        )
        if not found and device == "spacemouse":
            print("[viz server] WARNING: SpaceMouse not found", flush=True)
        elif found:
            print("[viz server] Input: SpaceMouse + keyboard", flush=True)
        else:
            print("[viz server] Input: keyboard only", flush=True)
    else:
        print("[viz server] Input: keyboard only", flush=True)

    # Import OpenCV after pygame (when used) so SDL2 is not initialized twice before pygame.
    cv2 = _opencv()

    # Flask in a daemon thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port,
                               threaded=True, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    print(f"[viz server] listening on http://127.0.0.1:{port}", flush=True)

    # cv2 display loop on the main thread (cv2 imported above after input backends)
    window = "DAgger"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)
    delay_ms = max(1, 1000 // fps)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for i, line in enumerate([
        "Waiting for frames...",
        "Keys: Space=toggle  R=reset  W/S/A/D/Q/E=arm  G/H=gripper  Q=quit",
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
        _handle_key(key)  # 255 = no key pressed → clears action
        _poll_pygame_controller()

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
    parser.add_argument("--device", default="spacemouse",
                        choices=["auto", "keyboard", "spacemouse", "pygame"],
                        help="spacemouse (default); pygame: game controller; auto: SpaceMouse if present else keyboard")
    parser.add_argument(
        "--dagger-config",
        default="spacemouse_default",
        metavar="NAME",
        help="Hydra dagger YAML stem — SpaceMouse spatial_mapping when --device spacemouse|auto; pygame when --device pygame.",
    )
    parser.add_argument(
        "--task",
        default=None,
        metavar="NAME",
        help="Optional data_collection task stem (e.g. square_mh) "
        "to merge pygame/* and spacemouse.spatial_mapping overrides into dagger-config.",
    )
    parser.add_argument(
        "--spacemouse-vid",
        type=lambda s: int(s, 0),
        default=None,
        metavar="VID",
        help="USB vendor id for SpaceMouse (decimal or 0x hex), e.g. 0x256f — use with --spacemouse-pid",
    )
    parser.add_argument(
        "--spacemouse-pid",
        type=lambda s: int(s, 0),
        default=None,
        metavar="PID",
        help="USB product id for SpaceMouse (decimal or 0x hex) — use with --spacemouse-vid",
    )
    args = parser.parse_args()
    if (args.spacemouse_vid is None) ^ (args.spacemouse_pid is None):
        parser.error("--spacemouse-vid and --spacemouse-pid must be passed together")
    serve(
        args.port,
        args.fps,
        args.device,
        args.dagger_config,
        task=args.task,
        spacemouse_vendor_id=args.spacemouse_vid,
        spacemouse_product_id=args.spacemouse_pid,
    )

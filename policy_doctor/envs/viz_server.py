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
# Flask app (runs in worker threads)
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.logger.disabled = True


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

def serve(port: int = 5002, fps: int = 30) -> None:
    # Flask in a daemon thread (handles incoming POST /frame)
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
    cv2.putText(blank, "Waiting for frames...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.imshow(window, blank)

    while True:
        _slot.wait(timeout=0.1)
        canvas = _slot.get_nowait()
        if canvas is not None:
            cv2.imshow(window, canvas)
        key = cv2.waitKey(delay_ms)
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

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
    args = parser.parse_args()
    serve(args.port, args.fps)

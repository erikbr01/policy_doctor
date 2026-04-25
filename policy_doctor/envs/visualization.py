"""DAgger visualization client — sends frames to the viz server over HTTP.

The viz server (viz_server.py) runs as a separate process with its own cv2
main thread.  This client sends frames in a fire-and-forget background thread
so the DAgger runner is never blocked by display latency.

Usage
-----
Start the server first:
    python -m policy_doctor.envs.viz_server --port 5002

Then construct DAggerVisualizer with the server URL:
    viz = DAggerVisualizer(server_url="http://localhost:5002")
"""

from __future__ import annotations

import json
import struct
import threading
from queue import Empty, Queue
from typing import Optional

import numpy as np

_DEFAULT_URL = "http://127.0.0.1:5002"


class DAggerVisualizer:
    """HTTP client that streams frames to the viz server.

    update() is non-blocking: it drops the frame into a single-slot queue
    and returns immediately.  A background thread drains the queue and POSTs
    to the server.  If the server is slow, intermediate frames are dropped
    (latest-frame semantics).

    Parameters
    ----------
    server_url : str
        URL of the running viz_server process.
    camera_names : list[str]
        Cameras to include in each frame.
    figsize : tuple
        Unused — kept for API compat.
    hw : tuple[int, int]
        (height, width) to render each camera at.
    """

    def __init__(
        self,
        server_url: str = _DEFAULT_URL,
        camera_names: list[str] = ["agentview"],
        figsize: tuple = (8, 5),
        hw: tuple[int, int] = (256, 256),
    ) -> None:
        self._url = server_url.rstrip("/") + "/frame"
        self.camera_names = camera_names
        self._hw = hw

        # Single-slot queue: newest frame wins
        self._queue: Queue = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._send_loop, name="dagger-viz-sender", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def update(
        self,
        camera_imgs: dict[str, np.ndarray],
        node_name: str,
        node_value: Optional[float],
        acting_agent: str,
        step: int,
        intervention_reason: str = "",
    ) -> None:
        """Non-blocking: enqueue the latest frame for sending."""
        payload = (camera_imgs, node_name, node_value, acting_agent, step, intervention_reason)
        try:
            self._queue.put_nowait(payload)
        except Exception:
            # Queue full — replace with newest frame
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def __del__(self) -> None:
        pass  # runner calls close() explicitly

    # ------------------------------------------------------------------
    # Sender thread — HTTP I/O only, no cv2, no MPS
    # ------------------------------------------------------------------

    def _send_loop(self) -> None:
        import requests
        session = requests.Session()
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                data = self._encode(*payload)
                session.post(self._url, data=data,
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=1)
            except Exception:
                pass  # server not running — silently drop

    def _encode(
        self,
        camera_imgs: dict,
        node_name: str,
        node_value: Optional[float],
        acting_agent: str,
        step: int,
        reason: str,
    ) -> bytes:
        h, w = self._hw
        meta = json.dumps({
            "node_name": node_name,
            "node_value": float(node_value) if node_value is not None else None,
            "acting_agent": acting_agent,
            "step": step,
            "reason": reason,
            "cameras": self.camera_names,
            "h": h,
            "w": w,
        }).encode()

        frames_bytes = b""
        for name in self.camera_names:
            img = camera_imgs.get(name)
            if img is None:
                frames_bytes += bytes(h * w * 3)
            else:
                img = np.asarray(img, dtype=np.uint8)
                if img.shape[:2] != (h, w):
                    import cv2
                    img = cv2.resize(img, (w, h))
                frames_bytes += img.tobytes()

        return struct.pack(">I", len(meta)) + meta + frames_bytes

"""Live OpenCV visualization for DAgger rollouts — decoupled render loop.

The main thread calls update() to push the latest frame + metadata into a
shared slot.  A background daemon thread reads from that slot and drives the
cv2 window at ~30 fps, independently of the sim step rate.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class DAggerVisualizer:
    """Decoupled cv2 display for DAgger rollouts.

    Usage
    -----
    - Call update() from the main thread after each sim step.  It is
      non-blocking: it just writes to a shared state dict under a lock.
    - A background thread reads that state and calls cv2.imshow at ~30 fps.
    - Call close() at the end to stop the render thread cleanly.

    Parameters
    ----------
    camera_names : list[str]
        Cameras to show side-by-side.
    figsize : tuple[float, float]
        Unused — kept for API compat with the old matplotlib version.
    fps : int
        Target display frame rate for the render thread.
    """

    def __init__(
        self,
        camera_names: list[str] = ["agentview"],
        figsize: tuple[float, float] = (8, 5),
        fps: int = 30,
    ) -> None:
        self.camera_names = camera_names
        self._window = "DAgger"
        self._fps = fps
        self._state: Optional[dict] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._render_loop, name="dagger-viz", daemon=True
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
        """Non-blocking: store the latest state for the render thread."""
        with self._lock:
            self._state = {
                "imgs": camera_imgs,
                "node_name": node_name,
                "node_value": node_value,
                "acting_agent": acting_agent,
                "step": step,
                "reason": intervention_reason,
            }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        cv2.destroyWindow(self._window)

    def __del__(self) -> None:
        pass  # runner calls close() explicitly

    # ------------------------------------------------------------------
    # Render thread
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, 640 * len(self.camera_names), 480)
        delay_ms = max(1, 1000 // self._fps)

        while not self._stop.is_set():
            with self._lock:
                state = self._state

            if state is not None:
                try:
                    canvas = self._build_canvas(state)
                    cv2.imshow(self._window, canvas)
                except Exception:
                    pass

            cv2.waitKey(delay_ms)

    def _build_canvas(self, state: dict) -> np.ndarray:
        frames = []
        for name in self.camera_names:
            img = state["imgs"].get(name)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                img = np.asarray(img, dtype=np.uint8)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)

        canvas = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]

        val_str = f"{state['node_value']:.3f}" if state["node_value"] is not None else "?"
        agent_color = (0, 200, 0) if state["acting_agent"] == "robot" else (0, 80, 255)
        lines = [
            (f"Step {state['step']}  |  Node: {state['node_name']}  (d={val_str})",
             (255, 255, 255)),
            (f"Agent: {state['acting_agent'].upper()}", agent_color),
        ]
        if state["reason"]:
            lines.append((f"[INTERVENTION] {state['reason']}", (0, 80, 255)))

        y = 24
        for text, color in lines:
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1, cv2.LINE_AA)
            y += 26

        return canvas

"""Live OpenCV visualization for DAgger rollouts.

On macOS, all cv2 window calls must happen on the main thread.  The heavy
work (color conversion, text overlay) runs in a background thread that
pre-builds canvases.  The main thread calls update() which:
  1. Writes the latest state (non-blocking, lock + dict).
  2. Displays the most recently pre-built canvas via cv2.imshow + waitKey(1).
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class DAggerVisualizer:
    """cv2 display with background canvas builder.

    Parameters
    ----------
    camera_names : list[str]
        Cameras to show side-by-side.
    figsize : tuple[float, float]
        Unused — kept for API compat with the old matplotlib version.
    """

    def __init__(
        self,
        camera_names: list[str] = ["agentview"],
        figsize: tuple[float, float] = (8, 5),
    ) -> None:
        self.camera_names = camera_names
        self._window = "DAgger"

        # Window must be created on the main thread
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, 640 * len(camera_names), 480)

        # Shared state: main thread writes, builder thread reads
        self._state: Optional[dict] = None
        self._state_lock = threading.Lock()
        self._state_event = threading.Event()  # wakes builder when new state arrives

        # Shared canvas: builder thread writes, main thread reads
        self._canvas: Optional[np.ndarray] = None
        self._canvas_lock = threading.Lock()

        self._stop = threading.Event()
        self._builder = threading.Thread(
            target=self._build_loop, name="dagger-canvas", daemon=True
        )
        self._builder.start()

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
        """Push new state and display the latest pre-built canvas.

        Fast on the main thread: one lock write + cv2.imshow.
        """
        # 1. Update state for the builder thread
        with self._state_lock:
            self._state = dict(
                imgs=camera_imgs,
                node_name=node_name,
                node_value=node_value,
                acting_agent=acting_agent,
                step=step,
                reason=intervention_reason,
            )
        self._state_event.set()

        # 2. Display the most recently built canvas (or a blank frame)
        with self._canvas_lock:
            canvas = self._canvas

        if canvas is not None:
            cv2.imshow(self._window, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        self._stop.set()
        self._state_event.set()  # unblock builder if waiting
        self._builder.join(timeout=2)
        cv2.destroyWindow(self._window)

    def __del__(self) -> None:
        pass  # runner calls close() explicitly

    # ------------------------------------------------------------------
    # Builder thread — numpy-only, no cv2 window calls
    # ------------------------------------------------------------------

    def _build_loop(self) -> None:
        while not self._stop.is_set():
            triggered = self._state_event.wait(timeout=0.5)
            self._state_event.clear()
            if not triggered or self._stop.is_set():
                continue

            with self._state_lock:
                state = dict(self._state) if self._state else None

            if state is None:
                continue

            canvas = self._build_canvas(state)

            with self._canvas_lock:
                self._canvas = canvas

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

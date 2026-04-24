"""Live OpenCV visualization for DAgger rollouts."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class DAggerVisualizer:
    """Live cv2 display of DAgger episode: camera frame + status overlay.

    Parameters
    ----------
    camera_names : list[str]
        List of cameras to display side-by-side.
    figsize : tuple[float, float]
        Unused — kept for API compatibility with the old matplotlib version.
    """

    def __init__(
        self,
        camera_names: list[str] = ["agentview"],
        figsize: tuple[float, float] = (8, 5),
    ) -> None:
        self.camera_names = camera_names
        self._window = "DAgger"
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, 640 * len(camera_names), 480)

    def update(
        self,
        camera_imgs: dict[str, np.ndarray],
        node_name: str,
        node_value: Optional[float],
        acting_agent: str,
        step: int,
        intervention_reason: str = "",
    ) -> None:
        frames = []
        for name in self.camera_names:
            img = camera_imgs.get(name)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # robosuite renders RGB; cv2 wants BGR
                img = np.asarray(img, dtype=np.uint8)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)

        canvas = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]

        # Status overlay
        val_str = f"{node_value:.3f}" if node_value is not None else "?"
        agent_color = (0, 255, 0) if acting_agent == "robot" else (0, 80, 255)
        lines = [
            (f"Step {step}  |  Node: {node_name}  (d={val_str})", (255, 255, 255)),
            (f"Agent: {acting_agent.upper()}", agent_color),
        ]
        if intervention_reason:
            lines.append((f"[INTERVENTION] {intervention_reason}", (0, 80, 255)))

        y = 24
        for text, color in lines:
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1, cv2.LINE_AA)
            y += 26

        cv2.imshow(self._window, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self._window)

    def __del__(self) -> None:
        pass  # runner calls close() explicitly; avoid double-close from GC

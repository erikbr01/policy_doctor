"""Live matplotlib visualization for DAgger rollouts."""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("TkAgg")  # Must be set before pyplot import; use "MacOSX" or "Agg" if Tk unavailable
import matplotlib.pyplot as plt
import numpy as np


class DAggerVisualizer:
    """Live matplotlib display of DAgger episode: camera + node assignment + status.

    Parameters
    ----------
    camera_names : list[str], default ["agentview"]
        List of cameras to display.
    figsize : tuple[float, float], default (8, 5)
        Figure size (width, height) in inches.
    """

    def __init__(
        self,
        camera_names: list[str] = ["agentview"],
        figsize: tuple[float, float] = (8, 5),
    ) -> None:
        self.camera_names = camera_names
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(
            1, len(camera_names), figsize=figsize, squeeze=False
        )
        self.axes = self.axes.flatten()
        self.images = [None] * len(camera_names)

        for i, ax in enumerate(self.axes):
            ax.set_title(camera_names[i])
            ax.axis("off")

        self.text_ax = None
        try:
            self.fig.canvas.manager.set_window_title("DAgger Rollout Monitor")
        except AttributeError:
            pass  # No window manager (headless or non-interactive backend)

    def update(
        self,
        camera_imgs: dict[str, np.ndarray],
        node_name: str,
        node_value: Optional[float] = None,
        acting_agent: str = "robot",
        step: int = 0,
        intervention_reason: str = "",
    ) -> None:
        """Update visualization with current state.

        Parameters
        ----------
        camera_imgs : dict[str, np.ndarray]
            Dict mapping camera names to RGB images.
        node_name : str
            Behavior graph node assignment.
        node_value : float, optional
            V-value or distance metric for the current node.
        acting_agent : str
            "robot" or "human".
        step : int
            Current step number.
        intervention_reason : str
            Reason if intervention was triggered.
        """
        # Update camera images
        for i, cam_name in enumerate(self.camera_names):
            if cam_name in camera_imgs:
                img = camera_imgs[cam_name]
                if self.images[i] is None:
                    self.images[i] = self.axes[i].imshow(img)
                else:
                    self.images[i].set_data(img)

                # Color border based on acting agent
                border_color = "green" if acting_agent == "robot" else "red"
                for spine in self.axes[i].spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(3)

        # Update status text
        status_text = f"Step: {step} | Node: {node_name} | Agent: {acting_agent}"
        if node_value is not None:
            status_text += f" | Value: {node_value:.3f}"
        if intervention_reason:
            status_text += f"\n[INTERVENTION] {intervention_reason}"

        self.fig.suptitle(status_text, fontsize=12, weight="bold")

        try:
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Brief pause to allow rendering
        except Exception:
            pass  # Ignore drawing errors (e.g., window closed)

    def close(self) -> None:
        """Close the visualization window."""
        plt.close(self.fig)

    def __del__(self):
        self.close()

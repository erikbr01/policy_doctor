"""
Standalone script to visualize the global influence matrix in a native interactive window.
Run this instead of the Streamlit app when you need to inspect the full matrix structure
without browser memory limitations.

Usage:
    python influence_visualizer/scripts/view_matrix.py \
        --eval_dir data/outputs/... \
        --demo_video_dir data/outputs/... \
        --dataset_path data/robomimic/...
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from influence_visualizer.data_loader import load_influence_data


def parse_args():
    parser = argparse.ArgumentParser(description="Global Influence Matrix Viewer")
    # Replicate necessary args from app.py
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--demo_video_dir", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--image_dataset_path", type=str, default=None)
    parser.add_argument("--exp_date", type=str, default="default")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--train_ckpt", type=str, default="latest")
    parser.add_argument("--include_holdout", type=bool, default=True)
    parser.add_argument("--mock", action="store_true")

    # Sampler config
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--max_train_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--pad_before", type=int, default=1)
    parser.add_argument("--pad_after", type=int, default=7)

    return parser.parse_known_args()[0]


class MatrixViewer:
    def __init__(self, data):
        self.data = data
        self.matrix = data.influence_matrix

        # Pre-calculate episode boundaries for fast lookup
        self.rollout_starts = np.array(
            [ep.sample_start_idx for ep in data.rollout_episodes]
        )
        self.demo_starts = np.array([ep.sample_start_idx for ep in data.demo_episodes])

        self.setup_plot()

    def format_coord(self, x, y):
        """Custom tooltip text showing Episode/Timestep info on hover."""
        x, y = int(x), int(y)

        # Bounds check
        if x < 0 or x >= self.matrix.shape[1] or y < 0 or y >= self.matrix.shape[0]:
            return f"x={x}, y={y}"

        val = self.matrix[y, x]

        # Find Demo Episode (X-axis)
        # searchsorted returns insertion point, so subtract 1 to get index
        d_ep_idx = np.searchsorted(self.demo_starts, x, side="right") - 1
        d_ep = self.data.demo_episodes[d_ep_idx]
        d_t = x - d_ep.sample_start_idx

        # Find Rollout Episode (Y-axis)
        r_ep_idx = np.searchsorted(self.rollout_starts, y, side="right") - 1
        r_ep = self.data.rollout_episodes[r_ep_idx]
        r_t = y - r_ep.sample_start_idx

        return (
            f"[Rollout Ep {r_ep.index} t={r_t}] influenced by "
            f"[Demo Ep {d_ep.index} t={d_t}] | "
            f"Score: {val:.4f}"
        )

    def setup_plot(self):
        print("Rendering matrix... (Window will open shortly)")

        fig, ax = plt.subplots(figsize=(14, 8))
        self.fig = fig
        self.ax = ax

        # Plot the main heatmap
        # 'nearest' interpolation is critical for scientific accuracy (pixels don't bleed)
        im = ax.imshow(
            self.matrix,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            origin="upper",
        )
        plt.colorbar(im, ax=ax, label="Influence Score")

        # Draw Lines
        # We use a collection for performance rather than individual plot calls
        from matplotlib.collections import LineCollection

        # Horizontal lines (Rollout boundaries)
        h_lines = []
        width = self.matrix.shape[1]
        for ep in self.data.rollout_episodes[:-1]:
            y = ep.sample_end_idx - 0.5
            h_lines.append([(0, y), (width, y)])

        lc_h = LineCollection(h_lines, colors="black", linewidths=0.5, alpha=0.3)
        ax.add_collection(lc_h)

        # Vertical lines (Demo boundaries)
        v_lines = []
        height = self.matrix.shape[0]
        for ep in self.data.demo_episodes[:-1]:
            x = ep.sample_end_idx - 0.5
            v_lines.append([(x, 0), (x, height)])

        lc_v = LineCollection(v_lines, colors="black", linewidths=0.5, alpha=0.3)
        ax.add_collection(lc_v)

        # Setup labels and interaction
        ax.set_title("Global Influence Matrix (Zoom/Pan enabled)")
        ax.set_xlabel("Demonstration Samples (Sequentially Stacked)")
        ax.set_ylabel("Rollout Samples (Sequentially Stacked)")

        # Override the status bar text format
        ax.format_coord = self.format_coord

        # Add a crosshair cursor
        self.cursor = Cursor(ax, useblit=True, color="red", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.show()


def main():
    args = parse_args()

    print("Loading data...")
    data = load_influence_data(
        eval_dir=args.eval_dir,
        demo_video_dir=args.demo_video_dir,
        dataset_path=args.dataset_path,
        image_dataset_path=args.image_dataset_path,
        exp_date=args.exp_date,
        use_mock=args.mock,
        train_dir=args.train_dir,
        train_ckpt=args.train_ckpt,
        include_holdout=args.include_holdout,
        val_ratio=args.val_ratio,
        max_train_episodes=args.max_train_episodes,
        seed=args.seed,
        horizon=args.horizon,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
    )

    MatrixViewer(data)


if __name__ == "__main__":
    main()

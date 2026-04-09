"""Compute trajectory-level influences for each rollout and demonstration.

This script loads the action influence matrix and computes trajectory-level
influences for each individual rollout-demonstration pair without aggregation.

The action influence matrix has shape (num_rollout_samples, num_demo_samples).
This script extracts influence submatrices for each (rollout_i, demo_j) pair,
which have shape (rollout_i_len, demo_j_len), and renders them as heatmap images.

Output files:
- trajectory_influences/rollout_<i>_demo_<j>.png: Individual influence heatmaps
- trajectory_influences/metadata.npz: Episode metadata and success masks

Usage:
    python influence_visualizer/scripts/compute_trajectory_influences.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
        --train_dir data/outputs/train/jan16/jan16_train_diffusion_unet_lowdim_lift_mh_0 \
        --demo_video_dir data/outputs/videos/lift_mh \
        --train_ckpt "epoch=0100-test_mean_score=1.000" \
        --include_holdout False \
        --output_dir trajectory_influences
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from influence_visualizer.data_loader import (
    load_demo_episodes_from_checkpoint,
    load_influence_matrix,
    load_rollout_episodes,
)


def render_trajectory_influence_heatmap(
    influence_matrix: np.ndarray,
    output_path: pathlib.Path,
    rollout_idx: int,
    demo_idx: int,
    rollout_success: bool,
    cmap: str = "RdBu_r",
    figsize: tuple = (8, 6),
    dpi: int = 100,
) -> None:
    """Render a single trajectory influence matrix as a heatmap.

    Args:
        influence_matrix: Shape (rollout_len, demo_len)
        output_path: Path to save the heatmap image
        rollout_idx: Rollout episode index
        demo_idx: Demonstration episode index
        rollout_success: Whether the rollout succeeded
        cmap: Matplotlib colormap name
        figsize: Figure size (width, height) in inches
        dpi: Image resolution
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute symmetric colorbar limits
    abs_max = max(abs(np.nanmin(influence_matrix)), abs(np.nanmax(influence_matrix)))
    vmin, vmax = -abs_max, abs_max

    # Create heatmap
    im = ax.imshow(
        influence_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Influence Score")

    # Labels and title
    ax.set_xlabel("Demonstration Timestep")
    ax.set_ylabel("Rollout Timestep")

    success_str = "Success" if rollout_success else "Failure"
    title = f"Rollout {rollout_idx} vs Demo {demo_idx} ({success_str})"
    ax.set_title(title)

    # Add statistics annotation
    stats_text = (
        f"Shape: {influence_matrix.shape}\n"
        f"Min: {np.nanmin(influence_matrix):.4f}\n"
        f"Max: {np.nanmax(influence_matrix):.4f}\n"
        f"Mean: {np.nanmean(influence_matrix):.4f}\n"
        f"Std: {np.nanstd(influence_matrix):.4f}"
    )
    ax.text(
        1.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def compute_trajectory_influences(
    action_influence_matrix: np.ndarray,
    test_ep_idxs: list,
    train_ep_idxs: list,
    test_ep_lens: np.ndarray,
    train_ep_lens: np.ndarray,
    success_mask: np.ndarray,
    output_dir: pathlib.Path,
    cmap: str = "RdBu_r",
    figsize: tuple = (8, 6),
    dpi: int = 100,
) -> None:
    """Extract and render individual trajectory influence matrices as heatmaps.

    For each (rollout_i, demo_j) pair, extracts the submatrix
    action_influence_matrix[test_idx[i], train_idx[j]] and renders it as a heatmap.

    Args:
        action_influence_matrix: Shape (num_rollout_samples, num_demo_samples)
        test_ep_idxs: List of arrays, each containing sample indices for a rollout episode
        train_ep_idxs: List of arrays, each containing sample indices for a demo episode
        test_ep_lens: Array of episode lengths for rollouts
        train_ep_lens: Array of episode lengths for demonstrations
        success_mask: Boolean array indicating which rollouts succeeded
        output_dir: Directory to save trajectory influence heatmaps
        cmap: Matplotlib colormap name
        figsize: Figure size (width, height) in inches
        dpi: Image resolution
    """
    num_test_eps = len(test_ep_idxs)
    num_train_eps = len(train_ep_idxs)

    # Verify matrix shape matches episode metadata
    assert action_influence_matrix.shape[0] == test_ep_lens.sum(), (
        f"Rollout dimension mismatch: matrix has {action_influence_matrix.shape[0]} rows "
        f"but metadata has {test_ep_lens.sum()} samples"
    )
    assert action_influence_matrix.shape[1] == train_ep_lens.sum(), (
        f"Demo dimension mismatch: matrix has {action_influence_matrix.shape[1]} cols "
        f"but metadata has {train_ep_lens.sum()} samples"
    )

    print(f"\nRendering trajectory influence heatmaps...")
    print(f"  Rollout episodes: {num_test_eps}")
    print(f"  Demo episodes: {num_train_eps}")
    print(f"  Total pairs: {num_test_eps * num_train_eps}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract and render each (rollout_i, demo_j) influence matrix
    for i, test_idx in enumerate(test_ep_idxs):
        if i % 10 == 0:
            print(f"  Processing rollout {i}/{num_test_eps}...")

        for j, train_idx in enumerate(train_ep_idxs):
            # Extract influence submatrix for this rollout-demo pair
            # Use np.ix_ to index both dimensions simultaneously
            influence_ij = action_influence_matrix[np.ix_(test_idx, train_idx)]

            # Verify shape
            assert influence_ij.shape == (test_ep_lens[i], train_ep_lens[j]), (
                f"Shape mismatch for rollout {i}, demo {j}: "
                f"expected {(test_ep_lens[i], train_ep_lens[j])}, got {influence_ij.shape}"
            )

            # Render and save heatmap
            output_path = output_dir / f"rollout_{i:04d}_demo_{j:04d}.png"
            render_trajectory_influence_heatmap(
                influence_matrix=influence_ij,
                output_path=output_path,
                rollout_idx=i,
                demo_idx=j,
                rollout_success=success_mask[i],
                cmap=cmap,
                figsize=figsize,
                dpi=dpi,
            )

    print(f"  Saved {num_test_eps * num_train_eps} trajectory influence heatmaps")


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory-level influences for each rollout and demonstration"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing rollout episodes and TRAK results",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Training directory containing .hydra/config.yaml",
    )
    parser.add_argument(
        "--train_ckpt",
        type=str,
        default=None,
        help="Specific checkpoint to use (e.g., 'epoch=0100-test_mean_score=1.000'). If not specified, uses 'latest'.",
    )
    parser.add_argument(
        "--demo_video_dir",
        type=str,
        required=True,
        help="Directory containing rendered demonstration videos",
    )
    parser.add_argument(
        "--include_holdout",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to include holdout set in demonstrations (True/False)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trajectory_influences",
        help="Output directory for trajectory influence heatmaps",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Matplotlib colormap (e.g., 'RdBu_r', 'viridis', 'coolwarm')",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8, 6],
        help="Figure size (width height) in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Image resolution (dots per inch)",
    )

    args = parser.parse_args()

    eval_dir = pathlib.Path(args.eval_dir)
    train_dir = pathlib.Path(args.train_dir)
    output_dir = eval_dir / args.output_dir

    print("=" * 80)
    print("Computing Trajectory Influences")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  eval_dir: {eval_dir}")
    print(f"  train_dir: {train_dir}")
    print(f"  train_ckpt: {args.train_ckpt or 'latest'}")
    print(f"  include_holdout: {args.include_holdout}")
    print(f"  output_dir: {output_dir}")

    # Load rollout episodes
    print(f"\nLoading rollout episodes from {eval_dir / 'episodes'}...")
    rollout_episodes, _, _, _ = load_rollout_episodes(eval_dir)
    print(f"  Loaded {len(rollout_episodes)} rollout episodes")
    print(f"  Total rollout samples: {sum(ep.num_samples for ep in rollout_episodes)}")

    # Load demonstration episodes
    print(f"\nLoading demonstration episodes from checkpoint...")

    # Construct checkpoint path
    checkpoint_dir = train_dir / "checkpoints"
    if args.train_ckpt:
        checkpoint_path = checkpoint_dir / f"{args.train_ckpt}.ckpt"
    else:
        checkpoint_path = checkpoint_dir / "latest.ckpt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    demo_video_dir = pathlib.Path(args.demo_video_dir)
    if not demo_video_dir.exists():
        raise FileNotFoundError(f"Demo video directory not found: {demo_video_dir}")

    demo_episodes, _, _, train_set_size_from_ckpt, holdout_set_size, horizon = (
        load_demo_episodes_from_checkpoint(
            checkpoint_path=checkpoint_path,
            demo_video_dir=demo_video_dir,
            include_holdout=args.include_holdout,
        )
    )
    print(f"  Loaded {len(demo_episodes)} demonstration episodes")
    print(f"  Train set size: {train_set_size_from_ckpt}")
    print(f"  Holdout set size: {holdout_set_size}")
    print(f"  Total demo samples: {sum(ep.num_samples for ep in demo_episodes)}")

    # Load action influence matrix
    print(f"\nLoading action influence matrix...")
    action_influence_matrix, train_set_size, test_set_size = load_influence_matrix(
        eval_dir=eval_dir,
    )
    print(f"  Matrix shape: {action_influence_matrix.shape}")
    print(f"  Train set size: {train_set_size}")
    print(f"  Test set size: {test_set_size}")

    # Verify matrix shape matches loaded episodes
    num_rollout_samples = sum(ep.num_samples for ep in rollout_episodes)
    num_demo_samples = sum(ep.num_samples for ep in demo_episodes)

    if action_influence_matrix.shape[0] != num_rollout_samples:
        raise ValueError(
            f"Rollout sample count mismatch: matrix has {action_influence_matrix.shape[0]} rows "
            f"but loaded {num_rollout_samples} rollout samples"
        )
    if action_influence_matrix.shape[1] != num_demo_samples:
        raise ValueError(
            f"Demo sample count mismatch: matrix has {action_influence_matrix.shape[1]} cols "
            f"but loaded {num_demo_samples} demo samples"
        )

    # Prepare episode metadata (both are now EpisodeInfo objects)
    test_ep_lens = np.array([ep.num_samples for ep in rollout_episodes], dtype=np.int64)
    test_ep_idxs = [
        np.arange(ep.sample_start_idx, ep.sample_end_idx) for ep in rollout_episodes
    ]
    success_mask = np.array([ep.success for ep in rollout_episodes], dtype=bool)

    train_ep_lens = np.array([ep.num_samples for ep in demo_episodes], dtype=np.int64)
    train_ep_idxs = [
        np.arange(ep.sample_start_idx, ep.sample_end_idx) for ep in demo_episodes
    ]

    # Compute and save trajectory influences
    compute_trajectory_influences(
        action_influence_matrix=action_influence_matrix,
        test_ep_idxs=test_ep_idxs,
        train_ep_idxs=train_ep_idxs,
        test_ep_lens=test_ep_lens,
        train_ep_lens=train_ep_lens,
        success_mask=success_mask,
        output_dir=output_dir,
        cmap=args.cmap,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )

    # Save metadata
    metadata_path = output_dir / "metadata.npz"
    print(f"\nSaving metadata to {metadata_path}...")
    np.savez(
        metadata_path,
        test_ep_lens=test_ep_lens,
        train_ep_lens=train_ep_lens,
        success_mask=success_mask,
        num_test_eps=len(rollout_episodes),
        num_train_eps=len(demo_episodes),
    )

    print(f"\n{'=' * 80}")
    print("Trajectory influence rendering complete!")
    print(f"{'=' * 80}")
    print(f"\nOutput files:")
    print(f"  Influence heatmaps: {output_dir}/rollout_<i>_demo_<j>.png")
    print(f"  Metadata: {metadata_path}")
    print(f"\nExample heatmap: {output_dir}/rollout_0000_demo_0000.png")
    print(f"\nTo load metadata:")
    print(f"  metadata = np.load('{metadata_path}')")
    print(f"  test_ep_lens = metadata['test_ep_lens']")
    print(f"  train_ep_lens = metadata['train_ep_lens']")
    print(f"  success_mask = metadata['success_mask']")


if __name__ == "__main__":
    main()

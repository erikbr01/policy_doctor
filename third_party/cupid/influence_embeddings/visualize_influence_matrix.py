"""Visualize the influence matrix as a heatmap.

This script can visualize two types of influence matrices:

1. Action Influence Matrix (default): State-action level influence
   - Relates each state-action pair of rollouts (rows) to each state-action
     pair of demonstrations (columns)
   - Shape: (num_rollout_samples, num_demo_samples)

2. Performance Influence Matrix (--performance_influence flag): Trajectory-level influence
   - Relates each rollout trajectory to each demonstration trajectory
   - This is the core CUPID metric used for data curation
   - Shape: (num_rollout_trajectories, num_demo_trajectories)
   - Computed by aggregating action influence using various aggregation functions

Usage (Action Influence):
    python influence_embeddings/visualize_influence_matrix.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/train_name/latest \
        --exp_date 25.01.17 \
        --train_set_size 1000 \
        --test_set_size 500 \
        --output_path action_influence_heatmap.png

Usage (Performance Influence):
    python influence_embeddings/visualize_influence_matrix.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/train_name/latest \
        --exp_date 25.01.17 \
        --performance_influence \
        --train_dir data/outputs/train/jan17/train_name \
        --aggr_fn mean_of_mean \
        --output_path influence_heatmap.png

    Note: --train_set_size and --test_set_size are auto-determined from metadata.
    Generates both action and performance influence heatmaps.
"""

import argparse
import pathlib
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from diffusion_policy.common import error_util, trak_util
from diffusion_policy.common.results_util import LOAD_TRAK_KWARGS, get_trak_scores
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset


def compute_performance_influence(
    action_influence_matrix: np.ndarray,
    test_ep_idxs: list,
    train_ep_idxs: list,
    test_ep_lens: np.ndarray,
    train_ep_lens: np.ndarray,
    success_mask: np.ndarray,
    aggr_fn: str = "mean_of_mean",
    return_dtype: type = np.float32,
) -> np.ndarray:
    """Compute performance influence matrix from action influence matrix.

    Aggregates state-action level influence to trajectory-level influence.
    This is the core CUPID computation that links demonstration trajectories
    to rollout trajectories.

    Args:
        action_influence_matrix: State-action level influence matrix
            of shape (num_rollout_samples, num_demo_samples)
        test_ep_idxs: List of arrays, each containing sample indices for a rollout episode
        train_ep_idxs: List of arrays, each containing sample indices for a demo episode
        test_ep_lens: Array of episode lengths for rollouts
        train_ep_lens: Array of episode lengths for demonstrations
        success_mask: Boolean array indicating which rollouts succeeded
        aggr_fn: Aggregation function name (e.g., "mean_of_mean", "sum_of_sum")
        return_dtype: Return data type

    Returns:
        Performance influence matrix of shape (num_rollout_eps, num_demo_eps)
    """
    # Map aggregation function name to actual function
    aggr_fn_map = {
        "mean_of_mean": error_util.mean_of_mean_influence,
        "mean_of_mean_success": error_util.mean_of_mean_influence_success,
        "sum_of_sum": error_util.sum_of_sum_influence,
        "sum_of_sum_success": error_util.sum_of_sum_influence_success,
        "min_of_max": error_util.min_of_max_influence,
        "max_of_min": error_util.max_of_min_influence,
    }

    if aggr_fn not in aggr_fn_map:
        raise ValueError(f"Unknown aggregation function: {aggr_fn}")

    performance_influence = error_util.pairwise_sample_to_trajectory_scores(
        pairwise_sample_scores=action_influence_matrix,
        num_test_eps=len(test_ep_idxs),
        num_train_eps=len(train_ep_idxs),
        test_ep_idxs=test_ep_idxs,
        train_ep_idxs=train_ep_idxs,
        test_ep_lens=test_ep_lens,
        train_ep_lens=train_ep_lens,
        success_mask=success_mask,
        aggr_fn=aggr_fn_map[aggr_fn],
        return_dtype=return_dtype,
    )

    return performance_influence


def load_episode_metadata(eval_dir: pathlib.Path, train_dir: pathlib.Path = None):
    """Load episode metadata for rollouts and optionally for demonstrations.

    Args:
        eval_dir: Directory containing rollout episodes
        train_dir: Optional path to training directory (containing .hydra/config.yaml)

    Returns:
        Dictionary with rollout metadata and optionally demo metadata
    """
    # Load rollout (test) metadata from episodes directory
    test_dataset = BatchEpisodeDataset(
        batch_size=1,
        dataset_path=eval_dir / "episodes",
        exec_horizon=1,
        sample_history=0,
    )

    test_metadata = {
        "success_mask": np.array(test_dataset.episode_successes, dtype=bool),
        "ep_lens": np.array(test_dataset.episode_lengths),
        "ep_idxs": test_dataset.episode_idxs,
        "num_eps": len(test_dataset.episode_idxs),
        "num_samples": len(test_dataset),
    }

    metadata = {"test": test_metadata}

    # Load demonstration metadata if train_dir is provided
    if train_dir is not None:
        config_path = train_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config
        cfg = OmegaConf.load(config_path)

        # Load training dataset and holdout dataset
        # Note: TRAK influence is computed on train + holdout combined
        train_dataset = hydra.utils.instantiate(cfg.task.dataset)
        holdout_dataset = train_dataset.get_holdout_dataset()

        # Get metadata using trak_util
        train_metadata = trak_util.get_dataset_metadata(cfg, train_dataset)
        holdout_metadata = trak_util.get_dataset_metadata(cfg, holdout_dataset)

        # Combine train and holdout metadata
        # The TRAK scores are computed on [train_samples, holdout_samples] concatenated
        combined_metadata = {
            "ep_lens": np.concatenate(
                [train_metadata["ep_lens"], holdout_metadata["ep_lens"]]
            ),
            "ep_idxs": train_metadata["ep_idxs"] + holdout_metadata["ep_idxs"],
            "num_eps": train_metadata["num_eps"] + holdout_metadata["num_eps"],
            "num_samples": train_metadata["num_samples"]
            + holdout_metadata["num_samples"],
        }

        metadata["train"] = combined_metadata
        metadata["train_only"] = train_metadata
        metadata["holdout"] = holdout_metadata

    return metadata


def visualize_influence_matrix(
    influence_matrix: np.ndarray,
    output_path: pathlib.Path,
    title: str = "Influence Matrix (Rollouts vs Demonstrations)",
    cmap: str = "RdBu_r",
    figsize: tuple = (12, 8),
    vmin: float = None,
    vmax: float = None,
    symmetric_colorbar: bool = True,
    subsample_rows: int = None,
    subsample_cols: int = None,
    xlabel: str = "Demonstration Sample Index",
    ylabel: str = "Rollout Sample Index",
) -> None:
    """Create and save a heatmap visualization of the influence matrix.

    Args:
        influence_matrix: Array of shape (num_rollout_samples, num_demo_samples)
        output_path: Path to save the output figure
        title: Title for the plot
        cmap: Matplotlib colormap name
        figsize: Figure size (width, height) in inches
        vmin: Minimum value for colorbar (None for auto)
        vmax: Maximum value for colorbar (None for auto)
        symmetric_colorbar: If True, center colorbar at 0
        subsample_rows: If set, subsample to this many rows for visualization
        subsample_cols: If set, subsample to this many columns for visualization
        xlabel: Label for x-axis
        ylabel: Label for y-axis
    """
    matrix = influence_matrix.copy()

    # Subsample if matrices are too large
    if subsample_rows is not None and matrix.shape[0] > subsample_rows:
        row_indices = np.linspace(0, matrix.shape[0] - 1, subsample_rows, dtype=int)
        matrix = matrix[row_indices, :]
        print(f"Subsampled rows: {influence_matrix.shape[0]} -> {subsample_rows}")

    if subsample_cols is not None and matrix.shape[1] > subsample_cols:
        col_indices = np.linspace(0, matrix.shape[1] - 1, subsample_cols, dtype=int)
        matrix = matrix[:, col_indices]
        print(f"Subsampled cols: {influence_matrix.shape[1]} -> {subsample_cols}")

    # Compute colorbar limits
    if symmetric_colorbar and vmin is None and vmax is None:
        abs_max = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        vmin, vmax = -abs_max, abs_max

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Influence Score")

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize influence matrix as a heatmap"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing TRAK results (e.g., data/outputs/eval_save_episodes/...)",
    )
    parser.add_argument(
        "--exp_date",
        type=str,
        required=True,
        help="Experiment date prefix (e.g., '25.01.17')",
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        default=None,
        help="Number of training (demonstration) samples (auto-determined from --train_dir if not provided)",
    )
    parser.add_argument(
        "--test_set_size",
        type=int,
        default=None,
        help="Number of test (rollout) samples (auto-determined from eval_dir if not provided)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="influence_heatmap.png",
        help="Output path for the heatmap image",
    )
    parser.add_argument(
        "--load_key",
        type=str,
        default="default_diffusion",
        choices=list(LOAD_TRAK_KWARGS.keys()),
        help="TRAK loading configuration key",
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
        default=[12, 8],
        help="Figure size (width height) in inches",
    )
    parser.add_argument(
        "--subsample_rows",
        type=int,
        default=None,
        help="Subsample to this many rows for visualization (default: no subsampling)",
    )
    parser.add_argument(
        "--subsample_cols",
        type=int,
        default=None,
        help="Subsample to this many columns for visualization (default: no subsampling)",
    )
    parser.add_argument(
        "--no_symmetric_colorbar",
        action="store_true",
        help="Disable symmetric colorbar centering at 0",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Influence Matrix (Rollouts vs Demonstrations)",
        help="Title for the heatmap",
    )
    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="Skip debug checks on intermediate TRAK files",
    )
    parser.add_argument(
        "--performance_influence",
        action="store_true",
        help="Compute and visualize performance influence (trajectory-level) instead of action influence (sample-level)",
    )
    parser.add_argument(
        "--aggr_fn",
        type=str,
        default="mean_of_mean",
        choices=[
            "mean_of_mean",
            "mean_of_mean_success",
            "sum_of_sum",
            "sum_of_sum_success",
            "min_of_max",
            "max_of_min",
        ],
        help="Aggregation function for computing performance influence",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Training directory containing .hydra/config.yaml (required for --performance_influence)",
    )

    args = parser.parse_args()

    # Auto-determine train_set_size and test_set_size if needed
    if args.train_set_size is None or args.test_set_size is None:
        print("Auto-determining dataset sizes from metadata...")

        # Load metadata to determine sizes
        if args.performance_influence and args.train_dir is not None:
            temp_metadata = load_episode_metadata(
                eval_dir=pathlib.Path(args.eval_dir),
                train_dir=pathlib.Path(args.train_dir),
            )
            if args.train_set_size is None:
                # TRAK is computed on train + holdout
                args.train_set_size = (
                    temp_metadata["train_only"]["num_samples"]
                    + temp_metadata["holdout"]["num_samples"]
                )
                print(f"  Auto-determined train_set_size: {args.train_set_size}")
            if args.test_set_size is None:
                args.test_set_size = temp_metadata["test"]["num_samples"]
                print(f"  Auto-determined test_set_size: {args.test_set_size}")
        else:
            # Can only auto-determine test_set_size without train_dir
            temp_metadata = load_episode_metadata(eval_dir=pathlib.Path(args.eval_dir))
            if args.test_set_size is None:
                args.test_set_size = temp_metadata["test"]["num_samples"]
                print(f"  Auto-determined test_set_size: {args.test_set_size}")
            if args.train_set_size is None:
                raise ValueError(
                    "Cannot auto-determine train_set_size without --train_dir. "
                    "Please provide either --train_set_size or --train_dir."
                )

    # Load influence matrix
    print(f"\nLoading influence matrix from: {args.eval_dir}")
    print(f"  exp_date: {args.exp_date}")
    print(f"  load_key: {args.load_key}")
    print(f"  train_set_size: {args.train_set_size}")
    print(f"  test_set_size: {args.test_set_size}")

    trak_kwargs = LOAD_TRAK_KWARGS[args.load_key].copy()
    influence_matrix = get_trak_scores(
        eval_dir=pathlib.Path(args.eval_dir),
        train_set_size=args.train_set_size,
        test_set_size=args.test_set_size,
        exp_date=args.exp_date,
        debug=not args.no_debug,
        **trak_kwargs,
    )

    print(f"Loaded action influence matrix with shape: {influence_matrix.shape}")
    print(f"  (rows=rollout samples, cols=demonstration samples)")

    # Compute performance influence if requested
    if args.performance_influence:
        print("\nComputing performance influence (trajectory-level)...")

        # Load demo metadata from train_dir
        if args.train_dir is not None:
            print(f"Loading demonstration metadata from: {args.train_dir}")
            metadata = load_episode_metadata(
                eval_dir=pathlib.Path(args.eval_dir),
                train_dir=pathlib.Path(args.train_dir),
            )
            test_metadata = metadata["test"]

            # Print dataset info
            if "train_only" in metadata and "holdout" in metadata:
                print(
                    f"  Train episodes: {metadata['train_only']['num_eps']} ({metadata['train_only']['num_samples']} samples)"
                )
                print(
                    f"  Holdout episodes: {metadata['holdout']['num_eps']} ({metadata['holdout']['num_samples']} samples)"
                )
                total_samples = (
                    metadata["train_only"]["num_samples"]
                    + metadata["holdout"]["num_samples"]
                )
                print(f"  Total: {total_samples} samples")

            # Following eval_demonstration_scores.py:online_trak_influence_routine pattern:
            # Use train_only metadata and slice the influence matrix to match
            train_metadata = metadata["train_only"]
            train_ep_lens = train_metadata["ep_lens"]
            train_ep_idxs = train_metadata["ep_idxs"]
            train_num_eps = train_metadata["num_eps"]

            # Slice influence matrix to only use train portion (exclude holdout)
            # The TRAK matrix has columns [train_samples | holdout_samples]
            print(f"\nSlicing influence matrix to train portion...")
            print(f"  Original matrix shape: {influence_matrix.shape}")
            print(f"  Using train columns [:, :{train_metadata['num_samples']}]")
            influence_matrix = influence_matrix[:, : train_metadata["num_samples"]]
            print(f"  Sliced matrix shape: {influence_matrix.shape}")

        else:
            raise ValueError(
                "For performance influence visualization, you must provide --train_dir"
            )

        print(f"  Rollout episodes: {test_metadata['num_eps']}")
        print(f"  Demo episodes: {train_num_eps}")
        print(f"  Success rate: {test_metadata['success_mask'].mean():.2%}")
        print(f"  Aggregation function: {args.aggr_fn}")

        # Compute performance influence
        performance_influence = compute_performance_influence(
            action_influence_matrix=influence_matrix,
            test_ep_idxs=test_metadata["ep_idxs"],
            train_ep_idxs=train_ep_idxs,
            test_ep_lens=test_metadata["ep_lens"],
            train_ep_lens=train_ep_lens,
            success_mask=test_metadata["success_mask"],
            aggr_fn=args.aggr_fn,
            return_dtype=np.float32,
        )

        print(
            f"Computed performance influence matrix with shape: {performance_influence.shape}"
        )
        print(f"  (rows=rollout trajectories, cols=demonstration trajectories)")

        # Visualize both action and performance influence
        output_path = pathlib.Path(args.output_path)

        # 1. Visualize action influence (sliced to train set)
        print(f"\nGenerating action influence heatmap...")
        action_output_path = (
            output_path.parent / f"{output_path.stem}_action{output_path.suffix}"
        )
        visualize_influence_matrix(
            influence_matrix=influence_matrix,
            output_path=action_output_path,
            title="Action Influence Matrix - Train Set"
            if args.title == "Influence Matrix (Rollouts vs Demonstrations)"
            else args.title,
            cmap=args.cmap,
            figsize=tuple(args.figsize),
            symmetric_colorbar=not args.no_symmetric_colorbar,
            subsample_rows=args.subsample_rows,
            subsample_cols=args.subsample_cols,
            xlabel="Demonstration Sample Index (Train Set)",
            ylabel="Rollout Sample Index",
        )

        # 2. Visualize performance influence
        print(f"\nGenerating performance influence heatmap...")
        performance_output_path = (
            output_path.parent / f"{output_path.stem}_performance{output_path.suffix}"
        )
        visualize_influence_matrix(
            influence_matrix=performance_influence,
            output_path=performance_output_path,
            title=f"Performance Influence Matrix - Train Set (aggr_fn={args.aggr_fn})"
            if args.title == "Influence Matrix (Rollouts vs Demonstrations)"
            else args.title,
            cmap=args.cmap,
            figsize=tuple(args.figsize),
            symmetric_colorbar=not args.no_symmetric_colorbar,
            subsample_rows=args.subsample_rows,
            subsample_cols=args.subsample_cols,
            xlabel="Demonstration Trajectory Index (Train Set)",
            ylabel="Rollout Trajectory Index",
        )

        print(f"\nGenerated both visualizations:")
        print(f"  Action influence: {action_output_path}")
        print(f"  Performance influence: {performance_output_path}")

        # Don't run the single visualization at the end
        return

    # Visualize action influence only (when --performance_influence is not set)
    visualize_influence_matrix(
        influence_matrix=influence_matrix,
        output_path=pathlib.Path(args.output_path),
        title=args.title,
        cmap=args.cmap,
        figsize=tuple(args.figsize),
        symmetric_colorbar=not args.no_symmetric_colorbar,
        subsample_rows=args.subsample_rows,
        subsample_cols=args.subsample_cols,
        xlabel="Demonstration Sample Index",
        ylabel="Rollout Sample Index",
    )


if __name__ == "__main__":
    """
    Example 1: Visualize action influence (state-action level):
    python influence_embeddings/visualize_influence_matrix.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
        --exp_date default \
        --train_set_size 27859 \
        --test_set_size 1242 \
        --output_path action_influence_heatmap.png \
        --no_debug

    Example 2: Visualize performance influence (generates both action and performance heatmaps):
    python influence_embeddings/visualize_influence_matrix.py \
        --eval_dir data/outputs/eval_save_episodes/jan17/jan16_train_diffusion_unet_lowdim_lift_mh_0/latest \
        --exp_date default \
        --performance_influence \
        --train_dir data/outputs/train/jan16/jan16_train_diffusion_unet_lowdim_lift_mh_0 \
        --aggr_fn mean_of_mean \
        --output_path influence_heatmap.png \
        --no_debug

    Note: --train_set_size and --test_set_size are auto-determined from metadata when using --train_dir.
    Performance influence mode generates two files: influence_heatmap_action.png and influence_heatmap_performance.png
    """
    main()

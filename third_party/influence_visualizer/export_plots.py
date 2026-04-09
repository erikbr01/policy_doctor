#!/usr/bin/env python3
"""Export all visualizations from the influence visualizer to PNG files.

This script generates all plots from the influence visualizer app for a given
configuration and organizes them into directories matching the app's tab structure.

Usage:
    python export_plots.py --config mock
    python export_plots.py --config pusht --output output/pusht_export --demo-split train
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.io as pio
from sklearn.manifold import TSNE
from tqdm import tqdm

from influence_visualizer.config import load_config
from influence_visualizer.data_loader import InfluenceDataLoader
from influence_visualizer.render_annotation import load_annotations
from influence_visualizer.render_behaviors import (
    compute_behavior_influence,
    get_behavior_statistics,
)
from influence_visualizer.render_clustering import (
    extract_demo_embeddings,
    extract_matrix_pair_embeddings,
    extract_rollout_embeddings,
)
from influence_visualizer.render_heatmaps import (
    compute_performance_influence,
    compute_trajectory_influence_matrix,
    compute_transition_level_statistics,
)
from influence_visualizer.render_local_behaviors import (
    extract_influence_matrix_embeddings,
    visualize_action_histograms,
    visualize_influence_matrix_grid,
    visualize_state_action_tsne,
    visualize_state_histograms,
)
from influence_visualizer.render_advanced_analysis import (
    analyze_failure_modes,
    analyze_local_structure,
)
from influence_visualizer import plotting


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Global list to collect figures for batch export
_figure_queue = []


def queue_figure(fig, filepath: Path):
    """Queue a figure for batch export."""
    _figure_queue.append((fig, filepath))


def flush_figures(verbose: bool = True):
    """Export all queued figures using batch write_images for performance."""
    global _figure_queue
    
    if not _figure_queue:
        return
    
    figures = [fig for fig, _ in _figure_queue]
    paths = [str(path) for _, path in _figure_queue]
    
    try:
        # Batch export all figures at once (much faster!)
        pio.write_images(figures, paths, format='png')
        if verbose:
            for _, path in _figure_queue:
                print(f"  ✓ Saved: {path.name}")
    except ValueError as e:
        # Kaleido/Chrome not available
        if "Kaleido" in str(e) or "Chrome" in str(e):
            if verbose:
                print(f"  ⚠ Skipped {len(_figure_queue)} plots (Kaleido requires Chrome)")
        else:
            if verbose:
                for _, path in _figure_queue:
                    print(f"  ✗ Error saving {path.name}: {e}")
    except Exception as e:
        if verbose:
            for _, path in _figure_queue:
                print(f"  ✗ Error saving {path.name}: {e}")
    finally:
        _figure_queue = []


def export_aggregated_influence(
    data,
    output_dir: Path,
    demo_split: str,
    annotation_file: str,
):
    """Export all plots from Aggregated Influence tab."""
    print("\n📊 Exporting Aggregated Influence visualizations...")
    tab_dir = ensure_dir(output_dir / "1_Aggregated_Influence")

    # Trajectory influence matrix
    print("  Computing trajectory influence matrix...")
    traj_matrix, demo_episodes = compute_trajectory_influence_matrix(data, split=demo_split)
    
    # Create labels
    x_labels = [f"D{i}" for i in range(traj_matrix.shape[1])]
    y_labels = [f"R{i}" for i in range(traj_matrix.shape[0])]
    
    fig = plotting.create_influence_heatmap(
        traj_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title="Trajectory-wise Influence Matrix",
        x_title="Demo Episode",
        y_title="Rollout Episode",
    )
    queue_figure(fig, tab_dir / "trajectory_influence_heatmap.png")

    # Performance influence
    print("  Computing performance influence...")
    perf_scores, demo_episodes = compute_performance_influence(
        data, split=demo_split, metric="net"
    )
    
    # Create labels for demos
    demo_labels = [f"Demo {ep.index}" for ep in demo_episodes]
    
    fig = plotting.create_performance_influence_bar_plot(
        perf_scores,
        labels=demo_labels,
        title="Performance Influence per Demonstration (Net Score)",
    )
    queue_figure(fig, tab_dir / "performance_influence_bar.png")

    # Transition statistics
    print("  Computing transition-level statistics...")
    stats_array, raw_metadata = compute_transition_level_statistics(data, split=demo_split)
    
    # Convert metadata to proper format for plotting
    metadata = []
    for idx, (rollout_idx, demo_idx, quality_label, success) in enumerate(raw_metadata):
        metadata.append({
            "rollout_idx": rollout_idx,
            "demo_idx": demo_idx,
            "quality": quality_label,
            "success": success,
            "success_str": "Success" if success else "Failure",
            "mean": stats_array[idx, 0],
            "std": stats_array[idx, 1],
            "min": stats_array[idx, 2],
            "max": stats_array[idx, 3],
        })
    
    fig = plotting.create_transition_statistics_scatter(
        x_values=stats_array[:, 0],  # mean
        y_values=stats_array[:, 1],  # std
        x_stat_name="mean",
        y_stat_name="std",
        metadata=metadata,
        title="Transition-Level Influence Statistics",
    )
    queue_figure(fig, tab_dir / "transition_statistics_scatter.png")

    # Influence distribution comparison
    print("  Computing influence distributions...")
    # Get all influence values for successful vs failed rollouts
    success_mask = np.array([ep.success if ep.success is not None else False for ep in data.rollout_episodes])
    
    # Get rollout episode ranges
    rollout_sample_idxs = []
    for ep in data.rollout_episodes:
        rollout_sample_idxs.extend(range(ep.sample_start_idx, ep.sample_end_idx))
    
    if demo_split == "train":
        influence_matrix = data.influence_matrix[:, : len(data.demo_sample_infos)]
    elif demo_split == "holdout":
        influence_matrix = data.influence_matrix[:, len(data.demo_sample_infos) :]
    else:  # both
        influence_matrix = data.influence_matrix

    # Collect influences for successful and failed episodes
    success_influences = []
    failure_influences = []
    for i, ep in enumerate(data.rollout_episodes):
        ep_influence = influence_matrix[ep.sample_start_idx : ep.sample_end_idx]
        if success_mask[i]:
            success_influences.append(ep_influence.flatten())
        else:
            failure_influences.append(ep_influence.flatten())
    
    if success_influences:
        success_influences = np.concatenate(success_influences)
    else:
        success_influences = np.array([])
    
    if failure_influences:
        failure_influences = np.concatenate(failure_influences)
    else:
        failure_influences = np.array([])

    # Compute histograms for both distributions
    if len(success_influences) > 0 and len(failure_influences) > 0:
        num_bins = 50
        all_values = np.concatenate([success_influences, failure_influences])
        bin_edges = np.histogram_bin_edges(all_values, bins=num_bins)
        success_counts, _ = np.histogram(success_influences, bins=bin_edges)
        failure_counts, _ = np.histogram(failure_influences, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        fig = plotting.create_distribution_comparison_plot(
            success_counts,
            failure_counts,
            bin_centers,
            bin_width,
            title="Influence Distribution: Success vs. Failure",
        )
        queue_figure(fig, tab_dir / "influence_distribution_comparison.png")

    # Transition density plots
    print("  Creating transition density plots...")
    # Extract std values for successful vs failed rollouts
    success_stds = []
    failure_stds = []
    for i, meta in enumerate(metadata):
        if meta["success"]:  # Use dict access instead of index
            success_stds.append(stats_array[i, 1])  # std is column 1
        else:
            failure_stds.append(stats_array[i, 1])
    
    fig = plotting.create_density_plots(
        np.array(success_stds),
        np.array(failure_stds),
        stat_name="Influence Std",
        plot_type="histogram",
        title="Transition Statistics Density: Success vs. Failure",
    )
    queue_figure(fig, tab_dir / "transition_density_plots.png")

    print(f"✓ Completed Aggregated Influence exports")


def export_episode_influence(
    data,
    output_dir: Path,
    demo_split: str,
    annotation_file: str,
):
    """Export episode-level influence visualizations (skipping per-sample details)."""
    print("\n📊 Exporting Episode Influence visualizations...")
    tab_dir = ensure_dir(output_dir / "2_Episode_Influence")

    # Export only episode-level heatmaps and magnitude plots
    for ep_idx, episode in enumerate(tqdm(data.rollout_episodes, desc="  Episodes")):
        ep_dir = ensure_dir(tab_dir / f"Episode_{ep_idx}")

        # Influence heatmap
        # Get the influence matrix for this rollout episode
        if demo_split == "train":
            ep_influence = data.influence_matrix[
                episode.sample_start_idx : episode.sample_end_idx,
                : len(data.demo_sample_infos),
            ]
        elif demo_split == "holdout":
            ep_influence = data.influence_matrix[
                episode.sample_start_idx : episode.sample_end_idx,
                len(data.demo_sample_infos) :,
            ]
        else:  # both
            ep_influence = data.influence_matrix[
                episode.sample_start_idx : episode.sample_end_idx, :
            ]

        # Create labels for the heatmap
        x_labels = [f"t{i}" for i in range(ep_influence.shape[0])]
        y_labels = [f"s{i}" for i in range(ep_influence.shape[1])]
        
        fig = plotting.create_influence_heatmap(
            ep_influence.T,
            x_labels=x_labels,
            y_labels=y_labels,
            title=f"Episode {ep_idx} Influence Heatmap",
            x_title="Rollout Timestep",
            y_title="Demo Sample",
        )
        queue_figure(fig, ep_dir / "influence_heatmap.png")

        # Magnitude over time
        pos_magnitude = np.sum(np.maximum(ep_influence, 0), axis=1)
        neg_magnitude = np.sum(np.minimum(ep_influence, 0), axis=1)

        fig = plotting.create_magnitude_over_time_plot(
            pos_magnitude,
            neg_magnitude,
            title=f"Episode {ep_idx} Influence Magnitude Over Time",
        )
        queue_figure(fig, ep_dir / "magnitude_over_time.png")

    print(f"✓ Completed Episode Influence exports")


def export_behaviors(
    data,
    output_dir: Path,
    demo_split: str,
    top_k: int,
    obs_key: str,
    annotation_file: str,
):
    """Export all plots from Behaviors tab."""
    print("\n📊 Exporting Behaviors visualizations...")
    tab_dir = ensure_dir(output_dir / "3_Behaviors")

    # Load annotations
    annotations = load_annotations(annotation_file, task_config=args.config)
    behavior_stats = get_behavior_statistics(annotations, data.rollout_episodes)

    if not behavior_stats:
        print("  No behavior annotations found, skipping behaviors export")
        return

    # Behavior pie chart
    print("  Creating behavior distribution pie chart...")
    labels = list(behavior_stats.keys())
    values = [len(behavior_stats[label]) for label in labels]
    fig = plotting.create_behavior_pie_chart(labels, values)
    queue_figure(fig, tab_dir / "behavior_distribution.png")

    # For each behavior, create bar chart
    for label in tqdm(behavior_stats.keys(), desc="  Behaviors"):
        behavior_slices = behavior_stats[label]
        mean_influences, _ = compute_behavior_influence(
            data, behavior_slices, split=demo_split
        )

        # Bar chart
        sample_labels = [f"s{i}" for i in range(len(mean_influences))]
        fig = plotting.create_behavior_influence_bar_chart(
            sample_labels,
            mean_influences,
            title=f"Influence for '{label}' Behavior",
        )
        safe_label = label.replace(" ", "_").replace("/", "_")
        queue_figure(
            fig, tab_dir / f"{safe_label}_influence_bar.png"
        )

    print(f"✓ Completed Behaviors exports")
    # Export all queued figures
    print("  Saving plots...")
    flush_figures()



def export_clustering(
    data,
    output_dir: Path,
    demo_split: str,
    annotation_file: str,
):
    """Export all plots from Clustering tab."""
    print("\n📊 Exporting Clustering visualizations...")
    tab_dir = ensure_dir(output_dir / "4_Clustering")

    # Demo embeddings t-SNE
    print("  Computing demo embeddings...")
    demo_embeddings, demo_metadata = extract_demo_embeddings(data, split=demo_split)
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(demo_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(demo_embeddings)
    
    fig = plotting.create_generic_tsne_plot(
        embeddings_2d,
        demo_metadata,
        color_by="quality_label",
        title="Demo Embeddings (t-SNE)",
        categorical_color=True,
    )
    queue_figure(fig, tab_dir / "demo_embeddings_tsne.png")

    # Rollout embeddings t-SNE
    print("  Computing rollout embeddings...")
    rollout_embeddings, rollout_metadata = extract_rollout_embeddings(
        data, split=demo_split
    )
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(rollout_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(rollout_embeddings)
    
    fig = plotting.create_generic_tsne_plot(
        embeddings_2d,
        rollout_metadata,
        color_by="success",
        title="Rollout Embeddings (t-SNE)",
        categorical_color=True,
    )
    queue_figure(fig, tab_dir / "rollout_embeddings_tsne.png")

    # Matrix pair embeddings t-SNE
    print("  Computing matrix pair embeddings...")
    matrix_embeddings, matrix_metadata = extract_matrix_pair_embeddings(
        data, split=demo_split, n_components=10, method="singular_values"
    )
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(matrix_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(matrix_embeddings)
    
    fig = plotting.create_generic_tsne_plot(
        embeddings_2d,
        matrix_metadata,
        color_by="success",
        title="Matrix Pair Embeddings (t-SNE)",
        categorical_color=True,
    )
    queue_figure(fig, tab_dir / "matrix_pair_embeddings_tsne.png")

    print(f"✓ Completed Clustering exports")


def export_local_behaviors(
    data,
    output_dir: Path,
    demo_split: str,
    top_k: int,
    obs_key: str,
    annotation_file: str,
):
    """Export all plots from Local Behaviors tab."""
    print("\n📊 Exporting Local Behaviors visualizations...")
    tab_dir = ensure_dir(output_dir / "5_Local_Behaviors")

    # Aggregated influence grid (all rollouts)
    print("  Creating aggregated influence grid...")
    
    # Get influence matrices for each rollout (aggregated across demos)
    matrices = []
    titles = []
    max_rollouts = min(20, data.num_rollout_episodes)
    
    for ep_idx in range(max_rollouts):
        episode = data.rollout_episodes[ep_idx]
        if demo_split == "train":
            ep_influence = data.influence_matrix[
                episode.sample_start_idx : episode.sample_end_idx,
                : len(data.demo_sample_infos),
            ]
        elif demo_split == "holdout":
            ep_influence = data.influence_matrix[
                episode.sample_start_idx : episode.sample_end_idx,
                len(data.demo_sample_infos) :,
            ]
        else:  # both
            ep_influence = data.influence_matrix[
                episode.sample_start_idx : episode.sample_end_idx, :
            ]
        
        matrices.append(ep_influence)
        titles.append(f"Rollout {ep_idx}")
    
    fig = plotting.create_aggregated_influence_grid(
        matrices,
        titles,
        main_title="Aggregated Influence Heatmaps",
        rollouts_per_row=2,
    )
    queue_figure(fig, tab_dir / "aggregated_influence_grid.png")

    # Per-episode influence matrices and embeddings
    for ep_idx, episode in enumerate(
        tqdm(data.rollout_episodes, desc="  Episode matrices")
    ):
        ep_dir = ensure_dir(tab_dir / f"Episode_{ep_idx}")

        # Influence matrices grid - with pagination
        from influence_visualizer.render_heatmaps import get_split_data
        _, demo_episodes, _, _ = get_split_data(data, demo_split)
        total_demos = len(demo_episodes)
        
        demos_per_page = 12
        num_pages = (total_demos + demos_per_page - 1) // demos_per_page
        
        # Export one PNG per page
        for page_num in range(num_pages):
            start_demo = page_num * demos_per_page
            fig = visualize_influence_matrix_grid(
                data, rollout_idx=ep_idx, split=demo_split,
                max_demos=demos_per_page, start_demo=start_demo
            )
            if num_pages > 1:
                queue_figure(fig, ep_dir / f"influence_matrices_grid_page_{page_num + 1}_of_{num_pages}.png")
            else:
                queue_figure(fig, ep_dir / "influence_matrices_grid.png")

        # Matrix embeddings
        embeddings, metadata = extract_influence_matrix_embeddings(
            data, rollout_idx=ep_idx, split=demo_split, method="stats"
        )
        if len(embeddings) > 1:
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            fig = plotting.create_generic_tsne_plot(
                embeddings_2d,
                metadata,
                color_by="mean_influence",
                title=f"Episode {ep_idx} Matrix Embeddings (t-SNE)",
                categorical_color=False,
            )
            queue_figure(
                fig, ep_dir / "matrix_embeddings_tsne.png"
            )

    # State-action distribution analysis
    print("  Creating state-action t-SNE...")
    fig = visualize_state_action_tsne(data, split=demo_split, perplexity=30)
    queue_figure(fig, tab_dir / "state_action_tsne.png")

    print("  Creating state histograms...")
    fig = visualize_state_histograms(data, split=demo_split)
    queue_figure(fig, tab_dir / "state_histograms.png")

    print("  Creating action histograms...")
    fig = visualize_action_histograms(data, split=demo_split)
    queue_figure(fig, tab_dir / "action_histograms.png")

    # Export all queued figures
    print("  Saving plots...")
    flush_figures()

    print(f"✓ Completed Local Behaviors exports")


def export_advanced_analysis(
    data,
    output_dir: Path,
    demo_split: str,
):
    """Export all plots from Advanced Analysis tab."""
    print("\n📊 Exporting Advanced Analysis visualizations...")
    tab_dir = ensure_dir(output_dir / "6_Advanced_Analysis")

    # Compute transition statistics for analysis
    print("  Computing transition statistics for failure mode analysis...")
    stats_array, metadata = compute_transition_level_statistics(data, split=demo_split)

    # Create a minimal data structure for the analysis functions
    failure_data = {
        "stats_array": stats_array,
        "metadata": metadata,
    }

    # Failure mode clustering
    print("  Creating failure mode clustering plot...")
    # Extract std values for successful vs failed rollouts
    success_stds = []
    failure_stds = []
    for i, meta in enumerate(metadata):
        if meta[3]:  # success
            success_stds.append(stats_array[i, 1])  # std is column 1
        else:
            failure_stds.append(stats_array[i, 1])

    if len(failure_stds) > 0 and len(success_stds) > 0:
        # create_comparison_histograms expects 2D arrays (n_samples, n_dims)
        # For simple histogram comparison, we'll use create_histogram instead
        fig = plotting.create_histogram(
            np.array(failure_stds),
            title="Influence Std Distribution (Failures)",
        )
        queue_figure(fig, tab_dir / "failure_mode_clustering.png")

    # Use a sample rollout-demo pair for local structure analysis
    if data.num_rollout_episodes > 0:
        print("  Creating local structure analysis plots...")
        selected_rollout = 0
        selected_demo = 0

        # Get the transition matrix
        rollout_ep = data.rollout_episodes[selected_rollout]
        if demo_split == "train":
            demo_samples = data.demo_sample_infos
        elif demo_split == "holdout":
            demo_samples = data.holdout_sample_infos
        else:
            demo_samples = data.demo_sample_infos + data.holdout_sample_infos

        if selected_demo < len(data.demo_episodes):
            demo_ep = data.demo_episodes[selected_demo]
            transition_matrix = data.influence_matrix[
                rollout_ep.sample_start_idx : rollout_ep.sample_end_idx,
                demo_ep.sample_start_idx : demo_ep.sample_end_idx,
            ]

            # Autocorrelation
            # Compute autocorrelation for both axes
            def autocorr(x, lag=1):
                """Compute autocorrelation at a given lag."""
                if lag >= len(x):
                    return 0
                return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag > 0 else 1.0

            max_lag = min(10, transition_matrix.shape[0] // 2, transition_matrix.shape[1] // 2)
            
            # Rollout axis autocorrelation (along columns, averaged)
            rollout_lags = list(range(max_lag + 1))
            rollout_autocorr = []
            for lag in rollout_lags:
                autocorrs = []
                for col in range(transition_matrix.shape[1]):
                    ac = autocorr(transition_matrix[:, col], lag)
                    if not np.isnan(ac):
                        autocorrs.append(ac)
                rollout_autocorr.append(np.mean(autocorrs) if autocorrs else 0)
            
            # Demo axis autocorrelation (along rows, averaged)
            demo_lags = list(range(max_lag + 1))
            demo_autocorr = []
            for lag in demo_lags:
                autocorrs = []
                for row in range(transition_matrix.shape[0]):
                    ac = autocorr(transition_matrix[row, :], lag)
                    if not np.isnan(ac):
                        autocorrs.append(ac)
                demo_autocorr.append(np.mean(autocorrs) if autocorrs else 0)
            
            fig = plotting.create_autocorrelation_plot(
                rollout_lags,
                rollout_autocorr,
                demo_lags,
                demo_autocorr,
                title=f"Autocorrelation (Rollout {selected_rollout}, Demo {selected_demo})",
            )
            queue_figure(fig, tab_dir / "autocorrelation.png")

            # Diagonal analysis
            diagonal_vals = np.diagonal(transition_matrix)
            mask = ~np.eye(transition_matrix.shape[0], transition_matrix.shape[1], dtype=bool)
            off_diagonal_vals = transition_matrix[mask]
            
            fig = plotting.create_diagonal_analysis_plot(
                diagonal_vals,
                off_diagonal_vals,
                title=f"Diagonal Analysis (Rollout {selected_rollout}, Demo {selected_demo})",
            )
            queue_figure(fig, tab_dir / "diagonal_analysis.png")

            # Gradient magnitude
            fig = plotting.create_gradient_magnitude_heatmap(
                transition_matrix,
                title=f"Gradient Magnitude (Rollout {selected_rollout}, Demo {selected_demo})",
            )
            queue_figure(fig, tab_dir / "gradient_magnitude.png")

            # Peak detection
            # Find local maxima and minima
            from scipy import ndimage
            # Use a simple threshold for peaks
            threshold = np.percentile(np.abs(transition_matrix), 90)
            max_coords = np.argwhere(transition_matrix > threshold)
            min_coords = np.argwhere(transition_matrix < -threshold)
            
            fig = plotting.create_peak_detection_plot(
                transition_matrix,
                max_coords,
                min_coords,
                title=f"Peak Detection (Rollout {selected_rollout}, Demo {selected_demo})",
            )
            queue_figure(fig, tab_dir / "peak_detection.png")

            # Asymmetry variance
            # Compute variance along each axis
            var_along_demo = np.var(transition_matrix, axis=1)
            var_along_rollout = np.var(transition_matrix, axis=0)
            
            fig = plotting.create_asymmetry_variance_plot(
                var_along_demo,
                var_along_rollout,
                title=f"Asymmetry Variance (Rollout {selected_rollout}, Demo {selected_demo})",
            )
            queue_figure(fig, tab_dir / "asymmetry_variance.png")

    # Demo variance
    print("  Creating demo variance plot...")
    # Group by demo and compute variance of std across rollouts
    demo_variance_data = {}
    for i, meta in enumerate(metadata):
        demo_idx = meta[1]
        if demo_idx not in demo_variance_data:
            demo_variance_data[demo_idx] = []
        demo_variance_data[demo_idx].append(stats_array[i, 1])  # std

    demo_indices = []
    demo_avg_stds = []
    demo_counts = []
    demo_qualities = []
    
    for demo_idx in sorted(demo_variance_data.keys()):
        demo_indices.append(demo_idx)
        demo_avg_stds.append(np.mean(demo_variance_data[demo_idx]))  # Average std
        demo_counts.append(len(demo_variance_data[demo_idx]))  # Count
        # Get quality label if available
        if demo_idx < len(data.demo_episodes):
            quality = getattr(data.demo_episodes[demo_idx], "quality_label", "unknown")
        else:
            quality = "unknown"
        demo_qualities.append(quality)

    fig = plotting.create_demo_variance_plot(
        demo_indices,
        demo_avg_stds,
        demo_counts,
        demo_qualities,

        title="Demonstration Variance in Influence Patterns",
    )
    queue_figure(fig, tab_dir / "demo_variance.png")

    print(f"✓ Completed Advanced Analysis exports")

    # Export all queued figures
    print("  Saving plots...")
    flush_figures()

def main():
    parser = argparse.ArgumentParser(
        description="Export all visualizations from the influence visualizer"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name (e.g., 'mock', 'pusht', 'lift')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--demo-split",
        type=str,
        default="train",
        choices=["train", "holdout", "both"],
        help="Demo split to use (default: train)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top influences to show (default: 20)",
    )
    parser.add_argument(
        "--obs-key",
        type=str,
        default="agentview_image",
        help="Image observation key (default: agentview_image)",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="annotations/behavior_labels.pkl",
        help="Path to annotation file (default: annotations/behavior_labels.pkl)",
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration: {args.config}")
    config = load_config(args.config)

    # Load data
    print("Loading influence data...")
    loader = InfluenceDataLoader(config)
    loader.load()
    data = loader.data

    # Create output directory
    output_base = Path(args.output) / args.config
    ensure_dir(output_base)

    print(f"\n{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_base}")
    print(f"Demo split: {args.demo_split}")
    print(f"Rollout episodes: {data.num_rollout_episodes}")
    print(f"Demo episodes: {len(data.demo_episodes)}")
    print(f"{'='*60}")

    # Export all tabs
    try:
        export_aggregated_influence(
            data, output_base, args.demo_split, args.annotation_file
        )
    except Exception as e:
        print(f"✗ Error exporting Aggregated Influence: {e}")

    try:
        export_episode_influence(
            data, output_base, args.demo_split, args.annotation_file
        )
    except Exception as e:
        print(f"✗ Error exporting Episode Influence: {e}")

    try:
        export_behaviors(
            data,
            output_base,
            args.demo_split,
            args.top_k,
            args.obs_key,
            args.annotation_file,
        )
    except Exception as e:
        print(f"✗ Error exporting Behaviors: {e}")

    try:
        export_clustering(data, output_base, args.demo_split, args.annotation_file)
    except Exception as e:
        print(f"✗ Error exporting Clustering: {e}")

    try:
        export_local_behaviors(
            data,
            output_base,
            args.demo_split,
            args.top_k,
            args.obs_key,
            args.annotation_file,
        )
    except Exception as e:
        print(f"✗ Error exporting Local Behaviors: {e}")

    try:
        export_advanced_analysis(data, output_base, args.demo_split)
    except Exception as e:
        print(f"✗ Error exporting Advanced Analysis: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print("Export completed!")
    print(f"Output directory: {output_base}")
    
    # Count files
    png_files = list(output_base.rglob("*.png"))
    print(f"Total PNG files created: {len(png_files)}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in png_files)
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

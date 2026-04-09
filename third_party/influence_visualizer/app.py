"""Streamlit web application for visualizing influence matrices.

This app allows users to:
1. Select a task configuration (e.g., robomimic_lift, pusht)
2. Select a policy rollout episode
3. View the rollout frame with a timeline/frame selector
4. For each frame, see a ranked grid of demonstration frames sorted by influence
5. View action chunks for both rollout and demo samples

The data loading uses the EXACT same code paths as train_trak_diffusion.py to ensure
perfect alignment between influence scores and displayed data.

Usage:
    streamlit run influence_visualizer/app.py

    Then select a config file from the sidebar and enter the data paths.
"""

import os
import pathlib
from typing import Optional

import numpy as np
import streamlit as st

from influence_visualizer import plotting
from influence_visualizer.config import (
    VisualizerConfig,
    list_configs,
    load_config,
)
from influence_visualizer.data_loader import InfluenceDataLoader
from influence_visualizer.profiling import get_profiler, profile
from influence_visualizer.render_advanced_analysis import (
    analyze_failure_modes,
    analyze_local_structure,
)
from influence_visualizer.render_annotation import (
    get_episode_annotations,
    get_label_for_frame,
    load_annotations,
    render_annotation_interface,
)
from influence_visualizer.render_behaviors import render_behaviors_tab
from influence_visualizer.render_clustering import render_clustering_tab
from influence_visualizer.render_comparison import render_comparison_tab
from influence_visualizer.render_frames import (
    frame_player,
    render_annotated_frame,
    render_label_timeline,
)
from influence_visualizer.render_heatmaps import (
    get_split_data,
    render_full_trajectory_influence_heatmap,
    render_influence_distribution_by_success,
    render_influence_heatmap,
    render_influence_magnitude_over_time,
    render_performance_influence,
    render_transition_statistics_density,
    render_transition_statistics_scatter,
)
from influence_visualizer.render_influences import (
    AGGREGATION_METHODS,
    rank_demos_by_slice_influence,
    render_slice_opponents,
    render_slice_opponents_slice,
    render_top_influences,
    render_top_influences_slice,
)
from influence_visualizer.render_learning import render_learning_tab
from influence_visualizer.render_local_behaviors import render_local_behaviors_tab
from influence_visualizer.render_ood_filtering import render_ood_filtering_tab
from influence_visualizer.render_video_export import render_video_export_tab
from influence_visualizer.video_export import export_slice_videos


def render_config_selector() -> tuple[Optional[str], Optional[VisualizerConfig]]:
    """Render config file selector in sidebar.

    Returns:
        Tuple of (selected_config_name, VisualizerConfig) if valid,
        (None, None) if data paths not set
    """
    st.header("Configuration")

    # List available configs
    config_names = list_configs()
    if not config_names:
        st.error("No config files found in influence_visualizer/configs/")
        return None, None

    # Config selector
    selected_config = st.selectbox(
        "Task Config",
        options=config_names,
        index=config_names.index("mock") if "mock" in config_names else 0,
        help="Select a task configuration file",
    )

    # Load the config
    config = load_config(selected_config)

    # Show task info
    st.caption(f"Task: {config.name}")

    # Display configured paths (read-only, from config file)
    if config.eval_dir:
        st.caption(f"Eval: {config.eval_dir}")
    if config.train_dir:
        st.caption(f"Train: {config.train_dir}")

    # If mock mode, return immediately
    if config.use_mock:
        st.info("Using mock data for demonstration")
        return selected_config, config

    st.divider()

    # Optional: advanced settings in expander
    with st.expander("Advanced Settings"):
        config.train_ckpt = st.text_input(
            "Checkpoint",
            value=config.train_ckpt,
            help="Checkpoint to use: 'latest', 'best', or epoch number",
        )
        config.exp_date = st.text_input(
            "Experiment Date",
            value=config.exp_date,
            help="Experiment date prefix for TRAK results",
        )
        config.image_dataset_path = (
            st.text_input(
                "Image Dataset Path",
                value=config.image_dataset_path or "",
                help="Optional: Path to image HDF5 if training used lowdim",
            )
            or None
        )

    # Validate paths
    if not config.eval_dir or not config.train_dir:
        st.warning("Please provide eval and train directories to load data")
        return None, None

    return selected_config, config


@st.cache_resource
def get_cached_loader(
    _config_hash: str, config: VisualizerConfig
) -> InfluenceDataLoader:
    """Load and cache the data loader.

    Args:
        _config_hash: Hash of config for cache invalidation
        config: VisualizerConfig with data paths

    Returns:
        Loaded InfluenceDataLoader
    """
    loader = InfluenceDataLoader(config)
    loader.load()
    return loader


def apply_soft_threshold_cached(
    influence_matrix: np.ndarray,
    lambda_threshold: float,
) -> np.ndarray:
    """Apply soft thresholding with session-state caching.

    Uses session_state to cache by (matrix_id, lambda) without slow array hashing.

    Args:
        influence_matrix: Original influence matrix
        lambda_threshold: Threshold parameter

    Returns:
        Thresholded influence matrix
    """
    from influence_visualizer.preprocessing import soft_threshold

    # Use session state for caching (keyed by matrix id + lambda)
    cache_key = f"soft_threshold_{id(influence_matrix)}_{lambda_threshold}"

    if cache_key not in st.session_state:
        # Compute and cache
        matrix_copy = influence_matrix.copy()
        result = soft_threshold(matrix_copy, lambda_threshold)
        st.session_state[cache_key] = result

    return st.session_state[cache_key]


def compute_sparsity_cached(
    influence_matrix: np.ndarray, lambda_threshold: float
) -> float:
    """Compute sparsity after thresholding (cached for preview).

    Args:
        influence_matrix: Original influence matrix
        lambda_threshold: Threshold parameter

    Returns:
        Sparsity percentage (0-100)
    """
    from influence_visualizer.preprocessing import compute_sparsity

    # Use session state for caching
    cache_key = f"sparsity_{id(influence_matrix)}_{lambda_threshold}"

    if cache_key not in st.session_state:
        # Reuse the thresholded matrix if already cached
        thresholded = apply_soft_threshold_cached(influence_matrix, lambda_threshold)
        result = compute_sparsity(thresholded)
        st.session_state[cache_key] = result

    return st.session_state[cache_key]


@st.cache_data(show_spinner=False, hash_funcs={np.ndarray: id})
def analyze_distribution_cached(_influence_matrix: np.ndarray) -> dict:
    """Analyze influence distribution (cached).

    Args:
        _influence_matrix: Influence matrix

    Returns:
        Dictionary with distribution statistics
    """
    from influence_visualizer.preprocessing import analyze_influence_distribution

    return analyze_influence_distribution(_influence_matrix)


def render_dataset_info(loader: InfluenceDataLoader, config: VisualizerConfig):
    """Render dataset info in sidebar."""
    from influence_visualizer.data_loader import (
        get_eval_dir_for_seed,
        load_rollout_success_stats,
    )

    st.header("Dataset Info")
    st.metric("Rollout Episodes", loader.num_rollout_episodes)
    # Policy rollout success rate for this task config
    if config.seeds and len(config.seeds) > 0:
        reference_seed = config.seeds[0]
        eval_dir_base = config.eval_dir or ""
        seed_stats = []
        for seed in config.seeds:
            eval_dir_seed = get_eval_dir_for_seed(
                eval_dir_base, seed=seed, reference_seed=reference_seed
            )
            try:
                n_success, n_total = load_rollout_success_stats(
                    pathlib.Path(eval_dir_seed)
                )
                pct = 100.0 * n_success / n_total if n_total else 0.0
                seed_stats.append((seed, n_success, n_total, pct))
            except (FileNotFoundError, OSError):
                seed_stats.append((seed, None, None, None))
        # Per-seed success rates
        st.markdown("**Rollout success rate**")
        for seed, n_success, n_total, pct in seed_stats:
            if pct is not None:
                st.caption(f"Seed {seed}: {pct:.1f}% ({n_success} / {n_total})")
            else:
                st.caption(f"Seed {seed}: —")
        # Average success rate across seeds
        valid = [(s, pct) for (s, _, _, pct) in seed_stats if pct is not None]
        if valid:
            avg_pct = sum(p for (_, p) in valid) / len(valid)
            st.metric("Avg success rate (seeds)", f"{avg_pct:.1f}%")
        else:
            st.metric("Avg success rate (seeds)", "—")
    else:
        n_rollouts = len(loader.data.rollout_episodes)
        if n_rollouts > 0:
            n_success = sum(
                1 for ep in loader.data.rollout_episodes
                if ep.success is True
            )
            success_pct = 100.0 * n_success / n_rollouts
            st.metric("Rollout Success Rate", f"{success_pct:.1f}%")
            st.caption(f"{n_success} / {n_rollouts} episodes")
        else:
            st.metric("Rollout Success Rate", "—")
    st.metric("Train Demo Episodes", len(loader.demo_episodes))
    st.metric("Holdout Demo Episodes", len(loader.holdout_episodes))
    st.metric("Rollout Samples", loader.num_rollout_samples)
    st.metric("Train Demo Samples", len(loader.data.demo_sample_infos))
    st.metric("Holdout Demo Samples", len(loader.data.holdout_sample_infos))

    st.divider()

    st.header("Config (from checkpoint)")
    st.text(f"Horizon: {loader.horizon}")
    st.text(f"Pad Before: {loader.data.pad_before}")
    st.text(f"Pad After: {loader.data.pad_after}")
    st.text(f"N Obs Steps: {loader.data.n_obs_steps}")


def render_settings(config: VisualizerConfig, loader: InfluenceDataLoader):
    """Render settings controls in sidebar.

    Args:
        config: VisualizerConfig with default settings
        loader: InfluenceDataLoader to get available image keys

    Returns:
        Tuple of (demo_split, top_k, obs_key)
    """
    st.header("Settings")

    # Demo split selector
    demo_split = st.selectbox(
        "Demo Split",
        options=["train", "holdout", "both"],
        index=0,
        help="Which demonstration set to analyze: train only, holdout only, or both combined",
    )

    top_k = st.slider(
        "Top K Influences",
        min_value=5,
        max_value=50,
        value=config.top_k,
    )

    # Get available image keys from the dataset
    available_image_keys = loader.get_available_image_keys()

    if available_image_keys:
        # If config.obs_key is in the list, use it as default
        default_index = 0
        if config.obs_key in available_image_keys:
            default_index = available_image_keys.index(config.obs_key)

        obs_key = st.selectbox(
            "Camera View",
            options=available_image_keys,
            index=default_index,
            help="Select which camera view to display in visualizations",
        )
    else:
        # Fallback to text input if no image keys available
        obs_key = st.text_input(
            "Image Obs Key",
            value=config.obs_key,
            help="Observation key for camera images",
        )

    # Preprocessing settings
    st.divider()
    st.subheader("Preprocessing")

    enable_soft_threshold = st.checkbox(
        "Enable soft thresholding",
        value=False,
        help="Apply soft thresholding to influence matrix (recommended by TRAK paper for sparsity)",
    )

    if enable_soft_threshold:
        # Simple manual input mode (fast - no analysis)
        st.markdown(
            "**Quick Start**: Enter λ manually or click 'Auto-suggest' for recommendations"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            lambda_threshold = st.number_input(
                "Threshold λ",
                min_value=0.0,
                max_value=1.0,
                value=0.001,
                step=0.000000001,
                format="%.9f",
                help="Soft threshold parameter. Higher = more sparsity. Start with 0.001 and adjust. Values can be very small (e.g., 0.000000001)",
            )
        with col2:
            show_suggestions = st.checkbox(
                "Auto-suggest",
                value=False,
                help="Analyze matrix to suggest optimal λ values (may take 30-60s for large matrices)",
            )

        # Optional: Show distribution analysis and suggestions
        if show_suggestions:
            with st.spinner(
                "Analyzing influence distribution... (this may take 30-60s for large matrices)"
            ):
                from influence_visualizer.preprocessing import suggest_threshold

                # Analyze distribution to suggest threshold (cached but slow first time)
                dist_stats = analyze_distribution_cached(loader.data.influence_matrix)

                st.success("✅ Analysis complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Sparsity", f"{dist_stats['sparsity']:.1f}%")
                with col2:
                    st.metric("Median |Score|", f"{dist_stats['abs_median']:.4f}")
                with col3:
                    st.metric("Max |Score|", f"{dist_stats['max']:.4f}")

                # Suggest thresholds for different sparsity levels
                suggested_50 = suggest_threshold(
                    loader.data.influence_matrix, target_sparsity=0.5
                )
                suggested_70 = suggest_threshold(
                    loader.data.influence_matrix, target_sparsity=0.7
                )
                suggested_90 = suggest_threshold(
                    loader.data.influence_matrix, target_sparsity=0.9
                )

                st.info(
                    f"💡 **Suggested λ values:**\n\n"
                    f"- **{suggested_50:.4f}** for 50% sparsity (conservative)\n"
                    f"- **{suggested_70:.4f}** for 70% sparsity \n"
                    f"- **{suggested_90:.4f}** for 90% sparsity (aggressive)"
                )

        # Always show preview (fast)
        preview_sparsity = compute_sparsity_cached(
            loader.data.influence_matrix, lambda_threshold
        )

        st.success(
            f"📊 **Preview**: {preview_sparsity:.1f}% sparsity with λ={lambda_threshold:.9f}"
        )

    else:
        lambda_threshold = 0.0

    # Performance settings
    st.divider()
    st.subheader("Performance")

    enable_detailed_hover = st.checkbox(
        "Enable detailed hover info",
        value=False,
        help="Show behavior labels in heatmap hover text. Disable for faster rendering of large heatmaps.",
    )

    return demo_split, top_k, obs_key, enable_detailed_hover, lambda_threshold


def get_annotation_file() -> str:
    """Get the hardcoded annotation file path.

    Returns:
        Path to the single annotation file
    """
    return "annotations/behavior_labels.json"


@st.fragment
def _render_traj_influence_section(data, demo_split, annotation_file, task_config):
    st.header("Trajectory-wise Influence Matrix")
    st.markdown("""
    This shows the **original CUPID performance influence matrix** where influences are
    aggregated from action-level to trajectory-level using `mean_of_mean_influence`.
    Each cell (i, j) represents the influence of demo trajectory j on rollout trajectory i.
    """)

    with st.expander("Load Trajectory Influence Matrix", expanded=False):
        show_traj_key = f"show_traj_influence_{demo_split}"
        if st.button(
            "Generate Trajectory Influence Matrix",
            key=f"gen_traj_influence_{demo_split}",
        ):
            st.session_state[show_traj_key] = True

        if st.session_state.get(show_traj_key, False):
            render_full_trajectory_influence_heatmap(
                data,
                split=demo_split,
                annotation_file=annotation_file,
                task_config=task_config,
            )


@st.fragment
def _render_performance_influence_section(data, demo_split):
    st.header("Performance Influence per Demonstration")
    st.markdown("""
    **Performance influence** measures how each demonstration contributes to policy performance.

    - Uses `sum_of_sum_influence` for trajectory-level aggregation
    - Computes quality scores: sum over successful rollouts minus sum over failed rollouts
    - Produces raw (un-normalized) scores that match the notebook's results

    Higher scores indicate demonstrations that positively influence successful rollouts more than failed ones.
    """)

    with st.expander("Load Performance Influence", expanded=False):
        # Add metric selector
        col_metric, col_top_k = st.columns([2, 1])
        with col_metric:
            perf_metric = st.selectbox(
                "Performance Metric",
                options=["net", "succ", "fail"],
                index=0,
                help="net: success - failure | succ: only successful rollouts | fail: only failed rollouts",
                key=f"perf_metric_select_{demo_split}",
            )
        with col_top_k:
            perf_top_k = st.slider(
                "Top/Bottom K",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of top and bottom demonstrations to display in tables",
                key=f"perf_top_k_slider_{demo_split}",
            )

        show_perf_key = f"show_perf_influence_{demo_split}"
        if st.button(
            "Generate Performance Influence",
            key=f"gen_perf_influence_{demo_split}",
        ):
            st.session_state[show_perf_key] = True

        if st.session_state.get(show_perf_key, False):
            render_performance_influence(
                data,
                split=demo_split,
                metric=perf_metric,
                top_k=perf_top_k,
            )


@st.fragment
def _render_transition_stats_section(data, demo_split, annotation_file, task_config):
    st.header("Transition-Level Influence Statistics")
    st.markdown("""
    **Transition-level statistics** analyze the raw influence matrices at the finest granularity.

    For each rollout-demo pair, this computes statistics (mean, std, min, max) across the
    transition-level influence matrix (rollout timesteps × demonstration timesteps).

    This can help identify patterns in how influence is distributed across different
    rollout-demonstration interactions.
    """)

    with st.expander("Load Transition Statistics", expanded=False):
        show_stats_key = f"show_transition_stats_{demo_split}"
        if st.button(
            "Compute Transition Statistics",
            key=f"gen_transition_stats_{demo_split}",
            help="Compute and display transition-level statistics. This may take a moment for large datasets.",
        ):
            st.session_state[show_stats_key] = True

        if st.session_state.get(show_stats_key, False):
            render_transition_statistics_scatter(
                data,
                split=demo_split,
                annotation_file=annotation_file,
                task_config=task_config,
            )


@st.fragment
def _render_influence_dist_section(data, demo_split):
    st.header("Influence Distribution: Success vs. Failure")
    st.markdown("""
    **Distribution comparison** shows how individual influence values differ between
    successful and failed rollouts.

    This visualization reveals whether failed rollouts have more variable (inconsistent)
    influence patterns compared to successful ones, which could indicate conflicting
    guidance from training data.
    """)

    with st.expander("Load Influence Distribution", expanded=False):
        show_dist_key = f"show_influence_dist_{demo_split}"
        if st.button(
            "Generate Distribution Comparison",
            key=f"gen_influence_dist_{demo_split}",
        ):
            st.session_state[show_dist_key] = True

        if st.session_state.get(show_dist_key, False):
            render_influence_distribution_by_success(data, split=demo_split)


@st.fragment
def _render_transition_density_section(data, demo_split):
    st.header("Transition Statistics Density: Success vs. Failure")
    st.markdown("""
    **Overlapping density histograms** show how transition-level influence statistics
    (mean, std, min, max) are distributed differently between successful and failed rollouts.

    This visualization helps identify whether certain statistical patterns (e.g., high std)
    are more common in failures versus successes.
    """)

    with st.expander("Load Transition Density", expanded=False):
        show_density_key = f"show_transition_density_{demo_split}"
        if st.button(
            "Generate Transition Density",
            key=f"gen_transition_density_{demo_split}",
        ):
            st.session_state[show_density_key] = True

        if st.session_state.get(show_density_key, False):
            render_transition_statistics_density(data, split=demo_split)


def render_aggregated_tab(data, demo_split, annotation_file, task_config):
    """Render aggregated influence visualizations using localized fragments."""
    _render_traj_influence_section(data, demo_split, annotation_file, task_config)
    st.divider()
    _render_performance_influence_section(data, demo_split)
    st.divider()
    _render_transition_stats_section(data, demo_split, annotation_file, task_config)
    st.divider()
    _render_influence_dist_section(data, demo_split)
    st.divider()
    _render_transition_density_section(data, demo_split)


def render_episode_tab(data, demo_split, top_k, obs_key, annotation_file, task_config):
    """Render episode-specific influence visualizations."""

    # Episode selection
    st.header("Select Rollout Episode")

    # Episode selector with success/failure indicator
    episode_options = []
    for ep in data.rollout_episodes:
        status = "✓" if ep.success else "✗" if ep.success is not None else "?"
        episode_options.append(
            f"Episode {ep.index} [{status}] ({ep.num_samples} samples)"
        )

    selected_episode_str = st.selectbox(
        "Choose a rollout episode to analyze:",
        options=episode_options,
        index=0,
    )
    selected_episode_idx = episode_options.index(selected_episode_str)
    selected_episode = data.rollout_episodes[selected_episode_idx]

    # Show episode info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Episode Index", selected_episode.index)
    with col2:
        status = (
            "Success"
            if selected_episode.success
            else "Failure"
            if selected_episode.success is not None
            else "Unknown"
        )
        st.metric("Status", status)
    with col3:
        st.metric("Samples", selected_episode.num_samples)

    st.divider()

    # Rollout frame player in its own fragment so playback doesn't reload the page
    # During playback: only the frame player reruns (fragment_scope=True)
    # During manual slider change: full page rerun updates the influence section below
    _render_rollout_frame_player(data, selected_episode, annotation_file, task_config)

    st.divider()

    # Influence section - NOT a fragment, so it updates when slider is manually changed
    # but NOT during playback (since frame_player uses fragment_scope=True)
    _render_episode_influence_content(
        data,
        selected_episode,
        selected_episode_idx,
        demo_split,
        top_k,
        obs_key,
        annotation_file,
        task_config,
    )


@st.fragment
def _render_rollout_frame_player(data, selected_episode, annotation_file, task_config):
    """Fragment for the rollout frame player - isolated to prevent full page reruns."""
    st.header("View Rollout & Select Frame")

    # Load annotations for label display
    with profile("load_annotations_frame_section"):
        annotations = load_annotations(annotation_file, task_config=task_config)
    episode_id_str = str(selected_episode.index)
    episode_annotations = get_episode_annotations(
        annotations, episode_id_str, split="rollout"
    )

    # Frame player with play/pause
    def _render_rollout_frame(sample_offset):
        abs_idx = selected_episode.sample_start_idx + sample_offset

        # Get label for this frame
        current_label = get_label_for_frame(sample_offset, episode_annotations)

        # Render label timeline
        with profile("render_label_timeline"):
            render_label_timeline(
                episode_annotations,
                num_frames=selected_episode.num_samples,
                current_frame=sample_offset,
                unique_key=f"tab_ep_timeline_rollout_ep{selected_episode.index}",
            )

        col_frame, col_action = st.columns([1, 1])
        with col_frame:
            with profile("get_rollout_frame"):
                rollout_frame = data.get_rollout_frame(abs_idx)
            lbl = f"t={sample_offset} | ep={selected_episode.index}"
            label_str = current_label if current_label != "no label" else "no label yet"
            with profile("render_annotated_frame"):
                render_annotated_frame(
                    rollout_frame,
                    lbl,
                    f"Global sample index: {abs_idx} | Label: {label_str}",
                )
        with col_action:
            with profile("get_rollout_action"):
                rollout_action = data.get_rollout_action(abs_idx)
            if rollout_action is not None:
                st.text(f"Action shape: {rollout_action.shape}")
            else:
                st.info("Action not available")

    # Key for the frame player - used to read current value from session state
    frame_player_key = f"rollout_ep{selected_episode.index}_frame"

    frame_player(
        label="Select sample (timestep) within episode:",
        min_value=0,
        max_value=selected_episode.num_samples - 1,
        key=frame_player_key,
        default_value=0,
        default_fps=3.0,
        help="Each sample corresponds to a state-action pair in the rollout",
        render_fn=_render_rollout_frame,
        fragment_scope=True,  # Use fragment-scoped rerun for playback
    )


@st.fragment
def _render_episode_heatmap_section(
    data, selected_episode_idx, demo_split, annotation_file, task_config
):
    with st.expander("Load Influence Heatmap", expanded=False):
        st.caption(
            "Shows influence aggregated by demo episode for the selected rollout episode. "
            "Each column is a demo episode, each row is a rollout timestep."
        )
        show_heatmap_key = f"show_heatmap_{selected_episode_idx}_{demo_split}"
        if st.button(
            "Generate Influence Heatmap",
            key=f"gen_influence_heatmap_{selected_episode_idx}_{demo_split}",
        ):
            st.session_state[show_heatmap_key] = True

        if st.session_state.get(show_heatmap_key, False):
            with profile("render_influence_heatmap"):
                render_influence_heatmap(
                    data,
                    selected_episode_idx,
                    split=demo_split,
                    annotation_file=annotation_file,
                    task_config=task_config,
                )


@st.fragment
def _render_episode_magnitude_section(
    data, selected_episode_idx, demo_split, annotation_file, task_config
):
    with st.expander("Load Influence Magnitude Over Time", expanded=False):
        st.caption(
            "Shows the magnitude of positive and negative influence aggregated across all demos for each rollout timestep."
        )
        show_magnitude_key = f"show_magnitude_{selected_episode_idx}_{demo_split}"
        if st.button(
            "Generate Influence Magnitude Plot",
            key=f"gen_influence_magnitude_{selected_episode_idx}_{demo_split}",
        ):
            st.session_state[show_magnitude_key] = True

        if st.session_state.get(show_magnitude_key, False):
            with profile("render_influence_magnitude_over_time"):
                render_influence_magnitude_over_time(
                    data,
                    selected_episode_idx,
                    split=demo_split,
                    annotation_file=annotation_file,
                    task_config=task_config,
                )


def _render_episode_heatmaps(
    data,
    selected_episode_idx,
    demo_split,
    annotation_file,
    task_config,
):
    """Render episode-level heatmaps using localized fragments."""
    st.header("Action-level Influence Overview")
    _render_episode_heatmap_section(
        data, selected_episode_idx, demo_split, annotation_file, task_config
    )
    _render_episode_magnitude_section(
        data, selected_episode_idx, demo_split, annotation_file, task_config
    )


@st.fragment
def _render_top_influences_section(
    data,
    selected_episode,
    selected_episode_idx,
    demo_split,
    top_k,
    obs_key,
    annotation_file,
    task_config,
    frame_player_key,
    analyzed_frame_key,
    show_influences_key,
    current_offset,
):
    analyzed_offset = st.session_state.get(analyzed_frame_key, current_offset)
    current_abs_idx = selected_episode.sample_start_idx + analyzed_offset

    with st.expander("Load Top Influential Demonstrations", expanded=False):
        # Show current vs analyzed frame status
        col_status, col_button = st.columns([3, 1])
        with col_status:
            if current_offset != analyzed_offset:
                st.info(
                    f"Currently viewing frame {current_offset}. "
                    f"Click button to analyze this frame."
                )
            else:
                st.caption(f"Ready to analyze frame {current_offset}")

        with col_button:
            if st.button(
                "Analyze Current Frame",
                key=f"analyze_frame_btn_{selected_episode.index}",
                type="primary" if current_offset != analyzed_offset else "secondary",
                help="Load top influences for the currently selected frame",
            ):
                st.session_state[analyzed_frame_key] = current_offset
                st.session_state[show_influences_key] = True
                st.rerun(scope="fragment")

        # Only show influences if button was clicked
        if st.session_state.get(show_influences_key, False):
            st.markdown(
                f"**For Rollout Sample {analyzed_offset} (Global Index: {current_abs_idx})**"
            )

            with profile("render_top_influences"):
                render_top_influences(
                    data,
                    current_abs_idx,
                    selected_episode_idx,
                    top_k=top_k,
                    obs_key=obs_key,
                    split=demo_split,
                    annotation_file=annotation_file,
                    task_config=task_config,
                    rollout_frame_key=frame_player_key,
                )


@st.fragment
def _render_slice_opponents_section(
    data,
    selected_episode,
    selected_episode_idx,
    demo_split,
    top_k,
    obs_key,
    annotation_file,
    task_config,
    frame_player_key,
    analyzed_frame_key,
    current_offset,
):
    analyzed_offset = st.session_state.get(analyzed_frame_key, current_offset)
    current_abs_idx = selected_episode.sample_start_idx + analyzed_offset
    show_opponents_key = f"show_opponents_ep{selected_episode.index}_{demo_split}"

    with st.expander("Load Slice Opponents (Lowest Influence)", expanded=False):
        st.markdown(
            f"**For Rollout Sample {analyzed_offset} (Global Index: {current_abs_idx})**"
        )
        if st.button(
            "Generate Slice Opponents",
            key=f"gen_slice_opponents_{selected_episode.index}_{demo_split}",
        ):
            st.session_state[show_opponents_key] = True

        if st.session_state.get(show_opponents_key, False):
            with profile("render_slice_opponents"):
                render_slice_opponents(
                    data,
                    current_abs_idx,
                    selected_episode_idx,
                    top_k=top_k,
                    obs_key=obs_key,
                    split=demo_split,
                    annotation_file=annotation_file,
                    task_config=task_config,
                    rollout_frame_key=frame_player_key,
                )


def _render_sample_influences(
    data,
    selected_episode,
    selected_episode_idx,
    demo_split,
    top_k,
    obs_key,
    annotation_file,
    task_config,
):
    """Render sample-level influences using localized fragments."""
    frame_player_key = f"rollout_ep{selected_episode.index}_frame"
    analyzed_frame_key = f"analyzed_frame_ep{selected_episode.index}"
    show_influences_key = f"show_influences_ep{selected_episode.index}"

    # Get current frame from player
    current_offset = st.session_state.get(f"{frame_player_key}_value", 0)

    # Initialize analyzed frame if needed
    if analyzed_frame_key not in st.session_state:
        st.session_state[analyzed_frame_key] = current_offset

    st.header("Sample-Level Influence Analysis")

    _render_top_influences_section(
        data,
        selected_episode,
        selected_episode_idx,
        demo_split,
        top_k,
        obs_key,
        annotation_file,
        task_config,
        frame_player_key,
        analyzed_frame_key,
        show_influences_key,
        current_offset,
    )

    _render_slice_opponents_section(
        data,
        selected_episode,
        selected_episode_idx,
        demo_split,
        top_k,
        obs_key,
        annotation_file,
        task_config,
        frame_player_key,
        analyzed_frame_key,
        current_offset,
    )


@st.fragment
def _render_slice_influence_section(
    data,
    selected_episode,
    selected_episode_idx,
    demo_split,
    top_k,
    obs_key,
    annotation_file,
    task_config,
):
    """Render slice-based influence analysis with sliding window aggregation."""
    with st.expander("Slice-Based Influence Analysis (Sliding Window)", expanded=False):
        st.caption(
            "Analyze influence for a slice of the rollout using sliding window aggregation "
            "over demonstration timesteps. This allows you to find demonstrations that are "
            "most/least influential for a specific segment of the rollout."
        )

        # Slice selection
        st.markdown("**Select Rollout Slice**")
        col_start, col_end = st.columns(2)
        max_offset = selected_episode.num_samples - 1

        with col_start:
            slice_start = st.number_input(
                "Start index (inclusive)",
                min_value=0,
                max_value=max_offset,
                value=0,
                key=f"slice_start_ep{selected_episode.index}_{demo_split}",
                help="Start index of the rollout slice to analyze",
            )
        with col_end:
            slice_end = st.number_input(
                "End index (exclusive)",
                min_value=1,
                max_value=selected_episode.num_samples,
                value=min(10, selected_episode.num_samples),
                key=f"slice_end_ep{selected_episode.index}_{demo_split}",
                help="End index of the rollout slice (exclusive)",
            )

        # Validate slice
        if slice_start >= slice_end:
            st.error("Start index must be less than end index")
            return

        slice_height = slice_end - slice_start
        st.info(f"Slice covers {slice_height} timesteps: [{slice_start}:{slice_end}]")

        st.divider()

        # Sliding window configuration
        st.markdown("**Sliding Window Configuration**")
        col_window, col_agg = st.columns(2)

        with col_window:
            window_width = st.number_input(
                "Window width (demo timesteps)",
                min_value=1,
                max_value=100,
                value=min(slice_height, 10),
                key=f"window_width_ep{selected_episode.index}_{demo_split}",
                help="Width of the sliding window along the demonstration timestep axis. "
                "The window has height equal to the rollout slice size and this width.",
            )

        with col_agg:
            aggregation_method = st.selectbox(
                "Aggregation method",
                options=list(AGGREGATION_METHODS.keys()),
                index=0,
                key=f"agg_method_ep{selected_episode.index}_{demo_split}",
                help="How to aggregate influence values within each window position",
            )

        # Explain the aggregation methods
        with st.expander("Aggregation Method Details", expanded=False):
            st.markdown("""
            - **sum**: Total sum of all influence values in the window
            - **mean**: Average of all influence values in the window
            - **max**: Maximum influence value in the window
            - **min**: Minimum influence value in the window
            - **sum_of_means**: Sum of row-wise means (each rollout timestep averaged, then summed)
            - **mean_of_sums**: Mean of row-wise sums (each rollout timestep summed, then averaged)
            - **abs_sum**: Sum of absolute influence values (ignores sign)
            - **abs_mean**: Mean of absolute influence values (ignores sign)
            """)

        st.divider()

        # Generate buttons
        show_slice_top_key = f"show_slice_top_ep{selected_episode.index}_{demo_split}"
        show_slice_opp_key = f"show_slice_opp_ep{selected_episode.index}_{demo_split}"
        show_slice_heatmap_key = (
            f"show_slice_heatmap_ep{selected_episode.index}_{demo_split}"
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button(
                "Visualize Local Matrix",
                key=f"gen_slice_heatmap_{selected_episode.index}_{demo_split}",
                help="Show the local influence matrix heatmap with aggregation scores",
            ):
                st.session_state[show_slice_heatmap_key] = True
                st.session_state[f"slice_params_ep{selected_episode.index}"] = {
                    "start": slice_start,
                    "end": slice_end,
                    "window": window_width,
                    "agg": aggregation_method,
                }
                st.rerun(scope="fragment")

        with col_btn2:
            if st.button(
                "Find Top Influential Demos",
                key=f"gen_slice_top_{selected_episode.index}_{demo_split}",
                type="primary",
            ):
                st.session_state[show_slice_top_key] = True
                st.session_state[f"slice_params_ep{selected_episode.index}"] = {
                    "start": slice_start,
                    "end": slice_end,
                    "window": window_width,
                    "agg": aggregation_method,
                }
                st.rerun(scope="fragment")

        with col_btn3:
            if st.button(
                "Find Slice Opponents (Lowest)",
                key=f"gen_slice_opp_{selected_episode.index}_{demo_split}",
            ):
                st.session_state[show_slice_opp_key] = True
                st.session_state[f"slice_params_ep{selected_episode.index}"] = {
                    "start": slice_start,
                    "end": slice_end,
                    "window": window_width,
                    "agg": aggregation_method,
                }
                st.rerun(scope="fragment")

        # Retrieve stored params (in case they differ from current UI state)
        stored_params = st.session_state.get(
            f"slice_params_ep{selected_episode.index}",
            {
                "start": slice_start,
                "end": slice_end,
                "window": window_width,
                "agg": aggregation_method,
            },
        )

        # Show local influence matrix heatmap
        if st.session_state.get(show_slice_heatmap_key, False):
            st.divider()
            st.subheader("Local Influence Matrix")

            with profile("render_slice_influence_heatmap"):
                # Get the influence matrix and episode data
                influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
                    data, demo_split
                )

                # Convert episode offsets to global indices
                rollout_start_idx = (
                    selected_episode.sample_start_idx + stored_params["start"]
                )
                rollout_end_idx = (
                    selected_episode.sample_start_idx + stored_params["end"]
                )

                # Extract the influence slice
                influence_slice = influence_matrix[rollout_start_idx:rollout_end_idx, :]

                # Compute aggregated scores for the visualization
                sorted_indices, sorted_scores, raw_scores = (
                    rank_demos_by_slice_influence(
                        data,
                        rollout_start_idx,
                        rollout_end_idx,
                        window_width=stored_params["window"],
                        aggregation_method=stored_params["agg"],
                        split=demo_split,
                        ascending=False,
                    )
                )

                # Compute demo episode boundaries for vertical lines
                demo_episode_boundaries = []
                cumsum = 0
                for ep_len in ep_lens:
                    cumsum += ep_len
                    demo_episode_boundaries.append(cumsum)

                # Create the heatmap
                heatmap_title = f"Influence Matrix (agg={stored_params['agg']})"
                fig = plotting.create_slice_influence_heatmap(
                    influence_slice=influence_slice,
                    rollout_start=stored_params["start"],
                    rollout_end=stored_params["end"],
                    aggregated_scores=raw_scores,
                    window_width=stored_params["window"],
                    title=heatmap_title,
                    show_aggregation_line=True,
                    highlight_top_k=min(top_k, 5),  # Highlight top 5 for visibility
                    top_k_indices=sorted_indices,
                    demo_episode_boundaries=demo_episode_boundaries,
                )

                st.plotly_chart(fig, width="stretch")

                # Add some summary stats
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric(
                        "Slice Shape",
                        f"{influence_slice.shape[0]} x {influence_slice.shape[1]}",
                    )
                with col_stats2:
                    st.metric("Max Agg. Score", f"{raw_scores.max():.4f}")
                with col_stats3:
                    st.metric("Min Agg. Score", f"{raw_scores.min():.4f}")

        # Show top influential demos for slice
        if st.session_state.get(show_slice_top_key, False):
            st.divider()

            # Add export button and FPS input
            export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
            with export_col1:
                do_export = st.button(
                    "Export Videos",
                    key=f"export_slice_videos_{selected_episode.index}_{demo_split}",
                    help="Export videos of rollout slice and top-k demo slices to outputs/slice_exports/",
                )
            with export_col2:
                export_fps = st.number_input(
                    "FPS",
                    min_value=1,
                    max_value=60,
                    value=10,
                    step=1,
                    key=f"export_fps_{selected_episode.index}_{demo_split}",
                    help="Frames per second for exported videos",
                )
            if do_export:
                with st.spinner("Exporting videos..."):
                    # Get the top influences
                    sorted_indices, sorted_scores, _ = rank_demos_by_slice_influence(
                        data,
                        selected_episode.sample_start_idx + stored_params["start"],
                        selected_episode.sample_start_idx + stored_params["end"],
                        window_width=stored_params["window"],
                        aggregation_method=stored_params["agg"],
                        split=demo_split,
                        ascending=False,
                    )

                    # Build influence details for top-k
                    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
                        data, demo_split
                    )
                    num_train_samples = len(data.demo_sample_infos)

                    top_influences = []
                    for i, local_sample_idx in enumerate(sorted_indices[:top_k]):
                        score = float(sorted_scores[i])

                        # Find episode
                        episode = None
                        for ep_idx, sample_idxs in enumerate(ep_idxs):
                            if local_sample_idx in sample_idxs:
                                episode = demo_episodes[ep_idx]
                                break

                        if episode is None:
                            continue

                        # Calculate global sample index
                        if demo_split == "train":
                            global_sample_idx = local_sample_idx
                        elif demo_split == "holdout":
                            global_sample_idx = local_sample_idx + num_train_samples
                        else:
                            global_sample_idx = local_sample_idx

                        if global_sample_idx < len(data.all_demo_sample_infos):
                            sample_info = data.all_demo_sample_infos[global_sample_idx]
                        else:
                            continue

                        top_influences.append(
                            {
                                "influence_score": score,
                                "demo_episode_idx": episode.index,
                                "demo_timestep": sample_info.timestep,
                                "local_demo_sample_idx": local_sample_idx,
                                "global_demo_sample_idx": global_sample_idx,
                                "sample_info": sample_info,
                                "episode": episode,
                            }
                        )

                    # Export videos
                    output_dir = pathlib.Path("outputs/slice_exports")
                    exported_paths, error_msg = export_slice_videos(
                        data=data,
                        rollout_episode_idx=selected_episode_idx,
                        rollout_start_offset=stored_params["start"],
                        rollout_end_offset=stored_params["end"],
                        demo_influences=top_influences,
                        output_dir=output_dir,
                        task_config_name=task_config,
                        obs_key=obs_key,
                        fps=int(export_fps),
                        demo_window_width=stored_params["window"],
                    )

                with export_col3:
                    if error_msg:
                        st.error(f"Export failed: {error_msg}")
                    else:
                        st.success(
                            f"Exported {len(exported_paths)} videos to {output_dir}"
                        )

            with profile("render_top_influences_slice"):
                render_top_influences_slice(
                    data,
                    rollout_episode_idx=selected_episode_idx,
                    rollout_start_offset=stored_params["start"],
                    rollout_end_offset=stored_params["end"],
                    window_width=stored_params["window"],
                    aggregation_method=stored_params["agg"],
                    top_k=top_k,
                    obs_key=obs_key,
                    split=demo_split,
                    annotation_file=annotation_file,
                    task_config=task_config,
                )

        # Show slice opponents
        if st.session_state.get(show_slice_opp_key, False):
            st.divider()
            with profile("render_slice_opponents_slice"):
                render_slice_opponents_slice(
                    data,
                    rollout_episode_idx=selected_episode_idx,
                    rollout_start_offset=stored_params["start"],
                    rollout_end_offset=stored_params["end"],
                    window_width=stored_params["window"],
                    aggregation_method=stored_params["agg"],
                    top_k=top_k,
                    obs_key=obs_key,
                    split=demo_split,
                    annotation_file=annotation_file,
                    task_config=task_config,
                )


def _render_episode_influence_content(
    data,
    selected_episode,
    selected_episode_idx,
    demo_split,
    top_k,
    obs_key,
    annotation_file,
    task_config,
):
    """Render the influence visualization section of the episode tab."""
    # Episode-level heatmaps (fragment - doesn't depend on current frame)
    _render_episode_heatmaps(
        data,
        selected_episode_idx,
        demo_split,
        annotation_file,
        task_config,
    )

    st.divider()

    # Sample-level influences (fragment with explicit refresh)
    _render_sample_influences(
        data,
        selected_episode,
        selected_episode_idx,
        demo_split,
        top_k,
        obs_key,
        annotation_file,
        task_config,
    )

    st.divider()

    # Slice-based influence analysis with sliding window
    st.header("Slice-Based Influence Analysis")
    _render_slice_influence_section(
        data,
        selected_episode,
        selected_episode_idx,
        demo_split,
        top_k,
        obs_key,
        annotation_file,
        task_config,
    )


@st.fragment
def render_advanced_analysis_tab(data, demo_split):
    """Render advanced analysis tab for exploring failure modes and local structure.

    This tab provides deep dives into:
    1. Failure mode differentiation using influence std patterns
    2. Local structure properties of transition matrices
    """
    st.markdown("""
    This tab explores two key research questions:

    **A) Failure Mode Differentiation:** Can we differentiate between different failure modes
    based on the standard deviation of local influence matrices?

    **B) Local Structure Analysis:** Which other ways can we leverage the local structure
    and properties of transition-wise influence matrices?
    """)

    st.divider()

    # Section selector
    analysis_mode = st.radio(
        "Select Analysis",
        options=["Failure Mode Analysis", "Local Structure Analysis"],
        horizontal=True,
        key=f"advanced_analysis_mode_{demo_split}",
    )

    st.divider()

    if analysis_mode == "Failure Mode Analysis":
        analyze_failure_modes(data, split=demo_split)
    else:
        analyze_local_structure(data, split=demo_split)


def render_annotation_tab(data, annotation_file, task_config, obs_key: str = "agentview_image"):
    """Render annotation interface for labeling demonstration and rollout segments.

    This tab provides tools for annotating slices of training demonstrations,
    holdout demonstrations, and rollouts with behavioral labels.
    """
    render_annotation_interface(data, annotation_file, task_config=task_config, obs_key=obs_key)


def main():
    st.set_page_config(
        page_title="Influence Visualizer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Influence Visualizer")

    # Sidebar: Config selection and data loading
    with st.sidebar:
        selected_config_name, config = render_config_selector()

        if config is None:
            st.stop()

        # Create a hash for caching - include the config file name to ensure
        # cache invalidation when switching between different task configs
        config_hash = f"{selected_config_name}_{config.task}_{config.eval_dir}_{config.train_dir}_{config.train_ckpt}"

        # Reset session state if config has changed
        if "last_config_hash" not in st.session_state:
            st.session_state.last_config_hash = config_hash
        elif st.session_state.last_config_hash != config_hash:
            # Clear all session state except for internal Streamlit keys if any
            # (Streamlit automatically re-runs when session_state is cleared)
            for key in list(st.session_state.keys()):
                if key != "last_config_hash":
                    del st.session_state[key]
            st.session_state.last_config_hash = config_hash
            st.rerun()

        # Load data
        with st.spinner("Loading influence data..."):
            with profile("load_data_cached_loader"):
                loader = get_cached_loader(config_hash, config)

        # Get the underlying data object for backwards compatibility
        data = loader.data

        st.divider()

        render_dataset_info(loader, config)

        st.divider()

        demo_split, top_k, obs_key, enable_detailed_hover, lambda_threshold = (
            render_settings(config, loader)
        )

        # Store in session state for access by rendering functions
        st.session_state["enable_detailed_hover"] = enable_detailed_hover

        # Apply soft thresholding if enabled (using cached version for speed)
        if lambda_threshold > 0:
            import copy

            from influence_visualizer.preprocessing import compute_sparsity

            # Create a shallow copy of data to avoid modifying cached object
            data = copy.copy(data)

            # Apply cached soft thresholding (fast after first call with same lambda)
            data.influence_matrix = apply_soft_threshold_cached(
                loader.data.influence_matrix, lambda_threshold
            )

            # Also apply to holdout if it exists
            if (
                hasattr(loader.data, "holdout_influence_matrix")
                and loader.data.holdout_influence_matrix is not None
            ):
                data.holdout_influence_matrix = apply_soft_threshold_cached(
                    loader.data.holdout_influence_matrix, lambda_threshold
                )

            final_sparsity = compute_sparsity(data.influence_matrix)

        st.divider()

    # Get annotation file path (hardcoded)
    annotation_file = get_annotation_file()

    # Main content: Tabs
    (
        tab_aggregated,
        tab_episode,
        tab_behaviors,
        tab_clustering,
        tab_local_behaviors,
        tab_advanced,
        tab_annotation,
        tab_video_export,
        tab_learning,
        tab_comparison,
        tab_ood_filtering,
        tab_diagnostics,
    ) = st.tabs(
        [
            "Aggregated Influence",
            "Episode Influence",
            "Behaviors",
            "Clustering",
            "Local Behaviors",
            "Analysis",
            "Annotation",
            "Video Export",
            "Learning",
            "Comparison",
            "OOD Filtering",
            "Diagnostics",
        ]
    )

    with tab_aggregated:
        with profile("tab_aggregated"):
            render_aggregated_tab(data, demo_split, annotation_file, selected_config_name)

    with tab_episode:
        with profile("tab_episode"):
            render_episode_tab(
                data, demo_split, top_k, obs_key, annotation_file, selected_config_name
            )

    with tab_behaviors:
        with profile("tab_behaviors"):
            render_behaviors_tab(
                data, demo_split, top_k, obs_key, annotation_file, selected_config_name
            )

    with tab_clustering:
        with profile("tab_clustering"):
            render_clustering_tab(data, demo_split, annotation_file, selected_config_name)

    with tab_local_behaviors:
        with profile("tab_local_behaviors"):
            render_local_behaviors_tab(data, demo_split, top_k, obs_key, annotation_file)

    with tab_advanced:
        with profile("tab_advanced"):
            render_advanced_analysis_tab(data, demo_split)

    with tab_annotation:
        with profile("tab_annotation"):
            render_annotation_tab(data, annotation_file, selected_config_name, obs_key)

    with tab_video_export:
        with profile("tab_video_export"):
            render_video_export_tab(data, selected_config_name, obs_key)

    with tab_learning:
        with profile("tab_learning"):
            render_learning_tab(
                data, demo_split, top_k, obs_key, annotation_file, selected_config_name
            )

    with tab_comparison:
        with profile("tab_comparison"):
            render_comparison_tab(selected_config_name, config)

    with tab_ood_filtering:
        with profile("tab_ood_filtering"):
            render_ood_filtering_tab(data, demo_split)

    with tab_diagnostics:
        with profile("tab_diagnostics"):
            from influence_visualizer.render_diagnostics import render_diagnostics_tab

            render_diagnostics_tab(data, demo_split)

    # Performance metrics: show after all tab content has run so current run's
    # profile() calls are included (sidebar runs first, so we print at the end)
    with st.sidebar:
        profiler = get_profiler()
        if st.button("Clear Metrics", key="clear_metrics"):
            profiler.reset()
        profiler.print_to_streamlit()


if __name__ == "__main__":
    main()

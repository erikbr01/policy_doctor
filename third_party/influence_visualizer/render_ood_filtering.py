"""OOD (Out-of-Distribution) filtering analysis for rollouts.

This module provides two approaches for detecting potential OOD behavior in rollouts:

1. Variance-based OOD detection: Analyzes the variance of influence across demo
   timesteps for each rollout timestep. High variance may indicate OOD behavior
   where the rollout doesn't match any demonstration consistently.

2. Diagonal detection: Uses template matching with a diagonal kernel (with noise)
   to find diagonal patterns in rollout-demo influence matrices. Strong diagonal
   patterns suggest the rollout closely follows a demonstration temporally.

DESIGN PATTERN NOTE:
- Follows the separation of concerns pattern
- Data collection and preprocessing happens in this Streamlit module
- Pure plotting logic lives in influence_visualizer.plotting
- Streamlit UI logic (st.button, st.checkbox, etc.) kept separate from plotting
"""

from typing import List, Tuple

import numpy as np
import streamlit as st
from scipy.ndimage import convolve, gaussian_filter

from influence_visualizer import plotting
from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.profiling import profile
from influence_visualizer.render_heatmaps import SplitType, get_split_data


def _get_n_action_steps(data: InfluenceData) -> int:
    """Temporal scaling: rollout sample = n_action_steps demo steps. Default 8 if missing."""
    return getattr(data, "n_action_steps", 8)


def compute_variance_across_demos(
    influence_matrix: np.ndarray,
    rollout_sample_indices: np.ndarray,
    ep_idxs: List[np.ndarray],
    apply_gaussian: bool = False,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """Compute variance of influence across demo timesteps for each rollout timestep.

    For each rollout timestep, computes the variance of influence values across
    all demonstration timesteps (flattened across all demos).

    Args:
        influence_matrix: Full influence matrix (rollout_samples x demo_samples)
        rollout_sample_indices: Indices of rollout samples to analyze
        ep_idxs: List of demo sample indices for each demo episode
        apply_gaussian: Whether to apply Gaussian smoothing before computing variance
        gaussian_sigma: Sigma for Gaussian smoothing

    Returns:
        Array of shape (num_rollout_timesteps,) containing variance values
    """
    num_rollout_timesteps = len(rollout_sample_indices)
    variances = np.zeros(num_rollout_timesteps)

    # Collect all demo sample indices
    all_demo_indices = np.concatenate(ep_idxs)

    # Extract the rollout-demos influence sub-matrix
    rollout_demos_influence = influence_matrix[
        np.ix_(rollout_sample_indices, all_demo_indices)
    ]

    # Apply Gaussian smoothing if requested
    if apply_gaussian and gaussian_sigma > 0:
        rollout_demos_influence = gaussian_filter(
            rollout_demos_influence, sigma=gaussian_sigma
        )

    # For each rollout timestep, compute variance across all demo timesteps
    for i in range(num_rollout_timesteps):
        variances[i] = np.var(rollout_demos_influence[i, :])

    return variances


def compute_variance_stats_across_rollouts(
    data: InfluenceData,
    split: SplitType = "train",
    apply_gaussian: bool = False,
    gaussian_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute variance statistics across all rollouts.

    For each rollout, computes variance per timestep, then aggregates across
    all rollouts to get mean, min, and max variance profiles.

    Args:
        data: InfluenceData object
        split: Which demo split to use
        apply_gaussian: Whether to apply Gaussian smoothing before computing variance
        gaussian_sigma: Sigma for Gaussian smoothing

    Returns:
        Tuple of (mean_variances, min_variances, max_variances) where each
        is an array aligned to the longest rollout length
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Find max rollout length
    max_rollout_len = max(ep.num_samples for ep in data.rollout_episodes)

    # Collect variance profiles for all rollouts
    all_variances = []

    for rollout_ep in data.rollout_episodes:
        rollout_sample_indices = np.arange(
            rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
        )

        variances = compute_variance_across_demos(
            influence_matrix,
            rollout_sample_indices,
            ep_idxs,
            apply_gaussian=apply_gaussian,
            gaussian_sigma=gaussian_sigma,
        )

        # Pad to max length with NaN
        padded_variances = np.full(max_rollout_len, np.nan)
        padded_variances[: len(variances)] = variances
        all_variances.append(padded_variances)

    all_variances = np.array(all_variances)  # Shape: (num_rollouts, max_rollout_len)

    # Compute statistics (ignoring NaN values)
    mean_variances = np.nanmean(all_variances, axis=0)
    min_variances = np.nanmin(all_variances, axis=0)
    max_variances = np.nanmax(all_variances, axis=0)

    return mean_variances, min_variances, max_variances


def create_diagonal_kernel(
    height: int,
    width: int,
    noise_scale: float = 0.1,
    diagonal_value: float = 1.0,
    smooth_sigma: float = 1.0,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Create a diagonal kernel with optional noise and smoothing.

    Creates a kernel with high values along the diagonal and optional
    Gaussian noise, then applies Gaussian smoothing.

    Args:
        height: Kernel height (rollout axis)
        width: Kernel width (demo axis)
        noise_scale: Scale of Gaussian noise to add (0 = no noise)
        diagonal_value: Value to place on the diagonal
        smooth_sigma: Sigma for Gaussian smoothing (0 = no smoothing)
        aspect_ratio: Temporal scaling ratio (n_action_steps).
                     Each row represents aspect_ratio steps in the column direction.
                     For aspect_ratio > 1, the diagonal becomes shallower (more horizontal).

    Returns:
        Diagonal kernel array of shape (height, width)
    """
    kernel = np.zeros((height, width))

    # Add scaled diagonal
    # For each row i, the corresponding column should be i * aspect_ratio
    for i in range(height):
        j = int(i * aspect_ratio)
        if j < width:
            kernel[i, j] = diagonal_value

    # Add Gaussian noise if requested
    if noise_scale > 0:
        noise = np.random.randn(height, width) * noise_scale
        kernel += noise

    # Apply Gaussian smoothing if requested
    if smooth_sigma > 0:
        kernel = gaussian_filter(kernel, sigma=smooth_sigma)

    return kernel


def compute_kernel_correlation(
    matrix: np.ndarray,
    kernel: np.ndarray,
) -> float:
    """Compute normalized correlation between matrix and kernel using convolution.

    Uses PyTorch conv2d with 'replicate' padding to handle size mismatches.

    Args:
        matrix: Input matrix
        kernel: Template kernel

    Returns:
        Maximum correlation value from the correlation map
    """
    import torch
    import torch.nn.functional as F

    # Normalize inputs
    matrix_norm = (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-10)
    kernel_norm = (kernel - np.mean(kernel)) / (np.std(kernel) + 1e-10)

    # Convert to torch tensors and add batch/channel dimensions
    # Shape: (1, 1, H, W)
    matrix_tensor = torch.from_numpy(matrix_norm).float().unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.from_numpy(kernel_norm).float().unsqueeze(0).unsqueeze(0)

    # Compute padding needed
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Manually pad the matrix with replicate mode
    # F.pad expects (left, right, top, bottom) for 2D
    padded_matrix = F.pad(matrix_tensor, (pad_w, pad_w, pad_h, pad_h), mode="replicate")

    # Apply convolution (no padding needed since we already padded)
    correlation_map = F.conv2d(
        input=padded_matrix, weight=kernel_tensor, bias=None, stride=1, padding=0
    )

    # Take the maximum correlation value as the global correlation
    correlation = float(torch.max(correlation_map).item())

    return correlation


def find_top_diagonals(
    correlation_map: np.ndarray,
    top_k: int = 5,
    aspect_ratio: float = 1.0,
) -> List[dict]:
    """Find top-k diagonal candidates in a correlation map.

    For each of the top-k peaks, tries all possible diagonal lengths
    and selects the one with the best score (length × mean_correlation).
    The diagonal slope is scaled by aspect_ratio: 1 step in y (rollout)
    corresponds to aspect_ratio steps in x (demo), matching temporal scaling.

    Args:
        correlation_map: Correlation map from sliding window (y=rollout, x=demo)
        top_k: Number of diagonal candidates to find
        aspect_ratio: Slope dx/dy (demo steps per rollout step), e.g. n_action_steps

    Returns:
        List of diagonal dicts with keys: 'score', 'length', 'mean_corr',
        'start_y', 'start_x', 'end_y', 'end_x', 'peak_y', 'peak_x'
    """
    if correlation_map.size == 0:
        return []

    h, w = correlation_map.shape
    ar = max(1e-6, aspect_ratio)  # avoid div by zero

    # Flatten and find top-k peaks
    flat_indices = np.argsort(correlation_map.flatten())[::-1]

    diagonals = []
    visited_peaks = set()

    for flat_idx in flat_indices:
        if len(diagonals) >= top_k:
            break

        peak_y, peak_x = np.unravel_index(flat_idx, (h, w))

        # Skip if too close to an already-found peak (within 3 pixels)
        too_close = False
        for vy, vx in visited_peaks:
            if abs(peak_y - vy) <= 3 and abs(peak_x - vx) <= 3:
                too_close = True
                break
        if too_close:
            continue

        visited_peaks.add((peak_y, peak_x))

        # Try all possible diagonal lengths through this peak
        # Diagonal: (y, x) = (peak_y + dy, peak_x + round(dy * aspect_ratio))
        # Bounds: up_left  dy in [-up_left_len, 0], down_right dy in [0, down_right_len]
        max_up_left = min(peak_y, int(peak_x / ar))
        max_down_right = min(h - peak_y - 1, int((w - peak_x - 1) / ar))

        best_score = -float("inf")
        best_diagonal = None

        for up_left_len in range(max_up_left + 1):
            for down_right_len in range(max_down_right + 1):
                total_len = up_left_len + down_right_len + 1
                if total_len < 2:
                    continue

                # Sample along scaled diagonal: dy in y, round(dy*ar) in x
                corr_values = []
                for dy in range(-up_left_len, down_right_len + 1):
                    y = peak_y + dy
                    x = peak_x + int(round(dy * ar))
                    x = max(0, min(w - 1, x))
                    corr_values.append(correlation_map[y, x])

                mean_corr = np.mean(corr_values)
                score = total_len * mean_corr

                if score > best_score:
                    best_score = score
                    start_y = peak_y - up_left_len
                    start_x = peak_x - int(round(up_left_len * ar))
                    end_y = peak_y + down_right_len
                    end_x = peak_x + int(round(down_right_len * ar))
                    best_diagonal = {
                        "score": score,
                        "length": total_len,
                        "mean_corr": mean_corr,
                        "start_y": start_y,
                        "start_x": start_x,
                        "end_y": end_y,
                        "end_x": end_x,
                        "peak_y": peak_y,
                        "peak_x": peak_x,
                    }

        if best_diagonal is not None:
            diagonals.append(best_diagonal)

    diagonals.sort(key=lambda d: d["score"], reverse=True)
    return diagonals


def compute_sliding_window_correlation(
    matrix: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """Compute sliding window correlation using fast convolution.

    Uses PyTorch convolution for efficient computation across all positions.

    Args:
        matrix: Input matrix (height x width)
        kernel: Template kernel (kernel_height x kernel_width)
        stride: Stride for sliding window

    Returns:
        Correlation map showing correlation at each window position
    """
    import torch
    import torch.nn.functional as F

    matrix_h, matrix_w = matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Check if kernel is larger than matrix
    if kernel_h > matrix_h or kernel_w > matrix_w:
        # Pad matrix to at least kernel size
        pad_h = max(0, kernel_h - matrix_h)
        pad_w = max(0, kernel_w - matrix_w)
        matrix = np.pad(matrix, ((0, pad_h), (0, pad_w)), mode="edge")

    # Normalize inputs
    matrix_norm = (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-10)
    kernel_norm = (kernel - np.mean(kernel)) / (np.std(kernel) + 1e-10)

    # Convert to torch tensors with batch/channel dimensions
    matrix_tensor = torch.from_numpy(matrix_norm).float().unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.from_numpy(kernel_norm).float().unsqueeze(0).unsqueeze(0)

    # Apply convolution (no padding for sliding window)
    correlation_map = F.conv2d(
        input=matrix_tensor, weight=kernel_tensor, stride=stride, padding=0
    )

    # Convert back to numpy and remove batch/channel dimensions
    correlation_map = correlation_map.squeeze().cpu().numpy()

    # Ensure it's at least 2D
    if correlation_map.ndim == 0:
        correlation_map = np.array([[correlation_map.item()]])
    elif correlation_map.ndim == 1:
        correlation_map = correlation_map.reshape(-1, 1)

    return correlation_map


def compute_diagonal_correlations(
    data: InfluenceData,
    rollout_idx: int,
    kernel_height: int,
    kernel_width: int,
    noise_scale: float,
    smooth_sigma: float,
    split: SplitType = "train",
    apply_gaussian: bool = False,
    gaussian_sigma: float = 1.0,
) -> Tuple[List[np.ndarray], List[float], np.ndarray, List[np.ndarray]]:
    """Compute diagonal kernel correlations for all rollout-demo pairs.

    Args:
        data: InfluenceData object
        rollout_idx: Index of rollout episode to analyze
        kernel_height: Height of diagonal kernel
        kernel_width: Width of diagonal kernel
        noise_scale: Noise scale for kernel
        smooth_sigma: Smoothing sigma for kernel
        split: Which demo split to use
        apply_gaussian: Whether to apply Gaussian smoothing to influence matrices
        gaussian_sigma: Sigma for Gaussian smoothing of influence matrices

    Returns:
        Tuple of (matrices, correlations, kernel, correlation_maps) where:
        - matrices: List of influence matrices for each demo
        - correlations: List of correlation scores for each demo
        - kernel: The diagonal kernel used
        - correlation_maps: List of sliding window correlation maps
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get rollout episode
    rollout_ep = data.rollout_episodes[rollout_idx]
    rollout_sample_indices = np.arange(
        rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
    )

    # Get temporal scaling ratio
    # Each rollout sample represents n_action_steps environment steps
    # Demo samples are already at environment step resolution
    aspect_ratio = float(_get_n_action_steps(data))

    # Create diagonal kernel with aspect ratio scaling
    kernel = create_diagonal_kernel(
        kernel_height,
        kernel_width,
        noise_scale,
        smooth_sigma=smooth_sigma,
        aspect_ratio=aspect_ratio,
    )

    matrices = []
    correlations = []
    correlation_maps = []

    for demo_ep_idx, demo_ep in enumerate(demo_episodes):
        demo_sample_idxs = ep_idxs[demo_ep_idx]

        # Get influence matrix for this rollout-demo pair
        rollout_demo_influence = influence_matrix[
            rollout_sample_indices[:, None], demo_sample_idxs
        ]

        # Apply Gaussian smoothing if requested
        if apply_gaussian and gaussian_sigma > 0:
            rollout_demo_influence = gaussian_filter(
                rollout_demo_influence, sigma=gaussian_sigma
            )

        # Compute global correlation (use original kernel, not resized)
        corr = compute_kernel_correlation(rollout_demo_influence, kernel)

        # Compute sliding window correlation map
        corr_map = compute_sliding_window_correlation(
            rollout_demo_influence, kernel, stride=1
        )

        matrices.append(rollout_demo_influence)
        correlations.append(corr)
        correlation_maps.append(corr_map)

    return matrices, correlations, kernel, correlation_maps


@st.fragment
def render_variance_based_ood_detection(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Fragment for variance-based OOD detection."""
    st.header("Variance-Based OOD Detection")

    st.markdown("""
    This analysis computes the **variance of influence across demonstration timesteps**
    for each rollout timestep. High variance may indicate that the rollout state is
    out-of-distribution (OOD) - it doesn't match any demonstration consistently.

    - **Low variance**: Rollout timestep has consistent influence pattern across demos
    - **High variance**: Rollout timestep has inconsistent influence, possibly OOD
    """)

    # Rollout selection
    st.subheader("Select Rollout Episode")

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
        key="variance_ood_episode_select",
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

    # Analysis section
    with st.expander("Load Variance Analysis", expanded=False):
        # Settings - Row 1
        col_metric, col_ref, col_smooth = st.columns(3)
        with col_metric:
            reference_metric = st.selectbox(
                "Reference line (red)",
                options=["mean", "min", "max"],
                index=0,
                key="variance_ood_reference_metric",
                help="Which aggregated statistic to show as reference line (computed across all rollouts)",
            )
        with col_ref:
            show_individual_demos = st.checkbox(
                "Show per-demo variance plots",
                value=False,
                key="variance_ood_show_individual",
                help="Show variance computed for each demo separately (in addition to aggregate)",
            )
        with col_smooth:
            apply_gaussian = st.checkbox(
                "Apply Gaussian smoothing",
                value=False,
                key="variance_ood_apply_gaussian",
                help="Apply Gaussian smoothing to influence matrices before computing variance",
            )

        # Settings - Row 2: Gaussian sigma and matrix display
        col_sigma, col_show_matrices = st.columns(2)
        with col_sigma:
            if apply_gaussian:
                gaussian_sigma = st.slider(
                    "Gaussian sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="variance_ood_gaussian_sigma",
                    help="Standard deviation of Gaussian kernel (larger = more smoothing)",
                )
            else:
                gaussian_sigma = 0.0
        with col_show_matrices:
            show_influence_matrices = st.checkbox(
                "Show individual influence matrices",
                value=True,
                key="variance_ood_show_matrices",
                help="Display individual rollout-demo influence matrices with variance plots on the side",
            )

        show_variance_key = f"show_variance_ood_{selected_episode_idx}_{demo_split}"
        if st.button(
            "Generate Variance Analysis",
            key=f"gen_variance_ood_{selected_episode_idx}_{demo_split}",
        ):
            st.session_state[show_variance_key] = True

        if st.session_state.get(show_variance_key, False):
            with profile("compute_variance_ood_detection"):
                influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
                    data, demo_split
                )

                # Compute variance for selected rollout
                rollout_sample_indices = np.arange(
                    selected_episode.sample_start_idx, selected_episode.sample_end_idx
                )

                variances = compute_variance_across_demos(
                    influence_matrix,
                    rollout_sample_indices,
                    ep_idxs,
                    apply_gaussian=apply_gaussian,
                    gaussian_sigma=gaussian_sigma,
                )

                # Compute reference statistics across all rollouts
                mean_variances, min_variances, max_variances = (
                    compute_variance_stats_across_rollouts(
                        data,
                        split=demo_split,
                        apply_gaussian=apply_gaussian,
                        gaussian_sigma=gaussian_sigma,
                    )
                )

                # Select reference line
                if reference_metric == "mean":
                    reference_line = mean_variances[: len(variances)]
                elif reference_metric == "min":
                    reference_line = min_variances[: len(variances)]
                else:  # max
                    reference_line = max_variances[: len(variances)]

                # Optionally compute per-demo variances
                per_demo_variances = None
                demo_labels = None
                if show_individual_demos:
                    per_demo_variances = []
                    demo_labels = []

                    for demo_ep_idx, demo_ep in enumerate(demo_episodes):
                        demo_sample_idxs = ep_idxs[demo_ep_idx]

                        # Get influence matrix for this rollout-demo pair
                        rollout_demo_influence = influence_matrix[
                            rollout_sample_indices[:, None], demo_sample_idxs
                        ]

                        # Apply Gaussian smoothing if requested
                        if apply_gaussian and gaussian_sigma > 0:
                            rollout_demo_influence = gaussian_filter(
                                rollout_demo_influence, sigma=gaussian_sigma
                            )

                        # Compute variance across demo timesteps for each rollout timestep
                        demo_variances = np.var(rollout_demo_influence, axis=1)
                        per_demo_variances.append(demo_variances)

                        # Get demo label
                        quality = "unlabelled"
                        if data.demo_quality_labels is not None:
                            quality = data.demo_quality_labels.get(
                                demo_ep.index, "unlabelled"
                            )
                        demo_labels.append(f"Demo {demo_ep.index} ({quality})")

                # Create visualization
                title_suffix = (
                    f" (Gaussian σ={gaussian_sigma})" if apply_gaussian else ""
                )
                fig = plotting.create_variance_ood_plot(
                    timesteps=np.arange(len(variances)),
                    variances=variances,
                    reference_line=reference_line,
                    reference_label=f"{reference_metric.capitalize()} (all rollouts)",
                    title=f"Variance Across Demos - Rollout {selected_episode_idx} [{status}]{title_suffix}",
                    per_demo_variances=per_demo_variances,
                    demo_labels=demo_labels,
                )

                st.plotly_chart(fig, width="stretch")

                # Show summary statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Mean Variance", f"{np.mean(variances):.6f}")
                with col_stat2:
                    st.metric("Max Variance", f"{np.max(variances):.6f}")
                with col_stat3:
                    st.metric("Min Variance", f"{np.min(variances):.6f}")
                with col_stat4:
                    st.metric("Std Variance", f"{np.std(variances):.6f}")

                # Show individual influence matrices if requested
                if show_influence_matrices:
                    st.divider()
                    st.subheader("Individual Rollout-Demo Influence Matrices")

                    # Pagination settings
                    col_page_settings1, col_page_settings2 = st.columns(2)
                    with col_page_settings1:
                        matrices_per_page = st.selectbox(
                            "Matrices per page",
                            options=[12, 16, 20, 24, 32],
                            index=0,
                            key="variance_ood_matrices_per_page",
                        )
                    with col_page_settings2:
                        total_demos = len(demo_episodes)
                        total_pages = (
                            total_demos + matrices_per_page - 1
                        ) // matrices_per_page
                        current_page = st.number_input(
                            f"Page (1-{total_pages})",
                            min_value=1,
                            max_value=total_pages,
                            value=1,
                            key="variance_ood_matrices_page",
                        )

                    start_idx = (current_page - 1) * matrices_per_page
                    end_idx = min(start_idx + matrices_per_page, total_demos)

                    # Prepare data for this page
                    page_matrices = []
                    page_variances = []
                    page_titles = []

                    for i in range(start_idx, end_idx):
                        demo_ep = demo_episodes[i]
                        demo_sample_idxs = ep_idxs[i]

                        # Get influence matrix
                        rollout_demo_influence = influence_matrix[
                            rollout_sample_indices[:, None], demo_sample_idxs
                        ]

                        # Apply Gaussian smoothing if requested
                        if apply_gaussian and gaussian_sigma > 0:
                            rollout_demo_influence = gaussian_filter(
                                rollout_demo_influence, sigma=gaussian_sigma
                            )

                        # Compute variance for this demo
                        demo_variance = np.var(rollout_demo_influence, axis=1)

                        page_matrices.append(rollout_demo_influence)
                        page_variances.append(demo_variance)

                        # Get demo label
                        quality = "unlabelled"
                        if data.demo_quality_labels is not None:
                            quality = data.demo_quality_labels.get(
                                demo_ep.index, "unlabelled"
                            )
                        page_titles.append(f"Demo {demo_ep.index} ({quality})")

                    # Create grid with variance plots
                    fig_grid = plotting.create_influence_matrices_with_variance(
                        matrices=page_matrices,
                        variances=page_variances,
                        titles=page_titles,
                        reference_variance=reference_line,
                        main_title=f"Rollout {selected_episode_idx} - Influence Matrices with Variance (Page {current_page}/{total_pages}){title_suffix}",
                        cols=4,
                    )
                    st.plotly_chart(fig_grid, width="stretch")


@st.fragment
def render_diagonal_detection(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Fragment for diagonal pattern detection using template matching."""
    st.header("Diagonal Pattern Detection")

    st.markdown("""
    This analysis uses **template matching with a diagonal kernel** to detect
    diagonal patterns in rollout-demo influence matrices.

    - **Strong diagonal pattern**: Rollout closely follows a demonstration temporally
    - **Weak diagonal pattern**: Rollout influence is more diffuse or follows different pattern

    The kernel can be configured with noise and smoothing to detect various diagonal structures.
    """)

    # Rollout selection
    st.subheader("Select Rollout Episode")

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
        key="diagonal_detection_episode_select",
    )
    selected_episode_idx = episode_options.index(selected_episode_str)
    selected_episode = data.rollout_episodes[selected_episode_idx]

    # Show episode info
    col1, col2, col3, col4 = st.columns(4)
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
    with col4:
        st.metric(
            "n_action_steps",
            _get_n_action_steps(data),
            help="Temporal scaling: each rollout sample = n_action_steps demo samples",
        )

    st.divider()

    # Kernel configuration
    st.subheader("Kernel Configuration")

    n_steps = _get_n_action_steps(data)
    st.info(
        f"**Temporal Scaling**: Kernel diagonal is scaled by aspect ratio = {n_steps}. "
        f"Each rollout timestep corresponds to {n_steps} demo timesteps."
    )

    col_kernel1, col_kernel2, col_kernel3 = st.columns(3)
    with col_kernel1:
        kernel_size = st.number_input(
            "Kernel size (rollout axis)",
            min_value=3,
            max_value=50,
            value=10,
            key="diagonal_kernel_size",
            help="Size along rollout axis; width is scaled by n_action_steps for temporal alignment",
        )
    with col_kernel2:
        noise_scale = st.slider(
            "Noise scale",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            key="diagonal_noise_scale",
            help="Scale of Gaussian noise to add to kernel (0 = pure diagonal)",
        )
    with col_kernel3:
        smooth_sigma = st.slider(
            "Smoothing sigma",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.5,
            key="diagonal_smooth_sigma",
            help="Gaussian smoothing applied to kernel (0 = no smoothing)",
        )

    # Scale kernel by temporal aspect: height = rollout axis, width = demo axis
    # 1 rollout step = n_steps demo steps, so kernel width = kernel_size * n_steps
    kernel_height = kernel_size
    kernel_width = max(kernel_size, int(round(kernel_size * n_steps)))

    st.caption(
        f"Kernel shape: **{kernel_height} × {kernel_width}** (rollout × demo, scaled by n_action_steps = {n_steps})"
    )

    st.divider()

    # Analysis section
    with st.expander("Load Diagonal Detection Analysis", expanded=False):
        # Display settings - Row 1
        col_display1, col_display2, col_display3, col_display4 = st.columns(4)
        with col_display1:
            max_demos_per_page = st.selectbox(
                "Demos per page",
                options=[12, 16, 20, 24, 32],
                index=0,
                key="diagonal_demos_per_page",
            )
        with col_display2:
            sort_by_correlation = st.checkbox(
                "Sort by correlation",
                value=True,
                key="diagonal_sort_by_correlation",
                help="Sort demos by correlation (highest first)",
            )
        with col_display3:
            show_diagonals = st.checkbox(
                "Show detected diagonals",
                value=True,
                key="diagonal_show_diagonals",
                help="Overlay detected diagonal lines on correlation maps",
            )
        with col_display4:
            diag_top_k = st.number_input(
                "Diagonals per demo",
                min_value=1,
                max_value=20,
                value=3,
                key="diagonal_top_k",
                help="Number of top diagonals to detect per demo",
            )

        # Display settings - Row 2: Gaussian smoothing for matrices
        col_smooth, col_sigma = st.columns(2)
        with col_smooth:
            apply_gaussian = st.checkbox(
                "Apply Gaussian smoothing to matrices",
                value=False,
                key="diagonal_apply_gaussian",
                help="Apply Gaussian smoothing to influence matrices before computing correlations",
            )
        with col_sigma:
            if apply_gaussian:
                gaussian_sigma = st.slider(
                    "Matrix smoothing sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="diagonal_gaussian_sigma",
                    help="Standard deviation of Gaussian kernel for matrix smoothing",
                )
            else:
                gaussian_sigma = 0.0

        show_diagonal_key = f"show_diagonal_{selected_episode_idx}_{demo_split}"
        if st.button(
            "Generate Diagonal Detection",
            key=f"gen_diagonal_{selected_episode_idx}_{demo_split}",
        ):
            st.session_state[show_diagonal_key] = True

        if st.session_state.get(show_diagonal_key, False):
            with profile("compute_diagonal_detection"):
                # Compute correlations
                matrices, correlations, kernel, correlation_maps = (
                    compute_diagonal_correlations(
                        data,
                        selected_episode_idx,
                        kernel_height,
                        kernel_width,
                        noise_scale,
                        smooth_sigma,
                        split=demo_split,
                        apply_gaussian=apply_gaussian,
                        gaussian_sigma=gaussian_sigma,
                    )
                )

                influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
                    data, demo_split
                )

                # Create list of (matrix, correlation, demo_idx, quality) tuples
                matrix_data = []
                for i, (matrix, corr) in enumerate(zip(matrices, correlations)):
                    demo_idx = demo_episodes[i].index
                    quality = "unlabelled"
                    if data.demo_quality_labels is not None:
                        quality = data.demo_quality_labels.get(demo_idx, "unlabelled")
                    matrix_data.append((matrix, corr, demo_idx, quality))

                # Sort by correlation if requested
                if sort_by_correlation:
                    matrix_data.sort(key=lambda x: x[1], reverse=True)

                # Show kernel
                st.subheader("Diagonal Kernel")
                col_kernel_vis, col_kernel_stats = st.columns([1, 1])

                with col_kernel_vis:
                    # Create a simple heatmap for the kernel
                    import plotly.graph_objects as go

                    kernel_fig = go.Figure(
                        data=go.Heatmap(
                            z=kernel,
                            colorscale=plotting.get_influence_colorscale(),
                            zmid=0,
                            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>",
                        )
                    )
                    kernel_fig.update_layout(
                        title="Diagonal Kernel",
                        xaxis_title="Demo t",
                        yaxis_title="Rollout t",
                        height=400,
                    )
                    st.plotly_chart(kernel_fig, width='stretch')

                with col_kernel_stats:
                    st.markdown("**Kernel Statistics**")
                    st.metric("Shape", f"{kernel.shape[0]} x {kernel.shape[1]}")
                    st.metric("Mean", f"{np.mean(kernel):.4f}")
                    st.metric("Std", f"{np.std(kernel):.4f}")
                    st.metric("Min", f"{np.min(kernel):.4f}")
                    st.metric("Max", f"{np.max(kernel):.4f}")

                st.divider()

                # Show correlation distribution
                st.subheader("Correlation Distribution")
                corr_array = np.array([x[1] for x in matrix_data])

                col_corr1, col_corr2, col_corr3 = st.columns(3)
                with col_corr1:
                    st.metric("Mean Correlation", f"{np.mean(corr_array):.4f}")
                with col_corr2:
                    st.metric("Max Correlation", f"{np.max(corr_array):.4f}")
                with col_corr3:
                    st.metric("Min Correlation", f"{np.min(corr_array):.4f}")

                st.divider()

                # Paginated grid of influence matrices with correlations
                st.subheader("Influence Matrices with Diagonal Correlation")

                total_demos = len(matrix_data)
                total_pages = (
                    total_demos + max_demos_per_page - 1
                ) // max_demos_per_page

                current_page = st.number_input(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="diagonal_page_number",
                )

                start_idx = (current_page - 1) * max_demos_per_page
                end_idx = min(start_idx + max_demos_per_page, total_demos)

                # Detect diagonals in all correlation maps if requested
                all_diagonals = []
                all_demo_labels = []
                if show_diagonals:
                    with st.spinner("Detecting diagonals..."):
                        aspect_ratio = float(_get_n_action_steps(data))
                        for i, (demo_ep, corr_map) in enumerate(
                            zip(demo_episodes, correlation_maps)
                        ):
                            diagonals = find_top_diagonals(
                                corr_map, top_k=diag_top_k, aspect_ratio=aspect_ratio
                            )
                            all_diagonals.append(diagonals)
                            quality = "unlabelled"
                            if data.demo_quality_labels is not None:
                                quality = data.demo_quality_labels.get(
                                    demo_ep.index, "unlabelled"
                                )
                            all_demo_labels.append(f"Demo {demo_ep.index} ({quality})")

                    # Show ranking chart
                    st.subheader("Diagonal Ranking Across All Demos")
                    ranking_fig = plotting.create_diagonal_ranking_chart(
                        all_diagonals=all_diagonals,
                        demo_labels=all_demo_labels,
                        title=f"Top Diagonals - Rollout {selected_episode_idx}",
                    )
                    st.plotly_chart(ranking_fig, width="stretch")

                    st.divider()

                # Prepare data for this page
                page_matrices = []
                page_correlation_maps = []
                page_diagonals = []
                page_titles = []

                for i in range(start_idx, end_idx):
                    matrix, corr, demo_idx, quality = matrix_data[i]
                    # Find the correlation map for this demo
                    demo_list_idx = [
                        j for j, ep in enumerate(demo_episodes) if ep.index == demo_idx
                    ][0]
                    corr_map = correlation_maps[demo_list_idx]

                    page_matrices.append(matrix)
                    page_correlation_maps.append(corr_map)
                    page_titles.append(
                        f"Demo {demo_idx} ({quality}) | Corr: {corr:.4f}"
                    )

                    # Add diagonals for this demo if available
                    if show_diagonals and all_diagonals:
                        page_diagonals.append(all_diagonals[demo_list_idx])
                    else:
                        page_diagonals.append([])

                # Create dual grid plot
                title_suffix = f" (Matrix σ={gaussian_sigma})" if apply_gaussian else ""
                fig = plotting.create_diagonal_detection_dual_grid(
                    matrices=page_matrices,
                    correlation_maps=page_correlation_maps,
                    titles=page_titles,
                    main_title=f"Rollout {selected_episode_idx} - Diagonal Detection (Page {current_page}/{total_pages}){title_suffix}",
                    cols=2,  # Reduce to 2 pairs per row since each pair takes 2 columns
                    diagonals_list=page_diagonals if show_diagonals else None,
                    show_diagonals=show_diagonals,
                )

                st.plotly_chart(fig, width="stretch")


def render_ood_filtering_tab(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Render OOD filtering tab with two sub-tabs."""
    # Sub-tabs for different OOD detection methods
    subtab_variance, subtab_diagonal = st.tabs(
        ["Variance-Based Detection", "Diagonal Detection"]
    )

    with subtab_variance:
        render_variance_based_ood_detection(data, demo_split)

    with subtab_diagonal:
        render_diagonal_detection(data, demo_split)

"""Local behaviors analysis: exploring intra-trajectory influence patterns.

This module explores behavior modes within individual influence matrices by:
1. Analyzing sliding windows within rollout-demo influence matrices
2. Computing different embeddings of sliding windows (flatten, HOG)
3. Visualizing influence matrix embeddings to find patterns

DESIGN PATTERN NOTE:
- This module follows the separation of concerns pattern used throughout the codebase
- Data collection and preprocessing happens in this Streamlit module
- Pure plotting logic lives in influence_visualizer.plotting (no Streamlit dependencies)
- When adding new visualizations:
  1. Create pure plotting functions in plotting/heatmaps.py or other plotting modules
  2. Export them in plotting/__init__.py
  3. Call them from this module with preprocessed data
  4. Keep Streamlit UI logic (st.button, st.checkbox, etc.) separate from plotting
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter, sobel
from sklearn.manifold import TSNE

from influence_visualizer import plotting
from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.profiling import profile
from influence_visualizer.render_heatmaps import SplitType, get_split_data


def compute_hog_features(window: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Compute Histogram of Oriented Gradients (HOG) for a 2D window.

    Args:
        window: 2D array of shape (height, width)
        n_bins: Number of orientation bins

    Returns:
        Flattened HOG feature vector
    """
    # Compute gradients
    gx = sobel(window, axis=1, mode="constant")
    gy = sobel(window, axis=0, mode="constant")

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    # Map orientations to [0, 2*pi)
    orientation = (orientation + 2 * np.pi) % (2 * np.pi)

    # Create histogram
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    histogram = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (orientation >= bin_edges[i]) & (orientation < bin_edges[i + 1])
        histogram[i] = np.sum(magnitude[mask])

    # Normalize
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)

    return histogram


def extract_sliding_window_features(
    influence_matrix: np.ndarray,
    window_height: int,
    window_width: int,
    stride: int = 1,
    method: Literal["flatten", "hog"] = "flatten",
    hog_bins: int = 8,
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract features from sliding windows over influence matrix.

    Args:
        influence_matrix: 2D array of shape (rollout_timesteps, demo_timesteps)
        window_height: Height of sliding window
        window_width: Width of sliding window
        stride: Stride for sliding window
        method: Embedding method ("flatten" or "hog")
        hog_bins: Number of bins for HOG (only used if method="hog")

    Returns:
        Tuple of (features, metadata) where:
        - features: array of shape (num_windows, feature_dim)
        - metadata: list of dicts with window position info
    """
    H, W = influence_matrix.shape
    features = []
    metadata = []

    for i in range(0, H - window_height + 1, stride):
        for j in range(0, W - window_width + 1, stride):
            window = influence_matrix[i : i + window_height, j : j + window_width]

            if method == "flatten":
                feature = window.flatten()
            elif method == "hog":
                feature = compute_hog_features(window, n_bins=hog_bins)
            else:
                raise ValueError(f"Unknown method: {method}")

            features.append(feature)
            metadata.append(
                {
                    "rollout_start": i,
                    "rollout_end": i + window_height,
                    "demo_start": j,
                    "demo_end": j + window_width,
                    "mean_influence": np.mean(window),
                    "std_influence": np.std(window),
                    "max_influence": np.max(window),
                    "min_influence": np.min(window),
                }
            )

    return np.array(features), metadata


def extract_influence_matrix_embeddings(
    data: InfluenceData,
    rollout_idx: int,
    split: SplitType = "train",
    method: Literal["flatten", "hog", "stats", "resized", "singular_values"] = "stats",
    hog_bins: int = 8,
    resize_shape: Tuple[int, int] = (20, 20),
    n_components: int = 10,
) -> Tuple[np.ndarray, List[Dict]]:
    """Extract embeddings for all influence matrices of a given rollout.

    For a single rollout, there are multiple influence matrices (one per demo).
    This function extracts embeddings for each rollout-demo influence matrix.

    Args:
        data: InfluenceData object
        rollout_idx: Index of rollout episode
        split: Which demo split to use
        method: Embedding method ("resized", "hog", "stats", or "singular_values")
        hog_bins: Number of bins for HOG
        resize_shape: Target shape for resizing matrices (only used if method="resized")
        n_components: Number of singular values to use (only used if method="singular_values")

    Returns:
        Tuple of (embeddings, metadata) where:
        - embeddings: array of shape (num_demos, feature_dim)
        - metadata: list of dicts with demo info
    """
    from scipy.ndimage import zoom

    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get rollout episode
    rollout_ep = data.rollout_episodes[rollout_idx]
    rollout_sample_indices = np.arange(
        rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
    )

    embeddings = []
    metadata = []

    for demo_ep_idx, demo_ep in enumerate(demo_episodes):
        demo_sample_idxs = ep_idxs[demo_ep_idx]

        # Get influence matrix for this rollout-demo pair
        # Shape: (rollout_timesteps, demo_timesteps)
        rollout_demo_influence = influence_matrix[
            rollout_sample_indices[:, None], demo_sample_idxs
        ]

        if method == "resized":
            # Resize to a common shape and then flatten
            h, w = rollout_demo_influence.shape
            target_h, target_w = resize_shape
            zoom_factors = (target_h / h, target_w / w)
            resized = zoom(rollout_demo_influence, zoom_factors, order=1)
            embedding = resized.flatten()
        elif method == "hog":
            embedding = compute_hog_features(rollout_demo_influence, n_bins=hog_bins)
        elif method == "stats":
            # Statistical features
            embedding = np.array(
                [
                    np.mean(rollout_demo_influence),
                    np.std(rollout_demo_influence),
                    np.min(rollout_demo_influence),
                    np.max(rollout_demo_influence),
                    np.median(rollout_demo_influence),
                    np.percentile(rollout_demo_influence, 25),
                    np.percentile(rollout_demo_influence, 75),
                ]
            )
        elif method == "singular_values":
            # Use top-k singular values as embedding
            U, s, Vt = np.linalg.svd(rollout_demo_influence, full_matrices=False)
            if len(s) < n_components:
                embedding = np.zeros(n_components)
                embedding[: len(s)] = s
            else:
                embedding = s[:n_components]
        else:
            raise ValueError(f"Unknown method: {method}")

        embeddings.append(embedding)

        # Get quality label if available
        quality_label = "unknown"
        if data.demo_quality_labels is not None:
            quality_label = data.demo_quality_labels.get(demo_ep.index, "unknown")

        metadata.append(
            {
                "demo_idx": demo_ep.index,
                "quality_label": quality_label,
                "num_samples": demo_ep.num_samples,
                "mean_influence": np.mean(rollout_demo_influence),
                "std_influence": np.std(rollout_demo_influence),
            }
        )

    return np.array(embeddings), metadata


def visualize_tsne_embeddings(
    embeddings: np.ndarray,
    metadata: List[Dict],
    color_by: str = "mean_influence",
    title: str = "t-SNE Visualization of Embeddings",
    perplexity: int = 30,
) -> go.Figure:
    """Visualize embeddings using t-SNE.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        metadata: List of metadata dicts for each embedding
        color_by: Metadata key to use for coloring points
        title: Plot title
        perplexity: t-SNE perplexity parameter

    Returns:
        Plotly figure
    """
    # Adjusted perplexity if needed
    perplexity = min(perplexity, len(embeddings) - 1, 50)

    if len(embeddings) < 2:
        # Not enough data for t-SNE
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data points for t-SNE (need at least 2)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Run t-SNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure using pure plotting function
    fig = plotting.create_generic_tsne_plot(
        embeddings_2d=embeddings_2d,
        metadata=metadata,
        color_by=color_by,
        title=title,
        categorical_color=(color_by == "quality_label"),
    )

    return fig


def visualize_influence_matrix_grid(
    data: InfluenceData,
    rollout_idx: int,
    split: SplitType = "train",
    max_demos: int = 12,
    start_demo: int = 0,
) -> go.Figure:
    """Visualize a grid of influence matrices for a single rollout.

    Shows all rollout-demo influence matrices for the selected rollout.

    Args:
        data: InfluenceData object
        rollout_idx: Index of rollout episode
        split: Which demo split to use
        max_demos: Maximum number of demos to show

    Returns:
        Plotly figure with subplots
    """
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    # Get rollout episode info
    rollout_ep = data.rollout_episodes[rollout_idx]
    rollout_sample_indices = np.arange(
        rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
    )

    # Limit number of demos
    end_demo = min(start_demo + max_demos, len(demo_episodes))
    num_demos = end_demo - start_demo

    matrices = []
    titles = []
    for i in range(start_demo, end_demo):
        demo_idx_val = demo_episodes[i].index
        demo_sample_idxs = ep_idxs[i]

        # Get influence matrix for this rollout-demo pair
        rollout_demo_influence = influence_matrix[
            rollout_sample_indices[:, None], demo_sample_idxs
        ]
        matrices.append(rollout_demo_influence)

        quality = "unlabelled"
        if data.demo_quality_labels is not None:
            quality = data.demo_quality_labels.get(demo_idx_val, "unlabelled")
        titles.append(f"Demo {demo_idx_val} ({quality})")

    fig = plotting.create_influence_grid_plot(
        matrices=matrices,
        titles=titles,
        main_title=f"Rollout {rollout_idx} - All Influence Matrices",
        cols=4,
    )

    return fig


def visualize_state_histograms(
    data: InfluenceData,
    split: SplitType = "train",
) -> go.Figure:
    """Visualize histograms of each state dimension for training vs rollout samples.

    Args:
        data: InfluenceData object
        split: Which demo split to use for training samples

    Returns:
        Plotly figure with histograms
    """
    from influence_visualizer.render_heatmaps import get_split_data

    # ... (same data collection logic as before, just using pure plotting function at the end) ...
    # [Skipping repeat of large data collection block for brevity in thoughts, but implementing it fully in the tool call]
    # (Actually I should provide the full replacement)

    # Collect ALL training observations
    _, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    train_obs_list = []
    if split == "train":
        total_train_samples = len(data.demo_sample_infos)
        sample_offset = 0
    elif split == "holdout":
        total_train_samples = len(data.holdout_sample_infos)
        sample_offset = len(data.demo_sample_infos)
    else:  # both
        total_train_samples = len(data.demo_sample_infos) + len(
            data.holdout_sample_infos
        )
        sample_offset = 0

    for idx in range(sample_offset, sample_offset + total_train_samples):
        obs = data.get_demo_obs(idx)
        if obs is not None:
            train_obs_list.append(obs)

    # Collect ALL rollout observations
    rollout_obs_list = []
    for idx in range(data.num_rollout_samples):
        obs = data.get_rollout_obs(idx)
        if obs is not None:
            rollout_obs_list.append(obs)

    if len(train_obs_list) == 0 or len(rollout_obs_list) == 0:
        # Return empty figure with annotation
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_annotation(
            text="No observation data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    train_obs = np.array(train_obs_list)
    rollout_obs = np.array(rollout_obs_list)
    state_dim = train_obs.shape[1]

    # Labels logic (same as before)
    state_labels = []
    if state_dim == 5:
        state_labels = ["Agent X", "Agent Y", "Block X", "Block Y", "Block Angle"]
    elif state_dim == 9:
        state_labels = [
            "Agent X",
            "Agent Y",
            "Block X",
            "Block Y",
            "Block Angle",
            "Block Sin",
            "Block Cos",
            "Agent Vel X",
            "Agent Vel Y",
        ]
    elif state_dim == 10:
        state_labels = [
            "Agent X",
            "Agent Y",
            "Agent Vel X",
            "Agent Vel Y",
            "Block X",
            "Block Y",
            "Block Vel X",
            "Block Vel Y",
            "Block Angle",
            "Block Angular Vel",
        ]
    elif state_dim == 20:
        state_labels = [
            "Agent X",
            "Agent Y",
            "Block X",
            "Block Y",
            "Block Angle",
            "Block Sin",
            "Block Cos",
            "Agent Vel X",
            "Agent Vel Y",
        ] + [f"Keypoint {i // 2} {'X' if i % 2 == 0 else 'Y'}" for i in range(11)]
    else:
        state_labels = [f"State {i}" for i in range(state_dim)]

    # Use pure plotting function
    fig = plotting.create_comparison_histograms(
        train_data=train_obs,
        rollout_data=rollout_obs,
        labels=state_labels,
        title="State Distribution Comparison (Training vs Rollout)",
    )

    return fig


def visualize_action_histograms(
    data: InfluenceData,
    split: SplitType = "train",
) -> go.Figure:
    """Visualize histograms of each action dimension for training vs rollout samples.

    Args:
        data: InfluenceData object
        split: Which demo split to use for training samples

    Returns:
        Plotly figure with histograms
    """
    from influence_visualizer.render_heatmaps import get_split_data

    # Collect ALL training actions
    _, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    train_action_list = []
    if split == "train":
        total_train_samples = len(data.demo_sample_infos)
        sample_offset = 0
    elif split == "holdout":
        total_train_samples = len(data.holdout_sample_infos)
        sample_offset = len(data.demo_sample_infos)
    else:  # both
        total_train_samples = len(data.demo_sample_infos) + len(
            data.holdout_sample_infos
        )
        sample_offset = 0

    for idx in range(sample_offset, sample_offset + total_train_samples):
        action_chunk = data.get_demo_action_chunk(idx)
        if action_chunk is not None:
            train_action_list.append(action_chunk.flatten())

    # Collect ALL rollout actions
    rollout_action_list = []
    for idx in range(data.num_rollout_samples):
        action = data.get_rollout_action(idx)
        if action is not None:
            if isinstance(action, np.ndarray):
                rollout_action_list.append(action.flatten())
            else:
                rollout_action_list.append(np.array([action]).flatten())

    if len(train_action_list) == 0 or len(rollout_action_list) == 0:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_annotation(
            text="No action data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    train_actions = np.array(train_action_list)
    rollout_actions = np.array(rollout_action_list)
    action_dim = train_actions.shape[1]

    # Action labels
    action_labels = [f"Action {i}" for i in range(action_dim)]
    task_name = data.cfg.get("task_name", "")
    if "pusht" in task_name.lower() and action_dim == 2:
        action_labels = ["Target X", "Target Y"]

    # Use pure plotting function
    fig = plotting.create_comparison_histograms(
        train_data=train_actions,
        rollout_data=rollout_actions,
        labels=action_labels,
        title="Action Distribution Comparison (Training vs Rollout)",
    )

    return fig


def visualize_state_action_tsne(
    data: InfluenceData,
    split: SplitType = "train",
    perplexity: int = 30,
) -> go.Figure:
    """Visualize t-SNE of state-action distributions for ALL training and rollout samples.

    Args:
        data: InfluenceData object
        split: Which demo split to use for training samples
        perplexity: t-SNE perplexity parameter

    Returns:
        Plotly figure with t-SNE visualization
    """
    from influence_visualizer.render_heatmaps import get_split_data

    # Collect ALL training samples
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

    train_state_actions = []

    # Determine which training samples to include based on split
    if split == "train":
        total_train_samples = len(data.demo_sample_infos)
        sample_offset = 0
    elif split == "holdout":
        total_train_samples = len(data.holdout_sample_infos)
        sample_offset = len(data.demo_sample_infos)
    else:  # both
        total_train_samples = len(data.demo_sample_infos) + len(
            data.holdout_sample_infos
        )
        sample_offset = 0

    train_indices = []  # Track sample indices
    train_episode_indices = []  # Track episode indices

    for idx in range(sample_offset, sample_offset + total_train_samples):
        state_action = data.get_demo_state_action(idx)
        if state_action is not None:
            train_state_actions.append(state_action)
            train_indices.append(idx)
            # Get episode index for this sample
            sample_info = data.get_demo_sample_info(idx)
            train_episode_indices.append(sample_info.episode_idx)

    # Collect ALL rollout samples
    rollout_state_actions = []
    rollout_indices = []  # Track sample indices
    rollout_episode_indices = []  # Track episode indices
    total_rollout_samples = data.num_rollout_samples

    for idx in range(total_rollout_samples):
        state_action = data.get_rollout_state_action(idx)
        if state_action is not None:
            rollout_state_actions.append(state_action)
            rollout_indices.append(idx)
            # Get episode index for this sample
            sample_info = data.get_rollout_sample_info(idx)
            rollout_episode_indices.append(sample_info.episode_idx)

    # Check if we have enough data
    if len(train_state_actions) == 0 and len(rollout_state_actions) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No state-action data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    if len(train_state_actions) + len(rollout_state_actions) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data points for t-SNE (need at least 2)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Combine all state-action vectors
    all_state_actions = train_state_actions + rollout_state_actions

    # Ensure all vectors have the same dimensionality
    max_dim = max(sa.shape[0] for sa in all_state_actions)
    padded_state_actions = []
    for sa in all_state_actions:
        if sa.shape[0] < max_dim:
            padded = np.zeros(max_dim)
            padded[: sa.shape[0]] = sa
            padded_state_actions.append(padded)
        else:
            padded_state_actions.append(sa[:max_dim])

    X = np.array(padded_state_actions)

    # Adjust perplexity if needed
    perplexity = min(perplexity, len(X) - 1, 50)

    # Run t-SNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)

    # Prepare metadata for t-SNE hover
    merged_metadata = []

    # Training samples metadata
    for i, (ep_idx, sample_idx) in enumerate(zip(train_episode_indices, train_indices)):
        sample_info = data.get_demo_sample_info(sample_idx)
        quality = "unknown"
        if data.demo_quality_labels and ep_idx in data.demo_quality_labels:
            quality = data.demo_quality_labels[ep_idx]

        merged_metadata.append(
            {
                "Source": "Training",
                "Episode": ep_idx,
                "Quality": quality,
                "SampleIdx": sample_idx,
                "Timestep": sample_info.timestep,
            }
        )

    # Rollout samples metadata
    for i, (ep_idx, sample_idx) in enumerate(
        zip(rollout_episode_indices, rollout_indices)
    ):
        sample_info = data.get_rollout_sample_info(sample_idx)
        status = "unknown"
        if ep_idx < len(data.rollout_episodes):
            rollout_ep = data.rollout_episodes[ep_idx]
            if rollout_ep.success is not None:
                status = "Success" if rollout_ep.success else "Failure"

        merged_metadata.append(
            {
                "Source": "Rollout",
                "Episode": ep_idx,
                "Status": status,
                "SampleIdx": sample_idx,
                "Timestep": sample_info.timestep,
            }
        )

    # Use pure plotting function
    fig = plotting.create_generic_tsne_plot(
        embeddings_2d=X_2d,
        metadata=merged_metadata,
        color_by="Source",
        title="t-SNE of State-Action Distributions (Training vs Rollout)",
        categorical_color=True,
    )

    return fig


@st.fragment
def render_influence_matrices_section(
    data: InfluenceData,
    demo_split: SplitType,
    selected_episode_idx: int,
):
    """Fragment for influence matrices grid visualization."""
    st.header("2. All Influence Matrices for This Rollout")

    st.markdown("""
    Each heatmap shows the influence matrix between this rollout and one demonstration.
    Rows are rollout timesteps, columns are demo timesteps.
    """)

    with st.expander("Load Influence Matrices Grid", expanded=False):
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )
        total_demos = len(demo_episodes)

        # Get rollout length for cutoff slider
        rollout_ep = data.rollout_episodes[selected_episode_idx]
        max_rollout_len = rollout_ep.num_samples

        col_settings1, col_settings2, col_settings3, col_settings4, col_settings5 = (
            st.columns(5)
        )
        with col_settings1:
            demos_per_page = st.selectbox(
                "Demos per page",
                options=[12, 16, 20, 24, 32],
                index=0,
                key="local_behaviors_demos_per_page",
            )
        with col_settings2:
            total_pages = (total_demos + demos_per_page - 1) // demos_per_page
            current_page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="local_behaviors_page_number",
            )
        with col_settings3:
            colorscheme = st.selectbox(
                "Color scheme",
                options=["Red-White-Green", "Red-White-Blue"],
                index=0,
                key="local_behaviors_colorscheme",
            )
        with col_settings4:
            apply_gaussian_individual = st.checkbox(
                "Apply Gaussian smoothing",
                value=False,
                key="local_behaviors_individual_gaussian_smoothing",
                help="Denoise matrices by convolving with a Gaussian kernel",
            )
        with col_settings5:
            if apply_gaussian_individual:
                gaussian_sigma_individual = st.slider(
                    "Gaussian sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="local_behaviors_individual_gaussian_sigma",
                    help="Standard deviation of Gaussian kernel (larger = more smoothing)",
                )
            else:
                gaussian_sigma_individual = 1.0

        # Rollout timestep cutoff
        st.markdown("**Rollout Timestep Filtering:**")
        col_cutoff1, col_cutoff2 = st.columns([1, 3])
        with col_cutoff1:
            enable_cutoff = st.checkbox(
                "Enable cutoff",
                value=False,
                key="local_behaviors_enable_cutoff",
                help="Enable rollout timestep cutoff to filter out late timesteps",
            )
        with col_cutoff2:
            if enable_cutoff:
                rollout_cutoff = st.slider(
                    "Max rollout timestep to display",
                    min_value=1,
                    max_value=max_rollout_len,
                    value=max_rollout_len,
                    key="local_behaviors_rollout_cutoff",
                    help="Only show rollout timesteps up to this value (useful to filter out late timesteps with large influence values)",
                )
            else:
                rollout_cutoff = max_rollout_len

        # Calculate which demos to show on this page
        start_demo_idx = (current_page - 1) * demos_per_page
        end_demo_idx = min(start_demo_idx + demos_per_page, total_demos)

        if enable_cutoff:
            st.caption(
                f"Showing demos {start_demo_idx + 1}-{end_demo_idx} of {total_demos} | "
                f"Rollout timesteps: 0-{rollout_cutoff - 1} (cutoff enabled)"
            )
        else:
            st.caption(
                f"Showing demos {start_demo_idx + 1}-{end_demo_idx} of {total_demos}"
            )

        show_matrices_key = f"show_matrices_grid_{selected_episode_idx}_{demo_split}"
        if st.button(
            "Generate Influence Matrices Grid",
            key=f"gen_influence_matrices_{selected_episode_idx}_{demo_split}",
        ):
            st.session_state[show_matrices_key] = True

        if st.session_state.get(show_matrices_key, False):
            # Prepare matrices and titles for grid
            rollout_sample_indices = np.arange(
                rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
            )
            # Apply rollout timestep cutoff
            rollout_sample_indices = rollout_sample_indices[:rollout_cutoff]

            matrices = []
            titles = []
            for i in range(start_demo_idx, end_demo_idx):
                demo_idx_val = demo_episodes[i].index
                demo_sample_idxs = ep_idxs[i]

                # Get influence matrix
                rollout_demo_influence = influence_matrix[
                    rollout_sample_indices[:, None], demo_sample_idxs
                ]

                # Apply Gaussian smoothing if requested
                if apply_gaussian_individual:
                    rollout_demo_influence = gaussian_filter(
                        rollout_demo_influence, sigma=gaussian_sigma_individual
                    )

                matrices.append(rollout_demo_influence)

                quality = "unlabelled"
                if data.demo_quality_labels is not None:
                    quality = data.demo_quality_labels.get(demo_idx_val, "unlabelled")
                titles.append(f"Demo {demo_idx_val} ({quality})")

            # Use pure plotting function
            title_parts = []
            if enable_cutoff:
                title_parts.append(f"rollout t ≤ {rollout_cutoff - 1}")
            if apply_gaussian_individual:
                title_parts.append(f"Gaussian σ={gaussian_sigma_individual}")
            title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""

            fig = plotting.create_influence_grid_plot(
                matrices=matrices,
                titles=titles,
                main_title=f"Rollout {selected_episode_idx} - Influence Matrices (Page {current_page}/{total_pages}){title_suffix}",
                colorscale=colorscheme,
                cols=4,
            )
            st.plotly_chart(fig, width="stretch")


@st.fragment
def render_aggregated_influence_grid(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Fragment for aggregated influence matrices grid across all rollouts."""
    st.header("Aggregated Influence Across All Rollouts")

    st.markdown("""
    Each heatmap shows the aggregated influence for one rollout, summing influence values
    across all demonstrations for each (rollout_timestep, demo_timestep) combination.
    """)

    with st.expander("Load Aggregated Influence Grid", expanded=False):
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )
        total_rollouts = len(data.rollout_episodes)

        # Controls
        col_settings1, col_settings2, col_settings3, col_settings4, col_settings5 = (
            st.columns(5)
        )
        with col_settings1:
            rollouts_per_page = st.selectbox(
                "Rollouts per page",
                options=[12, 16, 20, 24, 32],
                index=0,
                key="aggregated_rollouts_per_page",
            )
        with col_settings2:
            total_pages = (total_rollouts + rollouts_per_page - 1) // rollouts_per_page
            current_page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="aggregated_page_number",
            )
        with col_settings3:
            show_std_dev = st.checkbox(
                "Show Std Dev plots",
                value=True,
                key="aggregated_show_std_dev",
                help="Show standard deviation plots next to heatmaps",
            )
        with col_settings4:
            apply_gaussian = st.checkbox(
                "Apply Gaussian smoothing",
                value=False,
                key="aggregated_gaussian_smoothing",
                help="Denoise matrices by convolving with a Gaussian kernel",
            )
        with col_settings5:
            if apply_gaussian:
                gaussian_sigma = st.slider(
                    "Gaussian sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="aggregated_gaussian_sigma",
                    help="Standard deviation of Gaussian kernel (larger = more smoothing)",
                )
            else:
                gaussian_sigma = 1.0

        # Calculate which rollouts to show on this page
        start_rollout_idx = (current_page - 1) * rollouts_per_page
        end_rollout_idx = min(start_rollout_idx + rollouts_per_page, total_rollouts)

        st.caption(
            f"Showing rollouts {start_rollout_idx + 1}-{end_rollout_idx} of {total_rollouts}"
        )

        show_aggregated_grid_key = f"show_aggregated_grid_{demo_split}"
        if st.button(
            "Generate Aggregated Influence Grid",
            key=f"gen_aggregated_influence_grid_{demo_split}",
        ):
            st.session_state[show_aggregated_grid_key] = True

        if st.session_state.get(show_aggregated_grid_key, False):
            # Prepare data for aggregated grid
            max_demo_len = max(ep_lens)
            num_rollouts_on_page = end_rollout_idx - start_rollout_idx

            aggregated_matrices = []
            std_matrices = []
            titles = []

            for i in range(start_rollout_idx, end_rollout_idx):
                ep = data.rollout_episodes[i]
                rollout_sample_indices = np.arange(
                    ep.sample_start_idx, ep.sample_end_idx
                )
                rollout_len = len(rollout_sample_indices)

                agg_matrix = np.zeros((rollout_len, max_demo_len))
                for demo_idx, demo_ep in enumerate(demo_episodes):
                    demo_sample_idxs = ep_idxs[demo_idx]
                    demo_len = len(demo_sample_idxs)
                    rollout_demo_influence = influence_matrix[
                        rollout_sample_indices[:, None], demo_sample_idxs
                    ]
                    agg_matrix[:, :demo_len] += rollout_demo_influence

                if apply_gaussian:
                    agg_matrix = gaussian_filter(agg_matrix, sigma=gaussian_sigma)

                aggregated_matrices.append(agg_matrix)

                if show_std_dev:
                    std_matrices.append(np.std(agg_matrix, axis=1).reshape(-1, 1))

                status = "✓" if ep.success else "✗" if ep.success is not None else "?"
                titles.append(f"Rollout {ep.index} [{status}]")

            # Use pure plotting function
            title_suffix = f" (Gaussian σ={gaussian_sigma})" if apply_gaussian else ""
            fig = plotting.create_aggregated_influence_grid(
                matrices=aggregated_matrices,
                std_matrices=std_matrices if show_std_dev else None,
                titles=titles,
                main_title=f"Aggregated Influence Heatmaps (Page {current_page}/{total_pages}){title_suffix}",
                rollouts_per_row=2 if show_std_dev else 4,
            )
            st.plotly_chart(fig, width="stretch")


@st.fragment
def render_sliding_window_section(
    data: InfluenceData,
    demo_split: SplitType,
    selected_episode_idx: int,
):
    """Fragment for sliding window analysis."""
    st.header("3. Sliding Window Analysis")

    st.markdown("""
    Extract features from sliding windows over a single influence matrix.
    This helps identify local patterns and behavior modes within the matrix.
    """)

    with st.expander("Load Sliding Window Analysis", expanded=False):
        # Select a demo for sliding window analysis
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )
        selected_episode = data.rollout_episodes[selected_episode_idx]

        demo_options = [
            f"Demo {ep.index} ({ep.num_samples} samples)" for ep in demo_episodes
        ]
        selected_demo_str = st.selectbox(
            "Choose a demonstration for sliding window analysis:",
            options=demo_options,
            index=0,
            key="local_behaviors_demo_select",
        )
        selected_demo_idx = demo_options.index(selected_demo_str)
        selected_demo = demo_episodes[selected_demo_idx]

        # Sliding window parameters
        col_wh, col_ww, col_stride, col_method = st.columns(4)
        with col_wh:
            window_height = st.number_input(
                "Window height (rollout timesteps)",
                min_value=2,
                max_value=selected_episode.num_samples,
                value=min(5, selected_episode.num_samples),
                key="local_behaviors_window_height",
            )
        with col_ww:
            window_width = st.number_input(
                "Window width (demo timesteps)",
                min_value=2,
                max_value=selected_demo.num_samples,
                value=min(5, selected_demo.num_samples),
                key="local_behaviors_window_width",
            )
        with col_stride:
            stride = st.number_input(
                "Stride",
                min_value=1,
                max_value=10,
                value=2,
                key="local_behaviors_stride",
            )
        with col_method:
            window_method = st.selectbox(
                "Embedding method",
                options=["flatten", "hog"],
                index=0,
                key="local_behaviors_window_method",
                help="flatten: raw pixel values | hog: histogram of gradient orientations",
            )

        show_sliding_key = f"show_sliding_window_{selected_episode_idx}_{demo_split}"
        if st.button(
            "Generate Sliding Window Analysis",
            key=f"gen_sliding_window_{selected_episode_idx}_{selected_demo.index}",
        ):
            st.session_state[show_sliding_key] = True

        if st.session_state.get(show_sliding_key, False):
            # Get the influence matrix for this rollout-demo pair
            rollout_sample_indices = np.arange(
                selected_episode.sample_start_idx, selected_episode.sample_end_idx
            )
            demo_sample_idxs = ep_idxs[selected_demo_idx]
            rollout_demo_influence = influence_matrix[
                rollout_sample_indices[:, None], demo_sample_idxs
            ]

            # Extract sliding window features
            with profile("extract_sliding_window_features"):
                window_features, window_metadata = extract_sliding_window_features(
                    rollout_demo_influence,
                    window_height=window_height,
                    window_width=window_width,
                    stride=stride,
                    method=window_method,
                    hog_bins=8,
                )

            st.caption(
                f"Extracted {len(window_features)} windows with {window_features.shape[1]} features each"
            )

            # t-SNE visualization of sliding windows
            if len(window_features) >= 2:
                col_color, col_perplexity = st.columns([3, 1])
                with col_color:
                    window_color_by = st.selectbox(
                        "Color by",
                        options=[
                            "mean_influence",
                            "std_influence",
                            "max_influence",
                            "min_influence",
                        ],
                        index=0,
                        key="local_behaviors_window_color",
                    )
                with col_perplexity:
                    window_perplexity = st.slider(
                        "t-SNE perplexity",
                        min_value=5,
                        max_value=min(50, len(window_features) - 1),
                        value=min(30, len(window_features) - 1),
                        key="local_behaviors_window_perplexity",
                    )

                with profile("visualize_tsne_sliding_windows"):
                    fig_windows = visualize_tsne_embeddings(
                        window_features,
                        window_metadata,
                        color_by=window_color_by,
                        title=f"t-SNE of Sliding Windows (Rollout {selected_episode_idx}, Demo {selected_demo.index})",
                        perplexity=window_perplexity,
                    )
                    st.plotly_chart(fig_windows, width="stretch")
            else:
                st.info("Not enough windows for t-SNE visualization (need at least 2)")


@st.fragment
def render_matrix_embeddings_section(
    data: InfluenceData,
    demo_split: SplitType,
    selected_episode_idx: int,
):
    """Fragment for influence matrix embeddings."""
    st.header("4. Influence Matrix Embeddings")

    st.markdown("""
    Each point represents one rollout-demo influence matrix.
    This visualization helps identify patterns in how different demonstrations influence this rollout.
    """)

    with st.expander("Load Influence Matrix Embeddings", expanded=False):
        col_matrix_method, col_matrix_color, col_matrix_perplexity = st.columns(
            [2, 2, 1]
        )
        with col_matrix_method:
            matrix_method = st.selectbox(
                "Embedding method",
                options=["resized", "hog", "stats", "singular_values"],
                index=2,
                key="local_behaviors_matrix_method",
                help="resized: resize to common shape then flatten | hog: gradient histogram | stats: statistical features | singular_values: top-k singular values from SVD",
            )
        with col_matrix_color:
            matrix_color_by = st.selectbox(
                "Color by",
                options=[
                    "mean_influence",
                    "std_influence",
                    "quality_label",
                    "num_samples",
                ],
                index=0,
                key="local_behaviors_matrix_color",
            )
        with col_matrix_perplexity:
            matrix_perplexity = st.slider(
                "t-SNE perplexity",
                min_value=5,
                max_value=50,
                value=30,
                key="local_behaviors_matrix_perplexity",
            )

        # Add slider for number of components when using singular_values method
        if matrix_method == "singular_values":
            n_components = st.slider(
                "Number of singular values",
                min_value=5,
                max_value=50,
                value=10,
                key="local_behaviors_matrix_n_components",
                help="Number of top singular values to use as embedding features",
            )
        else:
            n_components = 10  # Default value when not using singular_values

        show_matrix_embeddings_key = (
            f"show_matrix_embeddings_{selected_episode_idx}_{demo_split}"
        )
        if st.button(
            "Generate Matrix Embeddings",
            key=f"gen_matrix_embeddings_{selected_episode_idx}_{demo_split}",
        ):
            st.session_state[show_matrix_embeddings_key] = True

        if st.session_state.get(show_matrix_embeddings_key, False):
            # Extract influence matrix embeddings
            with profile("extract_influence_matrix_embeddings"):
                matrix_embeddings, matrix_metadata = (
                    extract_influence_matrix_embeddings(
                        data,
                        selected_episode_idx,
                        split=demo_split,
                        method=matrix_method,
                        hog_bins=8,
                        n_components=n_components,
                    )
                )

            st.caption(
                f"Extracted {len(matrix_embeddings)} influence matrices with {matrix_embeddings.shape[1]} features each"
            )

            # t-SNE visualization of influence matrices
            with profile("visualize_tsne_influence_matrices"):
                fig_matrices = visualize_tsne_embeddings(
                    matrix_embeddings,
                    matrix_metadata,
                    color_by=matrix_color_by,
                    title=f"t-SNE of Influence Matrices (Rollout {selected_episode_idx})",
                    perplexity=matrix_perplexity,
                )
                st.plotly_chart(fig_matrices, width="stretch")


@st.fragment
def render_demo_influence_matrices_section(
    data: InfluenceData,
    demo_split: SplitType,
    selected_demo_idx: int,
):
    """Fragment for demo-centric influence matrices grid visualization."""
    st.header("2. All Influence Matrices for This Demo")

    st.markdown("""
    Each heatmap shows the influence matrix between this demonstration and one rollout.
    Rows are rollout timesteps, columns are demo timesteps.
    """)

    with st.expander("Load Influence Matrices Grid", expanded=False):
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )
        total_rollouts = len(data.rollout_episodes)

        col_settings1, col_settings2, col_settings3, col_settings4, col_settings5 = (
            st.columns(5)
        )
        with col_settings1:
            rollouts_per_page = st.selectbox(
                "Rollouts per page",
                options=[12, 16, 20, 24, 32],
                index=0,
                key="demo_centric_rollouts_per_page",
            )
        with col_settings2:
            total_pages = (total_rollouts + rollouts_per_page - 1) // rollouts_per_page
            current_page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="demo_centric_page_number",
            )
        with col_settings3:
            colorscheme = st.selectbox(
                "Color scheme",
                options=["Red-White-Green", "Red-White-Blue"],
                index=0,
                key="demo_centric_colorscheme",
            )
        with col_settings4:
            apply_gaussian_individual = st.checkbox(
                "Apply Gaussian smoothing",
                value=False,
                key="demo_centric_individual_gaussian_smoothing",
                help="Denoise matrices by convolving with a Gaussian kernel",
            )
        with col_settings5:
            if apply_gaussian_individual:
                gaussian_sigma_individual = st.slider(
                    "Gaussian sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="demo_centric_individual_gaussian_sigma",
                    help="Standard deviation of Gaussian kernel (larger = more smoothing)",
                )
            else:
                gaussian_sigma_individual = 1.0

        # Calculate which rollouts to show on this page
        start_rollout_idx = (current_page - 1) * rollouts_per_page
        end_rollout_idx = min(start_rollout_idx + rollouts_per_page, total_rollouts)

        st.caption(
            f"Showing rollouts {start_rollout_idx + 1}-{end_rollout_idx} of {total_rollouts}"
        )

        show_matrices_key = f"show_demo_matrices_grid_{selected_demo_idx}_{demo_split}"
        if st.button(
            "Generate Influence Matrices Grid",
            key=f"gen_demo_influence_matrices_{selected_demo_idx}_{demo_split}",
        ):
            st.session_state[show_matrices_key] = True

        if st.session_state.get(show_matrices_key, False):
            # Get demo sample indices
            demo_sample_idxs = ep_idxs[selected_demo_idx]

            # Prepare matrices and titles for grid
            matrices = []
            titles = []
            for i in range(start_rollout_idx, end_rollout_idx):
                rollout_ep = data.rollout_episodes[i]
                rollout_sample_indices = np.arange(
                    rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
                )

                # Get influence matrix (rollout timesteps x demo timesteps)
                rollout_demo_influence = influence_matrix[
                    rollout_sample_indices[:, None], demo_sample_idxs
                ]

                # Apply Gaussian smoothing if requested
                if apply_gaussian_individual:
                    rollout_demo_influence = gaussian_filter(
                        rollout_demo_influence, sigma=gaussian_sigma_individual
                    )

                matrices.append(rollout_demo_influence)

                status = (
                    "✓"
                    if rollout_ep.success
                    else "✗"
                    if rollout_ep.success is not None
                    else "?"
                )
                titles.append(f"Rollout {rollout_ep.index} [{status}]")

            # Use pure plotting function
            selected_demo = demo_episodes[selected_demo_idx]
            title_suffix = (
                f" (Gaussian σ={gaussian_sigma_individual})"
                if apply_gaussian_individual
                else ""
            )
            fig = plotting.create_influence_grid_plot(
                matrices=matrices,
                titles=titles,
                main_title=f"Demo {selected_demo.index} - Influence Matrices (Page {current_page}/{total_pages}){title_suffix}",
                colorscale=colorscheme,
                cols=4,
            )
            st.plotly_chart(fig, width="stretch")


def render_demo_centric_individual(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Render demo-centric individual view: fix a demo, scroll through rollouts."""

    # Demo selection
    st.header("1. Select Demonstration Episode")

    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, demo_split)

    demo_options = []
    for ep in demo_episodes:
        quality = "unlabelled"
        if data.demo_quality_labels is not None:
            quality = data.demo_quality_labels.get(ep.index, "unlabelled")
        demo_options.append(f"Demo {ep.index} ({quality}) - {ep.num_samples} samples")

    selected_demo_str = st.selectbox(
        "Choose a demonstration episode:",
        options=demo_options,
        index=0,
        key="demo_centric_demo_select",
    )
    selected_demo_idx = demo_options.index(selected_demo_str)
    selected_demo = demo_episodes[selected_demo_idx]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Demo Index", selected_demo.index)
    with col2:
        quality = "unlabelled"
        if data.demo_quality_labels is not None:
            quality = data.demo_quality_labels.get(selected_demo.index, "unlabelled")
        st.metric("Quality", quality)
    with col3:
        st.metric("Samples", selected_demo.num_samples)

    st.divider()

    # Render influence matrices section
    render_demo_influence_matrices_section(data, demo_split, selected_demo_idx)


@st.fragment
def render_demo_influence_distribution(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Fragment for demo influence distribution plots across rollout timesteps."""
    st.header("Influence Distribution per Demo Timestep")

    st.markdown("""
    For each demonstration, this shows how influence varies across rollout timesteps.

    - **X-axis**: Demo timesteps
    - **Y-axis**: Influence score distribution (aggregated across all rollout timesteps)
    - **Lines**: Mean, ±1 std, ±2 std, min, max

    This helps answer: How stable is the influence at each demo timestep across different rollout timesteps?
    """)

    with st.expander("Load Demo Influence Distribution Plots", expanded=False):
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )
        total_demos = len(demo_episodes)

        # Find max rollout length for cutoff slider
        max_rollout_len = max(ep.num_samples for ep in data.rollout_episodes)

        # Controls - Row 1
        col_settings1, col_settings2, col_settings3 = st.columns(3)
        with col_settings1:
            demos_per_page = st.selectbox(
                "Demos per page",
                options=[12, 16, 20, 24, 32],
                index=0,
                key="demo_dist_demos_per_page",
            )
        with col_settings2:
            total_pages = (total_demos + demos_per_page - 1) // demos_per_page
            current_page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="demo_dist_page_number",
            )
        with col_settings3:
            viz_mode = st.selectbox(
                "Visualization mode",
                options=["Lines & Bands", "Density Heatmap"],
                index=0,
                key="demo_dist_viz_mode",
                help="Lines & Bands: traditional line plots | Density Heatmap: color-coded density along y-axis",
            )

        # Controls - Row 2: Line toggles
        col_line1, col_line2, col_line3, col_line4, col_line5 = st.columns(5)
        with col_line1:
            show_mean = st.checkbox(
                "Show mean line",
                value=True,
                key="demo_dist_show_mean",
                help="Show mean influence line",
            )
        with col_line2:
            show_midpoint = st.checkbox(
                "Show midpoint line",
                value=False,
                key="demo_dist_show_midpoint",
                help="Show line for midpoint between min and max",
            )
        with col_line3:
            show_std_bands = st.checkbox(
                "Show std bands",
                value=True,
                key="demo_dist_show_std_bands",
                help="Show filled bands for ±1 and ±2 standard deviations",
            )
        with col_line4:
            show_percentile_bands = st.checkbox(
                "Show 1-99% bands",
                value=False,
                key="demo_dist_show_percentile_bands",
                help="Show filled bands for 1st and 99th percentiles",
            )
        with col_line5:
            show_minmax_bands = st.checkbox(
                "Show min/max bands",
                value=True,
                key="demo_dist_show_minmax_bands",
                help="Show filled bands for min and max values",
            )

        # Rollout timestep cutoff
        st.markdown("**Rollout Timestep Filtering:**")
        col_cutoff1, col_cutoff2 = st.columns([1, 3])
        with col_cutoff1:
            enable_cutoff = st.checkbox(
                "Enable cutoff",
                value=False,
                key="demo_dist_enable_cutoff",
                help="Enable rollout timestep cutoff to filter out late timesteps",
            )
        with col_cutoff2:
            if enable_cutoff:
                rollout_cutoff = st.slider(
                    "Max rollout timestep to include",
                    min_value=1,
                    max_value=max_rollout_len,
                    value=max_rollout_len,
                    key="demo_dist_rollout_cutoff",
                    help="Only include rollout timesteps up to this value when computing statistics",
                )
            else:
                rollout_cutoff = max_rollout_len

        # Temporal smoothing
        st.markdown("**Temporal Smoothing:**")
        col_smooth1, col_smooth2 = st.columns([1, 3])
        with col_smooth1:
            enable_smoothing = st.checkbox(
                "Enable smoothing",
                value=False,
                key="demo_dist_enable_smoothing",
                help="Apply Gaussian smoothing along demo timesteps",
            )
        with col_smooth2:
            if enable_smoothing:
                smoothing_sigma = st.slider(
                    "Gaussian sigma",
                    min_value=0.5,
                    max_value=10.0,
                    value=1.0,
                    step=0.5,
                    key="demo_dist_smoothing_sigma",
                    help="Standard deviation of Gaussian kernel (larger = more smoothing)",
                )
            else:
                smoothing_sigma = 0.0

        # Calculate which demos to show on this page
        start_demo_idx = (current_page - 1) * demos_per_page
        end_demo_idx = min(start_demo_idx + demos_per_page, total_demos)

        if enable_cutoff:
            st.caption(
                f"Showing demos {start_demo_idx + 1}-{end_demo_idx} of {total_demos} | "
                f"Stats computed from rollout timesteps: 0-{rollout_cutoff - 1}"
            )
        else:
            st.caption(
                f"Showing demos {start_demo_idx + 1}-{end_demo_idx} of {total_demos}"
            )

        show_dist_key = f"show_demo_dist_{demo_split}"
        if st.button(
            "Generate Demo Influence Distribution Plots",
            key=f"gen_demo_dist_{demo_split}",
        ):
            st.session_state[show_dist_key] = True

        if st.session_state.get(show_dist_key, False):
            # Prepare data for distribution plots
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            num_demos_on_page = end_demo_idx - start_demo_idx

            # Calculate grid layout
            cols = 4
            rows = (num_demos_on_page + cols - 1) // cols

            # Create subplots with quality labels
            subplot_titles = []
            for i in range(start_demo_idx, end_demo_idx):
                demo_idx = demo_episodes[i].index
                quality = "unlabelled"
                if data.demo_quality_labels is not None:
                    quality = data.demo_quality_labels.get(demo_idx, "unlabelled")
                subplot_titles.append(f"Demo {demo_idx} ({quality})")

            # Branch based on visualization mode
            # Collect data for all demos on this page
            demo_timestep_influences_list = []

            for i in range(start_demo_idx, end_demo_idx):
                demo_ep = demo_episodes[i]
                demo_sample_idxs = ep_idxs[i]
                demo_len = len(demo_sample_idxs)

                # Collect influence values for each demo timestep
                demo_timestep_influences = [[] for _ in range(demo_len)]

                for rollout_ep in data.rollout_episodes:
                    rollout_sample_indices = np.arange(
                        rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
                    )
                    rollout_len = min(len(rollout_sample_indices), rollout_cutoff)
                    rollout_sample_indices = rollout_sample_indices[:rollout_len]

                    rollout_demo_influence = influence_matrix[
                        rollout_sample_indices[:, None], demo_sample_idxs
                    ]

                    for demo_t in range(demo_len):
                        demo_timestep_influences[demo_t].extend(
                            rollout_demo_influence[:, demo_t].tolist()
                        )

                demo_timestep_influences_list.append(demo_timestep_influences)

            # Build title
            title_parts = []
            if enable_cutoff:
                title_parts.append(f"rollout t ≤ {rollout_cutoff - 1}")
            if viz_mode == "Lines & Bands" and enable_smoothing:
                title_parts.append(f"smoothed σ={smoothing_sigma}")
            title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""

            if viz_mode == "Density Heatmap":
                # Use plotting function for density heatmap
                title = f"Influence Density per Demo Timestep (Page {current_page}/{total_pages}){title_suffix}"
                fig = plotting.create_influence_density_heatmaps(
                    demo_timestep_influences_list=demo_timestep_influences_list,
                    subplot_titles=subplot_titles,
                    title=title,
                    cols=4,
                    nbinsy=200,
                )
                st.plotly_chart(fig, width="stretch")

            else:
                # Compute statistics for lines & bands visualization
                statistics_list = []

                for demo_timestep_influences in demo_timestep_influences_list:
                    demo_len = len(demo_timestep_influences)
                    demo_timesteps = np.arange(demo_len)
                    means = []
                    stds = []
                    mins = []
                    maxs = []
                    p1s = []
                    p99s = []

                    for demo_t in range(demo_len):
                        values = np.array(demo_timestep_influences[demo_t])
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                        mins.append(np.min(values))
                        maxs.append(np.max(values))
                        p1s.append(np.percentile(values, 1))
                        p99s.append(np.percentile(values, 99))

                    means = np.array(means)
                    stds = np.array(stds)
                    mins = np.array(mins)
                    maxs = np.array(maxs)
                    p1s = np.array(p1s)
                    p99s = np.array(p99s)

                    # Apply Gaussian smoothing if enabled
                    if enable_smoothing:
                        from scipy.ndimage import gaussian_filter1d

                        means = gaussian_filter1d(means, sigma=smoothing_sigma)
                        stds = gaussian_filter1d(stds, sigma=smoothing_sigma)
                        mins = gaussian_filter1d(mins, sigma=smoothing_sigma)
                        maxs = gaussian_filter1d(maxs, sigma=smoothing_sigma)
                        p1s = gaussian_filter1d(p1s, sigma=smoothing_sigma)
                        p99s = gaussian_filter1d(p99s, sigma=smoothing_sigma)

                    # Compute midpoint
                    midpoints = (mins + maxs) / 2.0

                    statistics_list.append(
                        {
                            "demo_timesteps": demo_timesteps,
                            "means": means,
                            "stds": stds,
                            "mins": mins,
                            "maxs": maxs,
                            "p1s": p1s,
                            "p99s": p99s,
                            "midpoints": midpoints,
                        }
                    )

                # Use plotting function for lines & bands
                title = f"Influence Distribution per Demo Timestep (Page {current_page}/{total_pages}){title_suffix}"
                fig = plotting.create_influence_distribution_lines(
                    statistics_list=statistics_list,
                    subplot_titles=subplot_titles,
                    title=title,
                    show_mean=show_mean,
                    show_midpoint=show_midpoint,
                    show_std_bands=show_std_bands,
                    show_percentile_bands=show_percentile_bands,
                    show_minmax_bands=show_minmax_bands,
                    cols=4,
                )
                st.plotly_chart(fig, width="stretch")


@st.fragment
def render_demo_centric_aggregated(
    data: InfluenceData,
    demo_split: SplitType,
):
    """Fragment for aggregated demo-centric influence matrices grid across all demos."""
    st.divider()
    st.header("Aggregated Influence Heatmaps Across All Demos")

    st.markdown("""
    Each heatmap shows the aggregated influence for one demonstration, summing influence values
    across all rollouts for each (rollout_timestep, demo_timestep) combination.
    """)

    with st.expander("Load Aggregated Demo-Centric Influence Grid", expanded=False):
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )
        total_demos = len(demo_episodes)

        # Find max rollout length for cutoff slider
        max_rollout_len = max(ep.num_samples for ep in data.rollout_episodes)

        # Controls - First row
        col_settings1, col_settings2, col_settings3, col_settings4, col_settings5 = (
            st.columns(5)
        )
        with col_settings1:
            demos_per_page = st.selectbox(
                "Demos per page",
                options=[12, 16, 20, 24, 32],
                index=0,
                key="demo_aggregated_demos_per_page",
            )
        with col_settings2:
            total_pages = (total_demos + demos_per_page - 1) // demos_per_page
            current_page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="demo_aggregated_page_number",
            )
        with col_settings3:
            show_std_dev = st.checkbox(
                "Show Std Dev plots",
                value=True,
                key="demo_aggregated_show_std_dev",
                help="Show standard deviation plots next to heatmaps",
            )
        with col_settings4:
            apply_gaussian = st.checkbox(
                "Apply Gaussian smoothing",
                value=False,
                key="demo_aggregated_gaussian_smoothing",
                help="Denoise matrices by convolving with a Gaussian kernel",
            )
        with col_settings5:
            if apply_gaussian:
                gaussian_sigma = st.slider(
                    "Gaussian sigma",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="demo_aggregated_gaussian_sigma",
                    help="Standard deviation of Gaussian kernel (larger = more smoothing)",
                )
            else:
                gaussian_sigma = 1.0

        # Second row - Rollout timestep cutoff
        st.markdown("**Rollout Timestep Filtering:**")
        col_cutoff1, col_cutoff2 = st.columns([1, 3])
        with col_cutoff1:
            enable_cutoff = st.checkbox(
                "Enable cutoff",
                value=False,
                key="demo_aggregated_enable_cutoff",
                help="Enable rollout timestep cutoff to filter out late timesteps",
            )
        with col_cutoff2:
            if enable_cutoff:
                rollout_cutoff = st.slider(
                    "Max rollout timestep to display",
                    min_value=1,
                    max_value=max_rollout_len,
                    value=max_rollout_len,
                    key="demo_aggregated_rollout_cutoff",
                    help="Only show rollout timesteps up to this value (useful to filter out late timesteps with large influence values)",
                )
            else:
                rollout_cutoff = max_rollout_len

        # Calculate which demos to show on this page
        start_demo_idx = (current_page - 1) * demos_per_page
        end_demo_idx = min(start_demo_idx + demos_per_page, total_demos)

        if enable_cutoff:
            st.caption(
                f"Showing demos {start_demo_idx + 1}-{end_demo_idx} of {total_demos} | "
                f"Rollout timesteps: 0-{rollout_cutoff - 1} (cutoff enabled)"
            )
        else:
            st.caption(
                f"Showing demos {start_demo_idx + 1}-{end_demo_idx} of {total_demos}"
            )

        show_aggregated_grid_key = f"show_demo_aggregated_grid_{demo_split}"
        if st.button(
            "Generate Aggregated Demo-Centric Influence Grid",
            key=f"gen_demo_aggregated_influence_grid_{demo_split}",
        ):
            st.session_state[show_aggregated_grid_key] = True

        if st.session_state.get(show_aggregated_grid_key, False):
            # Prepare data for aggregated grid
            num_demos_on_page = end_demo_idx - start_demo_idx

            aggregated_matrices = []
            std_matrices = []
            titles = []

            for i in range(start_demo_idx, end_demo_idx):
                demo_ep = demo_episodes[i]
                demo_sample_idxs = ep_idxs[i]
                demo_len = len(demo_sample_idxs)

                # Aggregate across all rollouts (only up to rollout_cutoff)
                agg_matrix = np.zeros((rollout_cutoff, demo_len))
                count_matrix = np.zeros(
                    (rollout_cutoff, demo_len)
                )  # Track how many rollouts contribute to each cell

                for rollout_ep in data.rollout_episodes:
                    rollout_sample_indices = np.arange(
                        rollout_ep.sample_start_idx, rollout_ep.sample_end_idx
                    )
                    # Only consider timesteps up to the cutoff
                    rollout_len = min(len(rollout_sample_indices), rollout_cutoff)
                    rollout_sample_indices = rollout_sample_indices[:rollout_len]

                    rollout_demo_influence = influence_matrix[
                        rollout_sample_indices[:, None], demo_sample_idxs
                    ]
                    agg_matrix[:rollout_len, :demo_len] += rollout_demo_influence
                    count_matrix[:rollout_len, :demo_len] += 1

                # Average instead of sum (to normalize for different numbers of rollouts)
                with np.errstate(divide="ignore", invalid="ignore"):
                    agg_matrix = np.divide(agg_matrix, count_matrix)
                    agg_matrix = np.nan_to_num(agg_matrix, nan=0.0)

                if apply_gaussian:
                    agg_matrix = gaussian_filter(agg_matrix, sigma=gaussian_sigma)

                aggregated_matrices.append(agg_matrix)

                if show_std_dev:
                    std_matrices.append(np.std(agg_matrix, axis=0).reshape(1, -1))

                quality = "unlabelled"
                if data.demo_quality_labels is not None:
                    quality = data.demo_quality_labels.get(demo_ep.index, "unlabelled")
                titles.append(f"Demo {demo_ep.index} ({quality})")

            # Use pure plotting function
            title_suffix = f" (Gaussian σ={gaussian_sigma})" if apply_gaussian else ""
            fig = plotting.create_aggregated_influence_grid(
                matrices=aggregated_matrices,
                std_matrices=std_matrices if show_std_dev else None,
                titles=titles,
                main_title=f"Aggregated Demo-Centric Influence Heatmaps (Page {current_page}/{total_pages}){title_suffix}",
                rollouts_per_row=2 if show_std_dev else 4,
            )
            st.plotly_chart(fig, width="stretch")


def render_local_behaviors_tab(
    data: InfluenceData,
    demo_split: SplitType,
    top_k: int,
    obs_key: str,
    annotation_file: str,
):
    """Render the local behaviors tab for intra-trajectory analysis.

    This tab explores:
    1. Sliding window features within influence matrices
    2. Influence matrix embeddings for a single rollout/demo
    3. Pattern discovery using t-SNE visualization
    """
    st.markdown("""
    This tab explores **local behavior modes** within individual influence matrices.

    Key questions:
    - How can we find coherent slices of behavior modes inside influence matrices?
    - What patterns emerge when we look at all influence matrices for a given rollout/demo?
    - Can we identify clusters of "how demonstrations influence rollouts"?
    """)

    # Create main tabs for rollout-centric vs demo-centric
    main_tab_rollout, main_tab_demo = st.tabs(["Rollout-Centric", "Demo-Centric"])

    with main_tab_rollout:
        st.markdown("""
        **Rollout-Centric View:** Fix a rollout and explore influence matrices across different demonstrations.
        """)
        # Create subtabs
        subtab_individual, subtab_aggregated = st.tabs(
            ["Individual Rollout", "Aggregated Rollout-Centric"]
        )

        with subtab_individual:
            st.divider()

            # Rollout selection
            st.header("1. Select Rollout Episode")

            episode_options = []
            for ep in data.rollout_episodes:
                status = "✓" if ep.success else "✗" if ep.success is not None else "?"
                episode_options.append(
                    f"Episode {ep.index} [{status}] ({ep.num_samples} samples)"
                )

            selected_episode_str = st.selectbox(
                "Choose a rollout episode:",
                options=episode_options,
                index=0,
                key="local_behaviors_episode_select",
            )
            selected_episode_idx = episode_options.index(selected_episode_str)
            selected_episode = data.rollout_episodes[selected_episode_idx]

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

            # Render fragments for each section
            render_influence_matrices_section(data, demo_split, selected_episode_idx)

            st.divider()

            render_sliding_window_section(data, demo_split, selected_episode_idx)

            st.divider()

            render_matrix_embeddings_section(data, demo_split, selected_episode_idx)

        with subtab_aggregated:
            st.divider()
            render_aggregated_influence_grid(data, demo_split)

            st.divider()
            render_state_action_tsne_section(data, demo_split)

    with main_tab_demo:
        st.markdown("""
        **Demo-Centric View:** Fix a demonstration and explore influence matrices across different rollouts.

        This helps answer: Given a fixed demonstration, do the influence matrices for that demo
        and different rollouts have similar patterns?
        """)
        # Create subtabs
        subtab_demo_individual, subtab_demo_aggregated = st.tabs(
            ["Individual Demo", "Aggregated Demo-Centric"]
        )

        with subtab_demo_individual:
            render_demo_centric_individual(data, demo_split)

        with subtab_demo_aggregated:
            render_demo_influence_distribution(data, demo_split)
            render_demo_centric_aggregated(data, demo_split)


@st.fragment
def _render_state_action_tsne_fragment(data, demo_split):
    st.header("State-Action Distribution (t-SNE)")
    st.markdown("""
    Visualize the distribution of state-action pairs from ALL rollouts compared to ALL training demonstrations.
    """)
    # Get dimensions for info display
    n_obs_steps = data.n_obs_steps
    sample_obs = data.get_demo_obs(0) if data.num_demo_samples > 0 else None
    obs_dim = (
        sample_obs.shape[0] // n_obs_steps
        if sample_obs is not None and sample_obs.shape[0] >= n_obs_steps
        else 19
    )
    action_horizon, action_dim = 16, 10
    if data.demo_dataset and len(data.demo_dataset.sampler) > 0:
        sample_data = data.demo_dataset.sampler.sample_sequence(0)
        if "action" in sample_data:
            action_shape = sample_data["action"].shape
            action_horizon, action_dim = (
                action_shape if len(action_shape) == 2 else (1, action_shape[0])
            )
    state_action_dim = (n_obs_steps * obs_dim) + (action_horizon * action_dim)

    st.info(
        f"**State-Action Vector Dimensions:** {state_action_dim}-dimensional vector ({n_obs_steps * obs_dim} obs + {action_horizon * action_dim} action)"
    )

    with st.expander("Load State-Action t-SNE", expanded=False):
        tsne_perplexity = st.slider(
            "t-SNE perplexity",
            min_value=5,
            max_value=500,
            value=30,
            key="aggregated_tsne_perplexity",
        )
        show_tsne_key = f"show_state_action_tsne_{demo_split}"
        if st.button("Generate t-SNE Visualization", key="aggregated_generate_tsne"):
            st.session_state[show_tsne_key] = True
        if st.session_state.get(show_tsne_key, False):
            with st.spinner("Running t-SNE..."):
                fig = visualize_state_action_tsne(
                    data, split=demo_split, perplexity=tsne_perplexity
                )
                st.plotly_chart(fig, width="stretch")


@st.fragment
def _render_state_histograms_fragment(data, demo_split):
    st.header("State Distribution Histograms")
    st.markdown("""
    Compare the distribution of each state dimension between training and rollout samples.
    """)
    with st.expander("Load State Distribution Histograms", expanded=False):
        show_histograms_key = f"show_state_histograms_{demo_split}"
        if st.button("Generate State Histograms", key="aggregated_generate_histograms"):
            st.session_state[show_histograms_key] = True
        if st.session_state.get(show_histograms_key, False):
            with st.spinner("Generating histograms..."):
                fig = visualize_state_histograms(data, split=demo_split)
                st.plotly_chart(fig, width="stretch")


@st.fragment
def _render_action_histograms_fragment(data, demo_split):
    st.header("Action Distribution Histograms")
    st.markdown("""
    Compare the distribution of each action dimension between training and rollout samples.
    """)
    with st.expander("Load Action Distribution Histograms", expanded=False):
        show_action_histograms_key = f"show_action_histograms_{demo_split}"
        if st.button(
            "Generate Action Histograms", key="aggregated_generate_action_histograms"
        ):
            st.session_state[show_action_histograms_key] = True
        if st.session_state.get(show_action_histograms_key, False):
            with st.spinner("Generating histograms..."):
                fig = visualize_action_histograms(data, split=demo_split)
                st.plotly_chart(fig, width="stretch")


def render_state_action_tsne_section(data: InfluenceData, demo_split: SplitType):
    """Render state-action distribution analysis using localized fragments."""
    _render_state_action_tsne_fragment(data, demo_split)
    st.divider()
    _render_state_histograms_fragment(data, demo_split)
    st.divider()
    _render_action_histograms_fragment(data, demo_split)


@st.fragment
def render_behavior_slice_search(
    data: InfluenceData,
    demo_split: SplitType,
    annotation_file: str,
    task_config: Optional[str] = None,
    obs_key: str = "agentview_image",
):
    """Fragment for searching demo slices by behavior label.

    This view is similar to the slice-based influence analysis in the episode influence tab,
    but focused on finding demo slices that match a specific behavior label.

    Args:
        data: InfluenceData object
        demo_split: Which demo split to use
        annotation_file: Path to annotation file
        task_config: Task config name for JSON annotation files
        obs_key: Camera view to use for frame rendering
    """
    from influence_visualizer.render_annotation import (
        get_episode_annotations,
        load_annotations,
    )
    from influence_visualizer.render_influences import (
        AGGREGATION_METHODS,
        sliding_window_aggregate,
    )

    st.header("Behavior Slice Search")

    st.markdown("""
    Search for demonstration slices matching a specific behavior label and rank them by
    their influence on rollout timesteps.

    **How it works:**
    1. Select a behavior label to search for
    2. Configure search parameters (aggregation method, window width, rollout aggregation)
    3. View ranking scores to determine optimal top-k
    4. Examine the top-k most influential demo slices for that behavior
    """)

    # Load annotations
    annotations = load_annotations(annotation_file, task_config=task_config)

    if not annotations:
        st.warning(
            f"No annotations found. Please annotate some demonstrations first using the Annotation tab."
        )
        return

    with st.expander("Behavior Slice Search Settings", expanded=True):
        # Collect all unique labels from ROLLOUT annotations
        all_labels = set()
        influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(
            data, demo_split
        )

        # Collect all labeled slices from ROLLOUTS (not demos)
        for rollout_ep in data.rollout_episodes:
            episode_id = str(rollout_ep.index)
            ep_annotations = get_episode_annotations(annotations, episode_id, "rollout")

            for slice_info in ep_annotations:
                all_labels.add(slice_info["label"])

        if not all_labels:
            st.warning(
                f"No labeled rollout slices found. "
                f"Please annotate some rollout episodes first in the Annotation tab."
            )
            return

        # Label selection
        st.subheader("1. Select Behavior Label")

        st.markdown("""
        Select a rollout behavior label. The search will find demo slices that have high
        influence on rollout timesteps labeled with this behavior.
        """)

        col_label, col_count = st.columns([2, 1])

        with col_label:
            selected_label = st.selectbox(
                "Rollout behavior to analyze",
                options=sorted(all_labels),
                key=f"slice_search_label_{demo_split}",
            )

        # Count how many rollout slices have this label
        rollout_slices_with_label = []
        for rollout_ep in data.rollout_episodes:
            episode_id = str(rollout_ep.index)
            ep_annotations = get_episode_annotations(annotations, episode_id, "rollout")
            for slice_info in ep_annotations:
                if slice_info["label"] == selected_label:
                    rollout_slices_with_label.append(
                        {
                            "rollout_idx": rollout_ep.index,
                            "rollout_ep": rollout_ep,
                            "start": slice_info["start"],
                            "end": slice_info["end"],
                        }
                    )

        with col_count:
            st.metric("Rollout slices with this label", len(rollout_slices_with_label))

        if not rollout_slices_with_label:
            st.info(f"No rollout slices found with label '{selected_label}'")
            return

        st.divider()

        # Search parameters
        st.subheader("2. Configure Search Parameters")

        col_agg, col_window = st.columns(2)

        with col_agg:
            aggregation_method = st.selectbox(
                "Aggregation method",
                options=list(AGGREGATION_METHODS.keys()),
                index=0,
                key=f"slice_search_agg_{demo_split}",
                help="How to aggregate influence values within each window",
            )

        with col_window:
            window_width = st.number_input(
                "Window width",
                min_value=1,
                max_value=50,
                value=5,
                key=f"slice_search_window_{demo_split}",
                help="Width of sliding window over demo timesteps",
            )

        # Top-k parameters
        col_per_slice_k, col_global_k, col_ascending = st.columns(3)

        with col_per_slice_k:
            per_slice_top_k = st.number_input(
                "Per-Slice Top K",
                min_value=1,
                max_value=len(data.demo_sample_infos),
                value=20,
                key=f"slice_search_per_slice_topk_{demo_split}",
                help="Number of top demos to collect from each behavior slice",
            )

        with col_global_k:
            global_top_k_input = st.number_input(
                "Global Top K",
                min_value=1,
                max_value=len(data.demo_sample_infos),
                value=10,
                key=f"slice_search_global_topk_{demo_split}",
                help="Number of top demos to display after global re-ranking",
            )

        with col_ascending:
            ascending = st.checkbox(
                "Sort ascending (lowest influence)",
                value=False,
                key=f"slice_search_ascending_{demo_split}",
                help="If checked, find slices with lowest influence instead of highest",
            )

        st.divider()

        # Search button
        st.subheader("3. Run Search")

        search_key = f"slice_search_results_{demo_split}_{selected_label}"
        if st.button(
            f"Search for '{selected_label}' slices",
            key=f"btn_slice_search_{demo_split}",
            type="primary",
        ):
            st.session_state[search_key] = True

        if st.session_state.get(search_key, False):
            with st.spinner("Computing influence scores..."):
                from influence_visualizer.render_influences import (
                    rank_demos_by_slice_influence,
                )

                if len(rollout_slices_with_label) == 0:
                    st.error("No rollout timesteps found for the selected behavior")
                    return

                # Collect top-k from each slice independently, then re-rank globally
                all_candidates = []

                for rollout_slice in rollout_slices_with_label:
                    rollout_ep = rollout_slice["rollout_ep"]
                    rollout_idx = rollout_slice["rollout_idx"]
                    start = rollout_slice["start"]
                    end = rollout_slice["end"]

                    # Convert episode-relative indices to global indices
                    rollout_start_idx = rollout_ep.sample_start_idx + start
                    rollout_end_idx = rollout_ep.sample_start_idx + end + 1

                    # Rank demos for this slice
                    sorted_indices, sorted_scores, _ = rank_demos_by_slice_influence(
                        data=data,
                        rollout_start_idx=rollout_start_idx,
                        rollout_end_idx=rollout_end_idx,
                        window_width=window_width,
                        aggregation_method=aggregation_method,
                        split=demo_split,
                        ascending=ascending,
                    )

                    # Collect per-slice top-k from this slice
                    for i in range(min(per_slice_top_k, len(sorted_indices))):
                        local_sample_idx = sorted_indices[i]
                        score = float(sorted_scores[i])

                        all_candidates.append(
                            {
                                "local_sample_idx": local_sample_idx,
                                "score": score,
                                "source_episode_idx": rollout_idx,
                                "source_start": start,
                                "source_end": end,
                            }
                        )

                # Re-rank globally
                all_candidates.sort(key=lambda x: x["score"], reverse=not ascending)

                # Take global top-k
                global_top_k = all_candidates[:global_top_k_input]

            # Video export (outside spinner to avoid UI issues)
            st.subheader("Export Videos")
            export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
            with export_col1:
                do_export = st.button(
                    "Export Videos",
                    key=f"export_behavior_search_{selected_label}_{demo_split}",
                    help="Export videos of behavior slices and top-k demos to outputs/behavior_search_exports/",
                )
            with export_col2:
                export_fps = st.number_input(
                    "FPS",
                    min_value=1,
                    max_value=60,
                    value=10,
                    step=1,
                    key=f"export_behavior_search_fps_{selected_label}_{demo_split}",
                    help="Frames per second for exported videos",
                )

            if do_export and st.session_state.get(search_key, False):
                import pathlib

                from influence_visualizer.video_export import export_slice_videos

                with st.spinner("Exporting videos..."):
                    all_exported_paths = []
                    for slice_idx, rollout_slice in enumerate(
                        rollout_slices_with_label
                    ):
                        rollout_idx = rollout_slice["rollout_idx"]
                        start_frame = rollout_slice["start"]
                        end_frame = rollout_slice["end"]

                        # Get top demos for this slice
                        slice_demos = [
                            c
                            for c in global_top_k
                            if c["source_episode_idx"] == rollout_idx
                            and c["source_start"] == start_frame
                            and c["source_end"] == end_frame
                        ][:10]

                        if not slice_demos:
                            continue

                        # Build demo influences
                        demo_influences = []
                        for candidate in slice_demos:
                            local_sample_idx = candidate["local_sample_idx"]

                            if demo_split == "train":
                                global_sample_idx = local_sample_idx
                            elif demo_split == "holdout":
                                global_sample_idx = local_sample_idx + len(
                                    data.demo_sample_infos
                                )
                            else:
                                global_sample_idx = local_sample_idx

                            if global_sample_idx >= len(data.all_demo_sample_infos):
                                continue

                            sample_info = data.all_demo_sample_infos[global_sample_idx]

                            demo_episode = None
                            for ep in data.all_demo_episodes:
                                if (
                                    ep.sample_start_idx
                                    <= global_sample_idx
                                    < ep.sample_end_idx
                                ):
                                    demo_episode = ep
                                    break

                            if demo_episode:
                                demo_influences.append(
                                    {
                                        "influence_score": candidate["score"],
                                        "demo_episode_idx": demo_episode.index,
                                        "demo_timestep": sample_info.timestep,
                                        "local_demo_sample_idx": local_sample_idx,
                                        "global_demo_sample_idx": global_sample_idx,
                                        "sample_info": sample_info,
                                        "episode": demo_episode,
                                    }
                                )

                        if demo_influences:
                            output_dir = pathlib.Path("outputs/behavior_search_exports")
                            exported_paths, error_msg = export_slice_videos(
                                data=data,
                                rollout_episode_idx=rollout_idx,
                                rollout_start_offset=start_frame,
                                rollout_end_offset=end_frame + 1,
                                demo_influences=demo_influences,
                                output_dir=output_dir,
                                task_config_name=f"{task_config or 'unknown'}_{selected_label}_slice{slice_idx}",
                                obs_key=obs_key,
                                fps=int(export_fps),
                                demo_window_width=window_width,
                            )
                            if not error_msg:
                                all_exported_paths.extend(exported_paths)

                    with export_col3:
                        if all_exported_paths:
                            st.success(f"✓ Exported {len(all_exported_paths)} videos")
                        else:
                            st.warning("No videos exported")

            st.divider()

            if st.session_state.get(search_key, False):
                # Display ranking scores visualization (on demand)
                st.subheader("4. Ranking Scores Visualization (Optional)")
                st.markdown("""
                Generate a plot showing all ranking scores to help determine the optimal top-k cutoff.
                """)

                ranking_viz_key = f"show_ranking_viz_{demo_split}_{selected_label}"
                if st.button(
                    "Generate Ranking Scores Plot",
                    key=f"btn_ranking_viz_{demo_split}_{selected_label}",
                ):
                    st.session_state[ranking_viz_key] = True

                if st.session_state.get(ranking_viz_key, False):
                    with st.spinner("Generating ranking visualization..."):
                        # Collect all scores from all candidates for the plot
                        all_scores = np.array([c["score"] for c in all_candidates])

                        # Create detailed labels with demo episode and timestep info
                        all_labels = []
                        for c in all_candidates:
                            local_sample_idx = c["local_sample_idx"]

                            # Find demo episode and timestep
                            demo_ep = None
                            demo_timestep = None
                            for ep_idx, sample_idxs in enumerate(ep_idxs):
                                if local_sample_idx in sample_idxs:
                                    demo_ep = demo_episodes[ep_idx]
                                    # Calculate global sample index
                                    if demo_split == "train":
                                        global_sample_idx = local_sample_idx
                                    elif demo_split == "holdout":
                                        global_sample_idx = (
                                            local_sample_idx + num_train_samples
                                        )
                                    else:
                                        global_sample_idx = local_sample_idx

                                    if global_sample_idx < len(
                                        data.all_demo_sample_infos
                                    ):
                                        sample_info = data.all_demo_sample_infos[
                                            global_sample_idx
                                        ]
                                        demo_timestep = sample_info.timestep
                                    break

                            # Create label: "Rollout ep0[10:20] → Demo ep5 t=120"
                            if demo_ep is not None and demo_timestep is not None:
                                label = (
                                    f"Rollout ep{c['source_episode_idx']}[{c['source_start']}:{c['source_end']}] "
                                    f"→ Demo ep{demo_ep.index} t={demo_timestep}"
                                )
                            else:
                                label = (
                                    f"Rollout ep{c['source_episode_idx']}[{c['source_start']}:{c['source_end']}] "
                                    f"→ Demo sample {local_sample_idx}"
                                )
                            all_labels.append(label)

                        fig_ranking = plotting.create_ranking_scores_plot(
                            scores=all_scores,
                            labels=all_labels,
                            title=f"Demo Influence on Rollout Behavior '{selected_label}' (per slice)",
                            highlight_top_k=global_top_k_input,
                            show_cumulative=False,
                        )
                        st.plotly_chart(fig_ranking, width="stretch")

                st.divider()

                # Aggregated view: Per demo episode analysis
                st.subheader("Demo Episode Aggregation")
                st.markdown("""
                Aggregates results by demonstration episode, showing the number of unique timesteps
                that appeared in the top-k results and their total influence.
                """)

                # Aggregate by demo episode
                demo_ep_stats = {}
                for c in all_candidates[
                    :global_top_k_input
                ]:  # Only consider the global top-k
                    local_sample_idx = c["local_sample_idx"]
                    score = c["score"]

                    # Find demo episode and timestep
                    for ep_idx, sample_idxs in enumerate(ep_idxs):
                        if local_sample_idx in sample_idxs:
                            demo_ep = demo_episodes[ep_idx]
                            demo_ep_id = demo_ep.index

                            # Calculate global sample index to get timestep
                            if demo_split == "train":
                                global_sample_idx = local_sample_idx
                            elif demo_split == "holdout":
                                global_sample_idx = local_sample_idx + num_train_samples
                            else:
                                global_sample_idx = local_sample_idx

                            if global_sample_idx < len(data.all_demo_sample_infos):
                                sample_info = data.all_demo_sample_infos[
                                    global_sample_idx
                                ]
                                demo_timestep = sample_info.timestep

                                if demo_ep_id not in demo_ep_stats:
                                    demo_ep_stats[demo_ep_id] = {
                                        "timesteps": set(),
                                        "total_influence": 0.0,
                                        "quality": None,
                                    }

                                # Add all timesteps in the window [demo_timestep, demo_timestep + window_width)
                                for t in range(
                                    demo_timestep,
                                    min(
                                        demo_timestep + window_width,
                                        demo_ep.num_samples,
                                    ),
                                ):
                                    demo_ep_stats[demo_ep_id]["timesteps"].add(t)

                                demo_ep_stats[demo_ep_id]["total_influence"] += score

                                # Get quality label if available
                                if (
                                    data.demo_quality_labels
                                    and demo_ep_id in data.demo_quality_labels
                                ):
                                    demo_ep_stats[demo_ep_id]["quality"] = (
                                        data.demo_quality_labels[demo_ep_id]
                                    )
                            break

                # Create unified scatter plot with dropdown for y-axis selection
                if demo_ep_stats:
                    import plotly.express as px
                    import plotly.graph_objects as go

                    st.markdown("---")
                    st.markdown("""
                    **Demo Episode Analysis:** Aggregates results by demonstration episode.
                    """)

                    # Add dropdown for y-axis selection
                    y_axis_mode = st.selectbox(
                        "Y-axis metric",
                        options=[
                            "Average Influence Score per Timestep",
                            "Unique Rollout Timesteps",
                            "Unique Rollout Episodes",
                        ],
                        index=0,
                        key=f"demo_ep_analysis_yaxis_{demo_split}_{selected_label}",
                        help="Choose the metric to display on the y-axis",
                    )

                    # Collect additional rollout data for each demo episode (merge with existing demo_ep_stats)
                    for c in all_candidates[:global_top_k_input]:
                        local_sample_idx = c["local_sample_idx"]
                        source_start = c["source_start"]
                        source_end = c["source_end"]
                        source_episode_idx = c["source_episode_idx"]

                        # Find demo episode
                        for ep_idx, sample_idxs in enumerate(ep_idxs):
                            if local_sample_idx in sample_idxs:
                                demo_ep = demo_episodes[ep_idx]
                                demo_ep_id = demo_ep.index

                                # Add rollout tracking to existing stats
                                if demo_ep_id in demo_ep_stats:
                                    if (
                                        "rollout_timesteps"
                                        not in demo_ep_stats[demo_ep_id]
                                    ):
                                        demo_ep_stats[demo_ep_id][
                                            "rollout_timesteps"
                                        ] = set()
                                        demo_ep_stats[demo_ep_id][
                                            "rollout_episodes"
                                        ] = set()

                                    # Add rollout timesteps from this slice
                                    for t in range(source_start, source_end + 1):
                                        demo_ep_stats[demo_ep_id][
                                            "rollout_timesteps"
                                        ].add(t)

                                    # Add rollout episode ID
                                    demo_ep_stats[demo_ep_id]["rollout_episodes"].add(
                                        source_episode_idx
                                    )
                                break

                    # Create scatter plot data based on y-axis selection
                    demo_timesteps_list = []
                    y_metric_list = []
                    qualities = []
                    hover_texts = []

                    for ep_id, stats in sorted(demo_ep_stats.items()):
                        n_demo_timesteps = len(stats["timesteps"])
                        quality = stats["quality"] if stats["quality"] else "unknown"

                        demo_timesteps_list.append(n_demo_timesteps)
                        qualities.append(quality)

                        # Choose metric based on dropdown selection
                        if y_axis_mode == "Average Influence Score per Timestep":
                            avg_influence = (
                                stats["total_influence"] / n_demo_timesteps
                                if n_demo_timesteps > 0
                                else 0
                            )
                            y_metric_list.append(avg_influence)
                            hover_texts.append(
                                f"Demo ep{ep_id}<br>"
                                f"Quality: {quality}<br>"
                                f"Unique demo timesteps: {n_demo_timesteps}<br>"
                                f"Total influence: {stats['total_influence']:.4f}<br>"
                                f"Avg influence: {avg_influence:.4f}"
                            )
                        elif y_axis_mode == "Unique Rollout Timesteps":
                            n_rollout_timesteps = len(
                                stats.get("rollout_timesteps", set())
                            )
                            y_metric_list.append(n_rollout_timesteps)
                            hover_texts.append(
                                f"Demo ep{ep_id}<br>"
                                f"Quality: {quality}<br>"
                                f"Unique demo timesteps: {n_demo_timesteps}<br>"
                                f"Unique rollout timesteps: {n_rollout_timesteps}"
                            )
                        else:  # Unique Rollout Episodes
                            n_rollout_episodes = len(
                                stats.get("rollout_episodes", set())
                            )
                            y_metric_list.append(n_rollout_episodes)
                            hover_texts.append(
                                f"Demo ep{ep_id}<br>"
                                f"Quality: {quality}<br>"
                                f"Unique demo timesteps: {n_demo_timesteps}<br>"
                                f"Unique rollout episodes: {n_rollout_episodes}"
                            )

                    # Create scatter plot
                    fig = go.Figure()

                    # Use Plotly's qualitative color palette for unique colors
                    colors = px.colors.qualitative.Plotly
                    unique_qualities = sorted(list(set(qualities)))
                    color_map = {
                        quality: colors[i % len(colors)]
                        for i, quality in enumerate(unique_qualities)
                    }

                    for quality in unique_qualities:
                        mask = [q == quality for q in qualities]
                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    demo_timesteps_list[i]
                                    for i in range(len(demo_timesteps_list))
                                    if mask[i]
                                ],
                                y=[
                                    y_metric_list[i]
                                    for i in range(len(y_metric_list))
                                    if mask[i]
                                ],
                                mode="markers",
                                name=quality.capitalize(),
                                marker=dict(
                                    size=10,
                                    color=color_map.get(quality.lower(), colors[0]),
                                ),
                                text=[
                                    hover_texts[i]
                                    for i in range(len(hover_texts))
                                    if mask[i]
                                ],
                                hovertemplate="%{text}<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        title=f"Demo Episode Analysis for Behavior '{selected_label}'",
                        xaxis_title="Number of Unique Demo Timesteps in Top-K",
                        yaxis_title=y_axis_mode,
                        hovermode="closest",
                        showlegend=True,
                        height=500,
                    )

                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No demo episode statistics to display")

                st.divider()

                # Display top-k results
                st.subheader(f"5. Top {global_top_k_input} Demo Windows")

                rank_label = "Lowest" if ascending else "Highest"
                total_timesteps = sum(
                    s["end"] - s["start"] + 1 for s in rollout_slices_with_label
                )
                st.markdown(f"""
                Showing the {rank_label.lower()} {global_top_k_input} influential demo windows (width={window_width})
                for rollout behavior **'{selected_label}'** (across {len(rollout_slices_with_label)} rollout slices, {total_timesteps} total timesteps).
                Collected top {per_slice_top_k} from each slice, then re-ranked globally.
                """)

                # Use the same rendering function as episode influence tab
                from influence_visualizer.render_influences import (
                    _render_slice_influence_detail,
                )

                num_train_samples = len(data.demo_sample_infos)

                for rank, candidate in enumerate(global_top_k):
                    local_sample_idx = candidate["local_sample_idx"]
                    score = candidate["score"]
                    source_ep = candidate["source_episode_idx"]
                    source_start = candidate["source_start"]
                    source_end = candidate["source_end"]

                    # Find which episode this sample belongs to
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

                    influence_dict = {
                        "influence_score": score,
                        "demo_episode_idx": episode.index,
                        "demo_timestep": sample_info.timestep,
                        "global_demo_sample_idx": global_sample_idx,
                        "sample_info": sample_info,
                        "episode": episode,
                    }

                    # Add source slice info header
                    st.markdown(
                        f"**📍 Source:** Rollout ep{source_ep} t[{source_start}:{source_end}]"
                    )

                    # Use the same rendering function as episode influence tab
                    _render_slice_influence_detail(
                        data=data,
                        influence=influence_dict,
                        rank=rank + 1,
                        rollout_episode_idx=source_ep,
                        rollout_start_offset=source_start,
                        rollout_end_offset=source_end,
                        window_width=window_width,
                        obs_key=obs_key,
                        split=demo_split,
                        demo_episodes=demo_episodes,
                        key_prefix=f"behavior_search_{selected_label}_{rank}",
                        annotation_file=annotation_file,
                        task_config=task_config,
                    )

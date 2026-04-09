"""Advanced analysis functions for exploring influence matrix patterns and failure modes."""

from typing import List, Tuple

import numpy as np
import streamlit as st

from influence_visualizer import plotting
from scipy import ndimage
from scipy.stats import pearsonr

from influence_visualizer.data_loader import InfluenceData
from influence_visualizer.render_heatmaps import (
    SplitType,
    compute_transition_level_statistics,
    get_split_data,
)


@st.fragment
def _render_failure_clustering_fragment(failure_data, split):
    with st.expander("Clustering Failures by Std", expanded=False):
        # Group failures by rollout and compute avg std
        rollout_std_map = {}
        for entry in failure_data:
            rollout_idx = entry["rollout_idx"]
            if rollout_idx not in rollout_std_map:
                rollout_std_map[rollout_idx] = []
            rollout_std_map[rollout_idx].append(entry["std"])

        # Compute average std per failed rollout
        failed_rollouts_avg_std = {
            rollout_idx: np.mean(stds) for rollout_idx, stds in rollout_std_map.items()
        }

        # Define threshold (median std)
        std_values = list(failed_rollouts_avg_std.values())
        if not std_values:
            st.info("No failure data available for clustering.")
            return

        std_threshold = np.median(std_values)

        st.caption(f"**Std threshold (median):** {std_threshold:.6f}")

        # Classify failures
        high_std_failures = [
            r for r, s in failed_rollouts_avg_std.items() if s > std_threshold
        ]
        low_std_failures = [
            r for r, s in failed_rollouts_avg_std.items() if s <= std_threshold
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("High Std Failures (Confused)", len(high_std_failures))
            st.caption("Policy received conflicting guidance")
        with col2:
            st.metric("Low Std Failures (Confidently Wrong)", len(low_std_failures))
            st.caption("Policy learned wrong strategy consistently")

        show_failure_dist_key = f"show_failure_dist_{split}"
        if st.button("Generate Failure Distribution", key=f"gen_failure_dist_{split}"):
            st.session_state[show_failure_dist_key] = True

        if st.session_state.get(show_failure_dist_key, False):
            # Visualize distribution using pure plotting function
            fig = plotting.create_histogram(
                values=std_values,
                title="Distribution of Average Std Across Failed Rollouts",
                xaxis_title="Average Std",
                yaxis_title="Count",
                color="red",
                nbins=30,
                vline_at=std_threshold,
                vline_label=f"Threshold: {std_threshold:.6f}",
            )
            st.plotly_chart(fig, width="stretch", key=f"failure_std_dist_{split}_plot")


@st.fragment
def _render_temporal_patterns_fragment(data, failure_data, split):
    with st.expander("Temporal Patterns: Std Over Rollout Timeline", expanded=False):
        st.markdown("""
        **Question:** Do failures show different std patterns early vs. late in the episode?

        This can reveal whether confusion happens at the start (uncertain initialization)
        or develops later (accumulated errors).
        """)

        # Re-derive needed variables within expander
        rollout_std_map = {}
        for entry in failure_data:
            rollout_idx = entry["rollout_idx"]
            if rollout_idx not in rollout_std_map:
                rollout_std_map[rollout_idx] = []
            rollout_std_map[rollout_idx].append(entry["std"])
        failed_rollouts_avg_std = {r: np.mean(s) for r, s in rollout_std_map.items()}
        if not failed_rollouts_avg_std:
            st.info("No failure data available for temporal analysis.")
            return

        std_threshold = np.median(list(failed_rollouts_avg_std.values()))
        high_std_failures = [r for r, s in failed_rollouts_avg_std.items() if s > std_threshold]
        low_std_failures = [r for r, s in failed_rollouts_avg_std.items() if s <= std_threshold]

        show_temporal_patterns_key = f"show_temporal_patterns_{split}"
        if st.button("Generate Temporal Patterns Plot", key=f"gen_temporal_patterns_{split}"):
            st.session_state[show_temporal_patterns_key] = True

        if st.session_state.get(show_temporal_patterns_key, False):
            # Gather data for temporal patterns
            data_list = []
            num_examples = min(5, len(high_std_failures), len(low_std_failures))
            example_high_std = sorted(high_std_failures, key=lambda r: failed_rollouts_avg_std[r], reverse=True)[:num_examples]
            example_low_std = sorted(low_std_failures, key=lambda r: failed_rollouts_avg_std[r])[:num_examples]

            influence_matrix, _, _, _ = get_split_data(data, split)

            for rollout_idx in example_high_std:
                rollout_ep = next(ep for ep in data.rollout_episodes if ep.index == rollout_idx)
                rollout_sample_indices = np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx)
                timestep_stds = [np.std(influence_matrix[t_idx, :]) for t_idx in rollout_sample_indices]
                data_list.append({
                    "x": list(range(len(timestep_stds))),
                    "y": timestep_stds,
                    "name": f"Rollout {rollout_idx}",
                    "col": 1
                })

            for rollout_idx in example_low_std:
                rollout_ep = next(ep for ep in data.rollout_episodes if ep.index == rollout_idx)
                rollout_sample_indices = np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx)
                timestep_stds = [np.std(influence_matrix[t_idx, :]) for t_idx in rollout_sample_indices]
                data_list.append({
                    "x": list(range(len(timestep_stds))),
                    "y": timestep_stds,
                    "name": f"Rollout {rollout_idx}",
                    "col": 2
                })

            # Create the figure using pure plotting function
            fig_temporal = plotting.create_temporal_patterns_subplots(
                data_list=data_list,
                title="Std Evolution Over Rollout Timeline",
                subplot_titles=("High Std Failures (Confused)", "Low Std Failures (Confidently Wrong)")
            )
            st.plotly_chart(fig_temporal, width="stretch", key=f"temporal_patterns_{split}_plot")


@st.fragment
def _render_demo_variance_fragment(failure_data, split):
    with st.expander("Demo-Specific Variance on Failures", expanded=False):
        st.markdown("""
        **Question:** Which demonstrations consistently create high variance when paired with failed rollouts?
        """)
        show_demo_variance_key = f"show_demo_variance_{split}"
        if st.button("Generate Demo Variance Plot", key=f"gen_demo_variance_{split}"):
            st.session_state[show_demo_variance_key] = True

        if st.session_state.get(show_demo_variance_key, False):
            # Prepare data for demo variance plot
            demo_std_map = {}
            demo_count_map = {}
            demo_quality_map = {}
            for entry in failure_data:
                demo_idx = entry["demo_idx"]
                if demo_idx not in demo_std_map:
                    demo_std_map[demo_idx] = []
                    demo_count_map[demo_idx] = 0
                    demo_quality_map[demo_idx] = entry["quality"]
                demo_std_map[demo_idx].append(entry["std"])
                demo_count_map[demo_idx] += 1

            demo_avg_std = {demo_idx: np.mean(stds) for demo_idx, stds in demo_std_map.items()}
            sorted_demos = sorted(demo_avg_std.items(), key=lambda x: x[1], reverse=True)
            top_n = min(20, len(sorted_demos))
            top_demos = sorted_demos[:top_n]

            # Use pure plotting function
            fig_demo = plotting.create_demo_variance_plot(
                demo_indices=[d[0] for d in top_demos],
                avg_stds=[d[1] for d in top_demos],
                counts=[demo_count_map[d[0]] for d in top_demos],
                qualities=[demo_quality_map[d[0]] for d in top_demos],
                title=f"Top {top_n} Demos Creating Highest Variance on Failures"
            )
            st.plotly_chart(fig_demo, width="stretch", key=f"demo_variance_{split}_plot")


@st.fragment
def _render_stat_signature_fragment(failure_data, split):
    with st.expander("Statistical Signature Comparison", expanded=False):
        st.markdown("""
        **Question:** How do high-std and low-std failures differ on other influence statistics?
        """)
        show_stat_comparison_key = f"show_stat_comparison_{split}"
        if st.button("Generate Statistical Comparison", key=f"gen_stat_comparison_{split}"):
            st.session_state[show_stat_comparison_key] = True

        if st.session_state.get(show_stat_comparison_key, False):
            rollout_std_map = {}
            for entry in failure_data:
                rollout_idx = entry["rollout_idx"]
                if rollout_idx not in rollout_std_map: rollout_std_map[rollout_idx] = []
                rollout_std_map[rollout_idx].append(entry["std"])
            failed_rollouts_avg_std = {r: np.mean(s) for r, s in rollout_std_map.items()}
            if not failed_rollouts_avg_std: return
            std_threshold = np.median(list(failed_rollouts_avg_std.values()))
            high_std_failures = [r for r, s in failed_rollouts_avg_std.items() if s > std_threshold]
            low_std_failures = [r for r, s in failed_rollouts_avg_std.items() if s <= std_threshold]

            metrics = ["mean", "std", "min", "max"]
            comparison_data = []
            for metric in metrics:
                high_vals = [e[metric] for e in failure_data if e["rollout_idx"] in high_std_failures]
                low_vals = [e[metric] for e in failure_data if e["rollout_idx"] in low_std_failures]
                comparison_data.append({
                    "Metric": metric.title(),
                    "High Std Failures": f"{np.mean(high_vals):.6f}",
                    "Low Std Failures": f"{np.mean(low_vals):.6f}",
                    "Difference": f"{np.mean(high_vals) - np.mean(low_vals):.6f}"
                })
            import pandas as pd
            st.table(pd.DataFrame(comparison_data))


def analyze_failure_modes(data: InfluenceData, split: SplitType = "train"):
    """Analyze and differentiate failure modes using localized fragments."""
    st.header("Failure Mode Analysis")
    st.markdown("""
    **Research Question:** Can we differentiate between different failure modes based on
    the standard deviation of local influence matrices?
    """)

    load_key = f"advanced_failure_analysis_loaded_{split}"
    with st.expander("Load failure mode analysis", expanded=False):
        st.caption(
            "Compute transition-level statistics for all rollout–demo pairs (can take several seconds)."
        )
        if st.button(
            "Compute and show failure analysis",
            key=f"advanced_failure_btn_{split}",
        ):
            st.session_state[load_key] = True

    if not st.session_state.get(load_key, False):
        return

    with st.spinner("Computing transition-level statistics for failure analysis..."):
        stats, metadata = compute_transition_level_statistics(data, split)

    failure_data = []
    for i, (rollout_idx, demo_idx, quality_label, success) in enumerate(metadata):
        if not success:
            failure_data.append({
                "rollout_idx": rollout_idx, "demo_idx": demo_idx, "quality": quality_label,
                "mean": stats[i, 0], "std": stats[i, 1], "min": stats[i, 2], "max": stats[i, 3],
            })

    if not failure_data:
        st.warning("No failed rollouts found in the dataset.")
        return

    st.divider()
    _render_failure_clustering_fragment(failure_data, split)
    st.divider()
    _render_temporal_patterns_fragment(data, failure_data, split)
    st.divider()
    _render_demo_variance_fragment(failure_data, split)
    st.divider()
    _render_stat_signature_fragment(failure_data, split)




def analyze_local_structure(
    data: InfluenceData,
    split: SplitType = "train",
):
    """Analyze local structure and properties of transition-wise influence matrices.

    This function explores the question: Which other ways can we leverage the local
    structure/properties of the transition-wise influence matrix?

    Beyond global statistics (mean, std, min, max), we examine:
    1. Temporal autocorrelation: Do high-influence moments cluster in time?
    2. Spatial patterns: Are there "hot zones" in the heatmap (e.g., diagonal patterns)?
    3. Influence gradients: How quickly does influence change across timesteps?
    4. Peak characteristics: Distribution of local maxima/minima
    5. Symmetry/reciprocity: Do rollout and demo timesteps have symmetric patterns?

    Args:
        data: InfluenceData object
        split: Which demo split to use ("train", "holdout", or "both")
    """
    st.header("Local Structure Analysis")

    st.markdown("""
    **Research Question:** Beyond global statistics (mean, std, min, max), what can we
    learn from the local structure and spatial patterns in transition-wise influence matrices?

    **Explorations:**
    1. **Temporal autocorrelation:** Do high-influence moments cluster in time?
    2. **Spatial patterns:** Are there diagonal structures or hot zones?
    3. **Influence gradients:** How smoothly does influence change?
    4. **Peak characteristics:** Where are local maxima/minima?
    5. **Asymmetry:** Do rollout and demo axes show different patterns?
    """)

    # Get influence matrix
    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)

@st.fragment
def _render_autocorr_fragment(transition_matrix, split, selected_rollout, selected_demo):
    with st.expander("1. Temporal Autocorrelation", expanded=False):
        st.markdown("""
        **Question:** Do high-influence moments cluster in time, or are they randomly distributed?
        """)
        show_autocorr_key = f"show_autocorr_{split}_{selected_rollout}_{selected_demo}"
        if st.button("Generate Temporal Autocorrelation", key=f"gen_autocorr_{split}_{selected_rollout}_{selected_demo}"):
            st.session_state[show_autocorr_key] = True

        if st.session_state.get(show_autocorr_key, False):
            rollout_profile = np.mean(transition_matrix, axis=1)
            demo_profile = np.mean(transition_matrix, axis=0)

            def autocorr(x, lag=1):
                return np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0.0

            max_lag = min(20, len(rollout_profile) - 1, len(demo_profile) - 1)
            lags = list(range(1, max_lag + 1))
            
            # Use pure plotting function
            fig = plotting.create_autocorrelation_plot(
                rollout_lags=lags,
                rollout_autocorr=[autocorr(rollout_profile, l) for l in lags],
                demo_lags=lags,
                demo_autocorr=[autocorr(demo_profile, l) for l in lags]
            )
            st.plotly_chart(fig, width="stretch", key=f"autocorr_{split}_plot")


@st.fragment
def _render_diagonal_fragment(transition_matrix, split, selected_rollout, selected_demo):
    with st.expander("2. Spatial Patterns: Diagonal Analysis", expanded=False):
        st.markdown("""
        **Question:** Is influence concentrated along the diagonal (temporally aligned) or spread throughout?
        """)
        show_diagonal_key = f"show_diagonal_{split}_{selected_rollout}_{selected_demo}"
        if st.button("Generate Diagonal Analysis", key=f"gen_diagonal_{split}_{selected_rollout}_{selected_demo}"):
            st.session_state[show_diagonal_key] = True

        if st.session_state.get(show_diagonal_key, False):
            min_dim = min(transition_matrix.shape)
            diagonal_vals = np.array([transition_matrix[i, i] for i in range(min_dim)])
            mask = np.ones_like(transition_matrix, dtype=bool)
            np.fill_diagonal(mask[:min_dim, :min_dim], False)
            off_diagonal_vals = transition_matrix[mask]
            
            # Use pure plotting function
            fig = plotting.create_diagonal_analysis_plot(
                diagonal_vals=diagonal_vals,
                off_diagonal_vals=off_diagonal_vals
            )
            st.plotly_chart(fig, width="stretch", key=f"diagonal_{split}_plot")


@st.fragment
def _render_gradients_fragment(transition_matrix, split, selected_rollout, selected_demo):
    with st.expander("3. Influence Gradients: Smoothness Analysis", expanded=False):
        st.markdown("""
        **Question:** How rapidly does influence change across adjacent timesteps?
        """)
        show_gradients_key = f"show_gradients_{split}_{selected_rollout}_{selected_demo}"
        if st.button("Generate Influence Gradients", key=f"gen_gradients_{split}_{selected_rollout}_{selected_demo}"):
            st.session_state[show_gradients_key] = True

        if st.session_state.get(show_gradients_key, False):
            grad_rollout = np.gradient(transition_matrix, axis=0)
            grad_demo = np.gradient(transition_matrix, axis=1)
            grad_magnitude = np.sqrt(grad_rollout**2 + grad_demo**2)
            
            # Use pure plotting function
            fig = plotting.create_gradient_magnitude_heatmap(
                grad_magnitude=grad_magnitude,
                title="Gradient Magnitude (Rate of Change)"
            )
            st.plotly_chart(fig, width="stretch", key=f"gradients_{split}_plot")


@st.fragment
def _render_peaks_fragment(transition_matrix, split, selected_rollout, selected_demo):
    with st.expander("4. Peak Detection: Local Maxima/Minima", expanded=False):
        st.markdown("""
        **Question:** Where are the local peaks in influence, and how are they distributed?
        """)
        show_peaks_key = f"show_peaks_{split}_{selected_rollout}_{selected_demo}"
        if st.button("Generate Peak Detection", key=f"gen_peaks_{split}_{selected_rollout}_{selected_demo}"):
            st.session_state[show_peaks_key] = True

        if st.session_state.get(show_peaks_key, False):
            from scipy.ndimage import maximum_filter, minimum_filter
            neighborhood_size = 3
            local_max = transition_matrix == maximum_filter(transition_matrix, neighborhood_size)
            local_min = transition_matrix == minimum_filter(transition_matrix, neighborhood_size)
            max_coords = np.argwhere(local_max)
            min_coords = np.argwhere(local_min)
            
            # Use pure plotting function
            fig = plotting.create_peak_detection_plot(
                transition_matrix=transition_matrix,
                max_coords=max_coords,
                min_coords=min_coords
            )
            st.plotly_chart(fig, width="stretch", key=f"peaks_{split}_plot")


@st.fragment
def _render_asymmetry_fragment(transition_matrix, split, selected_rollout, selected_demo):
    with st.expander("5. Asymmetry: Rollout vs Demo Axis", expanded=False):
        st.markdown("""
        **Question:** Do rollout and demo axes show different structural properties?
        """)
        show_asymmetry_key = f"show_asymmetry_{split}_{selected_rollout}_{selected_demo}"
        if st.button("Generate Asymmetry Analysis", key=f"gen_asymmetry_{split}_{selected_rollout}_{selected_demo}"):
            st.session_state[show_asymmetry_key] = True

        if st.session_state.get(show_asymmetry_key, False):
            var_along_demo = np.var(transition_matrix, axis=1)
            var_along_rollout = np.var(transition_matrix, axis=0)
            
            # Use pure plotting function
            fig = plotting.create_asymmetry_variance_plot(
                data_rollout=var_along_demo,
                data_demo=var_along_rollout,
                title="Asymmetry in Variance"
            )
            st.plotly_chart(fig, width="stretch", key=f"asymmetry_{split}_plot")


def analyze_local_structure(data: InfluenceData, split: SplitType = "train"):
    """Analyze local structure using localized fragments."""
    st.header("Local Structure Analysis")
    st.markdown("""
    **Research Question:** Beyond global statistics (mean, std, min, max), what can we
    learn from the local structure and spatial patterns in transition-wise influence matrices?
    """)

    influence_matrix, demo_episodes, ep_idxs, ep_lens = get_split_data(data, split)
    st.divider()
    st.subheader("Select Rollout-Demo Pair for Analysis")

    col_r, col_d = st.columns(2)
    with col_r:
        selected_rollout = st.selectbox("Rollout Episode", options=[ep.index for ep in data.rollout_episodes], key=f"local_struct_rollout_{split}")
    with col_d:
        selected_demo = st.selectbox("Demo Episode", options=[ep.index for ep in demo_episodes], key=f"local_struct_demo_{split}")

    rollout_ep = next(ep for ep in data.rollout_episodes if ep.index == selected_rollout)
    demo_ep_idx = next(i for i, ep in enumerate(demo_episodes) if ep.index == selected_demo)
    transition_matrix = influence_matrix[np.ix_(np.arange(rollout_ep.sample_start_idx, rollout_ep.sample_end_idx), ep_idxs[demo_ep_idx])]

    st.caption(f"Matrix shape: {transition_matrix.shape[0]} × {transition_matrix.shape[1]}")
    st.divider()

    _render_autocorr_fragment(transition_matrix, split, selected_rollout, selected_demo)
    st.divider()
    _render_diagonal_fragment(transition_matrix, split, selected_rollout, selected_demo)
    st.divider()
    _render_gradients_fragment(transition_matrix, split, selected_rollout, selected_demo)
    st.divider()
    _render_peaks_fragment(transition_matrix, split, selected_rollout, selected_demo)
    st.divider()
    _render_asymmetry_fragment(transition_matrix, split, selected_rollout, selected_demo)



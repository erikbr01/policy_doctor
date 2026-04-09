"""Curation tab: advantage-based slice selection, demo search, export curation config.

Workflow:
  1. Uses behavior graph + advantages from the Behavior Graph tab
  2. Set advantage threshold and selection percentile
  3. Preview selected rollout slices and their influence on training demos
  4. Run slice search (if influence data loaded)
  5. Export full curation config YAML for the pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from policy_doctor.config import VisualizerConfig
from policy_doctor.streamlit_app.config_io import resolve_task_config_stem


def render_tab(
    config: VisualizerConfig,
    data: Any,
    task_config_stem: Optional[str] = None,
) -> None:
    """Render the Curation tab."""
    st.header("Curation")
    task_stem = resolve_task_config_stem(config, task_config_stem)

    labels = st.session_state.get("clustering_labels")
    metadata = st.session_state.get("clustering_metadata")
    advantages = st.session_state.get("bg_advantages")
    graph = st.session_state.get("bg_graph")
    values = st.session_state.get("bg_values")

    if advantages is None or labels is None:
        st.info(
            "Build a behavior graph first in the **Behavior Graph** tab. "
            "This tab uses the computed advantages to select rollout slices for curation."
        )
        return

    st.subheader("Advantage-Based Selection")

    col1, col2, col3 = st.columns(3)
    with col1:
        adv_threshold = st.number_input(
            "Advantage threshold", value=-0.3, step=0.05,
            help="Select rollout slices with advantage >= this value",
            key="cur_adv_threshold",
        )
    with col2:
        selection_pct = st.number_input(
            "Selection percentile", 0.0, 100.0, 99.0, 1.0,
            help="Per-slice percentile for demo selection",
            key="cur_pct",
        )
    with col3:
        curation_mode = st.selectbox(
            "Curation mode",
            ["filter", "selection"],
            help="filter = exclude from train set; selection = add from holdout",
            key="cur_mode",
        )

    valid = np.isfinite(advantages) & (labels >= 0)
    selected_mask = valid & (advantages >= adv_threshold)
    n_selected = int(selected_mask.sum())
    n_valid = int(valid.sum())

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Valid slices", n_valid)
    col_m2.metric("Selected slices", n_selected)
    col_m3.metric("Selection rate", f"{100 * n_selected / max(n_valid, 1):.1f}%")

    _render_selection_preview(labels, metadata, advantages, selected_mask)

    if data is not None:
        _render_slice_search(
            data, labels, metadata, advantages, selected_mask, graph, values,
            adv_threshold, selection_pct, curation_mode, config,
        )
    else:
        st.info("Load influence data (set eval_dir/train_dir in config) to run slice search on training demos.")

    _render_export(adv_threshold, selection_pct, curation_mode, config, task_stem)


def _render_selection_preview(
    labels: np.ndarray,
    metadata: List[Dict],
    advantages: np.ndarray,
    selected_mask: np.ndarray,
) -> None:
    """Show which rollout slices are selected."""
    import plotly.graph_objects as go

    with st.expander("Selected slices by episode", expanded=False):
        ep_slices: Dict[int, list] = {}
        for i, m in enumerate(metadata):
            if selected_mask[i]:
                ep = m.get("rollout_idx", -1)
                ep_slices.setdefault(ep, []).append({
                    "start": m.get("start", 0),
                    "end": m.get("end", 0),
                    "advantage": float(advantages[i]),
                    "cluster": int(labels[i]),
                })

        if not ep_slices:
            st.warning("No slices selected with current threshold.")
            return

        import pandas as pd
        rows = []
        for ep, slices in sorted(ep_slices.items()):
            for s in slices:
                rows.append({
                    "Episode": ep,
                    "Start": s["start"],
                    "End": s["end"],
                    "Cluster": s["cluster"],
                    "Advantage": f"{s['advantage']:.4f}",
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
        st.caption(f"{len(rows)} slices across {len(ep_slices)} episodes")

    with st.expander("Advantage distribution of selected vs rejected"):
        valid = np.isfinite(advantages) & (labels >= 0)
        fig = go.Figure()
        rejected = valid & ~selected_mask
        if rejected.any():
            fig.add_trace(go.Histogram(
                x=advantages[rejected], nbinsx=40, name="Rejected", opacity=0.6,
                marker_color="salmon",
            ))
        if selected_mask.any():
            fig.add_trace(go.Histogram(
                x=advantages[selected_mask], nbinsx=40, name="Selected", opacity=0.6,
                marker_color="mediumseagreen",
            ))
        fig.update_layout(
            barmode="overlay", title="Advantage: Selected vs Rejected",
            xaxis_title="Advantage", yaxis_title="Count", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_slice_search(
    data, labels, metadata, advantages, selected_mask, graph, values,
    adv_threshold, selection_pct, curation_mode, config,
) -> None:
    """Run the full slice search on influence data."""
    st.subheader("Demo Slice Search")

    col1, col2 = st.columns(2)
    with col1:
        window_width = st.number_input("Demo window width", 1, 50, 5, key="cur_ww")
        agg_method = st.selectbox("Aggregation", ["mean", "sum", "max"], key="cur_agg")
    with col2:
        per_slice_top_k = st.number_input("Top-K per slice", 1, 1000, 100, key="cur_topk")
        ascending = st.checkbox("Ascending (lowest influence)", value=False, key="cur_asc")

    if not st.button("Run slice search", type="primary", key="cur_search"):
        return

    selected_indices = np.where(selected_mask)[0]
    if len(selected_indices) == 0:
        st.warning("No slices selected. Adjust the advantage threshold.")
        return

    with st.spinner("Running slice search on influence matrix..."):
        from policy_doctor.data.structures import EpisodeInfo, GlobalInfluenceMatrix
        from policy_doctor.behaviors.behavior_values import slice_indices_to_rollout_slices
        from policy_doctor.curation.attribution import (
            run_slice_search,
            per_slice_percentile_selection,
            resolve_candidates_to_demo_slices,
        )

        def _to_ep(ep):
            return EpisodeInfo(
                index=ep.index, num_samples=ep.num_samples,
                sample_start_idx=ep.sample_start_idx, sample_end_idx=ep.sample_end_idx,
                success=getattr(ep, "success", None),
                raw_length=getattr(ep, "raw_length", None) or ep.num_samples,
            )

        rollout_eps = [_to_ep(ep) for ep in data.rollout_episodes]
        demo_eps = [_to_ep(ep) for ep in data.demo_episodes]
        holdout_eps = [_to_ep(ep) for ep in data.holdout_episodes]
        all_demo_eps = demo_eps + holdout_eps
        num_train = len(data.demo_sample_infos)
        num_holdout = len(data.holdout_sample_infos)

        global_matrix = GlobalInfluenceMatrix(data.influence_matrix, rollout_eps, all_demo_eps)

        rollout_slices = slice_indices_to_rollout_slices(
            metadata, rollout_eps, labels, selected_indices,
        )

        if curation_mode == "filter":
            demo_start, demo_end = 0, num_train
        else:
            demo_start, demo_end = num_train, num_train + num_holdout

        all_candidates, per_slice_candidates = run_slice_search(
            global_matrix, rollout_slices, all_demo_eps,
            window_width_demo=window_width,
            per_slice_top_k=per_slice_top_k,
            ascending=ascending,
            demo_start_idx=demo_start, demo_end_idx=demo_end,
            use_all_demos_per_slice=True,
            show_progress=False,
            aggregation_method=agg_method,
        )

        raw_selection = per_slice_percentile_selection(per_slice_candidates, selection_pct)

        if curation_mode == "filter":
            resolve_infos = data.demo_sample_infos
            resolve_eps = demo_eps
        else:
            resolve_infos = data.holdout_sample_infos
            resolve_eps = holdout_eps

        resolved = resolve_candidates_to_demo_slices(
            raw_selection, resolve_infos, resolve_eps, window_width=window_width,
        )

    st.session_state["cur_resolved"] = resolved
    st.session_state["cur_raw_selection"] = raw_selection
    st.session_state["cur_all_candidates"] = all_candidates

    if not resolved:
        st.warning("No demo slices resolved. Try adjusting parameters.")
        return

    st.success(f"Found {len(resolved)} demo slices from {len(raw_selection)} candidates")

    import pandas as pd
    rows = [{"Episode": ep, "Start": s, "End": e} for ep, s, e in resolved]
    df = pd.DataFrame(rows)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
    with col2:
        ep_counts = df["Episode"].value_counts().reset_index()
        ep_counts.columns = ["Episode", "Slices"]
        st.bar_chart(ep_counts.set_index("Episode"))

    from policy_doctor import plotting
    try:
        fig_hist = plotting.create_curation_episode_histogram(
            [{"episode_idx": ep, "start": s, "end": e} for ep, s, e in resolved],
            title="Curation: Demo Episodes Targeted",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception:
        pass


def _render_export(
    adv_threshold: float,
    selection_pct: float,
    curation_mode: str,
    config: VisualizerConfig,
    task_stem: str,
) -> None:
    """Export full pipeline config for the curation step."""
    from policy_doctor.streamlit_app.config_io import render_config_export, render_save_to_disk

    st.subheader("Export Pipeline Config")

    clust_params = st.session_state.get("clustering_params", {})
    bg_params = st.session_state.get("bg_params", {})
    task_config = st.session_state.get("clustering_task_config", task_stem)

    pipeline_config = {
        "task_config": task_config,
        "curation_mode": curation_mode,
        "advantage_threshold": adv_threshold,
        "selection_percentile": selection_pct,
        "advantage_gamma": bg_params.get("gamma", 0.99),
        "advantage_reward_success": bg_params.get("reward_success", 1.0),
        "advantage_reward_failure": bg_params.get("reward_failure", -1.0),
        "advantage_reward_end": bg_params.get("reward_end", 0.0),
    }
    if clust_params:
        pipeline_config.update({
            "clustering_window_width": clust_params.get("window_width", 5),
            "clustering_stride": clust_params.get("stride", 2),
            "clustering_aggregation": clust_params.get("aggregation", "sum"),
            "clustering_normalize": clust_params.get("normalize", "none"),
            "clustering_umap_n_components": clust_params.get("umap_n_components", 100),
            "clustering_n_clusters": clust_params.get("n_clusters", 20),
        })

    col1, col2 = st.columns(2)
    with col1:
        render_config_export(
            pipeline_config,
            default_filename=f"curation_{task_config}_{curation_mode}.yaml",
            label="Download curation config",
            key="cur_export",
        )
    with col2:
        default_path = f"policy_doctor/policy_doctor/configs/experiment/{task_config}_{curation_mode}.yaml"
        render_save_to_disk(
            pipeline_config, default_path=default_path,
            label="Save to disk", key="cur_save",
        )

"""Behavior Graph tab: build graph from clustering, optional pruning, then MRP analysis.

Workflow:
  1. Load clustering (Clustering tab or disk)
  2. **Transition graph**: structure and transition probabilities (no rewards)
  3. **Pruning**: degree-1 graph simplification and slice relabeling
  4. **Markov reward process**: rewards, γ, V(s), transition values, advantages
  5. Export pipeline YAML
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.config import VisualizerConfig
from policy_doctor.streamlit_app.config_io import resolve_task_config_stem


def _clustering_level(metadata: List[Dict]) -> str:
    """Match ``behavior_values.build_behavior_graph_from_clustering`` rollout vs demo."""
    return "rollout" if (metadata and "rollout_idx" in metadata[0]) else "demo"


def render_tab(
    config: VisualizerConfig,
    data: Any,
    task_config_stem: Optional[str] = None,
) -> None:
    """Render the Behavior Graph tab."""
    st.header("Behavior Graph")
    task_stem = resolve_task_config_stem(config, task_config_stem)

    labels = st.session_state.get("clustering_labels")
    metadata = st.session_state.get("clustering_metadata")

    if labels is None:
        st.info(
            "No clustering loaded. Go to the **Clustering** tab first to run or load clustering, "
            "or load one below."
        )
        _render_load_clustering(config, task_stem)
        labels = st.session_state.get("clustering_labels")
        metadata = st.session_state.get("clustering_metadata")
        if labels is None:
            return

    n_clusters = len(set(labels) - {-1})
    st.caption(f"Clustering: {len(labels)} slices, {n_clusters} clusters")

    with st.expander("1. Transition graph (no rewards)", expanded=True):
        st.caption(
            "Build the graph from cluster sequences: states are behaviors, edges are empirical transition "
            "probabilities. Terminal nodes (success / failure / end) are **not** assigned numerical rewards here."
        )

        if st.button("Build transition graph", type="primary", key="bg_build_structure"):
            with st.spinner("Building transition graph..."):
                graph = BehaviorGraph.from_cluster_assignments(
                    labels,
                    metadata,
                    level=_clustering_level(metadata),
                )
            st.session_state["bg_graph"] = graph
            st.session_state["bg_slice_labels"] = np.asarray(labels, dtype=np.int64).copy()
            for k in (
                "bg_values",
                "bg_transition_values",
                "bg_q_values",
                "bg_advantages",
                "bg_params",
            ):
                st.session_state.pop(k, None)
            n_arcs = sum(len(tgts) for tgts in graph.transition_counts.values())
            n_steps = sum(sum(tgts.values()) for tgts in graph.transition_counts.values())
            st.success(
                f"Graph: {len(graph.nodes)} nodes, {n_arcs} transition types, "
                f"{n_steps} counted transitions"
            )

        if "bg_graph" not in st.session_state:
            return

        graph = st.session_state["bg_graph"]
        slice_labels = st.session_state.get("bg_slice_labels")
        if slice_labels is None:
            slice_labels = np.asarray(labels, dtype=np.int64).copy()
            st.session_state["bg_slice_labels"] = slice_labels
        _render_structure_visualizations(graph, slice_labels, metadata)

    if "bg_prune_rounds" in st.session_state:
        st.caption(
            f"**Last degree-1 prune:** **{st.session_state['bg_prune_rounds']}** iteration(s), "
            f"**{st.session_state['bg_prune_merges']}** cluster node merge(s). "
            "(Full controls in section 2 below.)"
        )

    with st.expander("2. Pruning (degree-1)", expanded=True):
        _render_pruning_section(labels, metadata)

    with st.expander("3. Markov reward process", expanded=True):
        graph = st.session_state["bg_graph"]
        slice_labels = st.session_state.get("bg_slice_labels")
        if slice_labels is None:
            slice_labels = np.asarray(labels, dtype=np.int64).copy()
            st.session_state["bg_slice_labels"] = slice_labels
        st.caption(
            "Treat the graph as an MRP: set rewards when episodes succeed, fail, or end without a label, "
            "choose discount γ, then solve for **V(s)** and per-transition **transition values** / **advantage** for curation."
        )

        col1, col2 = st.columns(2)
        with col1:
            gamma = st.slider("Discount factor (γ)", 0.0, 1.0, 0.99, 0.01, key="bg_gamma")
            reward_success = st.number_input("Reward (success)", value=1.0, key="bg_r_success")
        with col2:
            reward_failure = st.number_input("Reward (failure)", value=-1.0, key="bg_r_failure")
            reward_end = st.number_input("Reward (end / truncated)", value=0.0, key="bg_r_end")

        if st.button("Compute values, transition values, and advantages", type="primary", key="bg_compute_mrp"):
            with st.spinner("Solving for V(s), per-slice transition values, and advantages..."):
                values = graph.compute_values(
                    gamma=gamma,
                    reward_success=reward_success,
                    reward_failure=reward_failure,
                    reward_end=reward_end,
                )
                transition_values, advantages, _ = graph.compute_slice_values(
                    slice_labels, metadata, values, gamma=gamma
                )
            st.session_state["bg_values"] = values
            st.session_state["bg_transition_values"] = transition_values
            st.session_state.pop("bg_q_values", None)
            st.session_state["bg_advantages"] = advantages
            st.session_state["bg_params"] = {
                "gamma": gamma,
                "reward_success": reward_success,
                "reward_failure": reward_failure,
                "reward_end": reward_end,
            }
            st.success(
                "Computed state values **V(s)**, per-slice **transition values**, and **advantages**."
            )

        if "bg_values" in st.session_state:
            values = st.session_state["bg_values"]
            if "bg_transition_values" in st.session_state:
                transition_values = st.session_state["bg_transition_values"]
            else:
                transition_values = st.session_state.get("bg_q_values")
            advantages = st.session_state["bg_advantages"]
            params = st.session_state.get("bg_params") or {}
            gamma_used = float(params.get("gamma", gamma))

            st.caption(
                "Plots below use the **last** compute (change γ or rewards and compute again to refresh)."
            )
            if transition_values is None:
                st.warning(
                    "Transition values are missing from this session. Run **Compute values, transition values, "
                    "and advantages** again."
                )
            else:
                _render_mrp_visualizations(
                    graph,
                    values,
                    transition_values,
                    advantages,
                    slice_labels,
                    metadata,
                    gamma_used,
                )

    with st.expander("Export", expanded=False):
        _render_export(config, task_stem)


def _render_load_clustering(config: VisualizerConfig, task_stem: str) -> None:
    """Quick clustering loader for the behavior graph tab."""
    from policy_doctor.streamlit_app.config_io import (
        clustering_results_dir_for_task,
        discover_clustering_task_keys,
        list_clustering_results,
        load_task_clustering_result,
        sync_clustering_session_from_manifest,
    )

    task_config = st.text_input(
        "Task config key",
        value=task_stem,
        key="bg_task_cfg",
        help="Task YAML stem; must match the key used when clustering was saved.",
    )
    try:
        available = list_clustering_results(task_config)
    except Exception as e:
        st.error(f"Could not list clustering results: {e}")
        available = []
    if available:
        selected = st.selectbox("Clustering result", available, key="bg_clust_select")
        if st.button("Load clustering", key="bg_clust_load"):
            labels, metadata, manifest = load_task_clustering_result(task_config, selected)
            st.session_state["clustering_labels"] = labels
            st.session_state["clustering_metadata"] = metadata
            st.session_state["clustering_manifest"] = manifest
            st.session_state["clustering_task_config"] = task_config
            sync_clustering_session_from_manifest(manifest)
            st.success(f"Loaded: {len(labels)} slices, {len(set(labels) - {-1})} clusters")
            st.rerun()
    else:
        expect = clustering_results_dir_for_task(task_config)
        st.caption(f"No results for `{task_config}` — expected `{expect}`.")
        discovered = discover_clustering_task_keys()
        if discovered:
            st.caption("Keys with saved clustering: " + ", ".join(discovered))


def _render_pruning_section(
    clustering_labels: np.ndarray,
    metadata: List[Dict],
) -> None:
    """Degree-1 pruning: simplify graph and relabel slices (``BehaviorGraph`` API)."""
    graph = st.session_state["bg_graph"]
    sl = st.session_state.get("bg_slice_labels")
    if sl is None:
        sl = np.asarray(clustering_labels, dtype=np.int64).copy()
        st.session_state["bg_slice_labels"] = sl

    n_before = len(graph.nodes)
    n_clusters = len(set(int(x) for x in sl) - {-1})
    st.metric("Distinct behavior clusters in slice labels", n_clusters)

    if "bg_prune_rounds" in st.session_state:
        pm1, pm2 = st.columns(2)
        with pm1:
            st.metric(
                "Pruning iterations (last run)",
                st.session_state["bg_prune_rounds"],
            )
        with pm2:
            st.metric(
                "Cluster nodes merged (last run)",
                st.session_state["bg_prune_merges"],
            )

    st.caption(
        "Non-special nodes with **one** distinct outgoing neighbor **or** **one** distinct incoming "
        "neighbor (by transition counts) are merged into that neighbor; slice labels update accordingly. "
        "Each **round** merges every such node at once, then rebuilds the graph, until none remain "
        "(skipped merges into START / terminals stay as-is). "
        "**Re-run transition graph** in section 1 to reset from raw clustering."
    )

    if st.button("Run degree-1 pruning", type="primary", key="bg_prune_run"):
        with st.spinner("Pruning graph..."):
            prune_out = graph.simplify_by_degree_one_pruning(sl, metadata)
            if len(prune_out) == 4:
                new_graph, new_labels, n_rounds, n_merged = prune_out
                st.session_state["bg_prune_rounds"] = int(n_rounds)
                st.session_state["bg_prune_merges"] = int(n_merged)
            else:
                new_graph, new_labels = prune_out
                n_rounds, n_merged = 0, 0
                st.session_state.pop("bg_prune_rounds", None)
                st.session_state.pop("bg_prune_merges", None)
        st.session_state["bg_graph"] = new_graph
        st.session_state["bg_slice_labels"] = new_labels
        for k in (
            "bg_values",
            "bg_transition_values",
            "bg_q_values",
            "bg_advantages",
            "bg_params",
        ):
            st.session_state.pop(k, None)
        n_after = len(new_graph.nodes)
        k_after = len(set(int(x) for x in new_labels) - {-1})
        if len(prune_out) == 4:
            st.success(
                f"Pruned in **{n_rounds}** iteration(s) (**{n_merged}** cluster node merge(s)): "
                f"**{n_after}** graph nodes (was {n_before}), **{k_after}** behavior clusters in labels "
                f"(was {n_clusters})."
            )
        else:
            st.success(
                f"Pruned: **{n_after}** graph nodes (was {n_before}), **{k_after}** behavior clusters in labels "
                f"(was {n_clusters})."
            )
            st.info(
                "This environment’s `BehaviorGraph.simplify_by_degree_one_pruning` does not return "
                "iteration/merge stats. Use an editable install from the repo "
                "(`pip install -e …/policy_doctor`) to see those counts."
            )

    st.divider()
    st.markdown("**Interactive graph** (current build / after pruning)")
    viz_graph = st.session_state["bg_graph"]
    min_prob_prune = st.slider(
        "Min edge probability",
        0.0,
        0.5,
        0.0,
        0.01,
        key="bg_min_prob_prune",
    )
    prune_view = st.selectbox(
        "View",
        options=["plain", "timesteps"],
        format_func=lambda x: {
            "plain": "Cluster palette",
            "timesteps": "Timestep count (viridis)",
        }[x],
        key="bg_graph_view_prune",
    )
    from policy_doctor.plotting.plotly.behavior_graph import create_interactive_behavior_graph
    from policy_doctor.plotting.plotly.behavior_graph_timesteps import (
        create_timestep_colored_interactive_graph,
    )

    if prune_view == "timesteps":
        html_prune = create_timestep_colored_interactive_graph(
            viz_graph, min_probability=min_prob_prune
        )
    else:
        html_prune = create_interactive_behavior_graph(
            viz_graph,
            min_probability=min_prob_prune,
            layout_algorithm="layeredStatic",
            physics_enabled=False,
        )
    components.html(html_prune, height=560, scrolling=True)


def _render_structure_visualizations(
    graph: Any,
    labels: np.ndarray,
    metadata: List[Dict],
) -> None:
    """Pyvis + transition matrix only (no V(s), transition values, or advantage)."""
    from policy_doctor import plotting
    from policy_doctor.plotting.plotly.behavior_graph import create_interactive_behavior_graph
    from policy_doctor.plotting.plotly.behavior_graph_timesteps import (
        create_timestep_colored_interactive_graph,
    )

    st.markdown("**Interactive graph**")

    view_mode = st.selectbox(
        "View",
        options=["plain", "timesteps"],
        format_func=lambda x: {
            "plain": "Cluster palette (structure only)",
            "timesteps": "Timestep count (viridis)",
        }[x],
        key="bg_graph_view_structure",
        help="No value function — only topology and transition probabilities.",
    )

    min_prob = st.slider(
        "Min edge probability",
        0.0,
        0.5,
        0.0,
        0.01,
        key="bg_min_prob",
    )

    if view_mode == "timesteps":
        html = create_timestep_colored_interactive_graph(graph, min_probability=min_prob)
    else:
        html = create_interactive_behavior_graph(
            graph,
            min_probability=min_prob,
            layout_algorithm="layeredStatic",
            physics_enabled=False,
        )
    components.html(html, height=700, scrolling=True)

    with st.expander("Transition matrix", expanded=False):
        fig_trans = plotting.create_transition_matrix_heatmap(graph)
        st.plotly_chart(
            fig_trans, use_container_width=True, key="bg_plotly_trans_structure"
        )


def _render_mrp_visualizations(
    graph: Any,
    values: Dict[int, float],
    transition_values: np.ndarray,
    advantages: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict],
    gamma: float,
) -> None:
    """Value-colored graph, node values, advantage views, timelines."""
    from policy_doctor import plotting
    from policy_doctor.plotting.plotly.behavior_graph import (
        create_interactive_behavior_graph,
        create_value_colored_interactive_graph,
    )
    from policy_doctor.plotting.plotly.behavior_graph_timesteps import (
        create_timestep_colored_interactive_graph,
    )

    st.markdown("**Interactive graph (with values)**")

    view_mode = st.selectbox(
        "View",
        options=["value", "plain", "timesteps"],
        format_func=lambda x: {
            "value": "Value-colored (V(s) on nodes, transition values on edges)",
            "plain": "Cluster palette",
            "timesteps": "Timestep count (viridis)",
        }[x],
        key="bg_graph_view_mrp",
    )

    min_prob_mrp = st.slider(
        "Min edge probability",
        0.0,
        0.5,
        0.0,
        0.01,
        key="bg_min_prob_mrp",
    )

    if view_mode == "value":
        html = create_value_colored_interactive_graph(
            graph,
            values,
            gamma=gamma,
            min_probability=min_prob_mrp,
        )
    elif view_mode == "timesteps":
        html = create_timestep_colored_interactive_graph(
            graph, min_probability=min_prob_mrp
        )
    else:
        html = create_interactive_behavior_graph(
            graph,
            min_probability=min_prob_mrp,
            layout_algorithm="layeredStatic",
            physics_enabled=False,
        )
    components.html(html, height=700, scrolling=True)

    with st.expander("Node values (V per state)", expanded=False):
        fig_values = plotting.create_node_value_bar_chart(graph, values)
        st.plotly_chart(
            fig_values, use_container_width=True, key="bg_plotly_node_values_mrp"
        )

    with st.expander("Transition matrix", expanded=False):
        fig_trans = plotting.create_transition_matrix_heatmap(graph)
        st.plotly_chart(
            fig_trans, use_container_width=True, key="bg_plotly_trans_mrp"
        )

    with st.expander("Advantage matrix", expanded=False):
        fig_adv = plotting.create_advantage_matrix_heatmap(graph, values, gamma=gamma)
        st.plotly_chart(fig_adv, use_container_width=True, key="bg_plotly_adv_mrp")

    with st.expander("Episode timelines"):
        ep_indices = sorted(set(m.get("rollout_idx", -1) for m in metadata))
        if len(ep_indices) > 0:
            _n = len(ep_indices)
            _ep_key = "bg_ep_num"
            if _ep_key in st.session_state:
                st.session_state[_ep_key] = int(
                    min(max(st.session_state[_ep_key], 0), _n - 1)
                )
            _ep_i = st.number_input(
                "Episode index",
                min_value=0,
                max_value=_n - 1,
                step=1,
                key=_ep_key,
                help="0-based index into rollout episodes in this clustering (stepper: − / +).",
            )
            selected_ep = ep_indices[int(_ep_i)]
            st.caption(f"Rollout index **{selected_ep}** ({int(_ep_i) + 1} of {_n}).")
            fig_timeline = plotting.create_episode_q_advantage_timeline(
                labels,
                metadata,
                transition_values,
                advantages,
                ep_idx=selected_ep,
                level="rollout",
                height=480,
            )
            st.plotly_chart(
                fig_timeline,
                use_container_width=True,
                key=f"bg_plotly_timeline_mrp_{selected_ep}",
            )

    with st.expander("Advantage distribution"):
        valid = np.isfinite(advantages) & (labels >= 0)
        if valid.any():
            import plotly.graph_objects as go

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=advantages[valid], nbinsx=50, name="Advantage"))
            fig_hist.update_layout(
                title="Advantage distribution",
                xaxis_title="Advantage (transition value − V(s))",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(
                fig_hist, use_container_width=True, key="bg_plotly_adv_hist_mrp"
            )
            st.metric("Mean advantage", f"{advantages[valid].mean():.4f}")


def _render_export(config: VisualizerConfig, task_stem: str) -> None:
    """Export behavior graph config (caller may wrap in an expander)."""
    from policy_doctor.streamlit_app.config_io import render_config_export

    st.caption("Download a YAML snippet for the curation pipeline (uses last-computed rewards if any).")
    params = st.session_state.get("bg_params", {})
    clust_params = st.session_state.get("clustering_params", {})
    task_config = st.session_state.get("clustering_task_config", task_stem)

    pipeline_config = {
        "task_config": task_config,
        "advantage_gamma": params.get("gamma", 0.99),
        "advantage_reward_success": params.get("reward_success", 1.0),
        "advantage_reward_failure": params.get("reward_failure", -1.0),
        "advantage_reward_end": params.get("reward_end", 0.0),
    }
    if clust_params:
        pipeline_config.update({
            "clustering_window_width": clust_params.get("window_width", 5),
            "clustering_stride": clust_params.get("stride", 2),
            "clustering_n_clusters": clust_params.get("n_clusters", 20),
        })

    render_config_export(
        pipeline_config,
        default_filename=f"behavior_graph_{task_config}.yaml",
        label="Download behavior graph config",
        key="bg_export",
    )

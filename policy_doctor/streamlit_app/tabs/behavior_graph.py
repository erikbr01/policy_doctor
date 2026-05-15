"""Behavior Graph tab: build graph from clustering, optional pruning, then MRP analysis.

Workflow:
  1. Load clustering (Clustering tab or disk)
  2. **Transition graph**: structure and transition probabilities (no rewards)
  3. **Pruning**: degree-1 graph simplification and slice relabeling
  4. **Markov reward process**: rewards, γ, V(s), transition values, advantages
  5. Export pipeline YAML
"""

from __future__ import annotations

import pathlib
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

    with st.expander("4. Episode browser", expanded=False):
        _render_episode_browser(
            st.session_state.get("bg_slice_labels", labels),
            metadata,
            config,
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
    _prune_labels = st.session_state.get("bg_slice_labels")
    if _prune_labels is None:
        _prune_labels = np.asarray(clustering_labels, dtype=np.int64).copy()
    _render_graph_with_selector(
        viz_graph, _prune_labels, metadata,
        min_prob_key="bg_min_prob_prune",
        view_key="bg_graph_view_prune",
        graph_key="bg_graph_svg_prune",
        height=560,
    )


def _render_graph_with_selector(
    graph: Any,
    labels: np.ndarray,
    metadata: List[Dict],
    min_prob_key: str,
    view_key: str,
    graph_key: str,
    height: int = 700,
    show_value_option: bool = False,
    values: Optional[Dict] = None,
    gamma: float = 0.99,
) -> None:
    """Shared graph renderer: SVG component or Pyvis, with node panel."""
    from policy_doctor import plotting
    from policy_doctor.plotting.plotly.behavior_graph import create_interactive_behavior_graph
    from policy_doctor.plotting.plotly.behavior_graph_timesteps import (
        create_timestep_colored_interactive_graph,
    )

    renderer_col, view_col = st.columns([1, 2])
    with renderer_col:
        renderer = st.selectbox(
            "Renderer",
            options=["svg", "pyvis"],
            format_func=lambda x: {"svg": "SVG (clickable nodes)", "pyvis": "Pyvis (physics)"}[x],
            key=f"{view_key}_renderer",
        )
    with view_col:
        view_options = ["plain", "timesteps"]
        view_labels = {
            "plain": "Cluster palette",
            "timesteps": "Timestep count (viridis)",
        }
        if show_value_option and values:
            view_options = ["value"] + view_options
            view_labels["value"] = "Value-colored (V(s))"
        view_mode = st.selectbox(
            "Color mode",
            options=view_options,
            format_func=lambda x: view_labels.get(x, x),
            key=view_key,
        )

    min_prob = st.slider(
        "Min edge probability",
        0.0,
        0.5,
        0.0,
        0.01,
        key=min_prob_key,
    )

    # Show what representation was used for this clustering
    clust_src = st.session_state.get("clustering_influence_source") or st.session_state.get(
        "clustering_params", {}
    ).get("clustering_influence_source")
    if clust_src:
        _SRC_LABELS = {
            "infembed": "InfEmbed",
            "state": "State",
            "state_action": "State+Action",
            "trak": "TRAK",
            "policy_emb": "Policy embeddings",
        }
        emb_label = _SRC_LABELS.get(str(clust_src), str(clust_src))
        st.caption(f"Graph built from **{emb_label}** clustering")

    if renderer == "svg":
        from policy_doctor.streamlit_app.user_study.graph_plot import (
            compute_pruned_graph_nodes,
            render_graph_component,
        )
        excluded = compute_pruned_graph_nodes(
            graph, min_visit_prob=0.0, n_total=graph.num_episodes, min_edge_prob=min_prob
        )
        clicked = render_graph_component(
            graph,
            height=height,
            key=graph_key,
            excluded_node_ids=excluded,
            min_edge_prob=min_prob,
        )
        if clicked is not None and clicked in graph.nodes:
            _render_node_panel(clicked, graph, labels, metadata, key_prefix=graph_key)
        else:
            st.caption("Click a node to explore it — stats and transitions appear here.")
    else:
        if show_value_option and view_mode == "value" and values:
            from policy_doctor.plotting.plotly.behavior_graph import create_value_colored_interactive_graph
            html = create_value_colored_interactive_graph(graph, values, gamma=gamma, min_probability=min_prob)
        elif view_mode == "timesteps":
            html = create_timestep_colored_interactive_graph(graph, min_probability=min_prob)
        else:
            html = create_interactive_behavior_graph(
                graph,
                min_probability=min_prob,
                layout_algorithm="layeredStatic",
                physics_enabled=False,
            )
        components.html(html, height=height, scrolling=True)


def _render_node_panel(
    node_id: int,
    graph: Any,
    labels: np.ndarray,
    metadata: List[Dict],
    key_prefix: str,
) -> None:
    """Stats panel for a clicked node in the SVG graph."""
    import plotly.graph_objects as go
    from policy_doctor.behaviors.behavior_graph import FAILURE_NODE_ID, SUCCESS_NODE_ID

    node = graph.nodes.get(node_id)
    if node is None:
        return

    with st.container(border=True):
        header_col, close_col = st.columns([10, 1])
        with header_col:
            st.subheader(f"Node: {node.name}")
        with close_col:
            if st.button("✕", key=f"{key_prefix}_panel_close", help="Dismiss"):
                st.session_state.pop(f"{key_prefix}_selected", None)
                st.rerun()

        # Episode success stats
        ep_key = "rollout_idx" if graph.level == "rollout" else "demo_idx"
        ep_success: Dict = {}
        for meta in metadata:
            eidx = meta.get(ep_key)
            if eidx is not None and eidx not in ep_success:
                ep_success[eidx] = meta.get("success")

        success_count = sum(1 for eidx in node.episode_indices if ep_success.get(eidx) is True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Timesteps", node.num_timesteps)
        m2.metric("Episodes", node.num_episodes)
        m3.metric(
            "Success rate",
            f"{success_count / node.num_episodes:.0%}" if node.num_episodes else "—"
        )

        # Outgoing transitions bar chart
        outgoing = graph.transition_probs.get(node_id, {})
        if outgoing and not node.is_special:
            tgt_labels = [graph.nodes[t].name if t in graph.nodes else str(t) for t in outgoing]
            tgt_probs = list(outgoing.values())
            bar_colors = [
                "#2ca02c" if t == SUCCESS_NODE_ID else "#d62728" if t == FAILURE_NODE_ID else "#1f77b4"
                for t in outgoing
            ]
            fig_trans = go.Figure(go.Bar(
                x=tgt_probs, y=tgt_labels, orientation="h", marker_color=bar_colors
            ))
            fig_trans.update_layout(
                title="Where does this node lead?",
                height=max(100, 32 * len(tgt_labels) + 60),
                margin=dict(l=0, r=0, t=36, b=0),
                xaxis=dict(range=[0, 1], title="Probability"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_trans, use_container_width=True, key=f"{key_prefix}_node_trans")


def _render_structure_visualizations(
    graph: Any,
    labels: np.ndarray,
    metadata: List[Dict],
) -> None:
    """Graph structure (no value function)."""
    from policy_doctor import plotting

    st.markdown("**Interactive graph**")
    _render_graph_with_selector(
        graph, labels, metadata,
        min_prob_key="bg_min_prob",
        view_key="bg_graph_view_structure",
        graph_key="bg_graph_svg_structure",
        height=700,
    )

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

    st.markdown("**Interactive graph (with values)**")
    _render_graph_with_selector(
        graph, labels, metadata,
        min_prob_key="bg_min_prob_mrp",
        view_key="bg_graph_view_mrp",
        graph_key="bg_graph_svg_mrp",
        height=700,
        show_value_option=True,
        values=values,
        gamma=gamma,
    )

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


def _build_per_frame_labels(
    episode_idx: int,
    labels: np.ndarray,
    metadata: List[Dict],
    ep_key: str = "rollout_idx",
) -> np.ndarray:
    """Reconstruct a per-frame cluster label array for one episode.

    Each slice covers [window_start, window_end). Frames are assigned
    to the label of the last slice whose window_start ≤ frame.
    Returns an int64 array of length = last window_end for the episode.
    """
    ep_slices = [
        (m["window_start"], m["window_end"], int(labels[i]))
        for i, m in enumerate(metadata)
        if m.get(ep_key) == episode_idx and int(labels[i]) != -1
    ]
    if not ep_slices:
        return np.array([], dtype=np.int64)

    total = max(s[1] for s in ep_slices)
    per_frame = np.full(total, -1, dtype=np.int64)
    for ws, we, lbl in sorted(ep_slices, key=lambda x: x[0]):
        per_frame[ws:we] = lbl
    return per_frame


def _render_episode_browser(
    labels: np.ndarray,
    metadata: List[Dict],
    config: Any,
) -> None:
    """Browse episodes with multi-cluster timeline annotations and optional video."""
    from policy_doctor.plotting.plotly.clusters import CLUSTER_COLORS
    from policy_doctor.streamlit_app.components.mp4_player import cluster_timeline, mp4_player

    ep_key = "rollout_idx" if (metadata and "rollout_idx" in metadata[0]) else "demo_idx"
    ep_indices = sorted(set(m.get(ep_key, -1) for m in metadata if m.get(ep_key, -1) >= 0))
    if not ep_indices:
        st.info("No episode indices found in clustering metadata.")
        return

    cluster_ids = sorted(set(int(l) for l in labels if l >= 0))
    n_clusters = len(cluster_ids)

    st.caption(
        "Select an episode to see its cluster assignment timeline. "
        "Each color segment is a distinct behavior cluster. "
        "Point to a segment to see its cluster ID and time range."
    )

    col_ep, col_fps = st.columns([3, 1])
    with col_ep:
        ep_i = st.selectbox(
            "Episode",
            ep_indices,
            format_func=lambda i: f"Episode {i} — {'✓ success' if _ep_success(i, metadata, ep_key) is True else '✗ failure' if _ep_success(i, metadata, ep_key) is False else '?'}",
            key="bg_ep_browser_idx",
        )
    with col_fps:
        fps = st.number_input("FPS", 1, 60, 10, key="bg_ep_browser_fps")

    per_frame = _build_per_frame_labels(ep_i, labels, metadata, ep_key)
    if len(per_frame) == 0:
        st.warning(f"No labeled slices found for episode {ep_i}.")
        return

    total_frames = len(per_frame)

    # Cluster name map
    graph = st.session_state.get("bg_graph")
    cluster_names = {}
    if graph is not None:
        for nid, node in graph.nodes.items():
            if nid >= 0:
                cluster_names[nid] = node.name

    # Multi-cluster timeline (always shown)
    st.markdown(f"**Cluster timeline** — episode {ep_i}  ({total_frames} frames)")
    cluster_timeline(
        per_frame,
        cluster_colors=CLUSTER_COLORS,
        fps=fps,
        height=80,
        key=f"bg_ep_tl_{ep_i}",
        cluster_names=cluster_names or None,
    )

    # Cluster sequence summary
    with st.expander("Cluster sequence", expanded=False):
        _render_cluster_sequence(per_frame, cluster_ids, CLUSTER_COLORS, cluster_names)

    # Optional video with timeline
    st.markdown("**Video**")
    st.caption(
        "Point to a directory containing ``ep{N}.mp4`` files. "
        "Leave blank to skip video playback."
    )
    mp4_dir_str = st.text_input(
        "MP4 directory",
        value=st.session_state.get("bg_ep_mp4_dir", ""),
        key="bg_ep_mp4_dir_input",
        placeholder="/path/to/mp4_dir",
    )
    if mp4_dir_str:
        st.session_state["bg_ep_mp4_dir"] = mp4_dir_str
        mp4_dir = pathlib.Path(mp4_dir_str)
        mp4_path = mp4_dir / f"ep{ep_i}.mp4"
        if not mp4_path.exists():
            st.warning(f"Not found: `{mp4_path}`")
        else:
            mp4_player(
                mp4_path,
                key=f"bg_ep_vid_{ep_i}",
                max_height_px=360,
                fps=fps,
                total_frames=total_frames,
            )
            # Also show current-frame cluster info via frame slider
            frame_i = st.slider(
                "Inspect frame",
                0, total_frames - 1, 0,
                key=f"bg_ep_frame_{ep_i}",
            )
            cid = int(per_frame[frame_i])
            cname = cluster_names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled"
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] if cid >= 0 else "#555"
            st.markdown(
                f"Frame **{frame_i}** → "
                f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;">'
                f"{cname}</span>",
                unsafe_allow_html=True,
            )
    else:
        # No video: show frame-level cluster inspector from PKL if eval_dir is set
        if config.eval_dir:
            import pathlib as _pl
            from policy_doctor.paths import REPO_ROOT
            eval_path = _pl.Path(config.eval_dir)
            if not eval_path.is_absolute():
                eval_path = REPO_ROOT / eval_path
            pkls = sorted(eval_path.glob("episodes/ep*.pkl"))
            if pkls and ep_i < len(pkls):
                frame_i = st.slider(
                    "Inspect frame",
                    0, total_frames - 1, 0,
                    key=f"bg_ep_frame_{ep_i}",
                )
                cid = int(per_frame[frame_i])
                cname = cluster_names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled"
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] if cid >= 0 else "#555"
                st.markdown(
                    f"Frame **{frame_i}** → "
                    f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;">'
                    f"{cname}</span>",
                    unsafe_allow_html=True,
                )
                if st.button("Load frame from PKL", key=f"bg_ep_pklfr_{ep_i}"):
                    import pickle
                    with open(pkls[ep_i], "rb") as f:
                        df = pickle.load(f)
                    if frame_i < len(df):
                        row = df.iloc[frame_i]
                        if "obs" in row:
                            _show_obs_frame(row["obs"])


def _ep_success(ep_idx: int, metadata: List[Dict], ep_key: str) -> Optional[bool]:
    for m in metadata:
        if m.get(ep_key) == ep_idx:
            return m.get("success")
    return None


def _render_cluster_sequence(
    per_frame: np.ndarray,
    cluster_ids: List[int],
    colors: List[str],
    names: dict,
) -> None:
    """Show the run-length-encoded sequence of cluster visits."""
    import pandas as pd

    n = len(colors)
    rows = []
    prev = None
    for frame_i, cid in enumerate(per_frame):
        cid = int(cid)
        if cid != prev:
            rows.append({"cluster": names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled",
                         "start_frame": frame_i, "color": colors[cid % n] if cid >= 0 else "#555"})
            prev = cid
    for i in range(len(rows) - 1):
        rows[i]["end_frame"] = rows[i + 1]["start_frame"] - 1
    if rows:
        rows[-1]["end_frame"] = len(per_frame) - 1

    df = pd.DataFrame(rows)[["cluster", "start_frame", "end_frame"]]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _show_obs_frame(obs: Any) -> None:
    """Try to render an observation array as an image."""
    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        from PIL import Image
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        st.image(Image.fromarray(arr.astype(np.uint8)), use_container_width=True)
    elif arr.ndim == 3:
        # Might be (C, H, W) — try transposing
        arr_t = arr.transpose(1, 2, 0)
        if arr_t.shape[-1] in (1, 3, 4):
            _show_obs_frame(arr_t)
        else:
            st.caption(f"obs shape {arr.shape} — cannot render as image")
    else:
        st.caption(f"obs shape {arr.shape} — not an image")


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

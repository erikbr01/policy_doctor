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


def _resolve_mp4_dir(config: Any) -> Optional[pathlib.Path]:
    """Return eval_dir/media if it exists and contains ep*.mp4 files."""
    if not config.eval_dir:
        return None
    from policy_doctor.paths import REPO_ROOT
    eval_path = pathlib.Path(config.eval_dir)
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path
    media = eval_path / "media"
    if media.exists() and any(media.glob("ep*.mp4")):
        return media
    return None


def _find_ep_mp4(mp4_dir: pathlib.Path, ep_idx: int) -> Optional[pathlib.Path]:
    """Find the mp4 for episode ep_idx, handling both ep0.mp4 and ep0000_succ.mp4 conventions."""
    # Zero-padded with suffix (ep0042_succ.mp4 / ep0042_fail.mp4)
    matches = sorted(mp4_dir.glob(f"ep{ep_idx:04d}_*.mp4"))
    if matches:
        return matches[0]
    # Simple name (ep42.mp4)
    simple = mp4_dir / f"ep{ep_idx}.mp4"
    return simple if simple.exists() else None


def render_tab(
    config: VisualizerConfig,
    data: Any,
    task_config_stem: Optional[str] = None,
) -> None:
    """Render the Behavior Graph tab."""
    st.header("Behavior Graph")
    task_stem = resolve_task_config_stem(config, task_config_stem)

    # Auto-resolve mp4 dir from config once per session
    if "bg_ep_mp4_dir" not in st.session_state:
        auto = _resolve_mp4_dir(config)
        if auto:
            st.session_state["bg_ep_mp4_dir"] = str(auto)

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
    """Unified graph visualizer with a viz-type dropdown.

    Defaults to the clickable native-SVG trajectory tree (root → distinct
    leaves, no cycles). Other options: tree variants (sunburst / icicle /
    treemap) for compactness, and the original Markov graph (BFS-layered or
    temporal-mean) for transition-probability semantics.
    """
    from policy_doctor.plotting.plotly.behavior_graph import create_interactive_behavior_graph
    from policy_doctor.plotting.plotly.behavior_graph_timesteps import (
        create_timestep_colored_interactive_graph,
    )

    # ── Visualization-type dropdown ───────────────────────────────────────
    VIZ_OPTIONS = [
        "tree_native_svg",
        "tree_sunburst",
        "tree_icicle",
        "tree_treemap",
        "markov_svg_bfs",
        "markov_svg_temporal",
        "markov_pyvis",
    ]
    VIZ_LABELS = {
        "tree_native_svg":     "🌳 Trajectory tree (clickable nodes)  ← default",
        "tree_sunburst":       "🌞 Trajectory tree (sunburst)",
        "tree_icicle":         "📊 Trajectory tree (icicle)",
        "tree_treemap":        "🟦 Trajectory tree (treemap)",
        "markov_svg_bfs":      "🔁 Markov graph — BFS-layered (clickable)",
        "markov_svg_temporal": "🕒 Markov graph — temporal mean (clickable)",
        "markov_pyvis":        "🔧 Markov graph — Pyvis (physics)",
    }
    viz_type = st.selectbox(
        "Visualization",
        options=VIZ_OPTIONS,
        format_func=lambda v: VIZ_LABELS[v],
        index=0,
        key=f"{view_key}_viz",
        help=(
            "Tree views show each episode's run-length-collapsed sequence as a "
            "path from a shared START root to a per-branch terminal; no cycles. "
            "Markov-graph views show transition probabilities between clusters."
        ),
    )
    is_tree = viz_type.startswith("tree_")

    # ── Tree-only controls ────────────────────────────────────────────────
    if is_tree:
        c_mb, c_md = st.columns(2)
        with c_mb:
            min_branch = st.slider(
                "Hide branches reaching fewer than N episodes",
                1, 50, 2, key=f"{view_key}_minbranch",
            )
        with c_md:
            max_depth_cap = st.slider(
                "Max depth (rarely needs to cap)",
                2, 500, 500, key=f"{view_key}_maxdepth",
            )
    else:
        min_branch = 2
        max_depth_cap = 500

    # ── Markov-only controls (color mode + edge-prob threshold) ───────────
    if not is_tree:
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
    else:
        view_mode = "plain"

    min_prob = st.slider(
        "Min edge probability",
        0.0, 0.5, 0.0, 0.01,
        key=min_prob_key,
    )

    # ── Caption: which feature space the clustering came from ─────────────
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

    # ── Dispatch ──────────────────────────────────────────────────────────
    if is_tree:
        from policy_doctor.streamlit_app.components.trajectory_tree_view import (
            render_trajectory_tree,
        )
        # Resolve MP4s for the click-to-explore panel
        mp4_dir = st.session_state.get("mp4_dir")
        mp4_index = st.session_state.get("mp4_index") or {"episodes": []}
        tree_view = viz_type.replace("tree_", "")  # native_svg / sunburst / ...
        render_trajectory_tree(
            labels=np.asarray(labels, dtype=np.int64),
            metadata=metadata,
            view_mode=tree_view,
            min_branch=int(min_branch),
            max_depth_cap=int(max_depth_cap),
            cluster_names=None,
            mp4_dir=mp4_dir,
            mp4_index=mp4_index,
            height=height,
            level=getattr(graph, "level", "rollout"),
            key_prefix=f"{graph_key}_tree",
        )
        return

    # ── Markov graph branches ─────────────────────────────────────────────
    if viz_type in ("markov_svg_bfs", "markov_svg_temporal"):
        from policy_doctor.streamlit_app.user_study.graph_plot import (
            compute_pruned_graph_nodes,
            render_graph_component,
        )
        excluded = compute_pruned_graph_nodes(
            graph, min_visit_prob=0.0, n_total=graph.num_episodes, min_edge_prob=min_prob
        )
        pos = None
        if viz_type == "markov_svg_temporal":
            from policy_doctor.behaviors import graph_simplification as gs
            try:
                pos = gs.temporal_layout(
                    graph, np.asarray(labels, dtype=np.int64), metadata,
                    level=getattr(graph, "level", "rollout"),
                )
            except Exception as e:
                st.warning(f"Temporal layout failed ({e}); falling back to BFS-layered.")
                pos = None
        clicked = render_graph_component(
            graph,
            height=height,
            key=graph_key,
            excluded_node_ids=excluded,
            min_edge_prob=min_prob,
            pos=pos,
        )
        selected_edge = st.session_state.get(f"{graph_key}_selected_edge")
        if clicked is not None and clicked in graph.nodes:
            _render_node_panel(clicked, graph, labels, metadata, key_prefix=graph_key)
        elif selected_edge is not None:
            src_id, tgt_id = selected_edge
            if src_id in graph.nodes and tgt_id in graph.nodes:
                _render_edge_panel(src_id, tgt_id, graph, labels, metadata, key_prefix=graph_key)
            else:
                st.caption("Click a node or edge to explore it.")
        else:
            st.caption("Click a node or edge to explore it.")
    else:  # markov_pyvis
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
    from policy_doctor.streamlit_app.components.mp4_player import mp4_player

    node = graph.nodes.get(node_id)
    if node is None:
        return

    ep_key = "rollout_idx" if graph.level == "rollout" else "demo_idx"

    with st.container(border=True):
        header_col, close_col = st.columns([10, 1])
        with header_col:
            st.subheader(f"Node: {node.name}")
        with close_col:
            if st.button("✕", key=f"{key_prefix}_panel_close", help="Dismiss"):
                st.session_state.pop(f"{key_prefix}_selected", None)
                st.rerun()

        # Episode success stats
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

        # Videos
        if not node.is_special:
            mp4_dir_str = st.session_state.get("bg_ep_mp4_dir", "")
            if not mp4_dir_str:
                st.caption("No MP4 directory configured — check `eval_dir` in the task config YAML.")
            else:
                mp4_dir = pathlib.Path(mp4_dir_str)
                ep_slices = _episodes_for_node(node_id, labels, metadata, ep_key)
                ep_slices_map = {e[0]: (e[1], e[2]) for e in ep_slices}
                all_ep_idxs = sorted(ep_slices_map.keys())
                n_eps = len(all_ep_idxs)
                _VIDS_PER_PAGE = 3

                if all_ep_idxs:
                    _vp_key = f"{key_prefix}_vid_page_{node_id}"
                    _vp = st.session_state.get(_vp_key, 0)
                    _vp_total = max(1, (n_eps + _VIDS_PER_PAGE - 1) // _VIDS_PER_PAGE)
                    show_ep_idxs = all_ep_idxs[_vp * _VIDS_PER_PAGE:(_vp + 1) * _VIDS_PER_PAGE]

                    if _vp_total > 1:
                        _vc1, _vc2, _vc3 = st.columns([2, 8, 1])
                        with _vc1:
                            if st.button("←", disabled=(_vp == 0), key=f"{key_prefix}_vp_prev_{node_id}"):
                                st.session_state[_vp_key] = max(0, _vp - 1)
                                st.rerun()
                        _vc2.markdown(
                            f"<div style='text-align:center;padding-top:6px;color:#888;font-size:0.82em;'>"
                            f"Episodes {_vp * _VIDS_PER_PAGE + 1}–"
                            f"{min((_vp + 1) * _VIDS_PER_PAGE, n_eps)} of {n_eps}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        with _vc3:
                            if st.button("→", disabled=(_vp >= _vp_total - 1), key=f"{key_prefix}_vp_next_{node_id}"):
                                st.session_state[_vp_key] = min(_vp_total - 1, _vp + 1)
                                st.rerun()

                    vid_cols = st.columns(min(3, len(show_ep_idxs)))
                    available = 0
                    for col, ep_idx in zip(vid_cols, show_ep_idxs):
                        mp4_path = _find_ep_mp4(mp4_dir, ep_idx)
                        if mp4_path is None:
                            continue
                        available += 1
                        success = ep_success.get(ep_idx)
                        status = "✓ Success" if success is True else "✗ Failure" if success is False else ""
                        ts_range = ep_slices_map.get(ep_idx)
                        total_frames = _ep_total_frames(ep_idx, metadata, ep_key)
                        effective_end = (
                            total_frames if (ts_range and success is False and total_frames)
                            else (ts_range[1] if ts_range else None)
                        )
                        with col:
                            st.caption(f"Episode {ep_idx} — {status}")
                            mp4_player(
                                mp4_path,
                                key=f"{key_prefix}_panel_vid_{node_id}_{ep_idx}",
                                max_height_px=220,
                                slice_start=ts_range[0] if ts_range else None,
                                slice_end=effective_end,
                                total_frames=total_frames,
                            )
                    if available == 0:
                        st.info(f"No ep*.mp4 files found in `{mp4_dir}` for this node's episodes.")

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


def _episodes_for_node(
    node_id: int,
    labels: np.ndarray,
    metadata: List[Dict],
    ep_key: str = "rollout_idx",
) -> List[tuple]:
    """Return (ep_idx, ts_min, ts_max) for each episode that visits node_id."""
    from collections import defaultdict
    ep_slices: Dict = defaultdict(list)
    for i, meta in enumerate(metadata):
        if int(labels[i]) == node_id:
            ep_idx = meta.get(ep_key, -1)
            ts = meta.get("window_start", meta.get("timestep", 0))
            ep_slices[ep_idx].append(ts)
    result = []
    for ep_idx, timesteps in ep_slices.items():
        if ep_idx >= 0:
            result.append((ep_idx, min(timesteps), max(timesteps)))
    return sorted(result, key=lambda x: x[0])


def _ep_total_frames(
    ep_idx: int,
    metadata: List[Dict],
    ep_key: str = "rollout_idx",
) -> Optional[int]:
    """Return the max window_end across all slices of an episode."""
    ends = [
        m["window_end"]
        for m in metadata
        if m.get(ep_key) == ep_idx and m.get("window_end") is not None
    ]
    return max(ends) if ends else None


def _episodes_for_edge(
    src_id: int,
    tgt_id: int,
    labels: np.ndarray,
    metadata: List[Dict],
) -> List[tuple]:
    """Return (ep_idx, ts_src, ts_tgt, ts_tgt_end) for each episode containing src→tgt.

    ts_tgt_end is the window_start of the behavior after tgt (or None if tgt is last).
    Handles terminal edges (START/SUCCESS/FAILURE) the same way as the user study.
    """
    from policy_doctor.behaviors.behavior_graph import FAILURE_NODE_ID, START_NODE_ID, SUCCESS_NODE_ID

    ep_key = "rollout_idx" if any("rollout_idx" in m for m in metadata) else "demo_idx"
    ep_wins: Dict = {}
    ep_success: Dict = {}
    for i, m in enumerate(metadata):
        ep_idx = m.get(ep_key)
        if ep_idx is None:
            continue
        lab = int(labels[i])
        ts = m.get("window_start", m.get("timestep", 0))
        ep_wins.setdefault(ep_idx, []).append((ts, lab))
        if ep_idx not in ep_success:
            ep_success[ep_idx] = m.get("success")

    result = []
    for ep_idx, wins in ep_wins.items():
        wins.sort()
        rle: List[tuple] = []
        for ts, lab in wins:
            if lab == -1:
                continue
            if not rle or rle[-1][1] != lab:
                rle.append((ts, lab))
        if not rle:
            continue

        first_ts, first_lab = rle[0]
        last_ts, last_lab = rle[-1]
        success = ep_success.get(ep_idx)

        if src_id == START_NODE_ID:
            if tgt_id == first_lab:
                tgt_end = rle[1][0] if len(rle) > 1 else None
                result.append((ep_idx, first_ts, first_ts, tgt_end))
            continue
        if tgt_id == FAILURE_NODE_ID:
            if last_lab == src_id and success is False:
                # ts_tgt=None → extend orange bar to episode end in the panel
                result.append((ep_idx, last_ts, None, None))
            continue
        if tgt_id == SUCCESS_NODE_ID:
            if last_lab == src_id and success is True:
                result.append((ep_idx, last_ts, None, None))
            continue

        for i in range(len(rle) - 1):
            if rle[i][1] == src_id and rle[i + 1][1] == tgt_id:
                ts_tgt_end = rle[i + 2][0] if i + 2 < len(rle) else None
                result.append((ep_idx, rle[i][0], rle[i + 1][0], ts_tgt_end))
                break

    return sorted(result)


def _render_edge_panel(
    src_id: int,
    tgt_id: int,
    graph: Any,
    labels: np.ndarray,
    metadata: List[Dict],
    key_prefix: str,
) -> None:
    """Panel for a clicked edge: paginated videos showing the src→tgt transition."""
    from policy_doctor.behaviors.behavior_graph import FAILURE_NODE_ID, SUCCESS_NODE_ID
    from policy_doctor.streamlit_app.components.mp4_player import mp4_player

    src_node = graph.nodes.get(src_id)
    tgt_node = graph.nodes.get(tgt_id)
    src_name = src_node.name if src_node else str(src_id)
    tgt_name = tgt_node.name if tgt_node else str(tgt_id)
    prob = graph.transition_probs.get(src_id, {}).get(tgt_id, 0.0)

    ep_key = "rollout_idx" if graph.level == "rollout" else "demo_idx"
    mp4_dir_str = st.session_state.get("bg_ep_mp4_dir", "")
    mp4_dir = pathlib.Path(mp4_dir_str) if mp4_dir_str else None

    with st.container(border=True):
        header_col, close_col = st.columns([10, 1])
        with header_col:
            st.subheader(f"{src_name}  →  {tgt_name}")
            st.caption(f"Transition probability: {prob:.1%}")
        with close_col:
            if st.button("✕", key=f"{key_prefix}_edge_panel_close", help="Dismiss"):
                st.session_state.pop(f"{key_prefix}_selected_edge", None)
                st.rerun()

        all_triples = _episodes_for_edge(src_id, tgt_id, labels, metadata)
        n_eps = len(all_triples)

        if not all_triples:
            st.info("No episodes found for this transition.")
            return

        if mp4_dir is None:
            st.caption(f"{n_eps} episodes match this transition. Set an MP4 directory in the Episode Browser to see videos.")
            return

        _VIDS_PER_PAGE = 3
        _vp_key = f"{key_prefix}_edge_vid_page_{src_id}_{tgt_id}"
        _vp = st.session_state.get(_vp_key, 0)
        _vp_total = max(1, (n_eps + _VIDS_PER_PAGE - 1) // _VIDS_PER_PAGE)
        show_triples = all_triples[_vp * _VIDS_PER_PAGE:(_vp + 1) * _VIDS_PER_PAGE]

        if _vp_total > 1:
            _vc1, _vc2, _vc3 = st.columns([2, 8, 1])
            with _vc1:
                if st.button("←", disabled=(_vp == 0), key=f"{key_prefix}_ep_prev_{src_id}_{tgt_id}"):
                    st.session_state[_vp_key] = max(0, _vp - 1)
                    st.rerun()
            _vc2.markdown(
                f"<div style='text-align:center;padding-top:6px;color:#888;font-size:0.82em;'>"
                f"Episodes {_vp * _VIDS_PER_PAGE + 1}–{min((_vp + 1) * _VIDS_PER_PAGE, n_eps)} of {n_eps}"
                f"</div>", unsafe_allow_html=True)
            with _vc3:
                if st.button("→", disabled=(_vp >= _vp_total - 1), key=f"{key_prefix}_ep_next_{src_id}_{tgt_id}"):
                    st.session_state[_vp_key] = min(_vp_total - 1, _vp + 1)
                    st.rerun()

        vid_cols = st.columns(min(3, len(show_triples)))
        available = 0
        for col, (ep_idx, ts_src, ts_tgt, ts_tgt_end) in zip(vid_cols, show_triples):
            mp4_path = _find_ep_mp4(mp4_dir, ep_idx)
            if mp4_path is None:
                continue
            available += 1
            total_frames = _ep_total_frames(ep_idx, metadata, ep_key)
            # For terminal transitions (X→FAILURE/SUCCESS), ts_tgt is None — extend
            # the orange bar to episode end and omit the sky-blue target bar.
            effective_tgt = ts_tgt if ts_tgt is not None else total_frames
            with col:
                st.caption(f"Episode {ep_idx}")
                mp4_player(
                    mp4_path,
                    key=f"{key_prefix}_edge_vid_{src_id}_{tgt_id}_{ep_idx}",
                    max_height_px=220,
                    slice_start=ts_src,
                    slice_end=effective_tgt,
                    total_frames=total_frames,
                    slice2_start=ts_tgt if (ts_tgt is not None and ts_tgt_end is not None) else None,
                    slice2_end=ts_tgt_end,
                    bar1_label=src_name,
                    bar2_label=tgt_name if (ts_tgt is not None and ts_tgt_end is not None) else "",
                )
        if available == 0:
            st.info(f"No ep*.mp4 files found in `{mp4_dir}` for this transition's episodes.")


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
        "Select an episode to see its cluster assignment timeline and video. "
        "Each color segment is a distinct behavior cluster."
    )

    mp4_dir_str = st.session_state.get("bg_ep_mp4_dir", "")
    mp4_dir = pathlib.Path(mp4_dir_str) if mp4_dir_str else None

    with st.expander("MP4 directory override", expanded=(mp4_dir is None)):
        override = st.text_input(
            "MP4 directory",
            value=mp4_dir_str,
            key="bg_ep_mp4_dir_input",
            placeholder="/path/to/media",
            help="Leave blank to use eval_dir/media from the task config.",
        )
        if override and override != mp4_dir_str:
            st.session_state["bg_ep_mp4_dir"] = override
            mp4_dir = pathlib.Path(override)
    if mp4_dir:
        st.caption(f"Videos from: `{mp4_dir}`")

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

    # Cluster sequence summary
    with st.expander("Cluster sequence", expanded=False):
        _render_cluster_sequence(per_frame, cluster_ids, CLUSTER_COLORS, cluster_names)

    # Video + frame inspector
    if mp4_dir is not None:
        mp4_path = _find_ep_mp4(mp4_dir, ep_i)
        if mp4_path is None:
            st.warning(f"No ep*.mp4 found for episode {ep_i} in `{mp4_dir}`")
            # Fallback: show Plotly cluster timeline when no video
            st.markdown(f"**Cluster timeline** — episode {ep_i}  ({total_frames} frames)")
            cluster_timeline(per_frame, cluster_colors=CLUSTER_COLORS, fps=fps,
                             height=80, key=f"bg_ep_tl_{ep_i}", cluster_names=cluster_names or None)
        else:
            mp4_player(
                mp4_path,
                key=f"bg_ep_vid_{ep_i}",
                max_height_px=360,
                fps=fps,
                total_frames=total_frames,
                per_frame_labels=per_frame,
                cluster_colors=CLUSTER_COLORS,
                cluster_names=cluster_names or None,
            )
    else:
        st.caption("Enter an MP4 directory above to see video playback.")
        # Show Plotly cluster timeline when no video is available
        st.markdown(f"**Cluster timeline** — episode {ep_i}  ({total_frames} frames)")
        cluster_timeline(per_frame, cluster_colors=CLUSTER_COLORS, fps=fps,
                         height=80, key=f"bg_ep_tl_{ep_i}", cluster_names=cluster_names or None)


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

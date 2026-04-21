"""Graph Methods tab: compare BehaviorGraphs from different construction methods.

Loads completed ``build_behavior_graph`` outputs from a comparison run directory
(produced by ``scripts/compare_graph_methods.py``) and shows:

  - Summary stats table (nodes, edges, level, builder)
  - Side-by-side interactive network graphs
  - Per-method degree and edge-probability distributions
  - ENAP-specific edge action-prior magnitudes (when available)
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.paths import REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_completed_methods(base_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Return {method: graph_json_path} for every completed build_behavior_graph step."""
    found: Dict[str, pathlib.Path] = {}
    if not base_dir.is_dir():
        return found
    for method_dir in sorted(base_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        done = method_dir / "build_behavior_graph" / "done"
        graph_json = method_dir / "build_behavior_graph" / "behavior_graph.json"
        if done.exists() and graph_json.exists():
            found[method_dir.name] = graph_json
        else:
            # Per-seed graphs (cupid): look for behavior_graph_seed*.json
            seed_files = sorted(
                (method_dir / "build_behavior_graph").glob("behavior_graph_seed*.json")
            )
            if seed_files and done.exists():
                found[method_dir.name] = seed_files[0]  # first seed as representative
    return found


def _load_graph(path: pathlib.Path) -> Optional[BehaviorGraph]:
    try:
        with open(path) as f:
            return BehaviorGraph.from_dict(json.load(f))
    except Exception as e:
        st.error(f"Failed to load graph from {path}: {e}")
        return None


def _graph_stats(graph: BehaviorGraph) -> dict:
    cluster_nodes = [n for n in graph.nodes.values() if not n.is_special]
    all_edges = [
        e
        for src_edges in graph.edges.values()
        for e in src_edges.values()
    ]
    behavioral_edges = [
        e
        for src_id, src_edges in graph.edges.items()
        for tgt_id, e in src_edges.items()
        if not graph.nodes[src_id].is_special and not graph.nodes[tgt_id].is_special
    ]
    has_action_prior = any(
        e.action_prior is not None for e in all_edges
    )
    return {
        "nodes": len(cluster_nodes),
        "edges": len(behavioral_edges),
        "total_edges": len(all_edges),
        "level": graph.level,
        "builder": graph.builder,
        "episodes": graph.num_episodes,
        "has_action_prior": has_action_prior,
    }


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

def _edge_prob_histogram(
    graphs: Dict[str, BehaviorGraph],
    min_prob: float = 0.0,
) -> go.Figure:
    fig = go.Figure()
    for method, graph in graphs.items():
        probs = [
            e.probability
            for src_id, src_edges in graph.edges.items()
            for e in src_edges.values()
            if not graph.nodes[src_id].is_special
            and e.probability >= min_prob
        ]
        if probs:
            fig.add_trace(go.Histogram(
                x=probs,
                name=method,
                opacity=0.65,
                nbinsx=30,
                histnorm="probability density",
            ))
    fig.update_layout(
        barmode="overlay",
        title="Edge transition probability distribution",
        xaxis_title="Transition probability",
        yaxis_title="Density",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _out_degree_histogram(graphs: Dict[str, BehaviorGraph]) -> go.Figure:
    fig = go.Figure()
    for method, graph in graphs.items():
        degrees = [
            len(graph.edges.get(nid, {}))
            for nid in graph.nodes
            if not graph.nodes[nid].is_special
        ]
        if degrees:
            fig.add_trace(go.Histogram(
                x=degrees,
                name=method,
                opacity=0.65,
                xbins=dict(size=1),
            ))
    fig.update_layout(
        barmode="overlay",
        title="Out-degree distribution (behavior nodes)",
        xaxis_title="Out-degree",
        yaxis_title="Count",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _action_prior_magnitude_chart(graphs: Dict[str, BehaviorGraph]) -> Optional[go.Figure]:
    """Box plot of ||action_prior||₂ per method — ENAP methods only."""
    data: Dict[str, List[float]] = {}
    for method, graph in graphs.items():
        mags = [
            float(np.linalg.norm(e.action_prior))
            for src_edges in graph.edges.values()
            for e in src_edges.values()
            if e.action_prior is not None
        ]
        if mags:
            data[method] = mags

    if not data:
        return None

    fig = go.Figure()
    for method, mags in data.items():
        fig.add_trace(go.Box(y=mags, name=method, boxpoints="outliers"))
    fig.update_layout(
        title="Edge action-prior ‖a‖₂ (ENAP methods)",
        yaxis_title="‖action_prior‖₂",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_tab() -> None:
    st.header("Graph Methods")
    st.caption(
        "Compare BehaviorGraphs built by different methods (CuPID, ENAP-custom, ENAP). "
        "Point this tab at the output of ``scripts/compare_graph_methods.py``."
    )

    # --- Directory picker ---
    default_dir = str(REPO_ROOT / "data" / "compare_graph_methods")
    run_dir_str = st.text_input(
        "Compare run base directory",
        value=st.session_state.get("gm_run_dir", default_dir),
        key="gm_run_dir_input",
        help="Directory containing per-method subdirectories (cupid/, enap/, enap_custom/).",
    )
    st.session_state["gm_run_dir"] = run_dir_str
    base_dir = pathlib.Path(run_dir_str)

    completed = _find_completed_methods(base_dir)
    if not completed:
        st.info(
            f"No completed graph-building runs found under `{base_dir}`. "
            "Run `scripts/compare_graph_methods.py` first, or check the path."
        )
        return

    st.success(f"Found {len(completed)} completed method(s): {', '.join(completed)}")

    # Method selector
    all_methods = list(completed)
    selected = st.multiselect(
        "Methods to display",
        options=all_methods,
        default=all_methods,
        key="gm_selected_methods",
    )
    if not selected:
        return

    # Load graphs
    graphs: Dict[str, BehaviorGraph] = {}
    for method in selected:
        g = _load_graph(completed[method])
        if g is not None:
            graphs[method] = g

    if not graphs:
        return

    # --- Summary table ---
    st.subheader("Summary")
    rows = []
    for method, g in graphs.items():
        s = _graph_stats(g)
        rows.append({
            "Method": method,
            "Builder": s["builder"],
            "Level": s["level"],
            "Behavior nodes": s["nodes"],
            "Behavioral edges": s["edges"],
            "All edges (incl. terminals)": s["total_edges"],
            "Episodes": s["episodes"],
            "Action priors": "✓" if s["has_action_prior"] else "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # --- Graph visualizations ---
    st.subheader("Transition graphs")

    min_prob = st.slider(
        "Min edge probability to show",
        min_value=0.0, max_value=0.5, value=0.02, step=0.01,
        key="gm_min_prob",
    )
    graph_height = st.select_slider(
        "Graph height (px)",
        options=[400, 500, 600, 700, 800],
        value=600,
        key="gm_graph_height",
    )

    from policy_doctor.plotting.plotly.behavior_graph import create_interactive_behavior_graph

    # Lay out graphs in columns (max 2 per row)
    method_list = list(graphs.items())
    for row_start in range(0, len(method_list), 2):
        row_methods = method_list[row_start: row_start + 2]
        cols = st.columns(len(row_methods))
        for col, (method, graph) in zip(cols, row_methods):
            with col:
                s = _graph_stats(graph)
                st.markdown(
                    f"**{method}** · {s['builder']} · "
                    f"{s['nodes']} nodes · {s['edges']} edges"
                )
                html = create_interactive_behavior_graph(
                    graph,
                    min_probability=min_prob,
                    height=f"{graph_height}px",
                    width="100%",
                    layout_algorithm="layeredStatic",
                )
                components.html(html, height=graph_height + 20, scrolling=False)

    # --- Distribution charts ---
    with st.expander("Edge & degree distributions", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _edge_prob_histogram(graphs, min_prob=0.0),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _out_degree_histogram(graphs),
                use_container_width=True,
            )

        mag_fig = _action_prior_magnitude_chart(graphs)
        if mag_fig:
            st.plotly_chart(mag_fig, use_container_width=True)

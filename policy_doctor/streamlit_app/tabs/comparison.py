"""Comparison tab: baseline vs curated model success rates.

Loads eval_log.json from baseline and curated training runs, displays
success rate comparison across seeds as a table and bar chart.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from policy_doctor.config import VisualizerConfig
from policy_doctor.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT


def render_tab(config: VisualizerConfig, data: Any) -> None:
    """Render the Comparison tab."""
    st.header("Comparison: Baseline vs Curated")

    mode = st.radio(
        "Data source",
        ["Load from eval directories", "Load comparison JSON"],
        horizontal=True, key="cmp_mode",
    )

    if mode == "Load comparison JSON":
        _render_load_json()
    else:
        _render_load_from_dirs()


def _render_load_from_dirs() -> None:
    """Load eval_log.json from baseline and curated eval dirs."""
    st.subheader("Eval Directories")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline**")
        baseline_pattern = st.text_input(
            "Baseline eval dir pattern",
            value="data/outputs/eval_save_episodes/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_{seed}/latest",
            key="cmp_baseline_pattern",
        )
    with col2:
        st.markdown("**Curated**")
        curated_pattern = st.text_input(
            "Curated eval dir pattern",
            value="data/outputs/eval_save_episodes/curated/curated_train_diffusion_unet_lowdim_transport_mh_{seed}/latest",
            key="cmp_curated_pattern",
        )

    seeds = st.text_input("Seeds (comma-separated)", value="0,1,2", key="cmp_seeds")
    seed_list = [s.strip() for s in seeds.split(",")]

    if st.button("Load and compare", type="primary", key="cmp_load"):
        baseline_results = {}
        curated_results = {}

        for seed in seed_list:
            b_dir = pathlib.Path(baseline_pattern.format(seed=seed))
            if not b_dir.is_absolute():
                b_dir = _REPO_ROOT / b_dir
            b_rate = _load_success_rate(b_dir)
            if b_rate is not None:
                baseline_results[seed] = b_rate

            c_dir = pathlib.Path(curated_pattern.format(seed=seed))
            if not c_dir.is_absolute():
                c_dir = _REPO_ROOT / c_dir
            c_rate = _load_success_rate(c_dir)
            if c_rate is not None:
                curated_results[seed] = c_rate

        if not baseline_results and not curated_results:
            st.warning("No eval_log.json found in any of the specified directories.")
            return

        st.session_state["cmp_baseline"] = baseline_results
        st.session_state["cmp_curated"] = curated_results
        st.session_state["cmp_seeds"] = seed_list

    if "cmp_baseline" in st.session_state:
        _render_comparison(
            st.session_state["cmp_baseline"],
            st.session_state["cmp_curated"],
            st.session_state["cmp_seeds"],
        )


def _render_load_json() -> None:
    """Show comparison loaded from sidebar JSON upload."""
    st.info(
        "Upload a comparison JSON in the sidebar under **Comparison data** (expected keys: "
        "`baseline`, `curated` — maps of seed → success rate)."
    )

    if "cmp_baseline" in st.session_state:
        _render_comparison(
            st.session_state["cmp_baseline"],
            st.session_state["cmp_curated"],
            st.session_state["cmp_seeds"],
        )


def _load_success_rate(eval_dir: pathlib.Path) -> Optional[float]:
    """Load success rate from eval_log.json in an eval directory."""
    for log_name in ["eval_log.json", "eval_results.json"]:
        log_path = eval_dir / log_name
        if log_path.exists():
            try:
                with open(log_path) as f:
                    log = json.load(f)
                if isinstance(log, dict):
                    for key in ["success_rate", "mean_score", "rollout_success_rate"]:
                        if key in log:
                            return float(log[key])
                    if "results" in log and isinstance(log["results"], list):
                        successes = sum(1 for r in log["results"] if r.get("success", False))
                        return successes / max(len(log["results"]), 1)
            except Exception:
                continue

    # Try scanning for video_results pattern
    for jf in eval_dir.glob("**/video_results/*.json"):
        try:
            with open(jf) as f:
                log = json.load(f)
            if "success_rate" in log:
                return float(log["success_rate"])
        except Exception:
            continue
    return None


def _render_comparison(
    baseline: Dict[str, float],
    curated: Dict[str, float],
    seeds: List[str],
) -> None:
    """Render the comparison table and charts."""
    import pandas as pd
    import plotly.graph_objects as go

    st.subheader("Results")

    rows = []
    for seed in seeds:
        b = baseline.get(seed)
        c = curated.get(seed)
        delta = (c - b) if (b is not None and c is not None) else None
        rows.append({
            "Seed": seed,
            "Baseline": f"{100*b:.1f}%" if b is not None else "—",
            "Curated": f"{100*c:.1f}%" if c is not None else "—",
            "Delta": f"{100*delta:+.1f}pp" if delta is not None else "—",
        })

    b_vals = [v for v in baseline.values() if v is not None]
    c_vals = [v for v in curated.values() if v is not None]
    b_mean = np.mean(b_vals) if b_vals else None
    c_mean = np.mean(c_vals) if c_vals else None
    delta_mean = (c_mean - b_mean) if (b_mean is not None and c_mean is not None) else None
    rows.append({
        "Seed": "**Mean**",
        "Baseline": f"{100*b_mean:.1f}%" if b_mean is not None else "—",
        "Curated": f"{100*c_mean:.1f}%" if c_mean is not None else "—",
        "Delta": f"{100*delta_mean:+.1f}pp" if delta_mean is not None else "—",
    })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight mean metrics
    col1, col2, col3 = st.columns(3)
    if b_mean is not None:
        col1.metric("Baseline mean", f"{100*b_mean:.1f}%")
    if c_mean is not None:
        col2.metric("Curated mean", f"{100*c_mean:.1f}%")
    if delta_mean is not None:
        col3.metric("Delta", f"{100*delta_mean:+.1f}pp",
                     delta=f"{100*delta_mean:+.1f}pp",
                     delta_color="normal")

    # Bar chart
    fig = go.Figure()
    seed_labels = [s for s in seeds if s in baseline or s in curated]
    if baseline:
        fig.add_trace(go.Bar(
            name="Baseline",
            x=seed_labels,
            y=[100 * baseline.get(s, 0) for s in seed_labels],
            marker_color="steelblue",
        ))
    if curated:
        fig.add_trace(go.Bar(
            name="Curated",
            x=seed_labels,
            y=[100 * curated.get(s, 0) for s in seed_labels],
            marker_color="mediumseagreen",
        ))
    fig.update_layout(
        barmode="group",
        title="Success Rate by Seed",
        xaxis_title="Seed",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 100],
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Export
    with st.expander("Export comparison"):
        export_data = {"baseline": baseline, "curated": curated, "seeds": seeds}
        st.download_button(
            "Download comparison JSON",
            data=json.dumps(export_data, indent=2),
            file_name="comparison.json",
            mime="application/json",
            key="cmp_export_json",
        )

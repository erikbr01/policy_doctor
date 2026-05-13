from __future__ import annotations

from pathlib import Path
from typing import Union

import plotly.graph_objects as go
import streamlit as st
import yaml


def load_study_config(config_path: Union[str, Path]) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_strategies(config_path: Union[str, Path]) -> list[dict]:
    return load_study_config(config_path)["strategies"]


def render_strategy_allocator(
    strategies: list[dict],
    total_budget: int,
    allocation_step: int = 25,
    key_prefix: str = "alloc",
) -> dict[str, int]:
    allocations: dict[str, int] = {}

    for strategy in strategies:
        sid = strategy["id"]
        current = st.session_state.get(f"{key_prefix}_{sid}", 0)
        col_swatch, col_main = st.columns([0.04, 0.96])
        with col_swatch:
            st.markdown(
                f'<div style="background:{strategy["color"]};width:18px;height:18px;'
                f'border-radius:3px;margin-top:8px;"></div>',
                unsafe_allow_html=True,
            )
        with col_main:
            st.markdown(f"**{strategy['name']}**")
            st.caption(strategy["description"].strip())
            val = st.number_input(
                label="Demos",
                min_value=0,
                max_value=total_budget,
                value=current,
                step=allocation_step,
                key=f"{key_prefix}_{sid}",
                label_visibility="collapsed",
            )
        allocations[sid] = int(val)

    total_allocated = sum(allocations.values())
    remaining = total_budget - total_allocated
    progress_frac = min(total_allocated / total_budget, 1.0) if total_budget > 0 else 0.0

    st.progress(progress_frac)

    if total_allocated > total_budget:
        st.warning(f"Over budget by {total_allocated - total_budget} demos.")
    else:
        st.caption(f"{remaining} demos unallocated")

    return allocations


def render_strategy_summary(
    allocations: dict[str, int],
    strategies: list[dict],
    total_budget: int,
) -> None:
    names = [s["name"] for s in strategies]
    counts = [allocations.get(s["id"], 0) for s in strategies]
    colors = [s["color"] for s in strategies]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Data Collection Budget Allocation",
        xaxis_title="Demo count",
        xaxis=dict(range=[0, total_budget]),
        margin=dict(l=0, r=20, t=40, b=20),
        height=40 * len(strategies) + 80,
    )
    st.plotly_chart(fig, use_container_width=True)

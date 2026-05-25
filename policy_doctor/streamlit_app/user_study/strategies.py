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


def _render_example_demos(
    strategy: dict,
    demo_videos_dir: Path | None,
    key_prefix: str,
) -> None:
    """Render the example-demos expander for one strategy row.

    Each strategy may carry an ``example_demos`` block::

        example_demos:
          behavior_label: "Overhead grasp"
          initial_condition_label: "Object at center"
          video_paths:            # up to 3, relative to demo_videos_dir
            - ep001.mp4
            - ep042.mp4
            - ep087.mp4

    If ``video_paths`` is empty or the files don't exist yet, a styled
    placeholder is shown instead so the layout is visible before the
    kendama demos are filmed.
    """
    ex = strategy.get("example_demos")
    if not ex:
        return

    behavior_label = ex.get("behavior_label", "")
    ic_label = ex.get("initial_condition_label", "")
    video_paths: list[str] = ex.get("video_paths") or []
    sid = strategy["id"]
    color = strategy.get("color", "#888")

    tag_html = (
        f'<span style="background:{color}22;border:1px solid {color};'
        f'color:{color};border-radius:4px;padding:1px 8px;font-size:0.8em;'
        f'margin-right:6px;">{behavior_label}</span>'
        if behavior_label else ""
    )
    ic_html = (
        f'<span style="background:#1e293b;border:1px solid #334155;'
        f'color:#94a3b8;border-radius:4px;padding:1px 8px;font-size:0.8em;">'
        f'{ic_label}</span>'
        if ic_label else ""
    )

    label_line = (tag_html + ic_html) or "Example demos"
    with st.expander(f"▶ See example demos", expanded=False):
        if label_line:
            st.markdown(label_line, unsafe_allow_html=True)

        # Resolve paths and check existence
        resolved: list[Path | None] = []
        for vp in video_paths[:3]:
            if demo_videos_dir is not None:
                p = demo_videos_dir / vp
                resolved.append(p if p.exists() else None)
            else:
                resolved.append(None)

        # Pad to 3 slots so the layout is always 3-wide
        while len(resolved) < 3:
            resolved.append(None)

        from policy_doctor.streamlit_app.components.mp4_player import mp4_player

        cols = st.columns(3)
        for i, (col, path) in enumerate(zip(cols, resolved)):
            with col:
                if path is not None:
                    mp4_player(
                        path,
                        key=f"{key_prefix}_{sid}_demo_{i}",
                        max_height_px=160,
                    )
                else:
                    # Placeholder shown until real videos are added
                    st.markdown(
                        '<div style="height:120px;border:2px dashed #334155;'
                        'border-radius:6px;display:flex;align-items:center;'
                        'justify-content:center;color:#475569;font-size:0.8em;">'
                        'video coming soon</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Demo {i + 1}")


def render_strategy_allocator(
    strategies: list[dict],
    total_budget: int,
    allocation_step: int = 25,
    key_prefix: str = "alloc",
    demo_videos_dir: Path | None = None,
) -> dict[str, int]:
    """Render the strategy allocation widget.

    Args:
        demo_videos_dir: Base directory for example demo MP4s.  When set and
            a strategy has an ``example_demos.video_paths`` list, an expander
            shows up to 3 example clips below the strategy description.
            Pass ``None`` (default) to suppress the expander entirely.
    """
    allocations: dict[str, int] = {}

    # Show current remaining budget from last render (reads session state before widgets render)
    current_total = sum(
        int(st.session_state.get(f"{key_prefix}_{s['id']}", 0)) for s in strategies
    )
    remaining_header = total_budget - current_total
    cols_hdr = st.columns([3, 1])
    cols_hdr[0].caption(f"Each click adds/removes **{allocation_step} demos**.")
    if remaining_header > 0:
        cols_hdr[1].caption(f"**{remaining_header}** of {total_budget} remaining")
    elif remaining_header == 0:
        cols_hdr[1].caption(f"✓ fully allocated")
    else:
        cols_hdr[1].caption(f"⚠ over by {-remaining_header}")

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
            _render_example_demos(strategy, demo_videos_dir, key_prefix)
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

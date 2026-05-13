from __future__ import annotations

import pathlib

import plotly.graph_objects as go
import streamlit as st


def slice_indicator(
    slice_start: int,
    slice_end: int,
    total_frames: int,
    key: str = "slice_ind",
) -> go.Figure:
    pct = (slice_end - slice_start) / total_frames * 100 if total_frames > 0 else 0.0
    label = f"slice: frames {slice_start}–{slice_end} ({pct:.0f}%)"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[total_frames],
            y=[""],
            orientation="h",
            marker_color="lightgray",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[slice_end - slice_start],
            y=[""],
            orientation="h",
            base=slice_start,
            marker_color="#e87722",
            text=label,
            textposition="inside",
            insidetextanchor="middle",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        barmode="overlay",
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, total_frames], title=None),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig


def mp4_player(
    video_path: str | pathlib.Path,
    label: str = "",
    slice_start: int | None = None,
    slice_end: int | None = None,
    total_frames: int | None = None,
    key: str = "mp4p",
) -> None:
    if label:
        st.caption(label)

    with open(pathlib.Path(video_path), "rb") as f:
        bytes_data = f.read()
    st.video(bytes_data)

    if slice_start is not None and slice_end is not None and total_frames is not None:
        slice_indicator(slice_start, slice_end, total_frames, key=f"{key}_slice_ind")

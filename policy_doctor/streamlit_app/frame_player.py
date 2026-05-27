"""Video-style frame player widget for Streamlit.

Ported from ``influence_visualizer.render_frames.frame_player`` so the
``policy_doctor`` streamlit app keeps the play/pause + FPS + slider controls used
in the cluster-episode browser. UI helper only — no iv dependencies.
"""

from __future__ import annotations

import time
from typing import Optional

import streamlit as st


def frame_player(
    label: str,
    min_value: int,
    max_value: int,
    key: str,
    default_value: int = 0,
    default_fps: float = 3.0,
    help: Optional[str] = None,
    render_fn=None,
    fragment_scope: bool = False,
) -> int:
    """A video-style frame player with play/pause and a slider.

    Renders a row with a play/pause button, an FPS control, and a frame slider.
    When playing, the slider auto-increments at the specified FPS.

    All frame changes trigger a full page rerun, so downstream elements
    (like influence sections) always update when the frame changes.

    Args:
        label: Label for the slider.
        min_value: Minimum frame index.
        max_value: Maximum frame index.
        key: Unique key for session state (must be unique across the app).
        default_value: Initial frame value.
        default_fps: Default playback frames per second.
        help: Optional help text for the slider.
        render_fn: Optional callable(frame_idx) that renders the frame content.
        fragment_scope: If True, use ``st.rerun(scope="fragment")`` to only rerun
            the enclosing fragment instead of the full page.

    Returns:
        The current frame index (int).
    """
    value_key = f"{key}_value"
    playing_key = f"{key}_playing"
    fps_key = f"{key}_fps"
    slider_key = f"{key}_slider"

    # Initialize session state
    if value_key not in st.session_state:
        st.session_state[value_key] = min(default_value, max_value)
    else:
        # Clamp if bounds changed (e.g. different episode selected)
        st.session_state[value_key] = max(
            min_value, min(st.session_state[value_key], max_value)
        )
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False
    if fps_key not in st.session_state:
        st.session_state[fps_key] = default_fps

    # Advance frame if playing (before rendering)
    if st.session_state[playing_key] and max_value > min_value:
        current = st.session_state[value_key]
        st.session_state[value_key] = current + 1 if current < max_value else min_value

    # Sync slider to current value
    st.session_state[slider_key] = st.session_state[value_key]

    def _toggle_play():
        st.session_state[playing_key] = not st.session_state[playing_key]

    def _on_slider_change():
        if slider_key in st.session_state:
            st.session_state[value_key] = st.session_state[slider_key]

    # Render content FIRST (above the controls)
    if render_fn is not None:
        render_fn(st.session_state[value_key])

    # Layout: play/pause | FPS | slider
    col_btn, col_fps, col_slider = st.columns([0.6, 1.8, 5.6])

    with col_btn:
        is_playing = st.session_state[playing_key]
        btn_label = "⏸" if is_playing else "▶"
        st.button(btn_label, key=f"{key}_btn", on_click=_toggle_play)

    with col_fps:
        st.number_input(
            "FPS",
            min_value=1.0,
            max_value=60.0,
            step=1.0,
            key=fps_key,
            label_visibility="collapsed",
            help="Playback speed (frames per second)",
            format="%.0f",
        )

    with col_slider:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            key=slider_key,
            help=help,
            on_change=_on_slider_change,
            label_visibility="collapsed",
        )

    # Schedule next tick if playing
    if st.session_state[playing_key] and max_value > min_value:
        fps = st.session_state[fps_key]
        time.sleep(1.0 / fps)
        # Only use fragment scope if we're actually in a fragment context
        try:
            if fragment_scope:
                st.rerun(scope="fragment")
            else:
                st.rerun()
        except Exception:
            # Fallback to regular rerun if fragment scope fails
            st.rerun()

    return st.session_state[value_key]

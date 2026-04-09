"""Frame rendering utilities for the influence visualizer.

This module provides Streamlit-specific wrappers for frame and timeline visualization.
Pure plotting functions are in influence_visualizer.plotting.frames.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

from influence_visualizer import plotting

# Re-export color constants from plotting module for backwards compatibility
from influence_visualizer.plotting import (
    LABEL_COLORS,
    create_action_plot,
    create_annotated_frame,
    create_label_timeline,
    get_action_labels,
)
from influence_visualizer.plotting.common import (
    get_label_color as _get_label_color,
)


def render_annotated_frame(
    img: Optional[np.ndarray],
    label: str,
    caption: Optional[str] = None,
    font_size: int = 12,
):
    """Render a frame with annotation overlay.

    Args:
        img: RGB image as numpy array (H, W, 3).
        label: Text to overlay on the image (e.g., "t=5").
        caption: Optional caption to display below the image.
        font_size: Font size for the overlay text.
    """
    if img is None:
        st.warning("Frame not available")
        return

    # Create the annotated frame using the pure plotting function
    pil_img = create_annotated_frame(img, label, font_size=font_size)

    st.image(pil_img, caption=caption, width="stretch")


# Note: get_action_labels is handled by plotting package


@st.cache_data(show_spinner=False)
def render_action_chunk(
    actions: Optional[np.ndarray],
    title: str = "Action Chunk",
    unique_key: Optional[str] = None,
):
    """Render an action chunk as a plot.

    Args:
        actions: Action array to visualize
        title: Title for the plot
        unique_key: Optional unique key to avoid duplicate element IDs. If not provided,
                   will be generated from title (which may cause conflicts).
    """
    if actions is None:
        st.warning("Actions not available")
        return

    # Create the action plot using the pure plotting function
    action_labels = get_action_labels(actions.shape[-1])
    fig = create_action_plot(actions, action_labels=action_labels, title=title)

    # Generate a unique key
    if unique_key is None:
        import hashlib

        key_suffix = hashlib.md5(title.encode()).hexdigest()[:8]
        unique_key = f"action_chunk_{key_suffix}"

    st.plotly_chart(fig, width="stretch", key=unique_key)


def render_label_timeline(
    annotations: List[Dict],
    num_frames: int,
    current_frame: Optional[int] = None,
    unique_key: Optional[str] = None,
):
    """Render a color-coded timeline bar showing labels over time.

    Displays horizontal bars like a linear video editor, where each annotation
    slice is shown as a colored segment. Unlabeled regions are shown in light gray.
    A vertical marker indicates the current frame position.

    Args:
        annotations: List of annotation dicts with 'start', 'end', 'label' keys.
        num_frames: Total number of frames in the episode.
        current_frame: Optional current frame index to show a position marker.
        unique_key: Optional unique key for the plotly chart element.
    """
    if num_frames <= 0:
        return

    # Create the label timeline using the pure plotting function
    fig = create_label_timeline(
        annotations=annotations,
        num_frames=num_frames,
        current_frame=current_frame,
    )

    if unique_key is None:
        import hashlib

        key_suffix = hashlib.md5(f"timeline_{num_frames}".encode()).hexdigest()[:8]
        unique_key = f"label_timeline_{key_suffix}"

    st.plotly_chart(fig, width="stretch", key=unique_key)


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
        fragment_scope: If True, use st.rerun(scope="fragment") to only rerun
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

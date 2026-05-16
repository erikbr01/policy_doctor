"""NASA Task Load Index (TLX) survey block, rendered in the user-study pages.

The six standard NASA-TLX dimensions, each on a 0–100 visual-analog
scale with the canonical low / high anchors. Each dimension gets its
own slider; the helper returns the responses as a dict that the page
can serialize into the submission JSON.

Reference: Hart & Staveland (1988), "Development of NASA-TLX (Task Load
Index): Results of empirical and theoretical research".
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st


# (key, prompt, low_label, high_label)
_NASA_TLX_ITEMS: List[Tuple[str, str, str, str]] = [
    (
        "mental_demand",
        "Mental Demand",
        "Very Low",
        "Very High",
    ),
    (
        "physical_demand",
        "Physical Demand",
        "Very Low",
        "Very High",
    ),
    (
        "temporal_demand",
        "Temporal Demand",
        "Very Low",
        "Very High",
    ),
    (
        "performance",
        "Performance",
        "Perfect",
        "Failure",
    ),
    (
        "effort",
        "Effort",
        "Very Low",
        "Very High",
    ),
    (
        "frustration",
        "Frustration",
        "Very Low",
        "Very High",
    ),
]

_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "mental_demand": "How much mental and perceptual activity was required (e.g. thinking, deciding, looking, searching)? Was the task easy or demanding, simple or complex?",
    "physical_demand": "How much physical activity was required (e.g. clicking, scrolling)? Was the task easy or demanding, slow or brisk?",
    "temporal_demand": "How much time pressure did you feel due to the rate or pace at which the task elements occurred? Was the pace slow and leisurely or rapid and frantic?",
    "performance": "How successful do you think you were in accomplishing the goals of the task set by the experimenter? How satisfied were you with your performance? (Note: lower is better.)",
    "effort": "How hard did you have to work (mentally and physically) to accomplish your level of performance?",
    "frustration": "How insecure, discouraged, irritated, stressed, and annoyed (vs. secure, gratified, content, relaxed, and complacent) did you feel during the task?",
}


def render_nasa_tlx(key_prefix: str) -> Dict[str, int]:
    """Render the NASA-TLX block and return ``{item_key: rating(0..100)}``."""
    st.markdown(
        "Please rate the task load you experienced on each of the six "
        "dimensions below. Each rating is on a 0–100 scale anchored by the "
        "labels shown beside each slider."
    )
    responses: Dict[str, int] = {}
    for item_key, label, low, high in _NASA_TLX_ITEMS:
        st.markdown(f"**{label}**")
        st.caption(_ITEM_DESCRIPTIONS[item_key])
        cl, cs, ch = st.columns([1, 8, 1])
        cl.markdown(f"<div style='text-align:right;color:#888;'>{low}</div>",
                    unsafe_allow_html=True)
        with cs:
            responses[item_key] = st.slider(
                label,
                min_value=0, max_value=100, value=50, step=5,
                key=f"{key_prefix}_{item_key}",
                label_visibility="collapsed",
            )
        ch.markdown(f"<div style='color:#888;'>{high}</div>",
                    unsafe_allow_html=True)
    return responses

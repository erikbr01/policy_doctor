"""Streamlit tab renderers. Each module exposes render_tab(config, data).

Tabs orchestrate only: they call policy_doctor.computations, behaviors, curation,
and plotting; UI state and display live here; no business logic.
"""

from policy_doctor.streamlit_app.tabs import (
    behavior_graph,
    vlm_annotation,
    clustering,
    comparison,
    curation,
)

__all__ = [
    "behavior_graph",
    "vlm_annotation",
    "clustering",
    "comparison",
    "curation",
]

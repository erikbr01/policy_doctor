"""Pyvis-backed plotting: interactive behavior graph (vis.js).

The interactive, draggable behavior graph view is implemented with the **pyvis**
library (Python wrapper for vis.js). It returns HTML strings for embedding
e.g. via st.components.v1.html() in Streamlit.

Requires: pip install pyvis
"""

from policy_doctor.plotting.plotly.behavior_graph import (
    create_interactive_behavior_graph,
    create_value_colored_interactive_graph,
)
from policy_doctor.plotting.plotly.behavior_graph_timesteps import (
    create_timestep_colored_interactive_graph,
)

__all__ = [
    "create_interactive_behavior_graph",
    "create_timestep_colored_interactive_graph",
    "create_value_colored_interactive_graph",
]

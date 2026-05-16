"""Body of the Home page. Rendered by st.navigation in Home.py."""

import streamlit as st

st.set_page_config(page_title="Policy Doctor — Demo", layout="wide")

st.title("Policy Doctor — Demo")
st.caption(
    "Use the sidebar to switch between the user-study pages and the "
    "graph-testing playground."
)

st.markdown(
    """
This bundle contains three sibling pages, picked from the left-hand
**Navigation** sidebar:

- **User study A** — original *Group A* protocol (video-only condition).
  Participants watch rollout videos and allocate a data-collection budget.

- **User study B** — original *Group B* protocol (videos **plus** the
  behavior graph). Participants can click clusters, edges, and paths.

- **Graph demo** — interactive playground for exploring behavior-graph
  and trajectory-tree visualizations. Pick a task, switch between
  clusterings (representation × K × W × S × aggregation), choose a
  visualization, set color / pruning controls, and click any node or edge
  to drill in to its videos and stats.

All three pages share the same underlying clustering data and MP4
artifacts. Switching pages does not reset participant state.
"""
)

st.divider()
st.markdown(
    """
##### About the data

The pre-bundled clusterings live under
`third_party/influence_visualizer/configs/<task>/clustering/`. The
rendered rollout videos live under `/tmp/study_mp4s/<task>/`. Both are
mirrored into the docker image if you're running the containerized
build.

If you're running locally and either is missing, the page will tell you
which file it expected to find.
"""
)

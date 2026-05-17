"""Intro / consent gate shown before the actual user-study pages."""

from __future__ import annotations

import streamlit as st


_INTRO_TEXT = """
We would like to invite you to participate in a research study on robot
learning and data collection. The research investigates whether a visual
graph representation of a robot policy's behavioral modes helps human
data-collection planners make more targeted and effective strategy
decisions. Here, you get the chance to interact with a novel behavior
graph interface and recommend data collection strategies for a real
robot manipulation task, all through a standard computer interface.

The estimated time required is **25–35 minutes** per participant.

You will not receive any compensation for participation. Participation
is of course optional, and you will not have any disadvantages from not
participating.

**Group assignment.** Participants are randomly assigned to one of two
groups. One group sees rollout videos only; the other group additionally
sees a behavior-graph view of the robot's policy. Whether you see the
behavior graph depends on your assigned group — please proceed with the
condition you have been assigned.
"""


def gate_or_render(consent_key: str = "user_study_intro_acknowledged") -> None:
    """If the participant hasn't seen the intro, show it and ``st.stop()``.

    Once the participant clicks Continue, ``consent_key`` is set in
    ``st.session_state`` and the calling page renders normally on rerun.
    The key is shared across Group A and Group B so the intro shows
    exactly once per browser session.
    """
    if st.session_state.get(consent_key):
        return

    st.title(
        "Invitation to a user study on guiding robot data collection via "
        "behavior graphs"
    )
    st.markdown(_INTRO_TEXT)
    if st.button("Continue to the study", type="primary"):
        st.session_state[consent_key] = True
        st.rerun()
    st.stop()

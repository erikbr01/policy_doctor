"""Shared Streamlit appearance helpers (light mode CSS, theme, Plotly overrides)."""

from __future__ import annotations

import streamlit as st

LIGHT_MODE_KEY = "light_mode"

# Streamlit's runtime theme is set in config.toml; we override the surfaces
# that need to look right on white. The palette is deliberately muted (warm
# grays) so input chrome doesn't punch out.
LIGHT_MODE_CSS = """
<style>
  :root {
    --pd-text:    #2b2b2b;
    --pd-muted:   #6b6b6b;
    --pd-surface: #f4f4f5;
    --pd-border:  #d4d4d8;
  }
  /* Main canvas */
  [data-testid="stAppViewContainer"], [data-testid="stHeader"],
  section.main, .stApp, .block-container,
  body { background: #ffffff !important; color: var(--pd-text) !important; }
  /* Sidebar: muted panel so it doesn't disappear into the main canvas. */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"],
  [data-testid="stSidebarHeader"],
  [data-testid="stSidebarUserContent"] {
    background-color: var(--pd-surface) !important;
    color: var(--pd-text) !important;
  }
  [data-testid="stSidebar"] {
    border-right: 1px solid var(--pd-border) !important;
  }
  [data-testid="stSidebarCollapseButton"] button,
  [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
  [data-testid="stSidebarCollapseButton"] span {
    color: var(--pd-text) !important;
  }
  /* Text + headers in the content area */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
  .stApp p, .stApp label, .stApp .stMarkdown,
  [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: var(--pd-text) !important;
  }
  /* Captions sit slightly dimmer */
  .stApp [data-testid="stCaptionContainer"], .stApp small,
  .stApp [class*="caption"] { color: var(--pd-muted) !important; }
  /* Dividers + chip-style buttons need light borders, not dark */
  hr, [data-testid="stDivider"] { border-color: var(--pd-border) !important; }
  /* Sidebar nav: page links are very faint on white by default. */
  [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span,
  [data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] * {
    color: var(--pd-text) !important;
  }
  /* Sidebar widget labels (toggles/checkboxes) keep dark-theme near-white text. */
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
  [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
    color: var(--pd-text) !important;
  }
  /* Toggle track — off state is semi-transparent white on the dark theme. */
  [data-testid="stSidebar"] [data-testid="stCheckbox"] label > div:first-child {
    background-color: #d4d4d8 !important;
  }
  /* Standard checkboxes (consent, etc.) use a span for the box; toggles use div. */
  .stApp [data-testid="stCheckbox"] label > span {
    background-color: #ffffff !important;
    border: 1px solid var(--pd-border) !important;
    box-shadow: none !important;
  }
  .stApp [data-testid="stCheckbox"] label:has(input:checked) > span {
    background-color: #ff4b4b !important;
    border-color: #ff4b4b !important;
  }
  .stApp [data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
  .stApp [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p {
    color: var(--pd-text) !important;
  }

  /* ── Inputs ───────────────────────────────────────────────────
     By default Streamlit ships its dark theme through to widgets
     even after we flip the page bg, so dropdowns/buttons end up
     nearly black on the new white canvas. Force them onto the
     muted surface palette. */
  /* Selectbox (BaseWeb Select) */
  .stApp [data-baseweb="select"] > div {
    background-color: var(--pd-surface) !important;
    border-color: var(--pd-border) !important;
    color: var(--pd-text) !important;
  }
  .stApp [data-baseweb="select"] svg { fill: var(--pd-muted) !important; }
  /* Native popover for selectbox options */
  [data-baseweb="popover"] [role="listbox"],
  [data-baseweb="popover"] [role="option"] {
    background-color: #ffffff !important;
    color: var(--pd-text) !important;
  }
  [data-baseweb="popover"] [role="option"]:hover {
    background-color: var(--pd-surface) !important;
  }
  /* Text + number inputs */
  .stApp input[type="text"], .stApp input[type="number"],
  .stApp textarea {
    background-color: var(--pd-surface) !important;
    border-color: var(--pd-border) !important;
    color: var(--pd-text) !important;
  }
  /* Buttons — secondary uses the muted surface, primary keeps
     Streamlit's red accent so the active "highlight path"
     button is visibly differentiated from the inactive ones.
     stDownloadButton needs the same treatment (it renders as a
     separate testid and the default rule above misses it). */
  .stApp .stButton > button[kind="secondary"],
  .stApp .stButton > button:not([kind]),
  .stApp .stDownloadButton > button,
  .stApp [data-testid="stDownloadButton"] button {
    background-color: var(--pd-surface) !important;
    border: 1px solid var(--pd-border) !important;
    color: var(--pd-text) !important;
  }
  .stApp .stButton > button[kind="secondary"]:hover,
  .stApp .stButton > button:not([kind]):hover,
  .stApp .stDownloadButton > button:hover,
  .stApp [data-testid="stDownloadButton"] button:hover {
    background-color: #ebebed !important;
    border-color: #b4b4b8 !important;
  }
  /* Disabled state: lighter surface, dimmer text. */
  .stApp .stDownloadButton > button:disabled,
  .stApp [data-testid="stDownloadButton"] button:disabled {
    background-color: #fafafa !important;
    color: var(--pd-muted) !important;
    border-color: #e4e4e7 !important;
  }
  .stApp .stButton > button[kind="primary"] {
    background-color: #ff4b4b !important;
    border: 1px solid #ff4b4b !important;
    color: #ffffff !important;
  }
  .stApp .stButton > button[kind="primary"]:hover {
    background-color: #ff2b2b !important;
    border-color: #ff2b2b !important;
  }
  /* Sliders: leave the red track, tone down the thumb halo and the
     min/max labels. */
  .stApp [data-baseweb="slider"] [role="slider"] {
    border-color: var(--pd-border) !important;
  }
  /* Expander frame + body. Streamlit's stExpanderDetails container
     keeps the default dark fill even after our top-level bg flip,
     so we override every surface it draws on. */
  .stApp [data-testid="stExpander"],
  .stApp [data-testid="stExpander"] details,
  .stApp [data-testid="stExpander"] details > summary,
  .stApp [data-testid="stExpanderDetails"],
  .stApp [data-testid="stExpanderContent"] {
    background-color: #ffffff !important;
    color: var(--pd-text) !important;
    border-color: var(--pd-border) !important;
  }
</style>
"""


def is_light_mode() -> bool:
    return bool(st.session_state.get(LIGHT_MODE_KEY, False))


def get_theme() -> str:
    return "light" if is_light_mode() else "dark"


def muted_text_color(theme: str | None = None) -> str:
    return "#6b6b6b" if (theme or get_theme()) == "light" else "#888"


def plotly_layout_overrides(theme: str | None = None) -> dict:
    """Extra ``update_layout`` kwargs so Plotly text is readable on white."""
    if (theme or get_theme()) != "light":
        return {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        }
    text = "#2b2b2b"
    grid = "#e4e4e7"
    return {
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": text},
        "title": {"font": {"color": text}},
        "xaxis": {
            "gridcolor": grid,
            "linecolor": grid,
            "tickfont": {"color": text},
            "title": {"font": {"color": text}},
        },
        "yaxis": {
            "gridcolor": grid,
            "linecolor": grid,
            "tickfont": {"color": text},
            "title": {"font": {"color": text}},
        },
    }


def render_light_mode_toggle(*, sidebar: bool = True) -> bool:
    """Render the light-mode toggle; return whether light mode is active."""
    container = st.sidebar if sidebar else st
    return container.toggle(
        "Light mode",
        value=st.session_state.get(LIGHT_MODE_KEY, False),
        key=LIGHT_MODE_KEY,
    )


def apply_light_mode_if_enabled() -> None:
    """Inject light-mode CSS when the toggle is on."""
    if is_light_mode():
        st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)


def render_appearance_sidebar(*, show_colorblind: bool = True) -> tuple[bool, bool]:
    """Sidebar appearance block: light mode (+ optional colorblind toggle).

    Returns ``(light_mode, colorblind_mode)``.
    """
    st.sidebar.header("Appearance")
    light_mode = render_light_mode_toggle(sidebar=True)
    apply_light_mode_if_enabled()
    colorblind_mode = False
    if show_colorblind:
        colorblind_mode = st.sidebar.toggle(
            "Colorblind mode",
            value=st.session_state.get("colorblind_mode", False),
            key="colorblind_mode",
        )
    return light_mode, colorblind_mode

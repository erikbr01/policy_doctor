"""Shared step-flow machinery for the multi-step user study.

Provides:
  - Progress bar rendering
  - Per-step entry/exit time recording
  - Step advancement helper
  - Countdown timer (JS cosmetic + server-side enforcement)
"""

from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from policy_doctor.streamlit_app.appearance import get_theme

STEP_LABELS = ["Consent", "Introduction", "Rollout Info", "Data Collection", "Survey"]
N_STEPS = len(STEP_LABELS)


# ── Progress bar ──────────────────────────────────────────────────────────────

def _progress_styles(theme: str) -> dict[str, str]:
    if theme == "light":
        return {
            "bar_bg": "#f4f4f5",
            "bar_border": "1px solid #d4d4d8",
            "done": "#2ca02c",
            "current": "#0875c4",
            "current_border": "#0875c4",
            "pending": "#6b6b6b",
            "sep": "#b4b4b8",
            "timer_bg": "#f4f4f5",
            "timer_muted": "#6b6b6b",
        }
    return {
        "bar_bg": "#111827",
        "bar_border": "none",
        "done": "#2ca02c",
        "current": "#4db8ff",
        "current_border": "#4db8ff",
        "pending": "#666",
        "sep": "#444",
        "timer_bg": "#111827",
        "timer_muted": "#888",
    }


def render_progress_bar(
    current_step: int,
    labels: list[str] = STEP_LABELS,
    theme: str | None = None,
) -> None:
    theme = theme or get_theme()
    styles = _progress_styles(theme)
    items = []
    for i, label in enumerate(labels):
        num = f"{i + 1}"
        if i < current_step:
            items.append(
                f'<span style="color:{styles["done"]};font-weight:600;">'
                f'<span style="font-size:0.8em;opacity:0.8;">{num}.</span> ✓ {label}'
                f'</span>'
            )
        elif i == current_step:
            items.append(
                f'<span style="color:{styles["current"]};font-weight:700;'
                f'border-bottom:2px solid {styles["current_border"]};padding-bottom:2px;">'
                f'<span style="font-size:0.8em;opacity:0.8;">{num}.</span> {label}'
                f'</span>'
            )
        else:
            items.append(
                f'<span style="color:{styles["pending"]};font-weight:400;">'
                f'<span style="font-size:0.8em;">{num}.</span> {label}'
                f'</span>'
            )

    sep = f'<span style="color:{styles["sep"]};margin:0 10px;font-size:1.1em;">›</span>'
    bar = (
        f'<div style="background:{styles["bar_bg"]};border:{styles["bar_border"]};'
        'border-radius:8px;padding:12px 20px;'
        'margin-bottom:20px;font-size:0.94em;letter-spacing:0.02em;">'
        + sep.join(items) +
        '</div>'
    )
    st.markdown(bar, unsafe_allow_html=True)


# ── Step time tracking ────────────────────────────────────────────────────────

def _times_key(step_key: str) -> str:
    return step_key + "_times"


def record_step_entry(step: int, step_key: str) -> None:
    """Record the wall-clock time when a step is first entered."""
    tk = _times_key(step_key)
    times: dict = st.session_state.setdefault(tk, {})
    entry: dict = times.setdefault(step, {})
    if "start" not in entry:
        entry["start"] = time.time()


def record_step_exit(step: int, step_key: str) -> None:
    """Record the wall-clock time when a step is exited."""
    tk = _times_key(step_key)
    times: dict = st.session_state.setdefault(tk, {})
    times.setdefault(step, {})["end"] = time.time()


def advance_step(current: int, step_key: str) -> None:
    """Exit current step, move to next, and rerun."""
    record_step_exit(current, step_key)
    st.session_state[step_key] = current + 1
    st.rerun()


def get_step_durations(step_key: str) -> dict[int, float]:
    """Return elapsed seconds per completed step."""
    tk = _times_key(step_key)
    durations: dict[int, float] = {}
    for step, entry in st.session_state.get(tk, {}).items():
        if "start" in entry and "end" in entry:
            durations[int(step)] = round(entry["end"] - entry["start"], 1)
    return durations


# ── Countdown timer ───────────────────────────────────────────────────────────

def render_rollout_timer(
    start_time: float,
    allowed_seconds: int,
    key: str = "rollout_timer",
    theme: str | None = None,
) -> tuple[float, bool]:
    """Render a countdown timer and return ``(remaining_seconds, is_expired)``.

    The JS component shows a live countdown (purely cosmetic — the server-side
    check here is authoritative).  In the last 10 seconds the function forces
    a 1 s sleep + rerun to guarantee auto-advance even without user interaction.
    """
    theme = theme or get_theme()
    styles = _progress_styles(theme)
    elapsed = time.time() - start_time
    remaining = max(0.0, allowed_seconds - elapsed)
    expired = remaining <= 0

    if not expired:
        mins = int(remaining) // 60
        secs = int(remaining) % 60
        remaining_ms = int(remaining * 1000)
        border_col = "#d62728" if remaining < 60 else ("#f5a623" if remaining < 180 else "#2ca02c")
        text_col = border_col

        components.html(
            f"""
            <div style="
                display:inline-flex;align-items:center;gap:10px;
                background:{styles["timer_bg"]};
                border:2px solid {border_col};
                border-radius:8px;
                padding:8px 20px;
            ">
              <span style="font-size:1.3em;">⏱</span>
              <div>
                <div style="font-size:0.72em;color:{styles["timer_muted"]};letter-spacing:0.05em;">
                  TIME REMAINING
                </div>
                <div id="cd_{key}" style="
                    font-size:1.6em;font-weight:700;
                    color:{text_col};font-family:monospace;letter-spacing:0.08em;
                ">{mins:02d}:{secs:02d}</div>
              </div>
            </div>
            <script>
            (function(){{
              var endTime = Date.now() + {remaining_ms};
              var el = document.getElementById('cd_{key}');
              if (!el) return;
              function tick() {{
                var left = Math.max(0, endTime - Date.now());
                var m = Math.floor(left / 60000);
                var s = Math.floor((left % 60000) / 1000);
                el.textContent = (m<10?'0':'')+m+':'+(s<10?'0':'')+s;
                if (left > 0) setTimeout(tick, 500);
                else {{ el.textContent = '00:00'; el.style.color = '#d62728'; }}
              }}
              tick();
            }})();
            </script>
            """,
            height=72,
        )

        if remaining <= 10:
            time.sleep(1)
            st.rerun()
    else:
        st.error(
            "Time is up! Your viewing window has ended. "
            "Click **Proceed to Data Collection** below to continue."
        )

    return remaining, expired

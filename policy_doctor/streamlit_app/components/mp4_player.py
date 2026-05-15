from __future__ import annotations

import base64
import hashlib
import pathlib

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


def slice_indicator(
    slice_start: int,
    slice_end: int,
    total_frames: int,
    key: str = "slice_ind",
) -> go.Figure:
    pct = (slice_end - slice_start) / total_frames * 100 if total_frames > 0 else 0.0
    label = f"slice: frames {slice_start}–{slice_end} ({pct:.0f}%)"

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[total_frames], y=[""], orientation="h",
        marker_color="lightgray", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Bar(x=[slice_end - slice_start], y=[""], orientation="h",
        base=slice_start, marker_color="#e87722", text=label,
        textposition="inside", insidetextanchor="middle",
        showlegend=False, hoverinfo="skip"))
    fig.update_layout(barmode="overlay", height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, total_frames], title=None),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig


def mp4_player(
    video_path: str | pathlib.Path,
    label: str = "",
    slice_start: int | None = None,
    slice_end: int | None = None,
    total_frames: int | None = None,
    key: str = "mp4p",
    max_height_px: int | None = None,
    fps: int = 10,
    slice2_start: int | None = None,
    slice2_end: int | None = None,
    bar1_label: str = "",
    bar2_label: str = "",
) -> None:
    if label:
        st.caption(label)

    path = pathlib.Path(video_path)
    with open(path, "rb") as f:
        raw = f.read()

    # Determine video height
    vid_h = max_height_px if max_height_px else 400

    # Auto-seek and timeline — use components.html so JS runs
    start_sec = slice_start / fps if slice_start is not None else 0.0
    uid = "v" + hashlib.md5((str(video_path) + key).encode()).hexdigest()[:8]
    b64 = base64.b64encode(raw).decode()

    # Build timeline with live playhead
    if slice_start is not None and slice_end is not None and total_frames and total_frames > 0:
        pct_start = max(0, slice_start / total_frames * 100)
        pct_w = max(1, (slice_end - slice_start) / total_frames * 100)
        start_s = f"{slice_start / fps:.1f}s"
        end_s = f"{slice_end / fps:.1f}s"
        total_s = total_frames / fps

        bar2_html = ""
        legend2_html = ""
        if slice2_start is not None and slice2_end is not None and slice2_end > slice2_start:
            pct_start2 = max(0, slice2_start / total_frames * 100)
            pct_w2 = max(1, (slice2_end - slice2_start) / total_frames * 100)
            start_s2 = f"{slice2_start / fps:.1f}s"
            end_s2 = f"{slice2_end / fps:.1f}s"
            bar2_html = (
                f'<div style="position:absolute;left:{pct_start2:.1f}%;width:{pct_w2:.1f}%;'
                f'height:100%;background:#38bdf8;border-radius:5px;opacity:0.85;"></div>'
            )
            lbl2 = f" {bar2_label}" if bar2_label else ""
            legend2_html = (
                f'&nbsp;|&nbsp;<span style="color:#38bdf8;">■</span>'
                f'<span style="color:#aaa;">{lbl2} {start_s2}–{end_s2}</span>'
            )

        lbl1 = f" {bar1_label}" if bar1_label else ""
        timeline = f"""
        <div id="tl_{uid}" style="margin-top:6px;position:relative;height:10px;
             background:#333;border-radius:5px;cursor:pointer;">
          <div style="position:absolute;left:{pct_start:.1f}%;width:{pct_w:.1f}%;height:100%;
                      background:#f5a623;border-radius:5px;opacity:0.85;"></div>
          {bar2_html}
          <div id="ph_{uid}" style="position:absolute;top:-3px;left:0%;width:3px;height:16px;
               background:#fff;border-radius:2px;transition:left 0.1s linear;"></div>
        </div>
        <div style="font-size:10px;color:#888;margin-top:3px;">
          ▶ plays from start &nbsp;|&nbsp;<span style="color:#f5a623;">■</span><span style="color:#aaa;">{lbl1} {start_s}–{end_s}</span>{legend2_html}
        </div>"""
        playhead_js = f"""
  var ph=document.getElementById('ph_{uid}');
  var tl=document.getElementById('tl_{uid}');
  v.addEventListener('timeupdate',function(){{
    var pct=v.currentTime/{total_s:.3f}*100;
    if(ph) ph.style.left=Math.min(100,pct).toFixed(1)+'%';
  }});
  tl.addEventListener('click',function(e){{
    var rect=tl.getBoundingClientRect();
    var pct=(e.clientX-rect.left)/rect.width;
    v.currentTime=pct*{total_s:.3f};
  }});"""
        extra_h = 38
    else:
        timeline = ""
        playhead_js = ""
        extra_h = 4

    html = f"""
<video id="{uid}" controls preload="metadata"
  style="width:100%;max-height:{vid_h}px;border-radius:4px;background:#000;display:block;">
  <source src="data:video/mp4;base64,{b64}" type="video/mp4">
</video>
{timeline}
<script>
(function(){{
  var v=document.getElementById('{uid}');
  if(!v) return;
  function seek(){{ if({start_sec:.2f}>0) v.currentTime={start_sec:.2f}; }}
  v.readyState>=1 ? seek() : v.addEventListener('loadedmetadata',seek,{{once:true}});
  {playhead_js}
}})();
</script>"""

    components.html(html, height=vid_h + extra_h, scrolling=False)

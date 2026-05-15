from __future__ import annotations

import base64
import hashlib
import json
import pathlib
from typing import List, Optional

import numpy as np
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


def cluster_timeline(
    per_frame_labels: np.ndarray,
    cluster_colors: List[str],
    fps: int = 10,
    height: int = 80,
    key: str = "clust_tl",
    cluster_names: Optional[dict] = None,
) -> go.Figure:
    """Multi-colored timeline bar showing cluster assignment across an episode.

    Args:
        per_frame_labels: Integer cluster label per frame, shape (T,). -1 = unlabeled.
        cluster_colors: Color palette indexed by cluster id.
        fps: Frames-per-second, used to format hover labels.
        height: Plotly figure height in pixels.
        key: Streamlit widget key.
        cluster_names: Optional {cluster_id: name} mapping for hover text.

    Returns the Plotly figure (also renders it).
    """
    if len(per_frame_labels) == 0:
        return go.Figure()

    # Run-length encode to get contiguous segments
    labels = np.asarray(per_frame_labels, dtype=np.int64)
    total_frames = len(labels)
    segments: list[tuple[int, int, int]] = []  # (start, end_excl, label)
    i = 0
    while i < total_frames:
        j = i + 1
        while j < total_frames and labels[j] == labels[i]:
            j += 1
        segments.append((i, j, int(labels[i])))
        i = j

    fig = go.Figure()
    n_colors = len(cluster_colors)
    for seg_start, seg_end, cid in segments:
        color = "#555555" if cid < 0 else cluster_colors[cid % n_colors]
        name = (cluster_names or {}).get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled"
        width = seg_end - seg_start
        hover = (
            f"<b>{name}</b><br>"
            f"Frames {seg_start}–{seg_end - 1}<br>"
            f"Time {seg_start / fps:.1f}s–{(seg_end - 1) / fps:.1f}s"
        )
        fig.add_trace(go.Bar(
            x=[width], y=[""],
            base=seg_start,
            orientation="h",
            marker_color=color,
            marker_line_width=0,
            showlegend=False,
            hovertemplate=hover + "<extra></extra>",
            name=name,
        ))

    fig.update_layout(
        barmode="overlay",
        height=height,
        margin=dict(l=0, r=0, t=0, b=20),
        xaxis=dict(range=[0, total_frames], title="frame", tickfont=dict(size=10)),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.9)", font_color="white"),
    )
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
    per_frame_labels: Optional[np.ndarray] = None,
    cluster_colors: Optional[List[str]] = None,
    cluster_names: Optional[dict] = None,
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

    vid_h = max_height_px if max_height_px else 400
    start_sec = slice_start / fps if slice_start is not None else 0.0
    uid = "v" + hashlib.md5((str(video_path) + key).encode()).hexdigest()[:8]
    b64 = base64.b64encode(raw).decode()
    total_s = (total_frames / fps) if (total_frames and fps > 0) else None
    extra_h = 4

    # ── Cluster timeline with live cursor ────────────────────────────────────
    cluster_tl_html = ""
    cluster_tl_js = ""
    if (
        per_frame_labels is not None
        and cluster_colors is not None
        and total_s is not None
        and len(per_frame_labels) > 0
    ):
        labels_arr = np.asarray(per_frame_labels, dtype=np.int64)
        n_total = len(labels_arr)
        # RLE encode
        segs: list[tuple[int, int, int]] = []
        i = 0
        while i < n_total:
            j = i + 1
            while j < n_total and labels_arr[j] == labels_arr[i]:
                j += 1
            segs.append((i, j, int(labels_arr[i])))
            i = j

        n_c = len(cluster_colors)
        segs_html = ""
        segs_data = []
        for s0, s1, cid in segs:
            color = "#444" if cid < 0 else cluster_colors[cid % n_c]
            name = (cluster_names or {}).get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled"
            left_pct = s0 / n_total * 100
            w_pct = (s1 - s0) / n_total * 100
            segs_html += (
                f'<div style="position:absolute;left:{left_pct:.3f}%;width:{w_pct:.3f}%;'
                f'height:100%;background:{color};"></div>'
            )
            segs_data.append({"l": s0 / n_total, "r": s1 / n_total, "n": name})
        segs_json = json.dumps(segs_data)

        cluster_tl_html = f"""
<div style="margin-top:6px;position:relative;">
  <div id="ctlt_{uid}" style="position:absolute;bottom:22px;
       background:rgba(20,20,20,0.92);color:#fff;font-size:11px;padding:3px 8px;
       border-radius:4px;pointer-events:none;white-space:nowrap;display:none;
       transform:translateX(-50%);z-index:999;"></div>
  <div id="ctl_{uid}" style="position:relative;height:14px;border-radius:3px;
       overflow:hidden;cursor:pointer;background:#222;">
    {segs_html}
    <div id="ctlc_{uid}" style="position:absolute;top:0;left:0%;width:2px;height:100%;
         background:rgba(255,255,255,0.85);pointer-events:none;"></div>
  </div>
  <div style="font-size:10px;color:#666;margin-top:4px;">cluster timeline</div>
</div>"""

        cluster_tl_js = f"""
  var ctl=document.getElementById('ctl_{uid}');
  var ctlc=document.getElementById('ctlc_{uid}');
  var ctlt=document.getElementById('ctlt_{uid}');
  var segs={segs_json};
  if(ctl&&ctlc){{
    v.addEventListener('timeupdate',function(){{
      var dur=v.duration||{total_s:.3f};
      ctlc.style.left=(Math.min(1,v.currentTime/dur)*100).toFixed(2)+'%';
    }});
    ctl.addEventListener('click',function(e){{
      var dur=v.duration||{total_s:.3f};
      var r=ctl.getBoundingClientRect();
      v.currentTime=(e.clientX-r.left)/r.width*dur;
    }});
    ctl.addEventListener('mousemove',function(e){{
      if(!ctlt)return;
      var r=ctl.getBoundingClientRect();
      var frac=(e.clientX-r.left)/r.width;
      for(var i=0;i<segs.length;i++){{
        if(frac>=segs[i].l&&frac<segs[i].r){{
          ctlt.textContent=segs[i].n;
          ctlt.style.left=((e.clientX-r.left)/r.width*100).toFixed(1)+'%';
          ctlt.style.display='block';
          return;
        }}
      }}
      ctlt.style.display='none';
    }});
    ctl.addEventListener('mouseleave',function(){{if(ctlt)ctlt.style.display='none';}});
  }}"""
        extra_h += 40

    # ── Behavior slice bar with playhead ─────────────────────────────────────
    if slice_start is not None and slice_end is not None and total_s is not None:
        pct_start = max(0, slice_start / total_frames * 100)
        pct_w = max(1, (slice_end - slice_start) / total_frames * 100)
        start_s = f"{slice_start / fps:.1f}s"
        end_s = f"{slice_end / fps:.1f}s"

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
    var dur=v.duration||{total_s:.3f};
    var pct=v.currentTime/dur*100;
    if(ph) ph.style.left=Math.min(100,pct).toFixed(1)+'%';
  }});
  tl.addEventListener('click',function(e){{
    var dur=v.duration||{total_s:.3f};
    var rect=tl.getBoundingClientRect();
    var pct=(e.clientX-rect.left)/rect.width;
    v.currentTime=pct*dur;
  }});"""
        extra_h += 38
    else:
        timeline = ""
        playhead_js = ""

    html = f"""
<video id="{uid}" controls preload="metadata"
  style="width:100%;max-height:{vid_h}px;border-radius:4px;background:#000;display:block;">
  <source src="data:video/mp4;base64,{b64}" type="video/mp4">
</video>
{cluster_tl_html}
{timeline}
<script>
(function(){{
  var v=document.getElementById('{uid}');
  if(!v) return;
  function seek(){{ if({start_sec:.2f}>0) v.currentTime={start_sec:.2f}; }}
  v.readyState>=1 ? seek() : v.addEventListener('loadedmetadata',seek,{{once:true}});
  {cluster_tl_js}
  {playhead_js}
}})();
</script>"""

    components.html(html, height=vid_h + extra_h, scrolling=False)

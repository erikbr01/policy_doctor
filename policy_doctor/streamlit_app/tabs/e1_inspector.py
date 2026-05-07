"""E1 Cluster Coherence Inspector — Streamlit tab.

Visualises completed E1 evaluation runs:
  • Experiment overview (metrics, per-cluster accuracy)
  • Cluster browser  (example & query storyboards, VLM responses)
  • Episode player   (frame-by-frame with cluster assignment timeline)
  • Prompt inspector (reconstructed VLM prompt + raw response)

Data sources (all on disk, no VLM needed):
  <exp_dir>/metrics.json
  <exp_dir>/sample_plan.json
  <exp_dir>/predictions.jsonl
  <clustering_dir>/metadata.json       (slice → rollout_idx, window bounds)
  <clustering_dir>/cluster_labels.npy  (per-slice cluster id)
  <eval_dir>/episodes/ep*.pkl          (frames + obs + action per timestep)
"""

from __future__ import annotations

import io
import json
import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from policy_doctor.vlm.frames import (
    extract_window_frames,
    list_rollout_episode_pkls,
    resolve_window_indices,
)
from policy_doctor.vlm.storyboard import make_storyboard

# ── Cluster colour palette (10 colours, loops) ───────────────────────────────
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

def _cluster_color(cid: int) -> str:
    return _PALETTE[int(cid) % len(_PALETTE)]


# ── Experiment discovery ──────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

def _scan_experiments(repo_root: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Walk repo experiments/ recursively; keep dirs with all three E1 artefacts."""
    exp_root = repo_root / "experiments"
    results: Dict[str, pathlib.Path] = {}
    if not exp_root.is_dir():
        return results
    for d in sorted(exp_root.rglob("metrics.json")):
        exp_dir = d.parent
        if (exp_dir / "sample_plan.json").exists() and (exp_dir / "predictions.jsonl").exists():
            # relative label for the selector
            label = str(exp_dir.relative_to(repo_root))
            results[label] = exp_dir
    return results


# ── Data loaders (all cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_experiment(exp_dir_str: str):
    exp_dir = pathlib.Path(exp_dir_str)
    metrics = json.load(open(exp_dir / "metrics.json"))
    sample_plan = json.load(open(exp_dir / "sample_plan.json"))
    preds = [json.loads(l) for l in open(exp_dir / "predictions.jsonl")]
    pred_by_query = {p["query_idx"]: p for p in preds}
    return metrics, sample_plan, pred_by_query


@st.cache_data(show_spinner=False)
def _load_clustering_data(clustering_dir_str: str):
    cd = pathlib.Path(clustering_dir_str)
    metadata = json.load(open(cd / "metadata.json"))
    labels = np.load(cd / "cluster_labels.npy")
    return metadata, labels


@st.cache_data(show_spinner=False)
def _load_episode_pkl(eval_dir_str: str, rollout_idx: int):
    pkls = list_rollout_episode_pkls(pathlib.Path(eval_dir_str) / "episodes")
    with open(pkls[rollout_idx], "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def _load_episode_meta(eval_dir_str: str):
    """Return (episode_lengths, episode_successes) from episodes/metadata.yaml."""
    import yaml
    meta_path = pathlib.Path(eval_dir_str) / "episodes" / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    lengths = meta.get("episode_lengths", [])
    successes = meta.get("episode_successes", [None] * len(lengths))
    return lengths, successes


@st.cache_data(show_spinner=False)
def _make_storyboard_cached(
    eval_dir_str: str,
    slice_idx: int,
    metadata_json: str,   # serialised so it's hashable
    max_frames: int,
    composite_size: int,
    frame_seed: int,
) -> bytes:
    metadata = json.loads(metadata_json)
    eval_dir = pathlib.Path(eval_dir_str)
    rng = np.random.default_rng(frame_seed)
    r_idx, w0, w1 = resolve_window_indices(metadata[slice_idx])
    frames = extract_window_frames(eval_dir, r_idx, w0, w1, max_frames=max_frames, rng=rng)
    img = make_storyboard(frames, target_size=(composite_size, composite_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _get_episode_frames(eval_dir_str: str, rollout_idx: int) -> List[bytes]:
    """Return all frames of an episode as PNG bytes (one per timestep)."""
    eval_dir = pathlib.Path(eval_dir_str)
    pkls = list_rollout_episode_pkls(eval_dir / "episodes")
    with open(pkls[rollout_idx], "rb") as f:
        df = pickle.load(f)
    result = []
    for i in range(len(df)):
        row = df.iloc[i]
        if "img" not in row:
            result.append(b"")
            continue
        arr = np.asarray(row["img"])
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.01 else arr.astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        result.append(buf.getvalue())
    return result


def _make_episode_gif(frame_bytes: List[bytes], fps: int = 8) -> bytes:
    frames = [Image.open(io.BytesIO(b)) for b in frame_bytes if b]
    if not frames:
        return b""
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    return buf.getvalue()


# ── Timeline rendering ────────────────────────────────────────────────────────

def _render_cluster_timeline(
    episode_len: int,
    slice_assignments: List[Tuple[int, int, int]],  # (w0, w1, cluster_id)
    current_frame: Optional[int] = None,
) -> None:
    """Render a colour-coded HTML timeline bar for one episode."""
    # Build per-timestep colour (grey = unassigned)
    colours = ["#cccccc"] * episode_len
    labels = [-1] * episode_len
    for w0, w1, cid in slice_assignments:
        for t in range(max(0, w0), min(episode_len, w1)):
            colours[t] = _cluster_color(cid)
            labels[t] = cid

    block_w = max(2, min(8, 800 // episode_len))
    blocks = []
    for t, (col, cid) in enumerate(zip(colours, labels)):
        border = " border: 2px solid black;" if t == current_frame else ""
        tip = f"t={t} cluster={cid}" if cid >= 0 else f"t={t} unassigned"
        blocks.append(
            f'<span title="{tip}" style="display:inline-block;width:{block_w}px;'
            f'height:20px;background:{col};{border}"></span>'
        )

    legend_items = sorted({cid for _, _, cid in slice_assignments})
    legend = " ".join(
        f'<span style="background:{_cluster_color(c)};padding:2px 6px;'
        f'border-radius:3px;color:white;font-size:11px;">C{c}</span>'
        for c in legend_items
    )

    st.markdown(
        f'<div style="line-height:0;margin-bottom:4px">{"".join(blocks)}</div>'
        f'<div style="margin-top:4px">{legend}</div>',
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _episode_slice_assignments(
    rollout_idx: int,
    metadata: List[dict],
    labels: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """All (w0, w1, cluster_id) for slices belonging to this episode."""
    out = []
    for i, (meta, cid) in enumerate(zip(metadata, labels)):
        if int(meta.get("rollout_idx", -1)) == rollout_idx:
            _, w0, w1 = resolve_window_indices(meta)
            out.append((w0, w1, int(cid)))
    return out


def _cluster_at(t: int, assignments: List[Tuple[int, int, int]]) -> Optional[int]:
    for w0, w1, cid in assignments:
        if w0 <= t < w1:
            return cid
    return None


def _ep_len(eval_dir_str: str, rollout_idx: int) -> int:
    pkls = list_rollout_episode_pkls(pathlib.Path(eval_dir_str) / "episodes")
    with open(pkls[rollout_idx], "rb") as f:
        df = pickle.load(f)
    return len(df)


# ── Sub-renderers ─────────────────────────────────────────────────────────────

def _render_overview(metrics: dict, sample_plan: dict) -> None:
    st.subheader("Experiment metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Field | Value |
|---|---|
| Backend | `{sample_plan.get('backend', '?')}` |
| K | {len(sample_plan.get('cluster_ids', []))} |
| n_example | {sample_plan.get('n_example', '?')} |
| n_query | {sample_plan.get('n_query', '?')} |
| n_repetitions | {sample_plan.get('n_repetitions', '?')} |
| random_seed | {sample_plan.get('random_seed', '?')} |
| storyboard_mode | `{sample_plan.get('storyboard_mode', '?')}` |
| composite_size | {sample_plan.get('composite_target_size', '?')} |
| max_frames | {sample_plan.get('max_frames_per_storyboard', '?')} |
| global_ep_disjoint | {sample_plan.get('global_episode_disjoint', '?')} |
""")
    with col2:
        st.markdown(f"""
| Metric | Value |
|---|---|
| Top-1 accuracy | **{metrics.get('top1_accuracy', 0):.3f}** |
| n_valid | {metrics.get('n_valid', '?')} |
| n_unclear | {metrics.get('n_unclear', '?')} |
| chance level | {metrics.get('chance_level', '?'):.3f} |
| p-value | {metrics.get('binomial_test_pvalue', float('nan')):.2e} |
""")

    st.subheader("Per-cluster accuracy")
    per_cluster = metrics.get("per_cluster_accuracy", {})
    per_n = metrics.get("per_cluster_n_query", {})
    if per_cluster:
        import altair as alt
        import pandas as pd
        rows = [
            {"cluster": f"C{k}", "accuracy": v,
             "n": per_n.get(k, per_n.get(str(k), "?"))}
            for k, v in per_cluster.items()
        ]
        df = pd.DataFrame(rows)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("cluster:N", sort=None, title="Cluster"),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy"),
                color=alt.Color("cluster:N", legend=None),
                tooltip=["cluster", "accuracy", "n"],
            )
            .properties(height=250)
        )
        chance = alt.Chart(pd.DataFrame({"y": [metrics.get("chance_level", 0.1)]})).mark_rule(
            color="red", strokeDash=[4, 2]
        ).encode(y="y:Q")
        st.altair_chart(chart + chance, use_container_width=True)


def _render_cluster_browser(
    sample_plan: dict,
    pred_by_query: dict,
    metadata: List[dict],
    labels: np.ndarray,
    eval_dir: str,
    frame_seed: int,
) -> None:
    cluster_ids = sample_plan["cluster_ids"]
    cid = st.selectbox("Cluster", cluster_ids, format_func=lambda c: f"C{c}")
    cdata = sample_plan["clusters"][str(cid)]
    max_frames = sample_plan.get("max_frames_per_storyboard", 4)
    comp_size = sample_plan.get("composite_target_size", 512)
    metadata_json = json.dumps(metadata)

    st.markdown(f"**Examples** (n={len(cdata['example_indices'])})")
    cols = st.columns(min(len(cdata["example_indices"]), 4))
    for col, s_idx in zip(cols, cdata["example_indices"]):
        with col:
            img_bytes = _make_storyboard_cached(
                eval_dir, s_idx, metadata_json, max_frames, comp_size, frame_seed
            )
            st.image(img_bytes, caption=f"slice {s_idx}", use_container_width=True)

    st.markdown("---")
    st.markdown(f"**Queries** (n={len(cdata['query_indices'])})")
    for q_idx, origin in zip(cdata["query_indices"], cdata["query_origins"]):
        pred = pred_by_query.get(q_idx)
        if pred is None:
            continue
        correct = pred["is_correct"]
        majority = pred["majority_predicted_cluster_id"]
        agree = pred["agreement_rate"]
        border_color = "#3cb44b" if correct else "#e6194b"
        label = "✓ correct" if correct else f"✗ → C{majority}"
        origin_tag = f" [{origin}]" if origin != "tier1_global" else ""

        with st.expander(
            f"query_idx={q_idx}{origin_tag}  {label}  agree={agree:.0%}", expanded=False
        ):
            img_col, info_col = st.columns([1, 2])
            with img_col:
                img_bytes = _make_storyboard_cached(
                    eval_dir, q_idx, metadata_json, max_frames, comp_size, frame_seed
                )
                st.image(img_bytes, caption=f"slice {q_idx}", use_container_width=True)
                st.markdown(
                    f'<div style="background:{border_color};color:white;padding:4px 8px;'
                    f'border-radius:4px;text-align:center">{label}</div>',
                    unsafe_allow_html=True,
                )
            with info_col:
                st.markdown(f"**True cluster:** C{pred['true_cluster_id']}  "
                            f"**Majority prediction:** C{majority}  "
                            f"**Agreement:** {agree:.0%}")
                for rep in pred.get("repetitions", []):
                    pred_c = rep.get("predicted_cluster_id")
                    opaque = rep.get("predicted_opaque", "?")
                    is_ok = pred_c == cid
                    icon = "✓" if is_ok else "✗"
                    st.markdown(
                        f"**Rep {rep['rep']}** {icon} predicted C{pred_c} "
                        f"(opaque: *{opaque}*)"
                    )
                    st.caption(rep.get("raw_response", ""))


def _render_episode_player(
    sample_plan: dict,
    pred_by_query: dict,
    metadata: List[dict],
    labels: np.ndarray,
    eval_dir: str,
    frame_seed: int,
) -> None:
    eval_dir_path = pathlib.Path(eval_dir)
    ep_lengths, ep_successes = _load_episode_meta(eval_dir)
    n_episodes = len(ep_lengths)

    def _ep_label(i: int) -> str:
        succ = ep_successes[i] if i < len(ep_successes) else None
        tag = " ✓" if succ is True else (" ✗" if succ is False else "")
        return f"ep {i:03d}{tag}  ({ep_lengths[i]} steps)"

    rollout_idx = st.selectbox(
        "Episode",
        list(range(n_episodes)),
        format_func=_ep_label,
        key="ep_player_rollout_idx",
    )

    assignments = _episode_slice_assignments(rollout_idx, metadata, labels)
    if not assignments:
        st.warning(f"No slices found for episode {rollout_idx}.")
        return

    with st.spinner("Loading episode frames..."):
        frame_bytes = _get_episode_frames(eval_dir, rollout_idx)
    ep_len = len(frame_bytes)

    st.markdown(f"**Episode {rollout_idx}** — {ep_len} timesteps, "
                f"{len(assignments)} slices across "
                f"{len({c for _, _, c in assignments})} clusters")

    frame_t = st.slider("Timestep", 0, ep_len - 1, 0, key=f"ep_slider_{rollout_idx}")

    # Timeline
    st.markdown("**Cluster assignment timeline** (current frame = black border)")
    _render_cluster_timeline(ep_len, assignments, current_frame=frame_t)

    # Current frame
    current_cid = _cluster_at(frame_t, assignments)
    col_frame, col_info = st.columns([1, 1])
    with col_frame:
        if frame_bytes[frame_t]:
            st.image(frame_bytes[frame_t], caption=f"t={frame_t}", use_container_width=True)
        else:
            st.info("No image data at this timestep.")
    with col_info:
        if current_cid is not None:
            st.markdown(
                f'<div style="background:{_cluster_color(current_cid)};color:white;'
                f'padding:12px;border-radius:6px;font-size:18px;text-align:center">'
                f'Cluster {current_cid}</div>',
                unsafe_allow_html=True,
            )
            # Find the active slice
            active_slices = [(w0, w1, c) for w0, w1, c in assignments
                             if w0 <= frame_t < w1]
            if active_slices:
                w0, w1, c = active_slices[0]
                st.markdown(f"Window: **t={w0}…{w1-1}**")
                # Find slice index in metadata
                for s_idx, (meta, lbl) in enumerate(zip(metadata, labels)):
                    if (int(meta.get("rollout_idx", -1)) == rollout_idx
                            and int(meta.get("window_start", meta.get("start", -1))) == w0):
                        max_frames = sample_plan.get("max_frames_per_storyboard", 4)
                        comp_size = sample_plan.get("composite_target_size", 512)
                        metadata_json = json.dumps(metadata)
                        img_bytes = _make_storyboard_cached(
                            eval_dir, s_idx, metadata_json,
                            max_frames, comp_size, frame_seed,
                        )
                        st.image(img_bytes, caption="Slice storyboard", use_container_width=True)
                        break
        else:
            st.info("Timestep not covered by any slice window.")

    # GIF export
    st.markdown("---")
    fps = st.slider("GIF fps", 4, 24, 8, key=f"gif_fps_{rollout_idx}")
    if st.button("Export episode as GIF", key=f"gif_btn_{rollout_idx}"):
        with st.spinner("Encoding GIF..."):
            gif_bytes = _make_episode_gif(frame_bytes, fps=fps)
        st.download_button(
            "Download GIF", gif_bytes,
            file_name=f"episode_{rollout_idx}.gif", mime="image/gif",
        )

    # Cluster segments summary
    st.markdown("---")
    st.subheader("Slice segments for this episode")
    metadata_json = json.dumps(metadata)
    max_frames = sample_plan.get("max_frames_per_storyboard", 4)
    comp_size = sample_plan.get("composite_target_size", 512)

    cols = st.columns(min(len(assignments), 5))
    for col, (w0, w1, c) in zip(cols, sorted(assignments)):
        with col:
            # find slice index
            for s_idx, (meta, lbl) in enumerate(zip(metadata, labels)):
                if (int(meta.get("rollout_idx", -1)) == rollout_idx
                        and int(meta.get("window_start", meta.get("start", -1))) == w0):
                    img_b = _make_storyboard_cached(
                        eval_dir, s_idx, metadata_json, max_frames, comp_size, frame_seed
                    )
                    st.image(img_b, use_container_width=True)
                    st.markdown(
                        f'<div style="background:{_cluster_color(c)};color:white;'
                        f'padding:2px;border-radius:3px;text-align:center;font-size:12px">'
                        f'C{c} t={w0}–{w1-1}</div>',
                        unsafe_allow_html=True,
                    )
                    break


def _render_prompt_inspector(
    sample_plan: dict,
    pred_by_query: dict,
    metadata: List[dict],
    labels: np.ndarray,
    eval_dir: str,
    frame_seed: int,
) -> None:
    cluster_ids = sample_plan["cluster_ids"]
    cid = st.selectbox("Cluster", cluster_ids, format_func=lambda c: f"C{c}",
                       key="prompt_cid")
    cdata = sample_plan["clusters"][str(cid)]

    if not cdata["query_indices"]:
        st.info("No queries for this cluster.")
        return

    def _slice_label(s_idx: int) -> str:
        if s_idx < len(metadata):
            m = metadata[s_idx]
            r = m.get("rollout_idx", "?")
            w0 = m.get("window_start", m.get("start", "?"))
            w1 = m.get("window_end", m.get("end", "?"))
            pred = pred_by_query.get(s_idx)
            ok = "✓" if (pred and pred["is_correct"]) else ("✗" if pred else "")
            return f"slice {s_idx}  ep{r} t={w0}–{w1}  {ok}"
        return f"slice {s_idx}"

    q_idx = st.selectbox(
        "Query slice", cdata["query_indices"],
        format_func=_slice_label, key="prompt_qidx",
    )
    pred = pred_by_query.get(q_idx)
    if pred is None:
        st.warning(f"No prediction found for query_idx={q_idx}")
        return

    rep_n = st.selectbox(
        "Repetition",
        list(range(len(pred.get("repetitions", [])))),
        format_func=lambda r: f"Rep {r}",
        key="prompt_rep",
    )
    rep = pred["repetitions"][rep_n] if pred.get("repetitions") else {}

    st.subheader("VLM configuration")
    st.markdown(f"""
| | |
|---|---|
| Backend | `{sample_plan.get('backend', '?')}` |
| Storyboard mode | `{sample_plan.get('storyboard_mode', '?')}` |
| Composite size | {sample_plan.get('composite_target_size', '?')} |
| Max frames | {sample_plan.get('max_frames_per_storyboard', '?')} |
| n_example | {sample_plan.get('n_example', '?')} |
""")

    st.subheader("Label mapping (this rep)")
    label_map = rep.get("label_map", {})
    mapping_str = " | ".join(f"C{k}→*{v}*" for k, v in sorted(label_map.items(), key=lambda x: int(x[0])))
    st.markdown(mapping_str)

    st.subheader("Prompt (reconstructed)")
    st.caption("The exact prompt is not logged; this reconstruction shows the structure and label mapping sent to the VLM.")

    # Reconstruct approximate prompt
    opaque_labels = list(label_map.values())
    k = len(cluster_ids)
    system_prompt = (
        f"You are classifying robot rollout segments into {k} behavioral groups. "
        f"Each image shows a composite storyboard of frames from a short rollout window. "
        f"Groups are labelled: {', '.join(opaque_labels)}. "
        f"Given the example storyboards for each group and a query storyboard, "
        f"reply with ONLY the group label that best matches the query."
    )
    st.text_area("System prompt (reconstructed)", system_prompt, height=100, disabled=True)

    st.markdown("**Example storyboards shown in prompt:**")
    metadata_json = json.dumps(metadata)
    max_frames = sample_plan.get("max_frames_per_storyboard", 4)
    comp_size = sample_plan.get("composite_target_size", 512)

    for ex_cid in cluster_ids:
        ex_opaque = label_map.get(str(ex_cid), f"C{ex_cid}")
        ex_indices = sample_plan["clusters"][str(ex_cid)]["example_indices"]
        st.markdown(f"**Group *{ex_opaque}*** (cluster {ex_cid})")
        ex_cols = st.columns(min(len(ex_indices), 4))
        for col, s_idx in zip(ex_cols, ex_indices):
            with col:
                img_b = _make_storyboard_cached(
                    eval_dir, s_idx, metadata_json, max_frames, comp_size, frame_seed
                )
                st.image(img_b, use_container_width=True)

    st.markdown("**Query storyboard:**")
    q_img_b = _make_storyboard_cached(
        eval_dir, q_idx, metadata_json, max_frames, comp_size, frame_seed
    )
    col_q, _ = st.columns([1, 2])
    with col_q:
        st.image(q_img_b, use_container_width=True)

    st.subheader("VLM response")
    predicted_opaque = rep.get("predicted_opaque", "?")
    predicted_cid = rep.get("predicted_cluster_id")
    is_correct = predicted_cid == pred["true_cluster_id"]
    st.markdown(
        f'<div style="background:{"#3cb44b" if is_correct else "#e6194b"};'
        f'color:white;padding:8px 12px;border-radius:4px;display:inline-block">'
        f'Predicted: <b>{predicted_opaque}</b> → C{predicted_cid}  '
        f'{"✓" if is_correct else "✗ (true: C" + str(pred["true_cluster_id"]) + ")"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.text_area("Raw response", rep.get("raw_response", ""), height=80, disabled=True)


# ── Main render entry point ───────────────────────────────────────────────────

def render() -> None:
    st.header("E1 Cluster Coherence Inspector")

    experiments = _scan_experiments(_REPO_ROOT)
    if not experiments:
        st.warning(f"No E1 experiments found under {_REPO_ROOT / 'experiments'}.")
        return

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### E1 Inspector")
        exp_label = st.selectbox("Experiment", list(experiments.keys()))
        frame_seed = st.number_input(
            "Frame seed",
            min_value=0, max_value=9999, value=42, step=1,
            help="RNG seed for subsampling frames within each slice storyboard. "
                 "Changing this shows different frames from the same window; "
                 "predictions are unchanged.",
        )

    exp_dir = experiments[exp_label]
    metrics, sample_plan, pred_by_query = _load_experiment(str(exp_dir))

    # Validate paths from sample_plan
    clustering_dir = pathlib.Path(sample_plan["clustering_dir"])
    eval_dir = sample_plan["eval_dir"]

    if not clustering_dir.exists():
        st.error(f"Clustering dir not found: `{clustering_dir}`")
        return
    if not pathlib.Path(eval_dir, "episodes").exists():
        st.error(f"Eval dir / episodes not found: `{eval_dir}/episodes`")
        return

    metadata, labels = _load_clustering_data(str(clustering_dir))

    # ── Sub-tabs ─────────────────────────────────────────────────────────────
    tab_ov, tab_cb, tab_ep, tab_pr = st.tabs([
        "Overview", "Cluster Browser", "Episode Player", "Prompt Inspector"
    ])

    with tab_ov:
        _render_overview(metrics, sample_plan)

    with tab_cb:
        _render_cluster_browser(
            sample_plan, pred_by_query, metadata, labels, eval_dir, int(frame_seed)
        )

    with tab_ep:
        _render_episode_player(
            sample_plan, pred_by_query, metadata, labels, eval_dir, int(frame_seed)
        )

    with tab_pr:
        _render_prompt_inspector(
            sample_plan, pred_by_query, metadata, labels, eval_dir, int(frame_seed)
        )

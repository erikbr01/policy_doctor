"""Compute textual kinematic summaries for graph nodes.

Per the spec (Section 4.1), the ``get_node`` tool exposes a textual
``kinematic_summary`` so the agent can form a working hypothesis about a node
*before* spending visual budget on it.

Two strategies, both pure functions over the :class:`SessionContext`:

* ``raw_states`` (preferred) — pulls EE position / gripper / segment durations
  from the per-rollout ``raw_states/{rollout_id}.npz`` arrays produced by
  ``eval_save_episodes``. Reads only the slices belonging to the requested
  node (via :attr:`SessionContext.cluster_metadata`). Cheap; no torch.

* ``cluster_stats`` (fallback) — when raw_states_dir is unset or no metadata
  is available, returns a structural summary (n_timesteps, n_episodes, mean
  dwell duration, top successors). Always works.

The active strategy is selected by ``ctx.config.get('kinematic_summary_strategy', 'raw_states')``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from policy_doctor.vlm.proposals.agents.context import SessionContext


def kinematic_summary_for_node(ctx: "SessionContext", node_id: int) -> str:
    """Return a one-paragraph textual summary for ``node_id``.

    Tries (in order) raw .npz state files (legacy), the rollout .pkl produced
    by ``eval_save_episodes`` (the actual on-disk format), then the structural
    cluster_stats fallback. Never raises.
    """
    strategy = (ctx.config or {}).get("kinematic_summary_strategy", "raw_states")
    if strategy == "raw_states":
        out = _summary_from_raw_states(ctx, node_id)
        if out is not None:
            return out
        out = _summary_from_episode_pkls(ctx, node_id)
        if out is not None:
            return out
    return _summary_from_cluster_stats(ctx, node_id)


# ---------------------------------------------------------------------------
# raw_states strategy
# ---------------------------------------------------------------------------


def _summary_from_raw_states(ctx: "SessionContext", node_id: int) -> Optional[str]:
    """Return None when prerequisites are missing (caller falls back)."""
    if (
        ctx.cluster_labels is None
        or ctx.cluster_metadata is None
        or ctx.raw_states_dir is None
        or not Path(ctx.raw_states_dir).is_dir()
    ):
        return None

    # Locate slices belonging to this node.
    members: List[Dict[str, Any]] = []
    for i, meta in enumerate(ctx.cluster_metadata):
        if int(ctx.cluster_labels[i]) == int(node_id):
            members.append(meta)
    if not members:
        return None

    # Group by rollout to amortize the npz load.
    by_rollout: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for m in members:
        rid = m.get("rollout_idx", m.get("demo_idx"))
        if rid is None:
            continue
        by_rollout[int(rid)].append(m)

    # Sample at most this many rollouts to keep the summary cheap.
    sample = sorted(by_rollout.keys())[:8]

    ee_positions: List[np.ndarray] = []
    gripper_open_frac: List[float] = []
    durations: List[int] = []

    for ep_idx in sample:
        npz_path = _resolve_raw_states_path(ctx.raw_states_dir, ep_idx)
        if npz_path is None:
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as arr:
                # We try a few common key names from eval_save_episodes output.
                ee = _first_present(arr, ["robot0_eef_pos", "ee_pos", "eef_pos"])
                gripper = _first_present(arr, ["robot0_gripper_qpos", "gripper", "gripper_qpos"])
                # Note: shape may be (T, D) for ee, (T,) or (T, 1) for gripper.
        except Exception:
            continue

        for meta in by_rollout[ep_idx]:
            start, end = _slice_bounds(meta)
            if ee is not None and end >= start and start < len(ee):
                ee_positions.append(np.asarray(ee[start : end + 1]))
            if gripper is not None and end >= start and start < len(gripper):
                g = np.asarray(gripper[start : end + 1]).reshape(len(gripper[start : end + 1]), -1)
                # Heuristic: "open" when first DOF > median.
                col = g[:, 0]
                if col.size:
                    gripper_open_frac.append(float((col > np.median(col)).mean()))
            durations.append(end - start + 1)

    if not durations:
        return None

    parts = [f"Sampled from {len(sample)} rollouts; {len(durations)} segments."]
    parts.append(f"Mean segment duration: {np.mean(durations):.1f} timesteps.")

    if ee_positions:
        all_ee = np.concatenate(ee_positions, axis=0)
        if all_ee.ndim == 2 and all_ee.shape[1] >= 3:
            mean_ee = all_ee.mean(axis=0)
            parts.append(
                f"Mean end-effector position: ({mean_ee[0]:+.3f}, {mean_ee[1]:+.3f}, {mean_ee[2]:+.3f})."
            )

    if gripper_open_frac:
        frac = float(np.mean(gripper_open_frac))
        gripper_state = (
            "predominantly open" if frac > 0.7
            else ("predominantly closed" if frac < 0.3 else "mixed open/closed")
        )
        parts.append(f"Gripper {gripper_state} during segment ({frac:.0%} open frames).")

    return " ".join(parts)


def _first_present(arr_dict, names):
    for n in names:
        if n in arr_dict:
            return arr_dict[n]
    return None


def _slice_bounds(meta: Dict[str, Any]) -> tuple[int, int]:
    start = int(meta.get("window_start", meta.get("timestep", 0)))
    end = meta.get("window_end")
    if end is None:
        end = start + int(meta.get("window_width", 1)) - 1
    else:
        end = int(end) - 1
    return start, end


def _resolve_raw_states_path(dirpath: Path, episode_idx: int) -> Optional[Path]:
    """Try a few common naming schemes for raw-state files."""
    candidates = [
        Path(dirpath) / f"r{episode_idx:04d}.npz",
        Path(dirpath) / f"ep{episode_idx:04d}.npz",
        Path(dirpath) / f"{episode_idx:04d}.npz",
        Path(dirpath) / f"rollout_{episode_idx}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# episode_pkl strategy — reads what eval_save_episodes actually writes.
# ---------------------------------------------------------------------------


def _summary_from_episode_pkls(ctx: "SessionContext", node_id: int) -> Optional[str]:
    """Read EE position / gripper from the rollout pkls (pandas DataFrames).

    eval_save_episodes writes per-step DataFrames with columns that vary by
    task (``robot0_eef_pos``, ``robot0_gripper_qpos``, etc.). We probe a few
    common names and fall through silently on miss.
    """
    if ctx.cluster_labels is None or ctx.cluster_metadata is None:
        return None

    members: List[Dict[str, Any]] = []
    for i, meta in enumerate(ctx.cluster_metadata):
        if int(ctx.cluster_labels[i]) == int(node_id):
            members.append(meta)
    if not members:
        return None

    by_rollout: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for m in members:
        rid = m.get("rollout_idx", m.get("demo_idx"))
        if rid is None:
            continue
        by_rollout[int(rid)].append(m)

    sample = sorted(by_rollout.keys())[:8]

    try:
        import pandas as pd
    except ImportError:
        return None

    ee_positions: List[np.ndarray] = []
    gripper_open_frac: List[float] = []
    durations: List[int] = []

    for ep_idx in sample:
        try:
            entry = ctx.pool.entries[ep_idx]
        except (IndexError, AttributeError):
            continue
        pkl = getattr(entry, "episode_pkl", None)
        if pkl is None or not Path(pkl).exists():
            continue
        try:
            df = pd.read_pickle(str(pkl))
        except Exception:
            continue

        ee_col = _first_present_col(df, ["robot0_eef_pos", "ee_pos", "eef_pos"])
        gr_col = _first_present_col(df, ["robot0_gripper_qpos", "gripper", "gripper_qpos"])

        for meta in by_rollout[ep_idx]:
            start, end = _slice_bounds(meta)
            end = min(end, len(df) - 1)
            if end < start:
                continue
            durations.append(end - start + 1)
            if ee_col is not None:
                arrs = df[ee_col].iloc[start : end + 1].to_list()
                if arrs:
                    a = np.asarray(arrs)
                    if a.ndim == 2 and a.shape[1] >= 3:
                        ee_positions.append(a)
            if gr_col is not None:
                arrs = df[gr_col].iloc[start : end + 1].to_list()
                if arrs:
                    g = np.asarray(arrs).reshape(len(arrs), -1)
                    col = g[:, 0]
                    if col.size:
                        gripper_open_frac.append(float((col > np.median(col)).mean()))

    if not durations:
        return None

    parts = [f"Sampled from {len(sample)} rollouts; {len(durations)} segments."]
    parts.append(f"Mean segment duration: {np.mean(durations):.1f} timesteps.")
    if ee_positions:
        all_ee = np.concatenate(ee_positions, axis=0)
        if all_ee.ndim == 2 and all_ee.shape[1] >= 3:
            mean_ee = all_ee.mean(axis=0)
            parts.append(
                f"Mean end-effector position: ({mean_ee[0]:+.3f}, {mean_ee[1]:+.3f}, {mean_ee[2]:+.3f})."
            )
    if gripper_open_frac:
        frac = float(np.mean(gripper_open_frac))
        gripper_state = (
            "predominantly open" if frac > 0.7
            else ("predominantly closed" if frac < 0.3 else "mixed open/closed")
        )
        parts.append(f"Gripper {gripper_state} during segment ({frac:.0%} open frames).")
    return " ".join(parts)


def _first_present_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


# ---------------------------------------------------------------------------
# cluster_stats fallback
# ---------------------------------------------------------------------------


def _summary_from_cluster_stats(ctx: "SessionContext", node_id: int) -> str:
    """Structural summary when raw state data isn't available.

    Even without physical EE / gripper info, we can still anchor descriptions in
    structural / outcome statistics: V-value, failure likelihood, mean dwell,
    top predecessors and successors with probabilities, and the success rate
    among rollouts that pass through this node. Better than just node degree.
    """
    from policy_doctor.behaviors.behavior_graph import (
        FAILURE_NODE_ID,
        START_NODE_ID,
        SUCCESS_NODE_ID,
    )

    g = ctx.graph
    node = g.nodes.get(node_id)
    if node is None:
        return "(no statistics available for this node)"

    name_for = {
        START_NODE_ID: "START", SUCCESS_NODE_ID: "SUCCESS", FAILURE_NODE_ID: "FAILURE",
    }
    def _nm(n):
        return name_for.get(n, f"c{n}")

    parts = [f"{node.num_episodes} rollouts pass through this cluster ({node.num_timesteps} timesteps total)."]

    if node.num_episodes:
        parts.append(
            f"Mean dwell: {node.num_timesteps / max(node.num_episodes, 1):.1f} timesteps per rollout."
        )

    # V-value and failure likelihood — these are the agent's primary signal.
    try:
        v = ctx.node_values().get(node_id)
        fl = ctx.failure_likelihoods().get(node_id)
        if v is not None:
            parts.append(f"V-value: {v:+.3f} (Bellman value under the graph's transition matrix).")
        if fl is not None and fl > 0:
            parts.append(f"Failure likelihood: {fl:.0%} of rollouts that visit this cluster eventually reach FAILURE.")
    except Exception:
        pass

    # Pool-level success rate among rollouts that pass through this node.
    try:
        n_succ = n_fail = 0
        for entry in ctx.pool.entries:
            if not entry.cluster_path or node_id not in entry.cluster_path:
                continue
            if entry.success is True:
                n_succ += 1
            elif entry.success is False:
                n_fail += 1
        if n_succ + n_fail > 0:
            parts.append(
                f"Among {n_succ + n_fail} pool rollouts that visit this cluster, "
                f"{n_succ} succeed and {n_fail} fail "
                f"({n_succ / (n_succ + n_fail):.0%} task-success rate)."
            )
    except Exception:
        pass

    # Top predecessors — tells the agent what reaches this cluster.
    in_edges = g.get_incoming_transitions(node_id)
    if in_edges:
        top_in = sorted(in_edges, key=lambda x: -x[2])[:3]
        pred_str = ", ".join(f"{_nm(src)} (p={p:.2f})" for src, _, p in top_in)
        parts.append(f"Most-likely predecessors: {pred_str}.")

    # Top successors with advantage hints.
    out_edges = g.get_outgoing_transitions(node_id)
    if out_edges:
        top_out = sorted(out_edges, key=lambda x: -x[2])[:3]
        try:
            values = ctx.node_values()
            v_self = values.get(node_id, 0.0)
            succ_parts = []
            for tgt, _, p in top_out:
                v_tgt = values.get(tgt, 0.0)
                advantage = v_tgt - v_self
                tag = "↑" if advantage > 0.05 else ("↓" if advantage < -0.05 else "≈")
                succ_parts.append(f"{_nm(tgt)} (p={p:.2f}, Δv={advantage:+.2f} {tag})")
            parts.append(f"Most-likely successors: {', '.join(succ_parts)}.")
        except Exception:
            succ_str = ", ".join(f"{_nm(tgt)} (p={p:.2f})" for tgt, _, p in top_out)
            parts.append(f"Most-likely successors: {succ_str}.")

    return " ".join(parts)

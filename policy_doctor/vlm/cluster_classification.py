"""Experiment E1: VLM-based cluster coherence classification.

Tests whether a VLM can classify held-out slices into their correct influence-derived
cluster given example slices.  The claim under test: influence-based clusters recover
behaviorally meaningful structure.

Key design decisions (per experiment spec):
- Storyboard composites: each slice is compressed to one composite image (2×2 grid of
  up to 4 frames) to keep token budget manageable (K × n_example + 1 images per call).
- Pre-committed sample plan: ``sample_plan.json`` is written before any VLM call so the
  sampling is auditable and not cherry-picked.
- Episode disjointness: example and query slices for a cluster are drawn from different
  rollout episodes where possible; violations are logged in the sample plan.
- Label randomisation: the mapping cluster_id → opaque label (e.g. "Group A") is
  re-shuffled for every query to prevent positional bias.
- Each query is run ``n_repetitions`` times; results report majority vote and agreement.
"""

from __future__ import annotations

import functools
import json
import pathlib
import pickle
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.frames import (
    extract_window_frames,
    list_rollout_episode_pkls,
    resolve_window_indices,
)
from policy_doctor.vlm.storyboard import make_storyboard

# ---------------------------------------------------------------------------
# Prompt defaults
# ---------------------------------------------------------------------------

DEFAULT_CLASSIFICATION_SYSTEM = (
    "You are an expert in robot manipulation behavior analysis. "
    "You will be shown groups of short video clips from robot rollouts and asked to "
    "classify a query clip into one of those groups."
)

DEFAULT_CLASSIFICATION_PREAMBLE = (
    "Below are {n_groups} groups of robot behavior clips. "
    "Each image shows a 2×2 storyboard of 4 frames sampled from a short segment "
    "of a robot manipulation rollout (left-to-right, top-to-bottom chronological order). "
    "Study the behavioral pattern of each group carefully."
)

DEFAULT_CLASSIFICATION_QUESTION = (
    "Which group does the query clip most closely match in terms of the robot's "
    "behavioral pattern and object interactions?\n\n"
    "Reply with ONLY the group label (e.g. \"Group A\") on the first line.\n"
    "Optionally add a one-sentence justification on the second line.\n"
    "If the query clearly does not match any group, reply with \"unclear\" on the first line."
)

# ---------------------------------------------------------------------------
# Opaque label utilities
# ---------------------------------------------------------------------------

_ALPHA_LABELS = [f"Group {c}" for c in string.ascii_uppercase]


def _opaque_labels(n: int) -> List[str]:
    """Return n opaque labels like 'Group A', 'Group B', etc."""
    if n > len(_ALPHA_LABELS):
        raise ValueError(f"Too many clusters ({n}); max supported is {len(_ALPHA_LABELS)}")
    return _ALPHA_LABELS[:n]


def build_label_map(
    cluster_ids: Sequence[int],
    rng: np.random.Generator,
) -> Dict[int, str]:
    """Random mapping cluster_id → opaque label for one query."""
    ids = list(cluster_ids)
    labels = _opaque_labels(len(ids))
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return dict(zip(ids, shuffled))


# ---------------------------------------------------------------------------
# Sample plan
# ---------------------------------------------------------------------------

def _indices_for_cluster(
    cluster_labels: np.ndarray,
    cluster_id: int,
) -> np.ndarray:
    return np.where(cluster_labels == cluster_id)[0]


def _rollout_idx_of(meta: dict) -> Optional[int]:
    """Extract rollout_idx from slice metadata; None if absent."""
    v = meta.get("rollout_idx")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _split_by_episode(
    indices: np.ndarray,
    metadata: List[dict],
) -> Dict[int, List[int]]:
    """Group indices by rollout_idx."""
    by_ep: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        ep = _rollout_idx_of(metadata[idx])
        if ep is None:
            ep = -1
        by_ep[ep].append(int(idx))
    return dict(by_ep)


def _stratified_sample(
    pool: np.ndarray,
    n: int,
    distances: Optional[np.ndarray],
    rng: np.random.Generator,
) -> List[int]:
    """Sample *n* indices from *pool* stratified across the distance distribution.

    If *distances* is given (same length as pool), sorts pool by distance and
    picks evenly spaced positions.  Otherwise uses random sampling.
    """
    if len(pool) == 0:
        return []
    n = min(n, len(pool))
    if distances is not None and len(distances) == len(pool):
        order = np.argsort(distances)
        sorted_pool = pool[order]
        if n == 1:
            idxs = [int(sorted_pool[0])]
        else:
            positions = np.linspace(0, len(sorted_pool) - 1, n, dtype=int)
            idxs = [int(sorted_pool[p]) for p in positions]
    else:
        chosen = rng.choice(pool, size=n, replace=False)
        idxs = [int(i) for i in chosen]
    return idxs


def _centroid_distances(
    indices: np.ndarray,
    embeddings_reduced: np.ndarray,
) -> np.ndarray:
    """L2 distance from each sample to the empirical cluster centroid."""
    vecs = embeddings_reduced[indices]
    centroid = vecs.mean(axis=0)
    diffs = vecs - centroid
    return np.linalg.norm(diffs, axis=1)


def _select_one_cluster_legacy(
    cid: int,
    cluster_labels: np.ndarray,
    metadata: List[dict],
    embeddings_reduced: Optional[np.ndarray],
    *,
    n_example: int,
    n_query: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Pre-refactor per-cluster example+query selection.

    Identical logic and rng-consumption order to the original (pre two-pass)
    implementation. Used when ``global_episode_disjoint=False`` so old sample
    plans replay bit-for-bit when re-run with the same seed and inputs.
    """
    all_idx = _indices_for_cluster(cluster_labels, cid)
    if len(all_idx) == 0:
        return {
            "example_indices": [],
            "query_indices": [],
            "disjointness_status": "empty",
        }

    dists: Optional[np.ndarray] = None
    if embeddings_reduced is not None:
        dists = _centroid_distances(all_idx, embeddings_reduced)

    n_ex = min(n_example, len(all_idx))
    if dists is not None:
        sorted_order = np.argsort(dists)
        example_local = sorted_order[:n_ex]
        example_idx = all_idx[example_local].tolist()
    else:
        chosen = rng.choice(all_idx, size=n_ex, replace=False)
        example_idx = [int(i) for i in chosen]

    example_eps = {_rollout_idx_of(metadata[i]) for i in example_idx}

    remaining_mask = np.ones(len(all_idx), dtype=bool)
    ex_set = set(example_idx)
    for i, idx in enumerate(all_idx):
        if int(idx) in ex_set:
            remaining_mask[i] = False
    remaining = all_idx[remaining_mask]
    remaining_dists = dists[remaining_mask] if dists is not None else None

    disjoint_mask = np.array([
        _rollout_idx_of(metadata[int(idx)]) not in example_eps
        for idx in remaining
    ])
    disjoint_pool = remaining[disjoint_mask]
    disjoint_dists = (
        remaining_dists[disjoint_mask] if remaining_dists is not None else None
    )

    n_q = min(n_query, len(remaining))
    if len(disjoint_pool) >= n_q:
        query_idx = _stratified_sample(disjoint_pool, n_q, disjoint_dists, rng)
        disjointness = "full"
    elif len(disjoint_pool) > 0:
        query_idx = _stratified_sample(
            disjoint_pool, len(disjoint_pool), disjoint_dists, rng
        )
        same_ep_pool = remaining[~disjoint_mask]
        same_ep_dists = (
            remaining_dists[~disjoint_mask] if remaining_dists is not None else None
        )
        need = n_q - len(query_idx)
        query_idx += _stratified_sample(same_ep_pool, need, same_ep_dists, rng)
        disjointness = "partial"
    else:
        query_idx = _stratified_sample(remaining, n_q, remaining_dists, rng)
        disjointness = "same_episode" if len(remaining) > 0 else "insufficient"

    return {
        "example_indices": [int(i) for i in example_idx],
        "query_indices": [int(i) for i in query_idx],
        "disjointness_status": disjointness,
        "n_available": int(len(all_idx)),
    }


def build_sample_plan(
    cluster_labels: np.ndarray,
    metadata: List[dict],
    embeddings_reduced: Optional[np.ndarray],
    *,
    n_example: int,
    n_query: int,
    rng: np.random.Generator,
    max_clusters: Optional[int] = None,
    global_episode_disjoint: bool = False,
) -> Dict[str, Any]:
    """Build the pre-committed sampling plan for Experiment E1.

    For each cluster:
      - ``example_indices``: n_example prototype slices (centroid-proximal if
        ``embeddings_reduced`` is available, otherwise random).
      - ``query_indices``: n_query held-out slices (stratified; episode-disjoint
        from examples wherever possible).

    When ``global_episode_disjoint=True``, query selection runs in a second pass
    after all examples are chosen and prefers queries whose rollout episode
    appears in NO cluster's example pool. This blocks the cross-cluster
    episode-cue confound: at K=20/n_example=3, the prompt shows 60 example
    storyboards spread across 20 clusters; even if a query's episode is disjoint
    from its own cluster's examples, it can still appear in another cluster's
    examples and let the VLM cheat by visual episode-matching.

    Returns a JSON-serialisable dict. Always written as ``sample_plan.json``
    before any VLM call so sampling is pre-committed.
    """
    unique_ids = sorted(int(c) for c in set(cluster_labels.tolist()) if c >= 0)
    if max_clusters is not None:
        unique_ids = unique_ids[: int(max_clusters)]

    plan: Dict[str, Any] = {
        "n_example": n_example,
        "n_query": n_query,
        "cluster_ids": unique_ids,
        "clusters": {},
        "global_episode_disjoint": bool(global_episode_disjoint),
    }

    # Legacy path: when the flag is off, run the original per-cluster
    # interleaved selection (examples then queries, cluster by cluster). This
    # preserves bit-reproducibility of pre-refactor sample plans — the rng is
    # consumed in the same order as before.
    if not global_episode_disjoint:
        for cid in unique_ids:
            plan["clusters"][cid] = _select_one_cluster_legacy(
                cid, cluster_labels, metadata, embeddings_reduced,
                n_example=n_example, n_query=n_query, rng=rng,
            )
        return plan

    # Global-disjoint path: two-pass. Select examples for every cluster first
    # so we can compute the global example-episode pool before query selection.
    cluster_state: Dict[int, Dict[str, Any]] = {}
    for cid in unique_ids:
        all_idx = _indices_for_cluster(cluster_labels, cid)
        if len(all_idx) == 0:
            plan["clusters"][cid] = {
                "example_indices": [],
                "query_indices": [],
                "disjointness_status": "empty",
            }
            continue

        dists: Optional[np.ndarray] = None
        if embeddings_reduced is not None:
            dists = _centroid_distances(all_idx, embeddings_reduced)

        n_ex = min(n_example, len(all_idx))
        if dists is not None:
            sorted_order = np.argsort(dists)
            example_local = sorted_order[:n_ex]
            example_idx = all_idx[example_local].tolist()
        else:
            chosen = rng.choice(all_idx, size=n_ex, replace=False)
            example_idx = [int(i) for i in chosen]

        example_eps = {_rollout_idx_of(metadata[i]) for i in example_idx}

        remaining_mask = np.ones(len(all_idx), dtype=bool)
        ex_set = set(example_idx)
        for i, idx in enumerate(all_idx):
            if int(idx) in ex_set:
                remaining_mask[i] = False
        remaining = all_idx[remaining_mask]
        remaining_dists = dists[remaining_mask] if dists is not None else None

        cluster_state[cid] = {
            "all_idx": all_idx,
            "example_idx": example_idx,
            "example_eps": example_eps,
            "remaining": remaining,
            "remaining_dists": remaining_dists,
        }

    global_example_eps: set = set()
    for st in cluster_state.values():
        global_example_eps |= st["example_eps"]

    for cid in unique_ids:
        if cid not in cluster_state:
            continue  # empty cluster, already handled
        st = cluster_state[cid]
        example_idx = st["example_idx"]
        example_eps = st["example_eps"]
        remaining = st["remaining"]
        remaining_dists = st["remaining_dists"]

        # Episode label of each remaining slice.
        rem_eps = np.array([_rollout_idx_of(metadata[int(i)]) for i in remaining])
        local_disjoint_mask = np.array([ep not in example_eps for ep in rem_eps])
        global_disjoint_mask = np.array(
            [ep not in global_example_eps for ep in rem_eps]
        )

        # Tiered pools (each strictly stronger than the next).
        # tier1 = global-disjoint, tier2 = local-disjoint only (in another
        # cluster's examples), tier3 = same as own cluster's examples.
        tier1_mask = global_disjoint_mask
        tier2_mask = local_disjoint_mask & ~global_disjoint_mask
        tier3_mask = ~local_disjoint_mask

        def _sample_from_mask(mask: np.ndarray, n: int) -> List[int]:
            if n <= 0 or not mask.any():
                return []
            pool = remaining[mask]
            pool_d = remaining_dists[mask] if remaining_dists is not None else None
            return _stratified_sample(pool, n, pool_d, rng)

        n_q = min(n_query, len(remaining))
        query_idx: List[int] = []
        query_origins: List[str] = []
        used_tiers: List[str] = []
        for tier_name, tier_mask in (
            ("tier1_global", tier1_mask),
            ("tier2_local", tier2_mask),
            ("tier3_same_episode", tier3_mask),
        ):
            need = n_q - len(query_idx)
            if need <= 0:
                break
            picked = _sample_from_mask(tier_mask, need)
            if picked:
                query_idx.extend(picked)
                query_origins.extend([tier_name] * len(picked))
                used_tiers.append(f"{tier_name}:{len(picked)}")

        # Backward-compatible local-only status.
        if local_disjoint_mask.sum() == 0:
            local_status = "same_episode" if len(remaining) > 0 else "insufficient"
        elif tier3_mask.sum() > 0 and any(t.startswith("tier3") for t in used_tiers):
            # We had to dip into same-episode despite some local-disjoint being available.
            local_status = "partial"
        else:
            local_status = "full"

        picked_tiers = [t.split(":")[0] for t in used_tiers]
        if picked_tiers == ["tier1_global"]:
            global_status = "global_full"
        elif "tier1_global" in picked_tiers and "tier3_same_episode" not in picked_tiers:
            global_status = "global_partial_local"
        elif "tier1_global" in picked_tiers:
            global_status = "global_partial_same_ep"
        elif "tier2_local" in picked_tiers and "tier3_same_episode" not in picked_tiers:
            global_status = "local_only"
        elif "tier3_same_episode" in picked_tiers and "tier2_local" in picked_tiers:
            global_status = "local_partial_same_ep"
        elif "tier3_same_episode" in picked_tiers:
            global_status = "same_episode"
        else:
            global_status = "insufficient"

        plan["clusters"][cid] = {
            "example_indices": [int(i) for i in example_idx],
            "query_indices": [int(i) for i in query_idx],
            "query_origins": query_origins,
            "disjointness_status": local_status,
            "global_disjointness_status": global_status,
            "tiers_used": used_tiers,
            "n_available": int(len(st["all_idx"])),
        }

    return plan


# ---------------------------------------------------------------------------
# Frame loading + storyboard creation
# ---------------------------------------------------------------------------

def _load_slice_images(
    eval_dir: pathlib.Path,
    slice_idx: int,
    metadata: List[dict],
    max_frames: int,
    rng: np.random.Generator,
    *,
    view_window_extension: int = 0,
    storyboard_mode: str = "composite",
    composite_target_size: int = 512,
) -> "List[Image.Image]":
    """Load a slice as a list of images.

    ``storyboard_mode``:
      - ``"composite"`` (default): returns a single-element list ``[storyboard]``
        where the frames are tiled into one composite image of size
        ``(composite_target_size, composite_target_size)``. Per-cell resolution
        scales as ``composite_target_size / sqrt(max_frames)``.
      - ``"frames"``: returns the full list of individual frame images at
        their native resolution (no compositing). Costs more visual tokens
        but preserves per-frame resolution.

    ``view_window_extension`` widens the visual frame window symmetrically by
    that many timesteps on each side, **without changing the cluster window
    itself**.
    """
    from PIL import Image

    meta = metadata[slice_idx]
    r_idx, w0, w1 = resolve_window_indices(meta)
    if view_window_extension and view_window_extension > 0:
        w0 = max(0, w0 - int(view_window_extension))
        w1 = w1 + int(view_window_extension)  # extract_window_frames clips to ep length
    frames = extract_window_frames(
        eval_dir, r_idx, w0, w1, max_frames=max_frames, rng=rng
    )
    if not frames:
        return [Image.new("RGB", (64, 64))]
    if storyboard_mode == "frames":
        return list(frames)
    if storyboard_mode == "composite":
        return [make_storyboard(
            frames,
            target_size=(int(composite_target_size), int(composite_target_size)),
        )]
    raise ValueError(
        f"Unknown storyboard_mode={storyboard_mode!r}; expected 'composite' or 'frames'."
    )


def _load_storyboard(
    eval_dir: pathlib.Path,
    slice_idx: int,
    metadata: List[dict],
    max_frames: int,
    rng: np.random.Generator,
    *,
    view_window_extension: int = 0,
) -> "Image.Image":
    """Backward-compat shim — returns a single composite image.

    Prefer :func:`_load_slice_images` for new code; this preserves the older
    one-image-per-slice contract for callers that haven't been migrated.
    """
    imgs = _load_slice_images(
        eval_dir, slice_idx, metadata, max_frames, rng,
        view_window_extension=view_window_extension,
        storyboard_mode="composite",
    )
    return imgs[0]


# ---------------------------------------------------------------------------
# Optional state/action text for slice prompts
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _load_episode_df(eval_dir_str: str, rollout_idx: int):
    """Cache the pickled episode DataFrame so storyboard + extra-text loads share it."""
    pkls = list_rollout_episode_pkls(pathlib.Path(eval_dir_str) / "episodes")
    if rollout_idx < 0 or rollout_idx >= len(pkls):
        raise IndexError(
            f"rollout_idx={rollout_idx} out of range for {len(pkls)} episode files"
        )
    with open(pkls[rollout_idx], "rb") as f:
        return pickle.load(f)


def _format_vec(arr: np.ndarray, *, max_dim: Optional[int] = None, precision: int = 3) -> str:
    flat = np.asarray(arr).reshape(-1)
    if max_dim is not None and flat.size > max_dim:
        flat = flat[:max_dim]
        suffix = f"...({arr.size - max_dim} more)"
    else:
        suffix = ""
    formatted = ", ".join(f"{v:.{precision}f}" for v in flat.tolist())
    return f"[{formatted}]{suffix}"


def _load_slice_extra_text(
    eval_dir: pathlib.Path,
    slice_idx: int,
    metadata: List[dict],
    *,
    include_action_text: bool,
    include_state_text: bool,
    state_max_dim: Optional[int] = 64,
    action_max_dim: Optional[int] = 32,
) -> Optional[str]:
    """Format a per-timestep state/action text block for one slice.

    Returns ``None`` when both flags are off. The block has the form::

        [obs and action across t=0..T-1 within the cluster window]
        t=0 obs=[...]  action=[...]
        t=1 obs=[...]  action=[...]
        ...

    ``state_max_dim`` and ``action_max_dim`` truncate large per-timestep vectors
    to keep token cost bounded; truncation is annotated inline.
    """
    if not (include_action_text or include_state_text):
        return None
    meta = metadata[slice_idx]
    r_idx, w0, w1 = resolve_window_indices(meta)
    df = _load_episode_df(str(eval_dir), r_idx)
    lo = max(0, w0)
    hi = min(len(df), w1)
    if lo >= hi:
        return None

    lines: List[str] = []
    header_parts = []
    if include_state_text:
        header_parts.append("obs[-1] (current frame)")
    if include_action_text:
        header_parts.append("action[0] (executed)")
    lines.append("Numeric per-timestep " + " + ".join(header_parts) + ":")

    for local_t, t in enumerate(range(lo, hi)):
        row = df.iloc[t]
        parts: List[str] = [f"t={local_t}"]
        if include_state_text and "obs" in row:
            obs = np.asarray(row["obs"])
            obs_vec = obs[-1] if obs.ndim == 2 else obs.reshape(-1)
            parts.append(f"obs={_format_vec(obs_vec, max_dim=state_max_dim)}")
        if include_action_text and "action" in row:
            act = np.asarray(row["action"])
            act_vec = act[0] if act.ndim == 2 else act.reshape(-1)
            parts.append(f"action={_format_vec(act_vec, max_dim=action_max_dim)}")
        lines.append("  " + "  ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_classification_response(
    text: str,
    valid_labels: Sequence[str],
) -> Tuple[str, bool]:
    """Parse raw VLM response into (predicted_label, is_unclear).

    Looks for the first line containing a valid label or 'unclear'.
    Returns ``("unclear", True)`` when no valid label is found.
    """
    raw = (text or "").strip()
    first_line = raw.split("\n")[0].strip().lower()

    if "unclear" in first_line:
        return "unclear", True

    valid_lower = {lbl.lower(): lbl for lbl in valid_labels}
    for lbl_lower, lbl_orig in valid_lower.items():
        if lbl_lower in first_line:
            return lbl_orig, False

    # Fallback: search whole response
    for lbl_lower, lbl_orig in valid_lower.items():
        if lbl_lower in raw.lower():
            return lbl_orig, False

    return "unclear", True


# ---------------------------------------------------------------------------
# Single-query runner with pre-committed label maps
# ---------------------------------------------------------------------------

def run_query_with_label_maps(
    *,
    query_idx: int,
    true_cluster_id: int,
    cluster_ids: List[int],
    example_indices: Dict[int, List[int]],
    metadata: List[dict],
    eval_dir: pathlib.Path,
    backend: VLMBackend,
    n_repetitions: int,
    max_frames: int,
    system_prompt: Optional[str],
    user_preamble_template: str,
    user_prompt_question: str,
    frame_rng: np.random.Generator,
    label_maps: List[Dict[int, str]],
    view_window_extension: int = 0,
    include_action_text: bool = False,
    include_state_text: bool = False,
    storyboard_mode: str = "composite",
    composite_target_size: int = 512,
    query_storyboard_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one query with pre-determined per-rep label maps (for reproducibility).

    ``storyboard_mode`` controls visual rendering for *example* slices:
    ``"composite"`` packs frames into one image per slice; ``"frames"`` sends
    each frame as its own image. ``query_storyboard_mode`` overrides for the
    query side (default falls back to ``storyboard_mode``); the hybrid setup
    is ``storyboard_mode="composite", query_storyboard_mode="frames"``.
    """
    from PIL import Image as PILImage

    q_mode = query_storyboard_mode or storyboard_mode

    query_visual = _load_slice_images(
        eval_dir, query_idx, metadata, max_frames, frame_rng,
        view_window_extension=view_window_extension,
        storyboard_mode=q_mode,
        composite_target_size=composite_target_size,
    )
    query_extra_text = _load_slice_extra_text(
        eval_dir, query_idx, metadata,
        include_action_text=include_action_text,
        include_state_text=include_state_text,
    )

    # Each example slice yields a list of images. In composite mode this is
    # one image per slice; in frames mode it's max_frames per slice.
    example_visuals: Dict[int, List[List[PILImage.Image]]] = {}
    example_extra_text_by_cid: Dict[int, List[Optional[str]]] = {}
    for cid in cluster_ids:
        per_slice: List[List[PILImage.Image]] = []
        for ex_idx in example_indices.get(cid, []):
            per_slice.append(_load_slice_images(
                eval_dir, ex_idx, metadata, max_frames, frame_rng,
                view_window_extension=view_window_extension,
                storyboard_mode=storyboard_mode,
                composite_target_size=composite_target_size,
            ))
        example_visuals[cid] = per_slice
        example_extra_text_by_cid[cid] = [
            _load_slice_extra_text(
                eval_dir, ex_idx, metadata,
                include_action_text=include_action_text,
                include_state_text=include_state_text,
            )
            for ex_idx in example_indices.get(cid, [])
        ]

    preamble = user_preamble_template.format(n_groups=len(cluster_ids))

    def _flatten_with_text(
        per_slice_imgs: List[List[PILImage.Image]],
        per_slice_text: List[Optional[str]],
    ) -> Tuple[List[PILImage.Image], List[Optional[str]]]:
        """Flatten per-slice images, attaching slice-level text to that
        slice's last image (so the text appears between adjacent slices)."""
        flat_imgs: List[PILImage.Image] = []
        flat_text: List[Optional[str]] = []
        for slice_imgs, slice_text in zip(per_slice_imgs, per_slice_text):
            for j, im in enumerate(slice_imgs):
                flat_imgs.append(im)
                flat_text.append(slice_text if j == len(slice_imgs) - 1 else None)
        return flat_imgs, flat_text

    rep_records: List[Dict[str, Any]] = []
    for rep_i, label_map in enumerate(label_maps):
        inv_map = {v: k for k, v in label_map.items()}
        ordered_ids = sorted(label_map, key=lambda c: label_map[c])

        example_sets: List[Tuple[str, List[PILImage.Image]]] = []
        example_extra_texts: List[Optional[List[Optional[str]]]] = []
        for cid in ordered_ids:
            flat_imgs, flat_texts = _flatten_with_text(
                example_visuals[cid], example_extra_text_by_cid[cid]
            )
            example_sets.append((label_map[cid], flat_imgs))
            example_extra_texts.append(
                flat_texts if any(t is not None for t in flat_texts) else None
            )

        valid_labels = [label_map[cid] for cid in ordered_ids] + ["unclear"]

        raw_text = backend.classify_slice(
            query_images=query_visual,
            example_sets=example_sets,
            system_prompt=system_prompt,
            user_preamble=preamble,
            user_prompt=user_prompt_question,
            query_extra_text=query_extra_text,
            example_extra_texts=example_extra_texts,
        )
        pred_opaque, is_unclear = parse_classification_response(raw_text, valid_labels)
        if is_unclear:
            pred_cid = None
        else:
            pred_cid = inv_map.get(pred_opaque)

        rep_records.append({
            "rep": rep_i,
            "label_map": {str(k): v for k, v in label_map.items()},
            "predicted_opaque": pred_opaque,
            "predicted_cluster_id": pred_cid,
            "is_unclear": is_unclear,
            "raw_response": raw_text,
        })

    # Majority vote on cluster_id (None == unclear)
    from collections import Counter

    pred_cids = [r["predicted_cluster_id"] for r in rep_records]
    count = Counter(pred_cids)
    majority_cid, majority_n = count.most_common(1)[0]
    agreement_rate = majority_n / n_repetitions

    is_correct = (majority_cid == true_cluster_id) if majority_cid is not None else False

    return {
        "query_idx": query_idx,
        "true_cluster_id": true_cluster_id,
        "majority_predicted_cluster_id": majority_cid,
        "is_correct": is_correct,
        "is_unclear": (majority_cid is None),
        "agreement_rate": agreement_rate,
        "n_repetitions": n_repetitions,
        "repetitions": rep_records,
    }


# ---------------------------------------------------------------------------
# Label-map generation for full experiment
# ---------------------------------------------------------------------------

def generate_label_maps_for_plan(
    plan: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, List[Dict[int, str]]]:
    """Pre-generate per-rep label maps for every query in the plan.

    Returns ``{cluster_id_str: {query_idx_str: [label_map_rep0, rep1, ...]}}``.
    Stored in sample_plan to keep pre-commitment auditable.
    """
    cluster_ids = plan["cluster_ids"]
    n_repetitions = plan.get("n_repetitions", 3)
    maps: Dict[str, Any] = {}
    for cid in cluster_ids:
        maps[str(cid)] = {}
        for q_idx in plan["clusters"][cid]["query_indices"]:
            per_rep = []
            for _ in range(n_repetitions):
                lm = build_label_map(cluster_ids, rng)
                per_rep.append({str(k): v for k, v in lm.items()})
            maps[str(cid)][str(q_idx)] = per_rep
    return maps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    records: List[Dict[str, Any]],
    cluster_ids: List[int],
) -> Dict[str, Any]:
    """Compute accuracy, per-cluster accuracy, confusion matrix, and statistical tests.

    Args:
        records: List of per-query dicts as returned by ``run_query_with_label_maps``.
        cluster_ids: All cluster IDs in the experiment (determines confusion matrix size).

    Returns:
        Dict with keys: top1_accuracy, top1_accuracy_ci, chance_level,
        binomial_test_pvalue, per_cluster_accuracy, confusion_matrix,
        agreement_rate_mean, unclear_rate, n_total, n_unclear, n_valid.
    """
    try:
        from scipy.stats import binomtest
    except ImportError:
        binomtest = None

    n_clusters = len(cluster_ids)
    cid_to_row = {cid: i for i, cid in enumerate(sorted(cluster_ids))}

    n_total = len(records)
    n_unclear = sum(1 for r in records if r.get("is_unclear"))
    valid = [r for r in records if not r.get("is_unclear")]
    n_valid = len(valid)

    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    per_cluster_correct: Dict[int, int] = defaultdict(int)
    per_cluster_total: Dict[int, int] = defaultdict(int)
    per_cluster_valid: Dict[int, int] = defaultdict(int)
    agreement_rates: List[float] = []

    for r in records:
        agreement_rates.append(float(r.get("agreement_rate", 1.0)))
        true_cid = int(r["true_cluster_id"])
        per_cluster_total[true_cid] += 1
        if r.get("is_unclear"):
            continue
        per_cluster_valid[true_cid] += 1
        pred_cid = r.get("majority_predicted_cluster_id")
        if pred_cid is None:
            continue
        row = cid_to_row.get(true_cid)
        col = cid_to_row.get(int(pred_cid))
        if row is not None and col is not None:
            confusion[row, col] += 1
        if pred_cid == true_cid:
            per_cluster_correct[true_cid] += 1

    top1_correct = sum(1 for r in valid if r.get("is_correct"))
    top1_accuracy = top1_correct / n_valid if n_valid > 0 else 0.0
    chance = 1.0 / n_clusters if n_clusters > 0 else 0.0

    # Wilson score 95% CI
    z = 1.96
    if n_valid > 0:
        p = top1_accuracy
        denom = 1 + z**2 / n_valid
        centre = (p + z**2 / (2 * n_valid)) / denom
        margin = z * ((p * (1 - p) / n_valid + z**2 / (4 * n_valid**2)) ** 0.5) / denom
        ci_lower = max(0.0, centre - margin)
        ci_upper = min(1.0, centre + margin)
    else:
        ci_lower, ci_upper = 0.0, 1.0

    # Binomial test (one-sided: accuracy > chance)
    p_value: Optional[float] = None
    if binomtest is not None and n_valid > 0:
        result = binomtest(top1_correct, n_valid, chance, alternative="greater")
        p_value = float(result.pvalue)

    per_cluster_acc = {
        cid: (per_cluster_correct[cid] / per_cluster_total[cid])
        if per_cluster_total[cid] > 0
        else None
        for cid in cluster_ids
    }
    per_cluster_n_valid = {cid: int(per_cluster_valid[cid]) for cid in cluster_ids}

    return {
        "n_total": n_total,
        "n_unclear": n_unclear,
        "n_valid": n_valid,
        "unclear_rate": n_unclear / n_total if n_total > 0 else 0.0,
        "top1_accuracy": top1_accuracy,
        "top1_accuracy_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "chance_level": chance,
        "binomial_test_pvalue": p_value,
        "per_cluster_accuracy": {str(k): v for k, v in per_cluster_acc.items()},
        "per_cluster_n_query": {str(k): int(per_cluster_total[k]) for k in cluster_ids},
        "per_cluster_n_valid": {str(k): int(per_cluster_n_valid[k]) for k in cluster_ids},
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_cluster_ids": sorted(cluster_ids),
        "agreement_rate_mean": float(np.mean(agreement_rates)) if agreement_rates else None,
        "agreement_rates": agreement_rates,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cluster_coherence_classification(
    *,
    clustering_dir: pathlib.Path,
    eval_dir: pathlib.Path,
    backend: VLMBackend,
    n_example: int = 5,
    n_query: int = 5,
    n_repetitions: int = 3,
    max_frames_per_storyboard: int = 4,
    random_seed: int = 42,
    step_dir: pathlib.Path,
    system_prompt: Optional[str] = None,
    user_preamble_template: Optional[str] = None,
    user_prompt_question: Optional[str] = None,
    max_clusters: Optional[int] = None,
    dry_run: bool = False,
    global_episode_disjoint: bool = False,
    view_window_extension: int = 0,
    include_action_text: bool = False,
    include_state_text: bool = False,
    storyboard_mode: str = "composite",
    composite_target_size: int = 512,
    query_storyboard_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Experiment E1 for one clustering result.

    Workflow:
    1. Load clustering artifacts (labels, metadata, optional embeddings_reduced).
    2. Build and persist ``sample_plan.json`` (pre-commitment).
    3. For each query slice in the plan: run VLM classification n_repetitions times.
    4. Persist ``predictions.jsonl`` (one record per query).
    5. Compute and persist ``metrics.json``.
    6. Return a summary dict.
    """
    from policy_doctor.vlm.annotate import load_clustering_artifacts

    step_dir = pathlib.Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)

    # Load clustering
    cluster_labels, metadata, manifest = load_clustering_artifacts(clustering_dir)
    emb_path = pathlib.Path(clustering_dir) / "embeddings_reduced.npy"
    embeddings_reduced: Optional[np.ndarray] = (
        np.load(emb_path) if emb_path.exists() else None
    )

    if embeddings_reduced is not None:
        print(
            f"  embeddings_reduced loaded: shape={embeddings_reduced.shape} "
            "(centroid-proximal example selection enabled)"
        )
    else:
        print("  embeddings_reduced.npy absent — using random sampling for examples")

    # Resolve prompts
    sys_p = system_prompt or DEFAULT_CLASSIFICATION_SYSTEM
    preamble_t = user_preamble_template or DEFAULT_CLASSIFICATION_PREAMBLE
    question = user_prompt_question or DEFAULT_CLASSIFICATION_QUESTION

    # Build sample plan
    rng = np.random.default_rng(random_seed)
    plan = build_sample_plan(
        cluster_labels,
        metadata,
        embeddings_reduced,
        n_example=n_example,
        n_query=n_query,
        rng=rng,
        max_clusters=max_clusters,
        global_episode_disjoint=global_episode_disjoint,
    )
    plan["n_repetitions"] = n_repetitions
    plan["random_seed"] = random_seed
    plan["clustering_dir"] = str(clustering_dir)
    plan["eval_dir"] = str(eval_dir)
    plan["backend"] = getattr(backend, "name", type(backend).__name__)
    plan["max_frames_per_storyboard"] = int(max_frames_per_storyboard)
    plan["view_window_extension"] = int(view_window_extension)
    plan["include_action_text"] = bool(include_action_text)
    plan["include_state_text"] = bool(include_state_text)
    plan["storyboard_mode"] = str(storyboard_mode)
    plan["composite_target_size"] = int(composite_target_size)
    plan["query_storyboard_mode"] = query_storyboard_mode

    # Pre-generate label maps for all queries
    plan["label_maps"] = generate_label_maps_for_plan(plan, rng)

    plan_path = step_dir / "sample_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, default=str)
    print(f"  Sample plan written: {plan_path}")

    cluster_ids: List[int] = plan["cluster_ids"]
    n_queries_total = sum(
        len(plan["clusters"][cid]["query_indices"]) for cid in cluster_ids
    )

    if dry_run:
        print(
            f"[dry_run] validate_cluster_coherence_vlm: "
            f"K={len(cluster_ids)} clusters, {n_queries_total} queries × {n_repetitions} reps, "
            f"backend={plan['backend']}"
        )
        return {
            "sample_plan_path": str(plan_path),
            "predictions_path": None,
            "metrics_path": None,
            "n_clusters": len(cluster_ids),
            "n_queries_total": n_queries_total,
            "dry_run": True,
        }

    # Classification loop
    frame_rng = np.random.default_rng(random_seed + 1000)
    all_records: List[Dict[str, Any]] = []
    predictions_path = step_dir / "predictions.jsonl"

    with open(predictions_path, "w") as pred_f:
        for cid in cluster_ids:
            ex_idx = plan["clusters"][cid]["example_indices"]
            q_idxs = plan["clusters"][cid]["query_indices"]
            cid_str = str(cid)

            for q_idx in q_idxs:
                q_str = str(q_idx)
                raw_label_maps = plan["label_maps"][cid_str][q_str]
                label_maps_int: List[Dict[int, str]] = [
                    {int(k): v for k, v in lm.items()} for lm in raw_label_maps
                ]

                record = run_query_with_label_maps(
                    query_idx=q_idx,
                    true_cluster_id=cid,
                    cluster_ids=cluster_ids,
                    example_indices={c: plan["clusters"][c]["example_indices"] for c in cluster_ids},
                    metadata=metadata,
                    eval_dir=eval_dir,
                    backend=backend,
                    n_repetitions=n_repetitions,
                    max_frames=max_frames_per_storyboard,
                    system_prompt=sys_p,
                    user_preamble_template=preamble_t,
                    user_prompt_question=question,
                    frame_rng=frame_rng,
                    label_maps=label_maps_int,
                    view_window_extension=view_window_extension,
                    include_action_text=include_action_text,
                    include_state_text=include_state_text,
                    storyboard_mode=storyboard_mode,
                    composite_target_size=composite_target_size,
                    query_storyboard_mode=query_storyboard_mode,
                )
                all_records.append(record)
                pred_f.write(json.dumps(record, default=str) + "\n")
                pred_f.flush()

    print(f"  Predictions written: {predictions_path} ({len(all_records)} records)")

    # Metrics
    metrics = compute_classification_metrics(all_records, cluster_ids)
    metrics["backend"] = plan["backend"]
    metrics["n_clusters"] = len(cluster_ids)
    metrics["n_example_per_cluster"] = n_example
    metrics["n_query_per_cluster"] = n_query
    metrics["n_repetitions"] = n_repetitions
    metrics["random_seed"] = random_seed

    metrics_path = step_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics written: {metrics_path}")
    print(
        f"  Top-1 accuracy: {metrics['top1_accuracy']:.3f} "
        f"(chance={metrics['chance_level']:.3f}, "
        f"p={metrics.get('binomial_test_pvalue', 'n/a')})"
    )

    return {
        "sample_plan_path": str(plan_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "n_clusters": len(cluster_ids),
        "n_queries_total": len(all_records),
        "top1_accuracy": metrics["top1_accuracy"],
        "binomial_test_pvalue": metrics.get("binomial_test_pvalue"),
        "dry_run": False,
    }

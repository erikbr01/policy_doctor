"""Functional slice annotation loop — no Streamlit, no pipeline imports."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.frames import extract_window_frames, resolve_window_indices
from policy_doctor.vlm.prompts import (
    apply_slice_reasoning_to_templates,
    extract_slice_final_label,
    format_slice_prompts,
    merge_prompt_config,
    normalize_slice_reasoning_effort,
    prompt_fingerprint,
)


def load_clustering_artifacts(
    clustering_dir: pathlib.Path,
) -> Tuple[np.ndarray, List[dict], dict]:
    """Load labels, metadata list, manifest from a clustering result directory."""
    from policy_doctor.data.clustering_loader import load_clustering_result_from_path

    return load_clustering_result_from_path(clustering_dir)


def select_slice_indices(
    n_total: int,
    *,
    max_slices: Optional[int],
    random_seed: int,
) -> List[int]:
    if max_slices is None or max_slices >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(random_seed)
    return sorted(rng.choice(n_total, size=max_slices, replace=False).tolist())


def iter_slice_annotation_records(
    *,
    eval_dir: pathlib.Path,
    cluster_labels: np.ndarray,
    metadata: List[dict],
    backend: VLMBackend,
    task_hint: str,
    slice_system: Optional[str],
    slice_user_template: str,
    max_frames_per_slice: Optional[int],
    slice_indices: Sequence[int],
    random_seed: int,
    reasoning_effort: str = "none",
    prompt_version: Optional[str] = None,
    embed_source_eval_dir: bool = True,
    debug_plots_dir: Optional[pathlib.Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield one record dict per slice (for JSON lines).

    When *embed_source_eval_dir* is True, each row includes ``source_eval_dir`` (absolute
    string) so UIs can warn if task config ``eval_dir`` does not match.

    When *debug_plots_dir* is set, writes one PNG per slice showing the exact frame tensors
    passed to the backend and the formatted prompts + model output (see ``vlm.debug_plots``).
    """
    rng = np.random.default_rng(random_seed + 17)
    pver = prompt_version or prompt_fingerprint(
        slice_system, slice_user_template, reasoning_effort
    )
    backend_name = getattr(backend, "name", type(backend).__name__)
    eval_resolved = str(eval_dir.resolve()) if embed_source_eval_dir else None

    for i in slice_indices:
        if i < 0 or i >= len(metadata):
            continue
        meta = metadata[i]
        label_id = int(cluster_labels[i])
        if label_id < 0:
            continue
        r_idx, w0, w1 = resolve_window_indices(meta)
        images = extract_window_frames(
            eval_dir,
            r_idx,
            w0,
            w1,
            max_frames=max_frames_per_slice,
            rng=rng,
        )
        sys_p, user_p = format_slice_prompts(
            slice_system=slice_system,
            slice_user_template=slice_user_template,
            task_hint=task_hint,
            rollout_idx=r_idx,
            window_start=w0,
            window_end=w1,
            cluster_id=label_id,
            num_frames=len(images),
        )
        raw_text = backend.describe_slice(images, system_prompt=sys_p, user_prompt=user_p)
        label = extract_slice_final_label(raw_text, reasoning_effort=reasoning_effort)
        rec: Dict[str, Any] = {
            "slice_index": int(i),
            "rollout_idx": r_idx,
            "window_start": w0,
            "window_end": w1,
            "cluster_id": label_id,
            "label": label,
            "backend": backend_name,
            "prompt_version": pver,
        }
        if reasoning_effort == "high" and label != (raw_text or "").strip():
            rec["raw_model_output"] = raw_text
        if eval_resolved is not None:
            rec["source_eval_dir"] = eval_resolved
        if debug_plots_dir is not None:
            from policy_doctor.vlm.debug_plots import save_slice_annotation_debug_png

            plot_path = (
                debug_plots_dir
                / f"slice_{int(i):05d}_r{r_idx}_c{label_id}_w{w0}-{w1}.png"
            )
            save_slice_annotation_debug_png(
                plot_path,
                images=images,
                system_prompt=sys_p,
                user_prompt=user_p,
                model_label=raw_text,
                meta={
                    "slice_index": int(i),
                    "rollout_idx": r_idx,
                    "window_start": w0,
                    "window_end": w1,
                    "cluster_id": label_id,
                    "backend": backend_name,
                },
            )
            rec["debug_plot_path"] = str(plot_path.resolve())
        yield rec


def write_annotations_jsonl(path: pathlib.Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def run_slice_annotation_for_eval(
    *,
    eval_dir: pathlib.Path,
    clustering_dir: pathlib.Path,
    backend: VLMBackend,
    task_hint: str,
    prompts_file: Optional[str],
    prompts_inline: Optional[Dict[str, Any]],
    repo_root: Optional[pathlib.Path],
    max_slices: Optional[int],
    max_frames_per_slice: Optional[int],
    random_seed: int,
    debug_plots_dir: Optional[pathlib.Path] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Load clustering + episodes, annotate selected slices. Returns (records, prompt_version)."""
    labels, meta_list, _manifest = load_clustering_artifacts(clustering_dir)
    if len(labels) != len(meta_list):
        raise ValueError(
            f"cluster_labels length {len(labels)} != metadata length {len(meta_list)}"
        )

    sys_t, user_t = merge_prompt_config(
        prompts_file=prompts_file,
        prompts_inline=prompts_inline,
        repo_root=repo_root,
    )
    effort = "none"
    if prompts_inline:
        raw_eff = prompts_inline.get("reasoning_effort")
        if raw_eff is not None:
            effort = normalize_slice_reasoning_effort(raw_eff)
    sys_t, user_t = apply_slice_reasoning_to_templates(sys_t, user_t, effort)
    pver = prompt_fingerprint(sys_t, user_t, effort)
    idxs = select_slice_indices(len(meta_list), max_slices=max_slices, random_seed=random_seed)
    records = list(
        iter_slice_annotation_records(
            eval_dir=eval_dir,
            cluster_labels=labels,
            metadata=meta_list,
            backend=backend,
            task_hint=task_hint,
            slice_system=sys_t,
            slice_user_template=user_t,
            max_frames_per_slice=max_frames_per_slice,
            slice_indices=idxs,
            random_seed=random_seed,
            reasoning_effort=effort,
            prompt_version=pver,
            debug_plots_dir=debug_plots_dir,
        )
    )
    return records, pver


def resolve_source_eval_dir_for_jsonl(
    jsonl_path: pathlib.Path,
    records: Sequence[Dict[str, Any]],
) -> Optional[str]:
    """Best-effort rollout directory used when annotations were produced.

    Prefer ``source_eval_dir`` on rows (newer pipeline). Else read sibling
    ``result.json`` and match ``annotations_seed{N}.jsonl`` to ``per_seed[N]``.
    """
    for r in records:
        v = r.get("source_eval_dir")
        if v:
            return str(v)
    m = re.search(r"annotations_seed(\d+)\.jsonl$", jsonl_path.name, re.I)
    seed_key = m.group(1) if m else None
    result_path = jsonl_path.parent / "result.json"
    if not result_path.is_file() or not seed_key:
        return None
    try:
        with open(result_path) as f:
            blob = json.load(f)
        per = blob.get("per_seed") or {}
        info = per.get(seed_key) or per.get(str(int(seed_key)))
        if isinstance(info, dict):
            ed = info.get("eval_dir")
            if ed:
                return str(ed)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def eval_dirs_equivalent(a: pathlib.Path, b: pathlib.Path) -> bool:
    """True if paths resolve to the same location (for mismatch checks)."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False

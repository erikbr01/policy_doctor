"""VLM-based coherency judging for per-slice captions within each cluster."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.behavior_summarize import group_slice_labels_by_cluster
from policy_doctor.vlm.prompts import (
    coherency_prompt_fingerprint,
    format_coherency_prompts,
    merge_coherency_prompt_config,
)


def parse_coherency_json(text: str) -> Dict[str, Any]:
    """Parse model output into ``coherent`` / ``score`` / ``rationale`` (best-effort)."""
    raw = (text or "").strip()
    if not raw:
        return {
            "coherent": None,
            "score": None,
            "rationale": "empty model output",
            "parse_error": "empty",
        }
    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {
                "coherent": obj.get("coherent"),
                "score": obj.get("score"),
                "rationale": obj.get("rationale"),
                "parse_error": None,
            }
    except json.JSONDecodeError:
        pass
    # Try first {...} substring
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {
                    "coherent": obj.get("coherent"),
                    "score": obj.get("score"),
                    "rationale": obj.get("rationale"),
                    "parse_error": None,
                }
        except json.JSONDecodeError:
            pass
    return {
        "coherent": None,
        "score": None,
        "rationale": raw[:500],
        "parse_error": "json_decode_failed",
    }


def run_cluster_coherency_eval(
    slice_records: Sequence[Dict[str, Any]],
    *,
    backend: VLMBackend,
    task_hint: str,
    prompts_file: Optional[str],
    prompts_inline: Optional[Dict[str, Any]],
    repo_root: Optional[pathlib.Path],
    max_slice_labels_per_cluster: Optional[int],
    max_clusters: Optional[int],
) -> Tuple[List[Dict[str, Any]], str]:
    """One coherency judgment per cluster from slice-level annotation records."""
    sys_t, user_t = merge_coherency_prompt_config(
        prompts_file=prompts_file,
        prompts_inline=prompts_inline,
        repo_root=repo_root,
    )
    pver = coherency_prompt_fingerprint(sys_t, user_t)
    grouped = group_slice_labels_by_cluster(
        slice_records,
        max_labels_per_cluster=max_slice_labels_per_cluster,
    )
    cluster_ids = sorted(grouped.keys())
    if max_clusters is not None:
        cluster_ids = cluster_ids[: int(max_clusters)]

    rows: List[Dict[str, Any]] = []
    for cid in cluster_ids:
        labels = grouped[cid]
        if not labels:
            continue
        sys_p, user_p = format_coherency_prompts(
            coherency_system=sys_t,
            coherency_user_template=user_t,
            task_hint=task_hint,
            cluster_id=cid,
            slice_labels=labels,
        )
        raw = backend.evaluate_slice_caption_coherency(
            cluster_id=cid,
            slice_labels=labels,
            task_hint=task_hint,
            system_prompt=sys_p,
            user_prompt=user_p,
        )
        parsed = parse_coherency_json(raw)
        rows.append(
            {
                "cluster_id": cid,
                "num_slice_labels": len(labels),
                "raw_response": raw,
                "judgment": parsed,
                "prompt_version": pver,
                "backend": getattr(backend, "name", type(backend).__name__),
            }
        )
    return rows, pver

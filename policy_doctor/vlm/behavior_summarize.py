"""Behavior-level summaries from per-slice VLM captions (grouped by cluster id)."""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.prompts import (
    behavior_prompt_fingerprint,
    format_behavior_prompts,
    merge_behavior_prompt_config,
)


def load_slice_annotations_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_slice_labels_by_cluster(
    records: Sequence[Dict[str, Any]],
    *,
    max_labels_per_cluster: Optional[int],
) -> Dict[int, List[str]]:
    """Map cluster_id -> list of slice label strings (stable cluster id order)."""
    by_c: Dict[int, List[str]] = defaultdict(list)
    for r in records:
        if "cluster_id" not in r or "label" not in r:
            continue
        cid = int(r["cluster_id"])
        by_c[cid].append(str(r["label"]))
    out: Dict[int, List[str]] = {}
    for cid in sorted(by_c.keys()):
        labels = by_c[cid]
        if max_labels_per_cluster is not None and len(labels) > max_labels_per_cluster:
            labels = labels[: int(max_labels_per_cluster)]
        out[cid] = labels
    return out


def run_behavior_summarization(
    records: Sequence[Dict[str, Any]],
    *,
    backend: VLMBackend,
    task_hint: str,
    prompts_file: Optional[str],
    prompts_inline: Optional[Dict[str, Any]],
    repo_root: Optional[pathlib.Path],
    max_slice_labels_per_cluster: Optional[int],
    max_clusters: Optional[int],
) -> Tuple[List[Dict[str, Any]], str]:
    """Return (per-cluster summary records, behavior_prompt_version)."""
    sys_t, user_t = merge_behavior_prompt_config(
        prompts_file=prompts_file,
        prompts_inline=prompts_inline,
        repo_root=repo_root,
    )
    pver = behavior_prompt_fingerprint(sys_t, user_t)
    grouped = group_slice_labels_by_cluster(
        records,
        max_labels_per_cluster=max_slice_labels_per_cluster,
    )
    cluster_ids = sorted(grouped.keys())
    if max_clusters is not None:
        cluster_ids = cluster_ids[: int(max_clusters)]

    summaries: List[Dict[str, Any]] = []
    for cid in cluster_ids:
        labels = grouped[cid]
        if not labels:
            continue
        sys_p, user_p = format_behavior_prompts(
            behavior_system=sys_t,
            behavior_user_template=user_t,
            task_hint=task_hint,
            cluster_id=cid,
            slice_labels=labels,
        )
        text = backend.summarize_behavior_labels(
            cluster_id=cid,
            slice_labels=labels,
            task_hint=task_hint,
            system_prompt=sys_p,
            user_prompt=user_p,
        )
        summaries.append(
            {
                "cluster_id": cid,
                "summary": text,
                "num_slice_labels": len(labels),
                "prompt_version": pver,
                "backend": getattr(backend, "name", type(backend).__name__),
            }
        )
    return summaries, pver


def write_behavior_summaries_json(path: pathlib.Path, summaries: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(summaries), f, indent=2, default=str)

"""Cross-session aggregation for the agentic loop.

Reuses the same Jaccard-similarity machinery the one-shot
:mod:`policy_doctor.vlm.proposals.propose` aggregator uses, but adapts it to
operate over :class:`SessionResult` lists. The output mirrors the legacy
``selected_run.json`` / ``consistency_metrics.json`` artefacts so downstream
adherence scoring and pipeline steps don't need to know which mode produced
the requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from policy_doctor.vlm.proposals.agents.session import SessionResult
from policy_doctor.vlm.proposals.propose import (
    _RepetitionResult,
    _consistency_score,
    _union_aggregate,
)
from policy_doctor.vlm.proposals.request import DemonstrationRequest


@dataclass
class AgentAggregationResult:
    """Mirrors :class:`policy_doctor.vlm.proposals.propose.GenerationResult`."""

    selected_seed: int
    selected_requests: List[DemonstrationRequest] = field(default_factory=list)
    consistency_metrics: Dict[str, Any] = field(default_factory=dict)
    union_requests: List[DemonstrationRequest] = field(default_factory=list)


def aggregate_agent_sessions(
    sessions: Sequence[SessionResult],
    *,
    method: str = "best_consistency_run",
    similarity_threshold: float = 0.5,
) -> AgentAggregationResult:
    """Aggregate the strategies submitted by N agent sessions of one condition.

    Parameters
    ----------
    sessions:
        One :class:`SessionResult` per seed. Each carries
        ``submitted_requests`` (list of ``{"request": ..., "reasoning": ...}``)
        and ``seed``.
    method:
        ``"best_consistency_run"`` (default) selects the seed whose submissions
        have the most counterparts in every other seed.
        ``"union"`` deduplicates across all seeds; useful as a robustness check.

    Returns
    -------
    AgentAggregationResult
        Mirrors the one-shot generator's output so downstream code is uniform.
    """
    reps: List[_RepetitionResult] = []
    for s in sessions:
        try:
            reqs = [
                DemonstrationRequest.from_dict(sr["request"]) for sr in s.submitted_requests
            ]
        except Exception as e:  # malformed session — record but keep going.
            reqs = []
            err = f"deserialize_failed: {type(e).__name__}: {e}"
        else:
            err = s.error
        reps.append(
            _RepetitionResult(
                rep_idx=s.seed,
                requests=reqs,
                raw_response="",
                n_retries=0,
                error=err,
            )
        )

    best_idx, metrics = _consistency_score(reps, similarity_threshold=similarity_threshold)
    union = _union_aggregate(reps)

    if method == "union":
        selected_requests = union
        selected_seed = -1
    else:
        # Find the rep matching best_idx (rep_idx == seed in our adapter).
        selected = next(r for r in reps if r.rep_idx == best_idx)
        selected_requests = selected.requests
        selected_seed = selected.rep_idx

    metrics = {
        "method": method,
        "similarity_threshold": similarity_threshold,
        "n_sessions": len(sessions),
        "n_selected_requests": len(selected_requests),
        "n_union_requests": len(union),
        **metrics,
    }
    return AgentAggregationResult(
        selected_seed=selected_seed,
        selected_requests=selected_requests,
        consistency_metrics=metrics,
        union_requests=union,
    )


def write_aggregate_artefacts(
    result: AgentAggregationResult,
    *,
    out_dir: Path,
) -> None:
    """Write the same files the one-shot pipeline does, so downstream is uniform."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_run.json").write_text(
        json.dumps(
            {
                "selected_seed": result.selected_seed,
                "requests": [r.to_dict() for r in result.selected_requests],
            },
            indent=2,
            default=str,
        )
    )
    (out_dir / "consistency_metrics.json").write_text(
        json.dumps(result.consistency_metrics, indent=2, default=str)
    )
    (out_dir / "union_run.json").write_text(
        json.dumps(
            {"requests": [r.to_dict() for r in result.union_requests]},
            indent=2,
            default=str,
        )
    )

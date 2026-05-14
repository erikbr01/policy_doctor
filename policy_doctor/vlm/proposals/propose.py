"""Batch proposal generation for Experiment E2.

Pipeline per condition:

  1. Build messages via the configured :class:`VLMInputBuilder`.
  2. Call the VLM backend's :meth:`generate_structured` (text → JSON; retry on
     parse/validation failure with the error fed back into the prompt).
  3. Parse + validate against the request schema, tagging the source condition.
  4. Repeat ``n_repetitions`` times, then aggregate to a single chosen list using
     ``best_consistency_run`` (default) or ``union``.

The aggregation step never sees ``target_cluster`` — clustering of similar
requests is done over operator-facing text only, so the consistency metric
treats both conditions on equal footing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from policy_doctor.vlm.proposals.pool import RolloutPool
from policy_doctor.vlm.proposals.registry import (
    get_graph_representation,
    get_vlm_input_builder,
)
from policy_doctor.vlm.proposals.request import (
    DemonstrationRequest,
    REQUEST_TYPES,
    RequestValidationError,
    parse_and_validate_batch,
    request_json_schema,
)
from policy_doctor.vlm.proposals.vlm_input.base import Message


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class _RepetitionResult:
    """One VLM repetition's parsed batch + parse/validation diagnostics."""

    rep_idx: int
    requests: List[DemonstrationRequest]
    raw_response: str
    n_retries: int
    error: Optional[str] = None


@dataclass
class ProposalBatchResult:
    """End-to-end outcome of one condition's proposal generation."""

    condition: str
    selected_requests: List[DemonstrationRequest]
    repetitions: List[_RepetitionResult] = field(default_factory=list)
    consistency_metrics: Dict[str, Any] = field(default_factory=dict)
    aggregation: str = "best_consistency_run"
    selected_rep_idx: int = 0


# ---------------------------------------------------------------------------
# Backend-side structured generation interface
# ---------------------------------------------------------------------------
#
# Backends are expected to implement ``generate_structured`` with this contract:
#
#     def generate_structured(
#         self,
#         messages: list[Message],
#         json_schema: dict,
#         *,
#         max_retries: int = 3,
#         temperature: float = 0.3,
#         seed: int | None = None,
#     ) -> tuple[str, dict, int]:
#         """Return (raw_response_text, parsed_json, n_retries_used).
#         Retry-with-error-feedback is implemented by the backend (or by the
#         shim below if the backend only exposes a plain text-generation method).
#         """
#
# We provide a shim around any existing backend that has a generic
# ``generate(messages) -> str`` method, so we can keep this working without
# touching the per-backend code in vlm/backends/.
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object/array from a free-form response."""
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()
    # Walk for the first {...} or [...] balanced span.
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            c = text[j]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
    return text.strip()


def call_backend_structured(
    backend: Any,
    messages: List[Message],
    json_schema: Dict[str, Any],
    *,
    max_retries: int = 3,
    temperature: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], int]:
    """Adapter: prefer backend.generate_structured if defined, else fall back to
    a plain-text generate method with retry-on-parse loop."""

    if hasattr(backend, "generate_structured"):
        return backend.generate_structured(
            messages=messages,
            json_schema=json_schema,
            max_retries=max_retries,
            temperature=temperature,
            seed=seed,
        )

    if not hasattr(backend, "generate"):
        raise TypeError(
            f"VLM backend {type(backend).__name__} has neither "
            "generate_structured nor generate; cannot drive it for E2 proposals."
        )

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        msgs = list(messages)
        if last_err is not None:
            msgs = msgs + [
                Message(
                    role="user",
                    text_blocks=[
                        "Your previous response was not valid JSON or did not match the requested "
                        f"schema. Error:\n{last_err}\n\nReturn a valid JSON object with the "
                        "exact schema described in the system prompt and nothing else."
                    ],
                )
            ]
        raw = backend.generate(msgs, temperature=temperature, seed=seed)
        try:
            parsed = json.loads(_extract_json(raw))
            return raw, parsed, attempt
        except json.JSONDecodeError as e:
            last_err = f"JSONDecodeError: {e}"
            continue

    raise RuntimeError(
        f"backend.generate failed to produce valid JSON after {max_retries + 1} attempts; "
        f"last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Per-rep generation + parse + validate
# ---------------------------------------------------------------------------


def _generate_one_rep(
    *,
    backend: Any,
    builder: Any,
    graph_artefact: Any,
    pool: RolloutPool,
    condition: str,
    n_requests_per_type: Dict[str, int],
    rep_idx: int,
    temperature: float,
    seed: Optional[int],
    max_retries: int,
    task_hint: str,
) -> _RepetitionResult:
    schema = request_json_schema(with_target_cluster=(condition == "graph"))
    messages = builder.build_messages(
        graph_artefact=graph_artefact,
        pool=pool,
        condition=condition,
        n_requests_per_type=n_requests_per_type,
        json_schema=schema,
        history=None,
        task_hint=task_hint,
    )

    error: Optional[str] = None
    requests: List[DemonstrationRequest] = []
    raw = ""
    retries = 0
    try:
        raw, parsed, retries = call_backend_structured(
            backend, messages, schema,
            max_retries=max_retries, temperature=temperature, seed=seed,
        )
        requests = parse_and_validate_batch(
            parsed,
            source_condition=condition,
            allowed_rollout_ids=set(pool.rollout_ids),
        )
    except (RuntimeError, RequestValidationError, ValueError) as e:
        error = f"{type(e).__name__}: {e}"

    return _RepetitionResult(
        rep_idx=rep_idx,
        requests=requests,
        raw_response=raw,
        n_retries=retries,
        error=error,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return set(_TOKEN_RE.findall((s or "").lower()))


def _request_similarity(a: DemonstrationRequest, b: DemonstrationRequest) -> float:
    """Jaccard over normalized tokens of the operator-facing text. NEVER
    inspects target_cluster — the metric must be condition-agnostic."""
    if a.request_type != b.request_type:
        return 0.0
    if a.initial_conditions.reference_rollout_id != b.initial_conditions.reference_rollout_id:
        # Different reference rollouts → not the same demo to collect.
        return 0.0
    ta = _tokens(a.target_behavior)
    tb = _tokens(b.target_behavior)
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(1, union)


def _consistency_score(
    reps: Sequence[_RepetitionResult],
    *,
    similarity_threshold: float = 0.5,
) -> Tuple[int, Dict[str, Any]]:
    """Pick the rep with the highest count of requests that have a
    similarity-≥-threshold counterpart in EACH OTHER rep. Returns (best_idx, metrics)."""

    successful = [r for r in reps if not r.error and r.requests]
    if not successful:
        return 0, {"n_successful_reps": 0, "per_rep_match_counts": []}

    if len(successful) == 1:
        return successful[0].rep_idx, {
            "n_successful_reps": 1,
            "per_rep_match_counts": [len(successful[0].requests)],
            "consistency_rate": 1.0,
        }

    per_rep_counts: List[int] = []
    for r in successful:
        others = [o for o in successful if o.rep_idx != r.rep_idx]
        n_matched = 0
        for req in r.requests:
            ok = True
            for o in others:
                best_sim = max(
                    (_request_similarity(req, x) for x in o.requests),
                    default=0.0,
                )
                if best_sim < similarity_threshold:
                    ok = False
                    break
            if ok:
                n_matched += 1
        per_rep_counts.append(n_matched)

    best_local = max(range(len(successful)), key=lambda i: per_rep_counts[i])
    best_rep = successful[best_local]
    total = sum(len(r.requests) for r in successful)
    matched = sum(per_rep_counts)
    return best_rep.rep_idx, {
        "n_successful_reps": len(successful),
        "per_rep_match_counts": per_rep_counts,
        "consistency_rate": (matched / total) if total else 0.0,
        "similarity_threshold": similarity_threshold,
    }


def _union_aggregate(reps: Sequence[_RepetitionResult]) -> List[DemonstrationRequest]:
    """Take all valid requests across reps, dedup by (request_type, reference_rollout_id, target_behavior token set)."""
    seen: Dict[Tuple[str, str, frozenset], DemonstrationRequest] = {}
    for r in reps:
        if r.error:
            continue
        for req in r.requests:
            key = (
                req.request_type,
                req.initial_conditions.reference_rollout_id,
                frozenset(_tokens(req.target_behavior)),
            )
            seen.setdefault(key, req)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_proposals(
    *,
    backend: Any,
    pool: RolloutPool,
    behavior_graph: Any,
    condition: str,
    n_requests_per_type: Dict[str, int],
    output_dir: Path,
    n_repetitions: int = 2,
    aggregation: str = "best_consistency_run",
    temperature: float = 0.3,
    base_seed: Optional[int] = 42,
    max_retries: int = 3,
    task_hint: str = "",
    graph_representation_name: str = "combined",
    graph_representation_params: Optional[Dict[str, Any]] = None,
    vlm_input_builder_name: Optional[str] = None,
    vlm_input_builder_params: Optional[Dict[str, Any]] = None,
) -> ProposalBatchResult:
    """Generate one condition's proposal batch.

    Parameters
    ----------
    backend:
        Instance returned by :func:`policy_doctor.vlm.get_vlm_backend`.
    behavior_graph:
        :class:`policy_doctor.behaviors.behavior_graph.BehaviorGraph`. Required even
        in ``outcome_only`` condition because the graph_representation is rendered
        once per run and the condition's input builder chooses what to include.
    output_dir:
        Per-condition subdirectory; raw responses, selected list, consistency
        metrics, and the rendered graph artefact are written here.
    aggregation:
        ``best_consistency_run`` (default) or ``union``.
    """
    if condition not in ("graph", "outcome_only"):
        raise ValueError(f"condition must be 'graph' or 'outcome_only', got {condition!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render the graph artefact once — both conditions share the same render call,
    # but the outcome_condition input builder is expected to ignore the visual parts.
    graph_repr = get_graph_representation(
        graph_representation_name, graph_representation_params or {}
    )
    artefact = graph_repr.render(behavior_graph, pool, output_dir / "graph_artefact")

    builder_name = vlm_input_builder_name or (
        "graph_condition" if condition == "graph" else "outcome_condition"
    )
    builder = get_vlm_input_builder(builder_name, vlm_input_builder_params or {})

    reps: List[_RepetitionResult] = []
    for i in range(n_repetitions):
        seed_i = None if base_seed is None else int(base_seed + i)
        r = _generate_one_rep(
            backend=backend,
            builder=builder,
            graph_artefact=artefact,
            pool=pool,
            condition=condition,
            n_requests_per_type=n_requests_per_type,
            rep_idx=i,
            temperature=temperature,
            seed=seed_i,
            max_retries=max_retries,
            task_hint=task_hint,
        )
        reps.append(r)
        with open(output_dir / f"run_{i + 1}.json", "w") as f:
            json.dump(
                {
                    "rep_idx": i,
                    "n_retries": r.n_retries,
                    "error": r.error,
                    "raw_response": r.raw_response,
                    "requests": [req.to_dict() for req in r.requests],
                },
                f,
                indent=2,
                default=str,
            )

    # Aggregate
    if aggregation == "union":
        selected = _union_aggregate(reps)
        # Use the metrics block as a record but mark it union-mode
        _, metrics = _consistency_score(reps)
        metrics["aggregation"] = "union"
        selected_rep_idx = -1
    else:
        selected_rep_idx, metrics = _consistency_score(reps)
        for r in reps:
            if r.rep_idx == selected_rep_idx and not r.error:
                selected = list(r.requests)
                break
        else:
            selected = []
        metrics["aggregation"] = "best_consistency_run"

    with open(output_dir / "consistency_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "selected_run.json", "w") as f:
        json.dump(
            {
                "condition": condition,
                "aggregation": aggregation,
                "selected_rep_idx": selected_rep_idx,
                "n_requests": len(selected),
                "requests": [req.to_dict() for req in selected],
            },
            f,
            indent=2,
            default=str,
        )

    return ProposalBatchResult(
        condition=condition,
        selected_requests=selected,
        repetitions=reps,
        consistency_metrics=metrics,
        aggregation=aggregation,
        selected_rep_idx=selected_rep_idx,
    )

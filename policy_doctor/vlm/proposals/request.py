"""DemonstrationRequest schema for Experiment E2.

Contract between the VLM (proposal generator) and the operator (DAgger sim runner).
Operator-facing fields use only behaviorally-observable terms — ``target_cluster`` and
``source_condition`` are server-side metadata that **must not** leak to the operator.

Validation enforces a denylist on operator-facing strings to catch condition leaks
(``cluster``, ``node``, ``graph``, …) before a request is enqueued.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

# Strings forbidden in any operator-facing string field. Catches the obvious
# ways the graph condition could leak via VLM-emitted text. Case-insensitive.
DENYLIST_PATTERNS: Tuple[str, ...] = (
    r"\bcluster(s|ing)?\b",
    r"\bnode(s)?\b",
    r"\bgraph\b",
    r"\bbehavior\s+graph\b",
    r"\bumap\b",
    r"\bk[- ]?means\b",
    r"\bcentroid(s)?\b",
    r"\bembedding(s)?\b",
)

REQUEST_TYPES: Tuple[str, ...] = (
    "full_trajectory",
    "recovery",
    "alternative_strategy",
)

# Conditions accepted in ``source_condition``. The agentic conditions
# (A_G, A_NG, H_NG, H_G) are documented in
# :mod:`policy_doctor.vlm.proposals.agents.conditions`. The legacy one-shot
# names ("graph", "outcome_only") remain as the canonical names of the
# one-shot ablation path — they are not aliases.
CONDITIONS: Tuple[str, ...] = (
    "graph",
    "outcome_only",
    "A_G",
    "A_NG",
    "H_NG",
    "H_G",
)


@dataclass
class InitialConditions:
    """Where the sim should be reset to before the operator starts."""

    reference_rollout_id: str
    reference_frame: int = 0          # 0 = rollout start; >0 = mid-rollout (recovery)
    # Filled in by RolloutPool.resolve(): per-task description, opaque to operator
    # of any condition signal. Pulled from the reference rollout's recorded scene.
    object_poses: Dict[str, List[float]] = field(default_factory=dict)
    gripper_state: Dict[str, Any] = field(default_factory=dict)
    robot_pose: Dict[str, Any] = field(default_factory=dict)
    # Tolerances are nominal in sim (init_state reset is bit-exact); kept for
    # protocol parity with a real-robot version.
    tolerances: Dict[str, float] = field(default_factory=dict)


@dataclass
class DemonstrationRequest:
    """Single VLM-proposed demonstration request.

    Operator-visible fields:    target_behavior, prohibitions, success_criterion,
                                initial_conditions, request_id, request_type.
    Operator-HIDDEN fields:     target_cluster, source_condition.
    """

    request_id: str
    request_type: str               # full_trajectory | recovery | alternative_strategy
    initial_conditions: InitialConditions
    target_behavior: str            # behaviorally-observable, no cluster references
    prohibitions: List[str] = field(default_factory=list)   # operator-facing free text
    success_criterion: str = "task_success"

    # ---- server-side metadata (NEVER displayed to operator) -----------------
    target_cluster: Optional[int] = None     # set in graph condition; post-hoc in outcome
    source_condition: Optional[str] = None   # graph | outcome_only

    # ---- evidence (operator-facing in normalized form) ---------------------
    # The agent must cite at least N storyboard slices/rollouts it has
    # visually inspected. We persist these so the operator can see WHAT the
    # agent saw, and so the experimental record captures the evidence chain.
    # Operator endpoints normalize both into ``reference_storyboard_ids``
    # to avoid leaking the condition via field name.
    evidence_slice_ids: List[str] = field(default_factory=list)       # A_G / H_G
    evidence_rollout_ids: List[str] = field(default_factory=list)     # A_NG / H_NG

    @classmethod
    def new_id(cls) -> str:
        """Opaque UUIDv4. Never encode condition or any VLM output in the id."""
        return uuid.uuid4().hex[:12]

    # ---- (de)serialization --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["initial_conditions"] = asdict(self.initial_conditions)
        return d

    def to_operator_dict(self) -> Dict[str, Any]:
        """Strip server-side fields; this is what the operator UI sees.

        Evidence ids (slice or rollout) are normalized into a single
        ``reference_storyboard_ids`` field so the operator UI can render the
        same imagery for both conditions without leaking which path the agent
        took. The condition-specific source fields are dropped.
        """
        d = self.to_dict()
        d.pop("target_cluster", None)
        d.pop("source_condition", None)
        # Normalize: combine evidence_slice_ids + evidence_rollout_ids into
        # one operator-facing field. Drop the condition-revealing originals.
        ev_slices = d.pop("evidence_slice_ids", []) or []
        ev_rollouts = d.pop("evidence_rollout_ids", []) or []
        d["reference_storyboard_ids"] = list(ev_slices) + list(ev_rollouts)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DemonstrationRequest":
        ic = d["initial_conditions"]
        if isinstance(ic, dict):
            ic = InitialConditions(**ic)
        return cls(
            evidence_slice_ids=list(d.get("evidence_slice_ids") or []),
            evidence_rollout_ids=list(d.get("evidence_rollout_ids") or []),
            request_id=d["request_id"],
            request_type=d["request_type"],
            initial_conditions=ic,
            target_behavior=d["target_behavior"],
            prohibitions=list(d.get("prohibitions") or []),
            success_criterion=d.get("success_criterion", "task_success"),
            target_cluster=d.get("target_cluster"),
            source_condition=d.get("source_condition"),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class RequestValidationError(ValueError):
    """Raised when a request fails schema or denylist validation."""


_DENYLIST_RE = re.compile("|".join(DENYLIST_PATTERNS), flags=re.IGNORECASE)


def _check_denylist(field_name: str, value: str, errors: List[str]) -> None:
    if not isinstance(value, str):
        return
    m = _DENYLIST_RE.search(value)
    if m:
        errors.append(
            f"{field_name!r} contains forbidden term {m.group(0)!r} "
            f"(operator-facing fields must not reveal the experimental condition)"
        )


def validate_request(
    req: DemonstrationRequest,
    *,
    allowed_rollout_ids: Optional[set] = None,
) -> None:
    """Raise :class:`RequestValidationError` if *req* is malformed.

    Parameters
    ----------
    req:
        The request to validate.
    allowed_rollout_ids:
        Optional set of valid ``reference_rollout_id`` values; when provided, the
        request must reference a known rollout.
    """
    errors: List[str] = []

    # Required strings
    if not req.request_id:
        errors.append("request_id is empty")
    if req.request_type not in REQUEST_TYPES:
        errors.append(
            f"request_type={req.request_type!r} not in {REQUEST_TYPES}"
        )
    if not req.target_behavior:
        errors.append("target_behavior is empty")

    # Source condition (only checked if set; can be None pre-server-tagging)
    if req.source_condition is not None and req.source_condition not in CONDITIONS:
        errors.append(
            f"source_condition={req.source_condition!r} not in {CONDITIONS}"
        )

    # Operator-facing denylist
    _check_denylist("target_behavior", req.target_behavior, errors)
    _check_denylist("success_criterion", req.success_criterion, errors)
    for i, p in enumerate(req.prohibitions):
        _check_denylist(f"prohibitions[{i}]", p, errors)

    # Initial conditions
    ic = req.initial_conditions
    if not ic.reference_rollout_id:
        errors.append("initial_conditions.reference_rollout_id is empty")
    if ic.reference_frame < 0:
        errors.append(
            f"initial_conditions.reference_frame={ic.reference_frame} is negative"
        )
    if (
        allowed_rollout_ids is not None
        and ic.reference_rollout_id
        and ic.reference_rollout_id not in allowed_rollout_ids
    ):
        errors.append(
            f"reference_rollout_id={ic.reference_rollout_id!r} not in rollout pool"
        )

    if errors:
        raise RequestValidationError(
            "Invalid DemonstrationRequest:\n  - " + "\n  - ".join(errors)
        )


def parse_and_validate_batch(
    raw: Any,
    *,
    source_condition: str,
    allowed_rollout_ids: Optional[set] = None,
) -> List[DemonstrationRequest]:
    """Parse a JSON-shaped list into validated requests, tagging the condition.

    Accepts either a list or an object with key ``requests``. Raises
    :class:`RequestValidationError` on the first failure with a single-pass
    aggregated error message across all entries.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and "requests" in raw:
        raw = raw["requests"]
    if not isinstance(raw, list):
        raise RequestValidationError(
            f"Expected a list of requests, got {type(raw).__name__}"
        )

    out: List[DemonstrationRequest] = []
    errors: List[str] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            errors.append(f"requests[{i}] is not an object")
            continue
        try:
            entry = dict(entry)
            entry.setdefault("request_id", DemonstrationRequest.new_id())
            entry["source_condition"] = source_condition
            req = DemonstrationRequest.from_dict(entry)
            validate_request(req, allowed_rollout_ids=allowed_rollout_ids)
            out.append(req)
        except (KeyError, TypeError, RequestValidationError) as e:
            errors.append(f"requests[{i}]: {e}")
    if errors:
        raise RequestValidationError(
            "Batch validation failed:\n  - " + "\n  - ".join(errors)
        )
    return out


# ---------------------------------------------------------------------------
# JSON schema for VLM structured output
# ---------------------------------------------------------------------------


def request_json_schema(*, with_target_cluster: bool) -> Dict[str, Any]:
    """JSON schema we ask the VLM to emit. Same shape both conditions, but
    ``target_cluster`` is required only for the graph condition.
    """
    item_props: Dict[str, Any] = {
        "request_type": {"type": "string", "enum": list(REQUEST_TYPES)},
        "target_behavior": {"type": "string", "minLength": 1},
        "prohibitions": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "success_criterion": {"type": "string"},
        "initial_conditions": {
            "type": "object",
            "properties": {
                "reference_rollout_id": {"type": "string"},
                "reference_frame": {"type": "integer", "minimum": 0},
            },
            "required": ["reference_rollout_id"],
        },
    }
    required = ["request_type", "target_behavior", "initial_conditions"]
    if with_target_cluster:
        item_props["target_cluster"] = {"type": "integer"}
        required.append("target_cluster")
    return {
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": required,
                },
            },
        },
        "required": ["requests"],
    }

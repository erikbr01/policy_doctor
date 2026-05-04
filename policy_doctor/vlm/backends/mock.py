"""Deterministic mock VLM for tests and dry runs."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from policy_doctor.vlm.backends.base import (
    AssistantTurn,
    TokenUsage,
    ToolCall,
    VLMBackend,
)


class MockVLMBackend(VLMBackend):
    name = "mock"

    def __init__(self, prefix: str = "[mock]", **_kwargs) -> None:
        self.prefix = prefix

    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        sp = (system_prompt or "")[:40]
        up = user_prompt[:80].replace("\n", " ")
        return f"{self.prefix} frames={len(images)} user={up!r} system_hint={sp!r}"

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        n = len(slice_labels)
        head = slice_labels[0][:60] if slice_labels else ""
        return (
            f"{self.prefix} behavior cluster={cluster_id} n_slices={n} "
            f"task={task_hint!r} first_label={head!r}"
        )

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        n = len(slice_labels)
        return (
            '{"coherent": true, "score": 0.85, "rationale": "mock backend cluster '
            f"{cluster_id} n_slices={n}"
            '"}'
        )


    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
    ) -> str:
        # Always predicts the first group label — deterministic for tests.
        if example_sets:
            label = example_sets[0][0]
        else:
            label = "unclear"
        n_groups = len(example_sets)
        return (
            f"{self.prefix} classify: n_groups={n_groups} "
            f"n_query_frames={len(query_images)} predicted={label!r}"
        )


    def generate_structured(
        self,
        *,
        messages,
        json_schema: Dict[str, Any],
        max_retries: int = 3,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> Tuple[str, Any, int]:
        """Emit a deterministic, schema-shaped batch suitable for E2 plumbing tests.

        Reads ``schema.properties.requests.items`` to figure out which fields are
        required and whether ``target_cluster`` must be included. Pulls a few
        rollout ids from the messages text to keep the references_rollout_id
        validator happy (``parse_and_validate_batch`` checks them against the pool).
        """
        item_schema = (
            json_schema.get("properties", {})
            .get("requests", {})
            .get("items", {})
        )
        required = set(item_schema.get("required", []))
        with_target_cluster = "target_cluster" in required

        joined_text = "\n".join(
            tb for m in messages for tb in getattr(m, "text_blocks", []) or []
        )
        rollout_ids: List[str] = []
        for tok in joined_text.split():
            tok = tok.strip(".,;:[]()'\"")
            if tok.startswith("r") and tok[1:].isdigit() and tok not in rollout_ids:
                rollout_ids.append(tok)
            if len(rollout_ids) >= 4:
                break
        if not rollout_ids:
            rollout_ids = ["r0000"]

        types = ["full_trajectory", "recovery", "alternative_strategy"]
        out = []
        for i in range(min(3, len(rollout_ids))):
            entry = {
                "request_type": types[i % len(types)],
                "target_behavior": "approach the target object and complete the task",
                "prohibitions": [],
                "success_criterion": "task_success",
                "initial_conditions": {
                    "reference_rollout_id": rollout_ids[i],
                    "reference_frame": 0,
                },
            }
            if with_target_cluster:
                entry["target_cluster"] = i
            out.append(entry)
        payload = {"requests": out}
        raw = json.dumps(payload)
        return raw, payload, 0


    # ------------------------------------------------------------------
    # chat_with_tools — deterministic scripted agent for Tier 0
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> AssistantTurn:
        """Walk a fixed exploration script regardless of model state.

        The script visits each layer in order so Tier 0 exercises the entire
        plumbing path: ``get_graph_summary`` → ``list_nodes`` → 1-2 visual
        peeks → 3 ``propose_collection_request`` calls → ``finalize_strategy``.

        Tool selection is robust to the actual surface — if a tool isn't in
        the registry (e.g. running A_NG which has no ``list_nodes``), the
        script substitutes the closest available tool.
        """
        tool_names = {t["name"] for t in tools}
        n_assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

        # Pull rollout ids out of the message text for use in submissions.
        text_corpus = _flatten_text(messages)
        rollout_ids = _extract_rollout_ids(text_corpus, k=4)
        if not rollout_ids:
            rollout_ids = ["r0000", "r0001", "r0002", "r0003"]

        # Decide what to do this turn.
        script = _SCRIPT_A_G if "list_nodes" in tool_names else _SCRIPT_A_NG
        with_target_cluster = _propose_takes_target_cluster(tools)
        idx = n_assistant_turns

        if idx >= len(script):
            # Safety net: end the session if the script ran out.
            return _finalize_turn("scripted termination after exhausting plan")

        step = script[idx]
        kind = step["kind"]

        if kind == "tool":
            name = step["name"]
            if name not in tool_names:
                # Fall back to the first available submission/finalize tool.
                name = "finalize_strategy"
            args = _scripted_args(name, step, rollout_ids, idx, with_target_cluster)
            return _tool_use_turn(name, args)
        if kind == "finalize":
            return _finalize_turn(step.get("rationale", "scripted mock rationale"))
        return _finalize_turn("scripted termination")


def build_mock_backend(params: dict) -> MockVLMBackend:
    return MockVLMBackend(**(params or {}))


# ---------------------------------------------------------------------------
# Mock scripts
# ---------------------------------------------------------------------------


_SCRIPT_A_G = [
    {"kind": "tool", "name": "get_graph_summary"},
    {"kind": "tool", "name": "list_nodes", "args": {}},
    {"kind": "tool", "name": "list_paths", "args": {"from_node": "START", "to_node": "FAILURE", "top_k": 3}},
    # Inspect each cluster the script will later target — required by the
    # ``cluster_not_inspected`` gate on the submission validator.
    {"kind": "tool", "name": "get_node", "args_idx": 0},
    {"kind": "tool", "name": "get_node", "args_idx": 1},
    {"kind": "tool", "name": "get_node", "args_idx": 2},
    {"kind": "tool", "name": "propose_collection_request", "submission_idx": 0},
    {"kind": "tool", "name": "propose_collection_request", "submission_idx": 1},
    {"kind": "tool", "name": "propose_collection_request", "submission_idx": 2},
    {"kind": "tool", "name": "list_submitted_requests", "args": {}},
    {"kind": "finalize", "rationale": "Three requests target the highest-failure region; "
     "additional demonstrations there should reduce policy failure rate."},
]


_SCRIPT_A_NG = [
    {"kind": "tool", "name": "list_rollouts", "args": {"outcome": "failure", "n": 5}},
    {"kind": "tool", "name": "get_rollout_summary", "rollout_idx": 0},
    {"kind": "tool", "name": "propose_collection_request", "submission_idx": 0},
    {"kind": "tool", "name": "propose_collection_request", "submission_idx": 1},
    {"kind": "tool", "name": "propose_collection_request", "submission_idx": 2},
    {"kind": "finalize", "rationale": "Targeted three failure rollouts for recovery demonstrations."},
]


# Per-submission target_behavior text — must vary across submissions to pass
# the duplicate-target-behavior gate.
_TARGET_BEHAVIORS = [
    "approach the workspace from above with the gripper open and grasp the target object",
    "after the initial slip, stabilize the object and complete the placement cleanly",
    "use a side approach instead of overhead and lift the object onto the platform",
]


def _propose_takes_target_cluster(tools: List[Dict[str, Any]]) -> bool:
    for t in tools:
        if t["name"] == "propose_collection_request":
            return "target_cluster" in (t.get("input_schema", {}).get("properties") or {})
    return False


def _scripted_args(
    name: str,
    step: Dict[str, Any],
    rollout_ids: List[str],
    turn_idx: int,
    with_target_cluster: bool,
) -> Dict[str, Any]:
    if name == "get_graph_summary":
        return {}
    if name == "list_nodes":
        return step.get("args", {})
    if name == "list_paths":
        return step.get("args", {"from_node": "START", "to_node": "FAILURE", "top_k": 3})
    if name == "get_node":
        return {"node_id": int(step.get("args_idx", 0))}
    if name == "list_rollouts":
        return step.get("args", {"outcome": "failure", "n": 5})
    if name == "get_rollout_summary":
        idx = int(step.get("rollout_idx", 0)) % len(rollout_ids)
        return {"rollout_id": rollout_ids[idx]}
    if name == "list_submitted_requests":
        return {}
    if name == "propose_collection_request":
        sub_idx = int(step.get("submission_idx", 0))
        types = ["full_trajectory", "recovery", "alternative_strategy"]
        request_type = types[sub_idx % len(types)]
        rid = rollout_ids[sub_idx % len(rollout_ids)]
        body: Dict[str, Any] = {
            "request_type": request_type,
            "initial_conditions": {
                "reference_rollout_id": rid,
                # Recovery requires reference_frame >= 1; vary by submission so
                # successive recovery requests still differ in initial state.
                "reference_frame": (5 + sub_idx) if request_type == "recovery" else 0,
            },
            "target_behavior": _TARGET_BEHAVIORS[sub_idx % len(_TARGET_BEHAVIORS)],
            "prohibitions": ["do not push the object off the table"],
            "success_criterion": "task_success",
            "reasoning": (
                f"scripted submission {sub_idx + 1}: rollout {rid} reveals a recoverable failure mode; "
                "demonstrating the recovery should improve the policy."
            ),
        }
        # Synthesize evidence using the fixture's slice-id format. The A_G
        # script targets cluster=sub_idx, with windows at j*5 in the
        # fixture (clusters 0,1,2 → windows 0-4, 5-9, 10-14). For A_NG we
        # cite 3 distinct rollout_ids the test will pre-inspect.
        if with_target_cluster:
            body["target_cluster"] = sub_idx
            window_start = sub_idx * 5
            window_end = window_start + 4
            body["evidence_slice_ids"] = [
                f"{rollout_ids[i % len(rollout_ids)]}_t{window_start}_t{window_end}"
                for i in range(3)
            ]
        else:
            body["evidence_rollout_ids"] = list(rollout_ids[:3])
        return body
    return {}


def _tool_use_turn(name: str, arguments: Dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(
        text=None,
        tool_calls=[ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=64, output_tokens=32),
    )


def _finalize_turn(rationale: str) -> AssistantTurn:
    return AssistantTurn(
        text=None,
        tool_calls=[
            ToolCall(
                id=f"mock_{uuid.uuid4().hex[:8]}",
                name="finalize_strategy",
                arguments={"rationale": rationale},
            )
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=64, output_tokens=24),
    )


def _flatten_text(messages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    chunks.append(str(blk.get("text", "")))
                elif isinstance(blk, dict) and blk.get("type") == "tool_result":
                    inner = blk.get("content")
                    if isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                chunks.append(str(sub.get("text", "")))
                    elif isinstance(inner, str):
                        chunks.append(inner)
    return "\n".join(chunks)


_ROLLOUT_RE = re.compile(r"\br\d{3,5}\b")


def _extract_rollout_ids(text: str, *, k: int = 4) -> List[str]:
    seen: List[str] = []
    for tok in _ROLLOUT_RE.findall(text or ""):
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= k:
            break
    return seen

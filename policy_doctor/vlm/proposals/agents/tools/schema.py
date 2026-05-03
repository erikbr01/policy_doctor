"""JSON schemas for every agent tool.

These are the **frozen contract** between the agent and the experiment
runner. They are committed to the repo, hashed in ``pre_registration.yaml``,
and must not be edited after an experiment run starts.

Every schema is a JSON Schema (draft 2020-12 compatible) describing the tool's
``input_schema``. The ``description`` strings are rendered into the tool
declarations the agent sees and influence behavior, so they're treated as
prompt material — frozen along with the schemas.

Conventions:
* ``node_id`` uses the same integer cluster IDs as :mod:`behavior_graph`
  (``-2`` = START, ``-4`` = SUCCESS, ``-5`` = FAILURE, ``-3`` = END,
  ``>=0`` = behavior cluster).
* ``rollout_id`` is the opaque ``r{NNNN}`` form from
  :func:`policy_doctor.vlm.proposals.pool.episode_idx_to_rollout_id`.
* ``slice_id`` is ``"{rollout_id}_t{start_frame}_t{end_frame}"``.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Layer 1: graph topology
# ---------------------------------------------------------------------------


GET_GRAPH_SUMMARY: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


LIST_NODES: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "min_failure_likelihood": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.0,
            "description": "Only return nodes whose probability of reaching FAILURE within "
            "this graph is at least this value.",
        },
        "min_v": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.0,
            "description": "Only return nodes whose value (V) is at least this. V is the "
            "Bellman value computed under the graph's transition matrix with "
            "+1 for SUCCESS, -1 for FAILURE.",
        },
        "max_v": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "default": 1.0,
        },
    },
    "additionalProperties": False,
}


LIST_PATHS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "from_node": {
            "type": "string",
            "default": "START",
            "description": "Node id to enumerate paths from. Use 'START' for the source.",
        },
        "to_node": {
            "type": "string",
            "default": "FAILURE",
            "description": "Terminal node id to enumerate paths to. Use 'SUCCESS', "
            "'FAILURE', or 'END'.",
        },
        "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
    },
    "additionalProperties": False,
}


GET_NODE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "node_id": {"type": "integer"},
    },
    "required": ["node_id"],
    "additionalProperties": False,
}


GET_EDGE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "from_node": {"type": "integer"},
        "to_node": {"type": "integer"},
    },
    "required": ["from_node", "to_node"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Layer 2: slice and rollout access
# ---------------------------------------------------------------------------


LIST_SLICES_IN_NODE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "node_id": {"type": "integer"},
        "n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        "sort_by": {
            "type": "string",
            "enum": ["centroid_distance", "random"],
            "default": "centroid_distance",
            "description": "'centroid_distance' lists most-prototypical slices first.",
        },
    },
    "required": ["node_id"],
    "additionalProperties": False,
}


GET_SLICE_VIDEO: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "slice_id": {"type": "string"},
        "format": {
            "type": "string",
            "enum": ["storyboard", "video"],
            "default": "storyboard",
            "description": "'storyboard' returns a 4-frame composite (cheap). "
            "'video' returns an inline animation (counts against the video budget).",
        },
    },
    "required": ["slice_id"],
    "additionalProperties": False,
}


GET_ROLLOUT_SUMMARY: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "rollout_id": {"type": "string"},
    },
    "required": ["rollout_id"],
    "additionalProperties": False,
}


GET_ROLLOUT_VIDEO: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "rollout_id": {"type": "string"},
        "format": {
            "type": "string",
            "enum": ["storyboard", "video"],
            "default": "storyboard",
        },
    },
    "required": ["rollout_id"],
    "additionalProperties": False,
}


LIST_ROLLOUTS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": ["string", "null"],
            "enum": ["success", "failure", None],
            "default": None,
        },
        "passes_through": {
            "type": "array",
            "items": {"type": "integer"},
            "default": [],
            "description": "Optional list of node ids the rollout must traverse (in order).",
        },
        "n": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Layer 3: search and aggregation
# ---------------------------------------------------------------------------


FIND_FAILURE_NODES: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "min_failure_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.3},
    },
    "additionalProperties": False,
}


FIND_RECOVERY_PATHS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "from_node": {"type": "integer"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["from_node"],
    "additionalProperties": False,
}


FIND_UNDERREPRESENTED_MODES: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric": {
            "type": "string",
            "enum": ["rollout_count", "v"],
            "default": "rollout_count",
        },
        "threshold": {"type": "integer", "minimum": 1, "default": 5},
    },
    "additionalProperties": False,
}


COMPARE_PATHS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path_a": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
        "path_b": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
    },
    "required": ["path_a", "path_b"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Layer 4: strategy submission (the agent's output channel)
# ---------------------------------------------------------------------------


_INITIAL_CONDITIONS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reference_rollout_id": {"type": "string"},
        "reference_frame": {"type": "integer", "minimum": 0, "default": 0},
    },
    "required": ["reference_rollout_id"],
    "additionalProperties": False,
}


def propose_collection_request_schema(*, with_target_cluster: bool) -> Dict[str, Any]:
    """JSON schema for ``propose_collection_request``.

    The A_G surface emits ``target_cluster``; the A_NG surface omits it
    (the field is computed post-hoc from the reference rollout).
    """
    props: Dict[str, Any] = {
        "request_type": {
            "type": "string",
            "enum": ["full_trajectory", "recovery", "alternative_strategy"],
        },
        "initial_conditions": _INITIAL_CONDITIONS_SCHEMA,
        "target_behavior": {
            "type": "string",
            "minLength": 1,
            "description": "Behaviorally-observable description of what the operator should do. "
            "MUST NOT mention clusters, nodes, the graph, embeddings, or any "
            "internal representation — these terms leak the experimental condition.",
        },
        "prohibitions": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
            "description": "Optional list of behaviors the operator must avoid.",
        },
        "success_criterion": {
            "type": "string",
            "default": "task_success",
            "description": "Behaviorally-observable success criterion.",
        },
        "reasoning": {
            "type": "string",
            "minLength": 1,
            "description": "REQUIRED. One or two sentences explaining why this request matters "
            "for improving the policy. Logged to the trace; not shown to the operator.",
        },
    }
    required = [
        "request_type",
        "initial_conditions",
        "target_behavior",
        "reasoning",
    ]
    if with_target_cluster:
        props["target_cluster"] = {
            "type": "integer",
            "description": "Behavior-cluster id this request targets.",
        }
        required.append("target_cluster")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


LIST_SUBMITTED_REQUESTS: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


REVISE_REQUEST: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "target_behavior": {"type": "string"},
        "prohibitions": {"type": "array", "items": {"type": "string"}},
        "success_criterion": {"type": "string"},
        "reasoning": {
            "type": "string",
            "description": "Updated reasoning. Required when revising.",
        },
    },
    "required": ["request_id", "reasoning"],
    "additionalProperties": False,
}


DELETE_REQUEST: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
    },
    "required": ["request_id"],
    "additionalProperties": False,
}


FINALIZE_STRATEGY: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "minLength": 1,
            "description": "Brief overall rationale for the submitted strategy. Required.",
        },
    },
    "required": ["rationale"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# No-graph parallel surface (A_NG / H_NG)
# ---------------------------------------------------------------------------


NG_LIST_ROLLOUTS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": ["string", "null"],
            "enum": ["success", "failure", None],
            "default": None,
        },
        "n": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
    },
    "additionalProperties": False,
}


NG_GET_ROLLOUT_SUMMARY: Dict[str, Any] = {
    "type": "object",
    "properties": {"rollout_id": {"type": "string"}},
    "required": ["rollout_id"],
    "additionalProperties": False,
}


NG_GET_ROLLOUT_VIDEO: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "rollout_id": {"type": "string"},
        "format": {
            "type": "string",
            "enum": ["storyboard", "video"],
            "default": "storyboard",
        },
    },
    "required": ["rollout_id"],
    "additionalProperties": False,
}


NG_LIST_FAILURE_ROLLOUTS: Dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}},
    "additionalProperties": False,
}


NG_LIST_SUCCESS_ROLLOUTS: Dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}},
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Hashing — used to bind a pre-registration manifest.
# ---------------------------------------------------------------------------


def schema_hash() -> str:
    """SHA-256 of the canonical JSON encoding of every schema in this module.

    Recorded in ``pre_registration.yaml`` to prove the contract was not edited
    after an experiment started.
    """
    import hashlib
    import json as _json

    items = [
        ("get_graph_summary", GET_GRAPH_SUMMARY),
        ("list_nodes", LIST_NODES),
        ("list_paths", LIST_PATHS),
        ("get_node", GET_NODE),
        ("get_edge", GET_EDGE),
        ("list_slices_in_node", LIST_SLICES_IN_NODE),
        ("get_slice_video", GET_SLICE_VIDEO),
        ("get_rollout_summary", GET_ROLLOUT_SUMMARY),
        ("get_rollout_video", GET_ROLLOUT_VIDEO),
        ("list_rollouts", LIST_ROLLOUTS),
        ("find_failure_nodes", FIND_FAILURE_NODES),
        ("find_recovery_paths", FIND_RECOVERY_PATHS),
        ("find_underrepresented_modes", FIND_UNDERREPRESENTED_MODES),
        ("compare_paths", COMPARE_PATHS),
        ("propose_collection_request_with_cluster", propose_collection_request_schema(with_target_cluster=True)),
        ("propose_collection_request_no_cluster", propose_collection_request_schema(with_target_cluster=False)),
        ("list_submitted_requests", LIST_SUBMITTED_REQUESTS),
        ("revise_request", REVISE_REQUEST),
        ("delete_request", DELETE_REQUEST),
        ("finalize_strategy", FINALIZE_STRATEGY),
        ("ng_list_rollouts", NG_LIST_ROLLOUTS),
        ("ng_get_rollout_summary", NG_GET_ROLLOUT_SUMMARY),
        ("ng_get_rollout_video", NG_GET_ROLLOUT_VIDEO),
        ("ng_list_failure_rollouts", NG_LIST_FAILURE_ROLLOUTS),
        ("ng_list_success_rollouts", NG_LIST_SUCCESS_ROLLOUTS),
    ]
    canonical = _json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

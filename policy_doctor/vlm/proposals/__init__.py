"""Experiment E2: VLM-proposed demonstration requests for graph-informed data collection.

Top-level API:

    from policy_doctor.vlm.proposals import (
        DemonstrationRequest, InitialConditions,
        RolloutPool, generate_proposals, score_request_adherence,
        get_graph_representation, get_vlm_input_builder,
    )

Layout:

  request.py            DemonstrationRequest dataclass + denylist validator
  pool.py               RolloutPool index
  init_state.py         Mid-rollout sim_state extraction
  registry.py           Hydra-string registries for graph_representation/* and vlm_input/*
  graph_representation/ Pluggable BehaviorGraph → VLM artefact renderers
  vlm_input/            Pluggable per-condition message builders (graph / outcome_only)
  propose.py            Batch proposal generation (N reps, validation retry, aggregation)
  adherence.py          3-axis (init/cluster/success) automated scoring
  chat.py               Multi-turn message construction (dev mode)
"""

from policy_doctor.vlm.proposals.request import (
    CONDITIONS,
    REQUEST_TYPES,
    DemonstrationRequest,
    InitialConditions,
    RequestValidationError,
    parse_and_validate_batch,
    request_json_schema,
    validate_request,
)
from policy_doctor.vlm.proposals.pool import (
    RolloutEntry,
    RolloutPool,
    episode_idx_to_rollout_id,
    rollout_id_to_episode_idx,
)
from policy_doctor.vlm.proposals.init_state import (
    extract_object_pose_at_frame,
    extract_sim_state_at_frame,
    verify_sim_state_replays,
)
from policy_doctor.vlm.proposals.registry import (
    get_graph_representation,
    get_vlm_input_builder,
    list_graph_representations,
    list_vlm_input_builders,
    register_graph_representation,
    register_vlm_input_builder,
)

__all__ = [
    # request schema
    "CONDITIONS",
    "REQUEST_TYPES",
    "DemonstrationRequest",
    "InitialConditions",
    "RequestValidationError",
    "parse_and_validate_batch",
    "request_json_schema",
    "validate_request",
    # pool
    "RolloutEntry",
    "RolloutPool",
    "episode_idx_to_rollout_id",
    "rollout_id_to_episode_idx",
    # init_state helpers
    "extract_object_pose_at_frame",
    "extract_sim_state_at_frame",
    "verify_sim_state_replays",
    # registries
    "get_graph_representation",
    "get_vlm_input_builder",
    "list_graph_representations",
    "list_vlm_input_builders",
    "register_graph_representation",
    "register_vlm_input_builder",
]

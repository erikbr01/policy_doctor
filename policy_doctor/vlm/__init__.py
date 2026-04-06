"""VLM slice annotation — modular backends and functional helpers."""

from policy_doctor.vlm.annotate import (
    eval_dirs_equivalent,
    load_clustering_artifacts,
    resolve_source_eval_dir_for_jsonl,
    run_slice_annotation_for_eval,
    write_annotations_jsonl,
)
from policy_doctor.vlm.behavior_summarize import (
    group_slice_labels_by_cluster,
    load_slice_annotations_jsonl,
    run_behavior_summarization,
    write_behavior_summaries_json,
)
from policy_doctor.vlm.backends.base import VLMBackend
from policy_doctor.vlm.prompts import (
    behavior_prompt_fingerprint,
    default_task_hint_for_vlm,
    merge_behavior_prompt_config,
    merge_prompt_config,
    prompt_fingerprint,
    resolve_task_vlm_prompts_path,
    resolve_vlm_prompts_file_for_task,
)
from policy_doctor.vlm.registry import get_vlm_backend, list_vlm_backend_names, register_vlm_backend

__all__ = [
    "VLMBackend",
    "get_vlm_backend",
    "list_vlm_backend_names",
    "register_vlm_backend",
    "merge_prompt_config",
    "merge_behavior_prompt_config",
    "behavior_prompt_fingerprint",
    "prompt_fingerprint",
    "resolve_task_vlm_prompts_path",
    "resolve_vlm_prompts_file_for_task",
    "default_task_hint_for_vlm",
    "load_clustering_artifacts",
    "eval_dirs_equivalent",
    "resolve_source_eval_dir_for_jsonl",
    "run_slice_annotation_for_eval",
    "write_annotations_jsonl",
    "load_slice_annotations_jsonl",
    "group_slice_labels_by_cluster",
    "run_behavior_summarization",
    "write_behavior_summaries_json",
]

"""Prompt templates for slice annotation (easy YAML override)."""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

from policy_doctor.paths import PACKAGE_ROOT

# Task-specific prompt bundles: configs/vlm/tasks/registry.yaml + *.yaml
_VLM_TASKS_DIR = PACKAGE_ROOT / "configs" / "vlm" / "tasks"
_VLM_TASK_REGISTRY_PATH = _VLM_TASKS_DIR / "registry.yaml"


DEFAULT_SLICE_SYSTEM: Optional[str] = None

DEFAULT_SLICE_USER_TEMPLATE = """You are describing a short segment of a robot manipulation rollout.
Task context: {task_hint}

This segment is from rollout index {rollout_idx}, timesteps [{window_start}, {window_end}) in that rollout.
The clustering algorithm assigned cluster id {cluster_id} to this segment.

You see {num_frames} video frames sampled uniformly from the segment, in temporal order.
Give ONE concise phrase (under 25 words) describing the robot's behavior and relevant object states.
Do not mention cluster ids or timestep numbers in the answer."""


def load_prompts_yaml(path: pathlib.Path | str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompts file must be a mapping: {p}")
    return data


def _prompt_value_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, str) and value.strip().lower() in ("null", "none"):
        return False
    return True


def _load_task_prompt_registry() -> Dict[str, str]:
    """Return lowercase substring → prompts filename (e.g. transport_mh.yaml)."""
    if not _VLM_TASK_REGISTRY_PATH.is_file():
        return {}
    data = yaml.safe_load(_VLM_TASK_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    raw = data.get("by_substring") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(k).lower(): str(v) for k, v in raw.items()}


def resolve_task_vlm_prompts_path(task_config: Optional[str]) -> Optional[pathlib.Path]:
    """Resolve ``tasks/<file>.yaml`` from registry substring match (longest wins)."""
    if not task_config or not _VLM_TASKS_DIR.is_dir():
        return None
    reg = _load_task_prompt_registry()
    if not reg:
        return None
    tc = str(task_config).lower()
    best_sub: Optional[str] = None
    best_file: Optional[str] = None
    for sub, filename in reg.items():
        if sub and sub in tc:
            if best_sub is None or len(sub) > len(best_sub):
                best_sub = sub
                best_file = filename
    if not best_file:
        return None
    p = _VLM_TASKS_DIR / best_file
    return p if p.is_file() else None


def resolve_vlm_prompts_file_for_task(
    explicit_prompts_file: Optional[str],
    task_config: Optional[str],
    *,
    repo_root: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """Effective ``prompts_file`` string for merge_* helpers.

    Explicit Hydra/UI path wins when set. Otherwise use ``tasks/registry.yaml`` mapping
    for ``task_config`` (YAML stem, e.g. ``transport_mh_mar27``).
    """
    if _prompt_value_set(explicit_prompts_file):
        return str(explicit_prompts_file).strip()
    resolved = resolve_task_vlm_prompts_path(task_config)
    if resolved is None:
        return None
    abs_res = resolved.resolve()
    if repo_root is not None:
        try:
            return str(abs_res.relative_to(repo_root.resolve()))
        except ValueError:
            pass
    return str(abs_res)


def read_default_task_hint_from_prompts_file(path: Optional[pathlib.Path]) -> Optional[str]:
    """Optional ``default_task_hint`` key in a task prompts YAML."""
    if path is None or not path.is_file():
        return None
    blob = load_prompts_yaml(path)
    h = blob.get("default_task_hint")
    if not _prompt_value_set(h):
        return None
    return str(h).strip()


def default_task_hint_for_vlm(
    explicit_prompts_file: Optional[str],
    task_config: Optional[str],
    *,
    repo_root: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """``default_task_hint`` from the effective prompts file (explicit or registry-resolved)."""
    pf = resolve_vlm_prompts_file_for_task(explicit_prompts_file, task_config, repo_root=repo_root)
    if not pf:
        return None
    path = pathlib.Path(pf)
    if not path.is_absolute() and repo_root is not None:
        path = repo_root / path
    return read_default_task_hint_from_prompts_file(path.resolve())


def merge_prompt_config(
    *,
    prompts_file: Optional[str],
    prompts_inline: Optional[Dict[str, Any]],
    repo_root: Optional[pathlib.Path] = None,
) -> Tuple[Optional[str], str]:
    """Return (slice_system, slice_user_template) after optional file merge."""
    system: Optional[str] = DEFAULT_SLICE_SYSTEM
    user = DEFAULT_SLICE_USER_TEMPLATE

    if prompts_file and str(prompts_file).lower() not in ("null", "none", ""):
        path = pathlib.Path(prompts_file)
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        blob = load_prompts_yaml(path)
        pr = blob.get("prompts") or blob
        if isinstance(pr, dict):
            if _prompt_value_set(pr.get("slice_system")):
                system = pr["slice_system"]
            if _prompt_value_set(pr.get("slice_user_template")):
                user = str(pr["slice_user_template"])

    if prompts_inline:
        pr = prompts_inline.get("prompts", prompts_inline)
        if isinstance(pr, dict):
            if _prompt_value_set(pr.get("slice_system")):
                system = pr["slice_system"]
            if _prompt_value_set(pr.get("slice_user_template")):
                user = str(pr["slice_user_template"])

    return system, user


def format_slice_prompts(
    *,
    slice_system: Optional[str],
    slice_user_template: str,
    task_hint: str,
    rollout_idx: int,
    window_start: int,
    window_end: int,
    cluster_id: int,
    num_frames: int,
) -> Tuple[Optional[str], str]:
    fmt_kw = {
        "task_hint": task_hint,
        "rollout_idx": rollout_idx,
        "window_start": window_start,
        "window_end": window_end,
        "cluster_id": cluster_id,
        "num_frames": num_frames,
    }
    user = slice_user_template.format(**fmt_kw)
    return slice_system, user


def normalize_slice_reasoning_effort(value: Any) -> str:
    """Return ``none`` | ``medium`` | ``high`` for slice annotation."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return "none"
    s = str(value).strip().lower()
    if s in ("null", "none", "off", "false", "0"):
        return "none"
    if s in ("medium", "med", "balanced"):
        return "medium"
    if s in ("high", "max", "full"):
        return "high"
    raise ValueError(
        f"vlm_annotation.reasoning_effort must be none/medium/high, got {value!r}"
    )


# Appended to slice *system* template (before .format on user only — system has no placeholders).
_SLICE_REASONING_MEDIUM = """

Accuracy mode (medium): Before answering, mentally compare the first and last frames for each arm and
for lid, cube, and hammer (location + contact). If nothing clearly moves, say static or idle. Do not
describe steps that are not visible."""


_SLICE_REASONING_HIGH = """

High-accuracy / chain-of-thought protocol (required):
1. The frames are in strict time order—treat them as a short clip (not independent photos).
2. Privately: compare ONLY the first frame vs the last frame. For LEFT and RIGHT arm, note gripper
   pose and what (if anything) each contacts. For lid, cube, hammer: each is in a box (left vs right),
   in a gripper (which arm), on the table, or not visible.
3. State whether anything clearly changed between first and last. If motion is ambiguous, say so.
4. Do not narrate future steps, handovers, or placements that are not visible in these frames.

Then output EXACTLY one line in this form (no text after this line):
FINAL: <one factual sentence, max 50 words; no quotes>"""


def extract_slice_final_label(raw_text: str, *, reasoning_effort: str) -> str:
    """If *reasoning_effort* is ``high``, take the ``FINAL:`` line; else return stripped *raw_text*."""
    text = (raw_text or "").strip()
    if reasoning_effort != "high" or not text:
        return text
    for line in text.splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith("FINAL:"):
            rest = s.split(":", 1)[1].strip()
            if rest:
                return rest
    return text


def apply_slice_reasoning_to_templates(
    slice_system: Optional[str],
    slice_user_template: str,
    effort: str,
) -> Tuple[Optional[str], str]:
    """Augment prompts for deeper reasoning; *effort* is normalized (see ``normalize_slice_reasoning_effort``)."""
    if effort == "none":
        return slice_system, slice_user_template
    suffix = _SLICE_REASONING_HIGH if effort == "high" else _SLICE_REASONING_MEDIUM
    base_sys = (slice_system or "").rstrip()
    merged = f"{base_sys}{suffix}" if base_sys else suffix.strip()
    return merged, slice_user_template


def prompt_fingerprint(
    slice_system: Optional[str],
    slice_user_template: str,
    reasoning_effort: str = "none",
) -> str:
    # Omit effort from the hash when ``none`` so legacy prompt bundles keep stable fingerprints.
    if reasoning_effort == "none":
        raw = f"{slice_system or ''}\n{slice_user_template}"
    else:
        raw = f"{reasoning_effort}\n{slice_system or ''}\n{slice_user_template}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


DEFAULT_BEHAVIOR_SYSTEM: Optional[str] = None

DEFAULT_BEHAVIOR_USER_TEMPLATE = """Task context: {task_hint}

Unsupervised clustering assigned the same behavior cluster id ({cluster_id}) to several rollout segments.
Below are short visual descriptions for individual segments (one per line). They may agree or disagree.

{slice_labels_bulleted}

Write 2–4 sentences that:
- Summarize the common behavior when segments agree
- Explicitly note important discrepancies when they do not
Avoid repeating the list numbering or cluster id in your answer."""


def merge_behavior_prompt_config(
    *,
    prompts_file: Optional[str],
    prompts_inline: Optional[Dict[str, Any]],
    repo_root: Optional[pathlib.Path] = None,
) -> Tuple[Optional[str], str]:
    """Return (behavior_system, behavior_user_template)."""
    system: Optional[str] = DEFAULT_BEHAVIOR_SYSTEM
    user = DEFAULT_BEHAVIOR_USER_TEMPLATE

    if prompts_file and str(prompts_file).lower() not in ("null", "none", ""):
        path = pathlib.Path(prompts_file)
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        blob = load_prompts_yaml(path)
        pr = blob.get("prompts") or blob
        if isinstance(pr, dict):
            if _prompt_value_set(pr.get("behavior_system")):
                system = pr["behavior_system"]
            if _prompt_value_set(pr.get("behavior_user_template")):
                user = str(pr["behavior_user_template"])

    if prompts_inline:
        pr = prompts_inline.get("prompts", prompts_inline)
        if isinstance(pr, dict):
            if _prompt_value_set(pr.get("behavior_system")):
                system = pr["behavior_system"]
            if _prompt_value_set(pr.get("behavior_user_template")):
                user = str(pr["behavior_user_template"])

    return system, user


def format_behavior_prompts(
    *,
    behavior_system: Optional[str],
    behavior_user_template: str,
    task_hint: str,
    cluster_id: int,
    slice_labels: Sequence[str],
) -> Tuple[Optional[str], str]:
    lines = [f"- {s.strip()}" for s in slice_labels if str(s).strip()]
    bulleted = "\n".join(lines) if lines else "(no labels)"
    fmt_kw = {
        "task_hint": task_hint,
        "cluster_id": cluster_id,
        "num_slice_labels": len(lines),
        "slice_labels_bulleted": bulleted,
    }
    user = behavior_user_template.format(**fmt_kw)
    return behavior_system, user


def behavior_prompt_fingerprint(
    behavior_system: Optional[str],
    behavior_user_template: str,
) -> str:
    raw = f"{behavior_system or ''}\n{behavior_user_template}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

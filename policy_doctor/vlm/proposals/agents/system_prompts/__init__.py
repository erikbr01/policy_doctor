"""Frozen, committed system prompts for each agent condition.

Prompts are stored as plain ``.md`` files in this package directory rather
than as Python string literals. That makes diffs reviewable, hashes stable,
and editing risk-free for non-engineers.

Pre-registration:

* :func:`prompt_text(condition)` returns the canonical UTF-8 text.
* :func:`prompt_hash(condition)` returns the SHA-256 hex digest of that text.
* :func:`all_prompt_hashes()` returns ``{condition: hash}`` for every
  registered condition. Embedded in ``pre_registration.yaml``.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import Dict

from policy_doctor.vlm.proposals.agents.conditions import (
    Condition,
    parse_condition,
)


_PROMPT_FILE = {
    Condition.A_G: "A_G.md",
    Condition.A_NG: "A_NG.md",
    # H_NG / H_G use the same written instructions delivered via the
    # Streamlit operator UI; we route them through the same prompt files.
    Condition.H_NG: "A_NG.md",
    Condition.H_G: "A_G.md",
}


def prompt_text(condition: Condition | str) -> str:
    cond = parse_condition(condition)
    fname = _PROMPT_FILE.get(cond)
    if fname is None:
        raise KeyError(f"no system prompt registered for condition {cond}")
    pkg_files = files(__name__)
    return (pkg_files / fname).read_text(encoding="utf-8")


def prompt_hash(condition: Condition | str) -> str:
    return hashlib.sha256(prompt_text(condition).encode("utf-8")).hexdigest()


def all_prompt_hashes() -> Dict[str, str]:
    return {cond.value: prompt_hash(cond) for cond in (Condition.A_G, Condition.A_NG)}

"""Replay the golden-snapshot anchors and assert current code reproduces them.

This is the load-bearing regression check for the architecture refactor
(see REFACTOR_PLAN.md §5). It must pass at every phase boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.refactor import snapshot

GOLDEN_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize("anchor", [a[0] for a in snapshot._ANCHORS])
def test_golden_anchor_replay(anchor: str) -> None:
    """Each anchor: (a) compute current outputs; (b) load golden; (c) assert equal."""
    cfg = next(a for a in snapshot._ANCHORS if a[0] == anchor)
    _, inputs_fn, compute_fn, _save_fn, load_fn, cmp_fn = cfg

    actual = compute_fn(inputs_fn())
    expected = load_fn(GOLDEN_DIR / anchor)
    failures = cmp_fn(actual, expected)

    assert not failures, "\n".join([f"Golden mismatch for {anchor}:", *failures])

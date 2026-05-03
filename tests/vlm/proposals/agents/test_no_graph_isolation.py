"""Strict isolation test for the A_NG / H_NG tool surface.

Asserts that no cluster-level vocabulary leaks through the no-graph tools'
outputs. Mirrors the request-schema denylist in
:mod:`policy_doctor.vlm.proposals.request` but applied to *every* field of
*every* tool result.

This is the load-bearing guarantee of the experiment: A_NG vs A_G must
isolate the contribution of the graph itself, not "more information." If the
no-graph surface accidentally surfaces cluster ids or path information, the
comparison is contaminated.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable, List

from policy_doctor.vlm.proposals.agents.tools.registry import build_tool_registry
from policy_doctor.vlm.proposals.agents.tools.types import TextBlock, ToolResult
from tests.vlm.proposals.agents.conftest import build_fixture_context


# Same patterns as :mod:`policy_doctor.vlm.proposals.request._DENYLIST_RE`.
_FORBIDDEN = re.compile(
    r"\bcluster(s|ing)?\b|\bnode(s)?\b|\bgraph\b|\bbehavior\s+graph\b|"
    r"\bumap\b|\bk[- ]?means\b|\bcentroid(s)?\b|\bembedding(s)?\b",
    re.IGNORECASE,
)


# Field names that are themselves leaks (even if the value happens to be empty).
_FORBIDDEN_KEY_NAMES = {
    "cluster_path",
    "cluster_path_ids",
    "target_cluster",
    "node_id",
    "in_degree",
    "out_degree",
    "centroid_distance",
}


def _walk_strings(obj: Any) -> Iterable[str]:
    """Yield every string value reachable inside a JSON-shaped object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_strings(item)


def _walk_keys(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _walk_keys(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_keys(item)


def _result_text(result: ToolResult) -> List[str]:
    """Collect all string content from a ToolResult (including captions)."""
    out: List[str] = []
    for blk in result.content or []:
        if isinstance(blk, TextBlock):
            out.append(blk.text or "")
        else:
            cap = getattr(blk, "caption", None)
            if cap:
                out.append(cap)
    return out


class TestNoGraphIsolation(unittest.TestCase):
    """Run every A_NG tool on the fixture and scan their outputs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ctx = build_fixture_context(self.tmp, condition="A_NG")
        self.tools = build_tool_registry("A_NG", self.ctx)

    def _scan_result(self, tool_name: str, result: ToolResult) -> None:
        # 1. Free-text content must contain no forbidden tokens.
        for text in _result_text(result):
            m = _FORBIDDEN.search(text)
            self.assertIsNone(
                m,
                f"{tool_name}: free-text output contains forbidden term {m.group(0) if m else '?'}: "
                f"{text!r}",
            )

        # 2. JSON-shaped content (the common case for textual tools) — also
        #    inspect every field name and string value inside.
        for text in _result_text(result):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            for key in _walk_keys(payload):
                self.assertNotIn(
                    key.lower(),
                    {k.lower() for k in _FORBIDDEN_KEY_NAMES},
                    f"{tool_name}: response includes forbidden key {key!r}",
                )
            for s in _walk_strings(payload):
                m = _FORBIDDEN.search(s)
                self.assertIsNone(
                    m,
                    f"{tool_name}: nested string {s!r} contains forbidden term "
                    f"{m.group(0) if m else '?'}",
                )

    def test_list_rollouts_clean(self):
        r = self.tools["list_rollouts"].func({"outcome": "failure"})
        self._scan_result("list_rollouts", r)

    def test_list_failure_rollouts_clean(self):
        r = self.tools["list_failure_rollouts"].func({})
        self._scan_result("list_failure_rollouts", r)

    def test_list_success_rollouts_clean(self):
        r = self.tools["list_success_rollouts"].func({})
        self._scan_result("list_success_rollouts", r)

    def test_get_rollout_summary_clean(self):
        r = self.tools["get_rollout_summary"].func({"rollout_id": "r0001"})
        self._scan_result("get_rollout_summary", r)

    def test_get_rollout_video_no_cluster_metadata(self):
        # Even on the error path, no leaks.
        r = self.tools["get_rollout_video"].func({"rollout_id": "r0000", "format": "storyboard"})
        self._scan_result("get_rollout_video", r)

    def test_propose_collection_request_no_target_cluster_in_schema(self):
        # The A_NG submission tool must declare a schema without target_cluster.
        spec = self.tools["propose_collection_request"]
        props = spec.input_schema.get("properties") or {}
        self.assertNotIn("target_cluster", props)
        self.assertNotIn("target_cluster", spec.input_schema.get("required") or [])

    def test_no_a_g_tools_present(self):
        # A_NG registry must NOT include any Layer 1 / Layer 2 graph-aware tools.
        forbidden_a_g = {
            "get_graph_summary",
            "list_nodes",
            "list_paths",
            "get_node",
            "get_edge",
            "list_slices_in_node",
            "get_slice_video",
            "find_failure_nodes",
            "find_recovery_paths",
            "find_underrepresented_modes",
            "compare_paths",
        }
        present = set(self.tools.keys())
        leaked = forbidden_a_g & present
        self.assertFalse(leaked, f"A_NG registry leaks A_G tools: {leaked}")


if __name__ == "__main__":
    unittest.main()

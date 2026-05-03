"""Schema-level checks for the agent tool surface contract.

These tests catch contract drift: every schema must be a valid JSON Schema,
the propose-collection-request schema must vary correctly between A_G and
A_NG, and ``schema_hash()`` must be stable for an unchanged module.
"""

from __future__ import annotations

import json
import unittest

from policy_doctor.vlm.proposals.agents.tools import schema as S
from policy_doctor.vlm.proposals.agents.tools.types import (
    ImageBlock,
    TextBlock,
    ToolResult,
    ToolSpec,
)


_ALL_SCHEMAS = {
    "GET_GRAPH_SUMMARY": S.GET_GRAPH_SUMMARY,
    "LIST_NODES": S.LIST_NODES,
    "LIST_PATHS": S.LIST_PATHS,
    "GET_NODE": S.GET_NODE,
    "GET_EDGE": S.GET_EDGE,
    "LIST_SLICES_IN_NODE": S.LIST_SLICES_IN_NODE,
    "GET_SLICE_VIDEO": S.GET_SLICE_VIDEO,
    "GET_ROLLOUT_SUMMARY": S.GET_ROLLOUT_SUMMARY,
    "GET_ROLLOUT_VIDEO": S.GET_ROLLOUT_VIDEO,
    "LIST_ROLLOUTS": S.LIST_ROLLOUTS,
    "FIND_FAILURE_NODES": S.FIND_FAILURE_NODES,
    "FIND_RECOVERY_PATHS": S.FIND_RECOVERY_PATHS,
    "FIND_UNDERREPRESENTED_MODES": S.FIND_UNDERREPRESENTED_MODES,
    "COMPARE_PATHS": S.COMPARE_PATHS,
    "LIST_SUBMITTED_REQUESTS": S.LIST_SUBMITTED_REQUESTS,
    "REVISE_REQUEST": S.REVISE_REQUEST,
    "DELETE_REQUEST": S.DELETE_REQUEST,
    "FINALIZE_STRATEGY": S.FINALIZE_STRATEGY,
    "NG_LIST_ROLLOUTS": S.NG_LIST_ROLLOUTS,
    "NG_GET_ROLLOUT_SUMMARY": S.NG_GET_ROLLOUT_SUMMARY,
    "NG_GET_ROLLOUT_VIDEO": S.NG_GET_ROLLOUT_VIDEO,
    "NG_LIST_FAILURE_ROLLOUTS": S.NG_LIST_FAILURE_ROLLOUTS,
    "NG_LIST_SUCCESS_ROLLOUTS": S.NG_LIST_SUCCESS_ROLLOUTS,
}


class TestSchemaShape(unittest.TestCase):
    def test_every_schema_is_json_serializable_object(self):
        for name, sch in _ALL_SCHEMAS.items():
            with self.subTest(name=name):
                self.assertIsInstance(sch, dict)
                self.assertEqual(sch.get("type"), "object")
                # Round-trip JSON serialization (catches non-JSON values)
                json.dumps(sch)

    def test_every_required_field_exists_in_properties(self):
        for name, sch in _ALL_SCHEMAS.items():
            with self.subTest(name=name):
                props = set((sch.get("properties") or {}).keys())
                required = sch.get("required") or []
                missing = [r for r in required if r not in props]
                self.assertFalse(missing, f"{name}: required={missing} not in properties")

    def test_propose_request_schemas_differ_only_by_target_cluster(self):
        with_cluster = S.propose_collection_request_schema(with_target_cluster=True)
        without_cluster = S.propose_collection_request_schema(with_target_cluster=False)

        # target_cluster present iff with_target_cluster
        self.assertIn("target_cluster", with_cluster["properties"])
        self.assertNotIn("target_cluster", without_cluster["properties"])
        self.assertIn("target_cluster", with_cluster["required"])
        self.assertNotIn("target_cluster", without_cluster["required"])

        # Reasoning is required in both
        self.assertIn("reasoning", with_cluster["required"])
        self.assertIn("reasoning", without_cluster["required"])

    def test_schema_hash_is_stable_within_run(self):
        h1 = S.schema_hash()
        h2 = S.schema_hash()
        self.assertEqual(h1, h2)
        # Sanity: it's a hex sha-256.
        self.assertEqual(len(h1), 64)
        int(h1, 16)


class TestToolResultConstructors(unittest.TestCase):
    def test_text_constructor(self):
        r = ToolResult.text("get_graph_summary", "ok", latency_ms=12.5)
        self.assertTrue(r.ok)
        self.assertEqual(r.name, "get_graph_summary")
        self.assertEqual(len(r.content), 1)
        self.assertIsInstance(r.content[0], TextBlock)
        self.assertEqual(r.metadata["latency_ms"], 12.5)

    def test_error_constructor(self):
        r = ToolResult.error("get_node", "node_id not found", code="not_found")
        self.assertFalse(r.ok)
        self.assertIn("[error:not_found]", r.content[0].text)
        self.assertEqual(r.metadata["error_code"], "not_found")


class TestToolSpecDeclaration(unittest.TestCase):
    def test_declaration_shape(self):
        def _stub(_args):
            return ToolResult.text("noop", "")

        spec = ToolSpec(
            name="noop",
            description="does nothing",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            func=_stub,
        )
        decl = spec.declaration()
        self.assertEqual(decl["name"], "noop")
        self.assertEqual(decl["description"], "does nothing")
        self.assertEqual(decl["input_schema"]["type"], "object")
        self.assertEqual(spec.cost, "cheap")
        self.assertFalse(spec.is_terminal)


if __name__ == "__main__":
    unittest.main()

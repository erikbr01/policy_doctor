"""MCP server wiring tests.

The MCP SDK is an optional dependency, so we stub the ``mcp.types`` module
when needed. Tests focus on the SDK-free pieces:

* env-var → config translation
* SessionContext + tool registry build
* ToolResult → MCP content-item translation
* Submission persistence to disk
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _install_stub_mcp():
    """Install minimal stub modules for ``mcp.types`` so the server imports.

    The real SDK provides typed dataclasses; for the translation tests we just
    need objects with ``type``, ``text`` / ``data`` / ``mimeType`` attributes
    so the test assertions can read them.
    """
    if "mcp" in sys.modules:
        return
    mcp = types.ModuleType("mcp")
    types_mod = types.ModuleType("mcp.types")

    class TextContent:
        def __init__(self, *, type, text):
            self.type = type
            self.text = text

    class ImageContent:
        def __init__(self, *, type, data, mimeType):
            self.type = type
            self.data = data
            self.mimeType = mimeType

    class Tool:
        def __init__(self, *, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    types_mod.TextContent = TextContent
    types_mod.ImageContent = ImageContent
    types_mod.Tool = Tool
    mcp.types = types_mod
    sys.modules["mcp"] = mcp
    sys.modules["mcp.types"] = types_mod


class TestEnvConfig(unittest.TestCase):
    def test_required_env_vars_enforced(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            _load_config_from_env,
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                _load_config_from_env()
            self.assertIn("POLICY_DOCTOR_CLUSTERING_DIR", str(ctx.exception))

    def test_minimal_config_parses(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            _load_config_from_env,
        )

        with patch.dict(
            os.environ,
            {
                "POLICY_DOCTOR_CLUSTERING_DIR": "/tmp/clust",
                "POLICY_DOCTOR_POOL_EPISODES_DIR": "/tmp/pool",
            },
            clear=True,
        ):
            cfg = _load_config_from_env()
        self.assertEqual(cfg["condition"], "A_G")
        self.assertEqual(cfg["clustering_dir"], Path("/tmp/clust"))
        self.assertEqual(cfg["pool_episodes_dir"], Path("/tmp/pool"))
        self.assertGreaterEqual(cfg["max_tool_calls"], 1000)  # effectively unlimited
        self.assertEqual(cfg["kin_strategy"], "raw_states")

    def test_overrides_apply(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            _load_config_from_env,
        )

        with patch.dict(
            os.environ,
            {
                "POLICY_DOCTOR_CLUSTERING_DIR": "/tmp/clust",
                "POLICY_DOCTOR_POOL_EPISODES_DIR": "/tmp/pool",
                "POLICY_DOCTOR_CONDITION": "A_NG",
                "POLICY_DOCTOR_OUT_DIR": "/tmp/out",
                "POLICY_DOCTOR_MAX_VISUAL_CALLS": "5",
                "POLICY_DOCTOR_KIN_STRATEGY": "cluster_stats",
            },
            clear=True,
        ):
            cfg = _load_config_from_env()
        self.assertEqual(cfg["condition"], "A_NG")
        self.assertEqual(cfg["max_visual_calls"], 5)
        self.assertEqual(cfg["kin_strategy"], "cluster_stats")
        self.assertEqual(cfg["out_dir"], Path("/tmp/out"))


class TestToolResultTranslation(unittest.TestCase):
    def setUp(self):
        _install_stub_mcp()

    def test_text_block_round_trips(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            tool_result_to_mcp_content,
        )
        from policy_doctor.vlm.proposals.agents.tools.types import (
            TextBlock,
            ToolResult,
        )

        result = ToolResult(name="get_node", ok=True, content=[TextBlock(text="ok body")])
        items = tool_result_to_mcp_content(result)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].type, "text")
        self.assertEqual(items[0].text, "ok body")

    def test_image_block_emits_caption_then_image(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            tool_result_to_mcp_content,
        )
        from policy_doctor.vlm.proposals.agents.tools.types import (
            ImageBlock,
            ToolResult,
        )

        img = Image.new("RGB", (32, 32), color=(20, 30, 40))
        result = ToolResult(
            name="get_slice_video",
            ok=True,
            content=[ImageBlock(image=img, caption="r0001 storyboard")],
        )
        items = tool_result_to_mcp_content(result)
        # caption text + image content
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].type, "text")
        self.assertEqual(items[0].text, "r0001 storyboard")
        self.assertEqual(items[1].type, "image")
        self.assertEqual(items[1].mimeType, "image/jpeg")
        # base64 data round-trips back to a JPEG byte string.
        import base64
        decoded = base64.standard_b64decode(items[1].data)
        self.assertTrue(decoded.startswith(b"\xff\xd8"))  # JPEG magic

    def test_empty_content_emits_sentinel(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            tool_result_to_mcp_content,
        )
        from policy_doctor.vlm.proposals.agents.tools.types import ToolResult

        result = ToolResult(name="noop", ok=True, content=[])
        items = tool_result_to_mcp_content(result)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].type, "text")


class TestSessionPersistence(unittest.TestCase):
    """Submitted requests / rationale / budget are flushed to disk after each call."""

    def setUp(self):
        _install_stub_mcp()
        self.tmp = Path(tempfile.mkdtemp())

    def test_persist_writes_three_artefacts(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            persist_session,
        )
        from tests.vlm.proposals.agents.conftest import build_fixture_context

        ctx = build_fixture_context(self.tmp, condition="A_G")
        ctx.rationale = "test rationale"

        persist_session(ctx, self.tmp / "out")

        self.assertTrue((self.tmp / "out" / "submitted_requests.json").exists())
        self.assertTrue((self.tmp / "out" / "rationale.txt").exists())
        self.assertTrue((self.tmp / "out" / "budget_summary.json").exists())


class TestStatusDecoration(unittest.TestCase):
    """The MCP server appends a status line to every tool result, mirroring AgentSession."""

    def setUp(self):
        _install_stub_mcp()
        self.tmp = Path(tempfile.mkdtemp())

    def test_status_line_appended_normal_call(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            _decorate_with_status,
        )
        from policy_doctor.vlm.proposals.agents.tools.types import (
            TextBlock, ToolResult,
        )
        from tests.vlm.proposals.agents.conftest import build_fixture_context

        ctx = build_fixture_context(self.tmp)
        result = ToolResult(name="get_node", ok=True, content=[TextBlock(text="payload")])
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            tool_result_to_mcp_content,
        )
        items = _decorate_with_status(
            tool_result_to_mcp_content(result),
            ctx=ctx, target_n_submissions=4, warning=None,
        )
        # Last item is the status line.
        self.assertGreater(len(items), 1)
        last = items[-1]
        self.assertEqual(last.type, "text")
        self.assertIn("[session:", last.text)
        self.assertIn("0/4 requests submitted", last.text)
        self.assertNotIn("REMINDER", last.text)

    def test_warning_emits_reminder_prefix(self):
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            _decorate_with_status, tool_result_to_mcp_content,
        )
        from policy_doctor.vlm.proposals.agents.tools.types import (
            TextBlock, ToolResult,
        )
        from tests.vlm.proposals.agents.conftest import build_fixture_context

        ctx = build_fixture_context(self.tmp)
        result = ToolResult(name="get_node", ok=True, content=[TextBlock(text="payload")])
        items = _decorate_with_status(
            tool_result_to_mcp_content(result),
            ctx=ctx, target_n_submissions=4,
            warning={"warning": "approaching_visual_budget", "remaining": 3},
        )
        self.assertIn("REMINDER", items[-1].text)
        self.assertIn("3", items[-1].text)


class TestBudgetEnforcement(unittest.TestCase):
    """Pre-flight budget check: exhausted budget rejects calls before they run."""

    def setUp(self):
        _install_stub_mcp()
        self.tmp = Path(tempfile.mkdtemp())

    def _build_handler(self, *, max_tool_calls=2, max_visual_calls=10):
        """Construct the MCP call handler against the fixture context.

        Returns ``(call_tool, ctx, tools, call_log)``. ``call_log`` records
        every spec.func invocation so we can assert exhausted calls did NOT
        run the underlying tool.
        """
        import asyncio as _asyncio

        from policy_doctor.vlm.proposals.agents.budget import BudgetConfig
        from policy_doctor.vlm.proposals.agents.context import SessionContext
        from policy_doctor.vlm.proposals.agents.mcp_server.server import (
            _decorate_with_status, persist_session, tool_result_to_mcp_content,
        )
        from policy_doctor.vlm.proposals.agents.tools.registry import (
            build_tool_registry,
        )
        from tests.vlm.proposals.agents.conftest import (
            make_fixture_graph, make_fixture_labels_metadata, make_fixture_pool,
        )

        labels, metadata = make_fixture_labels_metadata()
        graph = make_fixture_graph()
        pool = make_fixture_pool(self.tmp)
        ctx = SessionContext.build(
            condition="A_G", graph=graph, pool=pool,
            cluster_labels=labels, cluster_metadata=metadata,
            budget_config=BudgetConfig(
                max_tool_calls=max_tool_calls,
                max_visual_calls=max_visual_calls,
                max_video_calls=10,
                max_session_duration_s=600,
            ),
            task_hint="test",
            config={"kinematic_summary_strategy": "cluster_stats"},
        )
        tools = build_tool_registry("A_G", ctx)

        # Wrap each tool's func to record invocations.
        call_log: List[str] = []
        for spec in tools.values():
            orig = spec.func
            def _wrapped(args, _orig=orig, _name=spec.name):
                call_log.append(_name)
                return _orig(args)
            spec.func = _wrapped

        # Build a minimal call_tool replicating the real one (simplified — no
        # MCP server scaffolding needed for unit tests).
        async def call_tool(name, arguments=None):
            spec = tools.get(name)
            if spec is None:
                from mcp.types import TextContent
                return [TextContent(type="text", text=f"[error:unknown_tool] {name!r}")]
            args = arguments or {}
            kind = spec.cost
            if kind == "visual" and args.get("format") == "video":
                kind = "video"
            cached = ctx.cache.get(name, args)
            if cached is not None:
                ctx.budget.note_cache_hit()
                return _decorate_with_status(
                    tool_result_to_mcp_content(cached),
                    ctx=ctx, target_n_submissions=4, warning=None,
                )
            bypass = spec.is_terminal or spec.bypass_budget
            err = ctx.budget.check(name, kind, bypass=bypass)
            if err is not None and spec.bypass_when_exhausted:
                err = None  # recovery affordance
            if err is not None:
                return _decorate_with_status(
                    tool_result_to_mcp_content(err),
                    ctx=ctx, target_n_submissions=4,
                    warning={"warning": f"{kind}_budget_exhausted", "remaining": 0},
                )
            result = await _asyncio.to_thread(spec.func, args)
            if result.ok:
                ctx.budget.charge(kind, bypass=bypass)
                ctx.cache.put(name, args, result)
            return _decorate_with_status(
                tool_result_to_mcp_content(result),
                ctx=ctx, target_n_submissions=4, warning=None,
            )

        return call_tool, ctx, tools, call_log

    def test_exhausted_budget_rejects_before_running_tool(self):
        import asyncio

        call_tool, ctx, tools, call_log = self._build_handler(max_tool_calls=2)

        async def _run():
            # First two budgeted calls go through.
            await call_tool("get_graph_summary", {})
            await call_tool("list_nodes", {})
            # Third call: budget is exhausted; find_failure_nodes is budgeted
            # (analysis tool) so it should NOT run.
            return await call_tool("find_failure_nodes", {"min_failure_prob": 0.3})

        items = asyncio.run(_run())
        # The third tool was NOT invoked.
        self.assertEqual(call_log, ["get_graph_summary", "list_nodes"])
        # Returned error mentions budget_exhausted.
        text = "\n".join(i.text for i in items if i.type == "text")
        self.assertIn("budget_exhausted", text)

    def test_bypass_when_exhausted_lookup_runs_after_exhaustion(self):
        """Read-only ID lookups marked bypass_when_exhausted run after exhaustion.

        Recovery affordance: when an agent realizes mid-submission that it
        needs to inspect a cluster, it can — without hitting the gate. During
        regular operation these tools charge normally (verified separately).
        """
        import asyncio

        call_tool, ctx, tools, call_log = self._build_handler(max_tool_calls=1)

        async def _run():
            # Burn the budget.
            await call_tool("get_graph_summary", {})
            # get_node is bypass_when_exhausted — must still run after exhaustion.
            return await call_tool("get_node", {"node_id": 1})

        items = asyncio.run(_run())
        self.assertIn("get_node", call_log)
        text = "\n".join(i.text for i in items if i.type == "text")
        self.assertNotIn("budget_exhausted", text)
        self.assertIn("node_id", text)

    def test_bypass_when_exhausted_charges_normally_when_budget_remains(self):
        """During regular operation (budget not yet exhausted), bypass_when_exhausted
        tools charge against the budget like everything else — they're not free.
        """
        import asyncio

        call_tool, ctx, tools, call_log = self._build_handler(max_tool_calls=10)

        async def _run():
            # 3 calls; all should charge normally because budget has room.
            await call_tool("get_graph_summary", {})
            await call_tool("get_node", {"node_id": 1})
            await call_tool("get_rollout_summary", {"rollout_id": "r0000"})

        asyncio.run(_run())
        # All three calls charged the budget.
        self.assertEqual(ctx.budget.state.n_tool_calls, 3)

    def test_terminal_finalize_strategy_bypasses_exhausted_budget(self):
        import asyncio

        call_tool, ctx, tools, call_log = self._build_handler(max_tool_calls=1)

        async def _run():
            # Burn the budget.
            await call_tool("get_graph_summary", {})
            # finalize_strategy is is_terminal=True — must still run.
            return await call_tool(
                "finalize_strategy", {"rationale": "done"}
            )

        items = asyncio.run(_run())
        self.assertIn("finalize_strategy", call_log)
        text = "\n".join(i.text for i in items if i.type == "text")
        self.assertIn('"ok": true', text)
        self.assertTrue(ctx.finalized)

    def test_propose_collection_request_bypasses_exhausted_budget(self):
        """Submission tools must run even when exploration budget is exhausted."""
        import asyncio

        call_tool, ctx, tools, call_log = self._build_handler(max_tool_calls=1)
        # Pre-inspect cluster 1 + populate evidence_slice_ids so the only
        # remaining check is the budget bypass we're testing.
        from tests.vlm.proposals.agents.conftest import (
            FIXTURE_SLICES_BY_CLUSTER,
            evidence_for_cluster,
        )
        ctx.inspected_nodes.add(1)
        for sid in FIXTURE_SLICES_BY_CLUSTER[1]:
            ctx.inspected_slices.add(sid)

        async def _run():
            # Burn the entire budget on exploration.
            r1 = await call_tool("get_graph_summary", {})
            # Now budget is exhausted. Confirm a non-bypass tool gets rejected.
            r2 = await call_tool("list_nodes", {})
            # And confirm propose_collection_request still runs.
            r3 = await call_tool(
                "propose_collection_request",
                {
                    "request_type": "full_trajectory",
                    "initial_conditions": {"reference_rollout_id": "r0000", "reference_frame": 0},
                    "target_behavior": "approach the cube and grasp it firmly",
                    "prohibitions": [],
                    "success_criterion": "task_success",
                    "target_cluster": 1,
                    "evidence_slice_ids": evidence_for_cluster(1, n=3),
                    "reasoning": "exploration is exhausted; submitting the strongest hypothesis from inspection so far",
                },
            )
            return r1, r2, r3

        r1, r2, r3 = asyncio.run(_run())

        # 1. get_graph_summary ran.
        self.assertIn("get_graph_summary", call_log)

        # 2. list_nodes was rejected by the budget gate (NOT in call_log).
        self.assertNotIn("list_nodes", call_log)
        text2 = "\n".join(i.text for i in r2 if i.type == "text")
        self.assertIn("budget_exhausted", text2)

        # 3. propose_collection_request DID run despite exhausted budget,
        #    and the submission landed.
        self.assertIn("propose_collection_request", call_log)
        text3 = "\n".join(i.text for i in r3 if i.type == "text")
        self.assertIn('"ok": true', text3)
        self.assertEqual(len(ctx.submitted), 1)

    def test_cached_call_does_not_charge_budget(self):
        import asyncio

        call_tool, ctx, tools, call_log = self._build_handler(max_tool_calls=2)

        async def _run():
            await call_tool("get_graph_summary", {})  # budget = 1/2
            await call_tool("get_graph_summary", {})  # cache hit; budget stays 1/2
            await call_tool("list_nodes", {})         # budget = 2/2
            # Fourth call: budget exhausted. find_failure_nodes is budgeted
            # (an analysis tool, not a bypass lookup) so it should be rejected.
            return await call_tool("find_failure_nodes", {"min_failure_prob": 0.3})

        asyncio.run(_run())
        # get_graph_summary ran exactly once; the second hit was cached.
        # find_failure_nodes was rejected by the budget gate.
        self.assertEqual(
            call_log,
            ["get_graph_summary", "list_nodes"],
        )
        self.assertEqual(ctx.budget.state.n_tool_calls, 2)
        self.assertEqual(ctx.budget.state.n_cache_hits, 1)


class TestServerBuildContextAndTools(unittest.TestCase):
    """Confirm the build_context_and_tools function wires the registry correctly."""

    def setUp(self):
        _install_stub_mcp()
        self.tmp = Path(tempfile.mkdtemp())

    def test_a_g_registry_includes_layer_one_tools(self):
        # Mock the heavy loaders so we don't need real artefacts on disk.
        import numpy as np

        from tests.vlm.proposals.agents.conftest import (
            make_fixture_graph,
            make_fixture_labels_metadata,
            make_fixture_pool,
        )

        labels, metadata = make_fixture_labels_metadata()
        pool = make_fixture_pool(self.tmp)
        graph = make_fixture_graph()

        with patch(
            "policy_doctor.data.clustering_loader.load_clustering_result_from_path",
            return_value=(labels, metadata, {"level": "rollout"}),
        ), patch(
            "policy_doctor.vlm.proposals.pool.RolloutPool.from_episodes_dir",
            return_value=pool,
        ), patch(
            "policy_doctor.behaviors.behavior_graph.BehaviorGraph.from_cluster_assignments",
            return_value=graph,
        ):
            from policy_doctor.vlm.proposals.agents.mcp_server.server import (
                build_context_and_tools,
            )

            ctx, tools = build_context_and_tools({
                "clustering_dir": self.tmp / "clust",
                "pool_episodes_dir": self.tmp / "pool",
                "condition": "A_G",
                "task_hint": "test task",
                "max_tool_calls": 100,
                "max_visual_calls": 100,
                "max_video_calls": 100,
                "kin_strategy": "cluster_stats",
            })

        self.assertEqual(ctx.condition, "A_G")
        # Layer 1 tools should be present.
        self.assertIn("get_graph_summary", tools)
        self.assertIn("list_nodes", tools)
        # Layer 4 tools should be present.
        self.assertIn("propose_collection_request", tools)
        self.assertIn("finalize_strategy", tools)


if __name__ == "__main__":
    unittest.main()

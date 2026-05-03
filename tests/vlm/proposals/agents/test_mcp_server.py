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

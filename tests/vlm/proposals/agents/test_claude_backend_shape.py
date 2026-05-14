"""Lightweight checks of the Claude tool-use response parser.

Does not call the real API — uses a stubbed ``client.messages.create`` so we
verify the (response → AssistantTurn) translation without network or keys.
The real-API smoke test lives at the integration tier.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestClaudeResponseParsing(unittest.TestCase):
    def _make_block(self, type_, **fields):
        b = MagicMock()
        b.type = type_
        for k, v in fields.items():
            setattr(b, k, v)
        return b

    def _make_response(self, content_blocks, stop_reason="tool_use",
                       input_tokens=10, output_tokens=20):
        resp = MagicMock()
        resp.content = content_blocks
        resp.stop_reason = stop_reason
        resp.usage.input_tokens = input_tokens
        resp.usage.output_tokens = output_tokens
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        return resp

    def test_parse_text_only_response(self):
        # Skip if anthropic isn't installed (CI environments).
        try:
            from policy_doctor.vlm.backends.claude import _response_to_assistant_turn
        except ImportError:
            self.skipTest("anthropic not installed in this env")

        resp = self._make_response(
            content_blocks=[self._make_block("text", text="hello")],
            stop_reason="end_turn",
        )
        turn = _response_to_assistant_turn(resp)
        self.assertEqual(turn.text, "hello")
        self.assertFalse(turn.has_tool_calls)
        self.assertEqual(turn.stop_reason, "end_turn")

    def test_parse_tool_use_response(self):
        try:
            from policy_doctor.vlm.backends.claude import _response_to_assistant_turn
        except ImportError:
            self.skipTest("anthropic not installed in this env")

        resp = self._make_response(
            content_blocks=[
                self._make_block("text", text="let me check"),
                self._make_block(
                    "tool_use",
                    id="toolu_abc",
                    name="get_graph_summary",
                    input={"foo": "bar"},
                ),
            ],
            stop_reason="tool_use",
        )
        turn = _response_to_assistant_turn(resp)
        self.assertEqual(turn.text, "let me check")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "get_graph_summary")
        self.assertEqual(turn.tool_calls[0].arguments, {"foo": "bar"})
        self.assertEqual(turn.tool_calls[0].id, "toolu_abc")

    def test_image_fallback_extractor(self):
        try:
            from policy_doctor.vlm.backends.claude import (
                _move_images_to_sibling_user_message,
            )
        except ImportError:
            self.skipTest("anthropic not installed in this env")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [
                            {"type": "text", "text": "summary"},
                            {"type": "image", "source": {"type": "base64", "data": "..."}},
                        ],
                    }
                ],
            }
        ]
        fixed = _move_images_to_sibling_user_message(messages)
        # Original tool_result keeps the text but loses the image.
        first_inner = fixed[0]["content"][0]["content"]
        self.assertEqual(len(first_inner), 1)
        self.assertEqual(first_inner[0]["type"], "text")
        # Sibling user message contains the image plus an attribution stub.
        self.assertEqual(fixed[1]["role"], "user")
        types = [blk["type"] for blk in fixed[1]["content"]]
        self.assertIn("image", types)


if __name__ == "__main__":
    unittest.main()

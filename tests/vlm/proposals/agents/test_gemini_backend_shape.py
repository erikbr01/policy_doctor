"""Lightweight shape checks for the Gemini tool-use backend.

No real API calls. Verifies:
* messages → Gemini Content translation, including image hoist
* tools → FunctionDeclaration translation + schema sanitization
* response → AssistantTurn parsing
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock


class TestGeminiTranslation(unittest.TestCase):
    """Pure-Python translation helpers, no SDK imports needed."""

    def test_text_message_translation(self):
        from policy_doctor.vlm.backends.gemini import _messages_to_gemini_contents

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi back"}]},
        ]
        contents, hoisted = _messages_to_gemini_contents(msgs)
        self.assertEqual(len(contents), 2)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"][0]["text"], "hello")
        # Anthropic 'assistant' → Gemini 'model'.
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(hoisted, [])

    def test_tool_use_then_tool_result_split_correctly(self):
        from policy_doctor.vlm.backends.gemini import _messages_to_gemini_contents

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "do it"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "get_graph_summary", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": "summary text"}],
                    }
                ],
            },
        ]
        contents, hoisted = _messages_to_gemini_contents(msgs)
        # Expect: user(text), model(function_call), function(function_response).
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[1]["role"], "model")
        self.assertIn("function_call", contents[1]["parts"][0])
        self.assertEqual(contents[2]["role"], "function")
        self.assertIn("function_response", contents[2]["parts"][0])
        # Function name must be looked up correctly from the tool_use.
        self.assertEqual(
            contents[2]["parts"][0]["function_response"]["name"], "get_graph_summary"
        )
        self.assertEqual(hoisted, [])

    def test_image_in_tool_result_hoisted_to_sibling_user_message(self):
        import base64

        from policy_doctor.vlm.backends.gemini import _messages_to_gemini_contents

        png_b64 = base64.b64encode(b"fake-jpeg-bytes").decode("ascii")
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "get_slice_video", "input": {"slice_id": "r0001_t0_t10"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": [
                            {"type": "text", "text": "see attached"},
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": png_b64},
                            },
                        ],
                    }
                ],
            },
        ]
        contents, hoisted = _messages_to_gemini_contents(msgs)
        # Expect: model(function_call), function(function_response), user(image).
        self.assertEqual(contents[0]["role"], "model")
        self.assertEqual(contents[1]["role"], "function")
        self.assertEqual(contents[2]["role"], "user")
        # Hoisted message contains the image.
        parts_types = ["inline_data" in p or "text" in p for p in contents[2]["parts"]]
        self.assertTrue(all(parts_types))
        # The function_response no longer carries the image.
        fr = contents[1]["parts"][0]["function_response"]
        self.assertEqual(fr["name"], "get_slice_video")
        self.assertEqual(len(hoisted), 1)

    def test_tool_translation_renames_input_schema_to_parameters(self):
        from policy_doctor.vlm.backends.gemini import _tools_to_function_declarations

        tools = [
            {
                "name": "get_node",
                "description": "...",
                "input_schema": {
                    "type": "object",
                    "properties": {"node_id": {"type": "integer"}},
                    "required": ["node_id"],
                    "additionalProperties": False,  # must be stripped
                },
            }
        ]
        decls = _tools_to_function_declarations(tools)
        self.assertEqual(decls[0]["name"], "get_node")
        self.assertIn("parameters", decls[0])
        self.assertNotIn("additionalProperties", decls[0]["parameters"])
        self.assertEqual(decls[0]["parameters"]["required"], ["node_id"])

    def test_schema_sanitizer_strips_recursively(self):
        from policy_doctor.vlm.backends.gemini import _sanitize_schema_for_gemini

        sch = {
            "type": "object",
            "additionalProperties": False,
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {
                "x": {"type": "object", "additionalProperties": False, "properties": {}},
            },
        }
        cleaned = _sanitize_schema_for_gemini(sch)
        self.assertNotIn("additionalProperties", cleaned)
        self.assertNotIn("$schema", cleaned)
        self.assertNotIn("additionalProperties", cleaned["properties"]["x"])


class TestGeminiResponseParsing(unittest.TestCase):
    """Parse a stubbed GenerateContentResponse into AssistantTurn."""

    def _make_part(self, text=None, function_call=None):
        p = MagicMock()
        p.text = text
        p.function_call = function_call
        return p

    def _make_function_call(self, name, args):
        fc = MagicMock()
        fc.name = name
        fc.args = args
        return fc

    def _make_response(self, parts, finish_name="STOP", input_tokens=10, output_tokens=20):
        candidate = MagicMock()
        candidate.content.parts = parts
        # Mimic the .name accessor on the enum-ish finish_reason.
        finish = MagicMock()
        finish.name = finish_name
        candidate.finish_reason = finish
        resp = MagicMock()
        resp.candidates = [candidate]
        resp.usage_metadata.prompt_token_count = input_tokens
        resp.usage_metadata.candidates_token_count = output_tokens
        resp.usage_metadata.cached_content_token_count = 0
        return resp

    def test_text_only_response(self):
        from policy_doctor.vlm.backends.gemini import _gemini_response_to_assistant_turn

        resp = self._make_response([self._make_part(text="hello there")], finish_name="STOP")
        turn = _gemini_response_to_assistant_turn(resp)
        self.assertEqual(turn.text, "hello there")
        self.assertFalse(turn.has_tool_calls)
        self.assertEqual(turn.stop_reason, "end_turn")
        self.assertEqual(turn.usage.input_tokens, 10)
        self.assertEqual(turn.usage.output_tokens, 20)

    def test_tool_use_response(self):
        from policy_doctor.vlm.backends.gemini import _gemini_response_to_assistant_turn

        fc = self._make_function_call("get_graph_summary", {"foo": "bar"})
        resp = self._make_response([
            self._make_part(text="let me check"),
            self._make_part(function_call=fc),
        ])
        turn = _gemini_response_to_assistant_turn(resp)
        self.assertEqual(turn.text, "let me check")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "get_graph_summary")
        self.assertEqual(turn.tool_calls[0].arguments, {"foo": "bar"})
        # Synthesized id is non-empty.
        self.assertTrue(turn.tool_calls[0].id.startswith("gem_"))
        self.assertEqual(turn.stop_reason, "tool_use")

    def test_max_tokens_finish_reason_maps(self):
        from policy_doctor.vlm.backends.gemini import _gemini_response_to_assistant_turn

        resp = self._make_response([self._make_part(text="...")], finish_name="MAX_TOKENS")
        turn = _gemini_response_to_assistant_turn(resp)
        self.assertEqual(turn.stop_reason, "max_tokens")


if __name__ == "__main__":
    unittest.main()

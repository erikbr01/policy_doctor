"""Day-one smoke test: Gemini tool-use with one storyboard image.

Why: Gemini's FunctionResponse channel is JSON-only — images can't ride inside
it. The Gemini backend hoists images out of the tool_result and into a
sibling user message. This script verifies the whole roundtrip works against
the real API on day one, before Tier 1 burns a real run.

Usage::

    export GOOGLE_API_KEY=...
    python scripts/check_gemini_tool_image.py
    # Or with your own storyboard image:
    python scripts/check_gemini_tool_image.py /path/to/storyboard.png

Exit code 0 = round-trip succeeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "third_party" / "cupid"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_or_synthesize(path: Path | None) -> Image.Image:
    if path is not None and path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (256, 256), color=(80, 80, 90))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storyboard", nargs="?", type=Path, default=None,
                        help="Path to a storyboard PNG. Synthesized if omitted.")
    parser.add_argument("--model", default="gemini-2.0-flash")
    args = parser.parse_args()

    storyboard = _load_or_synthesize(args.storyboard)

    from policy_doctor.vlm.backends.gemini import GeminiVLMBackend
    from policy_doctor.vlm.proposals.agents.session import _content_to_message_blocks
    from policy_doctor.vlm.proposals.agents.tools.types import (
        ImageBlock,
        TextBlock,
        ToolResult,
    )

    backend = GeminiVLMBackend(model_name=args.model, max_output_tokens=1024, temperature=0.0)

    tools = [
        {
            "name": "get_graph_summary",
            "description": "Returns a short text summary plus one preview storyboard image.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "finalize_strategy",
            "description": "End the session with a brief rationale.",
            "input_schema": {
                "type": "object",
                "properties": {"rationale": {"type": "string"}},
                "required": ["rationale"],
            },
        },
    ]

    msgs_initial = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Call get_graph_summary, then finalize_strategy."}],
        }
    ]
    turn1 = backend.chat_with_tools(
        messages=msgs_initial,
        tools=tools,
        system="You are an exploration agent. Use the tools.",
        max_tokens=512,
    )
    if not turn1.has_tool_calls:
        print(f"[FAIL] turn 1 did not produce tool calls; stop_reason={turn1.stop_reason}")
        return 1
    tc = turn1.tool_calls[0]
    print(f"[OK] turn 1 produced function_call: name={tc.name} id={tc.id[:12]}...")

    fake_result = ToolResult(
        name="get_graph_summary",
        ok=True,
        content=[
            TextBlock(text='{"n_cluster_nodes": 5, "n_paths_to_failure": 7}'),
            ImageBlock(image=storyboard, caption="overview of cluster c1"),
        ],
    )
    tool_result_blocks = _content_to_message_blocks(fake_result)

    msgs_with_image = msgs_initial + [
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": tool_result_blocks,
                "is_error": False,
            }],
        },
    ]
    turn2 = backend.chat_with_tools(
        messages=msgs_with_image,
        tools=tools,
        system="You are an exploration agent. Use the tools.",
        max_tokens=512,
    )
    print(
        f"[OK] turn 2 returned: stop_reason={turn2.stop_reason} "
        f"text={'<set>' if turn2.text else '<empty>'} "
        f"tool_calls={[t.name for t in turn2.tool_calls]}"
    )

    if turn2.stop_reason in {"tool_use", "end_turn"}:
        print("[PASS] image-in-tool_result (with sibling-message hoist) round-trip succeeded.")
        return 0
    print(f"[FAIL] unexpected stop_reason {turn2.stop_reason!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

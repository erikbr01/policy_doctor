"""Day-one smoke test: Claude tool-use with one storyboard image in the tool_result.

Why: Anthropic's API supports image content blocks inside tool_result, but the
exact shape is easy to get wrong, and "image-in-tool-result" is the dominant
cost driver of the agentic loop. This script makes one round-trip to confirm
the format works end-to-end *before* running Tier 1 — where a failure would
show up only after a long, expensive run.

Usage::

    export ANTHROPIC_API_KEY=...
    python scripts/check_claude_tool_image.py
    # (optional) supply your own storyboard image:
    python scripts/check_claude_tool_image.py /path/to/storyboard.png

Exit code 0 = round-trip succeeded; tool_result containing the image was
accepted, the agent's reply was parsed, and at least one tool_use was emitted.
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
    # Synthesize a small 4-frame storyboard so the test works without any data.
    img = Image.new("RGB", (256, 256), color=(80, 80, 90))
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storyboard", nargs="?", type=Path, default=None,
                        help="Path to a storyboard PNG. Synthesized if omitted.")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    storyboard = _load_or_synthesize(args.storyboard)

    from policy_doctor.vlm.backends.claude import ClaudeVLMBackend
    from policy_doctor.vlm.proposals.agents.session import (
        _content_to_message_blocks,
        _image_block_for_message,
    )
    from policy_doctor.vlm.proposals.agents.tools.types import (
        ImageBlock,
        TextBlock,
        ToolResult,
    )

    backend = ClaudeVLMBackend(model_name=args.model, max_tokens=1024, temperature=0.0)

    # ---- Build the conversation -------------------------------------
    # Two-turn structure: user asks for graph summary; tool returns text + image.
    tools = [
        {
            "name": "get_graph_summary",
            "description": "Returns a short text summary plus one preview storyboard image.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
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

    # First turn: model should call get_graph_summary.
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
    print(f"[OK] turn 1 produced tool_use: name={tc.name} id={tc.id[:12]}...")

    # Second turn: feed the tool_result with image embedded.
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
            "content": [
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": tool_result_blocks,
                    "is_error": False,
                }
            ],
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
        print("[PASS] image-in-tool_result round-trip succeeded.")
        return 0
    print(f"[FAIL] unexpected stop_reason {turn2.stop_reason!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

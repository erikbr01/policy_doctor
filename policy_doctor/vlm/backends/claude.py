"""Anthropic Claude VLM backend (optional dependency: anthropic)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from policy_doctor.vlm.backends.base import (
    AssistantTurn,
    TokenUsage,
    ToolCall,
    VLMBackend,
)

_INSTALL_HINT = (
    "Install with:  pip install anthropic\n"
    "Then set the ANTHROPIC_API_KEY environment variable (or pass api_key in backend_params)."
)


def _require_anthropic():
    try:
        import anthropic  # noqa: F401
        return anthropic
    except ImportError as exc:
        raise ImportError(
            f"anthropic is required for the Claude backend. {_INSTALL_HINT}"
        ) from exc


def _pil_to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class ClaudeVLMBackend(VLMBackend):
    """Anthropic Claude backend.

    Backend params (all optional):
      - ``model_name``: Claude model ID (default ``claude-sonnet-4-6``).
      - ``api_key``: Anthropic API key; falls back to ``ANTHROPIC_API_KEY`` env var.
      - ``max_tokens``: generation budget (default 1024).
      - ``temperature``: sampling temperature (default 0.2).
    """

    name = "claude"

    def __init__(
        self,
        *,
        model_name: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        anthropic = _require_anthropic()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _image_block(self, img: Image.Image) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _pil_to_b64_jpeg(img),
            },
        }

    def _call(self, system: Optional[str], user_content: list) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": user_content}],
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        if response.content and hasattr(response.content[0], "text"):
            return response.content[0].text or ""
        return ""

    # ------------------------------------------------------------------
    # VLMBackend interface
    # ------------------------------------------------------------------

    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        content: list = []
        for img in images:
            content.append(self._image_block(img))
        content.append({"type": "text", "text": user_prompt})
        return self._call(system_prompt, content)

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        content = [{"type": "text", "text": user_prompt}]
        return self._call(system_prompt, content)

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        content = [{"type": "text", "text": user_prompt}]
        return self._call(system_prompt, content)

    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
        query_extra_text: Optional[str] = None,
        example_extra_texts: Optional[
            Sequence[Optional[Sequence[Optional[str]]]]
        ] = None,
    ) -> str:
        """Classify query_images into one of the example groups."""
        content: list = []
        if user_preamble:
            content.append({"type": "text", "text": user_preamble + "\n\n"})
        for ci, (label, imgs) in enumerate(example_sets):
            content.append({"type": "text", "text": f"{label}:\n"})
            extras_for_group = (
                example_extra_texts[ci]
                if example_extra_texts is not None and ci < len(example_extra_texts)
                else None
            )
            for j, img in enumerate(imgs):
                content.append(self._image_block(img))
                if extras_for_group is not None and j < len(extras_for_group):
                    extra = extras_for_group[j]
                    if extra:
                        content.append({"type": "text", "text": "\n" + extra + "\n"})
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": "Query:\n"})
        for img in query_images:
            content.append(self._image_block(img))
        if query_extra_text:
            content.append({"type": "text", "text": "\n" + query_extra_text})
        content.append({"type": "text", "text": "\n" + user_prompt})
        return self._call(system_prompt, content)


    # ------------------------------------------------------------------
    # Tool-use primitive (used by the agentic proposal loop)
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        *,
        messages,
        tools,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> AssistantTurn:
        """Anthropic tool-use API call.

        ``messages`` and ``tools`` are passed through largely as-is — the
        session loop already produces them in Anthropic's content-block
        format (see :mod:`policy_doctor.vlm.proposals.agents.session`).

        Tool results may include images: each image is encoded as
        ``{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": ...}}``
        inside a ``tool_result`` block. The Anthropic API supports this
        natively; if a particular model rejects it we fall back to inlining
        the image in a sibling user-turn message keyed by tool_use_id.
        """
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "tools": tools,
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            # Conservative fallback: try a sibling-message remediation when an
            # image-in-tool-result is the likely culprit.
            if _looks_like_image_tool_result_failure(exc):
                fixed = _move_images_to_sibling_user_message(messages)
                kwargs["messages"] = fixed
                response = self._client.messages.create(**kwargs)
            else:
                raise

        return _response_to_assistant_turn(response)


def _response_to_assistant_turn(response: Any) -> AssistantTurn:
    """Parse Anthropic ``Message`` response into an :class:`AssistantTurn`."""
    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_chunks.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    arguments=dict(getattr(block, "input", {}) or {}),
                )
            )
        # Other block types (e.g. thinking) are dropped from the loop view but
        # remain in `raw` for trace inspection.

    usage_obj = getattr(response, "usage", None)
    usage = TokenUsage(
        input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(
            getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        ),
    )

    return AssistantTurn(
        text=("\n".join(text_chunks) or None),
        tool_calls=tool_calls,
        stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
        usage=usage,
        raw=response,
    )


def _looks_like_image_tool_result_failure(exc: Exception) -> bool:
    """Heuristic: detect 'image in tool_result' rejections to trigger fallback."""
    msg = str(exc).lower()
    return ("tool_result" in msg and "image" in msg) or "image" in msg and "media_type" in msg


def _move_images_to_sibling_user_message(messages: list) -> list:
    """Strip image blocks out of every tool_result and re-attach them as a
    plain user-turn message immediately after the tool_result, keyed by
    tool_use_id in a leading text block.
    """
    out: list = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        new_content: list = []
        sibling_blocks: list = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "tool_result":
                tr_inner = blk.get("content") or []
                if not isinstance(tr_inner, list):
                    new_content.append(blk)
                    continue
                kept_inner: list = []
                for sub in tr_inner:
                    if isinstance(sub, dict) and sub.get("type") == "image":
                        sibling_blocks.append({
                            "type": "text",
                            "text": f"[image attached to tool_use_id={blk.get('tool_use_id')}]",
                        })
                        sibling_blocks.append(sub)
                    else:
                        kept_inner.append(sub)
                if not kept_inner:
                    kept_inner = [{"type": "text", "text": "(see attached image below)"}]
                new_content.append({**blk, "content": kept_inner})
            else:
                new_content.append(blk)
        out.append({**m, "content": new_content})
        if sibling_blocks:
            out.append({"role": "user", "content": sibling_blocks})
    return out


def build_claude_backend(params: Optional[Dict[str, Any]]) -> ClaudeVLMBackend:
    """Factory function registered in the VLM registry."""
    p = dict(params or {})
    return ClaudeVLMBackend(
        model_name=p.get("model_name", "claude-sonnet-4-6"),
        api_key=p.get("api_key"),
        max_tokens=int(p.get("max_tokens", 1024)),
        temperature=float(p.get("temperature", 0.2)),
    )

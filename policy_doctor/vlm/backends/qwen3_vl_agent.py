"""Qwen3-VL agent backend — adds ``chat_with_tools`` to the existing Qwen
multimodal backend.

The base :class:`policy_doctor.vlm.backends.qwen2_vl.Qwen2VLBackend` covers
``describe_slice`` and ``classify_slice`` (the one-shot E1 paths). This
subclass adds the agentic E2 surface so a local Qwen3-VL checkpoint can
drive the tool-use loop without an API key.

Hermes-style tool calling: the model emits ``<tool_call>{json}</tool_call>``
blocks, the chat template injects the tool list into the system prompt
when given ``tools=[...]``. Verified against ``Qwen/Qwen3-VL-8B-Instruct``
and ``Qwen/Qwen3-VL-32B-Instruct`` (4-bit, multi-GPU via accelerate).
"""

from __future__ import annotations

import base64
import io
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from policy_doctor.vlm.backends.base import (
    AssistantTurn,
    TokenUsage,
    ToolCall,
)
from policy_doctor.vlm.backends.qwen2_vl import Qwen2VLBackend


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _tool_to_qwen(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Provider-neutral tool dict -> Qwen ``{"type":"function","function":...}``."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {
                "type": "object", "properties": {},
            },
        },
    }


def _extract_image_from_block(blk: Dict[str, Any]) -> Optional[Image.Image]:
    """Decode an Anthropic-shaped image block into a PIL.Image.

    Anthropic ships images as
      ``{"type": "image", "source": {"type": "base64", "media_type": ..., "data": <b64>}}``.
    Convenience: also accept a direct PIL handle under ``{"image": <PIL>}``.
    """
    src = blk.get("source")
    if isinstance(src, dict) and src.get("type") == "base64":
        data = base64.b64decode(src.get("data", ""))
        return Image.open(io.BytesIO(data)).convert("RGB")
    img = blk.get("image")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return None


def convert_messages_for_qwen(
    messages: List[Dict[str, Any]],
    system: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """Anthropic-shaped messages -> Qwen chat-template messages.

    Returns ``(qwen_messages, ordered_images)`` where the image list is
    flattened in the order the chat template will consume them.

    Mapping:
      * ``user`` text/image blocks → ``user`` content list (passes through).
      * ``assistant`` ``tool_use`` blocks → ``assistant`` text containing
        Hermes-style ``<tool_call>{json}</tool_call>`` blocks.
      * ``tool_result`` blocks → ``role=tool`` text messages (Qwen's tool
        role does not accept images, so any inline images are promoted into
        a follow-up ``user`` content block; the chat template emits
        ``<|image_pad|>`` placeholders for them in order).
    """
    out: List[Dict[str, Any]] = []
    all_images: List[Image.Image] = []

    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            parts: List[str] = []
            for blk in content:
                btype = blk.get("type")
                if btype == "text" and blk.get("text"):
                    parts.append(blk["text"])
                elif btype == "tool_use":
                    tc_json = json.dumps(
                        {"name": blk.get("name", ""),
                         "arguments": blk.get("input") or {}},
                        ensure_ascii=False,
                    )
                    parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
            out.append({
                "role": "assistant",
                "content": "\n".join(parts) if parts else "",
            })
            continue

        # role == "user" — may carry tool_results, text, or images.
        user_blocks: List[Dict[str, Any]] = []
        for blk in content:
            btype = blk.get("type")
            if btype == "tool_result":
                inner = blk.get("content") or []
                text_parts: List[str] = []
                inner_images: List[Image.Image] = []
                for ib in inner:
                    if ib.get("type") == "text":
                        txt = ib.get("text", "")
                        if txt:
                            text_parts.append(txt)
                    elif ib.get("type") == "image":
                        img = _extract_image_from_block(ib)
                        if img is not None:
                            inner_images.append(img)
                text_combined = "\n".join(text_parts).strip()
                if blk.get("is_error"):
                    text_combined = (
                        f"[error]\n{text_combined}" if text_combined else "[error]"
                    )
                if not text_combined and not inner_images:
                    text_combined = "(empty)"
                out.append({
                    "role": "tool",
                    "content": text_combined or "(see attached image)",
                })
                if inner_images:
                    img_blocks = [
                        {"type": "image", "image": img} for img in inner_images
                    ]
                    out.append({"role": "user", "content": img_blocks})
                    all_images.extend(inner_images)
            elif btype == "text":
                user_blocks.append({"type": "text", "text": blk.get("text", "")})
            elif btype == "image":
                img = _extract_image_from_block(blk)
                if img is not None:
                    user_blocks.append({"type": "image", "image": img})
                    all_images.append(img)
        if user_blocks:
            out.append({"role": "user", "content": user_blocks})

    return out, all_images


def parse_qwen_assistant(decoded: str) -> Tuple[Optional[str], List[ToolCall]]:
    """Pull ``<tool_call>{...}</tool_call>`` blocks out of *decoded*.

    Returns ``(text_remainder, tool_calls)``. Strips chat-template markers
    some decoders leave when special tokens are not skipped. Malformed JSON
    inside a tool_call falls through as plain text rather than killing the
    turn.
    """
    cleaned = decoded
    for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        cleaned = cleaned.replace(tok, "")

    tool_calls: List[ToolCall] = []
    parts: List[str] = []
    last_end = 0
    for m in _TOOL_CALL_RE.finditer(cleaned):
        parts.append(cleaned[last_end:m.start()])
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments")
            if args is None:
                args = obj.get("parameters") or {}
            tool_calls.append(ToolCall(
                id=f"qwen_{uuid.uuid4().hex[:12]}",
                name=str(obj.get("name", "")),
                arguments=dict(args) if isinstance(args, dict) else {},
            ))
        except (json.JSONDecodeError, TypeError, ValueError):
            parts.append(m.group(0))
        last_end = m.end()
    parts.append(cleaned[last_end:])
    text = "".join(parts).strip()
    return (text or None, tool_calls)


class Qwen3VLAgentBackend(Qwen2VLBackend):
    """Qwen2VLBackend plus ``chat_with_tools`` for the agentic loop."""

    name = "qwen3_vl_agent"

    def chat_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> AssistantTurn:
        """One agentic turn against a Qwen *-VL checkpoint."""
        import torch

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        qwen_msgs, all_images = convert_messages_for_qwen(messages, system)
        qwen_tools = [_tool_to_qwen(t) for t in tools]

        text = self._processor.apply_chat_template(
            qwen_msgs,
            tools=qwen_tools or None,
            add_generation_prompt=True,
            tokenize=False,
        )

        proc_extra: Dict[str, Any] = {}
        if self.image_max_pixels is not None or self.image_min_pixels is not None:
            ik: Dict[str, Any] = {}
            if self.image_min_pixels is not None:
                ik["min_pixels"] = int(self.image_min_pixels)
            if self.image_max_pixels is not None:
                ik["max_pixels"] = int(self.image_max_pixels)
            proc_extra["images_kwargs"] = ik

        inputs = self._processor(
            text=[text],
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
            **proc_extra,
        )

        target_dev = (
            next(self._model.parameters()).device
            if self.device_map is not None
            else self.device
        )
        inputs = inputs.to(target_dev)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_tokens)}
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False
        if seed is not None:
            torch.manual_seed(int(seed))

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        in_len = int(inputs["input_ids"].shape[1])
        trimmed = out_ids[:, in_len:]
        # skip_special_tokens=False so <tool_call> XML tags survive batch_decode.
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=False)[0]
        text_out, tool_calls = parse_qwen_assistant(decoded)

        return AssistantTurn(
            text=text_out,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            usage=TokenUsage(
                input_tokens=in_len,
                output_tokens=int(trimmed.shape[1]),
            ),
            raw=decoded,
        )


def build_qwen3_vl_agent_backend(
    params: Optional[Dict[str, Any]] = None,
) -> Qwen3VLAgentBackend:
    return Qwen3VLAgentBackend(**(params or {}))

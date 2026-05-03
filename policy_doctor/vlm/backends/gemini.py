"""Google Gemini VLM backend (optional dependency: google-generativeai)."""

from __future__ import annotations

import base64
import copy
import io
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from policy_doctor.vlm.backends.base import (
    AssistantTurn,
    TokenUsage,
    ToolCall,
    VLMBackend,
)

_INSTALL_HINT = (
    "Install with:  pip install google-generativeai\n"
    "Then set the GOOGLE_API_KEY environment variable (or pass api_key in backend_params)."
)


def _require_genai():
    try:
        import google.generativeai as genai  # noqa: F401

        return genai
    except ImportError as exc:
        raise ImportError(
            f"google-generativeai is required for the Gemini backend. {_INSTALL_HINT}"
        ) from exc


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class GeminiVLMBackend(VLMBackend):
    """Gemini backend via the google-generativeai SDK.

    Backend params (all optional):
      - ``model_name``: Gemini model ID (default ``gemini-2.0-flash``).
      - ``api_key``: Gemini API key; falls back to ``GOOGLE_API_KEY`` env var.
      - ``max_output_tokens``: generation budget (default 1024).
      - ``temperature``: sampling temperature (default 0.2).
    """

    name = "gemini"

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        genai = _require_genai()
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if key:
            genai.configure(api_key=key)
        self._genai = genai
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)
        self._gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

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
        """Describe a sequence of robot-rollout frames."""
        parts: list = []
        if system_prompt:
            parts.append(system_prompt + "\n\n")
        for img in images:
            parts.append({"mime_type": "image/jpeg", "data": _pil_to_bytes(img)})
        parts.append(user_prompt)
        response = self._model.generate_content(parts, generation_config=self._gen_config)
        return response.text or ""

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Summarize per-slice captions for one behavior cluster (text-only)."""
        parts: list = []
        if system_prompt:
            parts.append(system_prompt + "\n\n")
        parts.append(user_prompt)
        response = self._model.generate_content(parts, generation_config=self._gen_config)
        return response.text or ""

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Judge whether per-slice captions are mutually coherent (text-only)."""
        parts: list = []
        if system_prompt:
            parts.append(system_prompt + "\n\n")
        parts.append(user_prompt)
        response = self._model.generate_content(parts, generation_config=self._gen_config)
        return response.text or ""


    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
    ) -> str:
        """Classify query_images into one of the example groups."""
        parts: list = []
        if system_prompt:
            parts.append(system_prompt + "\n\n")
        if user_preamble:
            parts.append(user_preamble + "\n\n")
        for label, imgs in example_sets:
            parts.append(f"{label}:\n")
            for img in imgs:
                parts.append({"mime_type": "image/jpeg", "data": _pil_to_bytes(img)})
            parts.append("\n")
        parts.append("Query:\n")
        for img in query_images:
            parts.append({"mime_type": "image/jpeg", "data": _pil_to_bytes(img)})
        parts.append("\n" + user_prompt)
        response = self._model.generate_content(parts, generation_config=self._gen_config)
        return response.text or ""


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
        """Gemini tool-use call via google-generativeai.

        Translation strategy:

        * messages: Anthropic-shape (text/image/tool_use/tool_result content
          blocks) → Gemini ``Content(role, parts=[...])``.
        * tools: ``{name, description, input_schema}`` → Gemini
          ``FunctionDeclaration(name, description, parameters)``.
        * system: passed via ``system_instruction`` on a per-call model
          rebuild (Gemini's GenerativeModel takes it at instantiation).
        * Images inside tool_result: Gemini's ``FunctionResponse`` is JSON
          only — images get hoisted into a sibling user-turn message keyed
          by tool_use_id (same idea as the Claude image fallback).
        """
        contents, hoisted_user_msgs = _messages_to_gemini_contents(messages)
        # Hoisted user messages must follow their parent function_response in order.
        # _messages_to_gemini_contents already inserts them in place; nothing to do here.

        function_decls = _tools_to_function_declarations(tools)
        gemini_tools = [self._genai.types.Tool(function_declarations=function_decls)] if function_decls else None

        # Gemini takes system_instruction on the model object; rebuild per-call
        # to keep the call stateless.
        model = self._genai.GenerativeModel(
            self._model_name,
            system_instruction=system or None,
        )

        gen_cfg = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = model.generate_content(
            contents,
            generation_config=gen_cfg,
            tools=gemini_tools,
        )
        return _gemini_response_to_assistant_turn(response)


# ---------------------------------------------------------------------------
# Translation helpers (module-level for testability)
# ---------------------------------------------------------------------------


def _messages_to_gemini_contents(messages: List[Dict[str, Any]]):
    """Anthropic content-block messages → list of Gemini-shaped Content dicts.

    Returns ``(contents, hoisted)`` where ``hoisted`` is a list of additional
    user messages (one per image found inside a tool_result) that have already
    been inserted into ``contents`` in the right position.
    """
    out: List[Dict[str, Any]] = []
    hoisted: List[Dict[str, Any]] = []
    for m in messages or []:
        role_in = m.get("role")
        # Anthropic 'assistant' → Gemini 'model'; tool_result blocks live in
        # a 'user' message in Anthropic shape, become 'function' in Gemini.
        content = m.get("content")
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content or "")}]

        # Split blocks: tool_results have role 'function' in Gemini and must
        # not mix with regular text/image parts in the same Content. We emit
        # one Content per (role, block_group).
        text_image_parts: List[Dict[str, Any]] = []
        tool_use_parts: List[Dict[str, Any]] = []
        for blk in content:
            t = blk.get("type")
            if t == "text":
                text_image_parts.append({"text": blk.get("text") or ""})
            elif t == "image":
                src = blk.get("source") or {}
                if src.get("type") == "base64":
                    raw = base64.b64decode(src.get("data") or "")
                    text_image_parts.append({
                        "inline_data": {
                            "mime_type": src.get("media_type", "image/jpeg"),
                            "data": raw,
                        }
                    })
            elif t == "tool_use":
                tool_use_parts.append({
                    "function_call": {
                        "name": blk.get("name") or "",
                        "args": dict(blk.get("input") or {}),
                    }
                })
            elif t == "tool_result":
                # Build a function response Content. Strip images out of the
                # content list and hoist them into a sibling user message.
                tu_id = blk.get("tool_use_id") or ""
                inner = blk.get("content") or []
                text_chunks: List[str] = []
                hoisted_image_parts: List[Dict[str, Any]] = []
                for sub in inner if isinstance(inner, list) else []:
                    st = sub.get("type") if isinstance(sub, dict) else None
                    if st == "text":
                        text_chunks.append(str(sub.get("text") or ""))
                    elif st == "image":
                        src = sub.get("source") or {}
                        if src.get("type") == "base64":
                            raw = base64.b64decode(src.get("data") or "")
                            hoisted_image_parts.append({
                                "inline_data": {
                                    "mime_type": src.get("media_type", "image/jpeg"),
                                    "data": raw,
                                }
                            })
                response_payload: Dict[str, Any] = {"text": "\n".join(text_chunks) or ""}
                if blk.get("is_error"):
                    response_payload["error"] = True

                # We need the function name; Anthropic carries it on the
                # tool_use turn, not on tool_result. Look back at the most
                # recent assistant turn for the matching id.
                fn_name = _lookup_tool_name_for_id(messages, tu_id) or "unknown_tool"

                # Emit the function response as its own Content (Gemini role 'function').
                out.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": fn_name,
                            "response": response_payload,
                        }
                    }],
                })
                if hoisted_image_parts:
                    sibling = {
                        "role": "user",
                        "parts": [
                            {"text": f"[image attached to tool_use_id={tu_id}]"},
                            *hoisted_image_parts,
                        ],
                    }
                    out.append(sibling)
                    hoisted.append(sibling)

        # Now flush text/image and tool_use parts as needed.
        if text_image_parts:
            gemini_role = "model" if role_in == "assistant" else "user"
            out.append({"role": gemini_role, "parts": text_image_parts})
        if tool_use_parts:
            # tool_use only legitimately appears on assistant turns.
            out.append({"role": "model", "parts": tool_use_parts})

    return out, hoisted


def _lookup_tool_name_for_id(messages: List[Dict[str, Any]], tool_use_id: str) -> Optional[str]:
    """Walk the message history backwards to find the tool_use that matches ``tool_use_id``."""
    if not tool_use_id:
        return None
    for m in reversed(messages or []):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if blk.get("type") == "tool_use" and blk.get("id") == tool_use_id:
                return blk.get("name")
    return None


def _tools_to_function_declarations(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic tool spec → Gemini FunctionDeclaration dict shape.

    Anthropic tools: ``{name, description, input_schema}``.
    Gemini wants:    ``{name, description, parameters}`` where parameters is
    JSON Schema with a few restrictions (no ``additionalProperties: false``
    on older API versions; no ``$schema`` etc.). We strip the known offenders
    defensively.
    """
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        params = _sanitize_schema_for_gemini(t.get("input_schema") or {})
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": params,
        })
    return out


_GEMINI_DROP_KEYS = {"additionalProperties", "$schema", "$id", "$defs", "definitions"}


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    """Recursively strip JSON Schema keys Gemini's function calling rejects."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_DROP_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(v) for v in schema]
    return schema


_GEMINI_FINISH_REASON_MAP = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "stop_sequence",
    "RECITATION": "stop_sequence",
    "OTHER": "end_turn",
    # Newer SDK names also surface as ints; the response object usually
    # gives a name() method we can use first.
}


def _gemini_response_to_assistant_turn(response: Any) -> AssistantTurn:
    """Parse Gemini ``GenerateContentResponse`` into an :class:`AssistantTurn`."""
    text_chunks: List[str] = []
    tool_calls: List[ToolCall] = []

    candidates = getattr(response, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    finish_name = "STOP"
    if candidate is not None:
        fr = getattr(candidate, "finish_reason", None)
        # Newer SDK exposes an enum with .name; older returns a string already.
        if hasattr(fr, "name"):
            finish_name = fr.name
        elif fr is not None:
            finish_name = str(fr)

        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            fn_call = getattr(part, "function_call", None)
            if text:
                text_chunks.append(text)
            if fn_call is not None and getattr(fn_call, "name", ""):
                # `fn_call.args` is a Struct in protobuf land; coerce to dict.
                args = fn_call.args
                if hasattr(args, "items"):
                    args = {k: _to_python(v) for k, v in args.items()}
                else:
                    try:
                        args = dict(args) if args is not None else {}
                    except Exception:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        id=f"gem_{uuid.uuid4().hex[:8]}",  # Gemini doesn't give tool ids
                        name=fn_call.name,
                        arguments=args,
                    )
                )

    stop_reason = (
        "tool_use"
        if tool_calls
        else _GEMINI_FINISH_REASON_MAP.get(finish_name, "end_turn")
    )

    usage_obj = getattr(response, "usage_metadata", None)
    usage = TokenUsage(
        input_tokens=int(getattr(usage_obj, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage_obj, "candidates_token_count", 0) or 0),
        cache_read_input_tokens=int(getattr(usage_obj, "cached_content_token_count", 0) or 0),
        cache_creation_input_tokens=0,
    )

    return AssistantTurn(
        text=("\n".join(text_chunks) or None),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
        raw=response,
    )


def _to_python(value: Any) -> Any:
    """Best-effort coercion of protobuf Struct/Value to plain Python types."""
    # Recursively unwrap nested Struct/Value/ListValue. For non-protobuf
    # inputs, this is a no-op.
    if hasattr(value, "items"):
        try:
            return {k: _to_python(v) for k, v in value.items()}
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def build_gemini_backend(params: Optional[Dict[str, Any]]) -> GeminiVLMBackend:
    """Factory function registered in the VLM registry."""
    p = dict(params or {})
    return GeminiVLMBackend(
        model_name=p.get("model_name", "gemini-2.0-flash"),
        api_key=p.get("api_key"),
        max_output_tokens=int(p.get("max_output_tokens", 1024)),
        temperature=float(p.get("temperature", 0.2)),
    )

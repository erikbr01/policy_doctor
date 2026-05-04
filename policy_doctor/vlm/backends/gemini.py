"""Google Gemini VLM backend (optional dependency: google-genai).

Uses the modern ``google.genai`` SDK (not the deprecated ``google.generativeai``)
because Gemini-3-class models require ``thought_signature`` to be echoed back
in tool-use replays — and only ``google.genai`` exposes that field on response
parts. The legacy SDK silently drops it, leading to ``InvalidArgument 400``
on the second turn.
"""

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
    "Install with:  pip install google-genai\n"
    "Then set the GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable, or "
    "pass api_key in backend_params."
)


def _require_genai():
    """Import the modern google.genai SDK (handles thought_signature natively)."""
    try:
        from google import genai  # type: ignore[import]
        from google.genai import types as genai_types  # type: ignore[import]

        return genai, genai_types
    except ImportError as exc:
        raise ImportError(
            f"google-genai is required for the Gemini backend. {_INSTALL_HINT}"
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
        genai, genai_types = _require_genai()
        key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not key:
            raise ValueError(
                "Gemini backend requires an API key (api_key param, "
                "GOOGLE_API_KEY env var, or GEMINI_API_KEY env var)."
            )
        self._client = genai.Client(api_key=key)
        self._types = genai_types
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Internal helper for the legacy VLMBackend methods.
    # ------------------------------------------------------------------

    def _generate_text(
        self,
        parts: List[Any],
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """One-shot text generation via the new google.genai client.

        ``parts`` is a list of strings and/or ``{"mime_type", "data"}`` image
        dicts (the legacy interface). They get converted to ``types.Part``
        objects expected by the new SDK.
        """
        contents: List[Any] = []
        for p in parts:
            if isinstance(p, str):
                contents.append(self._types.Part.from_text(text=p))
            elif isinstance(p, dict) and "mime_type" in p and "data" in p:
                contents.append(self._types.Part.from_bytes(
                    data=p["data"], mime_type=p["mime_type"],
                ))
            else:
                contents.append(p)
        cfg = self._types.GenerateContentConfig(
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            system_instruction=system_prompt or None,
        )
        response = self._client.models.generate_content(
            model=self._model_name, contents=contents, config=cfg,
        )
        return response.text or ""

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
        for img in images:
            parts.append({"mime_type": "image/jpeg", "data": _pil_to_bytes(img)})
        parts.append(user_prompt)
        return self._generate_text(parts, system_prompt=system_prompt)

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        return self._generate_text([user_prompt], system_prompt=system_prompt)

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        return self._generate_text([user_prompt], system_prompt=system_prompt)

    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
    ) -> str:
        parts: list = []
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
        return self._generate_text(parts, system_prompt=system_prompt)


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
        """Gemini tool-use call via google.genai.

        Translation strategy:

        * messages: Anthropic-shape (text/image/tool_use/tool_result content
          blocks) → ``types.Content(role, parts=[...])``.
        * tools: ``{name, description, input_schema}`` → ``types.Tool``
          containing ``FunctionDeclaration`` objects.
        * system: passed via ``GenerateContentConfig.system_instruction``.
        * Images inside tool_result: Gemini's ``FunctionResponse`` is JSON
          only — images get hoisted into a sibling user-turn message keyed
          by tool_use_id (same idea as the Claude image fallback).
        * Gemini-3's ``thought_signature`` is captured on response and
          echoed back through ``ToolCall.provider_metadata`` on the next
          turn — required by the API for tool replays.
        """
        contents_dicts, _ = _messages_to_gemini_contents(messages)
        contents = [_dict_to_content(c, self._types) for c in contents_dicts]

        function_decls_dicts = _tools_to_function_declarations(tools)
        function_decls = [
            _dict_to_function_declaration(d, self._types) for d in function_decls_dicts
        ]
        gemini_tools = (
            [self._types.Tool(function_declarations=function_decls)]
            if function_decls else None
        )

        cfg = self._types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system or None,
            tools=gemini_tools,
        )
        response = self._client.models.generate_content(
            model=self._model_name, contents=contents, config=cfg,
        )
        return _gemini_response_to_assistant_turn(response)


# ---------------------------------------------------------------------------
# google.genai object converters — applied after the dict-level translation
# so the existing dict-based tests keep working without an SDK installed.
# ---------------------------------------------------------------------------


def _dict_to_content(c: Dict[str, Any], genai_types) -> Any:
    """Build a ``types.Content`` from one of our intermediate dicts.

    Our intermediate format:
      ``{"role": "user"|"model"|"function", "parts": [<part dict>, ...]}``
    where each part dict is one of:
      ``{"text": ...}``,
      ``{"inline_data": {"mime_type": ..., "data": <bytes>}}``,
      ``{"function_call": {"name", "args", optional "thought_signature", "id"}}``,
      ``{"function_response": {"name", "response": <dict>}}``.
    """
    role = c.get("role") or "user"
    parts = []
    for p in c.get("parts") or []:
        parts.append(_dict_to_part(p, genai_types))
    return genai_types.Content(role=role, parts=parts)


def _dict_to_part(p: Dict[str, Any], genai_types) -> Any:
    if "text" in p:
        return genai_types.Part.from_text(text=p.get("text") or "")
    if "inline_data" in p:
        d = p["inline_data"]
        return genai_types.Part.from_bytes(data=d["data"], mime_type=d["mime_type"])
    if "function_call" in p:
        fc = p["function_call"]
        kwargs: Dict[str, Any] = {
            "function_call": genai_types.FunctionCall(
                name=fc.get("name") or "",
                args=fc.get("args") or {},
                **({"id": fc["id"]} if fc.get("id") else {}),
            ),
        }
        # Echo Gemini-3's thought_signature back when present.
        if fc.get("thought_signature") is not None:
            kwargs["thought_signature"] = fc["thought_signature"]
        return genai_types.Part(**kwargs)
    if "function_response" in p:
        fr = p["function_response"]
        return genai_types.Part(
            function_response=genai_types.FunctionResponse(
                name=fr.get("name") or "",
                response=fr.get("response") or {},
                **({"id": fr["id"]} if fr.get("id") else {}),
            )
        )
    # Fallback: stringify.
    return genai_types.Part.from_text(text=str(p))


def _dict_to_function_declaration(d: Dict[str, Any], genai_types) -> Any:
    fn = d.get("function") or d
    return genai_types.FunctionDeclaration(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=_dict_to_schema(fn.get("parameters") or {"type": "object"}, genai_types),
    )


_TYPE_MAP = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "null": "TYPE_UNSPECIFIED",
}


def _dict_to_schema(s: Dict[str, Any], genai_types) -> Any:
    """JSON Schema dict → ``types.Schema``. Recurses on properties / items."""
    if not isinstance(s, dict):
        return None
    kwargs: Dict[str, Any] = {}
    if "type" in s:
        t = s["type"]
        if isinstance(t, str):
            kwargs["type"] = _TYPE_MAP.get(t.lower(), t.upper())
    for k in ("description", "format", "nullable"):
        if k in s:
            kwargs[k] = s[k]
    if "enum" in s and isinstance(s["enum"], list):
        kwargs["enum"] = [str(e) for e in s["enum"] if e is not None]
    if "required" in s:
        kwargs["required"] = list(s["required"])
    if "properties" in s and isinstance(s["properties"], dict):
        kwargs["properties"] = {
            k: _dict_to_schema(v, genai_types) for k, v in s["properties"].items()
        }
    if "items" in s and isinstance(s["items"], dict):
        kwargs["items"] = _dict_to_schema(s["items"], genai_types)
    return genai_types.Schema(**kwargs)


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
                fc: Dict[str, Any] = {
                    "name": blk.get("name") or "",
                    "args": dict(blk.get("input") or {}),
                }
                # Echo back any provider metadata the Gemini parser captured
                # on the original response (Gemini-3 thought_signature etc.).
                # The session loop preserves it under "provider_metadata".
                pmd = blk.get("provider_metadata") or {}
                if "thought_signature" in pmd:
                    fc["thought_signature"] = pmd["thought_signature"]
                if "function_call_id" in pmd:
                    fc["id"] = pmd["function_call_id"]
                tool_use_parts.append({"function_call": fc})
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


# JSON Schema keywords that the (older) google.generativeai Schema proto does
# not accept and will reject with "Unknown field for Schema: …" errors.
# Reference: only ``type``, ``format``, ``description``, ``nullable``, ``enum``,
# ``properties``, ``required``, ``items`` are supported. Everything else gets
# stripped recursively.
_GEMINI_DROP_KEYS = {
    "additionalProperties", "$schema", "$id", "$defs", "definitions",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "default",
    "anyOf", "oneOf", "allOf", "not",
    "const", "examples", "title",
    "multipleOf",
}


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    """Recursively strip JSON Schema keys Gemini's function calling rejects.

    Also normalizes:
    * ``type: ["string", "null"]`` → ``type: "string", nullable: true`` (Gemini's
      Schema proto requires a single ``type`` value).
    * ``enum: [..., null]`` → drop the ``null`` and set ``nullable: true``.
    """
    if isinstance(schema, dict):
        out: Dict[str, Any] = {}
        is_nullable = False

        for k, v in schema.items():
            if k in _GEMINI_DROP_KEYS:
                continue

            if k == "type" and isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                if len(non_null) != len(v):
                    is_nullable = True
                if len(non_null) == 1:
                    out["type"] = non_null[0]
                elif non_null:
                    # Pick the first non-null type; Gemini doesn't model unions.
                    out["type"] = non_null[0]
                continue

            if k == "enum" and isinstance(v, list):
                non_null = [e for e in v if e is not None]
                if len(non_null) != len(v):
                    is_nullable = True
                out["enum"] = [_sanitize_schema_for_gemini(e) for e in non_null]
                continue

            out[k] = _sanitize_schema_for_gemini(v)

        if is_nullable:
            out["nullable"] = True
        return out
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
                # Capture Gemini-3's thought_signature (and any other opaque
                # per-call metadata) so we can echo it back when this turn is
                # re-sent in the next request — Gemini-3 returns 400 InvalidArgument
                # if the signature is missing on tool_use replay.
                provider_md: Dict[str, Any] = {}
                sig = getattr(part, "thought_signature", None) or getattr(
                    fn_call, "thought_signature", None
                )
                if sig is not None:
                    provider_md["thought_signature"] = sig
                fc_id = getattr(fn_call, "id", None)
                # Only accept string ids; protobuf / SDK quirks can return
                # placeholder objects that would corrupt the call id.
                if isinstance(fc_id, str) and fc_id:
                    call_id = fc_id
                else:
                    call_id = f"gem_{uuid.uuid4().hex[:8]}"
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=fn_call.name,
                        arguments=args,
                        provider_metadata=provider_md or None,
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

"""Anthropic Claude VLM backend (optional dependency: anthropic)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from policy_doctor.vlm.backends.base import VLMBackend

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
    ) -> str:
        """Classify query_images into one of the example groups."""
        content: list = []
        if user_preamble:
            content.append({"type": "text", "text": user_preamble + "\n\n"})
        for label, imgs in example_sets:
            content.append({"type": "text", "text": f"{label}:\n"})
            for img in imgs:
                content.append(self._image_block(img))
            content.append({"type": "text", "text": "\n"})
        content.append({"type": "text", "text": "Query:\n"})
        for img in query_images:
            content.append(self._image_block(img))
        content.append({"type": "text", "text": "\n" + user_prompt})
        return self._call(system_prompt, content)


def build_claude_backend(params: Optional[Dict[str, Any]]) -> ClaudeVLMBackend:
    """Factory function registered in the VLM registry."""
    p = dict(params or {})
    return ClaudeVLMBackend(
        model_name=p.get("model_name", "claude-sonnet-4-6"),
        api_key=p.get("api_key"),
        max_tokens=int(p.get("max_tokens", 1024)),
        temperature=float(p.get("temperature", 0.2)),
    )

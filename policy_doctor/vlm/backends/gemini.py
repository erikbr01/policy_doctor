"""Google Gemini VLM backend (optional dependency: google-generativeai)."""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional, Sequence

from PIL import Image

from policy_doctor.vlm.backends.base import VLMBackend

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


def build_gemini_backend(params: Optional[Dict[str, Any]]) -> GeminiVLMBackend:
    """Factory function registered in the VLM registry."""
    p = dict(params or {})
    return GeminiVLMBackend(
        model_name=p.get("model_name", "gemini-2.0-flash"),
        api_key=p.get("api_key"),
        max_output_tokens=int(p.get("max_output_tokens", 1024)),
        temperature=float(p.get("temperature", 0.2)),
    )

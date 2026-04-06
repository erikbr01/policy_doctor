"""Abstract VLM backend — no heavy imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from PIL import Image


class VLMBackend(ABC):
    """Pluggable vision-language model for slice captions."""

    name: str = "base"

    @abstractmethod
    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Return a short semantic description for the given frames."""

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Summarize many per-slice captions for one behavior cluster (text-only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement behavior-level text summarization."
        )

"""Abstract VLM backend — no heavy imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

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

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Judge whether per-slice captions in one cluster are mutually coherent (text-only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement slice-caption coherency judging."
        )

    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
    ) -> str:
        """Classify *query_images* into one of the K example groups.

        *example_sets*: ordered list of ``(opaque_label, images)`` pairs where
        *opaque_label* is a display string like "Group A" and *images* are
        representative frames for that group (typically storyboard composites,
        one per example slice).

        The backend builds a single multimodal prompt::

            [system_prompt]
            [user_preamble]
            Group A: [img1] [img2] ...
            Group B: [img1] ...
            ...
            Query:
            [query_img1] ...
            [user_prompt]

        Returns the raw text response (parsed by the caller).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement slice classification."
        )

"""Deterministic mock VLM for tests and dry runs."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PIL import Image

from policy_doctor.vlm.backends.base import VLMBackend


class MockVLMBackend(VLMBackend):
    name = "mock"

    def __init__(self, prefix: str = "[mock]", **_kwargs) -> None:
        self.prefix = prefix

    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        sp = (system_prompt or "")[:40]
        up = user_prompt[:80].replace("\n", " ")
        return f"{self.prefix} frames={len(images)} user={up!r} system_hint={sp!r}"

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        n = len(slice_labels)
        head = slice_labels[0][:60] if slice_labels else ""
        return (
            f"{self.prefix} behavior cluster={cluster_id} n_slices={n} "
            f"task={task_hint!r} first_label={head!r}"
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
        n = len(slice_labels)
        return (
            '{"coherent": true, "score": 0.85, "rationale": "mock backend cluster '
            f"{cluster_id} n_slices={n}"
            '"}'
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
        # Always predicts the first group label — deterministic for tests.
        if example_sets:
            label = example_sets[0][0]
        else:
            label = "unclear"
        n_groups = len(example_sets)
        return (
            f"{self.prefix} classify: n_groups={n_groups} "
            f"n_query_frames={len(query_images)} predicted={label!r}"
        )


def build_mock_backend(params: dict) -> MockVLMBackend:
    return MockVLMBackend(**(params or {}))

"""Combined graph representation: PNG + text blocks (default for the graph condition)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from policy_doctor.vlm.proposals.graph_representation.base import (
    GraphRepresentation,
    VLMArtefact,
)
from policy_doctor.vlm.proposals.graph_representation.image_only import (
    ImageOnlyGraphRepresentation,
)
from policy_doctor.vlm.proposals.graph_representation.text_table import (
    TextTableGraphRepresentation,
)
from policy_doctor.vlm.proposals.registry import register_graph_representation

if TYPE_CHECKING:
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.vlm.proposals.pool import RolloutPool


class CombinedGraphRepresentation(GraphRepresentation):
    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        image_params = dict(params.get("image", {}))
        for k in ("figsize", "dpi", "min_probability", "filename"):
            if k in params and k not in image_params:
                image_params[k] = params[k]
        text_params = dict(params.get("text", {}))
        if "include_paths" in params and "include_paths" not in text_params:
            text_params["include_paths"] = params["include_paths"]
        self._image = ImageOnlyGraphRepresentation(image_params)
        self._text = TextTableGraphRepresentation(text_params)

    def render(
        self,
        graph: "BehaviorGraph",
        pool: "RolloutPool",
        output_dir: Path,
    ) -> VLMArtefact:
        img_artefact = self._image.render(graph, pool, output_dir)
        txt_artefact = self._text.render(graph, pool, output_dir)
        return VLMArtefact(
            images=img_artefact.images,
            text_blocks=txt_artefact.text_blocks,
            metadata={
                "representation": "combined",
                "image": img_artefact.metadata,
                "text": txt_artefact.metadata,
            },
        )


def build_combined(params: Dict[str, Any]) -> CombinedGraphRepresentation:
    return CombinedGraphRepresentation(params)


register_graph_representation("combined", build_combined)

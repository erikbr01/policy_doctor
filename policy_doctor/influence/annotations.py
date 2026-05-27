"""JSON annotation read helpers.

Pure-data extracts from the legacy ``influence_visualizer.render_annotation``
module — the streamlit-driven editing UI stays in ``streamlit_app/``; only the
read/load/inspect helpers (no Streamlit imports) live here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def collect_labels_from_annotations(annotations: Dict) -> List[str]:
    """Collect all unique labels from annotation slices.

    Args:
        annotations: Dictionary in format ``{"train": {ep: slices}, "holdout": {...}, "rollout": {...}}``.

    Returns:
        Sorted list of unique label strings found in all slices.
    """
    labels = set()
    for split in ["train", "holdout", "rollout"]:
        if split in annotations:
            for episode_slices in annotations[split].values():
                for slice_info in episode_slices:
                    if "label" in slice_info:
                        labels.add(slice_info["label"])
    return sorted(list(labels))


def load_annotations(filepath: str, task_config: Optional[str] = None) -> Dict:
    """Load annotations from a JSON file.

    If the task_config doesn't exist in the file, it will be automatically created
    with an empty annotation structure and saved to the file.

    Args:
        filepath: Path to the annotation file (must be .json)
        task_config: Task config name to load annotations for (required)

    Returns:
        Dictionary in format::

            {
                "train": {ep_id: [slices]},
                "holdout": {...},
                "rollout": {...},
                "labels": ["label1", "label2", ...],
            }

    Raises:
        FileNotFoundError: If the annotation file doesn't exist
        ValueError: If task_config is not provided or the annotation format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    filepath_obj = Path(filepath)
    if filepath_obj.suffix != ".json":
        raise ValueError(
            f"Annotation file must be JSON format (.json), got: {filepath}"
        )

    if task_config is None:
        raise ValueError("task_config parameter is required for loading annotations")

    with open(filepath, "r") as f:
        data = json.load(f)

    # Create task config if it doesn't exist
    if task_config not in data:
        print(
            f"Task config '{task_config}' not found in {filepath}. Creating new empty annotation structure."
        )
        data[task_config] = {"train": {}, "holdout": {}, "rollout": {}, "labels": []}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    task_data = data[task_config]

    if "labels" not in task_data:
        task_data["labels"] = []

    if not isinstance(task_data, dict):
        raise ValueError(
            f"Task config '{task_config}' data must be a dict, got {type(task_data).__name__}"
        )

    required_splits = {"train", "holdout", "rollout"}
    missing_splits = required_splits - set(task_data.keys())
    if missing_splits:
        raise ValueError(
            f"Task config '{task_config}' missing required splits: {missing_splits}. "
            f"All splits (train, holdout, rollout) must be present (can be empty dicts)."
        )

    for split in required_splits:
        if not isinstance(task_data[split], dict):
            raise ValueError(
                f"Split '{split}' in task config '{task_config}' must be a dict, "
                f"got {type(task_data[split]).__name__}"
            )

        for episode_id, slices in task_data[split].items():
            if not isinstance(slices, list):
                raise ValueError(
                    f"Episode '{episode_id}' in split '{split}' must have a list of slices, "
                    f"got {type(slices).__name__}"
                )

            for idx, slice_obj in enumerate(slices):
                if not isinstance(slice_obj, dict):
                    raise ValueError(
                        f"Slice {idx} in episode '{episode_id}' ({split}) must be a dict, "
                        f"got {type(slice_obj).__name__}"
                    )

                required_keys = {"start", "end", "label"}
                missing_keys = required_keys - set(slice_obj.keys())
                if missing_keys:
                    raise ValueError(
                        f"Slice {idx} in episode '{episode_id}' ({split}) missing required keys: {missing_keys}"
                    )

    # Sync labels: collect all labels from slices and update the labels list
    collected_labels = collect_labels_from_annotations(task_data)
    if set(task_data.get("labels", [])) != set(collected_labels):
        task_data["labels"] = collected_labels
        data[task_config] = task_data
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    return task_data


def get_episode_annotations(
    annotations: Dict, episode_id: str, split: str = "rollout"
) -> List[Dict]:
    """Get annotations for a specific episode and split.

    Args:
        annotations: Annotations dict in format ``{"train": {ep: slices}, "holdout": {...}, "rollout": {...}}``.
        episode_id: Episode ID (just the number, e.g., "0", "1", etc.)
        split: Split name ("train", "holdout", or "rollout")

    Returns:
        List of annotation slices for the episode (empty list if no annotations).

    Raises:
        ValueError: If annotations format is invalid or split is invalid.
    """
    if split not in ["train", "holdout", "rollout"]:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: train, holdout, rollout"
        )

    if not isinstance(annotations, dict):
        raise ValueError(
            f"Annotations must be a dict, got {type(annotations).__name__}"
        )

    if split not in annotations:
        raise ValueError(
            f"Split '{split}' not found in annotations. "
            f"Expected format: {{'train': {{}}, 'holdout': {{}}, 'rollout': {{}}}}. "
            f"Available keys: {list(annotations.keys())}"
        )

    split_data = annotations[split]
    if not isinstance(split_data, dict):
        raise ValueError(
            f"Split '{split}' data must be a dict, got {type(split_data).__name__}"
        )

    if episode_id in split_data:
        return split_data[episode_id]

    return []

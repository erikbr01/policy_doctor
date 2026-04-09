"""Annotation interface for labeling demonstration and rollout segments.

This module provides a video player interface for annotating behavioral segments
in training demonstrations, holdout demonstrations, and rollouts.

Annotations are stored in JSON format with the following structure:
{
    "task_config_name": {
        "labels": ["reaching", "grasping", "custom_label", ...],
        "train": {
            "episode_id": [{"start": 0, "end": 10, "label": "reaching"}, ...],
        },
        "holdout": {
            "episode_id": [{"start": 5, "end": 15, "label": "grasping"}, ...],
        },
        "rollout": {
            "episode_id": [{"start": 0, "end": 20, "label": "lifting"}, ...],
        }
    }
}

All splits (train, holdout, rollout) must be present at the task config level,
even if empty. Episodes without annotations should not appear in the split dict.

The "labels" key stores all unique labels that have been used in any annotation
slice. It is automatically synced when loading or saving annotations, and labels
are removed only when no slices with that label exist anymore.
"""

import json
import numbers
import os
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from influence_visualizer.render_frames import (
    frame_player,
    render_action_chunk,
    render_annotated_frame,
    render_label_timeline,
)


def collect_labels_from_annotations(annotations: Dict) -> List[str]:
    """Collect all unique labels from annotation slices.

    Args:
        annotations: Dictionary in format {"train": {ep: slices}, "holdout": {...}, "rollout": {...}}

    Returns:
        List of unique label strings found in all slices
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
        Dictionary in format: {
            "train": {ep_id: [slices]},
            "holdout": {...},
            "rollout": {...},
            "labels": ["label1", "label2", ...]
        }

    Raises:
        FileNotFoundError: If the annotation file doesn't exist
        ValueError: If task_config is not provided
        ValueError: If the annotation format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    # Validate file format
    filepath_obj = Path(filepath)
    if filepath_obj.suffix != ".json":
        raise ValueError(
            f"Annotation file must be JSON format (.json), got: {filepath}"
        )

    # task_config is required
    if task_config is None:
        raise ValueError("task_config parameter is required for loading annotations")

    # Load JSON format
    with open(filepath, "r") as f:
        data = json.load(f)

    # Create task config if it doesn't exist
    if task_config not in data:
        print(
            f"Task config '{task_config}' not found in {filepath}. Creating new empty annotation structure."
        )
        data[task_config] = {"train": {}, "holdout": {}, "rollout": {}, "labels": []}
        # Save the updated file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    task_data = data[task_config]

    # Initialize labels list if it doesn't exist (for backward compatibility)
    if "labels" not in task_data:
        task_data["labels"] = []

    # Validate structure
    if not isinstance(task_data, dict):
        raise ValueError(
            f"Task config '{task_config}' data must be a dict, got {type(task_data).__name__}"
        )

    # Validate required splits
    required_splits = {"train", "holdout", "rollout"}
    missing_splits = required_splits - set(task_data.keys())
    if missing_splits:
        raise ValueError(
            f"Task config '{task_config}' missing required splits: {missing_splits}. "
            f"All splits (train, holdout, rollout) must be present (can be empty dicts)."
        )

    # Validate each split is a dict
    for split in required_splits:
        if not isinstance(task_data[split], dict):
            raise ValueError(
                f"Split '{split}' in task config '{task_config}' must be a dict, "
                f"got {type(task_data[split]).__name__}"
            )

        # Validate episode data
        for episode_id, slices in task_data[split].items():
            if not isinstance(slices, list):
                raise ValueError(
                    f"Episode '{episode_id}' in split '{split}' must have a list of slices, "
                    f"got {type(slices).__name__}"
                )

            # Validate each slice
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

    # Check if labels list needs updating
    if set(task_data.get("labels", [])) != set(collected_labels):
        task_data["labels"] = collected_labels
        # Save the updated labels back to file
        data[task_config] = task_data
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    return task_data


def save_annotations(
    filepath: str, annotations: Dict, task_config: Optional[str] = None
):
    """Save annotations to a JSON file.

    The labels list will be automatically synced from all annotation slices.
    Labels that are no longer used in any slice will be removed.

    Args:
        filepath: Path to save the annotation file (must be .json)
        annotations: Dictionary in format {"train": {ep: slices}, "holdout": {...}, "rollout": {...}}
                    The "labels" key will be automatically added/updated.
        task_config: Task config name (required)

    Raises:
        ValueError: If filepath is not .json, task_config is missing, or annotations format is invalid
    """
    # Validate file format
    filepath_obj = Path(filepath)
    if filepath_obj.suffix != ".json":
        raise ValueError(
            f"Annotation file must be JSON format (.json), got: {filepath}"
        )

    # task_config is required
    if task_config is None:
        raise ValueError("task_config parameter is required for saving annotations")

    # Validate annotation structure
    if not isinstance(annotations, dict):
        raise ValueError(
            f"Annotations must be a dict, got {type(annotations).__name__}"
        )

    required_splits = {"train", "holdout", "rollout"}
    missing_splits = required_splits - set(annotations.keys())
    if missing_splits:
        raise ValueError(
            f"Annotations missing required splits: {missing_splits}. "
            f"All splits (train, holdout, rollout) must be present (can be empty dicts)."
        )

    # Validate each split
    for split in required_splits:
        if not isinstance(annotations[split], dict):
            raise ValueError(
                f"Split '{split}' must be a dict, got {type(annotations[split]).__name__}"
            )

        # Validate episode data
        for episode_id, slices in annotations[split].items():
            if not isinstance(slices, list):
                raise ValueError(
                    f"Episode '{episode_id}' in split '{split}' must have a list of slices, "
                    f"got {type(slices).__name__}"
                )

            # Validate each slice
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
                        f"Slice {idx} in episode '{episode_id}' ({split}) missing keys: {missing_keys}"
                    )

                # Validate types
                if not isinstance(slice_obj["start"], int):
                    raise ValueError(
                        f"Slice {idx} in episode '{episode_id}' ({split}): 'start' must be int, "
                        f"got {type(slice_obj['start']).__name__}"
                    )
                if not isinstance(slice_obj["end"], int):
                    raise ValueError(
                        f"Slice {idx} in episode '{episode_id}' ({split}): 'end' must be int, "
                        f"got {type(slice_obj['end']).__name__}"
                    )
                if not isinstance(slice_obj["label"], str):
                    raise ValueError(
                        f"Slice {idx} in episode '{episode_id}' ({split}): 'label' must be str, "
                        f"got {type(slice_obj['label']).__name__}"
                    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Load existing data first
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Collect all labels from the annotation slices
    collected_labels = collect_labels_from_annotations(annotations)

    # Ensure labels key exists and is synced
    if "labels" not in annotations:
        annotations["labels"] = []
    annotations["labels"] = collected_labels

    # Update the task config's annotations
    data[task_config] = annotations

    # Save back to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def validate_slice(
    start: int, end: int, episode_length: int, existing_slices: List[Dict]
) -> Optional[str]:
    """Validate a new annotation slice.

    Note: Multiple labels per timestamp are allowed, so overlaps are permitted.

    Args:
        start: Start frame
        end: End frame
        episode_length: Total number of frames in episode
        existing_slices: List of existing annotation slices for this episode (not used for overlap check)

    Returns:
        Error message if invalid, None if valid
    """
    if start is None or end is None:
        return "Please set both start and end frames"

    if start > end:
        return "Start frame must be <= end frame"

    if start < 0 or end >= episode_length:
        return f"Slice must be within [0, {episode_length - 1}]"

    # Multiple labels per timestamp are allowed, so no overlap checking
    return None


def get_episode_annotations(
    annotations: Dict, episode_id: str, split: str = "rollout"
) -> List[Dict]:
    """Get annotations for a specific episode and split.

    Args:
        annotations: Annotations dict in format {"train": {ep: slices}, "holdout": {...}, "rollout": {...}}
        episode_id: Episode ID (just the number, e.g., "0", "1", etc.)
        split: Split name ("train", "holdout", or "rollout")

    Returns:
        List of annotation slices for the episode (empty list if no annotations)

    Raises:
        ValueError: If annotations format is invalid or split is invalid
    """
    # Validate split
    if split not in ["train", "holdout", "rollout"]:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: train, holdout, rollout"
        )

    # Validate annotations structure
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

    # Return episode's annotations if it exists, otherwise empty list
    if episode_id in split_data:
        return split_data[episode_id]

    return []


def get_labels_for_frame(frame_idx: int, slices: List[Dict]) -> List[str]:
    """Get all labels for a specific frame.

    Args:
        frame_idx: Frame index
        slices: List of annotation slices

    Returns:
        List of label strings (empty list if not annotated)
    """
    labels = []
    for slice_info in slices:
        if slice_info["start"] <= frame_idx <= slice_info["end"]:
            labels.append(slice_info["label"])
    return labels


def get_label_for_frame(frame_idx: int, slices: List[Dict]) -> str:
    """Get the label for a specific frame as a string.

    If multiple labels exist for the same frame, returns the first one.

    Args:
        frame_idx: Frame index
        slices: List of annotation slices

    Returns:
        Label string (first label if multiple), or "no label" if not annotated
    """
    labels = get_labels_for_frame(frame_idx, slices)
    if not labels:
        return "no label"
    return labels[0]  # Return first label if multiple exist


def render_annotation_interface(
    data, annotation_file: str, task_config: Optional[str] = None, obs_key: str = "agentview_image"
):
    """Render annotation interface for labeling demonstration and rollout segments.

    This interface provides tools for annotating slices of training demonstrations,
    holdout demonstrations, and rollouts with behavioral labels.

    Args:
        data: DataLoader instance containing all episode and sample data.
        annotation_file: Path to the annotation file (.json or .pkl).
        task_config: Task config name (required for JSON files).
        obs_key: Camera/observation key for frame images (e.g. from sidebar Camera View).
    """
    st.header("Annotation Tool")
    st.markdown("""
    Label slices of demonstrations and rollouts with behavioral annotations.
    Multiple labels can be applied to the same timestamp (overlapping annotations are allowed).
    """)

    # Load annotations
    annotations = load_annotations(annotation_file, task_config=task_config)

    # Show file status
    if os.path.exists(annotation_file):
        # Count total annotated episodes across all splits
        total_episodes = (
            len(annotations.get("train", {}))
            + len(annotations.get("holdout", {}))
            + len(annotations.get("rollout", {}))
        )

        if total_episodes > 0:
            # Count total slices
            total_slices = sum(
                len(slices)
                for split in ["train", "holdout", "rollout"]
                for slices in annotations.get(split, {}).values()
            )
            st.success(
                f"Loaded {total_episodes} annotated episodes "
                f"({total_slices} slices) for task '{task_config}' from {annotation_file}"
            )
        else:
            st.info(
                f"No annotations found for task config '{task_config}' in {annotation_file}"
            )
    else:
        st.info(f"File does not exist yet. Will be created on first save.")

    st.divider()

    # Wrap entire interface body in a fragment for speed
    @st.fragment
    def _render_annotation_body():
        # Reload annotations inside fragment to get updates from save/delete
        annotations = load_annotations(annotation_file, task_config=task_config)

        # Source selection
        st.subheader("Select Episode")
        col_source, col_episode = st.columns([1, 1])

        with col_source:
            source_type = st.selectbox(
                "Data Source",
                options=["Train Demonstrations", "Holdout Demonstrations", "Rollouts"],
                key="annotation_source",
            )

        # Pick episode list based on source
        if source_type == "Train Demonstrations":
            episodes = data.demo_episodes
            data_type = "demo"
        elif source_type == "Holdout Demonstrations":
            episodes = data.holdout_episodes
            data_type = "holdout"
        else:
            episodes = data.rollout_episodes
            data_type = "rollout"

        with col_episode:
            if len(episodes) > 0:

                def format_ep(ep):
                    # Annotations are stored as annotations[split][episode_id] where
                    # split is "train"|"holdout"|"rollout" and episode_id is str(ep.index)
                    split = "train" if data_type == "demo" else data_type
                    episode_id_key = str(ep.index)
                    num_annotations = len(
                        annotations.get(split, {}).get(episode_id_key, [])
                    )

                    # Add success label for rollouts, quality label for demos
                    if data_type == "rollout":
                        status = (
                            "✓"
                            if ep.success
                            else "✗"
                            if ep.success is not None
                            else "?"
                        )
                        return f"Episode {ep.index} [{status}] ({ep.num_samples} samples, {num_annotations} slices)"
                    elif data_type in ["demo", "holdout"]:
                        # Try to get quality label from data
                        if (
                            hasattr(data, "demo_quality_labels")
                            and data.demo_quality_labels is not None
                        ):
                            quality = data.demo_quality_labels.get(ep.index, "N/A")
                            return f"Episode {ep.index} [Q: {quality}] ({ep.num_samples} samples, {num_annotations} slices)"
                        else:
                            return f"Episode {ep.index} ({ep.num_samples} samples, {num_annotations} slices)"
                    else:
                        return f"Episode {ep.index} ({ep.num_samples} samples, {num_annotations} slices)"

                # IMPORTANT: We use TWO separate keys to avoid Streamlit overwriting our integer index
                # 1. track_key: Our integer index (0, 1, 2, ...) - WE control this
                # 2. _widget_key: Streamlit's internal widget state - may contain formatted strings
                track_key = f"ann_ep_idx_{source_type}"
                _widget_key = f"_ep_widget_{source_type}"

                # Initialize tracking key
                if track_key not in st.session_state:
                    st.session_state[track_key] = 0

                # Validate tracking key
                stored_val = st.session_state[track_key]
                if not isinstance(stored_val, numbers.Integral):
                    st.warning(
                        f"Episode index had unexpected type {type(stored_val).__name__}: {stored_val!r}. Resetting to 0."
                    )
                    st.session_state[track_key] = 0
                elif stored_val < 0 or stored_val >= len(episodes):
                    st.warning(
                        f"Episode index {stored_val} out of bounds [0, {len(episodes) - 1}]. Resetting to 0."
                    )
                    st.session_state[track_key] = 0

                current_idx = int(st.session_state[track_key])

                # Callback to sync widget selection back to our tracking key
                def on_episode_select():
                    if _widget_key in st.session_state:
                        val = st.session_state[_widget_key]
                        if isinstance(val, numbers.Integral):
                            st.session_state[track_key] = int(val)
                        # If val is not integral (e.g., formatted string), just ignore it
                        # The track_key will retain its previous value

                # Prev/Next modify BOTH keys - track_key for our logic, _widget_key for the selectbox
                def go_prev():
                    current = st.session_state.get(track_key, 0)
                    if isinstance(current, numbers.Integral) and current > 0:
                        new_val = int(current) - 1
                        st.session_state[track_key] = new_val
                        st.session_state[_widget_key] = new_val

                def go_next():
                    current = st.session_state.get(track_key, 0)
                    if (
                        isinstance(current, numbers.Integral)
                        and current < len(episodes) - 1
                    ):
                        new_val = int(current) + 1
                        st.session_state[track_key] = new_val
                        st.session_state[_widget_key] = new_val

                # Use index= to control display, on_change to sync back
                selected_ep_idx = st.selectbox(
                    "Episode",
                    options=range(len(episodes)),
                    index=current_idx,
                    format_func=lambda i: format_ep(episodes[i]),
                    key=_widget_key,
                    on_change=on_episode_select,
                )
                selected_ep = episodes[selected_ep_idx]
            else:
                st.info(f"No episodes available for {source_type}.")
                return

        # Prev/Next episode buttons below the selectors
        if len(episodes) > 0:
            btn_cols = st.columns([1, 1, 6])
            with btn_cols[0]:
                st.button(
                    "← Prev",
                    key="btn_prev_ep",
                    disabled=current_idx == 0,
                    on_click=go_prev,
                )
            with btn_cols[1]:
                st.button(
                    "Next →",
                    key="btn_next_ep",
                    disabled=current_idx >= len(episodes) - 1,
                    on_click=go_next,
                )

        # Create episode ID for annotation storage
        episode_id_str = str(selected_ep.index)

        # Map data_type to split name
        split_map = {"demo": "train", "holdout": "holdout", "rollout": "rollout"}
        split = split_map.get(data_type, "rollout")

        # Get episode annotations using helper function
        episode_annotations = get_episode_annotations(
            annotations, episode_id_str, split
        )

        st.divider()

        # Video player for frame navigation
        st.subheader("Video Player")

        def _render_annotation_frame(frame_idx):
            """Render the current frame in the annotation player."""
            abs_idx = selected_ep.sample_start_idx + frame_idx

            # Get current label for this frame
            current_label = get_label_for_frame(frame_idx, episode_annotations)

            # Render label timeline at the top
            render_label_timeline(
                episode_annotations,
                num_frames=selected_ep.num_samples,
                current_frame=frame_idx,
                unique_key=f"tab_ann_timeline_{data_type}_ep{selected_ep.index}",
            )

            col_frame, col_action = st.columns([1, 1])

            with col_frame:
                # Get frame based on data type (use sidebar camera view)
                if data_type in ["demo", "holdout"]:
                    frame = data.get_demo_frame(abs_idx, obs_key=obs_key)
                else:  # rollout
                    frame = data.get_rollout_frame(abs_idx, obs_key=obs_key)

                lbl = f"t={frame_idx} | ep={selected_ep.index} | {current_label}"
                label_str = (
                    current_label if current_label != "no label" else "no label yet"
                )
                render_annotated_frame(
                    frame,
                    lbl,
                    f"Global sample index: {abs_idx} | Label: {label_str}",
                )

            with col_action:
                # Get action chunk based on data type
                if data_type in ["demo", "holdout"]:
                    action_chunk = data.get_demo_action_chunk(abs_idx)
                    action_title = f"{source_type} Sample {abs_idx}"
                else:  # rollout
                    action_chunk = data.get_rollout_action_chunk(abs_idx)
                    action_title = f"Rollout Sample {abs_idx}"

                # Render action chunk as a plot
                render_action_chunk(
                    action_chunk,
                    title=action_title,
                    unique_key=f"annotate_action_{data_type}_ep{selected_ep.index}_f{frame_idx}",
                )

        # Use frame_player with render function
        player_key = f"annotate_{data_type}_ep{selected_ep.index}"
        frame_player(
            label="Frame:",
            min_value=0,
            max_value=selected_ep.num_samples - 1,
            key=player_key,
            default_value=0,
            default_fps=3.0,
            help="Navigate through frames in the selected episode",
            render_fn=_render_annotation_frame,
            fragment_scope=True,
        )

        st.divider()

        # Slice annotation controls
        st.subheader("Annotate Slice")

        # Initialize session state for slice selection
        slice_start_key = f"slice_start_{data_type}_ep{episode_id_str}"
        slice_end_key = f"slice_end_{data_type}_ep{episode_id_str}"

        if slice_start_key not in st.session_state:
            st.session_state[slice_start_key] = None
        if slice_end_key not in st.session_state:
            st.session_state[slice_end_key] = None

        # Behavior label selection
        # Default preset labels
        preset_labels = [
            "reaching",
            "grasping",
            "lifting",
            "placing",
            "releasing",
            "idle",
            "recovery",
            "bad reach",
            "bad grasp",
            "other",
        ]

        # Get custom labels from annotations (these are labels that have been used and saved)
        custom_labels_from_file = annotations.get("labels", [])

        # Combine preset labels with custom labels (custom labels that aren't in preset)
        # Custom labels appear first to prioritize recently used labels
        all_labels = []
        for label in custom_labels_from_file:
            if label not in preset_labels:
                all_labels.append(label)
        all_labels.extend(preset_labels)

        # If no labels available, use preset only
        if not all_labels:
            all_labels = preset_labels

        col_label, col_custom = st.columns([1, 1])

        with col_label:
            selected_label = st.selectbox(
                "Behavior Label",
                options=all_labels,
                key="annotation_label_select",
                help="Select from preset labels or previously used custom labels",
            )

        with col_custom:
            custom_label = st.text_input(
                "Custom Label",
                placeholder="Enter custom label...",
                key="annotation_custom_input",
                help="Enter a new custom label (will be saved for future use)",
            )

        label_to_use = custom_label.strip() if custom_label.strip() else selected_label

        # Slice selection buttons
        col_start, col_end, col_reset, col_save = st.columns(4)

        def set_slice_start():
            # Read the current frame from session state at button click time
            frame = st.session_state.get(f"{player_key}_value", 0)
            st.session_state[slice_start_key] = frame

        def set_slice_end():
            # Read the current frame from session state at button click time
            frame = st.session_state.get(f"{player_key}_value", 0)
            st.session_state[slice_end_key] = frame

        def reset_slice():
            st.session_state[slice_start_key] = None
            st.session_state[slice_end_key] = None

        def save_slice():
            start = st.session_state[slice_start_key]
            end = st.session_state[slice_end_key]

            # Validate slice
            error = validate_slice(
                start, end, selected_ep.num_samples, episode_annotations
            )
            if error:
                st.session_state["annotation_error"] = error
                return

            # Ensure annotations has the new format structure
            if "train" not in annotations:
                annotations["train"] = {}
            if "holdout" not in annotations:
                annotations["holdout"] = {}
            if "rollout" not in annotations:
                annotations["rollout"] = {}

            # Ensure episode exists in the appropriate split
            if episode_id_str not in annotations[split]:
                annotations[split][episode_id_str] = []

            # Add annotation to the appropriate split
            annotations[split][episode_id_str].append(
                {
                    "start": start,
                    "end": end,
                    "label": label_to_use,
                }
            )

            # Sort by start frame
            annotations[split][episode_id_str].sort(key=lambda x: x["start"])

            # Save to file
            save_annotations(annotation_file, annotations, task_config=task_config)

            # Reset slice selection
            st.session_state[slice_start_key] = None
            st.session_state[slice_end_key] = None
            st.session_state["annotation_success"] = (
                f"Saved slice [{start}, {end}] with label '{label_to_use}'"
            )

        with col_start:
            st.button(
                "Set Start",
                key="btn_slice_start",
                on_click=set_slice_start,
            )

        with col_end:
            st.button(
                "Set End",
                key="btn_slice_end",
                on_click=set_slice_end,
            )

        with col_reset:
            st.button(
                "Reset",
                key="btn_slice_reset",
                on_click=reset_slice,
            )

        with col_save:
            save_disabled = (
                st.session_state[slice_start_key] is None
                or st.session_state[slice_end_key] is None
            )
            st.button(
                f"Save '{label_to_use}'",
                key="btn_slice_save",
                on_click=save_slice,
                disabled=save_disabled,
                type="primary",
            )

        # Show current slice selection
        start_val = st.session_state[slice_start_key]
        end_val = st.session_state[slice_end_key]

        if start_val is not None or end_val is not None:
            start_str = str(start_val) if start_val is not None else "?"
            end_str = str(end_val) if end_val is not None else "?"
            st.info(f"Current selection: [{start_str}, {end_str}]")

        # Show error or success messages
        if "annotation_error" in st.session_state:
            st.error(st.session_state["annotation_error"])
            del st.session_state["annotation_error"]

        if "annotation_success" in st.session_state:
            st.success(st.session_state["annotation_success"])
            del st.session_state["annotation_success"]

        st.divider()

        # Display existing annotations
        st.subheader("Existing Annotations")

        if len(episode_annotations) == 0:
            st.info("No annotations for this episode yet.")
        else:
            st.write(f"**{len(episode_annotations)} annotation slice(s)**")

            def delete_annotation(idx):
                # Remove this annotation from the appropriate split
                if split in annotations and episode_id_str in annotations[split]:
                    annotations[split][episode_id_str].pop(idx)

                    # Remove episode key from this split if no annotations left
                    if len(annotations[split][episode_id_str]) == 0:
                        del annotations[split][episode_id_str]

                # Save to file
                save_annotations(annotation_file, annotations, task_config=task_config)
                st.session_state["annotation_success"] = "Annotation deleted"

            for idx, slice_info in enumerate(episode_annotations):
                col_info, col_delete = st.columns([4, 1])

                with col_info:
                    num_frames = slice_info["end"] - slice_info["start"] + 1
                    st.text(
                        f"[{slice_info['start']}, {slice_info['end']}] "
                        f"({num_frames} frames) - {slice_info['label']}"
                    )

                with col_delete:
                    st.button(
                        "Delete",
                        key=f"btn_delete_annotation_{idx}",
                        on_click=delete_annotation,
                        args=(idx,),
                    )

    # Call the annotation body fragment
    _render_annotation_body()

"""Common plotting utilities.

This module provides shared constants and functions used across plotting modules,
including color palettes for labels and influence visualization.
"""

from typing import Dict, List, Tuple

# Color palette for annotation labels (behavior types)
LABEL_COLORS: Dict[str, str] = {
    "reaching": "#4CAF50",
    "grasping": "#2196F3",
    "lifting": "#FF9800",
    "placing": "#9C27B0",
    "idle": "#9E9E9E",
    "recovery": "#FFEB3B",
    "releasing": "#00BCD4",
    "bad reach": "#F44336",
    "bad grasp": "#E91E63",
    "other": "#795548",
}

# Extra colors for custom labels not in the predefined palette
EXTRA_COLORS: List[str] = [
    "#00BCD4",
    "#E91E63",
    "#3F51B5",
    "#8BC34A",
    "#FF5722",
    "#607D8B",
    "#CDDC39",
    "#673AB7",
]


def get_label_color(label: str, custom_map: Dict[str, str]) -> str:
    """Get the color for a label, assigning new colors for unknown labels.

    Args:
        label: The label string
        custom_map: Mutable dict to track assigned colors for custom labels

    Returns:
        Hex color string
    """
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    if label not in custom_map:
        idx = len(custom_map) % len(EXTRA_COLORS)
        custom_map[label] = EXTRA_COLORS[idx]
    return custom_map[label]


def get_influence_colorscale() -> List[Tuple[float, str]]:
    """Get the red-white-green colorscale for influence visualization.

    Returns:
        List of (position, color) tuples for Plotly colorscale
    """
    return [(0, "red"), (0.5, "white"), (1, "green")]


def get_action_labels(action_dim: int) -> List[str]:
    """Get semantic labels for action dimensions.

    For robomimic with abs_action=True and rotation_6d:
    - 10D single arm: pos(3) + rotation_6d(6) + gripper(1)
    - 20D dual arm: 2 * [pos(3) + rotation_6d(6) + gripper(1)]

    Args:
        action_dim: Total number of action dimensions

    Returns:
        List of labels for each dimension
    """
    if action_dim == 10:
        # Single arm with rotation_6d
        return [
            "pos_x",
            "pos_y",
            "pos_z",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "rot_4",
            "rot_5",
            "gripper",
        ]
    elif action_dim == 20:
        # Dual arm with rotation_6d
        labels = []
        for arm in ["arm0", "arm1"]:
            labels.extend(
                [
                    f"{arm}_pos_x",
                    f"{arm}_pos_y",
                    f"{arm}_pos_z",
                    f"{arm}_rot_0",
                    f"{arm}_rot_1",
                    f"{arm}_rot_2",
                    f"{arm}_rot_3",
                    f"{arm}_rot_4",
                    f"{arm}_rot_5",
                    f"{arm}_gripper",
                ]
            )
        return labels
    elif action_dim == 7:
        # Single arm with axis_angle (raw)
        return [
            "pos_x",
            "pos_y",
            "pos_z",
            "axis_angle_x",
            "axis_angle_y",
            "axis_angle_z",
            "gripper",
        ]
    elif action_dim == 14:
        # Dual arm with axis_angle (raw)
        labels = []
        for arm in ["arm0", "arm1"]:
            labels.extend(
                [
                    f"{arm}_pos_x",
                    f"{arm}_pos_y",
                    f"{arm}_pos_z",
                    f"{arm}_axis_angle_x",
                    f"{arm}_axis_angle_y",
                    f"{arm}_axis_angle_z",
                    f"{arm}_gripper",
                ]
            )
        return labels
    else:
        # Unknown format, use generic labels
        return [f"dim_{i}" for i in range(action_dim)]


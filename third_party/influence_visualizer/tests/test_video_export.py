"""Tests for video export functionality."""

import pathlib
import tempfile

import numpy as np
import pytest


def test_export_functions_exist():
    """Test that video export functions are properly defined."""
    from influence_visualizer.video_export import export_slice_videos

    assert callable(export_slice_videos)


def test_video_export_module_imports():
    """Test that the video export module can be imported without errors."""
    try:
        import influence_visualizer.video_export

        assert hasattr(influence_visualizer.video_export, "export_slice_videos")
        assert hasattr(influence_visualizer.video_export, "_export_rollout_slice_video")
        assert hasattr(influence_visualizer.video_export, "_export_demo_slice_video")
    except ImportError as e:
        pytest.fail(f"Failed to import video_export module: {e}")


if __name__ == "__main__":
    # Run basic smoke tests
    print("Testing video export module imports...")
    test_video_export_module_imports()
    print("✓ Module imports successful")

    print("Testing function existence...")
    test_export_functions_exist()
    print("✓ Functions exist")

    print("\nAll smoke tests passed!")

"""Tests for policy_doctor.plotting.plotly.frames."""

import unittest

import numpy as np

from policy_doctor.plotting.plotly import create_action_plot, create_annotated_frame


class TestPlotlyFrames(unittest.TestCase):
    def test_create_annotated_frame_returns_pil_image(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[::2, :, :] = 255
        out = create_annotated_frame(img, "test label")
        self.assertIsNotNone(out)
        self.assertEqual(out.__class__.__module__, "PIL.Image")
        self.assertEqual(out.__class__.__name__, "Image")
        self.assertEqual(out.size, (32, 32))

    def test_create_annotated_frame_grayscale(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        out = create_annotated_frame(img, "x")
        self.assertEqual(out.size, (16, 16))

    def test_create_action_plot_returns_figure(self):
        actions = np.array([[0.1, 0.2, -0.1]], dtype=np.float32)
        labels = ["a", "b", "c"]
        fig = create_action_plot(actions, labels, title="Actions")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, "Figure")

    def test_create_action_plot_has_scatter_or_bar(self):
        actions = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        fig = create_action_plot(actions, ["x", "y"])
        self.assertGreater(len(fig.data), 0)

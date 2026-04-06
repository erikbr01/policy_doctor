"""Tests for policy_doctor.plotting.common."""

import unittest

from policy_doctor.plotting.common import (
    EXTRA_COLORS,
    LABEL_COLORS,
    get_action_labels,
    get_influence_colorscale,
    get_label_color,
)


class TestCommon(unittest.TestCase):
    def test_label_colors_constants(self):
        self.assertIsInstance(LABEL_COLORS, dict)
        self.assertIn("reaching", LABEL_COLORS)
        self.assertTrue(all(s.startswith("#") for s in LABEL_COLORS.values()))

    def test_extra_colors_list(self):
        self.assertIsInstance(EXTRA_COLORS, list)
        self.assertGreater(len(EXTRA_COLORS), 0)
        self.assertTrue(all(s.startswith("#") for s in EXTRA_COLORS))

    def test_get_label_color_known(self):
        custom = {}
        self.assertEqual(get_label_color("reaching", custom), "#4CAF50")
        self.assertEqual(get_label_color("grasping", custom), "#2196F3")
        self.assertEqual(len(custom), 0)

    def test_get_label_color_unknown_assigns_from_extra(self):
        custom = {}
        c1 = get_label_color("custom_foo", custom)
        self.assertIn(c1, EXTRA_COLORS)
        self.assertEqual(custom["custom_foo"], c1)
        c2 = get_label_color("custom_bar", custom)
        self.assertIn(c2, EXTRA_COLORS)
        self.assertEqual(get_label_color("custom_foo", custom), c1)

    def test_get_influence_colorscale(self):
        scale = get_influence_colorscale()
        self.assertIsInstance(scale, list)
        self.assertEqual(len(scale), 3)
        self.assertEqual(scale[0], (0, "red"))
        self.assertEqual(scale[1], (0.5, "white"))
        self.assertEqual(scale[2], (1, "green"))

    def test_get_action_labels_10d(self):
        labels = get_action_labels(10)
        self.assertEqual(len(labels), 10)
        self.assertIn("pos_x", labels)
        self.assertIn("gripper", labels)

    def test_get_action_labels_7d(self):
        labels = get_action_labels(7)
        self.assertEqual(len(labels), 7)
        self.assertIn("axis_angle_x", labels)

    def test_get_action_labels_generic(self):
        labels = get_action_labels(5)
        self.assertEqual(labels, ["dim_0", "dim_1", "dim_2", "dim_3", "dim_4"])

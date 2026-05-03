"""Unit tests for policy_doctor.vlm.proposals.registry."""

from __future__ import annotations

import unittest

from policy_doctor.vlm.proposals.registry import (
    get_graph_representation,
    get_vlm_input_builder,
    list_graph_representations,
    list_vlm_input_builders,
)


class TestGraphRepresentationRegistry(unittest.TestCase):
    def test_lists_bundled_names(self):
        names = list_graph_representations()
        for expected in ("image_only", "text_table", "combined"):
            self.assertIn(expected, names)

    def test_get_combined_returns_renderable(self):
        repr_ = get_graph_representation("combined", {})
        self.assertTrue(callable(getattr(repr_, "render", None)))

    def test_get_image_only_returns_renderable(self):
        repr_ = get_graph_representation("image_only", {})
        self.assertTrue(callable(getattr(repr_, "render", None)))

    def test_get_text_table_returns_renderable(self):
        repr_ = get_graph_representation("text_table", {})
        self.assertTrue(callable(getattr(repr_, "render", None)))

    def test_unknown_graph_repr_raises_with_listing(self):
        with self.assertRaises(ValueError) as cm:
            get_graph_representation("not_a_real_repr", {})
        msg = str(cm.exception)
        # Helpful message lists the registered names.
        self.assertIn("Registered:", msg)
        for expected in ("combined", "image_only", "text_table"):
            self.assertIn(expected, msg)

    def test_case_insensitive_lookup(self):
        repr_ = get_graph_representation("Combined", {})
        self.assertTrue(callable(getattr(repr_, "render", None)))


class TestVLMInputBuilderRegistry(unittest.TestCase):
    def test_lists_bundled_names(self):
        names = list_vlm_input_builders()
        for expected in ("graph_condition", "outcome_condition"):
            self.assertIn(expected, names)

    def test_get_graph_condition_builder(self):
        b = get_vlm_input_builder("graph_condition", {})
        self.assertTrue(callable(getattr(b, "build_messages", None)))

    def test_get_outcome_condition_builder(self):
        b = get_vlm_input_builder("outcome_condition", {})
        self.assertTrue(callable(getattr(b, "build_messages", None)))

    def test_unknown_builder_raises_with_listing(self):
        with self.assertRaises(ValueError) as cm:
            get_vlm_input_builder("nope", {})
        msg = str(cm.exception)
        self.assertIn("Registered:", msg)
        for expected in ("graph_condition", "outcome_condition"):
            self.assertIn(expected, msg)


if __name__ == "__main__":
    unittest.main()

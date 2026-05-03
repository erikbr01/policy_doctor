"""Condition vocabulary: aliasing legacy → canonical names."""

from __future__ import annotations

import unittest

from policy_doctor.vlm.proposals.agents.conditions import (
    Condition,
    parse_condition,
)


class TestParseCondition(unittest.TestCase):
    def test_canonical_names(self):
        self.assertEqual(parse_condition("A_G"), Condition.A_G)
        self.assertEqual(parse_condition("A_NG"), Condition.A_NG)
        self.assertEqual(parse_condition("H_NG"), Condition.H_NG)
        self.assertEqual(parse_condition("H_G"), Condition.H_G)

    def test_legacy_aliases(self):
        # Legacy one-shot names map to the agentic canonical equivalents so
        # both vocabularies coexist without producing two enum members.
        self.assertEqual(parse_condition("graph"), Condition.A_G)
        self.assertEqual(parse_condition("outcome_only"), Condition.A_NG)

    def test_passes_through_enum(self):
        self.assertEqual(parse_condition(Condition.A_G), Condition.A_G)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            parse_condition("Z_X")


if __name__ == "__main__":
    unittest.main()

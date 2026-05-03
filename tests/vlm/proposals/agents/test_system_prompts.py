"""Frozen-prompt registration tests."""

from __future__ import annotations

import unittest

from policy_doctor.vlm.proposals.agents.system_prompts import (
    all_prompt_hashes,
    prompt_hash,
    prompt_text,
)


class TestSystemPrompts(unittest.TestCase):
    def test_a_g_prompt_loads(self):
        text = prompt_text("A_G")
        self.assertGreater(len(text), 100)
        # Frozen invariants — these must remain in the prompt to satisfy spec.
        self.assertIn("propose_collection_request", text)
        self.assertIn("finalize_strategy", text)
        self.assertIn("reasoning", text)
        # Must mention the leak rule (operator-facing fields).
        self.assertIn("operator", text.lower())

    def test_a_ng_prompt_loads(self):
        text = prompt_text("A_NG")
        self.assertGreater(len(text), 100)
        self.assertIn("propose_collection_request", text)
        # A_NG must NOT mention behavior-graph terms in the agent's instructions.
        self.assertNotIn("behavior graph", text.lower())
        self.assertNotIn("cluster", text.lower())

    def test_human_conditions_route_to_correct_prompt(self):
        # H_NG uses the same instruction text as A_NG, H_G as A_G.
        self.assertEqual(prompt_text("H_NG"), prompt_text("A_NG"))
        self.assertEqual(prompt_text("H_G"), prompt_text("A_G"))

    def test_hashes_are_stable_within_run(self):
        h1 = prompt_hash("A_G")
        h2 = prompt_hash("A_G")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        int(h1, 16)

    def test_all_prompt_hashes_includes_both_agents(self):
        hashes = all_prompt_hashes()
        self.assertIn("A_G", hashes)
        self.assertIn("A_NG", hashes)
        self.assertNotEqual(hashes["A_G"], hashes["A_NG"])


if __name__ == "__main__":
    unittest.main()

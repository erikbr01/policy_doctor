"""Tests for VLM coherency judging helpers."""

from __future__ import annotations

import json
import unittest

from policy_doctor.vlm.backends.mock import MockVLMBackend
from policy_doctor.vlm.coherency_eval import parse_coherency_json, run_cluster_coherency_eval


class TestParseCoherencyJson(unittest.TestCase):
    def test_plain_json(self):
        d = parse_coherency_json('{"coherent": false, "score": 0.2, "rationale": "x"}')
        self.assertEqual(d.get("parse_error"), None)
        self.assertFalse(d["coherent"])
        self.assertEqual(d["score"], 0.2)

    def test_fenced_json(self):
        raw = '```json\n{"coherent": true, "score": 1.0, "rationale": "ok"}\n```'
        d = parse_coherency_json(raw)
        self.assertEqual(d.get("parse_error"), None)
        self.assertTrue(d["coherent"])

    def test_empty(self):
        d = parse_coherency_json("")
        self.assertEqual(d["parse_error"], "empty")


class TestRunClusterCoherencyEvalMock(unittest.TestCase):
    def test_one_cluster(self):
        records = [
            {"cluster_id": 0, "label": "grasp"},
            {"cluster_id": 0, "label": "pick"},
        ]
        rows, pver = run_cluster_coherency_eval(
            records,
            backend=MockVLMBackend(),
            task_hint="test",
            prompts_file=None,
            prompts_inline=None,
            repo_root=None,
            max_slice_labels_per_cluster=None,
            max_clusters=None,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(pver)
        j = parse_coherency_json(rows[0]["raw_response"])
        self.assertEqual(j.get("parse_error"), None)
        self.assertTrue(j["coherent"])


if __name__ == "__main__":
    unittest.main()

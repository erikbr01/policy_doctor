"""Unit tests for policy_doctor.vlm.proposals.request."""

from __future__ import annotations

import json
import unittest

from policy_doctor.vlm.proposals.request import (
    DemonstrationRequest,
    InitialConditions,
    REQUEST_TYPES,
    RequestValidationError,
    parse_and_validate_batch,
    request_json_schema,
    validate_request,
)


def _good_request(**overrides):
    ic = InitialConditions(
        reference_rollout_id=overrides.pop("reference_rollout_id", "r0001"),
        reference_frame=overrides.pop("reference_frame", 0),
    )
    base = dict(
        request_id="abc123",
        request_type="full_trajectory",
        initial_conditions=ic,
        target_behavior="grasp the green cube and lift it above the table",
        prohibitions=["do not knock the cube off the table"],
        success_criterion="task_success",
        source_condition="graph",
        target_cluster=2,
    )
    base.update(overrides)
    return DemonstrationRequest(**base)


class TestValidateRequest(unittest.TestCase):
    def test_valid_request_passes(self):
        validate_request(_good_request())

    def test_empty_target_behavior_raises(self):
        req = _good_request(target_behavior="")
        with self.assertRaises(RequestValidationError):
            validate_request(req)

    def test_bad_request_type_raises(self):
        req = _good_request(request_type="not_a_type")
        with self.assertRaises(RequestValidationError):
            validate_request(req)

    def test_negative_reference_frame_raises(self):
        ic = InitialConditions(reference_rollout_id="r0001", reference_frame=-1)
        req = _good_request()
        req.initial_conditions = ic
        with self.assertRaises(RequestValidationError):
            validate_request(req)

    def test_empty_request_id_raises(self):
        req = _good_request(request_id="")
        with self.assertRaises(RequestValidationError):
            validate_request(req)


class TestDenylist(unittest.TestCase):
    """Each forbidden term must be caught in target_behavior, success_criterion,
    and any prohibitions[*] entry — case-insensitive."""

    FORBIDDEN_PHRASES = [
        "approach the cluster carefully",       # cluster
        "approach the Clusters carefully",      # cluster (caps + plural)
        "navigate to the node",                 # node
        "follow the graph carefully",           # graph
        "stay in the umap region",              # umap
        "use kmeans on the data",               # kmeans
        "use k-means on the data",              # kmeans (hyphen)
        "use k means on the data",              # kmeans (space)
        "approach the centroid",                # centroid
        "approach the centroids",               # centroids
        "look at the embedding space",          # embedding
        "look at the embeddings",               # embeddings
    ]

    def _assert_field_rejects(self, field_name, phrase):
        kw = {field_name: phrase}
        if field_name == "prohibitions":
            kw = {"prohibitions": [phrase]}
        req = _good_request(**kw)
        with self.assertRaises(RequestValidationError) as cm:
            validate_request(req)
        self.assertIn("forbidden term", str(cm.exception))

    def test_target_behavior_denylist(self):
        for phrase in self.FORBIDDEN_PHRASES:
            with self.subTest(phrase=phrase):
                self._assert_field_rejects("target_behavior", phrase)

    def test_success_criterion_denylist(self):
        for phrase in self.FORBIDDEN_PHRASES:
            with self.subTest(phrase=phrase):
                self._assert_field_rejects("success_criterion", phrase)

    def test_prohibitions_denylist(self):
        for phrase in self.FORBIDDEN_PHRASES:
            with self.subTest(phrase=phrase):
                self._assert_field_rejects("prohibitions", phrase)


class TestSerialization(unittest.TestCase):
    def test_to_operator_dict_strips_server_fields(self):
        req = _good_request(target_cluster=7, source_condition="graph")
        op = req.to_operator_dict()
        self.assertNotIn("target_cluster", op)
        self.assertNotIn("source_condition", op)
        # And the non-stripped variant keeps them
        full = req.to_dict()
        self.assertEqual(full["target_cluster"], 7)
        self.assertEqual(full["source_condition"], "graph")


class TestParseAndValidateBatch(unittest.TestCase):
    def _entry(self, **kw):
        e = {
            "request_type": "full_trajectory",
            "target_behavior": "grasp the cube and lift",
            "prohibitions": [],
            "success_criterion": "task_success",
            "initial_conditions": {
                "reference_rollout_id": "r0000",
                "reference_frame": 0,
            },
        }
        e.update(kw)
        return e

    def test_accepts_list_shape(self):
        out = parse_and_validate_batch(
            [self._entry(), self._entry()],
            source_condition="graph",
        )
        self.assertEqual(len(out), 2)
        for r in out:
            self.assertEqual(r.source_condition, "graph")
            # auto-fill request_id when missing
            self.assertTrue(r.request_id)

    def test_accepts_dict_with_requests_key(self):
        payload = {"requests": [self._entry(), self._entry(request_id="custom-id")]}
        out = parse_and_validate_batch(payload, source_condition="outcome_only")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].request_id, "custom-id")
        for r in out:
            self.assertEqual(r.source_condition, "outcome_only")

    def test_assigns_source_condition_overriding_input(self):
        payload = [self._entry()]
        # Even if entry contains source_condition, it is overwritten.
        payload[0]["source_condition"] = "outcome_only"
        out = parse_and_validate_batch(payload, source_condition="graph")
        self.assertEqual(out[0].source_condition, "graph")

    def test_string_input_decoded(self):
        raw = json.dumps([self._entry()])
        out = parse_and_validate_batch(raw, source_condition="graph")
        self.assertEqual(len(out), 1)


class TestRequestJsonSchema(unittest.TestCase):
    def test_with_target_cluster_required(self):
        s = request_json_schema(with_target_cluster=True)
        items = s["properties"]["requests"]["items"]
        self.assertIn("target_cluster", items["properties"])
        self.assertIn("target_cluster", items["required"])

    def test_without_target_cluster_excluded(self):
        s = request_json_schema(with_target_cluster=False)
        items = s["properties"]["requests"]["items"]
        self.assertNotIn("target_cluster", items["properties"])
        self.assertNotIn("target_cluster", items["required"])


if __name__ == "__main__":
    unittest.main()

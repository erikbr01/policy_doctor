"""Pipeline step: export Markov report from a synthetic clustering directory."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

try:
    import scipy  # noqa: F401

    from policy_doctor.behaviors.behavior_graph import (
        markov_test_result_to_jsonable,
        test_markov_property as run_markov_property_analysis,
    )

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    markov_test_result_to_jsonable = None  # type: ignore
    run_markov_property_analysis = None  # type: ignore
    _SCIPY_AVAILABLE = False


def _tiny_markov_labels_metadata(rng: np.random.RandomState):
    """Small rollout-labeled dataset for Markov tests."""
    P = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])
    labels_list = []
    metadata_list = []
    n_states = 3
    for ep in range(80):
        length = rng.randint(10, 21)
        state = int(rng.choice(n_states))
        success = rng.random() < 0.5
        for t in range(length):
            labels_list.append(state)
            metadata_list.append(
                {"rollout_idx": ep, "timestep": t, "success": success}
            )
            state = int(rng.choice(n_states, p=P[state]))
    return np.array(labels_list), metadata_list


@unittest.skipUnless(_SCIPY_AVAILABLE, "scipy required for Markov tests")
class TestExportMarkovReportStep(unittest.TestCase):
    def test_compute_writes_json(self):
        from policy_doctor.curation_pipeline.steps.export_markov_report import (
            ExportMarkovReportStep,
        )

        rng = np.random.RandomState(0)
        labels, metadata = _tiny_markov_labels_metadata(rng)

        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td) / "clustering"
            cdir.mkdir()
            np.save(cdir / "cluster_labels.npy", labels)
            manifest = {
                "level": "rollout",
                "algorithm": "kmeans",
                "n_clusters": 3,
                "n_samples": len(labels),
            }
            with open(cdir / "manifest.yaml", "w") as f:
                yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)
            with open(cdir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            run_dir = Path(td) / "pipeline_run"
            run_dir.mkdir()
            rc = run_dir / "run_clustering"
            rc.mkdir()
            with open(rc / "result.json", "w") as f:
                json.dump({"clustering_dirs": {"0": str(cdir)}}, f)

            cfg = OmegaConf.create(
                {
                    "repo_root": str(Path(td)),
                    "dry_run": False,
                    "seeds": [0],
                    "markov_export": {
                        "method": "chi2",
                        "significance_level": 0.05,
                        "exclude_terminals": True,
                    },
                    "env": "robomimic",
                    "task": None,
                }
            )
            step = ExportMarkovReportStep(cfg, run_dir)
            out = step.compute()
            self.assertIn("0", out["per_seed"])
            report = run_dir / "export_markov_report" / "markov_report_seed0.json"
            self.assertTrue(report.is_file())
            with open(report) as f:
                payload = json.load(f)
            self.assertIn("per_state", payload)
            self.assertIn("markov_holds", payload)


@unittest.skipUnless(_SCIPY_AVAILABLE, "scipy required for Markov tests")
class TestMarkovJsonable(unittest.TestCase):
    def test_roundtrip_keys(self):
        rng = np.random.RandomState(1)
        labels, metadata = _tiny_markov_labels_metadata(rng)
        raw = run_markov_property_analysis(
            labels, metadata, significance_level=0.05, exclude_terminals=True
        )
        js = markov_test_result_to_jsonable(raw)
        self.assertIn("per_state", js)
        self.assertIsInstance(js["per_state"], dict)
        json.dumps(js)


if __name__ == "__main__":
    unittest.main()

"""
Integration test: run the pipeline to recreate test_advantage_selection curation config
and compare slice set hash and dataset_fingerprint with the reference.

Strict test: uses the same dataset as the reference by reading eval_dir and train_dir
from reference metadata, so the recreated config must match fingerprint and slices.
The reference YAML must include in metadata: eval_dir, train_dir (paths that produced
the reference's dataset_fingerprint and total_raw_samples), and selection_method_metadata
with all pipeline parameters (global_top_k, window_width, aggregation_method, percentile).

Uses the same config format as the curation_pipeline runner: load base config from
``policy_doctor`` package ``configs/pipeline/config.yaml`` and merge overrides (task_config, clustering_dir,
policy_seeds, eval_dir, train_dir, etc.).
"""

import hashlib
import unittest

from pathlib import Path

from policy_doctor.paths import REPO_ROOT, iv_task_configs_base

_REPO_ROOT = REPO_ROOT
_IV_CFG = iv_task_configs_base()

# Clustering result used to create the reference: load from this folder
_REFERENCE_CLUSTERING_DIR = (
    _IV_CFG
    / "transport_mh_jan28"
    / "clustering"
    / "sliding_window_rollout_kmeans_k15_2026_03_05"
)


def _build_pipeline_cfg(overrides):
    """Build pipeline config the same way curation_pipeline runner does: base YAML + overrides."""
    from policy_doctor.curation_pipeline.config import get_pipeline_config_path
    import yaml
    path = get_pipeline_config_path()
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    cfg.update(overrides)
    return cfg


class TestRecreateCurationConfig(unittest.TestCase):
    """Run pipeline and compare recreated curation config with reference."""

    @classmethod
    def setUpClass(cls):
        cls.reference_path = (
            _IV_CFG
            / "transport_mh_jan28"
            / "curation"
            / "test_advantage_selection.yaml"
        )
        if _REFERENCE_CLUSTERING_DIR.is_dir() and (_REFERENCE_CLUSTERING_DIR / "manifest.yaml").exists():
            cls.clustering_dir = str(_REFERENCE_CLUSTERING_DIR)
        else:
            cls.clustering_dir = None

    def test_recreate_curation_config_matches_reference(self):
        """Run pipeline and assert slice set hash and fingerprint match reference."""
        if not self.reference_path.exists():
            self.skipTest(f"Reference config not found: {self.reference_path}")
        if not self.clustering_dir:
            self.skipTest(
                f"Clustering folder not found: {_REFERENCE_CLUSTERING_DIR}"
            )

        from policy_doctor.curation.config import load_curation_config_from_path
        from policy_doctor.curation_pipeline.steps.run_curation_config import run_pipeline_from_config

        reference = load_curation_config_from_path(self.reference_path)

        ref_eval = reference.metadata.get("eval_dir")
        ref_train = reference.metadata.get("train_dir")
        self.assertIsNotNone(
            ref_eval,
            "Reference metadata must include eval_dir so the test uses the same dataset.",
        )
        self.assertIsNotNone(
            ref_train,
            "Reference metadata must include train_dir so the test uses the same dataset.",
        )

        # Read all pipeline parameters from the reference so the test reproduces
        # the exact same result regardless of pipeline default changes.
        sel_meta = reference.metadata.get("selection_method_metadata") or {}
        per_slice_top_k = sel_meta.get("global_top_k", 100)
        use_all = per_slice_top_k >= 100
        overrides = {
            "task_config": "transport_mh_jan28",
            "config_root": "iv",
            "clustering_dir": self.clustering_dir,
            "policy_seeds": [0],
            "reference_seed": 0,
            "curation_output_name": "recreated_advantage_selection",
            "advantage_threshold": 0.1,
            "eval_dir": ref_eval,
            "train_dir": ref_train,
            "repo_root": str(_REPO_ROOT),
            "per_slice_top_k": per_slice_top_k,
            "use_all_demos_per_slice": use_all,
            "window_width": sel_meta.get("window_width", 5),
            "aggregation_method": sel_meta.get("aggregation_method", "mean"),
            "selection_percentile": sel_meta.get("percentile", 99.0),
        }
        cfg = _build_pipeline_cfg(overrides)
        try:
            out_paths = run_pipeline_from_config(cfg)
        except RuntimeError as e:
            if "influence_visualizer" in str(e) or "Influence data loading" in str(e):
                self.skipTest(f"Influence data loading requires influence_visualizer: {e}")
            raise

        self.assertIsInstance(out_paths, list, "run_pipeline_from_config returns list of paths per seed")
        self.assertGreater(len(out_paths), 0, "At least one curation config path expected")
        out_path = Path(out_paths[0]) if not isinstance(out_paths[0], Path) else out_paths[0]
        self.assertTrue(out_path.exists(), f"Pipeline did not write {out_path}")

        recreated = load_curation_config_from_path(out_path)

        # Clean up the output file so test runs don't accumulate artifacts
        try:
            out_path.unlink()
        except OSError:
            pass

        self.assertEqual(
            recreated.metadata.get("dataset_fingerprint"),
            reference.metadata["dataset_fingerprint"],
            "Dataset fingerprint must match reference (same dataset via eval_dir/train_dir).",
        )
        self.assertEqual(
            recreated.metadata.get("total_raw_samples"),
            reference.metadata["total_raw_samples"],
            "total_raw_samples must match reference (same dataset via eval_dir/train_dir).",
        )

        # Granular diagnostics: compare slices structurally before hash check
        ref_tuples = sorted((s.episode_idx, s.start, s.end) for s in reference.slices)
        rec_tuples = sorted((s.episode_idx, s.start, s.end) for s in recreated.slices)

        self.assertEqual(
            len(rec_tuples),
            len(ref_tuples),
            f"Slice count mismatch: recreated {len(rec_tuples)} vs reference {len(ref_tuples)}. "
            f"Recreated slice sizes: {set(e - s for _, s, e in rec_tuples)}, "
            f"Reference slice sizes: {set(e - s for _, s, e in ref_tuples)}. "
            "If recreated has many more same-size slices, merge_overlapping_slices may be missing.",
        )

        ref_episodes = sorted(set(t[0] for t in ref_tuples))
        rec_episodes = sorted(set(t[0] for t in rec_tuples))
        self.assertEqual(
            rec_episodes,
            ref_episodes,
            f"Episode set mismatch: {len(rec_episodes)} recreated vs {len(ref_episodes)} reference episodes.",
        )

        ref_set = set(ref_tuples)
        rec_set = set(rec_tuples)
        ref_only = sorted(ref_set - rec_set)
        rec_only = sorted(rec_set - ref_set)
        if ref_only or rec_only:
            msg_parts = [f"Slice tuple mismatch: {len(ref_only)} ref-only, {len(rec_only)} rec-only."]
            if ref_only:
                msg_parts.append(f"  Samples ref-only: {ref_only[:5]}")
            if rec_only:
                msg_parts.append(f"  Samples rec-only: {rec_only[:5]}")
            self.fail("\n".join(msg_parts))

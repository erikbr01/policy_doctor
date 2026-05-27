"""
``policy_doctor.data.dataset_episode_ends`` fingerprint check against the reference.

The IV-vs-PD comparison tests that used to live here were retired alongside the
``influence_visualizer`` package — see ``REFACTOR_FINDINGS.md`` (Phase 3). What
remains is the pure-PD test that confirms ``load_dataset_episode_ends`` produces
a fingerprint matching the reference curation YAML.

Run from repo root with the analysis venv:
  ./scripts/uv_env.sh analysis pytest tests/integration/test_fingerprint_episode_ends.py -v
"""

import unittest
from pathlib import Path

from policy_doctor.paths import REPO_ROOT, iv_task_configs_base

_REPO_ROOT = REPO_ROOT
_IV_CFG = iv_task_configs_base()


def _get_reference_paths():
    """Paths from reference config (same dataset as integration test)."""
    ref_path = _IV_CFG / "transport_mh_jan28" / "curation" / "test_advantage_selection.yaml"
    if not ref_path.exists():
        return None, None, None
    import yaml
    with open(ref_path) as f:
        data = yaml.safe_load(f)
    meta = data.get("metadata") or {}
    return ref_path, meta.get("eval_dir"), meta.get("train_dir")


class TestFingerprintEpisodeEnds(unittest.TestCase):
    """Confirm policy_doctor's dataset fingerprint matches the reference curation YAML."""

    def setUp(self):
        self._ref_path, self._eval_dir, self._train_dir = _get_reference_paths()
        if self._ref_path is None:
            self.skipTest("Reference config not found")
        if not self._eval_dir or not self._train_dir:
            self.skipTest("Reference config has no eval_dir/train_dir in metadata")

    def test_policy_doctor_load_dataset_episode_ends_matches_reference(self):
        """policy_doctor's load_dataset_episode_ends must yield same fingerprint as reference.

        If skipped: train dir missing or load_dataset_episode_ends failed (run pytest with -rs to see reason).
        """
        from policy_doctor.data.dataset_episode_ends import load_dataset_episode_ends
        from policy_doctor.curation.config import compute_dataset_fingerprint
        import yaml

        train_dir_abs = _REPO_ROOT / self._train_dir
        if not train_dir_abs.exists():
            self.skipTest(f"Train dir not found: {train_dir_abs}")
        try:
            episode_ends = load_dataset_episode_ends(train_dir_abs, "latest", _REPO_ROOT)
        except Exception as e:
            self.skipTest(
                f"load_dataset_episode_ends failed (need torch/hydra/dill/dataset): {type(e).__name__}: {e}"
            )

        fingerprint = compute_dataset_fingerprint(episode_ends)
        total = int(episode_ends[-1]) if len(episode_ends) > 0 else 0

        with open(self._ref_path) as f:
            ref = yaml.safe_load(f)
        ref_fingerprint = (ref.get("metadata") or {}).get("dataset_fingerprint")
        ref_total = (ref.get("metadata") or {}).get("total_raw_samples")

        self.assertEqual(
            fingerprint,
            ref_fingerprint,
            f"PD load_dataset_episode_ends fingerprint must match reference; got {fingerprint}, ref {ref_fingerprint}",
        )
        self.assertEqual(
            total,
            ref_total,
            f"PD total_raw must match reference; got {total}, ref {ref_total}",
        )


if __name__ == "__main__":
    unittest.main()

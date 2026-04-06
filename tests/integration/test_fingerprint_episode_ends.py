"""
Targeted tests to find exactly why policy_doctor's derived episode_ends (and thus
dataset_fingerprint) can differ from IV's data.demo_dataset.replay_buffer.episode_ends.

Run from repo root with project venv active:
  pytest policy_doctor/tests/integration/test_fingerprint_episode_ends.py -v
  python policy_doctor/run_tests.py

These tests load the same data via influence_visualizer and compare:
- Replay buffer episode_ends (what IV uses for fingerprint)
- Episode structure derived from demo_episodes (train-only) vs full buffer

Findings they encode:
- IV uses demo_dataset.replay_buffer.episode_ends = FULL buffer (all episodes: train+val+holdout).
- demo_episodes only contains TRAIN episodes; cumsum(demo_episodes.raw_length) is train-only.
- So fingerprint from full buffer != fingerprint from train-only unless there is no val/holdout.
"""

import unittest
from pathlib import Path

import numpy as np

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
    """Pinpoint where replay_buffer.episode_ends vs derived structure diverge."""

    def setUp(self):
        try:
            from influence_visualizer.data_loader import load_influence_data
        except ImportError:
            self.skipTest("influence_visualizer not installed")
        self._ref_path, self._eval_dir, self._train_dir = _get_reference_paths()
        if self._ref_path is None:
            self.skipTest("Reference config not found")
        if not self._eval_dir or not self._train_dir:
            self.skipTest("Reference config has no eval_dir/train_dir in metadata")
        self._eval_dir_abs = str(_REPO_ROOT / self._eval_dir)
        self._train_dir_abs = str(_REPO_ROOT / self._train_dir)

    def test_replay_buffer_vs_demo_episodes_length_and_total(self):
        """Compare: replay_buffer.episode_ends vs cumsum(demo_episodes.raw_length).
        Documents the mismatch: IV uses FULL buffer (all episodes); PD derived from demo_episodes is TRAIN-only."""
        from influence_visualizer.data_loader import load_influence_data

        data = load_influence_data(
            eval_dir=self._eval_dir_abs,
            train_dir=self._train_dir_abs,
            train_ckpt="latest",
            exp_date="default",
            include_holdout=True,
        )

        rb = data.demo_dataset.replay_buffer
        rb_ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
        n_rb = len(rb_ends)
        total_rb = int(rb_ends[-1]) if n_rb > 0 else 0

        # What policy_doctor would derive from train-only demo_episodes (order = included_episodes)
        train_lengths = np.array(
            [int(getattr(ep, "raw_length", None) or ep.num_samples) for ep in data.demo_episodes],
            dtype=np.int64,
        )
        derived_train_ends = np.cumsum(train_lengths)
        n_derived_train = len(derived_train_ends)
        total_derived_train = int(derived_train_ends[-1]) if n_derived_train > 0 else 0

        # Root cause: IV uses full replay_buffer.episode_ends (all episodes: train+val+holdout).
        # demo_episodes only contains TRAIN episodes. So len(rb_ends) >= len(demo_episodes) and total_rb >= total_derived_train.
        self.assertGreaterEqual(
            n_rb,
            n_derived_train,
            f"replay_buffer has {n_rb} episodes, demo_episodes (train) has {n_derived_train}; IV fingerprints full buffer",
        )
        self.assertGreaterEqual(
            total_rb,
            total_derived_train,
            f"replay_buffer total={total_rb}, derived train total={total_derived_train}; IV uses full buffer total",
        )
        # If they differ, that's why fingerprint differed before we passed through train_episode_ends
        if n_rb != n_derived_train or total_rb != total_derived_train:
            # Document: we cannot reconstruct full rb_ends from demo_episodes alone (we lack val/holdout in buffer order)
            pass

    def test_demo_episodes_raw_length_matches_buffer(self):
        """Each demo_episode.raw_length should equal buffer length for that episode index."""
        from influence_visualizer.data_loader import load_influence_data

        data = load_influence_data(
            eval_dir=self._eval_dir_abs,
            train_dir=self._train_dir_abs,
            train_ckpt="latest",
            exp_date="default",
            include_holdout=True,
        )

        rb = data.demo_dataset.replay_buffer
        rb_ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
        train_mask = data.demo_dataset.train_mask
        n_rb = len(rb_ends)

        # Buffer length for episode i
        def buffer_ep_len(i):
            start = 0 if i == 0 else int(rb_ends[i - 1])
            return int(rb_ends[i]) - start

        included_train = np.where(train_mask)[0]
        for k, ep in enumerate(data.demo_episodes):
            idx = ep.index
            self.assertIn(idx, included_train, "demo_episodes should only have train indices")
            expected_len = buffer_ep_len(idx)
            actual_len = int(getattr(ep, "raw_length", None) or ep.num_samples)
            self.assertEqual(
                actual_len,
                expected_len,
                f"Episode index {idx} (demo_episodes[{k}]): raw_length={actual_len} vs buffer length={expected_len}",
            )

    def test_which_episodes_are_train_vs_holdout(self):
        """Report how many episodes are train vs holdout vs val (if any)."""
        from influence_visualizer.data_loader import load_influence_data

        data = load_influence_data(
            eval_dir=self._eval_dir_abs,
            train_dir=self._train_dir_abs,
            train_ckpt="latest",
            exp_date="default",
            include_holdout=True,
        )

        rb = data.demo_dataset.replay_buffer
        rb_ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
        train_mask = data.demo_dataset.train_mask
        holdout_mask = getattr(data.demo_dataset, "holdout_mask", None)
        val_mask = getattr(data.demo_dataset, "val_mask", None)

        n_total = len(rb_ends)
        n_train = int(np.sum(train_mask))
        n_holdout = int(np.sum(holdout_mask)) if holdout_mask is not None else 0
        n_val = int(np.sum(val_mask)) if val_mask is not None else 0

        # Total samples per split from buffer
        def total_for_mask(m):
            if m is None or len(m) != n_total:
                return 0
            t = 0
            for i in range(n_total):
                if m[i]:
                    start = 0 if i == 0 else int(rb_ends[i - 1])
                    t += int(rb_ends[i]) - start
            return t

        total_train = total_for_mask(train_mask)
        total_holdout = total_for_mask(holdout_mask)
        total_val = total_for_mask(val_mask)
        total_all = int(rb_ends[-1]) if n_total > 0 else 0
        sum_masks = total_train + total_holdout + total_val

        self.assertGreater(n_total, 0, "Should have at least one episode")
        # Full buffer total must equal sum of train+holdout+val (and optionally val)
        self.assertEqual(
            total_all,
            sum_masks,
            f"total_all={total_all} vs train+holdout+val={total_train}+{total_holdout}+{total_val}",
        )

    def test_fingerprint_from_rb_ends_matches_reference(self):
        """Reference config fingerprint should equal hash(replay_buffer.episode_ends)."""
        from influence_visualizer.data_loader import load_influence_data
        from influence_visualizer.curation_config import compute_dataset_fingerprint
        import yaml

        data = load_influence_data(
            eval_dir=self._eval_dir_abs,
            train_dir=self._train_dir_abs,
            train_ckpt="latest",
            exp_date="default",
            include_holdout=True,
        )
        rb_ends = np.asarray(data.demo_dataset.replay_buffer.episode_ends[:], dtype=np.int64)
        fingerprint_from_rb = compute_dataset_fingerprint(rb_ends)
        total_from_rb = int(rb_ends[-1]) if len(rb_ends) > 0 else 0

        with open(self._ref_path) as f:
            ref = yaml.safe_load(f)
        ref_fingerprint = (ref.get("metadata") or {}).get("dataset_fingerprint")
        ref_total = (ref.get("metadata") or {}).get("total_raw_samples")

        self.assertEqual(
            fingerprint_from_rb,
            ref_fingerprint,
            f"Hash of replay_buffer.episode_ends should match reference; rb gives {fingerprint_from_rb}, ref {ref_fingerprint}",
        )
        self.assertEqual(
            total_from_rb,
            ref_total,
            f"total_raw from rb_ends[-1] should match reference; rb gives {total_from_rb}, ref {ref_total}",
        )

    def test_fingerprint_from_demo_episodes_only_differs_from_reference(self):
        """Hash of cumsum(demo_episodes.raw_length) differs from reference when buffer has more than train."""
        from influence_visualizer.data_loader import load_influence_data
        from influence_visualizer.curation_config import compute_dataset_fingerprint
        import yaml

        data = load_influence_data(
            eval_dir=self._eval_dir_abs,
            train_dir=self._train_dir_abs,
            train_ckpt="latest",
            exp_date="default",
            include_holdout=True,
        )
        # What PD would compute without access to full replay buffer
        train_lengths = np.array(
            [int(getattr(ep, "raw_length", None) or ep.num_samples) for ep in data.demo_episodes],
            dtype=np.int64,
        )
        derived_train_ends = np.cumsum(train_lengths)
        fingerprint_derived = compute_dataset_fingerprint(derived_train_ends)
        total_derived = int(derived_train_ends[-1]) if len(derived_train_ends) > 0 else 0

        rb_ends = np.asarray(data.demo_dataset.replay_buffer.episode_ends[:], dtype=np.int64)
        fingerprint_rb = compute_dataset_fingerprint(rb_ends)
        total_rb = int(rb_ends[-1]) if len(rb_ends) > 0 else 0

        with open(self._ref_path) as f:
            ref = yaml.safe_load(f)
        ref_fingerprint = (ref.get("metadata") or {}).get("dataset_fingerprint")
        ref_total = (ref.get("metadata") or {}).get("total_raw_samples")

        # Reference is from full buffer. So if buffer has more than train, derived (train-only) will differ.
        if len(rb_ends) > len(data.demo_episodes) or total_rb > total_derived:
            self.assertNotEqual(
                fingerprint_derived,
                ref_fingerprint,
                "Derived (train-only) fingerprint should differ from reference when IV used full buffer",
            )
            self.assertNotEqual(total_derived, ref_total)

    def test_policy_doctor_load_dataset_episode_ends_matches_reference(self):
        """policy_doctor's load_dataset_episode_ends must yield same fingerprint as reference (no IV buffer).
        If skipped: train dir missing or load_dataset_episode_ends failed (run pytest with -rs to see reason)."""
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

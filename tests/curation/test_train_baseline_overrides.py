"""Unit tests for TrainBaselineStep._subprocess_overrides."""

from __future__ import annotations

import unittest


class TestSubprocessOverrides(unittest.TestCase):
    """Verify the static helper that builds Hydra overrides for subprocess training.

    Subprocess mode (non-cupid conda env) omits HDF5-specific dataset_mask_kwargs
    that only apply to RobomimicReplayLowdimDataset and similar HDF5-backed classes.
    """

    def _call(self, **kw):
        from policy_doctor.curation_pipeline.steps.train_baseline import TrainBaselineStep

        defaults = dict(
            exp_name="test_exp",
            device="cuda:0",
            seed=0,
            num_epochs=100,
            checkpoint_topk=3,
            checkpoint_every=50,
            val_ratio=0.04,
            train_name="default_train",
            train_date="2026-01-01",
            task="square_mh",
            policy="diffusion_policy_cnn",
            project="test_project",
            run_output_dir="/tmp/runs/test",
        )
        defaults.update(kw)
        return TrainBaselineStep._subprocess_overrides(**defaults)

    def test_returns_list_of_strings(self):
        result = self._call()
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(s, str) for s in result))

    def test_length_14(self):
        result = self._call()
        self.assertEqual(len(result), 14)

    def test_key_fields_present(self):
        result = self._call(
            exp_name="my_exp",
            device="cuda:1",
            seed=42,
            num_epochs=500,
            val_ratio=0.1,
            run_output_dir="/tmp/out",
        )
        joined = " ".join(result)
        self.assertIn("name=my_exp", joined)
        self.assertIn("training.device=cuda:1", joined)
        self.assertIn("training.seed=42", joined)
        self.assertIn("training.num_epochs=500", joined)
        self.assertIn("task.dataset.val_ratio=0.1", joined)
        self.assertIn("multi_run.run_dir=/tmp/out", joined)

    def test_no_dataset_mask_kwargs(self):
        """Subprocess overrides must NOT include dataset_mask_kwargs (HDF5-only)."""
        result = self._call()
        joined = " ".join(result)
        self.assertNotIn("dataset_mask_kwargs", joined)
        self.assertNotIn("train_ratio", joined)
        self.assertNotIn("uniform_quality", joined)

    def test_logging_fields(self):
        result = self._call(
            train_name="myrun",
            train_date="2026-04-08",
            exp_name="exp",
            task="square_mh",
            project="corl",
        )
        joined = " ".join(result)
        self.assertIn("logging.name=myrun", joined)
        self.assertIn("logging.project=corl", joined)
        self.assertIn("multi_run.wandb_name_base=myrun", joined)


if __name__ == "__main__":
    unittest.main()

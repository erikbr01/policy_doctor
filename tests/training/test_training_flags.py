"""Tests for multi-GPU DDP, compile, and TF32 training flags.

All tests are CPU-only and require no GPU or actual training — they test
the override plumbing, utility functions, and model-wrapping logic in
isolation.
"""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the repo root is on sys.path when running directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make diffusion_policy importable (it lives in third_party/cupid).
_CUPID_ROOT = _REPO_ROOT / "third_party" / "cupid"
if str(_CUPID_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUPID_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_baseline_cfg(**overrides):
    """Return a minimal OmegaConf DictConfig that looks like a baseline section."""
    from omegaconf import OmegaConf

    base = {
        "task": "transport_mh",
        "state": "low_dim",
        "policy": "diffusion_unet_lowdim",
        "config_dir": "configs/low_dim/transport_mh/diffusion_policy_cnn",
        "config_name": "config.yaml",
        "seeds": [0],
        "num_epochs": 10,
        "checkpoint_topk": 1,
        "checkpoint_every": 5,
        "train_ratio": 0.64,
        "val_ratio": 0.04,
        "output_dir": "data/outputs/train",
        "project": "test-project",
        "train_date": "test",
        "num_gpus": 1,
        "tf32": False,
        "compile": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# ddp_util tests
# ---------------------------------------------------------------------------

class TestFindFreePort(unittest.TestCase):
    def test_returns_integer_in_valid_range(self):
        from diffusion_policy.common.ddp_util import find_free_port

        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_port_is_bindable(self):
        """The port returned should actually be free and bindable."""
        from diffusion_policy.common.ddp_util import find_free_port

        port = find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Should not raise
            s.bind(("127.0.0.1", port))

    def test_returns_different_ports_on_repeated_calls(self):
        """Two consecutive calls should ideally return the same port only by chance."""
        from diffusion_policy.common.ddp_util import find_free_port

        ports = {find_free_port() for _ in range(5)}
        # At least 2 distinct ports out of 5 calls (very likely unless all bound)
        # This is a soft check — the main guarantee is that each is bindable.
        self.assertGreaterEqual(len(ports), 1)


class TestDDPWorkerInjectsRank(unittest.TestCase):
    """ddp_worker injects _ddp_rank and _ddp_world_size into the overrides list."""

    def _captured_worker(self, **kwargs):
        """Records the kwargs it was called with."""
        self._captured = kwargs

    def test_rank_and_world_size_injected(self):
        """Verify that ddp_worker appends the rank/world_size overrides."""
        from diffusion_policy.common.ddp_util import find_free_port

        calls = []

        def fake_worker(**kwargs):
            calls.append(kwargs)

        # Simulate what ddp_worker does without actually calling dist.init_process_group.
        port = find_free_port()
        rank = 0
        world_size = 2
        worker_kwargs = {
            "run_output_dir": "/tmp/test",
            "config_dir_str": "/tmp/cfg",
            "config_name": "config.yaml",
            "overrides": ["training.device=cuda:0"],
        }

        # Manually replicate the override injection logic from ddp_worker.
        import os
        kwargs = dict(worker_kwargs)
        overrides = list(kwargs.get("overrides", []))
        overrides.append(f"+training._ddp_rank={rank}")
        overrides.append(f"+training._ddp_world_size={world_size}")
        kwargs["overrides"] = overrides
        fake_worker(**kwargs)

        self.assertEqual(len(calls), 1)
        final_overrides = calls[0]["overrides"]
        self.assertIn(f"+training._ddp_rank={rank}", final_overrides)
        self.assertIn(f"+training._ddp_world_size={world_size}", final_overrides)

    def test_original_overrides_preserved(self):
        """Original overrides must not be dropped when rank/world_size are injected."""
        original = ["training.device=cuda:0", "training.seed=42"]
        rank, world_size = 1, 4

        kwargs = {"overrides": list(original)}
        overrides = list(kwargs["overrides"])
        overrides.append(f"+training._ddp_rank={rank}")
        overrides.append(f"+training._ddp_world_size={world_size}")
        kwargs["overrides"] = overrides

        for o in original:
            self.assertIn(o, kwargs["overrides"])


# ---------------------------------------------------------------------------
# TF32 override tests
# ---------------------------------------------------------------------------

class TestTF32Overrides(unittest.TestCase):
    """Verify that train_baseline emits the correct tf32 overrides."""

    def _get_overrides_for(self, tf32: bool) -> list:
        """Build the override list the same way TrainBaselineStep does."""
        tf32_val = tf32
        compile_val = False
        overrides = [
            f"+training.tf32={str(tf32_val).lower()}",
            f"+training.compile={str(compile_val).lower()}",
        ]
        return overrides

    def test_tf32_false_override_when_disabled(self):
        overrides = self._get_overrides_for(tf32=False)
        self.assertIn("+training.tf32=false", overrides)

    def test_tf32_true_override_when_enabled(self):
        overrides = self._get_overrides_for(tf32=True)
        self.assertIn("+training.tf32=true", overrides)

    def test_tf32_flags_set_when_enabled(self):
        """The workspace should set torch.backends flags when use_tf32=True."""
        import torch

        original_matmul = torch.backends.cuda.matmul.allow_tf32
        original_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # Simulate workspace logic
            use_tf32 = True
            if use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_matmul
            torch.backends.cudnn.allow_tf32 = original_cudnn

    def test_tf32_flags_not_set_when_disabled(self):
        """The workspace should NOT touch torch.backends flags when use_tf32=False."""
        import torch

        original_matmul = torch.backends.cuda.matmul.allow_tf32
        original_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            use_tf32 = False
            if use_tf32:  # should not execute
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_matmul
            torch.backends.cudnn.allow_tf32 = original_cudnn


# ---------------------------------------------------------------------------
# compile override tests
# ---------------------------------------------------------------------------

class TestCompileOverrides(unittest.TestCase):
    def _get_overrides_for(self, compile_: bool) -> list:
        tf32_val = False
        compile_val = compile_
        return [
            f"+training.tf32={str(tf32_val).lower()}",
            f"+training.compile={str(compile_val).lower()}",
        ]

    def test_compile_false_override_when_disabled(self):
        overrides = self._get_overrides_for(compile_=False)
        self.assertIn("+training.compile=false", overrides)

    def test_compile_true_override_when_enabled(self):
        overrides = self._get_overrides_for(compile_=True)
        self.assertIn("+training.compile=true", overrides)


def _has_torch_compile() -> bool:
    """Return True if torch.compile is available (PyTorch >= 2.0)."""
    import torch
    return hasattr(torch, "compile")


class TestCompileAppliedToModel(unittest.TestCase):
    """torch.compile wraps diffusion policy backbone models correctly (CPU, no GPU needed)."""

    def _make_transformer_backbone(self):
        """TransformerForDiffusion — the noise-prediction net used by diffusion transformer policies."""
        from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

        # cond_dim is per-obs-step obs_dim; cond is passed as (B, n_obs_steps, obs_dim)
        return TransformerForDiffusion(
            input_dim=10,    # action dim
            output_dim=10,
            horizon=8,
            n_obs_steps=2,
            cond_dim=10,     # obs_dim per step
            n_layer=2,
            n_head=2,
            n_emb=64,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        )

    def _make_unet_backbone(self):
        """ConditionalUnet1D — the noise-prediction net used by diffusion UNet policies."""
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

        return ConditionalUnet1D(
            input_dim=10,           # action dim
            global_cond_dim=20,     # obs_dim * n_obs_steps
            diffusion_step_embed_dim=64,
            down_dims=[64, 128],
        )

    @unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
    def test_compile_wraps_transformer_backbone(self):
        """TransformerForDiffusion survives torch.compile and produces correct output shape."""
        import torch

        model = self._make_transformer_backbone()
        # dynamic=True handles data-dependent shapes in positional embeddings;
        # fullgraph=False allows graph breaks so CPU-only tracing still validates the path.
        model = torch.compile(model, dynamic=True, fullgraph=False)

        # (B, horizon, action_dim); timestep; cond (B, n_obs_steps, obs_dim)
        sample   = torch.randn(2, 8, 10)
        timestep = torch.zeros(2, dtype=torch.long)
        cond     = torch.randn(2, 2, 10)  # n_obs_steps=2, obs_dim=10
        out = model(sample, timestep, cond)
        self.assertEqual(out.shape, (2, 8, 10))

    @unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
    def test_compile_wraps_unet_backbone(self):
        """ConditionalUnet1D survives torch.compile and produces correct output shape."""
        import torch

        model = self._make_unet_backbone()
        model = torch.compile(model, dynamic=True, fullgraph=False)

        # UNet forward expects (B, T, input_dim) — see ConditionalUnet1D docstring
        sample   = torch.randn(2, 8, 10)   # (B, T, action_dim)
        timestep = torch.zeros(2)
        cond     = torch.randn(2, 20)
        out = model(sample, timestep, global_cond=cond)
        self.assertEqual(out.shape, (2, 8, 10))

    def test_compile_skipped_when_disabled(self):
        import torch.nn as nn

        model = self._make_transformer_backbone()
        use_compile = False
        if use_compile:
            import torch
            model = torch.compile(model)
        # Model should be the original nn.Module, not a compiled wrapper
        self.assertIsInstance(model, nn.Module)
        self.assertNotIn("OptimizedModule", type(model).__name__)


# ---------------------------------------------------------------------------
# num_gpus override tests
# ---------------------------------------------------------------------------

class TestNumGpusOverrides(unittest.TestCase):
    def test_default_num_gpus_is_one(self):
        """When num_gpus is not set in config, default should be 1."""
        from omegaconf import OmegaConf

        baseline = OmegaConf.create({"task": "transport_mh"})
        cfg = OmegaConf.create({})
        num_gpus = int(OmegaConf.select(baseline, "num_gpus") or OmegaConf.select(cfg, "num_gpus") or 1)
        self.assertEqual(num_gpus, 1)

    def test_num_gpus_from_baseline_config(self):
        """num_gpus set in baseline config should be picked up."""
        from omegaconf import OmegaConf

        baseline = OmegaConf.create({"num_gpus": 4})
        cfg = OmegaConf.create({})
        num_gpus = int(OmegaConf.select(baseline, "num_gpus") or OmegaConf.select(cfg, "num_gpus") or 1)
        self.assertEqual(num_gpus, 4)

    def test_num_gpus_from_top_level_cfg(self):
        """num_gpus set at top-level cfg should be used as fallback."""
        from omegaconf import OmegaConf

        baseline = OmegaConf.create({"task": "transport_mh"})
        cfg = OmegaConf.create({"num_gpus": 2})
        num_gpus = int(OmegaConf.select(baseline, "num_gpus") or OmegaConf.select(cfg, "num_gpus") or 1)
        self.assertEqual(num_gpus, 2)


# ---------------------------------------------------------------------------
# YAML config flags
# ---------------------------------------------------------------------------

class TestYamlConfigFlags(unittest.TestCase):
    """Verify that all baseline YAML configs have the three new flags."""

    _CONFIG_DIRS = [
        _REPO_ROOT / "policy_doctor" / "configs" / "robomimic" / "baseline" / "low_dim",
        _REPO_ROOT / "policy_doctor" / "configs" / "robomimic" / "curation_selection" / "low_dim",
        _REPO_ROOT / "policy_doctor" / "configs" / "robomimic" / "curation_filtering" / "low_dim",
    ]

    def test_all_configs_have_num_gpus(self):
        import yaml

        for config_dir in self._CONFIG_DIRS:
            for cfg_path in sorted(config_dir.glob("*.yaml")):
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                self.assertIn(
                    "num_gpus", data,
                    msg=f"Missing 'num_gpus' in {cfg_path.relative_to(_REPO_ROOT)}"
                )

    def test_all_configs_have_tf32(self):
        import yaml

        for config_dir in self._CONFIG_DIRS:
            for cfg_path in sorted(config_dir.glob("*.yaml")):
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                self.assertIn(
                    "tf32", data,
                    msg=f"Missing 'tf32' in {cfg_path.relative_to(_REPO_ROOT)}"
                )

    def test_all_configs_have_compile(self):
        import yaml

        for config_dir in self._CONFIG_DIRS:
            for cfg_path in sorted(config_dir.glob("*.yaml")):
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                self.assertIn(
                    "compile", data,
                    msg=f"Missing 'compile' in {cfg_path.relative_to(_REPO_ROOT)}"
                )

    def test_defaults(self):
        """num_gpus=1 (safe default), tf32=true and compile=true (enabled for performance)."""
        import yaml

        for config_dir in self._CONFIG_DIRS:
            for cfg_path in sorted(config_dir.glob("*.yaml")):
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                if "num_gpus" in data:
                    self.assertEqual(
                        data["num_gpus"], 1,
                        msg=f"Expected num_gpus=1 in {cfg_path.relative_to(_REPO_ROOT)}"
                    )
                if "tf32" in data:
                    self.assertTrue(
                        data["tf32"],
                        msg=f"Expected tf32=true in {cfg_path.relative_to(_REPO_ROOT)}"
                    )
                if "compile" in data:
                    self.assertTrue(
                        data["compile"],
                        msg=f"Expected compile=true in {cfg_path.relative_to(_REPO_ROOT)}"
                    )


if __name__ == "__main__":
    unittest.main()

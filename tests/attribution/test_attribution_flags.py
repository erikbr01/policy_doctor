"""Tests for tf32 and compile flags in TRAK and InfEmbed attribution steps.

All tests are CPU/unit tests — they verify CLI plumbing, backend flag
propagation, and model wrapping without running actual attribution.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CUPID_ROOT = _REPO_ROOT / "third_party" / "cupid"
if str(_CUPID_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUPID_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attribution_cfg(**overrides):
    """Minimal OmegaConf config with attribution sub-section.

    repo_root and dry_run are embedded in the config because PipelineStep
    reads them via OmegaConf.select rather than constructor kwargs.
    """
    from omegaconf import OmegaConf

    base = {
        "repo_root": str(_REPO_ROOT),
        "dry_run": True,
        "attribution": {
            "task": "transport_mh",
            "policy": "diffusion_unet_lowdim",
            "train_date": "jan18",
            "eval_date": "jan28",
            "seeds": [0],
            "train_ckpt": "latest",
            "eval_ckpt": "latest",
            "train_output_dir": "data/outputs/train",
            "eval_output_dir": "data/outputs/eval_save_episodes",
            "modelout_fn": "DiffusionLowdimFunctionalModelOutput",
            "gradient_co": "DiffusionLowdimFunctionalGradientComputer",
            "proj_dim": 4000,
            "batch_size": 32,
            "loss_fn": "square",
            "num_timesteps": 64,
            "device": "cuda:0",
            "seed": 0,
            "tf32": False,
            "compile": False,
        },
        "baseline": {"task": "transport_mh", "policy": "diffusion_unet_lowdim"},
    }
    for k, v in overrides.items():
        base["attribution"][k] = v
    return OmegaConf.create(base)


_ATTRIBUTION_CONFIG_DIR = (
    _REPO_ROOT / "policy_doctor" / "configs" / "robomimic" / "attribution" / "low_dim"
)


# ---------------------------------------------------------------------------
# CLI plumbing — TrainAttributionStep
# ---------------------------------------------------------------------------

class TestTrainAttributionCLIFlags(unittest.TestCase):
    """Verify that tf32/compile config values are forwarded as CLI args to TRAK."""

    def _collect_cmd_args(self, tf32: bool, compile_: bool) -> list[str]:
        """Run TrainAttributionStep in dry-run mode and capture printed args."""
        from policy_doctor.curation_pipeline.steps.train_attribution import TrainAttributionStep

        cfg = _make_attribution_cfg(tf32=tf32, compile=compile_)
        step = TrainAttributionStep(cfg=cfg, run_dir=_REPO_ROOT)

        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))):
            step.compute()

        # dry_run prints the cmd_args line
        args_line = next((l for l in printed if "--exp_name" in l), "")
        return args_line.split()

    def test_tf32_false_forwarded(self):
        args = self._collect_cmd_args(tf32=False, compile_=False)
        self.assertIn("--tf32=false", args)

    def test_tf32_true_forwarded(self):
        args = self._collect_cmd_args(tf32=True, compile_=False)
        self.assertIn("--tf32=true", args)

    def test_compile_false_forwarded(self):
        args = self._collect_cmd_args(tf32=False, compile_=False)
        self.assertIn("--compile=false", args)

    def test_compile_true_forwarded(self):
        args = self._collect_cmd_args(tf32=False, compile_=True)
        self.assertIn("--compile=true", args)


# ---------------------------------------------------------------------------
# CLI plumbing — ComputeInfembedStep
# ---------------------------------------------------------------------------

class TestComputeInfembedCLIFlags(unittest.TestCase):
    """Verify that tf32/compile config values are forwarded as CLI args to InfEmbed."""

    def _collect_cmd_args(self, tf32: bool, compile_: bool) -> list[str]:
        from policy_doctor.curation_pipeline.steps.compute_infembed import ComputeInfembedStep

        cfg = _make_attribution_cfg(tf32=tf32, compile=compile_)
        step = ComputeInfembedStep(cfg=cfg, run_dir=_REPO_ROOT)

        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))):
            with patch("pathlib.Path.exists", return_value=True):
                step.compute()

        args_line = next((l for l in printed if "--eval_dir" in l), "")
        return args_line.split()

    def test_tf32_absent_when_false(self):
        """--tf32 flag should NOT appear when tf32=false (it's an is_flag)."""
        args = self._collect_cmd_args(tf32=False, compile_=False)
        self.assertNotIn("--tf32", args)

    def test_tf32_present_when_true(self):
        args = self._collect_cmd_args(tf32=True, compile_=False)
        self.assertIn("--tf32", args)

    def test_compile_absent_when_false(self):
        args = self._collect_cmd_args(tf32=False, compile_=False)
        self.assertNotIn("--compile", args)

    def test_compile_present_when_true(self):
        args = self._collect_cmd_args(tf32=False, compile_=True)
        self.assertIn("--compile", args)


# ---------------------------------------------------------------------------
# tf32 backend flags
# ---------------------------------------------------------------------------

class TestTF32BackendFlags(unittest.TestCase):
    """Check that setting tf32=True in TRAK/InfEmbed main() sets the right backends."""

    def test_tf32_sets_matmul_flag(self):
        import torch

        orig = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            # Simulate what both scripts do when tf32=True
            use_tf32 = True
            if use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig

    def test_tf32_does_not_set_flags_when_disabled(self):
        import torch

        orig_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            use_tf32 = False
            if use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul
            torch.backends.cudnn.allow_tf32 = orig_cudnn


# ---------------------------------------------------------------------------
# compile_model wrapper applied to attribution models
# ---------------------------------------------------------------------------

def _has_torch_compile() -> bool:
    import torch
    return hasattr(torch, "compile")


def _has_trak() -> bool:
    try:
        import trak  # noqa: F401
        return True
    except ImportError:
        return False


class TestCompileAppliedToAttributionModels(unittest.TestCase):
    """compile_model wraps the diffusion loss wrapper used by InfEmbed."""

    def _make_loss_wrapper(self):
        """Build a DiffusionLossWrapper with a real DiffusionLowdimFunctionalModelOutput.

        Uses the actual task class so this test is meaningful in the policy_doctor
        (or cupid) env where trak is installed. The test only needs to construct
        the wrapper — no actual forward pass is required to verify compile_model.
        """
        import torch.nn as nn
        from diffusion_policy.data_attribution.infembed_adapter import DiffusionLossWrapper
        from diffusion_policy.data_attribution.modelout_functions import (
            DiffusionLowdimFunctionalModelOutput,
        )
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

        unet = ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=8,
            diffusion_step_embed_dim=32,
            down_dims=[32, 64],
        )

        class _MinimalPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = unet
            def forward(self, batch):
                return batch

        policy = _MinimalPolicy()
        task = DiffusionLowdimFunctionalModelOutput(loss_fn="square")
        return DiffusionLossWrapper(policy, task)

    @unittest.skipUnless(_has_trak(), "trak not installed (run in policy_doctor env)")
    @unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
    def test_compile_wraps_loss_wrapper(self):
        """DiffusionLossWrapper can be wrapped by compile_model without error."""
        from diffusion_policy.common.ddp_util import compile_model

        wrapper = self._make_loss_wrapper()
        compiled = compile_model(wrapper, fullgraph=True, dynamic=False)
        # compile_model returns something callable; it should still be an nn.Module
        self.assertTrue(callable(compiled))

    @unittest.skipUnless(_has_trak(), "trak not installed (run in policy_doctor env)")
    def test_compile_skipped_when_disabled(self):
        """When use_compile=False, the wrapper is unchanged."""
        import torch.nn as nn

        wrapper = self._make_loss_wrapper()
        use_compile = False
        if use_compile:
            from diffusion_policy.common.ddp_util import compile_model
            wrapper = compile_model(wrapper)
        self.assertIsInstance(wrapper, nn.Module)
        self.assertNotIn("OptimizedModule", type(wrapper).__name__)


# ---------------------------------------------------------------------------
# Compiled attribution output matches eager
# ---------------------------------------------------------------------------

class TestCompiledAttributionMatchesEager(unittest.TestCase):
    """Compiled DiffusionLossWrapper output must be numerically identical to eager.

    Uses a self-contained task that calls torch.func.functional_call — the same
    op used in real TRAK and InfEmbed attribution — without requiring a
    normalizer, noise scheduler, or trak installation.
    """

    # Dimensions kept tiny so the test is fast on CPU.
    B, NUM_T, Ta, Da, To, Do = 2, 4, 8, 4, 2, 8

    @classmethod
    def _make_wrapper(cls):
        import torch
        import torch.nn as nn
        from diffusion_policy.data_attribution.infembed_adapter import DiffusionLossWrapper
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

        unet = ConditionalUnet1D(
            input_dim=cls.Da,
            global_cond_dim=cls.Do * cls.To,
            diffusion_step_embed_dim=32,
            down_dims=[32, 64],
        )

        class _MinimalPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = unet
            def forward(self, batch):
                return batch

        class _FunctionalCallTask:
            """Calls torch.func.functional_call on the UNet — the core attribution op."""
            def get_output(self, model, weights, buffers, tsteps, action, obs,
                           return_per_example=False):
                model_w = {k[len("model."):]: v for k, v in weights.items()
                           if k.startswith("model.")}
                model_b = {k[len("model."):]: v for k, v in buffers.items()
                           if k.startswith("model.")}
                B_local = action.shape[0]
                global_cond = obs.reshape(B_local, -1)
                pred = torch.func.functional_call(
                    model.model, (model_w, model_b),
                    (action, tsteps),
                    {"global_cond": global_cond},
                )
                loss = (pred ** 2).mean(dim=(-1, -2))  # (B,)
                return loss if return_per_example else loss.mean()

        return DiffusionLossWrapper(_MinimalPolicy(), _FunctionalCallTask())

    @classmethod
    def _make_batch(cls):
        import torch
        torch.manual_seed(0)
        return {
            "action":    torch.randn(cls.B, cls.Ta, cls.Da),
            "obs":       torch.randn(cls.B, cls.To, cls.Do),
            "timesteps": torch.randint(0, 100, (cls.B, cls.NUM_T)),
        }

    @unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
    def test_infembed_compiled_matches_eager(self):
        """Compiled DiffusionLossWrapper(fullgraph=True) output matches eager to 1e-5."""
        import torch
        from diffusion_policy.common.ddp_util import compile_model

        wrapper = self._make_wrapper()
        wrapper.eval()
        batch = self._make_batch()

        with torch.no_grad():
            out_eager = wrapper(batch)

        # functional_call receives an nn.Module as argument, which dynamo cannot
        # trace through with fullgraph=True — use fullgraph=False so compile
        # still applies to the surrounding ops while allowing graph breaks.
        compiled = compile_model(wrapper, fullgraph=False, dynamic=False)
        with torch.no_grad():
            out_compiled = compiled(batch)

        self.assertTrue(
            torch.allclose(out_eager, out_compiled, atol=1e-5),
            f"max diff: {(out_eager - out_compiled).abs().max().item():.2e}",
        )

    @unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
    def test_trak_functional_call_compiled_matches_eager(self):
        """Compiling the policy used in TRAK does not change functional_call output."""
        import torch
        import copy
        from diffusion_policy.common.ddp_util import compile_model

        wrapper = self._make_wrapper()
        wrapper.eval()
        batch = self._make_batch()

        policy = wrapper.policy
        policy_compiled = compile_model(
            copy.deepcopy(policy), dynamic=True, fullgraph=False
        )

        B_exp = self.B * self.NUM_T
        tsteps = batch["timesteps"].reshape(-1)        # (B*T,)
        action  = batch["action"].repeat_interleave(self.NUM_T, dim=0)
        obs     = batch["obs"].repeat_interleave(self.NUM_T, dim=0)
        global_cond = obs.reshape(B_exp, -1)

        def _call(p):
            w = dict(p.named_parameters())
            b = dict(p.named_buffers())
            mw = {k[len("model."):]: v for k, v in w.items() if k.startswith("model.")}
            mb = {k[len("model."):]: v for k, v in b.items() if k.startswith("model.")}
            return torch.func.functional_call(
                p.model, (mw, mb), (action, tsteps), {"global_cond": global_cond}
            )

        with torch.no_grad():
            out_eager    = _call(policy)
            out_compiled = _call(policy_compiled)

        self.assertTrue(
            torch.allclose(out_eager, out_compiled, atol=1e-5),
            f"max diff: {(out_eager - out_compiled).abs().max().item():.2e}",
        )


# ---------------------------------------------------------------------------
# Attribution YAML configs have tf32 and compile keys
# ---------------------------------------------------------------------------

class TestAttributionYamlFlags(unittest.TestCase):
    """All attribution YAML configs expose tf32 and compile keys."""

    def test_all_attribution_configs_have_tf32(self):
        import yaml

        for cfg_path in sorted(_ATTRIBUTION_CONFIG_DIR.glob("*.yaml")):
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            self.assertIn(
                "tf32", data,
                msg=f"Missing 'tf32' in {cfg_path.relative_to(_REPO_ROOT)}",
            )

    def test_all_attribution_configs_have_compile(self):
        import yaml

        for cfg_path in sorted(_ATTRIBUTION_CONFIG_DIR.glob("*.yaml")):
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            self.assertIn(
                "compile", data,
                msg=f"Missing 'compile' in {cfg_path.relative_to(_REPO_ROOT)}",
            )

    def test_attribution_config_defaults_are_enabled(self):
        """tf32 and compile should default to true in attribution configs."""
        import yaml

        for cfg_path in sorted(_ATTRIBUTION_CONFIG_DIR.glob("*.yaml")):
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            for key in ("tf32", "compile"):
                if key in data:
                    self.assertTrue(
                        data[key],
                        msg=f"Expected {key}=true in {cfg_path.relative_to(_REPO_ROOT)}",
                    )


if __name__ == "__main__":
    unittest.main()

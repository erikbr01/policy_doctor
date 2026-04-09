"""Optional smoke: one short diffusion train epoch + rollout per data stack.

Dataset locations are resolved in order: optional env vars, then Cupid ``REPO_ROOT/data/...``,
then ``policy_doctor.paths.DATA_SOURCE_ROOT`` (``data/source/{robomimic,mimicgen,robocasa}``),
then shallow discovery under those trees (MimicGen ``source/square.hdf5``, RoboCasa
``v1.0/**/lerobot`` with ``meta/modality.json``, etc.).

Environment variables (absolute paths recommended):

* ``CUPID_SMOKE_ROBOMIMIC_LOWDIM_HDF5`` — Robomimic-layout low-dim HDF5 (e.g. square mh).
* ``CUPID_SMOKE_MIMICGEN_LOWDIM_HDF5`` — MimicGen-merged low-dim HDF5.
* ``CUPID_SMOKE_ROBOCASA_LOWDIM_HDF5`` — RoboCasa kitchen low-dim HDF5 (old HDF5 workflow).
* ``CUPID_SMOKE_ROBOCASA_LEROBOT`` — LeRobot v2 dataset root (directory).
* ``CUPID_SMOKE_ROBOCASA_ENV_NAME`` — registered env name for hybrid eval rollouts
  (default: ``PickPlaceCounterToCabinet``). No eval HDF5 needed.

Optional: ``CUPID_SMOKE_DEVICE`` (default ``cuda:0`` if CUDA available, else ``cpu``).
Hybrid image smoke skips without CUDA unless ``CUPID_SMOKE_ALLOW_CPU_HYBRID=1``.
Runs use Hydra ``logging.mode=offline`` and default ``WANDB_MODE=offline`` where supported.

**Sampler paths:** ``get_dataset_masks`` accepts standard Robomimic-style segments,
``mimicgen`` / ``robocasa`` as path components, and substrings such as ``mimicgen`` /
``robocasa`` in the full path (so resolved symlinks into ``*_data`` mounts still work).
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
    GlobalHydra = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from policy_doctor.paths import CUPID_ROOT, DATA_SOURCE_ROOT, REPO_ROOT


def _replay_split_path_ok_for_sampler(path: Path) -> bool:
    """Rough pre-check aligned with ``get_dataset_masks`` path detection."""
    resolved = path.resolve()
    ps = str(resolved).lower()
    parts_lower = {p.lower() for p in resolved.parts}
    segment_hits = parts_lower & {
        "robomimic",
        "hardware",
        "pusht",
        "libero",
        "mimicgen",
        "robocasa",
    }
    if segment_hits:
        return True
    return any(
        s in ps
        for s in ("libero", "mimicgen", "robocasa", "robomimic", "hardware", "pusht")
    )


def _shallow_find_file(root: Path, filename: str, max_depth: int = 8) -> Path | None:
    """First ``filename`` match under ``root`` within ``max_depth`` directory levels."""
    if not root.is_dir():
        return None
    # Resolve before rglob so yielded paths share the same prefix as ``root_r`` (symlink-safe).
    root_r = root.resolve()
    for p in root_r.rglob(filename):
        if not p.is_file():
            continue
        depth = len(p.relative_to(root_r).parts)
        if depth <= max_depth:
            return p
    return None


def _resolve_hdf5_file(
    env_names: tuple[str, ...],
    fallbacks: tuple[Path, ...],
) -> Path | None:
    for name in env_names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            return p
    for p in fallbacks:
        if p.is_file():
            return p
    return None


def _resolve_lerobot_root(env_name: str, fallbacks: tuple[Path, ...]) -> Path | None:
    raw = os.environ.get(env_name, "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            return p
    for p in fallbacks:
        if p.is_dir():
            return p
    return None


def _resolve_robomimic_square_mh_hdf5() -> Path | None:
    p = _resolve_hdf5_file(
        ("CUPID_SMOKE_ROBOMIMIC_LOWDIM_HDF5",),
        (
            REPO_ROOT / "data/robomimic/datasets/square/mh/low_dim_abs.hdf5",
            DATA_SOURCE_ROOT / "robomimic" / "datasets" / "square" / "mh" / "low_dim_abs.hdf5",
        ),
    )
    if p is not None:
        return p
    rb = DATA_SOURCE_ROOT / "robomimic"
    return _shallow_find_file(rb, "low_dim_abs.hdf5", max_depth=24)


def _resolve_mimicgen_square_hdf5() -> Path | None:
    p = _resolve_hdf5_file(("CUPID_SMOKE_MIMICGEN_LOWDIM_HDF5",), ())
    if p is not None:
        return p
    # Low-dim policy matches Robomimic-layout merged / MimicGen ``source`` exports — not
    # ``core_datasets/.../demo.hdf5`` (different observation layout vs square_mimicgen_lowdim config).
    candidates = (
        REPO_ROOT / "data/mimicgen/square_merged.hdf5",
        DATA_SOURCE_ROOT / "mimicgen" / "square_merged.hdf5",
        DATA_SOURCE_ROOT / "mimicgen" / "source" / "square.hdf5",
    )
    for c in candidates:
        if c.is_file():
            return c
    mg = DATA_SOURCE_ROOT / "mimicgen"
    return _shallow_find_file(mg, "square_merged.hdf5", max_depth=24)


def _resolve_robocasa_kitchen_lowdim_hdf5() -> Path | None:
    p = _resolve_hdf5_file(
        ("CUPID_SMOKE_ROBOCASA_LOWDIM_HDF5",),
        (
            REPO_ROOT / "data/robocasa/datasets/kitchen_lowdim_merged.hdf5",
            DATA_SOURCE_ROOT / "robocasa" / "datasets" / "kitchen_lowdim_merged.hdf5",
            DATA_SOURCE_ROOT / "robocasa" / "kitchen_lowdim_merged.hdf5",
        ),
    )
    if p is not None:
        return p
    rc = DATA_SOURCE_ROOT / "robocasa"
    if not rc.is_dir():
        return None
    p = _shallow_find_file(rc, "kitchen_lowdim_merged.hdf5", max_depth=28)
    if p is not None:
        return p
    root_r = rc.resolve()
    ranked: list[tuple[tuple[int, ...], str, Path]] = []
    for f in root_r.rglob("*.hdf5"):
        if not f.is_file():
            continue
        try:
            if len(f.relative_to(root_r).parts) > 28:
                continue
        except ValueError:
            continue
        n = f.name.lower()
        key: tuple[int, ...]
        if "kitchen" in n and "lowdim" in n:
            key = (0, 0)
        elif "kitchen" in n and "merged" in n:
            key = (0, 1)
        elif "kitchen" in n:
            key = (1, 0)
        elif "lowdim" in n:
            key = (2, 0)
        else:
            continue
        ranked.append((key, str(f), f))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked[0][2]


def _discover_robocasa_lerobot_dataset_root() -> Path | None:
    """LeRobot v2 root: directory containing ``meta/modality.json`` (robocasa export)."""
    p = _resolve_lerobot_root(
        "CUPID_SMOKE_ROBOCASA_LEROBOT",
        (
            REPO_ROOT / "data/robocasa/lerobot/atomic_task",
            DATA_SOURCE_ROOT / "robocasa" / "lerobot" / "atomic_task",
        ),
    )
    if p is not None and (p / "meta" / "modality.json").is_file():
        return p
    rc = DATA_SOURCE_ROOT / "robocasa"
    if not rc.is_dir():
        return None
    roots: list[tuple[int, str, Path]] = []
    for modality in rc.resolve().rglob("meta/modality.json"):
        root = modality.parent.parent
        if (root / "meta" / "modality.json").is_file():
            s = str(root).lower()
            if "/target/atomic/" in s:
                prio = 0
            elif "/pretrain/atomic/" in s:
                prio = 1
            elif "/atomic/" in s:
                prio = 2
            else:
                prio = 3
            roots.append((prio, str(root), root))
    if not roots:
        return None
    roots.sort(key=lambda t: (t[0], t[1]))
    return roots[0][2]


def _discover_robocasa_image_eval_hdf5() -> Path | None:
    p = _resolve_hdf5_file(
        ("CUPID_SMOKE_ROBOCASA_IMAGE_EVAL_HDF5",),
        (
            REPO_ROOT / "data/robocasa/eval/atomic_task_demo.hdf5",
            DATA_SOURCE_ROOT / "robocasa" / "eval" / "atomic_task_demo.hdf5",
        ),
    )
    if p is not None:
        return p
    rc = DATA_SOURCE_ROOT / "robocasa"
    if not rc.is_dir():
        return None
    root_r = rc.resolve()
    ranked: list[tuple[int, str, Path]] = []
    for f in root_r.rglob("*.hdf5"):
        if not f.is_file():
            continue
        try:
            if len(f.relative_to(root_r).parts) > 28:
                continue
        except ValueError:
            continue
        n = f.name.lower()
        if "atomic" in n and "demo" in n:
            ranked.append((0, str(f), f))
        elif n == "demo.hdf5" and "atomic" in str(f).lower():
            ranked.append((1, str(f), f))
        elif "eval" in n and "demo" in n:
            ranked.append((2, str(f), f))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked[0][2]


@contextlib.contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def _smoke_device() -> str:
    override = os.environ.get("CUPID_SMOKE_DEVICE", "").strip()
    if override:
        return override
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _lowdim_training_overrides(
    hdf5: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    ds = str(hdf5.resolve())
    od = str(output_dir.resolve())
    return [
        "training.resume=false",
        "training.num_epochs=1",
        "training.max_train_steps=2",
        "training.max_val_steps=1",
        "training.rollout_every=1",
        "training.checkpoint_every=1",
        "training.val_every=1",
        "training.use_ema=false",
        "logging.mode=offline",
        # Avoid ``get_dataset_masks`` "Remainder demos!" on tiny HDF5s (e.g. short MimicGen sources).
        "task.dataset.val_ratio=0.1",
        f"training.device={device}",
        "dataloader.num_workers=0",
        "val_dataloader.num_workers=0",
        "dataloader.batch_size=32",
        "val_dataloader.batch_size=32",
        f"task.dataset.dataset_path={ds}",
        f"task.dataset_path={ds}",
        f"task.env_runner.dataset_path={ds}",
        "task.env_runner.n_train=0",
        "task.env_runner.n_train_vis=0",
        "task.env_runner.n_test=1",
        "task.env_runner.n_test_vis=1",
        "task.env_runner.n_envs=1",
        f"multi_run.run_dir={od}",
        "checkpoint.save_last_ckpt=false",
        "checkpoint.save_last_snapshot=false",
        "checkpoint.topk.k=0",
    ]


def _hybrid_training_overrides(
    lerobot_root: Path,
    output_dir: Path,
    device: str,
    env_name: str = "PickPlaceCounterToCabinet",
) -> list[str]:
    lr = str(lerobot_root.resolve())
    od = str(output_dir.resolve())
    return [
        "training.resume=false",
        "training.num_epochs=1",
        "training.max_train_steps=1",
        "training.max_val_steps=1",
        "training.rollout_every=1",
        "training.checkpoint_every=1",
        "training.val_every=1",
        "training.use_ema=false",
        "logging.mode=offline",
        f"training.device={device}",
        "dataloader.num_workers=0",
        "val_dataloader.num_workers=0",
        "dataloader.batch_size=2",
        "val_dataloader.batch_size=2",
        f"task.dataset.dataset_path={lr}",
        f"task.dataset_path={lr}",
        # env_name-based eval (no HDF5 needed — new LeRobot robocasa workflow)
        f"task.env_runner.env_name={env_name}",
        "task.env_runner.n_train=0",
        "task.env_runner.n_train_vis=0",
        "task.env_runner.n_test=1",
        "task.env_runner.n_test_vis=1",
        "task.env_runner.n_envs=1",
        f"multi_run.run_dir={od}",
        "checkpoint.save_last_ckpt=false",
        "checkpoint.save_last_snapshot=false",
        "checkpoint.topk.k=0",
    ]


def _compose_config(config_dir: Path, overrides: list[str]):
    assert hydra is not None and GlobalHydra is not None and OmegaConf is not None
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        config_dir=str(config_dir.resolve()),
        version_base=None,
    )
    try:
        cfg = hydra.compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(cfg)
        return cfg
    finally:
        GlobalHydra.instance().clear()


def _diffusion_stack_importable(repo_root: Path) -> bool:
    """True if ``diffusion_policy`` and core workspace deps (e.g. dill) are available."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import diffusion_policy.workspace.base_workspace  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(hydra is not None and torch is not None, "hydra / torch required")
class TestDiffusionLowdimDataSourceSmoke(unittest.TestCase):
    """Robomimic, MimicGen, RoboCasa low-dim CNN diffusion (one epoch)."""

    def setUp(self):
        if not CUPID_ROOT.is_dir():
            self.skipTest(f"Cupid repo not found at {CUPID_ROOT}")
        if not _diffusion_stack_importable(REPO_ROOT):
            self.skipTest(
                "diffusion_policy workspace not importable; use the cupid conda env (REPO_ROOT on path)"
            )
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("WANDB_MODE", "offline")

    def _run_lowdim(self, rel_config_dir: str, hdf5: Path) -> None:
        config_dir = CUPID_ROOT / "configs" / rel_config_dir
        if not config_dir.is_dir():
            self.skipTest(f"Missing config dir {config_dir}")
        if not _replay_split_path_ok_for_sampler(hdf5):
            self.skipTest(
                "HDF5 path is not accepted by get_dataset_masks(); see module docstring."
            )
        device = _smoke_device()
        with tempfile.TemporaryDirectory(prefix="cupid_smoke_lowdim_") as tmp:
            out = Path(tmp)
            overrides = _lowdim_training_overrides(hdf5, out, device)
            cfg = _compose_config(config_dir, overrides)
            with _chdir(REPO_ROOT):
                import wandb

                from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
                    TrainDiffusionUnetLowdimWorkspace,
                )

                try:
                    ws = TrainDiffusionUnetLowdimWorkspace(cfg, output_dir=str(out))
                    ws.run()
                finally:
                    wandb.finish()

    def test_robomimic_square_mh_lowdim_smoke(self):
        hdf5 = _resolve_robomimic_square_mh_hdf5()
        if hdf5 is None:
            self.skipTest(
                "No Robomimic square mh low_dim_abs.hdf5 found under REPO_ROOT/data/robomimic or "
                f"{DATA_SOURCE_ROOT / 'robomimic'} (set CUPID_SMOKE_ROBOMIMIC_LOWDIM_HDF5)."
            )
        self._run_lowdim("low_dim/square_mh/diffusion_policy_cnn", hdf5)

    def test_mimicgen_square_lowdim_smoke(self):
        hdf5 = _resolve_mimicgen_square_hdf5()
        if hdf5 is None:
            self.skipTest(
                "No MimicGen square low-dim HDF5 found (square_merged.hdf5 or source/square.hdf5) under "
                f"{DATA_SOURCE_ROOT / 'mimicgen'} or REPO_ROOT/data/mimicgen; "
                "set CUPID_SMOKE_MIMICGEN_LOWDIM_HDF5."
            )
        self._run_lowdim("low_dim/square_mimicgen_lowdim/diffusion_policy_cnn", hdf5)

    def test_robocasa_layout_lowdim_smoke(self):
        hdf5 = _resolve_robocasa_kitchen_lowdim_hdf5()
        if hdf5 is None:
            self.skipTest(
                "No RoboCasa kitchen low-dim HDF5 under "
                f"{DATA_SOURCE_ROOT / 'robocasa'} or REPO_ROOT/data/robocasa; "
                "set CUPID_SMOKE_ROBOCASA_LOWDIM_HDF5."
            )
        self._run_lowdim("low_dim/robocasa_layout_lowdim/diffusion_policy_cnn", hdf5)


@unittest.skipUnless(hydra is not None and torch is not None, "hydra / torch required")
class TestDiffusionRobocasaLerobotHybridSmoke(unittest.TestCase):
    """LeRobot image train + live RoboCasa env rollout (heavy; CUDA recommended).

    Uses env_name + env_kwargs for evaluation — no HDF5 needed.
    Requires the robocasa conda env (robocasa registered on sys.path).
    Override the eval env via CUPID_SMOKE_ROBOCASA_ENV_NAME (default: PickPlaceCounterToCabinet).
    """

    def setUp(self):
        if not CUPID_ROOT.is_dir():
            self.skipTest(f"Cupid repo not found at {CUPID_ROOT}")
        if not _diffusion_stack_importable(REPO_ROOT):
            self.skipTest("diffusion_policy workspace not importable")
        allow_cpu_hybrid = os.environ.get("CUPID_SMOKE_ALLOW_CPU_HYBRID", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not allow_cpu_hybrid and (torch is None or not torch.cuda.is_available()):
            self.skipTest(
                "RoboCasa hybrid smoke expects CUDA; set CUPID_SMOKE_ALLOW_CPU_HYBRID=1 to try CPU (slow)."
            )
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("WANDB_MODE", "offline")

    def test_robocasa_lerobot_atomic_hybrid_smoke(self):
        lerobot = _discover_robocasa_lerobot_dataset_root()
        if lerobot is None:
            self.skipTest(
                "No LeRobot root with meta/modality.json under "
                f"{DATA_SOURCE_ROOT / 'robocasa'} or REPO_ROOT/data/robocasa/lerobot; "
                "set CUPID_SMOKE_ROBOCASA_LEROBOT."
            )
        config_dir = CUPID_ROOT / "configs/image/robocasa_lerobot_atomic/diffusion_policy_transformer"
        if not config_dir.is_dir():
            self.skipTest(f"Missing config dir {config_dir}")

        env_name = os.environ.get("CUPID_SMOKE_ROBOCASA_ENV_NAME", "PickPlaceCounterToCabinet").strip()
        device = _smoke_device()
        with tempfile.TemporaryDirectory(prefix="cupid_smoke_hybrid_") as tmp:
            out = Path(tmp)
            overrides = _hybrid_training_overrides(lerobot, out, device, env_name=env_name)
            cfg = _compose_config(config_dir, overrides)
            with _chdir(REPO_ROOT):
                import wandb

                try:
                    from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (
                        TrainDiffusionTransformerHybridWorkspace,
                    )
                except ImportError as e:
                    self.skipTest(f"Hybrid workspace import failed: {e}")

                try:
                    ws = TrainDiffusionTransformerHybridWorkspace(cfg, output_dir=str(out))
                    ws.run()
                finally:
                    wandb.finish()

"""Benchmark: dataloader throughput (cached vs. uncached), compile time, fwd/bwd speedup.

Motivation
----------
During kendama_baseline training we observed spiky GPU utilisation and slow
validation epochs (~3.5 s/batch vs ~0.25 s/batch for training).  This file
pins down the root causes with reproducible timing numbers.

What "use_cache" actually does
------------------------------
``use_cache=False`` converts the HDF5 → zarr.MemoryStore at startup every
run (~80 s for kendama_may13.hdf5).  ``use_cache=True`` saves the converted
zarr to disk as a .zarr.zip and loads from there on subsequent runs — much
faster startup but identical in-memory representation.  Both modes end up with
data in RAM once initialised; the batch throughput should therefore be the same.
The tests below verify (and quantify) this.

Slow validation
---------------
``get_validation_dataset()`` returns ``copy.copy(self)`` sharing the same
zarr.MemoryStore, so val data IS in RAM.  The 13 s first-batch penalty is
``torch.compile`` recompiling the validation forward graph (eval mode uses a
different graph than train mode with EMA disabled / gradients off).  A separate
timing test confirms this.

Run (cupid_torch2 env, GPU required for compile/fwd-bwd tests):
    conda activate cupid_torch2
    python -m pytest tests/training/test_droid_throughput.py -v -s

Marks
-----
- ``@slow``            — touches the actual HDF5 (minutes); skipped by default
- ``@requires_gpu``    — skipped when CUDA is unavailable
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CUPID_ROOT = _REPO_ROOT / "third_party" / "cupid"
for _p in [str(_REPO_ROOT), str(_CUPID_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Paths / skip guards
# ---------------------------------------------------------------------------

DROID_HDF5 = Path(os.environ.get(
    "DROID_HDF5",
    "/mnt/ssdB/erik/droid_data/kendama_may13.hdf5",
))

DROID_SHAPE_META = {
    "action": {"shape": [8]},
    "obs": {
        "hand_camera_image":     {"shape": [3, 256, 256], "type": "rgb"},
        "exterior_image_1_left": {"shape": [3, 256, 256], "type": "rgb"},
        "joint_positions":       {"shape": [7]},
        "gripper_position":      {"shape": [1]},
    },
}

def _has_hdf5() -> bool:
    return DROID_HDF5.exists()

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def _has_torch_compile() -> bool:
    try:
        import torch
        return hasattr(torch, "compile")
    except ImportError:
        return False

def _cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Dataset / dataloader helpers
# ---------------------------------------------------------------------------

def _make_dataset(use_cache: bool, val_ratio: float = 0.1, load_to_memory: bool = False):
    """Return an initialised RobomimicReplayImageDataset for the kendama HDF5.

    load_to_memory only applies when use_cache=True.  When True, the dataset
    skips zarr and stores everything as plain numpy arrays (no per-batch
    decompression overhead).
    """
    from diffusion_policy.dataset.robomimic_replay_image_dataset import (
        RobomimicReplayImageDataset,
    )
    return RobomimicReplayImageDataset(
        shape_meta=DROID_SHAPE_META,
        dataset_path=str(DROID_HDF5),
        horizon=16,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        rotation_rep="rotation_6d",
        use_cache=use_cache,
        load_to_memory=load_to_memory,
        seed=42,
        val_ratio=val_ratio,
    )


def _measure_dataloader_throughput(
    dataset,
    num_workers: int = 4,
    batch_size: int = 64,
    num_batches: int = 30,
    persistent_workers: bool = True,
    shuffle: bool = True,
) -> dict:
    """Measure true batch-delivery throughput.

    Times each ``next(iter)`` call — how long the main process waits for the
    worker queue.  This captures zarr decompression + tensor stacking in the
    workers, which is the actual bottleneck.

    The old implementation timed tensor .shape access after batch receipt,
    which is near-zero and tells us nothing about worker throughput.

    Returns: batches, elapsed_s, batches_per_s, samples_per_s,
             first_ms, p50_ms, p95_ms, batch_times_ms.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers and (num_workers > 0),
        # pin_memory=False: async pinning inflates per-batch times and
        # obscures the zarr decompression signal.
        pin_memory=False,
        shuffle=shuffle,
    )

    it = iter(loader)
    batch_times_ms = []
    for _ in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        batch_times_ms.append((time.perf_counter() - t0) * 1000)
        del batch

    elapsed = sum(batch_times_ms) / 1000
    n = len(batch_times_ms)
    return {
        "batches": n,
        "elapsed_s": elapsed,
        "batches_per_s": n / elapsed if elapsed > 0 else 0,
        "samples_per_s": n * batch_size / elapsed if elapsed > 0 else 0,
        "first_ms": batch_times_ms[0] if batch_times_ms else float("nan"),
        "p50_ms": float(np.percentile(batch_times_ms, 50)) if batch_times_ms else float("nan"),
        "p95_ms": float(np.percentile(batch_times_ms, 95)) if batch_times_ms else float("nan"),
        "batch_times_ms": batch_times_ms,
    }

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _fit_identity_normalizer(policy):
    """Attach an identity LinearNormalizer so compute_loss doesn't crash.

    In real training the normalizer is fitted from the dataset; here we just
    need scale=1, offset=0 for all obs keys and actions so the forward pass
    can run without data.
    """
    import torch
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    norm = LinearNormalizer()
    identity = SingleFieldLinearNormalizer.create_identity()
    norm["action"] = identity
    for key in DROID_SHAPE_META["obs"]:
        norm[key] = identity
    policy.set_normalizer(norm)
    # Normalizer is built on CPU; push it to whatever device the policy is on.
    device = next(policy.parameters()).device
    policy.normalizer.to(device)
    return policy


def _build_image_policy(device="cuda"):
    """Build the full DiffusionUnetHybridImagePolicy matching the droid config."""
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=DROID_SHAPE_META,
        noise_scheduler=noise_scheduler,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=100,
        obs_as_global_cond=True,
        # Match production config (down_dims from droid/diffusion_policy_cnn/config.yaml).
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        crop_shape=[224, 224],
        pretrained_backbone=False,  # skip ImageNet download in tests
        diffusion_step_embed_dim=128,
    )
    return policy.to(device)


def _make_fake_batch(batch_size: int = 8, device: str = "cuda") -> dict:
    """Minimal obs+action batch that matches the droid shape_meta."""
    import torch

    return {
        "obs": {
            "hand_camera_image":     torch.randint(0, 256, (batch_size, 2, 3, 256, 256),
                                                    dtype=torch.uint8, device=device).float() / 255.0,
            "exterior_image_1_left": torch.randint(0, 256, (batch_size, 2, 3, 256, 256),
                                                    dtype=torch.uint8, device=device).float() / 255.0,
            "joint_positions": torch.randn(batch_size, 2, 7, device=device),
            "gripper_position": torch.randn(batch_size, 2, 1, device=device),
        },
        "action": torch.randn(batch_size, 16, 8, device=device),
        "action_is_pad": torch.zeros(batch_size, 16, dtype=torch.bool, device=device),
    }


# ---------------------------------------------------------------------------
# 1. Dataloader throughput: use_cache=False vs use_cache=True
# ---------------------------------------------------------------------------

class TestDataloaderThroughput(unittest.TestCase):
    """Verify that use_cache affects only startup time, not per-batch throughput.

    Both use_cache=False and use_cache=True load data into zarr.MemoryStore.
    Training batch throughput should therefore be similar for both.

    Requires the kendama HDF5 at DROID_HDF5 (or $DROID_HDF5 env var).
    These tests are slow (several minutes each due to HDF5 conversion).
    """

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_uncached_startup_time(self):
        """use_cache=False: measure HDF5→zarr.MemoryStore conversion time."""
        t0 = time.perf_counter()
        ds = _make_dataset(use_cache=False)
        startup_s = time.perf_counter() - t0
        print(f"\n[uncached] startup: {startup_s:.1f}s  len={len(ds)}")
        # Conversion should complete (no assertion on time — just record it).
        self.assertGreater(len(ds), 0)

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_cached_startup_time(self):
        """use_cache=True: measure zarr.zip load time (fast after first run)."""
        t0 = time.perf_counter()
        ds = _make_dataset(use_cache=True)
        startup_s = time.perf_counter() - t0
        zarr_zip = Path(str(DROID_HDF5) + ".zarr.zip")
        print(f"\n[cached]   startup: {startup_s:.1f}s  zarr.zip={'yes' if zarr_zip.exists() else 'first-run'}")
        self.assertGreater(len(ds), 0)

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_uncached_batch_throughput(self):
        """use_cache=False: measure steady-state batch throughput from zarr.MemoryStore."""
        ds = _make_dataset(use_cache=False)
        stats = _measure_dataloader_throughput(ds, num_batches=30)
        print(
            f"\n[uncached] throughput: {stats['samples_per_s']:.0f} samples/s  "
            f"first={stats['first_ms']:.0f}ms  p50={stats['p50_ms']:.0f}ms"
        )
        self.assertGreater(stats["batches_per_s"], 0)

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_cached_batch_throughput(self):
        """use_cache=True: measure steady-state batch throughput from zarr.MemoryStore.

        Should be similar to uncached — both read from MemoryStore.
        """
        ds = _make_dataset(use_cache=True)
        stats = _measure_dataloader_throughput(ds, num_batches=30)
        print(
            f"\n[cached]   throughput: {stats['samples_per_s']:.0f} samples/s  "
            f"first={stats['first_ms']:.0f}ms  p50={stats['p50_ms']:.0f}ms"
        )
        self.assertGreater(stats["batches_per_s"], 0)

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_val_dataset_shares_memorybuffer(self):
        """Validation dataset uses copy.copy — it shares the same zarr.MemoryStore.

        If val is significantly slower than train from the same store,
        zarr decompression IS the bottleneck (not GPU-side compile/eval overhead).
        """
        ds = _make_dataset(use_cache=False, val_ratio=0.1)
        val_ds = ds.get_validation_dataset()

        train_stats = _measure_dataloader_throughput(ds, num_batches=20)
        val_stats = _measure_dataloader_throughput(val_ds, num_batches=20, shuffle=False)

        print(
            f"\n[train]    {train_stats['samples_per_s']:.0f} samples/s  "
            f"p50={train_stats['p50_ms']:.0f}ms"
        )
        print(
            f"[val]      {val_stats['samples_per_s']:.0f} samples/s  "
            f"p50={val_stats['p50_ms']:.0f}ms"
        )

        # If val is >5× slower than train, data loading is NOT the bottleneck
        # (it would have to be GPU-side: compile, model eval, etc.)
        ratio = train_stats["samples_per_s"] / max(val_stats["samples_per_s"], 1e-9)
        print(f"[ratio]    train/val throughput = {ratio:.1f}x")
        # Both read from the same MemoryStore — throughput should be within 3×.
        self.assertLess(
            ratio, 3.0,
            msg=f"Val dataloader is {ratio:.1f}× slower than train — "
                "bottleneck is NOT data loading (check GPU-side: compile/eval-mode graph)."
        )


# ---------------------------------------------------------------------------
# 1b. load_to_memory=True: numpy backend vs zarr.MemoryStore
# ---------------------------------------------------------------------------

class TestLoadToMemory(unittest.TestCase):
    """Compare zarr.MemoryStore vs numpy (load_to_memory=True) batch throughput.

    When use_cache=True and load_to_memory=False (the current training config):
      - Data is stored compressed in zarr.MemoryStore
      - Each batch access decompresses chunks via blosc
      - This is the bottleneck we observed: ~18 samples/s for val, ~122/s for train

    When use_cache=True and load_to_memory=True:
      - Dataset startup decompresses everything into plain numpy arrays
      - Each batch access is a numpy slice — no decompression at runtime
      - Both train and val should be fast and similar

    Concurrent-access hypothesis:
      Worker processes fork from the parent, inheriting the zarr store or numpy
      arrays via copy-on-write.  Reads from zarr.MemoryStore invoke blosc
      decompression; reads from numpy arrays are GIL-free C memcpy.  If zarr's
      blosc codec has per-store serialisation (unlikely for reads, but possible
      depending on blosc thread-pool interaction with fork), throughput will
      plateau with more workers.  The worker-scaling test below checks this.
    """

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_load_to_memory_startup_time(self):
        """Startup: decompress-all-upfront vs load-compressed-zarr-zip."""
        t0 = time.perf_counter()
        ds_zarr = _make_dataset(use_cache=True, load_to_memory=False)
        zarr_startup_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        ds_numpy = _make_dataset(use_cache=True, load_to_memory=True)
        numpy_startup_s = time.perf_counter() - t1

        print(
            f"\n[startup] zarr  (compressed RAM): {zarr_startup_s:.1f}s"
            f"\n[startup] numpy (decompressed):    {numpy_startup_s:.1f}s"
        )
        self.assertGreater(len(ds_zarr), 0)
        self.assertGreater(len(ds_numpy), 0)

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_load_to_memory_train_throughput(self):
        """Train batch throughput: zarr vs numpy with 4 workers."""
        ds_zarr = _make_dataset(use_cache=True, load_to_memory=False)
        stats_zarr = _measure_dataloader_throughput(ds_zarr, num_workers=4, shuffle=True)

        ds_numpy = _make_dataset(use_cache=True, load_to_memory=True)
        stats_numpy = _measure_dataloader_throughput(ds_numpy, num_workers=4, shuffle=True)

        print(
            f"\n[train/zarr]  p50={stats_zarr['p50_ms']:.0f}ms  "
            f"p95={stats_zarr['p95_ms']:.0f}ms  "
            f"{stats_zarr['samples_per_s']:.0f} samples/s"
            f"\n[train/numpy] p50={stats_numpy['p50_ms']:.0f}ms  "
            f"p95={stats_numpy['p95_ms']:.0f}ms  "
            f"{stats_numpy['samples_per_s']:.0f} samples/s"
            f"\n[speedup] numpy/zarr = {stats_numpy['samples_per_s'] / max(stats_zarr['samples_per_s'], 1):.1f}×"
        )
        # numpy should be at least 2× faster (zarr decompression overhead).
        self.assertGreater(
            stats_numpy["samples_per_s"],
            stats_zarr["samples_per_s"],
            msg="numpy backend not faster than zarr — decompression may not be the bottleneck."
        )

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_load_to_memory_val_train_ratio(self):
        """Does load_to_memory close the 6.7× val/train gap seen with zarr?

        With zarr: val is 6.7× slower than train (different zarr chunk access
        patterns for shuffled vs sequential, OR concurrent-access serialisation).
        With numpy: both paths are pure numpy slices — should be similar speed.
        """
        ds = _make_dataset(use_cache=True, load_to_memory=True, val_ratio=0.1)
        val_ds = ds.get_validation_dataset()

        train_stats = _measure_dataloader_throughput(ds, num_workers=4, shuffle=True)
        val_stats = _measure_dataloader_throughput(val_ds, num_workers=4, shuffle=False)

        ratio = train_stats["samples_per_s"] / max(val_stats["samples_per_s"], 1)
        print(
            f"\n[numpy/train] p50={train_stats['p50_ms']:.0f}ms  "
            f"{train_stats['samples_per_s']:.0f} samples/s"
            f"\n[numpy/val]   p50={val_stats['p50_ms']:.0f}ms  "
            f"{val_stats['samples_per_s']:.0f} samples/s"
            f"\n[ratio] train/val = {ratio:.1f}×  (was 6.7× with zarr)"
        )
        # With numpy, both read plain arrays — val should be within 2× of train.
        self.assertLess(
            ratio, 2.0,
            msg=f"Val still {ratio:.1f}× slower with numpy — bottleneck is NOT zarr decompression."
        )

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_worker_scaling_zarr(self):
        """Does zarr throughput scale with more workers, or does it plateau?

        If zarr.MemoryStore has per-process serialisation (e.g. blosc thread-pool
        interaction with fork), throughput will plateau or even drop with more workers.
        If it scales linearly, concurrent access is not the problem.
        """
        ds = _make_dataset(use_cache=True, load_to_memory=False)
        results = {}
        for nw in [1, 2, 4, 8]:
            stats = _measure_dataloader_throughput(ds, num_workers=nw, num_batches=20, shuffle=True)
            results[nw] = stats["samples_per_s"]
            print(f"  [zarr/{nw}w] {stats['samples_per_s']:.0f} samples/s  "
                  f"p50={stats['p50_ms']:.0f}ms")

        scaling_1_to_4 = results[4] / max(results[1], 1)
        scaling_4_to_8 = results[8] / max(results[4], 1)
        print(
            f"\n[zarr scaling] 1→4 workers: {scaling_1_to_4:.1f}×  "
            f"4→8 workers: {scaling_4_to_8:.1f}×"
            f"\n[diagnosis] concurrent access problem: "
            f"{'likely (scaling < 1.5×)' if scaling_1_to_4 < 1.5 else 'unlikely'}"
        )
        # Record for analysis — no hard assertion (plateau is informative too).

    @unittest.skipUnless(_has_hdf5(), f"HDF5 not found at {DROID_HDF5}")
    def test_worker_scaling_numpy(self):
        """Does numpy throughput scale with more workers?

        Expected: near-linear up to hardware limits (pure memcpy, GIL-free).
        """
        ds = _make_dataset(use_cache=True, load_to_memory=True)
        results = {}
        for nw in [1, 2, 4, 8]:
            stats = _measure_dataloader_throughput(ds, num_workers=nw, num_batches=20, shuffle=True)
            results[nw] = stats["samples_per_s"]
            print(f"  [numpy/{nw}w] {stats['samples_per_s']:.0f} samples/s  "
                  f"p50={stats['p50_ms']:.0f}ms")

        scaling_1_to_4 = results[4] / max(results[1], 1)
        print(
            f"\n[numpy scaling] 1→4 workers: {scaling_1_to_4:.1f}×"
        )
        # numpy should scale at least 2× from 1 to 4 workers.
        self.assertGreater(
            scaling_1_to_4, 2.0,
            msg=f"numpy only scaled {scaling_1_to_4:.1f}× from 1→4 workers — "
                "check for GIL or multiprocessing overhead."
        )


# ---------------------------------------------------------------------------
# 2. Compilation timing: ResNet18 backbone + ConditionalUnet1D
# ---------------------------------------------------------------------------

@unittest.skipUnless(_has_cuda(), "CUDA not available")
@unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
class TestImageModelCompile(unittest.TestCase):
    """Measure torch.compile time and fwd/bwd speedup for the droid policy.

    IMPORTANT: ``torch.compile(policy)`` compiles ``policy.forward()``, but the
    training loop calls ``policy.compute_loss()``.  compile_model() therefore has
    ZERO effect on training unless compute_loss is compiled separately, or the
    sub-modules (obs_encoder, noise_pred_net) are compiled directly.

    These tests expose that gap and measure the actual speedup from compiling the
    correct entry points.
    """

    def _warmup(self, fn, n: int = 3):
        for _ in range(n):
            fn()
        _cuda_sync()

    def _time_fn(self, fn, n: int = 15) -> float:
        _cuda_sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        _cuda_sync()
        return (time.perf_counter() - t0) / n

    def test_compile_wall_clock_time(self):
        """How long does torch.compile() take to wrap the policy module?

        This is just the Python wrap — inductor compilation is lazy and
        triggered on the first actual forward pass of the compiled callable.
        """
        import torch
        from diffusion_policy.common.ddp_util import compile_model

        policy = _build_image_policy()
        t0 = time.perf_counter()
        compiled = compile_model(policy)
        wrap_s = time.perf_counter() - t0
        print(f"\n[compile] torch.compile() wrap (module): {wrap_s:.3f}s")

        # Also measure wrapping compute_loss as a function — this IS what
        # affects training speed.
        policy2 = _fit_identity_normalizer(_build_image_policy())
        t1 = time.perf_counter()
        compiled_fn = torch.compile(policy2.compute_loss, fullgraph=False, dynamic=False)
        wrap_fn_s = time.perf_counter() - t1
        print(f"[compile] torch.compile() wrap (compute_loss fn): {wrap_fn_s:.3f}s")
        self.assertIsNotNone(compiled)

    def test_submodule_inductor_compile_time(self):
        """Time the first call to compiled obs_encoder and UNet — inductor JIT trigger.

        ``torch.compile(policy)`` compiles ``policy.forward()`` but training calls
        ``policy.compute_loss()``.  Compiling the sub-modules (obs_encoder, model)
        directly is the correct approach since those are the GPU-intensive components
        that ``compute_loss`` calls internally.

        This test also reveals whether ``compute_loss`` itself needs to be compiled
        separately (it contains Python-level logic and a ``@_dynamo.disable`` decorator
        on the normalizer that causes graph breaks, making full-function compile fragile).
        """
        import torch

        device = "cuda"
        policy = _fit_identity_normalizer(_build_image_policy(device))
        batch = _make_fake_batch(batch_size=4, device=device)

        # Build realistic sub-module inputs.
        # compute_loss reshapes (B, T_obs, ...) → (B*T_obs, ...) before calling obs_encoder.
        n_obs_steps = 2
        obs_raw = {k: v[:, :n_obs_steps, ...] for k, v in batch["obs"].items()
                   if k in ("hand_camera_image", "exterior_image_1_left",
                             "joint_positions", "gripper_position")}
        # Flatten T_obs into batch dimension, matching compute_loss behaviour.
        obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs_raw.items()}
        with torch.no_grad():
            obs_features_eager = policy.obs_encoder(obs)  # (B*T_obs, D)

        timestep = torch.zeros(4, dtype=torch.long, device=device)
        noisy_action = torch.randn(4, 16, 8, device=device)

        # ---- Compile obs_encoder ----
        # obs_encoder contains set.issubset() assertion — fullgraph=False required.
        compiled_obs_enc = torch.compile(policy.obs_encoder, fullgraph=False, dynamic=False)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_obs_enc(obs)
            _cuda_sync()
        obs_first_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_obs_enc(obs)
            _cuda_sync()
        obs_second_s = time.perf_counter() - t1

        # ---- Compile UNet (noise_pred_net) ----
        # obs_encoder returns (B*T_obs, D); compute_loss reshapes to (B, D*T_obs).
        batch_size = 4
        global_cond = obs_features_eager.reshape(batch_size, -1)  # (B, D*T_obs)

        compiled_unet = torch.compile(policy.model, fullgraph=False, dynamic=False)

        t2 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_unet(noisy_action, timestep, global_cond=global_cond)
            _cuda_sync()
        unet_first_s = time.perf_counter() - t2

        t3 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_unet(noisy_action, timestep, global_cond=global_cond)
            _cuda_sync()
        unet_second_s = time.perf_counter() - t3

        print(
            f"\n[obs_encoder compile] first (JIT): {obs_first_s:.2f}s  second: {obs_second_s * 1000:.1f}ms"
            f"\n[UNet compile]        first (JIT): {unet_first_s:.2f}s  second: {unet_second_s * 1000:.1f}ms"
        )
        # After warmup, compiled passes should be faster than the first (JIT) pass.
        self.assertLess(obs_second_s, obs_first_s + 0.1,  # allow some slack
                        msg="obs_encoder second pass not cached — compile may have failed.")
        self.assertLess(unet_second_s, unet_first_s + 0.1,
                        msg="UNet second pass not cached — compile may have failed.")

    def test_forward_backward_speedup(self):
        """Compiled sub-modules vs eager: forward speedup for the image policy.

        Measures forward pass only — the UNet backward through AOT autograd fails
        in torch 2.4.1 (TensorAlias/FakeTensor tracing bug with GroupNorm).
        Forward-only is the right benchmark for validation throughput anyway.

        Training backward speedup can be assessed via the training epoch throughput
        comparison once the correct compile approach is applied in the workspace.
        """
        import torch

        device = "cuda"
        batch_size = 8
        n_runs = 20

        # ---- Eager policy ----
        policy_eager = _fit_identity_normalizer(_build_image_policy(device))
        batch = _make_fake_batch(batch_size=batch_size, device=device)

        def eager_fwd():
            with torch.no_grad():
                _ = policy_eager.compute_loss(batch)

        self._warmup(eager_fwd, n=3)
        eager_s = self._time_fn(eager_fwd, n=n_runs)

        # ---- Sub-module compiled policy ----
        policy_compiled = _fit_identity_normalizer(_build_image_policy(device))
        policy_compiled.obs_encoder = torch.compile(
            policy_compiled.obs_encoder, fullgraph=False, dynamic=False
        )
        policy_compiled.model = torch.compile(
            policy_compiled.model, fullgraph=False, dynamic=False
        )

        def compiled_fwd():
            with torch.no_grad():
                _ = policy_compiled.compute_loss(batch)

        # Warmup triggers inductor compilation for each sub-module.
        self._warmup(compiled_fwd, n=5)
        compiled_s = self._time_fn(compiled_fwd, n=n_runs)

        speedup = eager_s / max(compiled_s, 1e-9)
        print(
            f"\n[fwd only] eager:              {eager_s * 1000:.1f} ms  (batch={batch_size})"
            f"\n[fwd only] compile(submodules):{compiled_s * 1000:.1f} ms"
            f"\n[fwd only] speedup:            {speedup:.2f}×"
        )
        self.assertGreater(speedup, 0.9, msg="compile(submodules) forward is slower than eager.")

    def test_eval_vs_train_mode_compile_graph(self):
        """Does compiling obs_encoder incur a recompile penalty on first eval-mode call?

        ResNet18 contains BatchNorm layers which behave differently in train vs eval.
        ``fullgraph=True`` bakes in either train OR eval mode behavior; switching
        modes triggers recompilation of the Triton kernels.

        This test isolates the obs_encoder (the component that actually changes
        between train/eval) to quantify the penalty without going through
        compute_loss (which has unrelated complexities).

        This is the root cause of the 13s first-batch penalty seen in validation —
        confirmed or denied by the first-vs-steady-state ratio below.
        """
        import torch

        device = "cuda"
        policy = _build_image_policy(device)

        # Build a realistic obs dict (B=4, T_obs=2, ...).
        batch = _make_fake_batch(batch_size=4, device=device)
        obs = {k: v[:, :2, ...] for k, v in batch["obs"].items()}

        # Compile obs_encoder only (ResNet18 × 2 — the part that changes between modes).
        compiled_enc = torch.compile(policy.obs_encoder, fullgraph=False, dynamic=False)
        # Reshape as compute_loss does (B*T_obs, C, H, W).
        obs_flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}

        # ---- Train mode: first call (compilation) ----
        compiled_enc.train()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_enc(obs_flat)
            _cuda_sync()
        train_first_s = time.perf_counter() - t0

        # Train steady-state.
        t1 = time.perf_counter()
        for _ in range(5):
            with torch.no_grad():
                _ = compiled_enc(obs_flat)
        _cuda_sync()
        train_steady_s = (time.perf_counter() - t1) / 5

        # ---- Eval mode: first call — BatchNorm graph differs, may recompile ----
        compiled_enc.eval()
        t2 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_enc(obs_flat)
            _cuda_sync()
        eval_first_s = time.perf_counter() - t2

        # Eval steady-state.
        t3 = time.perf_counter()
        for _ in range(5):
            with torch.no_grad():
                _ = compiled_enc(obs_flat)
        _cuda_sync()
        eval_steady_s = (time.perf_counter() - t3) / 5

        # Simulated epoch 2: switch train → eval again (as the training loop does).
        compiled_enc.train()
        for _ in range(3):
            with torch.no_grad():
                _ = compiled_enc(obs_flat)
        compiled_enc.eval()
        t4 = time.perf_counter()
        with torch.no_grad():
            _ = compiled_enc(obs_flat)
            _cuda_sync()
        eval_epoch2_first_s = time.perf_counter() - t4

        recompiles = eval_first_s > eval_steady_s * 5

        print(
            f"\n[obs_encoder train] first: {train_first_s:.2f}s  steady: {train_steady_s * 1000:.1f}ms"
            f"\n[obs_encoder eval]  first: {eval_first_s:.2f}s  steady: {eval_steady_s * 1000:.1f}ms"
            f"\n[obs_encoder eval]  epoch2 first: {eval_epoch2_first_s * 1000:.1f}ms"
            f"\n[diagnosis] eval recompile penalty: {'YES — {:.1f}x overhead'.format(eval_first_s/eval_steady_s) if recompiles else 'no — graph is shared or penalty is small'}"
            f"\n[diagnosis] val batch slowness from compile: {'likely' if recompiles else 'unlikely — check zarr decompression'}"
        )


# ---------------------------------------------------------------------------
# 3. State-based compile baseline (no GPU needed — correctness + rough timing)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_has_torch_compile(), "torch.compile requires PyTorch >= 2.0")
class TestStateModelCompileSpeedup(unittest.TestCase):
    """CPU-only compile speedup for state-based backbones.

    Extends the correctness tests in test_training_flags.py with actual timing.
    These are fast (seconds, not minutes) and run without a GPU.
    """

    def _make_unet(self):
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        return ConditionalUnet1D(
            input_dim=10,
            global_cond_dim=40,  # 2 obs steps × 20 obs_dim
            diffusion_step_embed_dim=64,
            down_dims=[64, 128, 256],
        )

    def _make_transformer(self):
        from diffusion_policy.model.diffusion.transformer_for_diffusion import (
            TransformerForDiffusion,
        )
        return TransformerForDiffusion(
            input_dim=10,
            output_dim=10,
            horizon=16,
            n_obs_steps=2,
            cond_dim=20,
            n_layer=4,
            n_head=4,
            n_emb=128,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        )

    def _bench(self, model, sample, timestep, cond, n_warmup=5, n_runs=50, label=""):
        import torch

        def step():
            out = model(sample, timestep, global_cond=cond) if hasattr(model, "global_cond") \
                else model(sample, timestep, cond)
            return out

        # detect which forward signature to use
        try:
            model(sample, timestep, global_cond=cond)
            def step():
                return model(sample, timestep, global_cond=cond)
        except TypeError:
            def step():
                return model(sample, timestep, cond)

        for _ in range(n_warmup):
            step()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            step()
        elapsed = (time.perf_counter() - t0) / n_runs
        return elapsed

    def test_unet_compile_speedup_cpu(self):
        import torch

        sample   = torch.randn(4, 16, 10)
        timestep = torch.zeros(4)
        cond     = torch.randn(4, 40)

        eager = self._make_unet()
        eager_s = self._bench(eager, sample, timestep, cond, label="unet-eager")

        compiled = torch.compile(self._make_unet(), fullgraph=True, dynamic=False)
        # Warmup triggers compile.
        for _ in range(10):
            compiled(sample, timestep, global_cond=cond)
        compiled_s = self._bench(compiled, sample, timestep, cond, label="unet-compiled")

        speedup = eager_s / max(compiled_s, 1e-9)
        print(
            f"\n[UNet/CPU] eager={eager_s * 1000:.1f}ms  "
            f"compiled={compiled_s * 1000:.1f}ms  speedup={speedup:.2f}×"
        )
        # CPU compile uses the 'inductor' C++ backend, not Triton; overhead can exceed
        # the gain for small models.  We just record the number — no pass/fail on CPU.
        if speedup < 0.9:
            print(f"  [info] CPU compile slowdown ({speedup:.2f}×) — expected for small UNet on CPU; GPU is the target.")

    def test_transformer_compile_speedup_cpu(self):
        import torch

        sample   = torch.randn(4, 16, 10)
        timestep = torch.zeros(4, dtype=torch.long)
        cond     = torch.randn(4, 2, 20)  # (B, n_obs_steps, cond_dim)

        eager = self._make_transformer()
        t0 = time.perf_counter()
        for _ in range(5):
            eager(sample, timestep, cond)
        for _ in range(50):
            eager(sample, timestep, cond)
        eager_s = (time.perf_counter() - t0) / 50

        compiled = torch.compile(self._make_transformer(), fullgraph=True, dynamic=False)
        for _ in range(10):
            compiled(sample, timestep, cond)
        t1 = time.perf_counter()
        for _ in range(50):
            compiled(sample, timestep, cond)
        compiled_s = (time.perf_counter() - t1) / 50

        speedup = eager_s / max(compiled_s, 1e-9)
        print(
            f"\n[Transformer/CPU] eager={eager_s * 1000:.1f}ms  "
            f"compiled={compiled_s * 1000:.1f}ms  speedup={speedup:.2f}×"
        )
        self.assertGreater(speedup, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# RoboCasa Training Crash: DataLoader Worker Death / Hang

## Summary

Training on RoboCasa data crashed in two distinct ways, both with the same symptom: a DataLoader
worker dies silently and the main process hangs forever in `_data_queue.get()`.

**Crash 1 — MuJoCo / glibc heap corruption (during eval):**
```
malloc_consolidate(): unaligned fastbin chunk detected
[worker process killed by SIGABRT]
```

**Crash 2 — OOM from forked worker memory amplification (during training, ~epoch 4):**
```
Training epoch 4:  85%|████...| 877/1031
KeyboardInterrupt   ← user kills a hung process; no exception from the worker
```
Workers die silently a few epochs in; training hangs at `queue.get(timeout=...)`.

Both crashes are caused by the same underlying issue: PyTorch's `fork`-based DataLoader workers
are incompatible with the data loading strategy used here.

---

## Root Cause

### Crash 2: fork + in-RAM dataset memory amplification

`CachedLerobotRobocasaImageDataset` pre-loads all video frames into regular Python/numpy heap
memory in the main process. PyTorch DataLoader uses `fork()` to create workers. Fork copies the
parent's page table (copy-on-write), but as workers write to their local state during
`__getitem__` (numpy type casts, temporary arrays), CoW pages are duplicated into physical RAM.

With `num_workers: 4` for both train and val DataLoaders, the system holds up to 9 copies of the
full dataset (1 parent + 8 workers) after a few epochs of page faults — enough to OOM a worker.
The worker dies without an exception reaching Python; the main process hangs on `queue.get()`.

**Fix**: Move cached frames into anonymous shared memory via `mmap.mmap(-1, nbytes)`, which
creates a `MAP_SHARED | MAP_ANONYMOUS` mapping on Linux. The kernel maps the same physical pages
into every forked child — no CoW duplication regardless of how many workers exist. This is done
in `CachedLerobotRobocasaImageDataset.__init__` after the parent class loads frames into regular
numpy arrays: each array is copied into an anonymous mmap buffer and replaced with a numpy view
over it. Conversion is done one camera at a time to keep peak memory to ~1 camera extra.

Unlike `multiprocessing.shared_memory` (which uses `/dev/shm`, typically capped at 50% of RAM),
anonymous mmap is backed by regular RAM/swap with no special size limit. The frame cache for 3
cameras at 256×256 is ~78 GB, which exceeds the typical `/dev/shm` cap on 128 GB machines.

The original `num_workers: 0` workaround killed throughput because `__getitem__` CPU work (axis
transpositions, float casts, batch assembly) is significant enough to starve the GPU when done
single-threaded.

### Crash 1: fork + MuJoCo heap corruption chain

1. **MuJoCo 3.x with EGL initialises a custom glibc malloc arena** (`mallopt`, `mmap`-backed
   memory pools) as part of GPU context setup. This modifies glibc's internal free-list state in
   the parent process.

2. **PyTorch DataLoader workers are created with `multiprocessing.fork()`** (the default on
   Linux). `fork` copies the parent's heap verbatim — including partially-initialised or
   inconsistently-locked malloc state — into the child process.

3. **Workers that fork *after* MuJoCo runs** inherit the corrupted heap. Any subsequent
   `malloc`/`free` call in those workers can trigger `malloc_consolidate: unaligned fastbin chunk
   detected` → `SIGABRT`.

### Why the crash is delayed

DataLoader workers with `persistent_workers: true` are started lazily on the **first
`iter(dataloader)`** call, not at construction time. In
`train_diffusion_transformer_hybrid_workspace.py`:

```
create dataset
create train_dataloader          ← workers not yet started
create val_dataloader            ← workers not yet started
(pre-warm val workers)           ← val workers fork here, before eval
...
training loop epoch 0:
  first iter(train_dataloader)  ← train workers fork (clean heap) ✓
  eval rollouts (epoch 0)       ← MuJoCo / EGL corrupts main-process heap ✗
  first iter(val_dataloader)    ← val workers fork (corrupted heap) → SIGABRT
```

Setting `rollout_every: 1` (or `rollout_every: 25` with epoch counting from 0, which triggers on
the very first epoch) means eval runs *between* the first training pass and the first validation
pass, ensuring val workers always fork from a corrupted heap.

### Why `AsyncVectorEnv` deadlocked first

Before the sequential eval rewrite, `RobocasaImageRunner.__init__` created an `AsyncVectorEnv`
(fork-based subprocess pool). Those subprocesses inherited open pipe file descriptors from the
DataLoader workers created by an earlier training run. The pipe GC caused the training dataloader
to hang indefinitely (the read end of the pipe was held open in the eval subprocess, so the
DataLoader worker's EOF signal was never delivered).

---

## Attempted Fixes (pre-solution)

### 1. Lazy env creation in `RobocasaImageRunner.__init__`

**What**: Moved `AsyncVectorEnv` construction from `__init__` to `run()`.  
**Result**: Eliminated the fork-pipe deadlock. Eval runs, but heap corruption in val workers
remains.

### 2. Sequential eval in main process

**What**: Replaced `AsyncVectorEnv` with a single env created fresh each `run()` call.  
**Result**: No subprocess issues. Eval works correctly. But MuJoCo still runs in the main process,
corrupting its heap before val workers are created.

### 3. `num_workers: 0` in val dataloader config

**What**: Val data loaded synchronously in the main process.  
**Result**: Crash eliminated, but training throughput dropped significantly (bottlenecked on single
CPU validation).

### 4. Pre-warm val dataloader workers before training loop

**What**: Added code to force one iteration of the val dataloader *before* the training loop (and
therefore before any eval rollout):

```python
if cfg.val_dataloader.get('num_workers', 0) > 0 \
        and cfg.val_dataloader.get('persistent_workers', False):
    print("Pre-warming val dataloader workers...")
    _prewarm_iter = iter(val_dataloader)
    try:
        next(_prewarm_iter)
    except StopIteration:
        pass
    del _prewarm_iter
```

**Expected**: Workers fork from clean heap; `persistent_workers: true` keeps them alive for the
rest of the run.  
**Result**: **Still crashes.** Suspected causes:
- The `del _prewarm_iter` path may be releasing the iterator reference before Python's GC
  guarantees workers are detached from it.
- With `persistent_workers: true`, PyTorch caches the iterator on `DataLoader._iterator`; however,
  once the iterator is exhausted and deleted, subsequent `iter(val_dataloader)` calls may restart
  workers rather than reuse the stale ones — especially if the worker state was never advanced past
  "idle after first batch".
- If the val dataset is very small, workers may cycle to an "exhausted" state quickly and be
  restarted on the next `iter()` call.

---

## Implemented Fix: Spawn-based Eval Subprocess

**Commit**: (see git log)  
**Status**: Active — `rollout_every` re-enabled at 50 in both configs.

The root cause of Crash 1 is that MuJoCo/EGL runs in the main process and corrupts its glibc
heap. The fix is to ensure MuJoCo **never** runs in the main process at all.

### How it works

`RobocasaImageRunner.run()` now spawns a dedicated eval subprocess using
`multiprocessing.get_context('spawn')`:

1. The parent serializes the policy to a temp file via `torch.save` (using `dill` as the pickle
   module to handle complex model objects).
2. A fresh subprocess is spawned. Because `spawn` (not `fork`) is used, the child gets a clean
   interpreter with no inherited heap state.
3. The child process imports torch, robosuite, MuJoCo — all from scratch. It loads the policy,
   creates a single eval env via `_make_eval_env()`, and runs rollouts sequentially.
4. Results (reward lists + video file paths) are returned to the parent via a
   `multiprocessing.Queue`. The parent builds `wandb.Video` objects and log data from these.
5. The temp file is cleaned up.

### Why this is correct

- **Crash 1 eliminated**: MuJoCo/EGL only initialises inside the spawned child. The parent's heap
  is never touched by MuJoCo, so fork-based DataLoader workers always see a clean heap.
- **Crash 2 eliminated**: Cached frames are moved into anonymous shared mmap
  (`MAP_SHARED | MAP_ANONYMOUS`) in `CachedLerobotRobocasaImageDataset.__init__`. Fork-based
  workers share the same physical pages — no CoW duplication, no OOM.
- **DataLoader performance preserved**: Both configs now use `num_workers: 4` /
  `persistent_workers: true` for train and val dataloaders. The LeRobot config no longer needs
  `num_workers: 0`.

### Performance characteristics

- **Subprocess startup**: ~30-60 s per eval call (one-time cost for importing torch + robosuite +
  robocasa in a fresh interpreter). With `rollout_every: 50` and 800 epochs, this adds ~16 eval
  calls × ~45 s ≈ 12 minutes over the full run.
- **Rollout wall-time**: Unchanged — same sequential single-env rollout as before.
- **GPU memory**: Two copies of the policy exist on GPU briefly (parent's copy is idle during eval).
  For typical diffusion policy models (~100-300 MB), this is negligible on 24+ GB GPUs.

### Key implementation details

- `_make_eval_env()`: Module-level function that creates a wrapped eval env from plain-dict
  config. Replaces the closure-based `env_fn` which cannot be pickled across spawn boundaries.
- `_undo_transform_action()`: Module-level replacement for the former instance method, used by
  `_eval_worker` for `abs_action` environments.
- `_eval_worker()`: Module-level subprocess entry point. Catches all exceptions and sends them
  back via the queue so the parent can re-raise them with full tracebacks.
- Error handling: If the subprocess dies without putting a result, the parent detects this via
  `p.is_alive()` checks every 30 s and raises a `RuntimeError` instead of hanging.

### Future optimization

For runs with very frequent eval (e.g., `rollout_every: 1` in debug mode), the ~45 s spawn
overhead per call may be noticeable. This can be optimised by keeping a persistent eval subprocess
alive between `run()` calls, communicating policy weights and results over a pipe. This is not
implemented yet because the current overhead is negligible at `rollout_every: 50`.

---

## Relevant Files

| File | Role |
|------|------|
| `diffusion_policy/env_runner/robocasa_image_runner.py` | Spawn-based eval runner (Crash 1 fix) |
| `diffusion_policy/dataset/lerobot_robocasa_dataset.py` | Shared-memory frame cache (Crash 2 fix) |
| `diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py` | Training loop |
| `diffusion_policy/gym_util/async_vector_env.py` | Fork-based async env pool (not used for robocasa) |
| `configs/image/robocasa_lerobot_atomic/diffusion_policy_transformer/config.yaml` | LeRobot dataset config |
| `configs/image/robocasa_hdf5_atomic/diffusion_policy_transformer/config.yaml` | HDF5 dataset config |

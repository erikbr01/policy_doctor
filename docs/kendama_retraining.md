# Kendama per-instruction retraining

Recipe for retraining four kendama diffusion policies — one per language
instruction — on top of the existing may13+may19+may20 baseline.

Each retraining arm uses the **same 247-episode baseline** plus **50 new
episodes** (selected randomly per language instruction) collected on
2026-05-21 / 2026-05-22.

---

## Overview

```
RAW DROID DATA (on this host)
─────────────────────────────
  ~/data/droid_data/data/{success,failure}/<date>/<uuid>/
    trajectory.h5          state + actions (DROID HDF5; current_task attr
                                            holds the language instruction)
    recordings/SVO/        14313307.svo2 (wrist) + 36716034.svo2 (ext1)

BASELINE (already converted)
────────────────────────────
  ~/kendama_may13_may20.hdf5            247 demos, ~33 GB, robomimic layout

PER-INSTRUCTION (this workflow)
───────────────────────────────
  ~/data/droid_data/staging_may22/<slug>/         50 symlinked traj dirs
  ~/data/droid_data/kendama_may22_<slug>_raw.hdf5 50 demos, ~4-5 GB each
  ~/data/droid_data/kendama_may13_may22_<slug>.hdf5
                                                  297-demo index file (~40 KB)
                                                  ExternalLinks → baseline + raw
```

Slugs used throughout:

| slug | DROID `current_task` attr |
|---|---|
| `one_grasp_close`  | `kendama - one grasp - close to box` |
| `one_grasp_away`   | `kendama - one grasp - away from box` |
| `two_grasps_close` | `kendama - two grasps - close to box` |
| `two_grasps_away`  | `kendama - two grasps - away from box` |

---

## Prerequisites

1. **Conda env**: `policy_doctor` (analysis / conversion). See
   [`droid_robot_setup.md`](droid_robot_setup.md) for general DROID setup.
2. **ZED SDK + pyzed + calibration**: required by the converter to decode
   `.svo2` recordings. Run [`scripts/install_droid_zed_deps.sh`](../scripts/install_droid_zed_deps.sh)
   once; it is idempotent. On this GCE host it also extracts
   `libnvcuvid.so.1` / `libnvidia-encode.so.1` from the matching driver-major
   `.deb` (the headless GCE driver doesn't ship those, but installing them
   system-wide is not required — the script just unpacks the libs locally).

   After installation, conversion needs:

   ```bash
   export LD_LIBRARY_PATH=~/zed_sdk_extracted/lib:~/nvidia_decode_extra/usr/lib/x86_64-linux-gnu
   ```

---

## Recipe

### 1. Audit the new data

Get the distribution of `current_task` across the new collection dates to
confirm you have ≥ 50 successes per target instruction:

```python
# In the policy_doctor env. Adjust roots for the dates of interest.
import glob, h5py, os
from collections import Counter

roots = ["~/data/droid_data/data/success/2026-05-21",
         "~/data/droid_data/data/success/2026-05-22"]
counts = Counter()
for r in roots:
    for traj in glob.glob(os.path.join(os.path.expanduser(r), "*", "trajectory.h5")):
        with h5py.File(traj, "r") as f:
            if not bool(f.attrs.get("success", False)):
                continue
            task = f.attrs.get("current_task")
            counts[task.decode() if isinstance(task, bytes) else task] += 1
print(counts.most_common())
```

Watch for typo variants of the instruction (e.g. `"turned right"`,
`"turned away"`) and the DROID default placeholder
(`"Do any task, and then reset the scene. ..."`). The conversion below
filters by exact-string match on `current_task`.

### 2. Stage 50 random episodes per instruction

Symlinks rather than copies so the converter's directory walk picks them up
without duplicating data. Fixed seed for reproducibility.

```python
import glob, h5py, os, random

TARGETS = {
    "one_grasp_close":  "kendama - one grasp - close to box",
    "one_grasp_away":   "kendama - one grasp - away from box",
    "two_grasps_close": "kendama - two grasps - close to box",
    "two_grasps_away":  "kendama - two grasps - away from box",
}
roots = ["/home/erbauer/data/droid_data/data/success/2026-05-21",
         "/home/erbauer/data/droid_data/data/success/2026-05-22"]
stage_root = "/home/erbauer/data/droid_data/staging_may22"

per_task = {slug: [] for slug in TARGETS}
for r in roots:
    for traj in sorted(glob.glob(os.path.join(r, "*", "trajectory.h5"))):
        folder = os.path.dirname(traj)
        with h5py.File(traj, "r") as f:
            if not bool(f.attrs.get("success", False)):
                continue
            task = f.attrs.get("current_task")
            task = task.decode() if isinstance(task, bytes) else task
        for slug, target in TARGETS.items():
            if task == target:
                per_task[slug].append(folder); break

for slug, folders in per_task.items():
    rng = random.Random(42); rng.shuffle(folders)
    stage_dir = os.path.join(stage_root, slug)
    os.makedirs(stage_dir, exist_ok=True)
    for folder in sorted(folders[:50]):
        link = os.path.join(stage_dir, os.path.basename(folder))
        if not os.path.exists(link):
            os.symlink(folder, link)
```

### 3. Convert SVO → robomimic HDF5, per instruction

Run [`scripts/convert_droid_hdf5_debug.py`](../scripts/convert_droid_hdf5_debug.py)
on each staging dir. Output resolution **256×256** matches the kendama base
config; 12 workers saturates the 12-core machine.

```bash
conda activate policy_doctor
export LD_LIBRARY_PATH=~/zed_sdk_extracted/lib:~/nvidia_decode_extra/usr/lib/x86_64-linux-gnu

for slug in one_grasp_close one_grasp_away two_grasps_close two_grasps_away; do
  python scripts/convert_droid_hdf5_debug.py \
    --input_path ~/data/droid_data/staging_may22/${slug} \
    --output_path ~/data/droid_data/kendama_may22_${slug}_raw.hdf5 \
    --zed_settings ~/zed_settings \
    --image_size 256 256 \
    --num_workers 12
done
```

> **Watch out:** if `--zed_settings` is omitted, the script defaults to the
> long-stale `/mnt/ssdB/erik/zed_settings` path. ZED then can't find
> calibration, every SVO open fails, and the converter silently produces a
> 0-demo file before raising `RuntimeError: No valid trajectories found.`

Each run takes ~5-8 min and produces a 4-5 GB raw HDF5 (50 demos).

### 4. Merge baseline + raw via ExternalLink (zero-copy)

`scripts/merge_droid_hdf5.py` defaults to writing
[`h5py.ExternalLink`](https://docs.h5py.org/en/stable/high/group.html#h5py.ExternalLink)
references rather than copying bytes. Each merged file is a **~40 KB index**
that points into the baseline and the per-instruction raw HDF5. The dataset
loader resolves the links transparently via `data_grp[key]`, so training is
none the wiser.

```bash
for slug in one_grasp_close one_grasp_away two_grasps_close two_grasps_away; do
  python scripts/merge_droid_hdf5.py \
    --inputs ~/data/droid_data/kendama_may13_may20.hdf5 \
             ~/data/droid_data/kendama_may22_${slug}_raw.hdf5 \
    --output ~/data/droid_data/kendama_may13_may22_${slug}.hdf5
done
```

**Why ExternalLink rather than copying?**
On this GCE host the persistent disk reads at ~50 MB/s. Naïvely copying
the 33 GB baseline four times (once per arm) projected to ~60 min of pure
I/O. The ExternalLink path takes ~3 s for all four merges; the only thing
written is the new index file. The trade-off is that each merged HDF5
**depends on the source files staying in place** — don't delete the
baseline or the `_raw` files. Pass `--copy_data` if you need a
self-contained output (e.g. to ship the file elsewhere).

Expected counts (sanity check):

```bash
python -c "
import h5py
for slug in ['one_grasp_close','one_grasp_away','two_grasps_close','two_grasps_away']:
    p = f'/home/erbauer/data/droid_data/kendama_may13_may22_{slug}.hdf5'
    with h5py.File(p,'r') as f:
        n = sum(1 for k in f['data'] if k.startswith('demo_'))
        print(slug, n, int(f['data'].attrs['total']))
"
# all 4 should print 297, with total_steps in the ~94-98k range
```

### 5. Launch training

One experiment YAML per arm; each writes outputs to a distinct
`train_date` so the four runs don't collide.

```bash
for exp in kendama_may22_one_grasp_close \
           kendama_may22_one_grasp_away \
           kendama_may22_two_grasps_close \
           kendama_may22_two_grasps_away; do
  python -m policy_doctor.scripts.run_pipeline \
    data_source=droid_hdf5 \
    +experiment=${exp} \
    steps=[train_baseline]
done
```

Output dirs land under
`data/outputs/train/<train_date>/<train_date>_train_<policy>_<task>_<seed>/`.

---

## Key files

| Path | Role |
|---|---|
| [`scripts/install_droid_zed_deps.sh`](../scripts/install_droid_zed_deps.sh) | One-shot ZED SDK + pyzed + calibration + driver-extras setup |
| [`scripts/convert_droid_hdf5_debug.py`](../scripts/convert_droid_hdf5_debug.py) | DROID SVO → robomimic HDF5 (filters `movement_enabled=False` steps) |
| [`scripts/merge_droid_hdf5.py`](../scripts/merge_droid_hdf5.py) | Zero-copy merge (ExternalLink default; `--copy_data` for standalone) |
| [`policy_doctor/configs/experiment/kendama_may20.yaml`](../policy_doctor/configs/experiment/kendama_may20.yaml) | Existing baseline experiment (may13+may19+may20 combined) |
| [`policy_doctor/configs/experiment/kendama_may22_one_grasp_close.yaml`](../policy_doctor/configs/experiment/kendama_may22_one_grasp_close.yaml) | Arm 1 — one grasp / close to box |
| [`policy_doctor/configs/experiment/kendama_may22_one_grasp_away.yaml`](../policy_doctor/configs/experiment/kendama_may22_one_grasp_away.yaml) | Arm 2 — one grasp / away from box |
| [`policy_doctor/configs/experiment/kendama_may22_two_grasps_close.yaml`](../policy_doctor/configs/experiment/kendama_may22_two_grasps_close.yaml) | Arm 3 — two grasps / close to box |
| [`policy_doctor/configs/experiment/kendama_may22_two_grasps_away.yaml`](../policy_doctor/configs/experiment/kendama_may22_two_grasps_away.yaml) | Arm 4 — two grasps / away from box |
| [`third_party/cupid/configs/image/droid/diffusion_policy_cnn/config.yaml`](../third_party/cupid/configs/image/droid/diffusion_policy_cnn/config.yaml) | Base training config (horizon=16, n_action_steps=8, 256×256 → 224×224 crop) |
| [`third_party/cupid/diffusion_policy/dataset/robomimic_replay_image_dataset.py`](../third_party/cupid/diffusion_policy/dataset/robomimic_replay_image_dataset.py) | Dataset class (resolves ExternalLinks transparently; builds zarr cache) |
| [`docs/droid_robot_setup.md`](droid_robot_setup.md) | General DROID conversion / training reference |

---

## Training log

A running record of how the four-arm retraining went: launch commands,
parallelism decisions, VRAM / time observations, and incidents. Append new
entries at the **bottom** (most recent last).

Output directory pattern per arm (set by `train_date` in each experiment YAML):

```
third_party/cupid/data/outputs/train/may22_<slug>/
    may22_<slug>_train_diffusion_unet_hybrid_image_droid_0/
        .hydra/                 # full resolved config
        checkpoints/            # topk + last
        wandb/                  # local wandb metadata
```

Stdout/stderr from the pipeline subprocess is captured at
`/tmp/kendama_train_logs/arm<N>_<slug>.log` for each run we kick off (see
`Capacity probe`, below, for the first one).

### Hardware snapshot at launch (2026-05-23)

- 2× NVIDIA A100-SXM4-40 GB (both idle at start). Driver 580.126.20.
- 24 CPU cores, 167 GiB RAM, 720 GiB free disk on `/`.
- Persistent disk read throughput ~50 MB/s — drives most pacing decisions
  (see "Why ExternalLink rather than copying?" earlier in this doc and the
  zarr-cache notes below).

### Capacity probe — does 2 runs / GPU fit?

The four merged HDF5s share 247 / 297 demos (the baseline), so the only
unknown driving parallelism is the steady-state VRAM of a single training
run with the production config (`batch_size=256`, two 224×224 image streams,
ResNet18 backbones, UNet `[256, 512, 1024]`, EMA on, tf32 on, `compile=true`).

**Procedure:** launch arm 1 (`kendama_may22_one_grasp_close`) on `cuda:0`
first; observe peak VRAM after compile finishes and a few epochs land.

Decision table:

| peak VRAM (per run) | plan |
|---|---|
| ≲ 18 GB | 4 runs in parallel, 2 per GPU. Stagger the launches so zarr-cache builds don't overlap (each cache is ~38 GB written at ~50 MB/s ≈ 13 min). |
| 18-25 GB | 2 in parallel (1 per GPU), then the other 2 sequentially. |
| > 25 GB | Strictly sequential. |

**Launch command (arm 1):**

```bash
conda run -n policy_doctor --no-capture-output python -m policy_doctor.scripts.run_pipeline \
  data_source=droid_hdf5 \
  experiment=kendama_may22_one_grasp_close \
  device=cuda:0 \
  steps=[train_baseline] \
  > logs/kendama_retraining/arm1_one_grasp_close.log 2>&1
```

The pipeline ran in `policy_doctor` and dispatched `train.py` in
`mimicgen_torch2` (per `data_source/droid_hdf5.yaml :: conda_env_train`).

### Incidents

#### `+experiment=…` rejected: "Multiple values for experiment"

First launch attempt used `+experiment=kendama_may22_one_grasp_close` (the
`+` prefix one adds when introducing a new defaults group). Hydra rejected
it because the base `config.yaml` already declares `experiment: null` in its
`defaults:` list, so the group is **defined** and we are **overriding** it,
not adding it.

> Fix: drop the `+`. Use `experiment=kendama_may22_one_grasp_close`.

#### Inline list override grammar (`+baseline.diffusion_compose_overrides=[...]`)

Tried to inject `dataloader.num_workers=6` from the CLI via:

```
+baseline.diffusion_compose_overrides=["dataloader.num_workers=6", ...]
```

Hydra parses the **inner** strings as override expressions and chokes on
the `=` inside list elements:

> `no viable alternative at input '[dataloader.num_workers='`

Workarounds when this is actually needed later:
- Edit the experiment YAML's `diffusion_compose_overrides:` list directly,
  *or*
- Append a single item with the indexed-list form
  `baseline.diffusion_compose_overrides.4=…` with the RHS suitably escaped.

For the capacity probe we just used the YAML defaults (12 workers); this
is well within budget on a single run and only matters once we go to
4-parallel.

#### `/tmp` is not durable on this GCE host — use the project `logs/` dir

Pipeline stdout/stderr was initially captured to `/tmp/kendama_train_logs/`,
which doesn't persist across reboots and gets auto-cleaned. Moved to
`<project>/logs/kendama_retraining/` and saved a feedback memory so future
sessions default there.

The live arm-1 log was `mv`'d while training was running: Linux keeps the
writer's open file descriptor tied to the inode, so the running process
kept appending at the new path. Only the `tail -F` monitor had to be
restarted to follow the new directory entry.

### Pre-warming caches for the other arms

Arm 1's zarr-cache build (`Cache does not exist. Creating!` →
`Loading image data: <progress>`) ran roughly **5 min to 42%** for ~190 k
image rows, meaning the full build is ~12-15 min. Each arm's cache is
independent (different output path next to its merged HDF5), so the other
three were started **in parallel** rather than serialized — page-cache
reuse on the shared baseline portion is the win, since the baseline HDF5
fits in OS page cache (~33 GB vs 167 GiB RAM).

Bypassing `train.py` to keep the GPU clean: added
[`scripts/build_droid_zarr_cache.py`](../scripts/build_droid_zarr_cache.py).
It Hydra-composes the same droid CNN config and instantiates only the
`RobomimicReplayImageDataset`, which is the code path that builds the
`<dataset>.hdf5.zarr.zip` cache. No model, no CUDA context, no compile.

```bash
# Run inside mimicgen_torch2 (the training env); CPU-only, no GPU needed.
for slug in one_grasp_away two_grasps_close two_grasps_away; do
  conda run -n mimicgen_torch2 --no-capture-output \
    python scripts/build_droid_zarr_cache.py \
      /home/erbauer/data/droid_data/kendama_may13_may22_${slug}.hdf5 \
    > logs/kendama_retraining/cache_${slug}.log 2>&1 &
done
```

Concurrent state ~30 s in: arm-1 training-launched cache at ~16 GB in its
tmp_dir, arms 2/3/4 cache-only at ~150-300 MB each (past torch import,
into image-loop).

#### Arm 1 training failed mid-init with `KeyError('meta')` after cache built

After the cache-build print `Saving cache to disk.`, the training process
exited with:

```
Error in call to target 'diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset':
KeyError('meta')
full_key: task.dataset
```

Hydra suppressed the inner traceback by default. Important note for next
time: re-run with `HYDRA_FULL_ERROR=1` set in the **outer** environment so
the inner subprocess sees it.

**Investigation:**
- The `.zarr.zip` file written by save_to_store was **complete and valid** —
  297 episodes, 94 611 steps, all data + meta keys present.
- The standalone `scripts/build_droid_zarr_cache.py` reproducer **succeeded**
  when run against the same merged HDF5 with the same cache present
  (`len(dataset)=84116`).
- So the failure is not in the cache build itself, and probably also not in
  the bare `RobomimicReplayImageDataset.__init__` — it appears in the path
  that the train.py subprocess takes but the standalone script does not.
  Likely candidates: the `+task.dataset.dataset_mask_kwargs.max_val_episodes=1`
  override propagating into a code branch the standalone doesn't exercise,
  or a downstream consumer (normalizer / sampler) that touches `meta` in a
  way the standalone skip.

**Followup:** killed the in-flight cache builders for arms 2/3/4 (SIGTERM,
exit 143) and deleted their corrupt partial `.zarr.zip` files + `.tmp_dir`
directories. Kept arm 1's valid cache. Relaunched arm 1 with
`HYDRA_FULL_ERROR=1` — it loaded the cache from disk and started training
normally.

**Root cause and patch:** the bug is in `RobomimicReplayImageDataset.__init__`
([cache-build branch](../third_party/cupid/diffusion_policy/dataset/robomimic_replay_image_dataset.py)).
After `_convert_robomimic_to_replay(store=zarr.DirectoryStore(tmp_dir))`
returns, the code immediately calls `shutil.rmtree(tmp_dir)` — but
`replay_buffer.root.store` is still backed by that very `tmp_dir`. The
next line in the constructor (`replay_buffer.n_episodes`) then traverses
`self.root['meta']` against a deleted backing store → `KeyError('meta')`.

Existing caches sidestepped the bug because the `if os.path.exists(...)`
branch read from the on-disk `.zarr.zip` directly. The bug was masked
until the first cold cache build on this host.

Patch: after `save_to_store(zip_store)` succeeds and **before** removing
`tmp_dir`, re-open the just-written `.zarr.zip` into a fresh
`ReplayBuffer` (in-memory `MemoryStore` by default, matching what the
cache-hit branch does). This way the in-memory `replay_buffer` no longer
references the tmp DirectoryStore, and `shutil.rmtree(tmp_dir)` is safe.

### VRAM observation (arm 1, post-warmup)

- `nvidia-smi`: **18.7 GB used / 40 GB on cuda:0**, ~91 % util in the
  middle of an epoch.
- Throughput: ~5.4 it/s after warmup, 361 train iters/epoch + 1 val iter.
  Epoch 0 took ~6 min (warmup + initial validation); epoch 1 onward is
  ~1 min/epoch. 1000 epochs → **~17 h** projected per arm.

**Parallelism decision:** 2 × 18.7 = 37.4 GB ≪ 40 GB headroom is **2.6 GB**,
which is tight given validation spikes and torch caching allocator
fragmentation. Going with **1 run per GPU** (2 in parallel: cuda:0 + cuda:1),
then the remaining 2 in a second wave. Projected wall clock: ~17 h × 2
waves ≈ 34 h.

#### Per-arm VRAM grows past steady-state after first val/sample

Took a second snapshot once arm 1 had been running ~50 min (epoch 34) and
arm 2 ~15 min (epoch 16):

| GPU | PID | VRAM | epoch |
|---|---|---|---|
| 0 | arm 1  | 37 078 MiB | 34 |
| 1 | arm 2  | 16 412 MiB | 16 |

Same code, same config — the asymmetry is **lifetime, not steady-state**.
Arm 1 has gone through (a) the validation pass at end of epoch 0
(`Validation epoch 0`) and (b) the `sample_every=50` sample-logging pass
that the base config defaults to. Both run the policy through a different
`torch.compile` graph (eval mode and/or full-horizon sampling) plus EMA
snapshotting and an optimizer-state copy for the checkpointer scratch.

Torch's caching allocator **retains all of that as reusable cache** after
the tensors are freed. `nvidia-smi` reports it as "used" but it isn't the
active working set — it's just available to reuse.

**Implication:** the true per-run peak is **~37 GB, not 18.7 GB**. Arm 2 is
on the same trajectory and will climb to a similar level once it hits its
first val / sample (around epoch 49-50 with the current schedule).
2-per-GPU was never going to fit (2 × 37 ≫ 40); 1-per-GPU is necessary,
not just preferred. cuda:0 is already saturating at ~91 % util / ~92 %
memory with a single run.

**Update (~1 h later, both arms running):** the 37 GB on cuda:0 was
**transient, not sustained**. After the val/sample pass completed, the
torch caching allocator released the cached blocks; arm 1 settled back to
~18 GB at epoch 36. Steady-state per-arm VRAM during pure training is
~16-18 GB on this config; the ~37 GB figure is the peak during periodic
val/sample/checkpoint events. 2-per-GPU is still off the table because of
those spikes, but average-case headroom is much larger than the snapshot
suggested.

#### Always pin per-arm jobs with `CUDA_VISIBLE_DEVICES`

`nvidia-smi --query-compute-apps` after both arms had been running for a
while:

```
pid    gpu_uuid     used_gpu_memory
5224   GPU-0 (arm 1)   34 482 MiB    # main allocation
18285  GPU-0 (arm 2)      552 MiB    # ← stray primary CUDA context
18285  GPU-1 (arm 2)   16 412 MiB    # actual training
```

Arm 2 was launched with `device=cuda:1` only — but the torch subprocess
**still initialized a 552 MiB primary CUDA context on GPU 0**. Some
combination of cudnn / nccl / lib imports probes all visible devices at
process startup; the only reliable way to prevent it is to make the other
GPU invisible via `CUDA_VISIBLE_DEVICES`.

By itself the 552 MiB is small. The problem is that arm 1 already runs at
~35 GB on GPU 0 and **spikes to ~37 GB** during val/sample passes. Anything
else co-residing on GPU 0 narrows the margin and risks OOMing arm 1 at
spike time. Trying to debug an arm-1 OOM caused by arm-2's stray
allocation would be confusing — better to make the isolation hard.

**Rule for this workflow:** every per-arm pipeline launch sets
`CUDA_VISIBLE_DEVICES=<physical_gpu_index>` *and* `device=cuda:0`. Under
that env var, the visible GPU is always presented to torch as `cuda:0`,
regardless of which physical GPU it is.

```bash
# Arm on physical GPU 1, isolated:
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 \
  conda run -n policy_doctor --no-capture-output \
  python -m policy_doctor.scripts.run_pipeline \
    data_source=droid_hdf5 \
    experiment=kendama_may22_<slug> \
    device=cuda:0 \
    steps=[train_baseline] \
  > logs/kendama_retraining/<arm>.log 2>&1 &
```

Killed and relaunched arm 2 (lost ~19 epochs / ~15 min). Verified after
restart that arm 2 no longer holds any GPU 0 memory.

#### Correction: the 37 GB / 16 GB asymmetry was a snapshot artifact

After arm 1 reached epoch 70+, took another snapshot: **arm 1 was back at
17.6 GB**, not 37 / 39.7 GB. The high readings earlier coincided with arm
1's checkpoint + sample windows (`checkpoint_every=50`, `sample_every=50`,
both fire at epoch 0 and 50). The sample step calls
`policy.predict_action(obs_dict)` which runs the full 100-step diffusion
inference loop — under `torch.compile`, this is a separate compiled graph
from the training loop with its own activation pattern, and torch's
caching allocator keeps that graph + its scratch around as cache. The
cache eventually shrinks back once the allocator decides the spikes
won't repeat soon.

So the two arms **never had different memory at steady state** — they
both sit at ~17 GB during pure training. The earlier "arm 1 = 37 GB, arm
2 = 16 GB" snapshot caught arm 1 mid-spike and arm 2 mid-training, and I
incorrectly told the user the difference was real and growing.

**Implications, restated correctly:**
- Steady-state per-arm VRAM: **~17 GB**, both arms identical.
- Periodic spike (epoch 0, 50, 100, …): **~37 GB**, lasts seconds.
- 2-per-GPU is mathematically feasible at steady state (2 × 17 = 34 GB <
  40 GB) but the spike scenarios (2 × 37 ≫ 40) are deadly if both arms
  spike on the same epoch. Continuing with 1-per-GPU is still the safe
  call, but the headroom story is much better than I claimed.

Lesson logged: take **multiple snapshots over time** before reporting a
VRAM number to the user, especially for workloads with periodic
checkpoint / sample / eval phases. A single `nvidia-smi` reading is a
sample of a time series, not a steady-state measurement.

#### Per-epoch VRAM creep — root cause and patch

Once both arms were past their first val + sample + checkpoint window we
started seeing a slow per-epoch ratchet: arm 1 climbed from ~17 GB after
the epoch-0 event up to ~30+ GB by epoch ~80. Arm 2 showed the same
pattern once it crossed its own epoch-0 boundary (16 → 23 GB jump). Both
arms were approaching OOM and the user pushed for an actual fix rather
than waiting it out (the `kendama_may20` baseline had survived 1000
epochs on the same code, so it's bounded, but cutting it close on a
40 GB card with parallel runs is unwise).

**Diagnosis** (in `train_diffusion_unet_hybrid_workspace.py`):

```python
for local_epoch_idx in range(num_epochs_to_run):
    ...
    prefetcher = DataPrefetcher(train_dataloader, device, transform_fn=...)  # ← new CUDA stream each epoch
```

`DataPrefetcher.__init__` creates a `torch.cuda.Stream()`, and its
`__next__` calls `record_stream` on each fetched batch. `record_stream`
tells the CUDA caching allocator: "this block is in use by the current
stream; don't reclaim it until that stream has caught up." Reclamation
is event-based and lazy — when the next epoch starts, the previous
epoch's prefetcher (and its stream) is GC'd, but the recorded events on
those (now-defunct) streams may never be queried again, so the allocator
holds the associated blocks indefinitely. The leak is small per epoch
(~few hundred MB) but accumulates over hundreds of epochs.

**Patch:** added `torch.cuda.empty_cache()` at the end of each epoch in
[`train_diffusion_unet_hybrid_workspace.py`](../third_party/cupid/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py)
(right after `self.epoch += 1`). `empty_cache()` walks the caching
allocator's free-block pool and returns blocks to the CUDA driver,
breaking the record_stream / dead-stream retention chain. Cost: ~50-200
ms per epoch (negligible at ~1 min/epoch). Keeps `torch.compile` on.

**Verification (post-restart, both arms on patched workspace):**
- Epoch 3-4 on both arms simultaneously: GPU 0 = 17 627 MiB, GPU 1 =
  17 629 MiB. Symmetric to within 2 MiB.
- Both past the epoch-0 event that previously caused the ratchet.
- 94 % GPU util on both.

Killed and relaunched both arms to pick up the patch — lost ~80 epochs
on arm 1 (~80 min) and ~30 epochs on arm 2 (~30 min). Worth it for the
stability headroom and to verify the fix; the previous baseline ran ~18
hours and we expect ~17-18 hours per arm.

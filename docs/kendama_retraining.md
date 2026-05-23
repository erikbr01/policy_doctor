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

# DROID Real-Robot Setup

End-to-end guide for collecting data from the Franka DROID setup, training policies, and running inference with sync or async action scheduling.

---

## Overview

```
DATA COLLECTION (droid conda env, on NUC / workstation)
────────────────────────────────────────────────────────
  DROID collection stack  →  <date>/<uuid>/trajectory.h5   (DROID HDF5 format)
                          └→  <date>/<uuid>/recordings/MP4/ (per-camera video)

FORMAT CONVERSION (policy_doctor env, on workstation)
──────────────────────────────────────────────────────
  scripts/convert_droid_to_robomimic.py
    →  data/source/droid/droid_dataset.hdf5               (robomimic HDF5 format)

TRAINING + ATTRIBUTION (cupid_torch2 env)
──────────────────────────────────────────
  python -m policy_doctor.scripts.run_pipeline data_source=droid_hdf5
    steps=[train_baseline, train_attribution, run_clustering, ...]

INFERENCE (cupid_torch2 env, with policy server running)
─────────────────────────────────────────────────────────
  DROIDInferenceRunner  →  DROIDInferenceEnv  →  RobotEnv (Franka)
                        └→  WebSocketPolicy  →  openpi server (pi0/pi0.5)
                         or HttpPolicy       →  policy_server.py (diffusion_policy)
```

---

## Step 1 — Convert raw trajectories to robomimic HDF5

```bash
conda activate policy_doctor
python scripts/convert_droid_to_robomimic.py \
    --input_path /path/to/droid/success \
    --output_path data/source/droid/droid_dataset.hdf5 \
    --wrist_serial 14313307 \
    --ext1_serial  36716034 \
    --ext2_serial  37617599 \
    --image_size 180 320 \
    --action_space joint_velocity \
    --train_frac 0.9 \
    --val_frac 0.05
```

| Parameter | Default | Description |
|---|---|---|
| `--input_path` | required | Root folder of DROID trajectories (crawls recursively for `trajectory.h5`) |
| `--output_path` | required | Destination HDF5 path |
| `--wrist_serial` | `14313307` | ZED serial for wrist camera |
| `--ext1_serial` | `36716034` | ZED serial for left exterior camera |
| `--ext2_serial` | `37617599` | ZED serial for right exterior camera |
| `--image_size` | `180 320` | H W to resize images to in the output HDF5 |
| `--action_space` | `joint_velocity` | `joint_velocity` (8-dim: 7+1) or `cartesian_velocity` (7-dim: 6+1) |
| `--train_frac` | `0.9` | Fraction of trajectories for the `mask/train` split |
| `--val_frac` | `0.05` | Fraction for `mask/valid`; remainder goes to `mask/test` |

The script reads images from `recordings/MP4/` when present, otherwise falls back to HDF5-embedded images. Steps where the controller had `movement_enabled=False` are filtered out automatically.

### Output HDF5 layout

```
data/
    attrs: total = <total timesteps>
    demo_0/
        actions          (T, 8)   — joint_velocity (7) + gripper_position (1)
        obs/
            joint_positions       (T, 7)
            cartesian_position    (T, 6)
            gripper_position      (T, 1)
            hand_camera_image       (T, H, W, 3)   uint8 RGB
            exterior_image_1_left   (T, H, W, 3)
            exterior_image_2_left   (T, H, W, 3)
        dones    (T,)   — 1 on final step
        rewards  (T,)   — 1 on final step for successful trajectories
    demo_1/ ...
mask/
    train / valid / test   — lists of demo keys
```

---

## Step 2 — Run the pipeline

```bash
conda activate policy_doctor
python -m policy_doctor.scripts.run_pipeline \
    data_source=droid_hdf5 \
    steps=[train_baseline,train_attribution,run_clustering] \
    task_config=droid
```

The `droid_hdf5` data source loads the baseline, attribution, and evaluation configs from `policy_doctor/configs/robomimic/*/image/droid.yaml`. These work identically to the robomimic/mimicgen data sources — TRAK, InfEmbed, clustering, and curation all run unchanged.

Override the dataset path per experiment:
```bash
python -m policy_doctor.scripts.run_pipeline \
    data_source=droid_hdf5 \
    baseline.diffusion_dataset_path=/abs/path/to/droid_dataset.hdf5 \
    steps=[train_baseline]
```

---

## Step 3 — Inference

The inference stack runs in `cupid_torch2`. It needs either the openpi server (for pi0/pi0.5) or the Flask policy server (for diffusion_policy checkpoints) running separately.

### Launch a policy server

**openpi (WebSocket, recommended for pi0/pi0.5):**
```bash
# In the openpi repo:
uv run scripts/serve_policy.py --env=DROID
```

**Flask (HTTP, for diffusion_policy checkpoints):**
```bash
conda activate policy_doctor
python -m policy_doctor.envs.policy_server \
    --checkpoint data/outputs/train/.../best.ckpt \
    --device cuda:0 \
    --port 5001
```

### Run inference

```python
from policy_doctor.envs.droid_env import DROIDInferenceEnv
from policy_doctor.envs.droid_runner import DROIDInferenceRunner, InferenceMode, StaleActionPolicy
from policy_doctor.envs.droid_policy_wrappers import WebSocketPolicy, HttpPolicy

# Choose policy backend
policy = WebSocketPolicy(host="127.0.0.1", port=8000, external_camera="ext1")
# or: policy = HttpPolicy(url="http://localhost:5001")

env = DROIDInferenceEnv(
    action_space="joint_velocity",
    wrist_serial="14313307",
    ext1_serial="36716034",
    ext2_serial="37617599",
    output_dir="data/droid_eval_runs",
)

runner = DROIDInferenceRunner(
    env=env,
    policy=policy,
    mode=InferenceMode.ASYNC_CHUNK,          # recommended
    open_loop_horizon=8,
    max_timesteps=600,
    control_hz=15.0,
    output_dir="data/droid_eval_runs",
)

record = runner.run_episode(instruction="pick up the red cup")
print(f"steps={record.n_steps}  avg_chunk_ms={record.avg_chunk_latency_ms:.0f}")
```

For a full multi-episode loop:
```python
records = runner.run(n_episodes=5, instructions="pick up the red cup")
```

---

## Key files

| File | Description |
|---|---|
| `scripts/convert_droid_to_robomimic.py` | Converts DROID `trajectory.h5` folders → robomimic HDF5 |
| `policy_doctor/envs/droid_env.py` | `DROIDInferenceEnv` — robot wrapper with obs preprocessing |
| `policy_doctor/envs/droid_policy_wrappers.py` | `WebSocketPolicy` (openpi) and `HttpPolicy` (Flask) |
| `policy_doctor/envs/droid_runner.py` | `DROIDInferenceRunner` — sync / async-chunk / async-streaming |
| `policy_doctor/configs/data_source/droid_hdf5.yaml` | Hydra pipeline data source config |
| `policy_doctor/configs/robomimic/baseline/image/droid.yaml` | Baseline training config |
| `policy_doctor/configs/robomimic/attribution/image/droid.yaml` | Attribution (TRAK/InfEmbed) config |
| `policy_doctor/configs/robomimic/evaluation/image/droid.yaml` | Evaluation config |

---

## Inference modes

### `SYNC`
Inference blocks the control loop. At every `open_loop_horizon` boundary the runner calls `policy.get()` before proceeding. Matches the original `pi_eval.py` behaviour. Use when latency is low enough that blocking doesn't miss the control deadline.

### `ASYNC_CHUNK` _(recommended)_
At the start of each chunk, inference for the *next* chunk is submitted immediately to a background thread. The robot executes the current chunk at `control_hz` while inference runs concurrently. At the chunk boundary the runner blocks for the result — which should already be available. Net effect: inference time is hidden inside execution time.

```
chunk N running:  [step 0][step 1]...[step H-1]
inference N+1:    [────────────────────────────submit → get────]
chunk N+1:                                                      [step 0]...
```

### `ASYNC_STREAMING`
A dedicated background thread runs inference in a tight loop, always using the latest observation. The control loop reads from a lock-protected buffer. Useful when inference is slower than one control step but faster than one chunk, so you want the freshest possible prediction at all times.

Configurable stale-action behaviour when the buffer is empty:

| `StaleActionPolicy` | Behaviour |
|---|---|
| `HOLD_LAST` _(default)_ | Repeat the last action from the current chunk |
| `ZERO_ON_STALE` | Send zero-velocity command (robot stops in place) |

---

## `DROIDInferenceEnv` parameters

| Parameter | Default | Description |
|---|---|---|
| `action_space` | `"joint_velocity"` | `"joint_velocity"` (8-dim) or `"cartesian_velocity"` (7-dim) |
| `wrist_serial` | `"14313307"` | ZED serial for wrist camera |
| `ext1_serial` | `"36716034"` | ZED serial for left exterior camera |
| `ext2_serial` | `"37617599"` | ZED serial for right exterior camera |
| `dry_run` | `False` | Skip hardware init; `get_obs()` returns random tensors, `step()` is a no-op |
| `record_data` | `True` | Accumulate per-step data for `save_episode()` |
| `output_dir` | `None` | Default directory for `save_episode()` |

`get_obs()` returns a dict with keys: `joint_position (7,)`, `gripper_position (1,)`, `cartesian_position (6,)`, `wrist_image (H, W, 3)`, `exterior_image_1_left (H, W, 3)`, `exterior_image_2_left (H, W, 3)`. Images are BGR→RGB converted but **not resized** — the policy backend handles resizing.

`save_episode()` writes a DROID-format `trajectory.h5` compatible with `convert_droid_to_robomimic.py` for re-ingestion into the attribution pipeline.

---

## `DROIDInferenceRunner` parameters

| Parameter | Default | Description |
|---|---|---|
| `mode` | `ASYNC_CHUNK` | `InferenceMode.SYNC / ASYNC_CHUNK / ASYNC_STREAMING` |
| `stale_action_policy` | `HOLD_LAST` | `StaleActionPolicy.HOLD_LAST / ZERO_ON_STALE` (streaming only) |
| `open_loop_horizon` | `8` | Actions per chunk before requesting a new one (SYNC / ASYNC_CHUNK) |
| `max_timesteps` | `600` | Hard step limit per episode |
| `control_hz` | `15.0` | Target control frequency; runner sleeps to maintain this rate |
| `output_dir` | `None` | If set, saves each episode via `env.save_episode()` |

### `EpisodeRecord` fields

| Field | Description |
|---|---|
| `episode_idx` | Episode index within the run |
| `instruction` | Language instruction passed to the policy |
| `n_steps` | Total environment steps taken |
| `n_new_chunks` | Number of inference calls completed |
| `n_stale_steps` | Steps where the stale-action policy was applied (streaming only) |
| `avg_chunk_latency_ms` | Mean inference round-trip time in milliseconds |
| `chunk_latencies_ms` | Per-chunk latency list |

---

## `WebSocketPolicy` parameters

| Parameter | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | openpi server host |
| `port` | `8000` | openpi server port |
| `external_camera` | `"ext1"` | Which exterior image to send: `"ext1"` (left) or `"ext2"` (right) |
| `wrist_key` | `"wrist_image"` | Key in the obs dict for the wrist image |

Images are resized via `openpi_client.image_tools.resize_with_pad` to 224×224 before sending, matching pi0/pi0.5 training resolution.

> **Note:** `HttpPolicy` is for diffusion_policy low-dim checkpoints only. It sends a state vector (joint + gripper) and ignores image keys. Do not use it with image policies.

---

## Conda environment

| Step | Env |
|---|---|
| Data collection | `droid` (on NUC / workstation, with ZED SDK + Franka RDK) |
| Format conversion | `policy_doctor` |
| Training + attribution | `cupid_torch2` |
| Inference (real robot) | `cupid_torch2` (needs `droid` package on `PYTHONPATH`) |

The inference env requires `~/src_droid` on `sys.path`. `DROIDInferenceEnv` adds it automatically via `_ensure_droid_on_path()`.

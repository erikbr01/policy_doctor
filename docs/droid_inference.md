# Running diffusion-policy inference on the real DROID rig

End-to-end guide for serving a trained image-diffusion-policy checkpoint to the
real Franka via the standalone client `scripts/run_droid_diffusion_inference.py`.
Companion to [`docs/droid_robot_setup.md`](droid_robot_setup.md) (data collection
+ training); this doc only covers the inference / rollout side.

The setup runs as three separate processes on the workstation, each in its own
conda / uv environment. They communicate over localhost:

```
  Terminal A   policy server      (cupid_torch25, GPU)
                  │
                  │ HTTP POST /infer_dict
                  ▼
  Terminal C   inference client   (zed_env)           ←──── ZMQ PULL ──── Terminal B
              run_droid_diffusion_inference.py                            pi_eval_viser.py
                                                      ────── ZMQ PUSH ───►   (viser dashboard,
                                                                              openpi/.venv)
```

The split exists because the ZED SDK + Franka RDK stack lives in one Python env,
the diffusion_policy + torch stack lives in another, and viser + pyroki need
Python ≥ 3.10. They never share a process.

---

## 1. One-time setup

### 1.1 Build the `cupid_torch25` env (server side)

Hosts the diffusion-policy checkpoint + Flask `/infer_dict` endpoint. Built
from scratch (does not depend on a pre-existing `cupid_torch2` env):

```bash
cd /home/hardware/code/erik/policy_doctor
bash scripts/create_cupid_torch25_from_yaml.sh
```

Takes ~5 min. Installs torch 2.5.1+cu124 plus everything in
`third_party/cupid/conda_environment.yaml` (minus the simulator deps —
inference doesn't need MuJoCo / robosuite simulation).

**On Blackwell-class GPUs (RTX 5090, sm_120):** torch 2.5.1+cu124 only ships
kernels up to sm_90, so the first `predict_action` call will fail with
`CUDA error: no kernel image is available for execution on the device`. Fix:

```bash
conda run -n cupid_torch25 pip install --upgrade \
    torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
conda run -n cupid_torch25 python -c "
import torch
print(torch.__version__, torch.cuda.get_arch_list())
print(torch.cuda.get_device_properties(0).name,
      torch.cuda.get_device_capability(0))
"
# Expect:  2.7.0+cu128 [..., 'sm_120', 'compute_120']
#          NVIDIA GeForce RTX 5090 (12, 0)
```

### 1.2 Client env (robot side)

The inference client runs in `zed_env` (or `zed_env_spacemouse` — anywhere
the DROID stack and pyzed already work). It needs only `numpy`, `cv2`,
`requests`, plus `pyzmq` + `msgpack` if you want viser. All four are already
present in `zed_env_spacemouse`.

**Do not install `policy_doctor` into the robot env.** The client script
imports only `droid.robot_env.RobotEnv` and stdlib + the four packages above;
it has no `policy_doctor` dependency. Mixing the policy_doctor / torch stack
into the ZED env is what we are explicitly avoiding.

### 1.3 Dashboard env (optional, for viser)

`pi_eval_viser.py` (from droid-spacemouse) needs `viser`, `pyroki`,
`yourdfpy`, `robot_descriptions`, `msgpack`, `zmq`. The openpi uv venv has
all of these:

```bash
/home/hardware/code/openpi/.venv/bin/python --version   # 3.11
/home/hardware/code/openpi/.venv/bin/python -c "import viser, pyroki, msgpack, zmq"
```

If you need a fresh env: see the docstring at the top of
`droid-spacemouse/scripts/pi_eval_viser.py` for two recipes
(uv pip into the openpi venv, or a dedicated conda env).

### 1.4 Checkpoint location

The default checkpoint used during validation lives at:

```
checkpoints/may13_droid.ckpt   (1.5 GB)
```

You can store checkpoints anywhere; just point `--checkpoint` at the right
path in step 2.1 below.

---

## 2. Start the three processes

### 2.1 Terminal A — policy server

```bash
conda activate cupid_torch25
cd /home/hardware/code/erik/policy_doctor
python -m policy_doctor.envs.policy_server \
    --checkpoint checkpoints/may13_droid.ckpt \
    --device cuda:0 \
    --port 5001
```

Wait for the line `[server] ready on cuda:0`. Health check from any shell:

```bash
curl http://127.0.0.1:5001/health
# {"device":"cuda:0","status":"ok"}
```

Notes:

- The server has two endpoints. `/infer` is for low-dim policies and is
  unused for DROID. `/infer_dict` accepts an npz-serialised dict-of-arrays
  matching the checkpoint's `cfg.shape_meta.obs` and is what
  `run_droid_diffusion_inference.py` calls.
- Checkpoints saved while the model was wrapped with `torch.compile` have
  `._orig_mod.` injected into `state_dict` keys. The server strips that
  prefix at load time so a non-compiled instance can load them — no manual
  pre-processing of the checkpoint required.

### 2.2 Terminal B — viser dashboard (optional but recommended)

```bash
/home/hardware/code/openpi/.venv/bin/python \
    /home/hardware/code/erik/droid-spacemouse/scripts/pi_eval_viser.py \
    --port 5556 --pause-port 5557
```

Open the URL it prints (typically `http://localhost:8080`).

The dashboard's URDF **does not auto-animate predicted action chunks** — it
shows the live joint state (which doesn't change in `--dry-run`). To see
where the policy wants to go, click **"Play preview"** in the GUI panel; the
URDF will animate through the latest received chunk. Adjust `Velocity scale`,
`Preview horizon`, and `Play speed` to taste.

The dashboard also exposes **Pause** (suppresses `env.step()` while inference
keeps flowing — useful for "freeze the arm, watch what the policy would do")
and **Reset** (ends the rollout, homes the arm).

### 2.3 Terminal C — inference client

Three run modes, mutually exclusive:

```bash
conda activate zed_env
cd /home/hardware/code/erik/policy_doctor

# (a) Wire-format check only — no robot, fake observations
python scripts/run_droid_diffusion_inference.py --no-robot --max-timesteps 32

# (b) Dry-run — real RobotEnv (arm homes once on init), real obs, NO env.step
python scripts/run_droid_diffusion_inference.py \
    --dry-run \
    --viser-port 5556 --pause-port 5557 \
    --external-camera left \
    --max-timesteps 300

# (c) Live — real everything, ARM MOVES
python scripts/run_droid_diffusion_inference.py \
    --viser-port 5556 --pause-port 5557 \
    --external-camera left \
    --max-timesteps 60 \
    --max-joint-vel 0.5
```

The script runs an interactive **session loop** — one process drives many
rollouts back-to-back. See §2.4 for keyboard controls and trial cadence.

Key flags:

| Flag                  | Default              | Notes                                                                                  |
|-----------------------|----------------------|----------------------------------------------------------------------------------------|
| `--server-url`        | `http://127.0.0.1:5001` | Where the policy server is listening.                                                 |
| `--inference-mode`    | `async_chunk`        | `async_chunk`: after step 0 of each chunk fires, submit inference for the next chunk in a background thread; await at the chunk boundary. `sync`: block at every chunk boundary (old behaviour). See §2.5. |
| `--external-camera`   | `left`               | `left` → ZED 36716034, `right` → ZED 37617599. Whichever you pick is sent to the policy as `exterior_image_1_left`. |
| `--wrist-serial`      | `14313307`           | Wrist ZED serial. Sent to the policy as `hand_camera_image`.                          |
| `--max-timesteps`     | `600`                | Hard step cap **per rollout**. ~40 s at 15 Hz. **Use 60 for first live runs.**        |
| `--open-loop-horizon` | `8`                  | Steps from each predicted chunk to execute before requesting a new chunk.             |
| `--control-hz`        | `15.0`               | DROID's standard data-collection rate. Don't change unless you know why.              |
| `--max-joint-vel`     | `0.5`                | Hard cap on \|joint velocity\| in rad/s. **Conservative; expect to relax to 1.0–2.0 later.** |
| `--no-robot`          | off                  | No RobotEnv init, no env.step, fake obs. Wire-format check only.                      |
| `--dry-run`           | off                  | Real RobotEnv (homes the arm on init), real obs, env.step suppressed.                 |
| `--viser-port`        | `None`               | If set, publishes per-step state + predicted chunks via ZMQ PUSH to the dashboard.    |
| `--pause-port`        | `None`               | If set, listens for Pause/Reset commands from the dashboard.                          |
| `--output-dir`        | `data/droid_eval_runs` | Root dir for per-rollout subfolders (see §2.6). Pass `--no-recording` to disable.   |
| `--no-recording`      | off                  | Skip MP4 + HDF5 capture entirely.                                                     |
| `--no-success-prompt` | off                  | Skip the y/n/0-100 success prompt at the end of live rollouts.                       |

Note: `RobotEnv.__init__` always homes the arm. The `--dry-run` flag only
gates `env.step()` calls during the rollout, not the init reset.

### 2.4 Interactive session loop

The script does **not** exit after a single rollout. It runs an outer loop:

```
[ready] press Enter ──►  rollout  ──►  stop arm  ──►  label  ──►  env.reset() (home)  ──►  back to top
                                                                                                │
                                                                              q+Enter at prompt ┘  → exit
```

The cadence stays the same across the whole session — set up the scene,
hit Enter, watch the rollout, label it, then the arm homes itself and you
get the next prompt. No conda re-activation between trials.

Mid-rollout keyboard controls (stdlib `termios` cbreak — the terminal
running Terminal C must have focus):

| Key   | Effect |
|-------|--------|
| `Space` (or `p`) | Toggle pause. While paused: `env.step()` is suppressed (zero-velocity sent once on the toggle) so you can rearrange the scene by hand. On resume: the obs-history deque is cleared and `pred_chunk` is dropped, so the next loop iteration triggers a fresh inference from the post-reset observation — no stale-chunk execution. |
| `r`   | End the current rollout. The session continues: stop arm → label prompt → `env.reset()` homes the arm → "press Enter to start the next rollout". |
| `q`   | End the current rollout AND exit the session after labelling. The arm is stopped and the recorder is finalized. No homing (you've quit). |

Between-rollout prompts (cooked-mode `input()`, line-buffered):

- **Start gate**: `press Enter to start the [first|next] rollout, q+Enter to quit`. Nothing moves until you confirm — explicit handshake before every rollout.
- **Success label** (live mode only, unless `--no-success-prompt`): `y` / `n` / a number in `[0, 1]` / a number in `[0, 100]` / `s` to skip. Written into `meta.json` and `data/demo_0/rewards[-1]` of the HDF5.

The viser **Pause** and **Reset** buttons still work in parallel and OR
with the keyboard pause: any pause source pauses, all sources must clear to
resume. The viser **Reset** button maps to the same end-of-rollout path as
keyboard `r`.

If stdin is not a TTY (script under `nohup`, redirected, etc.), the keyboard
listener disables itself silently — the only mid-rollout controls are then
viser (if wired) and `Ctrl+C`. The session loop still works; you'll just
hit `max_timesteps` on every rollout.

### 2.5 Inference modes (sync vs async_chunk)

`--inference-mode async_chunk` (default) submits inference in a background
thread one step into each chunk's execution, then awaits the result at the
chunk boundary. With `open_loop_horizon=8` at 15 Hz, the policy has ~470 ms
of overlap window before the foreground needs the next chunk — comfortably
above the observed 250 ms p50 latency, so the chunk boundary rarely
stalls. Mirrors the `ASYNC_CHUNK` path in
`policy_doctor/envs/droid_runner.py`.

`--inference-mode sync` reverts to the original behaviour (block on every
chunk boundary). Use it when debugging timing, or when you want every chunk
to be conditioned on the latest possible observation.

On pause: any in-flight async inference is **orphaned** (the executor still
drains it, but the foreground discards the result on resume). The executor
runs `max_workers=2` so an orphaned inference doesn't block a fresh submit
after resume. The Flask server itself runs `threaded=False` and serializes
on its end, so only one inference is ever truly in flight; orphaning is
purely a foreground bookkeeping concern.

### 2.6 Rollout recording layout

Every rollout (live or `--dry-run`) writes a self-contained directory under
`--output-dir`, default `data/droid_eval_runs/<timestamp>/`:

```
data/droid_eval_runs/20260516_181500/
├── trajectory.hdf5      one demo, training-schema layout (loadable by RobomimicReplayImageDataset)
├── wrist.mp4            wrist ZED at native res, RGB, control_hz
├── exterior.mp4         exterior ZED at native res, RGB, control_hz
└── meta.json            args, instruction, latency summary, success
```

The HDF5 stores exactly what the policy saw and what the policy returned:

```
data/
    attrs: total = T
    demo_0/
        attrs: num_samples = T, success = <float|nan>
        actions          (T, 8)   float32  — clipped action sent to env.step()
        raw_actions      (T, 8)   float32  — pre-clip action selected from the chunk
        dones            (T,)     float32  — 1 on final step
        rewards          (T,)     float32  — 1 on final step if success
        executed         (T,)     bool     — env.step() actually called this step
        t                (T,)     int32    — global control-loop step index
        obs/
            hand_camera_image       (T, 256, 256, 3) uint8 RGB
            exterior_image_1_left   (T, 256, 256, 3) uint8 RGB
            joint_positions         (T, 7)           float32
            gripper_position        (T, 1)           float32
            cartesian_position      (T, 6)           float32
        inference/
            step_indices            (N_chunks,)      int32   — t when each inference fired
            predicted_chunks        (N_chunks, 8, 8) float32 — full chunk returned by the policy
            latency_ms              (N_chunks,)      float32
mask/
    test = ["demo_0"]
```

The `obs/` group matches `scripts/convert_droid_to_robomimic.py`'s output
exactly — image storage is `uint8 HWC RGB` at the same `256×256` resolution
training used. The `/255` + `transpose(2,0,1)` step that converts it to
the policy's `[0, 1] CHW float32` happens in `RobomimicReplayImageDataset`
at load time, so these rollouts plug into the influence-computation
pipeline without any reformatting.

To reconstruct the exact `(1, n_obs_steps=2, *)` window that was sent to
`/infer_dict` for inference call at index `i`:

```python
import h5py
with h5py.File("trajectory.hdf5", "r") as f:
    t = int(f["data/demo_0/inference/step_indices"][i])
    # window = [obs at t-1, obs at t], padded with obs at t when t == 0
    prev = max(0, t - 1)
    hand_win = f["data/demo_0/obs/hand_camera_image"][[prev, t]]   # (2, 256, 256, 3) uint8
    # apply training's load-time normalization
    hand_win = (hand_win.astype("float32") / 255.0).transpose(0, 3, 1, 2)[None]  # (1, 2, 3, 256, 256)
```

`meta.json` carries the run-level metadata:

```json
{
  "timestamp":   "20260516_181500",
  "mode":        "live",
  "instruction": "pick up the red cup",
  "n_steps":     63,
  "n_chunks":    8,
  "success":     1.0,
  "latencies_ms": {"mean": 254.1, "p50": 251.9, "p95": 389.4},
  "args":        { ... CLI args ... }
}
```

---

## 3. Going from dry-run to live

Order of operations for the first live rollout against a new checkpoint:

1. **`--no-robot`**, short. Confirms server is up, wire format is intact,
   chunks come back the right shape. See `[client] t=0 inference ...ms
   chunk shape=(8, 8)` and a non-zero `mean=...ms` summary.
2. **`--dry-run --viser-port 5556`**. Real obs flow through the policy.
   Click Play preview in viser; the predicted trajectory should look like
   something the policy plausibly wants to do for the task.
3. **Live, `--max-timesteps 60 --max-joint-vel 0.5`.** First live attempt is
   intentionally short and slow. Stand within reach of Pause / Reset / Ctrl+C
   / hardware E-stop.
4. **Iterate one knob at a time** — usually just `max_joint_vel` upward and
   `max_timesteps` longer. The training-data joint-velocity range goes to
   roughly ±1.5–3 rad/s, so 0.5 will visibly saturate joint 4 (the elbow).

---

## 4. How obs and actions map between training and inference

The checkpoint stores its expected obs schema under `cfg.shape_meta.obs`:

| Policy key                 | Shape         | Source on the robot                                                       |
|----------------------------|---------------|---------------------------------------------------------------------------|
| `hand_camera_image`        | `(3, 256, 256)` float in [0, 1] | wrist ZED, BGRA → RGB, INTER_AREA-resized 1280×720→256×256, /255, HWC→CHW |
| `exterior_image_1_left`    | `(3, 256, 256)` float in [0, 1] | selected exterior ZED, same pipeline                                      |
| `joint_positions`          | `(7,)` float32                  | `RobotEnv.get_observation()["robot_state"]["joint_positions"]`            |
| `gripper_position`         | `(1,)` float32                  | `[state["gripper_position"]]`                                             |

Each key is sent to `/infer_dict` with a leading `(1, n_obs_steps=2, …)` —
the client maintains a 2-step history deque and stacks consecutive frames.

The policy internally crops `256×256 → 224×224` (center crop;
`crop_shape=[224, 224]`, `eval_fixed_crop=True`). Don't pre-crop on the
client side.

Action chunk returned: `(n_action_steps=8, action_dim=8)`. Layout is
`[joint_vel_0..6, gripper_position]`. The action normalizer is identity
(checked at load time), so values are in raw training scale — no
post-scaling needed before `env.step()`.

### Image normalization sanity check

- Conversion script `scripts/convert_droid_to_robomimic.py` stores `(T, H, W, 3) uint8 RGB` after `cv2.INTER_AREA` resize.
- Training dataset (`RobomimicReplayImageDataset`) does `np.moveaxis(..., -1, 1).astype(np.float32) / 255.0` → CHW float [0, 1].
- Image normalizer (`get_image_range_normalizer()`) applies `scale=2, offset=-1` → policy sees CHW float in **[-1, 1]**.
- Inference client does the same `INTER_AREA` resize, the same `/255` step, and the same HWC→CHW transpose. The normalizer runs server-side on the [0, 1] input. End-to-end: no double-normalization, no channel swap drift.

---

## 5. Safety considerations

The script applies **only one** safety layer beyond what `RobotEnv` itself
enforces: `clip_action_safe()` in
`scripts/run_droid_diffusion_inference.py`:

- Joint velocities clipped to `±max_joint_vel`.
- Gripper command binarized at 0.5 (matches openpi's reference; throws
  continuous signal but eliminates threshold drift).

There is **no** E-stop on inference timeout, **no** workspace-bounds check,
**no** torque or position limit at the script layer. The DROID `RobotEnv`'s
internal limits are the only deeper net.

Abort ladder (least to most disruptive):

1. **`Space`** in Terminal C, or **Pause** in viser dashboard — keeps
   inference running, suppresses `env.step()`, sends zero-velocity once.
   Reversible: `Space` again (or unpause in viser) clears history and
   resumes with a fresh inference.
2. **`r`** in Terminal C, or **Reset** in viser dashboard — ends the
   current rollout cleanly. The session loop then stops the arm,
   prompts for a success label, and `env.reset()`s the arm before
   prompting for the next rollout.
3. **`q`** in Terminal C — same as `r`, but exits the session after
   labelling instead of starting another rollout.
4. **`Ctrl+C`** in Terminal C — abort current rollout immediately. The
   `finally` blocks still send one `env.step(zeros)` and (if applicable)
   finalize the recorder before exiting.
5. **Hardware E-stop**.

---

## 6. Architecture / why this isn't a `policy_doctor` import

The script is deliberately standalone — modeled on
`openpi/examples/droid/main.py` rather than the `DROIDInferenceEnv` /
`DROIDInferenceRunner` / `ImageHttpPolicy` abstractions inside
`policy_doctor.envs.*`. Reasons:

- The robot side cannot import `policy_doctor`'s package surface without
  pulling torch / robomimic (via the dagger-runner re-exports in
  `policy_doctor/envs/__init__.py`).
- A standalone client lets the robot env stay free of torch entirely.
- The wire format between the two processes is the only contract that
  matters; the script is the entire contract surface.

If you want to use the `DROIDInferenceEnv` / `DROIDInferenceRunner`
abstractions for an in-process / Streamlit integration, those still exist
under `policy_doctor/envs/droid_*.py` and work with the same
`/infer_dict` endpoint. They're just not the recommended path for the
real-robot rollout loop.

---

## 7. Files referenced

| Path                                                     | Role                                          |
|----------------------------------------------------------|-----------------------------------------------|
| `policy_doctor/envs/policy_server.py`                    | Flask `/infer` + `/infer_dict` server          |
| `scripts/run_droid_diffusion_inference.py`               | Standalone robot-side client                   |
| `scripts/create_cupid_torch25_from_yaml.sh`              | One-shot env build for the server side        |
| `checkpoints/may13_droid.ckpt`                           | Reference checkpoint (kendama task)           |
| `droid-spacemouse/scripts/pi_eval_viser.py`              | Viser dashboard (run from openpi/.venv)       |
| `scripts/convert_droid_to_robomimic.py`                  | Authoritative source for training-data preprocessing — referenced by section 4 |

"""Standalone DROID real-robot inference for diffusion_policy image checkpoints.

Modeled on droid-spacemouse/scripts/pi_eval_steering.py (control flow + viser
wire format) and openpi/examples/droid/main.py (loop structure).  Runs in the
robot conda env (zed_env / zed_env_spacemouse) — imports nothing from
policy_doctor or torch.  Talks to a remote policy server over HTTP (see
policy_doctor/envs/policy_server.py:/infer_dict).

Three processes:

    Terminal A  (cupid_torch25)  — policy server, GPU
        conda activate cupid_torch25
        python -m policy_doctor.envs.policy_server \\
            --checkpoint checkpoints/may13_droid.ckpt \\
            --device cuda:0 \\
            --port 5001

    Terminal B  (any env with viser + pyroki + msgpack + zmq, e.g. openpi/.venv)
        /home/hardware/code/openpi/.venv/bin/python \\
            /home/hardware/code/erik/droid-spacemouse/scripts/pi_eval_viser.py

    Terminal C  (zed_env or zed_env_spacemouse)  — this script
        conda activate zed_env
        python scripts/run_droid_diffusion_inference.py \\
            --dry-run --viser-port 5556 --external-camera right

Run modes:
    --no-robot              fake observations, no robot init, no env.step.
                            Use to validate the wire format end-to-end.

    --dry-run               REAL RobotEnv (which homes the arm on init), REAL
                            camera + state observations, but skip env.step().
                            The arm only moves once: the home-pose reset at
                            __init__ time. Pair with --viser-port to preview
                            the policy's predicted actions on the dashboard.

    (no flag, --live)       real everything. ARM WILL MOVE.

Interactive session loop:
    The script runs an outer loop over rollouts so you don't have to restart
    between trials. Per-rollout: press Enter to start, the policy runs until
    --max-timesteps (or you hit a key), then you label the trial, the arm
    homes via env.reset(), and you're prompted for the next.

    Mid-rollout keys (requires foreground terminal; stdlib termios cbreak):
        Space  pause/resume action output. While paused, env.step is
               suppressed so you can reset the scene by hand. On resume the
               obs history is cleared and a fresh inference is requested.
        r      end this rollout early (then label + home + prompt for next).
        q      end this rollout AND exit the session after labelling.

    Between rollouts:
        Enter        start the next rollout
        q + Enter    quit the session
        success label prompt accepts y / n / number-in-[0,1] / number-in-[0,100] / s to skip

Inference mode:
    --inference-mode sync          blocking inference at every chunk boundary
    --inference-mode async_chunk   (default) submit next chunk after the first
                                   step of the current chunk has executed,
                                   overlap inference with action playback.
                                   See policy_doctor/envs/droid_runner.py for
                                   the in-tree sim-side equivalent.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import select
import signal
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Configuration matching may13_droid.ckpt shape_meta
# ---------------------------------------------------------------------------

POLICY_IMAGE_KEYS  = ("hand_camera_image", "exterior_image_1_left")
POLICY_STATE_KEYS  = ("joint_positions", "gripper_position")
POLICY_IMAGE_HW    = (256, 256)
ACTION_DIM         = 8                 # joint_velocity (7) + gripper (1)
DROID_CONTROL_HZ   = 15.0
N_OBS_STEPS        = 2
OPEN_LOOP_HORIZON  = 8

# ZED camera serials on this rig (match training data)
WRIST_SERIAL = "14313307"
EXT1_SERIAL  = "36716034"   # "left" exterior in DROID parlance
EXT2_SERIAL  = "37617599"   # "right" exterior


# ---------------------------------------------------------------------------
# Obs preprocessing
# ---------------------------------------------------------------------------

def resize_to_policy(img_hwc_uint8: np.ndarray, target_hw=POLICY_IMAGE_HW) -> np.ndarray:
    """HWC uint8 → HWC uint8 at target_hw. INTER_AREA matches the training
    conversion pipeline (`scripts/convert_droid_to_robomimic.py:_resize_image`).
    """
    h, w = target_hw
    if img_hwc_uint8.shape[:2] == (h, w):
        return img_hwc_uint8
    return cv2.resize(img_hwc_uint8, (w, h), interpolation=cv2.INTER_AREA)


def to_policy_image(img_hwc_uint8_resized: np.ndarray) -> np.ndarray:
    """Resized HWC uint8 → CHW float32 in [0, 1]. Matches the /255 + transpose
    step in `RobomimicReplayImageDataset.__getitem__`."""
    return (img_hwc_uint8_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)


def extract_droid_obs(env_obs: dict, *, wrist_serial: str, ext_serial: str):
    """RobotEnv.get_observation() → (policy_obs, recording).

    `policy_obs` is what gets sent over the wire to /infer_dict (CHW float32
    in [0, 1], state as plain float32).

    `recording` has the same data in the form the training HDF5 stores it
    (HWC uint8 RGB at policy resolution + state) plus the camera frames at
    native ZED resolution for the MP4 writers.
    """
    images = env_obs["image"]
    wrist_img = ext_img = None
    for key, img in images.items():
        if wrist_serial in key and "left" in key:
            wrist_img = img
        elif ext_serial in key and "left" in key:
            ext_img = img
    if wrist_img is None or ext_img is None:
        raise RuntimeError(
            f"Missing camera streams. Got keys: {list(images.keys())}; "
            f"expected wrist={wrist_serial}, ext={ext_serial}."
        )
    wrist_native = wrist_img[..., :3][..., ::-1].copy()   # drop alpha, BGR→RGB, HWC uint8
    ext_native   = ext_img[..., :3][..., ::-1].copy()
    wrist_256 = resize_to_policy(wrist_native)            # HWC uint8 256×256 — training schema
    ext_256   = resize_to_policy(ext_native)

    state = env_obs["robot_state"]
    joint_pos = np.asarray(state["joint_positions"], dtype=np.float32)
    gripper   = np.asarray([state["gripper_position"]], dtype=np.float32)
    cart_pos  = np.asarray(state.get("cartesian_position", np.zeros(6)), dtype=np.float32)

    policy_obs = {
        "hand_camera_image":     to_policy_image(wrist_256),
        "exterior_image_1_left": to_policy_image(ext_256),
        "joint_positions":       joint_pos,
        "gripper_position":      gripper,
    }
    recording = {
        "wrist_native_rgb":      wrist_native,        # HWC uint8 RGB native res
        "exterior_native_rgb":   ext_native,
        "hand_camera_image":     wrist_256,           # HWC uint8 RGB 256×256
        "exterior_image_1_left": ext_256,
        "joint_positions":       joint_pos,
        "gripper_position":      gripper,
        "cartesian_position":    cart_pos,
    }
    return policy_obs, recording


def fake_obs():
    """No-robot path: zeros for state, black images. Same return shape as extract_droid_obs."""
    h, w = POLICY_IMAGE_HW
    black_256 = np.zeros((h, w, 3), dtype=np.uint8)
    black_nat = np.zeros((480, 640, 3), dtype=np.uint8)   # fake native res
    policy_obs = {
        "hand_camera_image":     np.zeros((3, h, w), dtype=np.float32),
        "exterior_image_1_left": np.zeros((3, h, w), dtype=np.float32),
        "joint_positions":       np.zeros(7, dtype=np.float32),
        "gripper_position":      np.zeros(1, dtype=np.float32),
    }
    recording = {
        "wrist_native_rgb":      black_nat,
        "exterior_native_rgb":   black_nat,
        "hand_camera_image":     black_256,
        "exterior_image_1_left": black_256,
        "joint_positions":       np.zeros(7, dtype=np.float32),
        "gripper_position":      np.zeros(1, dtype=np.float32),
        "cartesian_position":    np.zeros(6, dtype=np.float32),
    }
    return policy_obs, recording


def stack_history(history: deque) -> dict:
    """deque of per-step obs → dict of (1, n_obs, *shape) arrays."""
    return {
        k: np.stack([h[k] for h in history], axis=0)[None]   # (1, T, ...)
        for k in history[0]
    }


def policy_infer(url: str, history: deque, timeout: float = 30.0) -> np.ndarray:
    """POST stacked history to /infer_dict; return action chunk (T, action_dim)."""
    batched = stack_history(history)
    buf = io.BytesIO()
    np.savez(buf, **batched)
    resp = requests.post(
        f"{url.rstrip('/')}/infer_dict",
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=timeout,
    )
    resp.raise_for_status()
    chunk = np.load(io.BytesIO(resp.content))   # (1, n_action_steps, action_dim)
    return chunk[0]


def clip_action_safe(action: np.ndarray, max_joint_vel: float) -> np.ndarray:
    """Hard safety clip: joint velocities ∈ [-max, max], gripper binarized at 0.5.

    Only safety net besides DROID's RobotEnv internal limits. Default
    max_joint_vel of 0.5 rad/s is conservative — Franka spec allows up to
    2.0 rad/s.
    """
    out = action.astype(np.float32).copy()
    out[:7] = np.clip(out[:7], -max_joint_vel, max_joint_vel)
    out[7] = 1.0 if out[7] > 0.5 else 0.0
    return out


# ---------------------------------------------------------------------------
# Rollout recorder — MP4 + HDF5 in the training data schema
# ---------------------------------------------------------------------------

class RolloutRecorder:
    """Captures one rollout as a self-contained directory.

    Layout:
        <output_dir>/<timestamp>/
            trajectory.hdf5      one demo group; matches the training HDF5
                                 schema (see scripts/convert_droid_to_robomimic.py)
                                 so this rollout can be loaded directly by
                                 RobomimicReplayImageDataset for influence
                                 computation.
            wrist.mp4            wrist ZED at native res, RGB, control_hz
            exterior.mp4         exterior ZED at native res, RGB, control_hz
            meta.json            CLI args, instruction, latency summary, success

    Data stored:
        per-step (matches training schema exactly):
            obs/hand_camera_image       (T, 256, 256, 3) uint8 RGB
            obs/exterior_image_1_left   (T, 256, 256, 3) uint8 RGB
            obs/joint_positions         (T, 7)           float32
            obs/gripper_position        (T, 1)           float32
            obs/cartesian_position      (T, 6)           float32
            actions                     (T, 8)           float32 — the action
                                                          actually sent to env.step
                                                          (safety-clipped)
            dones                       (T,)             float32 — 1 on final step
            rewards                     (T,)             float32 — 1 on final step
                                                          if success else 0
        extras (under same demo_0 group; not in training schema):
            raw_actions                 (T, 8)           float32 — pre-clip policy output
            executed                    (T,)             bool    — env.step actually called
            inference/step_indices      (N_chunks,)      int32   — t when each inference fired
            inference/predicted_chunks  (N_chunks, 8, 8) float32 — full chunk returned by
                                                                   the policy for that call
            inference/latency_ms        (N_chunks,)      float32

    Note: each /infer_dict call observed obs at history[t-1] and history[t].
    The (1, 2, *) window is reconstructible at consumption time from
    obs/* at indices [inference_step_index - 1, inference_step_index].
    """

    def __init__(self, output_dir: Path, args, instruction: str, mode: str):
        self.dir = Path(output_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.args = vars(args)
        self.instruction = instruction
        self.mode = mode  # "no_robot" / "dry_run" / "live"
        self.control_hz = float(args.control_hz)

        # Per-step buffers
        self._t_list:           list[int]        = []
        self._hand_img_list:    list[np.ndarray] = []
        self._ext_img_list:     list[np.ndarray] = []
        self._joint_pos_list:   list[np.ndarray] = []
        self._gripper_pos_list: list[np.ndarray] = []
        self._cart_pos_list:    list[np.ndarray] = []
        self._action_list:      list[np.ndarray] = []
        self._raw_action_list:  list[np.ndarray] = []
        self._executed_list:    list[bool]       = []

        # Per-inference buffers
        self._inf_t_list:        list[int]        = []
        self._inf_chunk_list:    list[np.ndarray] = []
        self._inf_latency_list:  list[float]      = []

        # MP4 writers (opened lazily on first frame so we know native HW)
        self._wrist_writer:    Optional[cv2.VideoWriter] = None
        self._exterior_writer: Optional[cv2.VideoWriter] = None

        print(f"[rec] writing to {self.dir}")

    # ------------------------------------------------------------------

    def push_step(self, *, t: int, recording: dict, action: np.ndarray,
                  raw_action: np.ndarray, executed: bool) -> None:
        self._t_list.append(t)
        self._hand_img_list.append(recording["hand_camera_image"])
        self._ext_img_list.append(recording["exterior_image_1_left"])
        self._joint_pos_list.append(recording["joint_positions"])
        self._gripper_pos_list.append(recording["gripper_position"])
        self._cart_pos_list.append(recording["cartesian_position"])
        self._action_list.append(np.asarray(action, dtype=np.float32))
        self._raw_action_list.append(np.asarray(raw_action, dtype=np.float32))
        self._executed_list.append(bool(executed))

        # Lazy-open the MP4 writers
        wrist_nat = recording["wrist_native_rgb"]
        ext_nat   = recording["exterior_native_rgb"]
        if self._wrist_writer is None:
            self._open_writers(wrist_nat.shape[:2], ext_nat.shape[:2])
        # cv2 needs BGR for writing
        self._wrist_writer.write(wrist_nat[..., ::-1])
        self._exterior_writer.write(ext_nat[..., ::-1])

    def push_inference(self, *, t: int, chunk: np.ndarray, latency_ms: float) -> None:
        self._inf_t_list.append(t)
        self._inf_chunk_list.append(np.asarray(chunk, dtype=np.float32))
        self._inf_latency_list.append(float(latency_ms))

    # ------------------------------------------------------------------

    def _open_writers(self, wrist_hw, ext_hw) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # cv2.VideoWriter wants (width, height)
        self._wrist_writer = cv2.VideoWriter(
            str(self.dir / "wrist.mp4"), fourcc, self.control_hz,
            (int(wrist_hw[1]), int(wrist_hw[0])),
        )
        self._exterior_writer = cv2.VideoWriter(
            str(self.dir / "exterior.mp4"), fourcc, self.control_hz,
            (int(ext_hw[1]), int(ext_hw[0])),
        )

    def finalize(self, *, latencies_ms: list[float], success: Optional[float],
                 notes: str = "") -> None:
        if self._wrist_writer is not None:
            self._wrist_writer.release()
            self._exterior_writer.release()

        T = len(self._t_list)
        if T == 0:
            print("[rec] no steps recorded; skipping HDF5 + meta.")
            return

        # Per-step stacks
        hand_img    = np.stack(self._hand_img_list,    axis=0).astype(np.uint8)
        ext_img     = np.stack(self._ext_img_list,     axis=0).astype(np.uint8)
        joint_pos   = np.stack(self._joint_pos_list,   axis=0).astype(np.float32)
        gripper_pos = np.stack(self._gripper_pos_list, axis=0).astype(np.float32)
        cart_pos    = np.stack(self._cart_pos_list,    axis=0).astype(np.float32)
        actions     = np.stack(self._action_list,      axis=0).astype(np.float32)
        raw_actions = np.stack(self._raw_action_list,  axis=0).astype(np.float32)
        executed    = np.array(self._executed_list,    dtype=bool)
        ts          = np.array(self._t_list,           dtype=np.int32)

        # Matches scripts/convert_droid_to_robomimic.py: dones/rewards are 1 on final step.
        dones = np.zeros(T, dtype=np.float32); dones[-1] = 1.0
        rewards = np.zeros(T, dtype=np.float32)
        if success is not None:
            rewards[-1] = float(success)

        hdf5_path = self.dir / "trajectory.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            data = f.create_group("data")
            data.attrs["total"] = T
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = T
            demo.attrs["success"] = float(success) if success is not None else float("nan")
            demo.create_dataset("actions", data=actions, compression="gzip")
            demo.create_dataset("dones",   data=dones,   compression="gzip")
            demo.create_dataset("rewards", data=rewards, compression="gzip")
            demo.create_dataset("raw_actions", data=raw_actions, compression="gzip")
            demo.create_dataset("executed",    data=executed)
            demo.create_dataset("t",           data=ts)

            obs = demo.create_group("obs")
            obs.create_dataset("hand_camera_image",     data=hand_img,    compression="gzip")
            obs.create_dataset("exterior_image_1_left", data=ext_img,     compression="gzip")
            obs.create_dataset("joint_positions",       data=joint_pos,   compression="gzip")
            obs.create_dataset("gripper_position",      data=gripper_pos, compression="gzip")
            obs.create_dataset("cartesian_position",    data=cart_pos,    compression="gzip")

            if self._inf_t_list:
                inf = demo.create_group("inference")
                inf.create_dataset("step_indices",
                                   data=np.array(self._inf_t_list, dtype=np.int32))
                inf.create_dataset("predicted_chunks",
                                   data=np.stack(self._inf_chunk_list, axis=0).astype(np.float32),
                                   compression="gzip")
                inf.create_dataset("latency_ms",
                                   data=np.array(self._inf_latency_list, dtype=np.float32))

            # Mask group for compatibility with RobomimicReplayImageDataset
            mask = f.create_group("mask")
            mask.create_dataset("test", data=np.array(["demo_0"], dtype="S"))

        # Meta
        lat = latencies_ms or []
        meta = {
            "timestamp":   self.dir.name,
            "mode":        self.mode,
            "instruction": self.instruction,
            "n_steps":     T,
            "n_chunks":    len(self._inf_t_list),
            "success":     success,
            "notes":       notes,
            "latencies_ms": {
                "mean": float(np.mean(lat)) if lat else None,
                "p50":  float(np.percentile(lat, 50)) if lat else None,
                "p95":  float(np.percentile(lat, 95)) if lat else None,
            },
            "args": self.args,
            "policy_obs_format": {
                "image_keys":        list(POLICY_IMAGE_KEYS),
                "image_storage":     "uint8 HWC RGB at 256×256 (matches training HDF5)",
                "image_at_policy":   "(img.astype(float32) / 255.0).transpose(2,0,1) → CHW [0,1]",
                "state_keys":        list(POLICY_STATE_KEYS),
                "n_obs_steps":       N_OBS_STEPS,
                "action_dim":        ACTION_DIM,
                "history_reconstruction":
                    "window at inference step t = obs/*[t-1], obs/*[t] stacked along axis 0",
            },
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[rec] wrote {hdf5_path.name} ({hand_img.nbytes / 1e6:.0f} MB images, "
              f"T={T}, n_chunks={len(self._inf_t_list)})")


# ---------------------------------------------------------------------------
# Viser/ZMQ wire (matches droid-spacemouse/scripts/pi_eval_steering.py)
# ---------------------------------------------------------------------------

class ViserPublisher:
    """Non-blocking ZMQ PUSH to the viser dashboard. No-op when disabled.

    Wire format matches pi_eval_steering._ViserPublisher so the existing
    pi_eval_viser.py dashboard can be used unmodified.
    """

    def __init__(self, port: Optional[int]):
        self.enabled = port is not None
        if not self.enabled:
            return
        import msgpack as _msgpack
        import zmq as _zmq
        self._msgpack, self._zmq = _msgpack, _zmq
        self._ctx = _zmq.Context()
        self._sock = self._ctx.socket(_zmq.PUSH)
        self._sock.setsockopt(_zmq.SNDHWM, 10)
        self._sock.setsockopt(_zmq.LINGER, 0)
        self._sock.connect(f"tcp://127.0.0.1:{port}")
        print(f"[viser] publishing to tcp://127.0.0.1:{port}")

    def publish(self, *, joint_position, gripper_position, step, chunk_step,
                instruction=None, action_chunk=None, experiment_id=None):
        if not self.enabled:
            return
        msg = {
            "joint_position":   np.asarray(joint_position,   dtype=np.float32).tolist(),
            "gripper_position": np.asarray(gripper_position, dtype=np.float32).tolist(),
            "step":             int(step),
            "chunk_step":       int(chunk_step),
        }
        if instruction is not None:
            msg["instruction"] = instruction
        if action_chunk is not None:
            msg["action_chunk"] = np.asarray(action_chunk, dtype=np.float32).tolist()
        if experiment_id is not None:
            msg["experiment_id"] = experiment_id
        try:
            self._sock.send(self._msgpack.packb(msg), flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass   # dashboard backpressured; drop


class PauseController:
    """Reverse channel: dashboard pushes {"paused": bool} / {"reset": True}.

    Mirrors pi_eval_steering._PauseController.
    """

    def __init__(self, port: Optional[int]):
        self._paused = False
        self._reset_requested = False
        self.enabled = port is not None
        self._sock = None
        if not self.enabled:
            return
        try:
            import msgpack as _msgpack
            import zmq as _zmq
        except ImportError as e:
            print(f"[pause] zmq/msgpack unavailable: {e}; pause disabled")
            self.enabled = False
            return
        self._msgpack, self._zmq = _msgpack, _zmq
        ctx = _zmq.Context.instance()
        sock = ctx.socket(_zmq.PULL)
        sock.setsockopt(_zmq.RCVHWM, 10)
        sock.setsockopt(_zmq.LINGER, 0)
        try:
            sock.bind(f"tcp://127.0.0.1:{port}")
        except _zmq.ZMQError as e:
            print(f"[pause] bind to :{port} failed ({e}); pause disabled")
            sock.close(linger=0)
            self.enabled = False
            return
        self._sock = sock
        print(f"[pause] listening for dashboard on tcp://127.0.0.1:{port}")

    def poll(self) -> bool:
        """Drain pending messages; return current paused state."""
        if not self.enabled:
            return False
        try:
            while True:
                raw = self._sock.recv(flags=self._zmq.NOBLOCK)
                try:
                    cmd = self._msgpack.unpackb(raw, raw=False)
                except Exception as e:
                    print(f"[pause] bad msg: {e}")
                    continue
                if cmd.get("reset"):
                    if not self._reset_requested:
                        print("[pause] RESET requested — ending rollout")
                    self._reset_requested = True
                if "paused" in cmd:
                    new = bool(cmd["paused"])
                    if new != self._paused:
                        self._paused = new
                        print(f"[pause] {'PAUSED (env.step suppressed)' if new else 'RESUMED'}")
        except self._zmq.Again:
            pass
        return self._paused

    def consume_reset(self) -> bool:
        r = self._reset_requested
        self._reset_requested = False
        return r


@contextmanager
def prevent_keyboard_interrupt():
    """Defer SIGINT so an HTTP call is never killed mid-flight."""
    interrupted = False
    orig = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, orig)
        if interrupted:
            raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Keyboard listener — single-char non-blocking stdin
# ---------------------------------------------------------------------------

class KeyboardListener:
    """Non-blocking single-key reader using termios cbreak mode.

    Use as a context manager so the original terminal settings are restored
    on exit (including on exception). Inside the context, poll() returns one
    pending character if available, else None.

    cbreak (vs raw) leaves ISIG enabled, so Ctrl-C still raises SIGINT —
    handled at the call site, not swallowed here.

    Disabled when stdin isn't a TTY (pipes / nohup); poll() then returns None.
    """

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_attrs = None

    def __enter__(self) -> "KeyboardListener":
        if self.enabled:
            import termios
            import tty
            self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc) -> None:
        if self._old_attrs is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def poll(self) -> Optional[str]:
        if not self.enabled or self._old_attrs is None:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def prompt_success_label() -> Optional[float]:
    """Blocking success prompt. Call between rollouts (in cooked mode).

    Returns 0.0 / 1.0 / fractional [0, 1], or None to skip / on EOF.
    """
    try:
        while True:
            raw = input("[rec] rollout success? (y / n / 0-100 / s to skip): ").strip().lower()
            if raw in ("s", "skip", ""):
                return None
            if raw == "y":
                return 1.0
            if raw == "n":
                return 0.0
            try:
                v = float(raw)
                if 0 <= v <= 1:
                    return v
                if 0 <= v <= 100:
                    return v / 100.0
            except ValueError:
                pass
            print("  expected: y, n, a number in [0, 1] or [0, 100], or 's' to skip")
    except (EOFError, KeyboardInterrupt):
        print("\n[rec] success prompt skipped.")
        return None


def stop_arm(env, args) -> None:
    """Send a single zero-velocity command. Safe to call on dry/no-robot paths."""
    if env is None or args.dry_run:
        return
    try:
        env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    except Exception as e:
        print(f"[client] warning: zero-velocity stop failed: {e}")


def run_one_rollout(
    *,
    env,
    args,
    recorder: Optional["RolloutRecorder"],
    history: deque,
    pause_ctrl: "PauseController",
    viser: "ViserPublisher",
    ext_serial: str,
    executor: Optional[ThreadPoolExecutor],
    latencies_ms: list,
) -> str:
    """Run one rollout. Mutates history + recorder + latencies_ms in place.

    Returns one of:
        "max_timesteps"  — hit args.max_timesteps
        "user_reset"     — keyboard 'r' (caller should home + prompt for next)
        "user_quit"      — keyboard 'q' (caller should label and exit session)
        "viser_reset"    — viser dashboard reset button
    """
    pred_chunk: Optional[np.ndarray] = None
    chunk_idx = 0
    pending_future: Optional[Future] = None
    pending_infer_start: float = 0.0
    step_dt = 1.0 / args.control_hz
    paused = False
    end_reason = "max_timesteps"

    with KeyboardListener() as kb:
        if kb.enabled:
            print("[kb] [Space]=pause/resume  [r]=end rollout (then home)  [q]=end + quit session")
        else:
            print("[kb] stdin not a TTY; interactive keys disabled "
                  "(use --pause-port for viser reset/pause).")

        for t in range(args.max_timesteps):
            loop_start = time.time()

            # 0a. drain keyboard
            quit_inner = False
            while True:
                key = kb.poll()
                if key is None:
                    break
                if key in (" ", "p"):
                    paused = not paused
                    if paused:
                        print("[kb] PAUSED — env.step suppressed; reset scene then press Space.")
                        stop_arm(env, args)
                    else:
                        print("[kb] RESUMED — clearing history; next step will re-inference.")
                        history.clear()
                        pred_chunk = None
                        chunk_idx = 0
                        pending_future = None  # orphan: executor drains it; we ignore the result
                elif key == "r":
                    print("[kb] RESET — ending rollout.")
                    end_reason = "user_reset"
                    quit_inner = True
                    break
                elif key == "q":
                    print("[kb] QUIT — ending rollout and session.")
                    end_reason = "user_quit"
                    quit_inner = True
                    break
            if quit_inner:
                break

            # 0b. drain viser pause/reset
            viser_paused = pause_ctrl.poll()
            if pause_ctrl.consume_reset():
                print("[viser] reset — ending rollout.")
                end_reason = "viser_reset"
                break
            effective_paused = paused or viser_paused

            # 1. observe
            if env is None:
                step_obs, recording = fake_obs()
                joint_pos_for_viser = np.zeros(7, dtype=np.float32)
                gripper_for_viser = np.zeros(1, dtype=np.float32)
            else:
                env_obs = env.get_observation()
                step_obs, recording = extract_droid_obs(
                    env_obs, wrist_serial=args.wrist_serial, ext_serial=ext_serial)
                joint_pos_for_viser = step_obs["joint_positions"]
                gripper_for_viser = step_obs["gripper_position"]
            history.append(step_obs)
            while len(history) < N_OBS_STEPS:
                history.append(step_obs)

            # 2a. async prefetch: fire once after step 0 of the current chunk has run
            if (args.inference_mode == "async_chunk" and executor is not None
                    and pred_chunk is not None
                    and chunk_idx == 1
                    and pending_future is None
                    and not effective_paused):
                pending_infer_start = time.time()
                # deque(history) is a fresh copy — background thread reads it without
                # contention while the foreground thread keeps rotating `history`.
                pending_future = executor.submit(policy_infer, args.server_url, deque(history))

            # 2b. refresh chunk at boundary
            new_chunk_this_step = False
            need_new_chunk = pred_chunk is None or chunk_idx >= args.open_loop_horizon
            if need_new_chunk:
                if args.inference_mode == "async_chunk" and pending_future is not None:
                    with prevent_keyboard_interrupt():
                        pred_chunk = pending_future.result(timeout=60)
                    latency_ms = (time.time() - pending_infer_start) * 1000
                    pending_future = None
                else:
                    infer_start = time.time()
                    with prevent_keyboard_interrupt():
                        pred_chunk = policy_infer(args.server_url, history)
                    latency_ms = (time.time() - infer_start) * 1000
                latencies_ms.append(latency_ms)
                assert pred_chunk.shape[-1] == ACTION_DIM, (
                    f"action_dim mismatch: server returned {pred_chunk.shape}, "
                    f"expected (..., {ACTION_DIM})"
                )
                chunk_idx = 0
                new_chunk_this_step = True
                if recorder is not None:
                    recorder.push_inference(t=t, chunk=pred_chunk, latency_ms=latency_ms)
                print(f"[client] t={t:4d}  inference {latency_ms:6.1f} ms  "
                      f"chunk shape={pred_chunk.shape}"
                      + ("  (paused)" if effective_paused else ""))

            # 3. select + safety-clip
            raw_action = pred_chunk[chunk_idx]
            action = clip_action_safe(raw_action, max_joint_vel=args.max_joint_vel)

            # 4. publish to viser (BEFORE we maybe step the robot)
            viser.publish(
                joint_position   = joint_pos_for_viser,
                gripper_position = gripper_for_viser,
                step             = t,
                chunk_step       = chunk_idx,
                instruction      = args.instruction or None,
                action_chunk     = pred_chunk if new_chunk_this_step else None,
                experiment_id    = args.experiment_id or None,
            )

            # 5. (maybe) execute on the robot
            will_execute = (env is not None) and (not args.dry_run) and (not effective_paused)
            if will_execute:
                env.step(action)
            elif env is None and (t < 4 or t % 50 == 0):
                print(f"[client] NO-ROBOT t={t:4d}  clipped={np.round(action, 3).tolist()}")

            # 6. record
            if recorder is not None:
                recorder.push_step(
                    t=t, recording=recording,
                    action=action, raw_action=raw_action,
                    executed=will_execute,
                )

            chunk_idx += 1

            # 7. pace to control_hz
            elapsed = time.time() - loop_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    return end_reason


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Server / loop params
    parser.add_argument("--server-url",   default="http://127.0.0.1:5001")
    parser.add_argument("--max-timesteps", type=int, default=600)
    parser.add_argument("--open-loop-horizon", type=int, default=OPEN_LOOP_HORIZON)
    parser.add_argument("--control-hz", type=float, default=DROID_CONTROL_HZ)
    parser.add_argument("--instruction", default="", help="Stamped into viser msg; not used by the diffusion policy.")
    # Inference
    parser.add_argument("--inference-mode", choices=["sync", "async_chunk"], default="async_chunk",
                        help="sync: block on every chunk boundary. async_chunk (default): "
                             "after the first step of each chunk has run, submit inference for the "
                             "next chunk in a background thread; await at the boundary (should "
                             "already be ready). Mirrors policy_doctor/envs/droid_runner.py.")
    # Cameras
    parser.add_argument("--wrist-serial", default=WRIST_SERIAL)
    parser.add_argument("--external-camera", choices=["left", "right"], default="left",
                        help="Which exterior ZED to feed the policy. Maps to ext1/ext2 serial.")
    parser.add_argument("--ext1-serial", default=EXT1_SERIAL, help="ZED serial used when --external-camera=left")
    parser.add_argument("--ext2-serial", default=EXT2_SERIAL, help="ZED serial used when --external-camera=right")
    # Safety
    parser.add_argument("--max-joint-vel", type=float, default=0.5,
                        help="Hard cap on |joint velocity| (rad/s). 0.5 is conservative for first runs.")
    # Run modes (mutually exclusive: --no-robot, --dry-run, default=live)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-robot", action="store_true",
                       help="Skip RobotEnv entirely; fake obs + no env.step. Validates the wire format only.")
    modes.add_argument("--dry-run", action="store_true",
                       help="Real RobotEnv (homes the arm on init), real obs, skip env.step. Pair with --viser-port.")
    # Viser
    parser.add_argument("--viser-port", type=int, default=None,
                        help="If set, publish per-step state + action chunks via ZMQ PUSH to the viser dashboard.")
    parser.add_argument("--pause-port", type=int, default=None,
                        help="If set, listen for Pause/Reset commands from the viser dashboard.")
    parser.add_argument("--experiment-id", default="", help="Stamped into viser msg.")
    # Recording
    parser.add_argument("--output-dir", default="data/droid_eval_runs",
                        help="Root dir for per-rollout subfolders (HDF5 + MP4 + meta.json).")
    parser.add_argument("--no-recording", action="store_true",
                        help="Disable rollout recording entirely.")
    parser.add_argument("--no-success-prompt", action="store_true",
                        help="Skip the interactive success prompt at end of live rollouts.")
    args = parser.parse_args()

    ext_serial = args.ext1_serial if args.external_camera == "left" else args.ext2_serial
    print(f"[client] external camera: {args.external_camera} ({ext_serial})")

    # Up-front server reachability check
    try:
        h = requests.get(f"{args.server_url.rstrip('/')}/health", timeout=5).json()
        print(f"[client] policy server: {h}")
    except Exception as e:
        print(f"[client] FATAL: cannot reach policy server at {args.server_url}: {e}")
        sys.exit(1)

    # Mode
    if args.no_robot:
        print("[client] NO-ROBOT mode: fake observations, no env.step.")
        env = None
    else:
        print("[client] initializing RobotEnv (joint_velocity, gripper position) — ARM WILL HOME.")
        from droid.robot_env import RobotEnv
        env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
        print("[client] RobotEnv ready.")
        if args.dry_run:
            print("[client] DRY-RUN mode: real obs, env.step() suppressed.")
        else:
            print("[client] LIVE mode: env.step() will execute on the robot. KEEP A HAND ON E-STOP.")

    viser = ViserPublisher(args.viser_port)
    pause = PauseController(args.pause_port)

    mode = "no_robot" if args.no_robot else ("dry_run" if args.dry_run else "live")

    # Single executor shared across all rollouts in this session. max_workers=2
    # so that orphaned (post-pause) inferences don't block a fresh submit; the
    # Flask server still serializes on its end, but the foreground never waits
    # for a stale request to complete.
    executor: Optional[ThreadPoolExecutor] = None
    if args.inference_mode == "async_chunk":
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="droid_infer")
        print("[client] inference mode: async_chunk (overlap inference w/ chunk execution)")
    else:
        print("[client] inference mode: sync (blocking at every chunk boundary)")

    print(f"[client] rollout params: max_timesteps={args.max_timesteps}, "
          f"open_loop_horizon={args.open_loop_horizon}, control_hz={args.control_hz}, "
          f"max_joint_vel={args.max_joint_vel}")
    print("[client] Ctrl+C ends the session.")

    rollout_idx = 0
    is_first = True

    try:
        while True:
            # Wait-for-go gate before each rollout (including the first).
            try:
                prompt = ("\n[ready] press Enter to start the "
                          + ("first" if is_first else "next")
                          + " rollout, q+Enter to quit: ")
                raw = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw in ("q", "quit"):
                break

            rollout_idx += 1
            is_first = False
            print(f"\n=== rollout {rollout_idx} ===")

            recorder: Optional[RolloutRecorder] = None
            if not args.no_recording:
                recorder = RolloutRecorder(
                    output_dir=Path(args.output_dir),
                    args=args,
                    instruction=args.instruction,
                    mode=mode,
                )

            history: deque = deque(maxlen=N_OBS_STEPS)
            latencies_ms: list = []
            end_reason = "max_timesteps"

            try:
                end_reason = run_one_rollout(
                    env=env, args=args, recorder=recorder, history=history,
                    pause_ctrl=pause, viser=viser, ext_serial=ext_serial,
                    executor=executor, latencies_ms=latencies_ms,
                )
            except KeyboardInterrupt:
                print("\n[client] interrupted by user.")
                end_reason = "user_quit"

            # Always stop the arm at end of rollout (no-op on dry/no-robot paths).
            stop_arm(env, args)

            # Latency summary for this rollout
            if latencies_ms:
                print(f"[client] inference latency: "
                      f"mean={np.mean(latencies_ms):.1f}ms  "
                      f"p50={np.percentile(latencies_ms, 50):.1f}ms  "
                      f"p95={np.percentile(latencies_ms, 95):.1f}ms  "
                      f"n_chunks={len(latencies_ms)}")

            # Label (live mode + recorder + prompt enabled)
            success: Optional[float] = None
            if recorder is not None and mode == "live" and not args.no_success_prompt:
                success = prompt_success_label()

            if recorder is not None:
                recorder.finalize(latencies_ms=latencies_ms, success=success)

            if end_reason == "user_quit":
                break

            # Home the arm before the next start prompt so the user isn't
            # racing the policy to set up the scene.
            if env is not None and not args.dry_run:
                print("[client] homing arm (env.reset())...")
                try:
                    env.reset()
                    print("[client] homed.")
                except Exception as e:
                    print(f"[client] WARNING: env.reset() failed: {e}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


if __name__ == "__main__":
    main()

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
"""

from __future__ import annotations

import argparse
import io
import signal
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Optional

import cv2
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

def preprocess_image(img_hwc_uint8: np.ndarray, target_hw=POLICY_IMAGE_HW) -> np.ndarray:
    """HWC uint8 RGB → CHW float32 in [0, 1], resized.

    Uses INTER_AREA to match scripts/convert_droid_to_robomimic.py:_resize_image
    (the same resampler the training HDF5 was built with). INTER_AREA is the
    correct choice for downsampling 1280x720 ZED native → 256x256.
    """
    h, w = target_hw
    if img_hwc_uint8.shape[:2] != (h, w):
        img_hwc_uint8 = cv2.resize(img_hwc_uint8, (w, h), interpolation=cv2.INTER_AREA)
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))   # (3, H, W)


def extract_droid_obs(env_obs: dict, *, wrist_serial: str, ext_serial: str) -> dict:
    """RobotEnv.get_observation() → policy obs (single timestep, preprocessed).

    Maps to the keys may13_droid.ckpt expects (see cfg.shape_meta.obs):
        wrist camera  → hand_camera_image
        ext camera    → exterior_image_1_left
        joint_positions, gripper_position pass through (DROID names already match).
    Drops alpha, BGR→RGB.
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
    wrist_img = wrist_img[..., :3][..., ::-1]   # drop alpha, BGR→RGB
    ext_img   = ext_img[..., :3][..., ::-1]

    state = env_obs["robot_state"]
    return {
        "hand_camera_image":     preprocess_image(wrist_img),
        "exterior_image_1_left": preprocess_image(ext_img),
        "joint_positions":       np.asarray(state["joint_positions"], dtype=np.float32),
        "gripper_position":      np.asarray([state["gripper_position"]], dtype=np.float32),
    }


def fake_obs() -> dict:
    """No-robot path: zeros for state, black images."""
    return {
        "hand_camera_image":     np.zeros((3, *POLICY_IMAGE_HW), dtype=np.float32),
        "exterior_image_1_left": np.zeros((3, *POLICY_IMAGE_HW), dtype=np.float32),
        "joint_positions":       np.zeros(7, dtype=np.float32),
        "gripper_position":      np.zeros(1, dtype=np.float32),
    }


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

    history: deque = deque(maxlen=N_OBS_STEPS)
    pred_chunk: Optional[np.ndarray] = None
    chunk_idx = 0
    step_dt = 1.0 / args.control_hz
    latencies_ms: list[float] = []

    print(f"[client] rollout params: max_timesteps={args.max_timesteps}, "
          f"open_loop_horizon={args.open_loop_horizon}, control_hz={args.control_hz}, "
          f"max_joint_vel={args.max_joint_vel}")
    print("[client] Ctrl+C to stop early.")

    try:
        for t in range(args.max_timesteps):
            loop_start = time.time()

            # Drain pause/reset commands from dashboard
            paused = pause.poll()
            if pause.consume_reset():
                print("[client] reset requested — ending rollout.")
                break

            # 1. observe
            if env is None:
                step_obs = fake_obs()
                joint_pos_for_viser = np.zeros(7, dtype=np.float32)
                gripper_for_viser = np.zeros(1, dtype=np.float32)
            else:
                env_obs = env.get_observation()
                step_obs = extract_droid_obs(env_obs, wrist_serial=args.wrist_serial, ext_serial=ext_serial)
                joint_pos_for_viser = step_obs["joint_positions"]
                gripper_for_viser = step_obs["gripper_position"]
            history.append(step_obs)
            while len(history) < N_OBS_STEPS:
                history.append(step_obs)

            # 2. refresh action chunk at boundaries
            new_chunk_this_step = False
            if pred_chunk is None or chunk_idx >= args.open_loop_horizon:
                infer_start = time.time()
                with prevent_keyboard_interrupt():
                    pred_chunk = policy_infer(args.server_url, history)
                latency_ms = (time.time() - infer_start) * 1000
                latencies_ms.append(latency_ms)
                assert pred_chunk.shape[-1] == ACTION_DIM, (
                    f"action_dim mismatch: server returned {pred_chunk.shape}, expected (..., {ACTION_DIM})"
                )
                chunk_idx = 0
                new_chunk_this_step = True
                print(f"[client] t={t:4d}  inference {latency_ms:6.1f} ms  chunk shape={pred_chunk.shape}"
                      + ("  (paused)" if paused else ""))

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
            if env is not None and not args.dry_run and not paused:
                env.step(action)
            elif env is None and (t < 4 or t % 50 == 0):
                # Diagnostic print on the fully-fake path
                print(f"[client] NO-ROBOT t={t:4d}  clipped={np.round(action, 3).tolist()}")

            chunk_idx += 1

            # 6. pace to control_hz
            elapsed = time.time() - loop_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
    except KeyboardInterrupt:
        print("\n[client] interrupted by user.")
    finally:
        if env is not None and not args.dry_run:
            # In live mode only: stop the arm by commanding zero velocity once.
            # In dry-run we never stepped, so this is unnecessary (and would
            # be the first env.step the user didn't ask for).
            try:
                env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            except Exception as e:
                print(f"[client] warning: zero-velocity stop failed: {e}")
        if latencies_ms:
            print(f"[client] inference latency: "
                  f"mean={np.mean(latencies_ms):.1f}ms  "
                  f"p50={np.percentile(latencies_ms, 50):.1f}ms  "
                  f"p95={np.percentile(latencies_ms, 95):.1f}ms  "
                  f"n_chunks={len(latencies_ms)}")


if __name__ == "__main__":
    main()

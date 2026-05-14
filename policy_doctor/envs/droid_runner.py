"""DROID real-robot inference runner.

Three inference modes
---------------------
SYNC
    Inference blocks the control loop.  At every open_loop_horizon boundary,
    submit + get() before executing the next chunk.  Matches the existing
    pi_eval.py behaviour.

ASYNC_CHUNK
    At the start of each chunk, submit inference for the *next* chunk
    immediately.  Execute the current chunk at control_hz while inference runs
    in the background.  Block for the new chunk only at the chunk boundary
    (should already be done by then).  Mirrors dagger_runner.py's _async path.

ASYNC_STREAMING
    A dedicated background thread calls submit → get in a tight loop, always
    using the latest observation.  The foreground control loop reads from a
    lock-protected buffer at control_hz.  If the buffer is empty (policy is
    slower than the control loop), applies the StaleActionPolicy.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from policy_doctor.envs.droid_env import DROIDInferenceEnv
from policy_doctor.envs.droid_policy_wrappers import PolicyBase


class InferenceMode(enum.Enum):
    SYNC = "sync"
    ASYNC_CHUNK = "async_chunk"
    ASYNC_STREAMING = "async_streaming"


class StaleActionPolicy(enum.Enum):
    HOLD_LAST = "hold_last"
    ZERO_ON_STALE = "zero_on_stale"


@dataclass
class EpisodeRecord:
    episode_idx: int
    instruction: str
    n_steps: int
    n_new_chunks: int
    n_stale_steps: int
    success: bool
    avg_chunk_latency_ms: float
    chunk_latencies_ms: list[float] = field(default_factory=list)


class DROIDInferenceRunner:
    """Run one or many rollouts with configurable inference mode.

    Parameters
    ----------
    env : DROIDInferenceEnv
    policy : PolicyBase
    mode : InferenceMode
    stale_action_policy : StaleActionPolicy
        Only used in ASYNC_STREAMING mode.
    open_loop_horizon : int
        Number of actions to execute from each chunk before requesting a new one.
        Used in SYNC and ASYNC_CHUNK modes.
    max_timesteps : int
    control_hz : float
        Target control frequency.  The runner sleeps after each env.step() to
        keep the loop at this rate.
    output_dir : Path, optional
        If set, saves each episode via env.save_episode().
    """

    def __init__(
        self,
        env: DROIDInferenceEnv,
        policy: PolicyBase,
        mode: InferenceMode = InferenceMode.ASYNC_CHUNK,
        stale_action_policy: StaleActionPolicy = StaleActionPolicy.HOLD_LAST,
        open_loop_horizon: int = 8,
        max_timesteps: int = 600,
        control_hz: float = 15.0,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.mode = mode
        self.stale_action_policy = stale_action_policy
        self.open_loop_horizon = open_loop_horizon
        self.max_timesteps = max_timesteps
        self.control_hz = control_hz
        self.output_dir = Path(output_dir) if output_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(self, instruction: str = "", episode_idx: int = 0) -> EpisodeRecord:
        """Run a single rollout and return a summary record."""
        obs = self.env.reset()
        self.policy.reset()

        if self.mode == InferenceMode.SYNC:
            return self._run_sync(obs, instruction, episode_idx)
        elif self.mode == InferenceMode.ASYNC_CHUNK:
            return self._run_async_chunk(obs, instruction, episode_idx)
        else:
            return self._run_async_streaming(obs, instruction, episode_idx)

    def run(self, n_episodes: int, instructions: list[str] | str = "") -> list[EpisodeRecord]:
        """Run multiple episodes."""
        if isinstance(instructions, str):
            instructions = [instructions] * n_episodes
        records = []
        for i in range(n_episodes):
            print(f"\n=== Episode {i+1}/{n_episodes} | instruction={instructions[i]!r} ===")
            rec = self.run_episode(instructions[i], episode_idx=i)
            print(
                f"  steps={rec.n_steps} chunks={rec.n_new_chunks} "
                f"stale={rec.n_stale_steps} "
                f"avg_chunk={rec.avg_chunk_latency_ms:.0f}ms"
            )
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # SYNC mode
    # ------------------------------------------------------------------

    def _run_sync(self, obs: dict, instruction: str, episode_idx: int) -> EpisodeRecord:
        dt = 1.0 / self.control_hz
        chunk: Optional[np.ndarray] = None
        chunk_idx = 0
        n_chunks = 0
        latencies: list[float] = []

        for step in range(self.max_timesteps):
            t0 = time.time()

            if chunk is None or chunk_idx >= self.open_loop_horizon or chunk_idx >= len(chunk):
                t_inf = time.time()
                self.policy.submit(obs, instruction)
                chunk = self.policy.get()
                latencies.append((time.time() - t_inf) * 1000)
                chunk_idx = 0
                n_chunks += 1

            action = chunk[chunk_idx]
            chunk_idx += 1
            obs, _, _, _ = self.env.step(action)

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        if self.output_dir:
            self.env.save_episode(
                self.output_dir / f"ep{episode_idx:04d}" / "trajectory.h5"
            )

        return self._make_record(episode_idx, instruction, self.max_timesteps, n_chunks, 0, latencies)

    # ------------------------------------------------------------------
    # ASYNC_CHUNK mode
    # ------------------------------------------------------------------

    def _run_async_chunk(self, obs: dict, instruction: str, episode_idx: int) -> EpisodeRecord:
        dt = 1.0 / self.control_hz
        n_chunks = 0
        latencies: list[float] = []

        # Submit the first chunk immediately.
        t_submit = time.time()
        self.policy.submit(obs, instruction)

        total_steps = 0
        while total_steps < self.max_timesteps:
            # Block for the pending chunk (should already be done when we arrive).
            t_wait = time.time()
            chunk = self.policy.get()
            latencies.append((time.time() - t_submit) * 1000)
            n_chunks += 1

            horizon = min(self.open_loop_horizon, len(chunk), self.max_timesteps - total_steps)

            for i in range(horizon):
                t0 = time.time()
                obs, _, _, _ = self.env.step(chunk[i])
                total_steps += 1

                # Submit next chunk immediately after the first step of this chunk
                # so inference overlaps with execution of steps 1..horizon-1.
                if i == 0:
                    t_submit = time.time()
                    self.policy.submit(obs, instruction)

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        if self.output_dir:
            self.env.save_episode(
                self.output_dir / f"ep{episode_idx:04d}" / "trajectory.h5"
            )

        return self._make_record(episode_idx, instruction, total_steps, n_chunks, 0, latencies)

    # ------------------------------------------------------------------
    # ASYNC_STREAMING mode
    # ------------------------------------------------------------------

    def _run_async_streaming(self, obs: dict, instruction: str, episode_idx: int) -> EpisodeRecord:
        dt = 1.0 / self.control_hz

        # Shared state between foreground and inference thread.
        _lock = threading.Lock()
        _obs_ref: list[dict] = [obs]           # latest obs (list for mutability)
        _instr_ref: list[str] = [instruction]
        _buffer: list[Optional[np.ndarray]] = [None]   # latest chunk
        _chunk_step: list[int] = [0]
        _last_action: list[Optional[np.ndarray]] = [None]
        _stop = threading.Event()
        latencies: list[float] = []
        n_chunks_ref: list[int] = [0]

        def _inference_thread():
            while not _stop.is_set():
                with _lock:
                    curr_obs = _obs_ref[0]
                    curr_instr = _instr_ref[0]
                t0 = time.time()
                try:
                    self.policy.submit(curr_obs, curr_instr)
                    new_chunk = self.policy.get()
                except Exception as e:
                    print(f"[droid_runner] inference error: {e}")
                    continue
                lat_ms = (time.time() - t0) * 1000
                with _lock:
                    _buffer[0] = new_chunk
                    _chunk_step[0] = 0  # discard unexecuted tail of previous chunk: latest beats stale
                    n_chunks_ref[0] += 1
                    latencies.append(lat_ms)

        thread = threading.Thread(target=_inference_thread, daemon=True, name="droid_infer")
        thread.start()

        n_stale = 0
        total_steps = 0
        action_dim = self.env._action_dim

        try:
            for step in range(self.max_timesteps):
                t0 = time.time()

                with _lock:
                    chunk = _buffer[0]
                    idx = _chunk_step[0]

                if chunk is not None and idx < len(chunk):
                    action = chunk[idx]
                    with _lock:
                        _chunk_step[0] += 1
                    _last_action[0] = action
                else:
                    # Buffer exhausted or not yet ready.
                    if self.stale_action_policy == StaleActionPolicy.HOLD_LAST and _last_action[0] is not None:
                        action = _last_action[0]
                        n_stale += 1
                    else:
                        action = np.zeros(action_dim, dtype=np.float32)
                        n_stale += 1

                new_obs, _, _, _ = self.env.step(action)
                total_steps += 1

                with _lock:
                    _obs_ref[0] = new_obs

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        finally:
            _stop.set()
            thread.join(timeout=5.0)

        if self.output_dir:
            self.env.save_episode(
                self.output_dir / f"ep{episode_idx:04d}" / "trajectory.h5"
            )

        return self._make_record(
            episode_idx, instruction, total_steps, n_chunks_ref[0], n_stale, latencies
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_record(
        self,
        episode_idx: int,
        instruction: str,
        n_steps: int,
        n_chunks: int,
        n_stale: int,
        latencies: list[float],
    ) -> EpisodeRecord:
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        return EpisodeRecord(
            episode_idx=episode_idx,
            instruction=instruction,
            n_steps=n_steps,
            n_new_chunks=n_chunks,
            n_stale_steps=n_stale,
            success=False,  # real-robot success is annotated externally
            avg_chunk_latency_ms=avg_lat,
            chunk_latencies_ms=list(latencies),
        )

"""Mid-rollout sim state extraction.

For ``recovery`` and ``alternative_strategy`` requests we need to start the sim
from a *mid-trajectory* state of a reference rollout, not just frame 0. The
DAgger / eval_save_episodes pkls already store ``sim_state`` per timestep — this
module is a thin indexer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_episode_df(episode_pkl: Path) -> pd.DataFrame:
    """Load the per-step DataFrame written by RobomimicDAggerEnv.save_episode /
    eval_save_episodes."""
    return pd.read_pickle(str(episode_pkl))


def extract_sim_state_at_frame(
    episode_pkl: Path,
    frame_idx: int,
) -> np.ndarray:
    """Return the MuJoCo ``sim_state`` vector at ``frame_idx`` of *episode_pkl*.

    Frame 0 = state immediately after the first env step (matches how rollouts
    were recorded). For exact rollout-start state, use the env's reset; ``frame_idx=0``
    here is the *post-first-step* state, which is fine for "start from rollout
    beginning" semantics in practice (the very first frame is identical to reset
    plus a tiny step).

    Raises
    ------
    IndexError
        If ``frame_idx`` is out of range for this episode.
    """
    df = load_episode_df(Path(episode_pkl))
    n = len(df)
    if not 0 <= frame_idx < n:
        raise IndexError(
            f"frame_idx={frame_idx} out of range for episode of length {n}"
        )
    state = df.iloc[frame_idx]["sim_state"]
    return np.asarray(state, dtype=np.float64).copy()


def extract_object_pose_at_frame(
    episode_pkl: Path,
    frame_idx: int,
    obs_key: str = "object",
) -> np.ndarray:
    """Return the recorded ``obs[obs_key]`` at ``frame_idx`` of *episode_pkl*.

    Used by :meth:`RolloutPool` to populate ``InitialConditions.object_poses``
    without needing to re-instantiate the env.
    """
    df = load_episode_df(Path(episode_pkl))
    if frame_idx < 0 or frame_idx >= len(df):
        raise IndexError(f"frame_idx={frame_idx} out of range")
    obs = df.iloc[frame_idx]["obs"]
    if isinstance(obs, dict) and obs_key in obs:
        return np.asarray(obs[obs_key], dtype=np.float32).copy()
    raise KeyError(f"obs_key={obs_key!r} not in episode obs dict")


def verify_sim_state_replays(
    episode_pkl: Path,
    sim_state: np.ndarray,
    expected_frame_idx: int,
    *,
    atol: float = 1e-6,
) -> bool:
    """Sanity check: returns True iff ``sim_state`` matches the recorded state at
    ``expected_frame_idx``. Used by Tier-1 smoke to verify init_state replay.
    """
    recorded = extract_sim_state_at_frame(episode_pkl, expected_frame_idx)
    if recorded.shape != sim_state.shape:
        return False
    return bool(np.allclose(recorded, sim_state, atol=atol))

"""Replay mar27 low-diffusion eval pickles into robomimic flat states for MimicGen-style HDF5.

**Run in the cupid conda stack**, not ``mimicgen``: this path imports ``diffusion_policy``
from the Cupid repo, robomimic 0.2-style envs, and MuJoCo 3.x. Output HDF5 is suitable
for validation and for ``materialize_*`` workflows; feeding it into MimicGen
``prepare_src_dataset`` requires a matching ``MG_*`` env interface (not in upstream MimicGen
for TwoArmTransport) and a consistent sim stack.

Eval episodes store policy ``action`` chunks (horizon × action_dim), not mujoco states.
This module rebuilds ``states`` / ``actions`` by stepping a robomimic transport env with the
same action transform as ``RobomimicLowdimRunner`` (absolute rotation_6d → env deltas).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Defaults for mar27 transport_mh seed-0 baseline (see ``transport_mh_mar27.yaml``).
MAR27_EVAL_REL = Path(
    "data/outputs/eval_save_episodes/mar27/mar27_train_diffusion_unet_lowdim_transport_mh_0/latest"
)
TRANSPORT_LOW_DIM_DATASET_REL = Path("data/robomimic/datasets/transport/mh/low_dim_abs.hdf5")
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
HORIZON = 16


def ensure_cupid_repo_on_path(cupid_root: Path) -> Path:
    root = cupid_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cupid repo not found: {root}")
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def default_cupid_root() -> Path | None:
    env = __import__("os").environ.get("CUPID_REPO_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    from policy_doctor.paths import PROJECT_ROOT

    candidate = (PROJECT_ROOT.parent / "cupid").resolve()
    return candidate if candidate.is_dir() else None


def mar27_eval_dir(cupid_root: Path) -> Path:
    return (cupid_root / MAR27_EVAL_REL).resolve()


def first_successful_episode_pkl(eval_dir: Path) -> Path:
    eps = sorted(eval_dir.glob("episodes/ep*_succ.pkl"))
    if not eps:
        raise FileNotFoundError(f"No ep*_succ.pkl under {eval_dir / 'episodes'}")
    return eps[0]


def _undo_transform_action(
    action: np.ndarray,
    rotation_transformer: Any,
) -> np.ndarray:
    """Match ``RobomimicLowdimRunner.undo_transform_action`` (dual-arm 20 → 14)."""
    raw_shape = action.shape
    x = action
    if raw_shape[-1] == 20:
        x = action.reshape(-1, 2, 10)
    d_rot = x.shape[-1] - 4
    pos = x[..., :3]
    rot = x[..., 3 : 3 + d_rot]
    gripper = x[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


def replay_transport_eval_pickle(
    cupid_root: Path,
    episode_pkl: Path,
    *,
    dataset_path: Path | None = None,
    n_obs_steps: int = N_OBS_STEPS,
    n_action_steps: int = N_ACTION_STEPS,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any], str]:
    """Return (states, actions, env_meta, model_file_xml) for one eval episode.

    ``states`` and ``actions`` have the same length (per MimicGen ``prepare_src_dataset``).
    """
    ensure_cupid_repo_on_path(cupid_root)
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    ds = dataset_path or (cupid_root / TRANSPORT_LOW_DIM_DATASET_REL).resolve()
    if not ds.is_file():
        raise FileNotFoundError(f"Transport dataset not found: {ds}")

    obs_keys = [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
    ]
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env_meta = json.loads(json.dumps(FileUtils.get_env_metadata_from_dataset(dataset_path=str(ds))))
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    rt = RotationTransformer("axis_angle", "rotation_6d")
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    with open(episode_pkl, "rb") as f:
        df = pickle.load(f)

    env.reset()
    model_xml = env.env.sim.model.get_xml()

    states_before: list[np.ndarray] = []
    actions_flat: list[np.ndarray] = []

    for _, row in df.iterrows():
        ap = np.asarray(row["action"], dtype=np.float32)
        if ap.shape != (HORIZON, 20):
            raise ValueError(f"Expected action ({HORIZON}, 20), got {ap.shape}")
        chunk = ap[n_obs_steps : n_obs_steps + n_action_steps]
        env_action = _undo_transform_action(chunk, rt)
        for act in env_action:
            states_before.append(np.array(env.get_state()["states"], dtype=np.float64))
            actions_flat.append(np.array(act, dtype=np.float32))
            env.step(act)

    states = np.stack(states_before, axis=0)
    actions = np.stack(actions_flat, axis=0)
    env_meta_out = env.serialize()
    return states, actions, env_meta_out, model_xml

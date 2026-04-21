"""Train ENAP visual encoder + HDBSCAN clustering — pipeline step.

Implements **Stage 1** of the ENAP E-step:

    obs (images + proprioception) → VisualEncoder → z_t
    z_t → HDBSCANClusterer → c_t  (discrete observation symbols)

Data source: the same post-rollout eval episode pickle files used by the
infembed/TRAK pipeline (``eval_dir/episodes/ep*.pkl``).

Saved outputs (in ``step_dir/``):
- ``feature_embeddings.npy``  — ``(N, feature_dim)`` visual features z_t
- ``symbol_assignments.npy``  — ``(N,)`` int8 cluster labels c_t
- ``actions.npy``             — ``(N, action_dim)`` continuous actions
- ``metadata.json``           — per-timestep dicts with ``rollout_idx``,
  ``timestep``, ``success``
- ``encoder_state.pt``        — saved VisualEncoder state dict
- ``result.json`` / ``done``  — standard PipelineStep outputs
"""

from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds
from policy_doctor.paths import PACKAGE_ROOT, iv_task_configs_base


# ---------------------------------------------------------------------------
# Episode loading helpers
# ---------------------------------------------------------------------------

def _load_episode_pickles(
    eval_dir: pathlib.Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[bool]], List[int]]:
    """Load all episode pickles from ``eval_dir/episodes/``.

    Returns:
        Tuple of (obs_list, actions_list, successes, episode_lengths):
        - ``obs_list``: per-episode ``(T, obs_dim)`` observation arrays
        - ``actions_list``: per-episode ``(T, action_dim)`` action arrays
        - ``successes``: per-episode success flags (or None)
        - ``episode_lengths``: per-episode timestep counts
    """
    episodes_dir = eval_dir / "episodes"
    episode_files = sorted(episodes_dir.glob("ep*.pkl"))
    if not episode_files:
        raise FileNotFoundError(f"No episode pickle files found in {episodes_dir}")

    obs_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    successes: List[Optional[bool]] = []
    episode_lengths: List[int] = []

    for pkl_path in episode_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        keys = set(data.keys()) if hasattr(data, "keys") else set()

        # --- Actions ---
        raw_actions = (
            data["action"] if "action" in keys else
            data["actions"] if "actions" in keys else
            None
        )
        if raw_actions is None:
            raise KeyError(f"No 'action' key in {pkl_path}; keys={list(keys)}")
        acts = _to_array(raw_actions)  # (T, action_dim) or (T,) or (T, chunk, a_dim)
        if acts.ndim == 1:
            acts = acts[:, None]
        elif acts.ndim == 3:
            # Chunked diffusion actions (T, chunk_size, a_dim) — take first in chunk
            acts = acts[:, 0, :]

        # --- Observations (low-dim state used as visual features in mlp mode) ---
        raw_obs = (
            data["obs"] if "obs" in keys else
            data["state"] if "state" in keys else
            data["observations"] if "observations" in keys else
            data["img"] if "img" in keys else
            data["image"] if "image" in keys else
            data["agentview_image"] if "agentview_image" in keys else
            None
        )
        if raw_obs is None:
            raise KeyError(f"No obs/state/img key in {pkl_path}; keys={list(keys)}")
        obs = _to_array(raw_obs)  # (T, ...) possibly multi-dim
        if obs.ndim > 2:
            obs = obs.reshape(len(obs), -1)

        T = min(len(acts), len(obs))
        obs_list.append(obs[:T])
        actions_list.append(acts[:T])

        # --- Success flag ---
        raw_success = data["success"] if "success" in keys else None
        if raw_success is not None:
            # pandas Series or scalar
            success_vals = raw_success.values if hasattr(raw_success, "values") else raw_success
            success_flag = bool(success_vals[-1]) if hasattr(success_vals, "__len__") else bool(success_vals)
        else:
            raw_rewards = (
                data["reward"] if "reward" in keys else
                data["rewards"] if "rewards" in keys else
                None
            )
            if raw_rewards is not None:
                success_flag = bool(_to_array(raw_rewards)[-1] > 0)
            else:
                success_flag = None
        successes.append(success_flag)
        episode_lengths.append(T)

    return obs_list, actions_list, successes, episode_lengths


def _to_array(series: Any) -> np.ndarray:
    """Convert various sequence types to a float32 numpy array.

    Handles: plain ndarray, pandas Series of scalars, pandas Series of
    ndarrays (object dtype — e.g. per-timestep obs/action arrays), and
    plain Python lists/iterables.
    """
    if isinstance(series, np.ndarray):
        return series.astype(np.float32)
    # pandas Series or DataFrame
    if hasattr(series, "values"):
        vals = series.values
        # Object array whose elements are ndarrays — must stack element-wise
        if vals.dtype == object:
            return np.stack([
                v.astype(np.float32) if isinstance(v, np.ndarray)
                else np.array(v, dtype=np.float32)
                for v in vals
            ], axis=0)
        return vals.astype(np.float32)
    # Generic iterable
    arr = []
    for item in series:
        if hasattr(item, "values"):
            arr.append(np.array(item.values, dtype=np.float32))
        elif isinstance(item, np.ndarray):
            arr.append(item.astype(np.float32))
        else:
            arr.append(np.array(item, dtype=np.float32))
    return np.stack(arr, axis=0)


def _flatten_obs(obs_list: List[np.ndarray]) -> List[np.ndarray]:
    """Flatten multi-dimensional observations to 1-D per timestep."""
    result = []
    for obs in obs_list:
        if obs.ndim > 2:
            obs = obs.reshape(len(obs), -1)
        result.append(obs.astype(np.float32))
    return result


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

class TrainENAPPerceptionStep(PipelineStep[Dict[str, Any]]):
    """Extract visual features and HDBSCAN symbols from eval rollout episodes.

    Config keys consumed (all under ``graph_building.enap``):
    - ``backbone``: ``"mlp"`` (default) or ``"dino"``
    - ``feature_dim``: fused feature output dim (default 128)
    - ``proprio_dim``: proprioception / state input dim (default inferred)
    - ``fourier_output_dim``: Fourier embedding dim (default 64)
    - ``hdbscan_min_cluster_size``: HDBSCAN param (default 10)
    - ``hdbscan_min_samples``: HDBSCAN param (default = min_cluster_size)
    - ``hdbscan_cluster_selection_epsilon``: HDBSCAN param (default 0.0)
    - ``device``: torch device string (default ``"cuda"`` if available)

    Reads eval data from the same task config used by ``run_clustering``:
    - ``task_config``: task YAML name
    - ``config_root``: ``"iv"`` or ``"pd"``
    - ``reference_seed``: reference seed for path resolution
    - ``policy_seeds``: list of seeds (uses first seed for ENAP; single model)
    """

    name = "train_enap_perception"

    def save(self, result: Dict[str, Any]) -> None:
        self.step_dir.mkdir(parents=True, exist_ok=True)
        with open(self.step_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        (self.step_dir / "done").touch()

    def load(self) -> Optional[Dict[str, Any]]:
        p = self.step_dir / "result.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def compute(self) -> Dict[str, Any]:
        from policy_doctor.enap.perception import (
            HDBSCANClusterer,
            VisualEncoder,
            extract_features,
        )
        from influence_visualizer.data_loader import get_eval_dir_for_seed

        cfg = self.cfg
        enap_cfg = OmegaConf.select(cfg, "graph_building.enap") or {}

        # --- Resolve eval dir ---
        task_config = OmegaConf.select(cfg, "task_config")
        config_root = OmegaConf.select(cfg, "config_root") or "iv"

        if config_root == "iv":
            base = iv_task_configs_base(self.repo_root)
        else:
            base = PACKAGE_ROOT / "configs"

        task_yaml = base / f"{task_config}.yaml"
        with open(task_yaml) as f:
            task_cfg = yaml.safe_load(f)

        eval_dir_base = task_cfg["eval_dir"]
        reference_seed = str(OmegaConf.select(cfg, "reference_seed") or 0)
        seeds = expand_seeds(
            OmegaConf.select(cfg, "policy_seeds")
            or OmegaConf.select(cfg, "seeds")
            or [0]
        )
        # ENAP perception uses the first seed only (single model hypothesis)
        seed = seeds[0]
        eval_dir_seed = get_eval_dir_for_seed(eval_dir_base, seed, reference_seed)
        eval_dir = self.repo_root / eval_dir_seed

        if self.dry_run:
            print(
                f"  [dry_run] TrainENAPPerceptionStep seed={seed} eval_dir={eval_dir}"
            )
            return {"dry_run": True, "eval_dir": str(eval_dir)}

        # --- Load episodes ---
        print(f"  Loading episodes from {eval_dir}")
        obs_list, actions_list, successes, episode_lengths = _load_episode_pickles(eval_dir)
        print(f"  Episodes: {len(obs_list)}, total timesteps: {sum(episode_lengths)}")

        # Flatten obs to 1-D per timestep
        obs_list_flat = _flatten_obs(obs_list)
        obs_dim = obs_list_flat[0].shape[1] if obs_list_flat[0].ndim > 1 else 1
        action_dim = actions_list[0].shape[1] if actions_list[0].ndim > 1 else 1

        # --- Build flat arrays + per-timestep metadata ---
        all_obs = np.concatenate(obs_list_flat, axis=0)   # (N, obs_dim)
        all_actions = np.concatenate(actions_list, axis=0)  # (N, action_dim)
        metadata: List[Dict] = []
        ep_offset = 0
        for ep_idx, (ep_len, success) in enumerate(zip(episode_lengths, successes)):
            for t in range(ep_len):
                metadata.append({
                    "rollout_idx": ep_idx,
                    "timestep": t,
                    "success": success,
                })
            ep_offset += ep_len

        # --- Config ---
        backbone = str(OmegaConf.select(enap_cfg, "backbone") or "mlp")
        feature_dim = int(OmegaConf.select(enap_cfg, "feature_dim") or 128)
        fourier_output_dim = int(OmegaConf.select(enap_cfg, "fourier_output_dim") or 64)
        hdbscan_min_cluster_size = int(
            OmegaConf.select(enap_cfg, "hdbscan_min_cluster_size") or 10
        )
        hdbscan_min_samples = OmegaConf.select(enap_cfg, "hdbscan_min_samples")
        if hdbscan_min_samples is not None:
            hdbscan_min_samples = int(hdbscan_min_samples)
        hdbscan_eps = float(
            OmegaConf.select(enap_cfg, "hdbscan_cluster_selection_epsilon") or 0.0
        )

        device_str = str(
            OmegaConf.select(enap_cfg, "device")
            or OmegaConf.select(cfg, "device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        device = torch.device(device_str)

        # In mlp mode the "image" input is the flat obs; proprio is a small fixed
        # slice (last <proprio_dim> dims) or we treat obs entirely as the image input
        # with a dummy 1-d proprio to keep the architecture consistent.
        proprio_dim_cfg = OmegaConf.select(enap_cfg, "proprio_dim")
        if proprio_dim_cfg is not None:
            proprio_dim = int(proprio_dim_cfg)
            image_input_dim = obs_dim - proprio_dim
            img_features = all_obs[:, :image_input_dim]
            proprio_features = all_obs[:, image_input_dim:]
        else:
            # Use full obs as image features; proprio = zeros (no split)
            image_input_dim = obs_dim
            proprio_dim = 1
            img_features = all_obs
            proprio_features = np.zeros((len(all_obs), 1), dtype=np.float32)

        print(
            f"  VisualEncoder: backbone={backbone}, img_dim={image_input_dim}, "
            f"proprio_dim={proprio_dim}, feature_dim={feature_dim}"
        )

        encoder = VisualEncoder(
            image_input_dim=image_input_dim,
            proprio_dim=proprio_dim,
            output_dim=feature_dim,
            backbone=backbone,
            fourier_output_dim=fourier_output_dim,
        ).to(device)

        # Extract features in batches
        imgs_t = torch.from_numpy(img_features)
        prop_t = torch.from_numpy(proprio_features)
        features = extract_features(encoder, imgs_t, prop_t, device=device)
        print(f"  Features: {features.shape}")

        # --- HDBSCAN ---
        print(
            f"  HDBSCAN: min_cluster_size={hdbscan_min_cluster_size}, "
            f"min_samples={hdbscan_min_samples}, eps={hdbscan_eps}"
        )
        clusterer = HDBSCANClusterer(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=hdbscan_eps,
        )
        symbols = clusterer.fit_predict(features)
        num_symbols = clusterer.num_symbols
        print(f"  Symbols: {num_symbols} clusters")

        # --- Persist ---
        self.step_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.step_dir / "feature_embeddings.npy", features)
        np.save(self.step_dir / "symbol_assignments.npy", symbols)
        np.save(self.step_dir / "actions.npy", all_actions)
        with open(self.step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        torch.save(encoder.state_dict(), self.step_dir / "encoder_state.pt")

        return {
            "num_symbols": num_symbols,
            "num_timesteps": int(len(symbols)),
            "num_episodes": len(episode_lengths),
            "action_dim": int(action_dim),
            "feature_dim": int(feature_dim),
            "obs_dim": int(obs_dim),
            "backbone": backbone,
            "eval_dir": str(eval_dir),
            "seed": str(seed),
        }

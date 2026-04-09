"""
RoboCasa LeRobot (v2) → diffusion_policy image batches.

Uses robocasa's GR00T-derived LeRobot loaders only (no language / lang_emb).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from robocasa.utils.groot_utils.groot_dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)

from diffusion_policy.common.normalize_util import (
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _modality_keys(dataset_path: Path, modality_filename: str) -> dict[str, list[str]]:
    modality_path = dataset_path / modality_filename
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)
    modality_dict: dict[str, list[str]] = {}
    for key in modality_meta.keys():
        modality_dict[key] = [f"{key}.{m}" for m in modality_meta[key]]
    return modality_dict


class LerobotRobocasaImageDataset(LeRobotSingleDataset, BaseImageDataset):
    """
    Single LeRobot dataset root (``.../lerobot`` with ``meta/``, ``data/``, ``videos/``).

    ``shape_meta`` uses ``lerobot_keys`` entries like ``video.robot0_eye_in_hand``,
    ``state.end_effector_position_relative``, matching keys produced by
    ``robocasa``'s modality.json layout.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",
        use_legacy_normalizer: bool = False,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        filter_key: Optional[str] = None,
        embodiment_tag: str = "oxe_droid",
    ):
        assert n_obs_steps and n_obs_steps > 0
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported for LeRobot RoboCasa adapter"
        dataset_path_p = Path(dataset_path).expanduser().resolve()
        delta_indices = list(range(-n_obs_steps + 1, horizon - n_obs_steps + 1))
        delta_indices_obs = list(range(-n_obs_steps + 1, 1))
        assert len(delta_indices_obs) == n_obs_steps

        modality_keys_dict = _modality_keys(dataset_path_p, LE_ROBOT_MODALITY_FILENAME)
        video_modality_keys = modality_keys_dict["video"]
        state_modality_keys = [
            k for k in modality_keys_dict["state"] if k != "state.dummy_tensor"
        ]
        action_modality_keys = modality_keys_dict["action"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
        }

        LeRobotSingleDataset.__init__(
            self,
            dataset_path=dataset_path_p,
            filter_key=filter_key,
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
        )
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

        rgb_keys: dict[str, list[str]] = {}
        lowdim_keys: dict[str, list[str]] = {}
        obs_shape_meta = copy.deepcopy(shape_meta["obs"])
        if "lang_emb" in obs_shape_meta:
            raise ValueError(
                "lang_emb is not supported in LerobotRobocasaImageDataset; remove language keys from shape_meta."
            )
        for key, attr in obs_shape_meta.items():
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                rgb_keys[key] = attr["lerobot_keys"]
            elif typ == "low_dim":
                lowdim_keys[key] = attr["lerobot_keys"]
            else:
                raise ValueError(f"Unsupported obs type {typ!r} for key {key!r}")

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_info = self.shape_meta["action"]
        self.lerobot_action_keys = self.action_info["lerobot_keys"]
        self.action_size = int(self.action_info["shape"][0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = LeRobotSingleDataset.__getitem__(self, idx)
        T_slice = slice(self.n_obs_steps)
        obs_dict: dict[str, np.ndarray] = {}

        for key, lerobot_keys in self.rgb_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            obs_dict[key] = (
                np.moveaxis(data[lerobot_key][T_slice], -1, 1).astype(np.float32) / 255.0
            )

        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            obs_dict[key] = data[lerobot_key][T_slice].astype(np.float32)

        action_parts = []
        for lr_key in self.lerobot_action_keys:
            if lr_key not in data:
                raise KeyError(f"Missing action key {lr_key!r} in LeRobot sample")
            action_parts.append(data[lr_key])
        action_concat = np.concatenate(action_parts, axis=-1)
        assert action_concat.shape[-1] == self.action_size, (
            f"action_concat shape mismatch: {action_concat.shape[-1]} != {self.action_size}"
        )

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action_concat.astype(np.float32)),
        }

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={},
        )

        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1
            lerobot_key = lerobot_keys[0].replace("state.", "")
            stat = self._metadata.statistics.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if isinstance(v, np.ndarray):
                    stat[k] = v.astype(np.float32)

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("sin") or key.endswith("cos"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim obs key for normalizer: {key!r}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_validation_dataset(self) -> BaseImageDataset:
        return self

    def __len__(self) -> int:
        return len(self.all_steps)


class LerobotRobocasaMixtureDataset(LeRobotMixtureDataset, BaseImageDataset):
    """
    Mixture of LeRobot roots with explicit per-source paths and weights (no registry soup).
    """

    def __init__(
        self,
        shape_meta: dict,
        sources: List[Dict[str, Any]],
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",
        use_legacy_normalizer: bool = False,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        ds_weights: Optional[List[float]] = None,
        ds_weights_alpha: float = 0.40,
        metadata_config: Optional[dict] = None,
        embodiment_tag: str = "oxe_droid",
    ):
        assert sources, "sources must be a non-empty list of {path, filter_key?, weight?}"
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported for LeRobot RoboCasa adapter"

        per_source_weights: List[Optional[float]] = []
        datasets: List[LerobotRobocasaImageDataset] = []
        for spec in sources:
            path = spec["path"]
            fk = spec.get("filter_key", None)
            w = spec.get("weight", None)
            per_source_weights.append(w)
            datasets.append(
                LerobotRobocasaImageDataset(
                    shape_meta=shape_meta,
                    dataset_path=path,
                    horizon=horizon,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    n_obs_steps=n_obs_steps,
                    abs_action=abs_action,
                    rotation_rep=rotation_rep,
                    use_legacy_normalizer=use_legacy_normalizer,
                    use_cache=use_cache,
                    seed=seed,
                    val_ratio=val_ratio,
                    filter_key=fk,
                    embodiment_tag=embodiment_tag,
                )
            )

        if ds_weights is None and all(w is not None for w in per_source_weights):
            ds_weights_arr = np.array([float(w) for w in per_source_weights], dtype=np.float64)
        elif ds_weights is not None:
            assert len(ds_weights) == len(datasets)
            ds_weights_arr = np.array(ds_weights, dtype=np.float64)
        else:
            ds_weights_arr = np.array(
                [np.power(len(ds), ds_weights_alpha) for ds in datasets], dtype=np.float64
            )
            ds_weights_arr = ds_weights_arr / ds_weights_arr[0]

        print("LerobotRobocasaMixtureDataset weights:", ds_weights_arr)

        mixture = list(zip(datasets, ds_weights_arr.tolist()))
        meta_cfg = metadata_config or {"percentile_mixing_method": "weighted_average"}
        LeRobotMixtureDataset.__init__(
            self,
            data_mixture=mixture,
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            metadata_config=meta_cfg,
        )

        rgb_keys: dict[str, list[str]] = {}
        lowdim_keys: dict[str, list[str]] = {}
        obs_shape_meta = copy.deepcopy(shape_meta["obs"])
        if "lang_emb" in obs_shape_meta:
            raise ValueError(
                "lang_emb is not supported in LerobotRobocasaMixtureDataset; remove language keys from shape_meta."
            )
        for key, attr in obs_shape_meta.items():
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                rgb_keys[key] = attr["lerobot_keys"]
            elif typ == "low_dim":
                lowdim_keys[key] = attr["lerobot_keys"]
            else:
                raise ValueError(f"Unsupported obs type {typ!r} for key {key!r}")

        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = int(n_obs_steps)  # type: ignore[arg-type]
        self.shape_meta = shape_meta
        self.action_info = self.shape_meta["action"]
        self.lerobot_action_keys = self.action_info["lerobot_keys"]
        self.action_size = int(self.action_info["shape"][0])
        self.lang_emb = None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset, trajectory_name, step = self.sample_step(idx)
        global_ds_index = self.to_global_index(dataset, trajectory_name, step)
        return dataset.__getitem__(global_ds_index)

    def to_global_index(self, dataset, trajectory_id: int, base_index: int) -> int:
        traj_idx = dataset.get_trajectory_index(trajectory_id)
        return int(dataset.start_indices[traj_idx] + base_index)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        tag = self.datasets[0].tag
        all_stats = self.merged_metadata[tag].statistics

        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={},
        )

        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1
            lerobot_key = lerobot_keys[0].replace("state.", "")
            stat = all_stats.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if isinstance(v, np.ndarray):
                    stat[k] = v.astype(np.float32)

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("sin") or key.endswith("cos"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim obs key for normalizer: {key!r}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_validation_dataset(self) -> BaseImageDataset:
        return self

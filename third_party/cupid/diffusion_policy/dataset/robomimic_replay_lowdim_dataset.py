import copy
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.common.hdf5_robomimic_layout import sorted_robomimic_demo_keys
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    robomimic_abs_action_only_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    build_combined_curation_masks,
    get_dataset_masks,
    load_sample_mask_from_curation_config,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys: List[str] = [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        abs_action=False,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=False,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        dataset_mask_kwargs: Dict[str, Any] = {},
        sample_curation_config: Optional[str] = None,
        holdout_selection_config: Optional[str] = None,
        sample_mask_drop_threshold: float = 1.0,
        verify_curation_dataset_identity: bool = True,
    ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file["data"]
            demo_keys = sorted_robomimic_demo_keys(demos)
            if not demo_keys:
                raise ValueError(
                    f"No demo_* groups under data/ in {dataset_path!r} "
                    "(expected Robomimic or MimicGen-merged HDF5)."
                )
            for dk in tqdm(demo_keys, desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[dk]
                episode = _data_to_obs(
                    raw_obs=demo["obs"],
                    raw_actions=demo["actions"][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                replay_buffer.add_episode(episode)

        train_mask, val_mask, holdout_mask = get_dataset_masks(
            dataset_path=dataset_path,
            num_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            seed=seed,
            **dataset_mask_kwargs,
        )

        # Build episode and sample masks: filter (exclude from train) and/or holdout selection (include from holdout).
        episode_mask = train_mask
        sample_mask = None
        if sample_curation_config is not None or holdout_selection_config is not None:
            episode_mask, sample_mask = build_combined_curation_masks(
                episode_ends=replay_buffer.episode_ends[:],
                train_mask=train_mask,
                holdout_mask=holdout_mask,
                sample_curation_config=sample_curation_config,
                holdout_selection_config=holdout_selection_config,
                sample_mask_drop_threshold=sample_mask_drop_threshold,
                verify_curation_dataset_identity=verify_curation_dataset_identity,
            )
            if sample_mask is not None:
                ep_ends = replay_buffer.episode_ends[:]
                n_train_eps = int(train_mask.sum())
                n_holdout_eps = int(holdout_mask.sum())
                n_train_samples = 0
                n_train_included = 0
                n_holdout_samples = 0
                n_holdout_included = 0
                n_holdout_eps_active = 0
                for ep_idx in range(replay_buffer.n_episodes):
                    start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
                    end = int(ep_ends[ep_idx])
                    ep_len = end - start
                    ep_included = int(np.sum(sample_mask[start:end]))
                    if train_mask[ep_idx]:
                        n_train_samples += ep_len
                        n_train_included += ep_included
                    elif holdout_mask[ep_idx]:
                        n_holdout_samples += ep_len
                        n_holdout_included += ep_included
                        if ep_included > 0:
                            n_holdout_eps_active += 1
                pct_train = 100.0 * n_train_included / n_train_samples if n_train_samples else 0.0
                pct_holdout = 100.0 * n_holdout_included / n_holdout_samples if n_holdout_samples else 0.0
                combined = n_train_included + n_holdout_included
                combined_total = n_train_samples + n_holdout_samples
                pct_combined = 100.0 * combined / combined_total if combined_total else 0.0
                lines = [
                    f"\n{'='*70}",
                    f"  [Curation] SAMPLE-LEVEL BREAKDOWN",
                    f"  Train split:    {n_train_included}/{n_train_samples} samples included ({pct_train:.1f}%)  [{n_train_eps} episodes]",
                ]
                if sample_curation_config:
                    lines.append(f"    filter config: {sample_curation_config}")
                lines.append(
                    f"  Holdout split:  {n_holdout_included}/{n_holdout_samples} samples included ({pct_holdout:.1f}%)  "
                    f"[{n_holdout_eps_active}/{n_holdout_eps} episodes active]"
                )
                if holdout_selection_config:
                    lines.append(f"    selection config: {holdout_selection_config}")
                lines += [
                    f"  Combined:       {combined}/{combined_total} samples included ({pct_combined:.1f}%)",
                    f"{'='*70}",
                ]
                print("\n".join(lines))

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
            sample_mask=sample_mask,
            sample_mask_drop_threshold=sample_mask_drop_threshold,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.holdout_mask = holdout_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self._dataset_path = dataset_path
        self._dataset_mask_kwargs = dataset_mask_kwargs

        # Visualization.
        self._return_image = False
        self._train_image_dataset: Optional[RobomimicReplayImageDataset] = None

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_holdout_dataset(self):
        holdout_set = copy.copy(self)
        holdout_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.holdout_mask,
        )
        holdout_set.train_mask = self.holdout_mask
        return holdout_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = (
                    robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
                )
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # aggregate obs stats
        obs_stat = array_to_stats(self.replay_buffer["obs"])

        normalizer["obs"] = normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        if self._return_image:
            assert isinstance(self._train_image_dataset, RobomimicReplayImageDataset)
            torch_data["img"] = self._train_image_dataset.__getitem__(idx)["img"]
        return torch_data


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([raw_obs[key] for key in obs_keys], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)

    data = {"obs": obs, "action": raw_actions}
    return data

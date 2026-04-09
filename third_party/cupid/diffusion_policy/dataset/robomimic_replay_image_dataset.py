import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
import os
import shutil
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.hdf5_robomimic_layout import sorted_robomimic_demo_keys
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
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
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

register_codecs()


class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",  # ignored when abs_action=False
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        load_to_memory=False,
        dataset_mask_kwargs: Dict[str, Any] = {},
        sample_curation_config: Optional[str] = None,
        holdout_selection_config: Optional[str] = None,
        sample_mask_drop_threshold: float = 1.0,
        verify_curation_dataset_identity: bool = True,
    ):
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        store = None if load_to_memory else zarr.MemoryStore()
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=store
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        train_mask, val_mask, holdout_mask = get_dataset_masks(
            dataset_path=dataset_path,
            num_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
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
            key_first_k=key_first_k,
            sample_mask=sample_mask,
            sample_mask_drop_threshold=sample_mask_drop_threshold,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
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
        self._render_obs_key = None

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

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W

            # Visualization.
            if self._return_image and key == self._render_obs_key:
                pass
            else:
                del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }

        if self._return_image:
            assert isinstance(self._render_obs_key, str), (
                "render obs key is not a string!"
            )
            torch_data["img"] = data[self._render_obs_key][T_slice].astype(np.uint8)
            del data[self._render_obs_key]

        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
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
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        demo_keys = sorted_robomimic_demo_keys(demos)
        if not demo_keys:
            raise ValueError(
                f"No demo_* groups under data/ in {dataset_path!r} "
                "(expected Robomimic or MimicGen-merged HDF5)."
            )
        episode_ends = list()
        prev_end = 0
        for dk in demo_keys:
            demo = demos[dk]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array(
            "episode_ends",
            episode_ends,
            dtype=np.int64,
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = list()
            for dk in demo_keys:
                demo = demos[dk]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                ), (
                    f"this_data.shape {this_data.shape} is not the same as {(n_steps,) + tuple(shape_meta['action']['shape'])}"
                )
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                ), (
                    f"this_data.shape {this_data.shape} is not the same as {(n_steps,) + tuple(shape_meta['obs'][key]['shape'])}"
                )
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
        ) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8,
                    )
                    for episode_idx, dk in enumerate(demo_keys):
                        demo = demos[dk]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )

import importlib.util
import os
import sys
import uuid
import time
import wandb
import numpy as np
import pandas as pd
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import pickle
import math
import yaml
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy


def _import_mimicgen_for_robosuite_registry() -> None:
    """MimicGen HDF5s may name envs ``Square_D1`` etc.; robosuite learns them only via ``import mimicgen``."""
    if importlib.util.find_spec("mimicgen") is not None:
        import mimicgen  # noqa: F401
        return
    # ``.../third_party/cupid/diffusion_policy/env_runner/this_file`` → ``.../third_party``
    third_party = pathlib.Path(__file__).resolve().parents[3]
    vendored = third_party / "mimicgen"
    if (vendored / "mimicgen").is_dir():
        sys.path.insert(0, str(vendored))
    if importlib.util.find_spec("mimicgen") is not None:
        import mimicgen  # noqa: F401


_import_mimicgen_for_robosuite_registry()


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env


class MimicgenLowdimRunner(BaseLowdimRunner):
    """
    MimicGen-sourced robomimic-layout HDF5 (same stack as RobomimicLowdimRunner; separate entry point).
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            save_episodes=False
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        # TODO: Hacky fix.
                        # filename = pathlib.Path(output_dir).joinpath(
                            # 'media', wv.util.generate_id() + ".mp4")
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f"{uuid.uuid4()}_{time.time()}_{os.getpid()}.mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    if save_episodes: 
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f"ep{i:04d}" + ".mp4")
                    else:
                        # TODO: Hacky fix.
                        # filename = pathlib.Path(output_dir).joinpath(
                            # 'media', wv.util.generate_id() + ".mp4")
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f"{uuid.uuid4()}_{time.time()}_{os.getpid()}.mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns, shared_memory=False)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

        # save episodes
        self.save_episodes = save_episodes
        self.episode_dir = None
        if save_episodes:
            assert n_train == n_train_vis == 0
            assert n_test > 0
            self.episode_dir = pathlib.Path(output_dir) / "episodes"
            self.episode_dir.mkdir()
            self.media_dir = pathlib.Path(output_dir) / "media"

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # total number of timesteps across episodes
        if self.save_episodes:
            total_timesteps = 0
            episode_lengths = []
            episode_successes = []
            all_sim_episodes: list = []  # for rollouts.hdf5

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])
            
            # save episodes
            if self.save_episodes:
                timestep = 0
                # One episode_data list per active env
                episode_data = [[] for _ in range(this_n_active_envs)]
                # Per-env sim-state/action lists for rollouts.hdf5
                sim_states = [[] for _ in range(this_n_active_envs)]
                sim_actions = [[] for _ in range(this_n_active_envs)]
                # Track "ever succeeded" to handle transient success states
                ever_succeeded = [False] * this_n_active_envs

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                
                # store timestep in episode dataset (all active envs)
                if self.save_episodes:
                    obs_all = np_obs_dict["obs"].copy().astype(np.float32)
                    if "action_pred" in np_action_dict:
                        act_all = np_action_dict["action_pred"].copy().astype(np.float32)
                    elif "action" in np_action_dict:
                        act_all = np_action_dict["action"].copy().astype(np.float32)
                    else:
                        raise ValueError(f"Actions for policy {type(policy)} cannot be retrieved.")
                    for env_i in range(this_n_active_envs):
                        episode_data[env_i].append({
                            "idx": total_timesteps,
                            "episode": chunk_idx * this_n_active_envs + env_i,
                            "timestep": timestep,
                            "obs": obs_all[env_i],
                            "action": act_all[env_i],
                        })

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)

                # Collect per-sub-step sim states from MultiStepWrapper buffer.
                # Each inference step covers n_action_steps MuJoCo steps; we save
                # all of them so rollouts.hdf5 has per-timestep resolution for
                # MimicGen trajectory replay.
                if self.save_episodes:
                    try:
                        # env.call returns list[list[state]] — one list per env
                        per_env_bufs = env.call('get_sim_states_buf')[:this_n_active_envs]
                    except Exception:
                        per_env_bufs = [None] * this_n_active_envs
                    for env_i in range(this_n_active_envs):
                        buf = per_env_bufs[env_i]
                        if buf:  # list of per-substep states for this env
                            for sub_i, st in enumerate(buf):
                                sim_states[env_i].append(st)
                                # corresponding action: env_action[env_i, sub_i, :]
                                a_idx = min(sub_i, env_action.shape[1] - 1)
                                sim_actions[env_i].append(env_action[env_i, a_idx])
                done = np.all(done)
                
                # Terminate on success if saving episodes.
                if self.save_episodes:
                    successes = env.call("_is_success")[:this_n_active_envs]
                    for env_i in range(this_n_active_envs):
                        if successes[env_i]:
                            ever_succeeded[env_i] = True
                    success = successes[0]  # used for episode[0] postfix below
                    done = np.all(ever_succeeded) if not done else done

                past_action = action

                # update pbar
                pbar.update(action.shape[1])
                if self.save_episodes:
                    timestep += action.shape[1]
                    total_timesteps += this_n_active_envs
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            # save episode data for each active env
            if self.save_episodes:
                for env_i in range(this_n_active_envs):
                    ep_global = chunk_idx * n_envs + env_i
                    ep_success = bool(ever_succeeded[env_i])  # use sticky success flag
                    postfix = "succ" if ep_success else "fail"
                    episode_lengths.append(len(episode_data[env_i]))
                    episode_successes.append(ep_success)
                    ep_df = pd.DataFrame(episode_data[env_i])
                    ep_df["reward"] = float(reward[env_i])
                    ep_df["success"] = ep_success
                    with open(self.episode_dir / f"ep{ep_global:04d}_{postfix}.pkl", "wb") as f:
                        pickle.dump(ep_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # Collect sim episode for rollouts.hdf5
                    if sim_states[env_i]:
                        all_sim_episodes.append({
                            "states": np.stack(sim_states[env_i], axis=0).astype(np.float64),
                            "actions": np.stack(sim_actions[env_i], axis=0).astype(np.float64),
                            "success": ep_success,
                            "model_file": None,
                        })
                del episode_data, sim_states, sim_actions

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            # TODO: Hacky fix.
            # if video_path is not None: 
            if video_path is not None and os.path.exists(video_path):
                time.sleep(1)
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        if self.save_episodes and all_sim_episodes:
            self._write_rollouts_hdf5(all_sim_episodes)

        if self.save_episodes:
            # rename media files based on success or failure (only if videos were recorded)
            episode_files = sorted([ep for ep in self.episode_dir.iterdir()])
            if self.media_dir.exists():
                media_files = sorted([ep for ep in self.media_dir.iterdir()])
                if len(media_files) == len(episode_files):
                    for episode_file, media_file in zip(episode_files, media_files):
                        media_file.rename(self.media_dir / f"{episode_file.stem}{media_file.suffix}")

            # save episode metadata
            metadata = {
                "length": int(sum(episode_lengths)),
                "episode_lengths": [int(x) for x in episode_lengths],
                "episode_successes": [bool(x) for x in episode_successes],
            }
            with open(self.episode_dir / "metadata.yaml", "w") as f:
                yaml.safe_dump(metadata, f)

        return log_data

    def _write_rollouts_hdf5(self, sim_episodes: list) -> pathlib.Path:
        """Write a MimicGen-compatible ``rollouts.hdf5`` from per-episode sim data."""
        import json as _json
        out_path = self.episode_dir / "rollouts.hdf5"
        with h5py.File(out_path, "w") as f:
            grp = f.create_group("data")
            grp.attrs["env_args"] = _json.dumps(self.env_meta)
            grp.attrs["total"] = len(sim_episodes)
            for i, ep in enumerate(sim_episodes):
                demo_grp = grp.create_group(f"demo_{i}")
                demo_grp.create_dataset("states", data=ep["states"])
                demo_grp.create_dataset("actions", data=ep["actions"])
                demo_grp.attrs["success"] = int(ep.get("success", False))
        print(f"[MimicgenLowdimRunner] rollouts.hdf5 → {out_path} ({len(sim_episodes)} episodes)")
        return out_path

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

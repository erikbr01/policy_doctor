import os
import io
import uuid
import time
import queue
import multiprocessing as mp
import tempfile
from typing import Any, Dict, Optional

import wandb
import numpy as np
import pandas as pd
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import yaml
import dill
import pickle
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from omegaconf import OmegaConf

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    import copy as _copy
    # Deep-copy env_meta: robocasa-support robomimic's create_env_from_metadata
    # mutates env_meta["env_kwargs"] (adds "env_name"). Without a copy, the
    # fallback path receives a poisoned dict with duplicate env_name.
    env_meta_safe = _copy.deepcopy(env_meta)
    try:
        # Normal path: works when robomimic's env_utils doesn't have the
        # buggy `action_dimension` print (cupid robomimic).
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta_safe,
            render=False,
            render_offscreen=enable_render,
            use_image_obs=enable_render,
        )
    except AttributeError:
        # Fallback for robocasa-support robomimic (new robosuite 1.5+): the
        # `action_dimension` print inside env_utils.create_env fails before
        # reset() because controllers are only initialized during reset().
        # Construct the EnvRobosuite wrapper directly, bypassing the buggy print.
        from robomimic.envs.env_robosuite import EnvRobosuite
        env_kwargs = _copy.deepcopy(env_meta.get("env_kwargs", {}))
        env = EnvRobosuite(
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=enable_render,
            use_image_obs=enable_render,
            postprocess_visual_obs=True,
            **env_kwargs,
        )
    return env


def _make_eval_env(env_meta, shape_meta, render_obs_key, fps, crf,
                   n_obs_steps, n_action_steps, max_steps):
    """Create a single wrapped eval environment from plain-dict config.

    Defined at module level so it can be called inside a spawn-ed subprocess
    (closures are not picklable across spawn boundaries).
    """
    robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
    robomimic_env.env.hard_reset = False

    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)

    return MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=render_obs_key,
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps, codec='h264', input_pix_fmt='rgb24',
                crf=crf, thread_type='FRAME', thread_count=1,
            ),
            file_path=None,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )


def _undo_transform_action(action, rotation_transformer):
    """Convert absolute-action representation back to env action space."""
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3:3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


def _eval_worker(result_queue, policy_path, eval_config):
    """Entry point for the spawn-ed eval subprocess.

    Runs in a completely fresh interpreter so that MuJoCo / EGL
    initialisation cannot corrupt the parent process's glibc heap.
    This fixes the ``malloc_consolidate(): unaligned fastbin chunk``
    crash that occurs when fork-based DataLoader workers inherit a
    heap dirtied by MuJoCo's custom malloc arenas.
    """
    try:
        import torch
        import numpy as np
        import dill
        import tqdm
        import pathlib
        import yaml
        import pickle
        from diffusion_policy.common.pytorch_util import dict_apply

        device = torch.device(eval_config['device'])

        # Reconstruct the policy from config YAML + state dict.
        # Avoids dill-serializing the full policy object, which fails when
        # the policy contains robomimic Config objects (their __setitem__
        # references __parent before the object is initialised by dill).
        import hydra
        from omegaconf import OmegaConf
        policy_cfg = OmegaConf.create(eval_config['policy_cfg_yaml'])
        policy = hydra.utils.instantiate(policy_cfg)
        state_dict = torch.load(policy_path, map_location=device,
                                weights_only=False)
        policy.load_state_dict(state_dict)
        policy.to(device)
        policy.eval()

        abs_action = eval_config['abs_action']
        rotation_transformer = None
        if abs_action:
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        single_env = _make_eval_env(
            env_meta=eval_config['env_meta'],
            shape_meta=eval_config['shape_meta'],
            render_obs_key=eval_config['render_obs_key'],
            fps=eval_config['fps'],
            crf=eval_config['crf'],
            n_obs_steps=eval_config['n_obs_steps'],
            n_action_steps=eval_config['n_action_steps'],
            max_steps=eval_config['max_steps'],
        )

        env_init_fn_dills = eval_config['env_init_fn_dills']
        n_inits = len(env_init_fn_dills)
        env_name = eval_config['env_meta']['env_name']
        max_steps = eval_config['max_steps']
        past_action_flag = eval_config['past_action']
        n_obs_steps = eval_config['n_obs_steps']
        tqdm_interval_sec = eval_config['tqdm_interval_sec']
        save_episodes = eval_config['save_episodes']
        episode_dir = eval_config.get('episode_dir')
        media_dir = eval_config.get('media_dir')

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        if save_episodes:
            episode_lengths = []
            episode_successes = []

        try:
            for ep_idx in range(n_inits):
                print(f"[eval] Episode {ep_idx + 1}/{n_inits} ({env_name})")

                init_fn = dill.loads(env_init_fn_dills[ep_idx])
                init_fn(single_env)

                if save_episodes:
                    timestep = 0
                    episode_data = []

                obs = single_env.reset()
                obs = {k: v[None] for k, v in obs.items()}
                past_action = None
                policy.reset()

                pbar = tqdm.tqdm(
                    total=max_steps,
                    desc=f"Eval {env_name} {ep_idx + 1}/{n_inits}",
                    leave=False,
                    mininterval=tqdm_interval_sec,
                )

                done = False
                ep_rewards = []
                info = {}
                while not done:
                    np_obs_dict = dict(obs)
                    if past_action_flag and past_action is not None:
                        np_obs_dict['past_action'] = past_action[
                            :, -(n_obs_steps - 1):].astype(np.float32)

                    obs_dict = dict_apply(
                        np_obs_dict,
                        lambda x: torch.from_numpy(x).to(device=device),
                    )

                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    np_action_dict = dict_apply(
                        action_dict, lambda x: x.detach().to('cpu').numpy()
                    )

                    action = np_action_dict['action']
                    if not np.all(np.isfinite(action)):
                        raise RuntimeError(f"Nan or Inf action: {action}")

                    env_action = action
                    if abs_action:
                        env_action = _undo_transform_action(action, rotation_transformer)

                    obs_raw, reward, done, info = single_env.step(env_action[0])
                    obs = {k: v[None] for k, v in obs_raw.items()}
                    ep_rewards.append(reward)
                    done = bool(done)

                    past_action = action
                    pbar.update(action.shape[1])

                    if save_episodes:
                        timestep += action.shape[1]

                pbar.close()

                all_rewards[ep_idx] = [float(r) for r in ep_rewards]
                single_env.env.video_recoder.stop()
                all_video_paths[ep_idx] = single_env.env.file_path

                if save_episodes:
                    success = bool(info.get('success', False))
                    postfix = "succ" if success else "fail"
                    episode_lengths.append(timestep)
                    episode_successes.append(success)
                    ep_dir = pathlib.Path(episode_dir)
                    with open(ep_dir / f"ep{ep_idx:04d}_{postfix}.pkl", "wb") as f:
                        pickle.dump(episode_data, f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            single_env.close()

        # Handle save_episodes post-processing inside the subprocess
        if save_episodes:
            ep_dir = pathlib.Path(episode_dir)
            m_dir = pathlib.Path(media_dir)
            episode_files = sorted(ep_dir.iterdir())
            media_files = sorted(m_dir.iterdir())
            assert len(episode_files) == len(media_files)
            for episode_file, media_file in zip(episode_files, media_files):
                media_file.rename(
                    m_dir / f"{episode_file.stem}{media_file.suffix}")
            metadata = {
                "length": int(sum(episode_lengths)),
                "episode_lengths": [int(x) for x in episode_lengths],
                "episode_successes": [bool(x) for x in episode_successes],
            }
            with open(ep_dir / "metadata.yaml", "w") as f:
                yaml.safe_dump(metadata, f)

        result_queue.put({
            'status': 'ok',
            'rewards': all_rewards,
            'video_paths': all_video_paths,
        })

    except Exception as e:
        import traceback
        result_queue.put({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
        })


class RobocasaImageRunner(BaseImageRunner):
    """
    RoboCasa kitchen image rollouts.

    Two env-meta sources (mutually exclusive):
    * ``dataset_path`` — read ``env_meta`` from a robomimic-layout HDF5 (old workflow).
      Also used to replay train-episode init states (``n_train > 0``).
    * ``env_name`` + ``env_kwargs`` — build ``env_meta`` directly from config params
      (new LeRobot-format workflow, no HDF5 needed).  ``n_train`` must be 0 when using
      this path because there are no demo init states to read.

    ``extra_env_kwargs`` can override ``env_kwargs`` in either case.
    """

    def __init__(self,
            output_dir,
            shape_meta: dict,
            dataset_path: Optional[str] = None,
            env_name: Optional[str] = None,
            env_kwargs: Optional[Dict[str, Any]] = None,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            save_episodes=False,
            extra_env_kwargs: Optional[Dict[str, Any]] = None,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # --- build env_meta ---
        if dataset_path is not None:
            # HDF5 path: read env_meta from file
            dataset_path = os.path.expanduser(dataset_path)
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        elif env_name is not None:
            # Direct config path: build env_meta from env_name + env_kwargs
            if n_train > 0:
                raise ValueError(
                    "n_train > 0 requires dataset_path to read demo init states from HDF5. "
                    "Use dataset_path or set n_train=0."
                )
            if env_kwargs is None:
                base_kwargs = {}
            elif OmegaConf.is_config(env_kwargs):
                base_kwargs = OmegaConf.to_container(env_kwargs, resolve=True)
            else:
                base_kwargs = dict(env_kwargs)
            env_meta = {
                "env_name": env_name,
                "type": 1,  # robomimic EnvType.ROBOSUITE_TYPE
                "env_kwargs": base_kwargs,
            }
        else:
            raise ValueError(
                "Provide either dataset_path (HDF5) or env_name to configure the eval environment."
            )

        if extra_env_kwargs is not None:
            ek = extra_env_kwargs
            if OmegaConf.is_config(ek):
                ek = OmegaConf.to_container(ek, resolve=True)
            if not isinstance(ek, dict):
                raise TypeError(f"extra_env_kwargs must be a dict, got {type(ek)}")
            env_meta["env_kwargs"].update(ek)

        # Ensure shape_meta is a plain dict (not OmegaConf) for subprocess pickling.
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)

        rotation_transformer = None
        if abs_action:
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
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
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
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
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train (requires HDF5 for demo init states)
        if n_train > 0:
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
                            filename = pathlib.Path(output_dir).joinpath(
                                'media', f"{uuid.uuid4()}_{time.time()}_{os.getpid()}.mp4")
                            filename.parent.mkdir(parents=False, exist_ok=True)
                            filename = str(filename)
                            env.env.file_path = filename

                        # switch to init_state reset
                        assert isinstance(env.env.env, RobomimicImageWrapper)
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
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # Env is created lazily in run() to avoid blocking at startup.
        # (AsyncVectorEnv with spawn context blocks for minutes while workers
        # re-import torch + robosuite; with fork it deadlocks with MuJoCo 3.x EGL.)
        self.dummy_env_fn = dummy_env_fn
        self.env_meta = env_meta
        self.shape_meta = shape_meta
        self.render_obs_key = render_obs_key
        self.env = None  # created lazily in run()
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

        # save episodes
        self.save_episodes = save_episodes
        self.episode_dir = None
        if save_episodes:
            assert n_envs == 1
            assert n_train == n_train_vis == 0
            assert n_test > 0 and n_test == n_test_vis
            self.episode_dir = pathlib.Path(output_dir) / "episodes"
            self.episode_dir.mkdir()
            self.media_dir = pathlib.Path(output_dir) / "media"

    def run(self, policy: BaseImagePolicy, policy_cfg=None):
        """Run eval rollouts in a spawn-ed subprocess.

        MuJoCo / EGL is initialised only inside the child process, so the
        parent's glibc heap stays clean for fork-based DataLoader workers.
        The one-time subprocess startup cost (~30-60 s for torch + robosuite
        imports) is negligible compared to the rollout wall-time.

        Args:
            policy: The policy to evaluate.
            policy_cfg: OmegaConf config used to instantiate the policy
                (cfg.policy from the workspace). Required — used to
                reconstruct the policy in the subprocess without dill.
        """
        if policy_cfg is None:
            raise ValueError(
                "policy_cfg is required for RobocasaImageRunner.run(). "
                "Pass cfg.policy from the workspace.")

        from omegaconf import OmegaConf
        device = policy.device
        n_inits = len(self.env_init_fn_dills)

        # Save only the state dict — avoids dill-serializing the full model,
        # which fails on robomimic Config objects.
        tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        try:
            torch.save(policy.state_dict(), tmp)
            tmp.close()
            policy_path = tmp.name

            eval_config = {
                'device': str(device),
                'policy_cfg_yaml': OmegaConf.to_yaml(policy_cfg),
                'env_meta': self.env_meta,
                'shape_meta': self.shape_meta,
                'render_obs_key': self.render_obs_key,
                'fps': self.fps,
                'crf': self.crf,
                'n_obs_steps': self.n_obs_steps,
                'n_action_steps': self.n_action_steps,
                'max_steps': self.max_steps,
                'abs_action': self.abs_action,
                'past_action': self.past_action,
                'tqdm_interval_sec': self.tqdm_interval_sec,
                'env_init_fn_dills': self.env_init_fn_dills,
                'save_episodes': self.save_episodes,
                'episode_dir': (str(self.episode_dir)
                                if self.episode_dir else None),
                'media_dir': (str(self.media_dir)
                              if getattr(self, 'media_dir', None) else None),
            }

            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            p = ctx.Process(target=_eval_worker,
                            args=(result_queue, policy_path, eval_config))
            print(f"[eval] Spawning eval subprocess for "
                  f"{n_inits} rollouts ...")
            p.start()

            # Wait for result, checking that the subprocess stays alive.
            result = None
            while result is None:
                try:
                    result = result_queue.get(timeout=30)
                except queue.Empty:
                    if not p.is_alive():
                        raise RuntimeError(
                            f"Eval subprocess died with exit code "
                            f"{p.exitcode}. Check stderr for "
                            f"MuJoCo/robosuite errors.")
            p.join()
        finally:
            os.unlink(policy_path)

        if result['status'] == 'error':
            raise RuntimeError(
                f"Eval subprocess failed:\n{result['traceback']}")

        # Build log_data in the main process (wandb objects must be
        # created here, where wandb is initialised).
        all_rewards = result['rewards']
        all_video_paths = result['video_paths']

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

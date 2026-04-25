"""Generate rollouts.hdf5 using parallel AsyncVectorEnv — no video overhead.

Runs num_episodes in chunks of n_envs parallel environments, batching policy
inference on GPU.  Captures raw simulator states at each sub-step via a thin
SimStateCapturingWrapper around RobomimicLowdimWrapper, then writes a
MimicGen-compatible rollouts.hdf5.

Usage:
    python scripts/gen_rollouts_hdf5.py \
        --train_dir third_party/cupid/data/outputs/train/... \
        --episode_dir .../latest/episodes \
        --num_episodes 100 \
        --n_envs 8 \
        --device cuda:0
"""
from __future__ import annotations
import json, math, pathlib, sys
import click, dill, h5py, numpy as np, torch

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


# ---------------------------------------------------------------------------
# State-capturing env wrapper (runs inside each AsyncVectorEnv subprocess)
# ---------------------------------------------------------------------------

class SimStateCapturingWrapper:
    """Thin wrapper around RobomimicLowdimWrapper that buffers sim states.

    All gym.Env methods are delegated; additionally exposes:
      reset_sim_buffers() — clear per-episode state/action/model_file buffers
      collect_episode_data() — return and clear buffers as a dict
    """

    def __init__(self, inner):
        self._inner = inner
        self._sim_states: list = []
        self._sim_actions: list = []

    # -- gym interface -------------------------------------------------------
    @property
    def observation_space(self):
        return self._inner.observation_space

    @property
    def action_space(self):
        return self._inner.action_space

    def seed(self, seed=None):
        return self._inner.seed(seed)

    def reset(self):
        self._sim_states = []
        self._sim_actions = []
        obs = self._inner.reset()
        # Capture model XML after reset (used by MimicGen's prepare_src_dataset)
        try:
            self._model_file = self._inner.env.get_state().get('model', None)
        except Exception:
            self._model_file = None
        return obs

    def step(self, action):
        # Capture sim state BEFORE applying action
        try:
            st = self._inner.env.get_state()['states']
            self._sim_states.append(np.array(st, dtype=np.float64))
            self._sim_actions.append(np.array(action, dtype=np.float64))
        except Exception:
            pass
        return self._inner.step(action)

    def close(self):
        return self._inner.close()

    def render(self, *a, **kw):
        return self._inner.render(*a, **kw)

    @property
    def metadata(self):
        return getattr(self._inner, 'metadata', {})

    # -- extra methods called via AsyncVectorEnv.call() ----------------------
    def collect_episode_data(self):
        return {
            'states':  np.array(self._sim_states,  dtype=np.float64) if self._sim_states  else np.empty((0,), dtype=np.float64),
            'actions': np.array(self._sim_actions, dtype=np.float64) if self._sim_actions else np.empty((0,), dtype=np.float64),
            'model_file': getattr(self, '_model_file', None),
        }

    def is_success(self):
        """Delegate to the environment's own success check."""
        try:
            result = self._inner.env.is_success()
            if isinstance(result, dict):
                return bool(result.get('task', False))
            return bool(result)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def _make_env_fn(env_meta, obs_keys, seed: int):
    def _fn():
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
        import mimicgen  # noqa: F401
        ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': obs_keys})
        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
        inner = RobomimicLowdimWrapper(
            env=robomimic_env, obs_keys=obs_keys, init_state=None,
            render_hw=(256, 256), render_camera_name='agentview')
        wrapper = SimStateCapturingWrapper(inner)
        wrapper.seed(seed)
        return wrapper
    return _fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option('--train_dir', required=True)
@click.option('--episode_dir', required=True)
@click.option('--num_episodes', default=100)
@click.option('--test_start_seed', default=100000)
@click.option('--train_ckpt', default='latest')
@click.option('--n_envs', default=8)
@click.option('--device', default='cuda:0')
def main(train_dir, episode_dir, num_episodes, test_start_seed, train_ckpt, n_envs, device):
    import hydra
    import robomimic.utils.file_utils as FileUtils
    from diffusion_policy.common.device_util import get_device
    from diffusion_policy.common.trak_util import get_best_checkpoint
    from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
    import mimicgen  # noqa: F401

    train_dir  = pathlib.Path(train_dir)
    episode_dir = pathlib.Path(episode_dir)
    out_path   = episode_dir / 'rollouts.hdf5'
    if out_path.exists():
        print('[gen_rollouts_hdf5] rollouts.hdf5 already exists — skipping.')
        return

    # --- load policy --------------------------------------------------------
    ckpt_dir = train_dir / 'checkpoints'
    if train_ckpt == 'latest':
        ckpt = ckpt_dir / 'latest.ckpt'
    elif train_ckpt == 'best':
        ckpt = get_best_checkpoint(list(ckpt_dir.iterdir()))
    else:
        ckpt = ckpt_dir / f'{train_ckpt}.ckpt'
    print(f'[gen_rollouts_hdf5] checkpoint: {ckpt}')

    payload = torch.load(str(ckpt), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(episode_dir.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if getattr(cfg.training, 'use_ema', False):
        policy = workspace.ema_model
    torch_device = get_device(device)
    policy.to(torch_device)
    policy.eval()
    dtype = policy.dtype

    # --- env config ---------------------------------------------------------
    runner_cfg = cfg.task.env_runner
    dataset_path_cfg = str(runner_cfg.dataset_path)
    if not pathlib.Path(dataset_path_cfg).is_absolute():
        candidate = train_dir
        for _ in range(8):
            candidate = candidate.parent
            if (candidate / dataset_path_cfg).exists():
                dataset_path = str(candidate / dataset_path_cfg); break
        else:
            raise FileNotFoundError(f'Cannot find {dataset_path_cfg!r} relative to {train_dir}')
    else:
        dataset_path = dataset_path_cfg

    obs_keys     = list(runner_cfg.obs_keys)
    n_obs_steps  = int(cfg.n_obs_steps)
    n_action_steps = int(runner_cfg.get('n_action_steps', 8))
    max_steps    = int(runner_cfg.get('max_steps', 400))
    env_meta     = FileUtils.get_env_metadata_from_dataset(dataset_path)

    print(f'[gen_rollouts_hdf5] {num_episodes} eps | n_envs={n_envs} | '
          f'max_steps={max_steps} | seed_start={test_start_seed}')
    print(f'[gen_rollouts_hdf5] env={env_meta["env_name"]}')

    all_episodes: list[dict] = [None] * num_episodes  # type: ignore

    # Build one pool of n_envs (re-seed each chunk via AsyncVectorEnv init fns)
    n_chunks = math.ceil(num_episodes / n_envs)

    for chunk_idx in range(n_chunks):
        start   = chunk_idx * n_envs
        end     = min(num_episodes, start + n_envs)
        active  = end - start

        seeds = [test_start_seed + start + i for i in range(active)]
        # pad to n_envs if last chunk is smaller (extras are discarded)
        seeds_padded = seeds + [seeds[-1]] * (n_envs - active)

        env_fns = [_make_env_fn(env_meta, obs_keys, s) for s in seeds_padded]
        vec_env = AsyncVectorEnv(env_fns)

        obs_batch = vec_env.reset()          # (n_envs, obs_dim)
        obs_buf   = np.stack([obs_batch] * n_obs_steps, axis=1)  # (n_envs, n_obs_steps, obs_dim)

        ep_done    = [False] * n_envs
        ep_success = [False] * n_envs
        ep_step    = [0]     * n_envs
        policy.reset()

        while not all(ep_done[i] or ep_step[i] >= max_steps for i in range(active)):
            obs_tensor = torch.from_numpy(obs_buf).to(torch_device, dtype=dtype)
            with torch.no_grad():
                action_np = policy.predict_action({'obs': obs_tensor})['action'].cpu().numpy()
            # action_np: (n_envs, n_action_steps, action_dim)

            for sub_i in range(n_action_steps):
                acts = action_np[:, sub_i, :]   # (n_envs, action_dim)
                # SimStateCapturingWrapper.step captures state before applying
                next_obs, _rew, dones, _info = vec_env.step(acts)

                for i in range(active):
                    if ep_done[i]:
                        continue
                    obs_buf[i] = np.roll(obs_buf[i], -1, axis=0)
                    obs_buf[i, -1] = next_obs[i]
                    ep_step[i] += 1
                    if dones[i]:
                        ep_done[i] = True
                    if ep_step[i] >= max_steps:
                        ep_done[i] = True

                # Check success for active envs
                succ_list = vec_env.call('is_success')
                for i in range(active):
                    if not ep_done[i] and succ_list[i]:
                        ep_success[i] = True
                        ep_done[i]    = True

                if all(ep_done[i] or ep_step[i] >= max_steps for i in range(active)):
                    break

        # Collect episode data
        ep_data_list = vec_env.call('collect_episode_data')
        vec_env.close()

        for i in range(active):
            ep_i = start + i
            all_episodes[ep_i] = {**ep_data_list[i], 'success': ep_success[i]}
            status = 'succ' if ep_success[i] else 'fail'
            T = len(ep_data_list[i]['states'])
            print(f'[gen_rollouts_hdf5] ep {ep_i+1:3d}/{num_episodes}  {status}  T={T}',
                  flush=True)

    # --- write HDF5 ---------------------------------------------------------
    n_succ = sum(1 for ep in all_episodes if ep['success'])
    print(f'[gen_rollouts_hdf5] writing {out_path} ({n_succ}/{num_episodes} success)...')
    with h5py.File(out_path, 'w') as f:
        grp = f.create_group('data')
        grp.attrs['env_args'] = json.dumps(env_meta)
        grp.attrs['total']    = len(all_episodes)
        for i, ep in enumerate(all_episodes):
            dg = grp.create_group(f'demo_{i}')
            if ep['states'].ndim >= 1 and ep['states'].shape[0] > 0:
                dg.create_dataset('states',  data=ep['states'])
                dg.create_dataset('actions', data=ep['actions'])
            dg.attrs['success'] = int(ep['success'])
            if ep.get('model_file') is not None:
                dg.attrs['model_file'] = ep['model_file']
    print(f'[gen_rollouts_hdf5] done → {out_path}')


if __name__ == '__main__':
    main()

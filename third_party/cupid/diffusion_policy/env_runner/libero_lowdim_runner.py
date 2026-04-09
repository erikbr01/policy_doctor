"""
LIBERO adapter for diffusion policy eval. Uses LIBERO envs and init states
while producing the same episode output format as RobomimicLowdimRunner
for compatibility with eval_save_episodes, TRAK, and the influence visualizer.
"""
import os
import sys
import pathlib
import collections
import math
import pickle
import tqdm
import numpy as np
import torch
import pandas as pd
import yaml

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.pytorch_util import dict_apply

# LIBERO is in third_party; ensure it is on path when running from project root.
_CUPID_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LIBERO_PATH = _CUPID_ROOT / "third_party" / "LIBERO"
if _LIBERO_PATH.exists() and str(_LIBERO_PATH) not in sys.path:
    sys.path.insert(0, str(_LIBERO_PATH))


def _obs_dict_to_lowdim(obs_dict, obs_keys, obs_key_mapping):
    """Build concatenated lowdim obs (1, obs_dim) from LIBERO env obs dict."""
    # obs_dict keys are env names (e.g. robot0_joint_pos); map to our key order.
    parts = []
    for key in obs_keys:
        env_key = obs_key_mapping.get(key, key)
        if env_key in obs_dict:
            parts.append(np.atleast_1d(obs_dict[env_key]).astype(np.float32))
        else:
            raise KeyError(f"Missing obs key {env_key} (our key {key}) in env obs.")
    return np.concatenate(parts, axis=-1)


class LiberoLowdimRunner(BaseLowdimRunner):
    """
    Eval runner for LIBERO benchmarks. Uses LIBERO env and init states;
    outputs episodes in the same layout as RobomimicLowdimRunner for
    eval_save_episodes / TRAK / influence visualizer compatibility.
    """

    def __init__(
        self,
        output_dir,
        benchmark_name,
        task_id,
        libero_datasets_dir,
        bddl_folder,
        init_states_folder,
        obs_keys,
        obs_key_mapping,
        n_train=0,
        n_train_vis=0,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=500,
        n_obs_steps=2,
        n_action_steps=8,
        n_latency_steps=0,
        render_hw=(128, 128),
        fps=10,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        save_episodes=False,
    ):
        super().__init__(output_dir)
        self.benchmark_name = benchmark_name
        self.task_id = int(task_id)
        self.libero_datasets_dir = pathlib.Path(libero_datasets_dir).expanduser()
        self.bddl_folder = pathlib.Path(bddl_folder).expanduser()
        self.init_states_folder = pathlib.Path(init_states_folder).expanduser()
        self.obs_keys = list(obs_keys)
        self.obs_key_mapping = dict(obs_key_mapping)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = n_obs_steps + n_latency_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.save_episodes = save_episodes
        self.render_hw = render_hw
        self.fps = fps

        # Resolve benchmark and task (LIBERO API).
        from libero.libero.benchmark import get_benchmark
        from libero.libero import get_libero_path

        self._libero_path_fn = get_libero_path
        benchmark = get_benchmark(benchmark_name)(0)
        self._task = benchmark.get_task(task_id)
        self._bddl_path = self.bddl_folder / self._task.problem_folder / self._task.bddl_file
        self._init_states_path = self.init_states_folder / self._task.problem_folder / self._task.init_states_file
        assert self._bddl_path.exists(), f"BDDL not found: {self._bddl_path}"
        # Init states are only required when run() is called (rollout/eval). Defer check so training can start without them.

        self.n_test = n_test
        self.test_start_seed = test_start_seed
        self._init_states = None  # loaded in run()
        self._env = None
        self._obs_history = []  # list of (obs_dim,) for current episode
        self._past_action = None

        if save_episodes:
            self.episode_dir = pathlib.Path(output_dir) / "episodes"
            self.episode_dir.mkdir(parents=True, exist_ok=True)
            self.media_dir = pathlib.Path(output_dir) / "media"
            self.media_dir.mkdir(parents=True, exist_ok=True)

    def _build_env(self):
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(
            bddl_file_name=str(self._bddl_path),
            camera_heights=self.render_hw[0],
            camera_widths=self.render_hw[1],
            has_renderer=False,
            has_offscreen_renderer=True,
            horizon=self.max_steps,
        )
        return env

    def _get_lowdim_obs(self, raw_obs):
        o = _obs_dict_to_lowdim(raw_obs, self.obs_keys, self.obs_key_mapping)
        return o  # (obs_dim,)

    def _obs_for_policy(self):
        """Stack last n_obs_steps into (1, n_obs_steps, obs_dim)."""
        n = min(len(self._obs_history), self.n_obs_steps)
        if n == 0:
            return None
        recent = self._obs_history[-self.n_obs_steps:]
        if len(recent) < self.n_obs_steps:
            pad = [recent[0]] * (self.n_obs_steps - len(recent))
            recent = pad + recent
        stacked = np.stack(recent, axis=0)
        return stacked[np.newaxis, ...].astype(np.float32)  # (1, n_obs_steps, obs_dim)

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        # Headless: set OpenGL backend before first env creation (avoids hang when no display)
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        if not self._init_states_path.exists():
            import logging
            logging.getLogger(__name__).warning(
                "Init states not found: %s. Skipping rollouts this epoch. "
                "Add the .pruned_init file (see LIBERO docs) to enable rollouts.",
                self._init_states_path,
            )
            return {"test/mean_score": 0.0}
        if self._init_states is None:
            loaded = torch.load(self._init_states_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.Tensor):
                self._init_states = loaded.numpy()
            elif isinstance(loaded, dict) and "states" in loaded:
                self._init_states = np.asarray(loaded["states"])
            else:
                self._init_states = np.asarray(loaded)
        init_states = self._init_states
        n_episodes = init_states.shape[0]
        n_test = min(n_episodes, self.n_test)
        test_start_seed = self.test_start_seed

        all_rewards = []
        all_video_paths = []
        episode_lengths = []
        episode_successes = []

        pbar = tqdm.tqdm(
            range(n_test),
            desc="Libero eval",
            leave=True,
            mininterval=self.tqdm_interval_sec,
        )
        for ep_idx in pbar:
            env = self._build_env()
            self._env = env
            idx = (test_start_seed + ep_idx) % n_episodes
            init_state = init_states[idx]
            # Single env: set_init_state takes one state (state_dim,)
            obs = env.set_init_state(init_state)
            # Dummy steps for physics settling (same as LIBERO metric.py)
            # Single env expects 1D action of length 7 (not (1, 7))
            dummy = np.zeros(7)
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            self._obs_history = []
            self._past_action = None
            policy.reset()
            done = False
            steps = 0
            episode_data = []
            reward_agg = 0.0
            success = False

            # Robosuite counts all step() calls; we do N_DUMMY steps before the loop, so total = N_DUMMY + steps.
            # Cap our loop so we never call step() when total would exceed env horizon (avoids "executing action in terminated episode").
            N_DUMMY = 5
            base_env = getattr(env, "env", env)
            env_horizon = getattr(base_env, "horizon", None)
            if env_horizon is not None:
                step_limit = min(self.max_steps, max(0, env_horizon - N_DUMMY))
            else:
                step_limit = self.max_steps

            step_pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"ep {ep_idx}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
                unit="step",
            )
            # Same pattern as LIBERO metric.evaluate_one_task_success: loop by step count, break when done
            while steps < step_limit:
                # Do not step again if episode already ended (robosuite raises otherwise)
                if done:
                    break
                # obs is a single-env dict (robot0_joint_pos, robot0_gripper_qpos, ...)
                lowdim = self._get_lowdim_obs(obs)
                self._obs_history.append(lowdim)
                obs_for_policy = self._obs_for_policy()
                if obs_for_policy is None:
                    obs, _, done, _ = env.step(dummy)  # dummy is (7,) for single env
                    continue

                np_obs_dict = {"obs": obs_for_policy}
                if self.past_action and self._past_action is not None:
                    np_obs_dict["past_action"] = self._past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                action = np_action_dict.get("action_pred", np_action_dict.get("action"))
                action = action[:, self.n_latency_steps :]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action")
                # Policy returns (batch, n_action_steps, action_dim); take first step for env
                action_1 = action[0, 0] if action.ndim == 3 else action[0]
                # Single env expects 1D action of length 7
                action_step = np.asarray(action_1, dtype=np.float64).flatten()

                if self.save_episodes:
                    try:
                        img = env.sim.render(width=self.render_hw[1], height=self.render_hw[0], camera_name="agentview")
                    except Exception:
                        img = np.zeros((self.render_hw[0], self.render_hw[1], 3), dtype=np.uint8)
                    episode_data.append({
                        "idx": len(episode_data),
                        "episode": ep_idx,
                        "timestep": steps,
                        "obs": np_obs_dict["obs"].copy().astype(np.float32)[0],
                        "action": action_1.copy().astype(np.float32),
                        "img": img,
                    })

                obs, reward, done, info = env.step(action_step)
                reward_agg += float(np.asarray(reward).flatten()[0])
                steps += 1
                step_pbar.update(1)
                # Break as soon as episode is done so we never step() again (robosuite raises otherwise)
                done_flat = np.asarray(done).flatten()
                if done_flat.any():
                    success = bool(done_flat[0])
                    if hasattr(env, "check_success"):
                        success = success or env.check_success()
                    break
                self._past_action = action

            step_pbar.close()
            if hasattr(env, "check_success") and not success:
                success = env.check_success()
            all_rewards.append(reward_agg)
            episode_lengths.append(len(episode_data) if episode_data else steps)
            episode_successes.append(success)

            n_done = len(all_rewards)
            n_succ = sum(episode_successes)
            pbar.set_postfix(
                succ=f"{n_succ}/{n_done}",
                mean_rew=f"{np.mean(all_rewards):.2f}",
                refresh=True,
            )

            if self.save_episodes and episode_data:
                postfix = "succ" if success else "fail"
                df = pd.DataFrame(episode_data)
                df["reward"] = reward_agg
                df["success"] = success
                with open(self.episode_dir / f"ep{ep_idx:04d}_{postfix}.pkl", "wb") as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

            env.close()
            del env
            self._env = None

        log_data = {}
        log_data["test/mean_score"] = np.mean(all_rewards)
        if self.save_episodes:
            metadata = {
                "length": int(sum(episode_lengths)),
                "episode_lengths": [int(x) for x in episode_lengths],
                "episode_successes": [bool(x) for x in episode_successes],
            }
            with open(self.episode_dir / "metadata.yaml", "w") as f:
                yaml.safe_dump(metadata, f)
        return log_data

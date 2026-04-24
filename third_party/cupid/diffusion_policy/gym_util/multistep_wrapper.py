import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
        # Simulator state / action buffers — populated if inner env supports
        # _get_simulator_state(); used to write rollouts.hdf5 for MimicGen.
        self._sim_states_buf: list = []
        self._sim_actions_buf: list = []
        self._episode_model_file: "str | None" = None
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        # Reset per-episode sim buffers and capture model file (once per episode).
        self._sim_states_buf = []
        self._sim_actions_buf = []
        self._episode_model_file = None
        if callable(getattr(self.env, '_get_episode_model_file', None)):
            try:
                self._episode_model_file = self.env._get_episode_model_file()
            except Exception:
                pass

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        _has_sim_state = callable(getattr(self.env, '_get_simulator_state', None))
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            # Capture simulator state BEFORE applying this action so that the
            # accumulated buffer matches the (state[t], action[t]) layout that
            # prepare_src_dataset / MimicGen expects.
            if _has_sim_state:
                try:
                    self._sim_states_buf.append(self.env._get_simulator_state())
                    self._sim_actions_buf.append(act.copy())
                except Exception:
                    pass
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

    def _render_frame(self, mode="rgb_array"):
        if callable(getattr(self.env, "_render_frame", None)):
            return self.env._render_frame(mode=mode)
        else:
            raise AttributeError(f"{self.env} does not have a callable method '_render_frame'.")
        
    def _is_success(self):
        if callable(getattr(self.env, "_is_success", None)):
            return self.env._is_success()
        else:
            raise AttributeError(f"{self.env} does not have a callable method '_is_success'.")

    def _get_episode_sim_data(self) -> dict:
        """Return accumulated simulator states, actions, and model XML for the current episode.

        States and actions are aligned so that ``states[t]`` is the simulator
        state at the moment ``actions[t]`` was applied (the layout expected by
        MimicGen's ``prepare_src_dataset``).

        Returns a dict with keys:
          ``states``     — np.ndarray of shape (T, state_dim), dtype float64
          ``actions``    — np.ndarray of shape (T, action_dim), dtype float64
          ``model_file`` — MJCF XML string (str | None if env doesn't support it)
        """
        states = (
            np.array(self._sim_states_buf, dtype=np.float64)
            if self._sim_states_buf
            else np.empty((0,), dtype=np.float64)
        )
        actions = (
            np.array(self._sim_actions_buf, dtype=np.float64)
            if self._sim_actions_buf
            else np.empty((0,), dtype=np.float64)
        )
        return {
            "states": states,
            "actions": actions,
            "model_file": self._episode_model_file,
        }
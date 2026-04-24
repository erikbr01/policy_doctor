# DAgger System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAgger Interactive Rollout                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  MonitoredPolicy (existing)              │
        │  ├─ Diffusion Policy (checkpoint)        │
        │  ├─ TrajectoryClassifier                 │
        │  │  └─ InfEmbedStreamScorer              │
        │  │     └─ Behavior Graph embedding       │
        │  └─ NodeValueThresholdRule               │
        │     └─ V-values from Bellman solve       │
        └─────────────────────────────────────────┘
                    │              │
           predict_action   classify & trigger
                    │              │
        ┌───────────▼──────────────▼──────────┐
        │   RobocasaDAggerRunner               │
        │   (Main Control Loop)                │
        │                                      │
        │   if auto_trigger or manual_override │
        │     → switch to human mode           │
        │   if human_done                      │
        │     → switch back to robot mode      │
        └──────────┬──────────────┬───────────┘
                   │              │
      ┌────────────▼──┐    ┌──────▼──────────┐
      │ RobocasaDAgger│    │ KeyboardIntervention
      │ Env           │    │ Device           │
      │               │    │ (pynput)         │
      │ Records:      │    │                  │
      │ - per-key obs │    │ Provides:        │
      │ - sim_state   │    │ - human action   │
      │ - acting_agent│    │ - is_intervening │
      │ - reward      │    │ - notify()       │
      └────────────┬──┘    └──────────────────┘
                   │
      ┌────────────▼──────────────┐
      │ Episode pkl (pandas DF)    │
      │ ├─ timestep               │
      │ ├─ obs (dict)             │
      │ ├─ action                 │
      │ ├─ reward, done           │
      │ ├─ acting_agent (key!)    │
      │ └─ sim_state              │
      └────────┬───────────────────┘
               │
    ┌──────────▼───────────┐
    │ build_dagger_dataset │
    │ Filter human only    │
    │ ↓                    │
    │ Robomimic HDF5       │
    │ (demo_0, demo_1...)  │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────────┐
    │ Retraining Pipeline      │
    │ D_0 ∪ D_1 ∪ D_2 ∪ ...   │
    │ (original + human demos) │
    └──────────────────────────┘
```

---

## Component Details

### 1. MonitoredPolicy (Existing)

**Purpose**: Wraps any policy to add per-timestep behavior monitoring.

**Key methods**:
- `predict_action(obs_dict)` → `{"action": (1, n_action_steps, action_dim)}`
  - Internally: calls wrapped policy, classifies via `TrajectoryClassifier`
  - Records in `episode_results[t]`:
    - `"result"`: full `MonitorResult` (embedding, influence scores, assignment)
    - `"intervention"`: `InterventionDecision` (triggered, node_id, node_value, reason)
- `reset()` → clears episode-level state

**Dependencies**: 
- TrajectoryClassifier (builds classifier from checkpoint)
- NodeValueThresholdRule (returns InterventionDecision at each step)

---

### 2. RobomimicDAggerEnv (New)

**Purpose**: Wraps `MultiStepWrapper(RobomimicLowdimWrapper(...))` to record DAgger-formatted episode data.

**Generic wrapper**: Works with ANY robomimic-compatible environment (robomimic, kitchen, robocasa, libero, blockpush, mimicgen, etc.) without modification. Backward compatibility alias: `RobocasaDAggerEnv`.

**Architecture** (wrapper stack):
```
RobocasaDAggerEnv ← (NEW) records per-step data
  ↓
MultiStepWrapper ← (existing) handles obs stacking (n_obs_steps)
  ↓
RobomimicLowdimWrapper ← (existing) handles obs extraction
  ↓
EnvRobosuite ← (robomimic) handles robosuite env interface
  ↓
robosuite.make() ← (existing) raw MuJoCo environment
```

**Key interface**:
```python
env = RobomimicDAggerEnv(inner_env, obs_keys, output_dir)
env.set_acting_agent("robot" or "human")  # Label controller
obs, reward, done, info = env.step(action)
path = env.save_episode()  # → ep0000.pkl (pandas DataFrame)
```

Works identically for all supported environments:
```python
# Kitchen stacking
env = RobomimicDAggerEnv(inner_env_square, obs_keys, output_dir)

# Robocasa pick-and-place
env = RobomimicDAggerEnv(inner_env_robocasa, obs_keys, output_dir)

# Libero long-horizon
env = RobomimicDAggerEnv(inner_env_libero, obs_keys, output_dir)
```

**Data recorded**:
- **obs**: `dict[key: np.ndarray]` — per-key observations (for HDF5)
  - Extracted from `RobomimicLowdimWrapper.env.get_observation()`
  - Keys: `["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]`
- **stacked_obs**: `np.ndarray(n_obs_steps, obs_dim)` — from MultiStepWrapper
- **sim_state**: `np.ndarray(state_dim,)` — from `EnvRobosuite.get_state()["states"]`
- **action, reward, done**: Standard gym interface
- **acting_agent**: `"robot"` or `"human"` (label for DAgger retraining)
- **timestep**: Step index in episode

---

### 3. KeyboardInterventionDevice (New)

**Purpose**: Provides human teleoperation via keyboard (pynput-based).

**Architecture**:
```
pynput.keyboard.Listener (background thread)
  ├─ on_press(key) → add to _keys_pressed
  ├─ on_release(key) → remove from _keys_pressed
  └─ Space key → toggle _is_intervening flag

Main loop (polling):
  ├─ device.get_action() → compose action from pressed keys
  └─ device.is_intervening → check if human is in control
```

**Key bindings** (10D OSC_POSE action space for PandaMobile):
```
Arm (6D):     W/S (z), A/D (x), Q/E (y)
Gripper (1D): G (close), H (open)
Base (3D):    I/K (x), J/L (rotation)
Control:      Space (toggle intervention)
```

**Key methods**:
- `get_action()` → `np.ndarray(10,)` or `None` (sum of pressed key bindings)
- `is_intervening` → `bool` (Space toggles this)
- `notify(msg)` → print intervention reason
- `reset()` → clear pressed keys and intervention flag

**Thread safety**: pynput listener runs in background; `_keys_pressed` and `_is_intervening` are checked atomically by main loop.

---

### 4. RobocasaDAggerRunner (New)

**Purpose**: Main control loop orchestrating policy rollout with intervention.

**State machine**:
```
ROBOT_MODE:
  1. Prepare obs: stack into (n_obs_steps, obs_dim)
  2. Call monitored_policy.predict_action() → action chunk
  3. Check intervention decision (auto-trigger from V-value)
  4. Check manual override (device.is_intervening)
  5. If either triggered: → HUMAN_MODE
  6. Else: execute action chunk, step through environment

HUMAN_MODE:
  1. Poll device.get_action()
  2. If action available: execute step
  3. Check if device.is_intervening still True
  4. If False: reset policy history, → ROBOT_MODE
```

**Key loop invariants**:
- obs_queue: `deque(maxlen=n_obs_steps)` maintains rolling obs history
- stacked_obs: `np.stack(obs_queue)` passed to policy as `{"obs": obs[None]}`
- Acting agent: always matches current mode (set via `env.set_acting_agent()`)
- Episode savings: triggered at end via `env.save_episode()`

**Integration points**:
- **MonitoredPolicy**: calls `predict_action()`, reads `episode_results[-1]["intervention"]`
- **RobocasaDAggerEnv**: calls `step()` and `set_acting_agent()`
- **KeyboardInterventionDevice**: polls `is_intervening`, calls `get_action()`
- **DAggerVisualizer**: optionally updates live display

---

### 5. DAggerVisualizer (New, Optional)

**Purpose**: Live matplotlib display of rollout (camera + node assignment + status).

**Implementation**:
- `plt.ion()` (interactive mode) + `fig.canvas.flush_events()`
- Imshow for camera feed (updates data in-place, doesn't redraw figure)
- Suptitle for status text and V-values
- Border color: green (robot) or red (human)

**No blocking**: uses minimal pauses (0.001s) to keep GUI responsive while polling main loop.

---

### 6. build_dagger_dataset.py (New Script)

**Purpose**: Convert collected DAgger episode pkls to robomimic HDF5 format for retraining.

**Pipeline**:
```
pkl files (ep0000.pkl, ep0001.pkl, ...)
  ↓ (read pandas DataFrames)
  ├─ Filter by acting_agent == "human" (if --filter_human_only)
  ├─ Extract contiguous human segments
  └─ Write as demo_N groups in HDF5
      ├─ obs/{object, robot0_eef_pos, ...}
      ├─ actions
      ├─ rewards, dones
      ├─ states (for replay)
      └─ acting_agent (custom extension)
```

**Two modes**:
1. **`--filter_human_only`** (default): Extract only human-controlled segments
   - Standard HG-DAgger: pure corrective demonstrations
2. **Full episodes**: Include entire rollout (robot + human)
   - For offline analysis or mixed training

---

## Data Flow

### Episode Recording

```
step t:
  ┌─ obs_dict = {"obs": stacked_obs[None]}
  ├─ action_chunk = monitored_policy.predict_action(obs_dict)
  ├─ intervention = monitored_policy.episode_results[-1]["intervention"]
  │
  ├─ for t in range(n_action_steps):
  │   ├─ obs, reward, done, info = env.step(action_chunk[t])
  │   ├─ env._episode_data.append({
  │   │    "timestep": t,
  │   │    "obs": {per_key},         ← from env.env.env.get_observation()
  │   │    "stacked_obs": stacked,   ← from env.step()
  │   │    "action": action,
  │   │    "reward": reward,
  │   │    "done": done,
  │   │    "acting_agent": env._acting_agent,  ← "robot" or "human"
  │   │    "sim_state": state,       ← from env.env.env.get_state()
  │   │  })
  │   ├─ obs_queue.append(obs)
  │   └─ check manual override mid-chunk
  │
  └─ visualizer.update(...)

episode end:
  └─ env.save_episode()
      └─ pd.DataFrame(env._episode_data).to_pickle("ep0000.pkl")
```

### HDF5 Conversion

```
ep0000.pkl (DataFrame with 100 rows)
  ├─ rows 0-30: acting_agent="robot"
  ├─ rows 31-50: acting_agent="human"  ← extracted as demo_0
  ├─ rows 51-70: acting_agent="robot"
  ├─ rows 71-90: acting_agent="human"  ← extracted as demo_1
  └─ rows 91-99: acting_agent="robot"

ep0001.pkl (50 rows, all human)
  └─ extracted as demo_2

Output HDF5:
  data/
    demo_0/  ← from ep0000 rows 31-50
      obs/{keys...}
      actions, rewards, dones
    demo_1/  ← from ep0000 rows 71-90
      obs/{keys...}
      actions, rewards, dones
    demo_2/  ← from ep0001 rows 0-49
      obs/{keys...}
      actions, rewards, dones
```

---

## Dependencies & Imports

### Existing (policy_doctor)
- `policy_doctor.monitoring.monitored_policy.MonitoredPolicy`
- `policy_doctor.monitoring.intervention.NodeValueThresholdRule`
- `policy_doctor.monitoring.trajectory_classifier.TrajectoryClassifier`
- `policy_doctor.behaviors.behavior_graph.BehaviorGraph`
- `policy_doctor.data.adapters.ensure_robocasa_on_path`
- `policy_doctor.gym_util.multistep_wrapper.MultiStepWrapper`

### Existing (cupid)
- `diffusion_policy.env.robomimic.robomimic_lowdim_wrapper.RobomimicLowdimWrapper`
- `robomimic.utils.env_utils.EnvUtils`
- `robomimic.utils.file_utils.FileUtils`

### New (third-party)
- `pynput.keyboard` — keyboard input (requires Accessibility on macOS)
- `matplotlib` — live visualization

### External
- `gym` — base env interface
- `numpy`, `pandas`, `h5py` — data handling

---

## Testing Strategy

Unit tests use **mocks** to avoid robosuite/robomimic dependencies:

- `test_robocasa_dagger_env.py`: `MockMultiStepWrapper` + `MockRobomimicWrapper`
- `test_intervention_device.py`: Direct action composition tests (no pynput needed)
- `test_dagger_runner.py`: Mocked policy + env, structure validation

Integration tests (future): require full `cupid_torch2` environment with checkpoint + infembed artifacts.

---

## Extension Points

### Add Alternative Intervention Devices

Subclass `InterventionDevice`:
```python
class SpaceMouseInterventionDevice(InterventionDevice):
    def __init__(self, vendor_id=9583, product_id=50741):
        # Connect to SpaceMouse HID
    
    @property
    def is_intervening(self): ...
    def get_action(self): ...
```

Then pass to `RobocasaDAggerRunner`:
```python
device = SpaceMouseInterventionDevice()
runner = RobocasaDAggerRunner(..., intervention_device=device)
```

### Custom Intervention Rules

Subclass `InterventionRule` in `policy_doctor.monitoring.intervention`:
```python
class ConfidenceThresholdRule(InterventionRule):
    def check(self, result, history):
        # Example: trigger on low influence score variance
        return InterventionDecision(triggered=...)
```

Pass to `MonitoredPolicy`:
```python
monitored_policy = MonitoredPolicy(
    ...,
    intervention_rule=ConfidenceThresholdRule(threshold=0.1)
)
```

---

## Known Limitations & Future Work

1. **Multi-agent handling**: Currently single robot only. Extend env wrapper for multi-arm setups.
2. **Episode merging**: Manual HDF5 merge for combining original + DAgger datasets. Utility script needed.
3. **Mid-episode policy updates**: Requires policy reload. Support online policy swapping.
4. **Real-time V-value visualization**: Currently shows distance to centroid, not V-value. Add V-value buffer.
5. **SpaceMouse support**: Keyboard is default; SpaceMouse extension module ready for implementation.

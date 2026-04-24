# MimicGen Data Generation Parameters

A comprehensive guide to steering the MimicGen data generation process, including variance control, initial pose constraints, and trajectory generation tuning.

## 1. Trajectory Variance & Stochasticity

| Parameter | Where | Description |
|-----------|-------|-------------|
| `action_noise` | Task spec per-subtask | Amount of action noise applied during trajectory execution (float, e.g., 0.05) |
| `apply_noise_during_interpolation` | Task spec per-subtask | Whether to apply action noise during interpolation between subtasks (boolean) |
| `experiment.seed` | Config / `run_mimicgen_generate.py` | Random seed for entire generation run (controls all randomness) |

## 2. Initial Pose Constraints

| Parameter | Where | Description |
|-----------|-------|-------------|
| Object pose in current scene | Environment reset | MimicGen samples new task instances on each `env.reset()` — configure robosuite environment's randomization |
| Robot initial joint angles | Environment reset | Configured in robosuite environment (not directly in MimicGen) |
| `experiment.generation.interpolate_from_last_target_pose` | Config | If True, start interpolation from last target pose (more variance); if False, from current robot pose |
| `experiment.generation.transform_first_robot_pose` | Config | If True, use first robot pose for each subtask segment interpolation (affects pose initialization) |

## 3. Subtask Boundary Randomization

| Parameter | Where | Description |
|-----------|-------|-------------|
| `subtask_term_offset_range` | Task spec per-subtask | 2-tuple (min, max) random offset added to subtask boundary detection (e.g., `(5, 10)`) — creates trajectory start/end variance |

## 4. Trajectory Interpolation Parameters

| Parameter | Where | Description |
|-----------|-------|-------------|
| `num_interpolation_steps` | Task spec per-subtask | Number of intermediate poses during linear interpolation between subtasks (e.g., 5) — smoother motion = more steps |
| `num_fixed_steps` | Task spec per-subtask | Additional steps with constant target pose at subtask start (e.g., 0) — allows robot settling time |

## 5. Source Demonstration Selection Strategy

### Main Strategy Parameters

| Parameter | Where | Description |
|-----------|-------|-------------|
| `selection_strategy` | Task spec per-subtask | How to pick source demo segment: `"random"`, `"nearest_neighbor_object"`, `"nearest_neighbor_robot_distance"` |
| `selection_strategy_kwargs` | Task spec per-subtask | Strategy-specific parameters (dict) |

### For `nearest_neighbor_object` Strategy

| Parameter | Where | Description |
|-----------|-------|-------------|
| `pos_weight` | Selection strategy kwargs | Weight on position distance (default 1.0) |
| `rot_weight` | Selection strategy kwargs | Weight on rotation distance (default 1.0) |
| `nn_k` | Selection strategy kwargs | Pick uniformly from top-k nearest neighbors (default 3) — higher k = more variance |

### For `nearest_neighbor_robot_distance` Strategy

| Parameter | Where | Description |
|-----------|-------|-------------|
| `pos_weight` | Selection strategy kwargs | Weight on position distance (default 1.0) |
| `rot_weight` | Selection strategy kwargs | Weight on rotation distance (default 1.0) |
| `nn_k` | Selection strategy kwargs | Pick uniformly from top-k nearest neighbors (default 3) |

### Global Selection Parameters

| Parameter | Where | Description |
|-----------|-------|-------------|
| `experiment.generation.select_src_per_subtask` | Config | If True, select different source demo for each subtask; if False, use same one for whole episode |

## 6. Number of Trials & Data Collection

| Parameter | Where | Description |
|-----------|-------|-------------|
| `num_trials` | Config or `run_mimicgen_generate.py` | Number of generation attempts (e.g., 50) |
| `experiment.generation.guarantee` | Config | If True, keep trying until `num_trials` **successes**; if False, stop after `num_trials` **attempts** |
| `experiment.generation.keep_failed` | Config | Whether to save failed trajectories (boolean) |
| `experiment.generation.max_num_failures` | Config | Maximum number of failed demos to keep |

## 7. Source Data Selection

| Parameter | Where | Description |
|-----------|-------|-------------|
| `experiment.source.dataset_path` | Config | Path to source HDF5 demonstrations |
| `experiment.source.n` | Config | Use only first N trajectories from source (None = use all) |
| `experiment.source.start` | Config | Skip first N trajectories, then start |
| `experiment.source.filter_key` | Config or `run_mimicgen_generate.py` | Filter key to select subset of trajectories |

## 8. Task & Environment Configuration

| Parameter | Where | Description |
|-----------|-------|-------------|
| `task_name` | Config / `run_mimicgen_generate.py` | Task name (e.g., `"square"`, `"coffee"`, `"threading"`) |
| `env_interface_name` | Config / `run_mimicgen_generate.py` | Environment interface (e.g., `"MG_Square"`, `"MG_Coffee"`) |
| `experiment.task.env_meta_update_kwargs` | Config | Override environment constructor arguments (object positions, task parameters, etc.) |
| `experiment.task.robot` | Config | Override robot type if different from source |
| `experiment.task.gripper` | Config | Override gripper type if different from source |

## 9. Object Pose Transformation

| Parameter | Where | Description |
|-----------|-------|-------------|
| Object reference frame | Task spec per-subtask: `object_ref` | Which object frame each subtask is relative to (e.g., `"square_nut"`, `"square_peg"`) — implicitly controls pose transformation |
| Current object pose | Environment sampling | Object is placed at different locations each reset — provides scene variation |

## 10. Subtask Termination Signals

| Parameter | Where | Description |
|-----------|-------|-------------|
| `subtask_term_signal` | Task spec per-subtask | Which environment signal detects subtask completion (e.g., `"grasp"`, `"insert_1"`) — used to split source demos |

## 11. Logging & Sampling

| Parameter | Where | Description |
|-----------|-------|-------------|
| `experiment.log_every_n_attempts` | Config | Logging frequency during generation |
| `experiment.render_video` | Config | Whether to render videos of selected trajectories |
| `experiment.num_demo_to_render` | Config | How many successful demos to render to video |
| `experiment.num_fail_demo_to_render` | Config | How many failed demos to render to video |

---

## Key Insights for Steering Variance

### High Variance Trajectories
- Increase `action_noise`
- Increase `subtask_term_offset_range`
- Use `nearest_neighbor_*` with higher `nn_k`
- Set `select_src_per_subtask=True`
- Set `interpolate_from_last_target_pose=True`

### Low Variance Trajectories
- Set `action_noise=0`
- Set `subtask_term_offset_range=(0,0)`
- Use `selection_strategy="random"` or low `nn_k=1`
- Set `select_src_per_subtask=False`
- Set `interpolate_from_last_target_pose=False`

### Constrain Object Poses
The MimicGen object pose transformation (`transform_source_data_segment_using_object_pose`) automatically handles pose adaptation. To constrain initial object poses:
- Modify `experiment.task.env_meta_update_kwargs` in the config
- Or configure robosuite environment's randomization parameters directly

### Constrain Robot Poses
- Use `interpolate_from_last_target_pose` to control interpolation behavior between subtasks
- Use `transform_first_robot_pose` to control whether first robot pose is included in each subtask segment
- Adjust `num_interpolation_steps` and `num_fixed_steps` to control robot settling behavior

---

## Implementation Details

### Where to Specify Parameters

1. **Task Spec Parameters** (`subtask_term_offset_range`, `action_noise`, `num_interpolation_steps`, etc.):
   - Defined in config task_spec dictionaries (e.g., `mimicgen/configs/robosuite.py`)
   - Can be modified via `config.task.task_spec.subtask_N = {dict}`

2. **Generation Parameters** (`num_trials`, `select_src_per_subtask`, etc.):
   - Set in `config.experiment.generation.*`

3. **Command-line Parameters** (when running `run_mimicgen_generate.py`):
   - `--num_trials`: Number of generation attempts
   - `--guarantee`: Keep trying until num_trials successes
   - `--filter_key`: Filter source demonstrations
   - `--seed_hdf5`: Path to seed demonstration

### Object-Centric Pose Transformation

MimicGen uses object-centric pose transformation to generalize trajectories to new object poses:
1. Compute end-effector poses relative to source object frame
2. Apply relative poses to current object frame in new scene
3. This preserves manipulation semantics while adapting to scene variation

This transformation is automatically applied when `object_ref` is specified for a subtask.


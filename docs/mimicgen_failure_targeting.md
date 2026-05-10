# MimicGen Failure-Targeting Experiment

## Hypothesis

The behavior graph identifies **where the policy breaks** — nodes with low V-values
and high incoming failure probability.  Targeting MimicGen generation toward the states
that precede those failures should fill coverage gaps and improve the retrained policy.

Two targeting mechanisms combine:

1. **Initial-condition (IC) targeting**: collect the initial object poses of failed
   episodes, cluster them, and constrain each generation job to start near a cluster
   center.
2. **Intermediate-state targeting**: find behavior-graph nodes that frequently
   transition into failure-prone nodes; collect the robot/object states at those
   nodes; use the same clustering and constraint machinery.

Seed selection uses `NearFailurePathHeuristic` — seeds drawn from success paths that
traversed high-failure-risk nodes — so the generator has source material appropriate
for the target region.

---

## Running the Experiment

### Prerequisites

An existing `run_clustering` result is required (or you can run the full pipeline
from scratch).  The experiment config assumes the Apr 23 baseline:
`train_date=apr23_mimicgen_pipeline_v2`.

### Full pipeline from scratch

```bash
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_failure_targeting_may10
```

### Against an existing clustering result

```bash
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_failure_targeting_may10 \
  run_dir=<path_to_existing_run_dir> \
  steps=[mimicgen_failure_targeting,mimicgen_random,eval_baseline]
```

### Run only the failure-targeting arm

```bash
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_failure_targeting_may10 \
  run_dir=<path_to_existing_run_dir> \
  steps=[mimicgen_failure_targeting]
```

### Run just the analysis step (inspect clusters before generating)

```bash
python -m policy_doctor.scripts.run_pipeline \
  data_source=mimicgen_square \
  experiment=mimicgen_square_failure_targeting_may10 \
  run_dir=<existing_run_dir> \
  steps=[analyze_failure_states]
```

The step writes `<run_dir>/mimicgen_failure_targeting/analyze_failure_states/result.json`
with cluster centers and per-cluster IC constraint suggestions.  Inspect this before
committing to a full generation run.

---

## Pipeline Structure

The `mimicgen_failure_targeting` arm runs these sub-steps sequentially:

```
analyze_failure_states
    ↓  (writes failure_clusters.json with per-cluster IC/subtask constraints)
select_mimicgen_seed
    ↓  (NearFailurePathHeuristic, per-cluster budget, per-seed IC constraints)
generate_mimicgen_demos
    ↓  (IC constraints + optional subtask rejection per seed)
train_on_combined_data
    ↓
eval_mimicgen_combined
```

Run directory layout:

```
<run_dir>/
  run_clustering/                         # shared upstream (must already exist or be run first)
  mimicgen_failure_targeting/
    analyze_failure_states/
      result.json                         # cluster info, pre-failure edges, suggested constraints
      done
    select_mimicgen_seed/
      seed.hdf5                           # all selected seeds (one demo per seed)
      result.json                         # per_seed_object_pose_ranges, per_seed_subtask_constraints
      done
    generate_mimicgen_demos/
      output/
        seed_0/  seed_1/  ...             # per-seed generation outputs
        demo.hdf5                         # merged successful demos
      done
    train_on_combined_data/
    eval_mimicgen_combined/
```

---

## Configuration Reference

All failure-analysis knobs live under `mimicgen_datagen.failure_analysis.*` in the
experiment YAML.  They are off by default (`enabled: false`) and activated
automatically by the arm's `cfg_overrides`.

### Graph Analysis

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Master switch.  Set to `true` to activate failure analysis. |
| `value_threshold` | `0.0` | V-value below which a node is considered failure-prone.  Nodes with V < threshold are "at-risk targets". |
| `min_transition_prob` | `0.05` | Minimum edge probability to count as a pre-failure transition.  Increase to focus on frequent failures only. |

**How V-values work**: `BehaviorGraph.compute_values()` solves linear Bellman
equations with `reward_success=1, reward_failure=-1`.  A node with V close to -1
almost always leads to failure; V close to +1 almost always leads to success.

### State Clustering

| Key | Default | Description |
|-----|---------|-------------|
| `targeting_mode` | `both` | Which states to cluster: `initial_state` (t=0 of failed episodes only), `intermediate_state` (states at pre-failure nodes), or `both`. |
| `n_clusters` | `5` | Number of target-state clusters.  Total seeds = `n_clusters × budget_per_cluster`. |
| `budget_per_cluster` | `4` | Seeds allocated per cluster. |
| `cluster_target_mode` | `centroid` | How to pick the target pose from a cluster: `centroid` (cluster mean) or `sample` (random cluster member). |
| `state_schema` | `null` | Per-object → qpos-index mapping for state parsing.  `null` uses the Square-task default (see below). |

### IC Constraint Slack

| Key | Default | Description |
|-----|---------|-------------|
| `slack_x` | `0.03` | ± offset around cluster center in world-frame x (metres). |
| `slack_y` | `0.03` | ± offset around cluster center in world-frame y (metres). |
| `slack_z_rot` | `0.5` | ± offset in z rotation (radians, ≈ ±29°). |

The IC constraint is applied relative to each seed's own initial pose (which was
selected near the cluster center), so tighter slack = generation stays closer to the
cluster center.

### Subtask Constraints

| Key | Default | Description |
|-----|---------|-------------|
| `subtask_constraint_idx` | `null` | Integer index of the subtask boundary to enforce (0-indexed).  `null` disables subtask constraints entirely. |
| `subtask_constraint_slack` | `1.5` | Multiplier on IC slack for the subtask check.  E.g., with `slack_x=0.03` and `subtask_constraint_slack=1.5`, the subtask window is ±0.045 m. |

**How subtask constraints work**: after subtask `subtask_constraint_idx` finishes
executing (but before data is recorded), the generation script checks whether each
constrained object's pose falls within the derived window centred on the cluster's
target pose.  Trials that miss the window are **rejected** (counted as failures).
The budget loop compensates by running more trials.

**Rejection rate warning**: tight constraints (`slack_multiplier` close to 1.0) can
push rejection rates above 90%, multiplying the required number of trials many-fold.
Start with the default 1.5× and inspect `stats.json` after a test run.

### Global subtask override (without failure analysis)

You can also set a fixed subtask constraint for all seeds from the experiment YAML,
bypassing failure analysis:

```yaml
mimicgen_datagen:
  fix_initial_object_poses: true
  subtask_constraints:
    "0":                    # after subtask 0 completes
      nut:
        x: [-0.05, 0.05]
        y: [-0.05, 0.05]
        z_rot: [-0.8, 0.8]
```

Per-seed constraints from `AnalyzeFailureStatesStep` take precedence over this global
setting when both are present.

---

## State Schema for Other Tasks

`state_schema` maps object names to qpos indices in the raw sim-state vector stored
in the rollout HDF5 (`data/demo_N/states`).  The Square-task default:

```yaml
failure_analysis:
  state_schema:
    nut:
      x_idx: 10      # world-frame x in qpos
      y_idx: 11      # world-frame y in qpos
      qw_idx: 13     # quaternion w (MuJoCo wxyz convention)
      qx_idx: 14
      qy_idx: 15
      qz_idx: 16
```

z_rot is derived from the quaternion as `atan2(2(qw·qz + qx·qy), 1 − 2(qy² + qz²))`.

To adapt to a different task, inspect the state vector variance across episodes
to locate the object free-joint entries:

```python
import h5py, numpy as np
f = h5py.File("rollouts.hdf5", "r")
states = np.stack([f["data"][k]["states"][0] for k in f["data"].keys()])
stds = states.std(axis=0)
# Components with high std (> 0.05) are object position/orientation entries
print(np.argwhere(stds > 0.05).flatten())
```

---

## Key Source Files

| File | Role |
|------|------|
| `policy_doctor/mimicgen/failure_targeting.py` | Pure analysis functions (graph→clusters→constraint dicts) |
| `policy_doctor/mimicgen/heuristics.py` | `NearFailurePathHeuristic`, `build_heuristic("near_failure")` |
| `policy_doctor/curation_pipeline/steps/analyze_failure_states.py` | Pipeline step — runs analysis, writes per-cluster result |
| `policy_doctor/curation_pipeline/steps/select_mimicgen_seed.py` | Extended to do per-cluster seed selection when analysis step is done |
| `policy_doctor/curation_pipeline/steps/generate_mimicgen_demos.py` | Extended to pass per-seed IC/subtask constraints to subprocess |
| `scripts/run_mimicgen_generate.py` | `--subtask_constraints` arg + `ConstrainedDataGenerator` wrapper |
| `policy_doctor/curation_pipeline/steps/mimicgen_arm.py` | `MimicgenFailureTargetingArmStep` |
| `policy_doctor/configs/experiment/mimicgen_square_failure_targeting_may10.yaml` | Experiment config |

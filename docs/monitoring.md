# Runtime Behavior Monitor

The monitoring pipeline assigns each timestep of a robot policy rollout to a node in the **behavior graph** — in real time, as the policy runs, or in batch over a saved rollout or demonstration.

It works by computing an **influence embedding** for each (obs, action) pair (a gradient-based fingerprint that captures which training demonstrations the current behavior resembles), then finding the nearest cluster centroid in that embedding space.

---

## Architecture overview

The pipeline has a hard split between offline (one-time) preparation and online (per-timestep) inference.

```
OFFLINE — one time, slow, requires cupid env
─────────────────────────────────────────────
  Policy checkpoint
    └─ InfEmbed fit  →  infembed_fit.pt         (Arnoldi eigenvectors)
    └─ Rollout embeddings  →  infembed_embeddings.npz
    └─ Clustering  →  cluster_labels.npy
                   →  clustering_models.pkl      (normalizer, prescaler, UMAP, KMeans)
    └─ Behavior graph  →  BehaviorGraph

ONLINE — per timestep, ~10–100 ms, requires cupid env
──────────────────────────────────────────────────────
  obs (To, Do) + action (Ta, Da)
    └─ StreamScorer.embed()  →  embedding (proj_dim,)   ← one forward+backward pass
    └─ StreamScorer.score()  →  influence_scores (N_train,)
    └─ GraphAssigner.assign()  →  AssignmentResult
         cluster_id, node_id, node_name, distance
```

**Environment requirements:**

| Component | Conda env |
|---|---|
| `InfEmbedStreamScorer` | `cupid_torch2` (needs `torch.func` for Arnoldi gradient projection) |
| `TRAKStreamScorer` | `cupid_torch2` |
| `FittedModelAssigner`, `NearestCentroidAssigner` | `policy_doctor` (pure numpy + sklearn) |
| `StreamMonitor`, `TrajectoryClassifier`, `MonitoredPolicy` | `cupid_torch2` (scorer is the bottleneck) |
| All unit tests | `policy_doctor` (mock scorer) |

`infembed` is installed as a package (`pip install -e third_party/cupid/third_party/infembed`) in `cupid_torch2`. The `cupid` env (torch 1.12) lacks `torch.func` and cannot run `InfEmbedStreamScorer`.

---

## Components

### `StreamScorer`

Abstract base. Two concrete implementations:

**`InfEmbedStreamScorer`** (recommended)
- `embed(batch)` → `(proj_dim,)`: one forward+backward pass through the policy, projected via the saved Arnoldi eigenvectors from `infembed_fit.pt`. `proj_dim` is typically 100.
- `score(batch)` → `(N_demo,)`: dot product of the embedding against `demo_embeddings` from `infembed_embeddings.npz`. Higher = more influential.
- Property `rollout_embeddings` → `(N_rollout, proj_dim)`: pre-computed training rollout embeddings, used to build a `NearestCentroidAssigner`.

**`TRAKStreamScorer`**
- `embed(batch)` → `(proj_dim,)`: gradient projected via the same JL random matrix used during featurization (reconstructed from seed). `proj_dim` is typically 2048.
- `score(batch)` → `(N_train,)`: dot product against `features.mmap`.

### `GraphAssigner`

Abstract base. Two concrete implementations:

**`FittedModelAssigner`** (preferred when available)
Applies the exact same pipeline used during clustering: `normalizer → prescaler → UMAP → KMeans.predict`. Requires `clustering_models.pkl` in the clustering result directory, produced by `RunClusteringStep` (from the version that added model persistence).

```python
assigner = FittedModelAssigner.from_paths(clustering_dir, graph)
```

**`NearestCentroidAssigner`** (fallback)
Computes per-cluster centroids in the raw InfEmbed embedding space from `rollout_embeddings + cluster_labels`. Assigns by nearest L2 centroid. No saved UMAP model needed. Approximate — works best when the raw embedding space geometry tracks the UMAP geometry.

```python
assigner = NearestCentroidAssigner.from_paths(
    rollout_embeddings=scorer.rollout_embeddings,
    clustering_dir=clustering_dir,
    graph=graph,
)
```

Both assigners accept a `cluster_id_to_node_id` dict for graphs where cluster IDs were remapped by degree-one pruning.

### `StreamMonitor`

Ties scorer + assigner together with per-stage timing:

```python
monitor = StreamMonitor(scorer=scorer, assigner=assigner)
result = monitor.process_sample(obs, action)  # obs: (To,Do), action: (Ta,Da)

result.embedding          # (proj_dim,) float32
result.influence_scores   # (N_train,) float32
result.assignment         # AssignmentResult or None
result.timing_ms          # dict: gradient_project_ms, score_ms, assign_ms, total_ms
```

Use `process_sample_embed_only()` to skip full influence scoring when only the graph assignment is needed (faster — no dot product against all training samples).

### `TrajectoryClassifier`

Higher-level wrapper that handles windowing and input transforms. The single `from_checkpoint()` classmethod reads `abs_action`, `rotation_rep`, `obs_keys`, `n_obs_steps`, `n_action_steps` from the checkpoint config automatically.

```python
classifier = TrajectoryClassifier.from_checkpoint(
    checkpoint="path/to/latest.ckpt",
    infembed_fit_path="path/to/infembed_fit.pt",
    infembed_embeddings_path="path/to/infembed_embeddings.npz",
    clustering_dir="path/to/clustering/<slug>",
    mode="rollout",   # or "demo"
    device="cuda:0",
)
```

Three classification entry points:

| Method | Input | When to use |
|---|---|---|
| `classify_sample(obs, action)` | pre-windowed `(To,Do)`, `(Ta,Da)` | live rollout, one step at a time |
| `classify_episode_from_pkl(df)` | episode DataFrame from `eval_save_episodes.py` | offline analysis of saved rollouts |
| `classify_demo_from_hdf5(demo_group)` | open h5py group at `data/demo_X` | offline analysis of demonstration HDF5 |
| `classify_sequence(obs_seq, action_seq)` | raw `(T,Do)` and `(T,Da)` | general purpose; builds windows internally |

### `MonitoredPolicy`

Wraps any `BaseLowdimPolicy` and intercepts `predict_action()`. Classifies the (obs, predicted_action) pair after each policy call and accumulates results in `episode_results`. Compatible with `RobomimicLowdimRunner` (the runner calls `policy.reset()` between episodes, which advances the episode counter).

```python
monitored = MonitoredPolicy(policy=policy, classifier=classifier, verbose=False)
# Pass `monitored` to env_runner.run() instead of the raw policy.
# After the run:
for entry in monitored.episode_results:
    print(entry["episode"], entry["timestep"], entry["node_name"])
```

---

## Input modes: rollout vs demo

The `mode` parameter of `TrajectoryClassifier` controls whether action transforms are applied.

**`mode="rollout"`** (default)

Data comes from the env or from a pkl saved by `eval_save_episodes.py`. The policy already outputs actions in `rotation_6d` format (it unnormalizes internally). No transforms needed — feed obs and action directly to the scorer.

**`mode="demo"`**

Data comes from an HDF5 demonstration file. When `abs_action=True` in the training config, actions are stored in raw `axis_angle` format and must be converted to `rotation_6d` before the scorer can compute gradients. The `TrajectoryClassifier` applies this transform automatically when `mode="demo"`:

```
raw HDF5 action (T, 7)     →    rotation_transformer.forward(rot)    →    (T, 10)
  pos(3) + rot_aa(3) + grip(1)     axis_angle → rotation_6d           pos(3) + rot_6d(6) + grip(1)
```

For dual-arm (14-dim): `(T, 14)` → `(T, 20)` via the same logic applied per arm.

When `abs_action=False` (delta actions), no rotation transform is applied even in `mode="demo"`.

`from_checkpoint()` reads `abs_action` from the checkpoint config, so the correct mode is set automatically.

---

## Quick start

### Offline: classify a saved rollout

```bash
# cupid conda env, from the project root
python scripts/monitor_offline.py \
    --episode <output_dir>/episodes/ep0000_succ.pkl \
    --checkpoint <train_dir>/checkpoints/latest.ckpt \
    --infembed_fit <eval_dir>/infembed_fit.pt \
    --infembed_npz <eval_dir>/infembed_embeddings.npz \
    --clustering_dir <configs_root>/clustering/<slug> \
    --output assignments.csv
```

### Offline: classify a demonstration from HDF5

```bash
python scripts/monitor_offline.py \
    --hdf5 <path>/dataset.hdf5 --demo demo_0 \
    --checkpoint <train_dir>/checkpoints/latest.ckpt \
    --infembed_fit <eval_dir>/infembed_fit.pt \
    --infembed_npz <eval_dir>/infembed_embeddings.npz \
    --clustering_dir <configs_root>/clustering/<slug>
```

### Online: live eval with monitoring

```bash
# Run from third_party/cupid/ — diffusion_policy must be on PYTHONPATH.
# cupid_torch2 env required (torch.func).
conda activate cupid_torch2
cd third_party/cupid
python ../../scripts/monitor_online.py \
    --output_dir /tmp/monitor_run \
    --train_dir <train_dir> \
    --train_ckpt best \
    --infembed_fit <eval_dir>/infembed_fit.pt \
    --infembed_npz <eval_dir>/infembed_embeddings.npz \
    --clustering_dir <configs_root>/clustering/<slug> \
    --num_episodes 10 \
    --verbose
```

If the clustering was done at the "rollout" (window) level — which is the default — pass `--episodes_dir` so the assigner can compute window-mean embeddings:

```bash
    --episodes_dir <eval_dir>/episodes    # directory containing metadata.yaml
```

Produces `monitor_assignments.csv` (one row per policy call) and `eval_log.json` alongside the standard episode output.

### Programmatic API

```python
from policy_doctor.monitoring import TrajectoryClassifier, MonitoredPolicy

# Build classifier once (expensive — loads model + artifacts)
classifier = TrajectoryClassifier.from_checkpoint(
    checkpoint="checkpoints/latest.ckpt",
    infembed_fit_path="infembed_fit.pt",
    infembed_embeddings_path="infembed_embeddings.npz",
    clustering_dir="clustering/my_run",
    mode="rollout",
    device="cuda:0",
)

# Single sample (live control loop)
result = classifier.classify_sample(obs_window, action_window)
print(f"{result.assignment.node_name}  ({result.timing_ms['total_ms']:.1f} ms)")

# Full HDF5 demo
import h5py
with h5py.File("dataset.hdf5", "r") as f:
    results = classifier.classify_demo_from_hdf5(f["data/demo_0"])

# Wrap a policy for live eval
monitored = MonitoredPolicy(policy=policy, classifier=classifier, verbose=True)
env_runner.run(monitored)
```

---

## Clustering pipeline: model persistence

`RunClusteringStep` now saves `clustering_models.pkl` alongside the standard clustering artifacts. This enables `FittedModelAssigner` to apply the exact same normalizer→prescaler→UMAP→KMeans pipeline that was used during clustering, giving geometrically correct assignments for new data points.

**Clustering result directory layout:**

```
<configs_root>/clustering/<slug>/
├── manifest.yaml            # algorithm, scaling, influence_source, level, …
├── cluster_labels.npy       # (N_rollout,) int32
├── metadata.json            # per-sample rollout_idx, timestep, window info
└── clustering_models.pkl    # ClusteringModels (new — required for FittedModelAssigner)
```

`clustering_models.pkl` is loaded with `joblib` and contains a `ClusteringModels` dataclass:

```python
@dataclass
class ClusteringModels:
    normalizer: Optional[Any]     # sklearn scaler or None
    normalizer_method: str        # "none" | "standard" | "minmax" | "robust"
    prescaler: Optional[Any]      # sklearn scaler or None
    prescaler_method: str
    reducer: Optional[Any]        # fitted UMAP / PCA model
    reducer_method: str           # "umap" | "pca"
    kmeans: Optional[Any]         # fitted KMeans model
```

Clustering runs that predate model saving can still be used with `NearestCentroidAssigner` (the fallback). `TrajectoryClassifier.from_checkpoint` and both scripts automatically fall back to `NearestCentroidAssigner` when `clustering_models.pkl` is missing.

---

## Performance

Run `python -m policy_doctor.monitoring.benchmark` for timing and storage numbers on your hardware.

Observed end-to-end timing running `monitor_online.py` with the jan28 transport_mh_0 policy (17M params, 100-dim InfEmbed, k=20 clustering) on an RTX 3090:

| Stage | Observed |
|---|---|
| `scorer.embed()` (forward + backward + project) | ~1950 ms |
| `scorer.score()` (100-dim dot product) | <1 ms |
| `assigner.assign()` (nearest centroid, k=20) | <0.1 ms |
| **Total per timestep** | **~2000 ms** |

This is too slow for hard real-time control (500 Hz). At 3 actions/step (n_action_steps=16, executed subset ~3–8) the monitor runs once per action chunk, which puts effective monitoring rate at ~1 classification per 1–3 seconds of execution — acceptable for behavior logging and offline analysis, but not for within-chunk intervention.

To reduce latency:
- Use `process_sample_embed_only()` to skip scoring (saves <1 ms for InfEmbed — scoring is not the bottleneck here).
- Profile the backward pass: the Arnoldi projection applies 100 dot products against the full parameter gradient (17M × 100 = 1.7B ops). Reducing `proj_dim` at fit time would directly reduce embed cost.
- `torch.compile` the policy + gradient computation (available in `cupid_torch2`, requires warm-up).

Storage requirements (typical):

| Artifact | Size |
|---|---|
| `infembed_fit.pt` | ~10–50 MB |
| `infembed_embeddings.npz` | ~5–20 MB |
| `clustering_models.pkl` | ~10–100 MB (UMAP model dominates) |
| Policy checkpoint | ~50–200 MB |
| `features.mmap` (TRAK) | ~500 MB–2 GB |

---

## Testing

Unit tests run in the `policy_doctor` env without GPU or real checkpoints. Integration tests require `cupid_torch2` and real jan28 artifacts on `/mnt/ssdB/`.

```bash
# Unit tests (policy_doctor env, no GPU needed)
conda activate policy_doctor
python run_tests.py --suite policy_doctor

# Integration tests (cupid_torch2 env, jan28 artifacts required)
conda activate cupid_torch2
python run_tests.py --suite cupid
```

**Test coverage by file:**

| Test file | Env | What's covered |
|---|---|---|
| `test_stream_monitor.py` | `policy_doctor` | `StreamMonitor` with mock scorer: result shape, timing keys, embed-only path |
| `test_graph_assigner.py` | `policy_doctor` | `NearestCentroidAssigner`: correct cluster assignment, noise exclusion, cluster→node mapping |
| `test_fitted_model_assigner.py` | `policy_doctor` | `FittedModelAssigner`: save/load roundtrip, consistency with sklearn predict |
| `test_trajectory_classifier.py` | `policy_doctor` | `TrajectoryClassifier`: window construction, timestep indices, end-of-episode padding, rotation transform passthrough and application (single-arm and dual-arm), pkl/HDF5 classification paths |
| `test_monitored_policy.py` | `policy_doctor` | `MonitoredPolicy`: action dict passthrough, result accumulation, `action_pred` key preference, batch B>1, `reset()` episode counting, `__getattr__` delegation |
| `test_pipeline_integration.py` | `policy_doctor` | End-to-end: `fit_*` → `save_clustering_models` → `load` → `FittedModelAssigner` → `StreamMonitor` → `TrajectoryClassifier` → `MonitoredPolicy` |
| `tests/integration/test_monitor_integration.py` | `cupid_torch2` | Real jan28 checkpoint + artifacts: `InfEmbedStreamScorer`, `NearestCentroidAssigner`, `StreamMonitor`, `TrajectoryClassifier.from_checkpoint` — all 20 tests |

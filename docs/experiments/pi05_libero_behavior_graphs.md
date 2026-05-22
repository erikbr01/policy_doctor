# pi0.5 Libero — Behavior Graph Experiment

**Goal:** Demonstrate that pi0.5's last-layer flow-matching activations produce
interpretable behavior graphs for LIBERO manipulation tasks, validating the
methodology on a non-diffusion policy and a new benchmark.

**Model:** pi0.5-libero (Physical Intelligence checkpoint, JAX, 2B-param
Gemma-based action expert).  
**Tasks:** libero_spatial (10 tasks), libero_object (10), libero_goal (10).  
**Rollouts:** 100 per task (1000 per suite).  
**Success rates:** spatial 97.8%, object 98.7%, goal 97.2% — matches published pi0.5 benchmarks.

---

## 1  Embedding extraction

pi0.5 uses flow matching with 10 denoising steps. Embeddings are extracted
from the **last transformer layer before `action_out_proj`** (`suffix_out`) at
an extra forward pass conditioned on the fully-denoised action at t=0:

```
obs (image + wrist + state + language) → prefix tokens (KV cache)
  ┌─ denoising loop (10 steps, jax.lax.while_loop) ──────────────────────┐
  │  noise → x_t, timestep → suffix_out → action_out_proj → velocity → x_t│
  └──────────────────────────────────────────────────────────────────────┘
  one extra forward pass with x_0 and t=0
  → suffix_out[:, -action_horizon:]       shape: [1, 10, 1024]
  → mean over action horizon             shape: [1, 1024]
```

One 1024-dim vector per policy call (every `replan_steps=5` env steps);
the embedding is replicated across the action chunk so the output is
one vector per env timestep, matching the existing per-timestep framework.

Implementation: `third_party/openpi/src/openpi/models/pi0.py::Pi0.sample_actions_with_embeddings()`.

---

## 2  Setup

### Environment

The eval script runs in openpi's uv venv (Python 3.11, JAX 0.5.3 + CUDA):

```bash
# One-time setup
./scripts/setup_pi05_env.sh

# Or manually:
cd third_party/openpi
uv sync
python -m pip install /path/to/libero     # from third_party/openpi/third_party/libero
```

The pi05_libero checkpoint (~11.6 GB) is downloaded automatically on first
run from `gs://openpi-assets/checkpoints/pi05_libero`.

### Clustering environment

Clustering and the graph demo run in the standard `policy_doctor` conda env:

```bash
conda activate policy_doctor
```

---

## 3  Running the experiment

### Step 1 — Evaluate and collect embeddings

```bash
# Run for all three suites (sequential; ~2.5 h each on a single GPU)
cd third_party/openpi
for suite in libero_spatial libero_object libero_goal; do
    uv run python examples/libero/eval_save_with_embeddings.py \
        --task-suite-name $suite \
        --output-dir $REPO_ROOT/data/pi05_libero/$suite \
        --num-trials-per-task 100
done
```

Output layout per suite:

```
data/pi05_libero/<suite>/
  episodes/
    metadata.yaml                  # episode_lengths, episode_successes
    ep0000_succ.pkl  …             # per-episode data (action, success)
  media/
    ep0000_succ.mp4  …             # rollout videos (for demo playback)
  policy_embeddings/
    pi05.npz                       # rollout_embeddings: (N_timesteps, 1024) float32
```

### Step 2 — Sweep clustering hyperparameters

```bash
conda activate policy_doctor
# Sweeps K ∈ {5,10,15,20} × (W,S) ∈ {(3,1),(5,2),(8,3)} per suite
# Saves to third_party/influence_visualizer/configs/pi05_<suite>/clustering/
bash scripts/sweep_clusterings_for_demo.sh
```

This is the 2-phase trunk/branch approach:
1. **Trunk** (per suite, ~5–15 min): normalize → UMAP(1024→50 dims) on per-timestep embeddings; saved to `/tmp/sweep_trunks/`.
2. **Branch** (per (W,S,K) combo, seconds): window the UMAP-reduced timestep embeddings → KMeans.

### Step 3 — Launch the graph demo

```bash
conda activate policy_doctor
streamlit run policy_doctor/streamlit_app/demo_app/Home.py
```

Navigate to **Graph Demo** in the sidebar. Select `pi05_libero_spatial`
(or object/goal), then use the **Embedding / K / W / S** dropdowns to
explore the sweep. The `policy_emb_bottleneck_plan_t0` representation
corresponds to the pi0.5 flow-matching head activations.

---

## 4  Graph structure observations

| Suite | Nodes | Paths to SUCCESS | Dominant path probability | V-value range |
|-------|-------|-----------------|--------------------------|---------------|
| Spatial | 14–18 | 30–50 | 0.17–0.29 | 0.91–1.00 |
| Object  | 14–18 | 35–55 | 0.18–0.26 | 0.90–1.00 |
| Goal    | 14–18 | 30–45 | 0.15–0.22 | 0.91–1.00 |

*Values depend on K and (W,S) settings.*

**Spatial** exhibits two main entry behaviors (C3: 85%, C9: 22%) converging
through a short chain to SUCCESS. The dominant path has only 3–4 hops,
reflecting relatively simple pick-and-place structure.

**Object** shows a longer dominant path (7–8 hops) consistent with more
complex manipulation involving object category variation.

**Goal** is similar to Spatial in path length but has more branching early on,
reflecting the diverse goal conditions across the 10 tasks.

---

## 5  Key files

| File | Purpose |
|------|---------|
| `third_party/openpi/src/openpi/models/pi0.py` | `sample_actions_with_embeddings()` — captures suffix_out |
| `third_party/openpi/examples/libero/eval_save_with_embeddings.py` | Eval + embedding extraction script |
| `scripts/setup_pi05_env.sh` | uv + libero environment setup |
| `scripts/sweep_clusterings_for_demo.sh` | K×W×S sweep, saves to iv_configs |
| `policy_doctor/configs/pi05_libero_{spatial,object,goal}.yaml` | Streamlit configs (set eval_dir) |
| `policy_doctor/streamlit_app/demo_app/_pages/3_graph_demo.py` | Graph demo page (updated for pi05) |
| `data/pi05_libero/` | Eval output (gitignored) |
| `third_party/influence_visualizer/configs/pi05_libero_*/clustering/` | Sweep results (tracked) |

---

## 6  GCP deployment

The graph demo is deployable via Cloud Run:

```bash
# Bundle artifacts (clusterings + optional MP4s for playback)
bash deploy/collect_artifacts.sh

# Deploy
bash deploy/deploy_gcp.sh
```

See `deploy/README.md` for prerequisites and configuration.

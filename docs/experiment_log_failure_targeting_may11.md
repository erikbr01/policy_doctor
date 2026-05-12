# Failure-Targeting Experiment — Square D1, May 11 2026

Running log of thoughts, decisions, and progress. Newest entries at the top.
Append, don't rewrite — design changes should be visible as a sequence.

---

## 2026-05-11 — Design pivot: failure trajectories, not pre-failure nodes

**Status:** code change pending; experiment not yet launched.

The original `failure_targeting.py` collects intermediate states from **all**
trajectories that visited graph nodes which lead into failure (`pre_failure_nodes`).
Erik's correction: that pool mixes successful trajectories that happened to pass
through risky regions — by definition those are *not* states we need coverage
for, because the policy navigated them fine.

**Revised pipeline:**

1. Build behavior graph (unchanged).
2. Top-K most-probable START→FAILURE paths in the graph.
3. Match these paths to FAILURE episodes (run-length-collapsed cluster sequence equals path).
   These are "the failure trajectories."
4. Per node along the path, collect states from these failure trajectories at the
   timesteps they spent in that node.
   - START node → IC pool.
   - Any non-START, non-FAILURE node → intermediate "pass-through" pool.
5. Seed selection (`NearFailurePathHeuristic`) unchanged — pick successful rollouts that
   traverse near each target.
6. MimicGen generation: IC constraint + per-intermediate-node subtask constraints.

The graph still selects *which* failures matter and *what nodes they visit*. The states
themselves come from failure trajectories — no V-value smoothing over success+failure
mixtures.

**Knob decisions (locked in):**
- Top-K failure paths: **5**.
- Intermediate constraints/demo: **pluggable heuristic, default "node closest to FAILURE"**.
  Other heuristics to try later: all non-START nodes, every-other-node, V-value-weighted picks.
- Per-node clustering: **KMeans with silhouette sweep, k ∈ [2, 10]**, run independently per (path, node).

**Still open:**
- Mapping graph-node → MimicGen subtask boundary: do they align 1:1 for Square,
  or do we need a separate mapping? Need to inspect the MimicGen Square subtask layout
  and the behavior-graph cluster semantics.
- Seed budget allocation: 20 total seeds; split = ? 5 paths × 4 seeds-per-path is the simplest.
- Per-cluster slack: currently ±0.03m / ±0.5rad in IC slack. Reasonable starting point.

---

## 2026-05-11 — Design escalation: intermediate states as *generation* constraints, not rejection filters

Erik observation: MimicGen is fundamentally an IC-randomisation + trajectory-warping framework.
The current intermediate-state mechanism is a rejection filter on top of warp output. That's wasteful.
The cleaner approach: treat each intermediate target as a "forced midpoint" — chain two MimicGen
warps with the target as a boundary IC.

```
Demo D:
  Run 1: seed subtasks [0..N-1], sampled IC₀, terminate when object pose ≈ T (target).
  Run 2: seed subtasks [N..M-1], starting IC = the actual object pose after Run 1 (T or T+ε).
  Demo D = concat(Run1, Run2).
```

Tractability check:
- Splitting the seed by subtask index: already supported by `subtask_term_signals` parsing.
- Generating from an explicit object pose: MimicGen's bread-and-butter — every IC sample does this.
- Hitting T at the end of Run 1: easiest path is **iterative IC₀ sampling** until the endpoint is
  within slack of T. Slightly more sophisticated: analytically backsolve IC₀ given T (the per-subtask
  transform T_k is a rigid object-frame transform — invertible). Start with the iterative version.

Implementation order:
1. Failure-trajectory selection (graph: top-K paths → matched failure episodes) and per-node
   state collection from failure trajectories.
2. Per-node clustering with silhouette-k.
3. Single-target "node-closest-to-FAILURE" picker as the default intermediate heuristic.
4. Constraint-aware MimicGen wrapper:
   - Phase 1: keep the current `ConstrainedDataGenerator` rejection-style filter, but route the
     target from the new selection pipeline. Verifies end-to-end wiring without touching MimicGen.
   - Phase 2: replace with the two-stage chained warp. Validates the more invasive idea against
     baseline numbers from Phase 1.

Sequence Phase 1 → Phase 2 deliberately: if Phase 2 underperforms or breaks the warp, we still have
Phase 1's numbers to fall back on.

**Tolerance for constraint matching:** we don't need exact matches at intermediate targets.
Implications:
- Slack-box check (object pose at subtask-N boundary within ±slack of target) is the acceptance criterion.
- Iterative IC₀ sampling — try K candidates, accept first that lands within slack. No analytical warp inversion needed.
- Slack derived from the per-node cluster: `slack = α × stddev` with a min/max clamp, α default = 1.5.
  Tight where failure trajectories were tightly clustered at that node; loose where they were spread out.
- Graceful degradation: if N IC₀ samples all miss, widen slack by `widen_factor` (e.g. 2×) and try once more.
  If still no match, fall back to unconstrained Run 1.
- Config knobs: `failure_analysis.slack_alpha`, `failure_analysis.max_resamples`, `failure_analysis.widen_factor`.

---

## 2026-05-11 — Infrastructure setup notes

- `cupid_torch2` was an empty stub; deleted. `cupid` / `cupid_torch2` / `mimicgen`
  legacy env names retired. Canonical set: `policy_doctor` (analysis +
  orchestration + InfEmbed; CUDA torch 2.8.0) and `mimicgen_torch2` (sim).
- `policy_doctor` env build snags worth knowing about:
  - `diffusion-policy==0.0.0`, `r3m==0.0.0`, `pybullet-svl==3.1.6.4` not on PyPI
    (or need `g++`). Stripped from `environment_policy_doctor.yaml`.
  - `plotly` missing from initial yaml; added.
  - `infembed`'s `pyproject.toml` has `[tool.setuptools.packages.find] where = ["infembed"]`,
    so pip's editable `.pth` points one level too deep and `import infembed` fails.
    Patched `scripts/install_policy_doctor_env.sh` to rewrite the `.pth` after install.
- `apr23` baseline outputs aren't on this machine; full filesystem scan came up empty.
  Decided to train fresh with `train_date=may11`, `eval_date=may11`, `max_train_episodes=100`,
  using `data/source/.../demo_src_square_task_D1/demo.hdf5` (1000-demo source, capped at 100).
- Conda env subprocess names in `policy_doctor/configs/data_source/*.yaml` migrated
  from `cupid` → `policy_doctor`. `mimicgen_square.yaml` already used `mimicgen_torch2`.
- `eval_save_episodes` mode in `MimicgenLowdimRunner` rewritten to support `n_envs > 1`
  on the flywheel branch — still useful for downstream collection but not part of this branch.
- New configs for the may11 100-demo baseline run:
  - `policy_doctor/configs/square_mh_may11_mimicgen_pipeline.yaml` (task config)
  - `policy_doctor/configs/experiment/mimicgen_square_failure_targeting_may11.yaml`
    (experiment yaml; runs full pipeline from `train_baseline` through eval).
- `scripts/subsample_hdf5_demos.py` added — `square_d1_60.hdf5` and similar are first-N
  demos of the 1000-demo source, sorted by numeric demo suffix (HDF5 group order is insertion
  order, not lexicographic).

---

## 2026-05-11 — First training launch (failed)

Launched `train_baseline` step. Crashed immediately: `wandb.errors.UsageError: No API key
configured.` Erik ran `wandb login`; relaunching.

---

## 2026-05-11 — Second training launch (failed: torch.compile / triton mismatch)

Relaunched with the full upstream chain
(`train_baseline → eval_policies → train_attribution → finalize_attribution → compute_infembed → run_clustering`).
wandb authenticated cleanly. But train.py crashed with
`ImportError: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler'`.

Same incompatibility we already worked around for attribution (commit `decd820`):
torch.compile uses a triton API that's not present in this env's triton build.

Quick-disable of compile got training running, but Erik (correctly) pushed back: compile is a
~1.5-2× speedup, worth fixing properly.

---

## 2026-05-11 — Root cause of triton/compile error: torch 2.6 + triton 3.4 mismatch

Diagnosis:

| Env | torch | triton | compile worked? |
|---|---|---|---|
| `mimicgen_torch2` | **2.6.0** | 3.4.0 | ✗ (AttrsDescriptor missing) |
| `policy_doctor`   | 2.8.0 | 3.4.0 | ✓ |

`AttrsDescriptor` was a triton API used by torch 2.6's `_inductor`, removed in triton 3.2+.
Torch 2.8 doesn't need it. So mimicgen_torch2 just had the wrong torch — should always have been 2.8.0
(the setup script `setup_torch2_envs.sh` pins `TORCH2_VERSION="2.8.0"` but had `torchvision==0.13.1`
which silently failed to update everything else).

**Fix sequence:**
1. `pip uninstall -y torch` in mimicgen_torch2.
2. `pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0` — clean install (had to
   nuke `torch/` site-packages dir manually first; pip's previous attempt left things in a half-installed
   state with `ATen.h` missing).
3. After torch upgrade, `torchvision` (still 0.21.0+cu118) was binary-incompatible with torch 2.8
   (`RuntimeError: operator torchvision::nms does not exist`). Reinstalled
   `torchvision==0.23.0` from the cu128 index.
4. Cleaned the leftover `torchvision-0.22.1+cu118.dist-info` dir so `pip list` matches runtime.
5. Re-exported `environment_mimicgen_torch2.yaml` to capture the corrected state
   (torch 2.8.0+cu128 + torchvision 0.23.0+cu128 + triton 3.4.0).
6. policy_doctor still has the legacy `torchvision==0.13.1` (it imports and works, but should
   be aligned with mimicgen_torch2 — TODO bump after relaunch).

Both envs now compile cleanly on a trivial `@torch.compile` test.

**Smoke test of the env-runner with the new env + the `shared_memory=False` patch:**
- Constructed `MimicgenLowdimRunner(n_envs=2, n_test=2, max_steps=40)`.
- Created the Square_D1 vector env successfully (no shared_memory warning).
- Ran 40-step rollout with a zero-action dummy policy → returned `test/mean_score=0.0` (expected).
- Verified before relaunching that in-training rollouts will actually work.

**Compile re-enabled in the experiment yaml** (`baseline.compile: true`).

---

## 2026-05-11 — Fourth training launch (failed: SIGABRT, cublasLtCreate symbol)

train_baseline aborted at the first batch with `Invalid handle. Cannot load symbol cublasLtCreate`
and SIGABRT (exit 134). torch.compile+matmul/conv triggers it; a plain `@torch.compile` on `x*2+1`
doesn't (too small to use cuBLAS).

**Diagnosis via `LD_DEBUG=files`:** torch's bundled cuDNN 9.10.2 (from `nvidia-cudnn-cu12==9.10.2.21`)
chains through `libcudnn_graph.so.9`. The bundled `libcudnn_graph.so.9` lives in
`nvidia/cudnn/lib/`, but the *system* cuDNN at `/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.13.0`
wins the dlopen — cuDNN 9.13 expects `libcublasLt.so.13` (CUDA 13), which isn't on this machine
(everything is CUDA 12.8/12.9). Result: dlopen("libcublasLt.so.13") fails, the cuDNN graph engine
returns "Invalid handle" for `cublasLtCreate`, and torch aborts.

**Root cause is system pollution**, not anything in our envs — system cuDNN gets pulled in because
the bundled cuDNN's dlopen search resolves the unversioned `libcudnn_graph.so.9` via the standard
LD path, not its own `$ORIGIN` RUNPATH. Likely an interaction between cuDNN's internal
`dlopen("libcudnn_graph.so.9", RTLD_LAZY)` and ldconfig's cache.

**Fix:** install a conda activate.d hook in both `policy_doctor` and `mimicgen_torch2` that prepends
`$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/*/lib` to `LD_LIBRARY_PATH`. Verified:
`@torch.compile`-decorated conv+matmul now runs in mimicgen_torch2 (loss = -2221.0 on the
test input).

`scripts/setup_torch2_envs.sh` updated to install the hook on env rebuild.

---

## 2026-05-11 — Fifth launch: training survives, but in-training rollouts crash (EGL)

Training survived the first batches (compile, matmul, conv all working). But every rollout
attempt during training printed a wall of:

```
ImportError: Cannot initialize a EGL device display. This likely means that your EGL
driver does not support the PLATFORM_DEVICE extension
```

Root cause: this GCP machine has only `nvidia-utils-580-server` installed; the user-space
GL libraries are all missing. No `libEGL.so`, no `libGL.so`, no `libOpenGL.so`. So robosuite's
`MjRenderContextOffscreen` couldn't create a headless rendering context for video recording.

Fix:
```
sudo apt-get install -y libnvidia-gl-580-server libegl1 libglvnd0 libgl1 libopengl0 libosmesa6
```

Verified end-to-end with `/tmp/test_runner_with_video.py`: runner with `n_test_vis>0`
constructs the offscreen context, rolls out 2 episodes, writes 2 MP4 videos.

Sixth launch in progress.

---

## 2026-05-12 — Phase 1 of the failure-targeting rewrite

Six new functions in `policy_doctor/mimicgen/failure_targeting.py`:

| Function | Role |
|---|---|
| `enumerate_failure_paths(graph, top_k=5, ...)` | thin wrapper over `BehaviorGraph.enumerate_paths_to_terminal(FAILURE)`, drops loop-paths by default |
| `match_failure_trajectories_to_paths(labels, metadata, paths, level)` | per-episode collapsed cluster sequence; match to path interior; failures only; first-match-wins |
| `collect_failure_trajectory_states_by_node(...)` | for matched eps: t=0 → START pool, per-node visited timesteps → intermediate pool |
| `silhouette_kmeans(features, tags, ..., k_min=2, k_max=10)` | KMeans with k picked by silhouette score; degenerate fallbacks for n=0/1/<k_min |
| `pick_intermediate_target_node(path, heuristic)` | "closest_to_failure" (default), "first", "middle" |
| `intermediate_nodes_for_path(path)` | full list of interior nodes (for Phase-2 multi-constraint use) |

`AnalyzeFailureStatesStep` now branches on `failure_analysis.mode`:
- `"path_based"` → new path: paths → matched failure trajectories → per-node state collection
  → per-node silhouette-k clustering → per-(path, cluster) constraint output.
- `"prefailure_node"` (default) → legacy code path, untouched.

New result schema for path_based mode:
```json
{
  "enabled": true, "mode": "path_based",
  "paths": [
    {"path_idx": 0, "path": [...], "probability": 0.42, "matched_episodes": [...],
     "intermediate_node_id": ..., "intermediate_heuristic": "closest_to_failure",
     "ic_pool":           {"node_id": -2, "n_states": 12, "k": 3, "silhouette": 0.41, "clusters": [...]},
     "intermediate_pool": {"node_id": 11, "n_states": 24, "k": 2, "silhouette": 0.55, "clusters": [...]}}
  ]
}
```

Each cluster carries `center_feature` and `stddev_feature` (used by Phase 2 to size the
constraint slack as `α × stddev`).

22 unit + integration tests in `tests/mimicgen/test_failure_paths.py` — all passing:
- path enumeration (top_k cap, ranking by probability, no-failures edge case)
- trajectory matching (success exclusion, run-length collapsing, first-match-wins)
- intermediate-target heuristic (all three modes, no-interior, unknown raises)
- silhouette-k (obvious 2-cluster, 3-cluster, n=0, n=1, n<k_min fallback)
- node-state collection from a tiny HDF5
- full `_compute_path_based` integration on a synthetic 7-episode graph

Experiment yaml updated to use `mode: path_based` with `top_k_paths: 5`,
`intermediate_heuristic: closest_to_failure`, `kmeans_k_min/max: 2/10`.

Phase 2 (chained-warp constraint enforcement) is next, after the upstream pipeline finishes
producing the actual clustering result we can run failure-targeting against.

---

## 2026-05-12 — Phase 2: ChainedWarpDataGenerator

**Mechanism**: subclass `mimicgen.datagen.data_generator.DataGenerator`, intercept the subtask
loop, and check the constraint after subtask `N` completes (the boundary is observed via
`env_interface.get_datagen_info()` at the *start* of subtask `N+1`). If the achieved object pose
lies outside the configured slack box, the trial early-aborts and returns an empty result.
The outer trial-budget loop in MimicGen's `generate_dataset` already resamples + retries.

When the constraint *is* met, the remaining subtasks run normally — MimicGen's per-subtask
warp transforms the seed segments using the current (constrained) object pose. That's the
"chained-warp" effect the user described: Run-1's achieved-near-target pose becomes the IC
for Run-2 without any explicit splitting.

**Why this beats the old rejection-after-full-trajectory mechanism:**
- Early abort skips executing subtasks N+1..M-1 on doomed trials — cuts wasted compute.
- Slack box is built from data (`slack = α × cluster_stddev`, with min/max clamps) instead
  of a global guess.
- Slack widens by `slack_widen_factor=2×` on retry-loop signal, allowing graceful degradation.

**Files:**
- `policy_doctor/mimicgen/chained_warp_generator.py`:
  - `IntermediateConstraint` (target_pose / slack / objects, with `is_satisfied()` and
    `widen()`).
  - `GenerationOutcome` (what the outer loop reads off `last_outcome`).
  - `make_chained_warp_generator_class()` — lazy factory; sim imports happen only when called.
  - `derive_slack_from_stddev()` — feature-space stddev → world slack with sin/cos→angular
    conversion + clamping.
  - `_datagen_info_to_xy_yaw()` + `_wrap_angle()` helpers.

- `scripts/run_mimicgen_generate.py`: new `--chained_warp_constraint` JSON flag (mutually
  exclusive with the legacy `--subtask_constraints`). When set, swaps the bound subclass into
  `mimicgen.datagen.data_generator.DataGenerator` and
  `mimicgen.scripts.generate_dataset.DataGenerator` before `generate_dataset()` runs.

**Tests:**
- `tests/mimicgen/test_chained_warp.py` — 16 unit tests covering `is_satisfied` (inside/on
  boundary/outside, angle wraparound, missing objects, `objects` filter), `widen`,
  `derive_slack_from_stddev` (typical / clamp-low / clamp-high), `_wrap_angle`,
  `_datagen_info_to_xy_yaw` (4x4 pose extraction, empty), `GenerationOutcome` defaults.
- `tests/mimicgen/test_chained_warp_e2e.py` — gated on `MIMICGEN_E2E=1` and sim-dep
  imports. Copies one demo from `~/data/mimicgen_data/source/square.hdf5`, runs
  `run_mimicgen_generate.py` as a subprocess with two constraints:
    1. impossible target at (9, 9) with tight slack → expect zero successes.
    2. loose slack around the seed's own subtask-0-end pose → expect some successes;
       walks the produced `demo.hdf5` and verifies the boundary pose of every successful
       demo is inside the slack box.

Run the e2e test with:
```
MIMICGEN_E2E=1 conda run -n mimicgen_torch2 --no-capture-output \
  python -m unittest tests.mimicgen.test_chained_warp_e2e -v
```

Currently the GPU is busy with the baseline training; we'll run the e2e once that finishes
or in parallel on a CPU-only mode.

**Not yet done:**
- Plumbing the path-based clusters (Phase 1 result JSON) into the actual generation call —
  i.e. translating each per-cluster constraint dict into an `IntermediateConstraint` and
  invoking `run_mimicgen_generate.py --chained_warp_constraint=...` from
  `generate_mimicgen_demos.py`. This is the last step before the failure-targeting arm
  end-to-end. Will land next.

---

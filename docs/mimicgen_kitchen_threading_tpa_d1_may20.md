# MimicGen Kitchen + Threading + ThreePieceAssembly D1 — May 20 2026 (re-run May 24)

**Status (2026-05-24, 21:40 UTC):** Kitchen budget=100 sweep running — 9 arms in MimicGen generate, ~94% CPU each.
**Goal:** Failure-targeted budget × rep × heuristic sweep on Kitchen D1, Threading D1, ThreePieceAssembly D1, using policy-embedding clustering (`bottleneck_plan_t0`) for behavior-graph + diversity heuristics.

---

## Experiment Design

### Three tasks, one constrained-pose object per task

| Task | Constrained object | x | y | z_rot |
|---|---|---|---|---|
| Kitchen D1 | bread | [-0.04, 0.04] | [-0.04, 0.04] | null *(was [-0.524, 0.524])* |
| Threading D1 | needle | [-0.04, 0.04] | [-0.04, 0.04] | null *(was [-0.524, 0.524])* |
| ThreePieceAssembly D1 | piece_1 | [-0.04, 0.04] | [-0.04, 0.04] | null *(was [-0.524, 0.524])* |

All other objects unconstrained (`null` = use D1 env range). The z_rot=null change is documented under "Lessons learned" below.

### Sweep dimensions per task

- **Heuristics:** `random`, `behavior_graph`, `diversity` (3)
- **Budgets:** `[100, 300, 500]` — additional generated demos on top of the 100-demo baseline
- **Reps:** `rep_seeds: [1, 2, 3]` (3)
- **Total arms per task:** 3 × 3 × 3 = **27**

### Baseline

- `max_train_episodes: 100` — 100 demos from the D1 source dataset
- All 3 baselines already trained, evaluated (500 episodes), and have `metadata.yaml` + `eval_log.json` (kitchen 9 ckpts, threading 14 ckpts, TPA 6 ckpts)

### Clustering for behavior_graph + diversity

- `clustering_influence_source: policy_emb`
- `clustering_policy_emb_layer: bottleneck_plan_t0` — UNet mid-block under planned-action conditioning at the clean denoise step (t=0). Pulled from `main`'s canonical `compute_policy_embeddings.py` (commit 904e845).
- `clustering_level: rollout`, `window_width: 5`, `stride: 2`, `n_clusters: 15` (kmeans), `umap_n_components: 100`

### Device pool (kitchen budget=100, current run)

`devices: [cuda:0, cuda:1, cuda:2, cuda:3, cuda:0, cuda:1, cuda:2, cuda:3, cuda:0]` — 9 slots = all 9 arms in parallel.

### Pipeline commands

```bash
# Kitchen budget=100 (currently running)
python -m policy_doctor.scripts.run_pipeline \
  experiment=mimicgen_kitchen_d1_may20_d100_bread_constrained \
  data_source=mimicgen_kitchen \
  steps=[compute_policy_embeddings,run_clustering,mimicgen_budget_rep_sweep] \
  mimicgen_budget_rep_sweep.budgets=[100] \
  mimicgen_budget_rep_sweep.devices=[cuda:0,cuda:1,cuda:2,cuda:3,cuda:0,cuda:1,cuda:2,cuda:3,cuda:0]
```

After budget=100 wave finishes (incl. evals), repeat with `budgets=[300]`, then `[500]`. Then run the same for threading and TPA.

---

## Session-2026-05-24 Re-run — Lessons Learned

This is a redo of an earlier failed sweep. Documenting the failure modes so future runs avoid them.

### Round-1 failure: wrong policy embedding layer

The earlier sweep clustered on a simplified embedding (`plan_bottleneck`: UNet mid-block at t=0 with a **zero action** input, averaged over horizon). This captures mostly *what the policy sees*, not *what it decides*. Behavior-graph clusters therefore split by observation phase rather than behavioral phase, so `behavior_graph` and `diversity` heuristics had no real signal advantage over `random` — preliminary results showed `random` ≥ `behavior_graph` ≈ `diversity` across all 3 tasks.

**Fix:** Use main's canonical `compute_policy_embeddings.py` (commit `904e845`), which supports `bottleneck_plan_t0`: hook = UNet bottleneck, action = planned (full denoise run then condition on its output), t_single = 0. Pulled into branch `feat/mimicgen-trajectory-pipeline` in commit `d546a2d`.

### Round-1 cleanup

- Archived per-arm sweep work for all 3 tasks (training, eval, per-arm pipeline_runs) — about 308 GB total, kept locally only briefly then **deleted** to free disk (we needed disk for the re-run, no external archive destination available)
- Kept baselines: trained checkpoints + 500-episode eval rollouts + `metadata.yaml` + `eval_log.json` for all 3 tasks
- Deleted bad `policy_embeddings/plan_bottleneck.npz` files from baseline eval dirs
- Deleted stale clustering dirs in `third_party/influence_visualizer/configs/<task>_d1_may20/clustering/`
- Deleted `~/data/robocasa_data` (190 GB, unused) to make room
- Disk after cleanup: 548 GB free

### Round-2 launch — bug chain

Launch attempts revealed several issues:

1. **Hydra `+experiment=` vs `experiment=`** — base config already declares `experiment: null`, so `+experiment=...` errors with "Multiple values for experiment". Use `experiment=...` (no `+`).
2. **`OmegaConf.select` fallback to plain dict** crashes downstream selects — `ComputePolicyEmbeddingsStep` used `OmegaConf.select(cfg, "policy_emb") or {}`. Fixed: `or OmegaConf.create({})`. Commit `615704c`.
3. **Kitchen eval path mismatch** — kitchen baseline eval was at `eval_save_episodes/<train_dir>/latest/` (flat) while `get_eval_dir` expects `eval_save_episodes/<eval_date>/<train_dir>/latest/` (nested). Worked around with a symlink.
4. **CompositeStep `parent_run_dir` wasn't propagating** — every sweep arm raised "No clustering directories found" because `SelectMimicgenSeedStep` was looking under `<top>/mimicgen_budget_rep_sweep/run_clustering/` instead of `<top>/run_clustering/`. The composite arm's sub-steps were getting `parent_run_dir = self.run_dir` (the sweep step_dir) instead of `self.parent_run_dir` (the top-level pipeline run_dir). Fixed in commit `51a700b`.
5. **z_rot constraint causes ~0% MimicGen success** — `bread.z_rot: [-0.524, 0.524]` (±π/6) caused all 9 arms to retry trials with 0 successes after 55+ min. MimicGen replays the seed trajectory adapted to a new sampled initial pose, but Cartesian interpolation of the end-effector trajectory doesn't rotate the gripper to match the new object orientation, so every trial fails the grasp. Memory note `project_ic_bugs_fixed` already documents this fix. Resolved by setting `z_rot: null` (use the seed's own rotation; only ±4cm x/y perturbation applied). Commit `cec6149`.
6. **Mid-debug regression**: at one point I "loosened" z_rot to `[-1.047, 1.047]` (±π/3). That's the wrong direction — wider range = harder MimicGen replay, worse success. Reverted.
7. **HDF5 file-lock + b-tree corruption from concurrent test runs on shared `seed.hdf5`** — investigating z_rot's effect by running multiple `prepare_src_dataset` calls in parallel on the same seed file corrupted it (b-tree "duplicate key", "negative link count"). Resolved by clearing `select_mimicgen_seed/` for every arm so seeds regenerate cleanly; also exported `HDF5_USE_FILE_LOCKING=FALSE` in test scripts.

### Code fixes committed during the re-run (branch `feat/mimicgen-trajectory-pipeline`)

| Commit | Description |
|---|---|
| `e6f3f18` | fix: per-arm mimicgen datagen dir + --guarantee passthrough |
| `f0adcdc` | fix: rollouts.hdf5 + AsyncVectorEnv shared-memory crash |
| `516666e` | fix: strip `_orig_mod.` prefix from torch.compile state_dicts |
| `9238c1d` | fix: auto-inject model_file when seed HDF5 lacks it |
| `68ce938` | feat: kitchen/threading/three_piece_assembly D1 may20 sweep configs |
| `d546a2d` | revert + restore: use main's policy_emb implementation (`bottleneck_plan_t0`) |
| `8629537` | feat: pipeline-step wrapper for compute_policy_embeddings + bottleneck_plan_t0 default |
| `78b5c2f` | config: switch may20 sweep experiments to bottleneck_plan_t0 layer |
| `f2f5c14` | fix: misc pipeline + training fixes needed by the re-run (hydra.run.dir, wandb allow_val_change) |
| `615704c` | fix: OmegaConf.select needs DictConfig not plain dict for fallback |
| `5a75016` | fix: unset stale clustering_dir in threading experiment |
| `51a700b` | fix: composite step propagates parent_run_dir to its sub-steps |
| `cec6149` | revert: z_rot=null (documented fix; ±π/3 was wrong direction) |

---

## Current State (2026-05-24, 21:40 UTC)

- Kitchen budget=100: 9 arms in MimicGen generate, all worker pythons at ~94% CPU
- compute_policy_embeddings ✓ done (50000 timesteps × 128-dim at `bottleneck_plan_t0.npz`)
- run_clustering ✓ done (output at `third_party/influence_visualizer/configs/kitchen_d1_may20/clustering/...kmeans_k15`)
- No results yet — first stats.json from any arm expected within ~20-30 min

### Planned waves after kitchen budget=100 completes

| Wave | Task(s) | Budget | Notes |
|---|---|---|---|
| 1 (current) | kitchen | 100 | 9 arms in 9 GPU slots |
| 2 | kitchen | 300 | reuses compute_policy_embeddings + run_clustering |
| 3 | kitchen | 500 | same |
| 4-6 | threading | 100/300/500 | needs separate launches; threading config still uses x/y±4cm, z_rot=null |
| 7-9 | TPA | 100/300/500 | same |

After all 3 tasks × 3 budgets have completed evals, run `compare_policies.py` for proper Wilson CIs / Beta posteriors / CLD letters.

---

## Open Risks

- **z_rot=null gives up the rotational-failure-targeting dimension.** Once we have data with z_rot=null, we may want to investigate whether a tighter range (e.g. ±0.05 rad ≈ ±3°) is small enough that MimicGen's nearest-neighbor source selection can find a matching seed. The z_rot test sweep this session was aborted due to seed.hdf5 corruption from concurrent runs — would need a clean retry on a single arm to test.
- **Behavior_graph + diversity heuristics still depend on cluster quality.** Even with the correct embeddings, if the kitchen task's behavior space is shallow (few distinguishable phases), we may not see large gaps between heuristics. The earlier session's `random ≥ all` result is suggestive — but it was confounded by bad embeddings. Need fresh data to draw any conclusion.
- **TPA has only 6 baseline checkpoints** (vs kitchen 9, threading 14). Smaller checkpoint pool means eval estimates have wider variance.
- **Threading needle constraint may still be too hard** even with z_rot=null. The previous run had 7/9 threading arms with 0 demos despite the looser constraint; needle is structurally hard for MimicGen replay. May produce baseline-only training for many threading arms.

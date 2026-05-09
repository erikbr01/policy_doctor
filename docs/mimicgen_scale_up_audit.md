# MimicGen scale-up audit — `worktree-feat+mimicgen-eef-pipeline`

**Date:** 2026-04-29
**Scope:** Full audit of the MimicGen pipeline on this worktree (18 commits ahead of `main`) before scaling up budget / demo-count / replicate experiments. The goal is *maximum confidence in results* — silent failures, subtle bugs, and config/path mixups are unacceptable.

Findings personally verified against source are marked ✓; findings reported by audit subagents but not personally reverified are marked ◇. Several subagent claims that turned out to be wrong on inspection have been corrected and downgraded — see notes inline.

---

## BLOCKERS — fix before any large-scale run

### B1 ✓ Heuristics give *asymmetric* stochasticity to "base" runs — variance estimation is biased

`policy_doctor/mimicgen/heuristics.py:121, 270, 403`

For the apr26 yaml `random_seed: null`:

- `RandomSelectionHeuristic.__init__` does `self.rng = np.random.default_rng(random_seed)`. With `random_seed=None`, NumPy seeds from system entropy → **non-deterministic across re-runs**.
- `BehaviorGraphPathHeuristic.__init__` does `self.rng = np.random.default_rng(random_seed) if random_seed is not None else None`, then `if self.rng is not None: self.rng.shuffle(idxs)` (line 194). With `None`, **no shuffle** → deterministic, picks first eligible.
- `DiversitySelectionHeuristic` same pattern (line 270, 340-341).

Pooling base + rep2 + rep3 to estimate variance is invalid:

- For `random` arm: 3 stochastic samples (but base is non-reproducible).
- For `behavior_graph` / `diversity` arms: base is degenerate (always picks first), only rep2/rep3 are stochastic. Variance estimator is biased downward; SE bars are wrong.

**Fix:** In `build_heuristic`, pass `random_seed=0` (not `None`) for the base run, so all three heuristics seed identically. Re-run base arms after the fix.

### B2 ✓ `train_on_combined_data` ignores `baseline.max_train_episodes` — demo-count sweep is invalid

`policy_doctor/curation_pipeline/steps/train_on_combined_data.py:174-198`

The training overrides list passes `++task.dataset.dataset_path={combined_hdf5_path}` but no `max_train_episodes` / `n_episodes` constraint. The combined HDF5 always contains `n_original + n_generated` demos. So when the apr26 sweep iterates `demo_counts: [60, 100, 300]`:

- The shell script renames `train_date` per demo count, but the combined dataset is `original_60 + generated_200 = 260` demos in every arm (only the *original* HDF5 differs).
- Whatever `max_train_episodes` lives in cupid's Hydra task config silently applies to the combined data. If it's a number, it subsamples; if not, you get the full 260 in every demo-count arm.

Either way, the apr26 demo-count sweep results are confounded. The header comment in the experiment yaml acknowledges this loosely on lines 34-35 ("train_on_combined_data always uses train_ratio=0.64").

**Fix:** Inject `++task.dataset.max_train_episodes={N + n_generated}` (or pre-subsample combined.hdf5 to the intended total) before training.

### B3 ✓ Empty / 0-demo `demo.hdf5` after merge passes silently into training

`policy_doctor/curation_pipeline/steps/generate_mimicgen_demos.py:667-675, 706-709` and `train_on_combined_data.py:87`

In the per-seed retry path, `_merge_hdf5s` writes a file with `total=0` if every per-seed pass produced empty output. The downstream check at `train_on_combined_data.py:87` is `generated_hdf5_path.exists()` — passes for an empty file. Then `combine_hdf5_datasets:68-70` (`if not gen_demo_keys: return _count_demos(output_path)`) silently returns `original` count and training proceeds on the unaugmented data.

**Fix:** After merge, raise `RuntimeError` if `merged_count == 0`. In `combine_hdf5_datasets`, raise on empty `gen_demo_keys` instead of silently no-op.

---

## HIGH

### H1 ✓ Adaptive retry can exhaust its trial cap with successes < budget — silently produces unequal generated-data sizes across arms

`generate_mimicgen_demos.py:550, 650-655`

`max_total_trials = success_budget * 20`. If observed success rate is much lower for one arm (e.g., random heuristic gives lower-quality seeds → lower MimicGen success rate), the loop prints a WARNING and exits with whatever it has. So `random_budget1000` arm may end up with 600 generated demos while `behavior_graph_budget1000` gets the full 1000. Eval then compares unequal training sizes.

**Fix:** Either (a) raise on shortfall, or (b) tighten the cap and explicitly subsample down to a *guaranteed-met* floor (the lowest realised count across arms).

### H2 ✓ "60-demo subset" requires `mimicgen_datagen.original_dataset_path` override — silent fallback to full D1 (1000 demos) if missing

`train_on_combined_data.py:240-291`

Priority order: `mimicgen_datagen.original_dataset_path` → `task.dataset.dataset_path` → Hydra config file. None of these is validated against the experiment's intent. A typo in a new yaml or forgotten override means combined.hdf5 = 1000 (full D1) + 200 generated = 1200 demos; results invalid but no warning.

**Fix:** Compare resolved demo count against an expected demo count in the config and raise on mismatch, OR require `original_dataset_path` to be explicit (no fallback) for any mimicgen experiment.

### H3 ✓ Per-seed `last pass wins` for `ep_length_*` stats is wrong

`generate_mimicgen_demos.py:642-644`

```python
for ep_k in ("ep_length_mean", "ep_length_std", "ep_length_max", "ep_length_3std"):
    if ep_k in s:
        acc[ep_k] = s[ep_k]  # last pass wins for episode lengths
```

`max` should be `max(acc.get(ep_k, -inf), s[ep_k])`; `mean`/`std` should be re-aggregated from accumulated sums (which the code does maintain). Reporting bug only — does not affect trained-policy results — but if you use these numbers to validate generation quality before training, you'll be misled.

### H4 ◇ `eval_baseline.py` returns `0.0` for missing/malformed `eval_log.json`

`policy_doctor/curation_pipeline/steps/eval_baseline.py:179-181`

Crashed eval → silent 0.0 in aggregation → arm appears worst, not "failed". Mean/SE silently include synthetic 0.0 from broken arms.

The newer `eval_mimicgen_combined.py` already raises (per fix `72535a7`). Backport that pattern to baseline.

### H5 ◇ `plot_mimicgen_budget_sweep.py` silently drops corrupt/missing `result.json`

`scripts/plot_mimicgen_budget_sweep.py:89-90`. Failed seeds disappear from the plot with no annotation; you can't distinguish "missing" from "0%".

### H6 ✓ Concurrent budget-sweep arms rely on cupid's `training.device` override only

`policy_doctor/curation_pipeline/steps/mimicgen_budget_sweep.py:116-130`, `train_on_combined_data.py:222-224`

Each arm gets a unique `cuda:N` via `OmegaConf.update`, but `subprocess.run` inherits the parent's full `os.environ` — `CUDA_VISIBLE_DEVICES` is not pinned. If cupid's `train.py` allocates on `cuda:0` *before* honoring `training.device`, two arms targeting `cuda:0` race on memory. Verify `train.py` calls `torch.cuda.set_device()` *before* any `.cuda()` allocation; otherwise wrap each arm with explicit `CUDA_VISIBLE_DEVICES`.

### H7 ✓ `is_done` sentinel does not validate that checkpoints actually exist

`policy_doctor/curation_pipeline/base_step.py:103-105`

If training writes the `done` sentinel and then the checkpoint disk fills / training crashes after sentinel-touch, `skip_if_done=True` will reuse the empty `train_dirs` and downstream eval evaluates a bogus / nonexistent ckpt. (The recent fix in `eval_mimicgen_combined` to raise on missing checkpoints catches this loudly — but the *training* step itself happily reports done with no model.)

**Fix:** In `TrainOnCombinedDataStep.compute()`, verify each `run_output_dir/checkpoints/*.ckpt` exists before returning. Apply same in `TrainBaselineStep`.

---

## MEDIUM

### M1 ✓ `select_mimicgen_seed.py` cross-step lookup uses `parent_run_dir`; `from_graph` step uses `run_dir` — divergent

`select_mimicgen_seed.py:94` vs `select_mimicgen_seed_from_graph.py`. Inside a CompositeStep these resolve to different paths. Currently only `select_mimicgen_seed` is wired into composites, so the live call chain is correct, but the dual-stepping is a foot-gun.

**Fix:** delete `select_mimicgen_seed_from_graph` if no longer used, or normalize both to `parent_run_dir`.

### M2 ✓ `output_dir` in `mimicgen_datagen` is read but ignored

`generate_mimicgen_demos.py:295` reads `output_dir_rel`, then line 476 explicitly says it's "intentionally ignored." Either delete the read or use it. Currently it's a config field that does nothing — easy way to confuse future-you when scaling.

### M3 ✓ Variance knob falsiness inconsistency

`generate_mimicgen_demos.py:282-318` mixes `OmegaConf.select(..., default=...)` and `OmegaConf.select(...) or fallback`. The former is correct; the latter silently rewrites legitimate `0` / `False` / `[]` to defaults. Most variance knobs use the safe form, but `success_budget` (282), `episode_budget`/`num_trials` (291-294), `task_name` (298), `output_dir` (296) etc. use the unsafe form. Set `task_name=""` or `success_budget=0` and you'll silently fall back.

### M4 ✓ Stats aggregation inflates `num_attempts` on subprocess failure

`generate_mimicgen_demos.py:618-619`: when a per-seed pass returns non-zero exit, the code does `total_trials += trials_per_seed` to count it as used. But the corresponding `total_successes += 0`. The retry loop will exhaust trial budget faster on flaky envs — could push the job into the H1 silent-shortfall regime.

### M5 ✓ `evaluation.overwrite: false` in apr26 yaml + `eval_date` not unique per arm

`policy_doctor/configs/experiment/mimicgen_square_sweep_apr26.yaml:64` sets `overwrite: false`. If a prior eval crashed and left partial output, `overwrite: false` reuses the partial state. Combined with the per-arm output dirs being keyed on `train_date` (not `train_date + arm_name`), there's a chance that two arms write to overlapping eval dirs. Worth tracing `get_eval_dir(...)` to be sure.

### M6 ◇ `combine_datasets.py` does not deduplicate trajectories

`combine_datasets.py:60-78`. Generated demos get appended as `demo_{n_existing+i}` — no check for content collisions or duplicate states (e.g., if generation accidentally re-emitted seed). Low likelihood at scale, but worth a hash check if you want bulletproof uniqueness.

### M7 ✓ `task_config: square_mh_apr26_sweep_demos60` requirement

The file exists at `third_party/influence_visualizer/configs/square_mh_apr26_sweep_demos60.yaml` (a subagent missed this — they only checked `policy_doctor/configs/`). The apr26 sweep is **NOT blocked** by missing config.

However, several pipeline steps still load from `iv_task_configs_base` (`compare.py`, `run_curation_config.py`, `annotate_slices_vlm.py`, `evaluate_cluster_coherency_vlm.py`, `export_markov_report.py`) — verify the iv yaml's contents (eval_dir, state_labels, seeds list) match the apr26 dataset before using any of those steps.

---

## LOW / NIT

- **N1** ✓ `combine_hdf5_datasets` uses `shutil.copy2` then `h5py.File(..., "a")` — safe, but if `original.hdf5` is a symlink across filesystems, the append happens on the local copy (correct, just noting).
- **N2** ◇ `episode_dataset.py` filter `file.endswith(".pkl")` excludes `.pkl.gz`. Currently fine; if you ever compress eval dumps, training silently sees zero episodes.
- **N3** ◇ `mimicgen_lowdim_runner.py` deferred AsyncVectorEnv creation: confirm `run()` is only ever called once per runner instance (statefulness risk).
- **N4** ✓ `wandb` env var leakage from parent into `subprocess.run(..., env={**os.environ, "WANDB_RESUME": "never"})` (line 222) — `WANDB_RUN_ID`, `WANDB_NAME`, etc. inherited. Concurrent arms can collide on wandb directory locks.
- **N5** ✓ Generated arms can return `< num_seeds` if eligibility is short — heuristics print warnings but composites continue with whatever they got. Logged `selection_info.num_seeds` in the result, so detectable post-hoc, not silently lost.
- **N6** ✓ The default `random_seed` in `apr26.yaml` (`random_seed: null`) is the source of B1 — config-level fix is to set `random_seed: 0` here as the explicit base seed.

---

## Architecture / orchestration concerns

1. **Composite arm explosion in `mimicgen_arm.py`**: 18 hand-written `*ArmStep` subclasses plus `_make_budget_arm_class` factory. The hand-written variants overlap with what the factory could produce (e.g., `MimicgenRandom20Rep2ArmStep` is just `(random, budget=20, random_seed=1)` — already expressible). Consider deleting all hand-written arm classes and driving everything via the budget-sweep config matrix. Less surface area to forget to register / update.

2. **Two seed-selection step classes coexist**: `SelectMimicgenSeedStep` (new) and `SelectMimicgenSeedFromGraphStep` (legacy). `generate_mimicgen_demos.py:339-363` reads from whichever is `is_done()`. At scale, having two write surfaces is a contamination risk — old runs may have one done sentinel, new arms write the other → wires cross. Recommend deleting legacy step or making `from_graph` raise if called.

3. **`run_dir` auto-generation**: `pipeline.py` auto-generates `run_dir` from `datetime.now()`. Subagent flagged collision in same-second invocations. The shell launchers explicitly set `run_dir`, so production paths are safe — but anyone running a manual `run_pipeline` invocation has the foot-gun. Consider appending a short uuid suffix.

4. **Stat semantics drift between baseline and combined eval**: baseline eval has `_read_mean_score → 0.0` fallback (H4); combined eval raises. After scaling, the analysis script reads both with the same parser. Unify the failure semantics.

5. **`run_clustering` was rightly decoupled from iv-task-config (commit `42da17e`)** — but the rest of the pipeline (compare, run_curation_config, etc.) still requires it. Documenting which steps need iv configs and which don't would help.

6. **Concurrency model**: `mimicgen_budget_sweep` uses `ThreadPoolExecutor` with subprocess shell-outs. That's fine, but the underlying `subprocess.run` is blocking per-thread, so you have N OS threads each holding a subprocess. The concurrency is real but coarse — if you scale to 16 GPUs, 16 active conda subprocesses each loading the cupid stack will stress disk I/O at startup. Consider `mp.Pool` with prewarmed workers or staggered launch (the apr26 shell already staggers 30s).

---

## Recommended fix order before scaling

1. **B1 + N6**: change `random_seed: null` → `random_seed: 0` in apr26 yaml (one-line) + audit that `BehaviorGraphPathHeuristic` and `DiversitySelectionHeuristic` actually shuffle when seeded. Re-run base arms.
2. **B2**: pre-subsample `combined.hdf5` to the intended total (most surgical), or pass `++task.dataset.max_train_episodes`.
3. **B3 + H1**: raise on empty merge / shortfall instead of warn.
4. **H6 + H7**: add `CUDA_VISIBLE_DEVICES` pinning per arm + checkpoint-existence check at end of training step.
5. **H4 + H5**: backport "raise on missing eval result" pattern to baseline + plotter.
6. **H3, M2, M3**: easy hygiene fixes.

Then, separately, inventory the iv-task-config dependencies (M7) and confirm the apr26 iv yaml is internally consistent with the apr26 experiment.

---

## What was NOT audited (recommend covering before scaling further)

- `run_mimicgen_generate.py` and the MimicGen subprocess in `mimicgen_torch2` env — particularly `prepare_src_dataset` correctness for D1 with multiple seeds in `seed.hdf5`, and how `--seed_object_poses` / `--object_pose_ranges` interact with `nn_k > 1`.
- The 4 modified `third_party/cupid` files line-by-line (high-level review only); especially `async_vector_env.py` (33 lines changed — may have non-obvious semantics around env reset / seed propagation).
- Whether the InfEmbed embeddings used by clustering are recomputed when the dataset changes (cache invalidation across demo-count arms).
- Whether `wandb` group/run-name collisions surface during concurrent budget-sweep arms.

---

## Audit methodology

- Codebase root audited: `/home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees/feat+mimicgen-eef-pipeline` at commit `72535a7` (18 commits ahead of `main`).
- Files read in full by primary auditor: `base_step.py`, `mimicgen_arm.py`, `mimicgen_budget_sweep.py`, `generate_mimicgen_demos.py`, `select_mimicgen_seed.py`, `train_on_combined_data.py`, `combine_datasets.py`, `heuristics.py`, `mimicgen_square_sweep_apr26.yaml`, `run_clustering.py`.
- Files audited via `Explore` subagents: `select_mimicgen_seed_from_graph.py`, `graph_seed.py`, `materializer.py`, `seed_trajectory.py`, `eval_mimicgen_combined.py`, `eval_baseline.py`, `plot_mimicgen_budget_sweep.py`, all experiment YAMLs, paths.py, `run_sweep_ordered.sh`, `run_mimicgen_budget_sweep.sh`, modified `third_party/cupid` files.
- Subagent claims that contradicted source on reverification have been corrected and demoted (notably the apr26 task_config "blocker" — the file does exist at `third_party/influence_visualizer/configs/`).

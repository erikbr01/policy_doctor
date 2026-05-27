# Repository Refactor Plan

Status: **Draft v1** — awaiting user approval before any code changes.
Branch (not yet created): `refactor/clean-architecture`
Tracking doc for findings as we work: `REFACTOR_FINDINGS.md` (created at phase start).

---

## 1. Goals

From the user request, restated as concrete acceptance criteria:

1. **Experiment-centric layout.** Every experiment owns one directory on disk that contains its config and every artifact it consumes or produces. `rsync experiments/<name>/ remote:` must give a fully-runnable experiment on the remote without reaching outside that directory.
2. **Single canonical data root.** No more `third_party/cupid/data/...` nested inside a submodule. One root at `<repo_root>/data/`, env-var overridable. The legacy `third_party/cupid/data/` stays on disk (frozen, read-only) but nothing new writes there.
3. **No date-keyed artifact paths.** `train_date` / `eval_date` removed from configs and code. Runs are identified by experiment name + timestamp.
4. **`influence_visualizer` removed.** The ~6 live functions absorbed into `policy_doctor/`; the rest deleted.
5. **uv workspace replaces conda.** One repo-level workspace with four env definitions (`analysis`, `cupid`, `mimicgen`, `robocasa`). `conda activate` and `conda run -n` eliminated.
6. **Code dedup + reorganization.** Pull out HDF5 readers, path utilities, plotly builders, clustering serialization into named modules; no more duplicate implementations across `policy_doctor/` and `third_party/influence_visualizer/`.
7. **Streamlit-only viz.** Delete the "main policy_doctor viz" surface — standalone plotting scripts and non-Streamlit render utilities. Keep `streamlit_app/` and the `plotting/` modules it consumes.

**Hard non-goals (explicit guardrails):**
- Old artifacts under `third_party/cupid/data/` are *frozen, not deleted*. New code reads from and writes to `<repo_root>/data/`. No backward-compat shim for the old paths — if a refactored step encounters an old path, it errors out rather than silently working.
- The deployed user-study apps (`study.behavior-graphs.com`, `demo.behavior-graphs.com`) must remain deployable from `main` until cutover. They cannot regress.
- This is a *refactor*. No new features, no scope creep. If we see a tempting cleanup that isn't on this list, log it in `REFACTOR_FINDINGS.md` and move on.

---

## 2. Current state (one-paragraph summary)

Configs live in three places (`policy_doctor/configs/`, `third_party/influence_visualizer/configs/`, plus ad-hoc YAMLs at the repo root). Artifacts land under `REPO_ROOT/data/` where `REPO_ROOT` magic-resolves to `third_party/cupid/` when that dir is populated — i.e., checkpoints end up nested inside a submodule. Run dirs are keyed by `train_date` / `eval_date`. There are four conda envs (`policy_doctor`, `mimicgen_torch2`, `cupid`, `robocasa`) and several setup scripts; pipeline steps shell out via `conda run -n <env>`. `third_party/influence_visualizer/` has 40 modules but only ~6 functions are imported by the rest of the codebase — the rest is dead code for a legacy standalone Streamlit app. There's substantial duplication between `policy_doctor/data/` and `influence_visualizer/` (clustering save/load, HDF5 loaders, plotly heatmap/frame builders). The deployed user-study Streamlit stack reads cluster results, influence matrices, and HDF5 demos from disk and writes survey responses to `/mnt/data/survey_responses/`.

(Detail: see the three exploration reports at the top of this transcript.)

---

## 3. Target architecture

### 3.1 On-disk layout

```
$POLICY_DOCTOR_DATA/                       # default <repo_root>/data, override via env var
    datasets/                              # source HDF5 datasets, shared across experiments
        robomimic/transport_mh.hdf5
        mimicgen/square_d0.hdf5
        ...
    experiments/
        <experiment_name>/                 # one self-contained experiment
            manifest.yaml                  # name, created_at, git_sha, env_name, source_dataset_path
            config/                        # resolved Hydra config snapshot (not a reference)
                config.yaml                # fully-resolved, including all overrides
                composed_overrides.yaml    # for traceability
            shared/                        # things shared across arms in this experiment
                baseline_ckpt/             # one copy, referenced by all arms
                source_dataset.hdf5        # symlink or copy of the source HDF5
            runs/                          # pipeline runs / experimental arms
                <run_name>/
                    <step_name>/
                        result.json
                        done               # sentinel
                        ...artifacts...
            logs/
                pipeline.log
                <run_name>.log
```

**Invariant:** every file an experiment needs to execute lives under `experiments/<name>/` *except* the source HDF5 dataset, which is symlinked from `datasets/`. The `bundle-experiment` CLI dereferences that symlink into a hard copy under `shared/` when packaging for transfer.

**Within an experiment:** if multiple arms share a baseline checkpoint, it lives once at `shared/baseline_ckpt/` and each arm's config references it via a relative path. No duplication inside an experiment.

**Across experiments:** the same baseline checkpoint is *copied* into each experiment's `shared/` (the user's explicit requirement — portability beats disk).

### 3.2 Config layout

Configs live in two places:

- **Repo: `policy_doctor/configs/`** — Hydra templates. This is what users edit and what's versioned with code.
- **Disk: `<experiment>/config/`** — the *resolved* config for a specific experiment, written at experiment-creation time. This is what `bundle-experiment` exports; this is what reproduces the experiment.

Removed from the schema:
- `train_date`, `eval_date` (replaced by experiment timestamp)
- `train_output_dir`, `eval_output_dir` (replaced by experiment-relative paths)
- `attribution.tf32`, `attribution.compile` stay (functional flags, not paths)

The `experiment:` Hydra group goes away as a separate concept — every Hydra invocation *creates or resumes* an experiment.

### 3.3 uv workspace

Repo root `pyproject.toml` becomes a uv workspace with two project members (each a sub-`pyproject.toml`):

- `policy_doctor/` — orchestration, clustering, behavior graph, influence ops, Streamlit. **No sim deps.**
- `third_party/cupid/` — diffusion policy training, TRAK, InfEmbed; depends on `policy_doctor`. **Sim-compatible.**
- (No `influence_visualizer` member — deleted.)

Four named envs declared via uv's optional-dependency groups (or workspace env extensions — exact mechanism nailed down in Phase 1). They differ primarily in their robosuite/robomimic/mujoco pinning, since these stacks are mutually incompatible:

| Env | Torch | Robosuite | Robomimic | Mujoco | Purpose |
|---|---|---|---|---|---|
| `analysis` | 2.8.0 (cu11.6) | — | — | — | Orchestration, clustering, Streamlit, attribution post-processing. The default; what you run unless you need sim. |
| `cupid` | 2.8.0 | 1.2.0 | 0.2.0 | 3.2.6 | Robomimic + diffusion-policy training/eval. Inherits the current `cupid` env (already torch-2). |
| `mimicgen` | 2.8.0 (cu12.9) | 1.4.1 | 0.3.0 | (via robosuite) | MimicGen seed selection + generation, MimicGen-trained policies (square/transport/coffee/threading). |
| `robocasa` | 2.7.1 (cu11.8) | 1.5.2 | 0.3.0 | 3.3.7 | RoboCasa manipulation tasks. |

`conda run -n X python …` becomes `uv run --extra X python …` (exact incantation TBD against current uv — verified in Phase 1).

The pre-torch-2 legacy `cupid` env (torch 1.12 / mujoco 2.3) is already gone; the `cupid` env above is the existing torch-2 env unchanged. We're torch-2 only.

### 3.4 Influence ops in `policy_doctor`

New module layout absorbing the live parts of `influence_visualizer`:

```
policy_doctor/
    influence/
        loader.py          # HDF5 loading, episode/sample enumeration (was iv/data_loader.py)
        lazy_hdf5.py       # was iv/lazy_hdf5.py
        clustering_io.py   # save/load clustering results (merges iv/clustering_results.py + pd/data/clustering_loader.py)
        paths.py           # get_eval_dir_for_seed, get_clustering_dir, etc. (de-dated)
```

Dead code from `influence_visualizer/render_*.py`, the standalone app, and unused plotting helpers — deleted.

---

## 4. Phased plan

Each phase ships as a sequence of commits onto `refactor/clean-architecture`. Phase boundaries are merge commits; the full test suite must pass at every boundary.

### Phase 0 — Scaffolding (low risk)

**Scope:**
- Create `refactor/clean-architecture` branch.
- Stub `REFACTOR_FINDINGS.md`.
- Write `tools/refactor/snapshot.py` — captures golden outputs from current code for the correctness anchors (see §5).
- Snapshot golden outputs from current main: a small clustering result, a behavior graph, a small influence matrix slice, a MimicGen seed selection. Commit fixtures to `tests/golden/`.

**Exit criteria:** golden fixtures committed; replaying them through *current* code reproduces them byte-for-byte.

**Risk:** very low — pure additions.

---

### Phase 1 — uv workspace and env unification

**Scope:**
- Write workspace-level `pyproject.toml` with four optional-dependency groups (`analysis`, `cupid`, `mimicgen`, `robocasa`).
- Lock dependencies (`uv lock`).
- Add `scripts/setup_uv.sh` that creates all four envs.
- Pick *one* pipeline step (e.g. `run_clustering`) and migrate its conda invocation to `uv run --extra analysis`.
- Run the full `policy_doctor` test suite under uv `analysis` env to prove parity.
- Once parity proven on one step, migrate the rest: `train_baseline` (mimicgen env), `train_attribution` (cupid env), `compute_infembed` (analysis env), `generate_mimicgen_demos` (mimicgen env), `eval_mimicgen_combined` (mimicgen env), DAgger entry points (cupid env), etc.
- Delete `environment_*.yaml`, `scripts/setup_torch2_envs.sh`, `scripts/create_cupid_torch25.sh`.

**Exit criteria:**
- `python run_tests.py --suite policy_doctor` passes under `uv run --extra analysis`.
- `python run_tests.py --suite cupid` passes under `uv run --extra cupid`.
- `python run_tests.py --suite mimicgen` passes under `uv run --extra mimicgen` (when GPU available).
- No `conda` references in any `.py` or `.sh` file.

**Risk:** medium. Mujoco + robosuite + mimicgen wheel availability under uv could bite. Mitigation: prototype the `mimicgen` env first as a smoke test before committing.

---

### Phase 2 — Experiment-centric artifact layout

**Scope:**
- Add `policy_doctor/experiment/` module:
  - `ExperimentDir` — dataclass wrapping `<DATA_ROOT>/experiments/<name>/`, with methods for `runs_dir()`, `shared_dir()`, `config_dir()`, etc.
  - `Experiment.create(name, config)` — materializes the dir, copies baseline checkpoints into `shared/`, writes resolved config.
  - `Experiment.load(name)` — opens an existing experiment.
- New `policy_doctor.paths` resolves `POLICY_DOCTOR_DATA` env var (default `~/policy_doctor_data/`).
- Update `PipelineStep` base class to take an `Experiment` instead of a freeform `run_dir`. Each step's `step_dir` becomes `<experiment>/runs/<run_name>/<step_name>/`.
- Strip `train_date` / `eval_date` from `config.yaml` and every step that reads them. Search-and-destroy: `grep -r 'train_date\|eval_date' policy_doctor/ third_party/cupid/`.
- Add `experiment-bundle` CLI: dereferences symlinks → hard copies, tar+xz the experiment dir for transfer.
- Add `experiment-init` CLI: creates a new experiment from a config name + baseline checkpoint path.

**Exit criteria:**
- One golden experiment (a small end-to-end `run_clustering` + `train_curated` pipeline on a tiny fixture) runs cleanly and produces a self-contained experiment dir under `experiments/`.
- Bundle/unpack roundtrips produce identical outputs.
- All other steps still work (golden snapshots match).

**Risk:** high — touches every pipeline step. Mitigation: golden snapshots from Phase 0 catch regressions immediately. Migrate one step at a time, run snapshot test after each.

---

### Phase 3 — Absorb `influence_visualizer`

**Scope:**
- Inventory the 6 live functions imported into `policy_doctor/`. Move them to `policy_doctor/influence/{loader,clustering_io,paths,lazy_hdf5}.py`.
- Merge `clustering_results.py` (IV) and `clustering_loader.py` (PD) into one canonical `policy_doctor/influence/clustering_io.py`. Keep the richer PD API (saves sklearn pipeline models).
- Update every import site (~13 files in `streamlit_app/`, `curation_pipeline/steps/`, tests).
- Delete `third_party/influence_visualizer/` entirely.
- Update root `pyproject.toml` to drop the editable install.

**Exit criteria:**
- `grep -r 'influence_visualizer' .` returns zero hits.
- Streamlit apps boot and all tabs render against a fresh experiment dir.
- Golden snapshots still match.

**Risk:** medium. Streamlit imports are the riskiest because they're checked at runtime, not test time. Mitigation: smoke-test the deployed Streamlit stack against a golden experiment as part of phase exit.

---

### Phase 4 — Viz cleanup

**Scope:**
- Identify every Python file under `scripts/` that produces a plot or HTML. Cross-reference against active Streamlit tabs and pipeline steps.
- Delete `scripts/plot_*.py`, `scripts/render_*.py`, `scripts/export_*_report.py` that have no live callers.
- Delete `policy_doctor/plotting/curation_scatter_mpl.py`, `policy_doctor/plotting/policy_comparison.py` if not consumed by Streamlit.
- Delete `policy_doctor/scripts/plot_*.py` likewise.
- Audit each plotting module: keep only those imported by `streamlit_app/` or an active pipeline step.

**Exit criteria:**
- Every `policy_doctor/plotting/*.py` has a Streamlit caller (verified by grep).
- No standalone plot scripts remain in `scripts/`.
- Streamlit apps still render.

**Risk:** low. Pure deletion; reverting is a `git revert`.

---

### Phase 5 — Code dedup and reorganization

**Scope:**
- HDF5 readers: ensure exactly one canonical reader in `policy_doctor/data/hdf5.py`. Delete or wrap any others.
- Path resolution: collapse `policy_doctor/paths.py`, the IV path module (now absorbed), and any inline path-construction logic into a single module.
- Hydra config helpers: consolidate `policy_doctor/curation_pipeline/config.py` and the various ad-hoc YAML loaders.
- Plotly figure builders: ensure `policy_doctor/plotting/plotly/` is the only home for figure construction. No `st.plotly_chart(go.Figure(...))` inline in render code (this is already a stated rule — enforce it via test/grep).
- Reorganize `scripts/` into `scripts/experiments/` (user-facing launchers) and `scripts/dev/` (one-off utilities). Delete anything not actively used.

**Exit criteria:**
- Static analysis pass: no two modules with overlapping responsibility (judged by maintainer review).
- All tests green.

**Risk:** low-medium. Most changes are local renames.

---

### Phase 6 — Documentation, conda cleanup, and merge to `main`

**Scope:**
- Update `CLAUDE.md` to reflect new layout (env names, paths, commands).
- Update `README.md` with `uv sync --extra analysis` quickstart.
- Update `deploy/README.md` if any deploy paths changed.
- Touch up `conda activate` references across `docs/*.md` and test docstrings.
- **From Phase 1 tail (deferred):** Delete `environment_*.yaml`, `scripts/setup_torch2_envs.sh`, `scripts/create_cupid_torch25.sh` once sim extras are validated on a Linux box.
- Final smoke test: deploy the Streamlit stack to a staging env (or local Docker compose) and click through both Group A and Group B flows.
- Open PR `refactor/clean-architecture` → `main`. Squash-merge if small, or merge with phase commits preserved if useful for archaeology.

**Exit criteria:** PR merged to `main`. `https://study.behavior-graphs.com` and `https://demo.behavior-graphs.com` redeployed and working.

**Risk:** low (almost docs at this point).

---

## 5. Correctness anchors (testing strategy)

The user's hard constraint: don't break correctness. Translated into tests:

### 5.1 Golden-output snapshot tests

Capture deterministic outputs from *current main* on small fixtures, commit to `tests/golden/`. At every phase boundary, replay through the refactored code and assert byte-equality (or numeric closeness with documented tolerance).

Targets:
| Anchor | Fixture | What's pinned |
|---|---|---|
| Clustering | 200 rollout samples × 50 demo samples influence matrix | `cluster_labels.npy`, `coords.npy`, `clustering_models.pkl` hash |
| Behavior graph | Cluster labels from above | Graph structure (node ids, edge counts, transition probabilities) as canonical JSON |
| Influence matrix | Small synthetic InfEmbed fit | `get_slice(0,10,0,5)` output array |
| MimicGen seed selection | Saved behavior graph + episodes | `seed_indices.json` from each of the 3 heuristics |
| TRAK featurize | One small policy + one HDF5 demo | Projected gradient hash (allowing for nondeterministic CUDA — TBD whether we can pin this) |

### 5.2 Existing test suites preserved

`tests/` is migrated as-is into the new env layout. `run_tests.py --suite policy_doctor` must pass under `uv run --extra analysis`. The `mimicgen` and `cupid` suites become a single `sim` suite under `uv run --extra mimicgen`.

### 5.3 End-to-end replay

One small canonical pipeline run (e.g. `run_clustering` → `train_curated` → `eval_curated` on a 50-demo fixture) is captured before refactoring. At every phase boundary, replay it through the refactored code and diff each step's `result.json`.

### 5.4 Streamlit smoke test

A Selenium / Playwright script that boots both Streamlit apps against a golden experiment dir and verifies each tab renders without exception. Run at the end of each phase.

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Mujoco / robosuite / mimicgen don't install cleanly under uv | medium | high (blocks Phase 1) | Prototype env spec early in Phase 1; if blocked, keep conda fallback for sim envs and only migrate analysis env |
| Hidden import from `influence_visualizer` we miss | medium | medium | Grep before deletion; CI runs Streamlit smoke test |
| Pipeline step writes to a path we don't know about | medium | medium | At Phase 2 entry, add `pathlib` wrappers that log every disk write; review the log before deleting old paths |
| User-study deploy breaks during refactor | low | high | Deploy lives off `main`. Long-lived branch doesn't touch `main` until Phase 6 |
| Golden snapshots are stochastic (CUDA, threading) | medium | medium | Pin seeds; use `allclose` with documented tolerance; for unavoidably stochastic outputs (TRAK gradients) snapshot a derived stat (cluster id, top-k indices) |
| New layout is wrong in some way we don't see yet | low | high | Phase 2 ships behind one step (`run_clustering`) before the full migration. Iterate the design there. |

---

## 7. Resolved decisions

All seven Phase-0-blocking questions answered:

1. **`POLICY_DOCTOR_DATA` default**: `<repo_root>/data/` (env-var overridable). Gitignored.
2. **Robocasa env**: keep — RoboCasa is still active.
3. **Cupid env**: keep as a torch-2 env (the pre-torch-2 legacy is already gone). Required for robomimic / diffusion-policy experiments because of the older robosuite 1.2.0 + robomimic 0.2.0 pinning that conflicts with the mimicgen stack.
4. **InfEmbed package location**: leave at `third_party/cupid/third_party/infembed/` for now. Move only if it has zero callers outside `cupid/` after Phase 5 cleanup.
5. **Streamlit apps**: leave UI code alone. Refactor changes only the data-loading layer they call.
6. **Old artifacts**: `third_party/cupid/data/` stays on disk, frozen. Stop writing to it.
7. **Currently-modified files**: committed in the Phase-0 prep commit.

## 8. Remaining open questions (non-blocking)

To revisit during execution, not before Phase 0:

- **Streamlit smoke-test harness**: Playwright vs. Selenium vs. just-import-and-render-via-Streamlit's-test-client. Decide in Phase 0 when writing the smoke test.
- **TRAK gradient determinism**: whether we can pin TRAK projected gradients exactly across CUDA runs, or whether we snapshot a derived statistic. Decide in Phase 0 when writing golden snapshots.
- **`experiment-init` CLI surface**: minimal (just `name + baseline_ckpt`) or richer (template-based, with Hydra overrides). Decide in Phase 2.

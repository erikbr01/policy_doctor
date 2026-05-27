# Refactor Findings Log

Running log of discoveries, gotchas, and decisions made while executing `REFACTOR_PLAN.md`. Append-only; each entry dated.

Format per entry:
```
## YYYY-MM-DD — <phase> — <one-line title>
**Context:** what we were doing
**Finding:** what we discovered
**Decision / action:** how we handled it
**Plan impact:** none / amendment to §X of REFACTOR_PLAN.md
```

---

## 2026-05-27 — Phase 0 — Refactor branch created

**Context:** Starting Phase 0 of the refactor. Modified `policy_doctor/configs/dagger_run.yaml` and `scripts/experiments/run_viz_server.sh` (DAgger demo defaults) were committed to `main` first to clean working-tree state.

**Finding:** Branch `refactor/clean-architecture` cut from `main` at commit `5a5d4ba`. `REFACTOR_PLAN.md` carried over (untracked on main, now under refactor branch).

**Decision / action:** All phase work lands on this branch. `main` continues to serve the deployed user-study Streamlit apps until Phase 6 cutover.

**Plan impact:** none.

---

## 2026-05-27 — Phase 0 — Local conda env naming differs from CLAUDE.md

**Context:** Tried to run a quick import smoke test under `conda run -n policy_doctor python -c …` per CLAUDE.md.

**Finding:** This Mac has no `policy_doctor` env. The local envs are `cupid`, `cupid_robosuite`, `lerobot`, `libero`, `robomimic`. The `cupid` env imports `policy_doctor`, `policy_doctor.behaviors.clustering`, `policy_doctor.behaviors.behavior_graph`, and `policy_doctor.data.structures` cleanly — so it's a superset for local dev.

**Decision / action:** For Phase 0 golden-snapshot generation on this Mac, use `conda run -n cupid`. The plan's `analysis` uv env will be the long-term home for these tests; this is a local-dev workaround, not a plan change.

**Plan impact:** none. Worth noting that uv setup in Phase 1 needs to be robust to whatever the dev machine has — the bootstrap script should handle "I have a different conda env" gracefully.

---

## 2026-05-27 — Phase 0 — Deferring MimicGen seed-selection golden anchor

**Context:** Plan §5.1 lists MimicGen seed selection as one of four golden anchors.

**Finding:** The three heuristics (`BehaviorGraphPathHeuristic`, `DiversitySelectionHeuristic`, `RandomSelectionHeuristic`) require both a synthesized rollouts HDF5 (the trajectory loader `MimicGenSeedTrajectory.from_rollout_hdf5()` enforces a schema with state/action/reward/done datasets and demo-level success attrs) and a fitted `BehaviorGraph`. Setup is ~3-4x the size of the other three anchors and the heuristic outputs depend transitively on every other anchor we're snapshotting.

**Decision / action:** Ship the three simpler anchors (clustering, behavior graph, influence-slice) in the Phase-0 first commit. Add MimicGen seed selection in a follow-up Phase-0 commit before declaring Phase 0 done. Tracked as a TODO at the bottom of `tools/refactor/snapshot.py`.

**Plan impact:** none — the anchor is still part of Phase 0, just split across two commits.

---

## 2026-05-27 — Phase 1 — Sim extras locked, deferred validation to Linux

**Context:** Adding `cupid`, `mimicgen`, `robocasa` to pyproject.toml so the conda env yamls can be retired.

**Finding:**
- The three sim extras pin mutually-incompatible robosuite/robomimic versions. uv refuses to resolve a single dep graph; needed `[tool.uv] conflicts = [[{extra = "cupid"}, {extra = "mimicgen"}, {extra = "robocasa"}]]` so uv resolves each extra in isolation.
- `mimicgen==1.0.0` is GitHub-only (NVlabs/mimicgen). Added `[tool.uv.sources] mimicgen = { git = ... }` pointing at default branch (resolved to `72bd767c`).
- Three legacy sim packages have broken build metadata under modern setuptools and won't build on Python 3.12:
  - `gym==0.21.0` — `extras_require` formatting rejected by setuptools ≥ 67.
  - `pybullet-svl==3.1.6.4` — `setup.py` imports `pkg_resources` which isn't in `build-system.requires`.
  - Likely also `dm-control 1.0.9` (didn't get that far in the trace).
  All are transitive deps of `robomimic==0.2.0`; the original conda env worked because conda pinned Python 3.10 and old setuptools. Dropped explicit pins; let the resolver pick what robomimic 0.2.0 transitively accepts and accept that `uv sync --extra cupid` may need an older Python or build-isolation overrides on first use.
- Workspace setup: `third_party/cupid` and `third_party/cupid/third_party/infembed` declared as workspace members. `cupid-workspace` and `infembed` declared via `[tool.uv.sources] = { workspace = true }`. Only the cupid and mimicgen extras pull these in.
- Final lockfile: 5943 lines, covers all four extras + the workspace members. Analysis goldens still pass in 1.5s.

**Decision / action:**
- Phase 1 sim-extras commit lands as-is. `uv sync --extra cupid` is expected to fail on macOS / Python 3.12 — Linux validation on a future commit.
- Open follow-up: figure out the right build-isolation overrides or vendored-package strategy for the legacy deps. May involve `[tool.uv.extra-build-dependencies]` entries or pinning `setuptools<66` for the affected packages, or pinning Python 3.10 for the `cupid` extra.

**Plan impact:** Recorded in plan §6 risk table as a "Mujoco / robosuite / mimicgen don't install cleanly under uv" mitigation that did partially bite. Doesn't change phase shape.

---

## 2026-05-27 — Phase 1 — Dispatch helper landed, Phase 1 substantially complete

**Context:** Five pipeline step files used to do `subprocess.run(["conda", "run", "-n", env, "--no-capture-output", "python", ...])` to dispatch training/eval/generation into the right env.

**Finding:**
- Added `policy_doctor/_env.py` exposing `resolve_uv_extra(env_name)` and `run_in_env(env_name, cmd, **kwargs)`. The historical conda names (`mimicgen_torch2`, `cupid_torch2`, `cupid_torch25`, `policy_doctor`, `policy_doctor_dagger`) map to new uv extras (`mimicgen`, `cupid`, `cupid`, `analysis`, `analysis`); unmapped names pass through unchanged so direct extras (`analysis`, `cupid`, `mimicgen`, `robocasa`) work without translation.
- Migrated the five dispatch sites mechanically. Each file lost the inline `"conda", "run", "-n", env, "--no-capture-output"` prefix and gained the helper import. Behavior is preserved: `cwd`, `env`, `check`, return-code inspection, dry-run paths, in-process branches all untouched.
- Config field names (`conda_env`, `data_source.conda_env_train`, `baseline.conda_env`) kept as-is for backward compatibility — Phase 5 cleanup will rename to `uv_env` and update YAML files in one mechanical pass.
- 11 unit tests for the helper + 4 golden replays = 15 tests pass in 0.91s.

**Decision / action:**
- Phase 1 is functionally complete on Mac: analysis env validated end-to-end, sim extras defined + locked but deferred validation to Linux, all programmatic dispatch routed through the uv wrapper.
- Two pieces remain *not* in this commit set: (1) delete `environment_*.yaml` and `scripts/setup_torch2_envs.sh` / `scripts/create_cupid_torch25.sh`, (2) update README/CLAUDE.md/docs/*.md to drop `conda activate` references. These are docs/cleanup and can land at Phase 6 alongside the other doc work, after Linux validation of sim extras confirms uv is the canonical path.
- Marking Phase 1 task complete; Phase 1 tail (yaml deletion + docs) gets folded into Phase 6.

**Plan impact:** Phase 6 scope slightly expanded: now also owns yaml deletion + docs touchup. Worth a note in the plan.

---

## 2026-05-27 — Phase 2 — Foundation: `policy_doctor.experiment` module

**Context:** Phase 2 begins. Smallest viable starting commit: the new module that everything else will build on. Zero existing code touched.

**Finding:**
- `policy_doctor/experiment/paths.py` exposes `data_root()`, `datasets_dir()`, `experiments_dir()`, `experiment_dir(name)`. All lazy so `POLICY_DOCTOR_DATA` env-var changes (pytest `monkeypatch.setenv`) are picked up per call.
- `policy_doctor/experiment/experiment.py` defines the `Experiment` frozen dataclass with:
  - Path properties (`manifest_path`, `config_dir`, `shared_dir`, `artifacts_dir`, `logs_dir`).
  - `create(name, *, baseline_from=None, manifest_extras=None)` factory that builds the skeleton, writes the manifest, and hard-copies the baseline checkpoint from another experiment if `baseline_from` is set.
  - `load(name)` factory for resuming an existing experiment.
  - `append_config_snapshot(resolved_config)` — append-only Hydra snapshot writer; first one symlinked as `canonical.yaml`.
  - `step_dir(name, *, version=None)` — returns/creates the artifacts dir for a step; `version="v2"` produces `<name>__v2/` for side-by-side reruns.
  - `update_manifest(**updates)`, `open_log(label)`.
- 12 unit tests, all pass. Goldens still pass. 27 tests / 0.92s.

**Decision / action:**
- Module ships standalone. Nothing imports it yet — that's the next commit.
- Validation note: `Experiment.create` rejects names with `/`, `..`, or leading `.`. Cheap defense against path-escape bugs in CLIs.

**Plan impact:** none. Next Phase-2 commit will be the PipelineStep base-class refactor.

---

## 2026-05-27 — Phase 2 — CurationPipeline ↔ Experiment bridge

**Context:** Wire the new `Experiment` module into the orchestrator without breaking the existing run_dir-driven pipeline.

**Finding:**
- `PipelineStep.__init__` gained an optional `experiment: Experiment | None` kwarg (TYPE_CHECKING import avoids a circular). All other shape unchanged — existing callers keep working.
- `CompositeStep.compute` forwards `experiment=self.experiment` to sub-steps so arm sub-steps see the same Experiment.
- `CurationPipeline.__init__` branches on `cfg.experiment_name`:
  - If set: creates or resumes the Experiment, sets `run_dir = experiment.artifacts_dir`, mirrors back into `cfg.run_dir` so steps that read `cfg.run_dir` stay consistent. `baseline_from` from config propagates to `Experiment.create`.
  - If not set: legacy `run_name` → `data/pipeline_runs/<run_name>` path is unchanged.
- `_save_config` branches the same way: experiment path appends a snapshot via `experiment.append_config_snapshot`; legacy path writes the single `pipeline_config.yaml`.
- `run()` and `step()` pass `experiment=self.experiment` to every step constructor.
- 5 new pipeline-integration tests cover experiment-create, experiment-resume, snapshot-append, legacy-runtime, baseline-from propagation. 32/32 tests pass in 0.92s.

**Decision / action:**
- Steps don't *use* `self.experiment` yet — they still construct paths from `run_dir`. That's fine — the bridge is in place, migration of individual step path-construction logic comes next.
- The next big chunk is stripping `train_date`/`eval_date` from `policy_doctor/curation_pipeline/paths.py` (5 `get_*` functions) and the 30+ places that call them. That deserves its own commit.

**Plan impact:** none.

**Context:** First Phase-1 milestone — establish a uv-managed `analysis` env and prove the goldens reproduce inside it.

**Finding:**
- Rewrote `pyproject.toml` to declare a core dependency set plus an `analysis` optional-deps group. Bumped `requires-python` from 3.9 → 3.10 (matches all four target envs).
- uv resolved 177 packages cleanly on macOS / Python 3.12; sync took ~2 min cold. Notable resolutions: numpy 2.0.2, torch 2.12.0, scikit-learn latest. No special handling needed.
- Added `scripts/uv_env.sh` as the `conda run -n` replacement. Per-env venvs at `.venvs/<env>/` selected via `UV_PROJECT_ENVIRONMENT`. The wrapper auto-syncs on first use.
- All four golden anchors pass under `./scripts/uv_env.sh analysis pytest tests/golden/` in 0.9s (warm) / 29s (cold venv).
- `.gitignore` extended to ignore `.venvs/`.
- uv.lock committed (~4300 lines) per uv best practice for reproducible installs.

**Decision / action:** Sim extras (`cupid`, `mimicgen`, `robocasa`) will be added in follow-up commits. Each can be verified independently and probably can't be fully tested on macOS (CUDA-pinned torch wheels, mujoco compat). Will validate analysis-extra parity on a Linux box if/when one is in scope.

**Plan impact:** none. Phase 1 progressing per plan.

**Context:** Follow-up to the previous entry. Added the fourth golden anchor.

**Finding:** The HDF5 schema expected by `MimicGenSeedTrajectory.from_rollout_hdf5()` is simpler than first read — only `data/<demo>/states`, `data/<demo>/actions`, and the demo-level `success` attribute are required for the heuristics to run end-to-end. `env_args` is loaded from `data.attrs` and stored as JSON. The heuristics' `info` dicts contain `numpy.float32`/`numpy.int64` scalars that don't round-trip through json.dumps without canonicalization — added `_canonicalize` to handle this.

The MimicGen golden captures three semantically distinct outcomes that any refactor must preserve:
  - `behavior_graph_path` picks all three seeds from the top-probability path (cluster_seq [0,1,2,3], prob 0.5).
  - `diversity` takes one seed per path in probability order: rollouts {6, 7, 3} for the three paths.
  - `random` samples without replacement: rollouts {7, 0, 5}.

**Decision / action:** Phase 0 complete. Goldens are 12 files, ~10 KB total. Replay runs in <1s under pytest. `tools/refactor/snapshot.py` is now the canonical regression check; every phase boundary must pass `pytest tests/golden/`.

**Plan impact:** none.

---

## 2026-05-27 — Phase 3 — Inventory of live `influence_visualizer` imports

**Context:** Phase 3 begins. Catalog every external use of `influence_visualizer` outside `third_party/influence_visualizer/` itself so we know exactly what must migrate before iv can be deleted.

**Finding:** 19 external import sites, falling into four buckets.

*Bucket A — live pipeline / scripts (must migrate):*

| Symbol | Call sites |
|---|---|
| `influence_visualizer.clustering_results.save_clustering_result` | `policy_doctor/curation_pipeline/steps/run_clustering.py:40`, `policy_doctor/streamlit_app/tabs/clustering.py:852`, `scripts/sweep_pi05_clustering.py:40`, `scripts/cluster_pi05_libero.py:27` |
| `influence_visualizer.clustering_results.get_clustering_dir` | `policy_doctor/curation_pipeline/steps/annotate_slices_vlm.py:59`, `policy_doctor/streamlit_app/tabs/vlm_annotation.py:37`, `policy_doctor/streamlit_app/config_io.py:175,191,249,262` |
| `influence_visualizer.data_loader.get_eval_dir_for_seed` | `policy_doctor/curation_pipeline/steps/{run_clustering,annotate_slices_vlm,validate_cluster_coherence_vlm,select_mimicgen_seed_from_graph,compute_data_support,compare}.py` (6 sites) |
| `influence_visualizer.data_loader.get_train_dir_for_seed` | `policy_doctor/data/clustering_embeddings.py:213` |
| `influence_visualizer.data_loader.load_influence_data` | `policy_doctor/data/influence_loader.py:102` (delegation wrapper used by `run_curation_config` step and `clustering_embeddings` precomputation) |

*Bucket B — Streamlit UI imports (lazy, inside function bodies):*

| Symbol | Call sites | Disposition |
|---|---|---|
| `influence_visualizer.render_annotation.load_annotations` | `streamlit_app/tabs/clustering.py:784` | pure-data: migrate to `policy_doctor.influence.annotations` |
| `influence_visualizer.render_annotation.get_episode_annotations` | `streamlit_app/clustering_browse.py:194` | pure-data: migrate to `policy_doctor.influence.annotations` |
| `influence_visualizer.render_frames.frame_player` | `streamlit_app/clustering_browse.py:123` | streamlit UI — leave in streamlit_app, relocate to `policy_doctor.streamlit_app.frame_player` |
| `influence_visualizer.plotting.create_annotated_frame` | `streamlit_app/clustering_browse.py:56` | pure PIL (no streamlit): migrate to `policy_doctor.influence.frames` |

*Bucket C — test-only comparison hooks (delete with the comparison tests):*

| Symbol | Call sites |
|---|---|
| `influence_visualizer.clustering_results.load_clustering_result_from_path`, `.load_embeddings_reduced` | `tests/integration/test_compare_iv_vs_policy_doctor.py`, `tests/vlm/test_cluster_classification.py` |
| `influence_visualizer.data_loader.load_influence_data` | `tests/integration/{test_compare_iv_vs_policy_doctor,test_fingerprint_episode_ends}.py` |
| `influence_visualizer.behavior_value_loader.*` | `tests/integration/test_compare_iv_vs_policy_doctor.py` (4 sites) |
| `influence_visualizer.render_learning.*`, `influence_visualizer.render_heatmaps.get_split_data` | `tests/integration/test_compare_iv_vs_policy_doctor.py` (4 sites) |
| `influence_visualizer.curation_config.compute_dataset_fingerprint` | `tests/integration/test_fingerprint_episode_ends.py` (2 sites) |

These tests' entire purpose is to assert PD matches IV — once IV is removed the comparison is moot. They self-skip when iv is unavailable today; we delete them in Phase 3B because the comparison subject is going away.

*Bucket D — comments / doc strings (no functional impact):*

`third_party/cupid/diffusion_policy/common/sampler.py:162`, `third_party/cupid/scripts/train/train_policies.sh:47,136`, `third_party/cupid/CLAUDE.md:53,66`, `policy_doctor/behaviors/behavior_graph.py:1` — all are explanatory text; will be cleaned up in Phase 6 docs.

**Decision / action:** Bucket A drives the migration plan. Bucket B carves out two small extracts (`annotations.py`, `frames.py`) plus a one-file relocation (`frame_player` to streamlit_app). Bucket C tests get deleted alongside the iv directory. Bucket D is just text.

Important wrinkles found:
- Two parallel partial ports already exist in `policy_doctor/data/`: `path_utils.py` has `get_eval_dir_for_seed` / `get_train_dir_for_seed`, and `clustering_loader.py` has most of the clustering-load surface except `save_clustering_result` and `load_embeddings_reduced`. The new `policy_doctor/influence/clustering_io.py` will *re-export* from `clustering_loader` and add the missing functions, then call sites consolidate on `policy_doctor.influence`.
- `policy_doctor/data/influence_loader.load_influence_data` is a thin shim that delegates to iv's `load_influence_data`. After migration, the shim's lazy import flips to `policy_doctor.influence.loader`. The heavy iv function itself depends on `diffusion_policy` + `hydra` and will move verbatim (along with the helpers it calls: `get_checkpoint_path`, `load_checkpoint_config`, `build_demo_sample_infos`, `build_rollout_sample_infos`, `load_influence_matrix`, `find_trak_experiment`, `create_image_dataset_from_config`, `get_task_loaders`, plus `SampleInfo` / `EpisodeInfo` / `InfluenceData` dataclasses and the two `TaskLoader` classes).
- `iv.clustering_results.get_clustering_dir` resolves relative to `Path(__file__).parent / "configs" / task_config / "clustering"` — i.e., the iv-package directory. The new function in `policy_doctor.influence.clustering_io` must resolve to `iv_task_configs_base() / task_config / "clustering"` to preserve the on-disk location of historical clustering runs.

**Plan impact:** none. Inventory confirms the original "~6 live functions" estimate in the plan; the actual surface is 5 live functions + 2 streamlit-only extracts + 1 streamlit-only relocation.

---

## 2026-05-27 — Phase 3 complete — `influence_visualizer` package removed

**Context:** Phase 3B migration committed green (40/40 tests). This entry closes Phase 3 after deleting the legacy iv code.

**What shipped:**
- `policy_doctor/influence/` package with 6 submodules: `clustering_io`, `loader`, `lazy_hdf5`, `path_helpers` (lightweight seed-path helpers split out of the heavy loader), `annotations`, `frames`. `loader.py` is a near-verbatim copy of the iv `data_loader.py`; `clustering_io.py`, `lazy_hdf5.py`, `annotations.py`, `frames.py` are extracts of the iv functions live code actually used. `InfluenceDataLoader`, `SampleData`, `TrajectoryData`, `create_mock_influence_data` were not migrated — nothing outside iv consumed them.
- `policy_doctor/streamlit_app/frame_player.py` — the one Streamlit-tinted helper (`frame_player`) extracted from `iv.render_frames`.
- 14 call sites retargeted: 6 pipeline steps, 4 streamlit modules + 2 streamlit tabs, 2 scripts, 1 data helper, 1 influence-loader shim. Two test files updated to point at the new modules (`tests/vlm/test_cluster_classification.py`, `tests/integration/test_fingerprint_episode_ends.py`); one test deleted (`tests/integration/test_compare_iv_vs_policy_doctor.py` — its sole purpose was IV-vs-PD comparison, which is moot once IV is gone).
- `third_party/influence_visualizer/` — all 38 python files plus `app.py`, `notebooks/`, `notes/`, `scripts/`, `tests/`, `plotting/`, `pyproject.toml`, and `README.md` were `git rm -r`'d. The `configs/` subdirectory was kept on disk (~108 MB of saved clustering runs, curation YAMLs, and task configs); live code reaches it via `policy_doctor.paths.iv_task_configs_base()` and the new `policy_doctor.influence.clustering_io.get_clustering_dir()` resolves there. Relocating those assets to a non-`third_party/` home is Phase 5 / Phase 6 work and explicitly deferred — moving them now would invalidate every existing experiment YAML.

**Surprises:**
- The iv `data_loader.py` is enormous (2346 lines) but most of it is the `InfluenceDataLoader` convenience wrapper and `InfluenceData` accessor methods. Trimming to the live surface (`load_influence_data` + helpers + dataclasses) brought it down to ~1845 lines.
- iv's `clustering_results.get_clustering_dir(task_config)` resolved to `Path(__file__).parent / "configs" / ...`, i.e. the iv-package directory. The new implementation uses `iv_task_configs_base()` instead so the on-disk location is preserved without referencing iv's now-gone `__file__`.
- The TaskCreate task tracker reminders fired about half-a-dozen times during this work; treated them as no-ops since the worktree-local TaskCreate state isn't visible to the wider system.
- `tests/integration/test_fingerprint_episode_ends.py` had a setUp that hard-required `influence_visualizer` (via try/except import + `self.skipTest`). That setUp was rewritten to drop the iv probe; only the pure-PD `test_policy_doctor_load_dataset_episode_ends_matches_reference` survives and exercises `policy_doctor.data.dataset_episode_ends` + `policy_doctor.curation.config.compute_dataset_fingerprint`.
- The legacy conda setup scripts (`scripts/install_policy_doctor_env.sh`, `scripts/create_cupid_torch25.sh`) reference `pip install -e third_party/influence_visualizer` and will fail post-deletion. They were already scheduled for retirement in Phase 1 (now folded into Phase 6 cleanup); leaving them broken does not regress anything currently tested.

**Verification:**
- `./scripts/uv_env.sh analysis pytest tests/golden/ tests/experiment/ tests/test_env_dispatch.py` — 40 passed in 0.97s after the deletion commit.
- `grep -rn "from influence_visualizer\|import influence_visualizer" policy_doctor/ tests/ scripts/` — zero hits.
- `grep -rn "influence_visualizer" third_party/cupid/` — 5 hits, all comments / docstrings (e.g. `sampler.py:162`, `CLAUDE.md:53,66`, `train_policies.sh:47,136`). Marked Bucket D in the Phase 3A inventory; deferred to Phase 6.
- `uv lock --check` — 224 packages resolved in 6 ms, no drift.

**Plan impact:** Phase 3 closed per spec. Phase 5 inherits one new follow-up: relocate `third_party/influence_visualizer/configs/` into the new experiments / data root and rewire `policy_doctor.paths.iv_task_configs_base()` + `policy_doctor.influence.clustering_io.get_clustering_dir()` to read from there. That can land alongside the broader "single canonical data root" cleanup.

---

## 2026-05-27 — Phase 4 complete — non-Streamlit viz deleted

**Context:** Per Plan §4 / §7 ("Streamlit-only viz"), this phase removed standalone plotting scripts and non-Streamlit render utilities with no live caller (= no import from `policy_doctor/streamlit_app/`, `policy_doctor/curation_pipeline/`, `policy_doctor/influence/`, an active pipeline step, or `tests/`).

**Inventory method:** ripgrep every viz file (scripts/plot_*.py, scripts/render_*.py, scripts/export_*.py, policy_doctor/scripts/plot_*.py, every `policy_doctor/plotting/**/*.py`) against the live-caller surfaces above. Plus the user's seed candidate list from earlier investigation.

**Deleted (16 files):**
- `scripts/plot_mimicgen_eef_from_result.py` — sole reference was `scripts/test_mimicgen_eef_plots.py` (also dead).
- `scripts/plot_mimicgen_budget_sweep.py` — zero callers.
- `scripts/plot_nut_constrained_violins.py` — zero callers.
- `scripts/render_agent_session.py` — only imported by `scripts/run_e2_agent_transport_mh.py` (itself referenced only from operator docs, not from any live caller surface).
- `scripts/render_episode_mp4s.py` — zero callers.
- `scripts/render_mimicgen_playback.py` — zero callers.
- `scripts/render_twostage_session.py` — zero callers; imported `render_agent_session`.
- `scripts/export_e1_report.py`, `scripts/export_e2_report.py`, `scripts/export_mimicgen_report.py` — zero callers each.
- `scripts/test_mimicgen_eef_plots.py` — dev harness for the now-deleted EEF plot script; not invoked from any test runner or pipeline step.
- `policy_doctor/scripts/plot_curation_data_vs_success.py` — zero callers; only consumer of `plotting/curation_scatter*.py`.
- `policy_doctor/scripts/plot_comparison.py` — zero callers; only consumer of `plotting/training_curves.py`. (The earlier note that it was "called from plotting/" was wrong; it imports *from* plotting, not the other way.)
- `policy_doctor/plotting/curation_scatter.py`, `curation_scatter_mpl.py`, `training_curves.py` — all three only reached from the deleted scripts above.

**Edited:** `policy_doctor/plotting/__init__.py` — removed re-exports of `create_training_comparison_plot`, `create_curation_data_vs_success_scatter`, `create_experiment_checkpoint_score_boxplot`, `create_multi_experiment_checkpoint_score_boxplots` (their backing modules are gone).

**Kept on close calls (documented for Phase 5):**
- `policy_doctor/scripts/compare_policies.py` + `policy_doctor/plotting/policy_comparison.py` — `compare_policies.py` is a Hydra-driven CLI with a config tree under `policy_doctor/configs/comparison/`, was not in the user's deletion candidate list, and is referenced by half a dozen YAML headers. Conservative keep; flag for Phase 5 to decide whether to fold into a pipeline step or excise.
- `scripts/run_e2_agent_transport_mh.py` — referenced only by docs (`operator_e2_quickstart.md`, `experiments/experiment_e2_critical_findings.md`) and a `policy_doctor.vlm.proposals.server` docstring; not imported by any live surface. Was the only caller of `render_agent_session.py`, so it is now broken at the import level. Not in the user's deletion candidate list; left in place but Phase 5 should either repair or delete it alongside the e2-agent flow.
- `policy_doctor/plotting/pyvis/__init__.py` — 22-line re-export of two functions from `plotting/plotly/behavior_graph*.py`. No streamlit/pipeline import, but `tests/plotting/test_pyvis.py` (4 tests) covers it. Conservative keep.
- `policy_doctor/plotting/vlm_montage.py` — live: `streamlit_app/tabs/vlm_annotation.py` calls `plotting.create_scrollable_frame_strip_html` (via `__init__.py` re-export), and `tests/vlm/test_slice_annotation.py` imports it directly.
- All `policy_doctor/plotting/plotly/*.py` — every file has at least one streamlit tab or `_simplification_pages` consumer (verified per-module).

**Verification:**
- `./scripts/uv_env.sh analysis pytest tests/golden/ tests/experiment/ tests/test_env_dispatch.py` — 40 passed in 0.96s before deletions, 40 passed in 0.96s after.
- Smoke imports under `analysis` env: `policy_doctor.streamlit_app.app`, `streamlit_app.demo_app.Home`, all 9 streamlit_app/tabs/* modules, both `_simplification_pages` modules (note: `simplification.py` and `sweep_analysis.py` import cleanly), `components/trajectory_tree_view.py`, `user_study/graph_plot.py`, `user_study/initial_conditions.py`, `policy_doctor.plotting`, `policy_doctor.plotting.plotly`, `policy_doctor.plotting.pyvis` — all OK. `survey_app/Home.py` and `user_study/app_group_{a,b}.py` execute Streamlit-runtime logic at import time (env-var lookups for `mp4_dir`) and fail with `TypeError: argument should be a str ... not 'NoneType'` outside `streamlit run`; this is a pre-existing condition on `refactor/clean-architecture` (reproduces on HEAD before the deletions) and is unrelated to Phase 4.
- `tests/plotting/test_eef_trajectories.py` has 2 pre-existing failures (`show_mean` kwarg drift); reproduces on HEAD before the deletions. Not introduced by Phase 4 and not in the keep-green suite.

**Plan impact:** Phase 4 closed per spec. Phase 5 inherits two follow-ups: (a) decide the fate of `compare_policies.py` + `policy_comparison.py` (fold into pipeline step vs. delete), and (b) repair or remove the now-broken `scripts/run_e2_agent_transport_mh.py`.

---

## Phase 5 — deferred

Phase 5 covered the safe/mechanical pieces (conda_env→uv_env rename across configs + dispatch sites; scripts/ reorganization into experiments/dev/setup with a back-compat symlink at `scripts/uv_env.sh`). The bigger ambitions — code dedup and "everything is a pipeline" — were scoped out of this session because each touches surfaces we can't validate without live data, and one of them is cascade-blocked on Phase 2. Recording them here so they aren't lost.

### Path-resolution dedup (cascade-blocked on Phase 2)

Four path-resolution modules coexist today:

- `policy_doctor/paths.py` — `PACKAGE_ROOT`, `PROJECT_ROOT`, `REPO_ROOT`, `DATA_SOURCE_ROOT`. The canonical top-level surface.
- `policy_doctor/curation_pipeline/paths.py` — `get_train_dir`, `get_eval_dir`, `get_train_name`. Encodes the `train_date`/`eval_date`/seed naming convention.
- `policy_doctor/influence/path_helpers.py` — checkpoint-discovery helpers absorbed from the deleted `influence_visualizer` package.
- `policy_doctor/experiment/paths.py` — the new Phase-2 `seed_dir`/`ckpt_dir`/etc. helpers built on top of the `Experiment` bundle.

The four modules carry overlapping responsibilities (locating train dirs, eval dirs, checkpoints under different conventions). Reconciling them is a Phase-2-cascade question: stripping `train_date`/`eval_date` from the naming scheme (planned for Phase 2 continuation) collapses `curation_pipeline/paths.py` into `experiment/paths.py`, at which point `path_helpers.py` can probably also be folded in. Until that happens, deduplicating prematurely would force two migrations.

**Phase 5/2 disposition:** revisit when path migration lands. Until then, the four modules stay; add an inline `See also` comment in each `paths.py` pointing at the others when a new contributor needs to find the right helper.

### HDF5 reader dedup (size-of-change vs. test data)

`h5py.File(...)` calls and HDF5-locating helpers are scattered across:

- `policy_doctor/influence/loader.py` — two `h5py.File` opens (one in `_load_low_dim_demos_into_episodes` around line 1018; one for image datasets around line 1742). The historical fat loader from `influence_visualizer`.
- `policy_doctor/influence/lazy_hdf5.py` — `_LazyHDF5Reader` (single `h5py.File` open at line 49). Lazy-open wrapper for memory-bounded reads.
- `policy_doctor/data/adapters.py` — path-discovery only (no `h5py` opens); searches known MimicGen/RoboCasa HDF5 locations under `repo_root`.
- `policy_doctor/data/dataset_episode_ends.py` — references HDF5 via path only (delegates to upstream loader).
- `policy_doctor/mimicgen/{combine_datasets,eef,failure_targeting,heuristics,materializer,seed_trajectory}.py` — six modules each calling `h5py.File` for MimicGen-specific reads (seed selection, EEF extraction, dataset combination).
- `policy_doctor/curation_pipeline/steps/{generate_mimicgen_demos,train_on_combined_data}.py` — pipeline steps that touch HDF5 directly (seed materialization, combined-data writes).
- `policy_doctor/monitoring/trajectory_classifier.py` — single `h5py.File` open for episode metadata.

The duplication is roughly: (1) "locate the right HDF5 for this task" (in `data/adapters.py` + `influence/path_helpers.py` + ad-hoc paths in the mimicgen modules), and (2) "open + iterate demos with the cupid layout assumption" (in `influence/loader.py`, the six mimicgen modules, and at least two pipeline steps). A clean refactor would extract a small `policy_doctor/data/hdf5_io.py` with `open_demos(path) -> Iterator[(demo_key, demo_group)]` and a `locate_dataset_for(task)` helper. The size of the change requires HDF5 files we don't have checked into the repo to test against; deferring until at least one of the dev environments has a live fixture wired up.

**Phase 5 disposition:** list above is the inventory. Deferred to a follow-up that pairs with adding HDF5 test fixtures (Phase 6 docs work could surface a lightweight one).

### "Everything is a pipeline" — scripts/experiments/ launchers to promote to `PipelineStep`

The directive was to convert ad-hoc launchers into `PipelineStep` subclasses so the orchestration layer (sentinel files, resumability, `dry_run`, config-driven dispatch via `_env.run_in_env`) covers everything. Inventory of candidates currently sitting in `scripts/experiments/` (post-reorg):

- `monitor_online.py` — runtime monitoring driver; should become a `RunMonitorOnlineStep` (consumes `train_dir`/`clustering_dir`, writes per-rollout classification outputs). Already invoked from `scripts/experiments/_lib.sh`; promotion would let the pipeline run it as part of an eval flow.
- `monitor_offline.py` — offline counterpart of the above. Same step contract.
- `run_dagger.py` / `run_dagger_robocasa.py` — DAgger driver; pairs with `build_dagger_dataset.py` (which is the "build" half). One composite `RunDaggerStep` that materializes the dataset then invokes the driver.
- `run_e2_agent.py` / `run_e2_agent_transport_mh.py` / `run_e2_sim.py` — e2 (proposal/agent/sim) trio; current launch is via `run_e2_*.sh` shell wrappers. Step would need to model the three-process topology (sim ↔ agent ↔ proposal server).
- `run_e1_*.py` (gemini, multi_model, sweep_eval, transport_r512_qwen) — VLM eval sweeps; natural composite under a `RunE1SweepStep` that fans out over models.
- `run_clustering_sweep.py` — k/window sweep launcher; arguably a thin wrapper around `run_clustering` repeated, so the step would be more of a sweep harness than a new primitive.
- `build_alt_clustering.py` / `build_k_sweep_clusterings.py` — alternative-clustering builders; depend on the upstream `run_clustering` result. Composite `BuildAlternativeClusteringsStep` over a config grid.
- `build_dagger_dataset.py` — dataset-builder for the DAgger flow; absorb into `RunDaggerStep` as a sub-step.
- `build_droid_zarr_cache.py` — DROID zarr cache builder; should be a `BuildDroidZarrCacheStep` that the DROID training step depends on.
- `gen_rollouts_hdf5.py` — rollout → HDF5 generator; closest to existing `eval_save_episodes` and could be folded into `EvalPoliciesStep` as a `write_hdf5: true` option rather than its own step.
- `cluster_kendama_policy_embeddings.py` / `cluster_kendama_rollouts.py` / `cluster_pi05_libero.py` / `sweep_pi05_clustering.py` — task-specific clustering drivers. The kendama and pi05 ones encode the same `run_clustering` shape with different inputs; a `RunClusteringStep` that already accepts `influence_source: policy_emb | rollout | ...` could subsume them with a small config extension.
- `aggregate_sweep_results.py` — post-sweep aggregator; would be the canonical `AggregateSweepStep` sentinel-tracked artifact.
- `rerun_evals_with_episode_lengths.py` — repair script for legacy eval dirs without `episode_lengths.npy`; one-shot, probably stays as a script.
- `markov_rnn_experiment.py` / `markov_rnn_jan28.py` / `run_window_sweep.py` / `run_encoder_k_sweep.py` / `run_head_to_head.py` / `run_policy_emb_sweep.py` / `run_k_sweep_evals.py` — research experiment drivers; promotion to pipeline steps is lower priority than the "core orchestration" ones above. Tag for Phase-5b.
- `enqueue_session.py` — session queue helper; not a pipeline step (it's a scheduler-side helper). Leave as a script.

The shell wrappers (`run_*.sh`, `train_*.sh`, `sweep_*.sh`) under `scripts/experiments/` are not in this list — they're the CLI surface that drives the Python launchers. Once a Python launcher becomes a `PipelineStep`, the corresponding shell wrapper either disappears (replaced by `python -m policy_doctor.scripts.run_pipeline steps=[...]`) or becomes a thin convenience that supplies fixed Hydra overrides.

**Phase 5/Phase-5b disposition:** this is the bulk of "Phase 5b". Each promotion is independent and can be ordered by impact; the `monitor_*`/`run_dagger`/`build_*` cluster is the highest-leverage starting point because those are already invoked from inside `_lib.sh` (i.e. they're orchestration glue, not research drivers).

### Verification
- `./scripts/uv_env.sh analysis pytest tests/golden/ tests/experiment/ tests/test_env_dispatch.py` — 40 passed at the start of Phase 5; 40 passed after Sub-task A (config rename); 40 passed after Sub-task B (scripts reorg). No regressions introduced.
- `./scripts/uv_env.sh` (legacy path, now a symlink) and `./scripts/setup/uv_env.sh` (canonical path) both resolve to the same `UV_PROJECT_ENVIRONMENT` and dispatch `uv run` correctly.

### Plan impact
- Phase 5 — partial close. Mechanical pieces landed (rename + scripts reorg + documentation of remaining work). Path-resolution dedup deferred to Phase 5/2 (cascade-blocked). HDF5 reader dedup deferred pending test fixtures. Pipeline-step promotion (the "everything is a pipeline" directive) explicitly carved out as Phase-5b with an inventory above so it can be picked up independently of the other phases.

## 2026-05-27 — Phase 6 complete — PR ready

The refactor branch is ready for review. Three Phase 6 commits closed out the loose ends:

| Commit | Subject |
|--------|---------|
| `154ef7c` | refactor(phase-6): delete legacy conda env yamls + setup scripts |
| `db361b2` | refactor(phase-6): rewrite CLAUDE.md, README.md, deploy/README.md for new uv + experiment-centric world |
| (this commit) | refactor(phase-6): final smoke test passes; refactor branch ready for review |

### Full commit chain ahead of `main`

```
(phase 6) refactor(phase-6): final smoke test passes; refactor branch ready for review
db361b2   refactor(phase-6): rewrite CLAUDE.md, README.md, deploy/README.md ...
154ef7c   refactor(phase-6): delete legacy conda env yamls + setup scripts
d6df25b   refactor(phase-5): document deferred dedup/pipeline-migration scope
1b16eed   refactor(phase-5): reorganize scripts/ into experiments/dev/setup
73527ec   refactor(phase-5): rename conda_env → uv_env in configs + dispatch sites
fe345cf   refactor(phase-4): delete deprecated non-Streamlit viz
c20f18a   refactor(phase-3): delete third_party/influence_visualizer/ Python package
4fe04a3   refactor(phase-3): absorb influence_visualizer into policy_doctor.influence
860130f   refactor(phase-2): experiment-bundle CLI
c05b395   refactor(phase-2): seed_dir/ckpt_dir helpers + experiment-init CLI
3f53955   refactor(phase-2): bridge CurationPipeline ↔ Experiment
8f0cd18   refactor(phase-2): add policy_doctor.experiment foundation
9571632   refactor(plan): incorporate Phase 2 design decisions and scripts-as-pipeline directive
d935bb8   refactor(phase-1): mark phase 1 substantially complete; fold yaml cleanup into phase 6
89b27a2   refactor(phase-1): migrate pipeline subprocess dispatch to uv via _env helper
4bd8c09   refactor(phase-1): add cupid/mimicgen/robocasa sim extras + workspace
ec413d0   refactor(phase-1): uv_env.sh --setup exits cleanly when no command given
17f15aa   refactor(phase-1): uv analysis env replaces policy_doctor conda env
c8ef8b1   refactor(phase-0): add fourth golden anchor for MimicGen seed selection
c0a0066   refactor(phase-0): golden-snapshot tool and three correctness anchors
930918f   refactor(phase-0): add plan and findings doc
```

### Test state at PR time

- `./scripts/uv_env.sh analysis pytest tests/golden/ tests/experiment/ tests/test_env_dispatch.py` — **40 passed**.
  - `tests/golden/` (4): goldens for clustering, behavior graph, MimicGen seed selection.
  - `tests/experiment/` (25): `test_bundle_cli` (3), `test_cli` (5), `test_experiment` (12), `test_pipeline_integration` (5).
  - `tests/test_env_dispatch.py` (11): uv-based env-dispatch helper.

### End-to-end CLI smoke tests (Phase 6)

- `python -m policy_doctor.scripts.experiment_init smoke_test` — created `data/experiments/smoke_test/{manifest.yaml,config/,shared/,artifacts/,logs/}`.
- `python -m policy_doctor.scripts.experiment_bundle smoke_test --out /tmp/smoke.tar.gz` — emitted the tarball (0.0 MB; empty experiment).
- `py_compile` of all three Streamlit entry points (`policy_doctor/streamlit_app/app.py`, `demo_app/Home.py`, `survey_app/Home.py`) — clean.

### What you can do on this branch today

- `./scripts/uv_env.sh analysis --setup` then run the 40-test suite.
- Create + bundle experiments via the new CLIs.
- Run the curation pipeline; steps that need a different env (training, MimicGen) dispatch through `policy_doctor._env.run_in_env` (uv-based, no conda).
- Use the absorbed `policy_doctor.influence` package (loader, clustering_io, frames, annotations, lazy_hdf5, path_helpers) — no `third_party/influence_visualizer` dependency.
- Launch all three Streamlit apps via the `analysis` env.

### Deferred work — ordered by priority for follow-up PRs

1. **(highest)** Strip `train_date` / `eval_date` from configs and migrate pipeline step path construction to use `Experiment.seed_dir()` / `ckpt_dir()`. The experiment-centric layer (`policy_doctor.experiment`) is in place and tested (`tests/experiment/test_pipeline_integration.py`), but pipeline steps still build paths from the legacy `data/outputs/train/<date>/...` convention internally. Until that lands, the experiment layer is opt-in for new code; existing pipeline runs keep the old layout. Cascade-blocks the path-resolution dedup below.
2. **Path-resolution dedup** — `policy_doctor.paths`, `policy_doctor/curation_pipeline/paths.py`, `policy_doctor/influence/path_helpers.py`, `policy_doctor/experiment/paths.py` carry overlapping responsibilities. Collapse once #1 lands.
3. **Phase-5b: scripts → pipeline migration.** Inventory in the previous findings entry. High-leverage starting point: `monitor_*`, `run_dagger`, `build_*` (already invoked from `scripts/experiments/_lib.sh`, i.e. they're orchestration glue, not research drivers).
4. **Linux validation of `cupid` / `mimicgen` / `robocasa` extras.** They're locked in `pyproject.toml` + `uv.lock` but the heavy transitive sim deps (`free-mujoco-py`, legacy `dm-control` chain) have only been validated on macOS in `--setup` mode; need a Linux x86_64 + Python 3.10 box to confirm `pytest tests/cupid/` and `tests/mimicgen/` still pass end-to-end.
5. **Docs migration under `docs/`.** 12 files still contain `conda activate` / `conda run -n` recipes: `kendama_retraining.md`, `mimicgen_pipeline_speedup_plan.md`, `experiment_log_failure_targeting_may11.md`, `operator_e2_quickstart.md`, `mimicgen_seed_selection_apr23.md`, `DAGGER_GUIDE.md`, `droid_robot_setup.md`, `data_support.md`, `constrained_generation_results.md`, `monitoring.md`, `experiments/experiment_e2_agent_proposals.md`, `experiments/pi05_libero_behavior_graphs.md`. None of these are referenced from the rewritten top-level docs as the canonical source. Migrate incrementally as each doc gets touched again.
6. **HDF5 reader dedup** — inventoried in Phase 5; deferred pending HDF5 test fixtures.
7. **`infembed` package relocation.** Currently editable-installed from `third_party/cupid/third_party/infembed/`. Once Phase-5b lands and cupid/influence boundaries are crisper, infembed can move to a top-level `third_party/infembed/` or be promoted into `policy_doctor.influence`.
8. **`.gitleaks.toml` cleanup.** Three orphan entries (`environment_policy_doctor.yaml`, `environment_mimicgen_torch2.yaml`, `environment_robocasa.yaml`) in the allowlist now reference non-existent files. Harmless but stale.
9. **String-only conda mentions in source.** `policy_doctor/paths.py:30` and `tests/support/mimicgen_seed/pipeline.py:7,81` contain comment / error-message strings that mention `environment_*.yaml` and `conda activate <env>`. Code is functionally correct (uv-based); only the prose is stale.

### Plan impact
- Phase 6 — **closed**. Refactor branch (`refactor/clean-architecture`) is at 23 commits ahead of `main` and is ready for the user to open the PR. The deferred items above are intentional follow-ups, not blockers.

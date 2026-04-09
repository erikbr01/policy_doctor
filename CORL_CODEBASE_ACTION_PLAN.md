# CoRL-aligned codebase action plan (policy_doctor)

This document translates [CORL_PLAN.md](./CORL_PLAN.md) into **concrete, ordered steps** against the **current** `policy_doctor` tree. It separates work that is **necessary for the paper claims**, work that is **already in place**, and work that is **optional or better deferred** so the repo stays flexible without carrying unnecessary surface area.

Assumptions: single-task diffusion policies, small data scale, a few Robomimic / MimicGen / (eventually) Robocasa-style tasks; attribution and training stay on the vendored **cupid** stack under `third_party/cupid/`.

---

## 1. Map goals to the repository

| CORL contribution | Primary location today | Gap vs plan |
|-------------------|------------------------|-------------|
| **1 — Behavior discovery** (clustering influence / embeddings) | `policy_doctor/behaviors/clustering*.py`, `computations/`, `data/clustering_*.py`, pipeline step `run_clustering`, IV clustering persistence | Hyperparameter + multi-task **sweep orchestration** is mostly “run Hydra many times”; no unified experiment matrix YAML required for v1. |
| **1 — Human-interpretable descriptions** | VLM: `policy_doctor/vlm/`, steps `annotate_slices_vlm`, `summarize_behaviors_vlm`, `vlm/registry.py` | **Second-stage VLM coherency judge** (plan: “VLM evaluates coherency of descriptions”) is **not** a dedicated pipeline step yet. |
| **2 — Behavior graph + temporal context** | `policy_doctor/behaviors/behavior_graph.py`, `behaviors/clustering_temporal.py`, Streamlit `streamlit_app/tabs/behavior_graph.py`, Plotly/pyvis plotting | Mostly sufficient; tie **Markov outputs** into saved run artifacts if you want paper figures without ad-hoc notebooks. |
| **2 — Markov assumption** | `test_markov_property` / `test_markov_property_pooled` in `behavior_graph.py`; `tests/integration/test_markov_property.py` | **Already implemented**; ensure each CoRL experiment run **records** JSON (per graph / per task) next to clustering outputs. |
| **3 — Data collection guidance** | MimicGen submodule `third_party/mimicgen/`, standalone cupid task `diffusion_policy/config/task/square_mimicgen_lowdim.yaml` (no Robomimic task defaults), HDF5 helpers, `tests/support/mimicgen_seed/`, diffusion smoke tests | **Heuristics from graphs → MimicGen / human protocol** are research, not code-complete; **dataset adapters** are the main engineering lever. |

---

## 2. Where configs live: `policy_doctor` vs `third_party/`

This split supports **parameterized experiments** (e.g. `data_source`: cupid Robomimic vs MimicGen HDF5 vs Robocasa-layout HDF5): one **diffusion** policy stack with **adapters**, one **attribution** stack, **visualization agnostic** to the data source. Config files should not be duplicated in two trees.

### 2.1 Canonical configs stay in `third_party`

| Stack | Owns | Rationale |
|-------|------|-----------|
| **Cupid** (`third_party/cupid/`) | Diffusion training / workspace Hydra (`configs/low_dim/...`, `diffusion_policy/config/task/...`), eval rollouts, TRAK workspace settings | `train.py` and workspace code resolve configs from the cupid tree. Copying full workspace YAMLs into `policy_doctor` causes drift. |
| **MimicGen** (`third_party/mimicgen/`) | Datagen: prepare/generate, env interfaces, pins aligned with that release | MimicGen entrypoints expect their native layout. |
| **Robocasa** (when used) | That project’s env + export conventions | Same rule: follow upstream config, not a second copy in policy_doctor. |

### 2.2 Experiment orchestration stays in `policy_doctor`

`policy_doctor/configs/` holds **cross-cutting experiment identity** and **composition**, not a second training stack:

- **`data_source`** (or equivalent defaults group): which dataset family / adapter applies for this run.
- **Pipeline**: steps, `task_config`, clustering / curation / VLM knobs.
- **Pointers** into cupid: `baseline.config_dir`, `baseline.config_name`, and Hydra-style **overrides** (e.g. `task.dataset.dataset_path`, diffusion `task=` name) where the pipeline already passes them into child processes.
- **Run metadata**: `run_name`, seeds, `train_date` / `eval_date`, `project`.

Treat `policy_doctor` as the **experiment composer**: it **selects** a cupid profile and **overrides** what differs per data source, instead of redefining the entire diffusion workspace.

### 2.3 What to duplicate vs not

- **Do not** mirror whole `third_party/cupid/configs/low_dim/.../config.yaml` files under `policy_doctor`.
- **Do** add small, **reusable** cupid-side definitions when something is shared across runs. **Architecture-only** reuse lives in `configs/low_dim/_shared/diffusion_unet_lowdim_square_arch.yaml` (composed with `@_here_`); **MimicGen vs Robomimic** stay separate experiment YAMLs and task slices (`square_mimicgen_lowdim` does **not** `defaults: square_lowdim`).
- **Do** keep the thin **`policy_doctor/configs/robomimic/`** task slices (`tasks/`, `baseline/`, `attribution/`, …) as the **adapter contract** for Python helpers (paths, `obs_dim`, naming)—not as a parallel definition of optimizer and model architecture.

### 2.4 Attribution and visualization

- **Attribution (TRAK, InfEmbed, eval episodes):** same code paths for all data sources; task-specific paths and dataset alignment live next to the **policy/dataset** they describe (mostly cupid + the robomimic YAML slices above).
- **Visualization / plotting / Streamlit:** remain in `policy_doctor`; they consume **artifacts** (clustering, graphs, rollouts) and stay **agnostic** to whether demos came from vanilla Robomimic, MimicGen, or Robocasa-layout HDF5.

### 2.5 Quick reference

| Concern | Location |
|---------|----------|
| Architecture, optimizer, horizon, normalizers | Cupid `configs/low_dim/_shared/…` + per-stack `config.yaml` (Robomimic `square_mh` vs MimicGen `square_mimicgen_lowdim` compose the shared block) |
| Dataset path, `obs_keys`, diffusion `task` for `train.py` | Cupid task/workspace YAMLs (**separate** Robomimic vs MimicGen runners/datasets); **policy_doctor** only **overrides** paths when an experiment needs a different HDF5 |
| MimicGen prepare / generate | MimicGen configs + glue scripts |
| Which stack + task + pipeline order | `policy_doctor/configs/` |

---

## 3. Necessary steps (do these for CoRL-shaped claims)

### 3.1 Contribution 1 — Clustering and annotation

1. **Freeze the “default” discovery path** for the paper: e.g. InfEmbed embeddings → clustering (document which embedding source and algorithm in one experiment preset under `policy_doctor/configs/experiment/`). Avoid maintaining multiple parallel discovery paths unless a reviewer-facing ablation needs them.

2. **VLM slice annotation** — keep using `annotate_slices_vlm` with real backends (`qwen3_vl`, `cosmos_reason2`, `molmo2` already registered in `policy_doctor/vlm/registry.py`).  
   - **Concrete code follow-ups:** extend `policy_doctor/vlm/frames.py` (and annotate path) so experiments can request **multi-camera** stacks and **higher resolution** via Hydra (`policy_doctor/configs/vlm/`), matching CORL_PLAN’s note about image quality.

3. **VLM coherency evaluation (missing piece)** — add a small, explicit stage after slice labels exist:
   - **New module** e.g. `policy_doctor/vlm/coherency_eval.py`: given per-cluster lists of slice captions (reuse `group_slice_labels_by_cluster` from `behavior_summarize.py`), call a VLM with a fixed rubric prompt (coherent / contradictory / mixed; optional 1–5 score).
   - **New pipeline step** e.g. `evaluate_cluster_coherency_vlm` in `policy_doctor/curation_pipeline/steps/`, registered in `curation_pipeline/pipeline.py` *after* `summarize_behaviors_vlm` (or after `annotate_slices_vlm` if you judge raw slice text only). Persist `result.json` + per-cluster scores under `<run_dir>/evaluate_cluster_coherency_vlm/`.
   - **Hydra** defaults in `policy_doctor/configs/vlm/` for prompts and `max_labels_per_cluster` to control cost.

4. **Hyperparameter and task sweeps** — do **not** require a new subsystem: use one shell/Makefile target or a thin loop over `python -m policy_doctor.scripts.run_pipeline` with `+experiment=...` and different `task_config` / clustering overrides. Optionally add `policy_doctor/configs/experiment/corl_cluster_sweep_template.yaml` as documentation only.

5. **Human intuition study (offline from code)** — protocol + UI: Streamlit tabs `clustering.py` / `vlm_annotation.py` already support browsing; add a minimal **export** (CSV/JSON of cluster id → example slice ids → paths) if you need Mechanical Turk or spreadsheet ratings.

### 3.2 Contribution 2 — Behavior graphs and Markov

1. **Treat Markov testing as a first-class artifact:** wrap `test_markov_property(...)` in a pipeline step or a `scripts/` entry point that reads the same `cluster_labels.npy` + `metadata.json` as `policy_doctor/data/clustering_loader.py`, writes `markov_report.json` (overall + per-state) under the run directory or next to the clustering slug. This avoids one-off Jupyter runs for every task.

2. **Figures for graph properties:** reuse `policy_doctor/plotting/plotly/behavior_graph*.py`; ensure the Streamlit behavior graph tab or a script can **export static images** for the paper (PNG/SVG) with deterministic styling.

3. **Temporal context:** keep using `behaviors/clustering_temporal.py` and existing Streamlit “temporal coherence” patterns (see `streamlit_app/clustering_browse.py`); only extend if a specific figure in the paper needs a new view.

### 3.3 Contribution 3 — Data collection and sim stacks

CORL_PLAN asks for **diffusion-only policy code** with **multiple dataset adapters** (classic MimicGen robosuite pin vs Robocasa-style robomimic HDF5). The realistic split:

1. **MimicGen → cupid training path (necessary minimal):**
   - **Implemented (Square low-dim):** `configs/low_dim/square_mimicgen_lowdim/diffusion_policy_cnn/` composes shared U-Net YAML only; `MimicgenLowdimRunner` / `MimicgenReplayLowdimDataset`; runner **`import mimicgen`** so MimicGen robosuite envs (e.g. `Square_D0`) register. Hydra task slice `diffusion_policy/config/task/square_mimicgen_lowdim.yaml` is **standalone** (not `defaults: square_lowdim`).
   - Default merged path in YAML remains `data/mimicgen/square_merged.hdf5`; smoke tests also resolve `data/source/mimicgen/source/square.hdf5` when present.
   - Use `diffusion_policy/common/hdf5_robomimic_layout.py` and `inspect_robomimic_hdf5.py` as the contract for HDF5 layout.
   - **Still open:** document one end-to-end recipe (MimicGen conda → merge → cupid train); **Attribution / TRAK / InfEmbed** path alignment for MimicGen HDF5 beyond Hydra overrides.

2. **Robocasa / Robocasa365 (necessary as “adapter,” not as full MimicGen fork):**
   - **Defer** unifying the Robocasa365 *branch of MimicGen* until you have a concrete task that requires it; the plan itself flags uncertainty.
   - **Do** implement or verify a **robomimic-layout HDF5** path: if Robocasa demos export to the same `data/demo_*` schema, extend `diffusion_policy/dataset/robomimic_replay_{lowdim,image}_dataset.py` only where keys or shapes differ (env-specific `obs_keys` in task YAML, not a second policy class).
   - The test helper `RobocasaRobomimicSeedMaterializer` in `tests/support/mimicgen_seed/robomimic_source.py` is the right *pattern*; promote it from tests into a small `policy_doctor/data/adapters/` (or cupid `dataset/`) module when you have a real Robocasa file.

3. **“Graph-guided” data collection (research step):**
   - Define one or two **simple heuristics** in prose first (e.g. under-sample transitions with high stationary probability; prioritize edges rarely visited). Then add a **script** that reads `BehaviorGraph` transition counts/ probs and outputs a **priority list of (behavior cluster, optional skill)** for MimicGen or human collectors — keep it outside the hot training path until the heuristic is validated.

---

## 4. Already sufficient — avoid extra churn

- **Full Hydra pipeline** (`train_baseline` → … → `compare`) — already matches CoRL-scale iteration; you do not need a second CLI.
- **Markov statistical tests** — implemented and unit-tested; don’t replace unless reviewers ask for a different null model.
- **Multiple VLM backends** via registry — prefer adding **Gemini** (below) over forking annotate logic per vendor.
- **Label coherency charts** (`plotting/plotly/clusters.py`, `create_label_coherency_chart`) — these measure **metadata / human label** statistics; they complement but do not replace **VLM coherency judging** from CORL_PLAN.

---

## 5. Defer or keep optional (scope discipline)

- **Pairwise VLM comparison across all slices in a cluster** — expensive; only add if per-cluster caption coherency is insufficient.
- **Single conda env for MimicGen + cupid** — generally **not** worth forcing; CORL_PLAN already acknowledges robosuite skew. Keep **`mimicgen`** for generation and **`cupid`** for training/eval, with documented HDF5 handoff.
- **Robocasa365 MimicGen branch** — treat as a separate spike, not a blocker for “diffusion on robomimic-layout Robocasa demos.”
- **Large-scale active-learning baselines** — out of scope for “small data, single task” unless you add a clearly scoped baseline subsection.

---

## 6. Proposed additions (sensible extensions)

| Item | Rationale | Where |
|------|-----------|--------|
| **Gemini backend** | CORL_PLAN names Gemini-ER; add `vlm/backends/gemini.py` + register in `vlm/registry.py` behind optional deps. | `policy_doctor/vlm/` |
| **`evaluate_cluster_coherency_vlm` step** | **Done:** annotate → per-cluster coherency JSON (`evaluate_cluster_coherency_vlm/`). | `curation_pipeline/steps/evaluate_cluster_coherency_vlm.py`, `vlm/coherency_eval.py`, `vlm_coherency_eval` in `configs/vlm/defaults.yaml` |
| **`export_markov_report` pipeline step** | **Done:** Markov report per seed under `export_markov_report/`. | `curation_pipeline/steps/export_markov_report.py`, `markov_export` in `configs/pipeline/config.yaml`, `markov_test_result_to_jsonable` in `behavior_graph.py` |
| **`policy_doctor/data/adapters/README.md` + thin modules** | Documents “one policy, many HDF5 sources” without scattering notes across cupid and tests. | New small package dir (optional) |
| **CORL experiment YAMLs** | One preset per paper figure row (clustering-only, full pipeline + VLM, MimicGen square). | `policy_doctor/configs/experiment/corl_*.yaml` |
| **`data_source` Hydra group** | **Implemented:** `cupid_robomimic`, `mimicgen_square`, `robocasa_layout` under `configs/data_source/`; see §2 config ownership. | `policy_doctor/configs/data_source/`, `scripts/experiments/*.sh` |

---

## 7. Cleanup / flexibility (if the repo feels “messy”)

These are **optional** hygiene moves; pick based on what actually hurt you in practice.

1. **Demote experimental presets:** move one-off `configs/experiment/*` that were for abandoned ideas into `configs/experiment/archive/` or delete if unused, and keep **3–5** canonical CoRL YAMLs at the top level.

2. **Clarify third-party boundaries in README:** one short subsection “What we edit for CoRL” (`policy_doctor/*`, `third_party/cupid/diffusion_policy/{config,dataset,common}`) vs “vendor fork” (`third_party/mimicgen`, bundled robomimic).

3. **Reduce duplicate plotting:** `influence_visualizer` and `policy_doctor` both ship cluster plotting; for CoRL, **import from `policy_doctor.plotting` only** in new code paths.

4. **Tests:** keep `run_tests.py` suites split (`policy_doctor` / `cupid` / `mimicgen`); add one **integration test** that runs mock VLM through annotate → summarize → (future) coherency with **no GPU**. Diffusion training smoke lives in `tests/integration/test_diffusion_data_source_smoke.py` — run with **`cupid`** conda Python (not `policy_doctor` env: diffusers mismatch).

---

## 7b. Diffusion / data-stack engineering — done vs remaining (working log)

### Done (landed in tree)

- **Separate env runners (no shared Robomimic config for MimicGen rollouts):** `RobomimicLowdimRunner` / `MimicgenLowdimRunner` (and image variants); MimicGen runners prepend vendored `third_party/mimicgen` on `sys.path` and `import mimicgen` so custom robosuite env names from HDF5 resolve.
- **Dataset adapters:** `MimicgenReplayLowdimDataset`, `RobocasaReplayLowdimDataset` (thin subclasses); `LerobotRobocasaImageDataset` for RoboCasa LeRobot v2; `lerobot_robocasa_dataset.py`.
- **Hydra experiment configs:** `square_mh` vs `square_mimicgen_lowdim` both default `../../_shared/diffusion_unet_lowdim_square_arch@_here_` then apply **stack-specific** `task` (paths, `_target_`, logging, `multi_run`). Robomimic MH is unchanged for backward compatibility; MimicGen has its own tags and output dirs.
- **Task slice:** `diffusion_policy/config/task/square_mimicgen_lowdim.yaml` is fully explicit — **no** inheritance from `square_lowdim`.
- **`get_dataset_masks`:** recognizes `mimicgen` / `robocasa` path segments and substring fallbacks (e.g. resolved `…/mimicgen_data/…`, `…/robocasa_data/…`) so splits work off symlinked `policy_doctor/data/source/…`.
- **`policy_doctor` `data_source` profiles:** `cupid_robomimic`, `mimicgen_square`, `robocasa_layout` (no duplicate `dataset_adapter` field in YAML — orchestration uses baseline pointers only).
- **Integration smoke tests:** `tests/integration/test_diffusion_data_source_smoke.py` — short train + rollout for Robomimic + MimicGen low-dim when HDF5 exists; resolves `DATA_SOURCE_ROOT` (`data/source/{robomimic,mimicgen,robocasa}`); optional hybrid skips without LeRobot + eval HDF5; `wandb.finish()` between tests; `logging.mode=offline`; `tests/config/__init__.py` fixes `run_tests.py` discovery; `third_party/cupid/wandb/settings` pydantic fix (`disabled` removed).
- **Tests (earlier):** policy_doctor Markov / pyvis collection fixes; `test_cupid_workspace_presence` asserts key cupid configs exist.
- **RoboCasa hybrid smoke test passing end-to-end** (`TestDiffusionRobocasaLerobotHybridSmoke`): train + live eval rollout via `PickPlaceCounterToCabinet` in the `robocasa` conda env (~73s). Verified against `data/source/robocasa/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot`.
- **`RobocasaImageRunner` refactored:** added `env_name` + `env_kwargs` parameters as HDF5-free env creation path; `OmegaConf.to_container` for nested `DictConfig` (e.g. `controller_configs`); deep copy of `env_meta` before `create_env_from_metadata` to prevent mutation; `AsyncVectorEnv(shared_memory=False)` for Dict observation spaces; try/except fallback to direct `EnvRobosuite` construction when robocasa_support robomimic raises `AttributeError` on `action_dimension`.
- **`AsyncVectorEnv` gym 0.26 compat** (`diffusion_policy/gym_util/async_vector_env.py`): `reset_async` / `reset_wait` accept `seed` and `options` kwargs; `concatenate` arg order corrected to `(space, items, out)`.
- **`CropRandomizer` import compat** (`diffusion_transformer_hybrid_image_policy.py`): try `rmbn.CropRandomizer`, fall back to `robomimic.models.obs_core.CropRandomizer` and patch `rmbn` for isinstance checks — handles both cupid and robocasa_support robomimic.
- **MimicGen `obs_dim` fix** (`configs/low_dim/square_mimicgen_lowdim/diffusion_policy_cnn/config.yaml`): MimicGen D1 `object` key is 17-dim (vs 14 in standard robomimic Square) → `obs_dim=26`, `policy.obs_dim=26`, `policy.model.global_cond_dim=52` (= 26 × 2 obs steps). Hardcoded `46` in `_shared` arch left unchanged; MimicGen config overrides at compose time.
- **Training scripts** (`scripts/experiments/`): `train_robomimic_square.sh` (cupid env, Square MH low-dim), `train_mimicgen_square.sh` (cupid env, MimicGen D1 low-dim), `train_robocasa_atomic.sh` (robocasa env, image/hybrid, auto-discovers latest LeRobot dataset). All use `--config-path` (not `--config-dir`) to replace the Hydra config root with the external cupid `configs/` tree.
- **patchelf symlink** for `mujoco_py` build in cupid env: `ln -sf /home/erbauer/miniforge3/bin/patchelf /home/erbauer/miniforge3/envs/cupid/bin/patchelf`.
- **README** updated with “Diffusion policy training” section covering all three scripts, Hydra key reference, env vars, and data path table.

### Remaining / follow-ups

- **RoboCasa low-dim smoke:** skips — user layout is LeRobot under `v1.0/`, not robomimic-layout kitchen HDF5. The old `robocasa_layout_lowdim` config targets a deprecated HDF5 format; no action needed for new robocasa data.
- **Default MimicGen training file:** Cupid YAML still documents `square_merged.hdf5`; training scripts use `core_datasets/.../demo.hdf5` (D1 generated). Document when to use merged vs source for diffusion — core `demo.hdf5` has different obs layout than merged.
- **Environments:** keep documenting **`mimicgen`** vs **`cupid`** conda split; `run_tests.py --suite policy_doctor` with **`policy_doctor`** env still errors on diffusion smoke (diffusers); use **`cupid`** Python for that module.
- **Attribution / TRAK / InfEmbed:** same adapters as training for MimicGen & Robocasa paths — not fully audited end-to-end.
- **CORL items elsewhere in this doc:** VLM multi-cam/resolution, graph→collection heuristic script, optional experiment YAMLs — unchanged by diffusion work.

---

## 8. Suggested implementation order

1. **Artifact plumbing:** Markov report export + clustering/VLM outputs in a fixed directory layout under `data/pipeline_runs/<run>/`.  
2. **VLM coherency step** + prompts + config.  
3. **Multi-cam / resolution** options for slice frames.  
4. **One MimicGen→cupid e2e recipe** in prose (README or this doc): datagen / merge → `square_mimicgen_lowdim` train; **partial guard:** `test_diffusion_data_source_smoke` + existing `test_mimicgen_square_e2e` (optional).  
5. **Robocasa HDF5 task YAML** once a sample dataset exists (adapter-only change).  
6. **Graph→collection heuristic script** last (depends on stable graph exports).

---

## 9. Success criteria (code-level)

- From a **single** `run_name`, you can point to: clustering manifest, optional VLM annotations, optional behavior summaries, **coherency scores**, **Markov summary**, and curated vs baseline `compare/result.json`.
- **One** diffusion experiment config (`square_mimicgen_lowdim`) trains on MimicGen-layout HDF5 with **MimicGen-specific** dataset/runner (shared weights architecture only); no reuse of Robomimic `task=square_lowdim` defaults.
- **No** requirement for multiple policy architectures—only dataset/task YAML and obs key differences.

This keeps the repository **flexible**: you can drop optional steps (VLM, MimicGen) for ablations without rewriting the core influence → cluster → graph → curation path.

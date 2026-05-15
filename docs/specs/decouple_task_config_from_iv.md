# Spec: Decouple pipeline from influence_visualizer task configs

## Problem

`RunClusteringStep` and several other pipeline steps depend on a per-task
YAML file that lives in `third_party/influence_visualizer/configs/`. Three
separate things are coupled through this file:

1. **Clustering output location** — `save_clustering_result(task_config=…)`
   writes results to `iv/configs/<task_config>/clustering/<name>/`. The
   `task_config` string is used as a directory namespace on disk.

2. **Eval/train path fallback** — `task_cfg["eval_dir"]` and
   `task_cfg.get("train_dir")` are used as last-resort fallbacks in
   `RunClusteringStep`, `SelectMimicgenSeedFromGraphStep`, and
   `SelectMimicgenSeedStep` when the Hydra config does not supply
   `evaluation.train_date`.

3. **TRAK embedding metadata** — `extract_trak_slice_windows` reads
   `task_cfg.get("train_ckpt", "latest")` and `task_cfg.get("exp_date",
   "default")` for TRAK-specific path construction.

Because of (1), every experiment must have a named YAML in the iv configs
directory, and that YAML must contain an `eval_dir` that is either correct
or benign. The apr26 sweep shipped with
`task_config: square_mh_apr23_mimicgen_pipeline` in the experiment YAML,
which pointed to the wrong `eval_dir` and silently corrupted seed-0 MimicGen
generation for all 24 completed arms.

The iv Streamlit app originated this design — it uses `task_config` to
browse clustering results stored inside its own configs tree. The curation
pipeline inherited the coupling without ever needing the Streamlit
integration.

---

## Goal

Remove `task_config` as a required parameter for pipeline execution.
All paths must be derivable from the Hydra config alone. The iv Streamlit app
must continue to work via explicit `clustering_dir` overrides (which it
already supports).

---

## Changes required

### 1. `third_party/influence_visualizer/clustering_results.py`

Add an overload to `save_clustering_result` that accepts an explicit
`output_dir: Path` instead of deriving the path from `task_config`:

```python
def save_clustering_result(
    name: str,
    cluster_labels: ...,
    metadata: ...,
    ...,
    output_dir: Path | None = None,   # NEW — explicit path
    task_config: str | None = None,   # kept for backwards compat
) -> Path:
    if output_dir is not None:
        result_dir = output_dir / name
    elif task_config is not None:
        result_dir = get_clustering_dir(task_config) / name
    else:
        raise ValueError("Either output_dir or task_config must be provided")
    ...
```

No existing callers break. The pipeline will pass `output_dir`; the Streamlit
app continues to pass `task_config`.

### 2. `policy_doctor/curation_pipeline/steps/run_clustering.py`

- Remove the `task_yaml` open and `task_cfg` dict entirely for the infembed
  path (which is the only path used in production).
- For the TRAK path, read `train_ckpt` and `exp_date` from Hydra config keys
  (`evaluation.train_ckpt`, `evaluation.exp_date`) with the same defaults
  (`"latest"`, `"default"`). Remove `task_cfg` argument from
  `extract_trak_slice_windows`.
- Remove the `task_cfg["eval_dir"]` fallback. The `evaluation.train_date` +
  `evaluation.task` + `evaluation.policy` triple is already required (enforced
  by `_resolve_rollouts_hdf5`). Apply the same requirement here: raise
  `ValueError` if none of the three override levels resolve an eval dir,
  with a message naming the required config keys.
- Pass `output_dir=self.step_dir / "clustering"` to `save_clustering_result`
  instead of `task_config=task_config`.
- Remove `from policy_doctor.paths import iv_task_configs_base` and the
  `config_root` branch.
- Remove `task_config` from the result JSON written by this step (it is not
  used by any downstream step — `clustering_dirs` path is what matters).

### 3. `policy_doctor/data/clustering_embeddings.py` — `extract_trak_slice_windows`

Remove the `task_cfg: dict` parameter. Add explicit `train_ckpt: str = "latest"`
and `exp_date: str = "default"` parameters. Update the single call site in
`run_clustering.py`.

### 4. `policy_doctor/curation_pipeline/steps/select_mimicgen_seed_from_graph.py`
### 5. `policy_doctor/curation_pipeline/steps/select_mimicgen_seed.py`

The `_resolve_rollouts_hdf5` helper already raises instead of falling back to
`task_cfg["eval_dir"]` (fixed in the bug-hardening commit). The remaining
`task_yaml` open (lines 103–105) now serves no purpose — remove it and the
`task_config` read entirely from these two files.

### 6. Other steps that open task YAMLs

These steps also read iv task configs but are not on the critical MimicGen
path. Defer them to a follow-up or update opportunistically:

| Step | What it reads from task YAML | Replacement |
|------|------------------------------|-------------|
| `compare.py` | `eval_dir`, display metadata | Hydra `evaluation.*` keys |
| `annotate_slices_vlm.py` | `task_config` for VLM hint, `iv_clustering_dir` for path | pass `clustering_dir` explicitly; drop `iv_clustering_dir` call |
| `run_curation_config.py` | `load_clustering_result(task_config, name)` fallback | already has `load_clustering_result_from_path(clustering_dir)` fast-path; remove the slow-path |

### 7. `policy_doctor/configs/experiment/mimicgen_square_sweep_apr26.yaml`

Remove `task_config` entirely (already done). The data_source default
(`square_mh_feb15`) would flow through for any step that still reads it;
those steps should be fixed by items 4–6 above so no step opens the file.

### 8. `policy_doctor/configs/data_source/mimicgen_square.yaml`

Remove `task_config: square_mh_feb15` once no pipeline step reads it.
The field can remain for Streamlit browsing if needed.

---

## What does NOT change

- `third_party/influence_visualizer/clustering_results.py` — the
  `task_config`-based path logic is kept for backwards compatibility with the
  Streamlit app. Only a new `output_dir` overload is added.
- Clustering result directory format on disk — same `manifest.yaml`,
  `cluster_labels.npy`, `metadata.yaml` layout.
- `policy_doctor/data/clustering_loader.py` — `load_clustering_result_from_path`
  reads from an explicit path and has no iv dependency; no change needed.
- The pipeline's `clustering_dirs` result JSON key — downstream steps already
  use this path directly.

---

## Affected files

**Must change (blocks correct MimicGen pipeline execution):**
- `third_party/influence_visualizer/clustering_results.py`
- `policy_doctor/curation_pipeline/steps/run_clustering.py`
- `policy_doctor/data/clustering_embeddings.py`
- `policy_doctor/curation_pipeline/steps/select_mimicgen_seed_from_graph.py`
- `policy_doctor/curation_pipeline/steps/select_mimicgen_seed.py`

**Should change (remove remaining iv config reads from pipeline):**
- `policy_doctor/curation_pipeline/steps/compare.py`
- `policy_doctor/curation_pipeline/steps/annotate_slices_vlm.py`
- `policy_doctor/curation_pipeline/steps/run_curation_config.py`

**Config cleanup (after code changes):**
- `policy_doctor/configs/data_source/mimicgen_square.yaml` (remove `task_config`)
- All other `data_source/*.yaml` files that set `task_config`

---

## Verification

1. `python run_tests.py --suite policy_doctor` — all pre-existing tests pass.
2. Dry-run clustering step with the apr26 sweep config:
   ```
   python -m policy_doctor.scripts.run_pipeline \
     data_source=mimicgen_square \
     +experiment=mimicgen_square_sweep_apr26 \
     run_dir=/tmp/test_no_task_config \
     seeds=[0] baseline.max_train_episodes=60 \
     evaluation.train_date=apr26_sweep_demos60 \
     dry_run=true steps=[run_clustering]
   ```
   Should succeed without reading any file from `influence_visualizer/configs/`.
3. Confirm clustering output lands in `run_dir/run_clustering/clustering/`
   (not in `iv/configs/<task_config>/clustering/`).
4. Run `select_mimicgen_seed` dry-run — should succeed without `task_config`
   in config.

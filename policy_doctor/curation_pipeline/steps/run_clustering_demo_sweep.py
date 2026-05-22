"""Demo clustering sweep — pipeline step class.

Runs a W × S × K × rep × ordering grid over one eval dir and writes results to:
  <step_dir>/clustering/<ordering>/<slug>/

Two orderings:
  umap_first   normalize → UMAP on per-timestep embeddings (trunk, shared across W/S/K)
               → window → K-means.  Fast: UMAP runs once per rep.
  agg_first    window → normalize → UMAP → K-means.
               UMAP runs once per (rep, W, S) combo.

Config is read from cfg.clustering_sweep (set via clustering_sweep=<task> on CLI).

Run with:
  python -m policy_doctor.scripts.run_pipeline \\
    clustering_sweep=transport_mh_jan28 \\
    steps=[run_clustering_demo_sweep]
"""
from __future__ import annotations

import os
if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

import pathlib
import time
from typing import Any, Dict, List

import numpy as np
import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.paths import PROJECT_ROOT

# Demo sweep results live in the policy_doctor project tree, not in cupid.
# CurationPipeline resolves run_dir relative to cfg.repo_root; we patch that
# here so data/demo_sweep/* stays out of third_party/cupid/.
_DEMO_ROOT = PROJECT_ROOT


class RunClusteringDemoSweepStep(PipelineStep[Dict[str, Any]]):
    """Run the full demo clustering sweep.

    Supports single-task (clustering_sweep=transport_mh_jan28) and
    multi-task (clustering_sweep=all_robomimic) configs.  When clustering_sweep.tasks
    is present, iterates over all listed tasks in sequence.
    """

    name = "run_clustering"

    def __init__(self, cfg, run_dir, parent_run_dir=None):
        super().__init__(cfg, run_dir, parent_run_dir)
        # Override step_dir: resolve from clustering_sweep.run_dir relative to
        # PROJECT_ROOT, not cfg.repo_root (which points at cupid for training steps).
        # For multi-task configs (tasks list), step_dir is per-task — set in _run_one.
        sw = OmegaConf.select(cfg, "clustering_sweep")
        run_dir_rel = OmegaConf.select(sw, "run_dir", default=None) if sw else None
        if run_dir_rel:
            self.step_dir = _DEMO_ROOT / run_dir_rel / self.name

    def compute(self) -> Dict[str, Any]:
        sw = OmegaConf.select(self.cfg, "clustering_sweep")
        if sw is None:
            raise ValueError(
                "clustering_sweep config is not set. "
                "Run with: clustering_sweep=all_robomimic (or a single task config)"
            )

        tasks_cfg = OmegaConf.select(sw, "tasks", default=None)
        if tasks_cfg is not None:
            # Multi-task: iterate over clustering_sweep.tasks list.
            # Each entry may override orderings, reps, umap_init from the sweep defaults.
            sw_base = OmegaConf.to_container(sw, resolve=True)
            all_results: Dict[str, Any] = {}
            for task_entry in OmegaConf.to_container(tasks_cfg, resolve=True):
                # Build per-task sw by merging task-level overrides on top of defaults
                sw_task = dict(sw_base)
                for key in ("orderings", "reps", "umap_init"):
                    if key in task_entry:
                        sw_task[key] = task_entry[key]
                result = self._run_one(
                    sw=sw_task,
                    task=task_entry["task"],
                    eval_dir=PROJECT_ROOT / task_entry["eval_dir"],
                    run_dir_rel=task_entry["run_dir"],
                )
                all_results[task_entry["task"]] = result
            return all_results

        # Single-task
        task = OmegaConf.select(sw, "task")
        eval_dir = PROJECT_ROOT / OmegaConf.select(sw, "eval_dir")
        run_dir_rel = OmegaConf.select(sw, "run_dir")
        return self._run_one(sw=sw, task=task, eval_dir=eval_dir, run_dir_rel=run_dir_rel)

    def _run_one(
        self,
        sw: Any,
        task: str,
        eval_dir: pathlib.Path,
        run_dir_rel: str,
    ) -> Dict[str, Any]:
        from policy_doctor.behaviors.clustering import (
            fit_cluster_kmeans,
            fit_normalize_embeddings,
            fit_reduce_dimensions,
        )
        from policy_doctor.data.clustering_loader import ClusteringModels
        from policy_doctor.data.slice_representations import (
            SliceWindowParams,
            get_slice_representation,
        )
        import joblib

        if not eval_dir.is_dir():
            raise FileNotFoundError(f"eval_dir not found: {eval_dir}")

        step_dir = _DEMO_ROOT / run_dir_rel / self.name
        # sw may be a plain dict (per-task override path) or an OmegaConf DictConfig
        _get = (lambda k: sw.get(k)) if isinstance(sw, dict) else (lambda k: OmegaConf.select(sw, k))
        k_values: List[int] = list(_get("k_values"))
        w_values: List[int] = list(_get("w_values"))
        s_values: List[int] = list(_get("s_values"))
        aggregation: str = _get("aggregation")
        orderings: List[str] = list(_get("orderings"))
        umap_n_components: int = int(_get("umap_n_components"))
        umap_n_jobs: int = int(_get("umap_n_jobs"))
        umap_init: str = _get("umap_init")
        reps_raw = _get("reps")
        reps = reps_raw if isinstance(reps_raw, list) else OmegaConf.to_container(reps_raw, resolve=True)

        clu_root = step_dir / "clustering"
        trunk_root = step_dir / "_trunks"

        written: List[str] = []

        for rep_cfg in reps:
            rep_id: str = rep_cfg["id"]
            representation: str = rep_cfg["representation"]
            rep_kwargs: Dict[str, Any] = {
                k: v for k, v in rep_cfg.items() if k not in ("id", "representation")
            }

            print(f"\n[demo_sweep] task={task}  rep={rep_id}", flush=True)
            rep_obj = get_slice_representation(representation)

            # ── umap_first: build trunk once, window in UMAP-reduced space ──────
            if "umap_first" in orderings:
                trunk_dir = trunk_root / rep_id
                trunk_dir.mkdir(parents=True, exist_ok=True)
                ts_emb_path = trunk_dir / "timestep_embeddings.npy"
                ep_meta_path = trunk_dir / "ep_meta.json"

                if ts_emb_path.exists():
                    print(f"  [umap_first] trunk exists: {trunk_dir.name}", flush=True)
                    import json
                    ts_emb = np.load(ts_emb_path).astype(np.float32)
                    ep_meta = json.load(open(ep_meta_path))
                else:
                    print(f"  [umap_first] building trunk for {rep_id} ...", flush=True)
                    t0 = time.time()
                    try:
                        per_ts, ep_lens, ep_succ = rep_obj.extract_per_timestep(
                            eval_dir, **rep_kwargs
                        )
                    except Exception as e:
                        print(f"    [warn] trunk extraction failed for {rep_id}: {e}; skipping rep", flush=True)
                        continue
                    print(f"    per-timestep: {per_ts.shape}", flush=True)
                    norm_x, normalizer = fit_normalize_embeddings(per_ts, method="none")
                    scaled_x, prescaler = fit_normalize_embeddings(norm_x, method="standard")
                    n_comp = min(umap_n_components, per_ts.shape[1] - 1)
                    ts_emb, reducer = fit_reduce_dimensions(
                        scaled_x, method="umap", n_components=n_comp,
                        n_jobs=umap_n_jobs, init=umap_init,
                    )
                    ts_emb = ts_emb.astype(np.float32)
                    print(f"    UMAP {per_ts.shape[1]}→{n_comp}d: {time.time()-t0:.1f}s", flush=True)

                    np.save(ts_emb_path, ts_emb)
                    import json
                    ep_meta = {
                        "episode_lengths": ep_lens,
                        "episode_successes": [
                            bool(s) if s is not None else None for s in ep_succ
                        ],
                    }
                    json.dump(ep_meta, open(ep_meta_path, "w"))
                    joblib.dump(
                        ClusteringModels(
                            normalizer=normalizer, normalizer_method="none",
                            prescaler=prescaler, prescaler_method="standard",
                            reducer=reducer, reducer_method="umap",
                            kmeans=None,
                            pipeline_steps=["normalize", "umap", "window", "kmeans"],
                        ),
                        trunk_dir / "trunk_models.pkl",
                    )

                from policy_doctor.data.clustering_embeddings import (
                    build_windows_from_rollout_timestep_embeddings,
                )
                for w in w_values:
                    for s in s_values:
                        slug_prefix = (
                            f"{rep_id}_seed0_kmeans_k"
                            if (w == 5 and s == 2)
                            else f"{rep_id}_w{w}_s{s}_seed0_kmeans_k"
                        )
                        windows, metadata = build_windows_from_rollout_timestep_embeddings(
                            ts_emb,
                            ep_meta["episode_lengths"],
                            ep_meta["episode_successes"],
                            w, s, aggregation,
                        )
                        windows = np.asarray(windows, dtype=np.float32)
                        for k in k_values:
                            slug = f"{slug_prefix}{k}"
                            out_dir = clu_root / "umap_first" / slug
                            if _is_done(out_dir, w, s, k, "umap_first"):
                                print(f"  [skip] umap_first/{slug}", flush=True)
                                continue
                            print(f"  [build] umap_first/{slug}", flush=True)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            labels, kmeans = fit_cluster_kmeans(windows, n_clusters=k)
                            n_actual = int(len(set(labels.tolist()) - {-1}))
                            _save(
                                out_dir, windows, metadata, labels, kmeans,
                                normalizer=None, normalizer_method="none",
                                prescaler=None, prescaler_method="standard",
                                reducer=None, reducer_method="umap",
                                pipeline_steps=["normalize", "umap", "window", "kmeans"],
                                manifest_extra={
                                    "task": task,
                                    "influence_source": rep_obj.name,
                                    "rep_id": rep_id,
                                    "window_width": w, "stride": s,
                                    "aggregation": aggregation,
                                    "n_clusters": n_actual,
                                    "umap_n_components": int(ts_emb.shape[1]),
                                    "umap_init": umap_init,
                                },
                            )
                            written.append(f"umap_first/{slug}")

            # ── agg_first: window first, then UMAP per (W, S) ────────────────
            if "agg_first" in orderings:
                for w in w_values:
                    for s in s_values:
                        params = SliceWindowParams(
                            window_width=w, stride=s, aggregation=aggregation
                        )
                        # Check if any K is missing before extracting features
                        slug_prefix = (
                            f"{rep_id}_seed0_kmeans_k"
                            if (w == 5 and s == 2)
                            else f"{rep_id}_w{w}_s{s}_seed0_kmeans_k"
                        )
                        missing_ks = [
                            k for k in k_values
                            if not _is_done(
                                clu_root / "agg_first" / f"{slug_prefix}{k}",
                                w, s, k, "agg_first"
                            )
                        ]
                        if not missing_ks:
                            print(
                                f"  [skip] agg_first/{rep_id} w={w} s={s} (all K done)",
                                flush=True,
                            )
                            continue

                        print(
                            f"  [agg_first] {rep_id} w={w} s={s} → UMAP ...",
                            flush=True,
                        )
                        t0 = time.time()
                        try:
                            features, metadata = rep_obj.extract(
                                eval_dir, params, **rep_kwargs
                            )
                        except Exception as e:
                            print(f"    [warn] extract failed: {e}; skipping", flush=True)
                            continue

                        print(f"    features: {features.shape}", flush=True)
                        norm_x, normalizer = fit_normalize_embeddings(features, method="none")
                        scaled_x, prescaler = fit_normalize_embeddings(norm_x, method="standard")
                        n_comp = min(umap_n_components, features.shape[1] - 1)
                        emb_red, reducer = fit_reduce_dimensions(
                            scaled_x, method="umap", n_components=n_comp,
                            n_jobs=umap_n_jobs, init=umap_init,
                        )
                        emb_red = emb_red.astype(np.float32)
                        print(
                            f"    UMAP {features.shape[1]}→{n_comp}d: {time.time()-t0:.1f}s",
                            flush=True,
                        )

                        for k in missing_ks:
                            slug = f"{slug_prefix}{k}"
                            out_dir = clu_root / "agg_first" / slug
                            print(f"  [build] agg_first/{slug}", flush=True)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            labels, kmeans = fit_cluster_kmeans(emb_red, n_clusters=k)
                            n_actual = int(len(set(labels.tolist()) - {-1}))
                            _save(
                                out_dir, emb_red, metadata, labels, kmeans,
                                normalizer=normalizer, normalizer_method="none",
                                prescaler=prescaler, prescaler_method="standard",
                                reducer=reducer, reducer_method="umap",
                                pipeline_steps=["window", "normalize", "umap", "kmeans"],
                                manifest_extra={
                                    "task": task,
                                    "influence_source": rep_obj.name,
                                    "rep_id": rep_id,
                                    "window_width": w, "stride": s,
                                    "aggregation": aggregation,
                                    "n_clusters": n_actual,
                                    "umap_n_components": int(emb_red.shape[1]),
                                    "umap_init": umap_init,
                                },
                            )
                            written.append(f"agg_first/{slug}")

        print(f"\n[demo_sweep] done — {len(written)} clustering(s) written", flush=True)
        return {"task": task, "clustering_root": str(clu_root), "written": written}


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_done(out_dir: pathlib.Path, w: int, s: int, k: int, ordering: str) -> bool:
    manifest = out_dir / "manifest.yaml"
    if not (out_dir / "cluster_labels.npy").exists() or not manifest.exists():
        return False
    try:
        m = yaml.safe_load(manifest.read_text()) or {}
        steps = m.get("pipeline_steps", [])
        expected = (
            ["normalize", "umap", "window", "kmeans"]
            if ordering == "umap_first"
            else ["window", "normalize", "umap", "kmeans"]
        )
        return (
            m.get("window_width") == w
            and m.get("stride") == s
            and m.get("n_clusters") == k
            and steps == expected
        )
    except Exception:
        return False


def _save(
    out_dir: pathlib.Path,
    emb_red: np.ndarray,
    metadata: list,
    labels: np.ndarray,
    kmeans: Any,
    normalizer: Any,
    normalizer_method: str,
    prescaler: Any,
    prescaler_method: str,
    reducer: Any,
    reducer_method: str,
    pipeline_steps: List[str],
    manifest_extra: Dict[str, Any],
) -> None:
    import json
    import joblib
    from policy_doctor.data.clustering_loader import ClusteringModels

    np.save(out_dir / "cluster_labels.npy", labels.astype(np.int32))
    np.save(out_dir / "embeddings_reduced.npy", emb_red.astype(np.float32))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(_to_jsonable(metadata), f)
    manifest = {
        "algorithm": "kmeans",
        "representation": "sliding_window",
        "level": "rollout",
        "n_samples": int(len(labels)),
        "pipeline_steps": pipeline_steps,
        **manifest_extra,
    }
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)
    joblib.dump(
        ClusteringModels(
            normalizer=normalizer, normalizer_method=normalizer_method,
            prescaler=prescaler, prescaler_method=prescaler_method,
            reducer=reducer, reducer_method=reducer_method,
            kmeans=kmeans,
            pipeline_steps=pipeline_steps,
        ),
        out_dir / "clustering_models.pkl",
    )


def _to_jsonable(metadata: list) -> list:
    out = []
    for m in metadata:
        item: Dict[str, Any] = {}
        for k, v in m.items():
            if isinstance(v, np.integer):
                item[k] = int(v)
            elif isinstance(v, np.floating):
                item[k] = float(v)
            elif isinstance(v, np.ndarray):
                item[k] = v.tolist()
            else:
                item[k] = v
        out.append(item)
    return out

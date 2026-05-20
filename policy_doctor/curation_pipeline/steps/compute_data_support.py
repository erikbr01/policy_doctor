"""Compute per-node data support for policy-embedding clusterings.

For each clustering produced by ``run_clustering`` we:

  1. Load the per-window rollout embeddings (recomputed from
     ``<eval_dir>/policy_embeddings/<layer>.npz`` so we can fit a fresh
     joint UMAP rather than reusing the rollout-only one).
  2. Load per-window demo embeddings (built from
     ``<train_dir>/policy_embeddings_demos/<layer>.npz``).
  3. Re-fit a UMAP on the union of demo + rollout windows.
  4. Compute every requested data-support metric in one pass against the
     demo cloud (count-in-radius, kNN distance, KDE log-density, binary
     coverage).
  5. Persist per-slice raw values + per-cluster summary statistics to
     ``<clustering_dir>/data_support.json`` so the Streamlit demo can colour
     by any metric / summary stat without re-running the pipeline.

The step no-ops with a ``data_support.skipped`` sentinel on any clustering
whose ``influence_source`` is not ``policy_emb`` — see CLAUDE.md for why
joint demo/rollout density only makes sense for the policy-embedding source.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from omegaconf import OmegaConf

from policy_doctor.behaviors.data_support import (
    aggregate_per_cluster,
    compute_all_metrics,
    fit_joint_umap,
)
from policy_doctor.behaviors.behavior_graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
)
from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.curation_pipeline.paths import expand_seeds, get_eval_dir, get_train_dir
from policy_doctor.data.clustering_embeddings import (
    build_windows_from_rollout_timestep_embeddings,
)


_SPECIAL_NODE_IDS = (SUCCESS_NODE_ID, FAILURE_NODE_ID, START_NODE_ID, END_NODE_ID)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_rollout_per_timestep(
    eval_dir: pathlib.Path, layer: str
) -> Tuple[np.ndarray, List[int], List[Any]]:
    """Load rollout per-timestep policy embeddings + episode lengths / successes."""
    emb_path = eval_dir / "policy_embeddings" / f"{layer}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Rollout policy embeddings not found: {emb_path}.  "
            f"Run compute_policy_embeddings.py --layer {layer} first."
        )
    with np.load(emb_path) as f:
        rollout_emb = np.asarray(f["rollout_embeddings"], dtype=np.float32)
    meta = _read_yaml(eval_dir / "episodes" / "metadata.yaml")
    ep_lens = meta.get("episode_lengths") or []
    ep_succ = meta.get("episode_successes", [None] * len(ep_lens))
    return rollout_emb, list(ep_lens), list(ep_succ)


def _load_demo_per_timestep(
    train_dir: pathlib.Path, layer: str
) -> Tuple[np.ndarray, List[int]]:
    emb_path = train_dir / "policy_embeddings_demos" / f"{layer}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Demo policy embeddings not found: {emb_path}.  "
            "Run the compute_policy_embeddings_demos step first."
        )
    with np.load(emb_path) as f:
        demo_emb = np.asarray(f["demo_embeddings"], dtype=np.float32)
        ep_lens = np.asarray(f["episode_lengths"], dtype=np.int64).tolist()
    return demo_emb, ep_lens


def _layer_for_clustering(manifest: Dict[str, Any], cfg) -> str:
    """Resolve the policy-emb layer for a given clustering result.

    Priority:
      1. ``manifest["policy_emb_layer"]``               — if the run wrote it.
      2. Suffix of the clustering name after ``policy_emb_``  — heuristic.
      3. ``cfg.clustering_policy_emb_layer``            — config default.
    """
    direct = manifest.get("policy_emb_layer")
    if direct:
        return str(direct)
    name = str(manifest.get("name") or "")
    if name.startswith("policy_emb_"):
        rest = name[len("policy_emb_"):]
        # Strip _seed{S}_kmeans_k{K} suffix if present.
        import re
        rest = re.sub(r"_seed\d+_(kmeans|gmm|hdbscan)_k\d+$", "", rest)
        if rest:
            return rest
    fallback = OmegaConf.select(cfg, "clustering_policy_emb_layer")
    return str(fallback or "bottleneck_plan_t0")


def _build_rollout_windows(
    rollout_per_timestep_emb: np.ndarray,
    ep_lens: List[int],
    ep_succ: List[Any],
    manifest: Dict[str, Any],
    cfg,
) -> np.ndarray:
    """Re-build the rollout windows that the clustering operated on.

    We deliberately re-build (rather than load ``embeddings_reduced.npy``)
    because we need the *pre-UMAP* high-dim windows for the joint refit.
    """
    window_width = int(manifest.get("window_width") or OmegaConf.select(cfg, "clustering_window_width") or 5)
    stride = int(manifest.get("stride") or OmegaConf.select(cfg, "clustering_stride") or 2)
    aggregation = str(manifest.get("aggregation") or OmegaConf.select(cfg, "clustering_aggregation") or "sum")
    windows, _ = build_windows_from_rollout_timestep_embeddings(
        rollout_per_timestep_emb,
        ep_lens,
        ep_succ,
        window_width=window_width,
        stride=stride,
        aggregation=aggregation,
    )
    return windows


def _build_demo_windows(
    demo_per_timestep_emb: np.ndarray,
    demo_ep_lens: List[int],
    manifest: Dict[str, Any],
    cfg,
) -> np.ndarray:
    """Build demo windows with the *same* (width, stride, aggregation) as the clustering."""
    window_width = int(manifest.get("window_width") or OmegaConf.select(cfg, "clustering_window_width") or 5)
    stride = int(manifest.get("stride") or OmegaConf.select(cfg, "clustering_stride") or 2)
    aggregation = str(manifest.get("aggregation") or OmegaConf.select(cfg, "clustering_aggregation") or "sum")
    # episode_successes is unused for demos (data_support module discards
    # the metadata) — pass Nones to keep the helper happy.
    windows, _ = build_windows_from_rollout_timestep_embeddings(
        demo_per_timestep_emb,
        demo_ep_lens,
        [None] * len(demo_ep_lens),
        window_width=window_width,
        stride=stride,
        aggregation=aggregation,
    )
    return windows


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


class ComputeDataSupportStep(PipelineStep[Dict[str, Any]]):
    """Compute per-node data support for every policy-embedding clustering.

    Result: ``{"data_support_paths": {(seed, k): "<clustering_dir>/data_support.json", ...},
               "skipped":            [<clustering_dir>, ...]}``
    """

    name = "compute_data_support"

    def compute(self) -> Dict[str, Any]:
        from policy_doctor.curation_pipeline.steps.compute_policy_embeddings_demos import (
            ComputePolicyEmbeddingsDemosStep,
        )
        from policy_doctor.curation_pipeline.steps.run_clustering import RunClusteringStep

        cfg = self.cfg

        # --- Config knobs ------------------------------------------------
        metrics = list(
            OmegaConf.select(cfg, "data_support_metrics")
            or [
                "count_in_radius",
                "knn_mean_distance",
                "knn_max_distance",
                "kde_log_density",
                "binary_coverage",
            ]
        )
        radius = float(OmegaConf.select(cfg, "data_support_radius") or 0.5)
        knn_k = int(OmegaConf.select(cfg, "data_support_knn_k") or 10)
        kde_bw_raw = OmegaConf.select(cfg, "data_support_kde_bandwidth")
        kde_bandwidth: Any = kde_bw_raw if kde_bw_raw not in (None, "") else "scott"
        umap_n_components = int(OmegaConf.select(cfg, "data_support_umap_n_components") or 10)
        umap_n_neighbors = int(OmegaConf.select(cfg, "data_support_umap_n_neighbors") or 15)
        umap_random_state = int(OmegaConf.select(cfg, "data_support_umap_random_state") or 0)
        umap_normalize = str(OmegaConf.select(cfg, "data_support_umap_normalize") or "standard")
        save_joint_umap = bool(OmegaConf.select(cfg, "data_support_save_joint_umap") or False)

        # --- Cross-step lookups -----------------------------------------
        prior_clustering = RunClusteringStep(self.cfg, self.parent_run_dir).load() or {}
        prior_demos = (
            ComputePolicyEmbeddingsDemosStep(self.cfg, self.parent_run_dir).load() or {}
        )
        demo_paths = prior_demos.get("demo_embeddings_paths") or {}
        if not demo_paths:
            raise RuntimeError(
                "compute_data_support: no demo embedding paths found.  Run "
                "compute_policy_embeddings_demos before compute_data_support, "
                "or set demo_embeddings_path in the step config."
            )

        # Build (seed, k, clustering_dir) tuples.
        dirs_by_k: Dict[str, Dict[str, str]] = prior_clustering.get("clustering_dirs_by_k") or {}
        flat_dirs: List[Tuple[str, str, pathlib.Path]] = []
        for k_str, by_seed in dirs_by_k.items():
            for seed, p in by_seed.items():
                p_path = pathlib.Path(p)
                if not p_path.is_absolute():
                    p_path = (self.repo_root / p_path).resolve()
                flat_dirs.append((str(seed), str(k_str), p_path))
        if not flat_dirs:
            # Fall back to flat clustering_dirs (no K-sweep).
            simple = prior_clustering.get("clustering_dirs") or {}
            for seed, p in simple.items():
                p_path = pathlib.Path(p)
                if not p_path.is_absolute():
                    p_path = (self.repo_root / p_path).resolve()
                flat_dirs.append((str(seed), "default", p_path))

        if not flat_dirs:
            raise RuntimeError(
                "compute_data_support: no clustering directories found from "
                "RunClusteringStep result.  Run run_clustering first."
            )

        # We also need the eval_dir per seed so we can load rollout embeddings.
        evaluation = OmegaConf.select(cfg, "evaluation") or {}
        attribution = OmegaConf.select(cfg, "attribution") or {}
        baseline = OmegaConf.select(cfg, "baseline") or {}
        eval_date = (
            OmegaConf.select(evaluation, "train_date")
            or OmegaConf.select(cfg, "evaluation.eval_date")
            or OmegaConf.select(cfg, "train_date")
        )
        eval_task = OmegaConf.select(evaluation, "task") or OmegaConf.select(cfg, "task")
        eval_policy = OmegaConf.select(evaluation, "policy") or OmegaConf.select(baseline, "policy")
        eval_output_dir = (
            OmegaConf.select(evaluation, "eval_output_dir") or "data/outputs/eval_save_episodes"
        )
        train_output_dir = (
            OmegaConf.select(attribution, "train_output_dir") or "data/outputs/train"
        )
        train_task = (
            OmegaConf.select(attribution, "task")
            or OmegaConf.select(baseline, "task")
            or eval_task
        )
        train_policy = (
            OmegaConf.select(attribution, "policy")
            or OmegaConf.select(baseline, "policy")
            or eval_policy
        )
        train_date = (
            OmegaConf.select(attribution, "train_date")
            or OmegaConf.select(cfg, "train_date")
            or eval_date
        )
        reference_seed = str(OmegaConf.select(cfg, "reference_seed") or 0)

        result: Dict[str, Any] = {
            "data_support_paths": {},
            "skipped": [],
        }

        for seed, k_str, clu_dir in flat_dirs:
            manifest = _read_yaml(clu_dir / "manifest.yaml")
            influence_source = str(manifest.get("influence_source") or "")
            # Only ``policy_emb`` is supported: the demo-extraction script
            # currently hooks ``DiffusionUnetLowdimPolicy`` U-Net activations,
            # which is the same code path as the rollout-side script.
            # ``pi05_activations`` clusterings would need a separate demo
            # extractor and are explicitly out of scope for v1.
            if influence_source != "policy_emb":
                print(
                    f"  [compute_data_support] skipping {clu_dir.name}: "
                    f"influence_source={influence_source!r} (not policy_emb)"
                )
                (clu_dir / "data_support.skipped").write_text(
                    f"Skipped: influence_source={influence_source!r}; "
                    f"data support v1 supports only the policy_emb source.\n"
                )
                result["skipped"].append(str(clu_dir))
                continue

            # Resolve layer and source dirs (eval / train) per seed.
            layer = _layer_for_clustering(manifest, cfg)
            seed_str = str(seed)

            from influence_visualizer.data_loader import get_eval_dir_for_seed
            eval_dir_base = get_eval_dir(
                eval_output_dir, eval_date, eval_task, eval_policy, 0,
                train_ckpt=OmegaConf.select(evaluation, "train_ckpt") or "latest",
                eval_as_train_seed=True,
            )
            eval_dir_seed = get_eval_dir_for_seed(eval_dir_base, seed_str, reference_seed)
            eval_dir_abs = (self.repo_root / eval_dir_seed).resolve()

            demo_path_str = (
                demo_paths.get(seed_str)
                or demo_paths.get(seed)
            )
            if demo_path_str is None:
                # Fall back to deriving from train_dir + layer.
                train_dir_abs = (
                    self.repo_root
                    / get_train_dir(train_output_dir, train_date, train_task, train_policy, seed_str)
                ).resolve()
                demo_path = train_dir_abs / "policy_embeddings_demos" / f"{layer}.npz"
            else:
                demo_path = pathlib.Path(demo_path_str)

            if not demo_path.exists():
                raise FileNotFoundError(
                    f"compute_data_support: demo embedding file missing for "
                    f"seed={seed_str}: {demo_path}"
                )
            train_dir_abs = demo_path.parent.parent

            print(
                f"  [compute_data_support] seed={seed_str} k={k_str} layer={layer}\n"
                f"    clustering: {clu_dir}\n"
                f"    eval_dir:   {eval_dir_abs}\n"
                f"    demo_emb:   {demo_path}"
            )

            if self.dry_run:
                continue

            cluster_labels = np.load(clu_dir / "cluster_labels.npy").astype(np.int64)

            rollout_per_ts, ep_lens, ep_succ = _load_rollout_per_timestep(eval_dir_abs, layer)
            demo_per_ts, demo_ep_lens = _load_demo_per_timestep(train_dir_abs, layer)

            rollout_windows = _build_rollout_windows(rollout_per_ts, ep_lens, ep_succ, manifest, cfg)
            demo_windows = _build_demo_windows(demo_per_ts, demo_ep_lens, manifest, cfg)
            print(
                f"    rollout windows: {rollout_windows.shape}  "
                f"demo windows: {demo_windows.shape}"
            )

            if rollout_windows.shape[0] != cluster_labels.shape[0]:
                raise RuntimeError(
                    f"Window/label count mismatch for {clu_dir.name}: "
                    f"{rollout_windows.shape[0]} windows vs {cluster_labels.shape[0]} labels. "
                    "Likely the clustering used different (window_width, stride, "
                    "aggregation) than the current config — recompute clustering "
                    "or override these knobs to match the manifest."
                )

            joint = fit_joint_umap(
                demo_windows,
                rollout_windows,
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                random_state=umap_random_state,
                normalize=umap_normalize,
            )

            per_metric, _ctx = compute_all_metrics(
                joint.demo_reduced,
                joint.rollout_reduced,
                metrics=metrics,
                radius=radius,
                knn_k=knn_k,
                kde_bandwidth=kde_bandwidth,
            )

            per_metric_aggregated: Dict[str, Dict[str, Any]] = {}
            for mname, vals in per_metric.items():
                per_cluster = aggregate_per_cluster(
                    vals,
                    cluster_labels,
                    exclude_labels=_SPECIAL_NODE_IDS,
                )
                # JSON-serialise cluster ids as strings.
                per_metric_aggregated[mname] = {str(cid): rec for cid, rec in per_cluster.items()}

            payload = {
                "_config": {
                    "radius": radius,
                    "metrics": metrics,
                    "knn_k": knn_k,
                    "kde_bandwidth": kde_bandwidth,
                    "umap_n_components": umap_n_components,
                    "umap_n_neighbors": umap_n_neighbors,
                    "umap_random_state": umap_random_state,
                    "umap_normalize": umap_normalize,
                    "umap_refit": True,
                    "demo_embeddings_path": str(demo_path),
                    "rollout_embeddings_path": str(
                        eval_dir_abs / "policy_embeddings" / f"{layer}.npz"
                    ),
                    "layer": layer,
                    "n_demo_windows": int(demo_windows.shape[0]),
                    "n_rollout_windows": int(rollout_windows.shape[0]),
                },
                "metrics": per_metric_aggregated,
            }
            out_path = clu_dir / "data_support.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"    → {out_path}")

            if save_joint_umap:
                import joblib
                joblib.dump(joint.umap_model, clu_dir / "joint_umap.joblib")

            result["data_support_paths"][f"{seed_str}__k{k_str}"] = str(out_path)

        return result

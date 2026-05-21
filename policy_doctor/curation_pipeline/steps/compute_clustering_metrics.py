"""Compute evaluation metrics for all demo-sweep clustering results.

Reads every clustering dir under data/demo_sweep/<task>/run_clustering/clustering/
and writes a metrics.json alongside each manifest.yaml.  Skips dirs that already
have an up-to-date metrics.json.

Metrics written per clustering:
  silhouette_mean         Mean silhouette coefficient (sklearn, subsampled ≤ 2000 pts)
  davies_bouldin          Davies-Bouldin index (lower = better)
  calinski_harabasz       Calinski-Harabasz index (higher = better)
  markov_holds            bool — does the majority of states pass the chi² Markov test?
  markov_fraction_holds   fraction of testable states where Markov property holds
  markov_num_tested       number of states that could be tested
  markov_violation_mean   Mean Cramér's V across testable nodes, weighted by transitions
                          through each node.  ∈ [0, 1].  0 = perfect Markov.
  markov_violation_max    Worst per-node Cramér's V — flags single bad-apple clusters.
  testable_fraction       Share of cluster states that pass Markov-testability gates
                          (≥2 distinct predecessors, ≥2 distinct successors, ≥5 transitions).
                          See docs/graph_evaluation.md §2.2.4. Higher = better data coverage.
  swap_rate_per_frame     Stride-fair fraction of frame-pairs where the label changes.
                          See docs/graph_evaluation.md §2.2.2. Lower = more temporally
                          coherent labels.
  distinct_per_episode    Mean # distinct cluster labels visited per episode after run-length
                          collapse. See docs/graph_evaluation.md §2.2.1. Should match the
                          true number of task phases.
  mi_success              Mutual information (nats) between window cluster and episode
                          success/failure outcome. See docs/graph_evaluation.md §2.2.3.
                          Higher = clusters discriminate outcomes better.
  start_v_value           V(START) from Bellman solve — expected success probability

Run with:
  python -m policy_doctor.scripts.run_pipeline \\
    clustering_sweep=all_robomimic \\
    steps=[run_clustering_demo_sweep,compute_clustering_metrics]
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.paths import PROJECT_ROOT

_DEMO_ROOT = PROJECT_ROOT
_SILHOUETTE_MAX_SAMPLES = 2000


class ComputeClusteringMetricsStep(PipelineStep[Dict[str, Any]]):
    """Compute and cache evaluation metrics for all demo-sweep clusterings."""

    name = "compute_clustering_metrics"

    def compute(self) -> Dict[str, Any]:
        sw = OmegaConf.select(self.cfg, "clustering_sweep")
        if sw is None:
            raise ValueError("clustering_sweep config is not set.")

        # Collect all task dirs from config
        tasks_cfg = OmegaConf.select(sw, "tasks", default=None)
        if tasks_cfg is not None:
            task_entries = OmegaConf.to_container(tasks_cfg, resolve=True)
        else:
            task_entries = [{"run_dir": OmegaConf.select(sw, "run_dir")}]

        summary: Dict[str, Any] = {}
        for entry in task_entries:
            run_dir = _DEMO_ROOT / entry["run_dir"]
            clu_root = run_dir / "run_clustering" / "clustering"
            if not clu_root.is_dir():
                print(f"  [skip] {clu_root} not found", flush=True)
                continue

            task_metrics: Dict[str, Any] = {}
            for ordering_dir in sorted(clu_root.iterdir()):
                if not ordering_dir.is_dir():
                    continue
                for clu_dir in sorted(ordering_dir.iterdir()):
                    if not (clu_dir / "cluster_labels.npy").exists():
                        continue
                    metrics = _compute_for_dir(clu_dir)
                    if metrics is not None:
                        task_metrics[f"{ordering_dir.name}/{clu_dir.name}"] = metrics

            task_key = run_dir.name
            summary[task_key] = {"computed": len(task_metrics)}
            print(
                f"  {task_key}: computed metrics for {len(task_metrics)} clusterings",
                flush=True,
            )

        return summary


def _compute_for_dir(clu_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    metrics_path = clu_dir / "metrics.json"
    manifest_path = clu_dir / "manifest.yaml"

    # Skip if metrics already exist and are newer than manifest
    if metrics_path.exists() and manifest_path.exists():
        if metrics_path.stat().st_mtime >= manifest_path.stat().st_mtime:
            print(f"    [skip] {clu_dir.name} (metrics up-to-date)", flush=True)
            return json.loads(metrics_path.read_text())

    print(f"    [metrics] {clu_dir.parent.name}/{clu_dir.name}", flush=True)

    try:
        labels = np.load(clu_dir / "cluster_labels.npy").astype(np.int64)
        emb = np.load(clu_dir / "embeddings_reduced.npy").astype(np.float32)
        with open(clu_dir / "metadata.json") as f:
            metadata = json.load(f)
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
    except Exception as e:
        print(f"    [warn] failed to load {clu_dir.name}: {e}", flush=True)
        return None

    level = manifest.get("level", "rollout")
    metrics: Dict[str, Any] = {}

    # ── Sklearn cluster quality ──────────────────────────────────────────────
    valid = labels != -1
    if valid.sum() >= 2 and len(set(labels[valid])) >= 2:
        emb_v, lab_v = emb[valid], labels[valid]

        # Subsample for silhouette (O(n²))
        if len(lab_v) > _SILHOUETTE_MAX_SAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(lab_v), _SILHOUETTE_MAX_SAMPLES, replace=False)
            emb_s, lab_s = emb_v[idx], lab_v[idx]
        else:
            emb_s, lab_s = emb_v, lab_v

        try:
            from sklearn.metrics import (
                calinski_harabasz_score,
                davies_bouldin_score,
                silhouette_score,
            )
            metrics["silhouette_mean"] = round(
                float(silhouette_score(emb_s, lab_s)), 4
            )
            metrics["davies_bouldin"] = round(
                float(davies_bouldin_score(emb_v, lab_v)), 4
            )
            metrics["calinski_harabasz"] = round(
                float(calinski_harabasz_score(emb_v, lab_v)), 2
            )
        except Exception as e:
            print(f"    [warn] sklearn metrics failed: {e}", flush=True)
    else:
        metrics["silhouette_mean"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None

    # ── Markov property test (chi² + Cramér's V) ───────────────────────────
    n_tested: Optional[int] = None
    try:
        from policy_doctor.behaviors.behavior_graph import test_markov_property

        markov = test_markov_property(
            labels, metadata, level=level,
            significance_level=0.05, method="chi2",
        )
        per_state = markov.get("per_state") or {}
        # Note: per_state values are MarkovTestResult dataclasses, NOT dicts.
        # The previous version used isinstance(r, dict), which was always False
        # — so markov_fraction_holds was always 0. Use attribute access.
        testable = [r for r in per_state.values() if getattr(r, "testable", False)]
        n_tested = len(testable)
        n_holds = sum(1 for r in testable if getattr(r, "markov_holds", False))

        # Cramér's V per node: sqrt(chi² / (N * (min(rows, cols) - 1))).
        # Weighted mean across testable nodes by transitions through the node.
        cramers_v_per_node = []
        weights = []
        for r in testable:
            ct = r.contingency_table
            if ct is None or r.chi2 is None:
                continue
            n_obs = int(ct.sum())
            denom = min(ct.shape) - 1
            if n_obs <= 0 or denom <= 0:
                continue
            v = float(np.sqrt(r.chi2 / (n_obs * denom)))
            v = min(max(v, 0.0), 1.0)  # clamp numerical drift
            cramers_v_per_node.append(v)
            weights.append(n_obs)
        if cramers_v_per_node:
            arr = np.array(cramers_v_per_node)
            wts = np.array(weights, dtype=float)
            metrics["markov_violation_mean"] = round(
                float((arr * wts).sum() / wts.sum()), 4
            )
            metrics["markov_violation_max"] = round(float(arr.max()), 4)
        else:
            metrics["markov_violation_mean"] = None
            metrics["markov_violation_max"] = None

        metrics["markov_holds"] = bool(markov.get("markov_holds"))
        metrics["markov_fraction_holds"] = (
            round(n_holds / n_tested, 4) if n_tested > 0 else None
        )
        metrics["markov_num_tested"] = n_tested
    except Exception as e:
        print(f"    [warn] Markov test failed: {e}", flush=True)
        metrics["markov_holds"] = None
        metrics["markov_fraction_holds"] = None
        metrics["markov_num_tested"] = None
        metrics["markov_violation_mean"] = None
        metrics["markov_violation_max"] = None

    # ── V(START) from Bellman solve ──────────────────────────────────────────
    try:
        from policy_doctor.behaviors.behavior_graph import (
            BehaviorGraph,
            START_NODE_ID,
        )

        graph = BehaviorGraph.from_cluster_assignments(labels, metadata, level=level)
        values = graph.compute_values()
        metrics["start_v_value"] = round(float(values.get(START_NODE_ID, 0.0)), 4)
    except Exception as e:
        print(f"    [warn] V(START) computation failed: {e}", flush=True)
        metrics["start_v_value"] = None

    # ── Coverage metrics (docs/graph_evaluation.md §2.2) ────────────────────
    try:
        K = max(int(labels.max()) + 1, 1) if len(labels) else 0
        if n_tested is not None and K > 0:
            metrics["testable_fraction"] = round(n_tested / K, 4)
        else:
            metrics["testable_fraction"] = None
    except Exception as e:
        print(f"    [warn] testable_fraction failed: {e}", flush=True)
        metrics["testable_fraction"] = None

    try:
        cov = _coverage_metrics(labels, metadata, level)
        metrics["swap_rate_per_frame"] = cov.get("swap_rate_per_frame")
        metrics["distinct_per_episode"] = cov.get("distinct_per_episode")
        metrics["mi_success"] = cov.get("mi_success")
    except Exception as e:
        print(f"    [warn] coverage metrics failed: {e}", flush=True)
        metrics["swap_rate_per_frame"] = None
        metrics["distinct_per_episode"] = None
        metrics["mi_success"] = None

    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _coverage_metrics(
    labels: np.ndarray, metadata: List[Dict], level: str
) -> Dict[str, Optional[float]]:
    """Per-frame swap rate, mean distinct clusters/episode, MI(label; success).

    All three operate on the labels + metadata. See docs/graph_evaluation.md
    §2.2.1, §2.2.2, §2.2.3.
    """
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"

    # Group windows by episode, sorted by window_start (or timestep).
    episodes: Dict[Any, List[Dict]] = {}
    for i, m in enumerate(metadata):
        if int(labels[i]) == -1:
            continue
        ep = m.get(ep_key)
        if ep is None:
            continue
        episodes.setdefault(ep, []).append({
            "i": i,
            "start": m.get("window_start", m.get("timestep", 0)),
            "end": m.get("window_end", m.get("timestep", 0) + 1),
            "success": m.get("success"),
        })
    for ep in episodes:
        episodes[ep].sort(key=lambda r: r["start"])

    # Swap rate per frame: sum(swaps) / sum(frames per episode).
    n_swaps = 0
    n_frames = 0
    distinct_counts: List[int] = []
    for ep, rows in episodes.items():
        if not rows:
            continue
        labs = [int(labels[r["i"]]) for r in rows]
        n_swaps += sum(1 for j in range(1, len(labs)) if labs[j] != labs[j - 1])
        # Frame total: highest window_end seen for the episode (1-indexed-style).
        n_frames += int(rows[-1]["end"])
        # Run-length-collapsed distinct count.
        rle: List[int] = []
        for lab in labs:
            if not rle or rle[-1] != lab:
                rle.append(lab)
        distinct_counts.append(len(set(rle)))

    swap_rate = (n_swaps / n_frames) if n_frames > 0 else None
    distinct_per_ep = (float(np.mean(distinct_counts)) if distinct_counts else None)

    # MI(cluster_label; success) in nats. Skip if all episodes share one outcome.
    mi_succ: Optional[float] = None
    succ = np.array(
        [bool(m.get("success")) for m in metadata],
        dtype=object,
    )
    keep = (labels != -1) & np.array(
        [m.get("success") is not None for m in metadata]
    )
    if keep.sum() >= 2:
        lab_k = labels[keep].astype(int)
        succ_k = succ[keep].astype(int)
        if len(set(succ_k)) >= 2:
            N = float(lab_k.size)
            # Joint counts via 2D histogram
            uniq_lab = np.unique(lab_k)
            joint = np.zeros((uniq_lab.size, 2), dtype=float)
            for j, c in enumerate(uniq_lab):
                m = lab_k == c
                joint[j, 0] = (succ_k[m] == 0).sum()
                joint[j, 1] = (succ_k[m] == 1).sum()
            p_joint = joint / N
            p_lab = p_joint.sum(axis=1, keepdims=True)
            p_succ = p_joint.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = p_joint / (p_lab @ p_succ)
                terms = np.where(p_joint > 0, p_joint * np.log(ratio), 0.0)
            mi_succ = float(np.nansum(terms))

    return {
        "swap_rate_per_frame": round(swap_rate, 6) if swap_rate is not None else None,
        "distinct_per_episode": round(distinct_per_ep, 3) if distinct_per_ep is not None else None,
        "mi_success": round(mi_succ, 4) if mi_succ is not None else None,
    }

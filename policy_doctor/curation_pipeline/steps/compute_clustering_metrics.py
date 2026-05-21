"""Compute evaluation metrics for all demo-sweep clustering results.

Reads every clustering dir under data/demo_sweep/<task>/run_clustering/clustering/
and writes a metrics.json alongside each manifest.yaml.  Skips dirs that already
have an up-to-date metrics.json.

Metrics written per clustering:
  silhouette_mean       Mean silhouette coefficient (sklearn, subsampled ≤ 2000 pts)
  davies_bouldin        Davies-Bouldin index (lower = better)
  calinski_harabasz     Calinski-Harabasz index (higher = better)
  markov_holds          bool — does the majority of states pass the chi² Markov test?
  markov_fraction_holds fraction of testable states where Markov property holds
  markov_num_tested     number of states that could be tested
  start_v_value         V(START) from Bellman solve — expected success probability

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

    # ── Markov property test (chi²) ──────────────────────────────────────────
    try:
        from policy_doctor.behaviors.behavior_graph import test_markov_property

        markov = test_markov_property(
            labels, metadata, level=level,
            significance_level=0.05, method="chi2",
        )
        n_tested = int(markov.get("num_states_tested") or 0)
        n_holds = sum(
            1 for r in (markov.get("per_state") or {}).values()
            if isinstance(r, dict) and r.get("markov_holds") is True
        )
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

    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics

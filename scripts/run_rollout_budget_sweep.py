"""Rollout-budget sweep — how many rollouts do we need for stable MV estimates?

For a fixed trunk (UMAP-50 over policy_emb_bottleneck_plan_t0 per-timestep
features), this script:

  1. Loads the trunk + episode metadata.
  2. For each N ∈ {20, 50, 100, 200, 300, 400, 500} and each K ∈ {5, 10, 15, 20}:
     - Subsamples N episodes (multiple subsample seeds for noise estimate).
     - Rebuilds a sliding-window dataset (w=5, s=1) restricted to those eps.
     - Fits KMeans (one shot).
     - Computes MV₁, MV₂, MV₃ with episode-bootstrap CIs.

Outputs to /mnt/ssdB/erik/cupid_data/graph_simplification/results/rollout_budget/
and an aggregated JSON at docs/rollout_budget_results/.

Usage:
    PYTHONPATH=. python scripts/run_rollout_budget_sweep.py \
      --eval_dir <500-rollout eval dir>      \
      --task transport_mh_graph_simplification \
      --rep policy_emb --layer bottleneck_plan_t0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SSD_ROOT = pathlib.Path("/mnt/ssdB/erik/cupid_data/graph_simplification")
TRUNKS_DIR = SSD_ROOT / "trunks_budget"
RESULTS_DIR = SSD_ROOT / "results" / "rollout_budget"
LOGS_DIR = SSD_ROOT / "logs" / "rollout_budget"
AGG_DIR = _REPO_ROOT / "docs" / "rollout_budget_results"

ROLLOUT_BUDGETS = [20, 50, 100, 200, 300, 400, 500]
K_VALUES = [5, 10, 15, 20]
WINDOW = 5
STRIDE = 1
N_SUBSAMPLE_SEEDS = 3       # estimate noise across episode draws
UMAP_N_COMPONENTS = 50
N_BOOTSTRAP = 100
ORDERS = (1, 2, 3)


def _build_windows(
    emb_ts: np.ndarray,
    ep_lengths: List[int],
    ep_successes: List[bool],
    keep_ep_idxs: List[int],
    w: int, s: int,
) -> Tuple[np.ndarray, List[Dict]]:
    """Build a sliding-window dataset over the SUBSET of episodes given by
    keep_ep_idxs, indexing into the concatenated per-timestep trunk."""
    offsets = np.cumsum([0] + list(ep_lengths))
    feats: List[np.ndarray] = []
    meta: List[Dict] = []
    for new_ep, old_ep in enumerate(keep_ep_idxs):
        a, b = int(offsets[old_ep]), int(offsets[old_ep + 1])
        ep_emb = emb_ts[a:b]
        T = ep_emb.shape[0]
        if T < w:
            continue
        for start in range(0, T - w + 1, s):
            feats.append(ep_emb[start:start + w].mean(axis=0))
            meta.append({
                "rollout_idx": int(new_ep),
                "window_start": int(start),
                "window_end": int(start + w),
                "window_width": int(w),
                "success": bool(ep_successes[old_ep]),
            })
    if not feats:
        return np.zeros((0, emb_ts.shape[1]), dtype=np.float32), []
    return np.asarray(feats, dtype=np.float32), meta


def _one_eval(args) -> Optional[Dict]:
    (task_label, trunk_path, N, K, subsample_seed) = args
    import joblib
    from sklearn.cluster import KMeans
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.behaviors.simplification.metrics import (
        compute_metrics, bootstrap_mv_ci, markov_violation_coverage,
    )

    trunk = pathlib.Path(trunk_path)
    emb_ts = np.load(trunk / "timestep_embeddings.npy").astype(np.float32)
    ep_meta = json.loads((trunk / "ep_meta.json").read_text())
    ep_lengths = list(ep_meta["episode_lengths"])
    ep_successes = list(ep_meta["episode_successes"])

    n_total_eps = len(ep_lengths)
    if N > n_total_eps:
        return None

    rng = np.random.RandomState(1000 * subsample_seed + N)
    keep = rng.choice(n_total_eps, size=N, replace=False).tolist()
    emb_w, meta = _build_windows(emb_ts, ep_lengths, ep_successes, keep, WINDOW, STRIDE)
    if len(emb_w) == 0:
        return None

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(emb_w).astype(np.int64)

    g = BehaviorGraph.from_cluster_assignments(labels, meta, level="rollout")
    n_nodes = len(g.cluster_nodes)
    metrics = compute_metrics(g, labels, meta, original_labels=labels, node_mapping={})

    out: Dict = {
        "task": task_label,
        "trunk": trunk.name,
        "N_rollouts": int(N),
        "K": int(K),
        "subsample_seed": int(subsample_seed),
        "n_windows": int(len(meta)),
        "n_cluster_nodes": int(n_nodes),
        "metrics": metrics.as_dict(),
    }
    for order in ORDERS:
        p, lo, hi = bootstrap_mv_ci(
            labels, meta, node_mapping={}, level="rollout", order=order,
            n_bootstrap=N_BOOTSTRAP, current_labels=labels, rng_seed=42 + order,
        )
        out[f"mv{order}_point"] = float(p)
        out[f"mv{order}_ci_lo"] = float(lo)
        out[f"mv{order}_ci_hi"] = float(hi)
        cov = markov_violation_coverage(
            labels, meta, node_mapping={}, level="rollout", order=order,
            current_labels=labels,
        )
        out[f"mv{order}_coverage_fraction"] = cov["coverage_fraction"]
        out[f"mv{order}_n_states_passing"] = cov["n_states_passing"]
        out[f"mv{order}_n_states_total"] = cov["n_states_total"]
    return out


def _budget_worker(args: Tuple[Tuple[str, str, int, int, int], str]) -> Tuple[Tuple, str]:
    u, out_dir_s = args
    try:
        r = _one_eval(u)
        if r is None:
            return u, "skip(N>n_eps)"
        task_label, _, N, K, seed = u
        out = pathlib.Path(out_dir_s) / f"{task_label}__N{N}__K{K}__seed{seed}.json"
        out.write_text(json.dumps(r, indent=2))
        return u, "ok"
    except Exception as e:  # noqa: BLE001
        return u, f"err:{type(e).__name__}:{e}"


def run_sweep(
    task_label: str,
    trunk_path: str,
    rollout_budgets: List[int],
    k_values: List[int],
    n_subsample_seeds: int,
    parallel: int,
    out_dir: pathlib.Path,
    force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    units: List[Tuple[str, str, int, int, int]] = []
    for N in rollout_budgets:
        for K in k_values:
            for seed in range(n_subsample_seeds):
                out = out_dir / f"{task_label}__N{N}__K{K}__seed{seed}.json"
                if not force and out.exists():
                    continue
                units.append((task_label, trunk_path, N, K, seed))

    print(f"=== Rollout-budget sweep on {task_label}: {len(units)} units, parallel={parallel} ===",
          flush=True)
    if not units:
        return

    inputs = [(u, str(out_dir)) for u in units]
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
        for u, status in ex.map(_budget_worker, inputs):
            done += 1
            _, _, N, K, seed = u
            print(f"  [{done}/{len(units)}] {status:20s} N={N} K={K} seed={seed}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_label", required=True,
                    help="Output name prefix (e.g. transport_mh_graph_simplification).")
    ap.add_argument("--trunk", required=True,
                    help="Trunk directory (UMAP-reduced per-timestep embeddings + ep_meta.json).")
    ap.add_argument("--budgets", type=int, nargs="+", default=ROLLOUT_BUDGETS)
    ap.add_argument("--ks", type=int, nargs="+", default=K_VALUES)
    ap.add_argument("--n_subsample_seeds", type=int, default=N_SUBSAMPLE_SEEDS)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out_dir", default=str(RESULTS_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_sweep(
        task_label=args.task_label,
        trunk_path=args.trunk,
        rollout_budgets=args.budgets,
        k_values=args.ks,
        n_subsample_seeds=args.n_subsample_seeds,
        parallel=args.n_jobs,
        out_dir=pathlib.Path(args.out_dir),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

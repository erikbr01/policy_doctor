"""Model-selection sweep for behavior-graph creation under Markov-violation loss.

Sweeps (representation × window × stride × K) across robomimic baseline tasks
(transport_mh_jan28, square_mh_feb5, lift_mh_jan26) and reports MV₁, MV₂, MV₃
with episode-bootstrap CIs.

Layout (all artifacts on /mnt/ssdB; docs/markdown only on the worktree):
  /mnt/ssdB/erik/cupid_data/graph_simplification/
    trunks/<task>__<rep>/                  # UMAP-50 per-timestep trunks (mode-A)
    clusterings/<task>__<rep>__w<w>_s<s>__K<K>/  # KMeans branches (mode-B)
    results/k_sweep/<task>__<rep>__w<w>_s<s>__K<K>.json
    logs/<phase>/<unit>.log
  docs/k_sweep_results/                     # final aggregated JSON + plots

Phases (run with --phase):
  trunks    : Build UMAP trunks per (task, rep). Sequential per task to avoid
              thrashing 32 cores. policy_emb requires precomputed embeddings
              under <eval_dir>/policy_embeddings/<layer>.npz (run
              `compute_policy_embeddings.py` first; see EVAL_DIRS below).
  cluster   : Build all (w, s, K) clustering branches per trunk. Parallel pool.
  eval      : Compute MV₁/MV₂/MV₃ with bootstrap CIs per clustering. Parallel.
  aggregate : Merge per-clustering JSONs into one summary at docs/k_sweep_results/.
  all       : Run trunks → cluster → eval → aggregate.

Usage:
  PYTHONPATH=. python scripts/run_simplification_model_selection.py --phase trunks
  PYTHONPATH=. python scripts/run_simplification_model_selection.py --phase cluster --n_jobs 8
  PYTHONPATH=. python scripts/run_simplification_model_selection.py --phase eval --n_bootstrap 100
  PYTHONPATH=. python scripts/run_simplification_model_selection.py --phase aggregate
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SSD_ROOT = pathlib.Path("/mnt/ssdB/erik/cupid_data/graph_simplification")
TRUNKS_DIR = SSD_ROOT / "trunks"
CLUSTERINGS_DIR = SSD_ROOT / "clusterings"
RESULTS_DIR = SSD_ROOT / "results" / "k_sweep"
LOGS_DIR = SSD_ROOT / "logs"
AGG_DIR = _REPO_ROOT / "docs" / "k_sweep_results"

EVAL_DIRS: Dict[str, str] = {
    "transport_mh_jan28": (
        "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/jan28"
        "/jan28_train_diffusion_unet_lowdim_transport_mh_0/latest"
    ),
    "square_mh_feb5": (
        "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/feb5"
        "/feb5_train_diffusion_unet_lowdim_square_mh_0/latest"
    ),
    "lift_mh_jan26": (
        "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/jan26"
        "/jan26_train_diffusion_unet_lowdim_lift_mh_0/latest"
    ),
}

TRAIN_DIRS: Dict[str, str] = {
    "transport_mh_jan28": (
        "/mnt/ssdB/erik/cupid_data/outputs/train/jan28"
        "/jan28_train_diffusion_unet_lowdim_transport_mh_0"
    ),
    "square_mh_feb5": (
        "/mnt/ssdB/erik/cupid_data/outputs/train/feb5"
        "/feb5_train_diffusion_unet_lowdim_square_mh_0"
    ),
    "lift_mh_jan26": (
        "/mnt/ssdB/erik/cupid_data/outputs/train/jan26"
        "/jan26_train_diffusion_unet_lowdim_lift_mh_0"
    ),
}

REPRESENTATIONS = ["infembed", "policy_emb"]
WINDOW_STRIDES: List[Tuple[int, int]] = [(3, 1), (5, 1), (8, 1)]
K_VALUES = [3, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30]
UMAP_N_COMPONENTS = 50
SEED = 42
POLICY_EMB_LAYER = "bottleneck_plan_t0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trunk_dir(task: str, rep: str) -> pathlib.Path:
    return TRUNKS_DIR / f"{task}__{rep}"


def clustering_dir(task: str, rep: str, w: int, s: int, K: int) -> pathlib.Path:
    return CLUSTERINGS_DIR / f"{task}__{rep}__w{w}_s{s}__K{K}"


def result_path(task: str, rep: str, w: int, s: int, K: int) -> pathlib.Path:
    return RESULTS_DIR / f"{task}__{rep}__w{w}_s{s}__K{K}.json"


def _run_subprocess(name: str, cmd: List[str], log_path: pathlib.Path,
                    env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


# ---------------------------------------------------------------------------
# Phase 1: trunks (UMAP per task, rep)
# ---------------------------------------------------------------------------

def _trunk_cmd(task: str, rep: str) -> List[str]:
    eval_dir = EVAL_DIRS[task]
    args = [
        "conda", "run", "-n", "policy_doctor", "--no-capture-output",
        "python", str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
        "--representation", rep,
        "--eval_dir", eval_dir,
        "--prescale", "standard",
        "--umap_n_components", str(UMAP_N_COMPONENTS),
        "--seed", str(SEED),
        "--timestep_embed_only",
        "--out_dir", str(trunk_dir(task, rep)),
    ]
    if rep == "policy_emb":
        args += ["--layer", POLICY_EMB_LAYER]
    return args


def _run_trunk_unit(unit: Tuple[str, str]) -> Tuple[Tuple[str, str], int, float]:
    task, rep = unit
    log = LOGS_DIR / "trunks" / f"{task}__{rep}.log"
    t0 = time.time()
    rc = _run_subprocess(f"trunk:{task}:{rep}", _trunk_cmd(task, rep), log)
    return unit, rc, time.time() - t0


def phase_trunks(parallel: int = 1, force: bool = False) -> None:
    units = [(t, r) for t in EVAL_DIRS for r in REPRESENTATIONS]
    todo = []
    for task, rep in units:
        out = trunk_dir(task, rep)
        if not force and (out / "embed_manifest.yaml").exists():
            print(f"  SKIP trunk: {out.name}")
            continue
        if rep == "policy_emb":
            emb_npz = pathlib.Path(EVAL_DIRS[task]) / "policy_embeddings" / f"{POLICY_EMB_LAYER}.npz"
            if not emb_npz.exists():
                print(f"  SKIP trunk (need policy_emb extraction first): {out.name}")
                print(f"    Run: conda run -n cupid_torch2 python third_party/cupid/compute_policy_embeddings.py "
                      f"--train_dir {TRAIN_DIRS[task]} --eval_dir {EVAL_DIRS[task]} --layer {POLICY_EMB_LAYER}")
                continue
        todo.append((task, rep))

    print(f"=== Phase trunks: {len(todo)} units, parallel={parallel} ===", flush=True)
    if not todo:
        return

    if parallel <= 1:
        for u in todo:
            _, rc, dt = _run_trunk_unit(u)
            print(f"  {'OK' if rc == 0 else f'FAIL({rc})'} trunk {u[0]}__{u[1]} {dt:.1f}s", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
            for u, rc, dt in ex.map(_run_trunk_unit, todo):
                print(f"  {'OK' if rc == 0 else f'FAIL({rc})'} trunk {u[0]}__{u[1]} {dt:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Phase 2: clusterings (KMeans branches)
# ---------------------------------------------------------------------------

def _cluster_cmd(task: str, rep: str, w: int, s: int, K: int) -> List[str]:
    return [
        "conda", "run", "-n", "policy_doctor", "--no-capture-output",
        "python", str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
        "--representation", rep,
        "--eval_dir", EVAL_DIRS[task],
        "--timestep_embed_dir", str(trunk_dir(task, rep)),
        "--window_width", str(w),
        "--stride", str(s),
        "--aggregation", "mean",
        "--n_clusters", str(K),
        "--seed", str(SEED),
        "--out_dir", str(clustering_dir(task, rep, w, s, K)),
    ]


def _run_cluster_unit(u: Tuple[str, str, int, int, int]) -> Tuple[Tuple, int, float]:
    task, rep, w, s, K = u
    log = LOGS_DIR / "cluster" / f"{task}__{rep}__w{w}_s{s}__K{K}.log"
    t0 = time.time()
    rc = _run_subprocess(
        f"cluster:{task}:{rep}:w{w}s{s}K{K}",
        _cluster_cmd(task, rep, w, s, K), log,
    )
    return u, rc, time.time() - t0


def phase_cluster(parallel: int = 8, force: bool = False) -> None:
    units = []
    for task in EVAL_DIRS:
        for rep in REPRESENTATIONS:
            if not (trunk_dir(task, rep) / "embed_manifest.yaml").exists():
                continue
            for (w, s) in WINDOW_STRIDES:
                for K in K_VALUES:
                    out = clustering_dir(task, rep, w, s, K)
                    if not force and (out / "manifest.yaml").exists():
                        continue
                    units.append((task, rep, w, s, K))

    print(f"=== Phase cluster: {len(units)} units, parallel={parallel} ===", flush=True)
    if not units:
        return

    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
        for u, rc, dt in ex.map(_run_cluster_unit, units):
            done += 1
            tag = f"{u[0]}__{u[1]}__w{u[2]}_s{u[3]}__K{u[4]}"
            print(f"  [{done}/{len(units)}] {'OK' if rc == 0 else f'FAIL({rc})'} {tag} {dt:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Phase 3: MV eval with bootstrap (single-process worker)
# ---------------------------------------------------------------------------

_MV_ORDERS = (1, 2, 3)


def _eval_one(unit: Tuple[str, str, int, int, int], n_bootstrap: int) -> Optional[Dict]:
    import numpy as np
    import yaml
    from policy_doctor.behaviors.behavior_graph import BehaviorGraph
    from policy_doctor.behaviors.simplification.metrics import (
        compute_metrics,
        bootstrap_mv_ci,
        markov_violation_against_original_bits,
        markov_violation_coverage,
    )

    task, rep, w, s, K = unit
    cdir = clustering_dir(task, rep, w, s, K)
    if not (cdir / "manifest.yaml").exists():
        return None
    labels = np.load(cdir / "cluster_labels.npy").astype(np.int64)
    meta = json.loads((cdir / "metadata.json").read_text())
    manifest = yaml.safe_load((cdir / "manifest.yaml").read_text()) or {}
    level = manifest.get("level", "rollout")
    ep_key = "rollout_idx" if level == "rollout" else "demo_idx"
    n_eps = len({m[ep_key] for m in meta})
    g = BehaviorGraph.from_cluster_assignments(labels, meta, level=level)
    n_nodes = len(g.cluster_nodes)

    metrics = compute_metrics(g, labels, meta, original_labels=labels, node_mapping={})
    out: Dict = {
        "task": task, "rep": rep, "w": w, "s": s, "K": K,
        "n_samples": int(len(labels)), "n_episodes": int(n_eps),
        "n_cluster_nodes": int(n_nodes),
        "level": level,
        "n_bootstrap": int(n_bootstrap),
        "metrics": metrics.as_dict(),
    }
    # MV1, MV2, MV3 bootstrap CIs + coverage diagnostic per order.
    for order in _MV_ORDERS:
        p, lo, hi = bootstrap_mv_ci(
            labels, meta, node_mapping={}, level=level, order=order,
            n_bootstrap=n_bootstrap, current_labels=labels, rng_seed=42 + order,
        )
        out[f"mv{order}_point"] = float(p)
        out[f"mv{order}_ci_lo"] = float(lo)
        out[f"mv{order}_ci_hi"] = float(hi)
        cov = markov_violation_coverage(
            labels, meta, node_mapping={}, level=level, order=order,
            current_labels=labels,
        )
        out[f"mv{order}_coverage_fraction"] = cov["coverage_fraction"]
        out[f"mv{order}_n_states_passing"] = cov["n_states_passing"]
        out[f"mv{order}_n_states_total"] = cov["n_states_total"]
        out[f"mv{order}_total_pairs"] = cov["total_pairs"]
        out[f"mv{order}_gated_pairs"] = cov["gated_pairs"]

    return out


def _eval_worker(args: Tuple[Tuple[str, str, int, int, int], int]) -> Tuple[Tuple, str]:
    u, n_bootstrap = args
    try:
        r = _eval_one(u, n_bootstrap)
        if r is None:
            return u, "missing"
        out = result_path(*u)
        out.write_text(json.dumps(r, indent=2))
        return u, "ok"
    except Exception as e:  # noqa: BLE001
        return u, f"error:{type(e).__name__}:{e}"


def phase_eval(parallel: int = 8, n_bootstrap: int = 100, force: bool = False) -> None:
    units = []
    for task in EVAL_DIRS:
        for rep in REPRESENTATIONS:
            for (w, s) in WINDOW_STRIDES:
                for K in K_VALUES:
                    out = result_path(task, rep, w, s, K)
                    cdir = clustering_dir(task, rep, w, s, K)
                    if not (cdir / "manifest.yaml").exists():
                        continue
                    if not force and out.exists():
                        continue
                    units.append((task, rep, w, s, K))

    print(f"=== Phase eval: {len(units)} units, parallel={parallel}, n_bootstrap={n_bootstrap} ===",
          flush=True)
    if not units:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    inputs = [(u, n_bootstrap) for u in units]
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
        for u, status in ex.map(_eval_worker, inputs):
            done += 1
            tag = f"{u[0]}__{u[1]}__w{u[2]}_s{u[3]}__K{u[4]}"
            print(f"  [{done}/{len(units)}] {status:12s} {tag}", flush=True)


# ---------------------------------------------------------------------------
# Phase 4: aggregate
# ---------------------------------------------------------------------------

def phase_aggregate() -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        rows.append(json.loads(p.read_text()))
    out = AGG_DIR / "k_sweep_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"  wrote {len(rows)} rows to {out}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["trunks", "cluster", "eval", "aggregate", "all"])
    ap.add_argument("--n_jobs", type=int, default=8,
                    help="Worker pool size for cluster/eval phases.")
    ap.add_argument("--n_jobs_trunks", type=int, default=2,
                    help="Parallel UMAP trunks. Keep low — UMAP uses many threads internally.")
    ap.add_argument("--n_bootstrap", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for d in (TRUNKS_DIR, CLUSTERINGS_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.phase in ("trunks", "all"):
        phase_trunks(parallel=args.n_jobs_trunks, force=args.force)
    if args.phase in ("cluster", "all"):
        phase_cluster(parallel=args.n_jobs, force=args.force)
    if args.phase in ("eval", "all"):
        phase_eval(parallel=args.n_jobs, n_bootstrap=args.n_bootstrap, force=args.force)
    if args.phase in ("aggregate", "all"):
        phase_aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

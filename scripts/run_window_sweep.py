"""Window-parameter sweep for E1 cluster coherence.

Pipeline architecture:
  UMAP is fitted ONCE per representation on per-timestep features (trunk).
  Window config and K are applied AFTER UMAP (branches).

  rep  →  extract per-timestep features  →  UMAP  ─┬─ w=(1,1) ─ K=5,10,15,20
                                                     ├─ w=(2,2) ─ K=5,10,15,20
                                                     ├─ w=(3,2) ─ K=5,10,15,20
                                                     └─ w=(5,5) ─ K=5,10,15,20

Usage:
    python scripts/run_window_sweep.py --phase build [--dry_run]
    python scripts/run_window_sweep.py --phase eval  [--dry_run]
    python scripts/run_window_sweep.py --phase report
    python scripts/run_window_sweep.py --phase all
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

EVAL_DIR = (
    "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27"
    "/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest"
)

# Architecture: "timestep_first" (UMAP on per-timestep features, then window)
#               "window_first"   (window, then UMAP — original pipeline, matches F1–F10)
ARCHITECTURE = "window_first"

# Roots differ by architecture so results from each don't clobber each other.
if ARCHITECTURE == "window_first":
    SWEEP_EMBED_ROOT = None  # not used in window_first mode
    SWEEP_CLUST_ROOT = pathlib.Path("/mnt/ssdB/erik/cupid_data/window_sweep_wf_clusterings")
    SWEEP_EVAL_ROOT  = _REPO_ROOT / "experiments" / "window_sweep_wf"
    REPORT_PATH      = _REPO_ROOT / "docs" / "experiments" / "results" / "e1_window_sweep_wf_results.md"
else:
    SWEEP_EMBED_ROOT = pathlib.Path("/mnt/ssdB/erik/cupid_data/window_sweep_embeddings")
    SWEEP_CLUST_ROOT = pathlib.Path("/mnt/ssdB/erik/cupid_data/window_sweep_clusterings_full")
    SWEEP_EVAL_ROOT  = _REPO_ROOT / "experiments" / "window_sweep"
    REPORT_PATH      = _REPO_ROOT / "docs" / "experiments" / "results" / "e1_window_sweep_results.md"

# window_first: infembed only (state/state_action already tested; old arch is the benchmark)
# timestep_first: all three representations
REPRESENTATIONS = ["infembed"] if ARCHITECTURE == "window_first" else ["infembed", "state", "state_action"]

# Paired (window_width, stride) — NOT a cartesian product.
WINDOW_PARAMS: List[Tuple[int, int]] = [(1, 1), (2, 2), (3, 2), (5, 5)]

KS = [5, 10, 15, 20]

# E1 eval config
SEEDS = [42, 43, 44]
N_EXAMPLE = 3
N_QUERY   = 3
N_REPS    = 3
COMPOSITE_SIZE   = 512
IMAGE_MAX_PIXELS = 512 * 512

# Clustering hyperparams
AGGREGATION       = "mean"
PRESCALE          = "standard"
# window_first uses 100D to match the original pipeline (F1–F10 used 100D).
# timestep_first uses 50D (safe for all reps; state=59D, state_action=79D).
UMAP_N_COMPONENTS = 100 if ARCHITECTURE == "window_first" else 50
CLUSTER_SEED      = 42

_REP_KWARGS: Dict[str, Dict[str, str]] = {
    "infembed":     {},
    "state":        {"obs_strategy": "current"},
    "state_action": {"obs_strategy": "current", "action_strategy": "executed"},
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _embed_dir(rep: str) -> pathlib.Path:
    return SWEEP_EMBED_ROOT / rep

def _clust_dir(rep: str, w: int, s: int, K: int) -> pathlib.Path:
    return SWEEP_CLUST_ROOT / f"{rep}__w{w}_s{s}__K{K}"

def _eval_out(rep: str, w: int, s: int, K: int, seed: int) -> pathlib.Path:
    return SWEEP_EVAL_ROOT / f"{rep}__w{w}_s{s}__K{K}__seed{seed}"

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _run_parallel(jobs: List[Tuple[str, List[str], pathlib.Path]], dry_run: bool) -> None:
    procs: List[Tuple[str, subprocess.Popen, pathlib.Path]] = []
    for name, cmd, log in jobs:
        if dry_run:
            print("DRY:", " ".join(cmd))
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((name, p, log))
        print(f"  started {name} (pid {p.pid})")
    for name, p, log in procs:
        rc = p.wait()
        print(f"  {'OK' if rc == 0 else f'FAILED(exit {rc})'}: {name}")
        if rc != 0:
            print(f"    log: {log}")

def _base_cmd(rep: str) -> List[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
        "--representation", rep,
        "--eval_dir", EVAL_DIR,
        "--prescale", PRESCALE,
        "--umap_n_components", str(UMAP_N_COMPONENTS),
        "--seed", str(CLUSTER_SEED),
    ]
    for k, v in _REP_KWARGS.get(rep, {}).items():
        cmd += [f"--{k}", v]
    return cmd

# ---------------------------------------------------------------------------
# Phase 1 — build
# ---------------------------------------------------------------------------

def phase_build(dry_run: bool = False) -> None:
    SWEEP_CLUST_ROOT.mkdir(parents=True, exist_ok=True)

    if ARCHITECTURE == "window_first":
        # Original pipeline: window → normalise → UMAP → kmeans.
        # Each (rep, w, s, K) is fully independent — no shared trunk.
        print(f"=== Build [window_first]: {len(REPRESENTATIONS)} reps × "
              f"{len(WINDOW_PARAMS)} windows × {len(KS)} K = "
              f"{len(REPRESENTATIONS)*len(WINDOW_PARAMS)*len(KS)} clusterings (parallel) ===")
        jobs = []
        for rep in REPRESENTATIONS:
            for (w, s) in WINDOW_PARAMS:
                for K in KS:
                    cout = _clust_dir(rep, w, s, K)
                    if (cout / "manifest.yaml").exists():
                        print(f"  SKIP (exists): {cout.name}")
                        continue
                    cmd = (
                        _base_cmd(rep) +
                        ["--window_width", str(w),
                         "--stride", str(s),
                         "--aggregation", AGGREGATION,
                         "--n_clusters", str(K),
                         "--out_dir", str(cout)]
                    )
                    log = SWEEP_CLUST_ROOT / f"build_{cout.name}.log"
                    jobs.append((cout.name, cmd, log))
        _run_parallel(jobs, dry_run)

    else:
        # Timestep-first pipeline: UMAP trunk once per rep, then window+kmeans branches.
        SWEEP_EMBED_ROOT.mkdir(parents=True, exist_ok=True)

        print("=== Stage 1: UMAP trunks (parallel) ===")
        trunk_jobs = []
        for rep in REPRESENTATIONS:
            eout = _embed_dir(rep)
            if (eout / "embed_manifest.yaml").exists():
                print(f"  SKIP (exists): {eout.name}")
                continue
            cmd = _base_cmd(rep) + ["--timestep_embed_only", "--out_dir", str(eout)]
            trunk_jobs.append((rep, cmd, SWEEP_EMBED_ROOT / f"embed_{rep}.log"))
        _run_parallel(trunk_jobs, dry_run)

        print("=== Stage 2: kmeans branches (parallel) ===")
        branch_jobs = []
        for rep in REPRESENTATIONS:
            trunk = _embed_dir(rep)
            if not (trunk / "embed_manifest.yaml").exists():
                print(f"  SKIP branches (trunk missing): {rep}")
                continue
            for (w, s) in WINDOW_PARAMS:
                for K in KS:
                    cout = _clust_dir(rep, w, s, K)
                    if (cout / "manifest.yaml").exists():
                        print(f"  SKIP (exists): {cout.name}")
                        continue
                    cmd = (
                        _base_cmd(rep) +
                        ["--window_width", str(w),
                         "--stride", str(s),
                         "--aggregation", AGGREGATION,
                         "--n_clusters", str(K),
                         "--timestep_embed_dir", str(trunk),
                         "--out_dir", str(cout)]
                    )
                    log = SWEEP_CLUST_ROOT / f"build_{cout.name}.log"
                    branch_jobs.append((cout.name, cmd, log))
        _run_parallel(branch_jobs, dry_run)

# ---------------------------------------------------------------------------
# Phase 2 — eval
# ---------------------------------------------------------------------------

def phase_eval(dry_run: bool = False) -> None:
    SWEEP_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    runner = str(_REPO_ROOT / "scripts" / "run_e1_transport_r512_qwen.py")
    total = len(REPRESENTATIONS) * len(WINDOW_PARAMS) * len(KS) * len(SEEDS)
    done = 0

    for rep in REPRESENTATIONS:
        for (w, s) in WINDOW_PARAMS:
            for K in KS:
                clust = _clust_dir(rep, w, s, K)
                if not (clust / "manifest.yaml").exists():
                    print(f"SKIP eval (no clustering): {clust.name}")
                    continue
                for seed in SEEDS:
                    out = _eval_out(rep, w, s, K, seed)
                    if (out / "metrics.json").exists():
                        done += 1
                        print(f"SKIP ({done}/{total}): {out.name}")
                        continue
                    cmd = [
                        "conda", "run", "-n", "policy_doctor", "--no-capture-output",
                        sys.executable, runner,
                        "--clustering_dir", str(clust),
                        "--max_clusters", str(K),
                        "--n_example", str(N_EXAMPLE),
                        "--n_query", str(N_QUERY),
                        "--n_repetitions", str(N_REPS),
                        "--global_episode_disjoint",
                        "--composite_target_size", str(COMPOSITE_SIZE),
                        "--image_max_pixels", str(IMAGE_MAX_PIXELS),
                        "--max_frames_per_storyboard", "4",
                        "--view_window_extension", "0",
                        "--storyboard_mode", "composite",
                        "--random_seed", str(seed),
                        "--out_dir", str(out),
                    ]
                    log = SWEEP_EVAL_ROOT / f"eval_{out.name}.log"
                    t0 = time.time()
                    done += 1
                    print(f"[{done}/{total}] {out.name} ...", flush=True)
                    if dry_run:
                        print("  DRY:", " ".join(cmd))
                        continue
                    log.parent.mkdir(parents=True, exist_ok=True)
                    rc = subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode
                    elapsed = (time.time() - t0) / 60
                    print(f"  {'OK' if rc == 0 else 'FAILED'} {elapsed:.1f}min", flush=True)

# ---------------------------------------------------------------------------
# Phase 3 — report
# ---------------------------------------------------------------------------

def _load_result(exp_dir: pathlib.Path, K: int) -> Optional[Dict[str, Any]]:
    try:
        m  = json.load(open(exp_dir / "metrics.json"))
        sp = json.load(open(exp_dir / "sample_plan.json"))
        preds = [json.loads(l) for l in open(exp_dir / "predictions.jsonl")]
    except FileNotFoundError:
        return None
    origin_map: Dict[int, str] = {}
    for cid, cdata in sp["clusters"].items():
        for qi, qidx in enumerate(cdata["query_indices"]):
            origin_map[qidx] = cdata["query_origins"][qi]
    clean = [p for p in preds if origin_map.get(p["query_idx"]) == "tier1_global"]
    correct = sum(1 for p in clean if p["is_correct"])
    n = len(clean)
    return {
        "headline": m["top1_accuracy"],
        "clean_acc": correct / n if n else float("nan"),
        "clean_n": n,
        "chance": 1.0 / K,
    }

def phase_report() -> None:
    rows = []
    for rep in REPRESENTATIONS:
        for (w, s) in WINDOW_PARAMS:
            for K in KS:
                seed_results = []
                for seed in SEEDS:
                    r = _load_result(_eval_out(rep, w, s, K, seed), K)
                    if r is not None:
                        seed_results.append(r)
                if not seed_results:
                    continue
                mean_clean = float(np.mean([r["clean_acc"] for r in seed_results]))
                std_clean  = float(np.std( [r["clean_acc"] for r in seed_results]))
                ratio = mean_clean / (1.0 / K)
                rows.append({"rep": rep, "w": w, "s": s, "K": K,
                             "mean_clean": mean_clean, "std_clean": std_clean,
                             "ratio": ratio, "n_seeds": len(seed_results)})

    lines = [
        "# E1 Window-Parameter Sweep Results",
        "",
        "UMAP fitted once per representation on per-timestep features.  "
        "Window config and K are applied post-UMAP.",
        "",
        f"Representations: {', '.join(REPRESENTATIONS)}  ",
        f"Window (w, s): {WINDOW_PARAMS}  ",
        f"K: {KS}  ",
        f"Seeds: {SEEDS}  ",
        f"Protocol: n_example={N_EXAMPLE}, n_query={N_QUERY}, n_reps={N_REPS}, "
        f"composite={COMPOSITE_SIZE}², global_episode_disjoint",
        "",
        "Clean accuracy = tier1_global queries. Values are mean ± std across seeds.",
        "",
    ]
    for K in KS:
        lines.append(f"## K = {K}  (chance = {1/K:.3f})")
        lines.append("")
        lines.append("| Rep | (w, s) | Mean clean | ±std | Ratio above chance | n_seeds |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted([x for x in rows if x["K"] == K], key=lambda x: -x["mean_clean"]):
            lines.append(
                f"| {r['rep']} | ({r['w']}, {r['s']}) "
                f"| {r['mean_clean']:.3f} | ±{r['std_clean']:.3f} "
                f"| {r['ratio']:.1f}× | {r['n_seeds']}/{len(SEEDS)} |"
            )
        lines.append("")

    lines += [
        "## Best window config per (rep, K)",
        "",
        "| Rep | K | Best (w,s) | Mean clean | Ratio |",
        "|---|---|---|---|---|",
    ]
    for rep in REPRESENTATIONS:
        for K in KS:
            subset = [r for r in rows if r["rep"] == rep and r["K"] == K]
            if not subset:
                continue
            best = max(subset, key=lambda r: r["mean_clean"])
            lines.append(
                f"| {rep} | {K} | ({best['w']}, {best['s']}) "
                f"| {best['mean_clean']:.3f} | {best['ratio']:.1f}× |"
            )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report → {REPORT_PATH}")
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "eval", "report", "all"], required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.phase in ("build", "all"):
        print("=== Phase 1: build ===")
        phase_build(dry_run=args.dry_run)
    if args.phase in ("eval", "all"):
        print("=== Phase 2: eval ===")
        phase_eval(dry_run=args.dry_run)
    if args.phase in ("report", "all"):
        print("=== Phase 3: report ===")
        phase_report()

if __name__ == "__main__":
    main()

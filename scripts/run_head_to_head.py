"""Head-to-head comparison at K=10, n_query=5, n_reps=5.

Seven representations, all using the same E1 protocol so results are
directly comparable:

  encoder_plan_t0      policy_emb (best from sweep)
  bottleneck_plan_t0   policy_emb (high variance across earlier runs)
  bottleneck_plan_t5   policy_emb (surprisingly strong in sweep)
  infembed_100d        InfEmbed 100D UMAP (primary baseline)
  state_full           state full_history (118D → UMAP 100D)
  state_action_full    state full_history + full_plan (438D → UMAP 100D)
  trak                 TRAK SVD(200D) → UMAP 50D (negative result control)

All existing clusterings are reused; only state_full and state_action_full
need to be built.  Results in experiments/head_to_head/.

Usage:
    conda run -n policy_doctor python scripts/run_head_to_head.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

EVAL_DIR = (
    "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27"
    "/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest"
)
RESULTS_ROOT = _REPO_ROOT / "experiments" / "head_to_head"
CLUST_ROOT = RESULTS_ROOT / "clusterings"
LOG_PATH = pathlib.Path("/tmp/head_to_head.log")

N_EXAMPLE = 3
N_QUERY   = 5
N_REPS    = 5
K         = 10
COMPOSITE = 768

# ── Clustering sources ────────────────────────────────────────────────────────
# Existing clustering dirs (absolute paths)
SWEEP_CLUST = _REPO_ROOT / "experiments" / "policy_emb_sweep" / "clusterings"

EXISTING_CLUSTERINGS = {
    "encoder_plan_t0":   SWEEP_CLUST / "encoder_plan_t0__w5_s2__K10",
    "bottleneck_plan_t0": SWEEP_CLUST / "bottleneck_plan_t0__w5_s2__K10",
    "bottleneck_plan_t5": SWEEP_CLUST / "bottleneck_plan_t5__w5_s2__K10",
    "infembed_100d":     pathlib.Path("/tmp/transport_mh_seed0_r512_clustering_k10"),
    "trak":              SWEEP_CLUST / "trak__w5_s2__K10",
}

# New clusterings to build
NEW_CLUSTERINGS = {
    "state_full": {
        "representation": "state",
        "obs_strategy": "full_history",
        "umap_n_components": "100",
    },
    "state_action_full": {
        "representation": "state_action",
        "obs_strategy": "full_history",
        "action_strategy": "full_plan",
        "umap_n_components": "100",
    },
}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _run(cmd: list[str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    return result.returncode


def build_new_clustering(label: str, kwargs: dict) -> pathlib.Path | None:
    clust_dir = CLUST_ROOT / f"{label}__w5_s2__K{K}"
    if (clust_dir / "manifest.yaml").exists():
        _log(f"  SKIP clustering (exists): {clust_dir.name}")
        return clust_dir
    _log(f"  build clustering: {clust_dir.name}")
    cmd = [
        "conda", "run", "-n", "policy_doctor", "--no-capture-output",
        "python", str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
        "--representation", kwargs["representation"],
        "--eval_dir", EVAL_DIR,
        "--window_width", "5", "--stride", "2", "--aggregation", "mean",
        "--prescale", "standard",
        "--umap_n_components", kwargs.get("umap_n_components", "100"),
        "--n_clusters", str(K), "--seed", "42",
        "--out_dir", str(clust_dir),
    ]
    if "obs_strategy" in kwargs:
        cmd += ["--obs_strategy", kwargs["obs_strategy"]]
    if "action_strategy" in kwargs:
        cmd += ["--action_strategy", kwargs["action_strategy"]]
    rc = _run(cmd, LOG_PATH.parent / f"build_{clust_dir.name}.log")
    if rc != 0:
        _log(f"  FAILED clustering: {clust_dir.name}")
        return None
    return clust_dir


def run_e1(label: str, clust_dir: pathlib.Path) -> bool:
    out_dir = RESULTS_ROOT / label
    if (out_dir / "metrics.json").exists():
        _log(f"  SKIP E1 (exists): {label}")
        return True
    _log(f"  E1: {label}")
    rc = _run(
        [
            "conda", "run", "-n", "policy_doctor", "--no-capture-output",
            "python", str(_REPO_ROOT / "scripts" / "run_e1_transport_r512_qwen.py"),
            "--clustering_dir", str(clust_dir),
            "--max_clusters", str(K),
            "--n_example", str(N_EXAMPLE),
            "--n_query", str(N_QUERY),
            "--n_repetitions", str(N_REPS),
            "--global_episode_disjoint",
            "--composite_target_size", str(COMPOSITE),
            "--max_frames_per_storyboard", "4",
            "--storyboard_mode", "composite",
            "--random_seed", "42",
            "--out_dir", str(out_dir),
        ],
        LOG_PATH.parent / f"e1_h2h_{label}.log",
    )
    if rc != 0:
        _log(f"  FAILED E1: {label}")
    return rc == 0


def collect_results() -> None:
    from scipy.stats import binomtest

    def get_clean(exp_dir):
        try:
            sp    = json.load(open(exp_dir / "sample_plan.json"))
            preds = [json.loads(l) for l in open(exp_dir / "predictions.jsonl")]
        except FileNotFoundError:
            return None
        om = {}
        for cid, cdata in sp["clusters"].items():
            for qi, qidx in enumerate(cdata["query_indices"]):
                om[qidx] = cdata["query_origins"][qi]
        clean = [p for p in preds if om.get(p["query_idx"]) == "tier1_global"]
        c = sum(1 for p in clean if p["is_correct"])
        n = len(clean)
        if n == 0:
            return None
        p = binomtest(c, n, 1 / K, alternative="greater").pvalue
        return c / n, c, n, p

    _log("\n=== HEAD-TO-HEAD RESULTS (K=10, n_query=5, n_reps=5) ===")
    _log(f"{'Label':<25} {'Clean':>10}  {'Ratio':>6}  p")
    _log("-" * 60)
    order = list(EXISTING_CLUSTERINGS) + list(NEW_CLUSTERINGS)
    for label in order:
        exp_dir = RESULTS_ROOT / label
        if not (exp_dir / "metrics.json").exists():
            _log(f"{label:<25}  (missing)")
            continue
        r = get_clean(exp_dir)
        if r:
            acc, c, n, p = r
            sig = "" if p < 0.05 else " NS"
            _log(f"{label:<25} {c}/{n}={acc:.3f}  {acc/(1/K):>5.1f}×  {p:.2e}{sig}")


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CLUST_ROOT.mkdir(exist_ok=True)
    _log("=== Head-to-head sweep starting (K=10, n_query=5, n_reps=5) ===")

    # ── Verify existing clusterings ───────────────────────────────────────────
    _log("\n--- Verifying existing clustering dirs ---")
    all_clusterings: dict[str, pathlib.Path] = {}
    for label, clust_dir in EXISTING_CLUSTERINGS.items():
        if not (clust_dir / "manifest.yaml").exists():
            _log(f"  MISSING: {label} → {clust_dir}")
        else:
            _log(f"  OK: {label}")
            all_clusterings[label] = clust_dir

    # ── Build new clusterings ─────────────────────────────────────────────────
    _log("\n--- Building new clusterings ---")
    for label, kwargs in NEW_CLUSTERINGS.items():
        clust = build_new_clustering(label, kwargs)
        if clust is not None:
            all_clusterings[label] = clust

    # ── Run E1 for all ────────────────────────────────────────────────────────
    _log("\n--- Running E1 evaluations ---")
    for label, clust_dir in all_clusterings.items():
        _log(f"\n=== {label} ===")
        run_e1(label, clust_dir)

    collect_results()
    _log("\n=== HEAD-TO-HEAD DONE ===")


if __name__ == "__main__":
    main()

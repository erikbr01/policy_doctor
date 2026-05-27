"""K sweep for encoder_plan_t0 (K=5, 15, 20 — K=10 already done).

Embedding already on disk; only clustering + E1 needed.
Results land in experiments/policy_emb_sweep/ alongside the main sweep.

Usage:
    conda run -n policy_doctor python scripts/run_encoder_k_sweep.py
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
SWEEP_ROOT = _REPO_ROOT / "experiments" / "policy_emb_sweep"
LOG_PATH = pathlib.Path("/tmp/encoder_k_sweep.log")

N_EXAMPLE, N_QUERY, N_REPS = 3, 3, 3

K_SWEEP = [5, 15, 20]  # K=10 already done
LAYER = "encoder_plan_t0"


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


def build_clustering(K: int) -> pathlib.Path | None:
    clust_dir = SWEEP_ROOT / "clusterings" / f"{LAYER}__w5_s2__K{K}"
    if (clust_dir / "manifest.yaml").exists():
        _log(f"  SKIP clustering (exists): {clust_dir.name}")
        return clust_dir
    _log(f"  build clustering: {clust_dir.name}")
    rc = _run(
        ["conda", "run", "-n", "policy_doctor", "--no-capture-output",
         "python", str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
         "--representation", "policy_emb", "--eval_dir", EVAL_DIR,
         "--window_width", "5", "--stride", "2", "--aggregation", "mean",
         "--prescale", "standard", "--umap_n_components", "50",
         "--n_clusters", str(K), "--seed", "42", "--layer", LAYER,
         "--out_dir", str(clust_dir)],
        LOG_PATH.parent / f"build_{clust_dir.name}.log",
    )
    if rc != 0:
        _log(f"  FAILED clustering: {clust_dir.name}")
        return None
    return clust_dir


def run_e1(clust_dir: pathlib.Path, K: int) -> bool:
    label = f"{LAYER}__w5_s2__K{K}"
    out_dir = SWEEP_ROOT / label
    if (out_dir / "metrics.json").exists():
        _log(f"  SKIP E1 (exists): {label}")
        return True
    _log(f"  E1: {label}")
    comp = 768 if K <= 10 else 512
    rc = _run(
        ["conda", "run", "-n", "policy_doctor", "--no-capture-output",
         "python", str(_REPO_ROOT / "scripts" / "run_e1_transport_r512_qwen.py"),
         "--clustering_dir", str(clust_dir), "--max_clusters", str(K),
         "--n_example", str(N_EXAMPLE), "--n_query", str(N_QUERY),
         "--n_repetitions", str(N_REPS), "--global_episode_disjoint",
         "--composite_target_size", str(comp),
         "--max_frames_per_storyboard", "4", "--storyboard_mode", "composite",
         "--random_seed", "42", "--out_dir", str(out_dir)],
        LOG_PATH.parent / f"e1_{label}.log",
    )
    if rc != 0:
        _log(f"  FAILED E1: {label}")
    return rc == 0


def collect_results() -> None:
    from scipy.stats import binomtest

    def get_clean(exp_dir, K):
        try:
            sp = json.load(open(exp_dir / "sample_plan.json"))
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
        return c / n, c, n, binomtest(c, n, 1 / K, alternative="greater").pvalue

    _log("\n=== encoder_plan_t0 K sweep results ===")
    _log(f"{'Label':<40} {'Clean':>8}  {'Ratio':>6}  p")
    _log("-" * 65)
    for K in [5, 10, 15, 20]:
        exp_dir = SWEEP_ROOT / f"{LAYER}__w5_s2__K{K}"
        if not (exp_dir / "metrics.json").exists():
            _log(f"{exp_dir.name:<40}  (missing)")
            continue
        r = get_clean(exp_dir, K)
        if r:
            acc, c, n, p = r
            _log(f"{exp_dir.name:<40} {c}/{n}={acc:.3f}  {acc / (1/K):>5.1f}x  {p:.2e}")


def main() -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    (SWEEP_ROOT / "clusterings").mkdir(exist_ok=True)
    _log("=== encoder_plan_t0 K sweep ===")
    for K in K_SWEEP:
        _log(f"\n--- K={K} ---")
        clust = build_clustering(K)
        if clust is None:
            continue
        run_e1(clust, K)
    collect_results()
    _log("=== DONE ===")


if __name__ == "__main__":
    main()

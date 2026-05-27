"""Overnight sweep for policy embedding representations.

Covers:
  - Hook layer:    bottleneck, decoder, encoder
  - Action scope:  plan (full 16-step), exec (action[0]×16), plan8 (first 8 + zero-pad)
  - Timestep:      t=0, t=5, t=10, t=25
  - K:             5, 10, 15, 20  (using best layer = bottleneck_plan_t0)
  - Window:        (w=1,s=1), (w=2,s=2), (w=3,s=2), (w=5,s=2)  at K=10

All E1 evals: n_example=3, n_query=3, n_reps=3, global_episode_disjoint,
              composite=768², seed=42. Results in experiments/policy_emb_sweep/.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n policy_doctor python scripts/run_policy_emb_sweep.py
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

TRAIN_DIR = (
    "/mnt/ssdB/erik/cupid_data/outputs/train/mar27"
    "/mar27_train_diffusion_unet_lowdim_transport_mh_0"
)
EVAL_DIR = (
    "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27"
    "/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest"
)
RESULTS_ROOT = _REPO_ROOT / "experiments" / "policy_emb_sweep"
LOG_PATH = pathlib.Path("/tmp/policy_emb_sweep.log")

# E1 eval constants
N_EXAMPLE, N_QUERY, N_REPS = 3, 3, 3
COMPOSITE_SIZE = 768

# ── Experiment grid ──────────────────────────────────────────────────────────

# Tier 1a: hook layer sweep (all with plan action, t=0, K=10, w=5,s=2)
LAYER_SWEEP = [
    "bottleneck_plan_t0",   # baseline (already computed)
    "decoder_plan_t0",
    "encoder_plan_t0",
]

# Tier 1b: action scope sweep (bottleneck hook, t=0, K=10, w=5,s=2)
ACTION_SWEEP = [
    "bottleneck_plan_t0",   # full plan  (already computed)
    "bottleneck_exec_t0",   # action[0] × horizon
    "bottleneck_plan8_t0",  # first 8 steps + zero
]

# Tier 1c: timestep sweep (bottleneck hook, plan action, K=10, w=5,s=2)
TIMESTEP_SWEEP = [
    "bottleneck_plan_t0",   # t=0  (already computed)
    "bottleneck_plan_t5",
    "bottleneck_plan_t10",
    "bottleneck_plan_t25",
]

# Tier 2a: K sweep (bottleneck_plan_t0, w=5,s=2)
K_SWEEP = [5, 10, 15, 20]  # K=10 already done

# Tier 2b: window sweep at K=10 (bottleneck_plan_t0)
WINDOW_SWEEP = [(1, 1), (2, 2), (3, 2), (5, 2)]  # (5,2) already done


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


def _embedding_exists(layer: str) -> bool:
    return (pathlib.Path(EVAL_DIR) / "policy_embeddings" / f"{layer}.npz").exists()


def _clustering_exists(layer: str, w: int, s: int, K: int) -> bool:
    return (_REPO_ROOT / "experiments" / "policy_emb_sweep"
            / f"clusterings" / f"{layer}__w{w}_s{s}__K{K}" / "manifest.yaml").exists()


def _eval_exists(label: str) -> bool:
    return (RESULTS_ROOT / label / "metrics.json").exists()


def compute_embedding(layer: str, device: str = "cuda:0") -> bool:
    if _embedding_exists(layer):
        _log(f"  SKIP compute (exists): {layer}")
        return True
    _log(f"  compute: {layer}")
    rc = _run(
        ["conda", "run", "-n", "cupid_torch2", "--no-capture-output",
         "python", str(_REPO_ROOT / "third_party" / "cupid" / "compute_policy_embeddings.py"),
         "--train_dir", TRAIN_DIR, "--eval_dir", EVAL_DIR,
         "--layer", layer, "--batch_size", "128", "--device", device],
        LOG_PATH.parent / f"compute_{layer}.log",
    )
    if rc != 0:
        _log(f"  FAILED compute: {layer}")
    return rc == 0


def build_clustering(layer: str, w: int, s: int, K: int) -> pathlib.Path | None:
    clust_dir = RESULTS_ROOT / "clusterings" / f"{layer}__w{w}_s{s}__K{K}"
    if (clust_dir / "manifest.yaml").exists():
        _log(f"  SKIP build (exists): {clust_dir.name}")
        return clust_dir
    _log(f"  build: {clust_dir.name}")
    rc = _run(
        ["conda", "run", "-n", "policy_doctor", "--no-capture-output",
         "python", str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
         "--representation", "policy_emb", "--eval_dir", EVAL_DIR,
         "--window_width", str(w), "--stride", str(s), "--aggregation", "mean",
         "--prescale", "standard", "--umap_n_components", "50",
         "--n_clusters", str(K), "--seed", "42", "--layer", layer,
         "--out_dir", str(clust_dir)],
        LOG_PATH.parent / f"build_{clust_dir.name}.log",
    )
    if rc != 0:
        _log(f"  FAILED build: {clust_dir.name}")
        return None
    return clust_dir


def run_e1(clust_dir: pathlib.Path, label: str, K: int) -> bool:
    out_dir = RESULTS_ROOT / label
    if (out_dir / "metrics.json").exists():
        _log(f"  SKIP E1 (exists): {label}")
        return True
    _log(f"  E1: {label}")
    # 768² OOMs at K≥15 (known constraint); drop to 512² for those.
    comp = COMPOSITE_SIZE if K <= 10 else 512
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


def run_experiment(layer: str, w: int, s: int, K: int) -> None:
    label = f"{layer}__w{w}_s{s}__K{K}"
    _log(f"=== {label} ===")
    if not compute_embedding(layer):
        return
    clust = build_clustering(layer, w, s, K)
    if clust is None:
        return
    run_e1(clust, label, K)


def collect_results() -> None:
    """Print a summary table of all completed E1 results."""
    from scipy.stats import binomtest

    def get_clean(exp_dir, K):
        try:
            m  = json.load(open(exp_dir / "metrics.json"))
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
        return c / n if n else float("nan"), c, n, binomtest(c, n, 1/K, alternative="greater").pvalue

    _log("\n=== RESULTS SUMMARY ===")
    _log(f"{'Label':<45} {'Clean':>8}  {'Ratio':>6}  p")
    _log("-" * 70)
    for exp_dir in sorted(RESULTS_ROOT.iterdir()):
        if not (exp_dir / "metrics.json").exists():
            continue
        # parse K from label
        parts = exp_dir.name.split("__K")
        if len(parts) < 2:
            continue
        try:
            K = int(parts[-1])
        except ValueError:
            continue
        r = get_clean(exp_dir, K)
        if r:
            acc, c, n, p = r
            _log(f"{exp_dir.name:<45} {c}/{n}={acc:.3f}  {acc/(1/K):>5.1f}x  {p:.2e}")


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "clusterings").mkdir(exist_ok=True)
    _log("=== Policy embedding sweep starting ===")

    # Collect unique layers that need computing
    all_layers = set(LAYER_SWEEP + ACTION_SWEEP + TIMESTEP_SWEEP)

    # ── Tier 1a: layer sweep at K=10, w=5,s=2 ──────────────────────────────
    _log("\n--- Tier 1a: hook layer sweep ---")
    for layer in LAYER_SWEEP:
        run_experiment(layer, w=5, s=2, K=10)

    # ── Tier 1b: action scope sweep ─────────────────────────────────────────
    _log("\n--- Tier 1b: action scope sweep ---")
    for layer in ACTION_SWEEP:
        run_experiment(layer, w=5, s=2, K=10)

    # ── Tier 1c: timestep sweep ──────────────────────────────────────────────
    _log("\n--- Tier 1c: timestep sweep ---")
    for layer in TIMESTEP_SWEEP:
        run_experiment(layer, w=5, s=2, K=10)

    # ── Tier 2a: K sweep (bottleneck_plan_t0) ───────────────────────────────
    _log("\n--- Tier 2a: K sweep (bottleneck_plan_t0) ---")
    for K in K_SWEEP:
        run_experiment("bottleneck_plan_t0", w=5, s=2, K=K)

    # ── Tier 2b: window sweep at K=10 (bottleneck_plan_t0) ──────────────────
    _log("\n--- Tier 2b: window sweep (bottleneck_plan_t0, K=10) ---")
    for w, s in WINDOW_SWEEP:
        run_experiment("bottleneck_plan_t0", w=w, s=s, K=10)

    # ── Spare time: TRAK clustering at K=5,10,15,20 (w=5,s=2) ──────────────
    _log("\n--- Spare time: TRAK representation (w=5,s=2, full→SVD200→UMAP50) ---")
    for K in K_SWEEP:
        label = f"trak__w5_s2__K{K}"
        _log(f"=== {label} ===")
        clust_dir = RESULTS_ROOT / "clusterings" / f"trak__w5_s2__K{K}"
        if not (clust_dir / "manifest.yaml").exists():
            _log(f"  build: {clust_dir.name}")
            rc = _run(
                ["conda", "run", "-n", "policy_doctor", "--no-capture-output",
                 "python", str(_REPO_ROOT / "scripts" / "build_alt_clustering.py"),
                 "--representation", "trak", "--eval_dir", EVAL_DIR,
                 "--window_width", "5", "--stride", "2", "--aggregation", "mean",
                 "--prescale", "standard", "--umap_n_components", "50",
                 "--n_clusters", str(K), "--seed", "42",
                 "--n_svd_components", "200", "--svd_seed", "42",
                 "--out_dir", str(clust_dir)],
                LOG_PATH.parent / f"build_{clust_dir.name}.log",
            )
            if rc != 0:
                _log(f"  FAILED build: {clust_dir.name}")
                continue
        else:
            _log(f"  SKIP build (exists): {clust_dir.name}")
        run_e1(clust_dir, label, K)

    collect_results()
    _log("=== Sweep complete ===")


if __name__ == "__main__":
    main()

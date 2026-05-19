#!/usr/bin/env python3
"""
Aggregate MimicGen sweep results from pipeline run directories.

Reads eval_mimicgen_combined/result.json and generate_mimicgen_demos/result.json
for each arm and prints summary tables covering:
  - Policy eval: best / top5_mean success rate
  - MimicGen generation: gen efficiency, mean demo length, throughput

Eval rollout duration (policy steps to success) is NOT available in the current
outputs because eval_save_episodes runs with save_episodes=False; episode lengths
are only written to metadata.yaml when save_episodes=True.

Usage:
    python scripts/aggregate_sweep_results.py
    python scripts/aggregate_sweep_results.py --run-dir /path/to/run_dir
    python scripts/aggregate_sweep_results.py --per-arm
    python scripts/aggregate_sweep_results.py --gen-stats
    # K-robustness sweep (run dirs must end in _k{N}):
    python scripts/aggregate_sweep_results.py --k-sweep \
      --run-dir .../tight_k5 --run-dir .../tight_k10 --run-dir .../tight_k15 \
      --run-dir .../tight_k20 --run-dir .../tight_k25
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


SSDБ_PIPELINE_ROOT = Path("/mnt/ssdB/erik/cupid_data/pipeline_runs")

DEFAULT_RUN_DIRS = [
    SSDБ_PIPELINE_ROOT / "mimicgen_square_apr26_seed1_d60_nut_constrained",
    SSDБ_PIPELINE_ROOT / "mimicgen_square_apr26_seed1_d60_budget300_nut_constrained",
    SSDБ_PIPELINE_ROOT / "mimicgen_square_apr26_seed1_d300_nut_constrained",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_arm_path(arm_dir: Path) -> dict | None:
    """Extract heuristic, budget, rep from an arm directory name."""
    m = re.match(
        r"mimicgen_(?P<heuristic>[a-z_]+)_budget(?P<budget>\d+)(?:_rep(?P<rep>\d+))?$",
        arm_dir.name,
    )
    if not m:
        return None
    rep_str = m.group("rep")
    return {
        "heuristic": m.group("heuristic"),
        "budget": int(m.group("budget")),
        # Phase A has no rep suffix → rep=0; Phase B _rep1/_rep2 → 1/2
        "rep": int(rep_str) if rep_str is not None else 0,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_eval_result(arm_dir: Path, topk: int = 5) -> dict | None:
    """
    Load eval_mimicgen_combined/result.json.

    Falls back to eval_mimicgen_combined.backup/result.json when the primary
    dir has been renamed during a re-run (e.g. rerun_evals_with_episode_lengths.py).

    Also computes topk_mean = mean of the top-`topk` checkpoints by success_rate,
    for a fair comparison when Phase A arms happen to have more checkpoints evaluated.
    """
    for step_dir_name in ("eval_mimicgen_combined", "eval_mimicgen_combined.backup"):
        path = arm_dir / step_dir_name / "result.json"
        if path.exists():
            break
    else:
        return None
    with open(path) as f:
        data = json.load(f)
    if "best_success_rate" not in data or "mean_success_rate" not in data:
        return None

    checkpoints = data.get("checkpoints", [])
    if checkpoints:
        top_scores = sorted([c["success_rate"] for c in checkpoints], reverse=True)[:topk]
        data["topk_mean_success_rate"] = float(np.mean(top_scores))

        # Episode lengths: average over checkpoints that have the field (post-patch evals only)
        ep_lens = [c["mean_episode_length"] for c in checkpoints if "mean_episode_length" in c]
        succ_ep_lens = [c["mean_success_episode_length"] for c in checkpoints
                        if "mean_success_episode_length" in c]
        data["mean_episode_length"] = float(np.mean(ep_lens)) if ep_lens else None
        data["mean_success_episode_length"] = float(np.mean(succ_ep_lens)) if succ_ep_lens else None
    else:
        data["topk_mean_success_rate"] = data["mean_success_rate"]
        data.setdefault("mean_episode_length", None)
        data.setdefault("mean_success_episode_length", None)

    return data


def load_gen_stats(arm_dir: Path) -> dict | None:
    """
    Load generate_mimicgen_demos/result.json and compute generation metrics.

    Returns:
      gen_efficiency   — num_success / num_attempts
      gen_ep_len_mean  — weighted mean episode length of generated demos (steps)
      gen_ep_len_std   — weighted mean of per-seed ep_length_std
      gen_time_hrs     — total wall-clock generation time (sum across all seeds)
      gen_throughput   — demos per hour (num_success / gen_time_hrs)
      num_generated    — num_success (demos actually written)
      num_attempts     — total datagen trials attempted
    """
    path = arm_dir / "generate_mimicgen_demos" / "result.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    stats = data.get("stats")
    if not stats:
        return None

    num_success = stats["num_success"]
    num_attempts = stats["num_attempts"]

    per_seed = stats.get("per_seed_stats", [])
    valid_seeds = [s for s in per_seed if s.get("num_success", 0) > 0]

    # Weighted mean episode length across seeds that produced at least one demo
    if valid_seeds:
        weights = np.array([s["num_success"] for s in valid_seeds], dtype=float)
        ep_len_means = np.array([s["ep_length_mean"] for s in valid_seeds])
        ep_len_stds = np.array([s["ep_length_std"] for s in valid_seeds])
        wtd_ep_len_mean = float(np.average(ep_len_means, weights=weights))
        wtd_ep_len_std = float(np.average(ep_len_stds, weights=weights))
    else:
        # Fall back to reading the HDF5 directly
        hdf5_path = data.get("generated_hdf5_path")
        if hdf5_path and Path(hdf5_path).exists():
            with h5py.File(hdf5_path, "r") as f:
                lengths = [f["data"][k]["actions"].shape[0] for k in f["data"].keys()]
            wtd_ep_len_mean = float(np.mean(lengths)) if lengths else float("nan")
            wtd_ep_len_std = float(np.std(lengths)) if lengths else float("nan")
        else:
            wtd_ep_len_mean = float("nan")
            wtd_ep_len_std = float("nan")

    # Some seeds may be missing "time spent (hrs)" if the run was interrupted
    total_time_hrs = sum(
        float(s["time spent (hrs)"])
        for s in per_seed
        if "time spent (hrs)" in s
    )

    return {
        "gen_efficiency": num_success / num_attempts if num_attempts > 0 else float("nan"),
        "gen_ep_len_mean": wtd_ep_len_mean,
        "gen_ep_len_std": wtd_ep_len_std,
        "gen_time_hrs": total_time_hrs,
        "gen_throughput": num_success / total_time_hrs if total_time_hrs > 0 else float("nan"),
        "num_generated": num_success,
        "num_attempts": num_attempts,
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_results(run_dir: Path) -> list[dict]:
    """Walk a run dir and collect per-arm eval + generation results."""
    results = []

    for phase_dir in sorted(run_dir.iterdir()):
        if not phase_dir.is_dir():
            continue
        if phase_dir.name not in ("mimicgen_budget_sweep", "mimicgen_budget_rep_sweep"):
            continue

        for arm_dir in sorted(phase_dir.iterdir()):
            if not arm_dir.is_dir():
                continue

            parsed = parse_arm_path(arm_dir)
            if parsed is None:
                continue

            eval_result = load_eval_result(arm_dir)
            if eval_result is None:
                continue

            gen_stats = load_gen_stats(arm_dir) or {}

            n_ckpts = len(eval_result.get("checkpoints", []))
            results.append(
                {
                    "run_dir": run_dir.name,
                    **parsed,
                    # Eval metrics
                    "best": eval_result["best_success_rate"],
                    "mean": eval_result["mean_success_rate"],
                    "topk_mean": eval_result["topk_mean_success_rate"],
                    "n_checkpoints": n_ckpts,
                    "ep_len": eval_result.get("mean_episode_length"),
                    "succ_ep_len": eval_result.get("mean_success_episode_length"),
                    # Generation metrics
                    **{f"gen_{k}": v for k, v in gen_stats.items()},
                }
            )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def agg(values: list[float]) -> dict:
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return {}
    arr = np.array(vals)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def fmt_agg(a: dict, decimals: int = 3) -> str:
    if not a:
        return "—"
    fmt = f"{{:.{decimals}f}}"
    mean_s = fmt.format(a["mean"])
    std_s = fmt.format(a["std"])
    min_s = fmt.format(a["min"])
    max_s = fmt.format(a["max"])
    return f"{mean_s} ± {std_s}  {min_s}  {max_s}  {a['n']}"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_eval_table(run_dir_name: str, all_results: list[dict]) -> None:
    run_results = [r for r in all_results if r["run_dir"] == run_dir_name]
    if not run_results:
        return

    print(f"\n{'='*72}")
    print(f"Run dir: {run_dir_name}")
    print(f"{'='*72}")

    for budget in sorted({r["budget"] for r in run_results}):
        bud = [r for r in run_results if r["budget"] == budget]
        print(f"\n  Budget = {budget}  [eval success rate, n=3]")
        print(f"  {'heuristic':<18} {'metric':<18}  {'mean±std':>13}  {'min':>6}  {'max':>6}  n")
        print(f"  {'-'*18} {'-'*18}  {'-'*13}  {'-'*6}  {'-'*6}  -")

        for heuristic in sorted({r["heuristic"] for r in bud}):
            rows = [r for r in bud if r["heuristic"] == heuristic]
            print(f"  {heuristic:<18} {'best':<18}  {fmt_agg(agg([r['best'] for r in rows]))}")
            print(f"  {'':<18} {'top5_mean':<18}  {fmt_agg(agg([r['topk_mean'] for r in rows]))}")
            succ_lens = [r["succ_ep_len"] for r in rows if r.get("succ_ep_len") is not None]
            if succ_lens:
                print(f"  {'':<18} {'succ_ep_len (steps)':<18}  {fmt_agg(agg(succ_lens), decimals=1)}")


def print_gen_table(run_dir_name: str, all_results: list[dict]) -> None:
    run_results = [r for r in all_results if r["run_dir"] == run_dir_name]
    if not run_results:
        return

    print(f"\n  Generation stats ({run_dir_name}):")
    print(f"  {'heuristic':<18} {'budget':>6} {'rep':>4}  "
          f"{'gen%':>6}  {'ep_len':>6}  {'hr':>5}  {'demos/hr':>8}")
    print(f"  {'-'*18} {'-'*6} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}")

    for r in sorted(run_results, key=lambda r: (r["budget"], r["heuristic"], r["rep"])):
        rep_label = "A" if r["rep"] == 0 else str(r["rep"])
        eff = r.get("gen_gen_efficiency", float("nan"))
        ep_len = r.get("gen_gen_ep_len_mean", float("nan"))
        hrs = r.get("gen_gen_time_hrs", float("nan"))
        tput = r.get("gen_gen_throughput", float("nan"))
        print(
            f"  {r['heuristic']:<18} {r['budget']:>6} {rep_label:>4}  "
            f"{eff:>6.1%}  {ep_len:>6.1f}  {hrs:>5.2f}  {tput:>8.0f}"
        )

    print()
    for budget in sorted({r["budget"] for r in run_results}):
        bud = [r for r in run_results if r["budget"] == budget]
        print(f"  Budget = {budget}  [generation metrics, n=3]")
        print(f"  {'heuristic':<18} {'metric':<12}  {'mean±std':>14}  {'min':>6}  {'max':>6}  n")
        print(f"  {'-'*18} {'-'*12}  {'-'*14}  {'-'*6}  {'-'*6}  -")

        for heuristic in sorted({r["heuristic"] for r in bud}):
            rows = [r for r in bud if r["heuristic"] == heuristic]
            gen_eff = [r.get("gen_gen_efficiency", float("nan")) for r in rows]
            ep_lens = [r.get("gen_gen_ep_len_mean", float("nan")) for r in rows]
            throughputs = [r.get("gen_gen_throughput", float("nan")) for r in rows]

            print(f"  {heuristic:<18} {'gen_efficiency':<12}  {fmt_agg(agg(gen_eff), decimals=3)}")
            print(f"  {'':<18} {'ep_len (steps)':<12}  {fmt_agg(agg(ep_lens), decimals=1)}")
            print(f"  {'':<18} {'demos/hr':<12}  {fmt_agg(agg(throughputs), decimals=0)}")
        print()


def extract_k_from_run_dir(run_dir_name: str) -> int | None:
    """Extract K value from a run dir name ending in _k{N}."""
    m = re.search(r"_k(\d+)$", run_dir_name)
    return int(m.group(1)) if m else None


def print_k_sweep_table(all_results: list[dict]) -> None:
    """Print a K-vs-heuristic comparison table across K-sweep run dirs."""
    # Attach K to each result
    k_tagged = []
    for r in all_results:
        k = extract_k_from_run_dir(r["run_dir"])
        if k is not None:
            k_tagged.append({**r, "k": k})

    if not k_tagged:
        print("[k-sweep] No run dirs with _k{N} suffix found.")
        return

    k_values = sorted({r["k"] for r in k_tagged})
    budgets = sorted({r["budget"] for r in k_tagged})
    heuristics = sorted({r["heuristic"] for r in k_tagged})

    for budget in budgets:
        print(f"\n{'='*72}")
        print(f"K Robustness Sweep  [budget={budget}, top5_mean success rate, n=3 reps per K]")
        print(f"{'='*72}")

        k_header = "  ".join(f"K={k:<4}" for k in k_values)
        print(f"  {'heuristic':<18}  {k_header}")
        print(f"  {'-'*18}  {'  '.join('-'*6 for _ in k_values)}")

        for heuristic in heuristics:
            cells = []
            for k in k_values:
                rows = [r for r in k_tagged
                        if r["k"] == k and r["budget"] == budget and r["heuristic"] == heuristic]
                scores = [r["topk_mean"] for r in rows if not np.isnan(r["topk_mean"])]
                if scores:
                    mean = float(np.mean(scores))
                    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                    cells.append(f"{mean:.3f}±{std:.3f}")
                else:
                    cells.append("  —   ")
            print(f"  {heuristic:<18}  {'  '.join(f'{c:<10}' for c in cells)}")

        # Also show best (peak checkpoint) for completeness
        print()
        print(f"  [best checkpoint]")
        print(f"  {'heuristic':<18}  {k_header}")
        print(f"  {'-'*18}  {'  '.join('-'*6 for _ in k_values)}")
        for heuristic in heuristics:
            cells = []
            for k in k_values:
                rows = [r for r in k_tagged
                        if r["k"] == k and r["budget"] == budget and r["heuristic"] == heuristic]
                scores = [r["best"] for r in rows if not np.isnan(r["best"])]
                if scores:
                    mean = float(np.mean(scores))
                    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                    cells.append(f"{mean:.3f}±{std:.3f}")
                else:
                    cells.append("  —   ")
            print(f"  {heuristic:<18}  {'  '.join(f'{c:<10}' for c in cells)}")


def print_per_arm_table(run_dir_name: str, all_results: list[dict]) -> None:
    run_results = sorted(
        [r for r in all_results if r["run_dir"] == run_dir_name],
        key=lambda r: (r["budget"], r["heuristic"], r["rep"]),
    )
    if not run_results:
        return

    print(f"\n  Per-arm detail ({run_dir_name}):")
    print(f"  {'heuristic':<18} {'budget':>6} {'rep':>4}  "
          f"{'best':>6}  {'t5mean':>6}  {'gen%':>6}  {'ep_len':>6}  {'n_ckpts':>7}")
    print(f"  {'-'*18} {'-'*6} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
    for r in run_results:
        rep_label = "A" if r["rep"] == 0 else str(r["rep"])
        eff = r.get("gen_gen_efficiency", float("nan"))
        ep_len = r.get("gen_gen_ep_len_mean", float("nan"))
        print(
            f"  {r['heuristic']:<18} {r['budget']:>6} {rep_label:>4}  "
            f"{r['best']:.3f}  {r['topk_mean']:.3f}  "
            f"{eff:>6.1%}  {ep_len:>6.1f}  {r['n_checkpoints']:>7}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        help="Pipeline run directory (can be repeated). Defaults to all three constrained runs.",
    )
    parser.add_argument("--per-arm", action="store_true", help="Print per-arm detail table")
    parser.add_argument("--gen-stats", action="store_true", help="Print generation stats tables")
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="Print K-robustness comparison table (run dirs must end in _k{N})",
    )
    args = parser.parse_args()

    run_dirs = args.run_dirs or DEFAULT_RUN_DIRS

    all_results = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[warn] run dir not found: {run_dir}")
            continue
        results = collect_results(run_dir)
        all_results.extend(results)
        print(f"[info] {run_dir.name}: loaded {len(results)} arms")

    if not all_results:
        print("No results found.")
        return

    if args.k_sweep:
        print_k_sweep_table(all_results)
    else:
        run_dir_names = list(dict.fromkeys(r["run_dir"] for r in all_results))
        for name in run_dir_names:
            if args.per_arm:
                print_per_arm_table(name, all_results)
            print_eval_table(name, all_results)
            if args.gen_stats:
                print_gen_table(name, all_results)


if __name__ == "__main__":
    main()

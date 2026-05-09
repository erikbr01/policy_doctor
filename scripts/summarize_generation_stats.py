#!/usr/bin/env python3
"""Aggregate MimicGen generation success rates across budget sweep arms.

Scans pipeline run directories for generate_mimicgen_demos/result.json files
and prints a table of success rates, attempt counts, and timing per arm.

Usage:
    python scripts/summarize_generation_stats.py
    python scripts/summarize_generation_stats.py --run-dirs path/to/runs ...
    python scripts/summarize_generation_stats.py --include-rep-sweep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CUPID_ROOT = REPO_ROOT / "third_party" / "cupid"

# Default search roots: main budget sweep + rep sweep dirs
DEFAULT_ROOTS = [
    CUPID_ROOT / "data" / "pipeline_runs",
    Path("/mnt/ssdB/erik/cupid_data/pipeline_runs"),
]

ARM_PATTERN = re.compile(
    r"mimicgen_(?P<heuristic>random|behavior_graph|diversity)"
    r"_budget(?P<budget>\d+)"
    r"(?:_rep(?P<rep>\d+))?"
)

SEED_PATTERN = re.compile(r"apr26_sweep_seed(?P<seed>\d+)")


def parse_arm_name(arm_name: str) -> dict:
    m = ARM_PATTERN.match(arm_name)
    if not m:
        return {}
    return {
        "heuristic": m.group("heuristic"),
        "budget": int(m.group("budget")),
        "rep": int(m.group("rep")) if m.group("rep") is not None else 0,
    }


def load_stats(result_path: Path) -> dict | None:
    try:
        with open(result_path) as f:
            d = json.load(f)
        stats = d.get("stats", {})
        return {
            "num_success": stats.get("num_success", 0),
            "num_failures": stats.get("num_failures", 0),
            "num_attempts": stats.get("num_attempts", 0),
            "success_rate": (
                100.0 * stats["num_success"] / stats["num_attempts"]
                if stats.get("num_attempts", 0) > 0
                else 0.0
            ),
            "time_hrs": sum(
                float(s.get("time spent (hrs)", 0) or 0)
                for s in stats.get("per_seed_stats", [])
            ),
            "seed_source": d.get("seed_source", "?"),
        }
    except Exception as e:
        print(f"  [warn] could not read {result_path}: {e}", file=sys.stderr)
        return None


def find_results(roots: list[Path]) -> list[dict]:
    seen_arms = set()
    rows = []

    for root in roots:
        if not root.exists():
            continue
        for result_path in sorted(root.rglob("generate_mimicgen_demos/result.json")):
            # Resolve symlinks to avoid double-counting ssd vs worktree paths
            try:
                real = result_path.resolve()
            except Exception:
                real = result_path
            if real in seen_arms:
                continue
            seen_arms.add(real)

            arm_dir = result_path.parent.parent
            arm_name = arm_dir.name

            # Determine pipeline_seed from grandparent run dir name
            run_dir = arm_dir.parent.parent
            seed_m = SEED_PATTERN.search(run_dir.name)
            pipeline_seed = int(seed_m.group("seed")) if seed_m else -1

            # Determine sweep type (budget_sweep vs budget_rep_sweep)
            sweep_dir = arm_dir.parent
            sweep_type = sweep_dir.name  # e.g. "mimicgen_budget_sweep"

            parsed = parse_arm_name(arm_name)
            if not parsed:
                continue

            stats = load_stats(result_path)
            if stats is None:
                continue

            rows.append({
                "pipeline_seed": pipeline_seed,
                "sweep": "rep" if "rep_sweep" in sweep_type else "main",
                "heuristic": parsed["heuristic"],
                "budget": parsed["budget"],
                "rep": parsed["rep"],
                **stats,
            })

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found.")
        return

    # Sort: sweep, heuristic, budget, pipeline_seed, rep
    rows.sort(key=lambda r: (
        r["sweep"],
        r["heuristic"],
        r["budget"],
        r["pipeline_seed"],
        r["rep"],
    ))

    hdr = f"{'sweep':<5} {'heuristic':<15} {'budget':>6} {'pseed':>5} {'rep':>3}  "
    hdr += f"{'success':>7} {'attempts':>8} {'rate%':>6} {'time_h':>6}"
    print(hdr)
    print("-" * len(hdr))

    last_group = None
    for r in rows:
        group = (r["sweep"], r["heuristic"], r["budget"])
        if group != last_group:
            if last_group is not None:
                print()
            last_group = group
        print(
            f"{r['sweep']:<5} {r['heuristic']:<15} {r['budget']:>6} "
            f"{r['pipeline_seed']:>5} {r['rep']:>3}  "
            f"{r['num_success']:>7} {r['num_attempts']:>8} "
            f"{r['success_rate']:>6.1f} {r['time_hrs']:>6.2f}"
        )

    print()
    # Summary: mean success rate per heuristic × budget (across seeds/reps)
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["heuristic"], r["budget"])].append(r["success_rate"])

    print("Mean success rate by heuristic × budget (across pipeline seeds / reps):")
    hdr2 = f"{'heuristic':<15} {'budget':>6}  {'mean%':>6} {'min%':>6} {'max%':>6} {'n':>3}"
    print(hdr2)
    print("-" * len(hdr2))

    for (heuristic, budget), rates in sorted(groups.items()):
        print(
            f"{heuristic:<15} {budget:>6}  "
            f"{sum(rates)/len(rates):>6.1f} {min(rates):>6.1f} "
            f"{max(rates):>6.1f} {len(rates):>3}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        help="Pipeline run root directories to search (default: standard apr26 locations)",
    )
    args = parser.parse_args()

    roots = [r.expanduser() for r in args.run_dirs] if args.run_dirs else DEFAULT_ROOTS
    print(f"Searching {len(roots)} root(s)...", file=sys.stderr)
    rows = find_results(roots)
    print(f"Found {len(rows)} arms with generation stats.\n", file=sys.stderr)
    print_table(rows)


if __name__ == "__main__":
    main()

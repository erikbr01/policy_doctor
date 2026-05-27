#!/usr/bin/env python3
"""
Back up existing eval outputs and re-run all checkpoint evals across the three
nut-constrained sweep experiments, using 4 concurrent workers on cuda:0.

The new evals use the patched mimicgen_lowdim_runner which terminates episodes
early on success and logs episode lengths to eval_log.json.

Steps:
  1. Rename each arm's eval_save_episodes dir to <name>.backup
  2. Rename each arm's eval_mimicgen_combined pipeline step dir to .backup
  3. Verify all backups exist — abort if any are missing
  4. Run all checkpoint evals (4 concurrent, all on cuda:0)
  5. Aggregate per-arm results and write eval_mimicgen_combined/result.json + done

Usage:
    python scripts/rerun_evals_with_episode_lengths.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SSDБ_ROOT = Path("/mnt/ssdB/erik/cupid_data")
PIPELINE_ROOT = SSDБ_ROOT / "pipeline_runs"
EVAL_OUTPUT_ROOT = SSDБ_ROOT / "outputs/eval_save_episodes"
CUPID_ROOT = Path(
    "/home/erbauer/refactor_cupid/policy_doctor/.claude/worktrees"
    "/feat+mimicgen-eef-pipeline/third_party/cupid"
)

RUN_DIRS = [
    "mimicgen_square_apr26_seed1_d60_nut_constrained",
    "mimicgen_square_apr26_seed1_d60_budget300_nut_constrained",
    "mimicgen_square_apr26_seed1_d300_nut_constrained",
]

NUM_EPISODES = 500
TEST_START_SEED = 100000
DEVICE = "cuda:0"
CONDA_ENV = "mimicgen_torch2"
WORKERS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_arm_dirs() -> list[Path]:
    arms = []
    for run_name in RUN_DIRS:
        for phase in ["mimicgen_budget_sweep", "mimicgen_budget_rep_sweep"]:
            phase_dir = PIPELINE_ROOT / run_name / phase
            if not phase_dir.exists():
                continue
            for arm in sorted(phase_dir.iterdir()):
                if (arm / "train_on_combined_data" / "result.json").exists():
                    arms.append(arm)
    return arms


def eval_save_episodes_dir(arm_dir: Path) -> Path | None:
    """Return the resolved eval_save_episodes dir for this arm, or None if unknown."""
    rj = arm_dir / "eval_mimicgen_combined" / "result.json"
    if not rj.exists():
        return None
    data = json.loads(rj.read_text())
    # Standard pipeline arms: output_dir in checkpoints list
    checkpoints = data.get("checkpoints", [])
    if checkpoints:
        od = Path(checkpoints[0]["output_dir"])
        return od.parent.resolve()
    # Manual eval arms: eval_output_dir key
    eod = data.get("eval_output_dir")
    if eod:
        return Path(eod).resolve()
    return None


def get_train_dirs_and_checkpoints(arm_dir: Path) -> list[tuple[Path, list[Path]]]:
    """Return [(train_dir, [ckpt_paths])] for each train dir in this arm."""
    rj = arm_dir / "train_on_combined_data" / "result.json"
    if not rj.exists():
        return []
    train_data = json.loads(rj.read_text())
    result = []
    for train_dir_str in train_data.get("train_dirs", []):
        train_dir = Path(train_dir_str)
        ckpt_dir = train_dir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  [warn] checkpoint dir missing: {ckpt_dir}")
            continue
        ckpts = sorted(p for p in ckpt_dir.iterdir()
                       if p.suffix == ".ckpt" and p.stem != "latest")
        result.append((train_dir, ckpts))
    return result


# ---------------------------------------------------------------------------
# Step 1+2: Backup
# ---------------------------------------------------------------------------

def backup_arm(arm_dir: Path, dry_run: bool) -> tuple[Path | None, Path]:
    """
    Rename:
      eval_save_episodes/<arm_eval_dir>  →  <same>.backup
      <arm_dir>/eval_mimicgen_combined   →  <arm_dir>/eval_mimicgen_combined.backup

    Returns (eval_save_dir_backed_up_or_None, step_dir_backed_up).
    """
    # --- eval_save_episodes dir ---
    eval_dir = eval_save_episodes_dir(arm_dir)
    eval_backup = None
    if eval_dir is not None and eval_dir.exists():
        target = eval_dir.parent / (eval_dir.name + ".backup")
        if not dry_run:
            eval_dir.rename(target)
        eval_backup = target

    # --- pipeline step dir ---
    step_dir = arm_dir / "eval_mimicgen_combined"
    step_backup = arm_dir / "eval_mimicgen_combined.backup"
    if step_dir.exists():
        if not dry_run:
            step_dir.rename(step_backup)

    return eval_backup, step_backup


def do_backups(arms: list[Path], dry_run: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"Step 1+2: Backing up {len(arms)} arm eval dirs")
    print(f"{'='*60}")
    all_ok = True
    for arm in arms:
        eval_backup, step_backup = backup_arm(arm, dry_run)
        prefix = "[dry-run] " if dry_run else ""
        if eval_backup:
            print(f"  {prefix}eval_save  → {eval_backup.name}")
        if step_backup:
            print(f"  {prefix}step_dir   → {arm.name}/eval_mimicgen_combined.backup")

    if dry_run:
        return True

    print(f"\nStep 3: Verifying backups...")
    for arm in arms:
        step_backup = arm / "eval_mimicgen_combined.backup"
        if (arm / "eval_mimicgen_combined").exists():
            print(f"  ERROR: step dir not renamed for {arm.name}")
            all_ok = False
        if not step_backup.exists():
            print(f"  WARN: no backup step dir for {arm.name} (may not have existed)")

    if all_ok:
        print("  All step dirs backed up.")
    return all_ok


# ---------------------------------------------------------------------------
# Step 4: Run evals
# ---------------------------------------------------------------------------

def run_one_eval(job: dict) -> tuple[dict, str]:
    output_dir: Path = job["output_dir"]
    existing_log = output_dir / "eval_log.json"
    if existing_log.exists():
        return job, "cached"

    cmd = [
        "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", str(CUPID_ROOT / "eval_save_episodes.py"),
        f"--output_dir={output_dir}",
        f"--train_dir={job['train_dir']}",
        f"--train_ckpt={job['ckpt_stem']}",
        f"--num_episodes={NUM_EPISODES}",
        f"--test_start_seed={TEST_START_SEED}",
        "--overwrite=False",
        f"--device={DEVICE}",
        "--save_episodes=False",
    ]
    result = subprocess.run(cmd, cwd=str(CUPID_ROOT))
    status = "done" if result.returncode == 0 else f"FAILED(rc={result.returncode})"
    return job, status


def collect_eval_jobs(arms: list[Path]) -> list[dict]:
    jobs = []
    for arm_dir in arms:
        for train_dir, ckpts in get_train_dirs_and_checkpoints(arm_dir):
            for ckpt_path in ckpts:
                output_dir = EVAL_OUTPUT_ROOT / train_dir.name / ckpt_path.stem
                jobs.append({
                    "arm": arm_dir.name,
                    "run": arm_dir.parent.parent.name,
                    "train_dir": train_dir,
                    "ckpt_stem": ckpt_path.stem,
                    "output_dir": output_dir,
                })
    return jobs


def do_evals(jobs: list[dict], dry_run: bool) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Step 4: Running {len(jobs)} checkpoint evals ({WORKERS} concurrent, {DEVICE})")
    print(f"{'='*60}")

    if dry_run:
        for j in jobs:
            print(f"  [dry-run] {j['arm']} / {j['ckpt_stem']}")
        return jobs

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one_eval, j): j for j in jobs}
        done_count = 0
        for fut in as_completed(futures):
            job, status = fut.result()
            done_count += 1
            print(f"  [{done_count}/{len(jobs)}] {job['arm']} / {job['ckpt_stem']}  → {status}")
            if status.startswith("FAILED"):
                failed.append(job)

    if failed:
        print(f"\n  {len(failed)} eval(s) FAILED:")
        for j in failed:
            print(f"    {j['arm']} / {j['ckpt_stem']}")
    else:
        print(f"\n  All {len(jobs)} evals completed.")

    return failed


# ---------------------------------------------------------------------------
# Step 5: Aggregate per-arm results
# ---------------------------------------------------------------------------

def read_eval_log(output_dir: Path) -> dict | None:
    log = output_dir / "eval_log.json"
    if not log.exists():
        return None
    data = json.loads(log.read_text())
    return data


def aggregate_arm(arm_dir: Path) -> bool:
    """Read checkpoint eval results and write eval_mimicgen_combined/result.json + done."""
    train_result_path = arm_dir / "train_on_combined_data" / "result.json"
    if not train_result_path.exists():
        return False
    train_data = json.loads(train_result_path.read_text())
    heuristic = train_data.get("heuristic", "unknown")

    all_checkpoint_results = []
    for train_dir, ckpts in get_train_dirs_and_checkpoints(arm_dir):
        for ckpt_path in ckpts:
            output_dir = EVAL_OUTPUT_ROOT / train_dir.name / ckpt_path.stem
            log_data = read_eval_log(output_dir)
            if log_data is None:
                print(f"  [warn] missing eval_log.json for {arm_dir.name} / {ckpt_path.stem}")
                continue
            rate = float(log_data.get("test/mean_score", 0.0))
            ep_len = log_data.get("test/mean_episode_length")
            succ_ep_len = log_data.get("test/mean_success_episode_length")
            entry = {
                "checkpoint": ckpt_path.stem,
                "output_dir": str(output_dir),
                "num_episodes": NUM_EPISODES,
                "num_success": round(rate * NUM_EPISODES),
                "success_rate": round(rate, 4),
            }
            if ep_len is not None:
                entry["mean_episode_length"] = round(float(ep_len), 1)
            if succ_ep_len is not None:
                entry["mean_success_episode_length"] = round(float(succ_ep_len), 1)
            all_checkpoint_results.append(entry)

    if not all_checkpoint_results:
        return False

    rates = [r["success_rate"] for r in all_checkpoint_results]
    mean_rate = round(sum(rates) / len(rates), 4)
    best_rate = round(max(rates), 4)
    ep_lens = [r["mean_episode_length"] for r in all_checkpoint_results if "mean_episode_length" in r]
    succ_ep_lens = [r["mean_success_episode_length"] for r in all_checkpoint_results
                    if "mean_success_episode_length" in r]
    mean_ep_len = round(sum(ep_lens) / len(ep_lens), 1) if ep_lens else None
    mean_succ_ep_len = round(sum(succ_ep_lens) / len(succ_ep_lens), 1) if succ_ep_lens else None

    result = {
        "heuristic": heuristic,
        "train_dir": str(get_train_dirs_and_checkpoints(arm_dir)[0][0]) if get_train_dirs_and_checkpoints(arm_dir) else "",
        "checkpoints": all_checkpoint_results,
        "mean_success_rate": mean_rate,
        "best_success_rate": best_rate,
    }
    if mean_ep_len is not None:
        result["mean_episode_length"] = mean_ep_len
    if mean_succ_ep_len is not None:
        result["mean_success_episode_length"] = mean_succ_ep_len

    step_dir = arm_dir / "eval_mimicgen_combined"
    step_dir.mkdir(exist_ok=True)
    (step_dir / "result.json").write_text(json.dumps(result, indent=2))
    (step_dir / "done").touch()

    succ_str = f"  ep_len={mean_succ_ep_len:.0f}" if mean_succ_ep_len is not None else ""
    print(f"  {arm_dir.name}: best={best_rate:.3f} mean={mean_rate:.3f}{succ_str}")
    return True


def do_aggregation(arms: list[Path], dry_run: bool) -> None:
    print(f"\n{'='*60}")
    print(f"Step 5: Aggregating results for {len(arms)} arms")
    print(f"{'='*60}")
    if dry_run:
        print("  [dry-run] skipping")
        return
    for arm in arms:
        aggregate_arm(arm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, don't execute")
    args = parser.parse_args()

    arms = all_arm_dirs()
    print(f"Found {len(arms)} arms across {len(RUN_DIRS)} run dirs")

    ok = do_backups(arms, dry_run=args.dry_run)
    if not ok and not args.dry_run:
        print("Backup verification failed — aborting.")
        sys.exit(1)

    jobs = collect_eval_jobs(arms)
    print(f"Total checkpoint evals to run: {len(jobs)}")

    failed = do_evals(jobs, dry_run=args.dry_run)

    if failed and not args.dry_run:
        print(f"\nWarning: {len(failed)} evals failed; aggregation may be incomplete.")

    do_aggregation(arms, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()

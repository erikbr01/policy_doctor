#!/usr/bin/env python3
"""Parallel post-training eval for all K-sweep arms.

Runs eval_save_episodes.py (500 episodes, n_envs=50, test_start_seed=100000)
on every top-k checkpoint for all 15 arms. Uses 2 concurrent workers so two
checkpoints are always in-flight simultaneously.

K=20 BG is skipped — it already has real 500-episode eval results.
Usage: conda run -n mimicgen_torch2 python scripts/run_k_sweep_evals.py
"""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = pathlib.Path(__file__).resolve().parents[1]
CUPID = REPO / "third_party" / "cupid"
PIPE_BASE = CUPID / "data" / "pipeline_runs"
TRAIN_BASE = CUPID / "data" / "outputs" / "train" / "apr26_sweep_demos60_budget300_nut_constrained_tight"
EVAL_BASE = pathlib.Path("/home/erbauer/data/cupid_data/outputs/eval_save_episodes")
LOG = REPO / "logs" / "k_sweep_evals.log"

N_ENVS = 50
NUM_EPISODES = 500
TEST_START_SEED = 100000
DEVICE = "cuda:0"
N_WORKERS = 2

# K=20 BG already has real eval — skip it.
SKIP = {("k20", "behavior_graph")}

log_lock = threading.Lock()


def log(msg: str) -> None:
    ts = time.strftime("%H:%M UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with log_lock:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def get_train_dir(k: int, arm: str) -> pathlib.Path | None:
    result_json = (
        PIPE_BASE
        / f"mimicgen_square_apr26_seed1_d60_budget300_nut_constrained_tight_k{k}"
        / "mimicgen_budget_sweep"
        / f"mimicgen_{arm}_budget300"
        / "train_on_combined_data"
        / "result.json"
    )
    if not result_json.exists():
        return None
    d = json.loads(result_json.read_text())
    dirs = d.get("train_dirs") or ([d.get("train_dir")] if d.get("train_dir") else [])
    if not dirs:
        return None
    p = pathlib.Path(dirs[0])
    return p if p.exists() else None


def get_checkpoints(train_dir: pathlib.Path) -> list[pathlib.Path]:
    ckpt_dir = train_dir / "checkpoints"
    return sorted(
        p for p in ckpt_dir.iterdir()
        if p.suffix == ".ckpt" and p.stem != "latest"
    )


def eval_checkpoint(train_dir: pathlib.Path, ckpt: pathlib.Path) -> dict:
    """Run eval_save_episodes.py for one checkpoint. Returns result dict."""
    output_dir = EVAL_BASE / train_dir.name / ckpt.stem
    eval_log = output_dir / "eval_log.json"

    if eval_log.exists():
        data = json.loads(eval_log.read_text())
        rate = float(data["test/mean_score"])
        ep_len = data.get("test/mean_episode_length")
        log(f"  [cached] {train_dir.name[-40:]} {ckpt.stem} rate={rate:.3f}")
        return {
            "checkpoint": ckpt.stem,
            "output_dir": str(output_dir),
            "num_episodes": NUM_EPISODES,
            "num_success": round(rate * NUM_EPISODES),
            "success_rate": round(rate, 4),
            **({"mean_episode_length": round(float(ep_len), 1)} if ep_len else {}),
        }

    # Remove partial output if it exists without eval_log.json
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    log(f"  [running] {train_dir.name[-40:]} {ckpt.stem}")
    cmd = [
        "conda", "run", "-n", "mimicgen_torch2", "--no-capture-output",
        "python", str(CUPID / "eval_save_episodes.py"),
        f"--output_dir={output_dir}",
        f"--train_dir={train_dir}",
        f"--train_ckpt={ckpt.stem}",
        f"--num_episodes={NUM_EPISODES}",
        f"--test_start_seed={TEST_START_SEED}",
        f"--n_envs={N_ENVS}",
        "--overwrite=False",
        f"--device={DEVICE}",
        "--save_episodes=False",
    ]
    result = subprocess.run(cmd, cwd=str(CUPID), capture_output=False)
    if result.returncode != 0:
        log(f"  [FAILED] {ckpt.stem} exit={result.returncode}")
        raise RuntimeError(f"eval failed for {ckpt.stem}")

    data = json.loads(eval_log.read_text())
    rate = float(data["test/mean_score"])
    ep_len = data.get("test/mean_episode_length")
    successes = round(rate * NUM_EPISODES)
    log(f"  [done]   {train_dir.name[-40:]} {ckpt.stem} successes~{successes}/{NUM_EPISODES} rate={rate:.3f}")
    return {
        "checkpoint": ckpt.stem,
        "output_dir": str(output_dir),
        "num_episodes": NUM_EPISODES,
        "num_success": successes,
        "success_rate": round(rate, 4),
        **({"mean_episode_length": round(float(ep_len), 1)} if ep_len else {}),
    }


def write_arm_result(k: int, arm: str, train_dir: pathlib.Path, ckpt_results: list[dict]) -> None:
    """Write result.json and eval sentinel for one arm."""
    arm_eval_dir = (
        PIPE_BASE
        / f"mimicgen_square_apr26_seed1_d60_budget300_nut_constrained_tight_k{k}"
        / "mimicgen_budget_sweep"
        / f"mimicgen_{arm}_budget300"
        / "eval_mimicgen_combined"
    )
    arm_eval_dir.mkdir(parents=True, exist_ok=True)

    rates = [r["success_rate"] for r in ckpt_results]
    mean_rate = round(sum(rates) / len(rates), 4)
    best_rate = round(max(rates), 4)
    ep_lens = [r["mean_episode_length"] for r in ckpt_results if "mean_episode_length" in r]
    mean_ep_len = round(sum(ep_lens) / len(ep_lens), 1) if ep_lens else None

    result = {
        "heuristic": arm,
        "train_dir": str(train_dir),
        "checkpoints": ckpt_results,
        "mean_success_rate": mean_rate,
        "best_success_rate": best_rate,
    }
    if mean_ep_len is not None:
        result["mean_episode_length"] = mean_ep_len

    result_path = arm_eval_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    (arm_eval_dir / "done").touch()
    log(f"=== K={k} {arm} COMPLETE: best={best_rate:.3f} mean={mean_rate:.3f} ===")


def build_task_list() -> list[tuple[int, str, pathlib.Path, pathlib.Path]]:
    """Returns list of (k, arm, train_dir, ckpt_path) tuples."""
    tasks = []
    for k in [5, 10, 15, 20, 25]:
        for arm in ["behavior_graph", "diversity", "random"]:
            if (f"k{k}", arm) in SKIP:
                log(f"Skipping K={k} {arm} (already evaluated)")
                continue
            train_dir = get_train_dir(k, arm)
            if train_dir is None:
                log(f"WARNING: no train_dir for K={k} {arm}, skipping")
                continue
            ckpts = get_checkpoints(train_dir)
            if not ckpts:
                log(f"WARNING: no checkpoints for K={k} {arm}, skipping")
                continue
            for ckpt in ckpts:
                tasks.append((k, arm, train_dir, ckpt))
    return tasks


def main() -> None:
    LOG.parent.mkdir(exist_ok=True)
    log(f"=== K-sweep post-training eval starting: N_WORKERS={N_WORKERS} n_envs={N_ENVS} episodes={NUM_EPISODES} ===")

    tasks = build_task_list()
    total = len(tasks)
    log(f"Total checkpoint evals to run: {total}")

    # Group by arm for result aggregation
    from collections import defaultdict
    arm_tasks: dict[tuple[int, str], list] = defaultdict(list)
    arm_train_dir: dict[tuple[int, str], pathlib.Path] = {}
    for k, arm, train_dir, ckpt in tasks:
        arm_tasks[(k, arm)].append(ckpt)
        arm_train_dir[(k, arm)] = train_dir

    # Track results per arm
    arm_results: dict[tuple[int, str], list[dict]] = defaultdict(list)
    arm_remaining: dict[tuple[int, str], int] = {key: len(ckpts) for key, ckpts in arm_tasks.items()}
    results_lock = threading.Lock()

    done_count = 0

    def run_task(k: int, arm: str, train_dir: pathlib.Path, ckpt: pathlib.Path) -> tuple:
        result = eval_checkpoint(train_dir, ckpt)
        return (k, arm, train_dir, result)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(run_task, k, arm, train_dir, ckpt): (k, arm, train_dir, ckpt)
            for k, arm, train_dir, ckpt in tasks
        }
        for future in as_completed(futures):
            try:
                k, arm, train_dir, result = future.result()
                with results_lock:
                    arm_results[(k, arm)].append(result)
                    arm_remaining[(k, arm)] -= 1
                    done_count += 1
                    log(f"Progress: {done_count}/{total} checkpoint evals done")
                    # Write arm result when all its checkpoints are done
                    if arm_remaining[(k, arm)] == 0:
                        write_arm_result(k, arm, train_dir, arm_results[(k, arm)])
            except Exception as e:
                k, arm, train_dir, ckpt = futures[future]
                log(f"ERROR: K={k} {arm} ckpt={ckpt.stem}: {e}")

    log("=== All evals complete ===")


if __name__ == "__main__":
    main()

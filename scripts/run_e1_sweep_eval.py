"""Evaluate every clustering dir under a sweep root via the E1 protocol.

Iterates each subdirectory of ``--sweep_root`` that contains a
``manifest.yaml`` and ``cluster_labels.npy``, infers ``K`` from the manifest's
``n_clusters`` field, and invokes ``scripts/run_e1_transport_r512_qwen.py``
against it. E1 outputs land under ``--results_root/<clustering_slug>/``.

Skips clusterings whose corresponding results dir already has ``metrics.json``
(unless ``--force``). Each eval is run sequentially on the single GPU pinned by
``CUDA_VISIBLE_DEVICES`` — this script does not parallelize across GPUs; pin
the GPU at invocation time.

Example:

    CUDA_VISIBLE_DEVICES=1 python scripts/run_e1_sweep_eval.py \\
      --sweep_root /tmp/clustering_sweeps/transport_r512_seed0_alt \\
      --results_root experiments/e1_sweep_transport_r512_seed0 \\
      --eval_dir /mnt/ssdB/.../latest \\
      --n_example 3 --n_query 3 --n_repetitions 3 \\
      --global_episode_disjoint \\
      --view_window_extension 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time
from typing import List

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True,
                    help="Directory containing per-combo clustering dirs (output of run_clustering_sweep.py).")
    ap.add_argument("--results_root", required=True,
                    help="Where E1 outputs land. Mirrors the sweep_root layout one-to-one.")
    ap.add_argument("--eval_dir", required=True,
                    help="Eval episodes dir (same one used to build the clusterings).")
    ap.add_argument("--runner", default=str(
        pathlib.Path(__file__).parent / "run_e1_transport_r512_qwen.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--n_example", type=int, default=3)
    ap.add_argument("--n_query", type=int, default=3)
    ap.add_argument("--n_repetitions", type=int, default=3)
    ap.add_argument("--max_frames_per_storyboard", type=int, default=4)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--image_max_pixels", type=int, default=1024 * 1024)
    ap.add_argument("--global_episode_disjoint", action="store_true")
    ap.add_argument("--view_window_extension", type=int, default=0)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0",
                    help="Pass cuda:0 (default); pin the actual GPU via CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--include_action_text", action="store_true")
    ap.add_argument("--include_state_text", action="store_true")
    ap.add_argument("--max_clusters_override", type=int, default=None,
                    help="If set, force this max_clusters on every eval; otherwise read from manifest n_clusters.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run evaluations whose results dir already has metrics.json.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print planned commands without running.")
    ap.add_argument("--include_pattern", default=None,
                    help="Optional substring filter on clustering dir name.")
    return ap.parse_args()


def _list_clustering_dirs(root: pathlib.Path) -> List[pathlib.Path]:
    dirs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "manifest.yaml").exists() or not (p / "cluster_labels.npy").exists():
            continue
        dirs.append(p)
    return dirs


def _max_clusters_from_manifest(manifest_path: pathlib.Path) -> int:
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    k = manifest.get("n_clusters")
    if k is None:
        raise RuntimeError(f"manifest missing n_clusters: {manifest_path}")
    return int(k)


def main() -> int:
    args = _parse_args()
    sweep_root = pathlib.Path(args.sweep_root).resolve()
    results_root = pathlib.Path(args.results_root).resolve()
    if not sweep_root.is_dir():
        raise SystemExit(f"sweep_root does not exist: {sweep_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    cdirs = _list_clustering_dirs(sweep_root)
    if args.include_pattern:
        cdirs = [d for d in cdirs if args.include_pattern in d.name]
    print(f"[e1_sweep_eval] {len(cdirs)} clustering dirs under {sweep_root}")

    n_skip = n_run = n_fail = 0
    summary_path = results_root / "sweep_eval_summary.jsonl"
    t0 = time.time()
    with open(summary_path, "a") as summary_f:
        for i, cdir in enumerate(cdirs, 1):
            slug = cdir.name
            res_dir = results_root / slug
            metrics_path = res_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"[e1_sweep_eval] [{i}/{len(cdirs)}] SKIP (metrics exist): {slug}")
                n_skip += 1
                continue

            try:
                k = args.max_clusters_override or _max_clusters_from_manifest(
                    cdir / "manifest.yaml"
                )
            except RuntimeError as e:
                print(f"  ! {e} — skipping")
                n_fail += 1
                continue

            cmd = [
                args.python, args.runner,
                "--clustering_dir", str(cdir),
                "--eval_dir", str(args.eval_dir),
                "--out_dir", str(res_dir),
                "--max_clusters", str(k),
                "--n_example", str(args.n_example),
                "--n_query", str(args.n_query),
                "--n_repetitions", str(args.n_repetitions),
                "--max_frames_per_storyboard", str(args.max_frames_per_storyboard),
                "--random_seed", str(args.random_seed),
                "--image_max_pixels", str(args.image_max_pixels),
                "--model_id", args.model_id,
                "--device", args.device,
                "--view_window_extension", str(args.view_window_extension),
            ]
            if args.global_episode_disjoint:
                cmd.append("--global_episode_disjoint")
            if args.load_in_4bit:
                cmd.append("--load_in_4bit")
            if args.load_in_8bit:
                cmd.append("--load_in_8bit")
            if args.device_map is not None:
                cmd.extend(["--device_map", args.device_map])
            if args.include_action_text:
                cmd.append("--include_action_text")
            if args.include_state_text:
                cmd.append("--include_state_text")

            print(f"\n[e1_sweep_eval] [{i}/{len(cdirs)}] {slug}  K={k}")
            print(f"  $ {shlex.join(cmd)}")
            if args.dry_run:
                n_run += 1
                continue

            res_dir.mkdir(parents=True, exist_ok=True)
            log_path = res_dir / "e1_eval.log"
            t_start = time.time()
            with open(log_path, "w") as logf:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
            t_elapsed = time.time() - t_start

            entry = {
                "clustering_dir": str(cdir),
                "results_dir": str(res_dir),
                "K": k,
                "exit_code": proc.returncode,
                "elapsed_s": round(t_elapsed, 1),
            }
            if proc.returncode == 0 and metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                entry.update({
                    "top1_accuracy": m.get("top1_accuracy"),
                    "binomial_test_pvalue": m.get("binomial_test_pvalue"),
                    "n_total": m.get("n_total"),
                    "unclear_rate": m.get("unclear_rate"),
                })
                n_run += 1
                print(
                    f"  done ({t_elapsed:.0f}s)  "
                    f"top1={m.get('top1_accuracy'):.3f}  "
                    f"p={m.get('binomial_test_pvalue'):.2e}"
                )
            else:
                entry["error"] = "metrics.json missing or runner failed"
                n_fail += 1
                print(f"  FAILED (exit {proc.returncode}); see {log_path}")

            summary_f.write(json.dumps(entry) + "\n")
            summary_f.flush()

    print(
        f"\n[e1_sweep_eval] done in {time.time()-t0:.0f}s  "
        f"run={n_run} skip={n_skip} fail={n_fail}\n"
        f"  per-combo summary: {summary_path}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

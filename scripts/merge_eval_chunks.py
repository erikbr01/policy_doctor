"""Merge parallel-launched eval chunks back into the canonical
`<task>_n500/episodes/` directory.

For each task, sweeps `<task>_n500/episodes/*.pkl` (existing main run) plus
`<task>_n500_chunk*/episodes/*.pkl` (parallel chunks) and copies them all
into `<task>_n500/episodes/` with sequential `ep{idx:04d}_{succ|fail}.pkl`
names. Idempotent — re-running won't duplicate or renumber existing files.

Symlinks instead of copies to avoid wasting SSD space.

Usage:
    python scripts/merge_eval_chunks.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

EVAL_ROOT = pathlib.Path(
    "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/graph_simplification"
)

TASKS = ["transport_mh_jan28", "square_mh_feb5", "lift_mh_jan26"]


def _parse_succ(name: str) -> str:
    """Return 'succ' or 'fail' from a filename like 'ep0042_succ.pkl'."""
    stem = pathlib.Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1] in {"succ", "fail"}:
        return parts[-1]
    return "succ"  # fallback


def merge_task(task: str, dry_run: bool = False) -> int:
    """Symlink chunk episode pkls into main_eps, de-duping by source path.

    Re-running is safe: if a chunk pkl is already symlinked into main_eps
    under any name, it's skipped on subsequent runs.
    """
    main_dir = EVAL_ROOT / f"{task}_n500"
    main_eps = main_dir / "episodes"
    main_eps.mkdir(parents=True, exist_ok=True)

    chunks: list[pathlib.Path] = []
    for chunk in sorted(EVAL_ROOT.glob(f"{task}_n500_chunk*")):
        eps = chunk / "episodes"
        if eps.is_dir():
            chunks.append(eps)

    print(f"\n=== {task}: {len(chunks)} chunk dirs ===")

    # Find next available index AND build a set of already-symlinked sources.
    next_idx = 0
    already_linked: set[str] = set()
    real_files = 0  # non-symlink pkls (e.g. from the original serial eval)
    for p in sorted(main_eps.iterdir()):
        if p.suffix != ".pkl":
            continue
        try:
            idx = int(p.stem.split("_")[0].lstrip("ep"))
            next_idx = max(next_idx, idx + 1)
        except ValueError:
            pass
        if p.is_symlink():
            try:
                tgt = p.resolve()
                already_linked.add(str(tgt))
            except FileNotFoundError:
                pass
        else:
            real_files += 1

    new_links = 0
    for chunk in chunks:
        for pkl in sorted(chunk.iterdir()):
            if pkl.suffix != ".pkl":
                continue
            src_abs = pkl.resolve()
            src_key = str(src_abs)
            if src_key in already_linked:
                continue
            succ = _parse_succ(pkl.name)
            new_name = f"ep{next_idx:04d}_{succ}.pkl"
            dst = main_eps / new_name
            # Bump past any existing name collision.
            while dst.exists() or dst.is_symlink():
                next_idx += 1
                new_name = f"ep{next_idx:04d}_{succ}.pkl"
                dst = main_eps / new_name
            if dry_run:
                print(f"  [dry] {src_abs}  ->  {dst}")
            else:
                os.symlink(src_abs, dst)
            already_linked.add(src_key)
            next_idx += 1
            new_links += 1

    total = real_files + len(already_linked)
    print(f"  total: {total} ({real_files} real + {len(already_linked)} symlinks; "
          f"+{new_links} new this run)")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    args = ap.parse_args()

    for task in args.tasks:
        merge_task(task, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

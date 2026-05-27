"""Cartesian-product hyperparam sweep over slice representations + clustering knobs.

Reads a YAML/JSON sweep spec, expands the grid, calls ``build_alt_clustering.py``
for each combination. Writes one clustering dir per combo under a sweep root.

Usage:

    python scripts/run_clustering_sweep.py --spec sweep_specs/transport_alt.yaml

Spec format (YAML):

    eval_dir: /path/to/eval/.../latest
    sweep_root: /tmp/clustering_sweeps/transport_r512_alt
    seed: 42
    grid:
      representation: [infembed, state, state_action]
      window_width: [3, 5, 10]
      stride: [2]
      aggregation: [sum, mean]
      prescale: [standard]
      umap_n_components: [50, 100]
      n_clusters: [10, 15, 20]
      # state-only kwargs (ignored by other reps)
      obs_strategy: [current]
      # state_action-only kwargs (ignored by other reps)
      action_strategy: [executed]

The cartesian product over the lists becomes one run each. Per-combo out_dir
names are derived from the param values so they're human-readable. Existing
out_dirs are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Knobs that are forwarded to build_alt_clustering as flags. Ordering also
# determines the slug component order — keep representation first for grouping.
_GRID_KEYS_ORDERED: List[str] = [
    "representation",
    "window_width",
    "stride",
    "aggregation",
    "prescale",
    "normalize",
    "reducer",
    "umap_n_components",
    "cluster_method",
    "n_clusters",
    "obs_strategy",
    "action_strategy",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="YAML or JSON sweep spec.")
    ap.add_argument("--builder", default=str(
        pathlib.Path(__file__).parent / "build_alt_clustering.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--force", action="store_true",
                    help="Re-run combos whose out_dir already has manifest.yaml.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the planned commands without running.")
    return ap.parse_args()


def _load_spec(path: pathlib.Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = [k for k in _GRID_KEYS_ORDERED if k in grid]
    extras = [k for k in grid if k not in _GRID_KEYS_ORDERED]
    keys.extend(extras)
    values = [list(grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    for tup in itertools.product(*values):
        combos.append(dict(zip(keys, tup)))
    return combos


def _slug_for(combo: Dict[str, Any]) -> str:
    rep = combo.get("representation", "rep")
    parts = [str(rep)]
    for k in _GRID_KEYS_ORDERED:
        if k == "representation" or k not in combo:
            continue
        v = combo[k]
        if k == "obs_strategy" and combo.get("representation") not in ("state", "state_action"):
            continue
        if k == "action_strategy" and combo.get("representation") != "state_action":
            continue
        # Skip "default-ish" values to keep slugs short
        if k == "normalize" and v == "none":
            continue
        if k == "reducer" and v == "umap":
            continue
        if k == "cluster_method" and v == "kmeans":
            continue
        if k == "stride" and v == 2:
            continue
        parts.append(f"{_short(k)}={v}")
    return "__".join(parts)


def _short(key: str) -> str:
    return {
        "window_width": "w",
        "stride": "s",
        "aggregation": "agg",
        "prescale": "pre",
        "normalize": "norm",
        "reducer": "red",
        "umap_n_components": "d",
        "cluster_method": "alg",
        "n_clusters": "K",
        "obs_strategy": "obs",
        "action_strategy": "act",
    }.get(key, key)


def _filter_kwargs_for_rep(combo: Dict[str, Any]) -> Dict[str, Any]:
    """Drop knobs that don't apply to the chosen representation."""
    rep = combo.get("representation", "infembed")
    out = dict(combo)
    if rep != "state" and rep != "state_action":
        out.pop("obs_strategy", None)
    if rep != "state_action":
        out.pop("action_strategy", None)
    return out


def _build_command(
    builder: str, python: str, eval_dir: str, out_dir: pathlib.Path,
    seed: int, combo: Dict[str, Any], task_config: str,
) -> List[str]:
    cmd = [
        python, builder,
        "--eval_dir", str(eval_dir),
        "--out_dir", str(out_dir),
        "--seed", str(seed),
        "--task_config", task_config,
    ]
    combo_filt = _filter_kwargs_for_rep(combo)
    for k, v in combo_filt.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def main() -> int:
    args = _parse_args()
    spec = _load_spec(pathlib.Path(args.spec))

    # Kendama rollouts use HDF5 episodes + optional policy ckpt, not cupid eval dirs.
    if "rollouts" in spec:
        from scripts.run_kendama_clustering_sweep import run_sweep

        return run_sweep(
            spec_path=pathlib.Path(args.spec),
            force=args.force,
            dry_run=args.dry_run,
        )

    eval_dir = spec["eval_dir"]
    sweep_root = pathlib.Path(spec["sweep_root"])
    sweep_root.mkdir(parents=True, exist_ok=True)
    seed = int(spec.get("seed", 42))
    task_config = spec.get("task_config", f"sweep_{sweep_root.name}")

    combos = _expand_grid(spec["grid"])
    print(f"[sweep] {len(combos)} combos -> {sweep_root}")

    # Deduplicate combos that resolve to the same filtered kwargs (e.g. multiple
    # obs_strategy values for representation=infembed are all equivalent).
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for c in combos:
        key = tuple(sorted(_filter_kwargs_for_rep(c).items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    print(f"[sweep] {len(deduped)} unique combos after filtering rep-specific kwargs")

    n_skip = n_run = n_fail = 0
    t0 = time.time()
    for i, combo in enumerate(deduped, 1):
        slug = _slug_for(combo)
        out_dir = sweep_root / slug
        manifest = out_dir / "manifest.yaml"
        cmd = _build_command(args.builder, args.python, eval_dir, out_dir, seed, combo, task_config)

        if manifest.exists() and not args.force:
            print(f"[sweep] [{i}/{len(deduped)}] SKIP (already done): {slug}")
            n_skip += 1
            continue

        print(f"\n[sweep] [{i}/{len(deduped)}] {slug}")
        print(f"  $ {shlex.join(cmd)}")
        if args.dry_run:
            n_run += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "build.log"
        with open(log_path, "w") as logf:
            res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print(f"  FAILED (exit {res.returncode}); see {log_path}")
            n_fail += 1
        else:
            n_run += 1

    print(
        f"\n[sweep] done in {time.time()-t0:.0f}s  "
        f"run={n_run} skip={n_skip} fail={n_fail}"
    )
    if n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compute metrics.json for demo-sweep clustering directories.

Wraps ``compute_clustering_metrics._compute_for_dir`` so Kendama (and other
one-off clusterings) get ``silhouette_mean`` without running the full Hydra
pipeline.

Usage:
    conda activate policy_doctor
    python scripts/compute_clustering_metrics.py --task kendama_may22
    python scripts/compute_clustering_metrics.py --clustering_dir \\
        data/demo_sweep/kendama_may22/run_clustering/clustering/aggregate_first/state_full_history_w20_s10_seed42_kmeans_k15
    python scripts/compute_clustering_metrics.py --task kendama_may22 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.curation_pipeline.steps.compute_clustering_metrics import (  # noqa: E402
    _compute_for_dir,
)


def _iter_clustering_dirs(root: Path):
    if (root / "cluster_labels.npy").exists():
        yield root
        return
    if not root.is_dir():
        return
    for labels_path in sorted(root.rglob("cluster_labels.npy")):
        yield labels_path.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--task",
        help="Task name under data/demo_sweep/<task>/run_clustering/clustering/",
    )
    g.add_argument(
        "--clustering_dir",
        type=Path,
        help="Single clustering directory (contains cluster_labels.npy)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "data" / "demo_sweep",
        help="Parent of task dirs when using --task (default: data/demo_sweep)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when metrics.json is newer than manifest.yaml",
    )
    args = ap.parse_args()

    if args.clustering_dir is not None:
        roots = [args.clustering_dir.resolve()]
    else:
        clu_root = args.root / args.task / "run_clustering" / "clustering"
        if not clu_root.is_dir():
            print(f"ERROR: {clu_root} not found", file=sys.stderr)
            return 1
        roots = list(_iter_clustering_dirs(clu_root))

    if not roots:
        print("ERROR: no clusterings found", file=sys.stderr)
        return 1

    n_ok = 0
    for clu_dir in roots:
        metrics_path = clu_dir / "metrics.json"
        if args.force and metrics_path.exists():
            metrics_path.unlink()
        try:
            rel = clu_dir.relative_to(_REPO_ROOT)
        except ValueError:
            rel = clu_dir
        print(f"[metrics] {rel}")
        metrics = _compute_for_dir(clu_dir)
        if metrics is None:
            print("  failed")
            continue
        n_ok += 1
        sil = metrics.get("silhouette_mean")
        sil_s = f"{sil:.4f}" if sil is not None else "n/a"
        print(f"  silhouette_mean={sil_s}")

    print(f"Done: {n_ok}/{len(roots)} clusterings")
    return 0 if n_ok == len(roots) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot silhouette score vs K for all K-sweep clusterings.

Replicates the K-sweep MV elbow plot style: one subplot per task,
one line per (rep, w, s) setting, K on x-axis.

Usage:
    conda run -n cupid python scripts/plot_silhouette_sweep.py \
        --data_dir /Users/erik/stanford/asl_rotation/data/graph_simplification/clusterings \
        --out /tmp/silhouette_sweep.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SLUG_RE = re.compile(
    r"^(?P<task>[a-z][a-z0-9_]+?)__(?P<rep>[a-z][a-z0-9_]+?)__w(?P<w>\d+)_s(?P<s>\d+)__K(?P<K>\d+)$"
)

REP_LABELS = {
    "policy_emb": "policy_emb",
    "infembed": "infembed",
}

STYLE = {
    # (rep, w, s) → (color, linestyle, marker)
}

_COLORS_POLICY = ["#60a5fa", "#3b82f6", "#1d4ed8"]  # blues
_COLORS_INFEMBED = ["#f59e0b", "#d97706", "#b45309"]  # ambers


def compute_silhouette(emb: np.ndarray, labels: np.ndarray,
                       sample_size: Optional[int] = 2000,
                       random_state: int = 0) -> float:
    from sklearn.metrics import silhouette_score
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    if sample_size and len(emb) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(emb), size=sample_size, replace=False)
        emb, labels = emb[idx], labels[idx]
    # Drop any -1 noise labels
    mask = labels >= 0
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return float("nan")
    return float(silhouette_score(emb[mask], labels[mask]))


def collect(data_dir: Path) -> List[Dict]:
    rows = []
    dirs = sorted(data_dir.iterdir())
    print(f"Scanning {len(dirs)} directories...", flush=True)
    for d in dirs:
        if not d.is_dir():
            continue
        m = _SLUG_RE.match(d.name)
        if not m:
            continue
        emb_path = d / "embeddings_reduced.npy"
        lbl_path = d / "cluster_labels.npy"
        if not emb_path.exists() or not lbl_path.exists():
            continue
        task, rep, w, s, K = m["task"], m["rep"], int(m["w"]), int(m["s"]), int(m["K"])
        try:
            emb = np.load(emb_path)
            lbl = np.load(lbl_path).astype(np.int64)
        except Exception as e:
            print(f"  skip {d.name}: {e}", flush=True)
            continue
        sil = compute_silhouette(emb, lbl)
        print(f"  {d.name}: sil={sil:.4f}", flush=True)
        rows.append(dict(task=task, rep=rep, w=w, s=s, K=K, silhouette=sil))
    return rows


def plot(rows: List[Dict], out: Path) -> None:
    # Group by task
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    tasks = sorted(by_task.keys())
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), squeeze=False)
    fig.patch.set_facecolor("#0f172a")

    for ax_col, task in enumerate(tasks):
        ax = axes[0][ax_col]
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        task_rows = by_task[task]
        # Group by (rep, w, s)
        by_setting: Dict[Tuple, List[Dict]] = defaultdict(list)
        for r in task_rows:
            by_setting[(r["rep"], r["w"], r["s"])].append(r)

        rep_ws_pairs = sorted(by_setting.keys())
        policy_ws = sorted({(w, s) for rep, w, s in rep_ws_pairs if rep == "policy_emb"})
        infembed_ws = sorted({(w, s) for rep, w, s in rep_ws_pairs if rep == "infembed"})

        color_map: Dict[Tuple, str] = {}
        for i, ws in enumerate(policy_ws):
            color_map[("policy_emb", *ws)] = _COLORS_POLICY[i % len(_COLORS_POLICY)]
        for i, ws in enumerate(infembed_ws):
            color_map[("infembed", *ws)] = _COLORS_INFEMBED[i % len(_COLORS_INFEMBED)]

        ls_map = {1: "-", 2: "--", 3: ":"}
        marker_map = {"policy_emb": "o", "infembed": "D"}

        for setting in rep_ws_pairs:
            rep, w, s = setting
            pts = sorted(by_setting[setting], key=lambda r: r["K"])
            Ks = [r["K"] for r in pts]
            sils = [r["silhouette"] for r in pts]
            color = color_map.get(setting, "#94a3b8")
            ls = ls_map.get(w, "-")
            marker = marker_map.get(rep, "o")
            label = f"{rep}, w={w}, s={s}"
            ax.plot(Ks, sils, color=color, linestyle=ls, marker=marker,
                    markersize=6, linewidth=1.8, label=label, alpha=0.9)

        ax.set_title(task, fontsize=11, pad=8)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Silhouette score", fontsize=10)
        ax.grid(True, color="#334155", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=7, framealpha=0.3, facecolor="#1e293b",
                  labelcolor="#cbd5e1", loc="upper right")

    fig.suptitle("Silhouette score vs K — all (rep, w, s) settings",
                 color="#f1f5f9", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path,
                        default=Path("/Users/erik/stanford/asl_rotation/data/graph_simplification/clusterings"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/silhouette_sweep.png"))
    parser.add_argument("--sample_size", type=int, default=2000,
                        help="Max samples per clustering for silhouette (0 = all)")
    args = parser.parse_args()

    rows = collect(args.data_dir)
    if not rows:
        print("No clusterings found.", file=sys.stderr)
        sys.exit(1)
    plot(rows, args.out)


if __name__ == "__main__":
    main()

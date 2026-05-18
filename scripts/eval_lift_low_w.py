"""Evaluate MV + coverage on the lift low-w clusterings.

Standalone, in-process, no orchestrator. Just walks the clustering dirs
matching the lift_mh_jan26__{rep}__w{w}_s1__K{K} pattern and computes the
same MV/coverage metrics the main K-sweep uses, then prints a comparison
table.
"""
from __future__ import annotations
import json
import pathlib
import re
import sys

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.behaviors.behavior_graph import BehaviorGraph
from policy_doctor.behaviors.simplification.metrics import (
    bootstrap_mv_ci,
    markov_violation_coverage,
)

ROOT = pathlib.Path("/mnt/ssdB/erik/cupid_data/graph_simplification/clusterings")
OUT = pathlib.Path("/mnt/ssdB/erik/cupid_data/graph_simplification/results/k_sweep")
SLUG = re.compile(r"^(?P<task>(lift_mh_jan26|square_mh_feb5|transport_mh_jan28))__(?P<rep>[a-z_]+)__w(?P<w>\d+)_s(?P<s>\d+)__K(?P<K>\d+)$")

for d in sorted(ROOT.iterdir()):
    m = SLUG.match(d.name)
    if not m:
        continue
    task = m["task"]
    w, s, K = int(m["w"]), int(m["s"]), int(m["K"])
    rep = m["rep"]
    if w not in (1, 2):
        continue  # only new low-w
    out = OUT / f"{task}__{rep}__w{w}_s{s}__K{K}.json"
    if out.exists():
        continue
    labels = np.load(d / "cluster_labels.npy").astype(np.int64)
    meta = json.loads((d / "metadata.json").read_text())
    manifest = yaml.safe_load((d / "manifest.yaml").read_text()) or {}
    level = manifest.get("level", "rollout")
    n_eps = len({m_["rollout_idx"] for m_ in meta})
    g = BehaviorGraph.from_cluster_assignments(labels, meta, level=level)
    row = {
        "task": task, "rep": rep, "w": w, "s": s, "K": K,
        "n_samples": int(len(labels)), "n_episodes": int(n_eps),
        "n_cluster_nodes": len(g.cluster_nodes),
        "level": level, "n_bootstrap": 100,
    }
    for order in (1, 2, 3):
        p, lo, hi = bootstrap_mv_ci(
            labels, meta, node_mapping={}, level=level, order=order,
            n_bootstrap=100, current_labels=labels, rng_seed=42 + order,
        )
        cov = markov_violation_coverage(
            labels, meta, node_mapping={}, level=level, order=order,
            current_labels=labels,
        )
        row[f"mv{order}_point"] = float(p)
        row[f"mv{order}_ci_lo"] = float(lo)
        row[f"mv{order}_ci_hi"] = float(hi)
        row[f"mv{order}_coverage_fraction"] = cov["coverage_fraction"]
        row[f"mv{order}_n_states_passing"] = cov["n_states_passing"]
        row[f"mv{order}_n_states_total"] = cov["n_states_total"]
    out.write_text(json.dumps(row, indent=2))
    print(f"  {d.name}: MV1={row['mv1_point']:.3f} cov={row['mv1_coverage_fraction']:.2f} samples={row['n_samples']}")
print("done")

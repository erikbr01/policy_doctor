"""Per-cluster confusion-structure analysis for an E1 metrics.json.

For each true cluster, classify the prediction structure as one of:
  - perfect          : all queries correct
  - mostly_correct   : > 50% correct
  - concentrated     : 0% correct, with >= 2/3 of misses landing on a single
                       other cluster (suggests over-clustering / near-duplicate
                       of that cluster)
  - diffuse          : 0% correct, misses spread across >= 3 different clusters
                       (suggests the cluster lacks a coherent visual identity
                       OR resembles many clusters equally)
  - mixed            : partial accuracy with no dominant confusion target

N.B. with n_query=3 per cluster the labels are noisy; this is a coarse
prioritization, not a final diagnosis.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        default="experiments/e1_transport_r512_seed0_qwen3vl8b_K20/metrics.json",
    )
    ap.add_argument(
        "--sample_plan",
        default=None,
        help="Optional sample_plan.json (sibling of metrics.json by default). "
             "If present, accuracy is also broken down by disjointness_status.",
    )
    args = ap.parse_args()

    metrics_path = pathlib.Path(args.metrics)
    m = json.loads(metrics_path.read_text())
    cm = m["confusion_matrix"]
    cluster_ids = m["confusion_matrix_cluster_ids"]
    K = len(cluster_ids)

    sp_path = (
        pathlib.Path(args.sample_plan)
        if args.sample_plan
        else metrics_path.parent / "sample_plan.json"
    )
    disjointness: dict = {}
    if sp_path.exists():
        plan = json.loads(sp_path.read_text())
        disjointness = {
            int(cid): plan["clusters"][cid].get("disjointness_status", "unknown")
            for cid in plan["clusters"]
        }

    rows = []
    for i, true_cid in enumerate(cluster_ids):
        row = cm[i]
        n_total = sum(row)
        n_correct = row[i]
        misses = {cluster_ids[j]: row[j] for j in range(K) if j != i and row[j] > 0}
        n_miss = sum(misses.values())
        acc = n_correct / n_total if n_total else 0.0

        if n_correct == n_total and n_total > 0:
            kind = "perfect"
        elif acc > 0.5:
            kind = "mostly_correct"
        elif n_correct == 0 and n_miss >= 2:
            top_target, top_count = max(misses.items(), key=lambda kv: kv[1])
            if top_count / n_miss >= 2 / 3:
                kind = "concentrated"
            elif len(misses) >= 3:
                kind = "diffuse"
            else:
                kind = "mixed"
        else:
            kind = "mixed"

        if misses:
            top_target, top_count = max(misses.items(), key=lambda kv: kv[1])
            top_str = f"{top_target}:{top_count}"
        else:
            top_str = "-"

        rows.append((true_cid, kind, n_correct, n_total, len(misses), top_str, misses))

    # Sort by kind for readability
    order = {"perfect": 0, "mostly_correct": 1, "mixed": 2, "concentrated": 3, "diffuse": 4}
    rows.sort(key=lambda r: (order.get(r[1], 9), r[0]))

    print(f"K = {K}, n_query/cluster = {rows[0][3]} (typical)\n")
    dj_col = "  disjoint" if disjointness else ""
    print(f"{'cid':>4}  {'kind':14s}  {'corr':>6}  {'#tgts':>5}  {'top miss':>10}{dj_col}  misses")
    print("-" * (70 + (12 if disjointness else 0)))
    kind_counts: Counter = Counter()
    for cid, kind, c, n, ntargets, top, misses in rows:
        miss_str = ", ".join(f"{k}:{v}" for k, v in sorted(misses.items()))
        dj = f"  {disjointness.get(int(cid), '?'):>9}" if disjointness else ""
        print(f"{cid:>4}  {kind:14s}  {c}/{n:<3}     {ntargets:>5}  {top:>10}{dj}  {miss_str}")
        kind_counts[kind] += 1

    print()
    print("Kind summary:")
    for k in ("perfect", "mostly_correct", "mixed", "concentrated", "diffuse"):
        if kind_counts[k]:
            print(f"  {k}: {kind_counts[k]}")

    # Confusion symmetry: is cluster A often confused with B, AND is B often confused with A?
    print("\nReciprocal-confusion pairs (A→B and B→A both > 0):")
    pairs_found = []
    for i, true_a in enumerate(cluster_ids):
        for j, true_b in enumerate(cluster_ids):
            if j <= i:
                continue
            ab = cm[i][j]
            ba = cm[j][i]
            if ab > 0 and ba > 0:
                pairs_found.append((true_a, true_b, ab, ba))
    if pairs_found:
        for a, b, ab, ba in pairs_found:
            print(f"  {a} ↔ {b}: A→B={ab}, B→A={ba}")
    else:
        print("  (none — no symmetric confusions at this sample size)")

    if disjointness:
        print("\nPer-cluster disjointness-stratified accuracy:")
        from collections import defaultdict
        bucket_correct: dict = defaultdict(int)
        bucket_total: dict = defaultdict(int)
        for i, true_cid in enumerate(cluster_ids):
            row = cm[i]
            n_total = sum(row)
            n_correct = row[i]
            dj = disjointness.get(int(true_cid), "unknown")
            bucket_correct[dj] += n_correct
            bucket_total[dj] += n_total
        for dj in sorted(bucket_total, key=lambda d: -bucket_total[d]):
            n = bucket_total[dj]
            c = bucket_correct[dj]
            acc = c / n if n else 0
            print(f"  {dj:>14}: {c:>3}/{n:<3} = {acc:.3f}")
        keep_correct = sum(c for dj, c in bucket_correct.items() if dj == "full")
        keep_total = sum(t for dj, t in bucket_total.items() if dj == "full")
        if keep_total:
            print(
                f"\n  Aggregate accuracy on disjointness=full only: "
                f"{keep_correct}/{keep_total} = {keep_correct / keep_total:.3f} "
                f"(chance = {1.0 / K:.3f})"
            )

    # Per-query origin bucketing (requires sample_plan with query_origins +
    # predictions.jsonl with is_correct). This is the per-query global-disjoint
    # bucketing — the precise version of the per-cluster summary above.
    pred_path = metrics_path.parent / "predictions.jsonl"
    if sp_path.exists() and pred_path.exists():
        plan = json.loads(sp_path.read_text())
        any_origin = any(
            "query_origins" in c for c in plan.get("clusters", {}).values()
        )
        if any_origin:
            print("\nPer-query origin-stratified accuracy (requires global_episode_disjoint plan):")
            origin_for: dict = {}
            for cid, c in plan["clusters"].items():
                origins = c.get("query_origins") or []
                for q_idx, origin in zip(c.get("query_indices", []), origins):
                    origin_for[(int(cid), int(q_idx))] = origin

            preds = [json.loads(l) for l in open(pred_path)]
            obc: Counter = Counter()
            obt: Counter = Counter()
            for p in preds:
                key = (int(p["true_cluster_id"]), int(p["query_idx"]))
                origin = origin_for.get(key, "unknown")
                obt[origin] += 1
                if p.get("is_correct"):
                    obc[origin] += 1
            for origin in sorted(obt, key=lambda o: -obt[o]):
                n = obt[origin]
                c = obc[origin]
                acc = c / n if n else 0
                print(f"  {origin:>20}: {c:>3}/{n:<3} = {acc:.3f}")
            tier1 = "tier1_global"
            if obt[tier1]:
                from scipy.stats import binomtest
                r = binomtest(obc[tier1], obt[tier1], p=1.0 / K, alternative="greater")
                ci = r.proportion_ci(method="wilson")
                print(
                    f"\n  Aggregate on tier1_global only: "
                    f"{obc[tier1]}/{obt[tier1]} = {obc[tier1] / obt[tier1]:.3f}  "
                    f"(chance={1.0 / K:.3f}, p={r.pvalue:.3e}, 95% CI=[{ci.low:.3f}, {ci.high:.3f}])"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build K-sweep clustering directories by refitting KMeans at multiple K
on the existing K=20 UMAP embedding.

Rationale: the slow non-deterministic step is UMAP. Holding UMAP fixed and
only varying K isolates the effect of K on E1 accuracy — an apples-to-apples
comparison the original pipeline doesn't naturally produce (each K=N run
re-fits UMAP from scratch).

Outputs to ``/mnt/ssdB/erik/cupid_data/clusterings/transport_mh_seed0_r512_clustering_k{K}/`` for each K:
  manifest.yaml, cluster_labels.npy, metadata.json,
  embeddings_reduced.npy (same as source), clustering_models.pkl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/ssdB/erik/cupid_data/clusterings/transport_mh_seed0_r512_clustering")
    ap.add_argument("--out_template", default="/mnt/ssdB/erik/cupid_data/clusterings/transport_mh_seed0_r512_clustering_k{K}")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 15])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from policy_doctor.data.clustering_loader import (
        ClusteringModels,
        load_clustering_models,
    )
    import joblib

    src = pathlib.Path(args.src)
    src_models = load_clustering_models(src)
    emb = src_models.reducer.embedding_.astype(np.float32)  # (N, 100)
    print(f"[k_sweep] source embedding: {emb.shape}")

    src_metadata = json.loads((src / "metadata.json").read_text())
    src_manifest = yaml.safe_load((src / "manifest.yaml").read_text())

    for K in args.ks:
        out = pathlib.Path(args.out_template.format(K=K))
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[k_sweep] K={K} → {out}")

        km = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
        labels = km.fit_predict(emb).astype(np.int32)
        print(f"  cluster sizes: {sorted(np.bincount(labels).tolist(), reverse=True)}")

        np.save(out / "cluster_labels.npy", labels)
        np.save(out / "embeddings_reduced.npy", emb)
        (out / "metadata.json").write_text(json.dumps(src_metadata))

        new_manifest = dict(src_manifest)
        new_manifest["n_clusters"] = K
        new_manifest["created"] = "k_sweep_from_K20"
        new_manifest["k_sweep_source"] = str(src)
        new_manifest["k_sweep_seed"] = int(args.seed)
        (out / "manifest.yaml").write_text(yaml.safe_dump(new_manifest))

        new_models = ClusteringModels(
            normalizer=src_models.normalizer,
            normalizer_method=src_models.normalizer_method,
            prescaler=src_models.prescaler,
            prescaler_method=src_models.prescaler_method,
            reducer=src_models.reducer,
            reducer_method=src_models.reducer_method,
            kmeans=km,
        )
        joblib.dump(new_models, out / "clustering_models.pkl")
        print(f"  wrote {out}/{{cluster_labels.npy, embeddings_reduced.npy, manifest.yaml, metadata.json, clustering_models.pkl}}")

    print("\n[k_sweep] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

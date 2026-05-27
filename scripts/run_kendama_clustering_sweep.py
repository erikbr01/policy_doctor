"""Efficient W × S × K sweep for Kendama clusterings.

Extracts per-timestep features once (state or policy_emb), then for each
(W, S) pair runs UMAP once and reuses the 2D coords for every K. Much faster
than spawning one full pipeline subprocess per combo.

For policy_emb the expensive GPU forwards happen exactly once. State trunks
build in seconds.

Usage:

    python scripts/run_kendama_clustering_sweep.py \\
        --spec sweep_specs/kendama_baseline250_policy_emb.yaml

Or reuse ``run_clustering_sweep.py`` after building the trunk:

    python scripts/build_kendama_clustering.py --timestep_embed_only ... \\
        --out_dir data/demo_sweep/kendama_baseline250/_trunks/policy_emb

    python scripts/run_clustering_sweep.py \\
        --spec sweep_specs/kendama_baseline250_policy_emb.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.kendama_clustering_lib import (  # noqa: E402
    aggregate_windows,
    build_manifest,
    cluster_coords,
    extract_policy_timesteps,
    extract_state_action_timesteps,
    extract_state_timesteps,
    fit_umap_coords,
    load_episodes,
    load_timestep_cache,
    save_clustering_dir,
    save_timestep_cache,
    slug_for_combo,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def _ensure_trunk(spec: dict[str, Any], sweep_root: Path, *, force: bool) -> Path:
    representation = spec.get("representation", "policy_emb")
    trunk_dir = sweep_root / "_trunks" / representation
    cache_path = trunk_dir / "timestep_cache.npz"
    manifest_path = trunk_dir / "embed_manifest.yaml"

    if cache_path.exists() and manifest_path.exists() and not force:
        print(f"[sweep] reusing trunk {trunk_dir}")
        return trunk_dir

    rollouts_dir = Path(spec["rollouts"])
    trunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] building trunk {representation} from {rollouts_dir}")

    if representation == "state":
        episodes = load_episodes(rollouts_dir, mode="state")
        per_ts, successes = extract_state_timesteps(episodes)
        meta = dict(representation="state", rollouts=str(rollouts_dir))
    elif representation == "state_action":
        episodes = load_episodes(rollouts_dir, mode="state_action")
        per_ts, successes = extract_state_action_timesteps(episodes)
        meta = dict(representation="state_action", rollouts=str(rollouts_dir))
    else:
        import torch

        ckpt = Path(spec["ckpt"])
        layer = spec.get("layer", "bottleneck_plan_t0")
        episodes = load_episodes(rollouts_dir, mode="policy")
        per_ts, successes = extract_policy_timesteps(
            episodes,
            ckpt_path=ckpt,
            layer=layer,
            batch_size=int(spec.get("batch_size", 128)),
            device=torch.device(spec.get("device", "cuda:0")),
        )
        meta = dict(
            representation="policy_emb",
            rollouts=str(rollouts_dir),
            ckpt=str(ckpt),
            layer=layer,
        )

    save_timestep_cache(cache_path, per_ts=per_ts, successes=successes,
                        representation=representation, meta=meta)
    manifest = dict(
        timestep_embed_only=True,
        n_episodes=len(per_ts),
        n_timesteps=int(sum(len(e) for e in per_ts)),
        **meta,
    )
    manifest_path.write_text(yaml.dump(manifest, sort_keys=False))
    print(f"[sweep] trunk ready: {len(per_ts)} episodes, "
          f"{manifest['n_timesteps']} timesteps")
    return trunk_dir


def run_sweep(*, spec_path: Path, force: bool = False, dry_run: bool = False) -> int:
    spec = _load_spec(spec_path)

    sweep_root = Path(spec["sweep_root"])
    sweep_root.mkdir(parents=True, exist_ok=True)
    seed = int(spec.get("seed", 42))
    task_config = spec.get("task_config", "kendama_baseline250")
    representation = spec.get("representation", "policy_emb")
    grid = spec["grid"]

    w_values = list(grid["window_width"])
    s_values = list(grid["stride"])
    k_values = list(grid["n_clusters"])
    ws_pairs = [(w, s) for w, s in itertools.product(w_values, s_values)]
    combos = [(w, s, k) for w, s, k in itertools.product(w_values, s_values, k_values)]

    print(f"[sweep] {len(combos)} combos ({len(ws_pairs)} W,S × {len(k_values)} K) → {sweep_root}")

    if dry_run:
        trunk_dir = sweep_root / "_trunks" / representation
        print(f"[sweep] dry run — trunk would be {trunk_dir}")
        for w, s, k in combos:
            layer = spec.get("layer", "bottleneck_plan_t0")
            slug = slug_for_combo(
                representation=representation,
                window=w, stride=s, n_clusters=k, seed=seed, layer=layer,
            )
            print(f"  {slug}")
        return 0

    t0 = time.time()
    trunk_dir = _ensure_trunk(spec, sweep_root, force=force)
    per_ts, successes, cache_meta = load_timestep_cache(trunk_dir / "timestep_cache.npz")
    layer = cache_meta.get("layer")

    umap_cache: dict[tuple[int, int], tuple] = {}
    n_skip = n_run = n_fail = 0

    for w, s in ws_pairs:
        features, metadata = aggregate_windows(
            per_ts, successes, window=w, stride=s, representation=representation,
        )
        if len(features) == 0:
            print(f"[sweep] SKIP W={w} S={s}: no windows")
            for k in k_values:
                n_fail += 1
            continue
        coords = fit_umap_coords(features, seed=seed)
        umap_cache[(w, s)] = (coords, metadata, len(features))
        print(f"[sweep] UMAP W={w} S={s}: {len(features)} windows → 2D")

    for w, s, k in combos:
        layer = layer or spec.get("layer", "bottleneck_plan_t0")
        slug = slug_for_combo(
            representation=representation,
            window=w, stride=s, n_clusters=k, seed=seed, layer=layer,
        )
        out_dir = sweep_root / slug
        manifest_path = out_dir / "manifest.yaml"
        if manifest_path.exists() and not force:
            n_skip += 1
            continue
        cached = umap_cache.get((w, s))
        if cached is None:
            n_fail += 1
            continue
        coords, metadata, n_samples = cached
        try:
            labels = cluster_coords(coords, n_clusters=k, seed=seed)
            manifest = build_manifest(
                representation=representation,
                n_clusters=k,
                n_samples=n_samples,
                window=w,
                stride=s,
                seed=seed,
                task_config=task_config,
                rollouts=str(spec["rollouts"]),
                ckpt=spec.get("ckpt"),
                layer=layer,
            )
            save_clustering_dir(out_dir, labels=labels, coords=coords,
                                metadata=metadata, manifest=manifest)
            n_run += 1
        except Exception as exc:
            print(f"[sweep] FAILED {slug}: {exc}")
            n_fail += 1

    print(
        f"\n[sweep] done in {time.time()-t0:.0f}s  "
        f"run={n_run} skip={n_skip} fail={n_fail}  "
        f"({len(umap_cache)} UMAP runs for {len(combos)} clusterings)"
    )
    return 1 if n_fail else 0


def main() -> int:
    args = _parse_args()
    return run_sweep(spec_path=Path(args.spec), force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

"""Cluster kendama rollouts by policy-bottleneck embeddings.

Companion to `cluster_kendama_rollouts.py` (state-based). This one loads
the actual policy checkpoint and extracts `bottleneck_plan_t0` embeddings
per rollout timestep — the activation at the UNet bottleneck when the
policy is asked to denoise the recorded action chunk at noise step t=0
(near-clean). Embeddings are then sliding-window aggregated, UMAP'd to
2D, and KMeans-clustered. Output goes to the standard demo_sweep path
so the graph-demo app picks it up automatically.

Policy: DiffusionUnetHybridImagePolicy (image obs encoder + 1D UNet),
loaded directly from a bare .ckpt via dill. Rollouts: DROID-style
trajectory.hdf5 + meta.json per episode (joint_positions, gripper,
two cameras at 256x256 RGB, executed actions).

Usage (policy_doctor env, GPU):
    conda activate policy_doctor
    python scripts/cluster_kendama_policy_embeddings.py \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final \\
        --ckpt     /mnt/ssdB/erik/rollouts/baseline_250_demos.ckpt \\
        --layer    bottleneck_plan_t0 \\
        --window 20 --stride 10 -K 8 \\
        --batch_size 32 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
# diffusion_policy / dill-pickled checkpoints expect cupid on sys.path
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "cupid"))


# ── Layer parsing (subset of compute_policy_embeddings.py) ────────────────
def _hook_module(model, hook: str):
    if hook == "bottleneck":
        return model.mid_modules[-1]
    if hook == "decoder":
        return model.up_modules[-1][1]
    if hook == "encoder":
        return model.down_modules[-1][1]
    raise ValueError(f"Unknown hook: {hook}")


def _parse_layer(layer: str) -> tuple[str, str | None, int | None]:
    """Parse 'bottleneck_plan_t0' → (hook, action_type, t_single)."""
    import re

    m = re.match(
        r"^(?P<hook>bottleneck|decoder|encoder)"
        r"(?:_(?P<action>plan8|plan|exec))?"
        r"(?:_t(?P<t>\d+))?$",
        layer,
    )
    if m is None:
        raise ValueError(f"Bad layer name: {layer!r}")
    return (
        m.group("hook"),
        m.group("action"),
        None if m.group("t") is None else int(m.group("t")),
    )


# ── Rollout loading ───────────────────────────────────────────────────────
def _load_episode(ep_dir: Path) -> dict | None:
    """Read one DROID-style episode. Returns None if malformed."""
    hdf5 = ep_dir / "trajectory.hdf5"
    meta_path = ep_dir / "meta.json"
    if not hdf5.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    # Some episodes have `success: null` (run aborted before label assigned).
    raw_success = meta.get("success")
    with h5py.File(hdf5, "r") as f:
        g = f["data/demo_0"]
        return dict(
            ep_dir=ep_dir,
            success=bool(raw_success) and float(raw_success) > 0,
            n_steps=int(meta.get("n_steps", g["actions"].shape[0])),
            # Pull bytes into RAM. Each episode ~0.5 GB across both cameras
            # (uint8); we release after each episode's batch is processed.
            joint_positions=g["obs/joint_positions"][:].astype(np.float32),
            gripper_position=g["obs/gripper_position"][:].astype(np.float32),
            hand_image=g["obs/hand_camera_image"][:],          # uint8 HWC
            exterior_image=g["obs/exterior_image_1_left"][:],  # uint8 HWC
            actions=g["data/demo_0/actions"][:].astype(np.float32)
                if "data/demo_0/actions" in g
                else g["actions"][:].astype(np.float32),
        )


def _build_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    """For each timestep t, take actions[t:t+horizon]; pad-by-repeat at end.

    Returns shape (T, horizon, action_dim).
    """
    T, D = actions.shape
    out = np.empty((T, horizon, D), dtype=np.float32)
    for t in range(T):
        end = min(t + horizon, T)
        chunk = actions[t:end]
        if len(chunk) < horizon:
            pad = np.tile(chunk[-1:], (horizon - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        out[t] = chunk
    return out


def _images_to_chw01(arr_uint8: np.ndarray) -> np.ndarray:
    """(T, H, W, C) uint8 → (T, C, H, W) float32 in [0, 1]."""
    return (arr_uint8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)


# ── Embedding extraction ──────────────────────────────────────────────────
def _extract_episode_embeddings(
    policy,
    episode: dict,
    *,
    hook: str,
    action_type: str | None,
    t_single: int | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Per-timestep embeddings for one episode. Returns (T, D)."""
    T = episode["n_steps"]
    horizon = policy.horizon
    To = policy.n_obs_steps  # = 1 for the kendama policy

    # Pre-stage obs arrays as float32 in policy-expected layout.
    hand_chw = _images_to_chw01(episode["hand_image"][:T])             # (T, 3, H, W)
    ext_chw = _images_to_chw01(episode["exterior_image"][:T])
    jp = episode["joint_positions"][:T]                                # (T, 7)
    gp = episode["gripper_position"][:T]                               # (T, 1)
    action_chunks = _build_action_chunks(
        episode["actions"][:T], horizon
    )                                                                  # (T, H, A)

    # Hook the bottleneck. The hook runs once per forward; we capture the
    # output and immediately mean-pool across the temporal axis (UNet 1D
    # output is (B, D, H_t)).
    captured: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        captured.append(feat.detach().float().mean(dim=-1).cpu())

    handle = _hook_module(policy.model, hook).register_forward_hook(_hook)
    scheduler = policy.noise_scheduler
    normalizer = policy.normalizer

    dtype = torch.float32
    embs: list[np.ndarray] = []
    try:
        for b0 in range(0, T, batch_size):
            b1 = min(b0 + batch_size, T)
            B = b1 - b0

            # Build obs dict; the hybrid image policy expects
            # (B, To, *shape). We have To=1.
            obs_dict = {
                "hand_camera_image": torch.from_numpy(hand_chw[b0:b1]).to(device=device, dtype=dtype)[:, None, ...],
                "exterior_image_1_left": torch.from_numpy(ext_chw[b0:b1]).to(device=device, dtype=dtype)[:, None, ...],
                "joint_positions": torch.from_numpy(jp[b0:b1]).to(device=device, dtype=dtype)[:, None, :],
                "gripper_position": torch.from_numpy(gp[b0:b1]).to(device=device, dtype=dtype)[:, None, :],
            }

            with torch.no_grad():
                nobs = normalizer.normalize(obs_dict)
                # (B, To, …) → (B*To, …) → encode → (B, To*Do); since To=1
                # this is effectively a per-frame encode.
                this_nobs = {
                    k: v[:, :To, ...].reshape(-1, *v.shape[2:])
                    for k, v in nobs.items()
                }
                nobs_features = policy.obs_encoder(this_nobs)
                global_cond = nobs_features.reshape(B, -1)

                # Normalize actions the same way the training loop would,
                # so the UNet sees them in the distribution it was trained on.
                actions_t = torch.from_numpy(action_chunks[b0:b1]).to(device=device)  # fp32
                nactions = normalizer["action"].normalize(actions_t)  # (B, H, A)

                if action_type == "plan":
                    clean = nactions
                elif action_type == "exec":
                    clean = nactions[:, 0:1, :].expand(-1, horizon, -1).contiguous()
                elif action_type == "plan8":
                    pad = torch.zeros_like(nactions)
                    pad[:, :8] = nactions[:, :8]
                    clean = pad
                else:
                    clean = None

                # Use the policy's own scheduler (DDPM, squaredcos) to add
                # noise — re-deriving the schedule risks mismatch with the
                # training timesteps the obs encoder + UNet expect.
                if t_single is not None:
                    timesteps = torch.full(
                        (B,), t_single, dtype=torch.long, device=device
                    )
                else:
                    # Average over n_noise_levels timesteps (rarely used here
                    # since the layer name pins t=0).
                    raise NotImplementedError("Only single-timestep mode is supported.")

                noise = torch.randn(
                    B, horizon, policy.action_dim,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(b0),
                )
                if clean is not None:
                    noisy = scheduler.add_noise(clean, noise, timesteps)
                else:
                    # Pure noise — match the diffusers convention by scaling
                    # std(noise) to match the schedule at this step.
                    noisy = scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)

                # Forward through the UNet. The hook fires here.
                policy.model(noisy, timesteps, global_cond=global_cond)
                emb = captured.pop(0).numpy().astype(np.float32)  # (B, D)
                embs.append(emb)
    finally:
        handle.remove()

    return np.concatenate(embs, axis=0)  # (T, D)


# ── Window aggregation ───────────────────────────────────────────────────
def _run_one(
    per_ts_embs: list[np.ndarray],
    successes: list[bool],
    W: int,
    S: int,
    *,
    args,
    ckpt_path: Path,
    rollouts_dir: Path,
    summary: list[dict],
) -> None:
    """One (W, S) iteration: aggregate, UMAP, pick K, save.

    Silhouette is scored on the 2D UMAP coords to match what the demo app's
    sweep_analysis page surfaces (compute_clustering_metrics.py loads
    `embeddings_reduced.npy` and silhouette-scores in that space).
    """
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    print(f"\n── W={W}, S={S} ──────────────────────────────")
    X, meta = _aggregate_windows(per_ts_embs, successes, W, S)
    if len(X) < args.k_max + 1:
        print(f"  too few windows ({len(X)}) for K up to {args.k_max}, skipping")
        return
    print(f"  {len(X)} windows, {X.shape[1]}-dim features")

    X_scaled = StandardScaler().fit_transform(X)
    print("  Running UMAP to 2D …")
    coords = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        random_state=args.seed, verbose=False,
    ).fit_transform(X_scaled).astype(np.float32)

    # Pick K. auto_k_kmeans internally silhouette-scores in the space
    # it KMeans'd on — we want that to be the 2D UMAP coords so the
    # reported score matches the demo app.
    if args.auto_k or args.sweep:
        from policy_doctor.behaviors.graph_simplification import auto_k_kmeans

        print(f"  silhouette-K sweep K ∈ [{args.k_min}, {args.k_max}] …")
        best_labels, best_k, scores = auto_k_kmeans(
            coords, k_range=(args.k_min, args.k_max), random_state=args.seed,
        )
        labels = best_labels.astype(np.int32)
        chosen_k = best_k
        sil = float(scores[best_k])
        silhouette_scores = scores
        print(f"    best K = {best_k}  silhouette(2D) = {sil:.4f}")
    else:
        from sklearn.cluster import KMeans

        chosen_k = args.n_clusters
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=chosen_k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(coords).astype(np.int32)
        sil = float(silhouette_score(coords, labels)) if chosen_k > 1 else float("nan")
        silhouette_scores = None

    sizes = np.bincount(labels).tolist()
    print(f"  sizes={sizes}")

    out_dir = _REPO_ROOT / args.out_root / (
        f"policy_emb_{args.layer}_w{W}_s{S}_seed{args.seed}_kmeans_k{chosen_k}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cluster_labels.npy", labels)
    np.save(out_dir / "embeddings_reduced.npy", coords)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    manifest = dict(
        algorithm="kmeans",
        scaling="none",
        umap_prescale="standard",
        influence_source="policy_emb",
        representation="sliding_window",
        slice_representation="policy_emb",
        rep_kwargs=dict(layer=args.layer),
        level="rollout",
        n_clusters=chosen_k,
        n_samples=len(X),
        auto_k=args.auto_k or args.sweep,
        silhouette_scores=silhouette_scores,
        silhouette_2d=round(sil, 4),
        window_width=W,
        stride=S,
        aggregation="mean",
        task_config="kendama_may22",
        seed=args.seed,
        ckpt=str(ckpt_path),
        rollouts=str(rollouts_dir),
        pipeline_steps=["embed", "window", "umap", "kmeans"],
    )
    (out_dir / "manifest.yaml").write_text(yaml.dump(manifest, sort_keys=False))

    from policy_doctor.curation_pipeline.steps.compute_clustering_metrics import (
        _compute_for_dir,
    )

    metrics = _compute_for_dir(out_dir)
    if metrics is not None:
        sil_m = metrics.get("silhouette_mean")
        sil_m_s = f"{sil_m:.4f}" if sil_m is not None else "n/a"
        print(f"  metrics.json written (silhouette_mean={sil_m_s})")

    summary.append(dict(w=W, s=S, k=chosen_k, silhouette=sil, sizes=sizes, path=out_dir))
    print(f"  → {out_dir.name}")


def _aggregate_windows(
    per_ts_embs: list[np.ndarray],  # one (T_ep, D) per episode
    successes: list[bool],
    window: int,
    stride: int,
) -> tuple[np.ndarray, list[dict]]:
    rows, meta = [], []
    for ep_idx, (emb, success) in enumerate(zip(per_ts_embs, successes)):
        T = len(emb)
        t = 0
        while t + window <= T:
            rows.append(emb[t:t + window].mean(axis=0))
            meta.append(
                dict(
                    rollout_idx=ep_idx,
                    window_start=t,
                    window_end=t + window,
                    window_width=window,
                    success=success,
                )
            )
            t += stride
    return np.stack(rows, axis=0).astype(np.float32), meta


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--rollouts",
        default="/mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final",
    )
    ap.add_argument(
        "--ckpt",
        default="/mnt/ssdB/erik/rollouts/baseline_250_demos.ckpt",
    )
    ap.add_argument(
        "--out_root",
        default="data/demo_sweep/kendama_may22/run_clustering/clustering/aggregate_first",
        help="Parent dir under data/demo_sweep/<task>/.../clustering/. The "
             "slug subdir is derived from layer + W + S + K.",
    )
    ap.add_argument("--layer", default="bottleneck_plan_t0")
    ap.add_argument("-K", "--n_clusters", type=int, default=8,
                    help="Fixed K. Ignored if --auto_k is set.")
    ap.add_argument("--auto_k", action="store_true",
                    help="Pick K by silhouette over [k_min, k_max] using "
                         "policy_doctor.behaviors.graph_simplification.auto_k_kmeans.")
    ap.add_argument("--k_min", type=int, default=4)
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--limit_episodes",
        type=int,
        default=0,
        help="If > 0, only process the first N episodes (smoke test).",
    )
    ap.add_argument(
        "--shard",
        default=None,
        help="`start:end` slice of the episode list to process (Python "
             "half-open). Lets you run two GPUs in parallel: shard 0:48 on "
             "cuda:0, shard 48:97 on cuda:1, then merge with --merge_only.",
    )
    ap.add_argument(
        "--embed_out",
        default=None,
        help="If set, write per-timestep embeddings + episode list to this "
             ".npz and exit (skip clustering). Use with --shard for parallel "
             "GPU runs.",
    )
    ap.add_argument(
        "--merge_only",
        nargs="+",
        default=None,
        help="Skip embedding. Load one or more --embed_out .npz files, "
             "concatenate in episode-index order, run windowing+UMAP+KMeans.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep (W, S) on top of auto-K. Runs UMAP+silhouette-K for "
             "each W,S in --sweep_w / --sweep_s. Writes every variant.",
    )
    ap.add_argument("--sweep_w", default="5,10,15,20,30,50",
                    help="Comma-separated window widths to sweep.")
    ap.add_argument("--sweep_s", default="1,2,5,10",
                    help="Comma-separated strides to sweep (skipped if S > W).")
    args = ap.parse_args()

    hook, action_type, t_single = _parse_layer(args.layer)
    if t_single is None:
        raise SystemExit("Pick a single-timestep layer (e.g. bottleneck_plan_t0).")

    rollouts_dir = Path(args.rollouts)
    ckpt_path = Path(args.ckpt)
    # The slug needs the final K. With --auto_k we don't know K until after
    # silhouette sweeps, so postpone the mkdir until then.
    device = torch.device(args.device)

    # ── 0. Merge-only path ────────────────────────────────────────────────
    if args.merge_only:
        print(f"Merging {len(args.merge_only)} shard files …")
        per_ts_embs: list[np.ndarray] = []
        successes: list[bool] = []
        ep_indices: list[int] = []
        for shard_path in args.merge_only:
            data = np.load(shard_path, allow_pickle=True)
            shard_idxs = data["episode_indices"].tolist()
            shard_succ = data["successes"].tolist()
            # Embeddings stored as a single concat with offsets per episode.
            offsets = data["offsets"].tolist()
            embs = data["embeddings"]
            for k, ep_idx in enumerate(shard_idxs):
                ep_indices.append(int(ep_idx))
                successes.append(bool(shard_succ[k]))
                per_ts_embs.append(embs[offsets[k]:offsets[k + 1]])
        # Reorder by ep_idx so the rollout_idx in metadata matches scan order.
        order = sorted(range(len(ep_indices)), key=lambda i: ep_indices[i])
        per_ts_embs = [per_ts_embs[i] for i in order]
        successes = [successes[i] for i in order]
        print(f"  {len(per_ts_embs)} episodes merged")
    else:
        # ── 1. Load policy ────────────────────────────────────────────────
        # Avoid the full workspace constructor — it imports lr_scheduler
        # which transitively breaks against current diffusers ('Union'
        # moved). Build the policy directly from cfg.policy + load just
        # the model state_dict.
        print(f"Loading policy from {ckpt_path.name} …")
        import dill
        import hydra

        with open(str(ckpt_path), "rb") as f:
            payload = torch.load(f, pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]
        policy = hydra.utils.instantiate(cfg.policy)
        use_ema = getattr(cfg.training, "use_ema", False)
        sd_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
        policy.load_state_dict(payload["state_dicts"][sd_key])
        policy.to(device)
        policy.eval()
        print(
            f"  horizon={policy.horizon} n_obs_steps={policy.n_obs_steps} "
            f"action_dim={policy.action_dim} hook={hook}"
        )

        # ── 2. Load episodes ──────────────────────────────────────────────
        print(f"Scanning {rollouts_dir} …")
        ep_dirs_all = sorted(d for d in rollouts_dir.iterdir() if d.is_dir())
        if args.limit_episodes:
            ep_dirs_all = ep_dirs_all[: args.limit_episodes]

        if args.shard:
            lo, hi = (int(s) for s in args.shard.split(":"))
            ep_dirs = ep_dirs_all[lo:hi]
            ep_indices_global = list(range(lo, lo + len(ep_dirs)))
            print(f"  shard {lo}:{hi} → {len(ep_dirs)} episodes (of {len(ep_dirs_all)})")
        else:
            ep_dirs = ep_dirs_all
            ep_indices_global = list(range(len(ep_dirs)))

        # ── 3. Per-episode embedding extraction ──────────────────────────
        per_ts_embs = []
        successes = []
        kept_indices: list[int] = []
        n_skipped = 0
        for global_idx, ep_dir in zip(ep_indices_global,
                                      tqdm.tqdm(ep_dirs, desc="Episodes")):
            ep = _load_episode(ep_dir)
            if ep is None:
                n_skipped += 1
                continue
            emb = _extract_episode_embeddings(
                policy,
                ep,
                hook=hook,
                action_type=action_type,
                t_single=t_single,
                batch_size=args.batch_size,
                device=device,
            )
            per_ts_embs.append(emb)
            successes.append(ep["success"])
            kept_indices.append(global_idx)
            del ep

        print(f"  embedded {len(per_ts_embs)} episodes, {n_skipped} skipped")
        if not per_ts_embs:
            raise SystemExit("No usable episodes.")

        # ── 3b. Shard write-out path (no clustering) ─────────────────────
        if args.embed_out:
            offsets = np.cumsum([0] + [len(e) for e in per_ts_embs]).astype(np.int64)
            embeddings = np.concatenate(per_ts_embs, axis=0)
            np.savez_compressed(
                args.embed_out,
                embeddings=embeddings,
                offsets=offsets,
                episode_indices=np.array(kept_indices, dtype=np.int32),
                successes=np.array(successes, dtype=bool),
            )
            print(f"Wrote shard → {args.embed_out}  "
                  f"({embeddings.shape[0]} timesteps, {embeddings.shape[1]}-dim)")
            return

    # ── 4. Resolve sweep grid ────────────────────────────────────────────
    if args.sweep:
        w_grid = [int(x) for x in args.sweep_w.split(",")]
        s_grid = [int(x) for x in args.sweep_s.split(",")]
        ws_pairs = [(w, s) for w in w_grid for s in s_grid if s <= w]
        print(f"Sweep grid: {len(ws_pairs)} (W, S) pairs × "
              f"K ∈ [{args.k_min}, {args.k_max}]")
    else:
        ws_pairs = [(args.window, args.stride)]

    summary: list[dict] = []  # one row per (W, S, K)
    for W, S in ws_pairs:
        _run_one(
            per_ts_embs, successes, W, S,
            args=args,
            ckpt_path=ckpt_path,
            rollouts_dir=rollouts_dir,
            summary=summary,
        )

    # Sweep-wide report sorted by silhouette so the picks are obvious.
    if args.sweep:
        print("\n=== Sweep summary (sorted by silhouette, 2D UMAP) ===")
        print(f"{'W':>4} {'S':>4} {'K':>4} {'silhouette':>12} {'sizes'}")
        for row in sorted(summary, key=lambda r: -r["silhouette"]):
            print(f"{row['w']:>4} {row['s']:>4} {row['k']:>4} "
                  f"{row['silhouette']:>12.4f}  {row['sizes']}")


if __name__ == "__main__":
    main()

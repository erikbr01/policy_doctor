"""Shared Kendama rollout loading, embedding, windowing, and clustering.

Both state-based and policy-embedding clusterings share episode I/O and the
aggregate-first pipeline (window → standardize → UMAP → K-means). Policy
embeddings are extracted once per checkpoint/layer; W/S/K sweeps reuse the
cached per-timestep features.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import h5py
import numpy as np
import yaml

Representation = Literal["state", "state_action", "policy_emb"]


@dataclass(frozen=True)
class KendamaEpisode:
    ep_dir: Path
    success: bool
    n_steps: int
    joint_positions: np.ndarray  # (T, 7) float32
    gripper_position: np.ndarray | None = None  # (T, 1) float32
    cartesian_position: np.ndarray | None = None  # (T, 6) float32
    hand_image: np.ndarray | None = None  # (T, H, W, C) uint8
    exterior_image: np.ndarray | None = None
    actions: np.ndarray | None = None  # (T, A) float32


def load_episodes(
    rollouts_dir: Path,
    *,
    mode: Literal["state", "state_action", "policy"] = "state",
) -> list[KendamaEpisode]:
    """Load DROID-style rollout episodes that have trajectory.hdf5 + meta.json."""
    episodes: list[KendamaEpisode] = []
    for ep_dir in sorted(rollouts_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        hdf5 = ep_dir / "trajectory.hdf5"
        meta_path = ep_dir / "meta.json"
        if not hdf5.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        raw_success = meta.get("success")
        success = bool(raw_success) and float(raw_success) > 0
        with h5py.File(hdf5, "r") as f:
            g = f["data/demo_0"]
            n_steps = int(meta.get("n_steps", g["actions"].shape[0]))
            jp = g["obs/joint_positions"][:].astype(np.float32)
            if mode == "state":
                episodes.append(
                    KendamaEpisode(
                        ep_dir=ep_dir,
                        success=success,
                        n_steps=n_steps,
                        joint_positions=jp,
                    )
                )
                continue
            gp = g["obs/gripper_position"][:].astype(np.float32)
            cart = g["obs/cartesian_position"][:].astype(np.float32)
            actions = g["actions"][:].astype(np.float32)
            if mode == "state_action":
                episodes.append(
                    KendamaEpisode(
                        ep_dir=ep_dir,
                        success=success,
                        n_steps=n_steps,
                        joint_positions=jp,
                        gripper_position=gp,
                        cartesian_position=cart,
                        actions=actions,
                    )
                )
                continue
            episodes.append(
                KendamaEpisode(
                    ep_dir=ep_dir,
                    success=success,
                    n_steps=n_steps,
                    joint_positions=jp,
                    gripper_position=gp,
                    cartesian_position=cart,
                    hand_image=g["obs/hand_camera_image"][:],
                    exterior_image=g["obs/exterior_image_1_left"][:],
                    actions=actions,
                )
            )
    return episodes


def extract_state_timesteps(episodes: list[KendamaEpisode]) -> tuple[list[np.ndarray], list[bool]]:
    """Per-episode joint-position trajectories."""
    return [ep.joint_positions for ep in episodes], [ep.success for ep in episodes]


def extract_state_action_timesteps(
    episodes: list[KendamaEpisode],
) -> tuple[list[np.ndarray], list[bool]]:
    """Per-timestep [joint, gripper, cartesian, executed action] vectors."""
    per_ts: list[np.ndarray] = []
    for ep in episodes:
        feats = np.concatenate(
            [ep.joint_positions, ep.gripper_position, ep.cartesian_position, ep.actions],
            axis=-1,
        ).astype(np.float32)
        per_ts.append(feats)
    return per_ts, [ep.success for ep in episodes]


def _parse_layer(layer: str) -> tuple[str, str | None, int | None]:
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


def _hook_module(model, hook: str):
    if hook == "bottleneck":
        return model.mid_modules[-1]
    if hook == "decoder":
        return model.up_modules[-1][1]
    if hook == "encoder":
        return model.down_modules[-1][1]
    raise ValueError(f"Unknown hook: {hook}")


def _build_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    t, d = actions.shape
    out = np.empty((t, horizon, d), dtype=np.float32)
    for ti in range(t):
        end = min(ti + horizon, t)
        chunk = actions[ti:end]
        if len(chunk) < horizon:
            pad = np.tile(chunk[-1:], (horizon - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        out[ti] = chunk
    return out


def _images_to_chw01(arr_uint8: np.ndarray) -> np.ndarray:
    return (arr_uint8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)


def load_policy_from_ckpt(ckpt_path: Path, device):
    import dill
    import hydra
    import torch

    with open(str(ckpt_path), "rb") as f:
        payload = torch.load(f, pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    use_ema = getattr(cfg.training, "use_ema", False)
    sd_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][sd_key])
    policy.to(device)
    policy.eval()
    return policy


def extract_policy_timesteps(
    episodes: list[KendamaEpisode],
    *,
    ckpt_path: Path,
    layer: str = "bottleneck_plan_t0",
    batch_size: int = 128,
    device,
    progress: bool = True,
) -> tuple[list[np.ndarray], list[bool]]:
    """Batched forward passes; returns one (T, D) array per episode."""
    import torch
    import tqdm

    hook, action_type, t_single = _parse_layer(layer)
    if t_single is None:
        raise ValueError("Pick a single-timestep layer (e.g. bottleneck_plan_t0).")

    policy = load_policy_from_ckpt(ckpt_path, device)
    horizon = policy.horizon
    to = policy.n_obs_steps
    scheduler = policy.noise_scheduler
    normalizer = policy.normalizer
    dtype = torch.float32

    per_ts: list[np.ndarray] = []
    successes: list[bool] = []
    ep_iter: Iterator[KendamaEpisode] = episodes
    if progress:
        ep_iter = tqdm.tqdm(episodes, desc="Policy embed")

    for ep in ep_iter:
        t = ep.n_steps
        hand_chw = _images_to_chw01(ep.hand_image[:t])
        ext_chw = _images_to_chw01(ep.exterior_image[:t])
        jp = ep.joint_positions[:t]
        gp = ep.gripper_position[:t]
        action_chunks = _build_action_chunks(ep.actions[:t], horizon)

        captured: list[torch.Tensor] = []

        def _hook_fn(_mod, _inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            captured.append(feat.detach().float().mean(dim=-1).cpu())

        handle = _hook_module(policy.model, hook).register_forward_hook(_hook_fn)
        embs: list[np.ndarray] = []
        try:
            for b0 in range(0, t, batch_size):
                b1 = min(b0 + batch_size, t)
                b = b1 - b0
                obs_dict = {
                    "hand_camera_image": torch.from_numpy(hand_chw[b0:b1]).to(device=device, dtype=dtype)[:, None, ...],
                    "exterior_image_1_left": torch.from_numpy(ext_chw[b0:b1]).to(device=device, dtype=dtype)[:, None, ...],
                    "joint_positions": torch.from_numpy(jp[b0:b1]).to(device=device, dtype=dtype)[:, None, :],
                    "gripper_position": torch.from_numpy(gp[b0:b1]).to(device=device, dtype=dtype)[:, None, :],
                }
                with torch.no_grad():
                    nobs = normalizer.normalize(obs_dict)
                    this_nobs = {
                        k: v[:, :to, ...].reshape(-1, *v.shape[2:])
                        for k, v in nobs.items()
                    }
                    nobs_features = policy.obs_encoder(this_nobs)
                    global_cond = nobs_features.reshape(b, -1)
                    actions_t = torch.from_numpy(action_chunks[b0:b1]).to(device=device)
                    nactions = normalizer["action"].normalize(actions_t)
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
                    timesteps = torch.full((b,), t_single, dtype=torch.long, device=device)
                    noise = torch.randn(
                        b, horizon, policy.action_dim,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(b0),
                    )
                    if clean is not None:
                        noisy = scheduler.add_noise(clean, noise, timesteps)
                    else:
                        noisy = scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)
                    policy.model(noisy, timesteps, global_cond=global_cond)
                    embs.append(captured.pop(0).numpy().astype(np.float32))
        finally:
            handle.remove()
        per_ts.append(np.concatenate(embs, axis=0))
        successes.append(ep.success)
    return per_ts, successes


def save_timestep_cache(
    path: Path,
    *,
    per_ts: list[np.ndarray],
    successes: list[bool],
    representation: Representation,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(e) for e in per_ts]).astype(np.int64)
    np.savez_compressed(
        path,
        embeddings=np.concatenate(per_ts, axis=0).astype(np.float32),
        offsets=offsets,
        successes=np.array(successes, dtype=bool),
        representation=representation,
        meta_json=json.dumps(meta),
    )


def load_timestep_cache(path: Path) -> tuple[list[np.ndarray], list[bool], dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    offsets = data["offsets"].tolist()
    embs = data["embeddings"]
    per_ts = [embs[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    meta = json.loads(str(data["meta_json"]))
    return per_ts, data["successes"].tolist(), meta


def aggregate_windows(
    per_ts: list[np.ndarray],
    successes: list[bool],
    *,
    window: int,
    stride: int,
    representation: Representation,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for ep_idx, (series, success) in enumerate(zip(per_ts, successes)):
        t_len = len(series)
        t = 0
        while t + window <= t_len:
            chunk = series[t:t + window]
            if representation in ("state", "state_action"):
                rows.append(chunk.reshape(-1))
            else:
                rows.append(chunk.mean(axis=0))
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
    if not rows:
        return np.empty((0, 0), dtype=np.float32), meta
    return np.stack(rows, axis=0).astype(np.float32), meta


def fit_umap_coords(
    features: np.ndarray,
    *,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    scaled = StandardScaler().fit_transform(features)
    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        verbose=False,
    ).fit_transform(scaled).astype(np.float32)


def cluster_coords(
    coords: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(coords).astype(np.int32)


def slug_for_combo(
    *,
    representation: Representation,
    window: int,
    stride: int,
    n_clusters: int,
    seed: int,
    layer: str | None = None,
) -> str:
    if representation == "state":
        return f"state_full_history_w{window}_s{stride}_seed{seed}_kmeans_k{n_clusters}"
    if representation == "state_action":
        return f"state_action_w{window}_s{stride}_seed{seed}_kmeans_k{n_clusters}"
    return f"policy_emb_{layer}_w{window}_s{stride}_seed{seed}_kmeans_k{n_clusters}"


def save_clustering_dir(
    out_dir: Path,
    *,
    labels: np.ndarray,
    coords: np.ndarray,
    metadata: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cluster_labels.npy", labels)
    np.save(out_dir / "embeddings_reduced.npy", coords)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "manifest.yaml").write_text(yaml.dump(manifest, sort_keys=False))


def build_manifest(
    *,
    representation: Representation,
    n_clusters: int,
    n_samples: int,
    window: int,
    stride: int,
    seed: int,
    task_config: str,
    rollouts: str,
    ckpt: str | None = None,
    layer: str | None = None,
) -> dict[str, Any]:
    influence_source = representation
    aggregation = "flatten" if representation in ("state", "state_action") else "mean"
    manifest: dict[str, Any] = dict(
        algorithm="kmeans",
        scaling="none",
        umap_prescale="standard",
        influence_source=influence_source,
        representation="sliding_window",
        slice_representation=influence_source,
        level="rollout",
        n_clusters=n_clusters,
        n_samples=n_samples,
        auto_k=False,
        window_width=window,
        stride=stride,
        aggregation=aggregation,
        task_config=task_config,
        seed=seed,
        rollouts=rollouts,
        pipeline_steps=["embed", "window", "umap", "kmeans"],
    )
    if representation == "state_action":
        manifest["rep_kwargs"] = dict(
            obs_strategy="proprio",
            action_strategy="executed",
        )
    if representation == "policy_emb":
        manifest["rep_kwargs"] = dict(layer=layer)
        manifest["ckpt"] = ckpt
    return manifest

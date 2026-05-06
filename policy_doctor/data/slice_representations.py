"""Pluggable slice-feature representations for clustering rollouts.

A *slice representation* turns a rollout eval directory into a flat
``(N_slices, D)`` feature matrix plus per-slice metadata in the same
format the existing clustering and E1 evaluation pipeline expects.
This decouples *what feature each slice is* from the downstream
``normalize → prescale → reduce → cluster → save`` pipeline.

Concrete impls:

- :class:`InfEmbedRepresentation` — wraps the existing
  ``extract_infembed_slice_windows``; per-timestep features are the
  Hessian-projected gradient embeddings the policy was trained with.
- :class:`StateRepresentation` — proprioceptive observations from the
  eval episode pickles. Default: most recent obs frame per timestep.
- :class:`StateActionRepresentation` — state vectors concatenated with
  the action the policy executed at that timestep.

All representations share the same window-aggregation engine
(``build_windows_from_rollout_timestep_embeddings``) so window_width,
stride, and aggregation behave identically across them.

The output ``metadata`` schema matches what the existing E1 planner
expects (``rollout_idx``, ``window_start``, ``window_end``,
``window_width``, ``success``).
"""

from __future__ import annotations

import pathlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from policy_doctor.data.clustering_embeddings import (
    build_windows_from_rollout_timestep_embeddings,
)


@dataclass
class SliceWindowParams:
    window_width: int = 5
    stride: int = 2
    aggregation: str = "sum"  # "sum" | "mean" | "max" | "min" | "std" | "median"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SliceRepresentation(ABC):
    """Produce per-slice feature vectors and metadata from rollout eval data.

    Output contract for :meth:`extract`:

    - ``features``: ``np.ndarray`` of shape ``(N_slices, D)``, dtype float32.
    - ``metadata``: list of ``N_slices`` dicts, each containing at minimum
      ``rollout_idx``, ``window_start``, ``window_end``, ``window_width``,
      ``success``. The slice planner used by E1 reads these keys.
    """

    name: str = "abstract"

    @abstractmethod
    def extract(
        self,
        eval_dir: pathlib.Path,
        params: SliceWindowParams,
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        ...

    @abstractmethod
    def extract_per_timestep(
        self,
        eval_dir: pathlib.Path,
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[int], List]:
        """Return (per_timestep_features, episode_lengths, episode_successes).

        per_timestep_features: (N_total_timesteps, D) float32 — raw, before
        windowing.  episode_lengths and episode_successes are needed later by
        :func:`build_windows_from_rollout_timestep_embeddings`.
        """
        ...

    def describe(self, params: SliceWindowParams, **method_kwargs: Any) -> Dict[str, Any]:
        """JSON-serializable fingerprint for the clustering manifest."""
        out = {
            "representation": self.name,
            "window_width": int(params.window_width),
            "stride": int(params.stride),
            "aggregation": params.aggregation,
        }
        out.update({k: _jsonable(v) for k, v in method_kwargs.items()})
        return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


# ---------------------------------------------------------------------------
# Episode loading helpers (shared by State / StateAction)
# ---------------------------------------------------------------------------


def _load_eval_episodes_meta(eval_dir: pathlib.Path) -> Tuple[List[int], List[Optional[bool]]]:
    """Return (episode_lengths, episode_successes) from episodes/metadata.yaml."""
    meta_path = eval_dir / "episodes" / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Episodes metadata not found: {meta_path}")
    with open(meta_path) as f:
        meta = yaml.safe_load(f) or {}
    if "episode_lengths" not in meta:
        raise KeyError(f"episode_lengths missing from {meta_path}")
    ep_lens = list(meta["episode_lengths"])
    ep_succ = list(meta.get("episode_successes") or [None] * len(ep_lens))
    return ep_lens, ep_succ


def _list_episode_pkls(eval_dir: pathlib.Path) -> List[pathlib.Path]:
    ep_dir = eval_dir / "episodes"
    pkls = sorted(ep_dir.glob("ep*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No ep*.pkl under {ep_dir}")
    return pkls


def _flatten_obs_row(obs: Any, *, obs_strategy: str) -> np.ndarray:
    """Reduce a per-timestep obs entry to a 1-D float32 vector.

    ``obs_strategy``:
      - ``"current"`` (default): if obs has shape ``(T, D)`` (history of T past
        frames), take the last (most recent) frame.  Otherwise pass through.
      - ``"full_history"``: flatten the entire (T, D) array to ``(T*D,)``.

    1-D vectors are returned unchanged.
    """
    arr = np.asarray(obs)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        if obs_strategy == "current":
            return arr[-1].astype(np.float32, copy=False)
        if obs_strategy == "full_history":
            return arr.reshape(-1).astype(np.float32, copy=False)
        raise ValueError(
            f"Unknown obs_strategy={obs_strategy!r}; expected 'current' or 'full_history'."
        )
    # Higher-dim (e.g., dict-of-arrays converted): flatten.
    return arr.reshape(-1).astype(np.float32, copy=False)


def _flatten_action_row(action: Any, *, action_strategy: str) -> np.ndarray:
    """Reduce a per-timestep action entry to a 1-D float32 vector.

    ``action_strategy``:
      - ``"executed"`` (default): if action has shape ``(H, A)`` (predicted
        H-step plan), take the first action ``action[0]`` — the one actually
        executed at this timestep.
      - ``"full_plan"``: flatten the full predicted plan to ``(H*A,)``.

    1-D vectors are returned unchanged.
    """
    arr = np.asarray(action)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        if action_strategy == "executed":
            return arr[0].astype(np.float32, copy=False)
        if action_strategy == "full_plan":
            return arr.reshape(-1).astype(np.float32, copy=False)
        raise ValueError(
            f"Unknown action_strategy={action_strategy!r}; expected 'executed' or 'full_plan'."
        )
    return arr.reshape(-1).astype(np.float32, copy=False)


def _build_per_timestep_matrix(
    eval_dir: pathlib.Path,
    *,
    feature_fn,
    expected_episode_lengths: Sequence[int],
) -> np.ndarray:
    """Concatenate per-timestep feature vectors across episodes into ``(T_total, D)``.

    ``feature_fn(df_row) -> np.ndarray (1-D)`` is called for each row of each
    episode pickle, in episode-then-timestep order. Episode count must match
    ``expected_episode_lengths``.
    """
    pkls = _list_episode_pkls(eval_dir)
    if len(pkls) != len(expected_episode_lengths):
        raise RuntimeError(
            f"Episode count mismatch: {len(pkls)} pkl files vs "
            f"{len(expected_episode_lengths)} entries in metadata.yaml"
        )
    rows: List[np.ndarray] = []
    for ep_i, (pkl_path, ep_len) in enumerate(zip(pkls, expected_episode_lengths)):
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        if len(df) != ep_len:
            raise RuntimeError(
                f"Episode {ep_i} ({pkl_path.name}) has {len(df)} rows but "
                f"metadata.yaml says {ep_len}"
            )
        for t in range(len(df)):
            rows.append(feature_fn(df.iloc[t]))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Concrete representations
# ---------------------------------------------------------------------------


class InfEmbedRepresentation(SliceRepresentation):
    """Per-timestep InfEmbed gradient embeddings, sliding-windowed."""

    name = "infembed"

    def extract(
        self,
        eval_dir: pathlib.Path,
        params: SliceWindowParams,
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        from policy_doctor.data.clustering_embeddings import (
            extract_infembed_slice_windows,
        )

        return extract_infembed_slice_windows(
            eval_dir,
            params.window_width,
            params.stride,
            params.aggregation,
        )

    def extract_per_timestep(
        self,
        eval_dir: pathlib.Path,
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[int], List]:
        from policy_doctor.data.clustering_embeddings import load_infembed_per_timestep

        return load_infembed_per_timestep(eval_dir)


class StateRepresentation(SliceRepresentation):
    """Proprioceptive observation vectors as the per-timestep feature.

    Method kwargs:
      - ``obs_strategy``: ``"current"`` (default) or ``"full_history"``. See
        :func:`_flatten_obs_row`.
    """

    name = "state"

    def extract(
        self,
        eval_dir: pathlib.Path,
        params: SliceWindowParams,
        *,
        obs_strategy: str = "current",
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        ep_lens, ep_succ = _load_eval_episodes_meta(eval_dir)

        def _feat(row):
            return _flatten_obs_row(row["obs"], obs_strategy=obs_strategy)

        per_ts = _build_per_timestep_matrix(
            eval_dir, feature_fn=_feat, expected_episode_lengths=ep_lens,
        )
        return build_windows_from_rollout_timestep_embeddings(
            per_ts, ep_lens, ep_succ,
            params.window_width, params.stride, params.aggregation,
        )

    def extract_per_timestep(
        self,
        eval_dir: pathlib.Path,
        *,
        obs_strategy: str = "current",
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[int], List]:
        ep_lens, ep_succ = _load_eval_episodes_meta(eval_dir)

        def _feat(row):
            return _flatten_obs_row(row["obs"], obs_strategy=obs_strategy)

        per_ts = _build_per_timestep_matrix(
            eval_dir, feature_fn=_feat, expected_episode_lengths=ep_lens,
        )
        return per_ts, ep_lens, ep_succ


class StateActionRepresentation(SliceRepresentation):
    """Concatenated [obs, action] vectors as the per-timestep feature.

    Method kwargs:
      - ``obs_strategy``: ``"current"`` (default) or ``"full_history"``.
      - ``action_strategy``: ``"executed"`` (default) or ``"full_plan"``.
    """

    name = "state_action"

    def extract(
        self,
        eval_dir: pathlib.Path,
        params: SliceWindowParams,
        *,
        obs_strategy: str = "current",
        action_strategy: str = "executed",
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        ep_lens, ep_succ = _load_eval_episodes_meta(eval_dir)

        def _feat(row):
            obs_v = _flatten_obs_row(row["obs"], obs_strategy=obs_strategy)
            act_v = _flatten_action_row(row["action"], action_strategy=action_strategy)
            return np.concatenate([obs_v, act_v], axis=0)

        per_ts = _build_per_timestep_matrix(
            eval_dir, feature_fn=_feat, expected_episode_lengths=ep_lens,
        )
        return build_windows_from_rollout_timestep_embeddings(
            per_ts, ep_lens, ep_succ,
            params.window_width, params.stride, params.aggregation,
        )

    def extract_per_timestep(
        self,
        eval_dir: pathlib.Path,
        *,
        obs_strategy: str = "current",
        action_strategy: str = "executed",
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[int], List]:
        ep_lens, ep_succ = _load_eval_episodes_meta(eval_dir)

        def _feat(row):
            obs_v = _flatten_obs_row(row["obs"], obs_strategy=obs_strategy)
            act_v = _flatten_action_row(row["action"], action_strategy=action_strategy)
            return np.concatenate([obs_v, act_v], axis=0)

        per_ts = _build_per_timestep_matrix(
            eval_dir, feature_fn=_feat, expected_episode_lengths=ep_lens,
        )
        return per_ts, ep_lens, ep_succ


class PolicyEmbeddingRepresentation(SliceRepresentation):
    """Pre-computed policy embeddings loaded from disk, sliding-windowed.

    The embeddings are produced by ``compute_policy_embeddings.py`` (cupid_torch2
    env, GPU) and saved as ``<eval_dir>/policy_embeddings/<layer>.npz`` with
    key ``rollout_embeddings``, shape ``(N_total_timesteps, D)``.

    Method kwargs:
      - ``layer``: which embedding to load (default: ``"bottleneck"``).
        Must match the filename produced by the compute script.
    """

    name = "policy_emb"

    def _load(
        self, eval_dir: pathlib.Path, *, layer: str = "bottleneck"
    ) -> Tuple[np.ndarray, List[int], List]:
        emb_path = eval_dir / "policy_embeddings" / f"{layer}.npz"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Policy embeddings not found: {emb_path}\n"
                f"Run compute_policy_embeddings.py --layer {layer} first."
            )
        with np.load(emb_path) as f:
            embeddings = np.asarray(f["rollout_embeddings"], dtype=np.float32)
        meta_path = eval_dir / "episodes" / "metadata.yaml"
        import yaml
        with open(meta_path) as fh:
            meta = yaml.safe_load(fh)
        ep_lens = meta["episode_lengths"]
        ep_succ = meta.get("episode_successes", [None] * len(ep_lens))
        return embeddings, ep_lens, ep_succ

    def extract(
        self,
        eval_dir: pathlib.Path,
        params: SliceWindowParams,
        *,
        layer: str = "bottleneck",
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        embeddings, ep_lens, ep_succ = self._load(eval_dir, layer=layer)
        return build_windows_from_rollout_timestep_embeddings(
            embeddings, ep_lens, ep_succ,
            params.window_width, params.stride, params.aggregation,
        )

    def extract_per_timestep(
        self,
        eval_dir: pathlib.Path,
        *,
        layer: str = "bottleneck",
        **method_kwargs: Any,
    ) -> Tuple[np.ndarray, List[int], List]:
        return self._load(eval_dir, layer=layer)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, SliceRepresentation] = {}


def register_slice_representation(rep: SliceRepresentation) -> None:
    if rep.name in _REGISTRY:
        raise ValueError(f"Slice representation {rep.name!r} already registered")
    _REGISTRY[rep.name] = rep


def get_slice_representation(name: str) -> SliceRepresentation:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown slice representation {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_slice_representations() -> List[str]:
    return sorted(_REGISTRY)


register_slice_representation(InfEmbedRepresentation())
register_slice_representation(StateRepresentation())
register_slice_representation(StateActionRepresentation())
register_slice_representation(PolicyEmbeddingRepresentation())


__all__ = [
    "SliceWindowParams",
    "SliceRepresentation",
    "InfEmbedRepresentation",
    "StateRepresentation",
    "PolicyEmbeddingRepresentation",
    "StateActionRepresentation",
    "register_slice_representation",
    "get_slice_representation",
    "list_slice_representations",
]

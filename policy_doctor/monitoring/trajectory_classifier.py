"""TrajectoryClassifier: classify each timestep of a trajectory via the behavior monitor.

Two input modes:
  - "rollout": data comes from the env (eval_save_episodes pkl or live policy).
    Actions are already in the policy's expected format (rotation_6d if abs_action,
    delta otherwise). No transforms needed.
  - "demo": data comes from an HDF5 demonstration file. If abs_action=True the
    rotation in actions is axis-angle and must be converted to rotation_6d before
    the scorer can compute gradients.

Use from_checkpoint() for the simplest setup — it reads abs_action, rotation_rep,
obs_keys, n_obs_steps, and n_action_steps directly from the checkpoint config.

Environment: requires the cupid conda env (diffusion_policy + infembed).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import yaml

from policy_doctor.monitoring.base import MonitorResult
from policy_doctor.monitoring.stream_monitor import StreamMonitor


def _compute_window_embeddings_for_assigner(
    rollout_embeddings: np.ndarray,
    metadata: list,
    episodes_dir: Optional[str],
) -> np.ndarray:
    """Compute window-mean embeddings so NearestCentroidAssigner has matching length.

    Called when the clustering metadata has fewer entries than rollout_embeddings
    (i.e., clustering was at the window/rollout level, not the timestep level).

    Args:
        rollout_embeddings: Timestep-level embeddings ``(N_timesteps, proj_dim)``.
        metadata: Clustering metadata list; each entry has ``rollout_idx``,
            ``window_start``, ``window_end``.
        episodes_dir: Directory containing ``metadata.yaml`` with
            ``episode_lengths``.  Required for correct global index computation.

    Returns:
        Window-mean embeddings ``(N_windows, proj_dim)``.
    """
    if episodes_dir is None:
        raise ValueError(
            "NearestCentroidAssigner: rollout_embeddings length "
            f"({len(rollout_embeddings)}) does not match cluster_labels length "
            f"({len(metadata)}).  The clustering was done at the window level.  "
            "Pass episodes_dir=<eval_episodes_dir> (directory with metadata.yaml) "
            "to TrajectoryClassifier.from_checkpoint() to enable window embedding "
            "computation."
        )
    ep_meta_path = Path(episodes_dir) / "metadata.yaml"
    if not ep_meta_path.exists():
        raise FileNotFoundError(
            f"episodes_dir/metadata.yaml not found: {ep_meta_path}"
        )
    with open(ep_meta_path) as f:
        ep_meta = yaml.safe_load(f)
    episode_lengths = ep_meta["episode_lengths"]
    global_starts = np.cumsum([0] + list(episode_lengths[:-1]))

    window_embs = []
    for entry in metadata:
        ep = entry["rollout_idx"]
        gs = int(global_starts[ep]) + entry["window_start"]
        ge = int(global_starts[ep]) + entry["window_end"]
        window_embs.append(rollout_embeddings[gs:ge].mean(axis=0))
    return np.array(window_embs, dtype=np.float32)


class TrajectoryClassifier:
    """Classify each timestep of a robot trajectory using the behavior monitor.

    Parameters
    ----------
    monitor:
        A :class:`~policy_doctor.monitoring.StreamMonitor` with scorer and assigner.
    mode:
        ``"rollout"`` or ``"demo"``. See module docstring.
    abs_action:
        Whether the policy was trained with absolute actions (rotation representation).
        Only affects ``mode="demo"``: when True, axis-angle rotations in HDF5 actions
        are converted to ``rotation_rep`` before scoring.
    rotation_rep:
        Target rotation representation (e.g. ``"rotation_6d"``).
    obs_keys:
        HDF5 observation keys to concatenate (used by :meth:`classify_demo_from_hdf5`).
    n_obs_steps:
        Observation horizon To. Used to build sliding windows in :meth:`classify_sequence`.
    n_action_steps:
        Action horizon Ta.
    """

    def __init__(
        self,
        monitor: StreamMonitor,
        mode: Literal["rollout", "demo"] = "rollout",
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",
        obs_keys: Optional[List[str]] = None,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.abs_action = abs_action
        self.rotation_rep = rotation_rep
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self._obs_keys = obs_keys
        self._rotation_transformer = None
        if mode == "demo" and abs_action:
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            self._rotation_transformer = RotationTransformer(
                from_rep="axis_angle", to_rep=rotation_rep
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        infembed_fit_path: str,
        infembed_embeddings_path: str,
        clustering_dir: str,
        mode: Literal["rollout", "demo"] = "rollout",
        device: str = "auto",
        episodes_dir: Optional[str] = None,
    ) -> "TrajectoryClassifier":
        """Build a TrajectoryClassifier from a policy checkpoint.

        Reads ``abs_action``, ``rotation_rep``, ``obs_keys``, ``n_obs_steps``, and
        ``n_action_steps`` directly from the checkpoint config.

        Args:
            checkpoint: Path to policy ``.ckpt`` file.
            infembed_fit_path: Path to ``infembed_fit.pt``.
            infembed_embeddings_path: Path to ``infembed_embeddings.npz``.
            clustering_dir: Path to clustering result directory (must contain
                ``cluster_labels.npy``; ``clustering_models.pkl`` used if present).
            mode: ``"rollout"`` or ``"demo"``.
            device: PyTorch device string.
            episodes_dir: Optional path to the eval episodes directory (must contain
                ``metadata.yaml`` with ``episode_lengths``).  Required when the
                clustering was done at the "rollout" (window) level and the
                ``infembed_embeddings.npz`` rollout embeddings are at the timestep
                level — used to compute window-mean embeddings for the
                ``NearestCentroidAssigner`` centroids.
        """
        import dill
        import torch
        from omegaconf import OmegaConf

        from policy_doctor.behaviors.behavior_graph import BehaviorGraph
        from policy_doctor.data.clustering_loader import load_clustering_result_from_path
        from policy_doctor.monitoring.graph_assigner import FittedModelAssigner, NearestCentroidAssigner
        from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer

        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        dataset_cfg = cfg.task.dataset
        runner_cfg = cfg.task.env_runner

        abs_action = bool(OmegaConf.select(dataset_cfg, "abs_action") or False)
        rotation_rep = str(OmegaConf.select(dataset_cfg, "rotation_rep") or "rotation_6d")
        obs_keys = list(
            OmegaConf.select(dataset_cfg, "obs_keys")
            or ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        )
        n_obs_steps = int(OmegaConf.select(runner_cfg, "n_obs_steps") or 2)
        # Use dataset.horizon as the action window size: the policy predicts `horizon`
        # steps (stored as action_pred in episode pkls), which is what the scorer expects.
        # env_runner.n_action_steps is only the executed subset, not the prediction horizon.
        n_action_steps = int(OmegaConf.select(dataset_cfg, "horizon") or
                             OmegaConf.select(runner_cfg, "n_action_steps") or 8)

        print(f"  [classifier] mode={mode}, abs_action={abs_action}, rotation_rep={rotation_rep}")
        print(f"  [classifier] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

        scorer = InfEmbedStreamScorer(
            checkpoint=checkpoint,
            infembed_fit_path=infembed_fit_path,
            infembed_embeddings_path=infembed_embeddings_path,
            device=device,
        )

        clustering_path = Path(clustering_dir)
        labels, metadata, manifest = load_clustering_result_from_path(clustering_path)
        graph = BehaviorGraph.from_cluster_assignments(
            labels, metadata, level=manifest.get("level", "rollout")
        )
        try:
            assigner = FittedModelAssigner.from_paths(clustering_path, graph)
            print("  [classifier] Assigner: FittedModelAssigner (exact pipeline)")
        except FileNotFoundError:
            rollout_emb = scorer.rollout_embeddings
            if rollout_emb is not None and len(rollout_emb) != len(labels):
                rollout_emb = _compute_window_embeddings_for_assigner(
                    rollout_emb, metadata, episodes_dir
                )
            assigner = NearestCentroidAssigner(
                rollout_embeddings=rollout_emb,
                cluster_labels=labels,
                graph=graph,
            )
            print("  [classifier] Assigner: NearestCentroidAssigner (no clustering_models.pkl found)")

        monitor = StreamMonitor(scorer=scorer, assigner=assigner)
        return cls(
            monitor=monitor,
            mode=mode,
            abs_action=abs_action,
            rotation_rep=rotation_rep,
            obs_keys=obs_keys,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _apply_action_transform(self, action: np.ndarray) -> np.ndarray:
        """Apply rotation transform for demo mode with abs_action=True.

        Matches the logic in robomimic_replay_lowdim_dataset._data_to_obs:
        axis-angle rotation is converted to rotation_6d, expanding action_dim
        from 10 → 13 (single-arm) or 14 → 20 (dual-arm).
        """
        if self._rotation_transformer is None:
            return action
        is_dual_arm = action.shape[-1] == 14
        if is_dual_arm:
            action = action.reshape(*action.shape[:-1], 2, 7)
        pos = action[..., :3]
        rot = action[..., 3:6]
        gripper = action[..., 6:]
        rot = self._rotation_transformer.forward(rot)
        action = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
        if is_dual_arm:
            action = action.reshape(*action.shape[:-2], 20)
        return action

    # ------------------------------------------------------------------
    # Classification entry points
    # ------------------------------------------------------------------

    def classify_sample_embed_only(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> MonitorResult:
        """Classify without computing influence scores (embedding + assignment only).

        ``MonitorResult.influence_scores`` is ``None``.  Use when only the graph node
        assignment is needed at runtime; call :meth:`score_embedding` separately if
        scores are needed (e.g. on intervention trigger).
        """
        if self.mode == "demo":
            action = self._apply_action_transform(action)
        return self.monitor.process_sample_embed_only(obs, action)

    def score_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Compute influence scores from a pre-computed embedding (no gradient pass).

        Returns ``(N_demo,)`` float32 array.  Requires the underlying scorer to
        support ``score_from_embedding`` (``InfEmbedStreamScorer`` does).
        """
        return self.monitor.score_from_embedding(embedding)

    def classify_sample(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> MonitorResult:
        """Classify a single pre-windowed (obs, action) pair.

        For ``mode="demo"`` with ``abs_action=True``, the rotation transform is
        applied here, so pass raw HDF5 actions as-is.

        Args:
            obs: ``(To, Do)`` observation window.
            action: ``(Ta, Da)`` action window.
        """
        if self.mode == "demo":
            action = self._apply_action_transform(action)
        return self.monitor.process_sample(obs, action)

    def classify_episode_from_pkl(
        self,
        episode_df,
    ) -> List[Tuple[int, MonitorResult]]:
        """Classify each timestep from an eval_save_episodes pkl DataFrame.

        The pkl stores the obs window ``(To, Do)`` and action window ``(Ta, Da)``
        per row, already in policy format (rotation_6d, unnormalized). No rotation
        transform is applied regardless of mode.

        Returns:
            List of ``(timestep_index, MonitorResult)`` tuples.
        """
        results = []
        for t, row in episode_df.iterrows():
            obs = np.asarray(row["obs"], dtype=np.float32)
            action = np.asarray(row["action"], dtype=np.float32)
            result = self.monitor.process_sample(obs, action)
            results.append((t, result))
        return results

    def classify_demo_from_hdf5(
        self,
        demo_group,
        obs_keys: Optional[List[str]] = None,
    ) -> List[Tuple[int, MonitorResult]]:
        """Classify each timestep from an open HDF5 demonstration group.

        Constructs sliding windows (size ``n_obs_steps`` / ``n_action_steps``) from
        the raw per-timestep arrays. Applies the rotation transform when
        ``mode="demo"`` and ``abs_action=True``.

        Args:
            demo_group: h5py group, e.g. ``file["data/demo_0"]``.
            obs_keys: Override ``self._obs_keys`` for this call.

        Returns:
            List of ``(timestep_index, MonitorResult)`` tuples, starting from
            timestep ``n_obs_steps - 1``.
        """
        keys = obs_keys or self._obs_keys
        if keys is None:
            raise ValueError("obs_keys must be set (via constructor or obs_keys argument)")

        obs_seq = np.concatenate(
            [demo_group["obs"][k][:] for k in keys], axis=-1
        ).astype(np.float32)
        action_seq = demo_group["actions"][:].astype(np.float32)
        action_seq = self._apply_action_transform(action_seq)

        return self.classify_sequence(obs_seq, action_seq)

    def classify_sequence(
        self,
        obs_seq: np.ndarray,
        action_seq: np.ndarray,
    ) -> List[Tuple[int, MonitorResult]]:
        """Classify a raw per-timestep sequence using sliding windows.

        The rotation transform is NOT applied here — use :meth:`classify_demo_from_hdf5`
        for HDF5 data (it handles the transform), or pre-apply via
        :meth:`_apply_action_transform`.

        Args:
            obs_seq: ``(T, Do)`` per-timestep obs (concatenated obs_keys).
            action_seq: ``(T, Da)`` per-timestep actions (already in policy format).

        Returns:
            List of ``(t, MonitorResult)`` tuples from ``t = n_obs_steps - 1`` onward.
        """
        T = len(obs_seq)
        To, Ta = self.n_obs_steps, self.n_action_steps
        results = []

        for t in range(To - 1, T):
            obs_window = obs_seq[t - To + 1 : t + 1]           # (To, Do)

            action_end = min(t + Ta, T)
            action_window = action_seq[t:action_end]            # (≤Ta, Da)
            if len(action_window) < Ta:
                pad = np.repeat(action_window[-1:], Ta - len(action_window), axis=0)
                action_window = np.concatenate([action_window, pad], axis=0)

            result = self.monitor.process_sample(obs_window, action_window)
            results.append((t, result))

        return results

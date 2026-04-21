"""Probabilistic Mealy Machine (PMM) with Extended L* learning.

Faithful port of the ENAP controller repository's ``agent/pmm_class.py``
(https://github.com/intelligent-control-lab/ENAP_controller), adapted to the
policy_doctor codebase.

Key classes:
- :class:`TrainableRNN` — vanilla RNN wrapper that loads from a pre-trained
  checkpoint saved by :func:`~policy_doctor.enap.rnn_encoder.train_pretrain_rnn`.
- :class:`PMM` — full Extended L* learning engine and PMM topology with
  replay-based edge assignment, pruning, and prediction helpers.

Data format expected by :meth:`PMM.learn_pmm`::

    trajectory_batch = [
        [{'action': np.array(a_dim,), 'state': np.array(s_dim,)}, ...],  # episode 0
        ...
    ]

where ``state`` is a **one-hot** vector over the HDBSCAN symbol alphabet.
"""

from __future__ import annotations

import math
import os
import pickle
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


# ---------------------------------------------------------------------------
# RNN
# ---------------------------------------------------------------------------

class TrainableRNN(nn.Module):
    """Vanilla RNN that loads weights from a ``train_pretrain_rnn`` checkpoint."""

    def __init__(self, action_dim: int, state_dim: int, embed_dim: int, h_dim: int) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.state_embed = nn.Embedding(state_dim, embed_dim)
        self.rnn = nn.RNN(action_dim + embed_dim, h_dim, batch_first=True)

    def load_weights(self, state_dict: Dict) -> None:
        """Load from a ``Pretrain`` checkpoint (enc.* → rnn.*)."""
        self.state_embed.weight.data = torch.as_tensor(state_dict["state_embed.weight"])
        self.rnn.weight_ih_l0.data = torch.as_tensor(state_dict["enc.weight_ih_l0"])
        self.rnn.weight_hh_l0.data = torch.as_tensor(state_dict["enc.weight_hh_l0"])
        self.rnn.bias_ih_l0.data = torch.as_tensor(state_dict["enc.bias_ih_l0"])
        self.rnn.bias_hh_l0.data = torch.as_tensor(state_dict["enc.bias_hh_l0"])

    def _to_tensors(
        self, A: np.ndarray, S: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
            S = torch.from_numpy(S).long()
        if A.dim() == 2:
            A = A.unsqueeze(0)
            S = S.unsqueeze(0)
        return A, S

    def forward_internal(
        self, A: np.ndarray, S: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A, S = self._to_tensors(A, S)
        s_emb = self.state_embed(S)
        x = torch.cat([A, s_emb], dim=-1)
        return self.rnn(x)

    def encode(self, A: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Return final hidden state (h_dim,) as numpy."""
        _, h_n = self.forward_internal(A, S)
        return h_n.squeeze(0).squeeze(0).detach().cpu().numpy()

    def forward_trajectory(self, A: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Return all hidden states ``(T, h_dim)`` as numpy."""
        out, _ = self.forward_internal(A, S)
        return out.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def forward_step(
        self,
        action: np.ndarray,
        state_idx: int,
        h_prev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Single-step update → new hidden state ``(h_dim,)`` as numpy."""
        a = torch.from_numpy(np.asarray(action, dtype=np.float32)).reshape(1, 1, -1)
        s = torch.tensor([[state_idx]], dtype=torch.long)
        s_emb = self.state_embed(s)
        x = torch.cat([a, s_emb], dim=-1)
        h0 = (
            torch.zeros(1, 1, self.h_dim)
            if h_prev is None
            else torch.from_numpy(np.asarray(h_prev, dtype=np.float32)).reshape(1, 1, -1)
        )
        _, h_n = self.rnn(x, h0)
        return h_n.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# PMM
# ---------------------------------------------------------------------------

class PMM:
    """Probabilistic Mealy Machine with Extended L* learning.

    Faithful port of the ENAP controller's ``agent/pmm_class.py``.  Manages
    the full L* pipeline: precomputing RNN embeddings, iterative closedness /
    hypothesis-build / equivalence-query cycles, Union-Find node merging,
    replay-based edge assignment, and pruning.

    Args:
        cos_tau_row: Cosine similarity threshold for closedness and
            representative matching (default 0.6).
        error_threshold: L2 action-distance threshold used in the equivalence
            query (default 0.3).
        max_inner_iters: Maximum L* outer iterations (default 20).
        stabil_required: Consecutive equivalence-query passes required before
            triggering merging and declaring convergence (default 2).
        use_observed_sigma: Only include symbols that actually appear in the
            data (default True).
        use_tqdm: Show progress bar during ``learn_pmm`` (default True).
        seed: RNG seed for reproducibility (default 42).
    """

    def __init__(
        self,
        cos_tau_row: float = 0.6,
        cos_tau_node: Optional[float] = None,
        error_threshold: float = 0.3,
        max_inner_iters: int = 20,
        stabil_required: int = 2,
        use_observed_sigma: bool = True,
        use_tqdm: bool = True,
        seed: int = 42,
    ) -> None:
        self.cos_tau_row = cos_tau_row
        # Node deduplication threshold: two prefix-reps that are ≥ cos_tau_node
        # cosine-similar are merged into one node.  Defaults to cos_tau_row when
        # not set.  Use a higher value than cos_tau_row to produce more nodes
        # (tighter distinctness requirement) while keeping wider coverage
        # neighborhoods for the closedness check.
        self.cos_tau_node = cos_tau_node if cos_tau_node is not None else cos_tau_row
        self.error_threshold = error_threshold
        self.max_inner_iters = max_inner_iters
        self.stabil_required = stabil_required
        self.use_observed_sigma = use_observed_sigma
        self.use_tqdm = use_tqdm
        self.seed = seed

        # Episode data loaded at learn time
        self.episodes: List[Dict] = []        # [{'S': (T,s_dim), 'A': (T,a_dim)}, ...]
        self.sigma: List[int] = []
        self.a_dim: Optional[int] = None
        self.s_dim: Optional[int] = None
        self.cluster_centers: Optional[np.ndarray] = None

        # RNN
        self.rnn: Optional[TrainableRNN] = None
        self.rnn_hidden: int = 64
        self.rnn_embed_dim: int = 16

        # Embedding database (rebuilt on load)
        self.db_embeddings: Optional[np.ndarray] = None        # (N, H)
        self.db_embeddings_norm: Optional[np.ndarray] = None   # (N, H) L2-norm
        self.db_indices_map: List[Tuple[int, int]] = []        # [(epi, t), ...]
        self._episode_offsets: List[int] = []                  # cumulative offsets

        # PMM topology
        self.pmm: Optional[Dict] = None
        self.S: List[Tuple] = []   # L* prefix set (transient during learning)

        # Replay caches (rebuilt after learn / load)
        self._edge_cache: Optional[Dict] = None    # {(q,x,q') -> [(epi,t),...]}
        self._step_cache: Optional[Dict] = None    # {(epi,t) -> (q,x,q')}
        self._qx_actions: Optional[Dict] = None   # {(q,x) -> [action,...]}

        # Representative embedding cache: prefix → embedding (filled during learn_pmm)
        self._rep_embed_cache: Dict[Tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _init_rnn(self, rnn_weights_path: Optional[str]) -> None:
        if not rnn_weights_path or not os.path.exists(rnn_weights_path):
            raise FileNotFoundError(
                f"RNN weights required but not found: {rnn_weights_path!r}"
            )
        ckpt = torch.load(rnn_weights_path, map_location="cpu")
        dims = ckpt["dims"]
        self.a_dim = dims["a"]
        self.s_dim = dims["s"]
        self.rnn_embed_dim = dims["e"]
        self.rnn_hidden = dims["h"]
        self.rnn = TrainableRNN(self.a_dim, self.s_dim, self.rnn_embed_dim, self.rnn_hidden)
        self.rnn.load_weights(ckpt["model_state"])
        self.rnn.eval()
        print(f"  [PMM] Loaded RNN weights from {rnn_weights_path}")

    def _build_sigma(self) -> None:
        if self.use_observed_sigma:
            self.sigma = sorted(
                {
                    int(np.argmax(ep["S"][t]))
                    for ep in self.episodes
                    for t in range(ep["S"].shape[0])
                }
            )
        else:
            self.sigma = list(range(self.s_dim))

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_inputs_for_window(
        self, ep: Dict, t_end_inclusive: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        A = ep["A"][: t_end_inclusive + 1]
        S = np.argmax(ep["S"][: t_end_inclusive + 1], axis=1)
        return A, S

    def _precompute_embeddings(self) -> None:
        all_embeds: List[np.ndarray] = []
        self.db_indices_map = []
        self._episode_offsets = [0]

        for epi, ep in enumerate(self.episodes):
            T = ep["S"].shape[0]
            if T > 0:
                full_A, full_S = self._get_inputs_for_window(ep, T - 1)
                h_seq = self.rnn.forward_trajectory(full_A, full_S)
                for t in range(T):
                    all_embeds.append(h_seq[t])
                    self.db_indices_map.append((epi, t))
            self._episode_offsets.append(self._episode_offsets[-1] + T)

        self.db_embeddings = np.stack(all_embeds)
        norms = np.linalg.norm(self.db_embeddings, axis=1, keepdims=True) + 1e-12
        self.db_embeddings_norm = self.db_embeddings / norms

        # --- Performance caches ---
        # Decoded symbol index per timestep, per episode: avoids repeated argmax
        self._ep_sidx: List[np.ndarray] = [
            np.argmax(ep["S"], axis=1).astype(np.int32)
            for ep in self.episodes
        ]
        # Prefix tuples for every (epi, t): _ep_prefix_tuples[epi][t] == tuple(Sidx[:t])
        # Used by _history_in_S to make the O(t) tuple-build a O(1) lookup.
        self._ep_prefix_tuples: List[List[tuple]] = [
            [tuple(sidx[:t].tolist()) for t in range(len(sidx) + 1)]
            for sidx in self._ep_sidx
        ]

    def _get_step_embedding(self, epi: int, t: int) -> Optional[np.ndarray]:
        if self.db_embeddings is None:
            return None
        idx = self._episode_offsets[epi] + t
        return self.db_embeddings[idx] if idx < len(self.db_embeddings) else None

    def _cosine_similarity(self, vec: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        v_norm = vec / (np.linalg.norm(vec) + 1e-12)
        return matrix_norm @ v_norm

    def _cosine(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(
            np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        )

    def _get_representative_embedding(
        self, seq_prefix: Tuple
    ) -> Optional[np.ndarray]:
        """Canonical RNN embedding for a symbolic prefix sequence.

        Results are memoised in ``_rep_embed_cache`` — the embedding for a
        given prefix is deterministic, so repeated calls within one
        ``learn_pmm`` run are free after the first computation.
        """
        if len(seq_prefix) == 0:
            return np.zeros(self.rnn_hidden)

        cached = self._rep_embed_cache.get(seq_prefix)
        if cached is not None:
            return cached

        seq_arr = np.array(seq_prefix, dtype=np.int32)
        L = len(seq_arr)
        match: Optional[Tuple[int, int]] = None

        ep_sidx = getattr(self, "_ep_sidx", None)
        for epi, ep in enumerate(self.episodes):
            Sidx = ep_sidx[epi] if ep_sidx is not None else np.argmax(ep["S"], axis=1).astype(np.int32)
            T = len(Sidx)
            if T < L:
                continue
            for t in range(L - 1, T):
                if np.array_equal(Sidx[t - L + 1 : t + 1], seq_arr):
                    match = (epi, t)
                    break
            if match:
                break

        if match is None:
            return None

        epi, t = match
        A, S = self._get_inputs_for_window(self.episodes[epi], t)
        z_seed = self.rnn.encode(A, S)

        sims = self._cosine_similarity(z_seed, self.db_embeddings_norm)
        mask = sims >= self.cos_tau_row
        result = np.mean(self.db_embeddings[mask], axis=0) if np.any(mask) else z_seed

        self._rep_embed_cache[seq_prefix] = result
        return result

    def _find_nearest_rep_index(
        self, z: np.ndarray, rep_embeddings: List[np.ndarray]
    ) -> int:
        if len(rep_embeddings) == 1:
            return 0
        # Vectorised: stack reps once, compute all cosine sims in one matmul
        mat = np.stack(rep_embeddings)          # (K, H)
        mat_norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat_n = mat / mat_norms                 # (K, H)
        z_n = z / (np.linalg.norm(z) + 1e-12)  # (H,)
        sims = mat_n @ z_n                      # (K,)
        return int(np.argmax(sims))

    def _find_nearest_rep_index_batch(
        self, zs: np.ndarray, rep_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Vectorised nearest-rep lookup for a batch of query vectors.

        Args:
            zs: (N, H) query embeddings.
            rep_embeddings: list of K representative embeddings.

        Returns:
            (N,) int array of nearest rep indices.
        """
        if len(rep_embeddings) == 1:
            return np.zeros(len(zs), dtype=np.int32)
        mat = np.stack(rep_embeddings)           # (K, H)
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        zs_n = zs / (np.linalg.norm(zs, axis=1, keepdims=True) + 1e-12)
        sims = zs_n @ mat_n.T                    # (N, K)
        return np.argmax(sims, axis=1).astype(np.int32)

    # ------------------------------------------------------------------
    # L* components
    # ------------------------------------------------------------------

    def _ensure_closed(
        self,
        reps: List[Tuple],
        rep_embeddings: List[np.ndarray],
    ) -> bool:
        """Add a new prefix to S if any one-step extension is not covered.

        Uses batch cosine similarity to avoid per-timestep Python loops.
        Scans all uncovered candidates to find the first genuinely new prefix
        (not already in S), preventing infinite duplicate insertions.
        """
        _sidx_cache = getattr(self, "_ep_sidx", None)
        _prefix_cache = getattr(self, "_ep_prefix_tuples", None)
        s_set = set(self.S)  # O(1) membership — prevents duplicate insertions

        # Precompute rep matrix for batch nearest-rep lookups
        rep_mat = np.stack(rep_embeddings)                           # (K, H)
        rep_mat_n = rep_mat / (np.linalg.norm(rep_mat, axis=1, keepdims=True) + 1e-12)

        for i, r_seq in enumerate(reps):
            z_r = rep_embeddings[i]

            if len(r_seq) == 0:
                indices = np.array(
                    [idx for idx, (_, t) in enumerate(self.db_indices_map) if t == 0],
                    dtype=np.int32,
                )
            else:
                sims = self._cosine_similarity(z_r, self.db_embeddings_norm)
                mask = sims >= self.cos_tau_row
                if not np.any(mask):
                    continue
                indices = np.where(mask)[0].astype(np.int32)

            if len(indices) == 0:
                continue

            # Filter to non-terminal timesteps (need t+1 for next-step check)
            non_terminal = np.array([
                idx for idx in indices
                if self.db_indices_map[idx][1] < len(self.episodes[self.db_indices_map[idx][0]]["S"]) - 1
            ], dtype=np.int32)
            if len(non_terminal) == 0:
                continue

            # Check coverage of NEXT-step embeddings (t+1, not t).
            # non_terminal guarantees t < T-1, so flat_idx+1 is within the same episode.
            z_nexts = self.db_embeddings[non_terminal + 1]           # (M, H) — next step
            z_nexts_n = z_nexts / (np.linalg.norm(z_nexts, axis=1, keepdims=True) + 1e-12)
            sims_to_reps = z_nexts_n @ rep_mat_n.T                   # (M, K)
            max_sims = sims_to_reps.max(axis=1)                      # (M,)

            # Find first uncovered next-step embedding whose prefix is new to S
            uncovered = np.where(max_sims < self.cos_tau_row)[0]
            for unc_i in uncovered:
                first_unc = non_terminal[unc_i]
                epi, t = self.db_indices_map[first_unc]
                new_seq = (
                    _prefix_cache[epi][t + 1]
                    if _prefix_cache is not None
                    else tuple(
                        _sidx_cache[epi][: t + 1].tolist()
                        if _sidx_cache is not None
                        else np.argmax(self.episodes[epi]["S"][: t + 1], axis=1)
                    )
                )
                if new_seq not in s_set:
                    self.S.append(new_seq)
                    return True

        return False

    def _build_pmm(self, S: List[Tuple]) -> Dict:
        """Build PMM hypothesis from prefix set S."""
        # 1. Representatives
        reps: List[Tuple] = []
        rep_embeddings: List[np.ndarray] = []
        for s in S:
            z = self._get_representative_embedding(s)
            if z is None:
                continue
            if not any(
                self._cosine(z, z_ex) >= self.cos_tau_node for z_ex in rep_embeddings
            ):
                reps.append(s)
                rep_embeddings.append(z)

        if not reps:
            reps = [tuple()]
            rep_embeddings = [np.zeros(self.rnn_hidden)]

        s_set = set(S)

        # Use cached prefix tuples when available (avoids O(t) tuple rebuild per timestep)
        _prefix_cache = getattr(self, "_ep_prefix_tuples", None)
        _sidx_cache = getattr(self, "_ep_sidx", None)

        def _history_in_S(epi: int, t: int) -> bool:
            if _prefix_cache is not None:
                return _prefix_cache[epi][t] in s_set
            ep = self.episodes[epi]
            history = tuple(np.argmax(ep["S"][:t], axis=1)) if t > 0 else ()
            return history in s_set

        z_init = np.zeros(self.rnn_hidden)
        q_init = self._find_nearest_rep_index(z_init, rep_embeddings)

        # 2. Scan embedding database — collect indices in S, then batch nearest-rep lookup
        in_s_indices: List[int] = []
        for idx in range(len(self.db_embeddings)):
            epi, t = self.db_indices_map[idx]
            if _history_in_S(epi, t):
                in_s_indices.append(idx)

        if not in_s_indices:
            edge_data: Dict = defaultdict(list)
        else:
            idx_arr = np.array(in_s_indices, dtype=np.int32)
            # Batch nearest-rep for current embeddings
            q_next_arr = self._find_nearest_rep_index_batch(
                self.db_embeddings[idx_arr], rep_embeddings
            )
            # Batch nearest-rep for previous embeddings (t>0: idx-1, t==0: q_init)
            prev_embs = np.stack([
                np.zeros(self.rnn_hidden) if self.db_indices_map[idx][1] == 0
                else self.db_embeddings[idx - 1]
                for idx in in_s_indices
            ])
            q_prev_arr = self._find_nearest_rep_index_batch(prev_embs, rep_embeddings)

            edge_data = defaultdict(list)
            for k, idx in enumerate(in_s_indices):
                epi, t = self.db_indices_map[idx]
                ep = self.episodes[epi]
                x = int(_sidx_cache[epi][t]) if _sidx_cache is not None else int(np.argmax(ep["S"][t]))
                action = ep["A"][t]
                next_x = (
                    int(_sidx_cache[epi][t + 1]) if _sidx_cache is not None
                    else int(np.argmax(ep["S"][t + 1]))
                ) if t < len(ep["S"]) - 1 else None
                edge_data[(int(q_prev_arr[k]), x, int(q_next_arr[k]))].append((action, next_x))

        # 3. Aggregate
        delta: Dict = {}
        qx_actions: Dict = defaultdict(list)
        edge_next_inputs: Dict = defaultdict(set)
        qx_dest_counts: Dict = defaultdict(lambda: defaultdict(int))

        for (q, x, q_next), samples in edge_data.items():
            qx_dest_counts[(q, x)][q_next] += len(samples)
            for action, next_x in samples:
                qx_actions[(q, x)].append(action)
                if next_x is not None:
                    edge_next_inputs[(q, x, q_next)].add(next_x)

        for (q, x), counts in qx_dest_counts.items():
            total = sum(counts.values())
            delta[(q, x)] = {nq: c / total for nq, c in counts.items()}

        Q = list(range(len(reps)))
        return {
            "Q": Q,
            "delta": delta,
            "reps": reps,
            "rep_embeddings": rep_embeddings,
            "_qx_actions": dict(qx_actions),
            "_edge_next_inputs": dict(edge_next_inputs),
        }

    def _equivalence_query(self, H: Dict) -> Optional[Tuple]:
        """NFA-style equivalence query: checks every episode against the hypothesis.

        Compares each true action to the mean action prior for (q, x) — not the
        nearest stored action.  Using nearest-neighbour over all stored actions
        makes the EQ trivially satisfied when many examples are stored, preventing
        the PMM from growing.  The mean action prior is the intended comparison
        target in the original ENAP formulation.
        """
        qx_actions = H["_qx_actions"]
        rep_embeddings = H["rep_embeddings"]
        _sidx_cache = getattr(self, "_ep_sidx", None)
        _prefix_cache = getattr(self, "_ep_prefix_tuples", None)

        # Compute mean action prior per (q, x)
        qx_mean_action: Dict = {
            key: np.stack(acts).mean(axis=0)
            for key, acts in qx_actions.items()
        }

        q_init = self._find_nearest_rep_index(np.zeros(self.rnn_hidden), rep_embeddings)

        for epi, ep in enumerate(self.episodes):
            T = ep["S"].shape[0]
            sidx = _sidx_cache[epi] if _sidx_cache is not None else np.argmax(ep["S"], axis=1)
            possible_states = {q_init}

            for t in range(T):
                x = int(sidx[t])
                true_action = ep["A"][t]
                next_possible: set = set()

                for q in possible_states:
                    mean_act = qx_mean_action.get((q, x))
                    if mean_act is None:
                        continue
                    dist = float(np.linalg.norm(mean_act - true_action))
                    if dist <= self.error_threshold:
                        for nq in H["delta"].get((q, x), {}):
                            next_possible.add(nq)

                possible_states = next_possible
                if not possible_states:
                    if _prefix_cache is not None:
                        return _prefix_cache[epi][t + 1]
                    return tuple(sidx[: t + 1].tolist())
        return None

    def _extract_prefix(self, ep: Dict, t: int) -> Tuple:
        return tuple(np.argmax(ep["S"][: t + 1], axis=1))

    def _merge_nodes(self, pmm: Dict) -> Dict:
        """Union-Find merging: collapse nodes with identical self-loop signatures."""
        reps = pmm["reps"]
        rep_embeddings = pmm["rep_embeddings"]
        edge_next = pmm.get("_edge_next_inputs", {})
        n_nodes = len(reps)

        parent = list(range(n_nodes))

        def find(i: int) -> int:
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> bool:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
                return True
            return False

        while True:
            changed = False
            for key, dist in pmm["delta"].items():
                q_src, x = key
                if q_src not in dist:
                    continue
                self_sig = tuple(sorted(edge_next.get((q_src, x, q_src), [])))
                if self_sig != (x,):
                    continue
                for q_dest in list(dist.keys()):
                    if q_dest == q_src:
                        continue
                    if (
                        tuple(sorted(edge_next.get((q_src, x, q_dest), [])))
                        == self_sig
                    ):
                        if union(q_src, q_dest):
                            changed = True
            if not changed:
                break

        # Rebuild with merged nodes
        old_roots = sorted(set(find(i) for i in range(n_nodes)))
        root_to_new = {r: i for i, r in enumerate(old_roots)}
        new_n = len(old_roots)

        new_reps: List[Optional[Tuple]] = [None] * new_n
        embed_groups: List[List[np.ndarray]] = [[] for _ in range(new_n)]
        for i in range(n_nodes):
            r = root_to_new[find(i)]
            if new_reps[r] is None:
                new_reps[r] = reps[i]
            if i < len(rep_embeddings):
                embed_groups[r].append(rep_embeddings[i])

        new_rep_embeddings = [
            np.mean(g, axis=0) if g else np.zeros(self.rnn_hidden)
            for g in embed_groups
        ]

        new_delta: Dict = defaultdict(lambda: defaultdict(float))
        new_edge_next: Dict = defaultdict(set)

        for key, dist in pmm["delta"].items():
            q_old, x = key
            q_new = root_to_new[find(q_old)]
            for q_dest_old, prob in dist.items():
                nq_new = root_to_new[find(q_dest_old)]
                new_delta[(q_new, x)][nq_new] += prob

        for key, nx_set in edge_next.items():
            q_old, x, q_dest_old = key
            q_new = root_to_new[find(q_old)]
            nq_new = root_to_new[find(q_dest_old)]
            new_edge_next[(q_new, x, nq_new)] |= nx_set

        # Renormalise probabilities
        final_delta: Dict = {}
        for (q, x), counts in dict(new_delta).items():
            total = sum(counts.values())
            final_delta[(q, x)] = {nq: c / total for nq, c in counts.items()}

        return {
            "Q": list(range(new_n)),
            "delta": final_delta,
            "reps": new_reps,
            "rep_embeddings": new_rep_embeddings,
            "_edge_next_inputs": dict(new_edge_next),
        }

    # ------------------------------------------------------------------
    # Replay (post-hoc edge assignment)
    # ------------------------------------------------------------------

    def replay_assign(self) -> Tuple[Dict, Dict]:
        """Walk each episode through delta, assigning (epi, t) to edges.

        Multi-edge disambiguation: pick the destination whose
        ``rep_embedding`` is closest to the current RNN embedding ``z_t``.
        """
        if not self.episodes or self.pmm is None:
            return {}, {}

        delta = self.pmm["delta"]
        rep_embeddings = self.pmm.get("rep_embeddings", [])

        edge_samples: Dict = defaultdict(list)
        step_to_edge: Dict = {}

        _sidx_cache = getattr(self, "_ep_sidx", None)

        for epi, ep in enumerate(self.episodes):
            T = ep["S"].shape[0]
            sidx = _sidx_cache[epi] if _sidx_cache is not None else np.argmax(ep["S"], axis=1)
            q = 0
            for t in range(T):
                x = int(sidx[t])
                dist = delta.get((q, x))

                if dist is None:
                    q_next = q
                elif len(dist) == 1:
                    q_next = next(iter(dist))
                else:
                    z_t = self._get_step_embedding(epi, t)
                    best_q, best_sim = None, -2.0
                    if z_t is not None and rep_embeddings:
                        z_t_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
                        for q_cand in dist:
                            if q_cand < len(rep_embeddings):
                                sim = float(
                                    np.dot(
                                        z_t_norm,
                                        rep_embeddings[q_cand]
                                        / (np.linalg.norm(rep_embeddings[q_cand]) + 1e-12),
                                    )
                                )
                                if sim > best_sim:
                                    best_sim = sim
                                    best_q = q_cand
                    if best_q is None:
                        best_q = next(iter(dist))
                    q_next = best_q

                edge_samples[(q, x, q_next)].append((epi, t))
                step_to_edge[(epi, t)] = (q, x, q_next)
                q = q_next

        return dict(edge_samples), step_to_edge

    def _rebuild_cache(self) -> None:
        self._edge_cache, self._step_cache = self.replay_assign()
        self._qx_actions = defaultdict(list)
        for (q, x, _), indices in self._edge_cache.items():
            for epi, t in indices:
                self._qx_actions[(q, x)].append(self.episodes[epi]["A"][t])
        self._qx_actions = dict(self._qx_actions)

    def _prune_after_replay(self) -> None:
        """Remove empty edges and unreachable nodes, remap indices."""
        old_delta = self.pmm["delta"]
        edge_cache = self._edge_cache or {}

        # 1. Rebuild delta from edge cache counts
        qx_counts: Dict = defaultdict(lambda: defaultdict(int))
        for (q, x, q_next), indices in edge_cache.items():
            if indices:
                qx_counts[(q, x)][q_next] += len(indices)

        new_delta: Dict = {}
        for (q, x), counts in qx_counts.items():
            total = sum(counts.values())
            new_delta[(q, x)] = {nq: c / total for nq, c in counts.items()}

        # 2. BFS reachability from q=0
        reachable: set = set()
        frontier = [0]
        while frontier:
            node = frontier.pop(0)
            if node in reachable:
                continue
            reachable.add(node)
            for (q, x), dist in new_delta.items():
                if q == node:
                    for q_next in dist:
                        if q_next not in reachable:
                            frontier.append(q_next)

        # 3. Remap to contiguous indices
        old_sorted = sorted(reachable)
        old_to_new = {old: new for new, old in enumerate(old_sorted)}

        remapped_delta: Dict = {}
        for (q, x), dist in new_delta.items():
            if q not in reachable:
                continue
            new_dist = {
                old_to_new[q_next]: prob
                for q_next, prob in dist.items()
                if q_next in reachable
            }
            if new_dist:
                total = sum(new_dist.values())
                remapped_delta[(old_to_new[q], x)] = {
                    nq: p / total for nq, p in new_dist.items()
                }

        old_reps = self.pmm["reps"]
        old_rep_emb = self.pmm["rep_embeddings"]
        new_reps = [old_reps[i] for i in old_sorted]
        new_rep_emb = [
            old_rep_emb[i] for i in old_sorted if i < len(old_rep_emb)
        ]

        n_removed_nodes = len(self.pmm["Q"]) - len(reachable)
        old_edge_count = sum(len(d) for d in old_delta.values())
        new_edge_count = sum(len(d) for d in remapped_delta.values())
        if n_removed_nodes > 0 or old_edge_count != new_edge_count:
            print(
                f"  [PMM] Prune: removed {n_removed_nodes} node(s), "
                f"{old_edge_count - new_edge_count} edge(s) "
                f"→ {len(reachable)} nodes, {new_edge_count} edges"
            )

        self.pmm = {
            "Q": list(range(len(reachable))),
            "reps": new_reps,
            "delta": remapped_delta,
            "rep_embeddings": new_rep_emb,
        }
        self._rebuild_cache()

    # ------------------------------------------------------------------
    # Main learning entry point
    # ------------------------------------------------------------------

    def learn_pmm(
        self,
        trajectory_batch: List[List[Dict]],
        rnn_weights_path: Optional[str] = None,
        cluster_centers: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run Extended L* on a batch of trajectories to learn a PMM.

        Args:
            trajectory_batch: List of episodes, each a list of dicts with
                keys ``"action"`` ``(a_dim,)`` and ``"state"`` ``(s_dim,)``
                (one-hot over HDBSCAN symbols).
            rnn_weights_path: Path to a ``train_pretrain_rnn`` checkpoint
                ``{'model_state': ..., 'dims': {'a','s','e','h'}}``.
            cluster_centers: ``(s_dim, feat_dim)`` HDBSCAN cluster centroids
                (used by the residual M-step; stored but not required here).

        Returns:
            The learned PMM dict ``{'Q', 'delta', 'reps', 'rep_embeddings'}``.
        """
        self._set_seed()
        self.cluster_centers = (
            np.asarray(cluster_centers) if cluster_centers is not None else None
        )

        # Convert to internal format
        self.episodes = []
        for ep in trajectory_batch:
            S = np.stack([t["state"] for t in ep], axis=0)
            A = np.stack([t["action"] for t in ep], axis=0)
            self.episodes.append({"S": S, "A": A})

        self.a_dim = len(trajectory_batch[0][0]["action"])
        self.s_dim = len(trajectory_batch[0][0]["state"])

        self._init_rnn(rnn_weights_path)
        self._build_sigma()
        self._precompute_embeddings()

        self.S = [tuple()]
        self._rep_embed_cache = {}  # reset cache at start of learning
        stabil = 0

        def _wandb_log_iter(it: int, n_prefixes: int, n_nodes: int, converged: bool) -> None:
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log(
                        {
                            "pmm/num_prefixes": n_prefixes,
                            "pmm/num_hypothesis_nodes": n_nodes,
                            "pmm/converged": int(converged),
                            "pmm/iteration": it + 1,
                        },
                        step=it + 1,
                    )
            except ImportError:
                pass

        for it in trange(
            self.max_inner_iters,
            desc="L* Iteration",
            disable=not self.use_tqdm,
        ):
            # Closedness check
            while True:
                temp_reps: List[Tuple] = []
                temp_embeds: List[np.ndarray] = []
                for s in self.S:
                    z = self._get_representative_embedding(s)
                    if z is not None and not any(
                        self._cosine(z, z_ex) >= self.cos_tau_node
                        for z_ex in temp_embeds
                    ):
                        temp_reps.append(s)
                        temp_embeds.append(z)
                if not self._ensure_closed(temp_reps, temp_embeds):
                    break

            H = self._build_pmm(self.S)
            ce = self._equivalence_query(H)

            if ce is None:
                stabil += 1
                print(f"  [PMM] Stabilized at iteration {it + 1}")
                _wandb_log_iter(it, len(self.S), len(H["Q"]), converged=False)
                if stabil >= self.stabil_required:
                    H = self._merge_nodes(H)
                    self.pmm = {
                        "Q": H["Q"],
                        "delta": H["delta"],
                        "reps": H["reps"],
                        "rep_embeddings": H["rep_embeddings"],
                    }
                    self._rebuild_cache()
                    self._prune_after_replay()
                    n_final = len(self.pmm["Q"])
                    n_edges = sum(len(d) for d in self.pmm["delta"].values())
                    print(f"  [PMM] Converged at iteration {it + 1}")
                    _wandb_log_iter(it, len(self.S), n_final, converged=True)
                    try:
                        import wandb as _wandb
                        if _wandb.run is not None:
                            _wandb.log(
                                {
                                    "pmm/final_nodes": n_final,
                                    "pmm/final_edges": n_edges,
                                    "pmm/converged_at_iter": it + 1,
                                }
                            )
                    except ImportError:
                        pass
                    return self.pmm
            else:
                _wandb_log_iter(it, len(self.S), len(H["Q"]), converged=False)
                for k in range(1, len(ce) + 1):
                    p = ce[:k]
                    if p not in self.S:
                        self.S.append(p)
                stabil = 0

        # Reached max iterations
        H = self._build_pmm(self.S)
        H = self._merge_nodes(H)
        self.pmm = {
            "Q": H["Q"],
            "delta": H["delta"],
            "reps": H["reps"],
            "rep_embeddings": H["rep_embeddings"],
        }
        self._rebuild_cache()
        self._prune_after_replay()
        n_final = len(self.pmm["Q"])
        n_edges = sum(len(d) for d in self.pmm["delta"].values())
        print("  [PMM] Max iterations reached.")
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.log(
                    {
                        "pmm/final_nodes": n_final,
                        "pmm/final_edges": n_edges,
                        "pmm/converged_at_iter": self.max_inner_iters,
                    }
                )
        except ImportError:
            pass
        return self.pmm

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, q: int, x: int) -> np.ndarray:
        """Return mean action for ``(q, x)`` from the replay cache."""
        if self._qx_actions is None:
            raise ValueError("No replay cache.  Call learn_pmm() first.")
        actions = self._qx_actions.get((q, x))
        if not actions:
            raise KeyError(f"No actions for (q={q}, x={x}) in replay cache.")
        return np.mean(actions, axis=0)

    def predict_list(self, q: int, x: int) -> List[np.ndarray]:
        """Return all observed actions for ``(q, x)``."""
        if self._qx_actions is None:
            raise ValueError("No replay cache.  Call learn_pmm() first.")
        actions = self._qx_actions.get((q, x))
        if not actions:
            raise KeyError(f"No actions for (q={q}, x={x}) in replay cache.")
        return actions

    def step(self, q: int, x: int, z_t: Optional[np.ndarray] = None) -> int:
        """Advance the PMM from state ``q`` on symbol ``x``.

        When multiple outgoing edges exist, picks the destination whose
        ``rep_embedding`` is closest to ``z_t`` (embedding-based disambiguation).
        """
        dist = self.pmm["delta"].get((q, x))
        if dist is None:
            return q  # self-loop
        if len(dist) == 1:
            return next(iter(dist))

        rep_embeddings = self.pmm.get("rep_embeddings", [])
        best_q, best_sim = None, -2.0
        if z_t is not None and rep_embeddings:
            z_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
            for q_cand in dist:
                if q_cand < len(rep_embeddings):
                    z_rep = rep_embeddings[q_cand]
                    sim = float(
                        np.dot(z_norm, z_rep / (np.linalg.norm(z_rep) + 1e-12))
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_q = q_cand
        return best_q if best_q is not None else next(iter(dist))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pmm(self, path: str) -> None:
        """Serialize PMM topology + RNN weights + episodes to a pickle."""
        if self.pmm is None:
            raise ValueError("PMM not trained; nothing to save.")

        rnn_state: Optional[Dict] = None
        rnn_type = "trainable"
        if isinstance(self.rnn, TrainableRNN):
            rnn_state = {
                k: v.detach().cpu().numpy()
                for k, v in self.rnn.state_dict().items()
            }

        payload = {
            "pmm": self.pmm,
            "sigma": self.sigma,
            "a_dim": self.a_dim,
            "s_dim": self.s_dim,
            "rnn_hidden": self.rnn_hidden,
            "rnn_embed_dim": self.rnn_embed_dim,
            "rnn_type": rnn_type,
            "rnn_state": rnn_state,
            "cos_tau_row": self.cos_tau_row,
            "cluster_centers": self.cluster_centers,
            "episodes": self.episodes,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [PMM] Saved to {path}")

    def load_pmm(self, path: str) -> None:
        """Load a serialized PMM, reconstruct RNN, rebuild replay cache."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        pmm_data = payload["pmm"]
        self.pmm = {
            "Q": pmm_data["Q"],
            "delta": pmm_data["delta"],
            "reps": pmm_data["reps"],
            "rep_embeddings": pmm_data["rep_embeddings"],
        }
        self.sigma = payload["sigma"]
        self.a_dim = payload["a_dim"]
        self.s_dim = payload["s_dim"]
        self.rnn_hidden = payload["rnn_hidden"]
        self.rnn_embed_dim = payload.get("rnn_embed_dim", 16)
        self.cos_tau_row = payload.get("cos_tau_row", self.cos_tau_row)
        self.cluster_centers = payload.get("cluster_centers")
        self.episodes = payload.get("episodes", [])

        rnn_state = payload.get("rnn_state")
        if rnn_state is not None:
            self.rnn = TrainableRNN(
                self.a_dim, self.s_dim, self.rnn_embed_dim, self.rnn_hidden
            )
            # Load weights (handles both tensor and numpy array formats)
            for k, v in rnn_state.items():
                if isinstance(v, np.ndarray):
                    rnn_state[k] = torch.from_numpy(v)
            self.rnn.load_state_dict(rnn_state, strict=False)
            self.rnn.eval()

        if self.episodes:
            self._precompute_embeddings()
            self._rebuild_cache()

    def to_json_serializable(self) -> Dict[str, Any]:
        """Return a JSON-safe dict of the PMM topology (no numpy arrays)."""
        if self.pmm is None:
            return {}
        delta_str = {
            f"{q}_{x}": {str(nq): p for nq, p in dist.items()}
            for (q, x), dist in self.pmm["delta"].items()
        }
        return {
            "Q": self.pmm["Q"],
            "delta": delta_str,
            "num_nodes": len(self.pmm["Q"]),
            "num_edges": sum(len(d) for d in self.pmm["delta"].values()),
            "sigma": self.sigma,
            "a_dim": self.a_dim,
            "s_dim": self.s_dim,
            "cos_tau_row": self.cos_tau_row,
        }

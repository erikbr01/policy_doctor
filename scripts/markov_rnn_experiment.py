"""Markov RNN experiment: policy embeddings → GRU → Markovian h_t.

Pipeline:
  1. Load InfEmbed policy embeddings z_t (100-dim), cluster labels c_t, actions a_t
     from transport_mh_seed0_r512_clustering + mar27 episode pkl files.
  2. Train a GRU:  [z_t ; a_t]  →  h_t  with phase-aware contrastive loss
     (using c_t to define phase boundaries).
  3. K-Means on h_t  →  new cluster labels.
  4. Run Markov chi2 test on baseline c_t vs. new h_t-cluster labels.
  5. Print comparison.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from policy_doctor.behaviors.behavior_graph import test_markov_property

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CLUSTER_DIR = Path(
    "/home/erbauer/refactor_cupid/policy_doctor/third_party/cupid/data/"
    "clusterings/transport_mh_seed0_r512_clustering"
)
_EPISODES_DIR = Path(
    "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/"
    "mar27/mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest/episodes"
)
_OUT_DIR = Path(
    "/home/erbauer/refactor_cupid/policy_doctor/data/"
    "markov_rnn_experiment"
)
_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _episode_file(rollout_idx: int) -> Path:
    prefix = f"ep{rollout_idx:04d}"
    for name in os.listdir(_EPISODES_DIR):
        if name.startswith(prefix):
            return _EPISODES_DIR / name
    raise FileNotFoundError(f"No episode file for rollout {rollout_idx}")


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[Dict], List[np.ndarray], List[np.ndarray]]:
    """Return (embeddings, labels, metadata, per_rollout_z, per_rollout_a).

    per_rollout_z: list of (n_windows, 100) arrays  (policy embeddings)
    per_rollout_a: list of (n_windows, action_dim) arrays  (mean action per window)
    """
    z_all = np.load(_CLUSTER_DIR / "embeddings_reduced.npy").astype(np.float32)  # (N, 100)
    c_all = np.load(_CLUSTER_DIR / "cluster_labels.npy")                          # (N,)
    meta = json.load(open(_CLUSTER_DIR / "metadata.json"))                         # list of N dicts

    # Group window indices by rollout
    from collections import defaultdict
    rollout_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, e in enumerate(meta):
        rollout_to_idx[int(e["rollout_idx"])].append(i)

    # Load actions per rollout, align with windows
    print(f"Loading actions for {len(rollout_to_idx)} rollouts ...")
    per_rollout_z: List[np.ndarray] = []
    per_rollout_a: List[np.ndarray] = []

    for rid in sorted(rollout_to_idx.keys()):
        indices = rollout_to_idx[rid]
        windows = [meta[i] for i in indices]
        # sort by window_start to get temporal order
        order = sorted(range(len(windows)), key=lambda j: windows[j]["window_start"])
        indices = [indices[j] for j in order]
        windows = [windows[j] for j in order]

        z_seq = z_all[indices]  # (T_w, 100)

        # load episode actions
        ep_df = pickle.load(open(_episode_file(rid), "rb"))
        ep_actions = np.stack([
            np.array(a, dtype=np.float32).reshape(-1) if not isinstance(a, np.ndarray)
            else a.astype(np.float32).reshape(-1)
            for a in ep_df["action"].values
        ])  # (T_ep, action_flat_dim)

        # for each window use action at window_start (clip to valid range)
        T_ep = len(ep_actions)
        a_seq = np.array([
            ep_actions[min(w["window_start"], T_ep - 1)]
            for w in windows
        ], dtype=np.float32)  # (T_w, action_flat_dim)

        per_rollout_z.append(z_seq)
        per_rollout_a.append(a_seq)

    return z_all, c_all, meta, per_rollout_z, per_rollout_a


# ---------------------------------------------------------------------------
# GRU encoder
# ---------------------------------------------------------------------------

class MarkovGRU(nn.Module):
    """GRU that maps [z_t ; a_t] history → Markovian h_t."""

    def __init__(self, z_dim: int, a_dim: int, h_dim: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.gru = nn.GRU(z_dim + a_dim, h_dim, num_layers=num_layers, batch_first=True)
        # Auxiliary heads for supervised signal
        self.z_head = nn.Linear(h_dim, z_dim)   # predict next z
        self.c_head: Optional[nn.Linear] = None  # set after knowing n_clusters

    def set_n_clusters(self, n: int) -> None:
        self.c_head = nn.Linear(self.h_dim, n).to(next(self.parameters()).device)

    def forward(
        self,
        z: torch.Tensor,  # (B, T, z_dim)
        a: torch.Tensor,  # (B, T, a_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([z, a], dim=-1)          # (B, T, z_dim+a_dim)
        h_seq, _ = self.gru(x)                  # (B, T, h_dim)
        z_pred = self.z_head(h_seq)             # (B, T, z_dim)
        c_pred = self.c_head(h_seq) if self.c_head is not None else None
        return h_seq, z_pred, c_pred


def phase_contrastive_loss(
    h_t: torch.Tensor,   # (N, D)
    h_t1: torch.Tensor,  # (N, D)
    c_t: torch.Tensor,   # (N,) int
    c_t1: torch.Tensor,  # (N,) int
    margin: float = 0.5,
) -> torch.Tensor:
    cos_sim = F.cosine_similarity(h_t, h_t1, dim=-1)
    dist = 1.0 - cos_sim
    same = (c_t == c_t1).float()
    loss = same * dist + (1.0 - same) * torch.clamp(margin - dist, min=0.0)
    return loss.mean()


def _pad_batch(
    seqs_z: List[np.ndarray],
    seqs_a: List[np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_T = max(s.shape[0] for s in seqs_z)
    B = len(seqs_z)
    z_dim = seqs_z[0].shape[1]
    a_dim = seqs_a[0].shape[1]
    Z = np.zeros((B, max_T, z_dim), dtype=np.float32)
    A = np.zeros((B, max_T, a_dim), dtype=np.float32)
    for i, (z, a) in enumerate(zip(seqs_z, seqs_a)):
        Z[i, :len(z)] = z
        A[i, :len(a)] = a
    return (
        torch.from_numpy(Z).to(device),
        torch.from_numpy(A).to(device),
    )


def train_gru(
    model: MarkovGRU,
    per_rollout_z: List[np.ndarray],
    per_rollout_a: List[np.ndarray],
    per_rollout_c: List[np.ndarray],
    num_epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    weights: Optional[Dict[str, float]] = None,
) -> None:
    weights = weights or {"z_pred": 1.0, "c_pred": 1.0, "contrast": 0.5}
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    n_eps = len(per_rollout_z)

    for epoch in range(num_epochs):
        idx = np.random.permutation(n_eps)
        total, total_n = 0.0, 0

        for start in range(0, n_eps, batch_size):
            batch = idx[start:start + batch_size]
            sz = [per_rollout_z[i] for i in batch]
            sa = [per_rollout_a[i] for i in batch]
            sc = [per_rollout_c[i] for i in batch]

            Z, A = _pad_batch(sz, sa, device)
            max_T = Z.shape[1]
            # pad c labels
            C = np.zeros((len(batch), max_T), dtype=np.int64)
            for j, c in enumerate(sc):
                C[j, :len(c)] = c
            C_t = torch.from_numpy(C).to(device)

            h_seq, z_pred, c_pred = model(Z, A)

            # next-z prediction (shift by 1)
            l_z = F.mse_loss(z_pred[:, :-1], Z[:, 1:])

            # next-c prediction
            B_, T_, V = c_pred.shape
            l_c = F.cross_entropy(
                c_pred[:, :-1].reshape(-1, V),
                C_t[:, 1:].reshape(-1),
            )

            # phase-contrastive on consecutive h_t pairs
            h_flat = h_seq[:, :-1].reshape(-1, model.h_dim)
            h1_flat = h_seq[:, 1:].reshape(-1, model.h_dim)
            c_flat = C_t[:, :-1].reshape(-1)
            c1_flat = C_t[:, 1:].reshape(-1)
            l_cont = phase_contrastive_loss(h_flat, h1_flat, c_flat, c1_flat)

            loss = (
                weights["z_pred"] * l_z
                + weights["c_pred"] * l_c
                + weights["contrast"] * l_cont
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(batch)
            total_n += len(batch)

        sched.step()
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1}/{num_epochs}  loss={total/total_n:.4f}  "
                  f"z={float(l_z):.4f}  c={float(l_c):.4f}  cont={float(l_cont):.4f}")

    model.eval()


@torch.no_grad()
def extract_h(
    model: MarkovGRU,
    per_rollout_z: List[np.ndarray],
    per_rollout_a: List[np.ndarray],
    device: torch.device = DEVICE,
) -> np.ndarray:
    model.eval()
    all_h = []
    for z, a in zip(per_rollout_z, per_rollout_a):
        Z = torch.from_numpy(z).unsqueeze(0).to(device)   # (1, T, z_dim)
        A = torch.from_numpy(a).unsqueeze(0).to(device)   # (1, T, a_dim)
        h_seq, _, _ = model(Z, A)
        all_h.append(h_seq.squeeze(0).cpu().numpy())
    return np.concatenate(all_h, axis=0)  # (N_total, h_dim)


# ---------------------------------------------------------------------------
# EM loop helpers
# ---------------------------------------------------------------------------

def _build_per_rollout_c(
    labels: np.ndarray,
    meta: List[Dict],
    rollout_to_idx: Dict[int, List[int]],
) -> List[np.ndarray]:
    per_rollout_c = []
    for rid in sorted(rollout_to_idx.keys()):
        indices = rollout_to_idx[rid]
        windows = [meta[i] for i in indices]
        order = sorted(range(len(windows)), key=lambda j: windows[j]["window_start"])
        per_rollout_c.append(labels[[indices[j] for j in order]])
    return per_rollout_c


def run_em(
    per_rollout_z: List[np.ndarray],
    per_rollout_a: List[np.ndarray],
    meta: List[Dict],
    rollout_to_idx: Dict[int, List[int]],
    c_init: np.ndarray,
    k: int,
    em_iters: int = 5,
    epochs_per_iter: int = 60,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """EM loop: alternate GRU training (M) and KMeans re-clustering (E).

    Returns (h_final, labels_final).
    """
    z_dim = per_rollout_z[0].shape[1]
    a_dim = per_rollout_a[0].shape[1]

    # Re-map c_init to [0, k-1] via fresh KMeans on z if k != current n_clusters
    if k != int(np.max(c_init)) + 1:
        c_cur = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
            np.vstack(per_rollout_z)
        ).astype(np.int64)
    else:
        c_cur = c_init.copy()

    model = MarkovGRU(z_dim=z_dim, a_dim=a_dim, h_dim=128)
    model.set_n_clusters(k)
    model.to(device)

    for em_iter in range(em_iters):
        per_rollout_c = _build_per_rollout_c(c_cur, meta, rollout_to_idx)
        # Rebuild c_head for current k (in case k changed; here k is fixed)
        if model.c_head.out_features != k:
            model.set_n_clusters(k)
        train_gru(
            model, per_rollout_z, per_rollout_a, per_rollout_c,
            num_epochs=epochs_per_iter, batch_size=32, device=device,
        )
        h_all = extract_h(model, per_rollout_z, per_rollout_a, device=device)
        c_cur = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(h_all).astype(np.int64)

        res = test_markov_property(c_cur, meta, level="rollout", significance_level=0.05)
        n_viols = sum(1 for r in res["per_state"].values() if r.testable and not r.markov_holds)
        n_tested = res["num_states_tested"]
        print(f"    EM iter {em_iter+1}/{em_iters}: violations {n_viols}/{n_tested}")

    h_final = extract_h(model, per_rollout_z, per_rollout_a, device=device)
    return h_final, c_cur


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Markov RNN Experiment — Transport Task (EM sweep)")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data ...")
    z_all, c_all, meta, per_rollout_z, per_rollout_a = load_dataset()

    from collections import defaultdict
    rollout_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, e in enumerate(meta):
        rollout_to_idx[int(e["rollout_idx"])].append(i)

    print(f"  windows: {len(z_all)}  rollouts: {len(per_rollout_z)}  "
          f"z_dim: {z_all.shape[1]}  a_dim: {per_rollout_a[0].shape[1]}")

    # 2. Baseline Markov test (original InfEmbed KMeans, k=20)
    print("\n[2] Baseline Markov test (InfEmbed KMeans, k=20) ...")
    res_base = test_markov_property(c_all, meta, level="rollout", significance_level=0.05)
    n_viols_base = sum(1 for r in res_base["per_state"].values() if r.testable and not r.markov_holds)
    print(f"  violations={n_viols_base}/{res_base['num_states_tested']}  markov_holds={res_base['markov_holds']}")

    # 3. EM sweep over k
    K_VALUES = [10, 15, 20, 30]
    EM_ITERS = 5
    EPOCHS_PER_ITER = 60

    print(f"\n[3] EM sweep: k in {K_VALUES}, {EM_ITERS} EM iters × {EPOCHS_PER_ITER} epochs each")

    results = {}
    for k in K_VALUES:
        print(f"\n  --- k={k} ---")
        h_final, labels_final = run_em(
            per_rollout_z, per_rollout_a, meta, rollout_to_idx,
            c_init=c_all, k=k, em_iters=EM_ITERS,
            epochs_per_iter=EPOCHS_PER_ITER, device=DEVICE,
        )
        res = test_markov_property(labels_final, meta, level="rollout", significance_level=0.05)
        n_viols = sum(1 for r in res["per_state"].values() if r.testable and not r.markov_holds)
        n_tested = res["num_states_tested"]
        viols = [(sid, r.p_value) for sid, r in res["per_state"].items() if r.testable and not r.markov_holds]
        results[k] = {"violations": n_viols, "tested": n_tested, "per_state": res["per_state"], "viols": viols}
        np.save(_OUT_DIR / f"h_em_k{k}.npy", h_final)
        np.save(_OUT_DIR / f"labels_em_k{k}.npy", labels_final)

    # 4. Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  {'method':30s}  {'violations':>12}  {'tested':>6}")
    print(f"  {'Baseline InfEmbed KMeans k=20':30s}  {n_viols_base:>5}/{res_base['num_states_tested']:<6}  {res_base['num_states_tested']:>6}")
    for k in K_VALUES:
        r = results[k]
        print(f"  {'RNN EM k='+str(k):30s}  {r['violations']:>5}/{r['tested']:<6}  {r['tested']:>6}")

    print("\nViolating states per k:")
    for k in K_VALUES:
        viols = sorted(results[k]["viols"])
        pvals = "  ".join(f"s{s}:{p:.1e}" for s, p in viols)
        print(f"  k={k:2d}: {pvals or '(none)'}")

    # Action dim (flattened)
    a_dim = per_rollout_a[0].shape[1]
    print(f"  a_dim (flat): {a_dim}")

    # 2. Baseline Markov test
    print("\n[2] Baseline Markov test (InfEmbed KMeans c_t) ...")
    res_baseline = test_markov_property(c_all, meta, level="rollout", significance_level=0.05)
    viols_base = [
        (sid, r.p_value) for sid, r in res_baseline["per_state"].items()
        if r.testable and not r.markov_holds
    ]
    print(f"  tested={res_baseline['num_states_tested']}  "
          f"untestable={res_baseline['num_states_untestable']}  "
          f"violations={len(viols_base)}")
    for sid, p in sorted(viols_base):
        print(f"    state {sid:2d}: p={p:.3e}")
    print(f"  markov_holds={res_baseline['markov_holds']}")

    # 3. Train GRU
    print("\n[3] Training GRU encoder ...")
    model = MarkovGRU(z_dim=z_all.shape[1], a_dim=a_dim, h_dim=128)
    model.set_n_clusters(n_clusters)
    train_gru(model, per_rollout_z, per_rollout_a, per_rollout_c,
              num_epochs=150, batch_size=32, device=DEVICE)

    # 4. Extract h_t and cluster
    print("\n[4] Extracting h_t and re-clustering ...")
    h_all = extract_h(model, per_rollout_z, per_rollout_a, device=DEVICE)
    print(f"  h_all shape: {h_all.shape}")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    h_labels = km.fit_predict(h_all).astype(np.int64)
    print(f"  h_labels unique clusters: {len(np.unique(h_labels))}")

    # Save h embeddings and labels
    np.save(_OUT_DIR / "h_embeddings.npy", h_all)
    np.save(_OUT_DIR / "h_cluster_labels.npy", h_labels)
    print(f"  Saved to {_OUT_DIR}")

    # 5. Markov test on h_t clusters
    print("\n[5] Markov test on h_t cluster labels ...")
    res_rnn = test_markov_property(h_labels, meta, level="rollout", significance_level=0.05)
    viols_rnn = [
        (sid, r.p_value) for sid, r in res_rnn["per_state"].items()
        if r.testable and not r.markov_holds
    ]
    print(f"  tested={res_rnn['num_states_tested']}  "
          f"untestable={res_rnn['num_states_untestable']}  "
          f"violations={len(viols_rnn)}")
    for sid, p in sorted(viols_rnn):
        print(f"    state {sid:2d}: p={p:.3e}")
    print(f"  markov_holds={res_rnn['markov_holds']}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline (InfEmbed KMeans):  violations {len(viols_base)}/{res_baseline['num_states_tested']} tested")
    print(f"  RNN h_t KMeans:              violations {len(viols_rnn)}/{res_rnn['num_states_tested']} tested")

    # Per-state comparison
    all_states = sorted(
        set(res_baseline["per_state"]) | set(res_rnn["per_state"])
    )
    print(f"\n  {'state':>5}  {'baseline':>10}  {'rnn_h_t':>10}")
    for s in all_states:
        br = res_baseline["per_state"].get(s)
        rr = res_rnn["per_state"].get(s)
        b_str = f"{br.p_value:.2e}" if (br and br.testable) else ("untestable" if br else "—")
        r_str = f"{rr.p_value:.2e}" if (rr and rr.testable) else ("untestable" if rr else "—")
        print(f"  {s:>5}  {b_str:>10}  {r_str:>10}")


if __name__ == "__main__":
    main()

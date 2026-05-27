"""Run RNN EM (k=10) on jan28 transport data for both infembed and policy_emb.

Writes results directly into the demo clustering directories:
  third_party/influence_visualizer/configs/transport_mh_jan28/clustering/
    infembed_rnn_em_w3_s1_seed0_kmeans_k10/
    policy_emb_rnn_em_w3_s1_seed0_kmeans_k10/
"""
from __future__ import annotations

import json, os, pickle, sys, yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from policy_doctor.behaviors.behavior_graph import test_markov_property

DEVICE = torch.device("cuda:0")

_TRUNKS   = Path("/mnt/ssdB/erik/cupid_data/graph_simplification/trunks")
_EPS_DIR  = Path("/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes"
                 "/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0/latest/episodes")
_SSD_CLU  = Path("/mnt/ssdB/erik/cupid_data/graph_simplification/clusterings")
_IV_CLU   = _REPO / "third_party/influence_visualizer/configs/transport_mh_jan28/clustering"

K          = 10
EM_ITERS   = 5
EPOCHS     = 60
H_DIM      = 128
SEED_K     = "w3_s1_K10"   # which SSD clustering to seed initial c_t from


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ep_file(rid: int) -> Path:
    prefix = f"ep{rid:04d}"
    for name in os.listdir(_EPS_DIR):
        if name.startswith(prefix):
            return _EPS_DIR / name
    raise FileNotFoundError(rid)


def load_jan28(rep: str):
    """Return (z_all, c_init, meta, per_rollout_z, per_rollout_a, per_rollout_c_init)."""
    trunk = _TRUNKS / f"transport_mh_jan28__{rep}"
    z_all = np.load(trunk / "timestep_embeddings.npy").astype(np.float32)  # (N, 50)
    ep_meta = json.load(open(trunk / "ep_meta.json"))
    ep_lens  = ep_meta["episode_lengths"]

    # Build per-timestep metadata matching the w=1 trunk
    meta = []
    for rid, L in enumerate(ep_lens):
        succ = ep_meta["episode_successes"][rid]
        for t in range(L):
            meta.append({"rollout_idx": rid, "window_start": t,
                         "window_end": t+1, "window_width": 1, "success": succ})
    assert len(meta) == len(z_all)

    # Seed c_t from existing w=3,s=1,k=10 clustering
    ssd_dir = _SSD_CLU / f"transport_mh_jan28__{rep}__{SEED_K.replace('_',''[0:0])}".replace("K","__K")
    # handle naming: transport_mh_jan28__infembed__w3_s1__K10
    ssd_dir = _SSD_CLU / f"transport_mh_jan28__{rep}__w3_s1__K{K}"
    c_init   = np.load(ssd_dir / "cluster_labels.npy").astype(np.int64)
    seed_meta = json.load(open(ssd_dir / "metadata.json"))

    # The seed clustering uses w=3 windows, but our trunk is w=1.
    # Map seed labels back to per-timestep by assigning each window's label
    # to its window_start timestep, then forward-fill.
    c_ts = np.full(len(z_all), -1, dtype=np.int64)
    for entry, lbl in zip(seed_meta, c_init):
        rid = entry["rollout_idx"]
        ws  = entry["window_start"]
        off = sum(ep_lens[:rid])
        if off + ws < len(c_ts):
            c_ts[off + ws] = lbl
    # forward-fill within each episode
    pos = 0
    for rid, L in enumerate(ep_lens):
        last = 0
        for t in range(L):
            if c_ts[pos + t] >= 0:
                last = c_ts[pos + t]
            else:
                c_ts[pos + t] = last
        pos += L

    # Load actions aligned per-timestep
    print(f"  [{rep}] loading actions ...")
    per_rollout_z, per_rollout_a, per_rollout_c = [], [], []
    pos = 0
    for rid, L in enumerate(ep_lens):
        ep_df = pickle.load(open(_ep_file(rid), "rb"))
        ep_acts = np.stack([
            np.array(a, dtype=np.float32).reshape(-1)
            for a in ep_df["action"].values
        ])  # (T, 320)
        T = min(L, len(ep_acts))
        per_rollout_z.append(z_all[pos:pos+T])
        per_rollout_a.append(ep_acts[:T])
        per_rollout_c.append(c_ts[pos:pos+T])
        pos += L

    return z_all, c_ts, meta, per_rollout_z, per_rollout_a, per_rollout_c


# ---------------------------------------------------------------------------
# GRU (same as markov_rnn_experiment.py)
# ---------------------------------------------------------------------------

class MarkovGRU(nn.Module):
    def __init__(self, z_dim, a_dim, h_dim=128):
        super().__init__()
        self.h_dim = h_dim
        self.gru = nn.GRU(z_dim + a_dim, h_dim, batch_first=True)
        self.z_head = nn.Linear(h_dim, z_dim)
        self.c_head: Optional[nn.Linear] = None

    def set_n_clusters(self, n):
        self.c_head = nn.Linear(self.h_dim, n).to(
            next(self.parameters()).device if len(list(self.parameters())) else "cpu"
        )

    def forward(self, z, a):
        h, _ = self.gru(torch.cat([z, a], -1))
        return h, self.z_head(h), self.c_head(h) if self.c_head else None


def _phase_contrast(h_t, h_t1, c_t, c_t1, margin=0.5):
    d = 1.0 - F.cosine_similarity(h_t, h_t1, dim=-1)
    same = (c_t == c_t1).float()
    return (same * d + (1 - same) * torch.clamp(margin - d, min=0)).mean()


def _pad(seqs_z, seqs_a, device):
    B, T = len(seqs_z), max(s.shape[0] for s in seqs_z)
    Z = np.zeros((B, T, seqs_z[0].shape[1]), np.float32)
    A = np.zeros((B, T, seqs_a[0].shape[1]), np.float32)
    for i, (z, a) in enumerate(zip(seqs_z, seqs_a)):
        Z[i, :len(z)] = z;  A[i, :len(a)] = a
    return torch.from_numpy(Z).to(device), torch.from_numpy(A).to(device)


def train_one_iter(model, per_z, per_a, per_c, epochs, bs=32):
    model.train()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n    = len(per_z)
    k    = model.c_head.out_features
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s+bs]
            Z, A = _pad([per_z[i] for i in b], [per_a[i] for i in b], DEVICE)
            T = Z.shape[1]
            C = np.zeros((len(b), T), np.int64)
            for j, i in enumerate(b):
                C[j, :len(per_c[i])] = per_c[i]
            C_t = torch.from_numpy(C).to(DEVICE)

            h, z_pred, c_pred = model(Z, A)
            l_z = F.mse_loss(z_pred[:, :-1], Z[:, 1:])
            l_c = F.cross_entropy(c_pred[:, :-1].reshape(-1, k), C_t[:, 1:].reshape(-1))
            l_ct = _phase_contrast(
                h[:, :-1].reshape(-1, model.h_dim), h[:, 1:].reshape(-1, model.h_dim),
                C_t[:, :-1].reshape(-1), C_t[:, 1:].reshape(-1))
            loss = l_z + l_c + 0.5 * l_ct
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep + 1) % 30 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={float(loss):.4f}  "
                  f"z={float(l_z):.4f}  c={float(l_c):.4f}  ct={float(l_ct):.4f}")
    model.eval()


@torch.no_grad()
def extract_h(model, per_z, per_a):
    return np.concatenate([
        model(torch.from_numpy(z).unsqueeze(0).to(DEVICE),
              torch.from_numpy(a).unsqueeze(0).to(DEVICE))[0]
              .squeeze(0).cpu().numpy()
        for z, a in zip(per_z, per_a)
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rep(rep: str):
    print(f"\n{'='*60}\n{rep.upper()}  —  jan28 transport, k={K}, {EM_ITERS} EM iters\n{'='*60}")

    z_all, c_init, meta, per_z, per_a, per_c = load_jan28(rep)
    a_dim = per_a[0].shape[1]
    print(f"  windows={len(z_all)}  rollouts={len(per_z)}  z_dim={z_all.shape[1]}  a_dim={a_dim}")

    # Baseline
    res0 = test_markov_property(c_init, meta, level="rollout", significance_level=0.05)
    v0 = sum(1 for r in res0["per_state"].values() if r.testable and not r.markov_holds)
    print(f"  Baseline (seed k={K}): violations {v0}/{res0['num_states_tested']}")

    model = MarkovGRU(z_all.shape[1], a_dim, H_DIM).to(DEVICE)
    model.set_n_clusters(K)

    c_cur = c_init.copy()
    per_c_cur = per_c

    for it in range(EM_ITERS):
        print(f"\n  EM iter {it+1}/{EM_ITERS}")
        train_one_iter(model, per_z, per_a, per_c_cur, EPOCHS)
        h = extract_h(model, per_z, per_a)
        c_cur = KMeans(K, n_init=10, random_state=42).fit_predict(h).astype(np.int64)
        # rebuild per-rollout c
        per_c_cur = []
        pos = 0
        for z in per_z:
            T = len(z); per_c_cur.append(c_cur[pos:pos+T]); pos += T
        res = test_markov_property(c_cur, meta, level="rollout", significance_level=0.05)
        vn = sum(1 for r in res["per_state"].values() if r.testable and not r.markov_holds)
        print(f"    violations {vn}/{res['num_states_tested']}")

    # Final
    h_final = extract_h(model, per_z, per_a)
    res_final = test_markov_property(c_cur, meta, level="rollout", significance_level=0.05)
    vf = sum(1 for r in res_final["per_state"].values() if r.testable and not r.markov_holds)
    print(f"\n  FINAL: baseline {v0}/{res0['num_states_tested']} → EM {vf}/{res_final['num_states_tested']}")

    # Write demo clustering directory
    out_name = f"{rep}_rnn_em_w3_s1_seed0_kmeans_k{K}"
    out_dir  = _IV_CLU / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cluster_labels.npy",     c_cur)
    np.save(out_dir / "embeddings_reduced.npy", h_final)
    # copy metadata from the w=1 trunk (same windows)
    import shutil
    shutil.copy(_SSD_CLU / f"transport_mh_jan28__{rep}__w3_s1__K{K}" / "metadata.json",
                out_dir / "metadata.json")
    manifest = {
        "algorithm": "kmeans_on_gru_h",
        "influence_source": rep,
        "level": "rollout",
        "n_clusters": K,
        "n_samples": int(len(c_cur)),
        "representation": "sliding_window",
        "window_width": 3,
        "stride": 1,
        "aggregation": "mean",
        "umap_n_components": H_DIM,
        "scaling": "none",
        "task_config": "transport_mh_jan28",
        "rnn_em_iters": EM_ITERS,
        "rnn_h_dim": H_DIM,
        "rnn_z_source": rep,
    }
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    print(f"  Saved → {out_dir}")
    return v0, res0["num_states_tested"], vf, res_final["num_states_tested"]


if __name__ == "__main__":
    results = {}
    for rep in ["infembed", "policy_emb"]:
        v0, t0, vf, tf = run_rep(rep)
        results[rep] = (v0, t0, vf, tf)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  {'rep':12s}  {'baseline':>12}  {'rnn_em_k10':>12}")
    for rep, (v0, t0, vf, tf) in results.items():
        print(f"  {rep:12s}  {v0:>4}/{t0:<6} {v0/t0:.0%}  {vf:>4}/{tf:<6} {vf/tf:.0%}")

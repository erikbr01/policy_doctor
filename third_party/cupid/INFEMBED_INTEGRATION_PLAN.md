# InfEmbed Integration Plan

This document outlines the plan to integrate [InfEmbed](https://arxiv.org/abs/2312.04712) (Error Discovery By Clustering Influence Embeddings) into the CUPID influence visualizer, and how it relates to the existing TRAK-based pipeline.

## 1. Summary: TRAK vs InfEmbed

| Aspect | TRAK (current) | InfEmbed |
|--------|----------------|----------|
| **Goal** | Attribution: score how much each training sample influences test (rollout) loss | Slice discovery: partition test/rollout data into coherent groups "wrong for the same reason" |
| **Mechanism** | Random projection (JL) of gradients → store features; scores = Φ·target_features | Low-rank Hessian inverse (Arnoldi) → embeddings μ(z) s.t. μ(z')·μ(z) = influence(z',z) |
| **Output** | Influence matrix (rollout_samples × demo_samples) | Low-dim embeddings (e.g. D=100) per sample; dot product = influence |
| **Clustering** | Current UI: build embeddings by aggregating TRAK matrix (high-dim) then PCA/UMAP + K-Means | InfEmbed paper: K-Means directly on influence embeddings (already low-dim) |

## 2. What We Reuse from TRAK

- **Model & checkpoint**: Same policy load (`get_policy_from_checkpoint`), same `cfg`.
- **Datasets**: Same train/holdout instantiation (`hydra.utils.instantiate(cfg.task.dataset)`), same rollout dataset (`BatchEpisodeDataset(eval_dir/episodes)`).
- **Loss**: Same diffusion loss (e.g. DDPM) used for gradients.
- **Gradient computation**: Same parameter set (`grad_wrt`) and same layers; we need ∇L(z;θ) per sample. InfEmbed then applies a *different* linear map (Hessian-based) than TRAK (random projection).

We **cannot** reuse TRAK’s projected features as InfEmbed embeddings: TRAK uses a random projector and (X^T X + λI)^{-1} for scoring; InfEmbed uses a low-rank Hessian inverse so that dot(embed_i, embed_j) = influence(i,j). So we need a separate computation path that produces InfEmbed embeddings.

## 3. What We Add

### 3.1 Script: Compute InfEmbed embeddings

- **Location**: e.g. `scripts/train/compute_infembed_embeddings.py` (or `influence_visualizer/scripts/compute_infembed_embeddings.py`).
- **Inputs**: Same as `train_trak_diffusion.py` (eval_dir, train_dir, train_ckpt, model_id, loss_fn, num_timesteps, batch_size, etc.).
- **Steps**:
  1. Load policy and config (same as TRAK).
  2. Build train and rollout DataLoaders in the **same order** as TRAK (no shuffle).
  3. Wrap the diffusion loss so InfEmbed’s embedder sees batches of the form expected by `ArnoldiEmbedder` (model, loss_fn, DataLoader). This may require a small adapter: a PyTorch `Dataset`/`DataLoader` that yields (batch_dict, loss_value_or_labels) and a `loss_fn` that computes per-sample loss for the diffusion policy.
  4. **Fit** InfEmbed’s `ArnoldiEmbedder` on the **training** dataloader (Hessian approximation from training data only).
  5. **Predict** embeddings for:
     - All training samples (in TRAK order),
     - All rollout samples (in TRAK order).
  6. Save to `eval_dir / <trak_exp_name> / infembed_embeddings.npz`:
     - `rollout_embeddings`: shape `(n_rollout_samples, D)`
     - `demo_embeddings`: shape `(n_demo_samples, D)` with train samples first, then holdout (if any), same indexing as TRAK.
     - Optional: `embedding_dim`, `train_set_size`, `test_set_size` for validation.

- **Dependencies**: Use `third_party/infembed` (ArnoldiEmbedder). Policy and data loading from existing CUPID/diffusion_policy code; gradient/loss from same modules as TRAK where possible.

**Implementation:** Script `compute_infembed_embeddings.py` (project root) and adapter `diffusion_policy/data_attribution/infembed_adapter.py`. Run with `--exp_name=auto` to detect the TRAK experiment under `eval_dir`. Optional: `bash scripts/train/compute_infembed.sh` (set dates/task to match TRAK).

**How to run:**
1. Run TRAK first (e.g. `bash scripts/train/train_trak.sh`) so `eval_dir` and the TRAK experiment directory exist.
2. From the **repo root**, with the same conda/env as for TRAK:  
   `python compute_infembed_embeddings.py --exp_name=auto --eval_dir=<path> --train_dir=<path> --modelout_fn=DiffusionLowdimFunctionalModelOutput --loss_fn=square --num_timesteps=64`  
   The script adds `third_party/infembed` to `sys.path` so InfEmbed is found without installing it. If InfEmbed’s own deps (e.g. `dill`) are missing, install them or `pip install -e third_party/infembed`.
3. In the influence visualizer, open Clustering → 7. Clustering Algorithms → Representation **InfEmbed** → Extract Embeddings (loads from `eval_dir/<trak_exp>/infembed_embeddings.npz`).

### 3.2 Adapter: Diffusion policy → InfEmbed

InfEmbed’s `ArnoldiEmbedder` expects:

- `model`: PyTorch module.
- `loss_fn`: Callable such that loss has a `reduction` attribute ('sum' or 'mean' for per-example grads).
- `DataLoader`: Yields batches; in `predict`, the embedder uses `batch[0:-1]` as features and `batch[-1]` as labels. For diffusion we don’t have a simple (x, y); we have a dict (obs, action, timesteps, etc.). So we need either:
  - A **wrapper DataLoader** that yields `(batch_dict, dummy_labels)` and a **wrapper loss_fn** that ignores the second argument and computes `task.get_output(model, ...)` on the batch_dict (with timesteps sampled as in TRAK), with `reduction='none'` or per-sample loss, or
  - A thin **custom embedder** that uses the same gradient computer as TRAK and only replaces the projection step with InfEmbed’s Hessian-based projection (more invasive).

Recommended: wrapper DataLoader + wrapper loss_fn that calls the existing diffusion modelout function and returns per-sample loss, then use InfEmbed’s ArnoldiEmbedder with that. InfEmbed’s embedder needs to compute gradients w.r.t. the same parameters as TRAK (`layers` or `grad_wrt`).

### 3.3 Visualizer: Clustering section

- **InfluenceData**: Add optional `trak_exp_name: Optional[str] = None` and set it in `load_influence_data` when we resolve the TRAK experiment. This allows the UI to look for InfEmbed files next to the TRAK results.
- **Clustering UI (Step 1 – Representation)**:
  - Add a third option: **"InfEmbed"** (label e.g. "InfEmbed (low-dim influence embeddings)").
  - When "InfEmbed" is selected:
    - **Level** (rollout vs demo) and **split** (train/holdout/both) behave as now.
    - "Extract Embeddings" tries to load `data.eval_dir / data.trak_exp_name / "infembed_embeddings.npz"`.
    - If found: load `rollout_embeddings` and `demo_embeddings`, slice demo by split, aggregate by episode (mean over timesteps) to get one vector per rollout or per demo, and store in session state. Embeddings are already low-dim (e.g. D=100); no extra PCA/UMAP needed unless we want to visualize in 2D.
    - If not found: show a short message: "InfEmbed embeddings not found. Run the InfEmbed script first (see INFEMBED_INTEGRATION_PLAN.md)."
  - When "Sliding Windows" or "Individual Timesteps" is selected, behavior stays as now (TRAK matrix–based embeddings).

- **Step 2 (Clustering algorithm)**: Added **"InfEmbed (K-Means on influence embeddings)"** to the algorithm dropdown. It runs K-Means with the same parameters; use it together with representation **InfEmbed** for the full paper method. Other algorithms (DBSCAN, Spectral, etc.) remain available on any representation.

## 4. File layout (convention)

- TRAK results: `eval_dir / <trak_exp_name> /` (scores, metadata, etc.).
- InfEmbed results: `eval_dir / <trak_exp_name> / infembed_embeddings.npz` (same experiment dir so we know they align with the same TRAK run and data order).

## 5. Ordering and alignment

- Demo samples: same order as in `train_trak_diffusion.py` (train loader, then holdout loader if present). Episode boundaries from `data.demo_episodes` and `data.holdout_episodes` (and `build_demo_sample_infos`).
- Rollout samples: same order as `BatchEpisodeDataset` in TRAK scoring. Episode boundaries from `data.rollout_episodes`.
- Aggregation to episode level: same as current `extract_demo_embeddings` / `extract_rollout_embeddings` (mean over timesteps per episode).

## 6. Optional: InfEmbed-Rule

The paper also describes InfEmbed-Rule (find slices satisfying a rule: accuracy &lt; A, size &gt; S). We can add this later as an extra clustering or filtering mode (e.g. "Discover slices with low accuracy" that runs RuleFind on the same embeddings).

## 7. References

- Paper: [Error Discovery By Clustering Influence Embeddings](https://arxiv.org/abs/2312.04712) (Wang, Adebayo et al.).
- Code: `third_party/infembed/` (ArnoldiEmbedder in `infembed/embedder/_core/arnoldi_embedder.py`).
- CUPID README: Stages 1–3 (train → eval rollouts → TRAK featurize + score).


---

# Project Specification: Robot Failure Mode Discovery via Influence Embeddings

## 1. Project Overview & Goal

**Objective:** Build a pipeline that automatically groups robotic failures into interpretable clusters (e.g., "Missed Grasp," "Wall Collision") and identifies which training demonstrations are responsible for those failures.

**The "Big Idea":**
We are combining two state-of-the-art techniques:

1. **CUPID (Robotics):** Uses the outcome of a robot's test run (Success vs. Failure) to query the training data.
2. **InfEmbed (Vision):** Uses vector embeddings of gradients to cluster data based on "how it affects the model," rather than just visual similarity.

**Output:** A script that takes a trained policy and dataset, and outputs:

1. Clusters of failure modes.
2. The top-ranked "suspect" training demonstrations for each failure mode.

---

## 2. Mathematical & Theoretical Context

*Since you do not have access to the papers, implement the following logic exactly.*

### A. The Influence Approximation

We determine the relationship between a **Test Rollout** (Query) and a **Training Demo** (Source) using an influence score.
The core formula for the influence of training point  on test point  is:

Where:

* : The gradient of the loss function with respect to the model parameters.
* : The inverse Hessian (curvature matrix), which dictates how gradients propagate.

### B. The "Embedding" Trick

Calculating the Hessian is too expensive. We approximate the term  using a low-rank projection matrix . This allows us to define an **Influence Embedding** vector  for any datapoint:

The influence score becomes a simple dot product:


### C. Handling Trajectories (The "Time" Dimension)

Robotics data is sequential. A trajectory  consists of  timesteps.

1. **Compute per-step embeddings:** .
2. **Aggregate:** Sum them up to get a trajectory embedding.


3. **Reward Weighting (CRITICAL):**
For **Test Rollouts** only, we multiply the embedding by the reward .
* If the robot **Failed** (), the vector flips direction.
* This forces the "Failure" vector to align with training data that *pushed* the model toward that failure.



---

## 3. Implementation Options

*Select Option A for this implementation. It is more robust for a coding agent.*

* **Option A: Random Projections (The "TRAK" method)**
* **Technique:** Instead of calculating the real Hessian, we use the Johnson-Lindenstrauss lemma. We project the high-dimensional gradients onto a fixed, random Gaussian matrix.
* **Why:** It is extremely fast, low memory, and mathematically proven to preserve distances (and thus influence scores).


* **Option B: Arnoldi Iteration (The "InfEmbed" method)**
* **Technique:** Iteratively compute eigenvectors of the Hessian.
* **Why:** Higher precision, but very difficult to implement for large Diffusion Policies without running out of GPU memory.



---

## 4. Step-by-Step Implementation Plan

### Phase 1: The Gradient Projector (The Engine)

**Goal:** Create a class that computes .

**Requirements:**

* Use `torch.func` (specifically `vmap` and `grad` or `functional_call`) to efficiently compute per-sample gradients. **Do not** use a standard `loss.backward()` loop, it will be too slow.

**Pseudocode Logic:**

```python
class GradientProjector:
    def __init__(self, model, layer_name, projection_dim=512):
        # 1. Identify the target layer (usually the bottleneck or final layer)
        # 2. Create a static random matrix P of shape [layer_params, projection_dim]
        self.P = torch.randn(layer_params.numel(), projection_dim).to(device)

    def get_embedding(self, state, action):
        # 1. Define a functional version of the model loss
        def compute_loss(params, s, a):
            pred = functional_call(model, params, s)
            return loss_fn(pred, a)

        # 2. Compute per-sample gradient
        # use torch.func.vmap(torch.func.grad(compute_loss))
        grads = compute_grads(model.params, state, action)

        # 3. Flatten grads and project
        # embedding = grads @ self.P
        return embedding

```

### Phase 2: Trajectory Aggregation

**Goal:** Process full sequences.

**Logic:**

1. Input: A batch of trajectories `[Batch, Time, Obs_Dim]`.
2. Loop or vectorization: Get embeddings for all `(s, a)` pairs.
3. **Masking:** If trajectories are padded, ensure you apply a mask so you don't sum up gradients for padding tokens (zeros).
4. **Aggregation:** Sum the vectors over the Time dimension.
5. **Normalization:** Normalize the final vector to unit length ( norm). This is crucial for K-Means clustering later.

### Phase 3: The Data Processing Loop

**Goal:** Create the two main embedding datasets.

1. **Process Training Data ():**
* Iterate through all successful expert demonstrations.
* Compute .
* Save tensor: `train_embeddings.pt` `[Num_Demos, Proj_Dim]`.


2. **Process Evaluation Data ():**
* Iterate through rollout data (which includes failures).
* Compute .
* **Apply Reward:** `final_vec = mu_traj * rollout_reward`.
* Save tensor: `eval_embeddings.pt` `[Num_Rollouts, Proj_Dim]`.



### Phase 4: Discovery (Clustering)

**Goal:** Find the failure modes.

**Logic:**

1. Filter `eval_embeddings` to keep only the **Failures** (where Reward = -1).
2. Run **K-Means Clustering** (e.g.,  or ) on these failure vectors.
3. Each Cluster Center represents a "Type of Error."

### Phase 5: Attribution (Linking back to Training Data)

**Goal:** "Who caused this error?"

**Logic:**
For each Cluster Center :

1. Compute Cosine Similarity between  and *all* `train_embeddings`.
2. Select the top  training examples with the highest similarity.
3. **Result:** These training examples are the ones effectively "voting" for that specific type of failure.

---

## 5. Technical Constraints & Tips for the Agent

1. **Model Architecture:** The system should work for a Diffusion Policy (U-Net based) or a standard MLP BC policy. If using Diffusion, compute the gradient of the **Noise Prediction Loss**.
2. **Memory Management:** The gradient of a whole trajectory can be large. Do not store the full gradients. Project them immediately into the lower dimension (e.g., 512) before storing.
3. **Layer Selection:** You do not need gradients for every parameter in the network.
* *Tip:* Select only the parameters of the **last 2 layers** of the policy network. This captures most semantic information while drastically reducing computation.


4. **Batching:** Ensure the `vmap` logic handles batches correctly. If the model runs out of memory, reduce the batch size used during the embedding generation phase.

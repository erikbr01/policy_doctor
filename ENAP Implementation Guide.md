# Emergent Neural Automaton Policies (ENAP): Architecture & Implementation Guide

This document provides a comprehensive breakdown of the ENAP framework, followed by a detailed, module-by-module implementation plan designed for an autonomous coding agent or software engineer.

---

## Part 1: ENAP Framework Outline

ENAP bridges the gap between high-level discrete reasoning (System 2) and low-level continuous control (System 1) by unsupervisedly extracting a Probabilistic Mealy Machine (PMM) from robotic demonstrations and coupling it with a continuous residual network.

### 1. Adaptive Symbol Abstraction (Discretizing the World)
* **Goal:** Convert continuous, high-dimensional observations into discrete symbols representing semantic task phases.
* **Process:**
    1.  Observations $o_t$ (images $I_t$, proprioception $p_t$) are encoded into feature vectors $z_t = \phi_\theta(o_t)$.
    2.  **HDBSCAN** clusters these features to create a discrete input alphabet $\Sigma$. Each step is assigned a cluster label $c_t \in \Sigma$.

### 2. Markovian History Encoding (RNN)
* **Goal:** Handle the variable temporal length of physical task phases.
* **Process:**
    1.  An RNN ingests the history of actions and symbols $(a_{0:t}, c_{0:t})$ and outputs a continuous hidden state $h_t$.
    2.  **Multi-Objective Loss:**
        * $\mathcal{L}_{act}$ (MSE): Predicts next action $a_{t+1}$.
        * $\mathcal{L}_{state}$ (Cross-Entropy): Predicts next symbol $c_{t+1}$.
        * $\mathcal{L}_{contrast}$ (Contrastive): Minimizes distance between $h_t, h_{t+1}$ if $c_t = c_{t+1}$ (self-loop); maximizes distance if $c_t \neq c_{t+1}$ (phase transition).

### 3. Structure Extraction (Extended $L^*$ Algorithm)
* **Goal:** Build the Probabilistic Mealy Machine (PMM) offline from the demonstration database $\mathcal{D}$.
* **Process:**
    1.  **Nodes (States):** Defined by grouping similar $h_t$ embeddings (using a similarity threshold $\tau_{sim}$). The algorithm expands its "Prefix Set" until *Closedness* is satisfied.
    2.  **Edges (Transitions):** Triggered by an observation symbol $c_t$. The algorithm calculates empirical transition probabilities between nodes and establishes a **Coarse Action Prior** ($a_{base}$), which is the empirical mean of continuous actions taking that specific transition.
    3.  **Non-deterministic EQ:** Tests the hypothesis graph against trajectories. If a trajectory cannot be explained by the graph, it adds the prefix as a counterexample and splits nodes.
    4.  **Pruning:** Merges redundant stable phases.

### 4. Iterative Residual Learning (EM Loop)
* **Goal:** Train the low-level controller and optimize the visual encoder (which can't be trained through HDBSCAN directly).
* **Process:**
    * **E-Step:** Freeze the encoder, run HDBSCAN, train the RNN, and extract the PMM.
    * **M-Step:** Freeze the PMM. Train a lightweight Residual Policy Network $\pi_\psi$ via Behavior Cloning to predict $\Delta a_t$ (where final action $\hat{a}_t = a_{base}^t + \Delta a_t$). Backpropagate this control loss into the visual encoder $\phi_\theta$ to improve features for the next E-step.

---

## Part 2: Detailed Implementation Plan for Coding Agents

This section outlines the software architecture, data structures, and training loops required to build ENAP from scratch.

### Module 1: Data & Perception (`perception.py`)

**Objective:** Handle raw datasets, visual encoding, and clustering.

1.  **Encoder Architectures ($\phi_\theta$):**
    * *Implementation:* Depending on the mode, initialize a frozen Vision Backbone (e.g., `DINOv2 ViT-S/14` or `Florence-2`).
    * *Visual Head:* Implement an MLP (e.g., Spatial Softmax -> Linear -> ReLU -> Linear outputting 128/256 dims).
    * *Proprioception Head:* Fourier feature mapping followed by Spectral Normalization and a Linear layer.
    * *Fusion:* Concatenate Visual and Proprioception features into a single vector $z_t$.
2.  **Clustering Engine (HDBSCAN):**
    * *Implementation:* Use `hdbscan` from `scikit-learn-contrib`.
    * *Wrapper Class:* Needs a method `fit_predict(dataset_features)` that stores the cluster centroids ($\mu_c$) and returns the discrete symbol assignments $c_t$ for all trajectories.

### Module 2: State Representation (`rnn_encoder.py`)

**Objective:** Compress sequences into Markovian embeddings.

1.  **Architecture:**
    * *Inputs:* Concatenated $[a_t, c_t]$. Note: $c_t$ should be passed through a learnable Embedding layer (e.g., 16 dimensions).
    * *Core:* A standard GRU or LSTM cell (e.g., hidden dimension 64).
    * *Heads:* Two linear layers projecting from $h_t$ to predict $\hat{a}_{t+1}$ and $\hat{c}_{t+1}$.
2.  **Custom Loss Function (Crucial):**
    * *Implementation:* The coding agent must implement the phase-aware contrastive loss meticulously.
    ```python
    def compute_contrastive_loss(h_t, h_t_plus_1, c_t, c_t_plus_1):
        cos_sim = cosine_similarity(h_t, h_t_plus_1)
        # If c_t == c_t_plus_1, minimize distance (maximize cos_sim)
        # If c_t != c_t_plus_1, maximize distance (minimize cos_sim)
        # Apply standard margin-based contrastive loss logic here.
    ```

### Module 3: Automaton Extraction (`extended_l_star.py`)

**Objective:** The core logic engine. This requires careful object-oriented design.

1.  **Data Structures:**
    * `class Node:` Stores its history embedding centroid $u$, a dictionary of outgoing `Edge` objects, and a unique ID.
    * `class Edge:` Stores the target `Node` ID, the triggering input symbol $c$, the calculated transition probability, the calculated **Action Prior** ($a_{base}$), and the Next-Input Set ($NIS$).
    * `class PMM:` Holds a list of `Node` objects and the start node. Contains the forward-pass inference logic: `step(current_node, symbol) -> (next_node, a_base)`.
2.  **Algorithm 1 Implementation (`ExtendedLStar` class):**
    * `_membership_query(prefix_u)`: Scans the dataset $\mathcal{D}$. Finds all $h_t$ within $\tau_{sim}$ of $u$. Returns empirical transitions.
    * `_check_closedness()`: Ensures all discovered destination embeddings map to an existing centroid in the Prefix Set $\mathcal{U}$. If not, expands $\mathcal{U}$.
    * `_equivalence_query(test_trajectories)`: Simulates the trajectory through the hypothesized PMM. If at step $t$, no valid edge exists that matches the observation symbol AND the physical action falls within a tolerance of $a_{base}$, flag as failure. Return the prefix as a counterexample.
    * `build_graph()`: The main while-loop managing MQ and EQ until stabilization.

### Module 4: Residual Control & EM Loop (`trainer.py`)

**Objective:** Tie the graph to continuous control and execute the alternating EM loop.

1.  **Residual Policy Network ($\pi_\psi$):**
    * *Architecture:* A lightweight MLP.
    * *Inputs:* Current Node ID (embedded), current continuous feature vector $z_t$, and the coarse action prior $a_{base}^t$ provided by the PMM.
    * *Output:* Predicts $\Delta a_t$. Final action is $\hat{a}_t = a_{base}^t + \Delta a_t$.
2.  **The EM Training Script (Algorithm 2):**
    * *Initialization:* Load dataset, initialize visual encoder $\phi_\theta$ and residual network $\pi_\psi$.
    * *Outer Loop (k = 0 to K):*
        * **E-Step:**
            1. Extract all features using current $\phi_\theta$.
            2. Run HDBSCAN to assign symbols $c_t$.
            3. Train the RNN on the newly generated sequence of symbols and actions.
            4. Run `ExtendedLStar` to build and freeze the PMM $\mathcal{M}^{(k+1)}$.
        * **M-Step:**
            1. Freeze the PMM. Set Residual Net and Visual Encoder to `train()` mode.
            2. Iterate over batches of the dataset: $(o_t, a_t, q_t)$.
            3. Query the frozen PMM for $a_{base}$.
            4. Forward pass the Residual Net to get $\Delta a_t$.
            5. Compute Behavior Cloning Loss: $\mathcal{L}_{BC} = ||a_t - (a_{base} + \Delta a_t)||^2$.
            6. Compute Regularization: $\mathcal{L}_{center} = ||\phi_\theta(o_t) - \mu_{c_t}||^2$.
            7. Backpropagate total loss $\mathcal{J}$ to update **both** $\pi_\psi$ and $\phi_\theta$.

### Development Milestones for the Agent

* **Milestone 1:** Implement and unit-test the Custom RNN with the contrastive loss function using synthetic sequential data.
* **Milestone 2:** Implement the `ExtendedLStar` class. Unit test by passing a synthetic dataset of embeddings and discrete symbols to ensure it successfully builds a valid, closed graph.
* **Milestone 3:** Implement the Visual Encoder and HDBSCAN pipeline.
* **Milestone 4:** Integrate all components into the EM training loop and execute on a small robotic dataset (e.g., a simple push task).

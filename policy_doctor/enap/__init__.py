"""ENAP (Emergent Neural Automaton Policies) graph-building modules.

This package implements the full ENAP framework from:

    Pan, Luo et al. "Emergent Neural Automaton Policies: Learning Symbolic
    Structure from Visuomotor Trajectories." arXiv:2603.25903 (2026).

Pipeline stages:

**E-step** — PMM graph extraction from rollout data:

1. **Perception** (:mod:`perception`): encode observations with a visual
   backbone + proprioception fusion, then run HDBSCAN to produce discrete
   observation symbols ``c_t``.

2. **RNN encoding** (:mod:`rnn_encoder`): train a vanilla RNN
   (:class:`~rnn_encoder.PretrainRNN`) with Prioritized Experience Replay
   (:class:`~rnn_encoder.PrioritizedReplayBuffer`) and phase-aware contrastive
   loss to produce Markovian history embeddings ``h_t``.

3. **PMM learning** (:mod:`pmm`): faithful port of the ENAP repository's
   ``agent/pmm_class.py`` — runs Extended L* on ``(h_t, c_t)`` sequences to
   extract a Probabilistic Mealy Machine.

**M-step** — residual policy refinement:

4. **Residual MLP** (:mod:`residual_policy`): train
   :class:`~residual_policy.ResidualMLP` to refine PMM action priors given
   current visual context.  :class:`~residual_policy.PMMAgent` wraps PMM +
   ResidualMLP for closed-loop deployment.

A :mod:`graph_adapter` converts the PMM into the shared
:class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph` format used by
all downstream pipeline steps.
"""

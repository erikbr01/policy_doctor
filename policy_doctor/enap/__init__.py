"""ENAP (Emergent Neural Automaton Policies) graph-building modules.

This package implements the E-step of the ENAP framework from:

    Pan, Luo et al. "Emergent Neural Automaton Policies: Learning Symbolic
    Structure from Visuomotor Trajectories." arXiv:2603.25903 (2026).

The E-step extracts a Probabilistic Mealy Machine (PMM) from demonstration
data via three stages:

1. **Perception** (:mod:`perception`): encode observations with a frozen
   visual backbone (DINOv2) + proprioception fusion, then run HDBSCAN to
   produce discrete observation symbols ``c_t``.

2. **RNN encoding** (:mod:`rnn_encoder`): train a GRU that ingests
   ``(a_t, c_t)`` history and outputs Markovian hidden states ``h_t``, using
   a phase-aware contrastive loss to make ``h_t`` cluster cleanly by task phase.

3. **Structure extraction** (:mod:`extended_l_star`): run the Extended L*
   algorithm on the ``h_t`` embeddings and ``c_t`` symbols to extract a PMM
   whose nodes correspond to stable task phases.

A :mod:`graph_adapter` converts the extracted PMM into the shared
:class:`~policy_doctor.behaviors.behavior_graph.BehaviorGraph` format used by
all downstream pipeline steps.
"""

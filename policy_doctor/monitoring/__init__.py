"""Runtime monitor: real-time attribution and behavior graph node assignment.

The monitor pipeline for a streaming test-time sample:
1. **Gradient + projection** (requires model in memory, cupid env):
   - TRAK path: JL-project the per-sample gradient → compare with cached
     ``features.mmap`` (N_train × proj_dim).
   - InfEmbed path: project gradient through saved Arnoldi eigenvectors →
     compare with cached ``demo_embeddings`` (N_demo × proj_dim).
2. **Graph assignment** (pure numpy, policy_doctor env):
   - Compute nearest centroid of the InfEmbed rollout embeddings.
   - Map centroid index to a behavior graph node.

Quick start::

    from policy_doctor.monitoring import InfEmbedStreamScorer, NearestCentroidAssigner, StreamMonitor

    scorer = InfEmbedStreamScorer(
        checkpoint="...checkpoints/latest.ckpt",
        infembed_fit_path=".../infembed_fit.pt",
        infembed_embeddings_path=".../infembed_embeddings.npz",
    )
    assigner = NearestCentroidAssigner.from_paths(
        rollout_embeddings=scorer.rollout_embeddings,
        clustering_dir=".../clustering/<name>",
        graph=behavior_graph,
    )
    monitor = StreamMonitor(scorer=scorer, assigner=assigner)
    result = monitor.process_sample(obs=obs_array, action=action_array)
    print(result.assignment.node_name, result.timing_ms)
"""

from policy_doctor.monitoring.base import AssignmentResult, GraphAssigner, MonitorResult, StreamScorer
from policy_doctor.monitoring.graph_assigner import NearestCentroidAssigner
from policy_doctor.monitoring.stream_monitor import StreamMonitor

__all__ = [
    "StreamScorer",
    "GraphAssigner",
    "AssignmentResult",
    "MonitorResult",
    "NearestCentroidAssigner",
    "StreamMonitor",
]

# Heavy imports (diffusion_policy / trak / infembed) are deferred to the scorer
# classes themselves so that the monitoring package can be imported in the
# ``policy_doctor`` conda env without the cupid sim stack.
try:
    from policy_doctor.monitoring.trak_scorer import TRAKStreamScorer
    __all__.append("TRAKStreamScorer")
except ImportError:
    pass

try:
    from policy_doctor.monitoring.infembed_scorer import InfEmbedStreamScorer
    __all__.append("InfEmbedStreamScorer")
except ImportError:
    pass

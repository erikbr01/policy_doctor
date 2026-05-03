"""Tool surface for the agentic proposal loop.

Layered design (Section 4 of the spec):

* :mod:`.topology` — Layer 1: graph topology (cheap, broad). No images.
* :mod:`.access` — Layer 2: slice and rollout access. May return images
  (counted against the visual budget).
* :mod:`.analysis` — Layer 3: search and aggregation. Cheap, computed.
* :mod:`.submission` — Layer 4: the agent's output channel.
* :mod:`.no_graph` — parallel surface for the A_NG condition. Same underlying
  rollouts and videos, but cluster-level information is stripped.

Every tool returns a :class:`policy_doctor.vlm.proposals.agents.tools.types.ToolResult`
with provider-neutral content blocks. The session loop (not the tool) is
responsible for translating these into backend-specific message format.
"""

"""Agentic proposal generation for Experiment E2.

Replaces the one-shot proposal flow in :mod:`policy_doctor.vlm.proposals.propose`
with a tool-use loop. The agent explores the behavior graph (or a parallel
graph-free surface for the A_NG condition) via structured tool calls and
submits :class:`DemonstrationRequest` objects through the same downstream
pipeline (operator -> adherence -> retrain -> evaluate) that the one-shot path
uses.

Sub-packages:

* :mod:`policy_doctor.vlm.proposals.agents.tools` — the four tool layers plus
  the no-graph parallel surface, all built on top of :class:`BehaviorGraph`
  and :class:`RolloutPool`.
* :mod:`policy_doctor.vlm.proposals.agents.session` — backend-agnostic
  tool-use loop (``AgentSession.run``).
* :mod:`policy_doctor.vlm.proposals.agents.context` — :class:`SessionContext`
  carries graph, pool, classifier, budget, submission state, trace.
* :mod:`policy_doctor.vlm.proposals.agents.budget` — call/visual/video budget
  tracker plus content-addressed result cache.
* :mod:`policy_doctor.vlm.proposals.agents.trace` — JSONL per-call trace.
* :mod:`policy_doctor.vlm.proposals.agents.system_prompts` — frozen prompts.
"""

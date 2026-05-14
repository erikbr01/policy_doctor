# Experiment E2: Agentic Proposals for Demonstration Collection

## Purpose

E1 established that influence-based clusters carry behavioral meaning. E2 asks the next question: **does giving an LLM agent a structured tool surface over the behavior graph help it propose better demonstration-collection actions than giving it the same underlying rollouts without graph access?**

In the agentic mode (the default), an LLM with tool access explores the behavior graph (or, in the no-graph condition, an outcome-only parallel surface) and submits `DemonstrationRequest`s through that tool surface. A trained operator executes the requests in the sim. Each demonstration is scored automatically against the saved behavior graph (`TrajectoryClassifier`). Curated demos retrain the policy. Success rates are then compared across conditions.

The legacy **one-shot** prompting mode (one VLM batch per condition, no tool use) is retained as an ablation under `mode: one_shot`; everything in this doc except §"Conditions" and §"Tool surface" applies to it unchanged.

The key automation is unchanged from the one-shot framing: per-demonstration **adherence scoring** is done by classifying the new demo into the existing behavior graph and checking whether the realized cluster path matches what was specified. No human grading.

## Conditions

Four pre-registered conditions. Three is the minimum defensible design; the fourth (`H_G`) strengthens attribution:

| Condition | Reasoner | Graph access | Tool surface |
|-----------|----------|--------------|--------------|
| `A_G`     | LLM agent | yes | full Layer 1–4 graph tools |
| `A_NG`    | LLM agent | no  | parallel no-graph tools (rollouts + outcomes only) |
| `H_NG`    | Trained operator | no | Streamlit rollout browser; same submission schema |
| `H_G` *(optional)* | Trained operator | yes | Streamlit graph + per-cluster examples |

Comparisons:
- `A_G` vs `A_NG` — value of graph access to an LLM
- `H_G` vs `H_NG` — value of graph access to a human
- `A_G` vs `H_G` — comparable reasoning given graph access
- `A_NG` vs `H_NG` — comparable reasoning without graph access

Both vocabularies coexist in the codebase: `parse_condition` maps the legacy `"graph"` / `"outcome_only"` strings (used by the one-shot path) to `A_G` / `A_NG` so old configs and tests keep working.

## Hypotheses

- **H1 (Primary)**: Policies retrained on `A_G`-condition demonstrations have higher evaluation success rate than those retrained on `A_NG` demonstrations.
- **H2 (Process)**: `A_G` demonstrations have higher cluster-adherence scores.
- **H3 (Coverage)**: `A_G` demonstrations cover a wider region of the behavior graph (more distinct clusters traversed).
- **H4 (Reasoning attribution)**: Where `H_G` is run, comparing `A_G` vs `H_G` and `A_NG` vs `H_NG` isolates the contribution of agent vs human reasoning at fixed graph access.

## Tool surface (agentic mode)

The agent reasons about the policy and constructs a strategy using *only* these tools. Tool design directly determines what the agent can express, so the surface is committed and hash-tracked in `pre_registration.yaml`.

### Layer 1 — graph topology (cheap, broad, no images)
- `get_graph_summary` — n nodes, terminal probabilities, V-value range, pool outcome counts. Almost always the agent's first call.
- `list_nodes(min_failure_likelihood, min_v, max_v)` — filterable list with V, failure likelihood, in/out degree, episode count.
- `list_paths(from_node, to_node, top_k)` — top-k highest-probability paths between two nodes.
- `get_node(node_id)` — full node info incl. a textual `kinematic_summary` computed from raw state trajectories. **Read this before spending visual budget.**
- `get_edge(from_node, to_node)` — probability, count, advantage, example rollouts.

### Layer 2 — slice and rollout access (visual; budgeted)
- `list_slices_in_node(node_id, n, sort_by)` — slice ids, ordered by centroid distance (most prototypical first) or random.
- `get_slice_video(slice_id, format)` — storyboard (4-frame composite, default) or video. **Counts against the visual budget.**
- `get_rollout_summary(rollout_id)` — outcome, length, cluster path.
- `get_rollout_video(rollout_id, format)` — same as above, full rollout.
- `list_rollouts(outcome, passes_through, n)` — filter by outcome and required cluster traversal.

### Layer 3 — search and aggregation (cheap, computed)
- `find_failure_nodes(min_failure_prob)` — nodes with high failure likelihood.
- `find_recovery_paths(from_node, top_k)` — top successful paths from a high-failure node.
- `find_underrepresented_modes(metric, threshold)` — nodes with low rollout count or low V.
- `compare_paths(path_a, path_b)` — shared prefix, divergence point, per-path outcome distributions.

### Layer 4 — strategy submission (the agent's output channel)
- `propose_collection_request(...)` — submit one `DemonstrationRequest`. `reasoning` is required; `target_cluster` is required for `A_G` and forbidden for `A_NG`.
- `list_submitted_requests()` — self-review.
- `revise_request(request_id, ...)` — modify a previously-submitted request; new reasoning required.
- `delete_request(request_id)`.
- `finalize_strategy(rationale)` — terminal; ends the session.

### Parallel no-graph surface (`A_NG` / `H_NG`)
- `list_rollouts(outcome, n)`, `list_failure_rollouts(n)`, `list_success_rollouts(n)`
- `get_rollout_summary(rollout_id)` — outcome, length only; **no `cluster_path`**.
- `get_rollout_video(rollout_id, format)` — same data as the A_G version.
- `propose_collection_request(...)` — same submission interface, **omits `target_cluster`** (computed post-hoc by classifying the reference rollout).

The asymmetry is exclusively in cluster-level information. Underlying rollouts and videos are identical, ensuring the comparison isolates the graph's contribution rather than information quantity. `tests/vlm/proposals/agents/test_no_graph_isolation.py` enforces this by scanning every A_NG tool's output for forbidden vocabulary.

### Token economics

Per-session budget (defaults; pre-registered):

```yaml
agentic.budget:
  max_tool_calls: 80
  max_visual_calls: 30
  max_video_calls: 5
  max_session_duration_s: 1200
  warning_remaining_threshold: 5
```

Cached visual results return immediately and **do not charge the budget on the second hit**, encouraging the agent to inspect a slice once and refer back to it. After total exhaustion, only `finalize_strategy` is callable.

### Budget visibility (status-line injection)

Every tool result the agent sees ends with a synthetic text block of the form

```
[session: 2/5 requests submitted, 17/40 tool calls used, 1/2 visual calls used. Call finalize_strategy when your strategy is complete.]
```

Without this, an early integration test against Qwen3-VL-8B showed that the agent had no visible signal of remaining budget or completion-target — and consequently kept exploring until the loop force-terminated, producing zero submissions. The status line makes the experiment's exit criterion ambient information the model cannot ignore.

When `BudgetTracker.warning_for(...)` triggers (default: ≤5 calls remain), the line is prefixed with a `REMINDER —` and an explicit instruction to submit any pending requests and call `finalize_strategy`. The threshold is configurable via `agentic.budget.warning_remaining_threshold`; raising it earlier in the session is appropriate for models prone to tool-call drift.

The injection is on by default and can be disabled per-session via `AgentSession(inject_status_line=False)` for tests that need byte-exact result content.

### Submission tools bypass the budget

When the budget is exhausted, an agent that explored too much could otherwise be locked out of its own output channel — making the experimental record empty. To prevent that, the four Layer 4 submission tools (`propose_collection_request`, `revise_request`, `delete_request`, `list_submitted_requests`) and `finalize_strategy` all bypass the budget gate. They never charge against the budget and never get rejected as exhausted.

The corresponding `budget_exhausted` error message instructs the agent accordingly:

> Budget for *kind* tool calls exhausted. Submission tools (propose_collection_request, revise_request, delete_request, list_submitted_requests) and finalize_strategy are still callable — submit any remaining strategy now and then call finalize_strategy with a brief rationale. Do not attempt further exploration.

This means an over-exploring agent can always commit a partial strategy on the way out, even if every exploration call has been spent. Verified by `tests/vlm/proposals/agents/test_mcp_server.py::test_propose_collection_request_bypasses_exhausted_budget`.

## Submission validation gates

In addition to the schema and denylist checks inherited from the one-shot path's `request.validate_request`, the agentic submission tool enforces these gates, all surfaced as structured errors the agent reads and recovers from:

| Gate | Error code | What it catches | Why |
|------|-----------|-----------------|-----|
| Recovery requires `reference_frame > 0` | `recovery_frame_zero` | A `recovery` request starting at frame 0 is `full_trajectory` mislabeled. | The cluster-adherence axis weights `recovery` differently from `full_trajectory`; mislabeled requests contaminate the score. |
| `target_behavior` text must be unique across submissions | `duplicate_target_behavior` | Two submissions with identical operator instructions. | Identical operator instructions provide no additional experimental signal. The check is whitespace- and case-normalized. |
| `target_cluster` requires prior inspection (A_G) | `cluster_not_inspected` | Submitting a `target_cluster` the agent never read about. | An agent that targets a cluster without inspecting it isn't using the graph; the experimental signal degrades. The agent must call `get_node`, `list_slices_in_node`, or `get_slice_video` for the cluster first. |
| Submission must cite ≥3 evidence storyboards | `insufficient_evidence` | A_G with fewer than 3 `evidence_slice_ids`; A_NG with fewer than 3 `evidence_rollout_ids`. | Without evidence, target_behavior prose is ungrounded and the operator has no way to verify the agent's reading. Three is the minimum that lets the agent point at a *pattern*, not a single example. |
| Cited evidence must have been visually inspected | `evidence_not_inspected` | Citing a slice or rollout id the agent never fetched in this session. | Stops fabrication; ensures the prose is grounded in imagery the agent actually saw. |
| A_G evidence slices must lie in `target_cluster` | `evidence_wrong_cluster` | Citing slices from cluster X while targeting cluster Y. | The evidence chain has to match the cluster claim. |

The evidence gates make visual inspection a *prerequisite* for submission rather than a measurement. This is intentional: the previously-considered alternative — measuring whether the agent *chose* to look at storyboards — produced runs in which models (Qwen3-VL-8B in particular) submitted ungrounded, hallucinated `target_behavior` prose. The current claim is therefore "given that the agent must ground in storyboards, does graph access change *what* it cites and *what* it proposes?" The control across conditions is the same minimum-evidence requirement (slice ids in A_G, rollout ids in A_NG), so the gate does not bias condition comparisons.

## Inspection bookkeeping

The inspection gates above require tracking what the agent has touched. `SessionContext` carries three sets:

* `inspected_nodes: Set[int]` — cluster IDs read via `get_node`, `list_slices_in_node`, or `get_slice_video` (the latter inspects the cluster owning the slice). Drives the A_G `cluster_not_inspected` gate.
* `inspected_slices: Set[str]` — slice IDs visually fetched via `get_slice_video`. Drives the A_G `evidence_not_inspected` gate.
* `inspected_rollouts: Set[str]` — rollout IDs visually fetched via `get_rollout_video`. Drives the A_NG `evidence_not_inspected` gate.

All three are append-only within a session and reset at session start. The `evidence_*_ids` arrays cited on a submission are checked against `inspected_slices` (A_G) or `inspected_rollouts` (A_NG); any unfamiliar id is rejected.

Operator-side, the persisted `DemonstrationRequest.to_operator_dict()` collapses `evidence_slice_ids` and `evidence_rollout_ids` into a single condition-blind field, `reference_storyboard_ids`. The operator UI renders the same imagery in both conditions and never sees which list it came from — the operator can't tell whether a request originated from A_G or A_NG by inspecting it.

## Kinematic-summary fallback

When per-rollout state arrays (`raw_states/*.npz` or per-step state in `eval_save_episodes` pkls) are not available, the `get_node` tool's `kinematic_summary` field falls back to a structural cluster_stats summary. To keep the fallback informative — concrete enough that the agent can write differentiated `target_behavior` prose — it includes:

* V-value (Bellman) and failure_likelihood for the cluster.
* Pool-level success rate among rollouts that visit the cluster.
* Top 3 most-likely predecessors with edge probabilities.
* Top 3 most-likely successors with edge probabilities and Δv (advantage) annotations.

When raw state data *is* available, the preferred path computes mean EE position, gripper open/close fraction, and segment duration from a sample of the cluster's slices.

## Adherence scoring (unchanged from one-shot mode)

The same three-axis scoring used in the one-shot version:

| Axis | Weight | What it asks |
|------|--------|--------------|
| `initial_condition`  | 0.25 | Did the env reset to the requested reference state? Trivially 1.0 in sim due to bit-exact `init_state` reset. |
| `cluster`            | 0.50 | Did the realized demonstration follow the specified cluster path? Computed by re-classifying the demo through the saved UMAP / k-means pipeline. **Load-bearing axis.** |
| `success`            | 0.25 | Did the demonstration achieve task success? |

Filter threshold: `overall < 0.6` excludes the demo from the retraining set. For requests without `target_cluster` (`A_NG` / `H_NG`), `target_cluster` is computed post-hoc by classifying the reference rollout, so both conditions have comparable annotations for analysis.

## Aggregation across sessions

Each condition runs N sessions (default 3) with different seeds. `aggregate_agent_sessions` (in `policy_doctor.vlm.proposals.agents.aggregate`) reuses the existing one-shot Jaccard-similarity machinery:

1. **Best-consistency selection (default)** — pick the seed whose submissions have the most counterparts in every other seed; that seed's requests become the final strategy. Cross-session consistency is reported as a secondary metric.
2. **Union (robustness check)** — deduplicate across all seeds.

Output: `proposals/<condition>/{selected_run,consistency_metrics,union_run}.json`. The format matches what the one-shot path produces, so `score_adherence_e2.py` consumes both modes uniformly.

## Operator blinding

In sim with a single operator, real blinding is impossible — but the channels we *can* close are closed:

- Opaque `request_id` (UUID-12). Per-request output dirs use only the id; never a condition string.
- Operator endpoint `GET /requests/active` strips `target_cluster` and `source_condition`.
- Denylist on operator-facing strings (`target_behavior`, `success_criterion`, `prohibitions`): rejects requests containing `cluster`, `node`, `graph`, `umap`, `kmeans`, `centroid`, `embedding` (case-insensitive) before they enter the queue. Enforced for both agent and human submissions.
- Queue is shuffled across all conditions; `queue_shuffle_seed` for reproducibility.
- Adherence summary returned to the operator after each demo shows only `success` and `overall`.

## Architecture

Three processes, all communicating over HTTP. Same pattern as the one-shot version — the agent loop is one extra endpoint on the existing proposal server, plus a standalone batch script.

```
Streamlit operator console     ┐
(policy_doctor env)            │  POST /agent_session, POST /human_request,
                               ▼  POST /propose (legacy)
                    ┌────────────────────────────┐
                    │  proposal_server (Flask)   │  ← VLM (Claude tool-use), pool index, queue,
                    │  policy_doctor env         │    behavior graph, adherence scorer
                    └────────────────────────────┘
                               ▲
        GET /requests/active   │   POST /requests/{id}/result
                               │
E2 sim runner (cupid env)      ┘  ← reuses run_dagger.py; deterministic init_state reset
        │ POST /frame
        ▼
viz_server (existing)
```

For batch runs, `scripts/run_e2_agent.py` skips the server and runs the agent loop in-process: N seeds × M conditions, writes traces directly under `<run_dir>/agent_sessions/...`, then aggregates into `<run_dir>/proposals/<condition>/selected_run.json` for the existing pipeline downstream.

**GPU sharing:** The agent backend is the Anthropic API (no local GPU). The legacy one-shot Qwen3-VL backend remains as before, with `vlm_lifecycle: unload` for single-GPU sharing with the sim.

## Module Layout

```
policy_doctor/vlm/proposals/
    request.py                         DemonstrationRequest, denylist validator, JSON schema
    pool.py                            RolloutPool index over eval_save_episodes output
    init_state.py                      Mid-rollout sim_state extraction
    propose.py                         Legacy one-shot proposal generator (ablation)
    adherence.py                       3-axis scoring + batch JSONL writer
    server.py                          Flask: /chat /propose /requests/* /adherence/*
                                       /agent_session /human_request

    agents/                            ── AGENTIC PATH ──
        conditions.py                  Condition enum (A_G/A_NG/H_NG/H_G), legacy aliasing
        context.py                     SessionContext (graph, pool, classifier, budget, ...)
        budget.py                      BudgetTracker + ResultCache
        session.py                     AgentSession.run() — backend-agnostic tool-use loop
        trace.py                       JSONL per-call trace
        aggregate.py                   Cross-session consistency selection + union
        run.py                         High-level run_one_session / run_condition
        system_prompts/
            __init__.py                prompt_text(cond), prompt_hash(cond)
            A_G.md                     Frozen, committed, hash-tracked
            A_NG.md
        tools/
            schema.py                  JSON schemas (the contract)
            types.py                   ToolResult, ToolSpec
            topology.py                Layer 1
            access.py                  Layer 2
            analysis.py                Layer 3
            submission.py              Layer 4
            no_graph.py                Parallel A_NG / H_NG surface
            kinematic_summary.py       Textual node summary from .pkl / .npz state arrays
            registry.py                build_tool_registry(condition, ctx)

    graph_representation/, vlm_input/  ── ONE-SHOT PATH (ablation) ──
        Same plugins as before; selected automatically when mode == "one_shot".

policy_doctor/vlm/backends/
    base.py                            VLMBackend ABC + chat_with_tools(), AssistantTurn,
                                       ToolCall, TokenUsage dataclasses
    claude.py                          ClaudeVLMBackend.chat_with_tools — Anthropic tool use
    mock.py                            MockVLMBackend — deterministic scripted agent for Tier 0
    qwen2_vl.py, gemini.py, ...        unchanged

policy_doctor/envs/
    e2_runner.py                       sim-side runner shim (unchanged)

policy_doctor/streamlit_app/tabs/
    e2_console.py                      operator UI; agent + human + legacy panels

policy_doctor/curation_pipeline/steps/
    build_rollout_pool.py             render storyboards + cluster paths
    score_adherence_e2.py             batch adherence + filter (consumes both modes' output)

policy_doctor/configs/e2/
    defaults.yaml                     mode: agentic|one_shot; agentic.* block; one-shot fields
    prompts/graph_outcome.yaml        legacy
    tasks/square_mh.yaml

policy_doctor/configs/experiment/
    e2_smoke_tier0.yaml               mock agent; CI-safe; seconds
    e2_smoke_tier1.yaml               real Claude tool use; tiny budget; no retraining
    e2_smoke_tier2.yaml               reduced agentic + retraining + eval
    e2_full.yaml                      pre-registered scale: 3 sessions × 3+ conditions × 50 eval × 3 retrain seeds

scripts/
    run_e2_agent.py                   Hydra batch entry — drives the whole agentic experiment
    check_claude_tool_image.py        Day-one Claude smoke: image-in-tool_result roundtrip
    run_e2_sim.py                     unchanged
    experiments/run_e2_proposal_server.sh
    experiments/run_e2_console.sh
    experiments/run_e2_sim.sh
```

## Reuse from existing infra

| E2 needs                                  | Existing module                                                                |
|-------------------------------------------|--------------------------------------------------------------------------------|
| Demonstration → cluster path              | `policy_doctor.monitoring.TrajectoryClassifier.classify_episode_from_pkl`      |
| Saved UMAP + k-means centroids            | `FittedModelAssigner.from_paths(clustering_dir, graph)`                        |
| Behavior graph V/A values, paths          | `behaviors/behavior_graph.py` — `compute_values`, `enumerate_paths`            |
| LLM tool-use client                       | `vlm/backends/claude.py` (`chat_with_tools`); pluggable per backend            |
| Storyboards (4-frame composites)          | `vlm/storyboard.make_storyboard`                                               |
| HTTP server pattern, viz overlay          | `envs/{policy_server,viz_server}.py`                                           |
| Operator interface, intervention devices  | `envs/{robomimic_dagger_env,dagger_runner}.py`                                 |
| Deterministic scene reset                 | `RobomimicLowdimWrapper(init_state=...)`                                       |
| Pipeline framework, pre-registration      | `curation_pipeline/base_step.py` + `done` sentinel                             |

## Running the Experiment

### Prerequisites

1. Completed `run_clustering` step (`clustering_dir` with `cluster_labels.npy`, `clustering_models.pkl`).
2. Satisfactory `validate_cluster_coherence_vlm` (E1) — without it, E2's adherence scoring is meaningless.
3. Eval episode pickle directory (`episodes_dir`) with `metadata.yaml` and per-rollout `ep*.pkl` from `eval_save_episodes`.
4. Trained base policy checkpoint and matching `infembed_fit.pt` / `infembed_embeddings.npz`.
5. For the agentic path: `ANTHROPIC_API_KEY` set (or `agent_backend: mock` for plumbing tests).

### Tier 0 — agent plumbing smoke (mock backend, CI-safe)

Runs in seconds; no GPU; no API calls. Drives the entire agentic loop end-to-end with a deterministic scripted mock:

```bash
PYTHONPATH=. conda run -n policy_doctor python -m unittest \
    discover -s tests/vlm/proposals
```

That covers 151 tests including 47 net-new agent tests (schemas, budget, every tool layer, no-graph isolation, mock end-to-end session, Claude response shape, system prompts, aggregation, condition aliasing).

### Day-one Claude check (manual; before Tier 1)

Confirms image-in-tool_result roundtrip with the real Anthropic API. Catches a class of failures that would otherwise only surface during a long Tier 1 run.

```bash
export ANTHROPIC_API_KEY=...
python scripts/check_claude_tool_image.py
# optionally: python scripts/check_claude_tool_image.py /path/to/storyboard.png
```

### Tier 1 — real Claude tool use, tiny budget, no retraining

Use to verify Claude tool calls land, image budget accounts correctly, GUI flow works, init_state replay round-trips.

```bash
# 1. Build pool + storyboards (one-time)
python -m policy_doctor.scripts.run_pipeline \
    +experiment=e2_smoke_tier1 \
    steps=[build_rollout_pool]

# 2. Run agent sessions for both conditions (in-process; writes proposals/ + agent_sessions/)
python scripts/run_e2_agent.py \
    +experiment=e2_smoke_tier1 \
    e2_proposals.pool_episodes_dir=<episodes_dir> \
    e2_proposals.clustering_dir=<clustering_dir> \
    e2_proposals.run_dir=<run_dir>

# 3. Start the proposal server to drain the queue (optional — only if you want a GUI)
./scripts/experiments/run_e2_proposal_server.sh \
    --config <run_dir>/e2_config.yaml --port 5003
./scripts/experiments/run_viz_server.sh --port 5002
./scripts/experiments/run_e2_console.sh --port 8501

# 4. Sim runner drains the queue
./scripts/experiments/run_e2_sim.sh \
    task=square_mh train_dir=/path/to/train_dir \
    proposal_server=http://127.0.0.1:5003 \
    viz_url=http://127.0.0.1:5002 \
    output_dir=<run_dir>/demonstrations
```

### Tier 2 — reduced-scale experimental run

Same as Tier 1 with `+experiment=e2_smoke_tier2`, then:

```bash
# 5. Score adherence in batch
python -m policy_doctor.scripts.run_pipeline \
    +experiment=e2_smoke_tier2 \
    steps=[score_adherence_e2]

# 6. Retrain + eval (existing steps)
python -m policy_doctor.scripts.run_pipeline \
    +experiment=e2_smoke_tier2 \
    steps=[run_curation_config, train_curated, eval_curated, compare]
```

### Tier 3 — full pre-registered scale

Same workflow with `+experiment=e2_full`. `<run_dir>/pre_registration.yaml` is written at run start (includes the schema hash, prompt hashes, budget, all knobs). Changing any value mid-run is a protocol violation.

### Human conditions (`H_NG`, optional `H_G`)

Open the Streamlit console; "Human exploration" panel exposes a rollout browser (with `cluster_path` shown only for `H_G`) and a `DemonstrationRequest` form with a required rationale. Submissions go through `POST /human_request`, are validated against the same denylist + rollout-id check the agent path uses, and land in the same operator queue.

```bash
./scripts/experiments/run_e2_console.sh --port 8501
# → "E2 Console" tab → "Human exploration" panel
```

## Choosing a Backend

Two distinct paths use different "backend" abstractions. They share the tool surface, the prompt files, and the submission schema — only the model invocation layer differs.

| Path | Use for | Model lives | Configuration |
|------|---------|-------------|---------------|
| **Headless `chat_with_tools`** (`scripts/run_e2_agent.py`) | Pre-registered experimental runs; reproducible; multi-seed; full trace | Direct API call from our code | `agentic.agent_backend` in YAML |
| **MCP server** (`python -m policy_doctor.vlm.proposals.agents.mcp_server`) | Demo, exploration, prompt iteration, ad-hoc use | The MCP client owns the model (Cursor / Claude Code / Claude Desktop / Inspector) | Env vars passed via the client's MCP config |

The same Layer 1–4 tools, the same prompts, and the same validation run on both paths.

### Headless backends (for `run_e2_agent.py`)

All headless backends implement `VLMBackend.chat_with_tools(messages, tools, system, max_tokens, temperature, seed) -> AssistantTurn`. Add a backend by writing one method; everything else is shared.

**Claude (default)**

```yaml
agentic:
  agent_backend: claude
  agent_backend_params:
    model_name: claude-sonnet-4-6      # or claude-opus-4-7
    max_tokens: 4096
```
Requires `ANTHROPIC_API_KEY`. Verify image-in-tool_result roundtrip with:
```bash
python scripts/check_claude_tool_image.py
```

**Gemini**

```yaml
agentic:
  agent_backend: gemini
  agent_backend_params:
    model_name: gemini-2.5-pro         # or gemini-2.0-flash, etc.
    max_output_tokens: 4096
```
Requires `GOOGLE_API_KEY` and `pip install google-generativeai`. Gemini's `FunctionResponse` channel is JSON-only, so the backend automatically hoists images into a sibling user-turn message — no special config needed. Verify with:
```bash
python scripts/check_gemini_tool_image.py
```

**Mock (Tier 0 / CI)**

```yaml
agentic:
  agent_backend: mock
  agent_backend_params: {}
```
Walks a fixed scripted exploration sequence (`get_graph_summary` → `list_nodes` → propose × N → `finalize_strategy`). Deterministic, no API key, runs in seconds. Used by `e2_smoke_tier0.yaml`.

**Adding another headless backend** (OpenAI, etc.) is one new file in `policy_doctor/vlm/backends/` implementing `chat_with_tools`. The session loop, tool registry, budget, trace, and aggregation are unchanged. See [`docs/E2_agents.md`](../E2_agents.md) §6.1 for the contract; the existing `claude.py` / `gemini.py` are the templates.

### Interactive backends via MCP

The MCP server exposes the same tool surface over the Model Context Protocol so any MCP-compatible client can drive an exploration session. The model lives **inside the client** (Cursor uses your Cursor-configured model, Claude Code uses Claude, etc.) — the server just answers tool calls.

**Install the SDK once** (it's an optional dependency):
```bash
pip install mcp
```

The server is launched per-session by the client over stdio; configuration goes through env vars. Common to every client:

| Env var | Required | What |
|---------|----------|------|
| `POLICY_DOCTOR_CLUSTERING_DIR`     | yes | path to the saved clustering result |
| `POLICY_DOCTOR_POOL_EPISODES_DIR`  | yes | path to `eval_save_episodes` output |
| `POLICY_DOCTOR_OUT_DIR`            | no  | where submissions land (default `./policy_doctor_mcp_session`) |
| `POLICY_DOCTOR_CONDITION`          | no  | `A_G` (default) or `A_NG` |
| `POLICY_DOCTOR_TASK_HINT`          | no  | task description |
| `POLICY_DOCTOR_STORYBOARD_DIR`     | no  | storyboard sidecar dir |
| `POLICY_DOCTOR_RAW_STATES_DIR`     | no  | `.npz` raw-state arrays for `kinematic_summary` |

Full env-var reference: [`policy_doctor/vlm/proposals/agents/mcp_server/README.md`](../../policy_doctor/vlm/proposals/agents/mcp_server/README.md).

#### Cursor

Add to `.cursor/mcp.json` in the project (or `~/.cursor/mcp.json` globally):
```json
{
  "mcpServers": {
    "policy-doctor-e2": {
      "command": "/abs/path/to/conda/envs/policy_doctor/bin/python",
      "args": ["-m", "policy_doctor.vlm.proposals.agents.mcp_server"],
      "env": {
        "POLICY_DOCTOR_CLUSTERING_DIR": "/abs/path/to/clustering_result",
        "POLICY_DOCTOR_POOL_EPISODES_DIR": "/abs/path/to/eval_episodes",
        "POLICY_DOCTOR_OUT_DIR": "/abs/path/to/cursor_session_output",
        "POLICY_DOCTOR_TASK_HINT": "Pick up the cube and place it on the platform."
      }
    }
  }
}
```
Restart Cursor; the `policy-doctor-e2` server appears in the MCP panel with all tools listed.

#### Claude Code

Two ways:

```bash
# 1. Per-project: drop a .mcp.json into the repo root with the same shape
#    Cursor uses, then run `claude` in that directory.

# 2. Global: register the server once with the CLI:
claude mcp add policy-doctor-e2 \
    /abs/path/to/conda/envs/policy_doctor/bin/python \
    -m policy_doctor.vlm.proposals.agents.mcp_server \
    -e POLICY_DOCTOR_CLUSTERING_DIR=/abs/path/to/clustering_result \
    -e POLICY_DOCTOR_POOL_EPISODES_DIR=/abs/path/to/eval_episodes \
    -e POLICY_DOCTOR_OUT_DIR=/abs/path/to/claude_code_session_output
```

After registering, run `claude` in any directory; the MCP server is reachable via `/mcp` and the tools appear in the slash-command list.

#### Claude Desktop

Same JSON shape as Cursor, in:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "policy-doctor-e2": {
      "command": "/abs/path/to/conda/envs/policy_doctor/bin/python",
      "args": ["-m", "policy_doctor.vlm.proposals.agents.mcp_server"],
      "env": { "POLICY_DOCTOR_CLUSTERING_DIR": "...", "POLICY_DOCTOR_POOL_EPISODES_DIR": "..." }
    }
  }
}
```
Restart the desktop app; the tools appear in the conversation tool picker.

#### MCP Inspector (debug / no model)

The Inspector is a UI for poking at an MCP server without a model in the loop. Useful for verifying the tools register correctly and for debugging schema mismatches:
```bash
npx @modelcontextprotocol/inspector \
  python -m policy_doctor.vlm.proposals.agents.mcp_server
```

#### When MCP-driven results matter for the experiment

MCP sessions are **interactive and not reproducible by design** — they're for demos, prompt iteration, and exploration, not for the headline experimental result. Each MCP-server invocation is one session with no seed control, no aggregation, no trace of the model's turn-by-turn reasoning (because the model lives in the client). The `submitted_requests.json` it writes is, however, in the same shape as `proposals/<condition>/selected_run.json`, so a Cursor exploration can bootstrap a downstream operator pipeline by copying that file into the run dir.

For pre-registered headline numbers, use `run_e2_agent.py` with a headless backend.

## Output Files

```
<run_dir>/
    pre_registration.yaml                    schema hash + prompt hashes + agentic config
    chat.jsonl                               only when chat_enabled=true
    agent_sessions/
        A_G/
            seed_0/
                trace.jsonl                  per-event trace (assistant_turn, tool_call, tool_result, ...)
                conversation.json            full provider-neutral message list (images stripped)
                submitted_requests.json      this seed's submissions
                rationale.txt                finalize_strategy summary
                budget_summary.json          n_tool_calls, n_visual_calls, n_cache_hits, ...
                session_summary.json         one-shot summary (stop_reason, n_turns, ...)
            seed_1/, seed_2/                 same shape
            summary.json                     per-condition aggregate (errors, stop_reasons, ...)
        A_NG/, H_NG/, H_G/                   same shape
    human_sessions/
        H_NG/submissions.jsonl               operator submissions + rationales
        H_G/submissions.jsonl
    proposals/
        A_G/
            selected_run.json                aggregation output (best-consistency seed)
            consistency_metrics.json
            union_run.json
        A_NG/, H_NG/, H_G/                   same shape
    demonstrations/
        <request_id>/                        opaque ids — no condition leak in path
            ep0000.pkl                       robomimic-style trajectory pkl
    score_adherence_e2/
        per_demo_scores.jsonl                one row per demo
        filtered_demos.jsonl                 demos with overall ≥ filter_threshold
        filter_summary.json                  pass rates by condition + request_type
```

## Statistical Analysis

Primary: **Wilcoxon signed-rank** on K paired retraining seeds (K=`n_retrain_seeds`) for the headline pairs:
- `A_G` vs `A_NG`
- `A_G` vs `H_NG` (if no `H_G`) or `A_G` vs `H_G` (if both human conditions present)

Secondary: **McNemar's test** on per-evaluation-condition success/failure pairs averaged across seeds.

Reported metrics:
- Per-policy success rates with Wilson 95% CIs.
- Cliff's delta or odds ratio (effect size).
- Per-axis adherence score distributions per condition.
- Cluster-path coverage of demonstrations per condition (H3).
- Cross-session consistency rate per condition (informative; difference between A_G and A_NG is itself a finding).

With three primary pairwise comparisons, apply Bonferroni or Benjamini-Hochberg correction.

K=2 in Tier 2 is statistically toothless on its own — point estimates and per-axis distributions still inform whether to commit to Tier 3.

## Empirical observations from integration testing

The features above (status-line injection, validation gates, richer kinematic_summary, operational system prompts) were added in response to specific failures observed when running Qwen3-VL-8B against the apr26 mimicgen-square sweep clustering (k=15, 500 rollouts, 11,122 slices). They are documented here because they materially affect what counts as a methodology-level result vs. a model-level result.

### Tool-call drift is real and model-dependent

Initial run: Qwen3-VL-8B with the original (advisory) `A_G.md` and a 35-call budget made 35 tool calls and submitted **zero** requests. The model meaningfully explored — calling `get_graph_summary` first, picking high-failure clusters via `find_failure_nodes`, querying `get_node` on each candidate, exploring `find_recovery_paths` — but never crossed from explore to commit.

After (status-line + operational prompt + tighter budget): same model, default `A_G.md`, 15-call budget with the budget warning firing at 8 remaining. Result: **still zero submissions** in 32 seconds — the model walked all 15 clusters via `get_node` and force-terminated. The agent saw the status line `[session: 0/4 requests submitted, …]` on every tool result. It did not act on it.

A separately-tested forcing prompt (one that explicitly enumerated the rollouts to use and the request types to submit) produced 3 valid submissions in 21 seconds, but the submissions were degenerate — identical `target_behavior` text across all three, only the cluster id varying. With the new `duplicate_target_behavior` gate active, two of those three submissions would be rejected.

Conclusion: the validation gates and infrastructure improvements are necessary but not sufficient for Qwen3-VL-8B to drive this experiment. **For Qwen-class models, an additional in-loop "stop-and-submit" injection at a budget threshold is likely required**; alternatively, the experiment should be run with stronger instruction-following models (Claude Sonnet/Opus, Gemini-Pro) which are known to follow tool-use prompts of this complexity.

### What this means for pre-registration

The original spec's pre-registration plan freezes prompts before any condition runs. The integration test suggests that approach is too tight: prompt iteration on Tier 1 (with smoke-tier budgets, no retraining) is methodologically necessary to find a prompt-budget combination the chosen agent backend can satisfy. The recommendation:

1. Treat **prompt iteration on Tier 1 as a documented methodology step**, not a violation. Iterate `A_G.md` and `A_NG.md` against multiple Tier 1 seeds until the agent produces 4–8 differentiated submissions on each seed.
2. Freeze the **post-iteration** version of the prompts before Tier 2/Tier 3.
3. Record the final prompt hashes and the iteration-count in `pre_registration.yaml` so the pre-registration claim is an audit trail of how the prompts converged, not a fiction.

This is what real method papers do; the rigid alternative does not survive contact with model variance.

### E1 dependency on apr26

The apr26 clustering used in the integration test has **not** had E1 (cluster-coherence VLM validation) run against it. E1 requires real rollout frames, which were not produced for this sweep (`eval_save_episodes` ran without image observations). For the actual experiment, E1 must precede E2 on the same clustering — without that, E2's adherence numbers are circular. This is the same caveat the spec already calls out in §"Self-validation risk."

### Operator-facing differentiation is a real constraint

Even when Qwen3-VL-8B did submit (under the forcing prompt, before the dedup gate was added), all three of its `target_behavior` strings were identical: "Pick up the square nut and place it on the rod without dropping it." The model varied `target_cluster` and `reference_rollout_id`, but the operator-readable instruction did not vary. The operator would do the same thing for all three demos.

The new `duplicate_target_behavior` gate forces the model to differentiate. For the experiment, the H1 hypothesis (graph-aware agents propose better demos) requires the operator-facing prose itself to be informative — not just the cluster annotations. The kinematic_summary fallback (structured V/A/predecessor/successor info) helps anchor the prose in concrete graph features.

## Failure Modes

| Symptom | Likely cause | Diagnostic |
|---------|--------------|------------|
| Agent fails to explore — submits ≤ 1 request and finalizes | System prompt too directive, or budget too small | Inspect `trace.jsonl`; if total tool calls < 5, raise budget or revise the prompt before unfreezing |
| Agent burns visual budget early | Watching videos before forming hypotheses | Check `trace.jsonl` for `cost: visual` calls in the first 5 turns; lower visual budget; emphasize `kinematic_summary` in prompt |
| `A_NG` strategies look graph-like | A capable VLM may form internal categorizations from videos | Diagnose: A_NG cluster coverage matches A_G's. **Genuine finding**, not a confound — report it |
| Conditions produce indistinguishable strategies | Graph adds no actionable info beyond outcomes at this scale | Low cross-condition divergence; negative result, but publishable, especially with E1 confirming the graph is structured |
| Image-in-tool_result rejected by Claude | API-shape mismatch | `chat_with_tools` has a fallback that moves images to a sibling user message; `scripts/check_claude_tool_image.py` verifies on day one |
| Cluster axis ~0 across the board | Saved UMAP / centroids stale or wrong | Run the §12 sanity check from the legacy doc: classifier on pool rollouts must reproduce stored cluster paths |
| Filter pass rates differ wildly between conditions | One condition consistently asks for unrealizable behaviors | Inspect `filter_summary.json`; re-look at trace |
| Adherence overall = 1.0 for every demo | `init_state` resolver missing → axis 1 defaults to 1.0 | Confirm `reference_pkl_resolver` is wired in `score_adherence_e2` |
| Operator can guess the condition | Leak in stdout / file path / overlay | `grep -ri "graph\|outcome\|cluster" <run_dir>` should return nothing in operator-facing files; `test_no_graph_isolation.py` enforces this for tool outputs |
| Recovery requests fail to start in the right state | `extract_sim_state_at_frame` indexing issue | Use `verify_sim_state_replays` in Tier 1 |
| Agent prompt sensitivity | Different prompts → different "expertise levels" | Lock the prompt before the experiment; hash committed to `pre_registration.yaml`. Run multiple seeds; report variance |
| Tool-call drift — agent explores indefinitely | No termination signal | `max_tool_calls` budget enforces; `max_turns` is a secondary cap |
| Operator distinguishes conditions from request style | A_G requests systematically reference cluster-like reasoning, A_NG don't | Denylist validation rejects on submission; operator interface strips condition fields; randomized queue order |

## Self-Validation Risk

Adherence scoring uses the same graph the agent reasoned over. If the graph mis-clusters certain behaviors, both proposal and verification inherit the same blind spots. Both conditions face this equally (the no-graph condition's `target_cluster` is also computed from the same graph), so the comparison is internally valid. But the absolute claims about adherence depend on the graph being a reasonable behavioral representation.

This is exactly what experiment E1 tests. **E1 must be run before E2.** The dependency is enforced in the README and in `pre_registration.yaml`.

## Scope and Limitations

- Sim-only for v1. Real-robot transfer is straightforward at the request-schema level but adds scene-validation and tolerance UI.
- Single operator (the implementer). Operator fatigue and priors from prompt-iteration phases are uncontrolled. Document.
- Single agent backend per run. The `chat_with_tools` interface is generic; cross-LLM checking (Gemini, etc.) is one backend implementation away.
- Adherence is fully automated. Any blind spot in the saved behavior graph propagates to E2 adherence — E1 is the protective measurement.
- Three primary conditions; `H_G` is optional (gated by operator availability). Adding the fourth condition is a config flip; infrastructure is in place.
- The legacy one-shot mode is retained for ablation but is not the headline result.

## Related

- E1 (`docs/experiments/experiment_e1_cluster_coherence.md`) — must pass before E2's adherence claims are meaningful.
- `docs/E2_agents.md` — the source spec this implementation realizes.
- `docs/monitoring.md` — runtime-monitoring components E2 reuses for trajectory classification.
- `docs/DAGGER_GUIDE.md` — the operator interface E2 inherits.

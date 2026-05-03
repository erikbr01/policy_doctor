# Experiment E3: Agentic Graph Exploration for Data Collection Strategy — Implementation Spec

This spec extends the behavior-graph codebase to support an agentic experiment in which a VLM agent, with structured tool access to the behavior graph and rollout pool, explores the policy's behavioral structure and proposes a data collection strategy. The agent's strategy is then executed by an operator.

The experiment compares three (or optionally four) conditions to isolate the contribution of (a) graph access and (b) agent vs human reasoning. Adherence is verified automatically via the same graph-classification pipeline used in E2.

This spec assumes E2's infrastructure exists (request schema, operator interface, adherence scoring, retraining, evaluation). E3 adds the agentic layer on top of those components.

## 0. Relationship to E1 and E2

- **E1 (cluster validation)** must run first. Adherence scoring depends on the graph being a meaningful behavioral representation.
- **E2 (one-shot VLM proposals)** shares request schema, operator protocol, adherence scoring, and evaluation. E3 reuses these directly.
- **E3 (this experiment)** replaces E2's one-shot prompting with agentic exploration via tool use. The resulting strategy is fed into the same downstream operator → adherence → retraining → evaluation pipeline.

Where E2 and E3 share infrastructure, this spec references E2 rather than re-specifying.

## 1. Conditions

Three primary conditions:

- **A_G (Agent-with-graph):** VLM agent with full access to the graph tool surface (Layers 1–4 in Section 4).
- **A_NG (Agent-without-graph):** VLM agent with access to a parallel tool surface that exposes rollouts, videos, and outcomes but no cluster-level structure.
- **H_NG (Human-without-graph):** Trained operator with access to the same underlying rollouts and videos as A_NG, producing a strategy in the same request format.

Optional fourth condition (recommended if budget allows):
- **H_G (Human-with-graph):** Trained operator with access to the graph + rollout videos.

The four-condition design enables clean attribution:
- A_G vs A_NG: graph value to an LLM.
- H_G vs H_NG: graph value to a human.
- A_G vs H_G: comparable reasoning given graph access.
- A_NG vs H_NG: comparable reasoning without graph access.

Three conditions (A_G, A_NG, H_NG) is the minimum defensible design. Four is preferred.

## 2. Module Structure

```
experiments/e3_agentic_proposals/
    __init__.py
    config.py
    tools/
        __init__.py
        schema.py              # JSON schemas for all tools
        graph_tools.py         # Layer 1: topology
        slice_tools.py         # Layer 2: slice/rollout access
        analysis_tools.py      # Layer 3: search/aggregation
        submission_tools.py    # Layer 4: strategy submission
        no_graph_tools.py      # Parallel surface for A_NG condition
        budget.py              # Token/call budget tracking
        cache.py               # Memoization for expensive calls
    agent/
        __init__.py
        client.py              # Tool-use loop with Claude
        system_prompts.py      # Per-condition system prompts (pre-registered)
        session.py             # Single agent session: setup, run, log, finalize
        trace.py               # Structured logging of every tool call
    human_interface/
        __init__.py
        explorer_ui.py         # H_NG / H_G interface for graph/rollout exploration
        strategy_form.py       # Same request schema, human-facing
    mcp_server/                # Optional, for interactive chat demo
        __init__.py
        server.py
        manifest.json
    run.py                     # Orchestrate one session per condition
    README.md
```

E2 modules (`request_schema`, `operator/`, `adherence/`, `retrain/`, `evaluate/`) are imported directly.

## 3. Pre-Experiment Artifacts

All E2 pre-experiment artifacts are reused:
- Base policy.
- Rollout pool (100–150 rollouts).
- Behavior graph + saved UMAP transformer + saved centroids.
- Evaluation suite of 50 initial conditions.

E3 adds:
- **Agent system prompts** for A_G and A_NG, frozen and committed before the experiment.
- **Human exploration interface** for H_NG and (if used) H_G, also frozen.
- **Tool budget configuration** specifying per-session limits on tool calls and visual inputs.

## 4. Tool Surface Design

The tool surface is the central design artifact for this experiment. The agent reasons about the policy and constructs a strategy using *only* these tools. Tool design directly determines what the agent can express.

### 4.1 Layer 1: Graph Topology (cheap, broad)

These tools are used to orient the agent at the start of a session and to navigate the graph structurally. They return small structured objects (no images), so they're cheap.

#### `get_graph_summary()`

Returns a high-level overview:
```json
{
  "n_nodes": 15,
  "n_paths_to_success": 4,
  "n_paths_to_failure": 7,
  "terminal_probabilities": {"success": 0.42, "failure": 0.58},
  "v_value_range": [0.05, 0.91],
  "advantage_range": [-0.62, 0.34],
  "n_rollouts": 142,
  "rollout_outcomes": {"success": 60, "failure": 82}
}
```

Almost always the agent's first call.

#### `list_nodes(min_failure_likelihood: float = 0.0, min_v: float = 0.0, max_v: float = 1.0)`

Returns a list of nodes with summary stats, optionally filtered:
```json
[
  {"node_id": "c3", "v": 0.71, "failure_likelihood": 0.18, "in_degree": 2, "out_degree": 3, "n_slices": 84},
  ...
]
```

#### `list_paths(from_node: str = "START", to_node: str = "FAILURE", top_k: int = 10)`

Returns the top-k paths between two nodes, ordered by Markov probability:
```json
[
  {"path": ["START", "c3", "c7", "c12", "FAILURE"], "probability": 0.18, "n_rollouts": 23},
  ...
]
```

#### `get_node(node_id: str)`

Full information for one node:
```json
{
  "node_id": "c3",
  "v": 0.71,
  "failure_likelihood": 0.18,
  "n_slices": 84,
  "predecessors": [{"node": "START", "edge_prob": 0.40}, {"node": "c1", "edge_prob": 0.22}],
  "successors": [{"node": "c7", "edge_prob": 0.55, "advantage": -0.12}, {"node": "c5", "edge_prob": 0.30, "advantage": 0.08}],
  "kinematic_summary": "End-effector approaches workspace from above; gripper open throughout segment; mean duration 18 timesteps."
}
```

The `kinematic_summary` is a textual description computed from state trajectories (not LLM-generated). This is critical for token economics: the agent should be able to form a working hypothesis about what a node represents *before* deciding to spend tokens on video.

#### `get_edge(from_node: str, to_node: str)`

Edge information:
```json
{
  "from": "c3",
  "to": "c7",
  "probability": 0.55,
  "advantage": -0.12,
  "example_rollouts": ["r_023", "r_041", "r_087", "r_112"]
}
```

### 4.2 Layer 2: Slice and Rollout Access (medium cost, specific)

These tools surface the underlying rollouts and slices. They return either textual summaries (cheap) or videos/storyboards (expensive — counted against the visual budget).

#### `list_slices_in_node(node_id: str, n: int = 20, sort_by: str = "centroid_distance")`

Returns slices assigned to a cluster:
```json
[
  {"slice_id": "r_023_t12_t30", "rollout_id": "r_023", "start_frame": 12, "end_frame": 30, "outcome": "success", "centroid_distance": 0.08},
  ...
]
```

`sort_by` accepts `"centroid_distance"` (most prototypical first) or `"random"`.

#### `get_slice_video(slice_id: str, format: str = "storyboard")`

Returns visual content for a slice. **This call counts against the visual budget.**

- `format="storyboard"`: 4-frame composite image (default; cheap in tokens).
- `format="video"`: short MP4 (more expensive; use sparingly).

#### `get_rollout_summary(rollout_id: str)`

Cluster path and metadata, no video:
```json
{
  "rollout_id": "r_023",
  "outcome": "failure",
  "length": 87,
  "cluster_path": ["c3", "c7", "c12", "FAILURE"],
  "initial_conditions": {"object_pose": [0.42, 0.18, 0.05, ...], "...": "..."}
}
```

#### `get_rollout_video(rollout_id: str, format: str = "storyboard")`

Same semantics as `get_slice_video` but for the full rollout. Counts against visual budget.

#### `list_rollouts(outcome: str = None, passes_through: list = None, n: int = 50)`

Filterable listing of rollouts:
```json
[
  {"rollout_id": "r_023", "outcome": "failure", "cluster_path": ["c3", "c7", "c12", "FAILURE"]},
  ...
]
```

`passes_through`: optional list of node_ids the rollout must traverse.

### 4.3 Layer 3: Search and Aggregation (cheap, computed)

Higher-level analytical queries. These are convenience wrappers that compute over the graph; the agent could in principle replicate them with Layer 1 + 2 calls, but exposing them directly saves tokens and makes the agent's reasoning more efficient.

#### `find_failure_nodes(min_failure_prob: float = 0.3)`

Nodes with high probability of leading to FAILURE within a few transitions:
```json
[
  {"node_id": "c12", "failure_likelihood": 0.78, "main_failure_path": ["c12", "FAILURE"], "n_rollouts": 31},
  ...
]
```

#### `find_recovery_paths(from_node: str, top_k: int = 5)`

Paths from `from_node` that reach SUCCESS:
```json
[
  {"path": ["c12", "c5", "c2", "SUCCESS"], "probability": 0.21, "n_example_rollouts": 4},
  ...
]
```

#### `find_underrepresented_modes(metric: str = "rollout_count", threshold: int = 5)`

Clusters with low rollout count or low V — candidates for additional data collection:
```json
[
  {"node_id": "c14", "n_rollouts": 3, "v": 0.31, "reason": "low rollout count"},
  ...
]
```

#### `compare_paths(path_a: list, path_b: list)`

Two-path comparison: divergence point, relative success rates, kinematic differences:
```json
{
  "shared_prefix": ["START", "c3"],
  "divergence_point": "c3",
  "path_a_outcome_distribution": {"success": 0.62, "failure": 0.38},
  "path_b_outcome_distribution": {"success": 0.18, "failure": 0.82}
}
```

### 4.4 Layer 4: Strategy Submission (the agent's output channel)

The agent must produce its strategy through these tools. Free-text output not submitted via these tools is ignored.

#### `propose_collection_request(initial_conditions: dict, target_behavior: str, target_cluster: str = None, prohibitions: list = None, success_criterion: str, request_type: str, reasoning: str)`

Submits a single demonstration request. Schema mirrors E2's `DemonstrationRequest`. The `reasoning` field is **required** — it forces the agent to articulate why each request matters and provides interpretable traces.

Returns: validation result. If validation fails (e.g., invalid `reference_rollout_id`, banned cluster references in `target_behavior`), the agent gets the error and can retry.

#### `list_submitted_requests()`

Returns the agent's currently-submitted requests, for self-review during exploration.

#### `revise_request(request_id: str, ...fields)`

Modifies a previously-submitted request. Useful when the agent learns something later that changes its earlier reasoning.

#### `delete_request(request_id: str)`

Removes a submitted request.

#### `finalize_strategy(rationale: str)`

Ends the session. The agent must write a brief rationale summarizing its overall strategy. After this call, no further tool calls are accepted.

### 4.5 Tools NOT Exposed

Deliberately omitted:
- Influence embedding internals.
- Clustering algorithm configuration.
- Direct retraining or evaluation triggers.
- Modifications to the graph itself.

The agent's role is bounded: explore the graph, propose a strategy. It cannot reconfigure the methodology or take downstream actions.

### 4.6 Parallel No-Graph Surface (for A_NG)

The A_NG condition uses tools that expose rollouts and videos but no cluster-level information. Same underlying data, narrower interface:

- `list_rollouts(outcome: str = None, n: int = 50)` — outcome filter only, no cluster filtering.
- `get_rollout_video(rollout_id, format)` — same as A_G.
- `get_rollout_summary(rollout_id)` — returns outcome and metadata, **omits cluster_path**.
- `list_failure_rollouts()`, `list_success_rollouts()` — convenience.
- `propose_collection_request(...)` — same submission interface, **omits `target_cluster`**.

The asymmetry is exclusively in cluster-level information. Underlying rollouts and videos are identical, ensuring the comparison isolates the graph's contribution rather than information quantity.

## 5. Token Economics and Budget Control

Agentic sessions over a graph with 15 clusters and ~150 rollouts can easily consume large token volumes. Without control, costs are unpredictable and conditions become non-comparable.

### 5.1 Budget Configuration

Pre-registered per-session limits:

```python
BUDGET = {
    "max_tool_calls": 80,           # Total tool calls per session
    "max_visual_calls": 30,         # Calls returning images/video
    "max_video_calls": 5,           # Of those, full-video calls (vs storyboards)
    "max_session_duration_min": 20, # Wall-clock cap
}
```

These limits apply identically across A_G and A_NG. The visual budget specifically must be matched, since it's the dominant cost driver.

### 5.2 Budget Enforcement

`tools/budget.py` implements:
- Counter incremented on every tool call, partitioned by type (cheap, visual, video).
- Tool calls that would exceed budget return a structured error: `{"error": "budget_exhausted", "type": "visual"}` rather than silently failing. The agent can adapt.
- Final 5 calls before exhaustion produce warnings: `{"warning": "approaching_visual_budget", "remaining": 5}`.
- After total budget exhaustion, only `finalize_strategy` is callable.

### 5.3 Caching

`tools/cache.py` memoizes:
- All Layer 1 tool calls (no token cost on repeat).
- All Layer 2 textual calls.
- All visual calls — repeat requests for the same slice/rollout video return cached content and **do not count against budget** on the second call.

This aligns incentives: the agent is encouraged to inspect a slice once and refer back to it, rather than re-fetching.

## 6. Agent Implementation

### 6.1 Tool-Use Loop (`agent/client.py`)

Standard Claude tool-use loop:
1. Initialize session with system prompt + initial user message ("Explore the graph and produce a data collection strategy. You have access to the following tools...").
2. Send messages; receive tool calls.
3. Execute tool calls via the function dispatcher.
4. Append results to conversation; continue.
5. Terminate when agent calls `finalize_strategy` or budget is exhausted.

Implementation notes:
- Use Claude's structured tool-use API.
- Temperature: 0.3 (pre-registered).
- Run 3 sessions per condition with different seeds; aggregate per Section 6.4.

### 6.2 System Prompts (`agent/system_prompts.py`)

Two pre-registered prompts: one for A_G, one for A_NG. Both must be:
- Frozen before the experiment.
- Identical except for the tool descriptions and the framing of "available information."
- Free of phrasing that biases the agent toward particular strategies.

Both prompts include:
- The task being learned by the policy.
- The agent's role: "advise on what data to collect to improve this policy."
- Description of the tool surface.
- Constraints: target distribution of request types, total request count, budget.
- Output format: structured requests via Layer 4 tools, ending with `finalize_strategy`.

The full prompts are committed to the repository as `system_prompts/A_G.md` and `system_prompts/A_NG.md` and never modified after experiment start.

### 6.3 Trace Logging (`agent/trace.py`)

Every tool call is logged with:
- Timestamp.
- Tool name and arguments.
- Result (or error).
- Token counts (input + output).
- Cumulative budget consumption.

Final trace per session:
```
experiment_runs/{run_id}/agent_sessions/{condition}/seed_{n}/
    trace.jsonl              # Per-call records
    conversation.json        # Full message history
    submitted_requests.json  # Final strategy
    rationale.txt            # Agent's finalize_strategy summary
    budget_summary.json
```

Traces are *the* qualitative evidence for the paper. They show what the agent inspected, what it concluded, what it ignored. Worth investing in clean trace formatting for figure generation.

### 6.4 Aggregation Across Sessions

Three sessions per condition, with different temperature seeds. Aggregation:
1. **Consistency measurement.** Embed each session's submitted requests and measure cross-session similarity. High consistency means the agent reliably produces similar strategies.
2. **Selection.** Use the session with highest internal consistency (most requests with cross-session counterparts) as the primary strategy for that condition.
3. **Report.** Cross-session consistency is reported as a secondary metric. Different consistency between A_G and A_NG is itself informative.

Alternative aggregation (union with deduplication, then prune to target distribution) is reported as a robustness check.

## 7. Human Conditions

### 7.1 Human-Without-Graph (H_NG)

A trained operator (separate from the demonstration-collection operator if possible, to avoid interface contamination) is given:
- Access to the rollout pool via a simple browser: filter by outcome, view storyboards/videos.
- The same task description and role framing as the agent (delivered as written instructions).
- The same time budget the agent has in wall-clock terms (20 min default, but pre-register).
- A form for submitting `DemonstrationRequest`s with the same schema.

The human does *not* see:
- The behavior graph.
- Cluster assignments.
- V/A values.

Instructions explicitly forbid the human from drawing diagrams or external structures during the session, to keep the comparison clean.

### 7.2 Human-With-Graph (H_G, optional)

Same as H_NG, but the human additionally has:
- Access to a graph visualization.
- Access to per-cluster example storyboards.
- Access to V/A values per node.
- Access to a tool that lists rollouts by cluster path.

The interface is designed to expose roughly the same affordances as the agent's tools, but in a human-readable form.

### 7.3 Human Variance Caveat

Even with a single trained operator, human exploration produces more variance than agent runs. To partly control:
- Run 1–2 humans per human condition (logistics permitting).
- Pre-register that humans receive the same written task framing as the agent.
- Time-box strictly.
- Report human session traces qualitatively for transparency.

This is a known limitation. The agent conditions carry the comparative-rigor burden; the human conditions provide a competitive-baseline anchor.

## 8. Strategy Execution by Operator

Once each condition has produced its set of `DemonstrationRequest`s (3 sessions per agent condition, aggregated to one strategy; 1–2 sessions per human condition), the operator (the demonstration-collection operator from E2) executes them.

This stage is **identical to E2 Section 7**:
- Operator interface displays requests one at a time, blind to source condition.
- Scene validation against initial conditions.
- Per-demonstration recording.
- One retry per failed request.
- Time budget per condition matched across conditions.

The operator never sees the agent's reasoning, the graph, or which condition a request came from. Strategy queues from all conditions are interleaved in randomized order.

## 9. Adherence Scoring

**Identical to E2 Section 6**, using the graph-classification pipeline:
- Each demonstration is classified against the saved graph (UMAP + k-means centroids reused, never refit).
- Adherence axes: initial-condition, target-cluster, success-criterion, prohibition.
- For requests without `target_cluster` (A_NG and H_NG conditions), `target_cluster` is computed post-hoc by classifying the reference rollout's slice. This gives both graph and no-graph conditions comparable cluster-level annotations for analysis.
- Filter threshold: 0.6 overall (pre-registered, identical to E2).

## 10. Retraining and Evaluation

**Identical to E2 Sections 7 and 8.** For each condition (A_G, A_NG, H_NG, optionally H_G):
- 3 retraining seeds → 3 retrained policies per condition.
- Evaluation on the fixed 50-condition suite, interleaved across policies.

Statistical comparison:
- Primary: paired Wilcoxon signed-rank across retraining seeds for the headline pairs (A_G vs A_NG; A_G vs H_NG).
- Secondary: McNemar's test on per-evaluation-condition success/failure.
- Effect size: Cliff's delta or odds ratio.

With three primary pairwise comparisons, apply Bonferroni or Benjamini-Hochberg correction.

## 11. Pre-Registration Checklist

Before running, freeze and commit:

- [ ] All E2 pre-registered items (base policy, rollout pool, graph, evaluation suite, request schema, adherence weights, filter threshold).
- [ ] Tool surface specifications (full JSON schemas for all tools).
- [ ] Tool budget configuration.
- [ ] Agent system prompts (verbatim, in repository).
- [ ] Human exploration interface design (screenshots in repository).
- [ ] Human task instructions (verbatim).
- [ ] Time budgets per condition.
- [ ] Aggregation method across agent sessions.
- [ ] Statistical tests and multiple-comparison correction.
- [ ] Number of retraining seeds.

A `pre_registration.yaml` file documents all of the above and is committed before any condition runs.

## 12. Suggested Implementation Order

1. **Tool schemas** (Section 4). Write the JSON schemas first; they're the contract.
2. **Layer 1 + Layer 2 implementation.** Test against the existing graph: every Layer 1 call should return correct topology; every Layer 2 call should return correct slice/rollout content.
3. **Sanity-check agent session.** Wire Layer 1 + Layer 2 to Claude's tool-use API. Run a 15-minute exploration session and read the trace. If the agent gets stuck, redesign before proceeding.
4. **Layer 3 + Layer 4.** Add aggregation tools and submission tools. Re-run sanity check.
5. **Budget control and caching.**
6. **No-graph parallel surface.** Implement the A_NG tools. Run a sanity-check session.
7. **Human exploration interface.** Build for H_NG (and H_G if used). Pilot with a teammate.
8. **Pre-registration and freeze.**
9. **Run all conditions.**
10. **Operator execution → adherence → retrain → evaluate** (reuses E2 infrastructure).
11. **Analysis, plots, writeup.**

The sanity-check session in step 3 is the highest-information step before committing to the experiment. If the agent calls the same tool repeatedly, fails to coherently explore, or commits to a strategy without inspection, the surface needs redesign.

## 13. MCP Server Wrapper (Optional, for Demo)

Once the function-tool layer is stable, wrap it as an MCP server:

```
mcp_server/
    server.py            # MCP server exposing the graph tools
    manifest.json        # Tool descriptions and schemas
```

This unlocks:
- Interactive chat demo with the graph: any MCP-compatible client (including Claude.ai with custom connectors) can drive a conversation about the policy.
- Reuse for future experiments without rewriting the tool surface.

The MCP wrapper is not required for the core experiment. Schedule it for after the experiment is functional, ideally before paper submission so the demo can be referenced.

## 14. Risks and Failure Modes

- **Agent fails to explore meaningfully.** Calls one or two tools and immediately submits requests without inspection. Diagnose: low diversity in tool calls, low cross-session consistency. Mitigation: revise system prompt to encourage exploration, or add an explicit minimum-exploration requirement.

- **Agent exhausts visual budget too early.** Watches videos before forming hypotheses. Diagnose: high visual-call count in early turns. Mitigation: emphasize Layer 1 + textual `kinematic_summary` in system prompt; consider lower visual budget.

- **No-graph agent reconstructs clustering implicitly.** A capable VLM watching many videos may form internal categorizations. Diagnose: A_NG strategies look graph-like in cluster coverage. This is a genuine finding rather than a confound — it would mean the explicit graph offers redundant information beyond what video inspection already provides at this scale. Worth reporting.

- **Conditions produce indistinguishable strategies.** A_G and A_NG submit similar requests. Diagnose: low cross-condition divergence. Indicates the graph isn't adding actionable information beyond what outcome-only access provides. Negative result, but publishable, especially paired with E1 confirming the graph is structured.

- **Agent prompt sensitivity.** Different system prompts yield different "expertise levels." Mitigation: lock the prompt before experiment, run multiple temperature samples, report variance, do not hand-tune prompts for one condition.

- **Tool-call drift.** Agent explores indefinitely without committing. Mitigation: max_tool_calls budget enforces termination.

- **Operator distinguishes conditions from request style.** If A_G requests systematically reference cluster-like reasoning while A_NG requests don't, the operator may infer condition. Mitigation: aggressive denylist on cluster-related vocabulary in `target_behavior` and `success_criterion`; structural blinding in operator interface; randomized request order across conditions.

- **Self-validation risk.** Adherence scoring uses the same graph the agent reasoned over. If the graph mis-clusters certain behaviors, both proposal and verification inherit the same blind spots. Mitigation: E1 must precede E3. Document the dependency.

## 15. Cost and Timeline Estimate

- **Tool surface implementation:** 1 week.
- **Sanity-check sessions and iteration:** 3–5 days.
- **Pre-registration:** 1 day.
- **Agent runs:** ~1 hour wall-clock per condition × 3 seeds × 2–3 conditions = 1 day.
- **Human runs:** 1–2 humans × 1–2 conditions × ~30 min each = 1 day.
- **Operator execution:** 2 hours per condition × 3–4 conditions = 1 day.
- **Adherence + retraining + evaluation:** ~3 days (reuses E2 infrastructure).
- **Analysis and writeup:** 3 days.

Total: roughly 3 weeks if E2 infrastructure is already in place. If E2 infrastructure must be built first, add ~1 additional week.

API cost: agent sessions are token-heavy (many tool calls, image inputs). Estimate $20–50 per session, $200–500 total.

## 16. Open Questions for the Author

Decisions to make before implementation:

- **Three vs four conditions.** The fourth (H_G) significantly strengthens attribution but adds operator time. My default recommendation: include H_G if a second human is available; otherwise stick with three conditions and note the limitation.
- **Visual budget split.** 30 storyboard calls + 5 video calls is my default. Worth piloting and adjusting based on the sanity-check session. Whatever you pick, lock it before pre-registration.
- **Agent vs human time budget.** Should the human have wall-clock parity with the agent's session, or task parity (same number of "things they can inspect")? I'd default to wall-clock parity since it more directly reflects deployment realities, but it's debatable.
- **Whether to allow agent to inspect E1 results.** The agent could in principle read documentation about cluster meaning. I'd default to *not* providing E1's cluster-coherence results to the agent — the agent should infer cluster meaning from the same kinematic summaries and example slices a human would.
- **Whether the agent should know the task succeeds 30–60% of the time.** If yes, it might calibrate strategies more sensibly. If no, it has to discover this from the rollout pool. I'd default to including base success rate in the system prompt — it's information any real operator would have.

---

**Critical reminders:**

- E1 must precede E3.
- All E2 infrastructure must exist before E3 runs.
- Pre-registration is non-negotiable: with multiple conditions, multiple comparisons, and tool-use variability, the forking-paths risk is high.
- Trace logging is the qualitative backbone of the paper. Invest in clean traces.

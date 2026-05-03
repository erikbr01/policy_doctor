# Policy Doctor MCP server

Exposes the agent tool surface (Layers 1–4, plus the no-graph parallel surface) over the Model Context Protocol so any MCP-compatible client — Cursor, Claude Code, Claude Desktop, the MCP Inspector — can drive an interactive exploration session.

This is the optional/demo path described in spec §13 of [`docs/E2_agents.md`](../../../../docs/E2_agents.md). It does not replace `scripts/run_e2_agent.py`, which remains the headless reproducible-experiment entry point. Both share the same underlying tools.

For the experiment-level overview of how this fits with the headless backends, see [`docs/experiments/experiment_e2_agent_proposals.md`](../../../../docs/experiments/experiment_e2_agent_proposals.md) §"Choosing a Backend".

## Install

```bash
pip install mcp        # or: pip install 'mcp[cli]'  (also installs mcp-inspector)
```

The `mcp` package is an optional dependency — the rest of the codebase doesn't require it.

## Run standalone (e.g. with the MCP Inspector)

```bash
export POLICY_DOCTOR_CLUSTERING_DIR=/path/to/clustering
export POLICY_DOCTOR_POOL_EPISODES_DIR=/path/to/eval_episodes
export POLICY_DOCTOR_OUT_DIR=/tmp/policy_doctor_session

# Talk to it via mcp-inspector:
npx @modelcontextprotocol/inspector \
  python -m policy_doctor.vlm.proposals.agents.mcp_server
```

## Cursor configuration

Add an entry to your project's `.cursor/mcp.json` (or your global `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "policy-doctor-e2": {
      "command": "python",
      "args": ["-m", "policy_doctor.vlm.proposals.agents.mcp_server"],
      "env": {
        "POLICY_DOCTOR_CLUSTERING_DIR": "/abs/path/to/clustering_result",
        "POLICY_DOCTOR_POOL_EPISODES_DIR": "/abs/path/to/eval_episodes",
        "POLICY_DOCTOR_OUT_DIR": "/abs/path/to/cursor_session_output",
        "POLICY_DOCTOR_TASK_HINT": "Pick up the green cube and place it on the platform.",
        "POLICY_DOCTOR_CONDITION": "A_G",
        "POLICY_DOCTOR_STORYBOARD_DIR": "/abs/path/to/storyboards",
        "POLICY_DOCTOR_RAW_STATES_DIR": "/abs/path/to/raw_states"
      }
    }
  }
}
```

If your environment uses a specific Python (conda env, venv), replace `"command": "python"` with the absolute interpreter path — Cursor doesn't honor your shell's PATH:

```json
"command": "/home/you/miniconda3/envs/policy_doctor/bin/python"
```

Restart Cursor, open the MCP panel, and you should see `policy-doctor-e2` connected with all the tools (`get_graph_summary`, `list_nodes`, `propose_collection_request`, `finalize_strategy`, …) listed.

## Claude Desktop configuration

Same shape, in `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "policy-doctor-e2": {
      "command": "/abs/path/to/python",
      "args": ["-m", "policy_doctor.vlm.proposals.agents.mcp_server"],
      "env": { "...": "..." }
    }
  }
}
```

## Environment variables

| Var | Required? | Default | Notes |
|-----|-----------|---------|-------|
| `POLICY_DOCTOR_CLUSTERING_DIR`     | yes | — | Saved clustering result dir (`cluster_labels.npy`, `clustering_models.pkl`, `metadata.json`) |
| `POLICY_DOCTOR_POOL_EPISODES_DIR`  | yes | — | `eval_save_episodes` output dir (`ep*.pkl`, `metadata.yaml`) |
| `POLICY_DOCTOR_CONDITION`          | no  | `A_G` | `A_G` (full graph surface) or `A_NG` (no-graph parallel surface) |
| `POLICY_DOCTOR_OUT_DIR`            | no  | `./policy_doctor_mcp_session` | Submissions and rationale flushed here |
| `POLICY_DOCTOR_TASK_HINT`          | no  | `""` | Free-text task description shown via `get_graph_summary` |
| `POLICY_DOCTOR_STORYBOARD_DIR`     | no  | none | Sidecar storyboard dir from `build_rollout_pool` step |
| `POLICY_DOCTOR_VIDEO_DIR`          | no  | none | Sidecar video dir |
| `POLICY_DOCTOR_RAW_STATES_DIR`     | no  | none | `.npz` raw-state arrays (kinematic_summary uses them when present) |
| `POLICY_DOCTOR_MAX_TOOL_CALLS`     | no  | 10000 | Cap total tool calls per session |
| `POLICY_DOCTOR_MAX_VISUAL_CALLS`   | no  | 10000 | Cap visual (storyboard / video) calls |
| `POLICY_DOCTOR_MAX_VIDEO_CALLS`    | no  | 10000 | Cap full-video calls (subset of visual) |
| `POLICY_DOCTOR_KIN_STRATEGY`       | no  | `raw_states` | `raw_states` or `cluster_stats` |

For interactive Cursor / Claude Desktop sessions the budget caps default to "effectively unlimited." Tighten them when you want to mirror the experimental rigor of the headless `run_e2_agent.py` runs.

## What the client sees

After connecting, the MCP client receives the full Layer 1–4 tool list (or the no-graph variant for `A_NG`):

- **Layer 1 — topology:** `get_graph_summary`, `list_nodes`, `list_paths`, `get_node`, `get_edge`
- **Layer 2 — slice/rollout access:** `list_slices_in_node`, `get_slice_video`, `get_rollout_summary`, `get_rollout_video`, `list_rollouts`
- **Layer 3 — analysis:** `find_failure_nodes`, `find_recovery_paths`, `find_underrepresented_modes`, `compare_paths`
- **Layer 4 — submission:** `propose_collection_request`, `list_submitted_requests`, `revise_request`, `delete_request`, `finalize_strategy`

Every `propose_collection_request`, `revise_request`, `delete_request`, `finalize_strategy` call triggers an immediate flush to:

```
$POLICY_DOCTOR_OUT_DIR/cursor_session/
    submitted_requests.json
    rationale.txt
    budget_summary.json
```

so you can watch progress in your terminal while the session runs in Cursor.

## Differences from the headless `run_e2_agent.py` path

| Concern | Headless (`run_e2_agent.py`) | MCP server |
|---------|------------------------------|------------|
| Model invocation | `chat_with_tools(...)` against Claude / Gemini SDK | The MCP **client** owns the model — Cursor uses your Cursor-configured model; Claude Desktop uses Claude |
| Budget defaults | Tight (80/30/5) — pre-registered for experimental rigor | Loose (10k/10k/10k) — interactive use |
| Trace logging | Full `trace.jsonl` with every assistant turn + tool call | Server doesn't see assistant turns; only tool calls land in `submitted_requests.json` |
| Reproducibility | Seeded; deterministic with mock backend | Driven by interactive prompts — not reproducible by design |
| Aggregation across N seeds | Yes, via `aggregate_agent_sessions` | No — single session per server launch |
| When to use | Pre-registered experimental runs | Demo, prompt iteration, debugging the tool surface, ad-hoc exploration |

If a Cursor session produces a strategy you want to feed into the operator pipeline, the `submitted_requests.json` it writes is in the same shape as the headless path's `proposals/<condition>/selected_run.json`. Copy it into the run dir to bootstrap from a Cursor exploration.

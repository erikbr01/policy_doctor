# Operator quickstart — Experiment E2 (agentic VLM proposals)

End-to-end recipe for running an agent session, viewing the proposed
demonstrations in the operator console, and executing them in a robomimic /
mimicgen / robocasa sim. Geared toward someone who has the codebase but
has not run E2 before.

The system is **four cooperating processes**: a stateful HTTP proposal
server, a one-shot agent driver, a stateless Streamlit console, and a sim
runner. Each owns a piece; the boundaries are deliberate. You do not need
all four for every test — see "Minimal smoke test" below.

---

## ⚠ Before you start: agent-backend reality check

The agent loop needs an LLM that implements `chat_with_tools`. As of
this commit, that means **Anthropic Claude or hosted Gemini only**:

- `Qwen2VLBackend` does **not** implement `chat_with_tools` (one-shot
  `classify_slice` only). Local-VLM-as-agent is a known gap, listed
  under "Open gaps" below.
- The free-tier Gemini quota for this project shows `limit: 0` for
  `gemini-2.0-flash` and `gemini-3.1-pro`; `gemini-flash-latest` works
  for ~8 turns before hitting per-minute caps. **An A_G session needs
  on the order of 25–40 turns**, so free-tier Gemini will NOT finalize.
  Either upgrade Gemini billing or use Anthropic.

Recommended setup before running anything:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # the path that actually works
# OR upgrade your Gemini key to a billable tier; flash-latest is fine.
```

The mock backend (`--backend mock`) is the cheapest plumbing check —
runs the full agent loop without any API calls, but its scripted
submissions are rejected by the evidence gate (that's the gate
working as designed; it confirms plumbing without exercising real
proposals).

---

## 0. Prerequisites

Three on-disk artefacts and three conda envs.

### Data
- **A trained policy** in a `train_dir` produced by
  `third_party/cupid/train.py`. Anything with `checkpoints/` and a `best`
  ckpt resolvable by `scripts/run_dagger.py:resolve_checkpoint` works.
- **An eval rollout pool**: a directory of `ep<NNNN>_<succ|fail>.pkl`
  produced by `eval_save_episodes`. Image columns are required (the
  agent's storyboard tools read frames from these pkls); a low-dim-only
  pool will degrade the agent to text-only inspection. Mid-rollout
  recovery requests additionally need `sim_state` per timestep, which
  `eval_save_episodes` does **not** save by default — DAgger-saved pkls do.
  Recovery / alternative_strategy requests against eval pkls fall back to
  `env.reset()` (rollout-start) with a console note.
- **A clustering directory** with `cluster_labels.npy`, `metadata.json`,
  `manifest.yaml`, and (for offline-friendly node summaries)
  `clustering_models.pkl`. Built by the
  `policy_doctor.curation_pipeline.steps.run_clustering` step against the
  same eval pool. The seed-0 r512 transport_mh clustering at
  `/tmp/transport_mh_seed0_r512_clustering` works as a reference example.

### Envs
- `policy_doctor` — proposal server, agent, Streamlit console.
- `cupid` (or `mimicgen_torch2`) — sim runner; needs the diffusion-policy
  stack + robosuite + robomimic.
- `cupid_torch2` — only if you also want the runtime classifier; the
  E2 pipeline does not require it for proposals or sim execution.

### Backend (the LLM that drives the agent)
- **Gemini** (default; uses `GEMINI_API_KEY` from `.env`). Working models
  in this repo as of writing: `gemini-flash-latest`, `gemini-2.5-flash`,
  `gemini-pro-latest`. The free tier hits 503 / 429s often — the backend
  retries with exponential backoff, but persistent overload still fails
  the session. `gemini-2.5-flash-lite` is too weak; it gives up after
  one tool call.
- **Claude** via `ANTHROPIC_API_KEY` (recommended for serious runs;
  `claude-sonnet-4-6` is the standard pick).
- **Qwen2/3-VL backends** **do not implement `chat_with_tools` yet** — they
  cover `classify_slice` only. Local-VLM-as-agent is a known gap; see the
  **Open gaps** section at the bottom.

---

## 1. Start the proposal server

```bash
conda activate policy_doctor

cat > /tmp/e2_cfg.yaml <<EOF
run_id: e2_$(date +%Y%m%d_%H%M%S)
run_dir: /tmp/e2_run/${run_id}
pool_episodes_dir: /path/to/eval_save_episodes/.../latest/episodes
clustering_dir: /tmp/transport_mh_seed0_r512_clustering
task: transport_mh
task_hint: "robomimic transport_mh: two Franka Panda arms cooperatively..."
mode: agentic
chat_enabled: false
EOF

python -m policy_doctor.vlm.proposals.server --config /tmp/e2_cfg.yaml --port 5003
```

It logs `[INFO] Running on http://127.0.0.1:5003`. Leave this terminal open.

The server owns the rollout pool, behavior graph, request queue, and
adherence scoring. It is the single source of truth — every other process
talks to it via HTTP.

---

## 2. Run an agent session

In a second terminal, drive one or more agent sessions. The session writes
the artefacts to disk; the server is *not* involved here.

```bash
conda activate policy_doctor

python scripts/run_e2_agent_transport_mh.py \
    --condition A_G --seed 0 \
    --backend gemini --model_id gemini-flash-latest \
    --max_tool_calls 30 --max_visual_calls 22 \
    --out_dir /tmp/e2_session_seed0
```

What happens:
1. Loads `cluster_labels.npy` + `metadata.json` from the clustering dir.
2. Builds a `BehaviorGraph` and a `RolloutPool` from the eval episodes.
3. Hands the agent a tool surface (graph topology, slice/rollout video,
   submission). The agent inspects the graph, looks at storyboards,
   submits 0–N `DemonstrationRequest` objects, and finalizes.
4. Writes `submitted_requests.json`, `trace.jsonl`, `rationale.txt`,
   `session_summary.json`, `conversation.json`, `budget_summary.json`,
   and a self-contained `session_report.html`.

Open the HTML report in a browser. **Read it as a critic** — see the
"Critical analysis" doc for the failure modes that have shown up in
practice. A passing run does not mean a useful run.

---

## 3. Enqueue requests onto the server

The agent run writes to `--out_dir`; the server doesn't auto-pick it up.
Bulk-import the whole session in one call. Convenient wrapper:

```bash
python scripts/enqueue_session.py /tmp/e2_session_seed0
# or with a non-default server URL:
python scripts/enqueue_session.py /tmp/e2_session_seed0 --server http://localhost:5003
```

Or hit the endpoint directly:

```bash
curl -s -X POST http://127.0.0.1:5003/requests/import_session \
    -H "Content-Type: application/json" \
    -d '{"session_dir": "/tmp/e2_session_seed0"}'
```

The endpoint reads `submitted_requests.json` from that dir, validates
every entry against the same denylist + rollout-id checks the server
applies elsewhere, and pushes all of them onto the operator queue.
Returns the list of added request_ids and any skipped entries with
reasons. Idempotent on `request_id`, so re-running is safe.

Single requests still work via `POST /human_request` (rationale required;
limited to `H_NG` / `H_G` conditions). For bulk import, prefer the
endpoint above.

---

## 4. View requests in the operator console (optional)

```bash
conda activate policy_doctor
streamlit run policy_doctor/streamlit_app/app.py
```

The "E2 Console" tab points at `http://127.0.0.1:5003`. The active
request panel shows the operator-facing fields plus the agent's reference
storyboards (the `reference_storyboard_ids` field, rendered in a 3-column
grid via `GET /storyboards/<id>`). The condition (`target_cluster`,
`source_condition`) is stripped server-side and never displayed.

The console is stateless. Reload-safe; close and reopen freely.

---

## 5. Drain the queue with the sim runner

In a fresh terminal with the **cupid** env (sim stack):

```bash
conda activate cupid          # or mimicgen_torch2

cd third_party/cupid          # required: third_party/cupid is on $REPO_ROOT
python ../../scripts/run_e2_sim.py \
    task=transport_mh \
    train_dir=/path/to/train_dir \
    proposal_server=http://localhost:5003 \
    output_dir=/tmp/e2_demos
```

Each request:
1. Pulled from `GET /requests/active` (operator-view, condition-stripped).
2. Sim env reset to the request's reference state. If the eval pkl has
   `sim_state` (DAgger-saved), exact replay; otherwise `env.reset()`
   (rollout-start) with a console note.
3. Operator teleoperates one episode (default keyboard controller; pass
   `dagger_config=spacemouse_default` for SpaceMouse).
4. Saved demo pkl + success flag posted back to
   `POST /requests/<id>/result`. The server runs adherence scoring (cluster
   axis re-classifies the demo; success axis is from the operator-reported
   bool; init_condition is 1.0 in sim).

When the queue is empty, the runner exits.

---

## 6. Configuration knobs

All under `policy_doctor/configs/e2/defaults.yaml`. The ones that change
the *experiment* (vs plumbing):

### Storyboard rendering — what the agent sees
```yaml
agentic:
  storyboard:
    n_frames: 5            # cells per camera row, sampled across the padded window
    pad_before: 12         # extend the slice window earlier (frames)
    pad_after: 12          # extend later. Wider context => agent sees the
                           # failure unfold rather than a near-static snapshot.
    target_size: [1024, 1024]
    cameras: null          # null = auto: agentview + every wrist camera that
                           # exists. For single-arm tasks, defaults to scene + wrist0.
```

The **temporal padding** is the biggest lever: a 5-frame slice
(`pad_before=0, pad_after=0`) is too narrow for transport — the storyboard
panels barely change. With `pad_before=pad_after=12` over 4–5 frames sampled,
the agent sees ~30 frames of context, which is enough to see a grasp
attempt unfold. The **wrist camera** is the next-biggest lever: at
512×512 the gripper-object interaction is legible only with the
eye-in-hand view; the agent-view alone is too far.

### Budget
```yaml
agentic:
  budget:
    max_tool_calls: 80      # total tool calls (read + visual + video)
    max_visual_calls: 30    # storyboard fetches; the expensive bucket
    max_video_calls: 5      # MP4 fetches; not implemented yet, keep low
    max_session_duration_s: 1200
```

### Submission gates (in submission tool, not config)
- `_RECOVERY_MIN_FRAME = 1` — recovery requests cannot start at frame 0.
- `_MIN_EVIDENCE_ITEMS = 3` — A_G submissions need ≥3 inspected
  `evidence_slice_ids`; A_NG needs ≥3 `evidence_rollout_ids`. The agent
  cannot cite an id it did not fetch.

### Choosing the sim
- `square_mh`, `lift_mh`, `transport_mh` — robomimic.
- `robocasa_layout_lowdim` — robocasa.
- For mimicgen tasks, swap the env via the cupid env's standard
  mimicgen wrappers and add a TASK_CONFIG entry in `scripts/run_dagger.py`
  with `dataset_path` pointing at the mimicgen low-dim hdf5.

### Sane defaults checklist (for a clean overnight run)
- Backend: `gemini-flash-latest` (or claude-sonnet-4-6 if you have a key).
- Budget: 30/22/0/1800 (tool/visual/video/seconds).
- Storyboard: defaults above (3 cameras, 5 frames, ±12 pad, 1024×1024).
- One A_G session and one A_NG session, seed=0 each, before any sweep.
- Read the HTML report critically before queueing for sim.

---

## Minimal smoke test (no sim, no operator)

If you only want to verify the agent emits sensible proposals against your
data:

```bash
python scripts/run_e2_agent_transport_mh.py \
    --condition A_G --seed 0 \
    --backend gemini --model_id gemini-flash-latest \
    --out_dir /tmp/e2_smoke
xdg-open /tmp/e2_smoke/session_report.html   # or open in browser manually
```

Inspect the storyboards in the report. For each submission, ask yourself:
- Does the cited evidence actually show the failure the prose describes?
- Are the 3 cited slices a *pattern*, or 3 lookalikes?
- Could a human operator follow `target_behavior` literally?
- Does the prohibition contradict the target_behavior? (This has bitten us.)

---

## Open gaps (worth knowing)

1. **Local-VLM-as-agent**: `Qwen2VLBackend` does not implement
   `chat_with_tools`. To run the agent locally on a 32B VLM, the Qwen
   function-calling output (Hermes-style or native) needs to be plumbed
   through the same `AssistantTurn` shape Claude/Gemini emit. Until then,
   only API-hosted models drive the agent loop.
2. **Mid-rollout init_state**: `eval_save_episodes` doesn't record
   `sim_state`, so recovery / alternative_strategy requests against eval
   pkls degrade to rollout-start. Re-render with DAgger to get exact
   replays, or add an `eval_save_with_states` flag to your eval pipeline.
3. **Free-tier Gemini 503/429**: persistent overload still kills sessions
   even with backoff. The pragmatic answer is a paid Gemini key or
   Anthropic. The retry handles transient burst limits; not quota
   exhaustion.
4. **Operator authentication**: the proposal server has no auth. Run on
   `127.0.0.1` only; do not expose to a network without a reverse proxy.

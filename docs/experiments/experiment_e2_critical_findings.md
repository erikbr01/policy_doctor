# E2 — Critical findings from overnight run-and-verify

Adversarial notes on the agent's actual outputs against `transport_mh`
seed-0 r512 clustering. Written from the position of "I doubt the agent;
let me see if I can disprove its claims by looking at the imagery itself."
Findings shaped the rendering refactor (cameras + temporal padding) and a
gate fix that landed in commit `5f9bd43`.

The overnight constraint: only Gemini API was available (no Anthropic key
in env, Qwen lacks `chat_with_tools`), and the free-tier Gemini quota
caps a session at 6–8 turns. So the **fresh** runs are diagnostic, not
final; the fully-finalized session I critique below is the prior
Claude-driven A_G run at `/tmp/qwen32b_4bit_plwiogm_/session/` (the
"qwen32b" in the dirname is misleading — Qwen has no agent loop).

## 1. The agent fabricated for two of three clusters

The session submitted three recovery requests, one each for clusters
c11, c14, c6. All three rationales told the same story: "the gripper
closes while misaligned over the hammer." I extracted and re-rendered
the cited storyboards with a wider temporal window (±12 frames around
the slice) and three cameras (agentview + both wrist cams):

- **c6 (sub3)**: defensible. The wrist camera clearly shows the hammer
  with a metallic head and wooden handle. Across the 3 cited slices the
  gripper is in active contact with the object, and at the rendering
  resolution I can read whether it's positioned over the head or the
  handle. The agent's distinction is at least *visible*.
- **c11 (sub1)** and **c14 (sub2)**: not defensible. In all 6 cited
  slices for these clusters, the wrist cam shows the gripper hovering
  in mid-air over the floor, no object engagement, and the agentview
  shows static bins with the hammer visible inside one bin but not
  being manipulated. **The "closes misaligned over the hammer's head"
  claim has no support in the imagery the agent itself cited.**

This is the core failure mode the instructions doc warns about: the
agent generates prose from a prior (the task description includes
"hammer," and a "misaligned grasp" story is a strong template), then
back-fills evidence by citing whatever slices the cluster contained,
without verifying that the cited frames actually show the failure mode.
The evidence gates (≥3 cited storyboards, all from inspected slices)
catch *some* of this — they ensure the agent looked at *something* —
but not whether the something was the failure it described.

**What the gates currently do not catch**: an A_G submission where
target_behavior describes failure mode X but the cited storyboards show
a different mode Y or no clear failure at all. Beating this requires
either an automated grounding check (e.g. a second-pass VLM call asking
"do these 3 storyboards depict <target_behavior failure>?") or a strong
"if you cannot see the failure happening in your evidence, do not
submit" rule in the system prompt with adversarial examples. Worth
trying both, ideally additively.

## 2. Internal contradictions the agent did not catch

Even when prose was plausible, the cross-field consistency was poor:

- **Sub2 (c14)**: target_behavior says "reposition the gripper directly
  above the hammer's handle, then close." Prohibition: "do not close the
  gripper while the end-effector is misaligned over the hammer's
  handle." The operator is told to align with the handle and then warned
  not to be aligned with the handle. The two fields contradict.
- **Sub3 (c6)**: reasoning text says "failure pattern: closes while
  misaligned over the hammer's head" — and then the request targets the
  head as the recovery, with a prohibition about handle misalignment.
  The remediation is on the wrong part.

Pattern: the agent treats target_behavior, prohibitions, and reasoning
as three independent slots and does not cross-check that they describe
*the same* operator action. A short post-submit consistency check
(non-LLM, just a string-similarity / contradiction sniff) would catch
the most obvious cases. The sniff itself is hard to write robustly
without false positives, so the more leveraged fix is to require the
agent to submit a single "operator script" string and have the schema
*derive* prohibitions / reasoning from it — but that's a bigger redesign.

## 3. The duplicate-target gate had a hole

`propose_collection_request` enforces case-/whitespace-normalized
unique target_behavior across submissions. `revise_request` did not.
Submissions #1 and #2 in the prior session ended with literally
identical target_behavior strings because the agent revised #2's text
post-submission and the revise path only ran the schema validator.
**Fixed in commit 5f9bd43**: revise now runs the same duplicate check
against `ctx.submitted` (excluding the request being revised) and rolls
back on conflict.

This was a one-line audit-and-fix, but it is a worked example of why
gate logic should live in *one* function applied at every write path,
not duplicated. The next gate that gets added will probably make the
same mistake; consolidating is on the punch list.

## 4. The original storyboards were unreadable for grasp judgments

The pre-fix renderer produced a 4-panel 512×512 storyboard from one
camera over the cited 5-frame slice. Two layered failures:

- **Camera**: the default picked `img` (which is the agent-view stitched
  composite at this dataset's r512 setting), not a wrist camera. The
  hammer at the agent-view distance is ~10 px — the agent literally
  could not read its head/handle even if it tried.
- **Temporal context**: a 5-frame slice means the 4-panel storyboard
  picks 4 frames at indices 0,1,3,4 within the window — adjacent
  timesteps where nothing has visibly changed. Several cited storyboards
  show 4 indistinguishable frames.

The combined effect: when the agent rationalized about "misalignment,"
it was rationalizing over imagery from which misalignment cannot be
judged. A model behaving honestly here would say "I cannot tell from
these frames" and decline. Instead, the model produced the prior-driven
template story.

**Fixed in 5f9bd43**: defaults are now 3 cameras (agentview + both
wrist cams), 5 frames sampled across a ±12-frame padded window, on a
1024×1024 canvas. Re-rendering the prior session's evidence with these
defaults makes the c6 grasp judgment legitimately legible; it also
makes c11/c14 *visibly disprove* the agent's claim — which the original
narrow rendering hid.

## 5. The agent did not signal uncertainty when imagery was uninformative

Across both the Claude-driven session (full finalize) and the truncated
flash-latest session (8 turns, quota-exhausted before submission), the
agent never emits "I cannot tell from this view" or "this storyboard
does not show what I expected." It always finds something to say.
This is consistent with general LLM behavior on high-confidence prior
domains. Two mitigations worth trying:

- **Calibration prompt**: explicit system-prompt rule "if a storyboard
  shows no robot-object interaction, treat it as evidence *against*
  submitting from this cluster, not as evidence for any particular
  failure mode." (Cheap; might help; might over-correct.)
- **Adversarial dummy slices**: include 1–2 deliberately uninformative
  slices in the inspectable set per cluster (e.g. frames before the
  episode starts) and check whether the agent still cites them as
  evidence. If yes, the fail-loud rule isn't holding. (Diagnostic, not
  a fix — but informative.)

## 6. The agent produced reasonable orientation behavior

Worth noting on the *positive* side: the tool-call sequence both
agents followed was sensible — `get_graph_summary` →
`find_failure_nodes` (low min_failure_prob threshold) → `get_node` on
high-failure clusters → `list_slices_in_node` → `get_slice_video` on
3+ slices per cluster. The agent loop, budget tracker, evidence gate,
and inspection bookkeeping all behaved as designed. The failure is in
the *grounding* step (does the prose match the imagery), not in the
plumbing.

## 7. What I am *not* claiming

- I am not claiming the agent will always fabricate. The c6 submission
  was at least visually defensible after re-rendering. With wider
  storyboards and a calibration prompt, the rate of grounded
  submissions should rise. Whether it rises *enough* for the experiment
  to differentiate A_G from A_NG is an empirical question I have not
  answered tonight.
- I am not claiming Gemini-flash-latest is the right backend. It got
  8 productive turns before the free-tier quota; it got further than
  flash-lite (which gave up after 1 turn) but neither finalized. A
  serious run needs Claude or a paid Gemini key; the current overnight
  quota is too tight even for one A_G session.
- I am not claiming the operator path is verified end-to-end. I read
  the code and patched the `sim_state`-missing fallback; I did not
  bring up the full server + agent + sim drainage stack against
  transport_mh tonight. The instructions doc walks through it; the
  user should expect to debug at the env-construction step the first
  time. The most likely failure points are (a) the
  `scripts/run_dagger.py:TASK_CONFIG[transport_mh]` obs_keys list
  missing the second arm's keys, and (b) the
  proposal_server's `boot()` requiring a `pool_episodes_dir` whose
  episodes share the indexing convention the cluster metadata uses.

## 7b. Update — live-agent verification of the rendering refactor

After committing the storyboard refactor I retried a fresh A_G session
on `gemini-flash-latest` against the same transport_mh seed-0 r512
clustering. Free-tier quota cut the run at turn 12, so it didn't
finalize, but the tool-call sequence was:

```
get_graph_summary → find_failure_nodes(0.3) → list_nodes(>=0.5)
→ get_node(c5) → get_node(c11) → get_node(c10)
→ list_slices_in_node(c5)
→ get_slice_video(r0147_t64_t68)
→ get_slice_video(r0120_t72_t76)
→ get_slice_video(r0187_t72_t76)
→ list_slices_in_node(c10)
→ get_slice_video(r0039_t66_t70)
→ [quota-exhausted before submission]
```

I extracted those four storyboards and looked at them
(`/tmp/run3_evidence_jpegs/`). With the new defaults:

- `r0147_t64_t68` (c5): the **top row** (agentview) shows the robot
  picking up the hammer, lifting it, and dropping it onto the floor
  across 5 frames; the hammer is visible mid-air and on the ground
  afterwards. The **middle row** (robot0 wrist) shows the gripper
  holding then releasing the hammer's wooden handle. The failure mode
  is now legible from the storyboard alone.
- `r0120_t72_t76` (c5): hammer slips out during pickup; visible across
  agentview and wrist cam. Same failure pattern as the previous slice,
  confirming that c5 *is* a coherent grasp-failure cluster.

Compare with the pre-refactor evidence I extracted earlier
(`/tmp/evidence_jpegs/sub1_*` from the prior session): those panels
were near-static and one-camera, and the failure was not visible in
them at all. The contrast is the load-bearing argument that the
refactor moved the needle: the agent now has imagery *capable* of
grounding its prose. Whether it will use that capability correctly in
finalized submissions depends on running long enough to submit, which
needs a paid backend.

## 7c. Update — Qwen3-VL-32B 4-bit ran the agent end-to-end

After committing `Qwen3VLAgentBackend` (`policy_doctor/vlm/backends/
qwen3_vl_agent.py`, commit 6f3f958), I ran the same A_G session driver
(`scripts/run_e2_agent_transport_mh.py --backend qwen --model_id
Qwen/Qwen3-VL-32B-Instruct`) on the rebuilt seed-0 r512 transport_mh
clustering. The model loaded in 4-bit NF4 across two GPUs, ran the full
agent loop for 28 turns, and **submitted 3 evidence-grounded requests
before OOMing during the final-turn generation**. Headline win: the
local-VLM-as-agent path now works without an API key.

### What I read in the storyboards (verbatim notes)

**Sub#1 (c5, recovery, target=head):** evidence
`r0147_t64_t68 / r0120_t72_t76 / r0187_t72_t76`. The agentview shows the
robot arm pick up the hammer, lift it, and drop it onto the floor across
the padded window. The robot0 wrist cam shows the gripper holding the
hammer by the **handle** at the moment of drop — the wooden stick is
clearly visible inside the gripper. The agent's prescription ("reposition
above the head") is therefore a coherent re-grasp suggestion grounded in
the imagery: the failure was a handle grasp; the fix is a head grasp.
**Verdict: defensible.**

**Sub#2 (c11, recovery, target=handle):** evidence `r0166_t42_t46 /
r0156_t22_t26 / r0194_t20_t24`. Same fabrication pattern as the prior
Claude session. The agentview shows static bins with the hammer sitting
inside; no active manipulation. The robot0 wrist cam shows empty wood
texture / table edge — the gripper is not engaged with the hammer. The
agent's claim "the gripper closes too early or misaligns" has **no visual
support in any of the three cited storyboards**.

**Sub#3 (c4, full_trajectory):** evidence `r0018_t6_t10 / r0151_t6_t10 /
r0091_t0_t4`. The agentview shows what looks like the start of a rollout
(robot in initial pose, hammer in bin); the wrist cam shows a generic bin
wall. The agent's reasoning text ("the same failure: the hammer is
dropped, the gripper closes too early or misaligns") **copy-pastes from
the recovery sub#2 template** even though this submission is a
`full_trajectory` request and the cited frames are early in the
trajectory (not failure frames at all). The text reuse is sloppy.

### What this changes about my prior conclusions

The c11 fabrication is reproducible across two distinct LLMs (Claude in
the original session, Qwen3-VL-32B here). Both models invent the same
"gripper closes misaligned" story. This is **not an LLM hallucination
issue per se** — it's a *data* issue: the slices the clustering chose to
represent c11 do not show the failure mode that c11 supposedly
characterises. Either:

* the cluster's centroid window is offset from the actual failure event
  in the rollout (and the failure happens 20+ frames before/after the
  cited window — the new ±12 padding partially mitigates this but not
  fully);
* or c11 is a "transition" cluster whose entries are *near* the failure
  but not *at* it, and the agent's prior about high failure_likelihood
  is what's driving the prose (post-hoc rationalization);
* or the eval pool genuinely contains slices in c11 where the gripper
  never engages, and the cluster is heterogeneous.

This is testable: pull all c11 slices, sort by `centroid_distance`, and
look at the closest 10 instead of just 3. If they all look like the
empty-floor wrist cams, the cluster is genuinely uninformative for grasp
judgement; the experiment should down-weight c11 (or any cluster whose
slices fail a "robot-object-engagement" heuristic) before the agent
sees it.

### Mechanical/operational notes from the Qwen run

* `revise_request` duplicate-gate fix from commit 5f9bd43 fired in
  production: the agent's first submission text for c5 was identical to
  the upcoming c11 text; the gate rejected it; the agent rolled back and
  revised to differentiate by "head" vs "handle". The revision_history
  field on sub#1 records this verbatim. The fix worked.
* OOM at turn 28 during the long final-turn generation. GPU 1 had 1 GiB
  free of 23.5 GiB; bnb activation memory grew with context length.
  Mitigation: lower `--max_new_tokens` (1024 → 512), or cap the
  context window with a sliding-window prune in the session loop after
  N turns. The 3 submissions had already persisted, so no work was lost.
* Run took ~5–7 minutes for model load + 28-turn agent loop.

## 9. Gemini-3.1-pro runs on Vertex AI — rendering and grounding experiments

After setting up a GCP billing account (`gcp-driven-data`, Vertex AI, global
region), a series of A_G sessions were run with gemini-3.1-pro-preview to
probe different visual input modes. All sessions: transport_mh seed-0 r512
clustering, same eval pool.

### 9a. Baseline composite storyboard (gemini31pro_A_G_seed0)

**11 turns, 3 submissions, stop=finalize.** Most efficient session so far.
Notable: the model fetched all three cluster nodes and all three slice lists
in parallel before touching visual budget, then watched 3 storyboards per
cluster in a single batched turn. The parallel execution pattern is
qualitatively better than any prior session.

c5 diagnosis: "Arm-1 holding hammer by the very tip of the handle, leading
to slipping." c11: same prior-driven "misaligned gripper during handoff"
fabrication as Qwen. c6: "Arm-0 closes too early while hovering."

### 9b. Per-frame storyboard + per-frame state/action text
(gemini31pro_A_G_seed0_frames_state)

Added two features: `mode="frames"` (5 separate images per slice at full
per-frame resolution instead of a composite grid) and `include_state_text=True`
(appends the 14-dim action vector + 24-dim obs tail at each displayed frame,
giving the model ground-truth gripper open/closed state).

**17 turns, 5 submissions.** The state text produced two diagnosis reversals
attributable to numeric grounding:
- **c5**: prior sessions said "grasp slips." With state text, the model read
  that the gripper is CLOSED at t=64-76 (the hammer is held, not slipping)
  and diagnosed the failure as "arm-1 holds hammer over goal bin but fails
  to lower and release." Completely different prescription.
- **c11**: prior sessions said "misaligned grasp." With state text, the model
  read the gripper as CLOSED and diagnosed "arm holds hammer in awkward
  vertical orientation post-handoff." Still unclear whether this is
  grounded — c11 slices remain the open question.

New clusters c14 and c19 also explored and given specific diagnoses.

### 9c. Inline MP4 video (gemini31pro_A_G_seed0_video_v2)

Replaced storyboard images with inline MP4 clips (cv2-rendered, 256×192 per
camera, 8 fps, ±12-frame window), passed as `inline_data` blobs to Vertex AI.

**8 turns, 4 submissions, stop=finalize.** The shortest and most efficient
session yet. Key finding: the model batch-parallelised EVERYTHING — 5×get_node,
5×list_slices, then 3×get_slice_video twice, then all 4 proposals submitted
in a single parallel turn. Total: 6 video calls, 8 turns to finalize.

Motion evidence enabled a new insight: the model proposed both a `recovery`
AND an `alternative_strategy` for the c11 handoff — "place hammer on surface
between bins, arm-1 picks it up" — indicating it inferred that the mid-air
coordination itself is the hard part. This alternative strategy was not
produced by any prior session. However, the c11 prose ("Arm-1 closes gripper
while misaligned during handoff") still looks like a prior-driven story; the
motion data was not enough to override the template.

### 9d. Two-stage session: description then proposals
(gemini31pro_A_G_seed0_twostage_v3)

A new pipeline splits the agent loop into two stages:

**Stage 1 (description):** a separate session with storyboard frames + state
text but NO submission tools. The agent's only output is `finalize_descriptions`
— a structured JSON array of literal per-cluster observations: gripper states,
contact/no-contact, object location, sequence of events. Explicitly forbidden
from using failure-mode language.

**Stage 2 (proposals):** a fresh A_G session with **zero visual budget**.
Receives Stage 1 descriptions as text in the user message. The `inspected_slices`
from Stage 1 are pre-populated so the evidence gate passes without re-watching.
Cannot deviate from the descriptions — has no images to reach for.

**Stage 1 results (5 clusters described, 10 visual calls, stop=finalize):**

| Cluster | Informative | Contact | Literal observation |
|---------|------------|---------|---------------------|
| c5  | ✓ | both arms | "Both arms' grippers closed around hammer, held horizontally above bins. Neither releases. Stationary." |
| c11 | ✗ | none | "Right arm hovering, gripper open, hammer on floor. No contact." |
| c14 | ✗ | none | "Right arm hovering, gripper open, hammer on floor. No contact." |
| c6  | ✗ | none | "Right arm hovering, gripper open, hammer on floor." |
| c12 | ✗ | none | "Right arm hovering, gripper open, hammer on floor." |

**This is the first session in which c11, c14, c6, and c12 were correctly
identified as uninformative.** Every prior session (Claude, Qwen, Gemini in
single-stage) invented failure-mode stories for these clusters. The description
constraint and the literal-observation requirement forced the model to report
"no contact" honestly.

**c5 received a new diagnosis**: "handoff deadlock — both arms hold the
hammer simultaneously, neither releases." This is different from every prior
session's c5 diagnosis. Whether it is more accurate than the single-stage
readings requires verification against the actual storyboards.

**Stage 2 results (18 turns, 5 submissions, zero visual/video budget):**

All 5 submissions target c5 exclusively (the only informative cluster). The
proposals are varied: 2× recovery from different rollout frames, 1× full
trajectory with explicit handoff timing instruction, 1× alternative strategy
(place-and-pick instead of mid-air handoff), 1× recovery focused on arm
release timing. Every submission is grounded in the single Stage 1 sentence
about c5 — the model cannot fabricate because it has no images.

**What the two-stage approach does and does not fix:**

Does fix: prior-driven fabrication for uninformative clusters. Does not fix:
the underlying data quality problem (c11 etc. slices genuinely don't show the
failure mode the cluster is supposed to characterise). A downstream fix is
to pre-filter clusters whose Stage 1 descriptions are uninformative before
the agent runs — the `run_exploration_session` taxonomy already provides
`recommended_for_submission: false` flags for this purpose.

Does not fix: the narrowness of Stage 2 output. With only one informative
cluster, Stage 2 produces 5 variations on one theme. The experiment needs
either better clustering (so informative slices represent each failure mode)
or a higher visual budget in Stage 1 (to watch more slices per cluster until
informative frames are found).

### 9e. Infrastructure added during this round

- Vertex AI mode for Gemini backend (`vertexai=True`, `location` param)
- `mode="frames"` and `include_state_text` for `get_slice_video`
- Inline MP4 via cv2 (`mode="video"`), with `VideoBlock` content type
- `run_description_session` + `run_two_stage_session` in `run.py`
- `finalize_descriptions` terminal tool + `description.md` system prompt
- `render_twostage_session.py` — combined Stage 1 + Stage 2 HTML report
- `build_description_tool_registry` in `registry.py`
- Bug fix: parallel `function_response` parts batched into single Gemini Content
- Default clustering path moved from `/tmp` to SSD

## 10. Recommended next experimental moves

In rough priority order:

1. **Investigate c11 cluster quality** — pull all c11 slices sorted by
   centroid distance, render the closest 10 with the three-camera ±12-frame
   storyboard. If they all show "gripper open, no contact, hammer on floor,"
   the cluster's representative window is offset from the actual failure event.
   Options: (a) down-weight c11 in the agent's tool surface, (b) fix the
   clustering window to capture the failure moment, (c) pre-filter via
   Stage 1 uninformative flag before the agent sees the cluster.
2. **Increase Stage 1 visual budget and slice coverage** so more than one
   cluster comes out informative. Current budget (12 visual calls) only
   covers ~5 clusters × 2 slices. 20 calls with the constraint "try 3
   slices if the first 2 are uninformative" would give better coverage.
3. **Run the two-stage pipeline on gemini-er-1.6** once Vertex AI access
   is granted for the `gcp-driven-data` project (or a billing-enabled
   Gemini API key is created). The robotics-specific model may produce
   more grounded Stage 1 descriptions.
4. **Run A_NG two-stage** to have the comparison pair. A_NG Stage 1 uses
   the no-graph tool surface for cluster discovery; Stage 2 proposals
   should still be grounded in the same descriptions.
5. **DAgger-save the eval pool** (eval_save_episodes with `save_states=true`)
   so mid-rollout init_state works for recovery requests.
6. **Skip H_G / H_NG conditions for now**. The agentic conditions are the
   load-bearing comparison; human conditions are a separate experiment.

# Refactor Findings Log

Running log of discoveries, gotchas, and decisions made while executing `REFACTOR_PLAN.md`. Append-only; each entry dated.

Format per entry:
```
## YYYY-MM-DD — <phase> — <one-line title>
**Context:** what we were doing
**Finding:** what we discovered
**Decision / action:** how we handled it
**Plan impact:** none / amendment to §X of REFACTOR_PLAN.md
```

---

## 2026-05-27 — Phase 0 — Refactor branch created

**Context:** Starting Phase 0 of the refactor. Modified `policy_doctor/configs/dagger_run.yaml` and `scripts/experiments/run_viz_server.sh` (DAgger demo defaults) were committed to `main` first to clean working-tree state.

**Finding:** Branch `refactor/clean-architecture` cut from `main` at commit `5a5d4ba`. `REFACTOR_PLAN.md` carried over (untracked on main, now under refactor branch).

**Decision / action:** All phase work lands on this branch. `main` continues to serve the deployed user-study Streamlit apps until Phase 6 cutover.

**Plan impact:** none.

---

## 2026-05-27 — Phase 0 — Local conda env naming differs from CLAUDE.md

**Context:** Tried to run a quick import smoke test under `conda run -n policy_doctor python -c …` per CLAUDE.md.

**Finding:** This Mac has no `policy_doctor` env. The local envs are `cupid`, `cupid_robosuite`, `lerobot`, `libero`, `robomimic`. The `cupid` env imports `policy_doctor`, `policy_doctor.behaviors.clustering`, `policy_doctor.behaviors.behavior_graph`, and `policy_doctor.data.structures` cleanly — so it's a superset for local dev.

**Decision / action:** For Phase 0 golden-snapshot generation on this Mac, use `conda run -n cupid`. The plan's `analysis` uv env will be the long-term home for these tests; this is a local-dev workaround, not a plan change.

**Plan impact:** none. Worth noting that uv setup in Phase 1 needs to be robust to whatever the dev machine has — the bootstrap script should handle "I have a different conda env" gracefully.

---

## 2026-05-27 — Phase 0 — Deferring MimicGen seed-selection golden anchor

**Context:** Plan §5.1 lists MimicGen seed selection as one of four golden anchors.

**Finding:** The three heuristics (`BehaviorGraphPathHeuristic`, `DiversitySelectionHeuristic`, `RandomSelectionHeuristic`) require both a synthesized rollouts HDF5 (the trajectory loader `MimicGenSeedTrajectory.from_rollout_hdf5()` enforces a schema with state/action/reward/done datasets and demo-level success attrs) and a fitted `BehaviorGraph`. Setup is ~3-4x the size of the other three anchors and the heuristic outputs depend transitively on every other anchor we're snapshotting.

**Decision / action:** Ship the three simpler anchors (clustering, behavior graph, influence-slice) in the Phase-0 first commit. Add MimicGen seed selection in a follow-up Phase-0 commit before declaring Phase 0 done. Tracked as a TODO at the bottom of `tools/refactor/snapshot.py`.

**Plan impact:** none — the anchor is still part of Phase 0, just split across two commits.

---

## 2026-05-27 — Phase 1 — Sim extras locked, deferred validation to Linux

**Context:** Adding `cupid`, `mimicgen`, `robocasa` to pyproject.toml so the conda env yamls can be retired.

**Finding:**
- The three sim extras pin mutually-incompatible robosuite/robomimic versions. uv refuses to resolve a single dep graph; needed `[tool.uv] conflicts = [[{extra = "cupid"}, {extra = "mimicgen"}, {extra = "robocasa"}]]` so uv resolves each extra in isolation.
- `mimicgen==1.0.0` is GitHub-only (NVlabs/mimicgen). Added `[tool.uv.sources] mimicgen = { git = ... }` pointing at default branch (resolved to `72bd767c`).
- Three legacy sim packages have broken build metadata under modern setuptools and won't build on Python 3.12:
  - `gym==0.21.0` — `extras_require` formatting rejected by setuptools ≥ 67.
  - `pybullet-svl==3.1.6.4` — `setup.py` imports `pkg_resources` which isn't in `build-system.requires`.
  - Likely also `dm-control 1.0.9` (didn't get that far in the trace).
  All are transitive deps of `robomimic==0.2.0`; the original conda env worked because conda pinned Python 3.10 and old setuptools. Dropped explicit pins; let the resolver pick what robomimic 0.2.0 transitively accepts and accept that `uv sync --extra cupid` may need an older Python or build-isolation overrides on first use.
- Workspace setup: `third_party/cupid` and `third_party/cupid/third_party/infembed` declared as workspace members. `cupid-workspace` and `infembed` declared via `[tool.uv.sources] = { workspace = true }`. Only the cupid and mimicgen extras pull these in.
- Final lockfile: 5943 lines, covers all four extras + the workspace members. Analysis goldens still pass in 1.5s.

**Decision / action:**
- Phase 1 sim-extras commit lands as-is. `uv sync --extra cupid` is expected to fail on macOS / Python 3.12 — Linux validation on a future commit.
- Open follow-up: figure out the right build-isolation overrides or vendored-package strategy for the legacy deps. May involve `[tool.uv.extra-build-dependencies]` entries or pinning `setuptools<66` for the affected packages, or pinning Python 3.10 for the `cupid` extra.

**Plan impact:** Recorded in plan §6 risk table as a "Mujoco / robosuite / mimicgen don't install cleanly under uv" mitigation that did partially bite. Doesn't change phase shape.

**Context:** First Phase-1 milestone — establish a uv-managed `analysis` env and prove the goldens reproduce inside it.

**Finding:**
- Rewrote `pyproject.toml` to declare a core dependency set plus an `analysis` optional-deps group. Bumped `requires-python` from 3.9 → 3.10 (matches all four target envs).
- uv resolved 177 packages cleanly on macOS / Python 3.12; sync took ~2 min cold. Notable resolutions: numpy 2.0.2, torch 2.12.0, scikit-learn latest. No special handling needed.
- Added `scripts/uv_env.sh` as the `conda run -n` replacement. Per-env venvs at `.venvs/<env>/` selected via `UV_PROJECT_ENVIRONMENT`. The wrapper auto-syncs on first use.
- All four golden anchors pass under `./scripts/uv_env.sh analysis pytest tests/golden/` in 0.9s (warm) / 29s (cold venv).
- `.gitignore` extended to ignore `.venvs/`.
- uv.lock committed (~4300 lines) per uv best practice for reproducible installs.

**Decision / action:** Sim extras (`cupid`, `mimicgen`, `robocasa`) will be added in follow-up commits. Each can be verified independently and probably can't be fully tested on macOS (CUDA-pinned torch wheels, mujoco compat). Will validate analysis-extra parity on a Linux box if/when one is in scope.

**Plan impact:** none. Phase 1 progressing per plan.

**Context:** Follow-up to the previous entry. Added the fourth golden anchor.

**Finding:** The HDF5 schema expected by `MimicGenSeedTrajectory.from_rollout_hdf5()` is simpler than first read — only `data/<demo>/states`, `data/<demo>/actions`, and the demo-level `success` attribute are required for the heuristics to run end-to-end. `env_args` is loaded from `data.attrs` and stored as JSON. The heuristics' `info` dicts contain `numpy.float32`/`numpy.int64` scalars that don't round-trip through json.dumps without canonicalization — added `_canonicalize` to handle this.

The MimicGen golden captures three semantically distinct outcomes that any refactor must preserve:
  - `behavior_graph_path` picks all three seeds from the top-probability path (cluster_seq [0,1,2,3], prob 0.5).
  - `diversity` takes one seed per path in probability order: rollouts {6, 7, 3} for the three paths.
  - `random` samples without replacement: rollouts {7, 0, 5}.

**Decision / action:** Phase 0 complete. Goldens are 12 files, ~10 KB total. Replay runs in <1s under pytest. `tools/refactor/snapshot.py` is now the canonical regression check; every phase boundary must pass `pytest tests/golden/`.

**Plan impact:** none.

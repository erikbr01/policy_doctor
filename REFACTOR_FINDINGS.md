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

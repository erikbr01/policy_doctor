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

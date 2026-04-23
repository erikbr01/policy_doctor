# Plan: Runtime Monitor Visualizer (Streamlit Tab)

## Goal

Add a **"Runtime Monitor"** tab to the Policy Doctor Streamlit app that lets a user:

1. Load per-timestep monitor assignment data (CSV from `monitor_online.py` / `monitor_offline.py`, or uploaded)
2. Browse rollout episodes **or** training demonstrations with their per-timestep behavior graph node assignments shown as a color-coded timeline
3. Overlay **intervention markers** on the timeline (timesteps where `NodeValueThresholdRule` fires)
4. Play through rollout frames (if image data is available) with the node assignment as an overlay
5. Explore the top-K training demonstrations most influential at each intervention point, with their own frame players

---

## Architecture Decisions

### Data Sources

The visualizer is designed to work with different levels of data richness:

| Level | What's needed | What's shown |
|-------|--------------|--------------|
| **Minimal** | `monitor_assignments.csv` + behavior graph from session state | Timeline, intervention simulation |
| **With influence** | + `influence_scores.npz` | Linked demo ranking at interventions |
| **With images** | + `data` object (`InfluenceDataContainer`) already loaded in app | Frame players for rollouts + demos |

The tab degrades gracefully when the richer data isn't available.

### Monitor Assignments CSV Format

The existing `monitor_assignments.csv` format (from `monitor_online.py`) is:

```
episode, timestep, cluster_id, node_id, node_name, distance, total_ms
```

The `episode` column is already written by `monitor_online.py`. The offline script currently only writes `timestep, cluster_id, node_id, node_name, distance, total_ms`. We need to extend the offline script to write `episode` as well.

### Interventions

Interventions are **computed in the UI** from the CSV + behavior graph values — no need to re-run the classifier:

1. Load the behavior graph's state values `V(s)` from session state (`bg_values` → `{node_id: float}`)
2. User sets a threshold slider
3. For each timestep: look up `V(node_id)` from the loaded assignments; mark as intervention if `V < threshold`

This means interventions work without any changes to the monitoring scripts, as long as the behavior graph has been built with `compute_values()` in the Behavior Graph tab.

### Influence Score Linking

For "which demos are linked to this intervention?":

1. **Via loaded `data` object**: If the full influence matrix is loaded (`InfluenceDataContainer.influence_matrix`), we slice row(s) corresponding to the intervention timestep(s) and rank demo indices by mean score. The existing `rank_demo_indices_by_slice_influence` computation from `policy_doctor.computations.slice_influence` handles this.

2. **Via `influence_scores.npz`** (optional file): A `(N_timesteps, N_demo)` float32 array saved by an extended monitor script (see Phase 4). This avoids needing the full TRAK matrix.

3. **Fallback**: If neither is available, show a "no influence data" notice and still allow browsing the demo by index.

### Frame Players

Rollout and demo frames are accessed via the existing `InfluenceDataContainer` API:

```python
data.get_rollout_frame(abs_sample_idx)  # → np.ndarray (H, W, 3) or None
data.get_demo_frame(abs_sample_idx)     # → np.ndarray (H, W, 3) or None
```

For demos, the absolute index is obtained by:
```python
ep = data.demo_episodes[demo_ep_idx]
abs_idx = ep.sample_start_idx + within_ep_timestep
```

Frame players are implemented as Streamlit `st.number_input` / `st.slider` scrubbers next to `st.image()` calls. No video rendering — just single-frame display with prev/next buttons.

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Runtime Monitor                                              │
│                                                               │
│  ┌── 1. Load Monitor Data ─────────────────────────────────┐ │
│  │  [Upload CSV] or [Path input]  [Load]                   │ │
│  │  ✓ 50 episodes, 3820 timesteps, 12 nodes                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌── 2. Intervention Settings ─────────────────────────────┐ │
│  │  Source: ○ Behavior graph V(s)  ○ Column from CSV       │ │
│  │  Threshold: ──●── 0.0   [Recompute]                     │ │
│  │  ⚡ 142 intervention timesteps across 50 episodes        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌── 3. Episode Timeline ──────────────────────────────────┐ │
│  │  Mode: ○ Rollout  ○ Demo                                │ │
│  │  Episode: [← 0 →]    Success: ✓                        │ │
│  │                                                         │ │
│  │  [Color-coded timeline bar: nodes over time]           │ │
│  │  [⚡ ⚡    ⚡ markers at intervention timesteps]        │ │
│  │                                                         │ │
│  │  Timestep: [── slider ──] 42                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌── 4. Frame Player (Rollout) ────────────────────────────┐ │
│  │  t=42  Node: Behavior 3 (Recovery)  V=−0.12  ⚡ INTERV │ │
│  │  [Frame image]                [action plot / metadata]  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌── 5. Linked Training Demos (at t=42) ───────────────────┐ │
│  │  Influence window: ±3 steps  [Update]                   │ │
│  │                                                         │ │
│  │  Demo #127 (ep 12, t=34)  score=0.84  ✓                │ │
│  │  [Frame image]            [prev] [42] [next]            │ │
│  │                                                         │ │
│  │  Demo #203 (ep 19, t=58)  score=0.71  ✓                │ │
│  │  [Frame image]            [prev] [58] [next]            │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1 — Core tab + CSV loading + timeline

**Files to create / modify:**

1. **`policy_doctor/streamlit_app/tabs/runtime_monitor.py`** (new)
   - `render_tab(config, data, task_config_stem)` entry point
   - Section 1: CSV upload / path input, parse into `pd.DataFrame`
   - Section 2: intervention threshold (reads `bg_values` from session state, falls back to CSV column `intervention_triggered` if present)
   - Section 3: episode selector + timeline plot (calls `create_monitoring_timeline`)

2. **`policy_doctor/plotting/plotly/monitoring.py`** (new, pure plotting)
   - `create_monitoring_timeline(df_episode, node_colors, intervention_mask, current_t)` → `go.Figure`
     - Horizontal segmented bar chart (one row), color-coded by node
     - Red ⚡ markers at intervention timesteps
     - Vertical line at `current_t`
   - `create_node_assignment_strip(df_episode, node_colors)` → `go.Figure`
     - Scatter strip of node assignments over time (alternative, more compact view)

3. **`policy_doctor/streamlit_app/app.py`** (modify)
   - Add `tab_runtime_monitor` to `st.tabs([...])` list
   - Import and call `runtime_monitor.render_tab()`
   - Add session-state reset keys for `rm_*` prefix

4. **`policy_doctor/plotting/__init__.py`** (modify)
   - Export `create_monitoring_timeline` and `create_node_assignment_strip`

**Session state keys** (add to `_PD_TASK_SWITCH_RESET_KEYS`):
```
"rm_df", "rm_path", "rm_threshold", "rm_episode", "rm_mode", "rm_timestep",
"rm_interventions", "rm_top_k", "rm_influence_window", "rm_demo_scrubbers"
```

### Phase 2 — Intervention computation + episode navigator

- Intervention computation from node values × threshold
- Episode success/failure annotation (read from CSV `success` column if present, or from `data.rollout_episodes`)
- Episode filter: all / successes only / failures only / episodes with interventions
- Intervention event list in sidebar: table of `(episode, timestep, node_name, V)`, clicking jumps to that episode/timestep

### Phase 3 — Rollout frame player

- If `data` is loaded and `data.get_rollout_frame()` is available:
  - Convert `(episode_idx, local_timestep)` to `abs_sample_idx` using `data.rollout_episodes`
  - `st.image(frame)` with node overlay rendered by `create_annotated_frame()` (already in `frames.py`)
  - prev/next buttons + timestep slider
  - When `data` is None or frames unavailable: show a "No frame data — load influence data via sidebar" notice

### Phase 4 — Linked training demo frame players

**Option A: Via full influence matrix** (works if `data` is loaded):
- At selected timestep: slice `data.influence_matrix[abs_rollout_idx, :]`
- Use `rank_demo_indices_by_slice_influence()` from `policy_doctor.computations.slice_influence`
- Map top-K demo sample indices → `(demo_episode_idx, within_ep_timestep)` via `data.demo_episodes`
- Show each as a frame player with `data.get_demo_frame(abs_demo_idx)`

**Option B: Via `influence_scores.npz`** (optional, lighter-weight file):
- Upload or specify path to `influence_scores.npz` (saved by extended monitor scripts)
- `(N_timesteps, N_demo_samples)` float32 matrix
- Same ranking logic, using the demo episode info from loaded data

**Frame player per demo:**
- Node assignment timeline for the demo episode (re-uses `create_monitoring_timeline` with demo data)
- Scrubber centered on the most-influential timestep
- Shows demo success status, episode index, influence score

### Phase 5 — Extended monitor script output (optional enhancement)

Update `scripts/monitor_online.py` and `scripts/monitor_offline.py`:

1. Add `--save_influence_scores PATH` flag: saves `(N_timesteps, N_demo)` float32 npz alongside the CSV
2. Add `--intervention_threshold FLOAT` flag: computes interventions in the script using `NodeValueThresholdRule` and adds `intervention_triggered`, `intervention_value`, `intervention_reason` columns to the CSV
3. The visualizer detects these extra columns and uses them directly (skipping the in-app computation)

This phase is optional for the visualizer — the tab works fully without it — but makes the visualizer usable even without the behavior graph values in session state.

---

## Key Design Rules (from CLAUDE.md)

- **No Streamlit imports in `policy_doctor/plotting/`** — all `st.*` calls live in tab files
- **No Plotly figure creation in tab/render files** — figures are returned by pure functions in `plotting/`
- Frame annotating reuses existing `create_annotated_frame()` from `plotting/plotly/frames.py`
- Node colors reuse `get_label_color()` from `plotting/common.py`
- Session state keys prefixed with `rm_` to avoid collisions

---

## File Checklist

| File | Action | Phase |
|------|--------|-------|
| `policy_doctor/streamlit_app/tabs/runtime_monitor.py` | Create | 1 |
| `policy_doctor/plotting/plotly/monitoring.py` | Create | 1 |
| `policy_doctor/streamlit_app/app.py` | Modify | 1 |
| `policy_doctor/plotting/__init__.py` | Modify | 1 |
| `policy_doctor/streamlit_app/app.py` (session keys) | Modify | 1 |
| `scripts/monitor_offline.py` | Modify (add `episode` col) | 1 |
| `scripts/monitor_online.py` | Modify (add `--save_influence_scores`) | 5 |

---

## Open Questions

1. **Demo mode**: For "demonstrations" (not rollouts), the monitor assigns each timestep of an HDF5 demo to a behavior node. Should the demo timeline be built from `classify_demo_from_hdf5` results (requires running the classifier) or from a saved CSV? For now: **CSV only** — user must have pre-generated demo assignments with `monitor_offline.py --hdf5`.

2. **Influence scores granularity**: The full influence matrix rows map to TRAK sequence windows (not raw timesteps). When mapping from `local_timestep` to `abs_sample_idx`, we need to handle the `horizon` / `pad_before` / `pad_after` windowing. `data.rollout_episodes[ep].sample_start_idx + t` gives the sequence sample index; this maps correctly if the clustering is at "timestep" level. At "rollout" level (window-mean), the mapping requires the same window-mean logic as in `TrajectoryClassifier`. **For Phase 4, assume timestep-level clustering or document the limitation.**

3. **Multi-environment rollouts**: `monitor_online.py` runs multiple parallel envs (`env_idx` column). The CSV has one row per `(episode, timestep, env_idx)`. The UI should show one timeline per `(episode, env_idx)` pair, or aggregate across env indices. **For Phase 1: treat `env_idx=0` as the primary and ignore others, with a note.**

4. **Behavior graph in session state vs standalone**: The intervention threshold relies on `bg_values` from session state. If the user opens the Runtime Monitor tab without having built a behavior graph, `bg_values` is None. Fallback: allow the user to upload a pre-saved `bg_values.json`, or skip the threshold and show assignments only.

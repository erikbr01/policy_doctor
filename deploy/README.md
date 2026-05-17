# Policy Doctor — Demo Bundle

A self-contained Streamlit app that wraps three pages — the existing
**Group A** and **Group B** user-study protocols, plus a new **Graph
demo** playground for interactive exploration of behavior graphs and
trajectory trees.

## What's in this folder

```
deploy/
├── Dockerfile               # Containerized build
├── requirements.txt         # Minimal runtime Python deps
├── .dockerignore            # Trim build context (drops .pkl, caches, etc.)
├── collect_artifacts.sh     # Re-bundles policy_doctor + clusterings + MP4s
├── policy_doctor/           # The package (populated by collect_artifacts.sh)
├── third_party/             # Pre-computed clusterings (populated)
└── data/
    └── study_mp4s/          # Rendered rollout MP4s (populated)
```

The three top-level Python folders (`policy_doctor/`, `third_party/`,
`data/`) are populated by `collect_artifacts.sh` — re-run it after any
upstream code or data change.

## One-time setup

```bash
# From the worktree root:
./deploy/collect_artifacts.sh
```

This rsyncs:

- `policy_doctor/` → `deploy/policy_doctor/`
- `third_party/influence_visualizer/configs/transport_mh_jan28/clustering/`
  (all 47+ variants) → `deploy/third_party/.../clustering/`
- `/tmp/study_mp4s/transport_mh_jan28/` → `deploy/data/study_mp4s/transport_mh_jan28/`

It excludes the ~9 MB `clustering_models.pkl` and other unused build
artifacts so the docker image stays small.

## Build the docker image

```bash
cd deploy
docker build -t policy-doctor-demo .
```

Expect a 1–2 GB image (mostly the MP4s + pip wheels).

## Run

```bash
docker run --rm -p 8501:8501 policy-doctor-demo
# then open http://localhost:8501
```

The sidebar shows four entries:

- **Home** — the landing page
- **👤 User Study A** — original Group A protocol
- **👤 User Study B** — original Group B protocol (videos + graph)
- **🌳 Graph Demo** — task / clustering / viz / color controls

## Adding more tasks

The Graph Demo auto-discovers any task that has a
`clusterings/<task>/clustering/<run>/` directory inside the bundle.
To add another task:

1. Drop the clustering directory under
   `deploy/third_party/influence_visualizer/configs/<task>/clustering/`
2. Drop its MP4s under `deploy/data/study_mp4s/<task>/` (with an
   `index.json` containing `[{ "index": int, "path": "...", "frame_count": int,
   "success": bool }]`).
3. Rebuild the image.

The user-study pages additionally need a session YAML at
`policy_doctor/configs/user_study/sessions/<task>.yaml` and a study config
at `policy_doctor/configs/user_study/<task>.yaml`. See
`transport_mh_jan28.yaml` for the format.

## Deployment options

The image listens on port `8501` (the streamlit default). It exposes a
health-check at `/_stcore/health`.

For GCP, see [`DEPLOY_PLAN.md`](DEPLOY_PLAN.md). Short version:

- **`./deploy/deploy_gcp_vm.sh`** — one-VM GCE deploy on `gcp-driven-data`
  (`e2-small`, `us-west1-a`), gated by a SHA-256-hashed shared password.
  Works with `roles/editor`. This is the default.
- **`./deploy/deploy_gcp.sh`** — Cloud Run alternative with Google
  Workspace domain restriction. Needs `run.services.setIamPolicy`
  (`roles/run.admin` or `roles/owner`) on the project.

Other hosts:

- **HuggingFace Spaces**: the `Dockerfile` works as-is; add the
  appropriate `README.md` frontmatter at the Space level.
- **Local intranet**: a single `docker run -d ...` is enough.

Behind a reverse proxy (nginx, Cloudflare), add the standard websocket
upgrade rules — Streamlit needs them for bidirectional state.

## Local dev (no docker)

If you've cloned the worktree and want to run the same app without
docker:

```bash
streamlit run policy_doctor/streamlit_app/demo_app/Home.py
```

This skips the bundle assembly — the app finds clusterings under the
worktree's `third_party/` and MP4s under `/tmp/study_mp4s/`.

# Policy Doctor — Deployment Guide

Two Streamlit apps are deployed from the same Docker image:

---

## Pre-deployment checklist (kendama)

Everything below must be done before the survey link can go out.

### Task data
- [ ] Render rollout MP4s + `index.json` for kendama → `data/study_mp4s/kendama_may20/`
- [ ] Run clustering pipeline for Group B; note the output directory path
- [ ] Film 3 representative training demos per strategy → `data/study_mp4s/kendama_may20/demo_videos/<ep_xxx.mp4>`

### Config
- [ ] Fill in strategies in `policy_doctor/configs/user_study/kendama_may20.yaml`:
  - Replace placeholder names/descriptions with real (behavior mode × initial condition) pairs
  - Add `video_paths` under each `example_demos` once demo MP4s exist
- [ ] Set `clustering_dir` in `policy_doctor/configs/user_study/sessions/kendama_may20.yaml`

### Infrastructure
- [ ] Create GCS bucket and grant IAM roles (see [GCS response storage](#gcs-response-storage))
- [ ] Set `SURVEY_GCS_BUCKET`, `SURVEY_PASSWORD_SHA256`, `APP_PASSWORD_SHA256` in `deploy/.env`
- [ ] Run `./deploy/collect_artifacts.sh` (add `kendama_may20` to `TASKS` array first)
- [ ] Run `./deploy/deploy_study_stack.sh` (or GCP VM deploy)

### Validation
- [ ] Open the survey app, complete all 5 steps, submit
- [ ] Confirm the response JSON appears in GCS (or local `study_responses/`)
- [ ] Open the Graph Demo → Survey Analytics page and verify the response shows up

---

| App | Port | Purpose |
|-----|------|---------|
| **Survey app** | 8501 | Participant-facing — randomly assigns Group A or B; this is the URL you share |
| **Graph demo** | 8502 | Researcher-facing — graph explorer + sweep analysis + survey analytics page |

Both read from (and write to) the same GCS response bucket when `SURVEY_GCS_BUCKET` is set.

---

## Contents of this folder

```
deploy/
├── Dockerfile              # Shared container image
├── docker-compose.yml      # Two-service compose config (survey :8501, demo :8502)
├── requirements.txt        # Python deps (includes google-cloud-storage)
├── .env.example            # Template for deployment environment variables
├── collect_artifacts.sh    # Bundle code + clusterings + MP4s into deploy/
├── deploy_study_stack.sh          # Build + launch both services locally via compose
├── deploy_local.sh         # Build + run single-app container (graph demo only)
├── deploy_gcp_vm.sh        # Push to GCE VM (single-app; see Two-app on GCP below)
├── deploy_gcp.sh           # Cloud Run alternative
└── sync_from_dev_and_deploy.sh
```

---

## Local development (no Docker)

Run either app directly from the worktree (no container, no artifact collection):

```bash
conda activate policy_doctor

# Survey app (randomized A/B):
streamlit run policy_doctor/streamlit_app/survey_app/Home.py

# Graph demo (with analytics page):
streamlit run policy_doctor/streamlit_app/demo_app/Home.py
```

Both apps discover clusterings under `third_party/` and MP4s under `/tmp/study_mp4s/`.
Leave `SURVEY_GCS_BUCKET` unset — responses are saved to `data/study_mp4s/<task>/study_responses/` locally.

---

## Local Docker deploy — both apps

The standard way to run both apps together for testing or local hosting:

```bash
# 1. Bundle artifacts (re-run after any code/data change)
./deploy/collect_artifacts.sh

# 2. Build + launch both services
./deploy/deploy_study_stack.sh
```

After startup:
- **Survey app** → http://localhost:8501  *(share this with participants)*
- **Graph demo** → http://localhost:8502  *(researchers)*

To stop: `docker compose -f deploy/docker-compose.yml down`

Flags for `deploy_study_stack.sh`:

| Flag | Effect |
|------|--------|
| `--no-collect` | Skip `collect_artifacts.sh` (faster rebuild after code-only changes) |
| `--no-build` | Reuse existing image |
| `--no-cache` | Force full Docker rebuild |

---

## Environment variables

Copy `deploy/.env.example` to `deploy/.env` and fill in values.
Docker Compose picks up `.env` automatically.

| Variable | Required | Description |
|----------|----------|-------------|
| `APP_PASSWORD_SHA256` | No | SHA-256 hash of the graph-demo password (omit = open) |
| `SURVEY_PASSWORD_SHA256` | No | SHA-256 hash of the survey app password (omit = open) |
| `SURVEY_GCS_BUCKET` | No | GCS bucket name for response storage (omit = local JSON files) |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to GCP service-account key JSON (GCP VMs use ADC instead) |

Generate a password hash:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```

---

## GCS response storage

When `SURVEY_GCS_BUCKET` is set, every survey submission writes a JSON blob to
`gs://<bucket>/survey_responses/group_{a|b}_{timestamp}_{participant_id}.json`.
The **Survey Analytics** page in the graph demo reads from the same bucket.

### One-time bucket setup

```bash
# Create the bucket (choose a region close to your VM)
gsutil mb -l us-west1 gs://your-bucket-name

# Grant write access to the VM's service account (or your ADC identity)
gsutil iam ch serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com:objectCreator \
    gs://your-bucket-name

# Grant read access (for the analytics page on the demo app)
gsutil iam ch serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com:objectViewer \
    gs://your-bucket-name
```

For local dev with Application Default Credentials:
```bash
gcloud auth application-default login
export SURVEY_GCS_BUCKET=your-bucket-name
```

### Viewing responses

```bash
# List all submissions
gsutil ls gs://your-bucket-name/survey_responses/

# Download all responses
gsutil -m cp 'gs://your-bucket-name/survey_responses/*.json' ./responses/
```

Or open the **Survey Analytics** page in the graph demo (port 8502) — it reads live from the bucket with a 60-second cache and a Refresh button.

---

## GCP VM deploy — both apps

The existing `deploy_gcp_vm.sh` deploys a single container. To run both apps on
the same VM, SSH in after the first deploy and swap to docker-compose:

### Initial VM creation (one-time)

```bash
# This creates the VM, builds and pushes the image, opens firewall for 8501.
./deploy/deploy_gcp_vm.sh
```

### Upgrade to two-service compose on the VM

```bash
# 1. Build and push the new image as usual
./deploy/deploy_gcp_vm.sh

# 2. SSH into the VM
gcloud compute ssh policy-doctor-demo --zone us-west1-a --project gcp-driven-data

# 3. On the VM: stop the single container, install compose, launch both services
docker rm -f policy-doctor 2>/dev/null || true

# Install docker compose plugin (COS already has docker; compose is separate)
mkdir -p ~/.docker/cli-plugins
curl -fsSL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Copy docker-compose.yml and .env to the VM
# (or write them inline with cat <<'EOF' ... EOF)
# Then launch:
SURVEY_PASSWORD_SHA256=<hash> \
APP_PASSWORD_SHA256=<hash> \
SURVEY_GCS_BUCKET=your-bucket-name \
    docker compose up -d
```

### Open the second port (8502) in the firewall

```bash
gcloud compute firewall-rules create policy-doctor-allow-8502 \
    --project=gcp-driven-data \
    --network=default \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:8502 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=policy-doctor-http \
    --quiet
```

### GCS credentials on the VM

GCE VMs use the default service account automatically (Application Default Credentials).
Ensure the VM's service account has `storage.objectCreator` + `storage.objectViewer` on
the bucket (see [GCS setup](#gcs-response-storage) above).
No credential files needed on the VM.

---

## Updating a running deployment

After code or data changes:

```bash
# Re-bundle + rebuild + restart
./deploy/deploy_study_stack.sh

# Or just rebuild without re-collecting artifacts
./deploy/deploy_study_stack.sh --no-collect
```

For the GCP VM, rebuild and push the image, then SSH in and pull:

```bash
./deploy/deploy_gcp_vm.sh          # builds + pushes new :latest
gcloud compute ssh policy-doctor-demo --zone us-west1-a --project gcp-driven-data \
    --command="docker compose pull && docker compose up -d"
```

---

## Session configuration

Each study session is a YAML file at
`policy_doctor/configs/user_study/sessions/<name>.yaml`:

```yaml
label: "Transport MH (Jan 28)"           # Display name in the session picker
mp4_dir: /tmp/study_mp4s/transport_mh_jan28
study_config: policy_doctor/configs/user_study/transport_mh_jan28.yaml
clustering_dir: third_party/influence_visualizer/configs/transport_mh_jan28/clustering/...
rollout_time_limit_seconds: 600          # Timer for the Rollout Info step (default: 600 = 10 min)
```

Changing `rollout_time_limit_seconds` only requires re-running `collect_artifacts.sh`
and rebuilding the image — no code changes.

---

## Artifact collection

`collect_artifacts.sh` copies everything the Docker build needs into `deploy/`:

```bash
./deploy/collect_artifacts.sh
```

It syncs:
- `policy_doctor/` source package
- All clustering directories for the configured tasks
- MP4 rollout videos (`data/study_mp4s/<task>/`) and their `index.json` files
- `data/clusterings/` (data-support features for the graph demo)
- `data/demo_sweep/` (sweep analysis results)

Re-run it whenever you add a new task, change clusterings, or add MP4s.
It excludes bulky artifacts (`clustering_models.pkl`, `joint_umap.joblib`) to keep
the image small.

---

## Adding a new study task

1. Add a session YAML: `policy_doctor/configs/user_study/sessions/<task>.yaml`
2. Add a study config: `policy_doctor/configs/user_study/<task>.yaml`
   (defines strategies, budget, allocation_step)
3. Place MP4s + `index.json` under `data/study_mp4s/<task>/`
4. For Group B: place clustering data under `third_party/influence_visualizer/configs/<task>/clustering/`
5. Add `<task>` to the `TASKS` array in `collect_artifacts.sh`
6. Re-collect + rebuild: `./deploy/deploy_study_stack.sh`

---

## Troubleshooting

**Container starts but survey responses aren't in GCS**
- Check `SURVEY_GCS_BUCKET` is set in the container: `docker exec <container> env | grep SURVEY`
- Verify the service account has `storage.objectCreator` on the bucket
- Check app logs: `docker compose logs survey`

**Port 8502 not accessible on GCP**
- Ensure the firewall rule for 8502 exists (see [Open the second port](#open-the-second-port-8502-in-the-firewall))
- Confirm the VM has the `policy-doctor-http` network tag

**`collect_artifacts.sh` fails on a task**
- Verify `data/study_mp4s/<task>/index.json` exists
- Verify the clustering directory path matches the session YAML

**Survey app always shows Group A (or B)**
- Group assignment is random per session, stored in `st.session_state`
- Each new browser session (or incognito window) gets a fresh random assignment

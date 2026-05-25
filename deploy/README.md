# Policy Doctor — Deployment Guide

Two Streamlit apps are deployed from the same Docker image:

| App | URL | Purpose |
|-----|-----|---------|
| **Survey app** | `https://study.behavior-graphs.com` | Participant-facing — randomly assigns Group A or B |
| **Graph demo** | `https://demo.behavior-graphs.com` | Researcher-facing — graph explorer, sweep analysis, survey analytics |

Both apps run as internal-only Docker services (no host port mapping). All traffic enters
through an **nginx reverse proxy** that routes by subdomain. In production, ingress comes
from a **Cloudflare Tunnel** sidecar: an outbound-only connection to Cloudflare's edge.
No static IP, open firewall ports, or TLS certificate management is required.

```
[ Browser ]
      │  HTTPS  (Cloudflare edge terminates TLS — no cert needed on VM)
      ▼
[ cloudflared container ]  ← outbound-only tunnel, zero open inbound ports
      │  HTTP
      ▼
[ nginx proxy :80 ]  (Docker-internal only — not reachable from internet directly)
   ├─ study.behavior-graphs.com ──►  survey container :8501
   └─  demo.behavior-graphs.com ──►    demo container :8501
```

Survey responses are written as JSON files to a **persistent disk** (`/mnt/data/survey_responses`)
that survives VM deletion. See [Response storage](#response-storage).

For the ordered TODO checklist, see [DEPLOY_PLAN.md](DEPLOY_PLAN.md).

---

## Contents of this folder

```
deploy/
├── Dockerfile                    # Shared container image (survey + demo)
├── docker-compose.yml            # Base stack: proxy + survey + demo
├── docker-compose.tunnel.yml     # Overlay: adds cloudflared sidecar (production)
├── cloudflared/
│   └── config.yml.example        # Tunnel config template — fill in TUNNEL_ID
├── nginx/
│   └── templates/                # nginx virtual-host config (envsubst'd at startup)
├── requirements.txt              # Python deps baked into the image
├── .env.example                  # Template for all deployment env vars
├── collect_artifacts.sh          # Bundle code + clusterings + MP4s into deploy/
├── deploy_study_stack.sh         # Build + launch the Compose stack on the VM
├── create_vm.sh                  # One-time: provision Debian 12 VM with Docker + AR auth
├── setup_data_disk.sh            # One-time: create + mount persistent data disk
├── push_deploy.sh                # Build image → push to AR → sync deploy/ → restart stack
├── deploy_gcp_vm.sh              # Legacy: single-container COS deploy (policy-doctor-demo)
└── sync_from_dev_and_deploy.sh
```

---

## Step-by-step setup walkthrough

This section documents **every step needed to go from zero to a live production deployment**.
Follow it in order for a brand-new VM.

---

### Step 0 — Prerequisites

You need these tools on your laptop:

| Tool | Install | Purpose |
|------|---------|---------|
| `gcloud` CLI | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install) | VM management, SSH, SCP |
| `docker` + Docker Desktop | [docs.docker.com](https://docs.docker.com/get-docker/) | Build the image locally |
| `cloudflared` | `brew install cloudflared` (macOS) | Create and manage the tunnel |

Authenticate gcloud:
```bash
gcloud auth login
gcloud config set project gcp-driven-data
```

---

### Step 1 — Create the Cloudflare Tunnel

A Cloudflare Tunnel is a long-lived outbound connection from your VM to Cloudflare's
edge. Because it's outbound-only, the VM never needs to open any inbound firewall ports.
Cloudflare holds the public DNS records and forwards HTTPS traffic down the tunnel.

**This step runs once from any machine with `cloudflared` and a Cloudflare account.**

#### 1a. Log in to Cloudflare

```bash
cloudflared tunnel login
```

This opens a browser OAuth flow. Select your Cloudflare account and authorise
`behavior-graphs.com`. `cloudflared` writes a certificate to
`~/.cloudflared/cert.pem` — this is your account credential for managing tunnels.

#### 1b. Create the tunnel

```bash
cloudflared tunnel create policy-doctor
```

Output looks like:
```
Created tunnel policy-doctor with id a1b2c3d4-e5f6-...
```

Two files are written:
- `~/.cloudflared/a1b2c3d4-e5f6-....json` — the **tunnel credentials JSON**. This
  file authorises the `cloudflared` daemon to run *this specific tunnel*. Treat it
  like a password — never commit it to git. It can be revoked at any time from the
  Cloudflare dashboard.
- The tunnel UUID (`a1b2c3d4-e5f6-...`) is the stable identifier you use everywhere.

#### 1c. Create DNS records

```bash
cloudflared tunnel route dns policy-doctor study.behavior-graphs.com
cloudflared tunnel route dns policy-doctor  demo.behavior-graphs.com
```

Each command creates a CNAME record in Cloudflare DNS:
```
study.behavior-graphs.com  CNAME  a1b2c3d4-e5f6-....cfargotunnel.com
 demo.behavior-graphs.com  CNAME  a1b2c3d4-e5f6-....cfargotunnel.com
```

Cloudflare proxies both hostnames through the tunnel. TLS is terminated at the
Cloudflare edge; the VM only ever sees plain HTTP.

#### 1d. Fill in the config template

On your laptop, in the repo:
```bash
cp deploy/cloudflared/config.yml.example deploy/cloudflared/config.yml
```

Edit `deploy/cloudflared/config.yml`:
```yaml
tunnel: a1b2c3d4-e5f6-...           # ← your tunnel UUID
credentials-file: /etc/cloudflared/a1b2c3d4-e5f6-....json

ingress:
  - hostname: study.behavior-graphs.com
    service: http://proxy:80
  - hostname: demo.behavior-graphs.com
    service: http://proxy:80
  - service: http_status:404
```

`config.yml` is gitignored. Only `config.yml.example` (with the `<TUNNEL_ID>` placeholder)
is committed.

---

### Step 2 — Create the GCE VM

`create_vm.sh` provisions a **Debian 12** VM and fully configures it via a
startup script. You never need to SSH in manually for this step.

```bash
VM_NAME=user-study-test ./deploy/create_vm.sh
```

**What the script does:**

1. Writes a bash startup script to a temp file. This runs as `root` on first boot.
2. Calls `gcloud compute instances create` with:
   - `--image-family=debian-12` / `--image-project=debian-cloud`
   - `--machine-type=e2-standard-2` (2 vCPU, 8 GB RAM — tunable at top of script)
   - `--boot-disk-size=20GB` (OS + Docker images)
   - `--scopes=cloud-platform` — grants the VM's service account access to
     Artifact Registry (and other GCP APIs) via Application Default Credentials.
     No credential files needed on the VM.
   - `--metadata-from-file startup-script=<tmpfile>` — the startup script runs once.

**What the startup script does (on the VM, runs as root):**

1. Logs to `/var/log/vm-setup.log` so you can verify it completed:
   ```bash
   gcloud compute ssh user-study-test --zone us-west1-a \
       --command "tail -f /var/log/vm-setup.log"
   ```
2. Installs **Docker CE** from Docker's official apt repo (not the older `docker.io`
   distro package). Enables and starts `docker.service`.
3. Adds all human users (UID 1000–65533) to the `docker` group so they can run
   `docker` without `sudo`.
4. Downloads `docker-credential-gcr` and writes `/etc/docker/config.json` with
   the Artifact Registry credential helper so `docker pull` authenticates
   automatically using the VM's service account.
5. Installs `docker compose` (the v2 plugin).
6. Creates `~/deploy/` and `~/deploy/cloudflared/` directories.

**Verify the startup script completed:**
```bash
# Wait ~2 min after VM creation, then:
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "docker --version && docker compose version && echo OK"
```
Expected: `Docker version 27.x.x` and `Docker Compose version v2.x.x`.

**Note on firewall rules:** No HTTP/HTTPS firewall rules are needed. The Cloudflare
Tunnel uses outbound-only connections. SSH (tcp:22) is open by the GCP project's
default rules and is needed for `gcloud compute ssh`.

---

### Step 3 — Create the persistent data disk

Survey responses must survive VM deletion, OS re-imaging, or instance recreation.
A separate **persistent disk** (independent of the boot disk) is the right tool:
GCP never auto-deletes it, and it can be re-attached to any replacement VM.

```bash
VM_NAME=user-study-test ./deploy/setup_data_disk.sh
```

**What the script does, step by step:**

1. **Creates the disk:**
   ```bash
   gcloud compute disks create policy-doctor-data \
       --size=10GB --type=pd-standard --zone=us-west1-a
   ```
   10 GB is more than enough for JSON survey responses.

2. **Attaches it to the VM:**
   ```bash
   gcloud compute instances attach-disk user-study-test \
       --disk=policy-doctor-data --device-name=data-disk --zone=us-west1-a
   ```

3. **Disables auto-delete** (critical — this is what prevents accidental deletion):
   ```bash
   gcloud compute instances set-disk-auto-delete user-study-test \
       --no-auto-delete --disk=policy-doctor-data --zone=us-west1-a
   ```
   With `auto-delete` off, deleting the VM with `gcloud compute instances delete`
   will leave the disk intact. Verify at any time:
   ```bash
   gcloud compute instances describe user-study-test \
       --zone=us-west1-a --format="value(disks[].autoDelete)"
   # → false  false    (boot disk | data disk)
   ```

4. **Formats and mounts** (SSHes into the VM):
   ```bash
   # Detect the new block device (typically /dev/sdb or /dev/disk/by-id/...-data-disk)
   sudo mkfs.ext4 -F /dev/disk/by-id/google-data-disk
   sudo mkdir -p /mnt/data
   sudo mount /dev/disk/by-id/google-data-disk /mnt/data
   ```

5. **Adds an fstab entry** so the disk auto-mounts on reboot:
   ```
   /dev/disk/by-id/google-data-disk  /mnt/data  ext4  defaults,nofail  0  2
   ```
   `nofail` means the VM still boots even if the disk is temporarily detached.

6. **Creates the responses directory:**
   ```bash
   sudo mkdir -p /mnt/data/survey_responses
   sudo chmod 777 /mnt/data/survey_responses
   ```
   `chmod 777` is required because the Docker containers run as root inside the
   container but map to a non-root UID on the host. The wide permissions allow
   the containers to write files without a `chown` dance.

**Reattaching to a new VM later:**
```bash
gcloud compute instances attach-disk NEW_VM_NAME \
    --disk=policy-doctor-data --zone=us-west1-a
# Then SSH in and re-run the mount + fstab steps, or re-run setup_data_disk.sh
# with CREATE_DISK=false (edit the variable at the top of the script).
```

---

### Step 4 — Copy tunnel credentials to the VM

`push_deploy.sh` handles this automatically on every run (it copies the credentials
JSON at sync time, excluded from the main tarball). But on first setup you may want
to do it manually:

```bash
# Find your tunnel UUID:
cloudflared tunnel list

# Copy the JSON:
gcloud compute scp ~/.cloudflared/<TUNNEL_ID>.json \
    user-study-test:~/deploy/cloudflared/ \
    --zone us-west1-a --project gcp-driven-data
```

The file must be readable by the `cloudflared` container (it mounts
`~/deploy/cloudflared/` as `/etc/cloudflared/:ro`).

Also push your filled-in `config.yml` at this point if you haven't already:
```bash
gcloud compute scp deploy/cloudflared/config.yml \
    user-study-test:~/deploy/cloudflared/config.yml \
    --zone us-west1-a --project gcp-driven-data
```

---

### Step 5 — Create `.env` on the VM

The `.env` file is never committed to git and is never overwritten by deploys.
Create it once directly on the VM:

```bash
gcloud compute ssh user-study-test --zone us-west1-a --project gcp-driven-data \
    --command "printf 'STUDY_DOMAIN=study.behavior-graphs.com\nDEMO_DOMAIN=demo.behavior-graphs.com\nSURVEY_RESPONSES_DIR=/mnt/data/survey_responses\n' > ~/deploy/.env"
```

**All available variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STUDY_DOMAIN` | No | `study.localhost` | Hostname routed to the survey container |
| `DEMO_DOMAIN` | No | `demo.localhost` | Hostname routed to the demo container |
| `SURVEY_RESPONSES_DIR` | No | `./survey_responses` | Host path bind-mounted into both containers |
| `APP_PASSWORD_SHA256` | No | *(open)* | SHA-256 hash of the graph-demo password |
| `SURVEY_PASSWORD_SHA256` | No | *(open)* | SHA-256 hash of the survey app password |
| `HTTP_PORT` | No | `80` | Host port the proxy binds (local testing only) |

Docker Compose reads `.env` automatically from the same directory as
`docker-compose.yml`. The file is loaded before container start, so a restart
is needed to pick up changes.

---

### Step 6 — Set the demo password

The graph-demo page is password-gated. `push_deploy.sh --set-password` hashes
the plaintext password with SHA-256, SSHes in, and does a `sed -i` on just
the `APP_PASSWORD_SHA256` line in `.env` — no full redeploy needed:

```bash
VM_NAME=user-study-test ./deploy/push_deploy.sh --set-password=bgproject26
```

**What it does:**
1. Runs `printf '%s' "bgproject26" | sha256sum` locally to get the hash.
2. SSHes into the VM.
3. If `APP_PASSWORD_SHA256=...` exists in `~/deploy/.env`, replaces that line in-place
   with `sed -i`. Otherwise appends it.
4. Restarts the Compose stack (`deploy_study_stack.sh --tunnel --no-collect --no-build`).

To generate a hash manually:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'bgproject26').hexdigest())"
# or:
printf '%s' 'bgproject26' | sha256sum
```

---

### Step 7 — First full deploy

```bash
VM_NAME=user-study-test ./deploy/push_deploy.sh
```

`push_deploy.sh` runs five stages in sequence:

#### Stage 1: Collect artifacts (`collect_artifacts.sh`)

Copies everything the Docker build needs from the worktree into `deploy/`:
- `policy_doctor/` Python source package
- Clustering directories from `third_party/influence_visualizer/configs/<task>/`
- MP4 rollout videos from `data/study_mp4s/<task>/` (with `index.json`)
- `data/clusterings/` (data-support features)
- `data/demo_sweep/` (sweep analysis results)
- Kendama rollout MP4s (runs `cluster_kendama_rollouts.py` if not yet run)

Bulky artifacts that the Streamlit app never loads (`clustering_models.pkl`,
`embedding_models.pkl`, `joint_umap.joblib`) are excluded to keep the image small.

Skip with `--no-collect` if you've already run this and only changed code.

#### Stage 2: Build image (`docker build --platform linux/amd64`)

```bash
docker build --platform linux/amd64 \
    -t us-west1-docker.pkg.dev/gcp-driven-data/policy-doctor/demo:latest \
    -t us-west1-docker.pkg.dev/gcp-driven-data/policy-doctor/demo:<git-sha> \
    deploy/
```

`--platform linux/amd64` is mandatory when building on an Apple Silicon Mac —
GCE instances are always x86-64. Two tags are pushed: `latest` (for deploy) and
the short git SHA (for rollback / audit trail).

The Dockerfile:
- Base: `python:3.10-slim`
- Installs: `libgl1`, `libglib2.0-0`, `ffmpeg` (for video frame extraction)
- Installs Python deps from `requirements.txt`
- Copies `policy_doctor/`, `third_party/`, `data/` into the image
- Sets `PYTHONPATH=/app`
- Healthcheck: `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"` every 30 s (uses stdlib — no `curl` needed)

#### Stage 3: Push to Artifact Registry

```bash
gcloud auth configure-docker us-west1-docker.pkg.dev
docker push us-west1-docker.pkg.dev/.../demo:latest
docker push us-west1-docker.pkg.dev/.../demo:<git-sha>
```

If the `policy-doctor` Artifact Registry repo doesn't exist yet, the script creates
it automatically. The VM pulls from here using its service account (no credential
files needed on the VM).

#### Stage 4: Sync `deploy/` folder to the VM

```bash
tar -czf - -C deploy/ \
    --exclude='.env' \
    --exclude='cloudflared/*.json' \
    --exclude='policy_doctor' \
    --exclude='third_party' \
    --exclude='data' \
    --exclude='survey_responses' \
    . \
| gcloud compute ssh user-study-test --command "tar -xzf - -C ~/deploy"
```

A tar pipe over SSH is used instead of `gcloud compute scp` because `scp` doesn't
support `--exclude`. The tarball contains scripts, nginx templates, and compose
files — everything except:
- `.env` — never overwritten; managed manually or via `--set-password`
- `cloudflared/*.json` — credentials; copied separately below
- `policy_doctor/`, `third_party/`, `data/` — baked into the Docker image, not needed on the VM host
- `survey_responses/` — live data, never touched

After the tar sync, the tunnel credentials JSON is copied separately:
```bash
gcloud compute ssh user-study-test --command \
    "cat > ~/deploy/cloudflared/<TUNNEL_ID>.json" < ~/.cloudflared/<TUNNEL_ID>.json
```

#### Stage 5: Pull image + restart stack on the VM

SSHes into the VM and runs:
```bash
# Configure the Artifact Registry credential helper for this user
docker-credential-gcr configure-docker \
    --registries=us-west1-docker.pkg.dev \
    --include-artifact-registry

# Pull new image and tag it with the local name Compose expects
docker pull us-west1-docker.pkg.dev/.../demo:latest
docker tag  us-west1-docker.pkg.dev/.../demo:latest policy-doctor-demo

# Restart the full Compose stack with the tunnel sidecar
cd ~/deploy
bash deploy_study_stack.sh --tunnel --no-collect --no-build
```

`deploy_study_stack.sh --tunnel` uses two compose files:
```bash
docker compose \
    -f docker-compose.yml \
    -f docker-compose.tunnel.yml \
    up -d --remove-orphans
```

`docker-compose.tunnel.yml` adds a `cloudflared` service:
```yaml
services:
  tunnel:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel --config /etc/cloudflared/config.yml run
    volumes:
      - ./cloudflared:/etc/cloudflared:ro
    depends_on:
      - proxy
```

On startup, `cloudflared` reads `config.yml`, opens a TLS connection to
`<TUNNEL_ID>.cfargotunnel.com`, and registers the tunnel with Cloudflare's edge.
From that point all traffic for `study.behavior-graphs.com` and
`demo.behavior-graphs.com` is forwarded by Cloudflare to `http://proxy:80` inside
the Docker network.

---

### Step 8 — Verify the deployment

```bash
# Check all four containers are running (tunnel, proxy, survey, demo):
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "docker ps --format 'table {{.Names}}\t{{.Status}}'"

# Check tunnel is connected (look for "Registered tunnel connection" log lines):
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "cd ~/deploy && docker compose logs tunnel --tail=20"

# Check the apps are responding (from the VM):
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "curl -fsS http://localhost:80/ -H 'Host: demo.behavior-graphs.com' | head -c 200"

# From your browser: https://demo.behavior-graphs.com
# Should prompt for a password.
```

---

## Security posture

| Surface | Status | Notes |
|---------|--------|-------|
| Inbound ports | **None open** | No GCP firewall rules open tcp:80 or tcp:443. The tunnel uses outbound-only connections. |
| SSH (tcp:22) | Open to 0.0.0.0/0 | GCP project default. Required for `gcloud compute ssh`. Protected by OS Login / project SSH keys. |
| Streamlit port 8501 | Not exposed | Containers use `expose` (Docker-internal), not `ports`. No host binding. |
| `policy-doctor-http` tag | Removed | This tag opened tcp:8501 via `policy-doctor-allow-8501` firewall rule. Removed from `user-study-test` — only `policy-doctor-demo` (old single-container instance) keeps it. |
| Cloudflare credentials JSON | Gitignored | Lives in `deploy/cloudflared/` on the VM. Not world-readable. Revocable from the Cloudflare dashboard. |
| `.env` (passwords + domains) | Gitignored | Never committed, never overwritten by deploys. `chmod 600` recommended. |
| Survey responses on disk | Persistent disk | Encrypted at rest by default (GCP). `chmod 777` on the directory is required for Docker-root container writes. |
| Pre-commit secret scanning | `gitleaks` | `.pre-commit-config.yaml` runs `gitleaks protect --staged` on every commit. Allowlist in `.gitleaks.toml`. |

---

## Pushing updates and making changes

### Quick-reference by change type

| What changed | Command |
|---|---|
| Python code (`policy_doctor/`) | `./deploy/push_deploy.sh --no-collect` |
| Data (new MP4s, clusterings) | `./deploy/push_deploy.sh` (full) |
| Session YAML (e.g. time limit) | `./deploy/push_deploy.sh` (full, YAML is baked into image) |
| nginx config | `./deploy/push_deploy.sh --no-collect --no-build` |
| Compose files / scripts | `./deploy/push_deploy.sh --no-collect --no-build --no-push` |
| Password only | `./deploy/push_deploy.sh --set-password=newpassword` |
| Env var (other) | SSH in + edit `.env` + `docker compose restart` |
| Just restart stuck container | SSH in + `docker compose restart survey` |
| Rollback to previous image | SSH in (see below) |

---

### Code change (Python / Streamlit)

When you edit files under `policy_doctor/` or `third_party/`:

```bash
# Artifacts (clusterings, MP4s) are already bundled — skip collect.
# This rebuilds the image, pushes it, and restarts the stack.
VM_NAME=user-study-test ./deploy/push_deploy.sh --no-collect
```

**What happens:** `push_deploy.sh` runs `docker build --platform linux/amd64`
(takes 3–10 min depending on layer cache hits), pushes to Artifact Registry,
tar-syncs the deploy folder, SSHes in, `docker pull`s the new image, tags it
`policy-doctor-demo`, and restarts the stack.

Use `--no-cache` if you suspect a stale pip install (forces full layer rebuild):
```bash
VM_NAME=user-study-test ./deploy/push_deploy.sh --no-collect --no-cache
```

---

### Data change (new MP4s, clustering updates)

When you add a new task, update rollout videos, or change clustering outputs:

```bash
# Full pipeline: collect artifacts → build → push → deploy.
VM_NAME=user-study-test ./deploy/push_deploy.sh
```

`collect_artifacts.sh` re-rsyncs all data directories into `deploy/` before the
build. If only one task changed you can still run the full collect — rsync skips
unchanged files and the Docker layer cache means unchanged layers are not re-pushed.

---

### Deploy-config change (compose files, nginx, shell scripts)

When you edit `docker-compose.yml`, `docker-compose.tunnel.yml`, nginx templates,
or shell scripts — but the image itself hasn't changed:

```bash
# Skip build and push entirely — just sync the files and restart.
VM_NAME=user-study-test ./deploy/push_deploy.sh --no-collect --no-build --no-push
```

**What happens:** The tar pipe syncs the updated scripts/configs to the VM,
then SSHes in and runs `deploy_study_stack.sh --tunnel --no-collect --no-build`.
Compose picks up any YAML changes on `up -d`.

---

### Password update

```bash
VM_NAME=user-study-test ./deploy/push_deploy.sh --set-password=newpassword
```

Takes ~10 seconds: no image rebuild, no file sync. It hashes the plaintext
locally, SSHes in, `sed -i`s the `APP_PASSWORD_SHA256=` line in `~/deploy/.env`,
and restarts the stack. If the variable doesn't exist yet it is appended.

---

### Other environment variable change

Variables in `.env` are never touched by deploys. SSH in and edit them directly:

```bash
gcloud compute ssh user-study-test --zone us-west1-a --project gcp-driven-data

# On the VM:
nano ~/deploy/.env
# Edit STUDY_DOMAIN, DEMO_DOMAIN, SURVEY_RESPONSES_DIR, etc.

# Pick up the change:
cd ~/deploy && docker compose -f docker-compose.yml -f docker-compose.tunnel.yml \
    up -d --force-recreate
```

Only the containers that read the changed variable need recreating. If you only
changed `DEMO_DOMAIN`, recreating `proxy` is enough:
```bash
cd ~/deploy && docker compose up -d --force-recreate proxy
```

---

### Hotfix — restart a stuck container without rebuilding

```bash
gcloud compute ssh user-study-test --zone us-west1-a --project gcp-driven-data \
    --command "cd ~/deploy && docker compose restart survey demo"
```

`restart` sends `SIGTERM` then `SIGKILL` to the container process and starts a
new one from the existing image. Takes ~5 seconds. No image pull required.

To restart the full stack (including the tunnel and proxy):
```bash
gcloud compute ssh user-study-test --zone us-west1-a --project gcp-driven-data \
    --command "cd ~/deploy && bash deploy_study_stack.sh --tunnel --no-collect --no-build"
```

---

### Rollback to a previous image

Every push tags the image with both `latest` and the short git SHA. To roll back:

```bash
# Find the SHA you want to roll back to:
gcloud artifacts docker images list \
    us-west1-docker.pkg.dev/gcp-driven-data/policy-doctor/demo \
    --include-tags

# On the VM — pull the old SHA and retag it as the running image:
gcloud compute ssh user-study-test --zone us-west1-a --project gcp-driven-data \
    --command "
        docker pull us-west1-docker.pkg.dev/gcp-driven-data/policy-doctor/demo:<old-sha>
        docker tag  us-west1-docker.pkg.dev/gcp-driven-data/policy-doctor/demo:<old-sha> \
                    policy-doctor-demo
        cd ~/deploy && docker compose up -d --force-recreate survey demo
    "
```

---

### Checking what's running on the VM

```bash
# All containers + health status:
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"

# Tunnel connection logs:
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "cd ~/deploy && docker compose logs tunnel --tail=30"

# App logs (survey or demo):
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "cd ~/deploy && docker compose logs demo --tail=50"

# Which image version is running:
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "docker inspect policy-doctor-demo --format '{{.Id}} {{.RepoTags}}'"

# Disk space:
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "df -h && docker system df"
```

---

`push_deploy.sh` flags reference:

| Flag | Effect |
|------|--------|
| `--no-collect` | Skip `collect_artifacts.sh` |
| `--no-build` | Skip `docker build` (reuse existing image) |
| `--no-push` | Skip Artifact Registry push |
| `--no-cache` | Force full Docker layer rebuild |
| `--set-password=<pw>` | Update `APP_PASSWORD_SHA256` in the VM's `.env` and restart stack; no image rebuild |

---

## Local development (no Docker)

Run either app directly from the worktree:

```bash
conda activate policy_doctor

# Survey app (randomized A/B):
streamlit run policy_doctor/streamlit_app/survey_app/Home.py

# Graph demo (with analytics page):
streamlit run policy_doctor/streamlit_app/demo_app/Home.py
```

Both apps discover clusterings under `third_party/` and MP4s under `/tmp/study_mp4s/`.

---

## Local Docker deploy

```bash
# 1. Bundle artifacts
./deploy/collect_artifacts.sh

# 2. Build + launch proxy + both services
./deploy/deploy_study_stack.sh
```

Proxy listens on port 80 and routes by subdomain:
- **Survey** → `http://study.localhost`
- **Demo** → `http://demo.localhost`

`*.localhost` resolves to 127.0.0.1 on most systems. If not, add to `/etc/hosts`:
```
127.0.0.1  study.localhost demo.localhost
```
Or use `lvh.me` (public DNS → 127.0.0.1): set `STUDY_DOMAIN=study.lvh.me` etc. in `.env`.

`deploy_study_stack.sh` flags:

| Flag | Effect |
|------|--------|
| `--no-collect` | Skip artifact bundling |
| `--no-build` | Reuse existing image |
| `--no-cache` | Force full Docker rebuild |
| `--tunnel` | Add `docker-compose.tunnel.yml` (needs `cloudflared/config.yml`) |

---

## Response storage

Survey responses are written as JSON files to the bind-mounted `SURVEY_RESPONSES_DIR`.

- **Locally:** `./deploy/survey_responses/` (gitignored)
- **Production:** `/mnt/data/survey_responses` on the persistent disk

Retrieve responses:
```bash
# From your laptop:
gcloud compute scp --recurse \
    user-study-test:/mnt/data/survey_responses ./local_responses \
    --zone us-west1-a --project gcp-driven-data
```

Each file is named `group_{a|b}_{timestamp}_{participant_id}.json`.

---

## Persistent disk reference

The `policy-doctor-data` disk is attached to `user-study-test` with `auto-delete=false`.

```bash
# Verify auto-delete is still off:
gcloud compute instances describe user-study-test \
    --zone=us-west1-a --format="value(disks[].autoDelete)"
# → false  false

# Re-attach to a new VM after VM recreation:
gcloud compute instances attach-disk NEW_VM_NAME \
    --disk=policy-doctor-data --zone=us-west1-a
# Then SSH in and mount:
gcloud compute ssh NEW_VM_NAME --zone=us-west1-a \
    --command "sudo mount /dev/disk/by-id/google-data-disk /mnt/data"
# (fstab entry from setup_data_disk.sh handles this automatically on subsequent reboots)
```

---

## Session configuration

Each study session is a YAML file at
`policy_doctor/configs/user_study/sessions/<name>.yaml`:

```yaml
label: "Transport MH (Jan 28)"
mp4_dir: /tmp/study_mp4s/transport_mh_jan28
study_config: policy_doctor/configs/user_study/transport_mh_jan28.yaml
clustering_dir: third_party/influence_visualizer/configs/transport_mh_jan28/clustering/...
rollout_time_limit_seconds: 600
```

Changing `rollout_time_limit_seconds` requires re-collecting and rebuilding the image;
no code changes.

---

## Adding a new study task

1. Add a session YAML: `policy_doctor/configs/user_study/sessions/<task>.yaml`
2. Add a study config: `policy_doctor/configs/user_study/<task>.yaml`
3. Place MP4s + `index.json` under `data/study_mp4s/<task>/`
4. For Group B: place clustering data under `third_party/influence_visualizer/configs/<task>/clustering/`
5. Add `<task>` to `TASKS` in `collect_artifacts.sh`
6. Re-collect + rebuild: `./deploy/push_deploy.sh`

---

## Troubleshooting

**Containers show `(unhealthy)` in `docker ps`**
- The healthcheck polls `http://localhost:8501/_stcore/health` every 30 s with a
  20 s grace period on startup. It uses `python -c urllib...` (no `curl` needed).
- If unhealthy after >2 min, check logs: `docker compose logs demo`

**`DNS_PROBE_FINISHED_NXDOMAIN` in the browser (local)**
- Add `127.0.0.1 study.localhost demo.localhost` to `/etc/hosts`, or
  set `STUDY_DOMAIN=study.lvh.me DEMO_DOMAIN=demo.lvh.me` in `.env`

**Subdomains not reachable in production**
- Check tunnel container: `docker compose logs tunnel --tail=30`
- `Registered tunnel connection` = tunnel is up
- `failed to connect` = credentials JSON or `config.yml` missing/wrong tunnel ID
- Verify DNS: `cloudflared tunnel info policy-doctor` — should show two CNAME routes

**`502 Bad Gateway` from nginx**
- Survey/demo container crashed: `docker compose logs survey` or `docker compose logs demo`
- Quick restart without rebuild: `docker compose restart survey demo`

**`collect_artifacts.sh` fails on a task**
- Verify `data/study_mp4s/<task>/index.json` exists
- Verify the clustering directory path in the session YAML is correct

**`push_deploy.sh` fails at `collect_artifacts.sh` but `deploy/data/` already exists**
- Use `--no-collect` to skip re-bundling: `./deploy/push_deploy.sh --no-collect`

**Survey app always shows Group A (or B)**
- Group assignment is random per browser session (`st.session_state`)
- Each new incognito window gets a fresh random assignment

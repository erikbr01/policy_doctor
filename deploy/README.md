# Policy Doctor — Deployment Guide

Two Streamlit apps are deployed from the same Docker image:

For the ordered checklist of remaining work to ship to participants, see
[DEPLOY_PLAN.md](DEPLOY_PLAN.md). This file is the **how-to reference**.

---

| App | URL (default) | Purpose |
|-----|---------------|---------|
| **Survey app** | `http://study.localhost` | Participant-facing — randomly assigns Group A or B; this is the URL you share |
| **Graph demo** | `http://demo.localhost` | Researcher-facing — graph explorer + sweep analysis + survey analytics page |

Both apps run as internal-only Docker services (no host port mapping). All
traffic enters through an **nginx reverse proxy** which routes by subdomain
to the right container.

For production, ingress goes through a **Cloudflare Tunnel**
(`docker-compose.tunnel.yml`): a `cloudflared` sidecar container opens an
outbound-only connection to Cloudflare's edge, so no static IP, open firewall
ports, or TLS certificate management is required.

Survey responses are written to a Docker volume on the VM and can be
retrieved at any time with `docker cp` (see [Response storage](#response-storage)).

```
[ Browser ]
      │  HTTPS (Cloudflare edge terminates TLS)
      ▼
[ cloudflared sidecar ]  ← outbound tunnel, no open ports on host
      │  HTTP
      ▼
[ nginx proxy :80 ]
   ├─ study.behavior-graphs.com ─►  survey  container :8501  (internal-only)
   └─  demo.behavior-graphs.com ─►  demo    container :8501  (internal-only)
```

---

## Contents of this folder

```
deploy/
├── Dockerfile                    # Shared container image (survey + demo)
├── docker-compose.yml            # Base stack: proxy + survey + demo (HTTP only)
├── docker-compose.tunnel.yml     # Overlay: Cloudflare Tunnel sidecar (production)
├── cloudflared/
│   └── config.yml.example        # Tunnel config template — copy to config.yml
├── nginx/
│   └── templates/                # nginx config — envsubst'd at container start
├── requirements.txt              # Python deps (includes google-cloud-storage)
├── .env.example                  # Template for deployment environment variables
├── collect_artifacts.sh          # Bundle code + clusterings + MP4s into deploy/
├── deploy_study_stack.sh         # Build + launch all services via compose (on VM)
├── create_vm.sh                  # One-time: create Debian VM with Docker CE + AR auth
├── setup_data_disk.sh            # One-time: create + mount persistent data disk on VM
├── push_deploy.sh                # Build + push image + sync deploy/ + restart stack
├── deploy_gcp_vm.sh              # Legacy: single-container COS deploy (policy-doctor-demo)
├── deploy_gcp.sh                 # Cloud Run alternative
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

## Local Docker deploy — proxy + both apps

```bash
# 1. Bundle artifacts (re-run after any code/data change)
./deploy/collect_artifacts.sh

# 2. Build + launch the proxy + both services
./deploy/deploy_study_stack.sh
```

After startup the proxy listens on port 80 and routes by subdomain:
- **Survey app** → http://study.localhost  *(share this with participants)*
- **Graph demo** → http://demo.localhost  *(researchers)*

### Local DNS for testing without a real domain

`*.localhost` resolves to 127.0.0.1 on most modern systems out of the box. If
your browser shows `DNS_PROBE_FINISHED_NXDOMAIN`, pick one of:

1. **Edit `/etc/hosts`** (one line, persistent):
   ```
   127.0.0.1  study.localhost demo.localhost
   ```
2. **Use a wildcard DNS service** (no host-file edits, needs internet):
   ```bash
   # deploy/.env
   STUDY_DOMAIN=study.lvh.me
   DEMO_DOMAIN=demo.lvh.me
   ```
   `lvh.me` is a public DNS entry that always points at 127.0.0.1.

If host port 80 is already taken, override `HTTP_PORT=8080` in `.env` — the
launcher prints the right URL with the port suffix.

### Running on a remote machine — SSH tunnel access

When the proxy is bound to port 80 on a remote host that doesn't expose
that port publicly, forward it over SSH and hit it from your laptop's
browser. The browser still sends the right Host header, so nginx routes
the request normally.

```bash
# On your laptop. (Use any unused local port; 8080 here.)
ssh -L 8080:localhost:80 user@remote-host
```

Then either:

1. **Edit your laptop's `/etc/hosts`** (works against the default
   `study.localhost` / `demo.localhost`):
   ```
   127.0.0.1  study.localhost demo.localhost
   ```
   Open `http://study.localhost:8080` and `http://demo.localhost:8080`.

2. **Use `lvh.me`** (no host-file edit, requires changing
   `STUDY_DOMAIN` / `DEMO_DOMAIN` in the remote's `.env` to `study.lvh.me`
   / `demo.lvh.me` and restarting the proxy):
   ```bash
   # On remote, after editing .env:
   cd deploy && docker compose up -d --force-recreate proxy
   ```
   Open `http://study.lvh.me:8080` and `http://demo.lvh.me:8080`.

Browsing `http://localhost:8080` directly returns nothing — the proxy
drops unknown Host headers (`return 444;`). You must hit one of the two
subdomains. Tunneling is only for the developer testing themselves; real
study URLs go through Cloudflare (see [Production deploy](#production-deploy--cloudflare-tunnel)).

To stop: `docker compose -f deploy/docker-compose.yml down`

Flags for `deploy_study_stack.sh`:

| Flag | Effect |
|------|--------|
| `--no-collect` | Skip `collect_artifacts.sh` (faster rebuild after code-only changes) |
| `--no-build` | Reuse existing image |
| `--no-cache` | Force full Docker rebuild |
| `--tunnel` | Layer in `docker-compose.tunnel.yml` (requires `cloudflared/config.yml` — see [Production deploy](#production-deploy--cloudflare-tunnel)) |

---

## Environment variables

Copy `deploy/.env.example` to `deploy/.env` and fill in values.
Docker Compose picks up `.env` automatically.

| Variable | Required | Description |
|----------|----------|-------------|
| `APP_PASSWORD_SHA256` | No | SHA-256 hash of the graph-demo password (omit = open) |
| `SURVEY_PASSWORD_SHA256` | No | SHA-256 hash of the survey app password (omit = open) |
| `STUDY_DOMAIN` | No | Hostname routed to the survey container (default `study.localhost`) |
| `DEMO_DOMAIN` | No | Hostname routed to the demo container (default `demo.localhost`) |
| `HTTP_PORT` | No | Host port the proxy binds for local HTTP testing (default `80`) |

Generate a password hash:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```

---

## Response storage

Survey responses are written as JSON files to a bind-mounted host path
(`SURVEY_RESPONSES_DIR`). The default for local testing is `./survey_responses`
(relative to the `deploy/` folder, gitignored).

In production, `SURVEY_RESPONSES_DIR` points at a **separate persistent disk**
(`/mnt/data/survey_responses`) that is never auto-deleted when the VM is
stopped, deleted, or recreated — see [Persistent disk setup](#persistent-disk-setup).

Retrieve responses at any time:

```bash
# On the VM:
ls /mnt/data/survey_responses/

# From your laptop:
gcloud compute scp --recurse \
    policy-doctor-demo:/mnt/data/survey_responses ./responses \
    --zone us-west1-a --project gcp-driven-data
```

Each file is named `group_{a|b}_{timestamp}_{participant_id}.json`.

> The **Survey Analytics** page in the demo app requires GCS and will show
> no data with local-volume storage.

---

## Persistent disk setup

Run once from your local machine **before the first deployment**:

```bash
./deploy/setup_data_disk.sh
```

This creates a `policy-doctor-data` persistent disk (10 GB, pd-standard),
attaches it to the VM with `--no-auto-delete`, formats it as ext4, mounts
it at `/mnt/data`, and adds an fstab entry so it remounts automatically on
reboot.

Then add to `~/deploy/.env` on the VM:
```
SURVEY_RESPONSES_DIR=/mnt/data/survey_responses
```

**Key properties of the disk:**
- `--no-auto-delete` means the disk is **never deleted** when the VM is
  deleted — you can reattach it to a replacement VM with:
  ```bash
  gcloud compute instances attach-disk NEW_VM_NAME \
      --disk=policy-doctor-data --zone=us-west1-a --no-auto-delete
  ```
- The disk is independent of the boot disk, so OS re-imaging or VM recreation
  doesn't touch your data.
- To verify `auto-delete` is off at any time:
  ```bash
  gcloud compute instances describe policy-doctor-demo \
      --zone=us-west1-a --format="value(disks[].autoDelete)"
  ```

---

## Production deploy — Cloudflare Tunnel

The recommended production setup uses a Cloudflare Tunnel instead of
Let's Encrypt. Advantages:

- **No static external IP required** — the tunnel uses outbound-only connections
- **No open firewall ports** — you can remove (or never add) tcp:80/443 GCP rules
- **No cert management** — TLS is terminated at Cloudflare's edge automatically
- **Free DDoS protection and Cloudflare Access** (optional participant gating)

### One-time tunnel setup (run from any machine with `cloudflared`)

Install `cloudflared` if needed:
```bash
# macOS
brew install cloudflared

# Linux (Debian/Ubuntu)
curl -L https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
  https://pkg.cloudflare.com/cloudflared any main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared
```

Create the tunnel and DNS records:
```bash
cloudflared tunnel login
cloudflared tunnel create policy-doctor
cloudflared tunnel route dns policy-doctor study.behavior-graphs.com
cloudflared tunnel route dns policy-doctor  demo.behavior-graphs.com
```

`tunnel create` prints a **tunnel UUID** and writes a credentials JSON to
`~/.cloudflared/<TUNNEL_ID>.json`. Copy both to the GCE host (see below).

### Fresh VM — end-to-end checklist

Run these once, in order, for a brand-new VM:

```bash
# 1. Create the VM (Debian 12, Docker CE, Artifact Registry auth pre-configured):
VM_NAME=user-study-test ./deploy/create_vm.sh
# Wait ~2 min for the startup script to finish, then confirm:
gcloud compute ssh user-study-test --zone us-west1-a \
    --command "docker --version && docker compose version"

# 2. Create and mount the persistent data disk (survives VM deletion):
VM_NAME=user-study-test ./deploy/setup_data_disk.sh

# 3. Copy the tunnel credentials JSON to the VM:
gcloud compute scp ~/.cloudflared/<TUNNEL_ID>.json \
    user-study-test:~/deploy/cloudflared/ \
    --zone us-west1-a --project gcp-driven-data
# Also fill in deploy/cloudflared/config.yml from config.yml.example and push it.

# 4. Create .env on the VM:
gcloud compute ssh user-study-test --zone us-west1-a --project gcp-driven-data \
    --command "printf 'STUDY_DOMAIN=study.behavior-graphs.com\nDEMO_DOMAIN=demo.behavior-graphs.com\nSURVEY_RESPONSES_DIR=/mnt/data/survey_responses\n' > ~/deploy/.env"

# 5. Set the demo password:
VM_NAME=user-study-test ./deploy/push_deploy.sh --set-password=bgproject26

# 6. First full deploy (collect + build + push + start stack):
VM_NAME=user-study-test ./deploy/push_deploy.sh
```

### Every deploy

```bash
VM_NAME=user-study-test ./deploy/push_deploy.sh
```

This collects artifacts, builds a `linux/amd64` image, pushes it to Artifact
Registry, syncs the `deploy/` folder (excluding `.env` and credentials) to the
VM, pulls the new image, and restarts the Compose stack with the tunnel sidecar.

### Updating the running deployment

```bash
# Full rebuild (collect + build + push + deploy):
VM_NAME=user-study-test ./deploy/push_deploy.sh

# Code-only change (artifacts already bundled):
VM_NAME=user-study-test ./deploy/push_deploy.sh --no-collect

# Update only the demo password (no image rebuild):
VM_NAME=user-study-test ./deploy/push_deploy.sh --set-password=newpassword
```

`push_deploy.sh` flags:

| Flag | Effect |
|------|--------|
| `--no-collect` | Skip `collect_artifacts.sh` (faster when only code changed) |
| `--no-build` | Reuse last-built image (skip `docker build`) |
| `--no-push` | Skip Artifact Registry push (re-deploy already-pushed image) |
| `--no-cache` | Force full Docker layer rebuild |
| `--set-password=<pw>` | Hash the password, update `APP_PASSWORD_SHA256` in the VM's `.env`, restart stack — no image rebuild |

The `.env` on the VM is **never overwritten** by a normal deploy (excluded from the sync tarball). Use `--set-password` or SSH in to change individual variables.

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

**`DNS_PROBE_FINISHED_NXDOMAIN` in the browser locally**
- `*.localhost` may not resolve automatically on every system
- Either add `127.0.0.1 study.localhost demo.localhost` to `/etc/hosts`,
  or set `STUDY_DOMAIN=study.lvh.me DEMO_DOMAIN=demo.lvh.me` in `deploy/.env`

**Subdomains not reachable on GCP (tunnel setup)**
- Check the tunnel container is running and connected: `docker compose logs tunnel`
- A `registered` log line means the tunnel is up; if you see `failed to connect`, verify the credentials JSON and config.yml are in `deploy/cloudflared/` and the tunnel ID matches
- Confirm Cloudflare DNS records exist: `cloudflared tunnel info policy-doctor`

**Proxy logs show errors / `502 Bad Gateway` from nginx**
- Check the proxy logs: `docker compose logs proxy`

**`502 Bad Gateway` from the proxy**
- The streamlit container probably crashed — `docker compose logs survey` / `demo`
- After a fix: `docker compose restart survey demo` (no need to rebuild)

**`collect_artifacts.sh` fails on a task**
- Verify `data/study_mp4s/<task>/index.json` exists
- Verify the clustering directory path matches the session YAML

**Survey app always shows Group A (or B)**
- Group assignment is random per session, stored in `st.session_state`
- Each new browser session (or incognito window) gets a fresh random assignment

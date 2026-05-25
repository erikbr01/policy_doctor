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
to the right container. The same setup transparently scales from local
testing (HTTP, `*.localhost`) to a public VM with real DNS + Let's Encrypt
certs — only the env vars change.

Both apps read from (and write to) the same GCS response bucket when
`SURVEY_GCS_BUCKET` is set.

```
[ Browser ]
      │  (DNS: study.* / demo.* → host IP)
      ▼
[ nginx proxy :80/:443 ]
   ├─ study.<domain> ─►  survey  container :8501  (internal-only)
   └─  demo.<domain> ─►  demo    container :8501  (internal-only)
```

---

## Contents of this folder

```
deploy/
├── Dockerfile                  # Shared container image (survey + demo)
├── docker-compose.yml          # Base stack: proxy + survey + demo (HTTP only)
├── docker-compose.tls.yml      # Override: enables HTTPS + Let's Encrypt sidecar
├── nginx/
│   ├── templates/              # nginx config (HTTP) — envsubst'd at start
│   └── templates-tls/          # nginx config (HTTPS) — used by TLS override
├── requirements.txt            # Python deps (includes google-cloud-storage)
├── .env.example                # Template for deployment environment variables
├── collect_artifacts.sh        # Bundle code + clusterings + MP4s into deploy/
├── deploy_study_stack.sh       # Build + launch all services locally via compose
├── init_letsencrypt.sh         # One-shot cert bootstrap for the TLS override
├── deploy_local.sh             # Legacy single-app container (graph demo only)
├── deploy_gcp_vm.sh            # Push to GCE VM (see "GCP VM deploy" below)
├── deploy_gcp.sh               # Cloud Run alternative
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
study URLs need DNS + the [Production deploy](#production-deploy--public-domain--https) flow below.

To stop: `docker compose -f deploy/docker-compose.yml down`

Flags for `deploy_study_stack.sh`:

| Flag | Effect |
|------|--------|
| `--no-collect` | Skip `collect_artifacts.sh` (faster rebuild after code-only changes) |
| `--no-build` | Reuse existing image |
| `--no-cache` | Force full Docker rebuild |
| `--tls` | Layer in `docker-compose.tls.yml` (requires `init_letsencrypt.sh` to have run) |

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
| `STUDY_DOMAIN` | No | Hostname routed to the survey container (default `study.localhost`) |
| `DEMO_DOMAIN` | No | Hostname routed to the demo container (default `demo.localhost`) |
| `HTTP_PORT` | No | Host port the proxy binds for HTTP (default `80`) |
| `HTTPS_PORT` | No | Host port the proxy binds for HTTPS (default `443`, TLS override only) |
| `LETSENCRYPT_EMAIL` | TLS only | Email passed to `certbot` for issuance + expiry notifications |

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

Or open the **Survey Analytics** page in the graph demo — it reads live from the bucket with a 60-second cache and a Refresh button.

---

## Production deploy — public domain + HTTPS

The production setup is the local setup plus three things:
1. Real DNS A records for both subdomains pointing at the host.
2. A reserved external IP on the VM so the records don't break on restart.
3. Let's Encrypt certs issued via the TLS compose override.

### One-time host prep (GCP VM example)

```bash
# 1. Promote the VM's ephemeral IP to static (GCP Console → VPC → IP addresses).
# 2. Open ports 80 + 443 in the firewall:
gcloud compute firewall-rules create policy-doctor-allow-https \
    --project=gcp-driven-data --network=default \
    --direction=INGRESS --action=ALLOW \
    --rules=tcp:80,tcp:443 --source-ranges=0.0.0.0/0 \
    --target-tags=policy-doctor-http --quiet

# 3. Point DNS at the static IP — at your registrar, create two A records:
#      study.<yourdomain>  →  <STATIC_IP>
#      demo.<yourdomain>   →  <STATIC_IP>
```

### Deploying to the VM

```bash
# Build + push the image (Artifact Registry).
./deploy/deploy_gcp_vm.sh

# SSH in and run the proxy + apps with TLS.
gcloud compute ssh policy-doctor-demo --zone us-west1-a --project gcp-driven-data
# On the VM, place the deploy/ folder + .env, then:
cd deploy
# .env must include:
#   STUDY_DOMAIN=study.yourdomain.com
#   DEMO_DOMAIN=demo.yourdomain.com
#   LETSENCRYPT_EMAIL=you@example.com
./init_letsencrypt.sh           # one-shot: dummy cert → ACME → real cert
./deploy_study_stack.sh --tls   # bring up proxy + apps with HTTPS
```

`init_letsencrypt.sh` will:
1. Drop dummy self-signed certs (so nginx can start with the TLS template).
2. Start the proxy + apps.
3. Delete the dummies and request real Let's Encrypt certs over HTTP-01.
4. Reload nginx so the real certs take effect.

Test against Let's Encrypt's staging server first to avoid rate limits:
```bash
STAGING=1 ./init_letsencrypt.sh
```
Once you see the staging cert served, re-run without `STAGING=1` for the real one.

After bootstrap, the **certbot sidecar** in `docker-compose.tls.yml` renews certs every 12h
and signals nginx to reload — no cron needed on the host.

### GCS credentials on the VM

GCE VMs use the default service account automatically (Application Default Credentials).
Ensure the VM's service account has `storage.objectCreator` + `storage.objectViewer` on
the bucket (see [GCS setup](#gcs-response-storage) above).
No credential files needed on the VM.

### Updating the running deployment

```bash
./deploy/deploy_gcp_vm.sh                           # build + push new :latest
gcloud compute ssh policy-doctor-demo --zone us-west1-a \
    --project gcp-driven-data --command="
        cd ~/deploy &&
        docker compose -f docker-compose.yml -f docker-compose.tls.yml pull &&
        docker compose -f docker-compose.yml -f docker-compose.tls.yml up -d
    "
```

---

## Updating a running deployment

After code or data changes:

```bash
# Re-bundle + rebuild + restart
./deploy/deploy_study_stack.sh

# Or just rebuild without re-collecting artifacts
./deploy/deploy_study_stack.sh --no-collect
```

For the GCP VM, rebuild and push the image, then SSH in and pull (see the
"Updating the running deployment" snippet under [Production deploy](#production-deploy--public-domain--https)).

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

**`DNS_PROBE_FINISHED_NXDOMAIN` in the browser locally**
- `*.localhost` may not resolve automatically on every system
- Either add `127.0.0.1 study.localhost demo.localhost` to `/etc/hosts`,
  or set `STUDY_DOMAIN=study.lvh.me DEMO_DOMAIN=demo.lvh.me` in `deploy/.env`

**Subdomains not reachable on GCP**
- Confirm DNS A records resolve to the VM's static IP (`dig study.<domain>`)
- Ensure the GCP firewall opens tcp:80 + tcp:443 with the `policy-doctor-http` tag
- Check the proxy logs: `docker compose logs proxy`

**`502 Bad Gateway` from the proxy**
- The streamlit container probably crashed — `docker compose logs survey` / `demo`
- After a fix: `docker compose restart survey demo` (no need to rebuild)

**Let's Encrypt issues `unauthorized` / `connection refused`**
- DNS hasn't propagated yet, or port 80 isn't reachable from the internet
- Test with `STAGING=1 ./init_letsencrypt.sh` first — staging has no rate limit

**`collect_artifacts.sh` fails on a task**
- Verify `data/study_mp4s/<task>/index.json` exists
- Verify the clustering directory path matches the session YAML

**Survey app always shows Group A (or B)**
- Group assignment is random per session, stored in `st.session_state`
- Each new browser session (or incognito window) gets a fresh random assignment

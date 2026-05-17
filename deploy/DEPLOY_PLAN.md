# Deploying policy-doctor-demo on GCP

Two supported targets:

- **GCE VM (`deploy_gcp_vm.sh`) — default.** Runs the docker image on a
  single `e2-small` VM in `us-west1-a`. Access is gated by a SHA-256-hashed
  shared password handled inside the Streamlit app. Works with
  `roles/editor`, costs ~$13/month.
- **Cloud Run (`deploy_gcp.sh`) — alternative.** Cheaper at idle but
  domain-restricted access (`stanford.edu`, `tri.global`, etc.) requires
  `run.services.setIamPolicy`, which `roles/editor` *does not include*. If
  the deployer doesn't have `roles/run.admin` / `roles/owner` /
  `roles/iam.securityAdmin`, this path can't grant access to anyone and
  the service returns 403 for everything. The VM path is the safe
  default for that reason.

Both scripts default to project `gcp-driven-data`, region `us-west1`
(Oregon — Tier 1 pricing, close to Bay Area participants). Override any
default with an env var (`PROJECT=...`, `REGION=...`, `ZONE=...`).

---

## One-time setup

### In the GCP Console

1. **Pick a project.** Console → project picker. Note the **Project ID**
   — `gcp-driven-data` for the current deploy.
2. **Enable billing.** Console → Billing → Link a billing account.
3. **(Recommended) Set a budget alert.** $50/month with alerts at
   50/90/100% is a reasonable first-deploy guardrail.

### In your terminal (gcloud)

```bash
gcloud auth login
gcloud config set project gcp-driven-data

# Enable the APIs we use. compute.googleapis.com is for the VM path;
# run.googleapis.com is only needed if you ever use the Cloud Run script.
gcloud services enable \
    compute.googleapis.com \
    artifactregistry.googleapis.com \
    run.googleapis.com

# Create the Artifact Registry repo (once per region).
gcloud artifacts repositories create policy-doctor \
    --repository-format=docker \
    --location=us-west1 \
    --description="policy-doctor demo images"

# Configure docker to authenticate against this region's AR.
gcloud auth configure-docker us-west1-docker.pkg.dev
```

---

## Per-release (VM path — the working default)

Single command:

```bash
./deploy/deploy_gcp_vm.sh
```

What it does, in order:

1. **Re-bundles** the worktree (`policy_doctor/`, clustering tree, MP4s)
   into `deploy/` via `collect_artifacts.sh`.
2. **Builds** the docker image as `linux/amd64` (forced — Apple Silicon
   builds get rejected by GCE/Cloud Run otherwise).
3. **Tags + pushes** the image as `:latest` and `:<git-sha>` to AR.
4. **Generates a password** (16 hex chars from `openssl rand -hex 8`) on
   first run, persists the plaintext in `deploy/.app_password` (gitignored,
   `chmod 600`). Subsequent runs reuse it so participants don't get locked
   out across redeploys. Pre-set `APP_PASSWORD=...` to pin a chosen value.
5. **Creates the firewall rule** `policy-doctor-allow-8501` allowing
   `tcp:8501` from `0.0.0.0/0` to instances tagged `policy-doctor-http`.
   Idempotent — skipped if it already exists.
6. **Creates or updates the VM** `policy-doctor-demo`
   (e2-small, `us-west1-a`, Container-Optimized OS):
   - First run: creates the VM with a startup script that pulls the image
     and runs the container with `APP_PASSWORD_SHA256` from instance
     metadata.
   - Subsequent runs: SSHs in, `docker pull`s the new image SHA, and
     `docker run`s a fresh container (the running container is replaced
     without rebooting the VM).
7. **Prints** the external IP, the URL (`http://<ip>:8501`), and the
   plaintext password (read from `deploy/.app_password`).

### Where the password lives

| Location | What's there |
|---|---|
| `deploy/.app_password` on the deployer's laptop | plaintext (chmod 600, gitignored) |
| GCE instance metadata `app-password-sha256` | SHA-256 hex digest |
| Container env `APP_PASSWORD_SHA256` | same SHA-256 hex digest |

The plaintext never leaves the deployer's machine. The Streamlit app
hashes the user-submitted password with SHA-256 and compares to the env
digest using `hmac.compare_digest` (constant-time). See
`policy_doctor/streamlit_app/demo_app/Home.py` (`_gate_on_password`).

### Changing the password

```bash
# Pin a specific one
APP_PASSWORD="new-shared-password" ./deploy/deploy_gcp_vm.sh

# Or rotate to a fresh random one
rm deploy/.app_password
./deploy/deploy_gcp_vm.sh
```

The script's idempotent update path updates both the instance metadata
and the running container's env, so the change is live as soon as the
script finishes.

---

## Operations

### URL / IP

```bash
gcloud compute instances describe policy-doctor-demo \
    --zone us-west1-a --project gcp-driven-data \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
```

### Logs

Startup-script log (image pull + first `docker run`):

```bash
gcloud compute ssh policy-doctor-demo --zone us-west1-a --project gcp-driven-data \
    --command='sudo tail -100 /var/log/policy-doctor-startup.log'
```

Container logs (Streamlit stdout/stderr):

```bash
gcloud compute ssh policy-doctor-demo --zone us-west1-a --project gcp-driven-data \
    --command='docker logs --tail=200 policy-doctor'
```

### Restart the container without redeploying

```bash
gcloud compute ssh policy-doctor-demo --zone us-west1-a --project gcp-driven-data \
    --command='docker restart policy-doctor'
```

### Tear it all down

```bash
gcloud compute instances delete policy-doctor-demo --zone us-west1-a --project gcp-driven-data --quiet
gcloud compute firewall-rules delete policy-doctor-allow-8501 --project gcp-driven-data --quiet
# (Optional) drop the AR images:
gcloud artifacts docker images list us-west1-docker.pkg.dev/gcp-driven-data/policy-doctor/demo \
    --project gcp-driven-data
```

### Custom domain / TLS

The VM publishes `http://<external-ip>:8501` — no TLS, no DNS. For a
study handout that's fine, but if you want `https://demo.example.com`:

- **Easiest, no extra infra:** put Cloudflare in front. Add an A record
  pointing your subdomain at the VM's external IP, turn Cloudflare proxy
  (orange cloud) on, and Cloudflare provides TLS to the browser and
  forwards to `http://<ip>:8501`. Bonus: Cloudflare Access (free tier)
  can also restrict by email/domain in front of the password gate.
- **In-VM TLS:** swap the startup script for one that runs Caddy as a
  sidecar and points it at the container; Caddy handles Let's Encrypt
  automatically once you have a DNS record. Not currently scripted.

---

## Per-release (Cloud Run path — when the deployer has run.admin)

Single command, identical defaults to the VM script:

```bash
./deploy/deploy_gcp.sh
```

Defaults: `PROJECT=gcp-driven-data`, `REGION=us-west1`, `AUTH_MODE=private`
(domains: `stanford.edu,tri.global`). Override any of them via env var.

Steps it runs:

1. Re-bundles via `collect_artifacts.sh`.
2. Builds the image as `linux/amd64`.
3. Tags + pushes `:latest` and `:<git-sha>` to AR.
4. `gcloud run deploy` with: `--memory=1Gi --cpu=1`,
   `--min-instances=1 --max-instances=3`, `--session-affinity` (Streamlit
   websockets must hit the same instance), `--no-cpu-throttling`,
   `--cpu-boost`, `--port=8501`.
5. If `AUTH_MODE=private`: grants Cloud Run Invoker to every domain in
   `$ALLOWED_DOMAINS`. **This step needs `run.services.setIamPolicy` —
   not in `roles/editor`. If the deployer hits PERMISSION_DENIED here,
   either ask a project owner to grant `roles/run.admin`, or switch to
   the VM path.**

### Open it up to the public instead

```bash
AUTH_MODE=public ./deploy/deploy_gcp.sh
```

(Still needs `setIamPolicy` to actually flip the binding.)

### Grant individual users

```bash
gcloud run services add-iam-policy-binding policy-doctor-demo \
    --region us-west1 \
    --member="user:participant1@example.com" \
    --role="roles/run.invoker"
```

### Roll back, logs, custom domain (Cloud Run)

```bash
# Rollback
gcloud run services update-traffic policy-doctor-demo \
    --region us-west1 \
    --to-revisions=policy-doctor-demo-<rev-id>=100

# List revisions (image SHAs come from the :<git-sha> tag)
gcloud run revisions list --service policy-doctor-demo --region us-west1

# Tail logs
gcloud run services logs tail policy-doctor-demo --region us-west1

# Map a custom domain (one-time)
gcloud run domain-mappings create \
    --service policy-doctor-demo \
    --domain demo.example.com \
    --region us-west1
```

---

## Troubleshooting

**`PERMISSION_DENIED ... run.services.setIamPolicy`** during Cloud Run
deploy → expected with `roles/editor`. Switch to the VM path or get
`roles/run.admin`.

**`Container manifest type 'application/vnd.oci.image.index.v1+json'
must support amd64/linux`** → you built on Apple Silicon and the daemon
emitted an arm64-only image. The scripts now force `--platform
linux/amd64`; if you build by hand, do the same.

**VM startup log shows `denied: Unauthenticated request` from
`*.pkg.dev`** → COS's docker only auto-authenticates against `gcr.io`,
not Artifact Registry. The startup script now runs
`docker-credential-gcr configure-docker --registries=us-west1-docker.pkg.dev`
before pulling; if you bring up a VM by hand, do that first.

**Connection to `<ip>:8501` times out** → the `policy-doctor-allow-8501`
firewall rule is missing, or the VM lost the `policy-doctor-http` tag.
Re-run `./deploy/deploy_gcp_vm.sh` — both are idempotent.

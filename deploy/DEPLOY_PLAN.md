# Deploying policy-doctor-demo to Cloud Run

Two phases: a **one-time setup** in the GCP Console + CLI, and a **per-release**
step that the `deploy_gcp.sh` script automates.

---

## One-time setup

### In the GCP Console

1. **Create or pick a project.** Console → top-bar project picker → New
   Project. Note the **Project ID** (not the display name) — used as `PROJECT`
   below.
2. **Enable billing.** Console → Billing → Link a billing account to the
   project. Cloud Run won't deploy without this.
3. **(Recommended) Set a budget alert.** Console → Billing → Budgets &
   alerts → Create budget. $50/month with alerts at 50/90/100% is sensible
   for the first deployment.

### In your terminal (gcloud)

```bash
# Log in (opens a browser tab)
gcloud auth login

# Set the project as the default
gcloud config set project YOUR_PROJECT

# Enable the two services we use
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com

# Create the Artifact Registry repo (one per region)
gcloud artifacts repositories create policy-doctor \
    --repository-format=docker \
    --location=us-central1 \
    --description="policy-doctor demo images"

# Tell docker to authenticate against this region's Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

If you want a different region (e.g. closer to your participants), substitute
it everywhere above and in the script's `REGION` env var.

---

## Per-release

The whole pipeline is a single script invocation:

```bash
PROJECT=YOUR_PROJECT ./deploy/deploy_gcp.sh
```

What it does:

1. **Re-bundles** the worktree's `policy_doctor/`, clustering tree, and MP4s
   into `deploy/` via `collect_artifacts.sh`.
2. **Builds** the docker image locally.
3. **Tags** the image with both `:latest` and `:<git-sha>` for traceability.
4. **Pushes** both tags to Artifact Registry.
5. **Deploys** to Cloud Run with the production flags:
   - `--memory=1Gi --cpu=1`
   - `--min-instances=1 --max-instances=3`
   - `--session-affinity` (Streamlit websockets must hit the same instance)
   - `--no-cpu-throttling` (responsive UI between user clicks)
   - `--cpu-boost` (snappier cold starts)
   - `--port=8501`
6. **Prints the deployed URL.**

Override any flag via environment variable:

```bash
PROJECT=my-proj REGION=us-west1 MEMORY=2Gi ./deploy/deploy_gcp.sh
```

### Restricting access to specific Google accounts

```bash
AUTH_MODE=private PROJECT=YOUR_PROJECT ./deploy/deploy_gcp.sh
```

The default `ALLOWED_DOMAINS` is `stanford.edu,tri.global` — anyone with a
signed-in Google Workspace account on either domain can access the demo
after a sign-in screen; everyone else gets 403. Override with a different
list:

```bash
AUTH_MODE=private \
    ALLOWED_DOMAINS="stanford.edu,tri.global,example.com" \
    PROJECT=YOUR_PROJECT \
    ./deploy/deploy_gcp.sh
```

To grant individual users instead of a whole domain, after the deploy:

```bash
gcloud run services add-iam-policy-binding policy-doctor-demo \
    --region us-central1 \
    --member="user:participant1@example.com" \
    --role="roles/run.invoker"
```

---

## Quick reference: rollbacks, logs, custom domain

**Roll back to a previous image:**

```bash
gcloud run services update-traffic policy-doctor-demo \
    --region us-central1 \
    --to-revisions=policy-doctor-demo-<rev-id>=100
```

`gcloud run revisions list --service policy-doctor-demo --region us-central1`
lists revisions and their image SHAs (set by the deploy script's `:<git-sha>`
tag).

**Tail logs:**

```bash
gcloud run services logs tail policy-doctor-demo --region us-central1
```

**Custom domain** (one-time, after the service exists):

```bash
gcloud run domain-mappings create \
    --service policy-doctor-demo \
    --domain demo.example.com \
    --region us-central1
```

You'll be prompted to add a DNS record at your registrar.

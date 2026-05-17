#!/usr/bin/env bash
# Deploy the policy-doctor demo to Cloud Run.
#
# Workflow (every release):
#   1. Re-run collect_artifacts.sh so deploy/policy_doctor, deploy/third_party,
#      deploy/data are fresh from the worktree.
#   2. Build the docker image locally.
#   3. Tag it for Artifact Registry with both `latest` and the current git SHA.
#   4. Push both tags.
#   5. `gcloud run deploy` with our chosen flags.
#
# Requirements (one-time, see DEPLOY_PLAN.md in this folder):
#   • `gcloud auth login` done
#   • Project's billing enabled
#   • Cloud Run + Artifact Registry APIs enabled
#   • Artifact Registry repo created (default: `policy-doctor` in $REGION)
#   • `gcloud auth configure-docker <region>-docker.pkg.dev` run once
#
# Usage:
#   ./deploy/deploy_gcp.sh                  # uses defaults from env / below
#   PROJECT=my-proj REGION=us-west1 ./deploy/deploy_gcp.sh
#   AUTH_MODE=private ./deploy/deploy_gcp.sh   # require IAM Cloud Run Invoker

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-gcp-driven-data}"
REGION="${REGION:-us-west1}"             # us-west1 (Oregon): Tier 1 pricing, close to CA
REPO="${REPO:-policy-doctor}"            # Artifact Registry repo name
SERVICE="${SERVICE:-policy-doctor-demo}" # Cloud Run service name
IMAGE_NAME="${IMAGE_NAME:-demo}"         # image inside the AR repo
MEMORY="${MEMORY:-1Gi}"
CPU="${CPU:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
AUTH_MODE="${AUTH_MODE:-private}"         # public | private
ALLOWED_DOMAINS="${ALLOWED_DOMAINS:-stanford.edu,tri.global}"  # comma- or space-separated list,
                                          # only used when AUTH_MODE=private
PORT=8501

# ── Sanity checks ────────────────────────────────────────────────────────────
command -v gcloud >/dev/null || { echo "gcloud not on PATH"; exit 1; }
command -v docker >/dev/null || { echo "docker not on PATH"; exit 1; }

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Not logged in to gcloud — run 'gcloud auth login' first." ; exit 1
fi

# ── Setup ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo dev)"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"
IMAGE_LATEST="${REGISTRY}/${IMAGE_NAME}:latest"
IMAGE_SHA="${REGISTRY}/${IMAGE_NAME}:${GIT_SHA}"

echo "================================================================"
echo "  project:     $PROJECT"
echo "  region:      $REGION"
echo "  service:     $SERVICE"
echo "  image:       $IMAGE_NAME"
echo "  git sha:     $GIT_SHA"
if [ "$AUTH_MODE" = "private" ]; then
    echo "  auth mode:   private (domains: ${ALLOWED_DOMAINS:-none})"
else
    echo "  auth mode:   public"
fi
echo "================================================================"

gcloud config set project "$PROJECT" >/dev/null

# ── 1. Bundle artifacts (clusterings + MP4s + policy_doctor) ────────────────
echo "→ Bundling artifacts via collect_artifacts.sh"
"$SCRIPT_DIR/collect_artifacts.sh"

# ── 2. Build docker image ────────────────────────────────────────────────────
# Cloud Run only runs amd64/linux. Force the build platform so Apple Silicon
# Macs don't push arm64-only images that Cloud Run rejects.
echo "→ Building image for linux/amd64 (this takes ~30s)"
docker build --platform linux/amd64 -t policy-doctor-demo "$SCRIPT_DIR"

# ── 3. Tag for Artifact Registry ─────────────────────────────────────────────
echo "→ Tagging $IMAGE_LATEST and $IMAGE_SHA"
docker tag policy-doctor-demo "$IMAGE_LATEST"
docker tag policy-doctor-demo "$IMAGE_SHA"

# ── 4. Push to Artifact Registry ─────────────────────────────────────────────
echo "→ Pushing to Artifact Registry"
docker push "$IMAGE_LATEST"
docker push "$IMAGE_SHA"

# ── 5. Deploy to Cloud Run ───────────────────────────────────────────────────
DEPLOY_ARGS=(
    "$SERVICE"
    --image "$IMAGE_SHA"
    --region "$REGION"
    --port "$PORT"
    --memory "$MEMORY"
    --cpu "$CPU"
    --min-instances "$MIN_INSTANCES"
    --max-instances "$MAX_INSTANCES"
    --session-affinity
    --no-cpu-throttling
    --cpu-boost
    --quiet
)
if [ "$AUTH_MODE" = "public" ]; then
    DEPLOY_ARGS+=(--allow-unauthenticated)
else
    DEPLOY_ARGS+=(--no-allow-unauthenticated)
fi

echo "→ gcloud run deploy ${DEPLOY_ARGS[*]}"
gcloud run deploy "${DEPLOY_ARGS[@]}"

# ── 6. Optional: grant access to one or more Google Workspace domains
#       (AUTH_MODE=private only) ─────────────────────────────────────────────
if [ "$AUTH_MODE" = "private" ] && [ -n "$ALLOWED_DOMAINS" ]; then
    # Split on commas and/or whitespace.
    for dom in ${ALLOWED_DOMAINS//,/ }; do
        [ -z "$dom" ] && continue
        echo "→ Granting Cloud Run Invoker to domain:$dom"
        gcloud run services add-iam-policy-binding "$SERVICE" \
            --region "$REGION" \
            --member="domain:${dom}" \
            --role="roles/run.invoker" \
            --quiet
    done
fi

# ── 7. Show the deployed URL ────────────────────────────────────────────────
URL=$(gcloud run services describe "$SERVICE" --region "$REGION" \
        --format='value(status.url)')
echo
echo "================================================================"
echo "  Deployed: $URL"
echo "  Image:    $IMAGE_SHA"
echo "================================================================"

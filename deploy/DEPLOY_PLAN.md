# Real-Deployment TODO

This is a checklist of what still has to happen before participants can
hit a real `https://study.behavior-graphs.com` URL. The **how-to** for each step
lives in [`README.md`](README.md) — this file is the project-management
view of remaining work, ordered by dependency.

For local / over-SSH testing, jump to
[Local Docker deploy](README.md#local-docker-deploy--proxy--both-apps) in
the README — no items below are needed for that.

Last updated: 2026-05-24.

---

## Status snapshot

| Area | State |
|------|-------|
| Architecture (nginx proxy + survey + demo + GCS responses) | ✅ Built and locally tested |
| Survey app code (5-step flow, randomized A/B, timer, analytics page) | ✅ Done |
| Local Docker stack (`./deploy/deploy_study_stack.sh`) | ✅ Working — verified end-to-end |
| Cloudflare Tunnel (`docker-compose.tunnel.yml`, `cloudflared/config.yml.example`) | ✅ Written, not yet exercised |
| Kendama task content (videos, clustering, session YAML) | ⚠️ Clustering + videos ready; strategy definitions are still placeholders |
| GCS response bucket | ❌ Not created |
| Tunnel credentials + DNS wired up | ❌ Not done |

---

## Outstanding work, in order

### 1. Finalize kendama study content

- [ ] Fill in real strategy definitions in
      `policy_doctor/configs/user_study/kendama_may22.yaml`:
      replace the 4 placeholder entries with real
      `(behavior mode × initial condition)` pairs.
- [ ] Film 3 representative training demos per strategy and place them
      at `data/study_mp4s/kendama_may22/demo_videos/<ep_xxx.mp4>`.
      File names go under each strategy's `example_demos.video_paths`.
- [ ] Re-run `./deploy/collect_artifacts.sh` so the new MP4s land in the
      image.

Blocker for: nothing technical — but the survey is misleading without
real strategies, so this gates "send link to participants."

### 2. ~~Provision response storage~~ (deferred)

Responses are written to a local Docker volume on the VM for now.
No setup needed — the volume is created automatically by Compose.
Retrieve responses at any time with `docker cp` (see README → Response storage).

### 3. Create and mount the persistent data disk

Run once from your local machine (VM must already exist):

```bash
./deploy/setup_data_disk.sh
```

- [ ] Script completes — disk is mounted at `/mnt/data` on the VM.
- [ ] Verify `auto-delete` is off:
      ```bash
      gcloud compute instances describe policy-doctor-demo \
          --zone=us-west1-a --format="value(disks[].autoDelete)"
      ```
      Should print `false` for the data disk (the boot disk will show `true`).

### 5. Create the Cloudflare Tunnel

Run once from any machine with `cloudflared` installed (not the VM):

```bash
cloudflared tunnel login
cloudflared tunnel create policy-doctor
cloudflared tunnel route dns policy-doctor study.behavior-graphs.com
cloudflared tunnel route dns policy-doctor  demo.behavior-graphs.com
```

- [ ] Tunnel created (`cloudflared tunnel list` shows `policy-doctor`).
- [ ] DNS CNAME records created (Cloudflare does this automatically via
      `tunnel route dns`; verify in the Cloudflare dashboard).

### 6. Configure the tunnel on the VM

```bash
# On the VM:
cd ~/deploy
cp cloudflared/config.yml.example cloudflared/config.yml
# Edit config.yml: fill in TUNNEL_ID and hostnames

# From your local machine — copy the credentials JSON to the VM:
gcloud compute scp ~/.cloudflared/<TUNNEL_ID>.json \
    policy-doctor-demo:~/deploy/cloudflared/ \
    --zone us-west1-a --project gcp-driven-data
```

- [ ] `cloudflared/config.yml` filled in with correct tunnel ID and domains.
- [ ] `<TUNNEL_ID>.json` credentials file present in `~/deploy/cloudflared/`
      on the VM (it is gitignored — never commit it).

### 7. Fill in the production `.env` on the VM

On the VM, in `~/deploy/.env`:

- [ ] `STUDY_DOMAIN=study.behavior-graphs.com`
- [ ] `DEMO_DOMAIN=demo.behavior-graphs.com`
- [ ] `SURVEY_RESPONSES_DIR=/mnt/data/survey_responses`
- [ ] `SURVEY_PASSWORD_SHA256=<hash>` (optional — gate the survey app)
- [ ] `APP_PASSWORD_SHA256=<hash>` (optional — gate the demo app)

Generate a hash with:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```

### 8. Deploy and smoke-test

```bash
# Build + push the image from your dev machine
./deploy/deploy_gcp_vm.sh

# On the VM: bring up the full stack with the tunnel sidecar
cd ~/deploy
./deploy_study_stack.sh --tunnel
```

- [ ] `docker compose logs tunnel` shows a `registered` / `connected` line.
- [ ] `https://study.behavior-graphs.com` loads in a browser (Cloudflare issues the cert
      automatically — no cert bootstrap step needed).
- [ ] `https://demo.behavior-graphs.com` loads.

### 9. End-to-end validation

- [ ] Open `https://study.behavior-graphs.com` in an incognito window, complete all
      5 survey steps, submit.
- [ ] Confirm the response JSON was written to the volume:
      ```bash
      docker compose exec survey ls /app/survey_responses/
      ```
- [ ] Repeat in a second incognito session to confirm Group A/B
      randomization (compare `study_group` field in the two JSON files).
- [ ] Time at least one full pass through the 10-min rollout timer
      (Group A) to confirm auto-advance fires server-side.

### 10. Pre-launch hardening

- [ ] Decide whether to leave both apps password-gated and ship the
      credentials to participants out-of-band, or open the survey app
      (`SURVEY_PASSWORD_SHA256=` blank) and keep only the demo gated.
- [ ] Set a GCP budget alert (~$30/mo for `e2-small` + minimal egress is
      typical; alerts at 50/90/100% are a reasonable guardrail).

---

## What's deliberately NOT in scope

- **Cloud Run** — `deploy_gcp.sh` targets it, but domain-restricted access
  needs `roles/run.admin` which the current deployer doesn't have. The VM +
  Cloudflare Tunnel path is the chosen approach.
- **Multi-VM / HA** — single `e2-small` is enough for the participant load
  anticipated for this study. If you need horizontal scaling, the proxy +
  apps already speak through Docker Compose service names, so swapping in a
  managed load balancer is mostly a DNS change.

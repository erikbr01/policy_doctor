# Real-Deployment TODO

This is a checklist of what still has to happen before participants can
hit a real `https://study.<domain>` URL. The **how-to** for each step
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
| TLS scaffold (`docker-compose.tls.yml`, `init_letsencrypt.sh`, certbot sidecar) | ✅ Written, not yet exercised |
| Kendama task content (videos, clustering, session YAML) | ⚠️ Clustering + videos ready; strategy definitions are still placeholders |
| GCS response bucket | ❌ Not created |
| Public DNS + static IP + cert issuance | ❌ Not done |

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

### 2. Provision GCS response storage

- [ ] `gsutil mb -l us-west1 gs://<bucket-name>` (region close to the VM).
- [ ] Grant the VM's service account `roles/storage.objectCreator` and
      `roles/storage.objectViewer` on the bucket.
- [ ] Drop `SURVEY_GCS_BUCKET=<bucket-name>` into `deploy/.env`.

Without this, responses fall back to a local Docker volume on the VM —
fine for a smoke test, but you lose them on `docker compose down -v` and
the analytics page can't read across machines.

### 3. Reserve a static IP + open the firewall

- [ ] GCP Console → VPC → External IPs → promote the VM's ephemeral IP
      to static (name it e.g. `user-study-static-ip`).
- [ ] Open ports 80 + 443 in the firewall with the
      `policy-doctor-http` network tag:
      ```bash
      gcloud compute firewall-rules create policy-doctor-allow-https \
          --project=gcp-driven-data --network=default \
          --direction=INGRESS --action=ALLOW \
          --rules=tcp:80,tcp:443 --source-ranges=0.0.0.0/0 \
          --target-tags=policy-doctor-http --quiet
      ```

### 4. Point real DNS records at the static IP

At your domain registrar (or in GCP Cloud DNS), create:

| Type | Host | Value |
|------|------|-------|
| A | `study` | `<STATIC_IP>` |
| A | `demo`  | `<STATIC_IP>` |

- [ ] Records created.
- [ ] DNS resolves: `dig study.<domain> +short` returns the static IP
      from a machine **outside** the VM. Wait if it doesn't — Let's
      Encrypt will fail otherwise.

### 5. Fill in the production `.env` on the VM

On the VM, in `~/deploy/.env`:

- [ ] `STUDY_DOMAIN=study.<domain>`
- [ ] `DEMO_DOMAIN=demo.<domain>`
- [ ] `LETSENCRYPT_EMAIL=<your-email>` (used for expiry notices)
- [ ] `SURVEY_GCS_BUCKET=<bucket-name>`
- [ ] `SURVEY_PASSWORD_SHA256=<hash>` (optional — gate the survey app)
- [ ] `APP_PASSWORD_SHA256=<hash>` (optional — gate the demo app)

Generate a hash with:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```

### 6. Issue Let's Encrypt certs

- [ ] **Dry run against staging** to catch DNS / firewall issues without
      hitting the production rate limit (5 certs / week / domain):
      ```bash
      STAGING=1 ./deploy/init_letsencrypt.sh
      ```
      `curl -kI https://study.<domain>` should return a cert from
      `Fake LE Intermediate X1` (untrusted but confirms the flow works).
- [ ] **Production issuance** once staging works:
      ```bash
      ./deploy/init_letsencrypt.sh
      ```
- [ ] Bring up the full stack with TLS:
      ```bash
      ./deploy/deploy_study_stack.sh --tls
      ```
- [ ] Verify `https://study.<domain>` and `https://demo.<domain>` both
      load and show a valid cert.

After bootstrap, the `certbot` sidecar in `docker-compose.tls.yml` renews
every 12h and signals nginx to reload — no host cron needed.

### 7. End-to-end validation

- [ ] Open `https://study.<domain>` in an incognito window, complete all
      5 survey steps, submit.
- [ ] Confirm the response JSON appears in GCS:
      `gsutil ls gs://<bucket>/survey_responses/`
- [ ] Open `https://demo.<domain>` → **Survey Analytics** page and
      verify the response shows up.
- [ ] Repeat in a second incognito session to confirm Group A/B
      randomization (compare `study_group` field in the GCS JSONs).
- [ ] Time at least one full pass through the 10-min rollout timer
      (Group A) to confirm auto-advance fires server-side.

### 8. Pre-launch hardening

- [ ] Decide whether to leave both apps password-gated and ship the
      credentials to participants out-of-band, or open the survey app
      (`SURVEY_PASSWORD_SHA256=` blank) and keep only the demo gated.
- [ ] Set a GCP budget alert (~$30/mo for `e2-small` + minimal egress is
      typical; alerts at 50/90/100% are a reasonable guardrail).
- [ ] Note the renewal date on the calendar — Let's Encrypt certs last
      90 days. The sidecar auto-renews, but it's worth having a manual
      verification 2 weeks before any high-traffic event.

---

## What's deliberately NOT in scope

- **Cloud Run** — the legacy `deploy_gcp.sh` script targets it, but
  domain-restricted access needs `roles/run.admin`, which the current
  deployer doesn't have. The VM + Cloudflare path was abandoned in
  favor of nginx + Let's Encrypt on the VM. Keep the script for
  reference but don't use it for this study.
- **Multi-VM / HA** — single `e2-small` is enough for the participant
  load anticipated for this study. If you need horizontal scaling, the
  proxy + apps already speak through Docker Compose service names, so
  swapping in a managed load balancer is mostly a DNS change.
- **Custom CDN / WAF in front** — nginx + Let's Encrypt is plenty for
  this study's threat model. If you later want Cloudflare in front,
  point the DNS at Cloudflare instead of the static IP and leave the
  origin on HTTP behind the proxy with `proxy_set_header`s already in
  place.

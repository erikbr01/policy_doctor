#!/usr/bin/env bash
# Bootstrap Let's Encrypt certs for $STUDY_DOMAIN and $DEMO_DOMAIN.
#
# Why this dance: nginx with the TLS template fails to start if no cert
# exists yet, but certbot needs nginx running to answer the ACME HTTP-01
# challenge. The well-known workaround:
#   1. Drop dummy self-signed certs at the expected paths.
#   2. Start nginx with the TLS template.
#   3. Delete the dummies, run certbot for real certs (HTTP-01 via webroot).
#   4. Reload nginx so it picks up the real certs.
#
# Prereqs:
#   - DNS A records for both subdomains already point at this host.
#   - LETSENCRYPT_EMAIL set in deploy/.env.
#   - Ports 80 and 443 open from the internet.
#
# Run once after DNS has propagated. The certbot sidecar in
# docker-compose.tls.yml handles automatic renewal after that.
#
# Re-running is safe; --force-renewal flag adds renewal on the same cert.
#   STAGING=1 ./init_letsencrypt.sh   # use Let's Encrypt staging (no rate limit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env into the shell so STUDY_DOMAIN / DEMO_DOMAIN / LETSENCRYPT_EMAIL
# are visible to certbot via docker compose run.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

STUDY_DOMAIN="${STUDY_DOMAIN:?STUDY_DOMAIN not set in .env}"
DEMO_DOMAIN="${DEMO_DOMAIN:?DEMO_DOMAIN not set in .env}"
LETSENCRYPT_EMAIL="${LETSENCRYPT_EMAIL:?LETSENCRYPT_EMAIL not set in .env}"

STAGING="${STAGING:-0}"
STAGING_FLAG=""
if [ "$STAGING" != "0" ]; then
    STAGING_FLAG="--staging"
    echo "→ Using Let's Encrypt STAGING (rate-limit-free, but cert is untrusted)"
fi

DC="docker compose -f docker-compose.yml -f docker-compose.tls.yml"

echo "================================================================"
echo "  Bootstrapping Let's Encrypt for:"
echo "    $STUDY_DOMAIN"
echo "    $DEMO_DOMAIN"
echo "  Email: $LETSENCRYPT_EMAIL"
echo "================================================================"

# ── 1. Download recommended TLS params (used by the TLS nginx template) ──
echo "→ Downloading recommended TLS params (options-ssl-nginx.conf, ssl-dhparams.pem)"
$DC run --rm --entrypoint "/bin/sh -c '\
    mkdir -p /etc/letsencrypt && \
    curl -fsSL https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf \
        > /etc/letsencrypt/options-ssl-nginx.conf && \
    openssl dhparam -out /etc/letsencrypt/ssl-dhparams.pem 2048'" certbot

# ── 2. Dummy self-signed certs so nginx can start ────────────────────────
for d in "$STUDY_DOMAIN" "$DEMO_DOMAIN"; do
    echo "→ Writing dummy cert for $d"
    $DC run --rm --entrypoint "/bin/sh -c '\
        mkdir -p /etc/letsencrypt/live/$d && \
        openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
            -keyout /etc/letsencrypt/live/$d/privkey.pem \
            -out    /etc/letsencrypt/live/$d/fullchain.pem \
            -subj \"/CN=localhost\"'" certbot
done

# ── 3. Start nginx with the TLS template so the ACME challenge resolves ──
echo "→ Starting proxy + apps so certbot can reach /.well-known/acme-challenge/"
$DC up -d proxy survey demo

# Give nginx a moment to come up.
sleep 3

# ── 4. Delete dummies + request real certs ───────────────────────────────
for d in "$STUDY_DOMAIN" "$DEMO_DOMAIN"; do
    echo "→ Deleting dummy cert for $d"
    $DC run --rm --entrypoint "/bin/sh -c '\
        rm -rf /etc/letsencrypt/live/$d /etc/letsencrypt/archive/$d /etc/letsencrypt/renewal/$d.conf'" certbot
done

echo "→ Requesting real certs from Let's Encrypt"
DOMAIN_ARGS=""
for d in "$STUDY_DOMAIN" "$DEMO_DOMAIN"; do DOMAIN_ARGS="$DOMAIN_ARGS -d $d"; done

$DC run --rm --entrypoint "certbot certonly --webroot -w /var/www/certbot \
    $STAGING_FLAG \
    --email $LETSENCRYPT_EMAIL --rsa-key-size 4096 --agree-tos --no-eff-email \
    $DOMAIN_ARGS --force-renewal" certbot

# ── 5. Reload nginx to pick up the real certs ────────────────────────────
echo "→ Reloading proxy"
$DC exec proxy nginx -s reload

echo
echo "================================================================"
echo "  Done. https://$STUDY_DOMAIN  and  https://$DEMO_DOMAIN"
echo "  Certs auto-renew via the certbot sidecar (every 12h)."
echo "================================================================"

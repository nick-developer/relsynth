#!/usr/bin/env bash
# End-to-end acceptance test on docker compose: Keycloak SSO, PostgreSQL source,
# API, two workers.  Creates throw-away secrets, runs e2e_flow.py, tears down.
#   deploy/e2e/compose-e2e.sh            # KEEP=1 leaves the stack running
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
compose="$here/../compose/docker-compose.yml"
mkdir -p "$here/../compose/secrets"
rand() { python3 -c "import secrets;print(secrets.token_urlsafe(24))"; }
export METADATA_DB_PASSWORD="$(rand)" KEYCLOAK_ADMIN_PASSWORD="$(rand)" SOURCE_DB_PASSWORD="$(rand)"
printf '%s' "$SOURCE_DB_PASSWORD" > "$here/../compose/secrets/source_db_password"
python3 -c "import base64,secrets;print(base64.b64encode(secrets.token_bytes(32)).decode())" > "$here/../compose/secrets/master_key"
chmod 0644 "$here/../compose/secrets/"*          # read by UID 10001 inside the containers (demo only)
cleanup() { [ "${KEEP:-0}" = 1 ] || docker compose -f "$compose" down -v --remove-orphans; }
trap cleanup EXIT
# NO_BUILD=1 uses an image built beforehand (tag relsynth-service:local), e.g. from the air-gapped bundle
if [ "${NO_BUILD:-0}" = 1 ]; then build=--no-build; else build=--build; fi
docker compose -f "$compose" up -d "$build" --wait --wait-timeout 600
python3 "$here/e2e_flow.py" --api http://localhost:8080 \
  --token-url http://localhost:8081/realms/relsynth/protocol/openid-connect/token \
  --source-url postgresql://reader@sources:5432/shop --secret-ref file:/run/secrets/source_db_password
# restart survival on the real stack: kill a worker mid-run, the other one finishes it
python3 "$here/restart_check.py" --compose "$compose"

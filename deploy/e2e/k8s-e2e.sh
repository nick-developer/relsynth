#!/usr/bin/env bash
# End-to-end acceptance test on Kubernetes (kind, k3d or k3s): installs the test
# dependencies and the Helm chart in namespace relsynth-e2e, then runs the same
# flow as compose-e2e.sh (SSO, PostgreSQL, run, approve, download, audit, tenant
# isolation) against NodePorts 30080 (API) and 30081 (Keycloak).
#   KUBECONFIG=... deploy/e2e/k8s-e2e.sh        # images relsynth-service:<v>, postgres:16-alpine,
#                                              # keycloak/keycloak:26.0.7 must be loaded on the nodes
# A kind cluster that maps the NodePorts to localhost: deploy/e2e/k8s/kind.yaml
# (CI: .github/workflows/service-e2e.yml). On failure the pods, events and logs
# of the namespace are printed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$here/../.."
ns=relsynth-e2e
kc() { kubectl -n "$ns" "$@"; }
diagnose() {
  rc=$?; [ "$rc" = 0 ] && return
  echo "::group::pods, events and logs of $ns (exit $rc)"
  kc get pods -o wide || true; kc get events --sort-by=.lastTimestamp | tail -60 || true
  for p in $(kc get pods -o name 2>/dev/null); do echo "--- $p"; kc logs "$p" --all-containers --tail=80 || true; done
  echo "::endgroup::"
}
trap diagnose EXIT
if command -v helm >/dev/null; then helm() { command helm "$@"; }
else helm() { docker run --rm --network host -v "$KUBECONFIG:/kube:ro" -v "$root:/src:ro" -e KUBECONFIG=/kube alpine/helm:3.16.3 "$@"; }; fi
chart=/src/deploy/helm/relsynth; command -v helm >/dev/null 2>&1 && [ "$(type -t helm)" = file ] && chart="$root/deploy/helm/relsynth"
rand() { python3 -c "import secrets;print(secrets.token_urlsafe(24))"; }
v=$(python3 -c "exec(open('$root/src/relsynth/_version.py').read());print(__version__)")
kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
mp=$(rand); sp=$(rand)
kc create secret generic e2e-passwords --from-literal=metadata="$mp" --from-literal=keycloak="$(rand)" --dry-run=client -o yaml | kc apply -f -
kc create secret generic shop-db --from-literal=password="$sp" --dry-run=client -o yaml | kc apply -f -
kc create secret generic relsynth-db --from-literal=url="postgresql+psycopg://relsynth:$mp@metadata:5432/relsynth" --dry-run=client -o yaml | kc apply -f -
kc create secret generic relsynth-master-key --from-literal=master.key="$(head -c32 /dev/urandom | base64)" --dry-run=client -o yaml | kc apply -f -
kc create configmap shop-sql --from-file=shop.sql="$here/shop.sql" --dry-run=client -o yaml | kc apply -f -
kc create configmap keycloak-realm --from-file=realm-relsynth.json="$root/deploy/compose/keycloak/realm-relsynth.json" --dry-run=client -o yaml | kc apply -f -
kc apply -f "$here/k8s/deps.yaml"
kc rollout status deploy/metadata deploy/sources deploy/keycloak --timeout=600s
helm upgrade --install e2e "$chart" -n "$ns" -f "${chart}/../../e2e/k8s/values-e2e.yaml" --set image.tag="$v" --wait --timeout 10m
helm test e2e -n "$ns"
python3 "$here/e2e_flow.py" --api http://localhost:30080 \
  --token-url http://localhost:30081/realms/relsynth/protocol/openid-connect/token \
  --source-url postgresql://reader@sources:5432/shop --secret-ref file:/run/secrets/sources/shop-db/password
# restart survival: delete the worker pod that runs a job; the other worker must finish it
python3 "$here/restart_check.py" --k8s-namespace "$ns" --api http://localhost:30080 \
  --token-url http://localhost:30081/realms/relsynth/protocol/openid-connect/token
kc wait --for=condition=available deploy -l app.kubernetes.io/component=worker --timeout=300s

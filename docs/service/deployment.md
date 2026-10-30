# Deploying the service

On-premises first: the same image runs in docker compose for small installs, on Kubernetes and OpenShift with the Helm chart, and in a customer's own AWS, Azure or GCP account. Nothing calls home, and every artefact can be carried into an air-gapped network.

## What every install needs

| Need | Small install (compose) | Kubernetes / OpenShift |
|---|---|---|
| Identity provider | Keycloak from `deploy/compose`, or the organisation's IdP | the organisation's IdP (OIDC; SAML through a broker, see [identity.md](identity.md)) |
| Metadata database | PostgreSQL container | PostgreSQL with HA (CloudNativePG, Crunchy, Patroni, or a managed service) |
| Artefact storage | Docker volume | ReadWriteMany PVC (or one PVC per tenant) |
| Master key or KMS | a file secret | a Kubernetes Secret, Vault transit, or a cloud KMS |
| Source credentials | file secrets | Kubernetes Secrets (`sourceSecrets`), Vault, or a cloud secret manager |
| TLS | a reverse proxy in front of port 8080 | the ingress controller or OpenShift Route |

## docker compose (one machine)

```bash
cd deploy/compose
cp .env.example .env                                 # set real passwords
mkdir -p secrets
head -c32 /dev/urandom | base64 > secrets/master_key # keep a copy in the organisation's secret store
printf '%s' "<source db password>" > secrets/source_db_password
docker compose up -d --build --wait
```

The UI is at `http://localhost:8080/ui/`, Keycloak at `http://localhost:8081`. The demo realm has one user per role (`erin` engineer, `pat` privacy officer, `cody` consumer, `olga` data owner, `tom` tenant admin in tenant `acme`; `bob` in `beta`; `root` platform admin) with passwords `<user>-Demo-Passw0rd`: replace them, and delete the `relsynth-e2e` client, before any real use. Containers run read-only, without capabilities, as UID 10001. Put a TLS proxy (nginx, Caddy, HAProxy) in front and set `OIDC_HOST` / `OIDC_ISSUER` to the public HTTPS names.

`deploy/e2e/compose-e2e.sh` brings the whole stack up, runs the acceptance flow and a worker kill, and tears it down.

## Helm (Kubernetes, OpenShift)

```bash
kubectl create namespace relsynth
kubectl -n relsynth create secret generic relsynth-db \
  --from-literal=url='postgresql+psycopg://relsynth:<pw>@pg-rw.db:5432/relsynth?sslmode=verify-full'
kubectl -n relsynth create secret generic relsynth-master-key --from-literal=master.key="$(head -c32 /dev/urandom | base64)"
kubectl -n relsynth create secret generic shop-db --from-literal=password='<source password>'
helm install relsynth deploy/helm/relsynth -n relsynth -f my-values.yaml
helm test relsynth -n relsynth
```

Minimal `my-values.yaml`:

```yaml
image: {repository: registry.example.org/relsynth/relsynth-service, tag: "1.0.1"}
config:
  oidc_issuer: https://sso.example.org/realms/relsynth
  allowed_hosts: "*.db.example.org"
sourceSecrets: [{name: shop-db}]          # connections use secret_ref file:/run/secrets/sources/shop-db/password
ingress: {enabled: true, className: nginx, host: relsynth.example.org, tlsSecret: relsynth-tls}
networkPolicy:
  ingressFrom: [{namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: ingress-nginx}}}]
tenants:                                  # optional: own worker pool and volume per tenant
  - {id: acme, workers: 2, storage: {existingClaim: relsynth-acme}}
```

What the chart creates: the API Deployment (2 replicas, anti-affinity, PodDisruptionBudget, startup/readiness/liveness probes), one worker Deployment per pool (shared pool plus one per entry of `tenants`, rolling with `maxUnavailable: 0` and a long grace period so running jobs finish), a pre-install/pre-upgrade migration Job, the Service, an Ingress or OpenShift Route, a default-deny NetworkPolicy with explicit egress, a ServiceAccount without an auto-mounted token, the data PVC (kept on uninstall), and a `helm test` pod. All pods run non-root with a read-only root filesystem, no capabilities, `RuntimeDefault` seccomp, and an `emptyDir` for each running job's plaintext outputs.

**OpenShift.** Set `pinUid: false` (OpenShift assigns a UID from the namespace range; the image works under any UID with group 0, so it fits the `restricted-v2` SCC unchanged) and `openshift.route.enabled: true`.

**Secrets from Vault or a cloud store.** Add `VAULT_ADDR` and `VAULT_K8S_ROLE` to `extraEnv` and use `vault:` references in connections; or use the External Secrets Operator or the Secrets Store CSI driver to project secrets as files and keep `file:` references. Set `RELSYNTH_SVC_SECRET_PREFIX_VAULT=vault:kv/data/relsynth/{tenant}/` so each tenant can only name its own secrets.

`deploy/e2e/k8s-e2e.sh` installs the chart with test dependencies (PostgreSQL, Keycloak) in a kind, k3d or k3s cluster and runs the acceptance flow.

## Air-gapped install

On a machine with internet access:

```bash
python tools/make_airgap_bundle.py --out relsynth-airgap --images
tar czf relsynth-airgap.tgz relsynth-airgap
```

The bundle has every wheel (relsynth and dependencies for Linux x86_64), the service image built offline from those wheels, the PostgreSQL and Keycloak images, the packaged Helm chart, the compose files, these docs and `SHA256SUMS`. Carry it in through the organisation's media-import process, then follow its `INSTALL.md`: verify the checksums, `docker load` (or `ctr images import`) the images, push them to the internal registry, `helm install` with `image.repository` pointing there. The OIDC JWKS can come from a file (`OIDC_JWKS_FILE`) when the pods cannot reach the IdP's discovery URL. The service makes no outbound calls of its own beyond the IdP, the KMS or Vault, and the source databases.

## In a customer's cloud account

The same chart runs on the managed Kubernetes services; only the storage, identity and key services change. relsynth is installed and operated by the customer in their own account; no data or telemetry leaves it.

| | AWS | Azure | GCP |
|---|---|---|---|
| Cluster | EKS | AKS | GKE |
| Metadata DB | RDS / Aurora PostgreSQL (Multi-AZ) | Azure Database for PostgreSQL Flexible Server (zone-redundant HA) | Cloud SQL for PostgreSQL (regional HA) |
| Artefact storage (RWX) | EFS CSI driver | Azure Files (Premium, NFS) | Filestore CSI driver |
| Key encryption | `kms: aws-kms`, `kms_key_id: arn:aws:kms:...` | `kms: azure-kv`, `kms_key_id: https://<vault>.vault.azure.net/keys/<key>/<version>` (one key per tenant) | `kms: gcp-kms`, `kms_key_id: projects/.../cryptoKeys/...` |
| Source secrets | `aws-sm:<secret id>#password` | `azure-kv:<vault>/<name>` | `gcp-sm:projects/<p>/secrets/<s>/versions/latest` |
| Pod identity | IRSA or EKS Pod Identity (`serviceAccount.annotations: eks.amazonaws.com/role-arn`) | Workload Identity (`azure.workload.identity/client-id`) | Workload Identity (`iam.gke.io/gcp-service-account`) |
| Identity provider | IAM Identity Center via OIDC app, Cognito, or the corporate IdP | Microsoft Entra ID (app registration, app roles → `relsynth_roles`) | Google Workspace / Cloud Identity via OIDC, or the corporate IdP |
| Ingress / TLS | AWS Load Balancer Controller + ACM | Application Gateway for Containers or NGINX + Key Vault certificates | GKE Gateway + Google-managed certificates |

Install the cloud SDK extras in the image (`EXTRAS=service,postgres,...` plus `boto3`, `azure-identity azure-keyvault-keys azure-keyvault-secrets`, or `google-cloud-kms google-cloud-secret-manager`). Grant the pod identity only `kms:Encrypt`/`kms:Decrypt` on the one key (AWS; equivalents on Azure and GCP) and read access to the relsynth secret path. Keep the database and storage endpoints private (VPC endpoints, Private Link, Private Service Connect).

## Upgrades and rollback

See [operations.md](operations.md#upgrades): the migration Job runs before new pods start, workers finish running jobs, `helm rollback` returns to the previous release.

# Tenants and isolation

A tenant is an organisational unit (a business line, a subsidiary, a customer of a shared platform team) whose projects, connections, specs, runs, reports, approvals and audit log no other tenant may see. Isolation is enforced in layers, so that a mistake in one layer does not expose data.

| Layer | Mechanism | Tested by |
|---|---|---|
| API | Every tenant route is `/api/v1/t/{tenant}/...`; a caller without a role in `{tenant}` gets 404 for every id, the same answer as for an id that does not exist, so ids cannot be probed. | `test_tenant_isolation_on_every_route` (19 routes, both tenants' paths); `deploy/e2e/e2e_flow.py` on compose and Kubernetes |
| Store | Every read and write takes the tenant id and filters on it (`Store._get/_list/_update`); handlers have no unscoped accessor. | `test_rows_are_invisible_across_tenants` |
| Files | Artefacts live under `tenants/<id>/`; the reader checks the row's tenant and that the resolved path lies inside the tenant folder. | `test_other_tenant_cannot_read_or_unwrap` |
| Keys | Each artefact has its own data key, wrapped by the tenant's key-encryption key (HKDF of the master key with the tenant id, or a KMS with the tenant as encryption context). A file copied into another tenant's folder cannot be decrypted with that tenant's key. | `test_other_tenant_cannot_read_or_unwrap`, `test_tampering_is_detected` |
| Sources | File sources must resolve (after symlinks and `..`) below `<file root>/<tenant>`; secret references can be confined to a per-tenant prefix (`RELSYNTH_SVC_SECRET_PREFIX_VAULT=vault:kv/data/relsynth/{tenant}/`); database hosts can be allow-listed. | `test_source_policy`, `test_secret_refs` |
| Compute | Each job runs in its own process with a clean environment (only the source secret of that job). Per-tenant worker pools (`WORKER_TENANTS`, Helm `tenants[]`) give a tenant its own pods, node pool and resource limits. | `test_worker_tenant_pools_and_retry_time` |
| Storage | Helm `tenants[].storage.existingClaim` gives a tenant its own volume, mounted only in the API and that tenant's workers. | chart template |
| Audit | One hash chain per tenant; a tenant's auditors see only their chain. | `test_audit_chain_verifies_and_detects_tampering` |

Stronger separation, when required, is one release of the chart per tenant in its own namespace (or cluster), with its own metadata database and keys. The same image and chart support both models.

## What the service persists

| Kept | Where | Contains real data? |
|---|---|---|
| tenants, projects, connections | metadata store | locations and secret *references*; no credentials |
| spec versions | metadata store | table and column names, roles, rules, inferred settings; introspection does not write values except category levels the spec declares (as in relsynth 1.0) |
| runs, approvals, service accounts | metadata store | ids, states, user ids, reasons; token hashes |
| audit log | metadata store | actions and ids; no values |
| scorecard, quality report | encrypted artefacts | aggregate statistics of real and synthetic data |
| synthetic tables | encrypted artefacts | synthetic rows only |

**Not kept:** source rows, samples, the diagnostics bundle, tracebacks (only the exception message, cut to 1,000 characters), and the job's work folder (deleted after every attempt; an `emptyDir` in Kubernetes). `test_no_source_row_is_persisted` runs a synthesis on a source whose every customer name contains a marker string and then searches the metadata store, every decrypted artefact and every file on disk for it.

**Encryption in transit.** TLS terminates at the ingress or route (HSTS is sent on every response); inside the cluster, use a service mesh with mTLS or the ingress controller's re-encryption, and `sslmode=verify-full` (or the driver's equivalent) for the metadata and source databases.

**Encryption at rest.** Artefacts are encrypted by the service (above). The metadata store holds no secrets or source values, but should still sit on encrypted volumes (the database's or the storage class's encryption).

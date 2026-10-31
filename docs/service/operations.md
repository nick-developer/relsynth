# Operating the service

## Health checks

| Endpoint | Meaning | Use |
|---|---|---|
| `GET /healthz` | the process answers | liveness probe |
| `GET /readyz` | the metadata store answers and the data folder is writable | readiness probe; 503 otherwise |
| `relsynth-service check` | settings valid and store reachable | worker liveness (exec probe), install scripts |

## Logs

One JSON object per line on stdout (`LOG_JSON=true`): `ts`, `level`, `logger`, `msg`, and where present `request_id`, `tenant`, `actor`, `action`, `target`, `outcome`, `status`, `path`, `ms`. Every response carries `X-Request-Id` (taken from the request when it is a safe token, otherwise generated), so a user's report can be matched to the log. Ship stdout with the platform's collector (Fluent Bit, Vector, OpenShift logging). Logs never contain tokens, secrets, spec bodies or data values; unhandled errors log a stack trace server-side and return only the request id.

## Audit log

Every change and every refusal is an entry in the tenant's audit log: `seq`, `ts`, `actor` (`user:<sub>` or `service:<id>`), `action` (for example `run.queued`, `approval.approve`, `data.downloaded`, `access.denied`, `download.refused`), `target`, `outcome`, `detail`, `prev_hash`, `hash`. Each entry's hash is SHA-256 over its canonical JSON including the previous entry's hash, so editing, deleting or reordering entries breaks the chain:

```bash
curl -H "Authorization: Bearer $T" https://relsynth/api/v1/t/acme/audit/verify   # {"ok": true, "entries": 1234, "head": "..."}
relsynth-service audit verify acme
relsynth-service audit export acme > acme-audit.jsonl                               # to the SIEM or WORM storage
```

Truncating the newest entries cannot be seen from the chain alone. Export the head hash on a schedule to append-only storage (S3 Object Lock, WORM share, SIEM); the governance build (T08) adds signed checkpoints for this (Prop. 803). Retention follows the organisation's policy; the audit table is append-only for the service, and the database role used by the service can be denied `UPDATE`/`DELETE` on it.

## Metrics

`GET /metrics` in Prometheus text format: `relsynth_http_requests_total{method,route,status}`, `relsynth_http_request_seconds` (histogram by route), `relsynth_jobs_total{state,kind}` (per worker process), `relsynth_span_seconds{span}`. Labels never carry tenant ids or user ids. Restrict `/metrics` to the monitoring namespace with the NetworkPolicy (it is not authenticated) or scrape through the ingress with an allow-list.

Suggested alerts: `/readyz` failing for 2 minutes; 5xx rate above 1 %; queue depth (`SELECT count(*) FROM runs WHERE state='queued'`) growing for 30 minutes; runs in `running` whose `lease_expires` is in the past for more than 10 minutes (no worker alive); audit verify not ok.

## Traces

`OTEL=true` with `relsynth[service-otel]` installed exports spans over OTLP/HTTP; configure the collector with the standard variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`). Spans: one per HTTP request, one per job attempt (`relsynth.job` with tenant, run and kind attributes).

## Jobs

- **States:** `queued → running → succeeded | failed | cancelled`; a retry goes back to `queued` with a `next_attempt_at`.
- **Leases:** a worker owns a job for `LEASE_S` (60 s) and renews it every `HEARTBEAT_S` (10 s). If the worker dies, the lease runs out and another worker claims the job (`attempts` + 1). Only the current lease holder can record an outcome, so a slow or partitioned worker cannot overwrite the new attempt.
- **Retries:** I/O errors (exit code 3: the source restarted, the network dropped) are retried with backoff 30 s, 60 s, 120 s ... up to `MAX_ATTEMPTS`; spec, engine and quality-gate failures are final.
- **Limits:** each attempt is a separate process (own process group, killed with the worker), with optional `JOB_MEMORY_MB` (address space), `JOB_CPU_S` and the wall-clock `JOB_TIMEOUT_S`. In Kubernetes the pod's limits bound all jobs of a worker.
- **Cancellation:** queued jobs stop at once; running jobs at the next heartbeat (SIGTERM, then SIGKILL after 10 s).
- **Shutdown:** SIGTERM drains: the worker takes no new job and finishes the current one (Helm `terminationGracePeriodSeconds`, compose `stop_grace_period`). `RELSYNTH_SVC_DRAIN=kill` gives the job back instead, so another worker resumes it.

## Backup and restore

What to back up:

1. **Metadata database** (PostgreSQL): `pg_dump -Fc` nightly plus WAL archiving (or the managed service's point-in-time recovery). It holds specs, runs, approvals and the audit log.
2. **Data volume** (`DATA_DIR/tenants`): encrypted artefacts; file-level or volume snapshots. Without the keys these are unreadable, which also makes off-site copies safe.
3. **Keys:** the master key secret (local KMS) or the KMS key policy. Losing it loses every artefact; store it in two places with separate access (for example Vault and an offline envelope).

Restore order: keys, then database, then volume; then `relsynth-service migrate`, start the API, check `/readyz`, run `audit verify` for each tenant, start workers. Runs that were `running` at backup time are resumed by the workers when their leases (from the old clock) are found expired. Test the restore every quarter (control mapping: A.8.13, CP-9, A1.2).

## High availability

| Component | HA approach |
|---|---|
| API | 2+ replicas behind the Service, pod anti-affinity, PodDisruptionBudget (`minAvailable: 1`); stateless |
| Workers | 2+ replicas per pool; a failed worker's job is resumed by another (lease expiry) |
| Metadata database | PostgreSQL with a synchronous standby (CloudNativePG, Crunchy, Patroni, or a managed service); the service reconnects (`pool_pre_ping`) |
| Data volume | ReadWriteMany storage with its own replication (CephFS, NFS on HA storage, EFS, Azure Files, Filestore) |
| Identity provider | the organisation's IdP; tokens are verified locally from cached JWKS, so a short IdP outage does not stop running sessions |

SQLite is for single-node installs and tests only: use PostgreSQL whenever there is more than one pod.

## Upgrades

`helm upgrade` runs the migrate job (pre-upgrade hook) before new pods start; workers roll one at a time (`maxUnavailable: 0`) and finish their current job first. Read CHANGELOG.md for any manual step. Roll back with `helm rollback` (schema changes are additive within a major version).

## Runbook

| Symptom | Check | Action |
|---|---|---|
| Runs stay `queued` | workers running? `relsynth_jobs_total` moving? pool tenant filters | scale workers; fix `WORKER_TENANTS` |
| Runs fail with "cannot prepare the job" | secret reference, `SECRET_PREFIX_*`, mounted secret | fix the connection's `secret_ref` or the mount |
| 401 for everyone | issuer URL, audience, JWKS reachable from the pods, clock skew | `OIDC_JWKS_URL` to an in-cluster address; NTP |
| 404 on a run a user can see in the UI of another tenant | expected: tenants are isolated | give the user a role in that tenant |
| `audit verify` not ok | which entry; database access logs | treat as a security incident (pentest plan §6) |

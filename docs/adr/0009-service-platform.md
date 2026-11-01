# ADR 0009: Service platform stack

- Status: accepted (roadmap thread T09, 2026-10-30; merged and released in 2.0.0)
- Numbered after the roadmap thread to avoid clashes with ADRs of parallel threads.

## Context

T09 turns the library into a service an organisation deploys on its own premises first (air-gapped included), with SSO, RBAC, multi-tenancy, a job queue that survives restarts, and a web UI. The service must add as little infrastructure as possible: every extra component is something an on-premises operations team must patch, back up and secure.

## Decisions

1. **API: FastAPI on Starlette and Uvicorn.** Typed request models produce the OpenAPI document the T09 scope asks for; MIT/BSD licences; small dependency set. gRPC is not built: no consumer needs it yet.
2. **Job queue in the metadata database, not a broker.** Jobs are rows; workers claim them with an atomic compare-and-set update and hold a lease renewed by heartbeats. This needs only the PostgreSQL the service already requires, gives exactly one live owner per job, survives worker and node loss (lease expiry), and keeps job state and audit in one transactional store. Celery with Redis or RabbitMQ was rejected: two more stateful systems to run in an air-gapped site, and at-least-once delivery without lease fencing. Throughput is ample: jobs run minutes to hours, so claims are rare.
3. **One process per job attempt.** Memory, CPU and time limits, cancellation and secret scoping apply to the job alone; a crash cannot take the worker down.
4. **Identity: OIDC only in the service; SAML through a broker.** Avoids XML-signature handling in relsynth; every enterprise IdP supports OIDC or can be brokered by Keycloak or Dex.
5. **Authorisation: a permission table in code plus an optional classification rule (ABAC).** Cedar or OPA for access control was considered (governance ADR 0002 left Cedar open for T09). The role model is small and fixed, and a second policy engine on every request adds latency and an operational dependency; the decision is revisited if organisations need custom access rules.
6. **Tenant isolation in the shared deployment, with per-tenant pools and volumes as options;** one release per tenant for the strongest isolation. Same image and chart for both.
7. **Encryption: envelope encryption in the application** (AES-256-GCM, per-tenant KEKs via HKDF or KMS), so isolation holds even when storage is shared or copied.
8. **Web UI without a build chain** (plain JavaScript): nothing to audit in node_modules, a strict CSP, and it ships inside the Python wheel.

## Consequences

PostgreSQL is mandatory for multi-node installs. Fair-share scheduling across tenants on the shared pool is not built (open item). Governance (T08) plugs in through `relsynth.service.release` (RFC-T09-01).

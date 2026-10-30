# OWASP ASVS 4.0.3 level 2: checklist with evidence

Scope: the relsynth service (API, web UI, workers) as of T09. Status: **Pass** (met, with evidence), **N/A** (does not apply, with the reason), **Deployer** (met by the platform the organisation runs; the deployment guide says how), **Open** (not yet met). Chapters are grouped by requirement area; every L2 requirement of ASVS 4.0.3 was reviewed and the table lists each one that applies to this service or needs an explanation. Requirements about features the service does not have (password storage, password reset, file upload, SOAP, GraphQL, WebSockets, deserialisation of untrusted objects) are N/A as a group in the last section.

No **high** finding is open. Open items are medium or low and listed at the end.

## V1 Architecture, design and threat modelling

| # | Requirement (short) | Status | Evidence |
|---|---|---|---|
| 1.1.2 | Threat modelling for design changes | Pass | [threat-model.md](threat-model.md) |
| 1.1.3 | User stories with security constraints | Pass | T09 prompt acceptance criteria; [identity.md](../identity.md) |
| 1.1.4 | Trust boundaries documented | Pass | threat model data-flow diagram (TB1–TB8) |
| 1.1.5 | High-level architecture documented | Pass | [README.md](../README.md#architecture) |
| 1.1.6 | Centralised security controls | Pass | `auth.py` (authn/authz), `store.py` (tenant scope), `crypto.py`, middleware in `app.py` |
| 1.2.1 | Unique low-privilege OS accounts for components | Pass | image runs as UID 10001 / arbitrary UID; no capabilities; read-only root fs |
| 1.2.2 | Component communication authenticated | Deployer | DB credentials via Secret; mTLS by service mesh ([tenancy.md](../tenancy.md)) |
| 1.2.3 | Single vetted authentication mechanism | Pass | OIDC via PyJWT; service accounts as the only alternative |
| 1.2.4 | Same strength on all authentication paths | Pass | every route depends on `principal()`; only health, metrics, ui-config and static UI are public |
| 1.4.1 | Access control enforced at trusted points | Pass | server-side `need()` per route and tenant-scoped store |
| 1.4.4 | Single well-vetted access-control mechanism | Pass | `PERMISSIONS` table; `test_rbac_table_is_consistent` |
| 1.4.5 | Attribute/feature-based access control | Pass | ABAC on classification ([identity.md](../identity.md#abac-optional)) |
| 1.5.1 | Input/output requirements defined | Pass | Pydantic models in OpenAPI ([openapi.json](../openapi.json)) |
| 1.5.2 | No untrusted serialisation | Pass | JSON and `yaml.safe_load` only |
| 1.5.3 | Input validation on a trusted layer | Pass | server-side Pydantic and `policy.check_source` |
| 1.6.1 | Explicit key management policy | Pass | [tenancy.md](../tenancy.md), `crypto.py` docstring, [operations.md](../operations.md#backup-and-restore) |
| 1.6.2 | Consumers of keys protect key material | Pass | KEK in KMS or a Secret mounted read-only (0440); never logged |
| 1.6.3 | Keys and passwords replaceable | Pass (partly) | KMS keys rotate in the KMS; local master-key rotation needs re-wrapping: Open item O3 |
| 1.6.4 | Symmetric keys not used as long-term secrets in clients | Pass | none in the browser |
| 1.7.1 | Common logging format | Pass | JSON logs ([operations.md](../operations.md#logs)) |
| 1.7.2 | Logs sent to a remote system | Deployer | stdout collected by the platform |
| 1.8.1 | Sensitive data identified and classified | Pass | assets A1–A8 in the threat model |
| 1.8.2 | Protection levels per class | Pass | [tenancy.md](../tenancy.md#what-the-service-persists) |
| 1.9.1 | Encrypted communications between components | Deployer | TLS at ingress; `sslmode=verify-full` documented |
| 1.11.1 | Business logic flows documented | Pass | README "The flow"; approval rules in `release.py` |
| 1.11.2 | High-value flows not shared unsafely across threads | Pass | atomic compare-and-set job claims; `test_claims_are_exclusive_under_contention` |
| 1.14.1 | Segregation of components | Pass | API vs workers vs job processes; per-tenant pools; NetworkPolicy |
| 1.14.2 | Signed binaries / trusted sources | Pass | T01 Sigstore, SBOM; offline image build from checked wheels |
| 1.14.3 | Build pipeline warns on outdated or insecure components | Pass | pip-audit in CI (T01) |
| 1.14.4 | Build pipeline with automated secure deployment | Pass | CI gates; Helm chart |
| 1.14.6 | No unsupported client-side tech | Pass | plain HTML/JS/CSS |

## V2 Authentication and V3 Session management

Authentication is delegated to the organisation's IdP (ASVS 2.x password, MFA, OTP, lock-out and recovery requirements are the IdP's: **Deployer**). The service checks tokens.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 2.2.1 | Anti-automation (brute force) | Deployer | Keycloak `bruteForceProtected: true` in the demo realm; IdP policy |
| 2.5.4 | No shared or default accounts | Pass (warning) | demo realm users are for evaluation; deployment guide says to replace them |
| 2.9.x | Cryptographic verifier (tokens) | Pass | asymmetric JWT signatures only; `test_alg_none_and_hmac_confusion_are_refused` |
| 2.10.1 | Service authentication secrets not static/in code | Pass | DB URL and keys from Secrets; secret references for sources |
| 2.10.2 | Service accounts not default credentials | Pass | random 256-bit tokens created per account |
| 2.10.3 | Passwords stored with protection | Pass | service-account tokens stored as SHA-256 of a 256-bit random value |
| 2.10.4 | Secrets from a vault, not source code | Pass | `secrets.py` (Vault, cloud managers, files) |
| 3.1.1 | No session tokens in URLs | Pass | bearer header only; OIDC code exchanged and removed from the URL (`history.replaceState`) |
| 3.2.x | Session binding and generation | N/A / Deployer | stateless bearer tokens; sessions live at the IdP |
| 3.3.1 | Logout invalidates | Pass (client) / Deployer | UI drops the token; IdP session end at the IdP |
| 3.3.2 | Re-authentication after idle/absolute timeout | Deployer | token lifetime 10 min, SSO idle 30 min in the demo realm |
| 3.5.2 | Stateless tokens, not static API keys, for sessions | Pass | OIDC JWTs; service accounts only for machines |
| 3.5.3 | Stateless tokens use signatures and protections | Pass | `jwt.decode` with `iss/aud/exp/nbf` required |
| 3.7.1 | Re-authentication for sensitive operations | Open (low) O5 | approvals rely on the current session; step-up via `acr` claim planned |

## V4 Access control

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 4.1.1 | Enforced on a trusted service layer | Pass | server-side only |
| 4.1.2 | Attributes cannot be manipulated by users | Pass | roles come from signed tokens; `test_tampered_token_is_refused` |
| 4.1.3 | Least privilege | Pass | role table; consumers see no specs, connections or privacy items |
| 4.1.5 | Access control fails securely | Pass | default 401/403/404; unknown tenant → 404 |
| 4.2.1 | Protection against IDOR | Pass | `test_tenant_isolation_on_every_route`; e2e checks |
| 4.2.2 | CSRF protection | Pass | bearer header (not cookies), so cross-site requests carry no credentials; CORS off by default |
| 4.3.1 | Admin interfaces use MFA | Deployer | IdP policy for `platform_admin`/`tenant_admin` |
| 4.3.2 | Directory browsing disabled | Pass | StaticFiles serves only the UI folder, no listing |
| 4.3.3 | Additional authorisation for high-value actions | Pass | separation of duties and quorum on approvals |

## V5 Validation, sanitisation and encoding

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 5.1.1 | HTTP parameter pollution | Pass | typed FastAPI parameters |
| 5.1.3 | Positive input validation | Pass | Pydantic models with lengths, ranges, enums; tenant id regex |
| 5.1.4 | Structured data strongly typed | Pass | OpenAPI schemas |
| 5.1.5 | URL redirects to allow-listed destinations | Pass | only `/` → `/ui/`; OIDC redirect URIs allow-listed at the IdP |
| 5.2.6 | SSRF protection | Pass (configure) | host allow-list, NetworkPolicy egress; `test_source_policy` |
| 5.3.1 | Output encoding for context | Pass | UI uses textContent; JSON responses |
| 5.3.3 | Context-aware escaping against XSS | Pass | no innerHTML; CSP |
| 5.3.4 | Parameterised queries | Pass | SQLAlchemy Core expressions only in the service |
| 5.3.8 | OS command injection | Pass | one fixed `subprocess` argv (runner), no shell |
| 5.3.9 | Local/remote file inclusion | Pass | path policy with resolve and tenant root |
| 5.5.2 | XML parsers restrict XXE | N/A | no XML (SAML via broker by design) |
| 5.5.3 | Deserialisation of untrusted data | Pass | `yaml.safe_load`, JSON |

## V7 Errors and logging, V8 Data protection

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 7.1.1 | No credentials or session tokens in logs | Pass | log fields are fixed; no headers logged |
| 7.1.2 | No other sensitive data in logs | Pass | no values, specs or errors with data in logs; job stderr discarded |
| 7.1.3 | Security events logged | Pass | `access.denied`, `approval.refused`, `download.refused`, `connection.refused` |
| 7.1.4 | Logs have enough context | Pass | request id, tenant, actor, action, target, outcome |
| 7.2.1 | Authentication decisions logged | Pass | failed token checks logged as `authentication failed` with the reason and request id; authorisation denials in the audit log |
| 7.2.2 | Access-control failures logged | Pass | `access.denied` audit entries; tenant refusals logged |
| 7.3.1 | Log injection prevention | Pass | JSON encoding of every field |
| 7.3.3 | Logs protected from modification | Pass | audit hash chain; export to WORM |
| 7.4.1 | Generic error messages with a reference | Pass | 500 returns `internal error` plus `request_id` |
| 7.4.2 | Exception handling across the code base | Pass | middleware catch-all; job errors captured per attempt |
| 8.1.1 | Sensitive data not cached server-side unprotected | Pass | `Cache-Control: no-store` |
| 8.1.6 | Backups protected | Pass | artefacts encrypted; [operations.md](../operations.md#backup-and-restore) |
| 8.2.1 | Anti-caching headers | Pass | `no-store` on every response |
| 8.2.2 | Browser storage holds no sensitive data | Pass (accepted risk) | access token in sessionStorage per tab, 10-minute lifetime; no data stored |
| 8.3.1 | Sensitive data in body/headers, not URL | Pass | ids only in URLs |
| 8.3.4 | Sensitive data identified with a policy | Pass | [tenancy.md](../tenancy.md) |
| 8.3.5 | Access to sensitive data audited | Pass | `report.read`, `data.downloaded`, `artifact.downloaded` |
| 8.3.7 | Sensitive data encrypted at rest | Pass | AES-256-GCM envelope encryption |
| 8.3.8 | Data retention | Deployer / Open O4 | retention follows the organisation; an automatic expiry job is not built |

## V6 Cryptography, V9 Communications

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 6.2.1 | Modules fail securely | Pass | `InvalidTag` on any tampering; `test_tampering_is_detected` |
| 6.2.2 | Industry-proven algorithms | Pass | AES-256-GCM, HKDF-SHA256 (RFC 5869), STREAM construction; `cryptography` library |
| 6.2.3 | Nonce/IV used correctly | Pass | fresh random key per artefact; per-chunk nonce = random prefix + counter |
| 6.2.5 | No insecure modes or padding | Pass | AEAD only |
| 6.3.1 | CSPRNG for random values | Pass | `secrets` module for keys, nonces, tokens |
| 6.4.1 | Key management solution | Pass | KMS options; per-tenant KEK |
| 6.4.2 | Key material not exposed to the application unnecessarily | Pass (KMS) | with a KMS, KEKs never leave it |
| 9.1.1 | TLS for all client connectivity | Deployer | ingress/route TLS; HSTS header |
| 9.2.1 | Trusted certificates for server connections | Deployer | `sslmode=verify-full`; Vault over https |

## V10 Malicious code, V12 Files, V13 API, V14 Configuration

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 10.3.2 | Integrity protections for code and libraries | Pass | signed releases, SBOM, offline image builds |
| 12.3.1 | Filenames from users not used directly | Pass | download names from artefact names with `Path.name`; storage paths are ids |
| 12.4.1 | Files stored outside the web root | Pass | data folder is not served |
| 13.1.1 | Same encoding and parsers everywhere | Pass | FastAPI/JSON |
| 13.1.3 | API URLs carry no sensitive data | Pass | ids only |
| 13.1.4 | Authorisation at URI and resource level | Pass | tenant in path plus row checks |
| 13.2.1 | Methods restricted | Pass | explicit routes per method |
| 13.2.2 | JSON schema validation | Pass | Pydantic |
| 13.2.5 | Content type checked | Pass | JSON bodies parsed by FastAPI only for `application/json` |
| 13.2.6 | Size limits | Pass | 16 MB body limit; field limits |
| 14.1.1 | Secure, repeatable build | Pass | Dockerfile, CI |
| 14.1.3 | Hardened configuration | Pass | non-root, read-only fs, caps dropped, seccomp; chart passes Pod Security "restricted" (server dry run on Kubernetes 1.31) |
| 14.2.1 | Components up to date | Pass | pip-audit, Dependabot (T01) |
| 14.2.3 | Third-party assets from trusted sources (SRI) | N/A | no third-party browser assets |
| 14.2.4 | Components from pre-defined trusted repositories | Pass | PyPI wheels pinned in the bundle with SHA-256 |
| 14.2.5 | SBOM | Pass | CycloneDX SBOM in releases (T01) |
| 14.3.2 | Debug modes off in production | Pass | no debug flag; dev sign-in needs two settings |
| 14.3.3 | Headers do not reveal versions | Pass | `server_header=False`; version only in `/readyz` and `ui-config` (Open O7, low) |
| 14.4.1 | Content-Type with charset | Pass | FastAPI JSON responses |
| 14.4.3 | Content-Security-Policy | Pass | `default-src 'self'; script-src 'self'; frame-ancestors 'none' ...` |
| 14.4.4 | `X-Content-Type-Options: nosniff` | Pass | middleware |
| 14.4.5 | HSTS | Pass | middleware |
| 14.4.6 | Referrer-Policy | Pass | `no-referrer` |
| 14.4.7 | Framing restricted | Pass | `frame-ancestors 'none'`, `X-Frame-Options: DENY` |
| 14.5.1 | Only required HTTP methods | Pass | routes |
| 14.5.3 | CORS allow-list | Pass | no CORS middleware; same-origin UI |

## Not applicable as a group

V2.1 (password security), V2.3 (authenticator lifecycle), V2.4 (credential storage of user passwords), V2.5 (credential recovery), V2.7/2.8 (OTP): handled by the IdP. V3.4 (cookie-based sessions): no cookies. V12.1–12.2, 12.5–12.6 (uploads, SSRF via uploads): no file upload. V13.3 (SOAP), V13.4 (GraphQL), WebSockets: none. V5.5.1 (serialised objects signed): none exchanged.

## Open items

| Id | Item | Severity | Plan |
|---|---|---|---|
| O1 | Per-tenant quotas and fair-share scheduling (DoS between tenants on the shared pool) | Medium | quota table and admission check; meanwhile use per-tenant pools |
| O2 | Signed audit checkpoints | Medium | T08 governance build (Prop. 803) |
| O3 | Local master-key rotation tool (re-wrap data keys) | Medium | `relsynth-service keys rotate`; KMS installs rotate in the KMS |
| O4 | Automatic retention expiry of artefacts | Low | scheduled job per tenant policy |
| O5 | Step-up authentication for approvals (`acr`/`auth_time` check) | Low | setting `approval_max_auth_age_s` |
| O7 | Version disclosure on `/readyz` and `ui-config` | Low | setting to hide |
| O8 | Third-party penetration test | Required before production | [pentest-plan.md](pentest-plan.md) |

# Identity and access

## Single sign-on

**OIDC (OpenID Connect Core 1.0).** The web UI uses the authorization-code flow with PKCE (RFC 7636) as a public client; there is no client secret in the browser. The API accepts the resulting access token (or an ID token) as `Authorization: Bearer <JWT>` and checks:

- the signature, against the issuer's JWKS (discovered from `<issuer>/.well-known/openid-configuration`, or `OIDC_JWKS_URL`, or a file `OIDC_JWKS_FILE` for air-gapped sites); unknown key ids trigger a refetch, so key rotation needs no restart;
- the algorithm, against an allow-list of asymmetric algorithms (`RS256, ES256, PS256, EdDSA`). `none` and HMAC algorithms are refused and cannot be configured, which rules out the key-confusion attacks of RFC 8725 §2.1–2.2;
- `iss`, `aud`, `exp`, `nbf`, `iat` and `sub`, with 30 s clock leeway.

Tested with Keycloak 26 (realm in `deploy/compose/keycloak/realm-relsynth.json`). Any OIDC provider works: Microsoft Entra ID, Okta, Ping, ADFS 2016+ (OIDC), Google Workspace, Dex, Authentik.

**SAML 2.0.** Connect SAML identity providers through an OIDC broker: Keycloak (identity brokering), Dex (SAML connector), or the IdP's own OIDC endpoint where it has one. relsynth deliberately does not parse SAML assertions itself: XML signature wrapping and canonicalisation bugs are a recurring source of authentication bypasses in SAML service providers (Somorovsky et al., USENIX Security 2012), and a broker that is already hardened and patched is the safer place for that code. The broker maps SAML attributes or groups to the `relsynth_roles` claim.

## Roles

Roles are per tenant. The token carries them in the claim named by `ROLES_CLAIM` (default `relsynth_roles`; dotted paths such as `resource_access.relsynth.roles` work) as strings `"<tenant>:<role>"`. With `TENANT_CLAIM`, bare role names apply to the tenant in that claim. `"*:platform_admin"` marks a platform operator, who can create tenants but has no access to any tenant's data unless also given a role in it.

| Permission | data_owner | privacy_officer | engineer | consumer | tenant_admin | agent |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| see projects and runs | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| create projects, connections, specs | ✓ | | ✓ | | | |
| see connections and specs | ✓ | ✓ | ✓ | | | specs only |
| start and cancel runs, introspect | ✓ | | ✓ | | | |
| read scorecards and quality reports | ✓ | ✓ | ✓ | ✓ | | ✓ |
| read privacy items of reports | ✓ | ✓ | | | | |
| approve, reject, revoke releases | ✓ | ✓ | | | | |
| download released synthetic data | ✓ | | ✓ | ✓ | | |
| read and verify the audit log | ✓ | ✓ | | | ✓ | |
| manage service accounts | | | | | ✓ | |
| ask an agent for a review or an explanation | ✓ | ✓ | ✓ | | | ✓ |
| see agent proposals | ✓ | ✓ | ✓ | | | ✓ |
| accept or reject agent proposals (people only) | ✓ | | ✓ | | | |

The `agent` role is held only by agent accounts (see [Agent accounts](#agent-accounts)). The table is `PERMISSIONS` and `AGENT_PERMISSIONS` in `src/relsynth/service/auth.py`; `test_rbac_table_is_consistent` keeps the two in step.

**Separation of duties.** Nobody approves a run they started, even when they hold both roles (`SEPARATION_OF_DUTIES=true`, governance decision D4). `APPROVAL_QUORUM` asks for more than one distinct approver. A rejection or revocation voids every earlier approval.

**Least privilege for consumers.** A consumer sees projects, run states, scorecards without their privacy items, and released data. They never see connections, specs or privacy assessments.

## ABAC (optional)

With `ABAC=true`, a connection may carry `labels.classification` in `public < internal < confidential < restricted`. A user can see or use it, and its runs, only if their token's `relsynth_attrs.clearance` is at least that level. Nobody can create a connection above their own clearance. The attribute comes from the IdP like the roles. Cedar was the governance design's candidate for richer attribute rules; the built-in rule covers the classification use case without a second policy engine (decision for the user in the T09 report).

## Service accounts

For CI pipelines and schedulers. A tenant admin creates one with a set of roles (`tenant_admin` cannot be delegated); the API returns the token once (`rss_<id>_<256-bit secret>`) and stores only its SHA-256. Tokens are revoked with `DELETE /service-accounts/{id}` and every use is audited as `service:<id>`. Prefer workload identity federation (an OIDC token from the CI system, accepted like any user token) where the CI system supports it.

## Agent accounts

An agent account is a service account created with the role `agent`. It holds that role only, whatever other roles the request lists, and OIDC tokens cannot claim it. It may read projects, specs, runs, reports and proposals and ask for agent reviews and explanations; every other request is refused and the refusal is audited. Agent output becomes proposals, which only a person accepts; an accepted proposal makes a new spec version written by that person. See [docs/agents.md](../agents.md#in-the-service).

## Development sign-in

`AUTH_MODE=dev` plus `ALLOW_DEV_AUTH=true` accepts a header `X-Relsynth-Dev-User: name; tenant:role,...; attr=value`. It exists for tests and local evaluation. The service refuses to start in dev mode without the second flag, and the Helm chart never sets it.

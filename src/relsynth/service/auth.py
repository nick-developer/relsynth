# SPDX-License-Identifier: Apache-2.0
"""Identity and access control.

Authentication
    * Users sign in at the organisation's identity provider with OpenID Connect
      (OIDC Core 1.0).  The API accepts the ID or access token as a Bearer JWT and
      checks its signature against the provider's JWKS, plus ``iss``, ``aud``,
      ``exp``/``nbf`` (with a small leeway) and the algorithm allow-list (no
      ``none``, no HMAC with a public key: RFC 8725 §3.1-3.2).
    * SAML 2.0 identity providers (AD FS, Shibboleth, Okta SAML apps) are
      connected through an OIDC broker such as Keycloak or Dex, which turns the
      SAML assertion into an OIDC token.  relsynth does not parse SAML XML itself
      (XML signature wrapping is a classic SAML flaw; docs/service/identity.md).
    * Service accounts (CI pipelines, schedulers) use opaque tokens
      ``rss_<id>_<secret>``; only their SHA-256 is stored.

Authorisation
    * RBAC: roles are per tenant, carried in the token claim ``relsynth_roles`` as
      ``"<tenant>:<role>"`` (``"*:platform_admin"`` for the platform operator).
    * ABAC (optional): a connection labelled ``classification: confidential``
      needs a principal whose ``clearance`` attribute is at least that level.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt

from .config import Settings

ROLES = ("data_owner", "privacy_officer", "engineer", "consumer", "tenant_admin")
PLATFORM_ADMIN = "platform_admin"

# permission -> roles that hold it (docs/service/identity.md has the same table)
PERMISSIONS: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in {
    "project:read": ROLES,
    "project:write": ("data_owner", "engineer"),
    "connection:read": ("data_owner", "privacy_officer", "engineer"),
    "connection:write": ("data_owner", "engineer"),
    "spec:read": ("data_owner", "privacy_officer", "engineer"),
    "spec:write": ("data_owner", "engineer"),
    "run:create": ("engineer", "data_owner"),
    "run:read": ROLES,
    "run:cancel": ("engineer", "data_owner"),
    "report:read": ("data_owner", "privacy_officer", "engineer", "consumer"),
    "privacy:read": ("data_owner", "privacy_officer"),
    "approval:decide": ("data_owner", "privacy_officer"),
    "data:download": ("data_owner", "engineer", "consumer"),
    "audit:read": ("data_owner", "privacy_officer", "tenant_admin"),
    "sa:manage": ("tenant_admin",),
}.items()}

LEVELS = ("public", "internal", "confidential", "restricted")
TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class AuthError(Exception):
    """401: no or invalid credentials."""


class Forbidden(Exception):
    """403: authenticated, but not allowed."""


@dataclass
class Principal:
    subject: str
    name: str
    kind: str = "user"                                   # user | service
    roles: dict[str, set[str]] = field(default_factory=dict)   # tenant -> roles
    attributes: dict[str, Any] = field(default_factory=dict)
    platform_admin: bool = False

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.subject}"

    def tenants(self) -> list[str]:
        return sorted(self.roles)

    def has(self, tenant: str, perm: str) -> bool:
        return bool(self.roles.get(tenant, set()) & PERMISSIONS[perm])

    def require(self, tenant: str, perm: str) -> None:
        if not self.has(tenant, perm):
            raise Forbidden(f"{perm} needs one of the roles {sorted(PERMISSIONS[perm])} in tenant {tenant}")

    def role_for(self, tenant: str, perm: str) -> str:
        held = sorted(self.roles.get(tenant, set()) & PERMISSIONS[perm])
        return held[0] if held else ""

    def cleared_for(self, labels: dict | None) -> bool:
        lvl = (labels or {}).get("classification")
        if lvl is None:
            return True
        have = self.attributes.get("clearance", "public")
        if lvl not in LEVELS or have not in LEVELS:
            return False
        return LEVELS.index(have) >= LEVELS.index(lvl)


def parse_roles(values, tenant_claim: str | None = None) -> tuple[dict[str, set[str]], bool]:
    roles: dict[str, set[str]] = {}
    admin = False
    if isinstance(values, str):
        values = [values]
    for v in values or []:
        if not isinstance(v, str):
            continue
        t, sep, r = v.partition(":")
        if not sep:              # a bare role applies to the token's single tenant claim
            t, r = tenant_claim or "", v
        if r == PLATFORM_ADMIN and t == "*":
            admin = True
        elif r in ROLES and TENANT_RE.match(t):
            roles.setdefault(t, set()).add(r)
    return roles, admin


class _JWKS:
    """JWKS cache: keys are fetched on first use and again (at most once a
    minute) when a token names an unknown key id, which handles key rotation."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._client: jwt.PyJWKClient | None = None
        self._static: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def key_for(self, token: str):
        if self.s.oidc_jwks_file:
            if self._static is None:
                self._static = json.loads(Path(self.s.oidc_jwks_file).read_text(encoding="utf-8"))
            kid = jwt.get_unverified_header(token).get("kid")
            keys = jwt.PyJWKSet.from_dict(self._static).keys
            for k in keys:
                if kid is None or k.key_id == kid:
                    return k.key
            raise AuthError("token signed by an unknown key")
        with self._lock:
            if self._client is None:
                url = self.s.oidc_jwks_url or _discover(self.s.oidc_issuer)["jwks_uri"]
                self._client = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)
        try:
            return self._client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientError as e:
            raise AuthError(f"cannot find the token's signing key: {e}") from e


_DISCOVERY: dict[str, tuple[float, dict]] = {}


def _discover(issuer: str) -> dict:
    import urllib.request
    hit = _DISCOVERY.get(issuer)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    if not url.startswith(("https://", "http://")):
        raise AuthError("the OIDC issuer must be an http(s) URL")
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310  # nosec B310 (scheme checked above)
        doc = json.loads(r.read())
    _DISCOVERY[issuer] = (time.time(), doc)
    return doc


class Authenticator:
    def __init__(self, settings: Settings, store):
        self.s, self.store = settings, store
        self.jwks = _JWKS(settings)
        self.algorithms = [a.strip() for a in settings.oidc_algorithms.split(",") if a.strip()]
        if any(a.upper().startswith("HS") or a.lower() == "none" for a in self.algorithms):
            raise ValueError("HMAC and 'none' algorithms are not accepted for OIDC tokens")

    def authenticate(self, authorization: str | None, dev_user: str | None = None) -> Principal:
        if self.s.auth_mode == "dev" and self.s.allow_dev_auth and dev_user:
            return self._dev(dev_user)
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("missing bearer token")
        token = authorization[7:].strip()
        if token.startswith("rss_"):
            return self._service(token)
        if self.s.auth_mode != "oidc":
            raise AuthError("OIDC is not configured")
        return self._oidc(token)

    def _oidc(self, token: str) -> Principal:
        try:
            key = self.jwks.key_for(token)
            claims = jwt.decode(token, key=key, algorithms=self.algorithms, audience=self.s.oidc_audience,
                                issuer=self.s.oidc_issuer, leeway=self.s.oidc_leeway_s,
                                options={"require": ["exp", "iat", "iss", "sub"]})
        except AuthError:
            raise
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid token: {e}") from e
        tclaim = claims.get(self.s.tenant_claim) if self.s.tenant_claim else None
        roles, admin = parse_roles(_claim(claims, self.s.roles_claim), tclaim)
        attrs = _claim(claims, self.s.attributes_claim) or {}
        return Principal(subject=str(claims["sub"]),
                         name=str(claims.get("preferred_username") or claims.get("email") or claims["sub"]),
                         roles=roles, attributes=attrs if isinstance(attrs, dict) else {}, platform_admin=admin)

    def _service(self, token: str) -> Principal:
        sa = self.store.service_account_by_token(token)
        if sa is None:
            raise AuthError("unknown or revoked service-account token")
        roles = {r for r in sa["roles"] if r in ROLES and r != "tenant_admin"}
        return Principal(subject=sa["id"], name=sa["name"], kind="service", roles={sa["tenant_id"]: roles})

    def _dev(self, header: str) -> Principal:
        # "alice; acme:engineer,acme:consumer; clearance=confidential"  (tests and local demos only)
        parts = [p.strip() for p in header.split(";")]
        roles, admin = parse_roles([r for r in (parts[1].split(",") if len(parts) > 1 else []) if r])
        attrs = dict(a.split("=", 1) for a in (parts[2].split(",") if len(parts) > 2 else []) if "=" in a)
        return Principal(subject=parts[0], name=parts[0], kind="user", roles=roles, attributes=attrs,
                         platform_admin=admin)


def _claim(claims: dict, path: str):
    """A claim by dotted path (Keycloak puts client roles at resource_access.<client>.roles)."""
    cur: Any = claims
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

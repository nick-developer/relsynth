# SPDX-License-Identifier: Apache-2.0
"""OIDC token checks (RFC 8725), role parsing, service accounts, RBAC table."""
from __future__ import annotations

import time

import jwt
import pytest

from relsynth.service.auth import PERMISSIONS, ROLES, Authenticator, AuthError, Principal, parse_roles

from .conftest import token


@pytest.fixture
def oidc(settings, store, rsa_jwks):
    settings.auth_mode = "oidc"
    settings.oidc_issuer = "https://idp.example"
    settings.oidc_jwks_file = str(rsa_jwks[1])
    return Authenticator(settings, store), rsa_jwks[0]


def test_valid_token_gives_roles_per_tenant(oidc):
    a, key = oidc
    p = a.authenticate("Bearer " + token(key, relsynth_roles=["acme:engineer", "acme:consumer", "beta:data_owner",
                                                                 "evil:root", "../x:engineer"],
                                         preferred_username="alice"))
    assert p.name == "alice" and p.roles == {"acme": {"engineer", "consumer"}, "beta": {"data_owner"}}
    assert p.has("acme", "run:create") and not p.has("acme", "approval:decide") and not p.has("evil", "run:read")


@pytest.mark.parametrize("bad", [
    {"aud": "other"}, {"iss": "https://evil.example"}, {"exp": int(time.time()) - 3600},
    {"nbf": int(time.time()) + 3600},
])
def test_bad_claims_are_refused(oidc, bad):
    a, key = oidc
    with pytest.raises(AuthError):
        a.authenticate("Bearer " + token(key, **bad))


def test_alg_none_and_hmac_confusion_are_refused(oidc, rsa_jwks):
    a, key = oidc
    now = int(time.time())
    claims = {"iss": "https://idp.example", "aud": "relsynth", "sub": "x", "iat": now, "exp": now + 60}
    unsigned = jwt.encode(claims, None, algorithm="none")
    with pytest.raises(AuthError):
        a.authenticate("Bearer " + unsigned)
    # HS256 signed with the public key as the secret (the classic key-confusion attack)
    from cryptography.hazmat.primitives import serialization
    pub = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    import hashlib
    import hmac
    import base64
    import json as _j
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
    head = b64(_j.dumps({"alg": "HS256", "kid": "k1", "typ": "JWT"}).encode())
    body = b64(_j.dumps(claims).encode())
    sig = b64(hmac.new(pub, head + b"." + body, hashlib.sha256).digest())
    with pytest.raises(AuthError):
        a.authenticate("Bearer " + (head + b"." + body + b"." + sig).decode())


def test_hmac_algorithms_cannot_be_configured(settings, store):
    settings.auth_mode, settings.oidc_issuer, settings.oidc_algorithms = "oidc", "https://i", "RS256,HS256"
    with pytest.raises(ValueError):
        Authenticator(settings, store)


def test_tampered_token_is_refused(oidc):
    a, key = oidc
    t = token(key, relsynth_roles=["acme:consumer"])
    h, b, s = t.split(".")
    import base64
    import json
    claims = json.loads(base64.urlsafe_b64decode(b + "=="))
    claims["relsynth_roles"] = ["acme:data_owner"]
    b2 = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    with pytest.raises(AuthError):
        a.authenticate(f"Bearer {h}.{b2}.{s}")


def test_missing_token_and_dev_header_without_dev_mode(oidc):
    a, _ = oidc
    with pytest.raises(AuthError):
        a.authenticate(None, "mallory; acme:data_owner")


def test_keycloak_nested_claim_and_bare_roles(oidc, settings):
    a, key = oidc
    settings.roles_claim, settings.tenant_claim = "resource_access.relsynth.roles", "tenant"
    p = a.authenticate("Bearer " + token(key, tenant="acme",
                                         resource_access={"relsynth": {"roles": ["engineer", "*:platform_admin"]}}))
    assert p.roles == {"acme": {"engineer"}} and p.platform_admin


def test_service_account_token_lifecycle(settings, store):
    a = Authenticator(settings, store)
    row, tok = store.add_service_account("acme", "ci", ["engineer"], "user:alice")
    p = a.authenticate("Bearer " + tok)
    assert p.kind == "service" and p.roles == {"acme": {"engineer"}}
    assert "token_hash" not in row and tok not in str(store.service_accounts("acme"))
    with pytest.raises(AuthError):
        a.authenticate("Bearer " + tok[:-2] + "xx")
    store.revoke_service_account("acme", row["id"])
    with pytest.raises(AuthError):
        a.authenticate("Bearer " + tok)


def test_rbac_table_is_consistent():
    assert set().union(*PERMISSIONS.values()) <= set(ROLES)
    assert PERMISSIONS["approval:decide"] == {"data_owner", "privacy_officer"}
    assert "consumer" not in PERMISSIONS["privacy:read"] | PERMISSIONS["run:create"] | PERMISSIONS["spec:read"]


def test_abac_clearance_order():
    p = Principal("u", "u", attributes={"clearance": "confidential"})
    assert p.cleared_for({}) and p.cleared_for({"classification": "internal"})
    assert not p.cleared_for({"classification": "restricted"}) and not p.cleared_for({"classification": "weird"})
    assert parse_roles(["*:platform_admin"]) == ({}, True)

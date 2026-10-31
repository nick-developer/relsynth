# SPDX-License-Identifier: Apache-2.0
"""Run manifests, approvals, revocations and release tokens (design secs. 6, 9).

Each is an in-toto Statement v1 in a DSSE envelope, stored in the registry
and named in its log.  A manifest is never edited after it is signed:
approvals, rejections, revocations and releases are later statements whose
subject is the manifest's digest.

No data goes into these statements: hashes, versions, settings and privacy
figures only (design sec. 2, principle 1).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import platform
import sys
import uuid
from datetime import datetime, timezone

from ..errors import GovernanceError, ReleaseDenied
from .canon import canonical, sha256
from .policy import duration
from .registry import now
from .sign import envelope, payload, verify_envelope

STATEMENT = "https://in-toto.io/Statement/v1"
BASE = "https://github.com/nick-developer/relsynth"
MANIFEST_TYPE = f"{BASE}/run-manifest/1"
APPROVAL_TYPE = f"{BASE}/approval/1"
REVOCATION_TYPE = f"{BASE}/revocation/1"
TOKEN_TYPE = f"{BASE}/release-token/1"


def new_run_id() -> str:
    """A ULID-like sortable id: 48-bit time plus 80 random bits, Crockford base32."""
    import secrets
    t = int(datetime.now(timezone.utc).timestamp() * 1000)
    n = (t << 80) | secrets.randbits(80)
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    out = []
    for _ in range(26):
        out.append(alphabet[n & 31])
        n >>= 5
    return "".join(reversed(out))


def table_bytes(df) -> bytes:
    """The canonical export of a synthetic table: CSV, header, LF line ends, UTF-8.
    A release package stores exactly these bytes, so a recipient can check
    each file against the signed manifest."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def subjects(tables: dict) -> list[dict]:
    return [{"name": t, "digest": {"sha256": sha256(table_bytes(df))}} for t, df in tables.items()]


def locator_hmac(registry, locator: str) -> str:
    """HMAC-SHA256 of a connection locator without credentials, keyed with the registry secret."""
    from ..data import mask_url
    key = registry.secret("locator_hmac")
    return hmac.new(key, mask_url(str(locator)).encode(), hashlib.sha256).hexdigest()


def schema_digest(spec: dict) -> str:
    """Hash of the table and column names and types: structure only."""
    def kind(cs):
        return str(cs.get("type", "")) if isinstance(cs, dict) else str(cs or "")
    tabs = {t: sorted((c, kind(cs)) for c, cs in (ts.get("columns") or {}).items())
            for t, ts in (spec.get("tables") or {}).items()}
    return sha256(canonical(tabs))


def _scrub(cmd: str) -> str:
    from ..data import mask_url
    import re
    return re.sub(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+", lambda m: mask_url(m.group(0)), cmd)


def software(plugins=(), accountant: str | None = None) -> dict:
    from .._version import __version__
    out: dict = {"relsynth": __version__, "python": sys.version.split()[0],
           "platform": f"{sys.platform}-{platform.machine()}"}
    if plugins:
        out["plugins"] = list(plugins)
    if accountant:
        out["accountant"] = accountant
    return out


def build_manifest(*, outputs: list[dict], run_id: str, started: str, finished: str, operator, command: str,
                   source: dict, spec_digest: str, ir_digest: str, ir_version: str, privacy: dict,
                   reports: list[dict], verdicts: dict, policy_info: dict, release: dict, chain_prev: str,
                   seeds: dict | None = None, public_facts_only: bool | None = None, plugins=(),
                   accountant: str | None = None) -> dict:
    """An in-toto statement for one run (schema run-manifest-1)."""
    spec: dict = {"digest": {"sha256": spec_digest}, "ir_digest": {"sha256": ir_digest}, "ir_version": ir_version}
    if public_facts_only is not None:
        spec["public_facts_only"] = bool(public_facts_only)
    pred = {"run": {"id": run_id, "started": started, "finished": finished,
                    "operator": {"name": operator.name, "key": operator.id}, "command": _scrub(command)},
            "source": source, "spec": spec, "software": software(plugins, accountant),
            "privacy": privacy, "reports": reports, "verdicts": verdicts, "policy": policy_info,
            "release": release, "chain": {"prev": chain_prev}}
    if seeds:
        pred["seeds"] = seeds
    if not outputs:
        raise GovernanceError("a manifest needs at least one output table")
    return {"_type": STATEMENT, "subject": outputs, "predicateType": MANIFEST_TYPE, "predicate": pred}


def check_manifest(st: dict) -> list[str]:
    """Problems of a manifest statement against the JSON Schema."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return []
    from importlib import resources
    sch = json.loads(resources.files("relsynth.governance").joinpath("schema/run-manifest-1.schema.json").read_text())
    return [f"{'/'.join(map(str, e.absolute_path)) or '(top)'}: {e.message}"
            for e in jsonschema.Draft202012Validator(sch).iter_errors(st)]


def record_manifest(registry, st: dict, operator) -> str:
    """Sign and store a manifest; returns its digest (the digest of the signed statement bytes)."""
    probs = check_manifest(st)
    if probs:
        raise GovernanceError("the run manifest is invalid:\n  - " + "\n  - ".join(probs))
    env = envelope(st, operator)
    digest = sha256(canonical(st))
    registry.put_json(env)
    p = st["predicate"]
    registry.append({"type": "manifest", "manifest": digest, "envelope": sha256(canonical(env)), "run": p["run"]["id"],
                     "domain": p["source"].get("domain", ""), "context": p["release"]["context"],
                     "mode": p["privacy"]["mode"]})
    return digest


# --------------------------------------------------------------------- lookup
def find_manifest(registry, run_or_digest: str) -> tuple[dict, dict, dict]:
    """(log record, envelope, statement) of a run id, a manifest digest or a unique prefix of either."""
    hits = [r for r in registry.entries("manifest")
            if r["entry"]["run"] == run_or_digest or r["entry"]["manifest"].startswith(run_or_digest)]
    if not hits:
        raise GovernanceError(f"no run or manifest {run_or_digest!r} in the registry")
    if len({h["entry"]["manifest"] for h in hits}) > 1:
        raise GovernanceError(f"{run_or_digest!r} matches several manifests; give more characters")
    rec = hits[-1]
    env = registry.get_json(rec["entry"]["envelope"])
    return rec, env, payload(env)


def statements_about(registry, manifest_digest: str, kind: str) -> list[tuple[dict, dict, dict]]:
    out = []
    for r in registry.entries(kind):
        if r["entry"].get("manifest") == manifest_digest:
            env = registry.get_json(r["entry"]["envelope"])
            out.append((r, env, payload(env)))
    return out


# --------------------------------------------------------------------- approvals
def approve(registry, policy, run: str, context: str, signer, *, reject: bool = False, reason: str = "") -> int:
    """Sign an approval (or a rejection) of one manifest for one context."""
    rec, _, st = find_manifest(registry, run)
    trusted = registry.trusted()
    if signer.id not in trusted:
        raise GovernanceError(f"key {signer.id} is not in the registry's trusted keys")
    ctx = policy.context(context)
    roles = trusted[signer.id]["roles"]
    if not set(roles) & set(ctx["approval"]["approver_roles"]):
        raise GovernanceError(f"{signer.name} has roles {roles}; context {context!r} needs one of "
                              f"{ctx['approval']['approver_roles']}")
    p = st["predicate"]
    sod = (policy.doc.get("approvals") or {}).get("separation_of_duties", True)
    if sod and signer.id == p["run"]["operator"]["key"]:
        raise GovernanceError("separation of duties: the operator of a run may not approve it")
    verdict = (p.get("verdicts") or {}).get(context, "missing")
    if not reject and verdict in ("fail", "blocked", "missing"):
        raise ReleaseDenied(f"the privacy verdict for context {context!r} is {verdict!r}; it cannot be approved")
    seen: dict = {"verdict": verdict, "mode": p["privacy"]["mode"]}
    if p["privacy"]["mode"] == "dp":
        seen["epsilon"] = p["privacy"]["stated"]["epsilon"]
        seen["delta"] = p["privacy"]["stated"]["delta"]
    pred: dict = {"manifest": rec["entry"]["manifest"], "run": p["run"]["id"], "context": context,
            "decision": "reject" if reject else "approve", "approver": {"name": signer.name, "key": signer.id},
            "saw": seen, "time": now(), "policy": policy.digest}
    if reason:
        pred["reason"] = reason
    subj = {"name": f"manifest:{p['run']['id']}", "digest": {"sha256": rec["entry"]["manifest"]}}
    st2 = {"_type": STATEMENT, "subject": [subj],
           "predicateType": APPROVAL_TYPE, "predicate": pred}
    env = envelope(st2, signer)
    d = registry.put_json(env)
    return registry.append({"type": "approval", "manifest": rec["entry"]["manifest"], "envelope": d,
                            "context": context, "decision": pred["decision"]})


def verified_approvals(registry, policy, manifest_digest: str, context: str, at: datetime | None = None) -> list[dict]:
    """Approvals of a manifest for a context whose signatures verify, as the policy input
    expects them: roles come from the trusted-keys list, not from the statement."""
    trusted = registry.trusted()
    keys = {i: v["public"] for i, v in trusted.items()}
    exp = (policy.doc.get("approvals") or {}).get("expires_after")
    at = at or datetime.now(timezone.utc)
    used = {r["entry"].get("approval") for r in registry.entries("release")}
    out = []
    for rec, env, st in statements_about(registry, manifest_digest, "approval"):
        p = st["predicate"]
        if p.get("context") != context:
            continue
        signed = verify_envelope(env, keys)
        if p["approver"]["key"] not in signed:
            continue
        t = datetime.fromisoformat(p["time"].replace("Z", "+00:00"))
        expired = bool(exp) and at - t > duration(str(exp)) and rec["entry"]["envelope"] not in used
        out.append({"key": p["approver"]["key"], "approver": p["approver"]["name"], "decision": p["decision"],
                    "manifest": p["manifest"], "roles": trusted[p["approver"]["key"]]["roles"], "expired": expired,
                    "reason": p.get("reason", ""), "envelope": rec["entry"]["envelope"]})
    return out


def revoke(registry, run: str, signer, reason: str) -> int:
    rec, _, st = find_manifest(registry, run)
    if signer.id not in registry.trusted():
        raise GovernanceError(f"key {signer.id} is not in the registry's trusted keys")
    pred = {"manifest": rec["entry"]["manifest"], "run": st["predicate"]["run"]["id"], "reason": reason,
            "by": {"name": signer.name, "key": signer.id}, "time": now(),
            "note": "revoking a release never refunds privacy budget: the data has already left"}
    st2 = {"_type": STATEMENT, "subject": [{"name": f"manifest:{pred['run']}", "digest": {"sha256": pred['manifest']}}],
           "predicateType": REVOCATION_TYPE, "predicate": pred}
    d = registry.put_json(envelope(st2, signer))
    return registry.append({"type": "revocation", "manifest": pred["manifest"], "envelope": d})


def revoked(registry, manifest_digest: str) -> bool:
    return any(r["entry"].get("manifest") == manifest_digest for r in registry.entries("revocation"))


# --------------------------------------------------------------------- release tokens
def issue_token(registry, manifest_digest: str, run: str, context: str, target: str, approvals: list[dict],
                package: list[str], signer, decision: dict) -> tuple[dict, str]:
    pred = {"manifest": manifest_digest, "run": run, "context": context, "target": target,
            "approvals": sorted(a["envelope"] for a in approvals if a["decision"] == "approve" and not a["expired"]),
            "package": sorted(package), "issued": now(), "nonce": uuid.uuid4().hex,
            "decision": {"allow": decision["allow"], "engine": decision.get("engine", "")}}
    st = {"_type": STATEMENT, "subject": [{"name": f"manifest:{run}", "digest": {"sha256": manifest_digest}}],
          "predicateType": TOKEN_TYPE, "predicate": pred}
    env = envelope(st, signer)
    d = registry.put_json(env)
    registry.append({"type": "release", "manifest": manifest_digest, "envelope": d, "context": context,
                     "target": target, "approval": ",".join(pred["approvals"])})
    return env, d


def check_token(registry, token_env: dict, context: str, target: str, package: list[str] | None = None) -> dict:
    """What a target writer of an external context checks before it opens a
    connection: a token signed by a trusted key, recorded in the registry, for
    this context and target, whose manifest is not revoked."""
    keys = registry.public_keys()
    if not verify_envelope(token_env, keys):
        raise ReleaseDenied("the release token is not signed by a trusted key")
    st = payload(token_env)
    if st.get("predicateType") != TOKEN_TYPE:
        raise ReleaseDenied("not a release token")
    p = st["predicate"]
    d = sha256(canonical(token_env))
    if not any(r["entry"]["envelope"] == d for r in registry.entries("release")):
        raise ReleaseDenied("the release token is not recorded in the registry")
    if p["context"] != context or p["target"] != target:
        raise ReleaseDenied(f"the token releases to {p['context']}/{p['target']}, not {context}/{target}")
    if revoked(registry, p["manifest"]):
        raise ReleaseDenied("the run of this token has been revoked")
    if package is not None and sorted(package) != p["package"]:
        raise ReleaseDenied("the files to write differ from the files the token releases")
    return p

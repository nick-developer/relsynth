# SPDX-License-Identifier: Apache-2.0
"""Governance: signed run manifests, the registry, the organisation's privacy
budget, release policy and approvals (roadmap T08; design in
docs/governance/design.md).

    import relsynth.governance as gov
    reg = gov.LocalRegistry.create("registry", owner)          # owner: gov.Signer
    gov.policy.sign(gov.policy.load("policy.yaml"), owner, reg)
    run = gov.record_run(reg, spec, synthetic, context="internal", target="dev-sandbox",
                         source_id="dwh-prod", operator=analyst, privacy_report=rep)
    gov.approve(reg, gov.current_policy(reg), run.run_id, "internal", dpo)
    gov.release(reg, run.run_id, "internal", "dev-sandbox", analyst, out="package/")
    assert not gov.verify_registry(reg)

Nothing here reads real data: a manifest holds hashes, versions, settings
and privacy figures only.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import GovernanceError, ReleaseDenied
from . import policy
from .accounting import dec, epsilon_of, ledger_epsilon, rho_of
from .canon import canonical, digest, sha256
from .decide import decide
from .ledger import Ledger
from .manifest import (approve, build_manifest, check_token, find_manifest, issue_token, locator_hmac, new_run_id,
                       record_manifest, revoke, revoked, schema_digest, statements_about, subjects, table_bytes,
                       verified_approvals)
from .policy import Policy
from .policy import current as current_policy
from .registry import LocalRegistry, now, open_registry
from .sign import Signer, envelope, payload, verify_envelope
from .verify import verify_package, verify_registry

__all__ = ["LocalRegistry", "open_registry", "Signer", "Policy", "Ledger", "policy", "current_policy", "decide",
           "record_run", "approve", "revoke", "release", "verify_registry", "verify_package", "show",
           "contexts_from_policy", "check_token", "RecordedRun", "canonical", "digest", "envelope", "payload",
           "verify_envelope", "ledger_epsilon", "epsilon_of", "rho_of", "new_run_id"]


def contexts_from_policy(pol: Policy) -> dict:
    """The organisation policy's release contexts in the form of privacy.yaml
    `contexts` (RFC-T06-03 item 1): the privacy assessment then judges each
    release against the organisation's limits."""
    out = {}
    for name, c in pol.contexts.items():
        out[name] = {"description": c.get("description", ""), "external": bool(c["external"]),
                     "attacker_knows": c.get("attacker_knows", "all"),
                     "per_person": "gate" if c["external"] else "info", "sign_off": True,
                     "thresholds": dict(c["limits"])}
    return out


@dataclass
class RecordedRun:
    run_id: str
    manifest: str                      # digest of the signed statement
    seq: int
    decisions: dict = field(default_factory=dict)
    statement: dict = field(default_factory=dict)


def _report_entry(reg, kind: str, schema: str, obj, external_safe: bool) -> dict:
    d = reg.put_blob(obj if isinstance(obj, (bytes, bytearray)) else canonical(_jsonable(obj)))
    return {"kind": kind, "schema": schema, "digest": {"sha256": d}, "external_safe": bool(external_safe)}


def _jsonable(o):
    from ..diagnostics import _jsonable as j
    return json.loads(json.dumps(j(o), default=str))


def _policy_input(reg, pol: Policy, st: dict, mdigest: str, context: str, target: str, package=(),
                  run_rho: float | None = None) -> dict:
    p = st["predicate"]
    inp = {"policy": pol.doc, "manifest": p, "manifest_digest": mdigest, "ledger": {},
           "privacy_checks": [], "quality": {}, "approvals": [],
           "request": {"action": "release", "context": context, "target": target, "package": list(package)}}
    dom = p["source"].get("domain")
    if p["privacy"]["mode"] == "dp" and dom in pol.domains:
        rho = float(p["privacy"]["run_total"]["rho"]) if run_rho is None else run_rho
        inp["ledger"] = Ledger(reg, pol).policy_input(dom, rho, run=p["run"]["id"])
    for r in p.get("reports", []):
        try:
            rep = json.loads(reg.get_blob(r["digest"]["sha256"]))
        except GovernanceError:  # nosec B112 - a missing report is reported by verify; the decision goes on without it
            continue
        if r["kind"] == "privacy":
            inp["privacy_checks"] = [{k: c.get(k) for k in ("criterion", "attack", "value", "limit", "status")}
                                     for c in (rep.get("contexts", {}).get(context, {}).get("checks") or [])]
        elif r["kind"] == "quality" and "score" in rep:
            inp["quality"] = {"pass_rate": rep["score"]}
    if mdigest:
        inp["approvals"] = [{k: v for k, v in a.items() if k != "envelope"}
                            for a in verified_approvals(reg, pol, mdigest, context)]
    return inp


def record_run(reg, spec: dict, outputs: dict, *, context: str, target: str, operator: Signer, source_id: str,
               locator: str = "", command: str = "", started: str | None = None, privacy_report: dict | None = None,
               scorecard: dict | None = None, privacy: dict | None = None, seeds: dict | None = None,
               dp_statement: dict | None = None, public_facts_only: bool | None = None, accountant: str | None = None,
               store_outputs: bool = True, pol: Policy | None = None, agents: list[dict] | None = None) -> RecordedRun:
    """Write the signed manifest of a finished run, record a non-DP release in the
    domain's ledger, evaluate the policy for the intended context and checkpoint
    the registry.  `privacy` is the DP engine's privacy section (mode dp, charges,
    ledger figures); without it the run is recorded as `mode: none`."""
    pol = pol or current_policy(reg)
    pol.context(context)
    domain = pol.domain_of_source(source_id) or ""
    run_id = (privacy or {}).get("_run_id") or new_run_id()
    started = started or now()
    outs = subjects(outputs)
    if store_outputs:
        for df in outputs.values():
            reg.put_blob(table_bytes(df))
    reports = []
    if scorecard is not None:
        reports.append(_report_entry(reg, "quality", "relsynth.scorecard/1", scorecard, external_safe=False))
    verdicts = {}
    if privacy_report is not None:
        from ..privacy.report import to_json
        reports.append(_report_entry(reg, "privacy", privacy_report.get("schema", "relsynth.privacy-report/1"),
                                     to_json(privacy_report).encode("utf-8"), external_safe=False))
        verdicts = {k: v["verdict"] for k, v in privacy_report["contexts"].items()}
    if dp_statement is not None:
        reports.append(_report_entry(reg, "dp_statement", "relsynth.dp-statement/1", dp_statement, external_safe=True))
    priv = {k: v for k, v in (privacy or {"mode": "none"}).items() if not k.startswith("_")}
    from ..ir import IR_VERSION
    spec_clean = {k: v for k, v in spec.items() if not str(k).startswith("_")}
    sd = sha256(canonical(_jsonable(spec_clean)))
    try:
        from ..ir import lift
        ird = sha256(canonical(_jsonable(lift(spec_clean).to_dict())))
    except Exception:  # noqa: BLE001 - a spec the IR cannot lift is still recorded; its IR digest is the spec's
        ird = sd
    src = {"id": source_id, "domain": domain,
           "fingerprint": {"locator_hmac": locator_hmac(reg, locator or source_id), "schema_sha256": schema_digest(spec)}}
    head = reg.head()[1]
    pinfo: dict = {"digest": {"sha256": pol.digest}, "version": pol.version, "engine": "", "decisions": {}}
    st = build_manifest(outputs=outs, run_id=run_id, started=started, finished=now(), operator=operator,
                        command=command or "relsynth (API)", source=src, spec_digest=sd, ir_digest=ird,
                        ir_version=IR_VERSION, privacy=priv, reports=reports, verdicts=verdicts, policy_info=pinfo,
                        release={"context": context, "target": target}, chain_prev=head, seeds=seeds,
                        public_facts_only=public_facts_only, accountant=accountant, agents=agents)
    # the decision before approval: what the run still needs (it is final only at `release`)
    dec_ = decide(_policy_input(reg, pol, st, "", context, target), pol.engine, pol.doc.get("rego") or ())
    pinfo["engine"] = dec_["engine"]
    pinfo["decisions"][context] = {"allow": dec_["allow"], "approval_required": dec_["approval_required"],
                                   "reasons": dec_["reasons"]}
    if priv["mode"] != "dp" and domain:
        Ledger(reg, pol).nondp(domain, run_id)
    md = record_manifest(reg, st, operator)
    reg.checkpoint(operator)
    seq = reg.head()[0]
    return RecordedRun(run_id=run_id, manifest=md, seq=seq, decisions=pinfo["decisions"], statement=st)


def release(reg, run: str, context: str, target: str, operator: Signer, out=None) -> dict:
    """Evaluate the policy for a release of a recorded run.  On allow, issue a
    signed release token and write the release package to `out` (synthetic data,
    manifest, approvals, token, checkpoint, DP statement); on deny, record the
    refusal and raise ReleaseDenied with the reasons."""
    pol = current_policy(reg)
    rec, menv, st = find_manifest(reg, run)
    md = rec["entry"]["manifest"]
    if revoked(reg, md):
        raise ReleaseDenied(f"run {st['predicate']['run']['id']} has been revoked")
    package = [s["digest"]["sha256"] for s in st["subject"]]
    package += [r["digest"]["sha256"] for r in st["predicate"].get("reports", []) if r["kind"] == "dp_statement"]
    inp = _policy_input(reg, pol, st, md, context, target, package)
    res = decide(inp, pol.engine, pol.doc.get("rego") or ())
    reg.append({"type": "decision", "manifest": md, "context": context, "target": target, "allow": res["allow"],
                "reasons": res["reasons"], "policy": pol.digest, "engine": res["engine"]})
    if not res["allow"]:
        raise ReleaseDenied(f"the policy refuses to release run {st['predicate']['run']['id']} to {context}/{target}",
                            res["reasons"])
    appr = verified_approvals(reg, pol, md, context)
    tok, tok_d = issue_token(reg, md, st["predicate"]["run"]["id"], context, target, appr, package, operator, res)
    cp = reg.checkpoint(operator)
    if out is not None:
        _write_package(reg, Path(out), st, menv, appr, tok, cp)
    return {"allow": True, "token": tok_d, "reasons": [], "approvals": len(appr), "package": str(out) if out else None,
            "engine": res["engine"]}


def _write_package(reg, f: Path, st, menv, appr, tok, cp) -> None:
    if f.exists() and any(f.iterdir()):
        raise GovernanceError(f"{f} is not empty")
    (f / "data").mkdir(parents=True, exist_ok=True)
    (f / "approvals").mkdir(exist_ok=True)
    for s in st["subject"]:
        (f / "data" / f"{s['name']}.csv").write_bytes(reg.get_blob(s["digest"]["sha256"]))
    w = lambda name, obj: (f / name).write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    w("manifest.dsse", menv)
    w("release-token.dsse", tok)
    w("checkpoint.dsse", cp)
    w("registry.json", {"id": reg.id, "root_key": reg.config["root_key"]})
    shutil.copy(reg.root / "keys" / "trusted.dsse", f / "trusted-keys.dsse")
    for a in appr:
        if a["decision"] == "approve" and not a["expired"]:
            w(f"approvals/{a['envelope'][:16]}.dsse", reg.get_json(a["envelope"]))
    for r in st["predicate"].get("reports", []):
        if r["kind"] == "dp_statement":
            (f / "dp_statement.json").write_bytes(reg.get_blob(r["digest"]["sha256"]))
    (f / "README.md").write_text(
        "# relsynth release package\n\nVerify it with `relsynth gov verify .` after checking that the root key id "
        f"`{reg.config['root_key']['id']}` in registry.json is the one your contact gave you.\n\n"
        "- data/: the synthetic tables, byte for byte as signed in manifest.dsse\n"
        "- manifest.dsse: the signed run manifest (no real data in it)\n"
        "- approvals/: the signed approvals of this release\n"
        "- release-token.dsse: the signed release decision\n"
        "- dp_statement.json: the differential privacy guarantee (DP releases only)\n", encoding="utf-8")


def show(reg, run: str) -> dict:
    """A run's manifest, its approvals, revocation, releases and decisions."""
    rec, _, st = find_manifest(reg, run)
    md = rec["entry"]["manifest"]
    appr = [s[2]["predicate"] for s in statements_about(reg, md, "approval")]
    return {"seq": rec["seq"], "manifest": md, "statement": st, "approvals": appr, "revoked": revoked(reg, md),
            "decisions": [r["entry"] for r in reg.entries("decision") if r["entry"]["manifest"] == md],
            "releases": [r["entry"] for r in reg.entries("release") if r["entry"]["manifest"] == md]}


def runs(reg, domain: str | None = None, state: str | None = None) -> list[dict]:
    out = []
    rel = {r["entry"]["manifest"] for r in reg.entries("release")}
    rev = {r["entry"]["manifest"] for r in reg.entries("revocation")}
    appr: dict = {}
    for r in reg.entries("approval"):
        appr.setdefault(r["entry"]["manifest"], []).append(r["entry"]["decision"])
    for r in reg.entries("manifest"):
        e = r["entry"]
        if domain and e["domain"] != domain:
            continue
        m = e["manifest"]
        s = ("revoked" if m in rev else "released" if m in rel else "rejected" if "reject" in appr.get(m, [])
             else "approved" if "approve" in appr.get(m, []) else "assessed")
        if state and s != state:
            continue
        out.append({"run": e["run"], "manifest": m, "domain": e["domain"], "context": e["context"], "mode": e["mode"],
                    "state": s, "time": e["time"], "seq": r["seq"]})
    return out


__all__ += ["runs", "dec"]

# SPDX-License-Identifier: Apache-2.0
"""relsynth.governance: canonical JSON, DSSE signing, the hash-chained
registry, the organisation policy, the budget ledger, release decisions,
approvals, release packages and the `relsynth gov` CLI (design
docs/governance/design.md, acceptance criteria of T08)."""
from __future__ import annotations

import base64
import copy
import json
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("cryptography")
pytest.importorskip("jsonschema")

from relsynth import governance as G  # noqa: E402
from relsynth.errors import GovernanceError, PrivacyError, ReleaseDenied  # noqa: E402
from relsynth.governance.canon import canonical, es_number  # noqa: E402
from relsynth.governance.decide import builtin, decide, invariants  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "docs" / "governance" / "examples" / "org-policy.yaml"


# ---------------------------------------------------------------- canonical JSON
@pytest.mark.parametrize("x,want", [
    (0.0, "0"), (-0.0, "0"), (1.0, "1"), (100.0, "100"), (0.5, "0.5"), (1e-9, "1e-9"), (1.5e21, "1.5e+21"),
    (1e21, "1e+21"), (1e20, "100000000000000000000"), (0.000001, "0.000001"), (1e-7, "1e-7"),
    (123.456, "123.456"), (-2.5e-8, "-2.5e-8"), (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"),
    (333333333.33333329, "333333333.3333333"), (9007199254740992, "9007199254740992")])
def test_es_number_matches_ecmascript(x, want):
    assert es_number(x) == want           # RFC 8785 appendix B and ECMA-262 Number::toString


def test_canonical_json_rfc8785():
    # RFC 8785 sec. 3.2.3: keys sorted by UTF-16 code units, minimal escapes
    obj = {"€": "Euro Sign", "\r": "Carriage Return", "דּ": "Hebrew Letter Dalet With Dagesh",
           "1": "One", "\U0001f600": "Emoji: Grinning Face", "\u0080": "Control", "ö": "Latin Small Letter O With Diaeresis"}
    keys = [k for k in json.loads(canonical(obj))]
    assert keys == ["\r", "1", "\u0080", "ö", "€", "\U0001f600", "דּ"]
    assert canonical({"b": [1, True, None, "a\n\"\u0001"], "a": {}}) == b'{"a":{},"b":[1,true,null,"a\\n\\"\\u0001"]}'
    with pytest.raises(ValueError):
        canonical(float("nan"))


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def people(tmp_path):
    owner = G.Signer.generate("Dana DPO")
    analyst = G.Signer.generate("Ana Analyst")
    dpo2 = G.Signer.generate("Second DPO")
    reg = G.LocalRegistry.create(tmp_path / "registry", owner, "dpo")
    reg.trust(owner, analyst.public, "Ana Analyst", ["operator"])
    reg.trust(owner, dpo2.public, "Second DPO", ["dpo"])
    pol = G.policy.load(EXAMPLE)
    G.policy.sign(pol, owner, reg)
    return {"reg": reg, "owner": owner, "analyst": analyst, "dpo2": dpo2, "policy": pol, "tmp": tmp_path}


def _tables():
    return {"Customer": pd.DataFrame({"CustomerId": [1, 2, 3], "Country": ["GR", "DE", "FR"]}),
            "Invoice": pd.DataFrame({"InvoiceId": [1, 2], "CustomerId": [1, 3], "Total": [9.9, 1.5]})}


def _spec():
    return {"name": "shop", "source": {"kind": "csv", "path": "real/"},
            "tables": {"Customer": {"columns": {"CustomerId": "id", "Country": "category"}, "primary_key": "CustomerId"},
                       "Invoice": {"columns": {"InvoiceId": "id", "CustomerId": "id", "Total": "float"},
                                   "primary_key": "InvoiceId"}}}


def _privacy_report(verdict="pass", contexts=("internal", "partner", "public")):
    return {"schema": "relsynth.privacy-report/1",
            "contexts": {c: {"verdict": verdict, "checks": [{"criterion": "membership", "attack": "domias_auc",
                                                             "value": 0.52, "ci": [0.5, 0.54], "limit": 0.6,
                                                             "status": "pass"}]} for c in contexts}}


def _record(p, context="internal", target="dev-sandbox", verdict="pass", privacy=None, **kw):
    return G.record_run(p["reg"], _spec(), _tables(), context=context, target=target, operator=p["analyst"],
                        source_id="dwh-prod", locator="postgresql://ana:secret@db.internal/dw",
                        privacy_report=_privacy_report(verdict), scorecard={"passed": 98, "total": 100, "score": 0.98},
                        privacy=privacy, **kw)


# ---------------------------------------------------------------- signing and registry
def test_dsse_signature_binds_payload_type_and_bytes():
    s = G.Signer.generate("x")
    env = G.envelope({"a": 1}, s)
    keys = {s.id: s.public}
    assert G.verify_envelope(env, keys) == [s.id]
    other = dict(env, payloadType="application/json")
    assert G.verify_envelope(other, keys) == []
    edited = dict(env, payload=base64.b64encode(canonical({"a": 2})).decode())
    assert G.verify_envelope(edited, keys) == []
    noncanon = dict(env, payload=base64.b64encode(b'{ "a": 1 }').decode())
    assert G.verify_envelope(noncanon, keys) == []


def test_key_file_roundtrip_and_mode(tmp_path):
    s = G.Signer.generate("Ana")
    p = s.save(tmp_path / "ana.key")
    if sys.platform != "win32":
        assert (p.stat().st_mode & 0o777) == 0o600
    t = G.Signer.load(p)
    assert t.id == s.id and t.name == "Ana"
    assert t.id.startswith("ed25519:SHA256:")


def test_only_root_key_signs_trusted_keys(people):
    with pytest.raises(GovernanceError):
        people["reg"].trust(people["analyst"], people["analyst"].public, "self", ["dpo"])
    assert people["reg"].trusted()[people["analyst"].id]["roles"] == ["operator"]


def test_policy_must_be_signed_by_privacy_owner(people):
    with pytest.raises(GovernanceError, match="privacy owner"):
        G.policy.sign(people["policy"], people["analyst"], people["reg"])
    assert G.current_policy(people["reg"]).digest == people["policy"].digest


TAMPER = ["edit", "delete", "insert", "swap", "truncate"]


@pytest.mark.parametrize("how", TAMPER)
def test_tampered_registry_is_detected(people, how):
    rr = _record(people)
    reg = people["reg"]
    assert G.verify_registry(reg) == []
    log = reg.root / "log.jsonl"
    lines = log.read_text().splitlines()
    j = len(lines) // 2
    if how == "edit":
        rec = json.loads(lines[j])
        rec["entry"]["time"] = "2000-01-01T00:00:00Z"
        lines[j] = json.dumps(rec)
    elif how == "delete":
        del lines[j]
    elif how == "insert":
        lines.insert(j, lines[j])
    elif how == "swap":
        lines[j], lines[j + 1] = lines[j + 1], lines[j]
    else:
        lines = lines[:j]
    log.write_text("\n".join(lines) + "\n")
    probs = G.verify_registry(reg)
    assert probs, how
    _ = rr


def test_tampered_report_or_manifest_object_is_detected(people):
    rr = _record(people)
    reg = people["reg"]
    rep = [r for r in rr.statement["predicate"]["reports"] if r["kind"] == "privacy"][0]
    p = reg._obj(rep["digest"]["sha256"])
    p.write_bytes(p.read_bytes().replace(b'"pass"', b'"PASS"'))
    assert any("privacy report does not match" in x for x in G.verify_registry(reg))


def test_resigned_manifest_by_untrusted_key_is_detected(people):
    rr = _record(people)
    reg = people["reg"]
    rec = [r for r in reg.entries("manifest")][-1]
    mallory = G.Signer.generate("Mallory")
    env = G.envelope(rr.statement, mallory)
    reg._obj(rec["entry"]["envelope"]).write_text(json.dumps(env))
    probs = G.verify_registry(reg)
    assert any("does not match its digest" in x or "no valid signature" in x for x in probs)


# ---------------------------------------------------------------- policy and caps
def test_policy_without_budget_is_refused_with_guidance():
    doc = yaml.safe_load(G.policy.skeleton())
    probs = G.policy.validate(doc)
    assert probs and "the organisation must set epsilon and delta" in probs[0]
    assert "NIST SP 800-226" in probs[0]


def test_policy_refuses_external_without_dp_and_large_delta():
    doc = yaml.safe_load(EXAMPLE.read_text())
    bad = copy.deepcopy(doc)
    bad["contexts"]["partner"]["dp_required"] = False
    assert G.policy.validate(bad)
    bad = copy.deepcopy(doc)
    bad["domains"]["customers"]["budget"]["delta"] = 1e-4
    assert G.policy.validate(bad)
    bad = copy.deepcopy(doc)
    bad["domains"]["customers"]["budget"]["per_release"]["epsilon_max"] = 7.0
    assert any("above the lifetime cap" in x for x in G.policy.validate(bad))


def test_caps_convert_epsilon_to_rho_conservatively(people):
    c = people["policy"].caps("customers")
    assert G.epsilon_of(c.lifetime_rho, c.delta) <= c.lifetime_epsilon
    assert G.epsilon_of(c.lifetime_rho * 1.001, c.delta) > c.lifetime_epsilon
    # the table in design sec. 5.2 (delta = 1e-8)
    for eps, rho in [(0.5, 0.0046), (1, 0.017), (2, 0.064), (4, 0.23), (8, 0.82)]:
        assert abs(G.rho_of(eps, 1e-8) - rho) / rho < 0.06


# ---------------------------------------------------------------- ledger
def test_ledger_reserve_touch_settle_refund(people):
    led = G.Ledger(people["reg"], people["policy"])
    caps = people["policy"].caps("customers")
    led.reserve("customers", "r1", 0.03)
    led.touch("customers", "r1")
    with pytest.raises(PrivacyError, match="cannot be refunded"):
        led.release("customers", "r1")
    with pytest.raises(PrivacyError, match="more than it reserved"):
        led.settle("customers", "r1", 0.031)
    led.settle("customers", "r1", 0.025)
    led.reserve("customers", "r2", 0.02)
    led.release("customers", "r2")                       # never read data: refunded
    s = led.summary("customers")
    assert abs(s["rho"] - 0.025) < 1e-12
    with pytest.raises(PrivacyError, match="per-release cap"):
        led.reserve("customers", "big", caps.per_release_rho * 1.01)
    assert G.verify_registry(people["reg"]) == []


def test_ledger_lifetime_cap_under_concurrency(people):
    led = G.Ledger(people["reg"], people["policy"])
    cap = people["policy"].caps("customers")
    each = cap.per_release_rho * 0.9
    ok, errs = [], []

    def go(i):
        try:
            led.reserve("customers", f"c{i}", each)
            ok.append(i)
        except PrivacyError as e:
            errs.append(str(e))
    th = [threading.Thread(target=go, args=(i,)) for i in range(24)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    total = led.summary("customers")["rho"]
    assert total <= cap.lifetime_rho + 1e-12
    assert len(ok) == int(cap.lifetime_rho // each) or len(ok) == int(cap.period_rho // each)
    assert errs and all("cap" in e for e in errs)


def test_nondp_runs_are_recorded(people):
    _record(people)
    s = G.Ledger(people["reg"], people["policy"]).summary("customers")
    assert len(s["nondp_releases"]) == 1 and s["rho"] == 0


# ---------------------------------------------------------------- decisions (acceptance: blocking with reasons)
def _dp_privacy(rho="0.05"):
    return {"mode": "dp", "unit": "Customer", "adjacency": "add_remove",
            "charges": [{"stage": "fit", "mechanism": "gaussian", "kind": "zcdp", "rho": rho}],
            "run_total": {"rho": rho, "delta": "0"}, "stated": {"epsilon": "1.9", "delta": "1e-9"},
            "ledger": {"seq": 0, "domain_rho": rho, "domain_delta": "0", "domain_epsilon": "1.9"}}


def test_partner_without_dp_is_refused(people):
    rr = _record(people, context="partner", target="partner-sftp")
    with pytest.raises(ReleaseDenied) as e:
        G.release(people["reg"], rr.run_id, "partner", "partner-sftp", people["analyst"])
    assert any("differential privacy is required, and this run has none" in r for r in e.value.reasons)
    assert any(r.startswith("invariant:") for r in e.value.reasons)


def test_failed_verdict_quality_and_target_are_refused(people):
    rr = _record(people, verdict="fail")
    with pytest.raises(ReleaseDenied, match="cannot be approved"):
        G.approve(people["reg"], people["policy"], rr.run_id, "internal", people["owner"])
    with pytest.raises(ReleaseDenied) as e:
        G.release(people["reg"], rr.run_id, "internal", "prod-db", people["analyst"])
    txt = "\n".join(e.value.reasons)
    assert 'the privacy verdict for context "internal" is "fail"' in txt
    assert 'target "prod-db" is not allowed for context "internal"' in txt
    rr2 = G.record_run(people["reg"], _spec(), _tables(), context="internal", target="dev-sandbox",
                       operator=people["analyst"], source_id="dwh-prod", privacy_report=_privacy_report("pass"),
                       scorecard={"passed": 80, "total": 100, "score": 0.8})
    with pytest.raises(ReleaseDenied) as e:
        G.release(people["reg"], rr2.run_id, "internal", "dev-sandbox", people["analyst"])
    assert any("quality pass rate 0.8 is below the organisation's minimum 0.95" in r for r in e.value.reasons)


def test_budget_exceeded_is_refused():
    pol = G.policy.load(EXAMPLE)
    led = {"domain": "customers", "delta": 1e-9, "run_rho": 0.05, "domain_rho_after": 0.5, "lifetime_rho_cap": 0.435,
           "lifetime_epsilon_cap": 6.0, "per_release_rho_cap": 0.056, "per_release_epsilon_cap": 2.0}
    inp = {"policy": pol.doc, "manifest": {"privacy": {"mode": "dp", "unit": "Customer"}, "source": {"domain": "customers"},
                                           "spec": {"public_facts_only": True}, "verdicts": {"partner": "pass"},
                                           "run": {"operator": {"key": "op"}}, "reports": []},
           "manifest_digest": "m", "ledger": led, "privacy_checks": [], "quality": {"pass_rate": 0.99},
           "approvals": [{"key": "k", "approver": "D", "decision": "approve", "manifest": "m", "roles": ["dpo"],
                          "expired": False}],
           "request": {"context": "partner", "target": "partner-sftp", "package": []}}
    res = decide(inp)
    assert not res["allow"]
    assert 'domain "customers" would reach rho 0.5, above its lifetime cap 0.435 (epsilon 6 at delta 1e-09)' in res["reasons"]
    assert "invariant: the domain's lifetime budget would be exceeded" in res["reasons"]
    ok = copy.deepcopy(inp)
    ok["ledger"]["domain_rho_after"] = 0.1
    assert decide(ok)["allow"]


def test_privacy_report_cannot_leave_the_organisation():
    pol = G.policy.load(EXAMPLE)
    inp = {"policy": pol.doc, "manifest": {"privacy": {"mode": "dp", "unit": "Customer"}, "source": {"domain": "customers"},
                                           "spec": {"public_facts_only": True}, "verdicts": {"partner": "pass"},
                                           "run": {"operator": {"key": "op"}},
                                           "reports": [{"kind": "privacy", "digest": {"sha256": "abc"}, "external_safe": False}]},
           "manifest_digest": "m", "ledger": {"domain": "customers", "run_rho": 0.01, "domain_rho_after": 0.01,
                                              "lifetime_rho_cap": 0.4, "per_release_rho_cap": 0.05},
           "approvals": [{"key": "k", "decision": "approve", "manifest": "m", "roles": ["dpo"], "expired": False}],
           "request": {"context": "partner", "target": "partner-sftp", "package": ["abc"]}}
    res = builtin(inp)
    assert "the privacy report was computed on real data without DP and cannot leave the organisation" in res["reasons"]


def test_invariants_hold_under_a_permissive_policy():
    inp = {"policy": {"contexts": {"partner": {"external": True}}}, "manifest": {"privacy": {"mode": "none"}},
           "request": {"context": "partner"}, "approvals": []}
    assert len(invariants(inp)) == 2
    assert not decide(inp, engine="nonsense")["allow"]


# ---------------------------------------------------------------- approvals and release
def test_internal_review_needs_named_approval_then_releases(people, tmp_path):
    rr = _record(people, verdict="review")
    assert not rr.decisions["internal"]["allow"]
    with pytest.raises(GovernanceError, match="separation of duties"):
        people["reg"].trust(people["owner"], people["analyst"].public, "Ana Analyst", ["operator", "dpo"])
        G.approve(people["reg"], people["policy"], rr.run_id, "internal", people["analyst"])
    with pytest.raises(ReleaseDenied, match="needs 1 approval"):
        G.release(people["reg"], rr.run_id, "internal", "dev-sandbox", people["analyst"])
    G.approve(people["reg"], people["policy"], rr.run_id, "internal", people["owner"])
    res = G.release(people["reg"], rr.run_id, "internal", "dev-sandbox", people["analyst"], out=tmp_path / "pkg")
    assert res["allow"] and res["approvals"] == 1
    assert G.verify_package(tmp_path / "pkg") == []
    data = tmp_path / "pkg" / "data" / "Invoice.csv"
    data.write_text(data.read_text().replace("9.9", "9.8"))
    assert any("does not match the signed manifest" in x for x in G.verify_package(tmp_path / "pkg"))
    assert G.verify_registry(people["reg"]) == []
    st = G.show(people["reg"], rr.run_id)
    assert st["releases"] and st["approvals"][0]["approver"]["name"] == "Dana DPO"


def test_rejection_and_revocation(people):
    rr = _record(people)
    G.approve(people["reg"], people["policy"], rr.run_id, "internal", people["owner"], reject=True, reason="too wide")
    with pytest.raises(ReleaseDenied, match="rejected by Dana DPO: too wide"):
        G.release(people["reg"], rr.run_id, "internal", "dev-sandbox", people["analyst"])
    rr2 = _record(people)
    G.revoke(people["reg"], rr2.run_id, people["owner"], "wrong source")
    with pytest.raises(ReleaseDenied, match="revoked"):
        G.release(people["reg"], rr2.run_id, "internal", "dev-sandbox", people["analyst"])
    assert {r["state"] for r in G.runs(people["reg"])} == {"rejected", "revoked"}


def test_external_dp_release_with_quorum_and_token(people, tmp_path):
    led = G.Ledger(people["reg"], people["policy"])
    run = G.new_run_id()
    led.reserve("customers", run, 0.04)
    led.touch("customers", run)
    led.settle("customers", run, 0.04)
    priv = _dp_privacy("0.04")
    priv["_run_id"] = run
    rr = _record(people, context="public", target="open-data-portal", privacy=priv, public_facts_only=True,
                 dp_statement={"schema": "relsynth.dp-statement/1", "epsilon": "1.9"})
    G.approve(people["reg"], people["policy"], rr.run_id, "public", people["owner"])
    with pytest.raises(ReleaseDenied, match="needs 2 approval"):
        G.release(people["reg"], rr.run_id, "public", "open-data-portal", people["analyst"])
    G.approve(people["reg"], people["policy"], rr.run_id, "public", people["dpo2"])
    res = G.release(people["reg"], rr.run_id, "public", "open-data-portal", people["analyst"], out=tmp_path / "pub")
    assert res["allow"]
    assert (tmp_path / "pub" / "dp_statement.json").exists()
    assert G.verify_package(tmp_path / "pub") == []
    tok = json.loads((tmp_path / "pub" / "release-token.dsse").read_text())
    assert G.check_token(people["reg"], tok, "public", "open-data-portal")["manifest"] == rr.manifest
    with pytest.raises(ReleaseDenied):
        G.check_token(people["reg"], tok, "partner", "partner-sftp")
    G.revoke(people["reg"], rr.run_id, people["owner"], "found an error")
    with pytest.raises(ReleaseDenied, match="revoked"):
        G.check_token(people["reg"], tok, "public", "open-data-portal")
    # revoking never refunds budget
    assert abs(led.summary("customers")["rho"] - 0.04) < 1e-12


def test_manifest_holds_no_data_or_credentials(people):
    rr = _record(people)
    raw = json.dumps(rr.statement)
    for v in ("secret", "db.internal", "GR", "9.9"):
        assert v not in raw
    assert G.verify_registry(people["reg"]) == []


# ---------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path, capsys):
    from relsynth.cli import main
    k = tmp_path / "keys"
    assert main(["gov", "keys", "new", str(k / "dpo.key"), "--name", "DPO"]) == 0
    assert main(["gov", "keys", "new", str(k / "ana.key"), "--name", "Ana"]) == 0
    assert main(["gov", "init", str(tmp_path / "reg"), "--owner-key", str(k / "dpo.key"),
                 "--policy-out", str(tmp_path / "skeleton.yaml")]) == 0
    assert main(["gov", "policy", "check", str(tmp_path / "skeleton.yaml")]) == 2
    assert "the organisation must set epsilon and delta" in capsys.readouterr().out
    r = ["--registry", str(tmp_path / "reg")]
    assert main(["gov", "policy", "check", str(EXAMPLE)]) == 0
    assert main(["gov", "policy", "sign", str(EXAMPLE), "--key", str(k / "dpo.key"), *r]) == 0
    assert main(["gov", "keys", "trust", str(k / "ana.pub.json"), "--roles", "operator", "--owner-key",
                 str(k / "dpo.key"), *r]) == 0
    capsys.readouterr()
    assert main(["gov", "budget", *r, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["budget"]["customers"]["rho"] == 0
    assert main(["gov", "verify", str(tmp_path / "reg")]) == 0
    with open(tmp_path / "reg" / "log.jsonl", "a") as fh:
        fh.write('{"seq": 99, "prev": "x", "entry": {"type": "manifest"}, "hash": "y"}\n')
    assert main(["gov", "verify", str(tmp_path / "reg")]) == 1


def test_docs_schemas_match_package():
    for name in ("org-policy-1.schema.json", "run-manifest-1.schema.json"):
        a = (ROOT / "docs" / "governance" / "schemas" / name).read_text()
        b = (ROOT / "src" / "relsynth" / "governance" / "schema" / name).read_text()
        assert json.loads(a) == json.loads(b), name

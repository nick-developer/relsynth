# SPDX-License-Identifier: Apache-2.0
"""The agents (T16): review answers with evidence, people's decisions,
explanations, doctor proposals, drafts and signed action records."""
from __future__ import annotations

import json

import pytest
import yaml

import relsynth as rs
from relsynth import agents
from relsynth.agents.evaluation import evaluate, truth_from_spec
from relsynth.agents.items import parse
from relsynth.agents.scope import Scope, for_roles

from .toydb import SECRET, make_toy


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    p = make_toy(tmp_path_factory.mktemp("toy") / "toy.db")
    draft = rs.introspect(f"sqlite:{p}", declared=False).raw
    truth = rs.introspect(f"sqlite:{p}").raw
    return p, draft, truth


def test_parse_kinds_of_review_lines():
    lines = [
        "Album: primary key AlbumId inferred (only unique non-null key-like column (score 1.00))",
        "Album.ArtistId -> Artist: foreign key inferred (100% of values in the parent, coverage 74%)",
        "ASK Customer.SupportRepId: looks like a foreign key but the data fits Employee (100% of values in the "
        "parent, coverage 38%), MediaType (100% of values, coverage 60%) equally well; declare it",
        "ASK Employee.ReportsTo: looks like a foreign key to Employee (100% of values in the parent), but the data "
        "does not establish it; declare it under relationships",
        "PlaylistTrack: unique key [PlaylistId, TrackId] inferred (8715 rows, no repeated pair)",
        "CurrencyExchange: unique key ['Date', 'FromCurrency', 'ToCurrency'] inferred (no repeat)",
        "Invoice.BillingCity: lookup CustomerId.City (equal on 100.0% of 412 sampled rows; 2.0% expected by chance)",
        "Invoice.Total = sum(UnitPrice * Quantity) of InvoiceLine: aggregate inferred (holds for every one of 400)",
        "orders: derive rule amount = qty * unit_price (holds on the sampled rows to the stored precision 0.01)",
        "Product: check Cost < Price (holds on 100.0% of 2,517 sampled rows, 66.3% by chance)",
        "Employee.Country: constant (a single value)",
        "Invoice.InvoiceDate: date (every value is at midnight: a calendar date stored as a date-time)",
        "Customer.Email: pii (name matches email)",
        "Genre: 25 rows of reference data, copied as-is (set mode: synthesize if it holds confidential data)",
        "NOTE: something",
        "a line nobody can parse",
    ]
    kinds = [i.kind for i in parse(lines)]
    assert kinds == ["pk", "fk", "fk_ask", "fk_ask", "unique", "unique", "lookup", "aggregate", "derive", "check",
                     "constant", "date", "privacy", "privacy", "note", "other"]
    items = parse(lines)
    assert items[2].candidates == ["Employee", "MediaType"] and items[3].candidates == ["Employee"]
    assert items[5].columns == ["Date", "FromCurrency", "ToCurrency"]
    # with the draft spec, a name the spec does not define is never answered
    spec = {"tables": {"Album": {}, "Artist": {}}}
    assert [i.kind for i in parse(lines[:2], spec)] == ["pk", "fk"]
    assert parse(["Ghost: primary key x inferred (y)"], spec)[0].kind == "other"


def test_scope_for_roles():
    assert for_roles({"consumer"}) == "schema"
    assert for_roles({"engineer"}) == "aggregates"
    assert for_roles({"data_owner"}) == "values"
    assert for_roles({"data_owner"}, ceiling="aggregates") == "aggregates"
    assert Scope("values", 10).value("x", 9) is None and Scope("values", 10).value("x", 10) == "x"
    assert Scope("aggregates").value("x", 10**6) is None and Scope("schema").number(5) is None
    with pytest.raises(ValueError):
        Scope("rows")


def test_review_toy_against_catalog(toy):
    p, draft, truth = toy
    res = agents.review(draft, source=f"sqlite:{p}")
    by = {it.kind: [] for it in res.items}
    for it, a in zip(res.items, res.answers):
        by[it.kind].append(a)
    assert all(a["verdict"] == "confirm" for a in by["pk"] + by["fk"])
    ask = by["fk_ask"][0]                                        # employee.reports_to: a self-reference
    assert ask["verdict"] == "answer" and ask["parent"] == "employee" and ask["change"]["op"] == "add_relationship"
    assert all(a["verdict"] == "explain" and a["attention"] == "high" for a in by["privacy"])
    assert all(a["evidence"] for a in by["pk"] + by["fk"] + by["lookup"] + by["derive"])
    ev = evaluate(res, truth_from_spec(truth))
    assert ev["wrong"] == 0 and ev["correct"] >= 9
    assert ev["approval"]["oracle_wrong_changes_applied"] == 0
    # the draft never asked about customer.support_rep: the evaluation reports it as missed
    assert ["customer", "support_rep", "employee"] in ev["links_missed_by_draft"]
    md = res.to_markdown()
    assert "Nothing has been applied" in md and SECRET not in md


def test_review_reads_no_rows_into_output(toy):
    p, draft, _ = toy
    for level in ("schema", "aggregates", "values"):
        res = agents.review(draft, source=f"sqlite:{p}", scope=level)
        out = json.dumps(res.to_dict(), default=str)
        assert SECRET not in out, level                     # no name, no email: values occur once each
        if level == "schema":
            assert all(not e.result.get("available") for evs in res.evidence.values() for e in evs)
            assert {a["verdict"] for a in res.answers} <= {"unsure", "explain"}


def test_values_scope_shows_only_frequent_values(toy):
    p, draft, _ = toy
    res = agents.review(draft, source=f"sqlite:{p}", scope="values")
    consts = [e for evs in res.evidence.values() for e in evs if e.tool == "distinct"]
    assert consts and all(e.result.get("value") == "active" for e in consts if e.args["column"] == "status")
    res2 = agents.review(draft, source=f"sqlite:{p}", scope="aggregates")
    assert all("value" not in e.result for evs in res2.evidence.values() for e in evs)


def test_evidence_gate_refuses_confirmation_of_a_false_inference(toy, tmp_path):
    p, draft, _ = toy
    bad = dict(draft)
    bad["review"] = list(draft["review"]) + ["customer: primary key region_id inferred (forced)",
                                             "orders.amount -> region: foreign key inferred (forced)"]
    always_yes = agents.ScriptedBackend(lambda s, u, sc: {"verdict": "confirm", "parent": None, "confidence": 1.0,
                                                          "evidence": ["E1"], "explanation": "fine"})
    res = agents.review(bad, source=f"sqlite:{p}", backend=always_yes)
    ans = {it.text: a for it, a in zip(res.items, res.answers)}
    assert ans["customer: primary key region_id inferred (forced)"]["verdict"] == "reject"
    assert ans["orders.amount -> region: foreign key inferred (forced)"]["verdict"] == "reject"


def test_apply_decisions_needs_a_person_and_rebuilds_changes(toy):
    p, draft, _ = toy
    res = agents.review(draft, source=f"sqlite:{p}")
    accept = {a["item"]: "accept" for a in res.answers}
    with pytest.raises(PermissionError):
        agents.apply_decisions(draft, res.answers, accept, decided_by="agent:relsynth")
    with pytest.raises(PermissionError):
        agents.apply_decisions(draft, res.answers, accept, decided_by="service:ci")
    # a tampered stored answer cannot smuggle in its own change
    tampered = [dict(a, change={"op": "add_relationship", "child": "orders", "child_key": "qty",
                                "parent": "region"}) for a in res.answers]
    new, log = agents.apply_decisions(draft, tampered, accept, decided_by="user:alice")
    rels = {(r["child"], r["child_key"], r["parent"]) for r in new["relationships"]}
    assert ("orders", "qty", "region") not in rels and ("employee", "reports_to", "employee") in rels
    assert new["review"] == [] and all(e["by"] == "user:alice" for e in log)
    assert rs.Spec(dict(new, source={"kind": "sqlite", "path": str(p)})).errors() == []
    # undecided answers change nothing
    same, log2 = agents.apply_decisions(draft, res.answers, {}, decided_by="user:alice")
    assert same == draft and log2 == []


def test_rejected_rule_is_proposed_for_removal(toy, tmp_path):
    p, draft, _ = toy
    import sqlite3
    q = tmp_path / "broken.db"
    q.write_bytes(p.read_bytes())
    db = sqlite3.connect(q)
    db.execute("UPDATE orders SET amount = amount + 1 WHERE order_id = 5")
    db.commit()
    db.close()
    res = agents.review(draft, source=f"sqlite:{q}", only=["derive"])
    a = res.answers[0]
    assert a["verdict"] == "reject" and a["change"]["op"] == "remove_rule"
    new, _ = agents.apply_decisions(draft, res.answers, {a["item"]: "accept"}, decided_by="user:alice")
    assert not any(r.get("type") == "derive" for r in new.get("rules") or [])


def test_explain_reports_are_grounded():
    q = {"vector": {"fidelity": {"checks": 10, "passed": 9, "failed": 1, "allowed_failures": 1, "pass": True,
                                 "expected_false_failures": 0.1},
                    "utility": {"checks": 2, "passed": 2, "failed": 0, "allowed_failures": 0, "pass": True},
                    "integrity": {"checks": 3, "passed": 3, "failed": 0, "allowed_failures": 0, "pass": True},
                    "verdict": {"pass": True}},
         "checks": [{"id": "w1:orders:amount", "area": "fidelity", "metric": "w1", "value": 0.31, "limit": 0.2,
                     "pass": False}]}
    e = agents.explain_report(q)
    assert e["kind"] == "quality" and e["source"] == "rules" and "9 of 10" in e["text"] and "0.31" in e["text"]
    liar = agents.ScriptedBackend(lambda s, u, sc: {"summary": "All 47 checks passed and the data are approved "
                                                               "for release."})
    e2 = agents.explain_report(q, backend=liar)
    assert e2["source"] == "rules" and any("numbers" in x for x in e2["problems"])
    honest = agents.ScriptedBackend(lambda s, u, sc: {"summary": "Fidelity passed 9 of 10 checks; one check, "
                                                                 "w1, measured 0.31 against a limit of 0.2."})
    e3 = agents.explain_report(q, backend=honest)
    assert e3["source"] == "model" and e3["problems"] == []
    priv = {"people": 120, "contexts": {"partner": {"verdict": "blocked", "external": True, "dp_required": True,
                                                    "blocked_reason": "needs DP", "checks": [],
                                                    "sign_off": {"required": True}}}}
    e4 = agents.explain_report(priv)
    assert "differential privacy" in e4["text"] and "cannot" in e4["text"]


def test_doctor_proposals_hold_back_privacy_fixes(monkeypatch):
    from relsynth.agents import drafts
    from relsynth.doctor import Issue

    class D:
        issues = [Issue("R1", "warn", "spec", "orders.email", "looks like an email address", {},
                        [{"path": "tables.orders.columns.email", "value": {"type": "pii", "pii": {"generator": "email"}}}]),
                  Issue("R2", "warn", "spec", "copula", "chains did not mix",
                        [], [{"path": "synthesis.copula.mode", "value": "rank"}]),
                  Issue("R3", "info", "spec", "text", "copies long phrases", {},
                        [{"path": "tables.t.columns.c.text.max_copy_ngram", "value": 4}])]
    monkeypatch.setattr("relsynth.api.doctor", lambda b: D())
    props = drafts.doctor_proposals({})
    assert props[0]["patches"] == [] and props[0]["advice_for_privacy_officer"]
    assert props[1]["patches"] == [{"path": "synthesis.copula.mode", "value": "rank"}]
    assert props[2]["patches"] == [] and props[2]["advice_for_privacy_officer"]
    spec = drafts.apply_doctor({"tables": {}}, props, ["D1", "D2", "D3"], decided_by="user:alice")
    assert spec["synthesis"]["copula"]["mode"] == "rank" and "orders" not in spec["tables"]
    with pytest.raises(PermissionError):
        drafts.apply_doctor({}, props, ["D2"], decided_by="agent:x")


def test_drafts_leave_epsilon_to_the_privacy_owner(toy):
    from relsynth.governance import policy as P
    _, _, truth = toy
    txt = agents.draft_policy(truth, organisation="Acme")
    doc = yaml.safe_load(txt)
    dom = doc["domains"]["people"]
    assert dom["budget"]["lifetime"]["epsilon"] is None and dom["budget"]["delta"] is None
    assert dom["unit"]["entity"] in ("customer", "employee")
    assert any("epsilon" in e or "budget" in e or "delta" in e for e in P.validate(doc))   # incomplete by design
    c = agents.draft_contract(truth, owner="alice")
    assert c["status"] == "draft" and c["kind"] == "DataContract"
    assert {o["name"] for o in c["schema"]} >= {"customer", "orders"}


def test_backend_configuration():
    assert agents.make_backend("rules").name == "rules"
    b = agents.make_backend({"kind": "openai", "model": "m"})
    assert b.local and b.base_url.startswith("http://localhost")
    with pytest.raises(ValueError):
        agents.make_backend({"kind": "openai", "base_url": "https://llm.example.com/v1"})
    with pytest.raises(ValueError):
        agents.make_backend({"kind": "openai", "base_url": "http://llm.example.com/v1", "allow_remote": True})
    r = agents.make_backend({"kind": "openai", "base_url": "https://llm.example.com/v1", "allow_remote": True})
    assert not r.local


def test_remote_model_never_gets_values(toy):
    p, draft, _ = toy
    seen = []

    class Remote(agents.ScriptedBackend):
        local = False

    b = Remote(lambda s, u, sc: seen.append(u) or {"verdict": "unsure", "parent": None, "confidence": 0,
                                                     "evidence": [], "explanation": ""})
    res = agents.review(draft, source=f"sqlite:{p}", backend=b, scope="values")
    assert res.scope == "aggregates"


def test_openai_backend_speaks_chat_completions(monkeypatch):
    import io
    sent = {}

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        msg = {"verdict": "unsure", "parent": None, "confidence": 0.1, "evidence": [], "explanation": "x"}
        return R(json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(msg) + "\n```"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    b = agents.OpenAICompatBackend(model="tiny")
    out = b.complete("sys", "user", {"type": "object"})
    assert out["verdict"] == "unsure" and sent["url"].endswith("/chat/completions")
    assert sent["body"]["response_format"]["type"] == "json_schema" and sent["body"]["temperature"] == 0


def test_cli_review_apply_explain_draft(toy, tmp_path):
    from relsynth.cli import main
    p, draft, _ = toy
    d = tmp_path / "draft.yaml"
    d.write_text(yaml.safe_dump(dict(draft, source={"kind": "sqlite", "path": str(p)})))
    assert main(["agent", "review", str(d), "--out", str(tmp_path / "a.json"), "--md", str(tmp_path / "a.md")]) == 0
    assert main(["agent", "apply", str(d), str(tmp_path / "a.json"), "--accept", "all-confirmed", "--by", "alice",
                 "--out", str(tmp_path / "s.yaml"), "--log", str(tmp_path / "log.json")]) == 0
    log = json.loads((tmp_path / "log.json").read_text())
    assert log and all(e["by"] == "user:alice" for e in log)
    assert main(["agent", "draft", "policy", str(d), "--out", str(tmp_path / "pol.yaml")]) == 0
    assert main(["agent", "draft", "contract", str(d), "--out", str(tmp_path / "c.yaml")]) == 0
    truth = tmp_path / "truth.yaml"
    truth.write_text(yaml.safe_dump(toy[2]))
    assert main(["agent", "eval", str(d), str(truth), "--min-accuracy", "0.9", "--out", str(tmp_path / "e.json")]) == 0


def test_signed_action_records_and_manifest_entry(toy, tmp_path):
    pytest.importorskip("cryptography")
    from relsynth import governance as G
    from relsynth.agents.record import action_statement, manifest_entry, record_action
    from relsynth.errors import GovernanceError
    p, draft, _ = toy
    owner, bot, analyst = (G.Signer.generate(n) for n in ("dpo", "review-bot", "analyst"))
    reg = G.LocalRegistry.create(tmp_path / "reg", owner)
    reg.trust(owner, bot.public, "review-bot", ["agent"])
    reg.trust(owner, analyst.public, "analyst", ["operator"])
    res = agents.review(draft, source=f"sqlite:{p}")
    st = action_statement(task="review", agent="agent:review-bot", on_behalf_of="user:alice",
                          backend=res.backend, scope=res.scope, inputs={"spec": draft}, output=res.answers,
                          proposals=len(res.answers))
    assert SECRET not in json.dumps(st)
    d, seq = record_action(reg, st, bot)
    assert seq > 0 and len(d) == 64 and G.verify_registry(reg) == []
    with pytest.raises(GovernanceError):
        record_action(reg, st, analyst)                      # only agent keys sign agent actions
    ent = manifest_entry(d, "agent:review-bot", "review", {"a": "accept", "b": "reject"}, "user:alice")
    assert ent["accepted"] == 1 and ent["rejected"] == 1

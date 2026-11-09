# SPDX-License-Identifier: Apache-2.0
"""Governed DP runs with several privacy units (RFC-G1-01, Prop. 1704): a
domain with `unit.entities`, budgets per unit set by hand (`unit_budgets`),
one ledger account per unit, per-unit epsilon in the spec, the manifest's
`privacy.units` and the release decision per unit."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

pytest.importorskip("cryptography")
pytest.importorskip("jsonschema")

import relsynth.dp as dp  # noqa: E402
from relsynth import governance as G  # noqa: E402
from relsynth.errors import PrivacyError, ReleaseDenied  # noqa: E402
from relsynth.governance.decide import decide, unit_views  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "docs" / "governance" / "examples" / "org-policy.yaml"


def _units_fixture():
    spec = importlib.util.spec_from_file_location("units_checks", ROOT / "tests" / "science" /
                                                  "test_privacy_units_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc(unit_budgets=None):
    doc = yaml.safe_load(EXAMPLE.read_text())
    d = doc["domains"]["customers"]
    d["unit"] = {"entities": ["customer", "employee"]}
    if unit_budgets:
        d["unit_budgets"] = unit_budgets
    return doc


@pytest.fixture
def gov(tmp_path):
    owner = G.Signer.generate("Dana DPO")
    analyst = G.Signer.generate("Ana Analyst")
    reg = G.LocalRegistry.create(tmp_path / "registry", owner, "dpo")
    reg.trust(owner, analyst.public, "Ana Analyst", ["operator"])
    # employees get a smaller lifetime budget, set by hand
    emp = {"accounting": "zcdp", "delta": 1.0e-9, "lifetime": {"epsilon": 3.0}, "per_release": {"epsilon_max": 1.5}}
    pol = G.policy.load(_doc({"employee": emp}))
    G.policy.sign(pol, owner, reg)
    return {"reg": reg, "owner": owner, "analyst": analyst, "policy": pol}


# ---------------------------------------------------------------- policy
def test_policy_names_several_units_and_budgets_per_unit(gov):
    pol = gov["policy"]
    assert pol.units("customers") == ["customer", "employee"] and pol.multi_unit("customers")
    assert pol.caps("customers", "customer").lifetime_epsilon == 6.0          # the domain's budget, per unit
    assert pol.caps("customers", "employee").lifetime_epsilon == 3.0          # its own, set by hand
    bad = _doc({"supplier": {"delta": 1e-9, "lifetime": {"epsilon": 1.0}}})
    assert any("not a unit of the domain" in p for p in G.policy.validate(bad))
    bad = _doc({"employee": {"delta": 1e-8, "lifetime": {"epsilon": 1.0}}})
    assert any("domain's delta" in p for p in G.policy.validate(bad))
    one = yaml.safe_load(EXAMPLE.read_text())
    one["domains"]["customers"]["unit_budgets"] = {"Customer": {"delta": 1e-9, "lifetime": {"epsilon": 1.0}}}
    assert any("needs unit.entities" in p for p in G.policy.validate(one))
    assert G.policy.validate(_doc()) == []


# ---------------------------------------------------------------- ledger
def test_ledger_keeps_one_account_per_unit(gov):
    led = G.Ledger(gov["reg"], gov["policy"])
    with pytest.raises(PrivacyError, match="name the unit"):
        led.reserve("customers", "r0", 0.01)
    led.reserve_units("customers", "r1", {"customer": (0.05, 0.0), "employee": (0.02, 0.0)})
    for u in ("customer", "employee"):
        led.touch("customers", "r1", unit=u)
    led.settle("customers", "r1", 0.04, unit="customer")
    led.settle("customers", "r1", 0.02, unit="employee")
    assert abs(led.summary("customers", unit="customer")["rho"] - 0.04) < 1e-12
    assert abs(led.summary("customers", unit="employee")["rho"] - 0.02) < 1e-12
    # the employee account has its own, smaller per-release cap: the whole reservation is refused
    cap = gov["policy"].caps("customers", "employee").per_release_rho
    with pytest.raises(PrivacyError, match="for employee, above the per-release cap"):
        led.reserve_units("customers", "r2", {"customer": (0.01, 0.0), "employee": (cap * 1.01, 0.0)})
    assert "r2" not in led.state("customers", "customer").runs              # nothing reserved
    assert G.verify_registry(gov["reg"]) == []


# ---------------------------------------------------------------- spec
def test_spec_takes_an_epsilon_per_unit():
    m = _units_fixture()
    s = m._spec(unit=["customer", "employee"], epsilon={"customer": 2.0, "employee": 0.5})
    cfg = dp.parse(s)
    assert cfg.unit_epsilon == {"customer": 2.0, "employee": 0.5}
    assert cfg.rho_of("employee") < cfg.rho_of("customer") == cfg.rho
    with pytest.raises(PrivacyError, match="one value per unit"):
        dp.parse(m._spec(unit=["customer", "employee"], epsilon={"customer": 2.0}))
    with pytest.raises(PrivacyError, match="one value per unit"):
        dp.parse(m._spec(unit="customer", epsilon={"customer": 2.0}))


def test_domain_override_accepts_the_domains_units(gov):
    m = _units_fixture()
    pol = gov["policy"]
    o = dp.domain_override(m._spec(unit=["customer", "employee"]), pol, "customers")
    assert o["unit"] == ["customer", "employee"] and o["delta"] == 1e-9
    assert dp.domain_override(m._raw_spec(), pol, "customers")["unit"] == ["customer", "employee"]
    with pytest.raises(PrivacyError, match="leaves out 'customer'"):
        dp.domain_override(m._spec(unit="employee"), pol, "customers")
    # a source that holds only the employees protects only them
    only = m._raw_spec()
    del only["tables"]["customer"]
    only["tables"]["sale"]["fact"].pop("anchor", None)
    assert dp.domain_override(only, pol, "customers")["unit"] == "employee"
    with pytest.raises(PrivacyError, match="'product'"):
        dp.domain_override(m._spec(unit=["customer", "product"]), pol, "customers")


# ---------------------------------------------------------------- decisions
def _inp(pol, units_ledger, unit=("customer", "employee")):
    priv = {"mode": "dp", "unit": list(unit), "adjacency": "add_remove",
            "charges": [{"stage": "fit", "mechanism": "m", "kind": "zcdp", "rho": "0.01", "unit": u} for u in unit],
            "run_total": {"rho": "0.05", "delta": "0"}, "stated": {"epsilon": "1.9", "delta": "1e-9"},
            "units": {u: {"run_total": {"rho": "0.05", "delta": "0"}, "stated": {"epsilon": "1.9", "delta": "1e-9"}}
                      for u in unit}}
    return {"policy": pol.doc, "manifest": {"privacy": priv, "source": {"domain": "customers"},
                                            "spec": {"public_facts_only": True}, "verdicts": {"partner": "pass"},
                                            "run": {"operator": {"key": "op"}}, "reports": []},
            "manifest_digest": "m", "ledger": {"domain": "customers", "units": units_ledger},
            "privacy_checks": [], "quality": {"pass_rate": 0.99},
            "approvals": [{"key": "k", "approver": "D", "decision": "approve", "manifest": "m", "roles": ["dpo"],
                           "expired": False}],
            "request": {"context": "partner", "target": "partner-sftp", "package": []}}


def _led(after, cap=0.4):
    return {"domain": "customers", "delta": 1e-9, "run_rho": 0.05, "domain_rho_after": after, "lifetime_rho_cap": cap,
            "lifetime_epsilon_cap": 6.0, "per_release_rho_cap": 0.056, "per_release_epsilon_cap": 2.0}


def test_release_is_decided_for_each_unit(gov):
    pol = gov["policy"]
    ok = _inp(pol, {"customer": _led(0.1), "employee": _led(0.1, cap=0.2)})
    views = unit_views(ok)
    assert [v["manifest"]["privacy"]["unit"] for v in views] == ["customer", "employee"]
    assert all(len(v["manifest"]["privacy"]["charges"]) == 1 for v in views)
    assert decide(ok)["allow"]
    over = _inp(pol, {"customer": _led(0.1), "employee": _led(0.3, cap=0.2)})
    res = decide(over)
    assert not res["allow"] and "invariant: the domain's lifetime budget would be exceeded" in res["reasons"]
    stranger = _inp(pol, {"customer": _led(0.1)}, unit=("customer", "supplier"))
    assert any('protects "supplier"' in r for r in decide(stranger)["reasons"])
    # a one-unit domain decides exactly as before
    one = copy.deepcopy(ok)
    one["policy"] = G.policy.load(EXAMPLE).doc
    assert unit_views(one) == [one]


# ---------------------------------------------------------------- end to end
def test_governed_two_unit_run_reserves_settles_and_records(gov):
    m = _units_fixture()
    pol, reg = gov["policy"], gov["reg"]
    spec = m._spec(unit=["customer", "employee"], epsilon={"customer": 1.0, "employee": 0.5},
                   public_tables=["reason", "product"])
    spec["source"] = {"kind": "csv", "path": "real/"}
    over = dp.domain_override(spec, pol, "customers", {"enabled": True})
    _, cfg, _ = dp.plan(spec, over)
    led, run = G.Ledger(reg, pol), G.new_run_id()
    led.reserve_units("customers", run, {u: (cfg.rho_of(u), cfg.delta * cfg.levels_delta_share) for u in cfg.units})
    seqs: dict = {}

    def settle(per_unit):
        for u, v in per_unit.items():
            seqs[u] = led.settle("customers", run, v["rho"], v["delta"], unit=u)
    res = dp.synthesize(spec, tables=m._tables(), override=over, write=False, evaluate=False,
                        before_read=lambda: [led.touch("customers", run, unit=u) for u in cfg.units],
                        after_fit_units=settle)
    assert set(seqs) == {"customer", "employee"}
    assert res.per_unit["employee"]["rho"] <= cfg.rho_of("employee") * (1 + 1e-9)
    notes = {}
    for u in cfg.units:
        s = led.summary("customers", unit=u)
        notes[u] = {"seq": seqs[u], "domain_rho": G.dec(s["rho"]), "domain_delta": G.dec(s["delta_spent"], 30),
                    "domain_epsilon": G.dec(s["epsilon"], 6)}
    priv = dp.manifest_privacy(res, run, notes)
    assert priv["unit"] == ["customer", "employee"] and set(priv["units"]) == {"customer", "employee"}
    assert priv["units"]["employee"]["ledger"]["seq"] == seqs["employee"]
    rr = G.record_run(reg, spec, dict(res.data), context="internal", target="dev-sandbox", operator=gov["analyst"],
                      source_id="dwh-prod", privacy=priv, dp_statement=res.statement, public_facts_only=False,
                      privacy_report={"schema": "relsynth.privacy-report/1",
                                      "contexts": {"internal": {"verdict": "pass", "checks": []}}},
                      scorecard={"passed": 99, "total": 100, "score": 0.99})
    assert rr.decisions["internal"]["allow"], rr.decisions["internal"]
    assert G.verify_registry(reg) == []
    # employees' smaller lifetime budget: a run that would exceed it is refused for both units
    big = gov["policy"].caps("customers", "employee").lifetime_rho
    with pytest.raises(PrivacyError):
        led.reserve_units("customers", "again", {"customer": (0.001, 0.0), "employee": (big, 0.0)})
    with pytest.raises(ReleaseDenied):
        G.release(reg, rr.run_id, "internal", "prod-db", gov["analyst"])


def test_cli_governed_two_unit_run(tmp_path, capsys):
    """`relsynth run --registry` on a two-unit domain: one reservation and one
    settle per unit, the manifest lists the units, `gov budget` shows each
    unit's account."""
    import json

    from relsynth.cli import main
    m = _units_fixture()
    data = tmp_path / "real"
    data.mkdir()
    for t, df in m._tables().items():
        df.to_csv(data / f"{t}.csv", index=False)
    spec = m._raw_spec()
    spec["source"] = {"kind": "csv", "path": str(data)}
    spec["synthesis"] = {"privacy": {"dp": {"unit": ["customer", "employee"], "public_tables": ["reason", "product"],
                                            "epsilon": {"customer": 1.0, "employee": 0.5}}}}
    sf = tmp_path / "spec.yaml"
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    k = tmp_path / "keys"
    for name in ("dpo", "ana"):
        assert main(["gov", "keys", "new", str(k / f"{name}.key"), "--name", name]) == 0
    reg = tmp_path / "reg"
    assert main(["gov", "init", str(reg), "--owner-key", str(k / "dpo.key")]) == 0
    doc = _doc({"employee": {"accounting": "zcdp", "delta": 1.0e-9, "lifetime": {"epsilon": 3.0}}})
    doc["quality"]["min_pass_rate"] = 0.0
    pf = tmp_path / "policy.yaml"
    pf.write_text(yaml.safe_dump(doc))
    r = ["--registry", str(reg)]
    assert main(["gov", "policy", "sign", str(pf), "--key", str(k / "dpo.key"), *r]) == 0
    assert main(["gov", "keys", "trust", str(k / "ana.pub.json"), "--roles", "operator", "--owner-key",
                 str(k / "dpo.key"), *r]) == 0
    rf = tmp_path / "privacy_report.json"
    rf.write_text(json.dumps({"schema": "relsynth.privacy-report/1",
                              "contexts": {"internal": {"verdict": "pass", "checks": []}}}))
    capsys.readouterr()
    code = main(["run", str(sf), *r, "--context", "internal", "--release-target", "dev-sandbox", "--source-id",
                 "dwh-prod", "--key", str(k / "ana.key"), "--privacy-report", str(rf), "--no-privacy",
                 "--no-write", "--bundle", "", "--json"])
    j = json.loads(capsys.readouterr().out)
    assert code == 0, j.get("error")
    assert set(j["dp"]["units"]) == {"customer", "employee"}
    assert main(["gov", "budget", *r, "--json"]) == 0
    b = json.loads(capsys.readouterr().out)["budget"]
    run = j["governance"]["run"]
    assert b["customers/customer"]["runs"][run]["state"] == "settled"
    assert b["customers/employee"]["runs"][run]["state"] == "settled"
    assert b["customers/employee"]["lifetime_epsilon_cap"] == 3.0
    assert main(["gov", "show", run, *r]) == 0
    shown = capsys.readouterr().out
    assert "DP: unit customer" in shown and "DP: unit employee" in shown
    assert main(["gov", "policy", "check", str(pf)]) == 0
    assert "unit employee" in capsys.readouterr().out
    assert main(["gov", "verify", str(reg)]) == 0

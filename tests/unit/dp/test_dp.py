# SPDX-License-Identifier: Apache-2.0
"""relsynth.dp: the DP section of a spec, the plan's refusals, the guards that
keep DP specs out of the ordinary pipeline, the DP statement and a small end
to end run (roadmap T07; the propositions are checked in
tests/science/test_dp_checks.py)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fakedata"))
import scenarios as S  # noqa: E402

import relsynth.dp as dp  # noqa: E402
from relsynth.errors import PrivacyError  # noqa: E402
from relsynth.governance.accounting import epsilon_of, rho_of  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


@pytest.fixture(scope="module")
def star():
    t, s, _ = S.SCENARIOS["S2_star"]()
    keep = set(t["customer"]["customer_id"].head(120))
    t["customer"] = t["customer"][t["customer"]["customer_id"].isin(keep)].reset_index(drop=True)
    t["sales"] = t["sales"][t["sales"]["customer_id"].isin(keep)].reset_index(drop=True)
    return t, s


def _spec(s, **kw):
    s = copy.deepcopy(s)
    s.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": "customer", "epsilon": 2.0, "delta": 1e-9, "public_tables": "all_unowned", **kw}
    return load_spec(s)


def test_config_needs_a_budget_unit_and_small_delta(star):
    _, s = star
    for kw, msg in (({"epsilon": None}, "no default epsilon"), ({"delta": 1e-4}, "delta"),
                    ({"unit": None}, "unit"), ({"rho": 0.1}, "not both"), ({"bogus": 1}, "unknown key"),
                    ({"split": {"discovery": 0.5}}, "adding up to 1"), ({"max_degree": {"sales": 0}}, "positive")):
        with pytest.raises(PrivacyError, match=msg):
            dp.parse(_spec(s, **kw))


def test_epsilon_is_converted_at_the_delta_left_after_level_selection(star):
    _, s = star
    c = dp.parse(_spec(s, epsilon=2.0, delta=1e-9))
    assert math.isclose(c.rho, rho_of(2.0, 0.5e-9))
    assert epsilon_of(c.rho, 0.5e-9) <= 2.0 + 1e-9
    assert math.isclose(dp.epsilon_of_config(c), 2.0)
    r = dp.parse(_spec(s, epsilon=None, rho=0.05))
    assert r.rho == 0.05 and dp.epsilon_of_config(r) == pytest.approx(epsilon_of(0.05, 0.5e-9))


def test_plan_traces_ownership_and_public_tables(star):
    _, s = star
    pl, cfg, _ = dp.plan(_spec(s))
    assert pl.unit == "customer" and pl.order == ["customer", "sales"]
    assert list(pl.public) == ["product"]
    with pytest.raises(PrivacyError):
        dp.plan(_spec(s, public_tables=[]))           # product is neither owned nor declared public


def test_ordinary_pipeline_refuses_a_dp_spec(star):
    from relsynth.pipeline import execute
    t, s = star
    with pytest.raises(PrivacyError, match="differential privacy mode"):
        execute(_spec(s), tables=t, write=False)
    from relsynth.privacy.assess import assess  # noqa: F401 - imported to show it is DP-aware (see dp.generator)


def test_stage2_spec_drops_the_dp_section(star):
    _, s = star
    s2 = dp.stage2_spec(_spec(s))
    assert "dp" not in s2["synthesis"]["privacy"] and s2["synthesis"]["pushdown"]["mode"] == "off"


def test_end_to_end_statement_and_charges(star):
    pytest.importorskip("opendp")
    t, s = star
    res = dp.synthesize(_spec(s, epsilon=1.0, spec_is_public=True), tables=t, write=False, evaluate=False)
    assert set(res.data) == {"customer", "product", "sales"}
    assert 0 < res.epsilon <= 1.0 + 1e-9 and res.rho <= res.config.rho * (1 + 1e-9)
    st = res.statement
    assert st["guarantee"]["epsilon"] and st["unit"] == "customer" and st["adjacency"] == "add_remove"
    assert "the spec was not declared public" not in " ".join(st["hazards"])
    assert all(c.rho >= 0 for c in res.charges) and {c.stage for c in res.charges} <= {"discovery", "fit"}
    pv = dp.manifest_privacy(res, "01RUN", {"seq": 1, "domain_rho": "0.1", "domain_delta": "0", "domain_epsilon": "1"})
    assert pv["mode"] == "dp" and pv["_run_id"] == "01RUN" and pv["charges"][0]["kind"] == "zcdp"


def test_domain_override_uses_the_domains_unit_and_delta(star):
    _, s = star

    class Pol:
        domains = {"people": {"unit": {"entity": "customer"}}}

        def caps(self, d):
            class C:
                delta = 1e-10
            return C

    o = dp.domain_override(_spec(s), Pol(), "people", {"epsilon": 1.0})
    assert o == {"epsilon": 1.0, "unit": "customer", "delta": 1e-10, "levels_delta_share": 0.01}
    Pol.domains = {"people": {"unit": {"entity": "product"}}}
    with pytest.raises(PrivacyError, match="privacy unit of domain"):
        dp.domain_override(_spec(s), Pol(), "people")


def test_cli_refuses_tune_and_export_model_on_a_dp_spec(star, tmp_path, capsys):
    import yaml

    from relsynth.cli import main
    _, s = star
    raw = copy.deepcopy(s)
    raw.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {"unit": "customer", "epsilon": 1, "delta": 1e-9}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    for argv in (["tune", str(p)], ["export-model", str(p)], ["verify", str(p)]):
        assert main(argv) == 2, argv
        assert "outside the DP accounting" in capsys.readouterr().err

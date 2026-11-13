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


def test_config_takes_one_unit_or_a_list(star):
    _, s = star
    c = dp.parse(_spec(s))
    assert c.unit == "customer" and c.units == ["customer"] and not c.multi
    for bad, msg in ((["customer", "customer"], "distinct"), (["customer", "nobody"], "not a table"), ([], "name")):
        with pytest.raises(PrivacyError, match=msg):
            dp.parse(_spec(s, unit=bad))


def test_pushdown_setting_is_parsed_and_picks_where_stage1_reads(star, star_sqlite, star_csv):
    """G2: auto (default) reads in the database when the source has SQL, on refuses
    files without DuckDB, off and tables given read in memory."""
    from relsynth.dp import rows as RW
    _, s = star
    assert dp.parse(_spec(s)).pushdown == "auto"
    assert dp.parse(_spec(s, pushdown=False)).pushdown == "off"
    with pytest.raises(PrivacyError, match="pushdown"):
        dp.parse(_spec(s, pushdown="always"))
    lite = _spec(star_sqlite[0])
    src, why = RW.source_for(lite, "auto")
    try:
        assert src.pushdown and why is None and src.unit("customer").n == len(S.SCENARIOS["S2_star"]()[0]["customer"])
    finally:
        src.close()
    src, why = RW.source_for(lite, "off")
    assert not src.pushdown and "off" in why
    src, why = RW.source_for(lite, "auto", tables=star[0])
    assert not src.pushdown and "memory" in why
    files = copy.deepcopy(star_csv[0])
    files["source"] = {**files["source"], "engine": "pandas"}
    files = _spec(files)
    src, why = RW.source_for(files, "auto")
    assert not src.pushdown and "SQL" in why
    with pytest.raises(PrivacyError, match="pushdown: on"):
        RW.source_for(files, "on")


def test_seed_keeps_a_row_when_every_row_breaks_a_rule(star):
    """A small seed where every order breaks a check rule (the dp-eval audit's crash):
    the seed keeps the row that breaks the fewest and re-attaches one child, so
    stage 2 has tables to fit instead of failing on an empty fact."""
    from relsynth.dp import relational as R
    from relsynth.pipeline import execute
    t, s = star
    s = copy.deepcopy(s)
    s["rules"] = list(s.get("rules") or []) + [{"table": "customer", "type": "check", "expr": "age < 0"},
                                               {"table": "sales", "type": "check", "expr": "quantity < 0"}]
    spec = _spec(s)
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    res = {k: v.copy() for k, v in t.items()}
    R._enforce_checks(spec, pl, res)
    assert len(res["customer"]) == 1 and len(res["sales"]) == 1
    assert res["sales"]["customer_id"].isin(res["customer"]["customer_id"]).all()
    s2 = dp.stage2_spec(spec)
    s2["rules"] = [r for r in s2["rules"] if r["expr"] not in ("age < 0", "quantity < 0")]
    run = execute(s2, tables=res, out=False, write=False, seed=1, evaluate_data=False, privacy=False)
    assert not run.exceptions, run.exceptions


def test_cli_reads_several_units_separated_by_commas():
    from types import SimpleNamespace

    from relsynth.cli import _dp_override
    assert _dp_override(SimpleNamespace(epsilon=1, dp_unit="A, B", dp_measure=None))["unit"] == ["A", "B"]
    assert _dp_override(SimpleNamespace(epsilon=None, dp_unit="A", dp_measure=None)) == {"unit": "A"}
    from relsynth.spec import SpecError
    with pytest.raises(SpecError, match="--dp-unit"):
        _dp_override(SimpleNamespace(epsilon=None, dp_unit=" , ", dp_measure=None))


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


def test_a_child_table_keeps_a_row_when_every_parent_draws_none(star, monkeypatch):
    """A noisy degree histogram of a small table can give every parent degree 0.  The
    seed keeps one row, as a root table keeps max(n_hat, 1), so stage 2 has a table to fit."""
    pytest.importorskip("opendp")
    import numpy as np

    import relsynth.dp.relational as R
    real_fit = R.fit

    def fit(*a, **kw):
        model = real_fit(*a, **kw)
        model.tables["customer"].marg["#sales"].draw = lambda z, rng: np.zeros(len(z))
        return model
    seeds = []
    monkeypatch.setattr(dp, "fit", fit)
    monkeypatch.setattr(dp, "decode", lambda spec, model: seeds.append(R.decode(spec, model)) or seeds[-1])
    t, s = star
    dp.synthesize(_spec(s, epsilon=1.0), tables=t, write=False, evaluate=False)
    assert len(seeds[0]["sales"]) == 1 and len(seeds[0]["customer"]) > 1


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


def test_cli_refuses_doctor_apply_on_a_dp_spec(star_sqlite, tmp_path, capsys):
    """doctor --apply would write spec values learnt from real data, so a DP spec is refused
    (docs/dp.md, "What DP mode refuses"); without --apply the report is still written."""
    import yaml

    import relsynth as rs
    from relsynth.cli import main
    from relsynth.diagnostics import write_bundle
    spec, _ = star_sqlite
    write_bundle(dict(rs.run(spec, write=False, privacy=False).bundle), tmp_path / "bundle.json")
    raw = copy.deepcopy(spec)
    raw.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {"unit": "customer", "epsilon": 1, "delta": 1e-9}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    out = tmp_path / "fixed.yaml"
    assert main(["doctor", str(tmp_path / "bundle.json"), "--spec", str(p), "--apply", "--out", str(out)]) == 2
    assert "doctor --apply reads real data outside the DP accounting" in capsys.readouterr().err
    assert not out.exists()
    assert main(["doctor", str(tmp_path / "bundle.json"), "--spec", str(p)]) == 0


def test_stage2_keeps_the_near_copy_guard_on_the_seed(star):
    """docs/privacy.md (small dimensions): in DP mode the near-copy guard runs in
    stage 2, which is fitted on the seed database only, so it is post-processing."""
    _, s = star
    spec = _spec(s)
    spec["synthesis"]["privacy"]["near_copy"] = "auto"
    s2 = dp.stage2_spec(spec)
    assert s2["synthesis"]["privacy"]["near_copy"] == "auto" and "dp" not in s2["synthesis"]["privacy"]


def test_dp_guard_reads_the_seed_rows_not_the_real_ones(star, monkeypatch):
    """In DP mode every near-copy guard is built from the decoded seed database:
    the rows it receives are the seed's, never the real tables."""
    pytest.importorskip("opendp")
    import relsynth.nearcopy as nc
    t, s = star
    seeds, seen = {}, []
    real_decode = dp.decode

    def spy_decode(spec, model):
        db = real_decode(spec, model)
        seeds.update(db)
        return db
    real_init = nc.NearCopyGuard.__init__

    def spy_init(self, real, tspec, cols, rho=1.0):
        seen.append((real, list(cols)))
        real_init(self, real, tspec, cols, rho)
    monkeypatch.setattr(dp, "decode", spy_decode)
    monkeypatch.setattr(nc.NearCopyGuard, "__init__", spy_init)
    spec = _spec(s, epsilon=1.0, spec_is_public=True)
    spec["synthesis"]["privacy"]["near_copy"] = "auto"
    dp.synthesize(spec, tables=t, write=False, evaluate=False)
    assert seen, "the scenario must build at least one guard"
    for real, cols in seen:
        match = [n for n, df in seeds.items() if len(df) == len(real) and set(cols) <= set(df.columns)
                 and all(sorted(map(str, df[c])) == sorted(map(str, real[c])) for c in cols)]
        assert match, "a guard was built from rows that are not a seed table"
        orig = [n for n in match if n in t and len(t[n]) == len(real)
                and all(sorted(map(str, t[n][c])) == sorted(map(str, real[c])) for c in cols)]
        assert not orig, "a guard was built from the real rows"

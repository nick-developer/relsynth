# SPDX-License-Identifier: Apache-2.0
"""Scenarios (relsynth.scenarios, Props. 1824-1828): interventions applied
while a dataset is sampled, recorded in the logs, and nothing changed when no
scenario is active."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.engine import Synthesizer
from relsynth.spec import SpecError, load_spec
from relsynth.timeseries.calendar import to_days

SCEN = {
    "grow": {"description": "more customers", "interventions": [{"type": "scale", "table": "Customer", "factor": 1.5}]},
    "mix": {"interventions": [{"type": "shares", "table": "Customer", "column": "Segment", "shares": {"Gov": 0.4}}]},
    "mixf": {"interventions": [{"type": "shares", "table": "Customer", "column": "Segment", "factors": {"SMB": 0.5}}]},
    "price": {"interventions": [{"type": "marginal", "table": "Orders", "column": "Amount",
                                 "quantiles": [[0.5, 1.0], [1.0, 1.5]]}]},
    "peak": {"interventions": [{"type": "intensity", "table": "Orders", "column": "OrderDate",
                                "start": "2021-01-01", "end": "2021-07-01",
                                "factor": 3.0}]},
    "lines": {"interventions": [{"type": "scale", "table": "Line", "factor": 2.5}]},
    "fewer": {"interventions": [{"type": "degree", "table": "Line", "key": "OrderKey", "rate": 0.5}]},
    "shipwin": {"interventions": [{"type": "intensity", "table": "Line", "column": "ShipDate", "start": "2021-01-01",
                                   "end": "2022-01-01", "factor": 2.0}]},
    "conc": {"interventions": [{"type": "degree", "table": "Line", "key": "ProductKey", "concentration": 3.0}]},
    "spread": {"interventions": [{"type": "degree", "table": "Line", "key": "ProductKey", "concentration": 0.0}]},
    "qty": {"interventions": [{"type": "marginal", "table": "Line", "column": "Qty", "multiply": 2}]},
}


def tables(seed=0, n_cust=400):
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({"CustomerKey": np.arange(1, n_cust + 1),
                         "Segment": rng.choice(["SMB", "Ent", "Gov"], n_cust, p=[0.6, 0.3, 0.1])})
    cust["Tier"] = np.where(cust["Segment"] == "Ent", rng.choice(["a", "b"], n_cust, p=[0.9, 0.1]),
                            rng.choice(["a", "b"], n_cust, p=[0.2, 0.8]))
    n = rng.poisson(np.where(cust["Segment"] == "Ent", 20, 6))
    ck = np.repeat(cust["CustomerKey"], n).to_numpy()
    d = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1095, len(ck)), "D")
    orders = pd.DataFrame({"OrderKey": np.arange(1, len(ck) + 1), "CustomerKey": ck, "OrderDate": d.date,
                           "Amount": np.round(rng.gamma(2, 30, len(ck)), 2)})
    prod = pd.DataFrame({"ProductKey": np.arange(1, 81), "Cat": rng.choice(["x", "y"], 80)})
    m = 3 * len(orders)
    pk = np.minimum(rng.zipf(1.6, m), 80)
    ok = rng.choice(orders["OrderKey"], m)
    sd = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1095, m), "D")
    line = pd.DataFrame({"OrderKey": ok, "ProductKey": pk, "Qty": rng.integers(1, 6, m), "ShipDate": sd.date})
    return {"Customer": cust, "Orders": orders, "Product": prod, "Line": line}


def spec(scenario=None, seq=True, scenarios=True, frailty="auto"):
    sp = {"version": 1, "tables": {
        "Customer": {"primary_key": "CustomerKey", "columns": {"CustomerKey": "id", "Segment": "categorical",
                                                               "Tier": "categorical"}},
        "Orders": {"primary_key": "OrderKey", "columns": {"OrderKey": "id", "CustomerKey": "id", "OrderDate": "date",
                                                          "Amount": "float"}},
        "Product": {"primary_key": "ProductKey", "columns": {"ProductKey": "id", "Cat": "categorical"}},
        "Line": {"role": "fact", "columns": {"OrderKey": "id", "ProductKey": "id", "Qty": "integer", "ShipDate": "date"},
                 "fact": {"anchor": "Orders"}}},
        "relationships": [{"child": "Orders", "child_key": "CustomerKey", "parent": "Customer"},
                          {"child": "Line", "child_key": "OrderKey", "parent": "Orders"},
                          {"child": "Line", "child_key": "ProductKey", "parent": "Product"}],
        "synthesis": {"seed": 3, "gibbs": {"iterations": 20, "final_iterations": 10},
                      "facts": {"frailty": frailty}}}
    if seq:
        sp["tables"]["Orders"]["sequence"] = {"entity": "CustomerKey", "time": "OrderDate"}
    if scenarios:
        sp["synthesis"]["scenarios"] = SCEN
    if scenario:
        sp["synthesis"]["scenario"] = scenario
    return sp


@pytest.fixture(scope="module")
def fitted():
    tabs = tables()
    S = Synthesizer(load_spec(spec())).fit({k: v.copy() for k, v in tabs.items()})
    return S, tabs, S.sample(index=0)


def frames_equal(a, b):
    assert sorted(a) == sorted(b)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])


def test_no_active_scenario_changes_nothing(fitted):
    """A spec with scenarios defined but none active gives the datasets of the
    spec without them (and the scenario name None)."""
    S, tabs, base = fitted
    plain = Synthesizer(load_spec(spec(scenarios=False))).fit({k: v.copy() for k, v in tabs.items()})
    frames_equal(base, plain.sample(index=0))
    assert "scenario" not in S.sample_log
    assert set(S.log["scenarios"]) == set(SCEN)
    assert S.log["scenarios"]["grow"]["description"] == "more customers"


def test_scale_resizes_entities_and_children_follow(fitted):
    S, _, base = fitted
    d = S.sample(index=0, scenario="grow")
    assert len(d["Customer"]) == 600
    assert 1.3 < len(d["Orders"]) / len(base["Orders"]) < 1.7
    eff = S.sample_log["scenario"]["interventions"][0]["effect"]
    assert eff["scale"] == 1.5 and S.sample_log["scenario"]["name"] == "grow"
    # the scenario is per call: the next dataset is the baseline again
    frames_equal(base, S.sample(index=0))


def test_target_shares_are_exact_and_conditionals_kept(fitted):
    """Prop. 1825: stratified rejection hits the quotas and keeps the law of the
    other columns given the tilted one (Tier given Segment)."""
    S, _, base = fitted
    d = S.sample(index=0, scenario="mix")
    sh = d["Customer"]["Segment"].value_counts(normalize=True)
    assert abs(sh["Gov"] - 0.4) < 1e-9
    assert abs(sh["SMB"] / sh["Ent"] - 0.6 / 0.3) < 0.6
    for seg in ("SMB", "Ent"):
        pb = (base["Customer"].query("Segment == @seg")["Tier"] == "a").mean()
        pd_ = (d["Customer"].query("Segment == @seg")["Tier"] == "a").mean()
        assert abs(pb - pd_) < 0.2
    # customers of a tilted-up segment keep their order rate: Gov share of orders rises
    o = d["Orders"].merge(d["Customer"], on="CustomerKey")
    ob = base["Orders"].merge(base["Customer"], on="CustomerKey")
    assert (o["Segment"] == "Gov").mean() > 2.5 * (ob["Segment"] == "Gov").mean()


def test_share_factors_tilt_and_keep_the_total(fitted):
    S, _, base = fitted
    d = S.sample(index=0, scenario="mixf")
    assert len(d["Customer"]) == len(base["Customer"])
    p0 = (base["Customer"]["Segment"] == "SMB").mean()
    p1 = (d["Customer"]["Segment"] == "SMB").mean()
    expect = 0.5 * p0 / (0.5 * p0 + 1 - p0)                   # exponential tilting
    assert abs(p1 - expect) < 0.07


def test_marginal_map_keeps_ranks(fitted):
    """Prop. 1826: the same rows (common random numbers) with the mapped values;
    ranks, and so the copula, unchanged; the top half scaled up."""
    S, _, base = fitted
    d = S.sample(index=0, scenario="price")
    a, b = base["Orders"]["Amount"].to_numpy(), d["Orders"]["Amount"].to_numpy()
    assert len(a) == len(b)
    assert (np.argsort(a, kind="stable") == np.argsort(b, kind="stable")).mean() > 0.999
    assert np.isclose(np.median(b), np.median(a), rtol=0.02)
    assert np.quantile(b, 0.99) > 1.35 * np.quantile(a, 0.99)
    assert (b >= a - 1e-9).all()
    eff = S.sample_log["scenario"]["interventions"][0]["effect"]
    assert eff["mean"] > eff["mean_without"]


def test_intensity_window_on_a_sequence_table(fitted):
    """Prop. 1827 (sequences): three times the rate in the window, the count
    raised by the integrated change."""
    S, _, base = fitted
    d = S.sample(index=0, scenario="peak")

    def inw(df):
        x = to_days(df["OrderDate"])
        lo, hi = to_days(pd.Series(["2021-01-01", "2021-07-01"]))
        return ((x >= lo) & (x < hi)).sum()
    g = S.sample_log["scenario"]["interventions"][0]["effect"]["count_factor"]
    assert 1.2 < g < 1.6
    assert 1.15 < len(d["Orders"]) / len(base["Orders"]) < 1.65
    assert 2.0 < inw(d["Orders"]) / inw(base["Orders"]) < 4.0
    out_b, out_d = len(base["Orders"]) - inw(base["Orders"]), len(d["Orders"]) - inw(d["Orders"])
    assert 0.8 < out_d / out_b < 1.25


def test_fact_rate_superposition_and_thinning(fitted):
    S, _, base = fitted
    up = S.sample(index=0, scenario="lines")
    assert 2.2 < len(up["Line"]) / len(base["Line"]) < 2.8
    assert S.sample_log["scenario"]["interventions"][0]["effect"]["copies"] == 3
    down = S.sample(index=0, scenario="fewer")
    assert 0.42 < len(down["Line"]) / len(base["Line"]) < 0.58


def test_intensity_window_on_a_fact_date_column(fitted):
    S, _, base = fitted
    d = S.sample(index=0, scenario="shipwin")
    lo, hi = to_days(pd.Series(["2021-01-01", "2022-01-01"]))

    def split(df):
        x = to_days(df["ShipDate"])
        w = (x >= lo) & (x < hi)
        return w.sum(), (~w).sum()
    (ib, ob), (id_, od) = split(base["Line"]), split(d["Line"])
    assert 1.7 < id_ / ib < 2.3 and 0.85 < od / ob < 1.15


def test_degree_concentration_tempers_parent_weights():
    """Prop. 1828: w^c within segments; c > 1 concentrates links on fewer
    products, c = 0 spreads them evenly; segment totals unchanged."""
    tabs = tables(1)
    S = Synthesizer(load_spec(spec(frailty="on"))).fit({k: v.copy() for k, v in tabs.items()})
    from relsynth.scenarios import _gini
    g = {}
    for sc in (None, "conc", "spread"):
        d = S.sample(index=0, scenario=sc)
        c = d["Line"]["ProductKey"].value_counts().reindex(d["Product"]["ProductKey"], fill_value=0).to_numpy()
        g[sc] = _gini(c)
        if sc:
            eff = S.sample_log["scenario"]["interventions"][0]["effect"]
            assert (eff["gini"] > eff["gini_without"]) == (sc == "conc")
    assert g["conc"] > g[None] > g["spread"]


def test_validation_errors():
    def errs(iv, extra=None, rules=None):
        sp = spec(scenarios=False, seq=False)
        sp["synthesis"]["scenarios"] = {"s": {"interventions": [iv]}}
        if extra:
            sp["synthesis"].update(extra)
        if rules:
            sp["rules"] = list(sp.get("rules") or []) + rules
        with pytest.raises(SpecError) as e:
            load_spec(sp)
        return str(e.value)
    assert "type must be one of" in errs({"type": "grow", "table": "Customer"})
    assert "factor must be a number > 0" in errs({"type": "scale", "table": "Customer", "factor": 0})
    assert "categorical or boolean" in errs({"type": "shares", "table": "Customer", "column": "CustomerKey",
                                             "factors": {"a": 1}})
    assert "exactly one of" in errs({"type": "shares", "table": "Customer", "column": "Segment"})
    assert "sum to at most 1" in errs({"type": "shares", "table": "Customer", "column": "Segment",
                                       "shares": {"SMB": 0.8, "Gov": 0.5}})
    assert "entity tables" in errs({"type": "shares", "table": "Line", "column": "Qty", "factors": {"1": 2}})
    assert "multiply, add or quantiles" in errs({"type": "marginal", "table": "Orders", "column": "Amount"})
    assert "add (days) only" in errs({"type": "marginal", "table": "Orders", "column": "OrderDate", "multiply": 2})
    assert "increasing" in errs({"type": "marginal", "table": "Orders", "column": "Amount",
                                 "quantiles": [[0.9, 1], [0.1, 2]]})
    assert "start < end" in errs({"type": "intensity", "table": "Orders", "column": "OrderDate",
                                  "start": "2021-02-01", "end": "2021-01-01", "factor": 2})
    assert "foreign key" in errs({"type": "degree", "table": "Line", "key": "Qty", "concentration": 2})
    assert "on a fact table" in errs({"type": "degree", "table": "Orders", "key": "CustomerKey", "rate": 2})
    assert "unknown key" in errs({"type": "scale", "table": "Customer", "factor": 2, "factr": 1})
    assert "not in the spec" in errs({"type": "scale", "table": "Nope", "factor": 2})
    assert "change its inputs" in errs({"type": "marginal", "table": "Orders", "column": "Amount", "multiply": 2},
                                       rules=[{"table": "Orders", "type": "derive", "expr": "Amount = 2 * 1"}])
    assert "is a key" in errs({"type": "marginal", "table": "Line", "column": "OrderKey", "multiply": 2})
    assert "aggregate rule" in errs({"type": "marginal", "table": "Orders", "column": "Amount", "multiply": 2},
                                    rules=[{"table": "Orders", "type": "aggregate", "column": "Amount", "agg": "sum",
                                            "from": "Line", "of": "Qty"}])
    assert "is not one of synthesis.scenarios" in errs({"type": "scale", "table": "Customer", "factor": 2},
                                                       {"scenario": "other"})


def test_api_and_run_select_a_scenario(tmp_path):
    tabs = tables(2, n_cust=150)
    src = tmp_path / "src"
    src.mkdir()
    for t, df in tabs.items():
        df.to_csv(src / f"{t}.csv", index=False)
    sp = spec(seq=False)
    sp["source"] = {"kind": "csv", "path": str(src)}
    m = rs.fit(sp)
    d = m.sample(dataset=0, scenario="grow")
    assert len(d["Customer"]) == 225 and d.sample_log["scenario"]["name"] == "grow"
    with pytest.raises(SpecError, match="not defined"):
        m.sample(dataset=0, scenario="nope")
    r = rs.run(sp, write=False, privacy=False, privacy_check=False, scenario="grow")
    assert len(r.data["Customer"]) == 225


def test_marginal_map_on_a_derived_column():
    """A column derived after the rows are complete (ratio_of, offset_from, a
    derive rule) is mapped once on the finished table; ranks kept."""
    from relsynth.scenarios import Scenario
    sc = object.__new__(Scenario)
    iv = {"type": "marginal", "table": "Lines", "column": "NetPrice", "quantiles": [[0.9, 1.0], [1.0, 1.2]]}
    rec = {"effect": {}}
    sc.maps, sc._mapped = {"Lines": [("NetPrice", iv, rec)]}, set()
    x = np.random.default_rng(0).lognormal(size=5000)
    out = {"Lines": pd.DataFrame({"NetPrice": x})}
    sc.map_columns("Lines", pd.DataFrame({"Other": [1.0]}))     # not there while drawing
    sc.map_derived(out)
    y = out["Lines"]["NetPrice"].to_numpy()
    assert (np.argsort(x, kind="stable") == np.argsort(y, kind="stable")).all()
    assert np.isclose(np.median(y), np.median(x))
    assert 1.1 < np.quantile(y, 0.995) / np.quantile(x, 0.995) <= 1.2
    assert "applied" in rec["effect"]


def test_fact_map_once_on_the_finished_rows_and_in_partitioned_runs(fitted):
    """A fact's marginal map acts on its finished rows (the same rows as the
    baseline), and a partitioned run maps the joined table once, not once per
    partition and again at the end."""
    from relsynth.runtime.partitioned import sample_partitioned
    S, _, base = fitted
    d = S.sample(index=0, scenario="qty")
    assert np.array_equal(d["Line"]["Qty"].to_numpy(), 2 * base["Line"]["Qty"].to_numpy())
    keep = S.cfg.get("scenario")
    pb = sample_partitioned(S, None, 0, executor={"executor": "local"}, partition_rows=400)[0]
    S.cfg["scenario"] = "qty"
    try:
        ps = sample_partitioned(S, None, 0, executor={"executor": "local"}, partition_rows=400)[0]
    finally:
        S.cfg["scenario"] = keep
    assert len(ps["Line"]) == len(pb["Line"])
    assert np.array_equal(ps["Line"]["Qty"].to_numpy(), 2 * pb["Line"]["Qty"].to_numpy())


def test_marginal_on_the_sequence_time_is_rejected():
    sp = spec(scenarios=False)
    sp["synthesis"]["scenarios"] = {"s": {"interventions": [
        {"type": "marginal", "table": "Orders", "column": "OrderDate", "add": 30}]}}
    with pytest.raises(SpecError, match="sequence time"):
        load_spec(sp)


def test_marginal_map_follows_the_rule_redraws():
    """The map acts after the check-rule redraws: every finished value is mapped
    (none is replaced by an unmapped redraw), and the rows the scenario pushes past
    the rule's bound are counted, not redrawn."""
    tabs = tables(4, n_cust=200)
    tabs["Orders"]["Amount"] = tabs["Orders"]["Amount"].clip(upper=100)
    sp = spec(seq=False)
    sp["rules"] = [{"table": "Orders", "type": "check", "expr": "Amount <= 100"}]
    sp["synthesis"]["scenarios"]["double"] = {"interventions": [
        {"type": "marginal", "table": "Orders", "column": "Amount", "multiply": 2}]}
    S = Synthesizer(load_spec(sp)).fit({k: v.copy() for k, v in tabs.items()})
    base = S.sample(index=0)
    d = S.sample(index=0, scenario="double")
    a, b = base["Orders"]["Amount"].to_numpy(), d["Orders"]["Amount"].to_numpy()
    assert (a <= 100 + 1e-9).all()
    assert np.allclose(b, 2 * a)
    eff = S.sample_log["scenario"]["interventions"][0]["effect"]
    assert eff["rule_violations"] == int((b > 100).sum()) > 0


def test_header_date_intensity_acts_on_the_header():
    """An intensity window on an order date of a fact split into header and lines
    tilts the headers (orders), and the lines follow: more orders in the window."""
    rng = np.random.default_rng(5)
    cust = pd.DataFrame({"CK": np.arange(1, 301)})
    no = rng.poisson(4, 300)
    so = np.arange(1, no.sum() + 1)
    od = (pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1095, len(so)), "D")).date
    nl = rng.integers(1, 4, len(so))
    line = pd.DataFrame({"SO": np.repeat(so, nl), "CK": np.repeat(np.repeat(cust["CK"], no).to_numpy(), nl),
                         "OrderDate": np.repeat(od, nl), "Qty": rng.integers(1, 6, nl.sum())})
    sp = {"version": 1, "tables": {
        "Cust": {"primary_key": "CK", "columns": {"CK": "id"}},
        "Line": {"role": "fact", "columns": {"SO": "id", "CK": "id", "OrderDate": "date", "Qty": "integer"},
                 "fact": {"anchor": "Cust", "header": {"key": "SO", "columns": ["CK", "OrderDate"]}}}},
        "relationships": [{"child": "Line", "child_key": "CK", "parent": "Cust"}],
        "synthesis": {"seed": 2, "gibbs": {"iterations": 10, "final_iterations": 5}, "scenarios": {
            "win": {"interventions": [{"type": "intensity", "table": "Line", "column": "OrderDate",
                                       "start": "2021-01-01", "end": "2022-01-01", "factor": 2.0}]}}}}
    S = Synthesizer(load_spec(sp)).fit({"Cust": cust, "Line": line})
    base, d = S.sample(index=0), S.sample(index=0, scenario="win")
    lo, hi = to_days(pd.Series(["2021-01-01", "2022-01-01"]))

    def orders(df):
        o = df.drop_duplicates("SO")
        x = to_days(o["OrderDate"])
        assert np.isfinite(x).all()
        w = (x >= lo) & (x < hi)
        return w.sum(), (~w).sum()
    (ib, ob), (id_, od_) = orders(base["Line"]), orders(d["Line"])
    assert 1.6 < id_ / ib < 2.4 and 0.85 < od_ / ob < 1.15


def test_sequence_windows_give_one_combined_gain(fitted):
    """Two windows on one sequence table: one integrated gain over both (what
    the sequence model applies), not the product of the separate gains."""
    from relsynth.scenarios import Scenario
    S, _, _ = fitted
    wins = [{"type": "intensity", "table": "Orders", "column": "OrderDate", "start": s_, "end": e_, "factor": 2.0}
            for s_, e_ in (("2020-01-01", "2020-07-01"), ("2021-01-01", "2021-07-01"))]
    sc = Scenario(S, "two", {"interventions": wins})
    m = S.seq["Orders"]["model"]
    one = [Scenario._seq_gain(m, [w[:3]]) for w in sc.seq_windows["Orders"]]
    both = Scenario._seq_gain(m, [w[:3] for w in sc.seq_windows["Orders"]])
    assert [r["effect"]["count_factor"] for r in sc.records] == [round(both, 6)] * 2
    assert both < one[0] * one[1]
    assert np.isclose(sc.scale_ent["Orders"], both)


def test_partitioned_rows_skip_the_sequence_count_record():
    from relsynth.scenarios import Scenario
    sc = object.__new__(Scenario)
    rec = {"effect": {}}
    sc.fact_rate, sc.fact_windows = {"F": [(1.3, None), (2.0, rec)]}, {}
    sc.note_fact_rows("F", 42)
    assert rec["effect"]["rows"] == 42


def test_share_tilts_are_scale_free_and_shares_must_be_allocatable(fitted):
    S, _, base = fitted
    keep = S.cfg["scenarios"] if "scenarios" in S.cfg else None
    S.cfg.setdefault("scenarios", {})
    try:
        S.cfg["scenarios"]["small"] = {"interventions": [{"type": "shares", "table": "Customer", "column": "Segment",
                                                          "factors": {"SMB": 0.001, "Ent": 0.001, "Gov": 0.001}}]}
        S.cfg["scenarios"]["noroom"] = {"interventions": [{"type": "shares", "table": "Customer", "column": "Segment",
                                                           "shares": {"SMB": 0.2, "Ent": 0.2, "Gov": 0.2}}]}
        d = S.sample(index=0, scenario="small")
        assert len(d["Customer"]) == len(base["Customer"])
        with pytest.raises(SpecError, match="remaining"):
            S.sample(index=0, scenario="noroom")
    finally:
        S.cfg["scenarios"].pop("small", None)
        S.cfg["scenarios"].pop("noroom", None)
        if keep is None and not S.cfg["scenarios"]:
            S.cfg.pop("scenarios")


def test_states_loglik_matrix_form():
    from scipy.special import gammaln, logsumexp

    from relsynth.timeseries.states import GH_LW, GH_X, StateModel
    rng = np.random.default_rng(0)
    m = StateModel()
    m.L, m.ref = 4, 0
    C, G, th = rng.poisson(2, (30, 4)).astype(float) + 1, rng.normal(size=(30, 4)), rng.normal(size=5) * 0.3
    d = np.r_[0.0, th[2:]]
    a = np.exp(np.clip(th[0] + np.exp(th[1]) * GH_X, -12, 8))[None, :, None] * np.exp(d)[None, None, :]
    D = (C[:, None, :] * a).sum(2)
    f = gammaln(D) - (C[:, None, :] * gammaln(a)).sum(2) + (a * G[:, None, :]).sum(2) + GH_LW[None, :]
    assert np.isclose(m._loglik(th, C, G), float(np.sum(logsumexp(f, axis=1))))

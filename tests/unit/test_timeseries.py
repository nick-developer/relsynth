# SPDX-License-Identifier: Apache-2.0
"""Time series and event sequences (relsynth.timeseries, roadmap T12): the
calendar time change, the sequence model and its engine hook, regular series,
metrics, incremental extension and the spec keys (Props. 1200-1206)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

import relsynth as rs
from relsynth.spec import SpecError, load_spec, validate
from relsynth.timeseries.calendar import Intensity, from_days, to_days
from relsynth.timeseries.events import SequenceModel, _Lifetime
from relsynth.timeseries.incremental import Extender, forward_events, hash_uniform
from relsynth.timeseries.metrics import Events, sequence_checks, series_checks, series_matrix
from relsynth.timeseries.series import SeriesModel, _ar_to_pacf, _pacf_to_ar, _seasonal_filter

D0 = float(to_days(pd.Series(["2021-01-01"]))[0])


def renewal(rng, n_ent, L, mu=-0.7, sd=0.3, rate=0.05):
    """Gamma renewal histories over [D0, D0 + L): shape alpha_i lognormal, rates gamma."""
    t, o = [], []
    for i in range(n_ent):
        a = np.exp(mu + sd * rng.standard_normal())
        r = rng.gamma(2.0, rate / 2)
        cur = D0 + rng.uniform(0, L / 2)
        while cur < D0 + L:
            t.append(cur)
            o.append(i)
            cur += rng.gamma(a, 1 / (a * r))
    return np.array(t), np.array(o)


# ------------------------------------------------------------------ calendar
def test_operational_time_round_trip_and_scale():
    rng = np.random.default_rng(1)
    t = D0 + rng.uniform(0, 730, 20000)
    seg = rng.integers(0, 2, 20000)
    I = Intensity().fit(t, seg, 2, D0, D0 + 730)
    u = I.forward(t, seg)
    back = I.inverse(u, seg)
    assert np.allclose(back, t, atol=1e-6)
    # one unit of operational time is one average day (Prop. 1200, normalisation)
    assert np.allclose(I.C[:, -1], I.ndays)
    assert np.all(np.diff(I.C, axis=1) > 0)


def test_time_change_makes_seasonal_poisson_flat():
    """Prop. 1200: a Poisson process with intensity lambda has uniform
    operational times; weekday-modulated events come out flat."""
    rng = np.random.default_rng(2)
    days = np.arange(int(D0), int(D0) + 700)
    w = np.array([1.5, 1.2, 1.0, 1.0, 1.0, 0.8, 0.3])
    lam = w[(days + 3) % 7] * 20
    cnt = rng.poisson(lam)
    t = np.repeat(days, cnt) + rng.random(cnt.sum())
    I = Intensity(weekly=True).fit(t, np.zeros(len(t), int), 1, D0, D0 + 700)
    u = I.forward(t, np.zeros(len(t), int)) / I.C[0, -1]
    assert ks_2samp(u, rng.random(len(u))).pvalue > 1e-3
    assert np.argmin(I.w) == 6 and np.argmax(I.w) == 0


def test_dates_round_trip():
    s = pd.Series(["2020-02-29", "2021-12-31 13:45:00", None], dtype=object)
    d = to_days(s)
    assert np.isnan(d[2])
    out = from_days(d[:2], True)
    assert str(out[1]) == "2021-12-31T13:45:00"
    assert str(from_days(d[:2], False)[1]).startswith("2021-12-31T00:00")


# ------------------------------------------------------------- the sequences
@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    t, o = renewal(rng, 2500, 900.0)
    m = SequenceModel().fit(t, o, np.zeros(2500, int), 1, None, datetime=True, t0=D0, t1=D0 + 900, rng=rng)
    return m, t, o


def test_sequence_model_recovers_rhythm(fitted):
    m, *_ = fitted
    th = m.fit_log["pooled"]["theta"]
    assert abs(th["mu_log_alpha"] - (-0.7)) < 0.15
    assert m.fit_log["pooled"]["converged"]


def test_bridge_is_gamma_renewal_given_endpoints():
    """Prop. 1201: interior gap shares of a gamma renewal given its first and
    last event are Dirichlet(alpha); the sampler draws exactly that law."""
    rng = np.random.default_rng(4)
    a, n = 0.4, 6
    g = rng.gamma(a, 1.0, (20000, n - 1))
    share = (g / g.sum(1, keepdims=True))[:, 0]
    m = SequenceModel()
    m.K, m.datetime = 1, True
    m.intensity = Intensity().fit(D0 + rng.uniform(0, 100, 1000), np.zeros(1000, int), 1, D0, D0 + 100)
    m.theta = np.array([[np.log(a), np.log(1e-6)]])
    m.life = [_Lifetime().fit(np.full(500, n), rng.uniform(0.05, 0.3, 500), rng.uniform(0.5, 0.9, 500))]
    m.t0, m.t1 = D0, D0 + 100
    m.intensity.w = np.ones(7)
    m.intensity.P[:] = 1.0 / m.intensity.B
    m.intensity._build()
    tt, ow = m.sample(np.full(20000, n), np.zeros(20000, int), np.full(20000, D0), np.full(20000, D0 + 100), rng)
    tt = tt.reshape(20000, n)
    u = m.intensity.forward(tt.reshape(-1), np.zeros(tt.size, int)).reshape(20000, n)
    s2 = (u[:, 1] - u[:, 0]) / (u[:, -1] - u[:, 0])
    assert ks_2samp(share, s2).pvalue > 1e-3
    assert np.all(np.diff(tt, axis=1) >= 0)


def test_sampled_gaps_match_real(fitted):
    m, t, o = fitted
    rng = np.random.default_rng(5)
    m.redraw(None)
    n = np.bincount(o, minlength=2500)
    st, so = m.sample(n, np.zeros(2500, int), np.full(2500, D0), np.full(2500, D0 + 900), rng)
    er, es = Events(t, o, D0, D0 + 900), Events(st, so, D0, D0 + 900)
    ch = {c["metric"]: c for c in sequence_checks(er, es, datetime=True, splits=12)}
    assert ch["gap_w1"]["pass"] and ch["burstiness"]["pass"] and ch["span_w1"]["pass"]


def test_posterior_draw_moves_parameters(fitted):
    m, *_ = fitted
    m.redraw(np.random.default_rng(6))
    a = m.theta.copy()
    m.redraw(np.random.default_rng(7))
    assert not np.allclose(a, m.theta)
    m.redraw(None)
    assert np.allclose(m.theta, m.theta_hat)


# ------------------------------------------------------------- engine hook
def shop_tables(seed=0, n_cust=400):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    end = start + pd.to_timedelta(rng.integers(200, 1500, n_cust), "D")
    cust = pd.DataFrame({"CustomerKey": np.arange(1, n_cust + 1), "Tier": rng.choice(["a", "b"], n_cust),
                         "EndDT": end.date})
    t, o = renewal(rng, n_cust, 1095.0, mu=-0.9, rate=0.08)
    d = pd.to_datetime(np.floor(t).astype("int64"), unit="D")
    keep = d.date < cust["EndDT"].to_numpy()[o]
    orders = pd.DataFrame({"OrderKey": np.arange(1, keep.sum() + 1), "CustomerKey": o[keep] + 1,
                           "OrderDate": d[keep].date, "Amount": np.round(rng.gamma(2, 30, keep.sum()), 2)})
    return {"Customer": cust, "Orders": orders}


def shop_spec(tmp_path, seq=True):
    tabs = shop_tables()
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    for t, df in tabs.items():
        df.to_csv(src / f"{t}.csv", index=False)
    spec = {"version": 1, "source": {"kind": "csv", "path": str(src)},
            "tables": {"Customer": {"primary_key": "CustomerKey",
                                    "columns": {"CustomerKey": "id", "Tier": "categorical", "EndDT": "date"}},
                       "Orders": {"primary_key": "OrderKey",
                                  "columns": {"OrderKey": "id", "CustomerKey": "id", "OrderDate": "date",
                                              "Amount": "float"}}},
            "relationships": [{"child": "Orders", "child_key": "CustomerKey", "parent": "Customer"}],
            "rules": [{"table": "Orders", "type": "check", "expr": "days(OrderDate) < days(CustomerKey.EndDT)"}],
            "synthesis": {"seed": 3, "gibbs": {"iterations": 20, "final_iterations": 10}}}
    if seq:
        spec["tables"]["Orders"]["sequence"] = {"entity": "CustomerKey", "time": "OrderDate"}
    return spec, tabs


def test_engine_sequence_keeps_rules_and_other_tables(tmp_path):
    spec, tabs = shop_spec(tmp_path)
    m = rs.fit(spec)
    info = m.synthesizer.seq["Orders"]
    assert info["bounds"] == [("end", "EndDT", True)]             # window read from the check rule
    d = m.sample(dataset=0)
    log = d.sample_log["sequences"]["Orders"]
    assert log["events"] == len(d["Orders"])
    o, c = d["Orders"], d["Customer"]
    end = c.set_index("CustomerKey")["EndDT"].reindex(o["CustomerKey"]).to_numpy()
    assert (pd.to_datetime(o["OrderDate"]).to_numpy() < pd.to_datetime(end)).all()
    # Prop. 1203: everything but the time column is what the run without sequences gives
    base = rs.fit(shop_spec(tmp_path, seq=False)[0]).sample(dataset=0)
    pd.testing.assert_frame_equal(base["Customer"], c)
    assert base["Orders"]["Amount"].tolist() == o["Amount"].tolist()
    assert base["Orders"]["CustomerKey"].tolist() == o["CustomerKey"].tolist()
    # and the histories look like the real ones (more than the 1.0 bucket model)
    er = Events(to_days(tabs["Orders"]["OrderDate"]), tabs["Orders"]["CustomerKey"])
    res = {}
    for name, df in (("seq", o), ("base", base["Orders"])):
        es = Events(to_days(df["OrderDate"]), df["CustomerKey"], er.t0, er.t1)
        res[name] = {x["metric"]: x["value"] for x in sequence_checks(er, es, splits=8)}
    assert res["seq"]["gap_w1"] < res["base"]["gap_w1"]


def test_sequence_on_header_and_calendar_key():
    """A sequence declared on a line fact moves to its order header; a time that
    is a lookup through a calendar key rewrites the key."""
    from relsynth.transform import Plan
    raw = {"tables": {
        "Cal": {"primary_key": "DateKey", "mode": "copy", "columns": {"DateKey": "id", "Date": "date"}},
        "Cust": {"primary_key": "CK", "columns": {"CK": "id"}},
        "Line": {"role": "fact", "columns": {"SO": "id", "CK": "id", "DateKey": "id", "Qty": "integer",
                                             "OrderDate": {"type": "lookup", "lookup": "DateKey.Date"}},
                 "fact": {"anchor": "Cust", "header": {"key": "SO", "columns": ["CK", "DateKey", "OrderDate"]}},
                 "sequence": {"entity": "CK", "time": "OrderDate"}}},
        "relationships": [{"child": "Line", "child_key": "CK", "parent": "Cust"},
                          {"child": "Line", "child_key": "DateKey", "parent": "Cal"}]}
    spec = load_spec(raw)
    assert validate(spec) == []
    p = Plan(spec)
    H = next(iter(p.headers.values()))["table"]
    assert p.spec["tables"][H]["sequence"]["time"] == "OrderDate"
    assert "sequence" not in p.spec["tables"]["Line"]


def test_spec_errors():
    raw = {"tables": {"C": {"primary_key": "k", "columns": {"k": "id"}},
                      "O": {"primary_key": "o", "columns": {"o": "id", "k": "id", "n": "integer"},
                            "sequence": {"entity": "x", "time": "n", "bucket": "fortnight", "redraws": -1}}},
           "relationships": [{"child": "O", "child_key": "k", "parent": "C"}]}
    with pytest.raises(SpecError) as e:
        load_spec(raw)
    for frag in ("sequence.entity x", "sequence.time n", "bucket", "redraws"):
        assert frag in str(e.value)
    raw["tables"]["O"]["sequence"] = {"entity": "k", "time": "n", "windw": "auto"}
    with pytest.raises(SpecError, match="windw"):
        load_spec(raw)


# ------------------------------------------------------------ regular series
def test_pacf_ar_round_trip():
    r = np.array([0.8, -0.3, 0.2])
    assert np.allclose(_ar_to_pacf(_pacf_to_ar(r)), r)


def test_seasonal_filter_keeps_unit_variance():
    rng = np.random.default_rng(8)
    V = rng.standard_normal((50, 24 * 200))
    Y = _seasonal_filter(V, np.full(50, 0.7), 24, rng)
    assert abs(Y[:, 24 * 50:].var() - 1) < 0.05
    lag = np.mean(Y[:, 24:] * Y[:, :-24])
    assert abs(lag - 0.7) < 0.05


def meters(seed=9, n=60, T=24 * 120):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=T, freq="h")
    h = np.arange(T) % 24
    lvl = rng.normal(4, 0.6, n)
    x = lvl[:, None] + 0.6 * np.sin(2 * np.pi * (h - 6) / 24)[None, :]
    f = np.convolve(rng.standard_normal(T + 48), np.ones(48) / 7, "valid")[:T]
    x = x + 0.25 * f[None, :] + 0.15 * rng.standard_normal((n, T))
    Y = np.round(np.expm1(x), 2)
    return pd.DataFrame({"m": np.repeat(np.arange(n), T), "ts": np.tile(ts, n), "kw": Y.reshape(-1)})


def test_series_model_and_tstr():
    df = meters()
    m = SeriesModel().fit(df, "m", "ts", ["kw"], np.random.default_rng(1))
    s = m.sample(60, np.random.default_rng(2))
    assert len(s) == len(df) and (s["kw"] >= 0).all()
    Yr, step = series_matrix(df, "m", "ts", "kw")
    Ys = s["kw"].to_numpy().reshape(60, -1)
    checks, ts = series_checks(Yr, Ys, step, splits=6)
    assert ts["gap"] < 0.25
    assert {c["metric"] for c in checks} == {"acf", "spectrum", "xcorr_w1", "value_w1"}
    m.redraw(np.random.default_rng(3))
    s2 = m.sample(10, np.random.default_rng(2))
    assert not np.allclose(s2["kw"].to_numpy()[:100], s["kw"].to_numpy()[:100])


def test_series_table_through_engine(tmp_path):
    df = meters(n=30, T=24 * 40).rename(columns={"m": "MeterKey", "ts": "Hour"})
    df["MeterKey"] += 1
    src = tmp_path / "src"
    src.mkdir()
    pd.DataFrame({"MeterKey": np.arange(1, 31)}).to_csv(src / "Meter.csv", index=False)
    df.to_csv(src / "Load.csv", index=False)
    spec = {"source": {"kind": "csv", "path": str(src)},
            "tables": {"Meter": {"primary_key": "MeterKey", "columns": {"MeterKey": "id"}},
                       "Load": {"role": "fact", "columns": {"MeterKey": "id", "Hour": "datetime", "kW": "float"},
                                "series": {"entity": "MeterKey", "time": "Hour", "values": ["kw"]}}},
            "relationships": [{"child": "Load", "child_key": "MeterKey", "parent": "Meter"}],
            "synthesis": {"scale": {"default": 2.0}}}
    spec["tables"]["Load"]["columns"] = {"MeterKey": "id", "Hour": "datetime", "kw": "float"}
    d = rs.fit(spec).sample()
    assert len(d["Meter"]) == 60 and len(d["Load"]) == 60 * 24 * 40
    assert set(d["Load"]["MeterKey"]) == set(d["Meter"]["MeterKey"])


# ------------------------------------------------------------- incremental
def test_hash_uniform_is_counter_based():
    u = hash_uniform(7, np.array([1, 2, 3], dtype=np.uint64), np.array([0, 0, 0]), 4)
    v = hash_uniform(7, np.array([3, 2, 1], dtype=np.uint64), np.array([0, 0, 0]), 4)
    assert np.allclose(u, v[::-1]) and ((u > 0) & (u < 1)).all()
    w = hash_uniform(7, np.arange(200000, dtype=np.uint64), np.zeros(200000, int), 1)
    assert ks_2samp(w, np.random.default_rng(0).random(200000)).pvalue > 1e-3


def test_forward_first_gap_conditions_on_age(fitted):
    """Prop. 1206: the first new gap of an entity of age c is Gamma | > c, so
    with alpha < 1 a recently active entity reorders sooner than one that has
    been away (the inspection paradox)."""
    m, *_ = fitted
    m.redraw(None)
    N = 40000
    seg = np.zeros(N, int)
    kh = np.arange(N, dtype=np.uint64)
    n = np.full(N, 5.0)
    U = np.full(N, 300.0)
    waits = {}
    for age in (1.0, 200.0):
        e, j, u = forward_events(m, seg, n, U, np.full(N, 1000.0 - age), np.full(N, 1000.0),
                                 np.full(N, 3000.0), kh, 11)
        first = pd.Series(u).groupby(e).min()
        waits[age] = float(np.median(first - 1000.0))
    assert waits[1.0] < waits[200.0]


def test_extension_is_step_invariant(tmp_path):
    spec, _ = shop_spec(tmp_path)
    m = rs.fit(spec)
    base = m.sample(dataset=0)
    end = pd.to_datetime(base["Orders"]["OrderDate"]).max()
    long = Extender(m, base, 0).run(end + pd.Timedelta(days=90), step="week")
    daily = Extender(m, base, 0).run(end + pd.Timedelta(days=30), step="day")
    a = pd.concat([p["Orders"] for *_, p in long], ignore_index=True)
    b = pd.concat([p["Orders"] for *_, p in daily], ignore_index=True)
    a = a[pd.to_datetime(a["OrderDate"]) < end + pd.Timedelta(days=30)].reset_index(drop=True)
    assert len(b) > 0
    pd.testing.assert_frame_equal(a.astype(str), b.astype(str))
    # new keys continue the base; entity windows hold
    assert a["OrderKey"].astype(int).min() > base["Orders"]["OrderKey"].astype(int).max()
    cend = base["Customer"].set_index("CustomerKey")["EndDT"].reindex(b["CustomerKey"]).to_numpy()
    assert (pd.to_datetime(b["OrderDate"]).to_numpy() < pd.to_datetime(cend)).all()


def test_ir_round_trip_keeps_sequence_and_series():
    from relsynth import ir
    raw = {"version": 1, "tables": {
        "C": {"primary_key": "k", "columns": {"k": "id"}},
        "O": {"primary_key": "o", "columns": {"o": "id", "k": "id", "d": "date"},
              "sequence": {"entity": "k", "time": "d", "bucket": "week"}},
        "M": {"primary_key": "m", "columns": {"m": "id"}},
        "L": {"role": "fact", "columns": {"m": "id", "h": "datetime", "kw": "float"},
              "series": {"entity": "m", "time": "h", "values": ["kw"], "ar": 3}}},
        "relationships": [{"child": "O", "child_key": "k", "parent": "C"},
                          {"child": "L", "child_key": "m", "parent": "M"}]}
    back, lost = ir.lower(ir.loads(ir.dumps(ir.lift(raw))))
    assert lost == []
    assert back["tables"]["O"]["sequence"] == raw["tables"]["O"]["sequence"]
    assert back["tables"]["L"]["series"] == raw["tables"]["L"]["series"]


def test_sequence_runs_in_the_partitioned_runtime(tmp_path):
    """The T12 sequence hook also runs when the facts are drawn by relsynth.runtime
    (T10): times are redrawn over the joined table, so the window rule holds and
    the result does not depend on the number of workers."""
    from relsynth.runtime import ProcessExecutor, sample_partitioned
    spec, _ = shop_spec(tmp_path)
    spec["tables"]["Orders"]["role"] = "fact"
    syn = rs.fit(spec).synthesizer
    assert "Orders" in syn.facts and "Orders" in syn.seq
    ref, _ = sample_partitioned(syn, index=0, executor="local", partition_rows=150)
    assert syn.sample_log["sequences"]["Orders"]["events"] == len(ref["Orders"])
    o, c = ref["Orders"], ref["Customer"]
    end = c.set_index("CustomerKey")["EndDT"].reindex(o["CustomerKey"]).to_numpy()
    assert (pd.to_datetime(o["OrderDate"]).to_numpy() < pd.to_datetime(end)).all()
    got, run = sample_partitioned(syn, index=0, executor=ProcessExecutor(2), partition_rows=150)
    assert any(len(p.blocks) > 1 for p in run.facts)
    for t in ref:
        pd.testing.assert_frame_equal(ref[t], got[t])

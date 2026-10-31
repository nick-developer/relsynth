# SPDX-License-Identifier: Apache-2.0
"""Checks C1200-C1206: the time-series propositions of docs/proofs (section
v3.2, time series), tested against known answers on simulated data.  About
five minutes on 4 cores.

    pytest -m science tests/science/test_timeseries_checks.py
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, ks_2samp

from relsynth.timeseries.calendar import Intensity, to_days
from relsynth.timeseries.events import SequenceModel, _Lifetime
from relsynth.timeseries.incremental import forward_events
from relsynth.timeseries.metrics import Events, sequence_checks, series_checks, series_matrix
from relsynth.timeseries.series import SeriesModel

pytestmark = [pytest.mark.science]

D0 = float(to_days(pd.Series(["2021-01-01"]))[0])


def renewal(rng, n_ent, L, mu=-0.7, sd=0.3, rate=0.05, start=0.5):
    """Gamma renewal histories over [D0, D0 + L): shape alpha_i lognormal, rate
    gamma, each entity acquired uniformly in the first `start` share."""
    t, o = [], []
    for i in range(n_ent):
        a = np.exp(mu + sd * rng.standard_normal())
        r = rng.gamma(2.0, rate / 2)
        cur = D0 + rng.uniform(0, L * start)
        while cur < D0 + L:
            t.append(cur)
            o.append(i)
            cur += rng.gamma(a, 1 / (a * r))
    return np.array(t), np.array(o)


def _flat_model(alpha, L=100.0, n_life=500, n=6, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    m = SequenceModel()
    m.K, m.datetime = 1, True
    m.intensity = Intensity().fit(D0 + rng.uniform(0, L, 1000), np.zeros(1000, int), 1, D0, D0 + L)
    m.intensity.w = np.ones(7)
    m.intensity.P[:] = 1.0 / m.intensity.B
    m.intensity._build()
    m.theta = np.array([[np.log(alpha), np.log(1e-6)]])
    m.life = [_Lifetime().fit(np.full(n_life, n), rng.uniform(0.05, 0.3, n_life), rng.uniform(0.5, 0.9, n_life))]
    m.t0, m.t1 = D0, D0 + L
    return m


def test_C1200_time_change_flattens_seasonal_poisson():
    """Prop. 1200: a Poisson process with weekday, monthly and trend modulation
    has uniform operational times under the fitted Lambda, and the inverse map
    returns the times."""
    rng = np.random.default_rng(1200)
    days = np.arange(int(D0), int(D0) + 1095)
    w = np.array([1.5, 1.2, 1.0, 1.0, 1.0, 0.8, 0.3])
    month = pd.to_datetime(days.astype("int64"), unit="D").month.to_numpy()
    lam = 20 * w[(days + 3) % 7] * (1 + 0.4 * np.cos(2 * np.pi * month / 12)) * (1 + (days - days[0]) / 1095)
    cnt = rng.poisson(lam)
    t = np.repeat(days, cnt) + rng.random(cnt.sum())
    I = Intensity().fit(t, np.zeros(len(t), int), 1, D0, D0 + 1095)
    u = I.forward(t, np.zeros(len(t), int))
    assert np.allclose(I.inverse(u, np.zeros(len(t), int)), t, atol=1e-6)
    assert ks_2samp(u / I.C[0, -1], rng.random(len(u))).pvalue > 1e-3


def test_C1201_bridge_is_gamma_renewal_given_endpoints():
    """Prop. 1201: interior gap shares of a gamma renewal given its first and
    last event are Dirichlet(alpha 1); the sampler draws that law, for bursty,
    Poisson and regular shapes."""
    for a in (0.3, 1.0, 3.0):
        rng = np.random.default_rng(int(a * 100))
        n, N = 6, 20000
        g = rng.gamma(a, 1.0, (N, n - 1))
        real = g / g.sum(1, keepdims=True)
        m = _flat_model(a, rng=rng)
        tt, _ = m.sample(np.full(N, n), np.zeros(N, int), np.full(N, D0), np.full(N, D0 + 100), rng)
        u = m.intensity.forward(tt, np.zeros(tt.size, int)).reshape(N, n)
        share = np.diff(u, axis=1) / (u[:, -1] - u[:, 0])[:, None]
        for j in (0, 2):
            assert ks_2samp(real[:, j], share[:, j]).pvalue > 1e-3


def test_C1202_posterior_recovers_parameters():
    """Prop. 1202: the MAP of (mu, log sigma) is consistent and its Laplace
    interval covers the true mu; the life-time marginals at their posterior
    mean are the empirical quantiles."""
    cover, est = 0, []
    R = 12
    for r in range(R):
        rng = np.random.default_rng(1202 + r)
        t, o = renewal(rng, 1500, 900.0, mu=-0.5, sd=0.3)
        m = SequenceModel(calibration=0).fit(t, o, np.zeros(1500, int), 1, None, datetime=True,
                                             t0=D0, t1=D0 + 900, rng=rng)
        th, cov = m.theta_hat[0], m.theta_cov[0]
        est.append(th[0])
        cover += abs(th[0] - (-0.5)) < 2.576 * np.sqrt(cov[0, 0]) + 0.05
    assert abs(np.mean(est) - (-0.5)) < 0.1
    assert cover >= R - 2
    x = np.random.default_rng(1).beta(2, 5, 4000)
    lf = _Lifetime().fit(np.ones(4000, np.int64), x, np.full(4000, np.nan))
    lf.draw(None)
    assert np.allclose(lf.strata[0]["h"].ppf([0.1, 0.5, 0.9]), np.quantile(x, [0.1, 0.5, 0.9]), atol=5e-3)


def test_C1203_hook_keeps_rules_and_other_tables(tmp_path):
    """Prop. 1203: the sequence hook changes only the times of the sequence
    table, keeps every check rule and leaves the other tables identical."""
    import relsynth as rs
    rng = np.random.default_rng(1203)
    n_c = 500
    start = pd.Timestamp("2020-01-01")
    end = start + pd.to_timedelta(rng.integers(200, 1500, n_c), "D")
    cust = pd.DataFrame({"CustomerKey": np.arange(1, n_c + 1), "Tier": rng.choice(["a", "b"], n_c),
                         "EndDT": end.date})
    t, o = renewal(rng, n_c, 1095.0, mu=-0.9, rate=0.08)
    d = pd.to_datetime(np.floor(t).astype("int64"), unit="D")
    keep = d.date < cust["EndDT"].to_numpy()[o]
    orders = pd.DataFrame({"OrderKey": np.arange(1, keep.sum() + 1), "CustomerKey": o[keep] + 1,
                           "OrderDate": d[keep].date, "Amount": np.round(rng.gamma(2, 30, keep.sum()), 2)})
    src = tmp_path / "src"
    src.mkdir()
    cust.to_csv(src / "Customer.csv", index=False)
    orders.to_csv(src / "Orders.csv", index=False)

    def spec(seq):
        s = {"version": 1, "source": {"kind": "csv", "path": str(src)},
             "tables": {"Customer": {"primary_key": "CustomerKey",
                                     "columns": {"CustomerKey": "id", "Tier": "categorical", "EndDT": "date"}},
                        "Orders": {"primary_key": "OrderKey",
                                   "columns": {"OrderKey": "id", "CustomerKey": "id", "OrderDate": "date",
                                               "Amount": "float"}}},
             "relationships": [{"child": "Orders", "child_key": "CustomerKey", "parent": "Customer"}],
             "rules": [{"table": "Orders", "type": "check", "expr": "days(OrderDate) < days(CustomerKey.EndDT)"}],
             "synthesis": {"seed": 5, "gibbs": {"iterations": 20, "final_iterations": 10}}}
        if seq:
            s["tables"]["Orders"]["sequence"] = {"entity": "CustomerKey", "time": "OrderDate"}
        return s
    for ds in (0, 1):
        a = rs.fit(spec(False)).sample(dataset=ds)
        b = rs.fit(spec(True)).sample(dataset=ds)
        pd.testing.assert_frame_equal(a["Customer"], b["Customer"])
        assert a["Orders"]["CustomerKey"].value_counts().sort_index().equals(
            b["Orders"]["CustomerKey"].value_counts().sort_index())
        e = b["Customer"].set_index("CustomerKey")["EndDT"].reindex(b["Orders"]["CustomerKey"]).to_numpy()
        assert (pd.to_datetime(b["Orders"]["OrderDate"]).to_numpy() < pd.to_datetime(e)).all()


def test_C1204_calibrated_limits_keep_their_level():
    """Prop. 1204: two independent populations of one law fail a check at the
    nominal 1% (binomial bound at 0.1%), and a planted Poisson-for-bursty
    defect fails the gap checks."""
    fails, tot = collections.Counter(), 0
    R = 24
    for r in range(R):
        rng = np.random.default_rng(1204 + r)
        t, o = renewal(rng, 1200, 900.0)
        t2, o2 = renewal(rng, 1200, 900.0)
        for c in sequence_checks(Events(np.floor(t), o, D0, D0 + 900), Events(np.floor(t2), o2, D0, D0 + 900),
                                 splits=20, seed=r):
            if c["pass"] is not None:
                tot += 1
                fails[c["metric"]] += not c["pass"]
    assert sum(fails.values()) <= binom.isf(0.001, tot, 0.01)
    rng = np.random.default_rng(1)
    t, o = renewal(rng, 1500, 900.0, mu=-1.2)
    t2, o2 = renewal(rng, 1500, 900.0, mu=0.0)          # Poisson gaps for a bursty population
    ch = {c["metric"]: c for c in sequence_checks(Events(t, o, D0, D0 + 900), Events(t2, o2, D0, D0 + 900),
                                                  datetime=True, splits=20)}
    assert not ch["burstiness"]["pass"] and not ch["gap_ks"]["pass"]


def test_C1204b_fitted_sequences_pass_their_checks():
    """Prop. 1204 applied: the fitted model of a growing bursty population
    passes every calibrated check (inter-arrival, life time, calendar)."""
    rng = np.random.default_rng(12045)
    t, o = renewal(rng, 2500, 900.0)
    m = SequenceModel().fit(t, o, np.zeros(2500, int), 1, None, datetime=True, t0=D0, t1=D0 + 900, rng=rng)
    m.redraw(None)
    n = np.bincount(o, minlength=2500)
    st, so = m.sample(n, np.zeros(2500, int), np.full(2500, D0), np.full(2500, D0 + 900), np.random.default_rng(5))
    ch = sequence_checks(Events(t, o, D0, D0 + 900), Events(st, so, D0, D0 + 900), datetime=True, splits=20)
    bad = [c["metric"] for c in ch if c["pass"] is False]
    assert bad == []


def test_C1205_series_model_keeps_seasonality_and_forecastability():
    """Prop. 1205: synthetic meters keep the daily ACF and spectrum within
    calibrated limits and a forecaster trained on them is within 15% of one
    trained on real data."""
    rng = np.random.default_rng(1205)
    n, T = 60, 24 * 120
    ts = pd.date_range("2024-01-01", periods=T, freq="h")
    h = np.arange(T) % 24
    dow = (np.arange(T) // 24) % 7
    f = np.zeros(T)
    for i in range(1, T):
        f[i] = 0.8 * f[i - 1] + rng.standard_normal() * 0.3
    rows = []
    for k in range(n):
        lvl = rng.lognormal(1, 0.5)
        e = np.zeros(T)
        for i in range(1, T):
            e[i] = 0.6 * e[i - 1] + rng.standard_normal() * 0.2
        y = lvl * np.exp(0.4 * np.sin(2 * np.pi * (h - 7) / 24) + 0.1 * (dow >= 5) + 0.5 * f + e)
        rows.append(pd.DataFrame({"m": k, "t": ts, "y": y}))
    real = pd.concat(rows, ignore_index=True)
    sm = SeriesModel().fit(real, "m", "t", ["y"], rng)
    syn = sm.sample(n, np.random.default_rng(7))
    Yr, step = series_matrix(real, "m", "t", "y")
    Ys = syn["y"].to_numpy().reshape(n, -1)
    checks, ts_ = series_checks(Yr, Ys, step, splits=20)
    ch = {c["metric"]: c for c in checks}
    assert ch["acf"]["pass"] and ch["spectrum"]["pass"]
    assert ts_["gap"] < 0.15


def test_C1206_incremental_equals_long_run():
    """Prop. 1206: (a) day-by-day forward runs are a prefix of one long run;
    (b) the forward process from a state observed at T has the law of the
    renewal continued past T (the first wait conditions on the age)."""
    rng = np.random.default_rng(1206)
    a, r = 0.5, 0.1
    m = _flat_model(a, L=4000.0, rng=rng)
    N = 20000
    m.rate_ab = np.array([[1e6, 1e6 / r]])             # rate known: Gamma(a + n, b + U) ~ point mass
    kh = np.arange(N, dtype=np.uint64)
    seg = np.zeros(N, int)
    # the true process: renewal from 0, observed up to T, continued
    T = 1000.0
    last = np.full(N, np.nan)
    nxt = np.empty(N)
    for i in range(N):
        cur = rng.gamma(a, 1 / (a * r))
        prev = np.nan
        while cur < T:
            prev, cur = cur, cur + rng.gamma(a, 1 / (a * r))
        last[i], nxt[i] = prev, cur
    has = np.isfinite(last)
    e, j, u = forward_events(m, seg, np.zeros(N), np.zeros(N), last, np.full(N, T), np.full(N, T + 400), kh, 3)
    first = pd.Series(u[j == 1], index=e[j == 1]).reindex(np.arange(N))
    ok = has & first.notna().to_numpy() & (nxt < T + 400)
    assert ks_2samp((first.to_numpy() - T)[ok], (nxt - T)[ok]).pvalue > 1e-3
    # (a) prefix stability: a shorter run is the prefix of the long one
    lo = forward_events(m, seg, np.zeros(N), np.zeros(N), last, np.full(N, T), np.full(N, T + 30), kh, 3)
    assert np.array_equal(np.sort(lo[2]), np.sort(u[u < T + 30]))


def test_C1204c_predictive_limits_hold_for_posterior_draws():
    """Prop. 1204 (predictive): datasets drawn from the posterior predictive of
    a well-specified fit fail the predictive limits at no more than the
    nominal rate (binomial bound at 0.1%)."""
    fails, tot = 0, 0
    for r in range(8):
        rng = np.random.default_rng(12040 + r)
        t, o = renewal(rng, 2000, 900.0)
        m = SequenceModel().fit(t, o, np.zeros(2000, int), 1, None, datetime=True, t0=D0, t1=D0 + 900, rng=rng)
        n = np.bincount(o, minlength=2000)
        for d in range(2):
            m.redraw(np.random.default_rng([r, d]))
            st, so = m.sample(n, np.zeros(2000, int), np.full(2000, D0), np.full(2000, D0 + 900),
                              np.random.default_rng([r, d, 1]))
            for c in sequence_checks(Events(t, o, D0, D0 + 900), Events(st, so, D0, D0 + 900), datetime=True,
                                     splits=20, seed=r, predictive=True):
                if c["pass"] is not None:
                    tot += 1
                    fails += not c["pass"]
    assert fails <= binom.isf(0.001, tot, 0.01)

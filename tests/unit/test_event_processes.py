# SPDX-License-Identifier: Apache-2.0
"""Event processes of the sequence model (Props. 1820-1823): the Bayesian
Hawkes interior (sequence.process: hawkes) and the semi-Markov status chain
(sequence.states)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from relsynth.engine import Synthesizer
from relsynth.spec import SpecError, load_spec
from relsynth.timeseries.calendar import to_days
from relsynth.timeseries.events import SequenceModel
from relsynth.timeseries.hawkes import HawkesPart, bridge, excitation, forest, profile_mu, simulate
from relsynth.timeseries.metrics import Events, sequence_checks
from relsynth.timeseries.states import StateModel

D0 = float(to_days(pd.Series(["2021-01-01"]))[0])


def hawkes_histories(rng, n_ent, T, mu=0.03, eta=0.6, beta=0.3):
    t, o = [], []
    for i in range(n_ent):
        x = simulate(mu, eta, beta, T, rng)
        t += list(D0 + x)
        o += [i] * len(x)
    return np.array(t), np.array(o, dtype=np.int64)


def flat(u_list):
    n = np.array([len(u) for u in u_list])
    return np.concatenate(u_list), np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64), n


# ----------------------------------------------------------------- Hawkes
def test_excitation_recursion_matches_direct_sum():
    rng = np.random.default_rng(0)
    u_list = [np.sort(rng.uniform(0, 50, k)) for k in (1, 4, 9)]
    u, start, n = flat(u_list)
    A = excitation(u, start, n, 0.7)
    for e, uu in enumerate(u_list):
        direct = [np.exp(-0.7 * (uu[j] - uu[:j])).sum() for j in range(len(uu))]
        assert np.allclose(A[start[e]:start[e] + n[e]], direct)


def test_profile_mu_solves_the_score_equation():
    rng = np.random.default_rng(1)
    e = rng.gamma(1.0, 0.05, 300)
    eid = np.repeat(np.arange(30), 10)
    S = rng.uniform(20, 200, 30)
    mu = profile_mu(e, eid, S)
    score = np.bincount(eid, 1 / (mu[eid] + e), 30) - S
    assert np.all(mu >= 0)
    assert np.allclose(score[mu > 0], 0, atol=1e-6 * S[mu > 0])
    assert np.all(score[mu == 0] <= 1e-9)            # boundary: the score at 0 is negative


def test_hawkes_fit_recovers_branching_ratio_and_decay():
    """Prop. 1820: the profile-likelihood posterior concentrates at the truth."""
    rng = np.random.default_rng(2)
    u_list = [x for x in (simulate(0.03, 0.6, 0.4, 900, rng) for _ in range(700)) if len(x) >= 3]
    u, start, n = flat(u_list)
    h = HawkesPart().fit(u, start, n, np.zeros(len(n), int), 1)
    eta, beta = h.log["pooled"]["eta"], h.log["pooled"]["beta"]
    sd = np.sqrt(np.diag(h.cov[0]))
    assert abs(eta - 0.6) < 0.06 and abs(np.log(beta / 0.4)) < 0.15
    assert np.all(sd < 0.2)
    assert h.log["pooled"]["residual_ks"] < 0.03     # time-rescaling residuals are Exp(1)


def test_forest_is_a_depth_first_forest_with_r_trees():
    rng = np.random.default_rng(3)
    for N, r in ((1, 1), (7, 1), (12, 4), (30, 30)):
        parent, tree = forest(N, r, rng)
        assert (parent < 0).sum() == r
        assert np.all(parent < np.arange(N))         # parents come first
        assert sorted(np.unique(tree)) == list(range(r))
        kids = parent >= 0
        assert np.all(tree[kids] == tree[parent[kids]])


def test_conditional_bridge_matches_conditioned_ogata_histories():
    """Prop. 1821: the cluster sampler given the count matches unconditional
    Hawkes histories (Ogata thinning) with that count, here through the gap
    shares and the share of the span's first half (edge effects are small for
    beta S = 40)."""
    rng = np.random.default_rng(4)
    eta, beta, S, n = 0.5, 4.0, 10.0, 10
    real = []
    while len(real) < 400:
        x = simulate(0.4, eta, beta, S, rng, history=[0.0])
        if len(x) == n - 2:                          # first event at 0, n - 2 interior events
            real.append(x / S)
    real = np.array(real)
    syn = bridge(np.full(4000, n), eta, beta * S, rng).reshape(4000, n - 2)
    g_r = np.diff(np.c_[np.zeros(len(real)), real], axis=1).ravel()
    g_s = np.diff(np.c_[np.zeros(len(syn)), syn], axis=1).ravel()
    assert ks_2samp(g_r, g_s).statistic < 0.06
    assert abs((real < 0.5).mean() - (syn < 0.5).mean()) < 0.04


def test_sequence_model_with_hawkes_fits_clustered_histories():
    rng = np.random.default_rng(5)
    t, o = hawkes_histories(rng, 900, 900.0)
    n = np.bincount(o, minlength=900)
    er = Events(t, o, D0, D0 + 900)
    res = {}
    for proc in ("renewal", "hawkes"):
        m = SequenceModel(process=proc).fit(t, o, np.zeros(900, int), 1, None, datetime=True, t0=D0,
                                            t1=D0 + 900, rng=np.random.default_rng(6))
        ts, ow = m.sample(n, np.zeros(900, int), np.full(900, D0), np.full(900, D0 + 900),
                          np.random.default_rng(7))
        assert np.all(np.diff(ts)[np.diff(ow) == 0] >= 0)
        res[proc] = {x["metric"]: x for x in sequence_checks(er, Events(ts, ow, D0, D0 + 900), splits=4)}
    assert res["hawkes"]["gap_w1"]["value"] < 0.5 * res["renewal"]["gap_w1"]["value"]
    assert res["hawkes"]["gap_ks"]["pass"] and res["hawkes"]["gap_w1"]["pass"]
    assert not res["renewal"]["gap_ks"]["pass"]
    # burstiness: within a few hundredths (the span truncation of Prop. 1821 makes
    # clusters a little shorter), several times closer than the renewal bridge
    assert res["hawkes"]["burstiness"]["value"] < 0.3 * res["renewal"]["burstiness"]["value"]


def test_hawkes_posterior_draws_move_and_mean_mode_is_the_mode():
    rng = np.random.default_rng(8)
    u_list = [x for x in (simulate(0.05, 0.4, 0.5, 400, rng) for _ in range(200)) if len(x) >= 3]
    u, start, n = flat(u_list)
    h = HawkesPart().fit(u, start, n, np.zeros(len(n), int), 1)
    h.draw(np.random.default_rng(1))
    assert not np.allclose(h.theta, h.mode)
    h.draw(None)
    assert np.allclose(h.theta, h.mode)


# ----------------------------------------------------------------- states
def semi_markov(rng, n_ent, L=1000.0):
    P = np.array([[0.1, 0.8, 0.1], [0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    pi = np.array([0.7, 0.2, 0.1])
    shape = np.array([0.5, 2.0, 6.0])               # common scale 10: mean 5, 20, 60 days
    t, o, s = [], [], []
    for i in range(n_ent):
        cur, st = D0 + rng.uniform(0, 300), rng.choice(3, p=pi)
        while cur < D0 + L:
            t.append(cur)
            o.append(i)
            s.append(st)
            cur += rng.gamma(shape[st], 10.0)
            st = rng.choice(3, p=P[st])
    return np.array(t), np.array(o), np.array(s), P, shape


def transitions(s, o, L=3):
    N = np.zeros((L, L))
    same = o[1:] == o[:-1]
    np.add.at(N, (s[:-1][same], s[1:][same]), 1)
    return N / N.sum(1, keepdims=True)


def test_state_chain_and_holding_shapes_are_recovered():
    """Props. 1822-1823: the transition posterior centres on the counts and the
    status-dependent Dirichlet bridge recovers the holding shapes."""
    rng = np.random.default_rng(9)
    t, o, s, P, shape = semi_markov(rng, 1200)
    m = SequenceModel(states={}).fit(t, o, np.zeros(1200, int), 1, None, datetime=True, t0=D0, t1=D0 + 1000,
                                     rng=np.random.default_rng(2), state=s, n_levels=3)
    sm = m.states
    assert np.abs(sm.P[0] - P).max() < 0.03
    th = sm.th_mode[0]
    d = np.zeros(3)
    d[np.arange(3) != sm.ref] = th[2:]
    alpha = np.exp(th[0] + d)
    assert np.allclose(np.log(alpha), np.log(shape), atol=0.15)
    n = np.bincount(o, minlength=1200)
    ts, ow, ss = m.sample_states(n, np.zeros(1200, int), np.full(1200, D0), np.full(1200, D0 + 1000),
                                 np.random.default_rng(3))
    assert np.abs(transitions(ss, ow) - transitions(s, o)).max() < 0.03
    same = ow[1:] == ow[:-1]
    g = np.diff(ts)[same]
    means = [g[ss[:-1][same] == k].mean() for k in range(3)]
    assert means[0] < means[1] < means[2]


def test_state_model_pools_small_segments_and_draws():
    s = np.array([0, 1, 0, 1, 1, 0, 2])
    start, n = np.array([0, 3]), np.array([3, 4])
    sm = StateModel(min_entities=5).fit(s, start, n, np.array([0, 1]), 2, 3)
    assert sm.pooled_seg.all()
    assert np.allclose(sm.P[0], sm.P[1])
    sm.draw(np.random.default_rng(0))
    assert np.allclose(sm.P.sum(2), 1) and np.allclose(sm.pi.sum(1), 1)


# ------------------------------------------------------------- engine hook
def shop(rng, n_cust=500):
    lv = np.array(["new", "repeat", "return"])
    P = np.array([[0.2, 0.7, 0.1], [0.3, 0.5, 0.2], [0.7, 0.2, 0.1]])
    t, o, st = [], [], []
    for i in range(n_cust):
        x = np.cumsum(rng.gamma(3, 5, 80))
        x = x[x < 1000]
        s = 0
        for xx in x:
            t.append(xx)
            o.append(i)
            st.append(s)
            s = rng.choice(3, p=P[s])
    d = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.floor(t), "D")
    cust = pd.DataFrame({"CustomerKey": np.arange(1, n_cust + 1), "Tier": rng.choice(["a", "b"], n_cust)})
    orders = pd.DataFrame({"OrderKey": np.arange(1, len(t) + 1), "CustomerKey": np.array(o) + 1,
                           "OrderDate": d.date, "Status": lv[st], "Amount": np.round(rng.gamma(2, 30, len(t)), 2)})
    return {"Customer": cust, "Orders": orders}, lv


def shop_spec(seq):
    sp = {"version": 1, "tables": {
        "Customer": {"primary_key": "CustomerKey", "columns": {"CustomerKey": "id", "Tier": "categorical"}},
        "Orders": {"primary_key": "OrderKey", "columns": {"OrderKey": "id", "CustomerKey": "id", "OrderDate": "date",
                                                          "Status": "categorical", "Amount": "float"}}},
        "relationships": [{"child": "Orders", "child_key": "CustomerKey", "parent": "Customer"}],
        "synthesis": {"seed": 3, "gibbs": {"iterations": 20, "final_iterations": 10}}}
    if seq:
        sp["tables"]["Orders"]["sequence"] = seq
    return sp


def status_transitions(df, lv):
    df = df.sort_values(["CustomerKey", "OrderDate", "OrderKey"], kind="stable")
    c = df["CustomerKey"].to_numpy()
    s = pd.Categorical(df["Status"], categories=lv).codes
    return transitions(s, c)


def test_engine_status_chain_orders_statuses_and_keeps_other_tables():
    tabs, lv = shop(np.random.default_rng(10))
    base = Synthesizer(load_spec(shop_spec({"entity": "CustomerKey", "time": "OrderDate"}))).fit(
        {k: v.copy() for k, v in tabs.items()})
    S = Synthesizer(load_spec(shop_spec({"entity": "CustomerKey", "time": "OrderDate", "states": "Status"}))).fit(
        {k: v.copy() for k, v in tabs.items()})
    b, d = base.sample(index=0), S.sample(index=0)
    log = S.sample_log["sequences"]["Orders"]
    assert 0 < log["status_rewritten"] < 0.2 * len(d["Orders"])
    pd.testing.assert_frame_equal(b["Customer"], d["Customer"])
    assert b["Orders"]["Amount"].tolist() == d["Orders"]["Amount"].tolist()
    real = status_transitions(tabs["Orders"], lv)
    assert np.abs(status_transitions(d["Orders"], lv) - real).max() < 0.05
    assert np.abs(status_transitions(b["Orders"], lv) - real).max() > 0.2      # without the chain: independent
    assert "status chain on Status" in S.log["decisions"]["Orders"]["sequence"]


def test_engine_hawkes_keeps_every_other_column():
    """Prop. 1203 holds for process hawkes: only the time column changes."""
    rng = np.random.default_rng(11)
    t, o = hawkes_histories(rng, 300, 900.0)
    d = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.floor(t - D0), "D")
    tabs = {"Customer": pd.DataFrame({"CustomerKey": np.arange(1, 301), "Tier": rng.choice(["a", "b"], 300)}),
            "Orders": pd.DataFrame({"OrderKey": np.arange(1, len(t) + 1), "CustomerKey": o + 1, "OrderDate": d.date,
                                    "Status": "x", "Amount": rng.gamma(2, 30, len(t)).round(2)})}
    a = Synthesizer(load_spec(shop_spec({"entity": "CustomerKey", "time": "OrderDate"}))).fit(
        {k: v.copy() for k, v in tabs.items()}).sample(index=0)
    S = Synthesizer(load_spec(shop_spec({"entity": "CustomerKey", "time": "OrderDate", "process": "hawkes"}))).fit(
        {k: v.copy() for k, v in tabs.items()})
    h = S.sample(index=0)
    assert "hawkes" in S.seq["Orders"]["model"].fit_log
    pd.testing.assert_frame_equal(a["Customer"], h["Customer"])
    pd.testing.assert_frame_equal(a["Orders"].drop(columns="OrderDate"), h["Orders"].drop(columns="OrderDate"))


def test_spec_errors_for_processes_and_states():
    sp = shop_spec({"entity": "CustomerKey", "time": "OrderDate", "process": "poisson"})
    with pytest.raises(SpecError, match="process must be renewal or hawkes"):
        load_spec(sp)
    sp = shop_spec({"entity": "CustomerKey", "time": "OrderDate", "states": "Amount"})
    with pytest.raises(SpecError, match="categorical, boolean or integer"):
        load_spec(sp)
    sp = shop_spec({"entity": "CustomerKey", "time": "OrderDate", "process": "hawkes", "states": "Status"})
    with pytest.raises(SpecError, match="semi-Markov"):
        load_spec(sp)
    sp = shop_spec({"entity": "CustomerKey", "time": "OrderDate", "states": {"column": "Status", "prior": 0}})
    with pytest.raises(SpecError, match="prior"):
        load_spec(sp)
    sp = shop_spec({"entity": "CustomerKey", "time": "OrderDate", "states": {"column": "Status", "prio": 1}})
    with pytest.raises(SpecError, match="prio"):
        load_spec(sp)

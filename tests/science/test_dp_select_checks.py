# SPDX-License-Identifier: Apache-2.0
"""Checks C1300-C1306: two-way marginals in DP stage 1 (roadmap T07b;
docs/proofs/sec_v26_dpselect.tex, Props. 1300-1306), tested on the shipped
code (`relsynth.dp.select`, `measure: mst | aim`).

C1301 is the central one, in the style of C902: for neighbouring databases
(one customer and all their sales removed) every new statistic, the score
vectors of each selection and every two-way histogram, moves by at most the
sensitivity it is charged for, with the noisy outputs (and choices) of the
full run replayed into the neighbour's run.

    pytest -m science tests/science/test_dp_select_checks.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

pytestmark = [pytest.mark.science]
opendp = pytest.importorskip("opendp")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import scenarios as S  # noqa: E402

import relsynth.dp as dp  # noqa: E402
from relsynth.dp import mechanisms as M  # noqa: E402
from relsynth.dp import relational as R  # noqa: E402
from relsynth.dp import select as SEL  # noqa: E402
from relsynth.errors import PrivacyError  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


# --------------------------------------------------------------------- C1300
def test_c1300_noisy_max_is_rho_zcdp_exactly():
    """Prop. 1300: report noisy max with Gumbel noise of scale b picks i with
    probability softmax(s / b).  For scores that move by at most D each, in
    either direction, every Renyi divergence of order alpha between the two
    choice distributions is at most alpha * rho with rho = D^2 / (2 b^2)
    (bounded range, eps = 2D/b, rho = eps^2/8), and OpenDP charges rho."""
    rng = np.random.default_rng(1)
    for D, rho in ((1.0, 0.05), (7.0, 0.01), (31.0, 0.2)):
        b = D / math.sqrt(2 * rho)
        worst = 0.0
        for _ in range(300):
            s = rng.normal(0, 3 * b, size=int(rng.integers(2, 12)))
            d = rng.uniform(-D, D, size=len(s))
            d[rng.random(len(s)) < 0.5] = np.sign(d[rng.random(len(s)) < 0.5].sum() or 1) * D
            lp = s / b - logsumexp(s / b)
            lq = (s + d) / b - logsumexp((s + d) / b)
            for a in (1.5, 2, 4, 16, 64):
                ren = logsumexp(a * lp + (1 - a) * lq) / (a - 1)
                worst = max(worst, ren / (a * rho))
                assert ren <= a * rho * (1 + 1e-9), (D, rho, a, ren)
        acct = M.Accountant(rho)
        i = M.noisy_max([0.0, 1e6, 0.0], D, rho, acct, "fit", "t")
        assert i == 1 and math.isclose(acct.spent[0], rho, rel_tol=1e-9)
        assert worst > 0.05, worst                       # the bound is not vacuous
    with pytest.raises(PrivacyError):
        M.noisy_max([1.0, 2.0], 1.0, 0.2, M.Accountant(0.1), "fit", "too much")


# ------------------------------------------------------- neighbouring databases
def _star(n_cust=150):
    t, s, _ = S.SCENARIOS["S2_star"]()
    keep = set(t["customer"]["customer_id"].head(n_cust))
    t["customer"] = t["customer"][t["customer"]["customer_id"].isin(keep)].reset_index(drop=True)
    t["sales"] = t["sales"][t["sales"]["customer_id"].isin(keep)].reset_index(drop=True)
    return t, s


def _dp_spec(s, **kw):
    s = copy.deepcopy(s)
    s.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {
        "unit": "customer", "epsilon": 2.0, "delta": 1e-9, "public_tables": ["product"], **kw}
    return load_spec(s)


def _public(tables):
    pub = {"customer.region": {"levels": sorted(tables["customer"]["region"].astype(str).unique())},
           "sales.channel": {"levels": sorted(tables["sales"]["channel"].astype(str).unique())}}
    for t, c in (("customer", "age"), ("customer", "income"), ("sales", "quantity"), ("sales", "amount")):
        x = tables[t][c]
        pub[f"{t}.{c}"] = {"min": float(x.min()), "max": float(x.max())}
    return pub


class _Gauss:
    """Stands in for mechanisms.gaussian (C902's recorder)."""

    def __init__(self, replay=None):
        self.calls, self.replay = [], replay

    def __call__(self, values, sensitivity, rho, acct, stage, mechanism):
        v = np.asarray(values, dtype=np.int64).ravel()
        acct.charge(M.Charge(stage, mechanism, rho))
        i = len(self.calls)
        self.calls.append((mechanism, v.copy(), float(sensitivity)))
        if self.replay is None:
            # a fixed, data-independent perturbation so that the run is not degenerate
            return v + np.random.default_rng(i).integers(-2, 3, size=len(v))
        name, out, _ = self.replay.calls[i]
        assert name == mechanism and len(out) == len(v), (name, mechanism)
        return self.replay.outs[i]

    @property
    def outs(self):
        return [v + np.random.default_rng(i).integers(-2, 3, size=len(v)) for i, (_, v, _) in enumerate(self.calls)]


class _Max:
    """Stands in for mechanisms.noisy_max: records the score vector and its
    sensitivity; picks the true maximum (first run) or the first run's choice."""

    def __init__(self, replay=None):
        self.calls, self.replay = [], replay

    def __call__(self, scores, sensitivity, rho, acct, stage, mechanism):
        s = np.asarray(scores, dtype=float).ravel()
        acct.charge(M.Charge(stage, mechanism, rho))
        i = len(self.calls)
        if self.replay is None:
            pick = int(np.argmax(s))
        else:
            name, s0, _, pick = self.replay.calls[i]
            assert name == mechanism and len(s0) == len(s), (name, mechanism, len(s0), len(s))
        self.calls.append((mechanism, s.copy(), float(sensitivity), pick))
        return pick


def _run(spec, tables, g, mx, monkeypatch):
    monkeypatch.setattr(R, "gaussian", g)
    monkeypatch.setattr(SEL, "noisy_max", mx)
    monkeypatch.setattr(R, "_hash_key", lambda: "0123456789abcdef")
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    acct = M.Accountant(cfg.rho, cfg.delta)
    real = {t: df.reset_index(drop=True) for t, df in tables.items()}
    return R.fit(spec, real, cfg, acct, pl), acct, cfg


def _without(tables, cid):
    t = {k: v.copy() for k, v in tables.items()}
    t["customer"] = t["customer"][t["customer"]["customer_id"] != cid].reset_index(drop=True)
    t["sales"] = t["sales"][t["sales"]["customer_id"] != cid].reset_index(drop=True)
    return t


@pytest.mark.parametrize("measure", ["mst", "aim"])
def test_c1301_scores_and_pairs_respect_their_sensitivity(monkeypatch, measure):
    """Props. 1301, 1302: with public ranges, levels and degree cap (fixed
    shapes), removing any one customer and their sales moves every selection's
    score vector by at most c in L-infinity and every two-way histogram by at
    most c in L2, given the earlier noisy outputs and choices.  Includes the
    busiest customer, where truncation binds."""
    tables, s = _star()
    spec = _dp_spec(s, public=_public(tables), max_degree={"sales": 4}, measure=measure, aim_rounds=12)
    g0, m0 = _Gauss(), _Max()
    _run(spec, tables, g0, m0, monkeypatch)
    pair_names = {n for n, _, _ in g0.calls if " x " in n}
    assert pair_names and any(n.startswith("sales: pair") and "^" in n for n in pair_names), pair_names
    assert {n for n, *_ in m0.calls} == {"customer: pair selection", "sales: pair selection"}
    counts = tables["sales"]["customer_id"].value_counts()
    ids = [counts.index[0]] + list(tables["customer"]["customer_id"].sample(8, random_state=5))
    worst: dict = {}
    for cid in ids:
        g1, m1 = _Gauss(replay=g0), _Max(replay=m0)
        _run(spec, _without(tables, cid), g1, m1, monkeypatch)
        assert len(g1.calls) == len(g0.calls) and len(m1.calls) == len(m0.calls)
        for (name, v0, sens), (_, v1, _) in zip(g0.calls, g1.calls):
            d = float(np.linalg.norm(v0 - v1))
            assert d <= sens + 1e-9, (cid, name, d, sens)
            if " x " in name:
                worst["pairs"] = max(worst.get("pairs", 0.0), d / sens)
        for (name, s0, sens, _), (_, s1, _, _) in zip(m0.calls, m1.calls):
            d = float(np.max(np.abs(s0 - s1)))
            assert d <= sens + 1e-9, (cid, name, d, sens)
            worst["scores"] = max(worst.get("scores", 0.0), d / sens)
    assert worst["pairs"] > 0.3 and worst["scores"] > 0.3, worst      # the bounds are not vacuous


@pytest.mark.parametrize("measure", ["mst", "aim"])
def test_c1302_the_run_spends_within_its_plan(measure):
    """Prop. 1302: with nothing public, a whole mst/aim run (discovery, one-way,
    degrees, selections, pairs, decode, stage 2) spends at most the stated
    budget, and the statement lists the new mechanisms."""
    tables, s = _star()
    res = dp.synthesize(_dp_spec(s, epsilon=1.0, measure=measure), tables=tables, write=False, evaluate=False)
    assert res.epsilon <= 1.0 + 1e-9 and res.rho <= res.config.rho * (1 + 1e-9)
    mech = set(res.statement["mechanisms"]["fit"])
    assert "customer: pair selection" in mech and any(" x " in m for m in mech)
    assert res.data["sales"]["customer_id"].isin(res.data["customer"]["customer_id"]).all()


# --------------------------------------------------------------------- C1303
def _toy_forest(rng):
    nodes = {n: SEL.Node(n, k) for n, k in (("a", 3), ("b", 4), ("c", 2), ("d", 3))}
    nodes["e"] = SEL.Node("e", 3, evidence=True)
    edges = [("a", "b"), ("b", "c"), ("b", "d"), ("a", "e")]
    J = rng.dirichlet(np.ones(3 * 4 * 2 * 3 * 3)).reshape(3, 4, 2, 3, 3)
    # project to the tree: p(a,b,c,d,e) = p(a,b) p(c|b) p(d|b) p(e|a)
    pab, pbc, pbd, pae = J.sum((2, 3, 4)), J.sum((0, 3, 4)), J.sum((0, 2, 4)), J.sum((1, 2, 3))
    T = np.einsum("ab,bc,bd,ae->abcde", pab, pbc / pbc.sum(1, keepdims=True), pbd / pbd.sum(1, keepdims=True),
                  pae / pae.sum(1, keepdims=True))
    f = SEL.Forest(nodes, edges)
    f.P = {("a", "b"): pab, ("b", "c"): T.sum((0, 3, 4)), ("b", "d"): T.sum((0, 2, 4)), ("a", "e"): T.sum((1, 2, 3))}
    f.p = {"a": T.sum((1, 2, 3, 4)), "b": T.sum((0, 2, 3, 4)), "c": T.sum((0, 1, 3, 4)), "d": T.sum((0, 1, 2, 4)),
           "e": T.sum((0, 1, 2, 3))}
    return f, T


def test_c1303_forest_model_estimate_and_sampler_are_exact():
    """Prop. 1303: (a) the model's pair of any two nodes is the tree's exact
    two-way marginal; (b) the sampler draws from the tree, and given evidence
    on a node from the exact conditional; (c) the estimate is consistent: node
    marginals sum to one and each edge joint has the node marginals as margins."""
    rng = np.random.default_rng(7)
    f, T = _toy_forest(rng)
    assert np.allclose(f.pair("c", "d"), T.sum((0, 1, 4)), atol=1e-12)
    assert np.allclose(f.pair("e", "c"), T.sum((0, 1, 3)).T, atol=1e-12)
    n = 200_000
    out = SEL.sample(f, n, {}, rng)
    # evidence nodes are drawn like the others when nothing is observed
    emp = np.bincount(out["c"] * 3 + out["d"], minlength=6).reshape(2, 3) / n
    assert np.abs(emp - T.sum((0, 1, 4))).max() < 0.006
    ev = rng.choice(3, size=n, p=f.p["e"])
    out = SEL.sample(f, n, {"e": ev}, rng)
    assert (out["e"] == ev).all()
    for e in range(3):
        cond = T[..., e].sum((0, 2, 3)) / T[..., e].sum()
        emp = np.bincount(out["b"][ev == e], minlength=4) / max((ev == e).sum(), 1)
        assert np.abs(emp - cond).max() < 0.012, (e, emp, cond)
    # (c)
    meas = {("a", "b"): (rng.integers(-3, 40, size=(3, 4)).astype(float), 4.0),
            ("b", "c"): (rng.integers(-3, 40, size=(4, 2)).astype(float), 4.0)}
    nodes = {n: SEL.Node(n, k, y=rng.integers(0, 60, size=k).astype(float), var=2.0) for n, k in
             (("a", 3), ("b", 4), ("c", 2))}
    est = SEL.estimate(nodes, meas, 100.0, iters=200)
    for v in nodes:
        assert math.isclose(est.p[v].sum(), 1.0) and (est.p[v] >= 0).all()
    for (a, b), Q in est.P.items():
        assert np.allclose(Q.sum(1), est.p[a], atol=1e-8) and np.allclose(Q.sum(0), est.p[b], atol=1e-8)


# --------------------------------------------------------------------- C1304
def test_c1304_compression_and_refinement_preserve_the_released_marginal():
    """Prop. 1304: the grouping keeps the null cell alone and makes at most K
    cells, ordered groups are runs; drawing a coarse cell from the grouped
    marginal and refining within it gives back the released fine marginal."""
    rng = np.random.default_rng(3)
    for ordered in (True, False):
        p = rng.dirichlet(np.ones(33) * 0.5)
        g = SEL.grouping(p, ordered, 8)
        assert g[0] == 0 and (g[1:] > 0).all() and g.max() + 1 <= 8
        if ordered:
            assert (np.diff(g[1:]) >= 0).all()
        pc = np.bincount(g, weights=p)
        coarse = rng.choice(len(pc), size=300_000, p=pc)
        fine = SEL.refine(coarse, g, p, rng)
        assert np.abs(np.bincount(fine, minlength=33) / len(fine) - p).max() < 0.004
        assert (g[fine] == coarse).all()


# --------------------------------------------------------------------- C1305
def test_c1305_two_way_error_falls_with_epsilon():
    """With two-way marginals the workload error falls as epsilon grows (the
    T07 copula path was flat): mean two-way TVD of the synthetic sales at
    epsilon 32 is well below that at epsilon 0.25, for both selections."""
    from dp_workload import workload_error
    tables, s = S.SCENARIOS["S2_star"]()[:2]
    for measure in ("mst", "aim"):
        err = {}
        for eps in (0.25, 32.0):
            v = []
            for _ in range(2):
                r = dp.synthesize(_dp_spec(s, epsilon=eps, measure=measure), tables=tables, write=False,
                                  evaluate=False)
                v.append(workload_error(tables["sales"], r.data["sales"],
                                        ["quantity", "amount", "channel", "product_id"])["tvd2"])
            err[eps] = float(np.mean(v))
        assert err[32.0] < 0.75 * err[0.25], (measure, err)


# --------------------------------------------------------------------- C1306
def test_c1306_bad_settings_are_refused_before_any_read():
    _, s = _star()
    with pytest.raises(PrivacyError, match="measure"):
        dp.plan(_dp_spec(s, measure="pgm"))
    with pytest.raises(PrivacyError, match="pair_bins"):
        dp.plan(_dp_spec(s, measure="mst", pair_bins=2))
    with pytest.raises(PrivacyError, match="select_share"):
        dp.plan(_dp_spec(s, measure="mst", select_share=1.0))

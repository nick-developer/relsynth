# SPDX-License-Identifier: Apache-2.0
"""Checks C1720-C1724: small-dimension privacy (gap plan G3;
docs/proofs/sec_g3_smalldims.tex, Props. 1720-1724), tested on the shipped
code (`marginals.segment_prior_rows`, `relsynth.nearcopy`, the engine's guard).

    pytest -m science tests/science/test_small_dims_checks.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.science]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import write_scenario  # noqa: E402

import relsynth as rs  # noqa: E402
from relsynth.marginals import Codec, Marginal  # noqa: E402
from relsynth.nearcopy import NearCopyGuard, nearest  # noqa: E402


def _own_mass(m, own):
    """Predictive mass of a continuous marginal draw on the segment's own atoms:
    the share of quantile knots that fall on the segment's values."""
    u = np.linspace(0, 1, 4001)[1:-1]
    v = m.from_uniform(u)
    return np.isin(np.round(v, 9), np.round(own, 9)).mean()


# --------------------------------------------------------------------- C1720
def test_c1720_segment_shrinkage_weight_and_stream():
    """Prop. 1720: the predictive mass on the segment's own rows is n / (n + a)
    (discrete: exactly, by the Dirichlet mean; continuous: E[V]), and the variates
    drawn do not depend on a (the generator's state after them can)."""
    rng = np.random.default_rng(5)
    table = pd.Series(rng.choice(list("abcdefgh"), 3000, p=[.3, .2, .15, .1, .1, .05, .05, .05]))
    cd = Codec("x", {"type": "categorical"}, table)
    X = cd.encode(table)
    top = Marginal.fit(cd, X, rng, dirichlet_a=0.25, bootstrap=False)
    seg = cd.encode(pd.Series(list("aaaaabbbcc")))           # n = 10, levels a, b, c only
    n = len(seg)
    for a in (0.25, 10.0, 20.0):
        draws = np.array([Marginal.fit(cd, seg, rng, prior=top, dirichlet_a=a).p for _ in range(20000)])
        own = draws[:, :3].sum(1) if list(cd.levels[:3]) == list("abc") else \
            draws[:, [cd.levels.index(v) for v in "abc"]].sum(1)
        w = n / (n + a)
        expect = w + (1 - w) * top.p[[cd.levels.index(v) for v in "abc"]].sum()
        assert abs(own.mean() - expect) < 4 * own.std() / np.sqrt(len(own)) + 1e-3, (a, own.mean(), expect)
    # continuous: E[V] = n / (n + a), V ~ Beta(n, a)
    vals = rng.gamma(2.0, size=3000)
    cc = Codec("y", {"type": "float"}, pd.Series(vals))
    topc = Marginal.fit(cc, vals, rng, bootstrap=False, grid=64)
    own = np.round(rng.gamma(2.0, size=12) * 100) / 100 + 50.0     # atoms far right of the table law
    # (b) the variates drawn do not depend on a ...
    class Counting:
        def __init__(self, seed):
            self.g, self.calls = np.random.default_rng(seed), []

        def __getattr__(self, name):
            f = getattr(self.g, name)
            return (lambda *x, **k: (self.calls.append(name), f(*x, **k))[1]) if callable(f) else f

    c0, c1 = Counting(11), Counting(11)
    Marginal.fit(cc, own, c0, prior=topc, dirichlet_a=0.25, grid=64)
    Marginal.fit(cc, own, c1, prior=topc, dirichlet_a=20.0, grid=64)
    assert c0.calls == c1.calls and "beta" in c0.calls
    # ... but the underlying random numbers can: the Beta is drawn by rejection, so
    # the state after it differs for some seeds, and only a = a0 keeps the stream
    differs = 0
    for seed in range(300):
        g0, g1, g2 = (np.random.default_rng(seed) for _ in range(3))
        g0.beta(len(own), 0.25), g1.beta(len(own), 40.0), g2.beta(len(own), 0.25)
        r0, r1, r2 = g0.random(), g1.random(), g2.random()
        differs += r0 != r1
        assert r0 == r2
    assert differs > 0
    for a in (0.25, 20.0):
        mass = []
        for _ in range(3000):
            m = Marginal.fit(cc, own, rng, prior=topc, dirichlet_a=a, grid=256)
            mass.append((m.from_uniform(np.linspace(0, 1, 2001)[1:-1]) >= 50.0).mean())
        w = len(own) / (len(own) + a)
        assert abs(np.mean(mass) - w) < 0.03, (a, np.mean(mass), w)


# --------------------------------------------------------------------- C1724
def test_c1724_shape_shrinkage_keeps_location_range_and_stream():
    """Prop. 1724: in shape mode the posterior mean predictive (bootstrap=False)
    of a continuous segment has its sample median, up to the grid (b), every value
    lies in the smallest interval containing the table's knots and the segment's
    values (c), the random stream does not change at a fixed a (d), a median
    beyond the table's range is pulled toward it, and the predictive moves by at most a / (n + a) whatever the two priors (e)."""
    rng = np.random.default_rng(7)
    vals = rng.gamma(2.0, size=4000)
    cc = Codec("y", {"type": "float"}, pd.Series(vals))
    top = Marginal.fit(cc, vals, rng, bootstrap=False, grid=64)
    lo, hi = top.knots_v[0], top.knots_v[-1]
    u = np.linspace(0, 1, 20001)[1:-1]
    for own in (np.array([6.1, 6.4, 6.6, 7.0, 7.2, 7.9]),          # right tail of the table
                np.array([0.20, 0.25, 0.3]),                       # left tail, three rows
                np.array([3.0] * 5)):                              # identical values
        for a in (20.0, 40.0):
            m = Marginal.fit(cc, own, rng, prior=top, dirichlet_a=a, grid=512, bootstrap=False,
                             shift_prior=True)
            v = m.from_uniform(u)
            assert abs(np.median(v) - np.median(own)) < 0.02 * (hi - lo), (own, a, np.median(v))
            assert v.min() >= min(lo, own.min()) - 1e-9 and v.max() <= max(hi, own.max()) + 1e-9
            # the table-mode prior moves the median toward the table's
            t = Marginal.fit(cc, own, rng, prior=top, dirichlet_a=a, grid=512, bootstrap=False)
            if own.std() > 0:
                assert abs(np.median(t.from_uniform(u)) - np.median(own)) > abs(np.median(v) - np.median(own))
        s0, s1 = np.random.default_rng(3), np.random.default_rng(3)
        Marginal.fit(cc, own, s0, prior=top, dirichlet_a=40.0, shift_prior=True)
        Marginal.fit(cc, own, s1, prior=top, dirichlet_a=40.0)
        assert s0.random() == s1.random()
    # (b) needs l <= m_g <= u: a segment beyond the table's largest knot keeps its
    # values, but the clipped prior sits at u and pulls the predictive median below m_g
    far = hi + np.array([5.0, 6.0, 7.0])
    m = Marginal.fit(cc, far, rng, prior=top, dirichlet_a=40.0, grid=512, bootstrap=False,
                     shift_prior=True)
    v = m.from_uniform(u)
    assert np.median(v) < np.median(far) - 1.0 and np.median(v) <= hi + 1e-9
    assert v.max() <= far.max() + 1e-9
    # (e) with a prior that changes with a, the predictive moves by at most a / (n + a)
    for _ in range(500):
        V = int(rng.integers(2, 30))
        e = rng.dirichlet(np.ones(V) * 0.3)
        G, H = rng.dirichlet(np.ones(V) * 0.3), rng.dirichlet(np.ones(V) * 0.3)
        n = int(rng.integers(1, 5000))
        a0 = float(rng.uniform(0.0, 5.0))
        a = float(rng.uniform(a0, 200.0))
        wa, w0 = n / (n + a), n / (n + a0)
        tv = 0.5 * np.abs(wa * e + (1 - wa) * G - (w0 * e + (1 - w0) * H)).sum()
        assert tv <= a / (n + a) + 1e-12
    # a positive segment_prior_rows below dirichlet_prior: the weight is a = dirichlet_prior,
    # and the bound is a / (n + a), not rows / (n + rows)
    dp, rows = 0.25, 0.01
    a = max(dp, rows)
    for _ in range(200):
        V = int(rng.integers(2, 30))
        e, G, H = (rng.dirichlet(np.ones(V) * 0.3) for _ in range(3))
        n = int(rng.integers(1, 50))
        wa, w0 = n / (n + a), n / (n + dp)
        tv = 0.5 * np.abs(wa * e + (1 - wa) * G - (w0 * e + (1 - w0) * H)).sum()
        assert tv <= a / (n + a) + 1e-12



def test_c1724_positive_tail_trim_can_leave_the_interval():
    """Prop. 1724(c) is stated for tail_trim 0: a positive tail_trim extrapolates the
    end knots on purpose, so a value can fall below both the table's lowest knot and
    the segment's lowest value (here the first trimmed quantile sits on that bound and
    the next one above it)."""
    rng = np.random.default_rng(7)
    vals = rng.uniform(0, 10, size=4000)
    cc = Codec("y", {"type": "float"}, pd.Series(vals))
    top = Marginal.fit(cc, vals, rng, bootstrap=False, grid=64)
    lo = top.knots_v[0]
    own = np.r_[np.full(6, lo), np.linspace(lo + 1, lo + 4, 20)]
    u = np.linspace(0, 1, 20001)[1:-1]
    kw = dict(prior=top, dirichlet_a=40.0, grid=16, bootstrap=False, shift_prior=True)
    trimmed = Marginal.fit(cc, own, rng, tail_trim=0.1, **kw).from_uniform(u)
    assert trimmed.min() < min(lo, own.min())
    plain = Marginal.fit(cc, own, rng, **kw).from_uniform(u)
    assert plain.min() >= min(lo, own.min()) - 1e-12

# --------------------------------------------------------------------- C1721
def test_c1721_large_segments_move_by_at_most_a_over_n():
    """Prop. 1721: TV(p_a, p_a0) = n (a - a0) / ((n + a)(n + a0)) TV(e, p0) <= (a - a0) / (n + a)."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        V = int(rng.integers(2, 30))
        p0 = rng.dirichlet(np.ones(V))
        n = int(rng.integers(1, 5000))
        c = rng.multinomial(n, rng.dirichlet(np.ones(V)))
        a0, a = 0.25, float(rng.uniform(0.25, 100))
        pa, pb = (c + a * p0) / (n + a), (c + a0 * p0) / (n + a0)
        tv = 0.5 * np.abs(pa - pb).sum()
        exact = n * (a - a0) / ((n + a) * (n + a0)) * 0.5 * np.abs(c / n - p0).sum()
        assert abs(tv - exact) < 1e-12 and tv <= (a - a0) / (n + a) + 1e-12


# --------------------------------------------------------------------- C1722
@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    spec = write_scenario("S9_small_shop", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    """A small product list in families (30 models x 10 variants that share a price,
    a cost and a line), the shape of DimProduct: as CSV with its spec."""
    d = tmp_path_factory.mktemp("catalogue")
    rng = np.random.default_rng(12)
    rows = []
    for m in range(30):
        price = float(np.round(rng.lognormal(5, 1), 2))
        line = rng.choice(list("MRTS"))
        for v in range(10):
            rows.append({"id": len(rows) + 1, "price": price, "cost": round(price * 0.6, 2),
                         "weight": float(np.round(rng.normal(10 + m, 0.3), 2)),
                         "color": rng.choice(["red", "black", "silver", "blue"]),
                         "size": str(38 + 2 * (v % 5)), "line": line})
    (d / "src").mkdir()
    pd.DataFrame(rows).to_csv(d / "src" / "product.csv", index=False)
    spec = {"version": 1, "name": "catalogue", "source": {"kind": "csv", "path": str(d / "src")},
            "target": {"kind": "csv", "path": str(d / "out")},
            "tables": {"product": {"role": "entity", "primary_key": "id", "columns": {
                "id": {"type": "id"}, "price": {"type": "float"}, "cost": {"type": "float"},
                "weight": {"type": "float"}, "color": {"type": "categorical"},
                "size": {"type": "categorical"}, "line": {"type": "categorical"}}}},
            "relationships": [], "rules": []}
    return spec


def _fit(spec, **priv):
    s = copy.deepcopy(spec)
    s.setdefault("synthesis", {}).setdefault("privacy", {}).update(priv)
    return rs.fit(s, seed=4)


def test_c1722_guard_accepts_only_rows_outside_B(catalogue):
    """Prop. 1722 (a, b): every accepted row of a guarded table is outside B, by
    brute force; on a one-dimensional example the accepted law is P(. | B^c)."""
    m = _fit(catalogue, near_copy="auto")
    syn = m.synthesizer
    out = syn.sample(index=0)
    log = syn.sample_log.get("near_copies", {})
    assert log, "no guarded table in the scenario"
    for t, r in log.items():
        g = syn._nc_guards[t]
        assert g is not None
        bad = g.near_copies(out[t].reset_index(drop=True)) if set(g.cols) <= set(out[t].columns) else None
        if bad is not None:
            # rows kept after the last round are the only rows allowed in B
            assert int(bad.sum()) <= r["kept"], (t, int(bad.sum()), r)
    # 1-D rejection sampling: law of accepted draws is the normal restricted to B^c
    real = pd.DataFrame({"a": [-1.0, 0.0, 0.4, 2.0], "b": list("xxxx"), "c": [0.0] * 4, "d": list("pppp")})
    ts = {"columns": {"a": {"type": "float"}, "b": {"type": "categorical"}, "c": {"type": "float"},
                      "d": {"type": "categorical"}}}
    g = NearCopyGuard(real, ts, ["a", "b", "c", "d"])
    rng = np.random.default_rng(9)
    prop = pd.DataFrame({"a": rng.normal(size=200000), "b": "x", "c": 0.0, "d": "p"})
    inB = g.near_copies(prop)
    acc = prop.a[~inB].to_numpy()
    # reference: the same proposal filtered by B computed by brute force from its
    # definition (every real row, its own positive-neighbour distance), without
    # relsynth.nearcopy.nearest
    Xr = g.X
    nu = np.array([min(np.linalg.norm(Xr[i] - Xr[k]) for k in range(len(Xr))
                       if np.linalg.norm(Xr[i] - Xr[k]) > 0) for i in range(len(Xr))])
    keep = []
    while len(keep) < 20000:
        x = pd.DataFrame({"a": rng.normal(size=5000), "b": "x", "c": 0.0, "d": "p"})
        D = np.linalg.norm(g.encode(x)[:, None, :] - Xr[None], axis=2)
        j = D.argmin(1)
        d = D[np.arange(len(x)), j]
        inb = ((d > 0) & (d < nu[j])) | ((d == 0) & ~g.has_twin[j])
        keep += x.a[~inb].tolist()
    from scipy.stats import ks_2samp
    assert ks_2samp(acc[:20000], np.array(keep[:20000])).pvalue > 1e-3
    assert 0 < inB.mean() < 1


def test_c1722_no_near_copy_draws_nothing(shop):
    """Prop. 1722 (c): when no row is a near copy the guard draws no random
    number, so the tables equal the unguarded ones byte for byte."""
    base = dict(_fit(shop, near_copy="off").sample())
    m = _fit(shop, near_copy="auto", near_copy_ratio=0.0)
    zero = dict(m.sample())
    flagged = sum(r["near_copies"] for r in (m.synthesizer.sample_log.get("near_copies") or {}).values())
    assert flagged == 0, "the scenario must have no near copies at ratio 0 for this check"
    for t in base:
        pd.testing.assert_frame_equal(base[t], zero[t])


# --------------------------------------------------------------------- C1723
def test_c1723_guard_leaves_a_hole_around_a_record():
    """Prop. 1723: adding a record x empties the ball around it in the guarded
    output; without x the same set has positive probability.  No finite epsilon."""
    ts = {"columns": {"a": {"type": "float"}, "b": {"type": "categorical"}, "c": {"type": "float"},
                      "d": {"type": "categorical"}}}
    X = pd.DataFrame({"a": [-2.0, -1.0, 1.0, 2.0], "b": list("xxxx"), "c": [0.0] * 4, "d": list("pppp")})
    x = pd.DataFrame({"a": [10.0], "b": ["x"], "c": [0.0], "d": ["p"]})      # an outlying record
    rng = np.random.default_rng(1)
    # a generator with atoms (as discrete-valued columns have): a tenth of its rows at 10
    a = rng.uniform(-3, 20, 400000)
    a[rng.random(len(a)) < 0.1] = 10.0
    prop = pd.DataFrame({"a": a, "b": "x", "c": 0.0, "d": "p"})
    with_x = NearCopyGuard(pd.concat([X, x], ignore_index=True), ts, ["a", "b", "c", "d"])
    without = NearCopyGuard(X, ts, ["a", "b", "c", "d"])
    in_with, in_without = with_x.near_copies(prop), without.near_copies(prop)
    # the ball around x in the guard's metric (rows whose nearest real row is x),
    # minus the guard regions of X
    j, d = nearest(with_x.encode(prop), with_x.X)
    hole = (j == len(X)) & (d < with_x.nn[len(X)]) & ~in_without
    assert hole.mean() > 0.01                       # positive probability without x
    assert in_with[hole].all()                      # probability 0 with x


def test_c1722_scorecard_marks_a_guarded_table(catalogue):
    """The dcr_ratio item of a guarded table says that it is not independent evidence."""
    m = _fit(catalogue, near_copy="auto")
    data = m.sample()
    rep = m.evaluate(data)
    items = [i for i in rep.scorecard["items"] if i["metric"] == "dcr_ratio"]
    guarded = set(data.sample_log.get("near_copies") or {})
    assert items and guarded
    for i in items:
        assert ("near-copy guard" in i.get("note", "")) == (i["area"] in guarded)


# --------------------------------------------------------------------- C1725
def _dp_rows(a, sizes, rng, levels=None):
    """Segments F_g ~ DP(a, F0): truncated stick breaking over N(0, 1) atoms, or
    Dir(a / L, ..., a / L) level probabilities for L levels."""
    x, g = [], []
    for k, s in enumerate(sizes):
        if levels:
            x += list(rng.choice(levels, size=s, p=rng.dirichlet(np.full(levels, a / levels))))
        else:
            V = rng.beta(1, a, size=400)
            w = V * np.cumprod(np.r_[1, 1 - V[:-1]])
            x += list(rng.normal(size=400)[rng.choice(400, size=s, p=w / w.sum())])
        g += [k] * s
    return np.array(x), np.array(g)


def test_c1725_empirical_bayes_concentration():
    """Prop. 1725: (a, c) the moment estimate recovers a for DP hierarchies of
    continuous and discrete columns (median of 30 replicates within 25 % of a,
    unbalanced segments of 5 to 60 rows); (b) the nested reference removes the
    coarse groups' effect; (d) the posterior mean at a = (1 - rho) / rho equals
    the credibility estimate; (e) no random number is drawn."""
    from relsynth.shrinkage import concentration, icc_indicators
    rng = np.random.default_rng(1725)
    for a in (0.5, 4.0, 20.0):
        for levels in (None, 8):
            est = [concentration("discrete" if levels else "continuous",
                                 *_dp_rows(a, rng.integers(5, 61, size=80), rng, levels), levels=levels or 0).a
                   for _ in range(30)]
            assert abs(np.median(est) / a - 1) < 0.25, (a, levels, np.median(est))
    # (b) the coarse groups differ, the segments inside them do not
    x = np.concatenate([rng.normal(0, 1, 3000), rng.normal(6, 1, 3000)])
    seg, coarse = np.repeat(np.arange(60), 100), np.repeat([0, 1], 3000)
    assert concentration("continuous", x, seg, coarse, a_max=1e5).a > 100
    # (d) credibility: Z = n / (n + k) with k = s2_w / s2_b = (1 - rho) / rho
    Y = rng.integers(0, 2, size=(400, 1)).astype(float)
    Y[:200] = (rng.random((200, 1)) < 0.8)
    g = np.repeat(np.arange(20), 20)
    rho, _, _ = icc_indicators(Y, g)
    k = (1 - rho) / rho
    for gg in range(20):
        yb, p = Y[g == gg, 0].mean(), Y[:, 0].mean()
        z = 20 / (20 + k)
        assert z * yb + (1 - z) * p == pytest.approx((20 * yb + k * p) / (20 + k))
    # (e)
    st = rng.bit_generator.state
    concentration("continuous", x, seg, coarse)
    assert rng.bit_generator.state == st


# --------------------------------------------------------------------- C1726
def _seg_fit(x_codes, groups, levels, floor, rows=0.0, bootstrap=False):
    """SegmentMarginals on one discrete column (table mode) with the given floor."""
    from types import SimpleNamespace

    from relsynth.generators.copula import SegmentMarginals
    s = pd.Series(np.asarray(levels, dtype=object)[x_codes])
    cd = Codec("x", {"type": "categorical"}, s)
    X = cd.encode(s)
    gen = SegmentMarginals()
    gen.data = SimpleNamespace(cfg={"marginals": {"dirichlet_prior": 0.25, "tail_trim": 0.0, "max_obs": 20000,
                                                  "segment_prior_rows": rows, "segment_floor_rows": floor,
                                                  "segment_prior_mode": "table"},
                                    "copula": {"grid": 48, "min_rows": 30}},
                               codecs=[cd], X=[X], cols=["x"], n_rows=len(X), role="entity")
    gen._groups, gen.parent, gen._pi_s = np.asarray(groups), {}, 0.0
    gen._fit_marginals(np.random.default_rng(0), bootstrap)
    return gen, cd, X


def test_c1726_privacy_floor_bounds_each_rows_weight():
    """Prop. 1726: (a) in the posterior mean marginal of a segment of n rows each
    row carries at most 1 / k (a level held by one row of the segment has mass at
    most 1 / k plus the prior's share); (b) segments of at least k rows are fitted
    as without the floor; (c) replacing one row moves the predictive by at most
    1 / k in total variation at a fixed weight."""
    rng = np.random.default_rng(1726)
    L, k = 12, 20.0
    sizes = [3, 5, 8, 12, 19, 20, 35, 60]
    groups = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.integers(0, L, size=len(groups))
    levels = [f"l{i}" for i in range(L)]
    gen, cd, X = _seg_fit(x, groups, levels, floor=k)
    p0 = gen.global_m[0].p
    for g, n in enumerate(sizes):
        p = gen.group_m[g][0].p
        cnt = np.bincount(X[groups == g], minlength=len(cd.levels)).astype(float)
        a = max(0.25, k - n)
        assert np.allclose(p, (cnt + a * p0 + 1e-9) / (n + a + len(p0) * 1e-9), atol=1e-9)
        assert 1.0 / (n + a) <= 1.0 / k + 1e-12
        own = (p - a * p0 / (n + a))[cnt > 0] / cnt[cnt > 0]       # mass per source row
        assert (own <= 1.0 / k + 1e-9).all()
    # (b) segments of n >= k rows: as with no floor
    nofl, _, _ = _seg_fit(x, groups, levels, floor=0.0, rows=0.25)
    for g, n in enumerate(sizes):
        same = np.allclose(gen.group_m[g][0].p, nofl.group_m[g][0].p)
        assert same == (n >= k), (n, same)
    # (c) one row replaced in the 3-row segment, prior law fixed (the global marginal moves by 1 / N)
    x2 = x.copy()
    x2[0] = (x[0] + 1) % L
    gen2, _, _ = _seg_fit(x2, groups, levels, floor=k)
    tv = 0.5 * np.abs(gen.group_m[0][0].p - gen2.group_m[0][0].p).sum()
    assert tv <= 1.0 / k + 1.0 / len(x) + 1e-9

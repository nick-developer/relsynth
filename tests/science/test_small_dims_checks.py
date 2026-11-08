# SPDX-License-Identifier: Apache-2.0
"""Checks C1720-C1723: small-dimension privacy (gap plan G3;
docs/proofs/sec_g3_smalldims.tex, Props. 1720-1723), tested on the shipped
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
    (discrete: exactly, by the Dirichlet mean; continuous: E[V]), and the number
    of random numbers drawn does not depend on a."""
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
    for a in (0.25, 20.0):
        st0 = np.random.default_rng(11)
        Marginal.fit(cc, own, st0, prior=topc, dirichlet_a=a, grid=64)
        st1 = np.random.default_rng(11)
        Marginal.fit(cc, own, st1, prior=topc, dirichlet_a=0.25 if a != 0.25 else 20.0, grid=64)
        assert st0.random() == st1.random()          # same stream whatever a is
        mass = []
        for _ in range(3000):
            m = Marginal.fit(cc, own, rng, prior=topc, dirichlet_a=a, grid=256)
            mass.append((m.from_uniform(np.linspace(0, 1, 2001)[1:-1]) >= 50.0).mean())
        w = len(own) / (len(own) + a)
        assert abs(np.mean(mass) - w) < 0.03, (a, np.mean(mass), w)


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
    # redraw loop with R = infinity gives the same law: compare with an independent run
    keep = []
    while len(keep) < 20000:
        x = pd.DataFrame({"a": rng.normal(size=5000), "b": "x", "c": 0.0, "d": "p"})
        keep += x.a[~g.near_copies(x)].tolist()
    from scipy.stats import ks_2samp
    assert ks_2samp(acc[:20000], np.array(keep[:20000])).pvalue > 1e-3
    assert 0 < inB.mean() < 1


def test_c1722_no_near_copy_draws_nothing(shop):
    """Prop. 1722 (c): when no row is a near copy the guard draws no random
    number, so the tables equal the unguarded ones byte for byte."""
    base = dict(_fit(shop, near_copy="off").sample())
    zero = dict(_fit(shop, near_copy="auto", near_copy_ratio=0.0).sample())
    m = _fit(shop, near_copy="auto", near_copy_ratio=0.0)
    m.sample()
    flagged = sum(r["near_copies"] for r in m.synthesizer.sample_log.get("near_copies", {}).values())
    if flagged == 0:
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

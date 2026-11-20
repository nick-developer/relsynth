# SPDX-License-Identifier: Apache-2.0
"""Empirical-Bayes strength of the segment prior (relsynth.shrinkage, Props.
1725-1726).  The statistical checks are C1725-C1726 in
tests/science/test_small_dims_checks.py."""
from __future__ import annotations

import numpy as np
import pytest

from relsynth.shrinkage import concentration, floor_weight, icc_indicators, indicators, standardise


def _dp_segments(a, sizes, rng, discrete=False, levels=6):
    """Rows of segments F_g ~ DP(a, F0) (truncated stick breaking), F0 = N(0, 1)
    or uniform over `levels` categories."""
    x, g = [], []
    for k, s in enumerate(sizes):
        if discrete:
            p = rng.dirichlet(np.full(levels, a / levels))
            x += list(rng.choice(levels, size=s, p=p))
        else:
            V = rng.beta(1, a, size=300)
            w = V * np.cumprod(np.r_[1, 1 - V[:-1]])
            atoms = rng.normal(size=300)
            x += list(atoms[rng.choice(300, size=s, p=w / w.sum())])
        g += [k] * s
    return np.array(x), np.array(g)


@pytest.mark.parametrize("a", [1.0, 5.0, 25.0])
def test_concentration_recovers_dp_and_dirichlet(a):
    rng = np.random.default_rng(int(a))
    for discrete in (False, True):
        est = []
        for _ in range(12):
            x, g = _dp_segments(a, rng.integers(10, 80, size=60), rng, discrete)
            est.append(concentration("discrete" if discrete else "continuous", x, g, levels=6).a)
        assert 0.6 * a < np.median(est) < 1.6 * a, (discrete, a, np.median(est))


def test_icc_matches_the_balanced_anova_formula():
    rng = np.random.default_rng(3)
    G, m = 8, 5
    Y = rng.integers(0, 2, size=(G * m, 1)).astype(float)
    g = np.repeat(np.arange(G), m)
    rho, _, _ = icc_indicators(Y, g)
    means = Y[:, 0].reshape(G, m).mean(1)
    msb = m * ((means - Y.mean()) ** 2).sum() / (G - 1)
    msw = (m * means * (1 - means)).sum() / (G * m - G)
    assert rho == pytest.approx((msb - msw) / (msb + (m - 1) * msw))


def test_nested_reference_removes_the_coarse_effect():
    """Segments that differ only through their coarse group: within the coarse
    groups there is no heterogeneity, so the nested estimate pools fully."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 1, 2000), rng.normal(8, 1, 2000)])
    coarse = np.repeat([0, 1], 2000)
    seg = np.repeat(np.arange(40), 100)
    flat = concentration("continuous", x, seg, a_max=1e4)
    nested = concentration("continuous", x, seg, coarse, a_max=1e4)
    assert flat.a < 2 and nested.a > 50


def test_no_information_gives_full_pooling_and_no_random_numbers():
    x = np.arange(10.0)
    one = concentration("continuous", x, np.zeros(10, int), a_max=10.0)
    assert one.a == 10.0 and one.groups == 1
    state = np.random.get_state()[1].copy()
    rng = np.random.default_rng(0)
    x, g = _dp_segments(3.0, [30] * 10, rng)
    r1 = concentration("continuous", x, g)
    r2 = concentration("continuous", x, g)
    assert r1 == r2 and (np.random.get_state()[1] == state).all()


def test_nulls_and_shape_mode():
    rng = np.random.default_rng(9)
    # same shape, different locations and scales: shape mode sees one shape (a large),
    # the plain estimate sees very different laws (a small)
    g = np.repeat(np.arange(20), 50)
    x = rng.normal(size=1000) * (1 + g) + 10 * g
    x[::7] = np.nan
    plain = concentration("continuous", x, g, a_max=1e4)
    shape = concentration("continuous", x, g, shape=True, a_max=1e4)
    assert plain.a < 1 and shape.a > 20 and plain.rows == int((~np.isnan(x)).sum())
    z, keep = standardise(np.array([1.0, 2, 3, 4, 5, 5, 5]), np.array([0, 0, 0, 0, 1, 1, 1]))
    assert keep.tolist() == [True] * 4 + [False] * 3 and np.median(z[:4]) == 0


def test_indicators():
    Y = indicators("discrete", np.array([0, 1, -1, 1]), levels=2)
    assert Y.shape == (4, 3) and Y.sum() == 4
    Y = indicators("continuous", np.linspace(0, 1, 100))
    assert Y.shape[0] == 100 and 0 < Y.shape[1] <= 15 and ((Y.min(0) == 0) & (Y.max(0) == 1)).all()


def test_floor_weight():
    assert floor_weight(5, 2.0, 20.0) == 15.0       # floor binds
    assert floor_weight(50, 2.0, 20.0) == 2.0       # large segment: the estimate
    assert floor_weight(5, 30.0, 20.0) == 30.0      # the estimate exceeds the floor
    assert floor_weight(5, 0.0, 0.0) == 0.0
    for n in range(0, 40):
        a = floor_weight(n, 0.25, 20.0)
        assert 1.0 / (n + a) <= 1.0 / 20.0 + 1e-12  # one row carries at most 1 / floor


def test_floor_counts_the_observations_the_fit_uses():
    """Prop. 1726: the floor is computed from the non-null values (at most max_obs)
    that Marginal.fit uses, not from the segment's rows, so observations plus
    prior weight reach the floor; a floor that does not bind keeps the 2.1
    options (no shape shift)."""
    from types import SimpleNamespace

    from relsynth.generators.copula import SegmentMarginals
    gen = SegmentMarginals()
    cont = SimpleNamespace(kind="continuous")
    kw = {"dirichlet_a": 0.25, "max_obs": 50}
    every = (lambda cd: True)
    x = np.r_[np.arange(10.0), np.full(10, np.nan)]          # 20 rows, 10 values
    out = gen._seg_kw(kw, cont, 0.0, x, (0.0, 20.0), "shape", every)
    assert 10 + out["dirichlet_a"] >= 20 and out["shift_prior"] is True
    big = np.arange(80.0)                                     # 80 values, the fit keeps 50
    out = gen._seg_kw(kw, cont, 0.0, big, (0.0, 70.0), "shape", every)
    assert 50 + out["dirichlet_a"] >= 70
    out = gen._seg_kw(kw, cont, 0.0, np.arange(30.0), (0.0, 20.0), "shape", every)
    assert out is kw                                          # the floor does not bind

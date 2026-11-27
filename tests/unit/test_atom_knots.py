# SPDX-License-Identifier: Apache-2.0
"""Continuous marginals of columns with point masses (zero-inflated amounts, wages, gains)."""
import numpy as np
import pandas as pd

from relsynth.marginals import Codec, Marginal, _atom_knots


def _fit_sample(x, grid=64, seed=0, tail_trim=0.0, on=True):
    c = Codec("x", {"type": "float"}, x)
    rng = np.random.default_rng(seed)
    m = Marginal.fit(c, c.encode(x), rng, grid=grid, tail_trim=tail_trim, point_mass_knots=on)
    return m, c.decode(m.sample(len(x), rng)).astype(float)


def _w1_sd(a, b):
    n = min(len(a), len(b))
    a = np.sort(np.random.default_rng(1).choice(a, n, replace=False))
    b = np.sort(np.random.default_rng(2).choice(b, n, replace=False))
    return float(np.abs(a - b).mean() / a.std())


def _zero_inflated(n, share_zero, seed):
    rng = np.random.default_rng(seed)
    x = np.where(rng.random(n) < share_zero, 0.0, np.minimum(np.round(rng.lognormal(6.5, 0.8, n)), 9999.0))
    x[:3] = 9999.0          # top-coded, as in survey wages
    return pd.Series(x)


def test_zero_inflated_tail_is_not_spread_to_the_maximum():
    for share_zero, seed in [(0.94, 0), (0.80, 1), (0.60, 2)]:
        x = _zero_inflated(40_000, share_zero, seed)
        m, s = _fit_sample(x)
        nz, snz = x[x > 0], s[s > 0]
        assert abs(s.mean() - x.mean()) / x.mean() < 0.10
        assert snz.quantile(0.99) < 1.8 * nz.quantile(0.99)
        assert _w1_sd(x.to_numpy(), s.to_numpy()) < 0.05
        assert abs((s == 0).mean() - share_zero) < 0.01


def test_no_atoms_keeps_the_uniform_grid():
    x = pd.Series(np.random.default_rng(3).normal(size=5000))
    m, _ = _fit_sample(x)
    assert len(m.knots_c) == 65
    obs = np.sort(x.to_numpy())
    w = np.full(len(obs), 1 / len(obs))
    cw = (np.cumsum(w) - w / 2) / w.sum()
    assert _atom_knots(obs, w, cw, 64).size == 0


def test_extra_knots_stay_inside_the_trimmed_range():
    x = _zero_inflated(20_000, 0.9, 4)
    m, s = _fit_sample(x, tail_trim=0.02)
    assert m.knots_c[0] == 0.0 and m.knots_c[-1] == 1.0
    assert np.all(np.diff(m.knots_c) >= 0) and np.all(np.diff(m.knots_v) >= 0)
    assert len(m.knots_c) > 65


def test_off_by_default_keeps_the_uniform_grid():
    x = _zero_inflated(20_000, 0.9, 5)
    c = Codec("x", {"type": "float"}, x)
    m = Marginal.fit(c, c.encode(x), np.random.default_rng(0), grid=64)
    assert len(m.knots_c) == 65
    from relsynth.spec import DEFAULT_SYNTHESIS
    assert DEFAULT_SYNTHESIS["marginals"]["point_mass_knots"] is False


def test_tail_trim_at_its_upper_bound_does_not_divide_by_zero():
    """marginals.tail_trim accepts 0.5: every knot is the median, and the knot levels stay finite."""
    x = pd.Series(np.random.default_rng(2).gamma(2.0, 3.0, 2000))
    for pmk in (False, True):
        m, _ = _fit_sample(x, tail_trim=0.5, on=pmk)
        assert np.isfinite(m.knots_c).all() and np.isfinite(m.knots_v).all()

# SPDX-License-Identifier: Apache-2.0
"""Posterior-draw plumbing (relsynth.posterior): mean mode, streams and the run cache."""
from __future__ import annotations

import numpy as np
import pytest

from relsynth import posterior as P


@pytest.fixture
def mean_rng():
    return P.MeanRNG(np.random.default_rng(0))


def test_mean_rng_returns_the_means_of_parameter_laws(mean_rng):
    assert mean_rng.gamma(3.0, 2.0) == 6.0
    assert mean_rng.standard_gamma(2.5) == 2.5
    assert mean_rng.beta(2.0, 6.0) == 0.25
    assert mean_rng.chisquare(4) == 4.0
    assert mean_rng.normal(1.5, 9.0) == 1.5
    assert mean_rng.standard_normal() == 0.0
    assert mean_rng.random() == 0.5
    np.testing.assert_allclose(mean_rng.dirichlet([1.0, 3.0]), [0.25, 0.75])


def test_mean_rng_keeps_shapes(mean_rng):
    assert mean_rng.gamma(np.array([1.0, 2.0]), 3.0).tolist() == [3.0, 6.0]
    assert mean_rng.beta(1.0, 1.0, size=(2, 3)).shape == (2, 3)
    assert mean_rng.standard_normal(4).tolist() == [0.0] * 4
    assert mean_rng.random(3).tolist() == [0.5] * 3
    assert mean_rng.dirichlet([1.0, 1.0], size=3).shape == (3, 2)
    assert mean_rng.normal(np.zeros(2), 1.0).shape == (2,)


def test_mean_rng_choice_takes_the_median_cell(mean_rng):
    assert mean_rng.choice(4, p=[0.1, 0.2, 0.3, 0.4]) == 2           # cdf 0.1, 0.3, 0.6 -> first >= 0.5
    assert mean_rng.choice(["a", "b", "c"], p=[0.5, 0.25, 0.25]) == "a"
    assert mean_rng.choice(3, p=[0.0, 0.0, 1.0]) == 2
    # with a size, or without probabilities, the wrapped generator draws
    assert len(mean_rng.choice(5, size=3)) == 3


def test_mean_rng_passes_other_calls_through():
    a = P.MeanRNG(np.random.default_rng(7)).integers(0, 1000, 5)
    b = np.random.default_rng(7).integers(0, 1000, 5)
    assert a.tolist() == b.tolist()


def test_param_rng_wraps_only_in_mean_mode():
    rng = np.random.default_rng(0)
    assert P.param_rng(rng, "draw") is rng
    assert isinstance(P.param_rng(rng, "mean"), P.MeanRNG)


def test_streams_are_reproducible_and_independent():
    p0, d0 = P.run_streams(11, None, 0)
    p0b, d0b = P.run_streams(11, None, 0)
    assert p0.integers(0, 2**31, 4).tolist() == p0b.integers(0, 2**31, 4).tolist()
    assert d0.integers(0, 2**31, 4).tolist() == d0b.integers(0, 2**31, 4).tolist()
    draws = {tuple(s.integers(0, 2**31, 4)) for s in
             [*P.run_streams(11, None, 0), *P.run_streams(11, None, 1), *P.run_streams(12, None, 0)]}
    assert len(draws) == 6                       # parameter and data streams, per dataset and seed
    # a sample seed replaces the synthesis seed
    a = P.run_streams(11, 99, 0)[0].integers(0, 2**31, 4).tolist()
    b = P.run_streams(5, 99, 0)[0].integers(0, 2**31, 4).tolist()
    assert a == b


def test_run_cache_draws_once_per_key():
    cache = P.RunCache()
    rng = np.random.default_rng(0)
    w = cache.bb("k", 5, rng)
    assert w.shape == (5,) and w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert cache.bb("k", 5, rng) is w            # fixed for the run
    assert len(cache.bb("k", 6, rng)) == 6       # redrawn when the support changes


def test_run_cache_mean_mode_uses_empirical_weights():
    w = P.RunCache("mean").bb("k", 4, np.random.default_rng(0))
    assert w.tolist() == [0.25] * 4

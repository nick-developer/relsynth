# SPDX-License-Identifier: Apache-2.0
"""Gaussian copula (relsynth.copula): vine parametrisation, truncated normals,
convergence diagnostics, and posterior recovery of a known correlation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from relsynth import copula as K


def _random_corr(p, rng):
    A = rng.standard_normal((p, p + 2))
    S = A @ A.T
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


@pytest.mark.parametrize("p", [2, 3, 5])
def test_vine_chol_and_cpc_are_inverse(p):
    rng = np.random.default_rng(p)
    C = _random_corr(p, rng)
    L = K.vine_chol(K.chol_to_cpc(C), p)
    np.testing.assert_allclose(L @ L.T, C, atol=1e-9)
    np.testing.assert_allclose(L, np.linalg.cholesky(C), atol=1e-9)


def test_vine_chol_gives_a_correlation_matrix_for_any_cpcs():
    rng = np.random.default_rng(0)
    w = rng.uniform(-0.99, 0.99, 6)
    L = K.vine_chol(w, 4)
    C = L @ L.T
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() > 0
    # the first column of the vine holds the correlations with variable 0
    np.testing.assert_allclose(C[1:, 0], w[:3], atol=1e-12)


def test_chol_to_cpc_clips_extreme_partial_correlations():
    C = np.array([[1.0, 0.99999], [0.99999, 1.0]])
    assert K.chol_to_cpc(C).tolist() == [0.999]


def test_rtruncnorm_matches_the_truncated_law_on_both_tails():
    rng = np.random.default_rng(1)
    n = 40_000
    for lo, hi in [(-0.5, 1.0), (2.0, 4.0), (-4.0, -2.5), (1.5, np.inf), (-np.inf, -1.0)]:
        x = K.rtruncnorm(np.zeros(n), 1.0, np.full(n, lo), np.full(n, hi), rng)
        assert ((x >= lo) & (x <= hi)).all()
        ref = stats.truncnorm(lo, hi)
        assert stats.kstest(x, ref.cdf).pvalue > 1e-3
        assert x.mean() == pytest.approx(ref.mean(), abs=0.02)


def test_rtruncnorm_scales_and_shifts():
    rng = np.random.default_rng(2)
    n = 20_000
    x = K.rtruncnorm(np.full(n, 3.0), 2.0, np.full(n, 3.0), np.full(n, np.inf), rng)
    assert x.min() >= 3.0
    assert x.mean() == pytest.approx(3.0 + 2.0 * np.sqrt(2 / np.pi), abs=0.04)    # half-normal mean


def test_nearest_corr_repairs_an_indefinite_matrix():
    C = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(C).min() < 0
    R = K.nearest_corr(C)
    np.testing.assert_allclose(np.diag(R), 1.0)
    np.testing.assert_allclose(R, R.T)
    assert np.linalg.eigvalsh(R).min() > 0
    good = _random_corr(3, np.random.default_rng(3))
    np.testing.assert_allclose(K.nearest_corr(good), good, atol=1e-6)


def test_split_rhat():
    rng = np.random.default_rng(4)
    assert np.isnan(K.split_rhat([1.0] * 7))
    assert K.split_rhat([2.0] * 20) == 1.0
    assert K.split_rhat(rng.standard_normal(4000)) == pytest.approx(1.0, abs=0.01)
    drift = np.r_[rng.standard_normal(500), 3 + rng.standard_normal(500)]
    assert K.split_rhat(drift) > 1.5
    # the formula: sqrt(((h-1)/h W + B/h) / W) with two halves
    x = np.array([0.0, 1, 2, 3, 10, 11, 12, 13])
    h = 4
    W = np.mean([np.var(x[:4], ddof=1), np.var(x[4:], ddof=1)])
    B = h * np.var([x[:4].mean(), x[4:].mean()], ddof=1)
    assert K.split_rhat(x) == pytest.approx(np.sqrt(((h - 1) / h * W + B / h) / W))


def test_batch_mcse():
    rng = np.random.default_rng(5)
    assert np.isnan(K.batch_mcse([1.0])) and np.isnan(K.batch_mcse([]))
    x = rng.standard_normal(10_000)
    assert K.batch_mcse(x) == pytest.approx(1 / np.sqrt(len(x)), rel=0.25)
    # positively autocorrelated chain: larger standard error than iid
    ar = np.zeros(10_000)
    for i in range(1, len(ar)):
        ar[i] = 0.9 * ar[i - 1] + rng.standard_normal()
    assert K.batch_mcse(ar) > 3 * K.batch_mcse(rng.standard_normal(10_000) * ar.std())
    # exact batch means: 4 batches of 2 for 8 draws (b = floor(sqrt 8) = 2)
    y = np.arange(8.0)
    m = y.reshape(4, 2).mean(1)
    assert K.batch_mcse(y) == pytest.approx(np.sqrt(2 * m.var(ddof=1) / 8))


def test_chain_converged():
    rng = np.random.default_rng(6)
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert K.chain_converged([C] * 5, [0.0] * 5, 1.05) == (True, {})
    good = [C + np.array([[0, e], [e, 0]]) for e in 0.01 * rng.standard_normal(200)]
    ok, det = K.chain_converged(good, [np.log(np.linalg.det(c)) for c in good], 1.05)
    assert ok and det["unconverged_entries"] == 0
    bad = [C + np.array([[0, e], [e, 0]]) for e in np.r_[np.full(100, -0.3), np.full(100, 0.3)] + 0.01 * rng.standard_normal(200)]
    ok, det = K.chain_converged(bad, [np.log(np.linalg.det(c)) for c in bad], 1.05)
    assert not ok and det["unconverged_pairs"] == [[0, 1]]


def _latent_bins(X, edges):
    """Bin bounds on the normal-score scale for each column (ordinal coarsening)."""
    lo = np.empty_like(X)
    hi = np.empty_like(X)
    for j in range(X.shape[1]):
        e = np.r_[-np.inf, edges, np.inf]
        k = np.searchsorted(edges, X[:, j])
        lo[:, j], hi[:, j] = e[k], e[k + 1]
    return lo, hi


@pytest.mark.parametrize("mode", ["bayes", "rank"])
def test_copula_recovers_a_known_correlation_from_coarsened_data(mode):
    rng = np.random.default_rng(7)
    R = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    X = rng.multivariate_normal(np.zeros(3), R, size=1500)
    lo, hi = _latent_bins(X, np.array([-1.0, -0.3, 0.3, 1.0]))
    cop = K.GaussianCopula(3)
    cop.fit(lo, hi, rng, iters=150, burn_in=50, mode=mode)
    M = cop.mean()
    assert np.abs(M - R).max() < 0.08
    # draws, the fixed per-dataset matrix and samples
    cop.set_current(rng, mode="mean")
    np.testing.assert_allclose(cop.draw(rng), M)
    Z = cop.sample(20_000, rng)
    np.testing.assert_allclose(np.corrcoef(Z.T), M, atol=0.03)
    # conditional draws given column 0 keep the regression on it
    za = rng.standard_normal((20_000, 1))
    zr = cop.sample_conditional(za, [0], [1, 2], rng)
    np.testing.assert_allclose([np.corrcoef(za[:, 0], zr[:, k])[0, 1] for k in range(2)], M[0, 1:], atol=0.03)

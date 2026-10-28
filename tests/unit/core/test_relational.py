# SPDX-License-Identifier: Apache-2.0
"""Relational latent-segment model (relsynth.relational): spike-and-slab draws
against their exact posteriors, the collapsed marginals, the grain test, and a
small star schema fitted end to end with known segments."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.special import betaln, gammaln

from relsynth import relational as R

N = 40_000


def test_zi_gamma_spike_probability_is_the_exact_posterior():
    rng = np.random.default_rng(0)
    a, b, pi, E = 2.0, 1.0, 0.3, 4.0
    lam = R.zi_gamma_draw(np.zeros(N), np.full(N, E), a, b, pi, rng)
    p_spike = pi / (pi + (1 - pi) * (b / (b + E)) ** a)
    assert (lam == 0).mean() == pytest.approx(p_spike, abs=0.01)
    slab = lam[lam > 0]
    assert slab.mean() == pytest.approx(a / (b + E), rel=0.03)                 # Gamma(a + 0, b + E)
    # a positive count rules the spike out; no spike prior gives the conjugate draw
    lam = R.zi_gamma_draw(np.full(N, 3.0), np.full(N, E), a, b, pi, rng)
    assert (lam > 0).all() and lam.mean() == pytest.approx((a + 3) / (b + E), rel=0.03)
    lam = R.zi_gamma_draw(np.zeros(N), np.full(N, E), a, b, 0.0, rng)
    assert (lam > 0).all()


def test_zoib_draw_component_probabilities():
    rng = np.random.default_rng(1)
    a, b, pi = 1.0, 1.0, 0.4
    for S, T in [(0, 5), (5, 5), (2, 5)]:
        p = R.zoib_draw(np.full(N, float(S)), np.full(N, float(T)), a, b, pi, rng)
        w0 = pi / 2 * (S == 0)
        w1 = pi / 2 * (S == T)
        ws = (1 - pi) * np.exp(betaln(a + S, b + T - S) - betaln(a, b))
        tot = w0 + w1 + ws
        assert (p == 0).mean() == pytest.approx(w0 / tot, abs=0.01)
        assert (p == 1).mean() == pytest.approx(w1 / tot, abs=0.01)
        free = p[(p > 0) & (p < 1)]
        assert free.mean() == pytest.approx((a + S) / (a + b + T), abs=0.01)
    # learned weights replace the fixed prior split
    p = R.zoib_draw(np.zeros(N), np.full(N, 5.0), a, b, pi, rng, w=[0.8, 0.1, 0.1])
    ws = 0.1 * np.exp(betaln(a, b + 5) - betaln(a, b))
    assert (p == 0).mean() == pytest.approx(0.8 / (0.8 + ws), abs=0.01)
    assert (R.zoib_draw(np.zeros(100), np.full(100, 5.0), a, b, 0.0, rng) > 0).all()


def test_grouped_zoib_shares_structural_zeros_within_a_group():
    rng = np.random.default_rng(2)
    S = np.array([[0.0, 3.0], [0.0, 2.0], [0.0, 1.0]])
    T = np.array([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    zero_together = 0
    for _ in range(400):
        p = R.grouped_zoib_draw(S, T, 1.0, 1.0, 0.5, np.array([0, 0, 0]), rng)
        assert (p[:, 1] > 0).all()                         # observed column is never a structural zero
        zero_together += (p[:, 0] == 0).all()
    # P(all zero) for the group: exact posterior of the shared indicator
    l0 = np.log(0.25)
    lm = np.logaddexp(np.log(0.25), np.log(0.5) + betaln(1, 1 + 2) - betaln(1, 1))
    lfree = 3 * lm
    p_all0 = 1 / (1 + np.exp(np.log(0.5) + lfree - l0))
    assert zero_together / 400 >= p_all0 - 0.08
    single = R.grouped_zoib_draw(S, T, 1.0, 1.0, 0.5, np.array([0, 1, 2]), rng)
    assert single.shape == S.shape


def test_eb_beta_rows_moment_estimator():
    S = np.array([[1.0, 2, 3, 4, 0], [5.0, 5, 5, 5, 5]])
    T = np.full((2, 5), 10.0)
    A, B = R.eb_beta_rows(S, T, 1.0, 1.0)
    s, t = S[0, :4], T[0, :4]
    m = s.sum() / t.sum()
    tau2 = (s / t).var(ddof=1) - np.mean(m * (1 - m) / t)
    c = 1e4 if tau2 <= 0 else np.clip(m * (1 - m) / tau2 - 1, 1, 1e4)
    assert A[0, 0] == pytest.approx(c * m) and B[0, 0] == pytest.approx(c * (1 - m))
    # row 1 is constant: no between-entry variance, maximum concentration
    assert A[1, 0] / (A[1, 0] + B[1, 0]) == pytest.approx(0.5) and A[1, 0] + B[1, 0] == pytest.approx(1e4)
    A, B = R.eb_beta_rows(np.zeros((1, 3)), np.full((1, 3), 4.0), 2.0, 3.0)
    assert A.tolist() == [[2.0] * 3] and B.tolist() == [[3.0] * 3]


def test_sparse_dirichlet_keeps_observed_levels_and_drops_absent_ones():
    rng = np.random.default_rng(3)
    counts = np.array([30.0, 10.0, 0.0, 0.0])
    keep_rate = np.zeros(4)
    for _ in range(300):
        p = R.sparse_dirichlet_draw(counts, np.ones(4), 0.5, rng)
        assert p.sum() == pytest.approx(1.0) and (p[:2] > 0).all()
        keep_rate += p > 0
    keep_rate /= 300
    assert keep_rate[2] < 0.5 and keep_rate[3] < 0.5     # 40 rows and none in level 2 or 3
    assert (R.sparse_dirichlet_draw(counts, np.ones(4), 0.0, rng) > 0).all()
    assert (R.sparse_dirichlet_draw(counts, np.ones(4), 1.0, rng)[2:] == 0).all()
    p = R.sparse_dirichlet_draw(np.zeros(3), np.ones(3), 0.5, rng)
    assert p.sum() == pytest.approx(1.0)


def test_gumbel_argmax_samples_the_softmax():
    rng = np.random.default_rng(4)
    logp = np.log(np.array([[0.2, 0.5, 0.3]] * N))
    z = R.gumbel_argmax(logp, rng)
    np.testing.assert_allclose(np.bincount(z, minlength=3) / N, [0.2, 0.5, 0.3], atol=0.01)


def test_kmeanspp_separates_clusters():
    rng = np.random.default_rng(5)
    X = np.r_[rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2))]
    z = R.kmeanspp(X, 2, rng)
    assert len(set(z[:50])) == 1 and len(set(z[50:])) == 1 and z[0] != z[-1]
    assert len(R.kmeanspp(np.zeros((0, 2)), 3, rng)) == 0
    assert set(R.kmeanspp(X[:2], 5, rng)) <= {0, 1}


def test_slab_rate_and_multiplicity_marginal():
    assert R.slab_rate([4.0, 6.0, 5.0], [2.0, 4.0, 0.0], 2.0, 9.0) == pytest.approx(2.0 / (6 / 10))
    assert R.slab_rate([4.0], [0.0], 2.0, 9.0) == 9.0
    E = np.array([[2.0, 0.0], [3.0, 1.0]])
    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    am, bm = 2.0, 1.0

    def gp(e, x):
        return am * np.log(bm) - gammaln(am) + gammaln(am + x) - (am + x) * np.log(bm + e)
    assert R.multiplicity_marginal(E, X, "block", am, bm) == pytest.approx(gp(2, 1) + gp(3, 0) + gp(1, 2))
    assert R.multiplicity_marginal(E, X, "partner", am, bm) == pytest.approx(gp(5, 1) + gp(1, 2))
    assert R.multiplicity_marginal(E, X, "anchor", am, bm) == pytest.approx(gp(2, 1) + gp(4, 2))
    assert R.multiplicity_marginal(E, X, "none", am, bm) == -np.inf
    assert R.multiplicity_marginal(E, 0 * X, "none", am, bm) == 0.0
    pi = 0.2
    b2 = R.slab_rate(E.ravel(), X.ravel(), am, bm)

    def gs(e, x):
        return am * np.log(b2) - gammaln(am) + gammaln(am + x) - (am + x) * np.log(b2 + e)
    want = sum(np.logaddexp(np.log(pi) if x == 0 else -np.inf, np.log1p(-pi) + gs(e, x))
               for e, x in [(2, 1), (3, 0), (1, 2)])
    assert R.multiplicity_marginal(E, X, "block", am, bm, pi_s=pi) == pytest.approx(want)


def test_poisson_dup_test_detects_a_grain():
    rng = np.random.default_rng(6)
    n1, n2 = 200, 50
    # Poisson facts: repeated pairs as the rank-1 model expects
    u = rng.integers(n1, size=3000)
    v = rng.integers(n2, size=3000)
    pairs, c = np.unique(np.c_[u, v], axis=0, return_counts=True)
    obs, exp_, z = R.poisson_dup_test(pairs[:, 0], pairs[:, 1], c, n1, n2, rng)
    assert abs(z) < 4 and obs == (c >= 2).sum()
    lam = 3000 / (n1 * n2)
    assert exp_ == pytest.approx(n1 * n2 * (1 - np.exp(-lam) * (1 + lam)), rel=0.1)
    # a grain: every pair at most once
    z_grain = R.poisson_dup_test(pairs[:, 0], pairs[:, 1], np.ones(len(pairs)), n1, n2, rng)[2]
    assert z_grain < -10
    # the sampled branch agrees with the exact one
    zs = R.poisson_dup_test(pairs[:, 0], pairs[:, 1], c, n1, n2, rng, max_pairs=5000)[2]
    assert zs == pytest.approx(z, abs=1.5)
    assert R.poisson_dup_test(np.array([0]), np.array([0]), np.array([0.0]), 1, 1, rng) == (0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def fitted_star():
    from conftest import scenario

    from relsynth import Synthesizer, load_spec
    t, s, truth = scenario("S2_star")
    syn = Synthesizer(load_spec(s))
    syn.fit(t)
    return t, syn, truth


def test_star_schema_end_to_end(fitted_star):
    real, syn, truth = fitted_star
    out = syn.sample()
    assert len(out["customer"]) == len(real["customer"]) and len(out["product"]) == len(real["product"])
    assert len(out["sales"]) == pytest.approx(len(real["sales"]), rel=0.05)
    assert out["sales"]["customer_id"].isin(out["customer"]["customer_id"]).all()
    assert out["sales"]["product_id"].isin(out["product"]["product_id"]).all()
    # the latent segments explain the customers' purchase counts: per-customer
    # dispersion is above Poisson in the real data and stays so
    def disp(d):
        c = d["sales"]["customer_id"].value_counts().reindex(d["customer"]["customer_id"], fill_value=0)
        return c.var() / c.mean()
    assert disp(real) > 1.1 and disp(out) == pytest.approx(disp(real), rel=0.15)
    # dependence between a customer attribute and purchase volume is kept
    def corr(d):
        c = d["sales"]["customer_id"].value_counts().reindex(d["customer"]["customer_id"], fill_value=0)
        return stats.spearmanr(d["customer"]["income"], c.to_numpy()).statistic
    assert corr(out) == pytest.approx(corr(real), abs=0.15)

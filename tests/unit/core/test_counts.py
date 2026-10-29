# SPDX-License-Identifier: Apache-2.0
"""Count-model choice (relsynth.counts) against closed forms and brute force.

The Polya-urn marginal is checked against the sequential product that defines
it, the Poisson and negative-binomial likelihoods against scipy, and the
decision rule against its Bayes-factor threshold.  These run in milliseconds,
so mutation testing can use them (docs/testing.md).
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from relsynth import counts as C

count_lists = st.lists(st.integers(min_value=0, max_value=25), min_size=1, max_size=30)


def urn_sequential(t, a):
    """log p(t_1..t_n) = sum_i log((a f0(t_i) + #{j < i: t_j = t_i}) / (a + i - 1)), f0 = Poisson(mean)."""
    t = list(t)
    lam = max(float(np.mean(t)), 1e-12)
    out, seen = 0.0, {}
    for i, x in enumerate(t, start=1):
        out += np.log(a * stats.poisson.pmf(x, lam) + seen.get(x, 0)) - np.log(a + i - 1)
        seen[x] = seen.get(x, 0) + 1
    return out


@given(count_lists, st.sampled_from([0.1, 1.0, 5.0]))
@settings(max_examples=150, deadline=None)
def test_dp_marginal_equals_the_urn_product(t, a):
    assert C.dp_logmarginal(t, a) == pytest.approx(urn_sequential(t, a), rel=1e-9, abs=1e-9)


@given(count_lists)
@settings(max_examples=60, deadline=None)
def test_dp_marginal_is_exchangeable(t):
    rng = np.random.default_rng(len(t))
    assert C.dp_logmarginal(rng.permutation(t)) == pytest.approx(C.dp_logmarginal(t), abs=1e-9)


def test_dp_marginal_edge_cases():
    assert C.dp_logmarginal([]) == 0.0
    # one observation: the urn draws it from the base measure
    assert C.dp_logmarginal([3], 2.0) == pytest.approx(np.log(stats.poisson.pmf(3, 3.0)))
    # counts far in the Poisson tail stay finite (log-space computation)
    assert np.isfinite(C.dp_logmarginal([0] * 50 + [400]))


@given(count_lists, st.floats(min_value=0.05, max_value=30))
@settings(max_examples=100, deadline=None)
def test_poisson_loglik_matches_scipy(t, lam):
    assert C.poisson_loglik(t, lam) == pytest.approx(stats.poisson.logpmf(t, lam).sum(), rel=1e-9, abs=1e-9)


def test_poisson_loglik_degenerate_rate():
    assert C.poisson_loglik([0, 0, 0], 0.0) == 0.0
    assert C.poisson_loglik([0, 1], 0.0) == -np.inf


def test_nb_loglik_is_the_profile_maximum():
    rng = np.random.default_rng(1)
    t = rng.negative_binomial(2, 2 / (2 + 6.0), size=400)     # mean 6, strongly over-dispersed
    m = t.mean()
    nb = C.nb_loglik(t)
    grid = [stats.nbinom.logpmf(t, r, r / (r + m)).sum() for r in np.exp(np.linspace(-3, 6, 400))]
    assert nb >= max(grid) - 1e-6                              # the optimiser finds the maximum
    assert nb == pytest.approx(max(grid), abs=0.5)
    assert nb > C.poisson_loglik(t, m) + 100                   # and it beats Poisson clearly


def test_nb_loglik_falls_back_to_poisson_when_not_overdispersed():
    t = np.array([3, 4, 5, 4, 3, 5, 4])                        # variance < mean
    assert C.nb_loglik(t) == C.poisson_loglik(t, t.mean())
    assert C.nb_loglik([7]) == C.poisson_loglik([7], 7.0)      # a single count


def test_evidence_skips_tiny_and_empty_segments():
    lbf, det = C.count_model_evidence([[5], [0, 0, 0], []])
    assert lbf == 0.0 and det["n"] == 0
    lbf2, det2 = C.count_model_evidence([[1, 2], [5], [0, 0]])
    assert det2["n"] == 2


def test_evidence_penalises_each_family_by_its_parameters():
    t = np.array([2, 3, 2, 4, 3, 2, 3, 3])
    n = len(t)
    lbf, det = C.count_model_evidence([t], a=1.0)
    pen = 0.5 * np.log(n)
    dp = C.dp_logmarginal(t, 1.0) - pen
    pois = C.poisson_loglik(t, t.mean()) - pen
    nb = C.nb_loglik(t) - 2 * pen
    assert lbf == pytest.approx(dp - max(pois, nb))
    assert det["parametric"] == ("nb" if nb > pois else "poisson")


def test_regular_counts_choose_the_dp_and_poisson_counts_do_not():
    # every parent has exactly 7 children: far more regular than Poisson
    state, det = C.choose_count_model([[7] * 200])
    assert state is not None and det["log_bf_dp"] > np.log(100)
    rng = np.random.default_rng(0)
    state, det = C.choose_count_model([rng.poisson(4.0, 300)])
    assert state is None and det["log_bf_dp"] < np.log(100)


def test_choice_settings_override_the_evidence():
    rng = np.random.default_rng(0)
    pois = [rng.poisson(4.0, 300)]
    assert C.choose_count_model([[7] * 200], setting="poisson") == (None, {})
    state, _ = C.choose_count_model(pois, setting="dp", a=2.5)
    assert state["a"] == 2.5 and np.array_equal(state["counts"][0], pois[0])


def test_bayes_factor_threshold_is_strict():
    groups = [[7] * 20]
    lbf, _ = C.count_model_evidence(groups)
    assert C.choose_count_model(groups, bf=np.exp(lbf) * 1.001)[0] is None
    assert C.choose_count_model(groups, bf=np.exp(lbf) * 0.999)[0] is not None


def _evidence_by_hand(groups, a):
    dp = pois = nb = 0.0
    for t in groups:
        t = np.asarray(t)
        pen = 0.5 * np.log(len(t))
        dp += C.dp_logmarginal(t, a) - pen
        pois += C.poisson_loglik(t, t.mean()) - pen
        nb += C.nb_loglik(t) - 2 * pen
    return dp - max(pois, nb)


def test_evidence_sums_segments_and_skips_only_the_unusable_ones():
    rng = np.random.default_rng(3)
    good = [rng.poisson(3.0, 40), [7] * 30, rng.negative_binomial(2, 0.3, 50)]
    lbf, det = C.count_model_evidence([[4], good[0], [0, 0], good[1], [], good[2]], a=2.0)
    assert lbf == pytest.approx(_evidence_by_hand(good, 2.0))
    assert det["n"] == 120


def test_evidence_uses_the_concentration_given():
    t = [[7] * 25 + [3]]
    lo, _ = C.count_model_evidence(t, a=0.2)
    hi, _ = C.count_model_evidence(t, a=20.0)
    assert lo == pytest.approx(_evidence_by_hand(t, 0.2)) and hi == pytest.approx(_evidence_by_hand(t, 20.0))
    assert lo != pytest.approx(hi)
    # the choice passes it on: the threshold is crossed at one concentration and not at the other
    bf = np.exp((lo + hi) / 2)
    assert (C.choose_count_model(t, a=0.2, bf=bf)[0] is None) != (C.choose_count_model(t, a=20.0, bf=bf)[0] is None)


def test_nb_loglik_on_float_counts_and_zero_mean():
    assert C.nb_loglik([0, 0, 0]) == 0.0
    t = np.array([0.0, 1.0, 9.0, 0.0, 14.0])
    assert C.nb_loglik(t) == pytest.approx(C.nb_loglik(t.astype(int)))
    assert C.nb_loglik(t) > C.poisson_loglik(t, t.mean())

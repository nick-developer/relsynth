# SPDX-License-Identifier: Apache-2.0
"""Neural generator plug-ins, the parts that need no torch: the DP-SGD accountant
(Prop. 1803), the diffusion schedule, and the registry's report of a missing torch.
Kept apart from test_neural.py, which is skipped without torch, so that these run in
every CI job."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from relsynth.generators import Registry, check
from relsynth.generators.neural import accountant as acc
from relsynth.generators.neural.ddpm import cosine_abar


# ---------------------------------------------------------------- accountant (Prop. 1803)
def test_full_batch_is_the_gaussian_mechanism():
    """q = 1: eps_RDP(alpha) = alpha / (2 sigma^2), i.e. rho = T / (2 sigma^2) zCDP."""
    r = acc.rdp_sampled_gaussian(1.0, 2.0, orders=(2, 5, 64))
    assert np.allclose(r, np.array([2, 5, 64]) / 8.0)
    assert acc.zcdp_rho(2.0, 10) == pytest.approx(10 / 8)


@pytest.mark.parametrize("q,sigma,alpha", [(0.01, 1.0, 2), (0.05, 0.8, 8), (0.2, 2.0, 16), (0.5, 3.0, 5)])
def test_sampled_gaussian_matches_numerical_integration(q, sigma, alpha):
    """The closed form equals E_{mu0}[(mu/mu0)^alpha] with mu0 = N(0, s^2), mu = (1-q) mu0 + q N(1, s^2),
    integrated numerically (Mironov, Talwar and Zhang 2019, sec. 3.3)."""
    def f(x):     # mu(x) / mu0(x) = (1 - q) + q exp((2x - 1) / (2 s^2))
        return norm.pdf(x, 0, sigma) * ((1 - q) + q * math.exp((2 * x - 1) / (2 * sigma ** 2))) ** alpha
    hi = 12 * sigma + alpha        # the integrand's mass lies below alpha + 12 sigma
    val, _ = integrate.quad(f, -12 * sigma, hi, limit=400, epsabs=0, epsrel=1e-10)
    want = math.log(val) / (alpha - 1)
    got = acc.rdp_sampled_gaussian(q, sigma, orders=(alpha,))[0]
    assert got == pytest.approx(want, rel=1e-6, abs=1e-12)


def test_epsilon_is_monotone_and_calibration_inverts_it():
    e1 = acc.epsilon(0.01, 1.0, 1000, 1e-5)[0]
    e2 = acc.epsilon(0.01, 1.5, 1000, 1e-5)[0]
    e3 = acc.epsilon(0.01, 1.0, 2000, 1e-5)[0]
    assert e2 < e1 < e3
    s = acc.calibrate_sigma(2.0, 1e-5, 0.01, 1000)
    assert acc.epsilon(0.01, s, 1000, 1e-5)[0] <= 2.0
    assert acc.epsilon(0.01, s * 0.99, 1000, 1e-5)[0] > 2.0
    with pytest.raises(ValueError, match="out of reach"):
        acc.calibrate_sigma(1e-6, 1e-5, 1.0, 10 ** 6)


def test_cosine_schedule():
    ab = cosine_abar(1000)
    assert ab[0] == 1.0 and np.all(np.diff(ab) < 0) and 0 < ab[-1] < 1e-3


# ---------------------------------------------------------------- registry without torch
def test_neural_generators_are_registered_and_report_a_missing_torch(monkeypatch):
    reg = Registry(entry_points=False)
    assert {"tvae", "ddpm"} <= set(reg.names())
    info = {g["name"]: g for g in reg.list()}
    assert info["ddpm"]["capabilities"]["gpu"] == "optional"
    from relsynth.generators.neural import train
    monkeypatch.setattr(train, "torch_missing", lambda: train.INSTALL_HINT)
    spec = {"tables": {"t": {"columns": {"x": {"type": "float"}}}}, "synthesis": {}}
    errs, _ = check(reg.entry("tvae").cls, spec, "t")
    assert any("relsynth[neural]" in e for e in errs)


# ---------------------------------------------------------------- encoding helpers (no torch)
def test_score_map_is_increasing_on_small_samples():
    """Prop. 1805's calibration map must be strictly increasing for any sample size, also when the
    sample is smaller than the 1999-point inner grid (the end knots stay outside it)."""
    from relsynth.generators.neural.base import score_map
    rng = np.random.default_rng(0)
    for m in (5, 300, 999, 1000, 20000):
        x, y = score_map(rng.normal(size=m))
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0), m


def test_missing_parent_values_are_neutral_in_the_conditioning_vector():
    """A missing value in a parent column without nulls at fit time is not read as level 0 or
    as the lowest score: its one-hot block is all zero and its score is 0."""
    from types import SimpleNamespace as NS

    from relsynth.generators.neural.features import FeatureSpace

    class Marg:                       # a continuous marginal without nulls: N(0, 1) scores
        p_null = 0.0

        def latent_bounds(self, x):
            x = np.asarray(x, float)
            return np.where(np.isnan(x), -np.inf, x - 0.01), np.where(np.isnan(x), -np.inf, x + 0.01)

    data = NS(X=[np.array([0, 1, 0, 1]), np.array([0.1, -0.2, 0.3, 0.0])],
              codecs=[NS(kind="discrete", levels=["a", "b"]), NS(kind="continuous")])
    fs = FeatureSpace(data, [0, 1], 64, [None, Marg()])
    A = fs.dense([np.array([-1, 1]), np.array([np.nan, 0.5])], np.arange(2), np.random.default_rng(0))
    # columns: score of column 1, then the one-hot block of column 0 (a, b)
    assert A[0].tolist() == [0.0, 0.0, 0.0]
    assert A[1, 1:].tolist() == [0.0, 1.0] and abs(A[1, 0] - 0.5) < 0.05


def test_a_broken_unavailable_hook_does_not_break_the_registry():
    from relsynth.generators.independent import IndependentGenerator

    class Broken(IndependentGenerator):
        name = "broken-hook"

        def unavailable(self):        # an instance method by mistake: calling it on the class fails
            return "never"

    reg = Registry(entry_points=False)
    reg.register(Broken)
    info = {g["name"]: g for g in reg.list()}
    assert "unavailable() failed" in info["broken-hook"]["unavailable"]
    errs, _ = check(Broken, {"tables": {"t": {"columns": {}}}, "synthesis": {}}, "t")
    assert any("unavailable() failed" in e for e in errs)


def test_dp_options_reject_non_finite_values():
    from relsynth.generators import GeneratorError
    from relsynth.generators.neural.train import DPOptions
    for bad in ({"epsilon": 1, "clip": float("nan")}, {"epsilon": 1, "clip": float("inf")},
                {"epsilon": 1, "clip": float("-inf")}, {"epsilon": float("inf")}, {"epsilon": 1, "clip": 0},
                {"epsilon": 1, "delta": 1.5}, {"epsilon": 1, "clip": "x"}):
        with pytest.raises(GeneratorError):
            DPOptions.parse(bad, "t")
    assert DPOptions.parse({"epsilon": 2, "clip": 0.5}, "t").clip == 0.5
    for bad_pop in (0, -3, 2.5, True, "100"):
        with pytest.raises(GeneratorError):
            DPOptions.parse({"epsilon": 1, "population": bad_pop}, "t")
    d = DPOptions.parse({"epsilon": 1, "population": 500}, "t")
    assert d.public_count(37) == 500 and DPOptions(1.0).public_count(37) == 37


def test_dp_noise_comes_from_the_os_csprng(monkeypatch):
    """The privacy-relevant draws of DP-SGD: uniforms with 53 random bits and Box-Muller normals
    from os.urandom.  The distribution checks (moments and a Kolmogorov-Smirnov test) run on fixed
    test-local bytes so they are reproducible; production draws stay unseeded."""
    from scipy.stats import kstest
    from relsynth.generators.neural import train
    calls = []
    real = train.os.urandom

    def spy(k):
        calls.append(k)
        return real(k)
    monkeypatch.setattr(train.os, "urandom", spy)
    assert not np.array_equal(train.os_uniform(8), train.os_uniform(8))
    assert calls == [64, 64]                       # 8 bytes per uniform, from the OS
    fixed = np.random.default_rng(20261121)
    monkeypatch.setattr(train.os, "urandom", lambda k: fixed.bytes(k))
    u = train.os_uniform(200_000)
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.005
    z = train.os_normal(200_001)
    assert len(z) == 200_001 and abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert kstest(z, "norm").pvalue > 1e-4

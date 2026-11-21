# SPDX-License-Identifier: Apache-2.0
"""Renyi-DP accountant of DP-SGD (the Poisson-subsampled Gaussian mechanism).

DP-SGD (Abadi et al. 2016) clips every row's gradient to L2 norm C, sums the
clipped gradients of a Poisson sample (each row in the batch independently with
probability q) and adds N(0, sigma^2 C^2) noise.  One step is the sampled
Gaussian mechanism with noise multiplier sigma.  Its Renyi divergence of
integer order alpha >= 2 is exactly (Mironov, Talwar and Zhang 2019, sec. 3.3)

    eps_RDP(alpha) = log A_alpha / (alpha - 1),
    A_alpha = sum_{k=0}^{alpha} C(alpha, k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2)),

computed here in log space.  T steps compose additively (Mironov 2017,
Prop. 1), and the RDP curve converts to (epsilon, delta)-DP by the conversion
of Canonne, Kamath and Steinke (2020, Prop. 12; also Balle et al. 2020, Thm. 21):

    epsilon = min_alpha  T eps_RDP(alpha) + log((alpha - 1) / alpha) - (log delta + log alpha) / (alpha - 1).

With q = 1 (no sampling) the mechanism is the Gaussian mechanism and
eps_RDP(alpha) = alpha / (2 sigma^2): rho-zCDP with rho = T / (2 sigma^2)
(Bun and Steinke 2016), the unit of relsynth's DP ledger.

Everything is plain numpy and scipy so that it can be checked without torch
(Prop. 1803, tests/unit/test_neural_accountant.py).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = tuple(range(2, 257)) + (320, 384, 448, 512)


def rdp_sampled_gaussian(q: float, sigma: float, orders=ORDERS) -> np.ndarray:
    """eps_RDP(alpha) of one step, for each integer order alpha >= 2."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("sampling rate q must lie in [0, 1]")
    if sigma <= 0:
        raise ValueError("noise multiplier sigma must be positive")
    out = np.empty(len(orders))
    for i, a in enumerate(orders):
        a = int(a)
        if a < 2:
            raise ValueError("orders must be integers >= 2")
        if q == 0.0:
            out[i] = 0.0
            continue
        if q == 1.0:
            out[i] = a / (2.0 * sigma * sigma)
            continue
        k = np.arange(a + 1, dtype=float)
        logc = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
        terms = logc + (a - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2.0 * sigma * sigma)
        out[i] = float(logsumexp(terms)) / (a - 1)
    return out


def epsilon(q: float, sigma: float, steps: int, delta: float, orders=ORDERS) -> tuple[float, int]:
    """(epsilon, best order) of `steps` DP-SGD steps at the given delta."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if steps <= 0:
        return 0.0, int(orders[0])
    a = np.asarray(orders, dtype=float)
    rdp = steps * rdp_sampled_gaussian(q, sigma, orders)
    eps = rdp + np.log1p(-1.0 / a) - (math.log(delta) + np.log(a)) / (a - 1.0)
    i = int(np.argmin(eps))
    return max(float(eps[i]), 0.0), int(orders[i])


def calibrate_sigma(target_epsilon: float, delta: float, q: float, steps: int, lo: float = 0.3,
                    hi: float = 2000.0, tol: float = 1e-4) -> float:
    """The smallest noise multiplier whose epsilon after `steps` steps is at most
    target_epsilon (bisection; epsilon falls monotonically in sigma)."""
    if target_epsilon <= 0:
        raise ValueError("target epsilon must be positive")
    if epsilon(q, hi, steps, delta)[0] > target_epsilon:
        raise ValueError(f"epsilon {target_epsilon} is out of reach with {steps} steps at sampling rate {q:.4g}: "
                         "use fewer epochs or a smaller batch")
    if epsilon(q, lo, steps, delta)[0] <= target_epsilon:
        return lo
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if epsilon(q, mid, steps, delta)[0] <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return hi


def zcdp_rho(sigma: float, steps: int) -> float:
    """rho of `steps` full-batch (q = 1) Gaussian steps: steps / (2 sigma^2)."""
    return steps / (2.0 * sigma * sigma)

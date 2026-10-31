# SPDX-License-Identifier: Apache-2.0
"""Privacy-loss accounting across mechanisms and releases (Props. 800, 801).

Every charge is converted to delta-approximate rho-zCDP and charges add up
(Bun and Steinke 2016).  The total is stated as epsilon at a target delta with
the conversion of Canonne, Kamath and Steinke (2020, Prop. 12), which is never
looser than the classic rho + 2 sqrt(rho log 1/delta); when every charge is
pure or approximate DP, basic composition is used instead if it is smaller.

All conversions round *against* the claim: a stated epsilon is an upper
bound of the exact conversion, and a cap converted to rho is a lower bound,
so a ledger built on these functions is never optimistic.
"""
from __future__ import annotations

import math
from decimal import Decimal

__all__ = ["to_azcdp", "compose", "cks_delta", "epsilon_of", "rho_of", "ledger_epsilon", "dec"]

_SAFETY = 1e-9


def dec(x, places: int = 12) -> str:
    """A privacy figure as a decimal string, as manifests store them (exact to hash)."""
    if x == math.inf:
        return "inf"
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places)).normalize()
    return format(d, "f")


def to_azcdp(charge: dict) -> tuple[float, float]:
    """One charge as delta-approximate rho-zCDP: (rho, delta).

    pure eps          -> (eps^2/2, 0)    Bun and Steinke 2016, Prop. 1.4
    zcdp rho          -> (rho, 0)
    approx (eps, del) -> (eps^2/2, del)  approximate zCDP, Bun and Steinke 2016, Def. 1.10
    """
    kind = charge["kind"]
    if kind == "pure":
        return float(charge["epsilon"]) ** 2 / 2, 0.0
    if kind == "zcdp":
        return float(charge["rho"]), 0.0
    if kind == "approx":
        return float(charge["epsilon"]) ** 2 / 2, float(charge["delta"])
    raise ValueError(f"unknown charge kind {kind!r}")


def compose(charges) -> tuple[float, float]:
    """Adaptive composition: rho and delta add (Prop. 800 a)."""
    rho = dlt = 0.0
    for c in charges:
        r, d = to_azcdp(c)
        rho += r
        dlt += d
    return rho, dlt


def _logd(rho, eps, la):
    a = 1 + math.exp(la)
    return (a - 1) * (a * rho - eps) - math.log(a - 1) + a * math.log1p(-1 / a)


def cks_delta(rho: float, eps: float) -> float:
    """delta(eps) of rho-zCDP, CKS 2020 Prop. 12:
    inf over a > 1 of exp((a-1)(a rho - eps)) / (a-1) * (1 - 1/a)^a.

    Every a gives a valid delta, so the numerical minimum found here is an
    upper bound of the exact infimum (never optimistic)."""
    if rho <= 0:
        return 0.0
    from scipy.optimize import minimize_scalar
    grid = [-12 + 0.05 * i for i in range(481)]
    vals = [_logd(rho, eps, x) for x in grid]
    i = min(range(len(vals)), key=vals.__getitem__)
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(lambda x: _logd(rho, eps, x), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return min(1.0, math.exp(min(float(r.fun), vals[i])))


def epsilon_of(rho: float, delta: float) -> float:
    """Smallest epsilon (up to 1e-9, rounded up) with cks_delta(rho, eps) <= delta."""
    if rho <= 0:
        return 0.0
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    from scipy.optimize import brentq
    hi = rho + 2 * math.sqrt(rho * math.log(1 / delta)) + 1
    eps = brentq(lambda e: cks_delta(rho, e) - delta, 0.0, hi, xtol=1e-11) + _SAFETY
    while cks_delta(rho, eps) > delta:            # guard the root against round-off
        eps += _SAFETY * 10
    return min(eps, rho + 2 * math.sqrt(rho * math.log(1 / delta)))


def rho_of(epsilon: float, delta: float) -> float:
    """Largest rho (rounded down) whose stated epsilon at `delta` is at most `epsilon`:
    the conversion of an epsilon cap into the rho the ledger compares against."""
    if epsilon <= 0:
        return 0.0
    from scipy.optimize import brentq
    f = lambda r: epsilon_of(r, delta) - epsilon
    lo, hi = 1e-12, epsilon * epsilon / 2 + epsilon     # epsilon_of(eps^2/2+eps) > eps for delta < 1/e
    while f(hi) < 0:
        hi *= 2
    r = brentq(f, lo, hi, xtol=1e-13)
    while epsilon_of(r, delta) > epsilon:
        r *= 1 - 1e-9
    return r


def ledger_epsilon(charges, delta: float) -> float:
    """The epsilon the ledger states for `charges` at total delta <= `delta` (Prop. 800 b, c)."""
    charges = list(charges)
    rho, d0 = compose(charges)
    if d0 >= delta:
        return math.inf
    eps = epsilon_of(rho, delta - d0) if rho > 0 else 0.0
    if charges and all(c["kind"] in ("pure", "approx") for c in charges):
        eps = min(eps, sum(float(c["epsilon"]) for c in charges))
    return eps

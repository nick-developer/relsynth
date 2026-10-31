# SPDX-License-Identifier: Apache-2.0
"""The noise mechanisms of DP mode and the in-run privacy accountant.

Every statistic released in DP mode is an integer vector; its noise is the
discrete Gaussian of Canonne, Kamath and Steinke (2020), drawn by OpenDP from
the operating system's CSPRNG.  For an integer query with L2 sensitivity D,
noise of scale sigma gives rho = D^2 / (2 sigma^2) zCDP exactly (Prop. 900);
OpenDP's own privacy map is read back and charged, so the charge is the
library's figure, not ours.  Integers avoid the floating-point attacks on
continuous noise (Mironov 2012).

Noise is never seeded: a DP run is not bit-reproducible (design sec. 5.6).

`Accountant` holds the run's plan (rho, delta) and refuses any mechanism that
would exceed it, which Prop. 802 of the ledger relies on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import PrivacyError

_LIB = None


def library() -> str:
    try:
        from importlib.metadata import version
        return f"opendp {version('opendp')}"
    except Exception:  # noqa: BLE001
        return "opendp"


def _dp():
    global _LIB
    if _LIB is None:
        try:
            import opendp.prelude as dp
        except ImportError as e:
            raise PrivacyError("differential privacy needs OpenDP: pip install 'relsynth[dp]'") from e
        dp.enable_features("contrib")
        _LIB = dp
    return _LIB


@dataclass
class Charge:
    stage: str            # discovery | fit (manifest stages)
    mechanism: str
    rho: float
    delta: float = 0.0

    def to_manifest(self) -> dict:
        from ..governance.accounting import dec
        out = {"stage": self.stage, "mechanism": self.mechanism, "kind": "zcdp", "rho": dec(self.rho)}
        if self.delta:
            out["delta"] = dec(self.delta, 30)
        return out


@dataclass
class Accountant:
    """The run's privacy budget: charges add up (zCDP composition, Prop. 800 a)
    and may never exceed the plan."""
    rho: float
    delta: float = 0.0
    charges: list = field(default_factory=list)

    @property
    def spent(self) -> tuple[float, float]:
        return sum(c.rho for c in self.charges), sum(c.delta for c in self.charges)

    def charge(self, c: Charge) -> None:
        r, d = self.spent
        if r + c.rho > self.rho * (1 + 1e-9) + 1e-15 or d + c.delta > self.delta * (1 + 1e-9) + 1e-30:
            raise PrivacyError(f"mechanism {c.mechanism!r} would spend rho {r + c.rho:.6g}, delta {d + c.delta:g}: "
                               f"more than the run's plan (rho {self.rho:.6g}, delta {self.delta:g})")
        self.charges.append(c)


def gaussian(values, sensitivity: float, rho: float, acct: Accountant, stage: str, mechanism: str) -> np.ndarray:
    """values + discrete Gaussian noise for an integer vector query of L2 sensitivity
    `sensitivity`, spending `rho`.  Returns the noisy integers."""
    v = np.asarray(values, dtype=np.int64).ravel()
    if rho <= 0:
        raise PrivacyError(f"{mechanism}: no budget left for this statistic")
    sens = int(math.ceil(sensitivity))
    sigma = sens / math.sqrt(2 * rho)
    dp = _dp()
    m = dp.m.make_gaussian(dp.vector_domain(dp.atom_domain(T="i64")), dp.l2_distance(T="i64"), scale=float(sigma))
    spent = float(m.map(sens))
    acct.charge(Charge(stage, mechanism, spent))
    if v.size == 0:
        return v
    out = np.asarray(m([int(x) for x in v]), dtype=np.int64)
    return out


def sigma_for(sensitivity: float, rho: float) -> float:
    return math.ceil(sensitivity) / math.sqrt(2 * rho)


def _dgauss_tail(sigma: float, t: int, lim: int) -> float:
    """P(X > t) for the discrete Gaussian of scale sigma, summed on [-lim, lim]."""
    from scipy.special import logsumexp
    x = np.arange(-lim, lim + 1, dtype=float)
    lp = -x ** 2 / (2 * sigma ** 2)
    return float(np.exp(logsumexp(lp[x > t]) - logsumexp(lp))) if t < lim else 0.0


def threshold_for(sensitivity: int, sigma: float, delta: float, k: int) -> float:
    """Threshold of Gaussian partition selection (Prop. 903): a level held by the
    removed unit only has count at most `sensitivity`, so it is released when
    sensitivity + X > tau, X discrete Gaussian; a unit adds at most k such
    levels, and tau is the smallest integer with k P(X > tau - sensitivity) <=
    delta.  The tail is summed exactly: the continuous Gaussian's is slightly
    smaller than the discrete one's, so Phi alone would not be conservative
    (check C903)."""
    from scipy.stats import norm
    target = delta / max(k, 1)
    lo = int(math.floor(sigma * norm.isf(target)))
    hi = lo + int(math.ceil(2 * sigma)) + 2
    lim = hi + int(math.ceil(40 * sigma)) + 10
    while _dgauss_tail(sigma, hi, lim) > target:      # widen until hi is safe (never in practice)
        hi += int(math.ceil(sigma)) + 1
        lim = hi + int(math.ceil(40 * sigma)) + 10
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _dgauss_tail(sigma, mid, lim) <= target:
            hi = mid
        else:
            lo = mid
    return float(sensitivity + hi)

# SPDX-License-Identifier: Apache-2.0
"""Rates, risks and their confidence intervals for privacy attacks.

Every number in a privacy report is an estimate from a finite number of attacks,
so each one carries an interval:

* attack success rates: Wilson score interval (Wilson 1927), as in Anonymeter
  (Giomi et al. 2023), so results are comparable with that tool;
* residual risk (attack rate above the control rate, normalised by the room left
  above the control rate) with the propagated error of Giomi et al. (eq. 2);
* TPR at a fixed FPR: the threshold is set on calibration scores and the TPR is
  a binomial proportion on fresh scores, so its Clopper-Pearson interval is
  exact (Prop. 604);
* AUC: DeLong et al. (1988) variance;
* paired differences of accuracies (attribute inference against a baseline):
  normal interval on the per-target differences (Prop. 606).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta, norm


@dataclass(frozen=True)
class Rate:
    value: float
    error: float
    n: int
    k: float

    @property
    def ci(self) -> tuple[float, float]:
        return (max(self.value - self.error, 0.0), min(self.value + self.error, 1.0))


@dataclass(frozen=True)
class Risk:
    value: float
    ci: tuple[float, float]
    attack: Rate
    control: Rate | None
    baseline: Rate | None

    def to_dict(self) -> dict:
        d = {"value": self.value, "ci": list(self.ci), "attack_rate": asdict(self.attack)}
        d["control_rate"] = asdict(self.control) if self.control else None
        d["baseline_rate"] = asdict(self.baseline) if self.baseline else None
        # an attack no better than random guessing measures nothing (Giomi et al. sec. 4)
        d["attack_beats_baseline"] = bool(self.baseline is None or self.attack.value > self.baseline.value)
        return d


def z_of(level: float) -> float:
    return float(norm.ppf(0.5 * (1.0 + level)))


def wilson(n: int, k: float, level: float = 0.95) -> Rate:
    """Wilson score estimate of a Bernoulli rate: k successes in n trials."""
    if n <= 0:
        return Rate(0.0, 1.0, 0, 0.0)
    z = z_of(level)
    z2 = z * z
    den = n + z2
    rate = (k + 0.5 * z2) / den
    err = z / den * np.sqrt(max(k * (n - k) / n, 0.0) + 0.25 * z2)
    return Rate(float(rate), float(err), int(n), float(k))


def residual(attack: Rate, control: Rate) -> tuple[float, float]:
    """(attack - control) / (1 - control) and its propagated error."""
    c = min(control.value, 1 - 1e-12)
    r = (attack.value - c) / (1 - c)
    d_a = 1 / (1 - c)
    d_c = (attack.value - 1) / (1 - c) ** 2
    return float(r), float(np.hypot(attack.error * d_a, control.error * d_c))


def risk(n_attack: int, k_attack: float, n_control: int | None = None, k_control: float | None = None,
         n_base: int | None = None, k_base: float | None = None, level: float = 0.95) -> Risk:
    """Residual risk against the control set (or the attack rate itself without
    one), clipped to [0, 1] with the clipped interval, as in Anonymeter."""
    a = wilson(n_attack, k_attack, level)
    c = wilson(n_control, k_control, level) if n_control else None
    b = wilson(n_base, k_base, level) if n_base else None
    v, e = residual(a, c) if c is not None else (a.value, a.error)
    clip = lambda x: float(min(max(x, 0.0), 1.0))  # noqa: E731
    return Risk(clip(v), (clip(v - e), clip(v + e)), a, c, b)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def auc_delong(score, y) -> tuple[float, float]:
    """AUC (ties count half) and its DeLong (1988) standard error, floored by
    the Hanley-McNeil (1982) error (see below)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    # placement values through ranks: O((m+n) log(m+n))
    from scipy.stats import rankdata
    r_all = rankdata(np.concatenate([pos, neg]))
    r_pos, r_neg = rankdata(pos), rankdata(neg)
    v10 = (r_all[:m] - r_pos) / n          # share of negatives below each positive
    v01 = 1 - (r_all[m:] - r_neg) / m      # share of positives above each negative
    auc = float(v10.mean())
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n) if m > 1 and n > 1 else float("nan")
    # DeLong's variance is 0 at a perfect separation of few points; the Hanley and
    # McNeil (1982) variance with the AUC moved half a pair from 0 and 1 is used as
    # a floor, so a perfect score on 8 shadows does not get a zero-width interval
    a = min(max(auc, 0.5 / (m * n)), 1 - 0.5 / (m * n))
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    hm = np.sqrt(max(a * (1 - a) + (m - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a), 0.0) / (m * n))
    se = float(np.nanmax([se, hm]))
    return auc, float(se)


def auc_interval(auc: float, se: float, level: float = 0.95) -> list[float]:
    z = z_of(level)
    if not np.isfinite(se):
        return [float("nan"), float("nan")]
    return [float(max(auc - z * se, 0.0)), float(min(auc + z * se, 1.0))]


def tpr_at_fpr(score_cal, y_cal, score_test, y_test, fpr: float, level: float = 0.95) -> dict:
    """TPR at a target FPR.  The threshold is the (1 - fpr) quantile of the
    calibration non-member scores; TPR and realised FPR are then counted on the
    test scores, which are independent of the threshold, so both are binomial
    proportions with exact Clopper-Pearson intervals (Prop. 604).  With fewer
    than 1/fpr calibration non-members the threshold is not identified and the
    entry says so instead of reporting a number."""
    sc, yc = np.asarray(score_cal, float), np.asarray(y_cal, bool)
    st, yt = np.asarray(score_test, float), np.asarray(y_test, bool)
    neg = np.sort(sc[~yc])
    if len(neg) * fpr < 1:
        return {"fpr_target": fpr, "identified": False,
                "reason": f"needs at least {int(np.ceil(1 / fpr))} calibration non-members, has {len(neg)}"}
    thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    tp, P = int((st[yt] > thr).sum()), int(yt.sum())
    fp, N = int((st[~yt] > thr).sum()), int((~yt).sum())
    return {"fpr_target": fpr, "identified": True, "threshold": float(thr),
            "tpr": tp / max(P, 1), "tpr_ci": list(clopper_pearson(tp, P, level)),
            "fpr": fp / max(N, 1), "fpr_ci": list(clopper_pearson(fp, N, level)),
            "n_members": P, "n_non_members": N}


def paired_difference(a, b, level: float = 0.95) -> dict:
    """Mean of a - b over paired outcomes (per target) with a normal interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n == 0:
        return {"value": float("nan"), "ci": [float("nan")] * 2, "n": 0}
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    z = z_of(level)
    return {"value": m, "ci": [m - z * se, m + z * se], "n": int(n)}


def dp_tpr_bound(fpr: float, epsilon: float, delta: float = 0.0) -> float:
    """Largest TPR any membership test can reach at this FPR against an
    (epsilon, delta)-DP release (Kairouz, Oh and Viswanath 2015, Thm 2.1)."""
    return float(min(1.0, np.exp(epsilon) * fpr + delta, 1 - np.exp(-epsilon) * (1 - delta - fpr)))

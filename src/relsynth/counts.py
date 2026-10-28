# SPDX-License-Identifier: Apache-2.0
"""Choice of the count model for rows per parent (children per parent row,
fact rows per anchor row).

Two families compete within each segment:
  * parametric: Poisson(lambda), or negative binomial when the counts are more
    spread than Poisson (the gamma-frailty model the engine already uses);
  * nonparametric: a Dirichlet process with concentration a and Poisson base
    measure, whose posterior predictive reproduces the observed count support
    (e.g. every customer has exactly 7 invoices, or lines per invoice in
    {1, 2, 4, 6, 9, 14}) and puts mass a/(n+a) on the Poisson base.

The DP marginal likelihood is the Polya-urn product
    p(t_1..t_n) = prod_i (a g0(t_i) + #{j < i: t_j = t_i}) / (a + i - 1),
which is exchangeable, so it does not depend on the order of the counts.  Each
family is scored by its log marginal (DP) or maximised log likelihood (Poisson,
NB) minus (d/2) log n for its d plug-in parameters (Schwarz), summed over the
segments.  The DP is chosen when it beats the better parametric model by a Bayes
factor of at least `bf` (default 100, "decisive" on Jeffreys' scale).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def poisson_loglik(t, lam):
    t = np.asarray(t, float)
    if lam <= 0:
        return 0.0 if not t.any() else -np.inf
    return float((t * np.log(lam) - lam - gammaln(t + 1)).sum())


def nb_loglik(t):
    """Maximised NB log likelihood (mean fixed at the sample mean, its MLE);
    falls back to Poisson when the counts are not over-dispersed."""
    t = np.asarray(t, float)
    m = t.mean()
    if m <= 0 or len(t) < 2 or t.var() <= m:
        return poisson_loglik(t, m)

    def nll(lr):
        r = np.exp(lr)
        return -float((gammaln(t + r) - gammaln(r) - gammaln(t + 1)
                       + r * np.log(r / (r + m)) + t * np.log(m / (r + m))).sum())

    res = minimize_scalar(nll, bounds=(-8.0, 12.0), method="bounded")
    return max(-res.fun, poisson_loglik(t, m))


def nb_size_pooled(groups):
    """Maximum-likelihood NB size r shared by the segments, each with its own mean
    (the sample mean, its MLE): one dispersion for a key's degrees instead of one
    per segment, so small segments borrow strength.  inf when the pooled counts
    are not over-dispersed (the Poisson limit)."""
    gs = [np.asarray(t, float) for t in groups if len(t) >= 2 and np.asarray(t).sum() > 0]
    if not gs:
        return np.inf
    ms = [t.mean() for t in gs]
    if sum(t.var() * len(t) for t in gs) <= sum(m * len(t) for t, m in zip(gs, ms)):
        return np.inf

    def nll(lr):
        r = np.exp(lr)
        return -sum(float((gammaln(t + r) - gammaln(r) + r * np.log(r / (r + m)) + t * np.log(m / (r + m))).sum())
                    for t, m in zip(gs, ms))

    res = minimize_scalar(nll, bounds=(-8.0, 12.0), method="bounded")
    return float(np.exp(res.x)) if res.x < 11.9 else np.inf


def dp_logmarginal(t, a=1.0):
    """Polya-urn marginal of the counts under DP(a, Poisson(mean))."""
    t = np.asarray(t, np.int64)
    if not len(t):
        return 0.0
    lam = max(float(t.mean()), 1e-12)
    # draw i equals value x with probability (a f0(x) + c_x) / (a + i), c_x earlier
    # copies of x; the product over the m_x copies of x is
    # prod_{c<m_x} (b_x + c) = b_x Gamma(b_x + m_x) / Gamma(b_x + 1), b_x = a f0(x),
    # computed with log b_x in log space (a f0(x) underflows for counts far out
    # in the Poisson tail, which made the marginal -inf)
    x, m = np.unique(t, return_counts=True)
    lb = np.log(a) + x * np.log(lam) - lam - gammaln(x + 1)
    b = np.exp(lb)
    out = (lb + gammaln(b + m) - gammaln(b + 1)).sum() - (gammaln(a + len(t)) - gammaln(a))
    return float(out)


def count_model_evidence(groups, a=1.0):
    """log Bayes factor of the DP model against the better of Poisson and NB,
    summed over the segments (each a list of counts).  Returns (log_bf, detail)."""
    dp = pois = nb = 0.0
    n_tot = 0
    for t in groups:
        t = np.asarray(t, np.int64)
        n = len(t)
        if n < 2 or t.sum() == 0:
            continue
        n_tot += n
        pen = 0.5 * np.log(n)
        m = t.mean()
        dp += dp_logmarginal(t, a) - pen
        pois += poisson_loglik(t, m) - pen
        nb += nb_loglik(t) - 2 * pen
    best = max(pois, nb)
    lbf = dp - best if n_tot else 0.0
    return float(lbf), {"log_bf_dp": round(float(lbf), 3), "n": int(n_tot),
                        "parametric": "nb" if nb > pois else "poisson"}


def choose_count_model(groups, setting="auto", a=1.0, bf=100.0):
    """Returns the DP state {"counts": [...], "a": a} or None (Poisson/frailty)."""
    if setting == "poisson":
        return None, {}
    lbf, det = count_model_evidence(groups, a)
    if setting == "dp" or lbf > np.log(bf):
        return {"counts": [np.asarray(t, np.int64) for t in groups], "a": a}, det
    return None, det

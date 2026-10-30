# SPDX-License-Identifier: Apache-2.0
"""Event sequences per entity: when each customer's orders happen.

The relational engine decides how many events an entity has (its fan-out law,
with frailty and the count model) and what each event holds.  The 1.0 engine
then draws every event's date independently from a seasonal profile of the
entity's segment, so a customer's three orders are three independent dates:
the inter-arrival times, the burstiness and the life span of real customers
are lost.  The sequence model replaces those dates by a draw from a model of
the whole history of the entity, given its count n and its window.

Model (per entity i of segment k, in operational time u = Lambda_k(t), see
relsynth.timeseries.calendar, on the entity's window [A_i, B_i], U_i = B_i - A_i)

* n = 1: the position p = (u_1 - A) / U ~ Beta(beta_h, beta_t).
* n >= 2: the head, active span and tail of the history
      (h, s, r) = (u_1 - A, u_n - u_1, B - u_n) / U ~ Dirichlet(beta_h, beta_s(n), beta_t),
      log beta_s(n) = b0 + b1 log(n - 1):
  acquisition (head), life span and churn (tail), with a span that grows with n.
* n >= 3: the n - 1 gaps inside the span, as shares of it,
      (g_1, ..., g_{n-1}) / s ~ Dirichlet(alpha_i, ..., alpha_i),
      log alpha_i ~ Normal(mu, sigma^2)                     (entity random effect).
  This is the exact law of a gamma renewal process with shape alpha_i given
  its first and last event (Lukacs 1955: normalised independent gamma gaps are
  Dirichlet and independent of their sum; Prop. 1201).  alpha = 1 is Poisson
  (uniform order statistics), alpha < 1 bursty (repeat orders close together,
  then a long pause), alpha > 1 regular (a monthly subscription).  The random
  effect lets a segment hold both bursty and regular entities.

The parameters theta_k = (log beta_h, log beta_t, b0, b1, mu, log sigma) of
each segment are fitted by maximum a posteriori with a Normal prior centred on
the pooled fit (strength `shrinkage`), the per-entity random effect integrated
out by Gauss-Hermite quadrature; a dataset draws theta_k from the Laplace
approximation of its posterior (Bernstein-von Mises; Prop. 1202).

Date columns are observed to the day: fitting adds U(0, 1) of a day to each
event (dequantisation, so that two orders on one day have a positive gap), and
synthetic times are floored back to the day.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .calendar import Intensity

NPAR = 6
NAMES = ("log_beta_head", "log_beta_tail", "span_b0", "span_b1", "mu_log_alpha", "log_sigma_alpha")
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)        # nodes for E f(Z), Z ~ N(0, 1)
GH_LW = np.log(GH_W / GH_W.sum())
PRIOR_MEAN = np.array([0.0, 0.0, 0.0, 0.5, 0.0, -0.5])
PRIOR_SD = np.array([2.0, 2.0, 2.0, 1.0, 2.0, 1.0])


def _ldir(x_log_sum, a, m):
    """log Dirichlet(a 1_m) density at a point whose log coordinates sum to x_log_sum."""
    return gammaln(m * a) - m * gammaln(a) + (a - 1) * x_log_sum


class _Data:
    """Sufficient statistics of the sequences of one segment."""

    def __init__(self, n, A, B, u_first, u_last, gap_log_sum):
        U = np.maximum(B - A, 1e-9)
        eps = 1e-9
        self.one = n == 1
        self.p1 = np.clip((u_first[self.one] - A[self.one]) / U[self.one], eps, 1 - eps)
        two = n >= 2
        h = np.clip((u_first[two] - A[two]) / U[two], eps, None)
        s = np.clip((u_last[two] - u_first[two]) / U[two], eps, None)
        r = np.clip((B[two] - u_last[two]) / U[two], eps, None)
        tot = h + s + r
        self.h, self.s, self.r = np.log(h / tot), np.log(s / tot), np.log(r / tot)
        self.ln1 = np.log(n[two] - 1.0)
        three = n >= 3
        self.m = (n[three] - 1).astype(float)
        self.gsum = gap_log_sum[three]
        self.n_entities = int((n >= 1).sum())

    def loglik(self, th):
        bh, bt = np.exp(th[0]), np.exp(th[1])
        ll = 0.0
        if len(self.p1):
            ll += np.sum(gammaln(bh + bt) - gammaln(bh) - gammaln(bt)
                         + (bh - 1) * np.log(self.p1) + (bt - 1) * np.log1p(-self.p1))
        if len(self.h):
            bs = np.exp(np.clip(th[2] + th[3] * self.ln1, -20, 20))
            ll += np.sum(gammaln(bh + bs + bt) - gammaln(bh) - gammaln(bs) - gammaln(bt)
                         + (bh - 1) * self.h + (bs - 1) * self.s + (bt - 1) * self.r)
        if len(self.m):
            la = np.clip(th[4] + np.exp(th[5]) * GH_X[None, :], -12, 8)           # entities x nodes
            a = np.exp(la)
            f = _ldir(self.gsum[:, None], a, self.m[:, None]) + GH_LW[None, :]
            ll += np.sum(logsumexp(f, axis=1))
        return float(ll)


def _fit_theta(data, mean, sd):
    def nlp(th):
        v = -data.loglik(th) + 0.5 * np.sum(((th - mean) / sd) ** 2)
        return v if np.isfinite(v) else 1e300
    res = minimize(nlp, mean.copy(), method="L-BFGS-B")
    th = res.x
    # Laplace: numerical Hessian of the negative log posterior at the mode
    H = np.zeros((NPAR, NPAR))
    e = 1e-3
    f0 = nlp(th)
    for i in range(NPAR):
        for j in range(i, NPAR):
            if i == j:
                ei = np.zeros(NPAR)
                ei[i] = e
                H[i, i] = (nlp(th + ei) - 2 * f0 + nlp(th - ei)) / e ** 2
            else:
                ei, ej = np.zeros(NPAR), np.zeros(NPAR)
                ei[i], ej[j] = e, e
                H[i, j] = H[j, i] = (nlp(th + ei + ej) - nlp(th + ei - ej) - nlp(th - ei + ej)
                                     + nlp(th - ei - ej)) / (4 * e ** 2)
    w, V = np.linalg.eigh((H + H.T) / 2)
    w = np.maximum(w, 1.0 / sd.max() ** 2)            # the prior bounds the curvature from below
    cov = (V / w) @ V.T
    return th, cov, bool(res.success)


class SequenceModel:
    """Sequence model of the events of one table, per entity segment.

    fit(t, owner, seg_of_owner, K, window) with t the event times in days,
    owner the entity row of each event (-1: no entity), seg_of_owner the
    segment of each entity row and window (A, B) the entity windows in days.
    """

    def __init__(self, bucket="month", weekly=True, hourly=None, shrinkage=1.0, min_entities=30):
        self.intensity = Intensity(bucket, weekly, hourly)
        self.shrinkage = float(shrinkage)
        self.min_entities = int(min_entities)

    # ------------------------------------------------------------ fitting
    def fit(self, t, owner, seg_of_owner, K, window, datetime=False, rng=None, t0=None, t1=None):
        rng = np.random.default_rng(0) if rng is None else rng
        t = np.asarray(t, dtype=float).copy()
        owner = np.asarray(owner, dtype=np.int64)
        seg_of_owner = np.asarray(seg_of_owner, dtype=np.int64)
        self.datetime = bool(datetime)
        self.K = int(K)
        ok = np.isfinite(t) & (owner >= 0)
        if not self.datetime:
            t = np.floor(t) + rng.random(len(t))                # dequantise days
        t0 = float(np.nanmin(t[ok])) if t0 is None else float(t0)
        t1 = float(np.nanmax(t[ok])) + 1e-6 if t1 is None else float(t1)
        if not self.datetime:
            t0, t1 = np.floor(t0), np.floor(t1 - 1e-9) + 1.0
        self.t0, self.t1 = t0, t1
        A, B = self._windows(window, len(seg_of_owner))
        ok &= (t >= np.maximum(A[np.maximum(owner, 0)], t0)) & (t < np.minimum(B[np.maximum(owner, 0)], t1))
        self.null_rate = float(np.mean(~np.isfinite(np.asarray(t, dtype=float)))) if len(t) else 0.0
        self.intensity.fit(t[ok], seg_of_owner[owner[ok]], self.K, t0, t1, self.datetime)
        self.stats = self._stats(t[ok], owner[ok], seg_of_owner, A, B)
        self._fit_params()
        self._fit_rates(np.bincount(owner[ok], minlength=len(seg_of_owner)), seg_of_owner, A, B)
        self.redraw(None)
        self.fitted = True
        return self

    def _windows(self, window, n_ent):
        A = np.full(n_ent, self.t0)
        B = np.full(n_ent, self.t1)
        if window is not None:
            a, b = window
            if a is not None:
                a = np.asarray(a, dtype=float)
                A = np.where(np.isfinite(a), np.maximum(a, self.t0), self.t0)
            if b is not None:
                b = np.asarray(b, dtype=float)
                B = np.where(np.isfinite(b), np.minimum(b, self.t1), self.t1)
        B = np.maximum(B, A + 1e-6)
        return A, B

    def _stats(self, t, owner, seg_of_owner, A, B):
        """Per entity with events: segment, n, window and the statistics the
        likelihood reads (operational first/last times, sum of log gap shares)."""
        if not len(t):
            return None
        o = np.argsort(owner * 1e7 + (t - self.t0), kind="stable")
        t, owner = t[o], owner[o]
        ents, start, n = np.unique(owner, return_index=True, return_counts=True)
        seg = seg_of_owner[ents]
        u = self.intensity.forward(t, seg_of_owner[owner])
        uA = self.intensity.forward(A[ents], seg)
        uB = self.intensity.forward(B[ents], seg)
        last = start + n - 1
        uf, ul = u[start], u[last]
        gl = np.zeros(len(ents))
        multi = n >= 3
        if multi.any():
            d = np.diff(u)
            same = owner[1:] == owner[:-1]
            span = np.maximum(ul - uf, 1e-12)
            share = np.where(same, d, 1.0) / span[np.searchsorted(ents, owner[1:])]
            lg = np.where(same, np.log(np.clip(share, 1e-12, None)), 0.0)
            gl = np.bincount(np.searchsorted(ents, owner[1:]), weights=lg, minlength=len(ents))
        return {"seg": seg, "n": n, "A": uA, "B": uB, "uf": uf, "ul": ul, "gl": gl}

    def _fit_params(self):
        st = self.stats
        self.theta_hat = np.tile(PRIOR_MEAN, (self.K, 1))
        self.theta_cov = np.tile(np.diag(PRIOR_SD ** 2), (self.K, 1, 1))
        self.fit_log = {"segments": {}}
        if st is None:
            return
        pooled = _Data(st["n"], st["A"], st["B"], st["uf"], st["ul"], st["gl"])
        th0, cov0, ok0 = _fit_theta(pooled, PRIOR_MEAN, PRIOR_SD)
        self.pooled = th0
        self.fit_log["pooled"] = {"theta": dict(zip(NAMES, np.round(th0, 4).tolist())), "converged": ok0,
                                  "entities": pooled.n_entities}
        sd = np.full(NPAR, self.shrinkage)
        for k in range(self.K):
            m = st["seg"] == k
            if m.sum() < self.min_entities:
                self.theta_hat[k], self.theta_cov[k] = th0, cov0
                self.fit_log["segments"][k] = {"entities": int(m.sum()), "pooled": True}
                continue
            dk = _Data(st["n"][m], st["A"][m], st["B"][m], st["uf"][m], st["ul"][m], st["gl"][m])
            th, cov, ok = _fit_theta(dk, th0, sd)
            self.theta_hat[k], self.theta_cov[k] = th, cov
            self.fit_log["segments"][k] = {"entities": int(m.sum()), "pooled": False, "converged": ok,
                                           "theta": dict(zip(NAMES, np.round(th, 4).tolist()))}

    def _fit_rates(self, n, seg, A, B):
        """Event rate of an entity (events per unit of operational time) for the
        forward process of incremental runs: r ~ Gamma(a_k, b_k) per segment, fitted
        by maximum likelihood of the counts n ~ NegBin (the Poisson-gamma mixture
        with exposure U = Lambda(B) - Lambda(A)), entities without events included."""
        from scipy.special import gammaln as lg
        U = np.maximum(self.intensity.forward(B, seg) - self.intensity.forward(A, seg), 1e-9)
        self.rate_ab = np.ones((self.K, 2))

        def nll(x, nn, uu):
            a, b = np.exp(x)
            return -np.sum(lg(nn + a) - lg(a) + a * np.log(b / (b + uu)) + nn * np.log(uu / (b + uu)))
        pooled = minimize(nll, np.log([1.0, max(U.mean() / max(n.mean(), 1e-9), 1e-6)]), args=(n, U),
                          method="Nelder-Mead").x
        for k in range(self.K):
            m = seg == k
            x = minimize(nll, pooled, args=(n[m], U[m]), method="Nelder-Mead").x \
                if m.sum() >= self.min_entities else pooled
            self.rate_ab[k] = np.exp(x)

    def redraw(self, rng=None):
        """One posterior draw of every parameter (rng given) or posterior means."""
        self.intensity.draw(rng)
        if rng is None:
            self.theta = self.theta_hat.copy()
        else:
            self.theta = np.stack([rng.multivariate_normal(self.theta_hat[k], self.theta_cov[k])
                                   for k in range(self.K)])
        return self

    # ----------------------------------------------------------- sampling
    def sample(self, n, seg, A, B, rng):
        """Event times (days) of entities with n[i] events, segment seg[i] and
        window [A[i], B[i]) in days.  Returns (times, owner) with the times of
        each entity in increasing order; owner indexes the entities."""
        n = np.asarray(n, dtype=np.int64)
        seg = np.asarray(seg, dtype=np.int64)
        A, B = self._windows((A, B), len(n))
        uA = self.intensity.forward(A, seg)
        uB = self.intensity.forward(B, seg)
        U = np.maximum(uB - uA, 1e-9)
        th = self.theta[seg]
        bh, bt = np.exp(th[:, 0]), np.exp(th[:, 1])
        owner = np.repeat(np.arange(len(n)), n)
        u = np.empty(int(n.sum()))
        start = np.concatenate([[0], np.cumsum(n)[:-1]])
        one = n == 1
        if one.any():
            p = rng.beta(bh[one], bt[one])
            u[start[one]] = uA[one] + p * U[one]
        two = n >= 2
        if two.any():
            bs = np.exp(np.clip(th[two, 2] + th[two, 3] * np.log(n[two] - 1.0), -20, 20))
            g = np.stack([rng.gamma(bh[two]), rng.gamma(bs), rng.gamma(bt[two])], 1)
            g = g / g.sum(1, keepdims=True)
            first = uA[two] + g[:, 0] * U[two]
            span = g[:, 1] * U[two]
            idx = np.flatnonzero(two)
            u[start[two]] = first
            u[start[two] + n[two] - 1] = first + span
            three = n[idx] >= 3
            if three.any():
                ii = idx[three]
                alpha = np.exp(np.clip(th[ii, 4] + np.exp(th[ii, 5]) * rng.standard_normal(len(ii)), -12, 8))
                m = n[ii] - 1
                gam = rng.gamma(np.repeat(alpha, m))
                gam = np.maximum(gam, 1e-300)
                off = np.concatenate([[0], np.cumsum(m)[:-1]])
                tot = np.add.reduceat(gam, off)
                share = gam / np.repeat(tot, m)
                cum = np.cumsum(share)
                cum -= np.repeat(cum[off] - share[off], m)          # cumulative share within each entity
                pos = np.repeat(start[ii], m) + (np.arange(m.sum()) - np.repeat(off, m)) + 1
                inner = np.ones(m.sum(), bool)
                inner[off + m - 1] = False                           # the last share ends at the last event
                f_rep = np.repeat(first[three], m)
                s_rep = np.repeat(span[three], m)
                u[pos[inner]] = f_rep[inner] + cum[inner] * s_rep[inner]
        t = self.intensity.inverse(u, seg[owner])
        t = np.clip(t, np.repeat(A, n), np.nextafter(np.repeat(B, n), -np.inf))
        return t, owner

    def summary(self):
        return {"calendar": self.intensity.summary(), "parameters": NAMES, **self.fit_log}

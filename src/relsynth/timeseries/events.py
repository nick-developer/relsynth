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

* the life time of the entity: the position of its first event and, for
  n >= 2, the share of the rest of the window that its active span takes,
      h = (u_1 - A) / U,      q = (u_n - u_1) / (B - u_1)     (1 - q is the churn share),
  with a Gaussian copula of (h, q) per count stratum (1, 2, 3-4, 5-8, ...),
  so entities with many events can start early and stay long.  Marginals are
  quantile grids, as in the engine's marginal summaries, with a Bayesian
  bootstrap posterior (Rubin 1981); the copula correlation has the Fisher-z
  posterior;
* n >= 3: the n - 1 gaps inside the span, as shares of it,
      (g_1, ..., g_{n-1}) / s ~ Dirichlet(alpha_i, ..., alpha_i),
      log alpha_i ~ Normal(mu, sigma^2)                     (entity random effect).
  This is the exact law of a gamma renewal process with shape alpha_i given
  its first and last event (Lukacs 1955: normalised independent gamma gaps are
  Dirichlet and independent of their sum; Prop. 1201).  alpha = 1 is Poisson
  (uniform order statistics), alpha < 1 bursty (repeat orders close together,
  then a long pause), alpha > 1 regular (a monthly subscription).  The random
  effect lets a segment hold both bursty and regular entities.

The gap parameters theta_k = (mu, log sigma) of each segment are fitted by maximum a posteriori with a Normal prior centred on
the pooled fit (strength `shrinkage`), the per-entity random effect integrated
out by Gauss-Hermite quadrature; a dataset draws theta_k from the Laplace
approximation of its posterior (Bernstein-von Mises; Prop. 1202), and the
life-time copula from its posterior.

Date columns are observed to the day: fitting adds U(0, 1) of a day to each
event (dequantisation, so that two orders on one day have a positive gap), and
synthetic times are floored back to the day.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .calendar import Intensity

NPAR = 2
NAMES = ("mu_log_alpha", "log_sigma_alpha")
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)        # nodes for E f(Z), Z ~ N(0, 1)
GH_LW = np.log(GH_W / GH_W.sum())
PRIOR_MEAN = np.array([0.0, -0.5])
PRIOR_SD = np.array([2.0, 1.0])


NQ = 200                        # quantile cells of each life-time marginal
EDGES = (1, 2, 3, 5, 9, 17, 33)  # count strata: 1, 2, 3-4, 5-8, 9-16, 17-32, 33+
MIN_STRATUM = 30


class _Marginal:
    """A distribution on [0, 1] kept as NQ equal-mass cells between quantile
    knots (no entity value is stored).  Its posterior draw gives the cells
    Dirichlet(m / NQ) masses (a Bayesian bootstrap of the cells, Rubin 1981)."""

    def __init__(self, x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        self.m = len(x)
        self.knots = np.quantile(x, np.linspace(0, 1, NQ + 1)) if len(x) else np.linspace(0, 1, NQ + 1)
        self.cdf = np.linspace(0, 1, NQ + 1)

    def draw(self, rng):
        if rng is None:
            self.cdf = np.linspace(0, 1, NQ + 1)
        else:
            w = rng.dirichlet(np.full(NQ, max(self.m, NQ) / NQ))
            self.cdf = np.concatenate([[0.0], np.cumsum(w)])
            self.cdf[-1] = 1.0

    def ppf(self, p):
        return np.interp(p, self.cdf, self.knots)


class _Lifetime:
    """The life time of an entity given its count n: the head position h and,
    for n >= 2, the span share q, with a Gaussian copula per count stratum
    (strata of fewer than MIN_STRATUM entities are merged into their
    neighbour).  Marginals are quantile grids with Bayesian-bootstrap
    posteriors; the copula correlation has the Fisher-z posterior
    N(atanh r, 1 / (m - 3)) (Prop. 1202)."""

    def fit(self, n, h, q):
        n = np.asarray(n, dtype=np.int64)
        ok = n >= 1
        n, h, q = n[ok], np.asarray(h, float)[ok], np.asarray(q, float)[ok]
        groups = [[lo, hi] for lo, hi in zip(EDGES, EDGES[1:] + (np.iinfo(np.int64).max,))]
        cnt = [int(((n >= lo) & (n < hi)).sum()) for lo, hi in groups]
        merged, c = [], []
        for g, k in zip(groups, cnt):              # merge small strata upward, keeping 1 apart from >= 2
            if merged and (c[-1] < MIN_STRATUM) and merged[-1][0] >= 2:
                merged[-1][1] = g[1]
                c[-1] += k
            else:
                merged.append(list(g))
                c.append(k)
        if len(merged) > 1 and c[-1] < MIN_STRATUM and merged[-2][0] >= 2:
            merged[-2][1] = merged[-1][1]
            merged.pop()
            c[-1] += c.pop()
        self.strata = []
        for lo, hi in merged:
            m = (n >= lo) & (n < hi)
            if not m.any() and len(n):
                continue
            hh = _Marginal(h[m])
            if lo >= 2:
                qq = _Marginal(q[m])
                r = 0.0
                if m.sum() > 3:
                    from scipy.stats import norm, rankdata
                    zh = norm.ppf(rankdata(h[m]) / (m.sum() + 1))
                    zq = norm.ppf(rankdata(q[m]) / (m.sum() + 1))
                    r = float(np.clip(np.corrcoef(zh, zq)[0, 1], -0.99, 0.99)) if zh.std() > 0 and zq.std() > 0 else 0.0
            else:
                qq, r = None, 0.0
            self.strata.append({"lo": lo, "hi": hi, "m": int(m.sum()), "h": hh, "q": qq, "z": np.arctanh(r),
                                "r": r})
        if not self.strata:
            self.strata = [{"lo": 1, "hi": np.iinfo(np.int64).max, "m": 0, "h": _Marginal(np.full(1, 0.5)),
                            "q": _Marginal(np.full(1, 0.5)), "z": 0.0, "r": 0.0}]
        self.n_entities = int(len(n))
        return self

    def draw(self, rng):
        for st in self.strata:
            st["h"].draw(rng)
            if st["q"] is not None:
                st["q"].draw(rng)
            st["r"] = float(np.tanh(st["z"])) if rng is None else \
                float(np.tanh(st["z"] + rng.standard_normal() / np.sqrt(max(st["m"] - 3, 1))))
        return self

    def _stratum(self, n):
        lo = np.array([st["lo"] for st in self.strata])
        i = np.searchsorted(lo, n, side="right") - 1
        return np.clip(i, 0, len(lo) - 1)

    def sample(self, n, rng):
        """(h, q) per entity (q = NaN for n = 1)."""
        from scipy.stats import norm
        n = np.asarray(n, dtype=np.int64)
        h = np.empty(len(n))
        q = np.full(len(n), np.nan)
        si = self._stratum(n)
        for j in np.unique(si):
            st = self.strata[j]
            m = si == j
            z1 = rng.standard_normal(m.sum())
            h[m] = st["h"].ppf(norm.cdf(z1))
            two = m & (n >= 2)
            if two.any():
                qm = st["q"] if st["q"] is not None else _Marginal(np.full(1, 0.5))
                zz = z1[two[m]]
                z2 = st["r"] * zz + np.sqrt(1 - st["r"] ** 2) * rng.standard_normal(two.sum())
                q[two] = qm.ppf(norm.cdf(z2))
        return h, q

    def summary(self):
        return {"strata": [{"counts": [int(st["lo"]), None if st["hi"] > 10 ** 9 else int(st["hi"]) - 1],
                            "entities": st["m"], "head_median": round(float(st["h"].ppf(0.5)), 4),
                            "span_median": None if st["q"] is None else round(float(st["q"].ppf(0.5)), 4),
                            "copula_r": round(float(st["r"]), 4)} for st in self.strata]}


def _ldir(x_log_sum, a, m):
    """log Dirichlet(a 1_m) density at a point whose log coordinates sum to x_log_sum."""
    return gammaln(m * a) - m * gammaln(a) + (a - 1) * x_log_sum


class _Data:
    """Sufficient statistics of the sequences of one segment."""

    def __init__(self, n, A, B, u_first, u_last, gap_log_sum):
        U = np.maximum(B - A, 1e-9)
        eps = 1e-9
        one_plus = n >= 1
        # position of the first event in the window (n >= 1)
        self.n = n[one_plus]
        self.h = np.clip((u_first[one_plus] - A[one_plus]) / U[one_plus], eps, 1 - eps)
        two = n >= 2
        # share of the rest of the window that the active span takes (n >= 2)
        rest = np.maximum(B[two] - u_first[two], 1e-12)
        self.q = np.full(len(self.n), np.nan)
        self.q[self.n >= 2] = np.clip((u_last[two] - u_first[two]) / rest, eps, 1 - eps)
        self.ln1 = np.log(n[two] - 1.0)
        three = n >= 3
        self.m = (n[three] - 1).astype(float)
        self.gsum = gap_log_sum[three]
        self.n_entities = int((n >= 1).sum())

    def loglik(self, th):
        ll = 0.0
        if len(self.m):
            la = np.clip(th[0] + np.exp(th[1]) * GH_X[None, :], -12, 8)           # entities x nodes
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

    def __init__(self, bucket="month", weekly=True, hourly=None, shrinkage=1.0, min_entities=30, calibration=3):
        self.intensity = Intensity(bucket, weekly, hourly)
        self.shrinkage = float(shrinkage)
        self.min_entities = int(min_entities)
        self.calibration = int(calibration)
        self.fit_log = {}

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
        expo, inner = self._exposure(t[ok], owner[ok], seg_of_owner, t0, t1)
        self.intensity.fit(t[ok], seg_of_owner[owner[ok]], self.K, t0, t1, self.datetime,
                           exposure=expo, interior=inner)
        self.stats = self._stats(t[ok], owner[ok], seg_of_owner, A, B)
        self._fit_params()
        self._calibrate(t[ok], owner[ok], seg_of_owner, A, B, rng)
        self._fit_rates(np.bincount(owner[ok], minlength=len(seg_of_owner)), seg_of_owner, A, B)
        self.redraw(None)
        self.fitted = True
        return self

    def _calibrate(self, t, owner, seg_of_owner, A, B, rng, iters=None):
        """Moment-match the calendar profile: the bridge itself puts more events
        near the ends of an entity's span (alpha < 1) and the life-time law
        decides when entities are active, so the rate per entity at risk is not
        yet the calendar.  Simulate the fitted model at its posterior means with
        the real counts and windows, and multiply each bucket's profile by
        (real + 1) / (simulated + 1) until the bucket counts agree; operational
        time and the life-time and gap statistics are refitted each round
        (Prop. 1200)."""
        iters = self.calibration if iters is None else iters
        if self.stats is None or iters <= 0:
            return
        I = self.intensity
        n_ent = np.bincount(owner, minlength=len(seg_of_owner))
        seg_e = seg_of_owner
        b_real = I.bucket_of_day[np.clip(np.floor(t).astype(np.int64) - I.d0, 0, I.ndays - 1)]
        real = np.zeros((self.K, I.B))
        np.add.at(real, (seg_of_owner[owner], b_real), 1.0)
        self.fit_log["calibration"] = []
        for _ in range(int(iters)):
            self.redraw(None)
            sim = np.zeros((self.K, I.B))
            for r in range(2):
                st, so = self.sample(n_ent, seg_e, A, B, np.random.default_rng([int(rng.integers(2 ** 31)), r]))
                bs = I.bucket_of_day[np.clip(np.floor(st).astype(np.int64) - I.d0, 0, I.ndays - 1)]
                np.add.at(sim, (seg_e[so], bs), 0.5)
            ratio = (real + 1.0) / (sim + 1.0)
            tvd = float(0.5 * np.abs(real / max(real.sum(), 1) - sim / max(sim.sum(), 1)).sum())
            self.fit_log["calibration"].append(round(tvd, 5))
            I.adj = I.adj * ratio
            I.adj = I.adj / I.adj.mean(1, keepdims=True)
            I._build()
            self.stats = self._stats(t, owner, seg_of_owner, A, B)
            self._fit_params(keep_log=True)
        self.redraw(None)

    def _exposure(self, t, owner, seg_of_owner, t0, t1):
        """Entities at risk per day and segment: an entity is at risk from its
        first to its last event, so the calendar profile is a rate per active
        entity and the growth of the active population is left to the life-time
        law (Prop. 1200)."""
        if not len(t):
            return None, None
        d0, d1 = int(np.floor(t0)), int(np.ceil(t1))
        first = np.full(len(seg_of_owner), np.inf)
        last = np.full(len(seg_of_owner), -np.inf)
        np.minimum.at(first, owner, t)
        np.maximum.at(last, owner, t)
        ok = np.isfinite(first) & (last > first)
        if ok.sum() < 2:
            return None, None
        # the first and last event of an entity bound its risk period: only the
        # events between them are counted against it
        inner = (t > first[owner]) & (t < last[owner])
        if inner.sum() < 2:
            return None, None
        L = d1 - d0
        E = np.zeros((self.K, L + 1))
        a = np.clip(first[ok] - d0, 0, L)
        b = np.clip(last[ok] - d0, 0, L)
        sg = seg_of_owner[ok]
        ia, ib = np.floor(a).astype(np.int64), np.floor(b).astype(np.int64)
        # fractional day coverage by a difference array of whole days plus end corrections
        np.add.at(E, (sg, ia), 1.0)
        np.add.at(E, (sg, ib), -1.0)
        E = np.cumsum(E, 1)
        np.add.at(E, (sg, ia), -(a - ia))
        np.add.at(E, (sg, ib), (b - ib))
        return E[:, :L], inner

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

    def _fit_params(self, keep_log=False):
        st = self.stats
        self.theta_hat = np.tile(PRIOR_MEAN, (self.K, 1))
        self.theta_cov = np.tile(np.diag(PRIOR_SD ** 2), (self.K, 1, 1))
        cal = self.fit_log.get("calibration") if keep_log else None
        self.fit_log = {"segments": {}}
        if cal is not None:
            self.fit_log["calibration"] = cal
        self.life = [_Lifetime().fit(np.ones(1, np.int64), np.full(1, 0.5), np.full(1, np.nan))] * self.K
        if st is None:
            return
        pooled = _Data(st["n"], st["A"], st["B"], st["uf"], st["ul"], st["gl"])
        th0, cov0, ok0 = _fit_theta(pooled, PRIOR_MEAN, PRIOR_SD)
        self.pooled = th0
        life0 = _Lifetime().fit(pooled.n, pooled.h, pooled.q)
        self.life = [life0] * self.K
        self.fit_log["pooled"] = {"theta": dict(zip(NAMES, np.round(th0, 4).tolist())), "converged": ok0,
                                  "entities": pooled.n_entities}
        sd = np.full(NPAR, self.shrinkage)
        for k in range(self.K):
            m = st["seg"] == k
            if m.sum() < self.min_entities:
                self.theta_hat[k], self.theta_cov[k] = th0, cov0
                self.life[k] = _Lifetime().fit(pooled.n, pooled.h, pooled.q)
                self.fit_log["segments"][k] = {"entities": int(m.sum()), "pooled": True}
                continue
            dk = _Data(st["n"][m], st["A"][m], st["B"][m], st["uf"][m], st["ul"][m], st["gl"][m])
            th, cov, ok = _fit_theta(dk, th0, sd)
            self.theta_hat[k], self.theta_cov[k] = th, cov
            self.life[k] = _Lifetime().fit(dk.n, dk.h, dk.q)
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
        self.life = [lf if any(lf is x for x in self.life[:k]) else lf.draw(rng) for k, lf in enumerate(self.life)]
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
        owner = np.repeat(np.arange(len(n)), n)
        u = np.empty(int(n.sum()))
        start = np.concatenate([[0], np.cumsum(n)[:-1]])
        h = np.full(len(n), 0.5)
        q = np.full(len(n), np.nan)
        for k in np.unique(seg[n >= 1]):
            mk = (seg == k) & (n >= 1)
            h[mk], q[mk] = self.life[k].sample(n[mk], rng)
        first_all = uA + h * U
        one = n == 1
        if one.any():
            u[start[one]] = first_all[one]
        two = n >= 2
        if two.any():
            first = first_all[two]
            span = q[two] * (uB[two] - first)
            idx = np.flatnonzero(two)
            u[start[two]] = first
            u[start[two] + n[two] - 1] = first + span
            three = n[idx] >= 3
            if three.any():
                ii = idx[three]
                alpha = np.exp(np.clip(th[ii, 0] + np.exp(th[ii, 1]) * rng.standard_normal(len(ii)), -12, 8))
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
        return {"calendar": self.intensity.summary(), "parameters": NAMES, **self.fit_log,
                "lifetime": {k: lf.summary() for k, lf in enumerate(self.life)}}

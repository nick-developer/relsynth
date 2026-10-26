# SPDX-License-Identifier: Apache-2.0
# Reference copy of relsynth/relational.py as it was before the v0.8 row-restricted
# occupancy likelihood and in-place Gumbel argmax (its count, Dirichlet and Gaussian
# kernels are checked against plain numpy separately).  Used only by check C69.
"""Relational latent-segment model (layer 1 algorithms 3 and 4).

Every entity table gets a latent segment per row.  Every link between tables
becomes a Poisson block model on segment pairs, fitted from aggregated tuple
statistics (one GROUP BY per relation):

  edge   child entity -> parent entity (many-to-one FK, snowflake levels)
  fact   fact table with an anchor dimension and a partner dimension
  extra  further dimensions of a fact (anchor x extra dimension)
  single fact table with only one dimension (counts per anchor row)
  tree   self-referencing FK (multi-type Galton-Watson process)

Blocked Gibbs: all rows of one entity type are conditionally independent given
the other types' segments (tree types: given the opposite depth parity).
Over-specified caps + sparse Dirichlet priors, then a greedy merge that scores
the collapsed marginal likelihood exactly from block statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln, logsumexp

from .kernels import count_table, dirichlet_rows, gauss_ll

GAMMA_DIR = 1.0          # total Dirichlet concentration for segment weights
NIG = dict(k0=0.1, a0=2.0, b0=1.0)   # attribute normal-score prior
PCLIP = 1e-12
CHUNK_CELLS = 2_000_000  # rows x segments per scoring chunk in the Gibbs update


# ---------------------------------------------------------------------------
# Structural zeros and ones.  Relational data is full of combinations that can
# never occur (a cell of another form version, a date outside a reference
# period) and of ones that always occur (every submission files the module's
# templates).  A conjugate prior puts positive mass on every rate, so a
# posterior draw leaks O(1/n) probability into impossible combinations.  The
# priors below add point masses: a rate is structurally 0 (or a presence
# probability structurally 1) with prior probability `pi_s`; the posterior
# weight of each component is the exact Bayes factor of the observed counts.
# ---------------------------------------------------------------------------
def zi_gamma_draw(C, E, a, b, pi_s, rng):
    """Poisson rate lambda with prior pi_s*delta_0 + (1-pi_s)*Gamma(a, b);
    data: count C over exposure E.  Elementwise over arrays."""
    C, E = np.asarray(C, float), np.asarray(E, float)
    b = np.broadcast_to(np.asarray(b, float), C.shape)
    a = np.broadcast_to(np.asarray(a, float), C.shape)
    slab = rng.gamma(a + C, 1.0 / (b + E))
    if pi_s <= 0:
        return slab
    # P(spike | C = 0) = pi / (pi + (1 - pi) * (b / (b + E))^a)
    log_bf = a * (np.log(b) - np.log(b + E))
    p_spike = np.where(C == 0, pi_s / (pi_s + (1 - pi_s) * np.exp(log_bf)), 0.0)
    return np.where(rng.random(C.shape) < p_spike, 0.0, slab)


def zoib_draw(S, T, a, b, pi_s, rng):
    """Bernoulli probability with prior (pi_s/2) delta_0 + (pi_s/2) delta_1 +
    (1 - pi_s) Beta(a, b); data: S successes in T trials.  Elementwise."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    slab = rng.beta(a + S, b + T - S)
    if pi_s <= 0:
        return slab
    lw_slab = np.log1p(-pi_s) + betaln(a + S, b + T - S) - betaln(a, b)
    lw0 = np.where(S == 0, np.log(pi_s / 2), -np.inf)
    lw1 = np.where(S == T, np.log(pi_s / 2), -np.inf)
    L = np.stack([lw0, lw1, lw_slab])
    P = np.exp(L - L.max(0))
    P /= P.sum(0)
    u = rng.random(S.shape)
    return np.where(u < P[0], 0.0, np.where(u < P[0] + P[1], 1.0, slab))


def eb_beta_rows(S, T, a0, b0, min_support=3):
    """Empirical-Bayes Beta slab for each row k of a block matrix of presence
    counts (partial pooling of p[k, l] across partner segments l).

    On the support of row k (entries with S > 0, the others being candidates for
    a structural zero), the mean m = sum S / sum T and the between-entry variance
    tau^2 = var(S/T) - mean(m(1-m)/T) give the Beta-binomial concentration
    c = m(1-m)/tau^2 - 1 (Kleinman 1973 moment estimator), bounded to
    [1, 1e4].  The slab is then Beta(c m, c (1 - m)): entries whose observed
    rates differ only by binomial noise are shrunk to the row mean, entries that
    genuinely differ keep their own rate.  Rows with fewer than min_support
    supported entries keep the Beta(a0, b0) prior."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    A = np.full(S.shape, float(a0))
    B = np.full(S.shape, float(b0))
    for k in range(S.shape[0]):
        sup = (S[k] > 0) & (T[k] > 0)
        if sup.sum() < min_support:
            continue
        s_, t_ = S[k, sup], T[k, sup]
        m = s_.sum() / t_.sum()
        if m <= 0 or m >= 1:
            continue
        r = s_ / t_
        tau2 = r.var(ddof=1) - np.mean(m * (1 - m) / t_)
        c = 1e4 if tau2 <= 0 else float(np.clip(m * (1 - m) / tau2 - 1, 1.0, 1e4))
        A[k], B[k] = c * m, c * (1 - m)
    return A, B


def grouped_zoib_draw(S, T, a, b, pi_s, groups, rng):
    """Bernoulli probabilities p[k, l] whose structural support is shared by the
    rows k of a group (e.g. the segments of one stratum):

        G_gl ~ (pi_s/2) 'all zero' + (pi_s/2) 'all one' + (1 - pi_s) 'free'
        p_kl = 0 / 1 for every k in g if G_gl is all zero / all one,
        p_kl ~ zoib prior independently per k if G_gl is free.

    The posterior of G_gl is exact: 'all zero' needs S_kl = 0 for every k in g,
    'all one' needs S_kl = T_kl, and 'free' has the product of the per-row zoib
    marginals pi_s/2 [S=0] + pi_s/2 [S=T] + (1-pi_s) B(a+S, b+T-S)/B(a, b)
    (a group without exposure gets its prior back from the same formula).
    Pooling the evidence of all rows of the group makes a combination that never
    occurs anywhere in the stratum structurally absent with high posterior
    probability even when each row alone has little exposure.  Groups with one
    row use the per-row draw (the two models coincide for one row)."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    out = zoib_draw(S, T, a, b, pi_s, rng)
    if pi_s <= 0 or groups is None:
        return out
    groups = np.asarray(groups)
    lh, lq = np.log(pi_s / 2), np.log1p(-pi_s)
    lslab = lq + betaln(a + S, b + T - S) - betaln(a, b)
    lm = np.logaddexp(np.logaddexp(np.where(S == 0, lh, -np.inf), np.where(S == T, lh, -np.inf)), lslab)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        if len(rows) < 2 or g < 0:
            continue
        all0 = (S[rows] == 0).all(0)
        all1 = (S[rows] == T[rows]).all(0)
        L = np.stack([np.where(all0, lh, -np.inf), np.where(all1, lh, -np.inf), lq + lm[rows].sum(0)])
        P = np.exp(L - L.max(0))
        P /= P.sum(0)
        u = rng.random(L.shape[1])
        st = np.where(u < P[0], 0, np.where(u < P[0] + P[1], 1, 2))
        out[np.ix_(rows, np.flatnonzero(st == 0))] = 0.0
        out[np.ix_(rows, np.flatnonzero(st == 1))] = 1.0
    return out


def sparse_dirichlet_draw(counts, alpha, pi_s, rng):
    """Categorical probabilities with a Dirichlet(alpha) slab over the included
    levels, every level being structurally absent a priori with probability pi_s
    (independently).  Observed levels are included; the inclusion set J of the
    unobserved levels is drawn from its exact joint posterior

        P(J | c) propto prod_{j in J} rho * Gamma(A_o + A_J) / Gamma(A_o + A_J + n),
        rho = (1 - pi_s) / pi_s,  A_o = sum of alpha over observed levels,

    (Dirichlet-multinomial likelihood; the factors of unobserved levels are 1).
    Writing Gamma(x)/Gamma(x+n) = B(x, n)/Gamma(n) as an integral over u in (0, 1)
    makes the levels independent given u:
        u ~ p(u) propto u^(A_o-1) (1-u)^(n-1) prod_j (1 + rho u^alpha_j),
        j in J independently with probability rho u^alpha_j / (1 + rho u^alpha_j),
    so J is sampled exactly (u by inverse cdf on a fine grid of logit u).  Then
    p_J+obs ~ Dirichlet(c + alpha) on the included levels."""
    c = np.asarray(counts, float)
    al = np.maximum(np.asarray(alpha, float), 1e-12)
    n = c.sum()
    keep = np.ones(len(c), bool)
    zero = c == 0
    if pi_s > 0 and n > 0 and zero.any():
        if pi_s >= 1:
            keep = ~zero
        else:
            lrho = np.log1p(-pi_s) - np.log(pi_s)
            Ao = al[~zero].sum()
            az = al[zero]

            def logf(y):
                lu, l1u = -np.logaddexp(0, -y), -np.logaddexp(0, y)       # log u, log(1-u)
                # density of y = logit(u): includes the Jacobian u (1 - u)
                out = Ao * lu + n * l1u
                for i in range(0, len(az), 2048):
                    out = out + np.logaddexp(0, lrho + np.outer(lu, az[i:i + 2048])).sum(1)
                return out
            y = np.linspace(-60, 60, 2401)
            ly = logf(y)
            m = np.flatnonzero(ly > ly.max() - 40)
            y = np.linspace(y[max(m[0] - 1, 0)], y[min(m[-1] + 1, len(y) - 1)], 2049)
            ly = logf(y)
            w = np.exp(ly - ly.max())
            cell = (w[:-1] + w[1:]) / 2                                     # trapezoid mass per grid cell
            i = rng.choice(len(cell), p=cell / cell.sum())
            yy = y[i] + rng.random() * (y[i + 1] - y[i])
            lu = -np.logaddexp(0, -yy)
            pin = 1 / (1 + np.exp(-(lrho + az * lu)))                     # rho u^a / (1 + rho u^a)
            keep[zero] = rng.random(len(az)) < pin
    out = np.zeros(len(c))
    if keep.any():
        out[keep] = rng.dirichlet(c[keep] + al[keep])
    return out


def gumbel_argmax(logp, rng):
    return np.argmax(logp - np.log(-np.log(rng.random(logp.shape))), axis=1)


def kmeanspp(X, K, rng, iters=25):
    n = len(X)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    K = min(K, n)
    cent = [X[rng.integers(n)]]
    d2 = ((X - cent[0]) ** 2).sum(1)
    for _ in range(1, K):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        cent.append(X[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((X - cent[-1]) ** 2).sum(1))
    C = np.array(cent)
    for _ in range(iters):
        D = ((X[:, None, :] - C[None]) ** 2).sum(2) if n * K * X.shape[1] < 5e7 else \
            np.stack([((X - c) ** 2).sum(1) for c in C], 1)
        z = D.argmin(1)
        for k in range(K):
            if (z == k).any():
                C[k] = X[z == k].mean(0)
    return z


class Entity:
    def __init__(self, name, n, K, Xn, Xc, ncat, strata=None, caps=None):
        """strata: optional observed stratum per row (e.g. the row of a small copied
        parent).  Segments are then nested in strata: segment k belongs to stratum
        seg_stratum[k] and only rows of that stratum may take it.  This conditions
        the latent-class model on an observed covariate, which leaves the data
        likelihood unchanged and makes links to the stratifying parents exact."""
        self.name, self.n = name, n
        self.Xn, self.Xc, self.ncat = Xn, Xc, ncat
        self.strata = None if strata is None else np.asarray(strata, dtype=np.int64)
        if self.strata is not None:
            caps = np.asarray(caps, dtype=np.int64)
            self.seg_stratum = np.repeat(np.arange(len(caps)), caps)
            self.K = int(caps.sum())
        else:
            self.K = max(1, min(K, n))
            self.seg_stratum = np.zeros(self.K, dtype=np.int64)
        self.z = np.zeros(n, dtype=np.int64)
        if self.strata is not None:
            first = np.searchsorted(self.seg_stratum, np.arange(len(caps)))
            self.z = first[self.strata]
        self.rels = []           # (relation, role)
        self.tree = None
        self.fixed = False       # copied reference table: segments = rows, never updated

    @property
    def seg_group(self):
        """Group of each segment for pooling structural support (its stratum;
        one global group without strata).  Can be overridden."""
        return self._seg_group if getattr(self, "_seg_group", None) is not None else self.seg_stratum

    @seg_group.setter
    def seg_group(self, v):
        self._seg_group = v

    def allowed(self, lo=0, hi=None):
        """n x K mask of segments each row may take (None = all)."""
        if self.strata is None:
            return None
        return self.seg_stratum[None, :] == self.strata[lo:hi, None]

    def counts(self):
        return np.bincount(self.z, minlength=self.K).astype(float)

    # ---- attribute model: independent normal scores + Dirichlet categoricals
    def draw_attr_params(self, rng):
        K = self.K
        nk = self.counts()
        p = self.Xn.shape[1]
        self.mu = np.zeros((K, p))
        self.s2 = np.ones((K, p))
        for j in range(p):
            x = self.Xn[:, j]
            sx = np.bincount(self.z, weights=x, minlength=K)
            sxx = np.bincount(self.z, weights=x * x, minlength=K)
            kn = NIG["k0"] + nk
            mn = sx / kn
            an = NIG["a0"] + nk / 2
            bn = NIG["b0"] + 0.5 * (sxx - sx ** 2 / np.maximum(nk, 1e-12) * (nk > 0)) \
                + 0.5 * NIG["k0"] * nk / kn * (sx / np.maximum(nk, 1e-12)) ** 2 * (nk > 0)
            s2 = 1 / rng.gamma(an, 1 / bn)
            self.s2[:, j] = s2
            self.mu[:, j] = mn + np.sqrt(s2 / kn) * rng.standard_normal(K)
        self.theta = []
        for j, V in enumerate(self.ncat):
            cnt = count_table(self.z, self.Xc[:, j], K, V)
            self.theta.append(np.log(dirichlet_rows(rng, 0.5 + cnt)))

    def attr_loglik(self, lo=0, hi=None):
        hi = self.n if hi is None else hi
        L = np.zeros((hi - lo, self.K))
        gauss_ll(L, self.Xn[lo:hi], self.mu, self.s2)
        for j in range(len(self.ncat)):
            L += self.theta[j][:, self.Xc[lo:hi, j]].T
        return L

    def attr_stats(self, z, K):
        """Additive per-segment sufficient statistics of the attributes (merge score)."""
        nk = np.bincount(z, minlength=K).astype(float)
        st = {"nk": nk}
        for j in range(self.Xn.shape[1]):
            x = self.Xn[:, j]
            st[f"sx{j}"] = np.bincount(z, weights=x, minlength=K)
            st[f"sxx{j}"] = np.bincount(z, weights=x * x, minlength=K)
        for j, V in enumerate(self.ncat):
            st[f"c{j}"] = count_table(z, self.Xc[:, j], K, V)
        return st, {k: (1,) + (0,) * (np.ndim(v) - 1) for k, v in st.items()}

    def attr_marginal_stats(self, st):
        tot = 0.0
        nk = st["nk"]
        k0, a0, b0 = NIG["k0"], NIG["a0"], NIG["b0"]
        for j in range(self.Xn.shape[1]):
            sx, sxx = st[f"sx{j}"], st[f"sxx{j}"]
            kn, an = k0 + nk, a0 + nk / 2
            mean = np.where(nk > 0, sx / np.maximum(nk, 1), 0)
            bn = b0 + 0.5 * (sxx - nk * mean ** 2) + 0.5 * k0 * nk * mean ** 2 / kn
            tot += (gammaln(an) - gammaln(a0) + a0 * np.log(b0) - an * np.log(bn)
                    + 0.5 * (np.log(k0) - np.log(kn)) - nk / 2 * np.log(2 * np.pi)).sum()
        for j, V in enumerate(self.ncat):
            cnt = st[f"c{j}"]
            a = 0.5
            tot += (gammaln(V * a) - gammaln(V * a + cnt.sum(1))
                    + (gammaln(a + cnt) - gammaln(a)).sum(1)).sum()
        return tot

    def attr_marginal(self, z, K):
        """Collapsed attribute log marginal likelihood of partition z (merge score)."""
        tot = 0.0
        nk = np.bincount(z, minlength=K).astype(float)
        k0, a0, b0 = NIG["k0"], NIG["a0"], NIG["b0"]
        for j in range(self.Xn.shape[1]):
            x = self.Xn[:, j]
            sx = np.bincount(z, weights=x, minlength=K)
            sxx = np.bincount(z, weights=x * x, minlength=K)
            kn, an = k0 + nk, a0 + nk / 2
            mean = np.where(nk > 0, sx / np.maximum(nk, 1), 0)
            bn = b0 + 0.5 * (sxx - nk * mean ** 2) + 0.5 * k0 * nk * mean ** 2 / kn
            tot += (gammaln(an) - gammaln(a0) + a0 * np.log(b0) - an * np.log(bn)
                    + 0.5 * (np.log(k0) - np.log(kn)) - nk / 2 * np.log(2 * np.pi)).sum()
        for j, V in enumerate(self.ncat):
            cnt = count_table(z, self.Xc[:, j], K, V)
            a = 0.5
            tot += (gammaln(V * a) - gammaln(V * a + cnt.sum(1))
                    + (gammaln(a + cnt) - gammaln(a)).sum(1)).sum()
        return tot


class PairRel:
    """Poisson block model between e1 (anchor) and e2 with optional measures."""

    def __init__(self, name, kind, e1, e2, u, v, c, S=None, Q2=None, table=None):
        self.name, self.kind, self.e1, self.e2 = name, kind, e1, e2
        self.u, self.v, self.c = u, v, c.astype(float)
        self.S = S if S is not None else np.zeros((len(u), 0))
        self.Q2 = Q2 if Q2 is not None else np.zeros((len(u), 0))
        self.q = self.S.shape[1]
        self.table = table
        rate = self.c.sum() / max(e1.n * e2.n, 1)
        self.a_eta, self.b_eta = 0.5, 0.5 / max(rate, 1e-12)
        self.s2 = np.ones(self.q)

    def block_stats(self, z1, z2, K1, K2):
        b = z1[self.u] * K2 + z2[self.v]
        C = np.bincount(b, weights=self.c, minlength=K1 * K2).reshape(K1, K2)
        S = np.stack([np.bincount(b, weights=self.S[:, j], minlength=K1 * K2).reshape(K1, K2)
                      for j in range(self.q)], -1) if self.q else np.zeros((K1, K2, 0))
        Q = np.stack([np.bincount(b, weights=self.Q2[:, j], minlength=K1 * K2).reshape(K1, K2)
                      for j in range(self.q)], -1) if self.q else np.zeros((K1, K2, 0))
        return C, S, Q

    def draw_params(self, rng, pi_s=0.0):
        e1, e2 = self.e1, self.e2
        C, S, Q = self.block_stats(e1.z, e2.z, e1.K, e2.K)
        E = np.outer(e1.counts(), e2.counts())
        self.eta = zi_gamma_draw(C, E, self.a_eta, self.b_eta, pi_s, rng) if pi_s > 0 else \
            rng.gamma(self.a_eta + C, 1 / (self.b_eta + E))
        self.eta = np.maximum(self.eta, 1e-300)
        self.m = np.zeros((e1.K, e2.K, self.q))
        for j in range(self.q):
            lam = 1.0 + C / self.s2[j]
            mean = (S[..., j] / self.s2[j]) / lam
            self.m[..., j] = mean + rng.standard_normal(C.shape) / np.sqrt(lam)
            resid = (Q[..., j] - 2 * self.m[..., j] * S[..., j] + C * self.m[..., j] ** 2).sum()
            self.s2[j] = 1 / rng.gamma(2.0 + 0.5 * C.sum(), 1 / (1.0 + 0.5 * max(resid, 0)))

    def loglik_rows(self, role, lo=0, hi=None):
        """(rows lo..hi of the entity in `role`) x K contribution (0 = e1, 1 = e2)."""
        me, other = (self.e1, self.e2) if role == 0 else (self.e2, self.e1)
        ui, vi = (self.u, self.v) if role == 0 else (self.v, self.u)
        hi = me.n if hi is None else hi
        sel = (ui >= lo) & (ui < hi) if (lo > 0 or hi < me.n) else slice(None)
        eta = self.eta if role == 0 else self.eta.T
        m = self.m if role == 0 else self.m.transpose(1, 0, 2)
        Ko = other.K
        nr = hi - lo
        idx = (ui[sel] - lo) * Ko + other.z[vi[sel]]
        Crow = np.bincount(idx, weights=self.c[sel], minlength=nr * Ko).reshape(nr, Ko)
        L = Crow @ np.log(eta).T - (eta @ other.counts())[None, :]
        for j in range(self.q):
            Srow = np.bincount(idx, weights=self.S[sel, j], minlength=nr * Ko).reshape(nr, Ko)
            L += (Srow @ m[..., j].T - 0.5 * Crow @ (m[..., j] ** 2).T) / self.s2[j]
        return L

    def merge_stats(self, z1, z2, K1, K2):
        C, S, Q = self.block_stats(z1, z2, K1, K2)
        st = {"C": C, "S": S, "Q": Q, "n1": np.bincount(z1, minlength=K1).astype(float),
              "n2": np.bincount(z2, minlength=K2).astype(float)}
        # tags: the side (1 = e1, 2 = e2) indexed by each axis of each statistic
        return st, {"C": (1, 2), "S": (1, 2), "Q": (1, 2), "n1": (1,), "n2": (2,)}

    def marginal_stats(self, st):
        C, S, Q = st["C"], st["S"], st["Q"]
        E = np.outer(st["n1"], st["n2"])
        a, b = self.a_eta, self.b_eta
        tot = (a * np.log(b) - gammaln(a) + gammaln(a + C) - (a + C) * np.log(b + E))[E > 0].sum()
        for j in range(self.q):
            s2, k0 = self.s2[j], self.s2[j] / 1.0
            tot += (0.5 * np.log(k0 / (k0 + C)) - 0.5 / s2 * (Q[..., j] - S[..., j] ** 2 / (k0 + C))).sum() \
                - 0.5 * C.sum() * np.log(2 * np.pi * s2)
        return tot

    def marginal(self, z1, z2, K1, K2):
        C, S, Q = self.block_stats(z1, z2, K1, K2)
        E = np.outer(np.bincount(z1, minlength=K1), np.bincount(z2, minlength=K2)).astype(float)
        a, b = self.a_eta, self.b_eta
        tot = (a * np.log(b) - gammaln(a) + gammaln(a + C) - (a + C) * np.log(b + E))[E > 0].sum()
        for j in range(self.q):
            s2, k0 = self.s2[j], self.s2[j] / 1.0
            tot += (0.5 * np.log(k0 / (k0 + C)) - 0.5 / s2 * (Q[..., j] - S[..., j] ** 2 / (k0 + C))).sum() \
                - 0.5 * C.sum() * np.log(2 * np.pi * s2)
        return tot


class OccRel:
    """Occupancy (grain) model for a fact whose rows are pairs of an anchor and a
    partner row, each pair present at most once or with a modelled multiplicity:

        A_i  ~ Bernoulli(pi_k)                anchor i is active (has any row)
        B_ij ~ Bernoulli(p_kl) | A_i = 1      pair (i, j) is present
        M_ij = 1 + Poisson(mu_g) | B_ij = 1   rows of the pair (g: partner, anchor
                                              or block group; mu = 0 for a unique grain)

    A zero-inflated, multiplicity-augmented Bernoulli block model.  Unlike the
    Poisson block model it never repeats a pair whose grain is unique, and counts
    per anchor are Poisson-binomial (variance <= mean), so regular structures such
    as 'exactly the module's templates' are representable (p = 1).
    Activity of anchors without rows is data-augmented in the Gibbs cycle.

    Structural zeros (set_eligibility): when a key constraint makes some pairs
    impossible (a cell of another template than the anchor's filing), each anchor
    row i and partner row j carry a group code and only pairs with g1_i = g2_j
    are at risk.  Every 'pairs at risk' count (T, the non-present pairs of a row)
    is then taken over eligible pairs only: the quasi-independence model of a
    contingency table with structural zeros (Goodman 1968)."""

    kind = "occ"

    def __init__(self, name, e1, e2, u, v, c, S=None, Q2=None, table=None, mult_by="block",
                 a=0.5, b=0.5):
        self.name, self.e1, self.e2 = name, e1, e2
        self.u, self.v = np.asarray(u, np.int64), np.asarray(v, np.int64)
        self.c = np.asarray(c, float)
        self.x = self.c - 1.0                       # excess rows per present pair
        self.S = S if S is not None else np.zeros((len(self.u), 0))
        self.Q2 = Q2 if Q2 is not None else np.zeros((len(self.u), 0))
        self.q = self.S.shape[1]
        self.table = table
        self.mult_by = mult_by
        self.a, self.b = a, b
        self.prior_mean = None      # optional (strata x K2) centre of the presence prior
        self.prior_conc = 2.0
        self.has = np.bincount(self.u, minlength=e1.n) > 0
        self.act = self.has.copy()
        xm = self.x.sum() / max(len(self.x), 1)
        self.am, self.bm = 0.5, 0.5 / max(xm, 1e-3)
        self.s2 = np.ones(self.q)
        self.g1 = self.g2 = None
        self.G = 0

    def set_eligibility(self, g1, g2):
        """Group codes (0..G-1) of the anchor and partner rows; only pairs within a
        group can be present.  Observed pairs must all be eligible."""
        g1, g2 = np.asarray(g1, np.int64), np.asarray(g2, np.int64)
        if len(self.u) and np.any(g1[self.u] != g2[self.v]):
            raise ValueError(f"{self.name}: an observed pair is outside its eligibility group")
        self.g1, self.g2 = g1, g2
        self.G = int(max(g1.max(initial=-1), g2.max(initial=-1))) + 1

    def elig_counts(self, z1, z2, K1, K2, act=None):
        """N1g[k, g] (active anchor rows of segment k in group g), N2g[l, g]."""
        act = self.act if act is None else act
        G = self.G
        N1g = np.bincount(z1[act] * G + self.g1[act], minlength=K1 * G).reshape(K1, G).astype(float)
        N2g = np.bincount(z2 * G + self.g2, minlength=K2 * G).reshape(K2, G).astype(float)
        return N1g, N2g

    def beta_ab(self, K1=None):
        """Beta prior of p[k, l]: Beta(a, b), or, for segments nested in strata with a
        known stratum-level rate m[s, l] (two-stage refinement), the hierarchical
        prior Beta(c m + eps, c (1 - m) + eps) centred on the stratum's rate."""
        if self.prior_mean is None:
            return self.a, self.b
        st = self.e1.seg_stratum
        m = self.prior_mean[st]
        c, eps = self.prior_conc, 1e-3
        return c * m + eps, c * (1 - m) + eps

    # ---- sufficient statistics under partitions z1, z2
    def stats(self, z1, z2, K1, K2, act=None):
        act = self.act if act is None else act
        b = z1[self.u] * K2 + z2[self.v]
        E = np.bincount(b, minlength=K1 * K2).reshape(K1, K2).astype(float)
        A = np.bincount(z1[act], minlength=K1).astype(float)
        n1 = np.bincount(z1, minlength=K1).astype(float)
        m2 = np.bincount(z2, minlength=K2).astype(float)
        if self.g1 is not None:
            N1g, N2g = self.elig_counts(z1, z2, K1, K2, act)
            T = N1g @ N2g.T                     # eligible pairs at risk per block
        else:
            T = np.outer(A, m2)
        X = np.bincount(b, weights=self.x, minlength=K1 * K2).reshape(K1, K2)
        return E, T, A, n1, m2, X

    def _mult_groups(self, E, X):
        if self.mult_by == "partner":
            return E.sum(0), X.sum(0)
        if self.mult_by == "anchor":
            return E.sum(1), X.sum(1)
        if self.mult_by == "block":
            return E, X
        return None, None

    def draw_params(self, rng, pi_s=0.0):
        e1, e2 = self.e1, self.e2
        E, T, A, n1, m2, X = self.stats(e1.z, e2.z, e1.K, e2.K)
        if pi_s > 0:
            pa, pb = eb_beta_rows(E, T, self.a, self.b)
            grp = getattr(e1, "seg_group", None)
            self.pi = grouped_zoib_draw(A[:, None], n1[:, None], self.a, self.b, pi_s, grp, rng)[:, 0]
            self.p = grouped_zoib_draw(E, T, pa, pb, pi_s, grp, rng)
        else:
            pa, pb = self.beta_ab()
            self.pi = rng.beta(self.a + A, self.b + n1 - A)
            self.p = rng.beta(pa + E, pb + T - E)
        self.row_active = (np.bincount(self.u, minlength=e1.n)[:e1.n] > 0) if getattr(e1, "copied", False) else None
        if getattr(e1, "fixed", False) and e1.K == e1.n:
            # a copied anchor (one segment per row) reappears row for row: whether a known row takes part at
            # all is that row's observed property, not a draw (an empty playlist stays
            # empty); its pairs are still drawn from the block probabilities
            seen = np.bincount(e1.z[self.u], minlength=e1.K)[:e1.K] > 0
            self.pi = seen.astype(float)
        Eg, Xg = self._mult_groups(E, X)
        if Eg is None:
            self.mu = np.zeros(1)
        else:
            bm = slab_rate(Eg, Xg, self.am, self.bm) if pi_s > 0 else self.bm
            self.mu = zi_gamma_draw(Xg, Eg, self.am, bm, pi_s, rng) if pi_s > 0 else \
                rng.gamma(self.am + Xg, 1 / (self.bm + Eg))
        # measures (normal scores per present pair), as in the Poisson block model
        b = e1.z[self.u] * e2.K + e2.z[self.v]
        C = np.bincount(b, weights=self.c, minlength=e1.K * e2.K).reshape(e1.K, e2.K)
        self.m = np.zeros((e1.K, e2.K, self.q))
        for j in range(self.q):
            Sj = np.bincount(b, weights=self.S[:, j], minlength=e1.K * e2.K).reshape(e1.K, e2.K)
            Qj = np.bincount(b, weights=self.Q2[:, j], minlength=e1.K * e2.K).reshape(e1.K, e2.K)
            lam = 1.0 + C / self.s2[j]
            self.m[..., j] = (Sj / self.s2[j]) / lam + rng.standard_normal(C.shape) / np.sqrt(lam)
            resid = (Qj - 2 * self.m[..., j] * Sj + C * self.m[..., j] ** 2).sum()
            self.s2[j] = 1 / rng.gamma(2.0 + 0.5 * C.sum(), 1 / (1.0 + 0.5 * max(resid, 0)))
        # data augmentation: activity of anchors without any pair
        pc = np.clip(self.p, PCLIP, 1 - PCLIP)
        k = e1.z[~self.has]
        if self.g1 is not None:
            _, N2g = self.elig_counts(e1.z, e2.z, e1.K, e2.K)
            logq = (np.log1p(-pc) @ N2g)[k, self.g1[~self.has]]     # eligible partners only
        else:
            logq = (np.log1p(-pc) * m2[None, :]).sum(1)[k]   # log P(no pair | active, segment)
        pik = np.clip(self.pi[k], PCLIP, 1 - PCLIP)
        w1 = pik * np.exp(logq)
        pa = w1 / (w1 + 1 - pik)
        self.act = self.has.copy()
        self.act[~self.has] = rng.random(len(k)) < pa

    def _rows(self, role, lo=0, hi=None):
        me, other = (self.e1, self.e2) if role == 0 else (self.e2, self.e1)
        ui, vi = (self.u, self.v) if role == 0 else (self.v, self.u)
        hi = me.n if hi is None else hi
        sel = (ui >= lo) & (ui < hi) if (lo > 0 or hi < me.n) else slice(None)
        Ko = other.K
        nr = hi - lo
        idx = (ui[sel] - lo) * Ko + other.z[vi[sel]]
        Erow = np.bincount(idx, minlength=nr * Ko).reshape(nr, Ko).astype(float)
        Xrow = np.bincount(idx, weights=self.x[sel], minlength=nr * Ko).reshape(nr, Ko)
        Crow = np.bincount(idx, weights=self.c[sel], minlength=nr * Ko).reshape(nr, Ko)
        return me, other, sel, idx, Erow, Xrow, Crow

    def loglik_rows(self, role, lo=0, hi=None):
        me, other, sel, idx, Erow, Xrow, Crow = self._rows(role, lo, hi)
        hi = me.n if hi is None else hi
        pc = np.clip(self.p, PCLIP, 1 - PCLIP)
        lp, l1p = np.log(pc), np.log1p(-pc)
        if role == 0:
            if self.g1 is not None:
                # partners at risk of each row: the eligible ones (its group)
                N2g = np.bincount(other.z * self.G + self.g2, minlength=other.K * self.G).reshape(other.K, self.G)
                M2r = N2g.T[self.g1[lo:hi]].astype(float)                 # n1 x K2
                logq = M2r @ l1p.T                                        # n1 x K1
            else:
                m2 = other.counts()
                M2r = np.broadcast_to(m2[None, :].astype(float), (hi - lo, other.K))
                logq = (l1p * m2[None, :]).sum(1)[None, :]
            pres = Erow @ lp.T + (M2r - Erow) @ l1p.T                    # n1 x K1
            pik = np.clip(self.pi, PCLIP, 1 - PCLIP)
            L = np.where(self.has[lo:hi, None], np.log(pik)[None, :] + pres,
                         np.logaddexp(np.log1p(-pik)[None, :], np.log(pik)[None, :] + logq))
            mu, lpT, l1pT = self.mu, lp, l1p
            if self.mult_by == "anchor":
                L += Xrow.sum(1)[:, None] * np.log(np.maximum(mu, 1e-300))[None, :] - Erow.sum(1)[:, None] * mu[None, :]
            elif self.mult_by == "block":
                L += Xrow @ np.log(np.maximum(mu, 1e-300)).T - Erow @ mu.T
            m_blk, s2 = self.m, self.s2
        else:
            if self.g1 is not None:
                a_ = self.act
                N1g = np.bincount(self.e1.z[a_] * self.G + self.g1[a_], minlength=self.e1.K * self.G)
                Arow = N1g.reshape(self.e1.K, self.G).T[self.g2[lo:hi]].astype(float)   # n2 x K1
            else:
                Arow = np.bincount(self.e1.z[self.act], minlength=self.e1.K).astype(float)[None, :]
            L = Erow @ lp + (Arow - Erow) @ l1p                          # n2 x K2
            mu = self.mu
            if self.mult_by == "partner":
                L += Xrow.sum(1)[:, None] * np.log(np.maximum(mu, 1e-300))[None, :] - Erow.sum(1)[:, None] * mu[None, :]
            elif self.mult_by == "block":
                L += Xrow @ np.log(np.maximum(mu, 1e-300)) - Erow @ mu
            m_blk, s2 = self.m.transpose(1, 0, 2), self.s2
        for j in range(self.q):
            Srow = np.bincount(idx, weights=self.S[sel, j], minlength=(hi - lo) * other.K).reshape(hi - lo, other.K)
            L += (Srow @ m_blk[..., j].T - 0.5 * Crow @ (m_blk[..., j] ** 2).T) / s2[j]
        return L

    def merge_stats(self, z1, z2, K1, K2):
        E, T, A, n1, m2, X = self.stats(z1, z2, K1, K2)
        st = {"E": E, "A": A, "n1": n1, "m2": m2, "X": X}
        tags = {"E": (1, 2), "A": (1,), "n1": (1,), "m2": (2,), "X": (1, 2)}
        if self.g1 is not None:
            st["N1g"], st["N2g"] = self.elig_counts(z1, z2, K1, K2)
            tags["N1g"], tags["N2g"] = (1,), (2,)
        if self.q:
            b_ = z1[self.u] * K2 + z2[self.v]
            st["C"] = np.bincount(b_, weights=self.c, minlength=K1 * K2).reshape(K1, K2)
            tags["C"] = (1, 2)
            for j in range(self.q):
                st[f"S{j}"] = np.bincount(b_, weights=self.S[:, j], minlength=K1 * K2).reshape(K1, K2)
                st[f"Q{j}"] = np.bincount(b_, weights=self.Q2[:, j], minlength=K1 * K2).reshape(K1, K2)
                tags[f"S{j}"] = tags[f"Q{j}"] = (1, 2)
        return st, tags

    def marginal_stats(self, st):
        E, A, n1, m2, X = st["E"], st["A"], st["n1"], st["m2"], st["X"]
        T = st["N1g"] @ st["N2g"].T if "N1g" in st else np.outer(A, m2)
        K1 = len(n1)
        a, b = self.a, self.b
        tot = (betaln(a + A, b + n1 - A) - betaln(a, b))[n1 > 0].sum()
        ok = T > 0
        if self.prior_mean is not None and K1 == len(self.e1.seg_stratum):
            pa, pb = self.beta_ab()
            tot += (betaln(pa + E, pb + T - E) - betaln(pa, pb))[ok].sum()
        else:
            tot += (betaln(a + E, b + T - E) - betaln(a, b))[ok].sum()
        Eg, Xg = self._mult_groups(E, X)
        if Eg is not None:
            Eg, Xg = np.atleast_1d(Eg), np.atleast_1d(Xg)
            g = Eg > 0
            tot += (self.am * np.log(self.bm) - gammaln(self.am) + gammaln(self.am + Xg[g])
                    - (self.am + Xg[g]) * np.log(self.bm + Eg[g])).sum()
        for j in range(self.q):
            C, S, Q = st["C"].ravel(), st[f"S{j}"].ravel(), st[f"Q{j}"].ravel()
            s2 = self.s2[j]
            tot += (0.5 * np.log(s2 / (s2 + C)) - 0.5 / s2 * (Q - S ** 2 / (s2 + C))).sum() \
                - 0.5 * C.sum() * np.log(2 * np.pi * s2)
        return tot

    def marginal(self, z1, z2, K1, K2):
        """Collapsed log marginal likelihood given the augmented activity."""
        E, T, A, n1, m2, X = self.stats(z1, z2, K1, K2)
        a, b = self.a, self.b
        tot = (betaln(a + A, b + n1 - A) - betaln(a, b))[n1 > 0].sum()
        ok = T > 0
        if self.prior_mean is not None and K1 == len(self.e1.seg_stratum):
            pa, pb = self.beta_ab()
            tot += (betaln(pa + E, pb + T - E) - betaln(pa, pb))[ok].sum()
        else:
            tot += (betaln(a + E, b + T - E) - betaln(a, b))[ok].sum()
        Eg, Xg = self._mult_groups(E, X)
        if Eg is not None:
            Eg, Xg = np.atleast_1d(Eg), np.atleast_1d(Xg)
            g = Eg > 0
            tot += (self.am * np.log(self.bm) - gammaln(self.am) + gammaln(self.am + Xg[g])
                    - (self.am + Xg[g]) * np.log(self.bm + Eg[g])).sum()
        if self.q:
            b_ = z1[self.u] * K2 + z2[self.v]
            C = np.bincount(b_, weights=self.c, minlength=K1 * K2)
            for j in range(self.q):
                S = np.bincount(b_, weights=self.S[:, j], minlength=K1 * K2)
                Q = np.bincount(b_, weights=self.Q2[:, j], minlength=K1 * K2)
                s2 = self.s2[j]
                tot += (0.5 * np.log(s2 / (s2 + C)) - 0.5 / s2 * (Q - S ** 2 / (s2 + C))).sum() \
                    - 0.5 * C.sum() * np.log(2 * np.pi * s2)
        return tot


def slab_rate(Eg, Xg, am, default):
    """Empirical-Bayes rate of the Gamma slab in a spike-and-slab Poisson prior:
    the slab describes groups that are not structurally zero, so its mean is
    estimated from the groups that certainly belong to it (X > 0), not from the
    pooled mean that the structural zeros dilute."""
    Eg, Xg = np.atleast_1d(np.asarray(Eg, float)), np.atleast_1d(np.asarray(Xg, float))
    pos = (Xg > 0) & (Eg > 0)
    if not pos.any():
        return default
    return am / (Xg[pos].sum() / Eg[pos].sum())


def multiplicity_marginal(E, X, by, am, bm, pi_s=0.0):
    """Collapsed log marginal of excess rows X over present pairs E, grouped by
    'partner' (columns), 'anchor' (rows), 'block' or 'none', under the same prior
    as the final draw: pi_s*delta_0 + (1-pi_s)*Gamma(am, bm) per group (spike and
    slab; a group with X = 0 has marginal pi_s + (1-pi_s)(bm/(bm+E))^am)."""
    if by == "none":
        return 0.0 if X.sum() == 0 else -np.inf
    if by == "partner":
        Eg, Xg = E.sum(0), X.sum(0)
    elif by == "anchor":
        Eg, Xg = E.sum(1), X.sum(1)
    else:
        Eg, Xg = E.ravel(), X.ravel()
    g = Eg > 0
    if pi_s > 0:
        bm = slab_rate(Eg, Xg, am, bm)
    slab = am * np.log(bm) - gammaln(am) + gammaln(am + Xg[g]) - (am + Xg[g]) * np.log(bm + Eg[g])
    if pi_s <= 0:
        return float(slab.sum())
    spike = np.where(Xg[g] == 0, np.log(pi_s), -np.inf)
    return float(np.logaddexp(spike, np.log1p(-pi_s) + slab).sum())


def poisson_dup_test(u, v, c, n1, n2, rng, max_pairs=4_000_000):
    """Goodness of fit of the degree-corrected (rank-1) Poisson model to the number
    of repeated pairs.  Under Poisson(lambda_ij = r_i s_j / N) the expected number
    of pairs with two or more rows is sum 1 - e^-l (1 + l).  Returns
    (observed, expected, z); z << 0 means far fewer repeats than any Poisson
    block model allows, i.e. the fact has a grain (occupancy model)."""
    r = np.bincount(u, weights=c, minlength=n1).astype(float)
    s = np.bincount(v, weights=c, minlength=n2).astype(float)
    N = c.sum()
    obs = float((c >= 2).sum())
    if N == 0:
        return obs, 0.0, 0.0
    rr, ss = r[r > 0], s[s > 0]
    tot_pairs = len(rr) * len(ss)
    if tot_pairs <= max_pairs:
        L = np.outer(rr, ss) / N
        q = 1 - np.exp(-L) * (1 + L)
        exp_, var = q.sum(), (q * (1 - q)).sum()
    else:
        i = rng.integers(len(rr), size=max_pairs)
        j = rng.integers(len(ss), size=max_pairs)
        L = rr[i] * ss[j] / N
        q = 1 - np.exp(-L) * (1 + L)
        exp_, var = q.mean() * tot_pairs, (q * (1 - q)).mean() * tot_pairs
    return obs, float(exp_), float((obs - exp_) / np.sqrt(max(var, 1e-9)))


class SingleRel:
    """Fact with a single dimension: N_u ~ Pois(lambda_{z_u}), measures ~ N(m_k, s2)."""

    def __init__(self, name, e, N, S, Q2, table=None):
        self.name, self.kind, self.e1 = name, "single", e
        self.N, self.S, self.Q2 = N.astype(float), S, Q2
        self.q = S.shape[1]
        self.table = table
        self.a, self.b = 0.5, 0.5 / max(self.N.mean(), 1e-9)
        self.s2 = np.ones(self.q)

    def draw_params(self, rng, pi_s=0.0):
        e = self.e1
        nk = e.counts()
        Nk = np.bincount(e.z, weights=self.N, minlength=e.K)
        self.lam = np.maximum(rng.gamma(self.a + Nk, 1 / (self.b + nk)), 1e-300)
        self.m = np.zeros((e.K, self.q))
        for j in range(self.q):
            Sk = np.bincount(e.z, weights=self.S[:, j], minlength=e.K)
            Qk = np.bincount(e.z, weights=self.Q2[:, j], minlength=e.K)
            lam = 1 + Nk / self.s2[j]
            self.m[:, j] = Sk / self.s2[j] / lam + rng.standard_normal(e.K) / np.sqrt(lam)
            resid = (Qk - 2 * self.m[:, j] * Sk + Nk * self.m[:, j] ** 2).sum()
            self.s2[j] = 1 / rng.gamma(2 + 0.5 * Nk.sum(), 1 / (1 + 0.5 * max(resid, 0)))

    def loglik_rows(self, role, lo=0, hi=None):
        sl = slice(lo, hi)
        L = self.N[sl, None] * np.log(self.lam)[None] - self.lam[None]
        for j in range(self.q):
            L += (self.S[sl, j][:, None] * self.m[:, j][None]
                  - 0.5 * self.N[sl, None] * self.m[:, j][None] ** 2) / self.s2[j]
        return L

    def marginal(self, z, K):
        nk = np.bincount(z, minlength=K).astype(float)
        Nk = np.bincount(z, weights=self.N, minlength=K)
        a, b = self.a, self.b
        return (a * np.log(b) - gammaln(a) + gammaln(a + Nk) - (a + Nk) * np.log(b + nk))[nk > 0].sum()

    def merge_stats(self, z, K):
        st = {"nk": np.bincount(z, minlength=K).astype(float), "Nk": np.bincount(z, weights=self.N, minlength=K)}
        return st, {"nk": (1,), "Nk": (1,)}

    def marginal_stats(self, st):
        nk, Nk, a, b = st["nk"], st["Nk"], self.a, self.b
        return (a * np.log(b) - gammaln(a) + gammaln(a + Nk) - (a + Nk) * np.log(b + nk))[nk > 0].sum()


class TreeRel:
    """Self-reference: z_child ~ Cat(psi_{z_parent}), #children ~ Pois(lambda_{z})."""

    def __init__(self, name, e, par, table=None):
        self.name, self.kind, self.e1, self.par = name, "tree", e, par
        self.table = table
        n = e.n
        self.nchild = np.bincount(par[par >= 0], minlength=n).astype(float)
        self.depth = tree_depth(par)
        self.a, self.b = 0.5, 0.5 / max(self.nchild.mean(), 1e-9)

    def draw_params(self, rng, pi_s=0.0):
        e, K = self.e1, self.e1.K
        nonroot = self.par >= 0
        B = np.zeros((K, K))
        np.add.at(B, (e.z[self.par[nonroot]], e.z[nonroot]), 1)
        if pi_s > 0:
            self.psi = np.log(np.maximum(np.stack([sparse_dirichlet_draw(b, np.full(K, 0.5), pi_s, rng)
                                                   for b in B]), 1e-300))
        else:
            self.psi = np.log(np.stack([rng.dirichlet(0.5 + b) for b in B]))
        nk = e.counts()
        Nk = np.bincount(e.z, weights=self.nchild, minlength=K)
        self.lam = np.maximum(rng.gamma(self.a + Nk, 1 / (self.b + nk)), 1e-300)
        roots = ~nonroot
        self.pi_root = rng.dirichlet(GAMMA_DIR / K + np.bincount(e.z[roots], minlength=K))

    def loglik_rows(self, role, lo=0, hi=None):
        if lo > 0 or (hi is not None and hi < self.e1.n):
            return self.loglik_rows(role)[lo:hi]
        e, K = self.e1, self.e1.K
        n = e.n
        nonroot = self.par >= 0
        L = np.zeros((n, K))
        L[~nonroot] += np.log(self.pi_root)[None]
        L[nonroot] += self.psi[e.z[self.par[nonroot]]]
        Bc = np.zeros((n, K))
        np.add.at(Bc, (self.par[nonroot], e.z[nonroot]), 1)
        L += Bc @ self.psi.T
        L += self.nchild[:, None] * np.log(self.lam)[None] - self.lam[None]
        return L

    def marginal(self, z, K):
        nonroot = self.par >= 0
        B = np.zeros((K, K))
        np.add.at(B, (z[self.par[nonroot]], z[nonroot]), 1)
        a = 0.5
        tot = (gammaln(K * a) - gammaln(K * a + B.sum(1)) + (gammaln(a + B) - gammaln(a)).sum(1)).sum()
        nk = np.bincount(z, minlength=K).astype(float)
        Nk = np.bincount(z, weights=self.nchild, minlength=K)
        tot += (self.a * np.log(self.b) - gammaln(self.a) + gammaln(self.a + Nk)
                - (self.a + Nk) * np.log(self.b + nk))[nk > 0].sum()
        return tot

    def merge_stats(self, z, K):
        nonroot = self.par >= 0
        B = np.zeros((K, K))
        np.add.at(B, (z[self.par[nonroot]], z[nonroot]), 1)
        st = {"B": B, "nk": np.bincount(z, minlength=K).astype(float),
              "Nk": np.bincount(z, weights=self.nchild, minlength=K)}
        return st, {"B": (1, 1), "nk": (1,), "Nk": (1,)}

    def marginal_stats(self, st):
        B, nk, Nk = st["B"], st["nk"], st["Nk"]
        K, a = B.shape[0], 0.5
        tot = (gammaln(K * a) - gammaln(K * a + B.sum(1)) + (gammaln(a + B) - gammaln(a)).sum(1)).sum()
        tot += (self.a * np.log(self.b) - gammaln(self.a) + gammaln(self.a + Nk)
                - (self.a + Nk) * np.log(self.b + nk))[nk > 0].sum()
        return tot


def tree_depth(par):
    """Depth of every node; cycles in the source are cut (the node where a
    cycle is detected becomes a root) and par is modified in place."""
    n = len(par)
    depth = np.full(n, -1)
    for s in range(n):
        while depth[s] < 0:
            path, onpath, u = [], set(), s
            while u >= 0 and depth[u] < 0 and u not in onpath:
                path.append(u)
                onpath.add(u)
                u = par[u]
            if u >= 0 and u in onpath:
                par[u] = -1          # cut the cycle and retry
                continue
            base = depth[u] + 1 if u >= 0 else 0
            for i, w in enumerate(reversed(path)):
                depth[w] = base + i
    return depth


class TiedCount:
    """Children per parent row when the child entity's segment is tied to its parent's
    (an order header takes its customer's segment): m_i ~ Pois(mu_{z_i})."""

    def __init__(self, name, parent, idx):
        self.name, self.kind, self.e1, self.table = name, "tied", parent, None
        self.m = np.bincount(idx, minlength=parent.n).astype(float)
        self.a, self.b = 0.5, 0.5 / max(self.m.mean(), 1e-9)

    def draw_params(self, rng, pi_s=0.0):
        e = self.e1
        Mk = np.bincount(e.z, weights=self.m, minlength=e.K)
        self.mu = np.maximum(rng.gamma(self.a + Mk, 1 / (self.b + e.counts())), 1e-300)

    def loglik_rows(self, role, lo=0, hi=None):
        sl = slice(lo, hi)
        return self.m[sl, None] * np.log(self.mu)[None] - self.mu[None]

    def marginal(self, z, K):
        nk = np.bincount(z, minlength=K).astype(float)
        Mk = np.bincount(z, weights=self.m, minlength=K)
        a, b = self.a, self.b
        return (a * np.log(b) - gammaln(a) + gammaln(a + Mk) - (a + Mk) * np.log(b + nk))[nk > 0].sum()

    def merge_stats(self, z, K):
        st = {"nk": np.bincount(z, minlength=K).astype(float), "Mk": np.bincount(z, weights=self.m, minlength=K)}
        return st, {"nk": (1,), "Mk": (1,)}

    def marginal_stats(self, st):
        nk, Mk, a, b = st["nk"], st["Mk"], self.a, self.b
        return (a * np.log(b) - gammaln(a) + gammaln(a + Mk) - (a + Mk) * np.log(b + nk))[nk > 0].sum()


def _merge_axis(arr, axis, a, b):
    """Sum slice b into slice a along an axis and drop b (segments a and b merged)."""
    arr = np.array(arr, dtype=float, copy=True)
    sa = [slice(None)] * arr.ndim
    sb = [slice(None)] * arr.ndim
    sa[axis], sb[axis] = a, b
    arr[tuple(sa)] += arr[tuple(sb)]
    return np.delete(arr, b, axis=axis)


class RelationalModel:
    def __init__(self, entities: dict, relations: list, cfg, rng):
        self.E, self.R, self.cfg, self.rng = entities, relations, cfg, rng
        for r in relations:
            if r.kind == "tree":
                r.e1.tree = r
                r.e1.rels.append((r, 0))
            elif r.kind == "single":
                r.e1.rels.append((r, 0))
            else:
                r.e1.rels.append((r, 0))
                if not getattr(r, "child_only", False):
                    # child_only: the link of a grain entity to the parents of its unique key;
                    # the parents' side is modelled by the entity's occupancy relation instead
                    r.e2.rels.append((r, 1))
        self.trace = []
        self.history = {}

    # ---- tied entities
    def tie(self, child, parent, idx, edge):
        """Tie child's segments to its parent's: z_child = z_parent[idx] (a virtual
        order header takes its customer's segment, so the lines of an order depend
        on the customer as they did before the header was split off).  The child's
        relations then score the parent's segments: their log-likelihoods are summed
        into the parent rows in Gibbs and their marginals into the merge score; the
        number of children per parent becomes a Poisson count per parent segment."""
        child.tie = (parent, np.asarray(idx, dtype=np.int64))
        parent.tied = getattr(parent, "tied", []) + [child]
        child.rels = [(r, ro) for r, ro in child.rels if r is not edge]
        parent.rels = [(r, ro) for r, ro in parent.rels if r is not edge]
        tc = TiedCount(f"{edge.name} (tied)", parent, idx)
        self.R.append(tc)
        parent.rels.append((tc, 0))
        self._sync(parent)

    def _sync(self, p):
        for c in getattr(p, "tied", []):
            c.z = p.z[c.tie[1]]
            c.K = p.K
            c.seg_stratum = np.zeros(p.K, dtype=np.int64)

    def _tied_loglik(self, p):
        """Rows of p x K: log-likelihood of each row's tied children under each segment."""
        tot = np.zeros((p.n, p.K))
        for c in getattr(p, "tied", []):
            idx = c.tie[1]
            step = max(1, int(CHUNK_CELLS // max(p.K, 1)))
            for lo in range(0, c.n, step):
                hi = min(c.n, lo + step)
                L = c.attr_loglik(lo, hi)
                for r, role in c.rels:
                    L = L + r.loglik_rows(role, lo, hi)
                np.add.at(tot, idx[lo:hi], L)
        return tot

    def _tied_score(self, p, z, K):
        s = 0.0
        for c in getattr(p, "tied", []):
            # the children's partition is a function of the parent's, so it adds no
            # partition prior of its own: only its data terms
            zc = z[c.tie[1]]
            s += c.attr_marginal(zc, K)
            for r, role in c.rels:
                if r.kind in ("fact", "edge", "extra", "occ"):
                    s += r.marginal(zc, r.e2.z, K, r.e2.K) if role == 0 else r.marginal(r.e1.z, zc, r.e1.K, K)
                else:
                    s += r.marginal(zc, K)
        return s

    # ---- initialisation: k-means++ on attributes + degree profile
    def init(self):
        for e in self.E.values():
            feats = [e.Xn]
            if e.Xc.shape[1]:
                oh = [np.eye(V)[e.Xc[:, j]] * 0.7 for j, V in enumerate(e.ncat)]
                feats += oh
            for r, role in e.rels:
                if r.kind in ("fact", "edge", "extra", "occ"):
                    ui = r.u if role == 0 else r.v
                    deg = np.bincount(ui, weights=r.c, minlength=e.n)
                    f = [np.log1p(deg)]
                    for j in range(r.q):
                        s = np.bincount(ui, weights=r.S[:, j], minlength=e.n)
                        f.append(s / np.maximum(deg, 1))
                    feats.append(np.column_stack(f))
                elif r.kind == "single":
                    feats.append(np.column_stack([np.log1p(r.N)] + [r.S[:, j] / np.maximum(r.N, 1) for j in range(r.q)]))
                elif r.kind == "tree":
                    feats.append(np.column_stack([np.log1p(r.nchild), np.minimum(r.depth, 6) / 2.0]))
                elif r.kind == "tied":
                    feats.append(np.log1p(r.m)[:, None])
            if e.fixed or getattr(e, "tie", None) is not None:
                continue
            X = np.column_stack(feats) if feats else np.zeros((e.n, 1))
            sd = X.std(0)
            X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
            if e.strata is None:
                e.z = kmeanspp(X, e.K, self.rng) if e.n else e.z
            else:
                for st in np.unique(e.strata):
                    rows = np.flatnonzero(e.strata == st)
                    segs = np.flatnonzero(e.seg_stratum == st)
                    e.z[rows] = segs[kmeanspp(X[rows], len(segs), self.rng)]
        for e in self.E.values():
            self._sync(e)

    def draw_params(self, pi_s=0.0):
        for e in self.E.values():
            e.pi = self.rng.dirichlet(GAMMA_DIR / e.K + e.counts())
            e.draw_attr_params(self.rng)
        for r in self.R:
            r.draw_params(self.rng, pi_s)

    def update_z(self, e):
        if e.n == 0 or e.fixed or getattr(e, "tie", None) is not None:
            return
        Lt = self._tied_loglik(e) if getattr(e, "tied", None) else None
        if e.tree is not None:
            parity = e.tree.depth % 2
            blocks = [parity == 0, parity == 1]
        else:
            blocks = [np.ones(e.n, dtype=bool)]
        # rows are conditionally independent: score them in chunks so that memory
        # stays bounded (rows x segments matrices) for large tables with many segments
        step = max(1, int(CHUNK_CELLS // max(e.K, 1)))
        for blk in blocks:
            znew = np.empty(e.n, dtype=np.int64)
            for lo in range(0, e.n, step):
                hi = min(e.n, lo + step)
                L = e.attr_loglik(lo, hi)
                if e.tree is None:
                    with np.errstate(divide="ignore"):   # structural zeros: pi = 0 -> log 0 = -inf
                        L += np.log(e.pi)[None]
                for r, role in e.rels:
                    L += r.loglik_rows(role, lo, hi)
                if Lt is not None:
                    L += Lt[lo:hi]
                al = e.allowed(lo, hi)
                if al is not None:
                    L = np.where(al, L, -np.inf)
                znew[lo:hi] = gumbel_argmax(L, self.rng)
            e.z = np.where(blk, znew, e.z)
        self._sync(e)

    def loglik(self):
        tot = 0.0
        for e in self.E.values():
            step = max(1, int(CHUNK_CELLS // max(e.K, 1)))
            for lo in range(0, e.n, step):
                hi = min(e.n, lo + step)
                with np.errstate(divide="ignore"):
                    L = e.attr_loglik(lo, hi) + (np.log(e.pi)[None] if e.tree is None else 0)
                tot += L[np.arange(hi - lo), e.z[lo:hi]].sum()
        return float(tot)

    def run(self, iters, label="Gibbs"):
        import time as _time
        from .engine import say
        t0 = _time.time()
        last = t0
        for t in range(iters):
            self.draw_params()
            for e in self.E.values():
                self.update_z(e)
            if t % 5 == 0 or t == iters - 1:
                self.trace.append(self.loglik())
            now = _time.time()
            if now - last > 60 or (t == 0 and iters > 1):
                # long fits report their pace: sweeps done and the time each one takes
                say(f"fit: {label} sweep {t + 1}/{iters} ({(now - t0) / (t + 1):.1f}s per sweep)")
                last = now

    # ---- algorithm 4: greedy merge on the collapsed score
    def score(self, e, z, K):
        s = K * np.log(1.0) + gammaln(np.bincount(z, minlength=K)[np.bincount(z, minlength=K) > 0]).sum()
        s += e.attr_marginal(z, K)
        if getattr(e, "tied", None):
            s += self._tied_score(e, z, K)
        for r, role in e.rels:
            if r.kind in ("fact", "edge", "extra", "occ"):
                if r.e1 is r.e2:
                    s += r.marginal(z, z, K, K)
                elif role == 0:
                    s += r.marginal(z, r.e2.z, K, r.e2.K)
                else:
                    s += r.marginal(r.e1.z, z, r.e1.K, K)
            else:
                s += r.marginal(z, K)
        return s

    # ---- merge score from additive block statistics: a merge of segments a and b
    # sums their rows of every statistic, so each candidate costs O(K x K_other)
    # instead of a pass over all rows of every relation (identical value to score()).
    def _merge_terms(self, e):
        """Terms (statistics, merged axes, score function, separable) of the merge
        score of entity e.  A term is separable when its score is a sum over the
        merged entity's segments plus parts that depend on the statistics only
        through their sums over those segments (invariant under any merge); the
        gain of merging a and b is then the term evaluated on slices a and b alone."""
        z, K = e.z, e.K
        terms = []
        nk = np.bincount(z, minlength=K).astype(float)
        terms.append(({"nk": nk}, {"nk": (0,)}, lambda st: gammaln(st["nk"][st["nk"] > 0]).sum(), True))
        st, tg = e.attr_stats(z, K)
        terms.append((st, {k: (0,) for k in st}, e.attr_marginal_stats, True))
        groups = [(e, z, e.rels)]
        for c in getattr(e, "tied", []):
            zc = z[c.tie[1]]
            st, tg = c.attr_stats(zc, K)
            terms.append((st, {k: (0,) for k in st}, c.attr_marginal_stats, True))
            groups.append((c, zc, c.rels))
        for ent, zz, rels in groups:
            for r, role in rels:
                if r.kind in ("fact", "edge", "extra", "occ"):
                    if r.e1 is r.e2:
                        st, tg = r.merge_stats(zz, zz, K, K)
                        sides = (1, 2)
                    elif role == 0:
                        st, tg = r.merge_stats(zz, r.e2.z, K, r.e2.K)
                        sides = (1,)
                    else:
                        st, tg = r.merge_stats(r.e1.z, zz, r.e1.K, K)
                        sides = (2,)
                else:
                    st, tg = r.merge_stats(zz, K)
                    sides = (1,)
                # trees (both axes are the entity's) and self-links couple segments; an
                # occupancy prior centred per stratum indexes all segments
                sep = r.kind != "tree" and r.e1 is not getattr(r, "e2", None) and getattr(r, "prior_mean", None) is None
                terms.append((st, {k: tuple(i for i, sd in enumerate(v) if sd in sides) for k, v in tg.items()},
                              r.marginal_stats, sep))
        return terms

    @staticmethod
    def _merge_terms_ab(terms, a, b):
        out = []
        for st, tg, fn, sep in terms:
            st2 = {}
            for k, v in st.items():
                for ax in tg.get(k, ()):
                    v = _merge_axis(v, ax, a, b)
                st2[k] = v
            out.append((st2, tg, fn, sep))
        return out

    @staticmethod
    def _terms_score(terms):
        return float(sum(fn(st) for st, tg, fn, sep in terms))

    @staticmethod
    def _merge_gain(terms, cur_t, a, b):
        """Score change of merging segments a and b: separable terms on the two
        slices only (O(K_other) each), the others in full."""
        d = 0.0
        for (st, tg, fn, sep), c in zip(terms, cur_t):
            if sep:
                two, one = {}, {}
                for k, v in st.items():
                    axs = tg.get(k, ())
                    if axs:
                        v2 = np.take(v, [a, b], axis=axs[0])
                        two[k], one[k] = v2, v2.sum(axis=axs[0], keepdims=True)
                    else:
                        two[k] = one[k] = v
                d += fn(one) - fn(two)
            else:
                st2 = {}
                for k, v in st.items():
                    for ax in tg.get(k, ()):
                        v = _merge_axis(v, ax, a, b)
                    st2[k] = v
                d += fn(st2) - c
        return d

    def compact(self, e):
        if getattr(e, "tie", None) is not None:
            return                      # follows its parent (see _sync)
        used = np.unique(e.z)
        remap = -np.ones(max(e.K, 1), dtype=np.int64)
        remap[used] = np.arange(len(used))
        e.z = remap[e.z]
        e.seg_stratum = e.seg_stratum[used] if len(used) else e.seg_stratum[:1]
        e.K = max(len(used), 1)
        self._sync(e)

    def merge(self):
        from .engine import say
        say("fit: merging segments")
        hist = {}
        for e in self.E.values():
            self.compact(e)
        changed = True
        while changed:
            changed = False
            for e in self.E.values():
                if e.fixed or getattr(e, "tie", None) is not None:
                    continue
                terms = self._merge_terms(e)
                while e.K > 1:
                    cur_t = [None if sep else fn(st) for st, tg, fn, sep in terms]
                    best, arg = -np.inf, None
                    for a in range(e.K):
                        for b in range(a + 1, e.K):
                            if e.seg_stratum[a] != e.seg_stratum[b]:
                                continue          # segments never span strata
                            d = self._merge_gain(terms, cur_t, a, b)
                            if d > best:
                                best, arg = d, (a, b)
                    if best <= 0:
                        break
                    a, b = arg
                    terms = self._merge_terms_ab(terms, a, b)
                    z2 = np.where(e.z == b, a, e.z)
                    z2 = np.where(z2 > b, z2 - 1, z2)
                    e.z, e.K = z2, e.K - 1
                    e.seg_stratum = np.delete(e.seg_stratum, b)
                    self._sync(e)
                    hist.setdefault(e.name, []).append({"merged": [int(a), int(b)], "delta": float(best)})
                    changed = True
        self.history["merges"] = hist

    def fit(self):
        g = self.cfg["gibbs"]
        caps = {e.name: e.K for e in self.E.values()}
        self.init()
        self.run(g["iterations"])
        occupied = {e.name: int(len(np.unique(e.z))) for e in self.E.values()}
        if g.get("merge", True):
            self.merge()
        else:
            for e in self.E.values():
                self.compact(e)
        self.run(g["final_iterations"], "final Gibbs")
        for e in self.E.values():
            self.compact(e)
        # one posterior draw given the final partition, under the structural priors
        self.draw_params(pi_s=self.cfg.get("structure", {}).get("structural_prior", 0.0))
        self.history.update({"caps": caps, "occupied_before_merge": occupied,
                             "segments_final": {e.name: e.K for e in self.E.values()},
                             "loglik_trace": self.trace})
        return self


def dispersion_alpha(totals, z, K, mu_k, threshold):
    """Gamma frailty shape per segment from the dispersion of per-row totals:
    Var = mu + mu^2/alpha.  Returns (alpha_k or inf, dispersion index D_k)."""
    alpha = np.full(K, np.inf)
    D = np.ones(K)
    for k in range(K):
        t = totals[z == k]
        if len(t) < 30:        # a dispersion estimate from fewer rows is too noisy to act on
            continue
        m, v = t.mean(), t.var(ddof=1)
        mu = mu_k[k] if mu_k is not None else m
        D[k] = v / max(m, 1e-9)
        if D[k] > threshold and v > m:
            alpha[k] = max(m ** 2 / (v - m), 0.05)
    return alpha, D

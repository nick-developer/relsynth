# SPDX-License-Identifier: Apache-2.0
"""Status transitions along an entity's events: a multi-state semi-Markov
model inside the sequence model (``sequence.states: <column>``).

Event tables often carry a status that evolves along an entity's history:
an account goes open -> active -> dormant -> closed, an order line placed ->
shipped -> returned, a patient visit triage -> treatment -> follow-up.  The
attribute model draws each event's status on its own, so the order of a
history's statuses is lost.  This model draws the sequence of statuses of
each entity's events as a Markov chain per segment (a discrete-time
multi-state model; Andersen and Keiding 2002), and lets the gap that follows
an event depend on its status (semi-Markov; Janssen and Manca 2006).

Chain (per segment k, L status levels, Prop. 1822):

    s_1 ~ Categorical(pi_k),   s_{j+1} | s_j ~ Categorical(P_k[s_j, .]),
    pi_k ~ Dirichlet(n0_k + c p0),  P_k[a, .] ~ Dirichlet(N_k[a, .] + c P0[a, .]),

the conjugate posterior of the transition counts N_k (Anderson and Goodman
1957; Lee, Judge and Zellner 1970) with a prior of strength c centred on the
pooled chain (p0, P0), which is itself the posterior mean under a uniform
Dirichlet(1/L) prior.  Segments with fewer than min_entities entities use the
pooled chain.

Holding times (Prop. 1823).  The renewal bridge draws the n - 1 gap shares of
an entity's span as Dirichlet(alpha_i, ..., alpha_i), the law of normalised
i.i.d. Gamma(alpha_i) gaps.  Here the gap after an event in status s has shape
alpha_i exp(delta_s), with delta_ref = 0 for the most frequent status:

    (g_1, ..., g_{n-1}) / span ~ Dirichlet(alpha_i e^{delta_{s_1}}, ..., alpha_i e^{delta_{s_{n-1}}}),

which is the exact law of independent Gamma(alpha_i e^{delta_s}, scale)
holding times given their sum (Lukacs 1955), so a status with delta_s > 0
holds longer and more regularly.  log alpha_i ~ Normal(mu, sigma^2) is the
entity effect of the renewal bridge, integrated out by Gauss-Hermite
quadrature; theta_k = (mu, log sigma, delta) has a Normal prior centred on
the pooled fit and a Laplace posterior.  delta = 0 is the renewal bridge.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

from ._bayes import draw as _draw
from ._bayes import map_laplace

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)
GH_LW = np.log(GH_W / GH_W.sum())
MAX_LEVELS = 50


class StateModel:
    """Markov chain of statuses and status-dependent gap shapes per segment."""

    def __init__(self, prior=1.0, shrinkage=1.0, min_entities=30):
        self.prior = float(prior)
        self.shrinkage = float(shrinkage)
        self.min_entities = int(min_entities)

    # ------------------------------------------------------------ fitting
    def fit(self, s, start, n, seg, K, L, log_share=None):
        """s: status codes 0..L-1 (-1: unknown) of the events, sorted within each
        entity (entity e at s[start[e]:start[e] + n[e]]), seg[e] its segment;
        log_share: log of each gap's share of the entity's span, stored at the
        event that starts the gap (NaN elsewhere)."""
        self.K, self.L = int(K), int(L)
        s = np.asarray(s, dtype=np.int64)
        n = np.asarray(n, dtype=np.int64)
        seg = np.asarray(seg, dtype=np.int64)
        ent = np.repeat(np.arange(len(n)), n)
        pos = np.arange(len(s)) - np.repeat(start, n)
        first = pos == 0
        nxt = np.flatnonzero(pos[:-1] < np.repeat(n, n)[:-1] - 1) if len(s) > 1 else np.zeros(0, np.int64)
        ok = (s[nxt] >= 0) & (s[nxt + 1] >= 0)
        a, b, k = s[nxt][ok], s[nxt + 1][ok], seg[ent[nxt]][ok]
        N = np.zeros((self.K, L, L))
        np.add.at(N, (k, a, b), 1.0)
        f = first & (s >= 0)
        N0 = np.zeros((self.K, L))
        np.add.at(N0, (seg[ent[f]], s[f]), 1.0)
        self.N, self.N0 = N, N0
        # pooled chain: posterior mean under Dirichlet(1/L)
        Np, N0p = N.sum(0), N0.sum(0)
        self.P0 = (Np + 1.0 / L) / (Np + 1.0 / L).sum(1, keepdims=True)
        self.p0 = (N0p + 1.0 / L) / (N0p + 1.0 / L).sum()
        n_ent = np.bincount(seg, minlength=self.K)
        self.pooled_seg = n_ent < self.min_entities
        freq = np.bincount(s[s >= 0], minlength=L).astype(float)
        self.ref = int(np.argmax(freq)) if freq.sum() else 0
        self.levels_seen = freq
        self.log = {"levels": L, "transitions": int(N.sum()), "segments_pooled": int(self.pooled_seg.sum()),
                    "pooled_transition": np.round(self.P0, 4).tolist()}
        self._fit_shapes(s, start, n, seg, log_share)
        self.draw(None)
        return self

    def _shape_stats(self, s, start, n, log_share, m):
        """Per entity of mask m with n >= 3: gaps per starting status (C) and their
        log-share sums (G); entities with an unknown status in a gap are left out."""
        idx = np.flatnonzero(m & (n >= 3))
        C = np.zeros((len(idx), self.L))
        G = np.zeros((len(idx), self.L))
        keep = np.ones(len(idx), bool)
        for r, e in enumerate(idx):
            sl = slice(start[e], start[e] + n[e] - 1)
            ss, lg = s[sl], log_share[sl]
            if (ss < 0).any() or not np.isfinite(lg).all():
                keep[r] = False
                continue
            np.add.at(C[r], ss, 1.0)
            np.add.at(G[r], ss, lg)
        return C[keep], G[keep]

    def _loglik(self, th, C, G):
        if not len(C):
            return 0.0
        d = np.zeros(self.L)
        d[np.arange(self.L) != self.ref] = th[2:]
        la = np.clip(th[0] + np.exp(th[1]) * GH_X, -12, 8)              # nodes
        a = np.exp(la)[None, :, None] * np.exp(d)[None, None, :]       # 1 x nodes x L shapes
        D = (C[:, None, :] * a).sum(2)                                # entities x nodes
        f = gammaln(D) - (C[:, None, :] * gammaln(a)).sum(2) + (a * G[:, None, :]).sum(2) + GH_LW[None, :]
        return float(np.sum(logsumexp(f, axis=1)))

    def _fit_shapes(self, s, start, n, seg, log_share):
        p = 2 + self.L - 1
        mean0 = np.zeros(p)
        mean0[1] = -0.5
        sd0 = np.r_[2.0, 1.0, np.ones(self.L - 1)]
        self.th_mode = np.tile(mean0, (self.K, 1))
        self.th_cov = np.tile(np.diag(sd0 ** 2), (self.K, 1, 1))
        if log_share is None:
            return
        log_share = np.asarray(log_share, dtype=float)
        C, G = self._shape_stats(s, start, n, log_share, np.ones(len(n), bool))
        th0, cov0, ok0 = map_laplace(lambda th: self._loglik(th, C, G), mean0, sd0)
        self.th_mode[:], self.th_cov[:] = th0, cov0
        self.log["holding"] = {"pooled": self._named(th0), "converged": ok0, "entities": int(len(C))}
        sd = np.full(p, self.shrinkage)
        for k in range(self.K):
            if self.pooled_seg[k]:
                continue
            Ck, Gk = self._shape_stats(s, start, n, log_share, seg == k)
            if len(Ck) < self.min_entities:
                continue
            th, cov, _ok = map_laplace(lambda th: self._loglik(th, Ck, Gk), th0, sd, x0=th0)
            self.th_mode[k], self.th_cov[k] = th, cov

    def _named(self, th):
        d = np.zeros(self.L)
        d[np.arange(self.L) != self.ref] = th[2:]
        return {"mu_log_alpha": round(float(th[0]), 4), "log_sigma_alpha": round(float(th[1]), 4),
                "delta": np.round(d, 4).tolist()}

    # ------------------------------------------------------- posterior draws
    def draw(self, rng=None):
        """Transition matrices, initial laws and holding-time shapes from their
        posterior (rng given) or at their posterior means."""
        c = self.prior * self.L
        P = np.empty((self.K, self.L, self.L))
        pi = np.empty((self.K, self.L))
        for k in range(self.K):
            Nk = self.N.sum(0) if self.pooled_seg[k] else self.N[k]
            N0k = self.N0.sum(0) if self.pooled_seg[k] else self.N0[k]
            A = Nk + c * self.P0 + 1e-9
            a0 = N0k + c * self.p0 + 1e-9
            if rng is None:
                P[k] = A / A.sum(1, keepdims=True)
                pi[k] = a0 / a0.sum()
            else:
                P[k] = np.stack([rng.dirichlet(r) for r in A])
                pi[k] = rng.dirichlet(a0)
        self.P, self.pi = P, pi
        self.theta = np.stack([_draw(self.th_mode[k], self.th_cov[k], rng) for k in range(self.K)])
        return self

    # ------------------------------------------------------------ sampling
    def sample_states(self, n, seg, rng):
        """Status codes of every event of entities with n events (flattened, in
        event order)."""
        n = np.asarray(n, dtype=np.int64)
        seg = np.asarray(seg, dtype=np.int64)
        start = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64) if len(n) else n
        s = np.zeros(int(n.sum()), dtype=np.int64)
        if not len(s):
            return s
        cpi = np.cumsum(self.pi, 1)
        cP = np.cumsum(self.P, 2)
        has = np.flatnonzero(n >= 1)
        u = rng.random(len(has))
        s[start[has]] = np.minimum((u[:, None] > cpi[seg[has]]).sum(1), self.L - 1)
        for j in range(1, int(n.max())):
            act = np.flatnonzero(n > j)
            idx = start[act] + j
            prev = s[idx - 1]
            u = rng.random(len(act))
            s[idx] = np.minimum((u[:, None] > cP[seg[act], prev]).sum(1), self.L - 1)
        return s

    def gap_shapes(self, s, n, seg, rng):
        """Gamma shapes of the n - 1 gaps of every entity (flattened; the gap after
        event j has the shape of status s_j), with the entity effect drawn."""
        n = np.asarray(n, dtype=np.int64)
        seg = np.asarray(seg, dtype=np.int64)
        start = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64) if len(n) else n
        th = self.theta[seg]
        la = np.clip(th[:, 0] + np.exp(th[:, 1]) * rng.standard_normal(len(n)), -12, 8)
        D = np.zeros((self.K, self.L))
        D[:, np.arange(self.L) != self.ref] = self.theta[:, 2:]
        m = np.maximum(n - 1, 0)
        ent = np.repeat(np.arange(len(n)), m)
        pos = np.arange(m.sum()) - np.repeat(np.concatenate([[0], np.cumsum(m)[:-1]]) if len(m) else m, m)
        st = s[np.repeat(start, m) + pos]
        return np.exp(np.clip(la[ent] + D[seg[ent], st], -12, 8))

    def summary(self):
        return {**self.log, "reference_level": self.ref}

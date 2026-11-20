# SPDX-License-Identifier: Apache-2.0
"""Self-exciting event histories: a Bayesian Hawkes process with an
exponential kernel, inside the sequence model (``sequence.process: hawkes``).

The renewal bridge of relsynth.timeseries.events treats the gaps of an entity
as independent given its shape.  Many event streams are clustered instead: an
order triggers a return and a re-order within days, a claim triggers follow-up
claims, a login triggers a burst of transactions.  A Hawkes process (Hawkes
1971) has the conditional intensity, in operational time u (Prop. 1200),

    lambda_i(u) = mu_i + eta * beta * sum_{u_k < u} exp(-beta (u - u_k)),

with a baseline (immigrant) rate mu_i per entity, a branching ratio
0 <= eta < 1 (the expected number of events each event triggers) and a decay
rate beta (1 / beta is the mean delay of a triggered event).  It is a
Poisson cluster process (Hawkes and Oakes 1974): immigrants arrive at rate
mu_i and each event has Poisson(eta) children at Exp(beta) delays.

Fitting (per segment, Prop. 1820).  As in the renewal bridge the life-time
law gives each entity's first and last event; the Hawkes part models the
n - 2 events between them on the span (u_1, u_n], with the first event in
the history.  Its log likelihood (Ozaki 1979; Daley and Vere-Jones 2003,
Prop. 7.2.III) is

    l_i = sum_{1 < j < n} log(mu_i + eta beta A_j) - mu_i S_i
          - eta sum_{j < n} (1 - exp(-beta (u_n - u_j))),
    A_j = exp(-beta (u_j - u_{j-1})) (1 + A_{j-1}),  A_1 = 0     (O(n) recursion),

with S_i = u_n - u_1.  The baseline mu_i is profiled out: l_i is concave in
mu_i and its maximiser solves sum_j 1 / (mu_i + eta beta A_j) = S_i (Newton
in log mu; mu_i = 0 when the left side at 0 is at most S_i).  The segment
parameters theta_k = (logit eta, log beta) get a Normal prior centred on the
pooled fit (strength `shrinkage`) and a Laplace posterior (relsynth.timeseries._bayes).

Sampling given the count (Prop. 1821).  The engine fixes how many events an
entity has, so the interior events are drawn from the cluster representation
conditioned on their number.  With N = n - 1 nodes (the first event and the
n - 2 interior events) and r = m + 1 clusters (the first event plus m interior
immigrants):

1. m from p(m | N) proportional to Poisson(m; nu) BT(N; m + 1, eta), where
   BT is the Borel-Tanner law of the total progeny of r Poisson(eta)
   Galton-Watson trees, BT(N; r) = r / N exp(-eta N) (eta N)^(N - r) / (N - r)!
   (Tanner 1961; Haight and Breuer 1960), and nu = max(N (1 - eta) - 1, 0) the
   expected number of interior immigrants given N (E N = r / (1 - eta));
2. the forest: offspring counts xi ~ Multinomial(N - r; 1/N, ..., 1/N) (i.i.d.
   Poisson counts given their sum), turned into a forest of r trees in
   depth-first order by the cycle lemma (Dvoretzky and Motzkin 1947; Pitman
   2006, Lemma 6.1), with the trees in uniform random order (they are
   exchangeable);
3. times: the first tree is rooted at the first event, the other roots are
   uniform on the span (Poisson immigrants given their number), and every
   child follows its parent by an Exp(beta) delay truncated to the span.

Steps 1 and 2 are exact for the cluster process given N; the truncation in
step 3 is the conditioning on the span, exact for each child given its
parent and approximate jointly (an edge effect that vanishes when
1 / beta << S).  ``simulate`` is Ogata's (1981) thinning algorithm for the
unconditional process, used by the tests (relsynth extend does not continue Hawkes histories yet).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, logit

from ._bayes import draw as _draw
from ._bayes import map_laplace

NAMES = ("logit_eta", "log_beta")
MAX_FIT_EVENTS = 2000            # events per entity read by the fit (the first ones)


def excitation(u, start, n, beta):
    """A_j = sum_{k < j} exp(-beta (u_j - u_k)) of every event, for histories
    stored one after another (u sorted within each entity, entity e at
    u[start[e]:start[e] + n[e]]); beta a scalar or one value per entity."""
    u = np.asarray(u, dtype=float)
    A = np.zeros(len(u))
    if not len(n):
        return A
    b = np.broadcast_to(np.asarray(beta, dtype=float), n.shape)
    for j in range(1, int(n.max())):
        act = np.flatnonzero(n > j)
        idx = start[act] + j
        A[idx] = np.exp(-b[act] * (u[idx] - u[idx - 1])) * (1.0 + A[idx - 1])
    return A


def profile_mu(e, eid, S, iters=60):
    """The baseline mu_i >= 0 that maximises sum_j log(mu_i + e_j) - mu_i S_i for
    every entity i (eid: the entity of each term).  The objective is concave in
    mu_i; its stationary point solves g(mu) = sum_j 1/(mu + e_j) - S_i = 0, found
    by Newton steps in x = log mu (g decreases in x).  mu_i = 0 when g(0) <= 0."""
    E = len(S)
    e = np.asarray(e, dtype=float)
    with np.errstate(divide="ignore", over="ignore"):
        g0 = np.bincount(eid, weights=1.0 / e, minlength=E) - S
    pos = g0 > 0
    cnt = np.bincount(eid, minlength=E).astype(float)
    ebar = np.bincount(eid, weights=e, minlength=E) / np.maximum(cnt, 1)
    # Jensen: mean 1/(mu + e) >= 1/(mu + mean e), so cnt/S - mean e is below the root
    x = np.log(np.maximum(cnt / np.maximum(S, 1e-12) - ebar, 1e-8 * np.maximum(cnt / np.maximum(S, 1e-12), 1e-12)))
    for _ in range(iters):
        mu = np.exp(x)
        r = 1.0 / (mu[eid] + e)
        s1 = np.bincount(eid, weights=r, minlength=E)
        s2 = np.bincount(eid, weights=r * r, minlength=E)
        h = np.log(np.maximum(s1, 1e-300)) - np.log(np.maximum(S, 1e-300))
        dh = -mu * s2 / np.maximum(s1, 1e-300)
        step = np.clip(h / np.where(dh < 0, dh, -1.0), -5.0, 5.0)
        x = x - step
        if np.max(np.abs(step[pos]), initial=0.0) < 1e-10:
            break
    return np.where(pos, np.exp(x), 0.0)


class _Data:
    """The histories of a segment's entities with at least three events, in
    operational time, sorted (the first MAX_FIT_EVENTS events of each)."""

    def __init__(self, u, start, n):
        keep = n >= 3
        n = np.minimum(n[keep], MAX_FIT_EVENTS)
        st = start[keep]
        idx = np.concatenate([np.arange(s, s + k) for s, k in zip(st, n)]) if len(n) else np.zeros(0, np.int64)
        self.u = np.asarray(u, float)[idx]
        self.n = n.astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.n)[:-1]]).astype(np.int64) if len(n) else self.n
        self.E = len(self.n)
        last = self.start + self.n - 1
        self.S = np.maximum(self.u[last] - self.u[self.start], 1e-9) if self.E else np.zeros(0)
        pos = np.arange(len(self.u)) - np.repeat(self.start, self.n)
        ent = np.repeat(np.arange(self.E), self.n)
        self.interior = (pos > 0) & (pos < np.repeat(self.n, self.n) - 1)
        self.before_last = pos < np.repeat(self.n, self.n) - 1
        self.eid = ent[self.interior]
        self.ent = ent
        self.u_last = np.repeat(self.u[last], self.n) if self.E else np.zeros(0)
        # weakly informative scale of beta: the median event rate of the span
        self.beta0 = float(np.median((self.n - 1) / self.S)) if self.E else 1.0

    def loglik(self, th):
        if not self.E:
            return 0.0
        eta, beta = float(expit(th[0])), float(np.exp(th[1]))
        if not (np.isfinite(beta) and beta > 0):
            return -np.inf
        A = excitation(self.u, self.start, self.n, beta)
        e = eta * beta * A[self.interior]
        mu = profile_mu(e, self.eid, self.S)
        with np.errstate(divide="ignore"):
            ll = np.sum(np.log(mu[self.eid] + e)) - np.sum(mu * self.S)
        comp = 1.0 - np.exp(-beta * (self.u_last - self.u)[self.before_last])
        return float(ll - eta * comp.sum())

    def residuals(self, th):
        """Compensator increments tau_j = Lambda(u_j) - Lambda(u_{j-1}) of the
        interior events at theta (profiled baselines): Exp(1) under the model
        (time-rescaling theorem; Brown et al. 2002)."""
        if not self.E:
            return np.zeros(0)
        eta, beta = float(expit(th[0])), float(np.exp(th[1]))
        A = excitation(self.u, self.start, self.n, beta)
        e = eta * beta * A[self.interior]
        mu = profile_mu(e, self.eid, self.S)
        idx = np.flatnonzero(self.interior)
        d = self.u[idx] - self.u[idx - 1]
        return mu[self.eid] * d + eta * (1.0 + A[idx - 1]) * (1.0 - np.exp(-beta * d))


class HawkesPart:
    """The Hawkes part of a sequence model: theta_k = (logit eta, log beta) per
    segment with Laplace posteriors, fitted from the histories in operational
    time (see the module docstring)."""

    def __init__(self, shrinkage=1.0, min_entities=30):
        self.shrinkage = float(shrinkage)
        self.min_entities = int(min_entities)

    def fit(self, u, start, n, seg, K):
        """u: operational times sorted within each entity, entity e at
        u[start[e]:start[e] + n[e]], seg[e] its segment."""
        self.K = int(K)
        pooled = _Data(u, start, n)
        b0 = np.log(max(pooled.beta0, 1e-6))
        mean0 = np.array([logit(0.3), b0])
        sd0 = np.array([2.0, 2.5])
        th0, cov0, ok0 = map_laplace(pooled.loglik, mean0, sd0)
        self.mode = np.tile(th0, (self.K, 1))
        self.cov = np.tile(cov0, (self.K, 1, 1))
        self.log = {"pooled": {**self._named(th0), "converged": ok0, "entities": pooled.E}, "segments": {}}
        if pooled.E:
            r = np.sort(pooled.residuals(th0))
            if len(r):
                ks = float(np.max(np.abs((np.arange(1, len(r) + 1) / len(r)) - (1 - np.exp(-r)))))
                self.log["pooled"]["residual_ks"] = round(ks, 4)
        sd = np.full(2, self.shrinkage)
        for k in range(self.K):
            m = seg == k
            if (m & (n >= 3)).sum() < self.min_entities:
                self.log["segments"][k] = {"entities": int((m & (n >= 3)).sum()), "pooled": True}
                continue
            dk = _Data(u, start[m], n[m])
            th, cov, ok = map_laplace(dk.loglik, th0, sd, x0=th0)
            self.mode[k], self.cov[k] = th, cov
            self.log["segments"][k] = {"entities": dk.E, "pooled": False, "converged": ok, **self._named(th)}
        self.draw(None)
        return self

    @staticmethod
    def _named(th):
        return {"eta": round(float(expit(th[0])), 4), "beta": round(float(np.exp(th[1])), 6)}

    def draw(self, rng=None):
        self.theta = np.stack([_draw(self.mode[k], self.cov[k], rng) for k in range(self.K)])
        return self

    def bridge(self, n, seg, S, rng):
        """Interior event shares (see shares()) for entities with n >= 3 events,
        segment seg and span S (operational time)."""
        eta = expit(self.theta[seg, 0])
        beta = np.exp(self.theta[seg, 1])
        return bridge(n, eta, beta * S, rng)


# ------------------------------------------------------------------ sampling
def _immigrants(N, eta, rng):
    """Number of interior immigrants m given N nodes (step 1 of Prop. 1821)."""
    nu = max(N * (1.0 - eta) - 1.0, 0.0)
    if nu <= 0 or N <= 1:
        return 0
    m = np.arange(0, N)                      # r = m + 1 trees, 1 <= r <= N
    r = m + 1
    lw = m * np.log(nu) - gammaln(m + 1) + np.log(r) + (N - r) * np.log(max(eta * N, 1e-300)) - gammaln(N - r + 1)
    if eta <= 0:
        lw = np.where(r == N, 0.0, -np.inf) + m * np.log(nu) - gammaln(m + 1)
    lw -= lw.max()
    p = np.exp(lw)
    return int(rng.choice(len(p), p=p / p.sum()))


def forest(N, r, rng):
    """Parent of each of N nodes (-1 for a root) of a forest of r trees whose
    offspring counts are i.i.d. Poisson given their total N - r, nodes in
    depth-first order with the trees in uniform random order (step 2 of
    Prop. 1821).  Also returns the tree of each node."""
    xi = rng.multinomial(N - r, np.full(N, 1.0 / N)) if N > r else np.zeros(N, np.int64)
    W = np.cumsum(xi - 1)
    k = int(np.argmin(W))                    # first time of the minimum: a valid rotation (cycle lemma)
    xi = np.roll(xi, -(k + 1))
    parent = np.full(N, -1, dtype=np.int64)
    tree = np.zeros(N, dtype=np.int64)
    stack = []                               # (node, children still to attach)
    t = -1
    for j in range(N):
        if stack:
            p = stack[-1]
            parent[j] = p[0]
            tree[j] = tree[p[0]]
            p[1] -= 1
            if p[1] == 0:
                stack.pop()
        else:
            t += 1
            tree[j] = t
        if xi[j] > 0:
            stack.append([j, int(xi[j])])
    order = rng.permutation(r)               # exchangeable trees in random order
    return parent, order[tree]


def bridge(n, eta, bS, rng):
    """Interior shares of entities with n >= 3 events (Prop. 1821): per entity the
    n - 2 positions in (0, 1), sorted, concatenated.  eta: branching ratio, bS:
    beta times the span (the span in units of the mean delay)."""
    n = np.asarray(n, dtype=np.int64)
    eta = np.broadcast_to(np.asarray(eta, dtype=float), n.shape)
    bS = np.broadcast_to(np.asarray(bS, dtype=float), n.shape)
    out = []
    for i in range(len(n)):
        N = int(n[i]) - 1
        if N < 2:
            continue
        m = _immigrants(N, float(eta[i]), rng)
        r = m + 1
        parent, tree = forest(N, r, rng)
        x = np.zeros(N)                      # times in units of the span
        roots = np.flatnonzero(parent < 0)
        rt = tree[roots]
        x[roots] = np.where(rt == 0, 0.0, rng.random(len(roots)))
        lam = max(float(bS[i]), 1e-9)        # delays ~ Exp(lam) in span units
        for j in range(N):                   # depth-first order: parents first
            p = parent[j]
            if p < 0:
                continue
            room = 1.0 - x[p]
            q = -np.expm1(-lam * room)       # P(delay <= room)
            x[j] = x[p] - np.log1p(-rng.random() * q) / lam
        first = int(np.flatnonzero((parent < 0) & (tree == 0))[0])
        inner = np.delete(x, first)
        out.append(np.sort(np.clip(inner, 1e-12, 1 - 1e-12)))
    return np.concatenate(out) if out else np.zeros(0)


def simulate(mu, eta, beta, T, rng, history=()):
    """One unconditional Hawkes history on (0, T] by Ogata's (1981) thinning:
    between events the intensity decays, so its value just after the current
    time bounds it until the next event.  history: earlier event times (<= 0)."""
    h = np.asarray(history, dtype=float)
    A = float(np.exp(beta * h[h <= 0]).sum()) if len(h) else 0.0
    t = 0.0
    out = []
    while True:
        lam_bar = mu + eta * beta * A
        if lam_bar <= 0:
            break
        w = rng.exponential(1.0 / lam_bar)
        if t + w > T:
            break
        A *= np.exp(-beta * w)
        t += w
        if rng.random() * lam_bar <= mu + eta * beta * A:
            out.append(t)
            A += 1.0
    return np.array(out)

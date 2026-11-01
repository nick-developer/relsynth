# SPDX-License-Identifier: Apache-2.0
"""Two-way marginals in stage 1 of DP mode (roadmap T07b; docs/dp.md sec. 7, Props. 1300-1306).

With `synthesis.privacy.dp.measure: mst` or `aim`, stage 1 releases, for each
owned table, the one-way histograms of T07 and a set of two-way histograms
chosen privately, instead of one Gaussian correlation matrix.  The nodes of a
table are its attribute coordinates, its truncated degrees (rows per row in
each owned child table) and, for a child table, its parent row's attribute
cells; a pair of two parent nodes is never a candidate (it is the parent
table's business).

- **mst** (McKenna, Miklau and Sheldon 2021): score every pair by the L1
  distance between its real two-way histogram and the one the independent
  model predicts, then pick a maximum spanning tree edge by edge with the
  exponential mechanism (report noisy max, Gumbel noise, OpenDP's zCDP map),
  and measure the chosen pairs with discrete Gaussian noise.
- **aim** (McKenna et al. 2022): workload-adaptive rounds; each round picks the
  pair whose real histogram the current model gets most wrong, less the noise
  its measurement would add (sqrt(2/pi) sigma cells), measures it, refits the
  model, and spends more per round (sigma / 2) when the last measurement did
  not change the model's estimate by more than the noise.  relsynth keeps
  the model a forest (a pair that would close a cycle is not a candidate,
  a pair already in the forest may be measured again), so the estimate and the
  sampler are exact.  AIM proper allows any junction tree within a size limit.

The estimate (Private-PGM for a tree, McKenna, Sheldon and Miklau 2019) is
post-processing: per node, the inverse-variance combination of its one-way
measurement and the margins of its two-way measurements, projected onto the
simplex; per edge, the noisy table clipped at zero and fitted to both node
estimates by iterative proportional fitting.  The seed rows are drawn from the
forest, a child row given its parent row's cells (belief propagation with the
parent's cells as evidence).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mechanisms import Accountant, noisy_max


@dataclass
class Node:
    name: str               # coordinate name; "^c" for the parent row's attribute c
    k: int                  # number of cells (the first is null)
    evidence: bool = False  # a parent node: observed when a child row is drawn
    y: np.ndarray | None = None     # noisy one-way counts
    var: float = 1.0                # noise variance of each cell of y
    zero: np.ndarray | None = None  # cells that are empty by construction (the null cell of a degree)


@dataclass
class Forest:
    """A tree-structured model over cells: node marginals p[v] and edge joints P[(a, b)]."""
    nodes: dict
    edges: list = field(default_factory=list)          # (a, b) pairs, a < b in node order
    p: dict = field(default_factory=dict)
    P: dict = field(default_factory=dict)
    n: float = 0.0
    groups: dict = field(default_factory=dict)        # node -> coarse cell of each fine cell

    def adj(self) -> dict:
        out: dict = {v: [] for v in self.nodes}
        for a, b in self.edges:
            out[a].append(b)
            out[b].append(a)
        return out

    def components(self) -> dict:
        comp, c = {}, 0
        ad = self.adj()
        for v in self.nodes:
            if v in comp:
                continue
            stack = [v]
            comp[v] = c
            while stack:
                u = stack.pop()
                for w in ad[u]:
                    if w not in comp:
                        comp[w] = c
                        stack.append(w)
            c += 1
        return comp

    def cond(self, u, v) -> np.ndarray:
        """P(x_v | x_u) as a (k_u, k_v) matrix for an edge u-v."""
        J = self.P[(u, v)] if (u, v) in self.P else self.P[(v, u)].T
        return J / np.maximum(J.sum(axis=1, keepdims=True), 1e-300)

    def pair(self, a, b) -> np.ndarray:
        """The model's two-way distribution of nodes a, b (any two nodes)."""
        if (a, b) in self.P:
            return self.P[(a, b)]
        if (b, a) in self.P:
            return self.P[(b, a)].T
        path = self._path(a, b)
        if path is None:
            return np.outer(self.p[a], self.p[b])
        J = np.diag(self.p[a])
        for u, v in zip(path, path[1:]):
            J = J @ self.cond(u, v)
        return J

    def _path(self, a, b):
        ad = self.adj()
        prev = {a: None}
        stack = [a]
        while stack:
            u = stack.pop()
            if u == b:
                break
            for w in ad[u]:
                if w not in prev:
                    prev[w] = u
                    stack.append(w)
        if b not in prev:
            return None
        out = [b]
        while out[-1] != a:
            out.append(prev[out[-1]])
        return out[::-1]


# ------------------------------------------------------------------ compression
def grouping(p: np.ndarray, ordered: bool, K: int) -> np.ndarray:
    """Coarse cell of each fine cell of a released one-way marginal `p` (cell 0
    is null and stays on its own): ordered cells are cut into K - 1 runs of
    about equal mass, unordered levels keep the K - 2 most frequent and pool
    the rest.  Depends on released statistics only (post-processing; the
    domain compression of MST)."""
    k = len(p)
    if k <= K:
        return np.arange(k)
    g = np.zeros(k, dtype=int)
    q = np.asarray(p[1:], dtype=float)
    q = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / len(q))
    if ordered:
        mid = np.cumsum(q) - q / 2
        g[1:] = 1 + np.minimum((mid * (K - 1)).astype(int), K - 2)
        _, g[1:] = np.unique(g[1:], return_inverse=True)
        g[1:] += 1
    else:
        top = np.argsort(-q, kind="stable")[:K - 2]
        g[1:] = K - 1
        g[1 + top] = 1 + np.arange(len(top))
    return g


def refine(coarse: np.ndarray, group: np.ndarray, p: np.ndarray, rng) -> np.ndarray:
    """A fine cell in each coarse cell, drawn from the fine marginal within it."""
    out = np.zeros(len(coarse), dtype=int)
    for G in np.unique(coarse):
        members = np.nonzero(group == G)[0]
        w = np.asarray(p, dtype=float)[members]
        w = w / w.sum() if w.sum() > 0 else np.full(len(members), 1.0 / len(members))
        at = coarse == G
        out[at] = members[rng.choice(len(members), size=int(at.sum()), p=w)]
    return out


# ------------------------------------------------------------------ estimation
def _simplex(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    s = x.sum()
    return x / s if s > 0 else np.full(len(x), 1.0 / len(x))


def estimate(nodes: dict, meas: dict, n: float, iters: int = 30) -> Forest:
    """The forest estimate from one-way measurements (in `nodes`) and two-way
    measurements meas[(a, b)] = (y, var) with y shaped (k_a, k_b).
    Post-processing only."""
    f = Forest(nodes, sorted(meas, key=lambda e: (list(nodes).index(e[0]), list(nodes).index(e[1]))), n=n)
    for v, nd in nodes.items():
        num = np.zeros(nd.k)
        den = 0.0
        if nd.y is not None:
            num += nd.y / nd.var
            den += 1.0 / nd.var
        for (a, b), (y, var) in meas.items():
            if v == a:
                kk = y.shape[1]
                num += y.sum(axis=1) / (kk * var)
                den += 1.0 / (kk * var)
            elif v == b:
                kk = y.shape[0]
                num += y.sum(axis=0) / (kk * var)
                den += 1.0 / (kk * var)
        est = num / den if den > 0 else np.ones(nd.k)
        if nd.zero is not None:
            est = np.where(nd.zero, 0.0, est)
            if not (est > 0).any():
                est = np.where(nd.zero, 0.0, 1.0)
        f.p[v] = _simplex(est)
    for (a, b), (y, _var) in meas.items():
        Q = np.clip(y.astype(float), 0, None)
        Q = Q + max(Q.sum(), 1.0) * 1e-6 / Q.size
        Q = Q * (f.p[a] > 0)[:, None] * (f.p[b] > 0)[None, :]
        Q = Q / Q.sum()
        pa, pb = f.p[a], f.p[b]
        for _ in range(iters):
            Q *= (pa / np.maximum(Q.sum(axis=1), 1e-300))[:, None]
            Q *= (pb / np.maximum(Q.sum(axis=0), 1e-300))[None, :]
        f.P[(a, b)] = Q
    return f


# ------------------------------------------------------------------ selection
def pair_counts(cells: dict, a: str, b: str, ka: int, kb: int) -> np.ndarray:
    return np.bincount(cells[a] * kb + cells[b], minlength=ka * kb).reshape(ka, kb)


def candidates(nodes: dict, max_cells: int) -> list:
    names = list(nodes)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if nodes[a].evidence and nodes[b].evidence:
                continue
            if nodes[a].k * nodes[b].k > max_cells:
                continue
            out.append((a, b))
    return out


def _l1(real: np.ndarray, model_counts: np.ndarray) -> float:
    return float(np.abs(real - model_counts).sum())


def _combine(meas: dict, e, y, var):
    if e in meas:
        y0, v0 = meas[e]
        w0, w1 = 1 / v0, 1 / var
        meas[e] = ((y0 * w0 + y * w1) / (w0 + w1), 1 / (w0 + w1))
    else:
        meas[e] = (y.astype(float), var)


def select_and_measure(method: str, table: str, nodes: dict, cells: dict, n_hat: float, c: int, rho: float,
                       acct: Accountant, gaussian, *, max_cells: int, select_share: float, rounds: int | None,
                       report: dict) -> Forest:
    """Spend `rho` on choosing and measuring two-way marginals of one table;
    return the fitted forest.  `cells[v]` holds each row's cell of node v
    (truncated rows; parent nodes from the parent row).  Every released score
    vector has L-infinity sensitivity c and every two-way histogram L2
    sensitivity c (Prop. 1301)."""
    cand = candidates(nodes, max_cells)
    meas: dict = {}
    sigma_of = (lambda r: c / math.sqrt(2 * r))
    if not cand or rho <= 0:
        return estimate(nodes, meas, n_hat)
    exact = {e: pair_counts(cells, e[0], e[1], nodes[e[0]].k, nodes[e[1]].k) for e in cand}
    base = estimate(nodes, meas, n_hat)
    chosen: list = []
    if method == "mst":
        steps = max(min(len(nodes) - 1, len(cand)), 1)
        r_sel = rho * select_share / steps
        forest = Forest(nodes, [])
        comp = forest.components()
        scores = {e: _l1(exact[e], n_hat * np.outer(base.p[e[0]], base.p[e[1]])) for e in cand}
        for _ in range(steps):
            live = [e for e in cand if comp[e[0]] != comp[e[1]]]
            if not live:
                break
            i = noisy_max([scores[e] for e in live], c, r_sel, acct, "fit", f"{table}: pair selection")
            e = live[i]
            chosen.append(e)
            forest.edges.append(e)
            comp = forest.components()
        r_meas = rho * (1 - select_share) / max(len(chosen), 1) if chosen else 0.0
        for e in chosen:
            y = gaussian(exact[e].ravel(), c, r_meas, acct, "fit", f"{table}: pair {e[0]} x {e[1]}")
            _combine(meas, e, y.reshape(exact[e].shape), sigma_of(r_meas) ** 2)
        model = estimate(nodes, meas, n_hat)
        report["pairs"] = [f"{a} x {b}" for a, b in chosen]
        return model
    # aim
    T = rounds or max(16 * len(nodes), 8)
    alpha = 0.9
    r_round = rho / T
    left = rho
    model = base
    hist: list = []
    while left > 1e-12 * rho:
        if left < 2 * r_round:
            r_round = left
        sig = sigma_of(alpha * r_round)
        forest = Forest(nodes, list(meas))
        comp = forest.components()
        live = [e for e in cand if e in meas or comp[e[0]] != comp[e[1]]]
        if not live:
            break
        pen = math.sqrt(2 / math.pi) * sig
        est = {e: n_hat * model.pair(*e) for e in live}
        q_aim = [_l1(exact[e], est[e]) - pen * exact[e].size for e in live]
        i = noisy_max(q_aim, c, (1 - alpha) * r_round, acct, "fit", f"{table}: pair selection")
        e = live[i]
        y = gaussian(exact[e].ravel(), c, alpha * r_round, acct, "fit", f"{table}: pair {e[0]} x {e[1]}")
        left -= r_round
        _combine(meas, e, y.reshape(exact[e].shape), sig ** 2)
        old = est[e]
        model = estimate(nodes, meas, n_hat)
        hist.append(f"{e[0]} x {e[1]}")
        if _l1(n_hat * model.pair(*e), old) <= pen * exact[e].size:
            r_round = min(4 * r_round, left) if left > 0 else r_round
    report["pairs"] = sorted(set(hist))
    report["rounds"] = len(hist)
    return model


# ------------------------------------------------------------------ sampling
def sample(f: Forest, n: int, evidence: dict, rng) -> dict:
    """n rows of cells from the forest; evidence[v] gives each row's cell of an
    observed node v (shape (n,)).  Belief propagation upwards, then ancestral
    sampling downwards, one row per array row."""
    ad = f.adj()
    out: dict = {}
    seen: set = set()
    names = list(f.nodes)
    for root in [v for v in names if f.nodes[v].evidence] + names:
        if root in seen:
            continue
        order, par = [], {root: None}
        stack = [root]
        while stack:
            u = stack.pop()
            order.append(u)
            for w in ad[u]:
                if w not in par:
                    par[w] = u
                    stack.append(w)
        seen |= set(order)
        lam: dict = {}
        for u in reversed(order):
            k = f.nodes[u].k
            if u in evidence:
                L = np.zeros((n, k))
                L[np.arange(n), np.clip(evidence[u], 0, k - 1)] = 1.0
            else:
                L = np.ones((n, k))
            for w in ad[u]:
                if par.get(w) == u:
                    L *= lam[w] @ f.cond(u, w).T
                    L /= np.maximum(L.max(axis=1, keepdims=True), 1e-300)
            lam[u] = L
        for u in order:
            if u in evidence:
                out[u] = np.asarray(evidence[u], dtype=int)
                continue
            if par[u] is None:
                w = f.p[u][None, :] * lam[u]
            else:
                w = f.cond(par[u], u)[out[par[u]]] * lam[u]
            out[u] = _draw(w, rng)
    return out


def _draw(w: np.ndarray, rng) -> np.ndarray:
    s = w.sum(axis=1, keepdims=True)
    bad = (s[:, 0] <= 0)
    if bad.any():
        w = w.copy()
        w[bad] = 1.0
        s = w.sum(axis=1, keepdims=True)
    cum = np.cumsum(w / s, axis=1)
    u = rng.random((len(w), 1))
    return np.minimum((cum < u).sum(axis=1), w.shape[1] - 1)

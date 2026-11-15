# SPDX-License-Identifier: Apache-2.0
"""Set arrays of scalars: whether their elements are modelled jointly (multi-hot)
and in which order a synthetic array lists them (Prop. 1785).

A set array (tags, genres, verifications) is one 0/1 column per element value
on its holder (multi-hot), which the copula models jointly with the holder's
other attributes (a multivariate probit; Ashford and Sowden 1970): a child
table draws its elements one by one with replacement and cannot hold a set
without repairs.  For the model report:

* How much the joint model carries is summarised by the best tree of pairwise
  dependences (Chow and Liu 1968: the maximum spanning tree of the pairwise
  mutual informations is the maximum-likelihood tree-structured law).  Its
  log-likelihood gain over independent elements is N * sum of the tree's
  mutual informations, for one extra parameter per edge, so by the Bayesian
  information criterion (Schwarz 1978) the dependence is supported when
  ``2 N sum MI - (edges) log N > 0``.
* A synthetic array lists its elements in the order the source uses.  When the
  source's arrays never disagree on the order of two elements (airbnb host
  verifications: no conflict among 18,355 ordered pairs) that order is a
  partial order and every array follows one linear extension of it.  When they
  disagree, the order is a Plackett-Luce draw (Luce 1959, Plackett 1975) with
  Bradley-Terry worths fitted to the precedence counts by the
  minorisation-maximisation algorithm of Hunter (2004); a Plackett-Luce draw is
  the order of log-worth plus Gumbel noise (Yellott 1977).

Everything here is computed from counts the profile already holds; nothing
draws a random number except `order_items` in the Plackett-Luce case, from the
generator it is given.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _mi(n11: float, na: float, nb: float, N: float) -> float:
    """Mutual information (nats) of two 0/1 indicators from their 2x2 counts."""
    cells = [(n11, na, nb), (na - n11, na, N - nb), (nb - n11, N - na, nb), (N - na - nb + n11, N - na, N - nb)]
    out = 0.0
    for nij, ni, nj in cells:
        if nij > 0 and ni > 0 and nj > 0:
            out += nij / N * np.log(nij * N / (ni * nj))
    return float(max(out, 0.0))


def dependence_gain(vocab: list, counts: Counter, pairs: Counter, N: int) -> tuple[float, int]:
    """(BIC gain, edges) of the Chow-Liu tree of the elements' indicators over
    independent indicators: 2 N sum MI - edges * log N.  Prim's algorithm on the
    complete graph of the V elements (V^2 work)."""
    V = len(vocab)
    if V < 2 or N < 2:
        return 0.0, 0
    na = np.array([counts.get(v, 0) for v in vocab], float)
    M = np.zeros((V, V))
    for i in range(V):
        for j in range(i + 1, V):
            n11 = pairs.get((vocab[i], vocab[j]), 0) + pairs.get((vocab[j], vocab[i]), 0)
            M[i, j] = M[j, i] = _mi(float(n11), na[i], na[j], float(N))
    inside = np.zeros(V, bool)
    inside[0] = True
    best = M[0].copy()
    total, edges = 0.0, 0
    for _ in range(V - 1):
        cand = np.where(inside, -1.0, best)
        j = int(np.argmax(cand))
        total += float(cand[j])
        edges += 1
        inside[j] = True
        best = np.maximum(best, M[j])
    return 2.0 * N * total - edges * np.log(N), edges


def element_order(vocab: list, pairs: Counter, iters: int = 200) -> dict:
    """The order model of a set array's elements: {"kind": "fixed", "rank": [...]}
    (a linear extension of the precedence order the source never contradicts,
    ties broken by the net number of elements an element precedes, then by the
    vocabulary order) or {"kind": "plackett_luce", "log_worth": [...]} (Bradley-
    Terry worths by MM with one pseudo-comparison each way per co-observed pair,
    so that worths stay finite)."""
    V = len(vocab)
    W = np.zeros((V, V))                       # W[i, j]: arrays with i before j
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            if i != j:
                W[i, j] = pairs.get((a, b), 0)
    conflict = bool(((W > 0) & (W.T > 0)).any())
    net = (W > 0).sum(1) - (W > 0).sum(0)
    if not conflict:
        # Kahn's algorithm; the precedence graph is acyclic when no pair conflicts
        # and the relation is transitive in the data, else the cycle is broken by net
        indeg = (W > 0).sum(0).astype(int)
        left, rank = set(range(V)), []
        while left:
            ready = [i for i in left if indeg[i] == 0] or list(left)
            i = min(ready, key=lambda k: (-net[k], k))
            rank.append(i)
            left.discard(i)
            for j in np.flatnonzero(W[i] > 0):
                indeg[j] -= 1
        pos = np.empty(V, int)
        pos[rank] = np.arange(V)
        return {"kind": "fixed", "rank": pos.tolist()}
    n = W + W.T
    prior = (n > 0).astype(float) * 0.5            # half a comparison each way per observed pair
    Wp, npair = W + prior, n + 2 * prior
    w = np.ones(V)
    wins = Wp.sum(1)
    for _ in range(iters):
        den = (npair / np.maximum(w[:, None] + w[None, :], 1e-300)).sum(1)
        new = np.where(den > 0, wins / np.maximum(den, 1e-300), w)
        new = new / np.exp(np.mean(np.log(np.maximum(new, 1e-300))))
        if np.allclose(new, w, rtol=1e-10, atol=0):
            w = new
            break
        w = new
    return {"kind": "plackett_luce", "log_worth": np.log(np.maximum(w, 1e-300)).tolist()}


def order_items(held: list[int], model: dict | None, rng: np.random.Generator | None = None) -> list[int]:
    """The held elements (indices into the vocabulary) in a synthetic order."""
    if not held or not model:
        return list(held)
    if model["kind"] == "fixed":
        r = model["rank"]
        return sorted(held, key=lambda i: (r[i], i))
    lw = np.asarray(model["log_worth"])[held]
    g = rng.gumbel(size=len(held)) if rng is not None else np.zeros(len(held))
    return [held[i] for i in np.argsort(-(lw + g), kind="stable")]

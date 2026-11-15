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
  partial order, every array follows one linear extension of it, and a 0/1 cell
  per element holds all there is to know.  When they disagree, the order is
  information of each document: the cell holds 1 + the element's position, and
  the copula models it with the other attributes.

Everything here is computed from counts the profile already holds and draws
no random number.
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


def element_order(vocab: list, pairs: Counter) -> dict:
    """The order model of a set array's elements: {"kind": "fixed", "rank": [...]}
    when no two elements appear in both orders (a linear extension of the
    source's precedence order: Kahn's algorithm, ties broken by the net number of
    elements an element precedes, then by the vocabulary order), else
    {"kind": "positions"} (the cells hold positions)."""
    V = len(vocab)
    W = np.zeros((V, V))                       # W[i, j]: arrays with i before j
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            if i != j:
                W[i, j] = pairs.get((a, b), 0)
    if ((W > 0) & (W.T > 0)).any():
        return {"kind": "positions"}
    net = (W > 0).sum(1) - (W > 0).sum(0)
    indeg = (W > 0).sum(0).astype(int)
    left, rank = set(range(V)), []
    while left:
        # without a conflicting pair the precedence graph can still hold a cycle of
        # three or more elements (a before b, b before c, c before a in different
        # arrays): then the element with the largest net count goes first
        ready = [i for i in left if indeg[i] == 0] or list(left)
        i = min(ready, key=lambda k: (-net[k], k))
        rank.append(i)
        left.discard(i)
        for j in np.flatnonzero(W[i] > 0):
            indeg[j] -= 1
    pos = np.empty(V, int)
    pos[rank] = np.arange(V)
    if any(pos[i] > pos[j] for i in range(V) for j in range(V) if W[i, j] > 0):
        return {"kind": "positions"}           # a cycle: no single order lists every array
    return {"kind": "fixed", "rank": pos.tolist()}

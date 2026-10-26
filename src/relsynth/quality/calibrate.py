"""Null limits from real-versus-real splits.

For a metric T comparing a sample of size a (real) with a sample of size b
(synthetic), the null hypothesis is that both are draws of one law.  The real
data gives draws of T under that hypothesis: split the real rows at random into
halves A and B (child rows follow their parent row, so that a split keeps whole
families) and compute T(A, B).  With R splits the limit is the one-sided
normal prediction bound

    c + kappa * (mean(T_r - c) + t_{R-1}(q) * sd(T_r) * sqrt(1 + 1/R)),

where c is the value of T for identical laws (0 for a distance, 0.5 for a
detection accuracy), q the confidence level, and kappa = sqrt((1/a + 1/b) /
(1/a' + 1/b')) rescales from the split sizes (a', b') to the compared sizes
(Prop. 106 in docs/proofs/sec_v20.tex).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..evaluate import keystr
from .multi import relations


def descendants(spec, t):
    """Tables below t through simple relationships, parents before children:
    [(child, fk, parent, pk)]."""
    rel = relations(spec)
    out, frontier, seen = [], [t], {t}
    while frontier:
        nxt = []
        for p in frontier:
            for _n, pp, pk, c, fk in rel:
                if pp == p and c not in seen:
                    out.append((c, fk, p, pk))
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
    return out


def anchor_chain(spec, tables, t, min_rows=20):
    """The path from t up to the entity its rows cluster under: at each step the
    parent named by the fact's anchor, else the parent with the most rows;
    copied tables and tables under min_rows rows (lookups) end the path.
    [(child, fk, parent, pk)], empty when t is a root."""
    rel = relations(spec)
    T = spec["tables"]
    out, cur, seen = [], t, {t}
    while True:
        ups = [(c, fk, p, pk) for _n, p, pk, c, fk in rel if c == cur and p not in seen
               and T[p].get("mode") != "copy" and len(tables.get(p, ())) >= min_rows]
        if not ups:
            return out
        anchor = (T[cur].get("fact") or {}).get("anchor")
        pick = next((u for u in ups if u[2] == anchor), None) or max(ups, key=lambda u: len(tables[u[2]]))
        out.append(pick)
        seen.add(pick[2])
        cur = pick[2]


def cluster_ids(spec, tables, t):
    """For each row of t, the key of the root row it hangs from (its own row
    number when t is a root): rows of one cluster are split together, because
    rows that share an ancestor are not independent (Prop. 106, condition (i))."""
    chain = anchor_chain(spec, tables, t)
    if not chain:
        return np.arange(len(tables[t])).astype(str)
    ids = keystr(tables[t][chain[0][1]])
    for (_c0, _f0, _p0, own_pk), (c, fk, _p, _pk) in zip(chain, chain[1:]):
        # c is the parent of the previous step; own_pk is the key the previous step points at
        up = pd.Series(keystr(tables[c][fk]).to_numpy(), index=keystr(tables[c][own_pk]).to_numpy())
        up = up[~up.index.duplicated()]
        ids = ids.map(up)
    return ids.fillna("__null__").astype(str).to_numpy()


def cluster_halves(ids, rng):
    """Boolean mask of half A: whole clusters assigned at random to the halves."""
    u, inv = np.unique(ids, return_inverse=True)
    pick = np.zeros(len(u), dtype=bool)
    pick[rng.permutation(len(u))[: len(u) // 2]] = True
    return pick[inv]


def split_db_root(spec, tables, t, rng):
    """split_db at the root of t's anchor chain, so that whole clusters move."""
    chain = anchor_chain(spec, tables, t)
    return split_db(spec, tables, chain[-1][2] if chain else t, rng)


def split_db(spec, tables, t, rng, follow=True):
    """Two half-databases: the rows of t split at random into halves, every
    table below t (with follow) keeps the rows whose parent is in the half;
    other tables are shared by both halves."""
    n = len(tables[t])
    perm = rng.permutation(n)
    halves = []
    for idx in (np.sort(perm[: n // 2]), np.sort(perm[n // 2:])):
        d = dict(tables)
        d[t] = tables[t].iloc[idx].reset_index(drop=True)
        for c, fk, p, pk in (descendants(spec, t) if follow else []):
            keep = keystr(tables[c][fk]).isin(set(keystr(d[p][pk]).dropna())).to_numpy()
            d[c] = tables[c][keep].reset_index(drop=True)
        halves.append(d)
    return halves


def kappa(sizes, split_sizes):
    """Scale factor from the null law at split sizes to the compared sizes."""
    (a, b), (a2, b2) = sizes, split_sizes
    v = 1 / max(a, 1) + 1 / max(b, 1)
    v2 = 1 / max(a2, 1) + 1 / max(b2, 1)
    return float(np.sqrt(v / v2))


def prediction_limit(draws, center=0.0, confidence=0.99, k=1.0, log=False):
    """One-sided normal prediction bound for one more draw, rescaled by k.
    log=True (a distance, center 0): the bound is computed on log(T), where the
    right-skewed law of a distance is close to normal, then exponentiated."""
    x = np.asarray([d for d in draws if d is not None and np.isfinite(d)], dtype=float)
    if len(x) < 3:
        return None
    R = len(x)
    q = float(student_t.ppf(confidence, R - 1)) * np.sqrt(1 + 1 / R)
    if log and center == 0 and (x > 0).all():
        lx = np.log(x)
        return float(k * np.exp(lx.mean() + q * lx.std(ddof=1)))
    return float(center + k * ((x - center).mean() + q * x.std(ddof=1)))


def exceed_prob(draws, limit, center=0.0, k=1.0, log=False):
    """Level of a limit under the same predictive law as prediction_limit."""
    x = np.asarray([d for d in draws if d is not None and np.isfinite(d)], dtype=float)
    R = len(x)
    if R < 3:
        return None
    sc = np.sqrt(1 + 1 / R)
    if log and center == 0 and (x > 0).all():
        lx = np.log(x)
        if limit <= 0:
            return 1.0
        sd = lx.std(ddof=1) * sc
        if not sd > 0:
            return float(limit < k * np.exp(lx.mean()))
        return float(student_t.sf((np.log(limit / k) - lx.mean()) / sd, R - 1))
    sd = k * x.std(ddof=1) * sc
    if not sd > 0:
        return None
    return float(student_t.sf((limit - center - k * (x - center).mean()) / sd, R - 1))


def null_summary(draws, center, confidence, k, log=False):
    x = [float(d) for d in draws if d is not None and np.isfinite(d)]
    return {"method": "real-vs-real splits" + (" (log scale)" if log and center == 0 and x and min(x) > 0 else ""),
            "splits": len(x),
            "mean": float(np.mean(x)) if x else None, "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "center": center, "kappa": round(k, 4), "confidence": confidence,
            "limit": prediction_limit(x, center, confidence, k, log)}

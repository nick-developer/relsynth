# SPDX-License-Identifier: Apache-2.0
"""Column encodings shared by the quality metrics.

Every encoding takes the real column as its reference, so that real and
synthetic values are mapped by the same function:

* rank-normal scores: z = Phi^-1(F_real(x)) with the real mid-ECDF on n + 1
  points (the normal scores of docs/proofs, Section v1.0), so that a synthetic
  value outside the real range maps to the most extreme real score;
* discrete codes: a categorical keeps its `max_levels` most frequent real
  spellings (the rest pooled), a numeric column its real quantile bins, and a
  null is a code of its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..evaluate import _canon, _is_cat, _num
from ..marginals import MODELLED

OTHER = "__other__"
NULL = "__null__"


def attribute_columns(spec, t):
    """The modelled, non-key columns of table t (the columns the 1.0 evaluator scores)."""
    ts = spec["tables"][t]
    keys = {c for r in spec["relationships"] if r["child"] == t
            for c in (r["child_key"] if r.get("composite") else [r["child_key"]])}
    return [c for c, cs in ts["columns"].items()
            if cs["type"] in MODELLED and c not in keys and c != ts.get("primary_key")]


def is_categorical(spec, t, c, real_col):
    cs = spec["tables"][t]["columns"][c]
    if "_categorical" in cs:                 # frozen on the whole real table (freeze_kinds)
        return cs["_categorical"]
    return _is_cat(cs, real_col)


def freeze_kinds(spec, real):
    """A copy of spec whose columns record whether they are categorical, decided
    once on the whole real table.  The rule for integers depends on the data
    (at most 30 distinct values), so a real half or a sample could otherwise
    treat a column differently from the full table."""
    out = dict(spec)
    out["tables"] = {}
    for t, ts in spec["tables"].items():
        ts = dict(ts)
        cols = {}
        for c, cs in ts.get("columns", {}).items():
            cs = dict(cs)
            if t in real and c in real[t]:
                cs["_categorical"] = bool(_is_cat(cs, real[t][c]))
            cols[c] = cs
        ts["columns"] = cols
        out["tables"][t] = ts
    return out


def _is_null(x):
    return x is None or x is pd.NA or x is pd.NaT or (isinstance(x, float) and np.isnan(x))


def spellings(s):
    """One text spelling per value (numbers by value), None for null.  Each
    distinct value is spelled once (the hot path of the k-hop trends)."""
    v = s.to_numpy(dtype=object)
    if not len(v):
        return np.array([], dtype=object)
    codes, uniq = pd.factorize(v, use_na_sentinel=True)
    is_bool = np.fromiter((type(x) in (bool, np.bool_) for x in v), bool, len(v))
    if is_bool.any() and (~is_bool & (codes >= 0)).any():
        # True == 1 under hashing, so factorize would merge them: spell value by value
        return np.array([None if _is_null(x) else _canon(x) for x in v], dtype=object)
    spelled = np.array([None if _is_null(x) else _canon(x) for x in uniq] + [None], dtype=object)
    return spelled[codes]                                   # -1 (missing) picks the trailing None


def rank_normal(ref, x):
    """Normal scores of x under the mid-ECDF of the non-null reference values;
    NaN where x is NaN.  ref and x are float arrays."""
    r = np.sort(ref[~np.isnan(ref)])
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    if len(r) == 0 or not ok.any():
        return out
    lo = np.searchsorted(r, x[ok], side="left")
    hi = np.searchsorted(r, x[ok], side="right")
    out[ok] = ndtri((0.5 * (lo + hi) + 0.5) / (len(r) + 1))
    return out


def codes(real, other, categorical, max_levels=20, bins=5):
    """Discrete codes of two columns on the real column's levels or quantile
    bins.  Returns (code_real, code_other, k) with codes in 0..k-1."""
    if categorical:
        a, b = spellings(real), spellings(other)
        vc = pd.Series(a[a != None]).value_counts()  # noqa: E711
        top = list(vc.index[:max_levels])
        lut = {v: i for i, v in enumerate(top)}
        k_other, k_null = len(top), len(top) + 1

        def enc(v):
            sv = pd.Series(v, dtype=object)
            c = sv.map(lut).fillna(k_other).to_numpy(dtype=np.int64, copy=True)
            c[sv.isna().to_numpy()] = k_null
            return c
        return enc(a), enc(b), len(top) + 2
    xa, xb = _num(real), _num(other)
    ok = xa[~np.isnan(xa)]
    edges = np.unique(np.quantile(ok, np.linspace(0, 1, bins + 1)[1:-1])) if len(ok) else np.array([])
    k_null = len(edges) + 1

    def enc(x):
        c = np.searchsorted(edges, x, side="right").astype(np.int64)
        c[np.isnan(x)] = k_null
        return c
    return enc(xa), enc(xb), len(edges) + 2


def feature_matrix(real, other, spec, t, cols, max_levels=20, quadratic=True):
    """Design matrices (real, other) on the real column's encoding: rank-normal
    score (and its square), a null indicator, and one-hot real levels scaled by
    1/sqrt(2) so that two different levels are at distance 1."""
    fr, fo = [], []
    for c in cols:
        if is_categorical(spec, t, c, real[c]):
            a, b, k = codes(real[c], other[c], True, max_levels=max_levels)
            # every code gets a column, so that the layout depends on the real column only
            fr.append(np.eye(k)[a] / np.sqrt(2))
            fo.append(np.eye(k)[b] / np.sqrt(2))
        else:
            xa, xb = _num(real[c]), _num(other[c])
            za, zb = rank_normal(xa, xa), rank_normal(xa, xb)
            na, nb = np.isnan(za), np.isnan(zb)
            za, zb = np.nan_to_num(za), np.nan_to_num(zb)
            parts_a, parts_b = [za], [zb]
            if quadratic:
                parts_a.append(za ** 2 - 1)
                parts_b.append(zb ** 2 - 1)
            parts_a.append(na.astype(float))
            parts_b.append(nb.astype(float))
            fr.append(np.column_stack(parts_a))
            fo.append(np.column_stack(parts_b))
    if not fr:
        return np.zeros((len(real), 0)), np.zeros((len(other), 0))
    return np.column_stack(fr), np.column_stack(fo)

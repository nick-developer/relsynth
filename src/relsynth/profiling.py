# SPDX-License-Identifier: Apache-2.0
"""Dependency discovery for introspect (data profiling).

Identifying a data model from a database is the discovery of its dependencies
(Abedjan, Golab & Naumann, "Profiling relational data: a survey", VLDB J. 2015):

* unique column combinations (keys, grains),
* inclusion dependencies (foreign keys), also conditional ones,
* functional dependencies, exact or approximate (g3 error, Kivinen & Mannila
  1995): order headers in a fact, values looked up from a parent,
* algebraic constraints between numeric columns (Brown & Haas, BHUNT, VLDB
  2003): amount = price x quantity, tax = 8% of sales,
* order dependencies within groups: line numbers 1..n, running balances, the
  gap between two dates.

Every test here is made on a uniform random sample (or on complete groups, for
dependencies within groups) and is accepted only when (a) it holds on at least
`MIN_SHARE` of the rows that can show it and (b) that agreement is beyond chance:
with c the share expected if the columns were unrelated (the collision
probability sum_v p(v) q(v) for an equality, the agreement within groups for a
functional dependency, the agreement with permuted inputs for an algebraic
rule), a one-sided binomial test of the observed agreement against c must have
p <= 1e-9.  Condition (b) separates structure from degenerate columns: a flag
that is 0 on 99.95% of rows agrees with any other such flag on about 99.9% of
rows by chance, and is rejected; a 4-level category that matches its parent's on
all of 2,700 rows is accepted although c = 0.6, since 0.6^2700 is negligible.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

MIN_SHARE = 0.999
MIN_ROWS = 30
ALPHA = 1e-9


def significant(share, chance, n, alpha=ALPHA) -> bool:
    """Is agreement on `share` of n trials beyond chance?  One-sided binomial
    test: P(X >= share n | X ~ Bin(n, chance)) <= alpha (1e-9: with the hundreds
    of candidates tested per table, a false dependency stays improbable)."""
    from scipy.stats import binom
    if n <= 0:
        return False
    k = int(round(share * n))
    if chance >= 1.0:
        return False
    return float(binom.logsf(k - 1, n, max(chance, 1e-12))) <= np.log(alpha)


# ---------------------------------------------------------------- values
def is_dateish(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    v = s.dropna()
    if not len(v):
        return False
    head = v.head(200)
    import datetime as _dt
    if head.map(lambda x: isinstance(x, (_dt.date, _dt.datetime, pd.Timestamp))).mean() > 0.95:
        return True
    if head.map(lambda x: isinstance(x, str)).mean() > 0.95 and head.astype(str).str.len().median() >= 8 and \
            head.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.95:
        return pd.to_datetime(head, errors="coerce", format="mixed").notna().mean() > 0.95
    return False


def canon(s, as_date=None) -> np.ndarray:
    """Comparable values: dates as day numbers (with fraction), numbers as rounded
    floats, everything else as strings; None for nulls."""
    if isinstance(s, pd.Series) and isinstance(s.dtype, np.dtype) and s.dtype.kind == "M" and as_date is not False:
        # naive datetime64 column: the object round trip below gives back the same
        # values in the same unit, so it is skipped (identical result)
        d = s.reset_index(drop=True)
        x = ((d - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        out = np.empty(len(d), dtype=object)
        ok = np.isfinite(x)
        out[ok] = np.round(x[ok], 6)
        out[~ok] = None
        return out
    if isinstance(s, pd.Series) and isinstance(s.dtype, np.dtype) and s.dtype.kind in "iuf" and as_date is not True:
        # plain numeric column: every non-null value is a number (identical result)
        v = s.to_numpy(dtype=float)
        out = np.empty(len(v), dtype=object)
        ok = ~np.isnan(v)
        out[ok] = np.round(v[ok], 9)
        out[~ok] = None
        return out
    s = pd.Series(np.asarray(s, dtype=object)).reset_index(drop=True)
    if as_date is None:
        as_date = is_dateish(s)
    out = np.empty(len(s), dtype=object)
    na = s.isna().to_numpy()
    if as_date:
        d = pd.to_datetime(s, errors="coerce", format="mixed")
        x = ((d - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        ok = np.isfinite(x)
        out[ok] = np.round(x[ok], 6)
        out[~ok] = None
        return out
    num = pd.to_numeric(s, errors="coerce")
    isnum = num.notna().to_numpy() & ~na
    if isnum.sum() >= 0.98 * max((~na).sum(), 1):
        out[isnum] = np.round(num.to_numpy(dtype=float)[isnum], 9)
        out[~isnum] = [None if n_ else str(v) for n_, v in zip(na[~isnum], s.to_numpy(dtype=object)[~isnum])]
        return out
    out[:] = [None if n_ else str(v) for n_, v in zip(na, s.to_numpy(dtype=object))]
    return out


def _codes(a):
    """Integer codes of canonical values (None is its own level)."""
    return pd.factorize(pd.Series(a, dtype=object), use_na_sentinel=False)[0]


def equality(a, b):
    """(share of equal rows, chance share) for two aligned canonical arrays, nulls equal."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    n = len(a)
    if n == 0:
        return 0.0, 1.0
    same = np.asarray(a == b, dtype=bool)
    pa = pd.Series(a, dtype=object).fillna("\x00").value_counts(normalize=True)
    pb = pd.Series(b, dtype=object).fillna("\x00").value_counts(normalize=True)
    chance = float((pa * pb.reindex(pa.index).fillna(0)).sum())
    return float(same.mean()), chance


# ---------------------------------------------------------------- FDs
def fd_profile(key, col):
    """Functional dependency key -> col on the groups with at least two rows:
    (share of such groups where col is constant, share expected by chance, number
    of such groups).  Chance: a group of m rows agrees with probability
    sum_v p_v^m when values are drawn independently from col's distribution."""
    k = _codes(np.asarray(key, dtype=object))
    v = _codes(np.asarray(col, dtype=object))
    df = pd.DataFrame({"k": k, "v": v})
    g = df.groupby("k")["v"]
    size = g.size()
    nun = g.nunique()
    multi = size >= 2
    nm = int(multi.sum())
    if nm == 0:
        return 0.0, 1.0, 0
    agree = float((nun[multi] == 1).mean())
    p = np.bincount(v) / len(v)
    ms = size[multi].to_numpy()
    um, cnt = np.unique(ms, return_counts=True)
    chance = float(sum(c * np.sum(p ** m) for m, c in zip(um, cnt)) / nm)
    return agree, chance, nm


def sequence_share(key, vals, start=1):
    """(share of groups whose values are exactly start..start+m-1 in any order,
    number of groups with m >= 2).  Groups of one row only say 'the value is
    start', which a constant column also satisfies: the evidence for a line number
    is in the larger groups, so callers require enough of them."""
    df = pd.DataFrame({"k": _codes(np.asarray(key, dtype=object)),
                       "v": pd.to_numeric(pd.Series(np.asarray(vals, dtype=object)), errors="coerce")})
    if df.v.isna().any():
        return 0.0, 0
    g = df.groupby("k")["v"]
    n = g.size()
    ok = (g.min() == start) & (g.max() == n + start - 1) & (g.nunique() == n)
    return float(ok.mean()), int((n >= 2).sum())


# ---------------------------------------------------------------- numbers
def decimals(v) -> int | None:
    """Stored precision: the fewest decimals (<= 6) that represent every value."""
    x = pd.to_numeric(pd.Series(np.asarray(v, dtype=object)), errors="coerce").dropna().to_numpy(dtype=float)[:20000]
    if not len(x):
        return None
    return next((d for d in range(7) if np.allclose(x, np.round(x, d), atol=1e-9, rtol=0)), None)


def nice_constant(k, x, y, tol):
    """The constant with the fewest significant digits (<= 6) such that y = k x within tol."""
    if not np.isfinite(k) or k == 0:
        return None
    for sig in range(1, 7):
        kk = float(f"{k:.{sig}g}")
        if np.mean(np.abs(y - kk * x) <= tol) >= MIN_SHARE:
            return kk
    return None


def _share(y, f, tol):
    m = np.isfinite(y) & np.isfinite(f)
    if m.sum() < MIN_ROWS:
        return 0.0, int(m.sum())
    return float(np.mean(np.abs(y[m] - f[m]) <= (tol[m] if np.ndim(tol) else tol))), int(m.sum())


def algebraic_rules(df: pd.DataFrame, cols: list[str], dec: dict, rng=None, screen=2000):
    """Candidate rules lhs = expression over the numeric columns `cols`:
    a copy (y = a), a scaling (y = k a, k with few significant digits), a sum, a
    difference or a product of two columns.  A rule holds when |y - f| <= tol on
    at least MIN_SHARE of the rows where every term is known, tol being one unit
    of y's stored precision (10^-d for d decimals; 1e-9 relative otherwise), and
    is kept only if its agreement is significantly above the agreement with the
    inputs permuted (see `significant`), which rules out identities that hold
    because a column is (almost) constant.  Returns [(lhs, expr, inputs, complexity, tolerance)]."""
    rng = rng or np.random.default_rng(0)
    X = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols}
    cols = [c for c in cols if np.isfinite(X[c]).sum() >= MIN_ROWS and len(np.unique(X[c][np.isfinite(X[c])])) >= 3]
    n = len(df)
    idx_s = np.arange(n) if n <= screen else rng.choice(n, screen, replace=False)
    perm = rng.permutation(n)
    out = []

    def tol_of(c, y):
        d = dec.get(c)
        return (10.0 ** -d if d else 0.0) + 1e-9 * np.abs(y) + 1e-9

    def holds(y, f, tol, chance_f):
        if _share(y[idx_s], f[idx_s], tol[idx_s] if np.ndim(tol) else tol)[0] < MIN_SHARE:
            return False
        share, n_ = _share(y, f, tol)
        if share < MIN_SHARE:
            return False
        chance = _share(y, chance_f, tol)[0]
        return significant(share, chance, n_)

    # two-column rules: each pair's four expressions are computed once on the
    # screening rows and screened against every target at once (the same test as
    # holds() on those rows), then confirmed one by one on all rows
    tols = {c: tol_of(c, X[c]) for c in cols}
    pairs = {}
    if len(cols) >= 3:
        Ys = np.stack([X[c][idx_s] for c in cols], 1)
        Ts = np.stack([tols[c][idx_s] for c in cols], 1)
        fin_y = np.isfinite(Ys)
        for a, b in itertools.combinations(cols, 2):
            xa, xb = X[a], X[b]
            for expr, op in ((f"{a} + {b}", np.add), (f"{a} - {b}", np.subtract), (f"{b} - {a}", None),
                             (f"{a} * {b}", np.multiply)):
                fs = (xb[idx_s] - xa[idx_s]) if op is None else op(xa[idx_s], xb[idx_s])
                ok = fin_y & np.isfinite(fs)[:, None]
                cnt = ok.sum(0)
                hit = ((np.abs(Ys - fs[:, None]) <= Ts) & ok).sum(0)
                pass_ = (cnt >= MIN_ROWS) & (hit / np.maximum(cnt, 1) >= MIN_SHARE)
                for j in np.flatnonzero(pass_):
                    c = cols[j]
                    if c in (a, b):
                        continue
                    f = (xb - xa) if op is None else op(xa, xb)
                    fp = (xb - xa[perm]) if op is None else op(xa, xb[perm])
                    if holds(X[c], f, tols[c], fp):
                        pairs.setdefault(c, []).append((c, expr, [a, b], 2,
                                                        float(10.0 ** -dec[c]) if dec.get(c) else 0.0))
    for c in cols:
        y = X[c]
        tol = tols[c]
        others = [a for a in cols if a != c]
        for a in others:
            x = X[a]
            if holds(y, x, tol, x[perm]):
                out.append((c, a, [a], 0, float(10.0 ** -dec[c]) if dec.get(c) else 0.0))
                continue
            m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-12)
            if m.sum() >= MIN_ROWS:
                k = nice_constant(float(np.median(y[m] / x[m])), x[m], y[m], tol[m])
                if k is not None and k not in (0.0, 1.0) and holds(y, k * x, tol, k * x[perm]):
                    out.append((c, f"{k:g} * {a}", [a], 1, float(10.0 ** -dec[c]) if dec.get(c) else 0.0))
        out += pairs.get(c, [])
    return out


def choose_rules(cands, basic=()):
    """A set of derive rules without cycles that defines as many columns as
    possible: starting from the columns no rule defines (and `basic` columns),
    repeatedly accept the simplest rule whose inputs are all available; if every
    remaining rule waits on another, the column with most candidate definitions is
    taken as given (it is modelled) and the search continues."""
    rules = {}
    lhs_all = {c for c, *_ in cands}
    avail = set(basic) | {i for _, _, ins, _, _ in cands for i in ins if i not in lhs_all}
    left = [x for x in cands if x[0] not in avail]
    while left:
        ready = [x for x in left if all(i in avail for i in x[2])]
        if ready:
            best = min(ready, key=lambda x: (x[3], len(x[2]), x[1]))
            rules[best[0]] = best
            avail.add(best[0])
            left = [x for x in left if x[0] != best[0]]
            continue
        cnt = pd.Series([x[0] for x in left]).value_counts()
        given = cnt.index[0]
        avail.add(given)
        left = [x for x in left if x[0] != given]
    return list(rules.values())


def cumulative_rule(df, group, order, cols):
    """y_t = y_{t-1} + step_t within each group ordered by `order`, step being a
    column, its negative, or the difference of two columns.  Tested on complete
    groups (cluster sample).  Returns (y, step) or None for each y."""
    out = []
    if group not in df or order not in df:
        return out
    d = df.copy()
    d["_o"] = canon(d[order])
    d = d.dropna(subset=["_o"])
    d = d.sort_values([group, "_o"], kind="mergesort")
    X = {c: pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float) for c in cols}
    first = (d[group] != d[group].shift()).to_numpy()
    for y in cols:
        yy = X[y]
        diff = yy - np.r_[np.nan, yy[:-1]]
        diff[first] = np.nan
        if np.isfinite(diff).sum() < MIN_ROWS or np.nanstd(diff) == 0:
            continue
        others = [c for c in cols if c != y]
        cands = [(a, X[a]) for a in others] + [(f"0 - {a}", -X[a]) for a in others] + \
                [(f"{a} - {b}", X[a] - X[b]) for a in others for b in others if a != b]
        for name, s in cands:
            m = np.isfinite(diff) & np.isfinite(s)
            share = float(np.mean(np.abs(diff[m] - s[m]) <= 1e-6 + 1e-9 * np.abs(s[m]))) if m.sum() else 0.0
            if m.sum() >= MIN_ROWS and share >= MIN_SHARE:
                rp = np.random.default_rng(0).permutation(len(s))
                m2 = m & np.isfinite(s[rp])
                if significant(share, float(np.mean(np.abs(diff[m2] - s[rp][m2]) <= 1e-6)), int(m.sum())):
                    out.append((y, name))
                    break
    return out

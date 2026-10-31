# SPDX-License-Identifier: Apache-2.0
"""Structure-aware fidelity, integrity and privacy evaluation.

Works for any spec: the checks that run are decided by the structures present
(columns, many-to-one links, facts, trees, time columns, text columns).
The report contains aggregates only, never rows.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr, ndtri
from scipy.stats import ks_2samp, rankdata, spearmanr

from .marginals import MODELLED
from .spec import inherit_scale

THRESHOLDS = {
    "ks": 0.08, "tvd": 0.08, "null_diff": 0.03, "assoc_max": 0.15, "auc": 0.65,
    "card_ks": 0.12, "card_mean_rel": 0.10, "join_tvd": 0.10, "fk_integrity": 1.0,
    "tree_depth_tvd": 0.12, "fanout_ks": 0.12, "time_tvd": 0.10, "tstr_ratio": 0.85,
    "text_rare_leaks": 0, "dcr_ratio": 0.5, "exact_match": 0.01, "rule_violation": 0.0,
    "pair_repeat": 0.01, "novel_combination": 0.005, "rows_removed": 0.01,
}


def _num(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 86400e9
        out[s.isna().to_numpy()] = np.nan
        return out
    if s.dtype == bool or str(s.dtype) == "boolean":
        return s.astype(float).to_numpy()
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _is_cat(cs, s=None):
    return cs["type"] in ("categorical", "boolean") or (
        cs["type"] == "integer" and s is not None and s.nunique() <= 30)


def _canon(v):
    """One spelling per value: numbers by value (10000, 10000.0 and Decimal('10000.00')
    are the same level), everything else as text."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return "<null>"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)) or type(v).__name__ == "Decimal":
        f = float(v)
        if np.isfinite(f):
            return str(int(f)) if f == int(f) and abs(f) < 2 ** 53 else repr(round(f, 12))
    return str(v)


def _canon_counts(a):
    """Series(a).map(_canon).value_counts(normalize=True), with _canon applied once
    per distinct value instead of once per row.  Values are grouped by equality
    first (1, 1.0 and Decimal('1'), which _canon spells the same), counted, and the
    counts summed per spelling; the result is then built exactly as value_counts
    builds it (labels in order of first appearance, sorted by count with the
    same sort), so it is identical, index order included."""
    arr = a.to_numpy(dtype=object) if isinstance(a, (pd.Series, pd.Index)) else np.asarray(list(a), dtype=object)
    if len(arr) == 0:
        return pd.Series(arr, dtype=object).map(_canon).value_counts(normalize=True)
    kind = pd.api.types.infer_dtype(arr, skipna=True)
    if kind not in ("string", "integer", "floating", "mixed-integer-float", "decimal", "datetime", "datetime64",
                    "date", "empty", "boolean") and any(isinstance(v, (bool, np.bool_)) for v in arr):
        # True == 1 under hashing, but _canon spells them differently: per row
        return pd.Series(arr, dtype=object).map(_canon).value_counts(normalize=True)
    codes, uniq = pd.factorize(arr, use_na_sentinel=True)
    first = {}
    cnt = {}
    if len(uniq):
        _, pos = np.unique(codes[codes >= 0], return_index=True)
        rows = np.flatnonzero(codes >= 0)[pos]
        k = np.bincount(codes[codes >= 0], minlength=len(uniq))
        for u, r, c in zip(uniq, rows, k):
            lab = _canon(u)
            first[lab] = min(first.get(lab, r), r)
            cnt[lab] = cnt.get(lab, 0) + int(c)
    for r in np.flatnonzero(codes < 0):
        lab = _canon(arr[r])
        first.setdefault(lab, r)
        first[lab] = min(first[lab], r)
        cnt[lab] = cnt.get(lab, 0) + 1
    labels = sorted(first, key=first.get)
    counts = np.array([cnt[l] for l in labels], dtype=np.int64)
    out = pd.Series(counts, index=pd.Index(labels, dtype=object)).sort_values(ascending=False)
    return out / counts.sum()


def tvd(a, b):
    pa = _canon_counts(a)
    pb = _canon_counts(b)
    idx = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())


def ks(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def _eff(n, m):
    """Variance factor of a real-minus-synthetic difference when the synthetic
    sample (m rows) is a posterior predictive draw given the n real rows:
    Var = v * (1/(n+1) + 1/m).  The real sample's own noise is shared by the fit
    and cancels in the difference (Bayesian bootstrap: Var(F_post - F_n | data) =
    F_n(1-F_n)/(n+1)); the same factor holds when the synthetic rows are an
    independent draw from the law of the real ones (1/n + 1/m)."""
    return 1.0 / (max(n, 1) + 1) + 1.0 / max(m, 1)


def _sidak_z(alpha, k, two_sided=True):
    """z such that the largest of k independent |N(0,1)| (or N(0,1)) exceeds it
    with probability alpha (Sidak 1967)."""
    a = 1 - (1 - alpha) ** (1 / max(int(k), 1))
    return float(ndtri(1 - a / 2)) if two_sided else float(ndtri(1 - a))


Z25_ALPHA = float(2 * ndtr(-2.5))      # two-sided level of a 2.5-sd limit (0.0124)


def ks_crit(n, m, alpha=0.01, pp=False):
    """Two-sample KS critical value c(alpha) * sqrt(eff).  pp=True uses the
    calibrated factor _eff(n, m) = 1/(n+1) + 1/m (the scorecard's null); the
    default keeps the older, conservative 2/n + 1/m used by callers outside the
    scorecard."""
    v = _eff(n, m) if pp else 2 / max(n, 1) + 1 / max(m, 1)
    return float(np.sqrt(-np.log(alpha / 2) / 2) * np.sqrt(v))


def _ks_alpha(lim, n, m):
    """Asymptotic two-sample KS level of a limit (Smirnov): 2 exp(-2 lim^2 / eff)."""
    if lim is None or not np.isfinite(lim):
        return None
    return float(min(1.0, 2 * np.exp(-2 * lim * lim / _eff(n, m))))


def tvd_crit(p, n, m, z=2.5, pp=False):
    """Expected TVD between two multinomial samples (normal approximation),
    times z; pp=True uses the calibrated variance factor _eff (see ks_crit)."""
    p = np.asarray(p, dtype=float)
    v = _eff(n, m) if pp else 2 / max(n, 1) + 1 / max(m, 1)
    return float(z * 0.5 * np.sum(np.sqrt(2 * p * (1 - p) / np.pi * v)))


def tvd_null(p, n, m):
    """Mean and sd of the TVD under the null (normal approximation per level,
    levels treated as independent): used only to report a check's level."""
    p = np.asarray(p, dtype=float)
    s = np.sqrt(p * (1 - p) * _eff(n, m))
    return float(0.5 * np.sqrt(2 / np.pi) * s.sum()), float(0.5 * np.sqrt(((1 - 2 / np.pi) * s * s).sum()))


def _z_alpha(lim, sd, two_sided=True):
    if lim is None or sd is None or not np.isfinite(lim) or not sd > 0:
        return None
    return float((2 if two_sided else 1) * ndtr(-lim / sd))


# ---------------------------------------------------------------- columns
def column_metrics(real, syn, tspec, cols):
    out = {}
    for c in cols:
        cs = tspec["columns"][c]
        r, s = real[c], syn[c]
        d = {"type": cs["type"], "null_diff": float(abs(r.isna().mean() - s.isna().mean()))}
        # pooled null share (+1/2 continuity): a real sample without nulls must not give
        # a zero limit when the law has a small null share
        xr, xs = int(r.isna().sum()), int(s.isna().sum())
        pn = (xr + xs + 0.5) / (len(r) + len(s) + 1)
        d["null_sd"] = float(np.sqrt(pn * (1 - pn) * _eff(len(r), len(s))))
        d["null_crit"] = 2.5 * d["null_sd"]
        d["n"], d["m"] = int(r.notna().sum()), int(s.notna().sum())
        if _is_cat(cs, r):
            d["tvd"] = tvd(r.dropna(), s.dropna())
            pr = r.dropna().astype(str).value_counts(normalize=True).to_numpy()
            d["tvd_crit"] = tvd_crit(pr, r.notna().sum(), s.notna().sum(), pp=True)
            d["tvd_mu"], d["tvd_sd"] = tvd_null(pr, r.notna().sum(), s.notna().sum())
            d["levels_real"] = int(r.nunique())
            d["levels_unseen_in_syn"] = int(len(set(r.dropna().astype(str)) - set(s.dropna().astype(str))))
        else:
            a, b = _num(r), _num(s)
            d["ks"] = ks(a, b)
            d["n"], d["m"] = int((~np.isnan(a)).sum()), int((~np.isnan(b)).sum())
            d["ks_crit_1pct"] = ks_crit(max(d["n"], 1), max(d["m"], 1), pp=True)
            m, sd = np.nanmean(a), np.nanstd(a)
            d["mean_shift_sd"] = float(abs(np.nanmean(b) - m) / sd) if sd > 0 else 0.0
            d["sd_ratio"] = float(np.nanstd(b) / sd) if sd > 0 else 1.0
            d["distinct_ratio_real"] = float(r.nunique() / max(r.notna().sum(), 1))
        out[c] = d
    return out


def _assoc(x, y, cx, cy):
    """Spearman for num-num, bias-corrected Cramer's V for cat-cat, correlation
    ratio for num-cat; all in [0,1] except Spearman in [-1,1]."""
    ok = pd.notna(x) & pd.notna(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return 0.0
    if not cx and not cy:
        r = spearmanr(_num(x), _num(y)).statistic
        return 0.0 if np.isnan(r) else float(r)
    if cx and cy:
        ct = pd.crosstab(x.astype(str), y.astype(str)).to_numpy().astype(float)
        n = ct.sum()
        if min(ct.shape) < 2:
            return 0.0
        e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
        chi2 = ((ct - e) ** 2 / np.maximum(e, 1e-12)).sum()
        r_, k_ = ct.shape
        phi2 = max(chi2 / n - (k_ - 1) * (r_ - 1) / (n - 1), 0)
        rc, kc = r_ - (r_ - 1) ** 2 / (n - 1), k_ - (k_ - 1) ** 2 / (n - 1)
        den = min(kc - 1, rc - 1)
        return float(np.sqrt(phi2 / den)) if den > 0 else 0.0
    if cx:
        x, y = y, x
    # correlation ratio on ranks (like Spearman for two numeric columns), so that a
    # single extreme value cannot dominate the measure
    v = pd.Series(_num(x)).rank().to_numpy()
    g = y.astype(str).to_numpy()
    df = pd.DataFrame({"v": v, "g": g})
    m = df.groupby("g").v.agg(["mean", "count"])
    tot = np.nanvar(v) * len(v)
    if tot <= 0:
        return 0.0
    return float(np.sqrt((m["count"] * (m["mean"] - np.nanmean(v)) ** 2).sum() / tot))


def _collapse_rare(r, s, cols, iscat, k):
    """Rows with a level seen in fewer than k real rows are left out of the pairs
    that involve that column, in both tables: dependence on so few rows is
    deliberately not modelled (it would single out those rows), so the comparison
    is made on the levels that carry estimable dependence."""
    if k <= 1:
        return r, s
    r, s = r.copy(), s.copy()
    for c in cols:
        if not iscat[c]:
            continue
        vc = r[c].dropna().astype(str).value_counts()
        rare = set(vc.index[vc < k])
        if not rare or len(rare) == len(vc):
            continue
        for d in (r, s):
            v = d[c].astype(object)
            d[c] = v.where(~v.astype(str).isin(rare) | v.isna(), None)   # left out of every pair
    return r, s


def _prep_col(d, c, iscat):
    """Per-column arrays for _assoc_fast: non-null mask and either the codes of
    the text spellings (categorical) or the numeric values."""
    s = d[c]
    ok = s.notna().to_numpy()
    if iscat:
        codes = np.full(len(s), -1, dtype=np.int64)
        if ok.any():
            codes[ok] = pd.factorize(s[ok].astype(str))[0]
        return ok, codes
    return ok, _num(s)


def _assoc_fast(A, B, cx, cy):
    """_assoc on prepared columns (same statistic, without per-pair pandas work):
    contingency tables by bincount, ranks by rankdata.  Returns None when the
    numeric side has values that _num could not convert (then _assoc is used)."""
    (oka, va), (okb, vb) = A, B
    ok = oka & okb
    if ok.sum() < 10:
        return 0.0
    if not cx and not cy:
        x, y = va[ok], vb[ok]
        if np.isnan(x).any() or np.isnan(y).any():
            return None
        r = spearmanr(x, y).statistic
        return 0.0 if np.isnan(r) else float(r)
    if cx and cy:
        a, b = va[ok], vb[ok]
        kb = int(b.max()) + 1
        ct = np.bincount(a * kb + b, minlength=(int(a.max()) + 1) * kb).reshape(-1, kb).astype(float)
        ct = ct[ct.sum(1) > 0][:, ct.sum(0) > 0]
        n = ct.sum()
        if min(ct.shape) < 2:
            return 0.0
        e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
        chi2 = ((ct - e) ** 2 / np.maximum(e, 1e-12)).sum()
        r_, k_ = ct.shape
        phi2 = max(chi2 / n - (k_ - 1) * (r_ - 1) / (n - 1), 0)
        rc, kc = r_ - (r_ - 1) ** 2 / (n - 1), k_ - (k_ - 1) ** 2 / (n - 1)
        den = min(kc - 1, rc - 1)
        return float(np.sqrt(phi2 / den)) if den > 0 else 0.0
    if cx:
        va, vb = vb, va
    x, g = va[ok], vb[ok]
    if np.isnan(x).any():
        return None
    v = rankdata(x)
    cnt = np.bincount(g)
    m = np.bincount(g, weights=v)[cnt > 0] / cnt[cnt > 0]
    tot = np.var(v) * len(v)
    if tot <= 0:
        return 0.0
    return float(np.sqrt((cnt[cnt > 0] * (m - v.mean()) ** 2).sum() / tot))


def assoc_metrics(real, syn, tspec, cols, max_rows=20000, rng=None, min_level_rows=0, floor=None):
    rng = rng or np.random.default_rng(0)
    floor = THRESHOLDS["assoc_max"] if floor is None else floor
    if len(cols) < 2:
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "worst_pair": None}
    r = real.sample(min(len(real), max_rows), random_state=1)
    s = syn.sample(min(len(syn), max_rows), random_state=1) if len(syn) else syn
    iscat = {c: _is_cat(tspec["columns"][c], real[c]) for c in cols}
    r, s = _collapse_rare(r, s, cols, iscat, min_level_rows)

    def neff(d, c):
        # rows that carry information on dependence: non-null rows, and for a
        # categorical only those away from its modal level (a flag set on 5 rows
        # has 5 informative rows, however long the table)
        x = d[c].dropna()
        if not iscat[c] or not len(x):
            return max(len(x), 1)
        return max(int(len(x) - x.astype(str).value_counts().iloc[0]), 1)
    ne_r = {c: neff(r, c) for c in cols}
    ne_s = {c: neff(s, c) for c in cols} if len(s) else {}
    pr = {c: _prep_col(r, c, iscat[c]) for c in cols}
    ps = {c: _prep_col(s, c, iscat[c]) for c in cols} if len(s) else {}
    # the limit of every pair is Sidak-adjusted for the number of pairs, so that the
    # family (any pair beyond its limit) keeps the level of one 2.5-sd limit
    K = len(cols) * (len(cols) - 1) // 2
    zk = _sidak_z(Z25_ALPHA, K)
    diffs = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ra = _assoc_fast(pr[a], pr[b], iscat[a], iscat[b])
            if ra is None:
                ra = _assoc(r[a], r[b], iscat[a], iscat[b])
            if len(s):
                sa = _assoc_fast(ps[a], ps[b], iscat[a], iscat[b])
                if sa is None:
                    sa = _assoc(s[a], s[b], iscat[a], iscat[b])
            else:
                sa = 0.0
            nr = min(ne_r[a], ne_r[b])
            ns = min(ne_s[a], ne_s[b]) if len(s) else 1
            diffs[f"{a}~{b}"] = (ra, sa, abs(ra - sa), zk * np.sqrt(_eff(nr, ns)))
    # the worst pair relative to its own limit (the floor included): the scorecard
    # passes iff every pair is within max(floor, its crit)
    worst = max(diffs.items(), key=lambda kv: kv[1][2] / max(kv[1][3], floor, 1e-9))
    return {"max_abs_diff": float(worst[1][2]), "mean_abs_diff": float(np.mean([d[2] for d in diffs.values()])),
            "worst_pair": worst[0], "worst_real": worst[1][0], "worst_syn": worst[1][1],
            "worst_crit": float(worst[1][3]), "k_pairs": K, "z": zk,
            "pairs": {k: [round(v[0], 3), round(v[1], 3)] for k, v in diffs.items()}}


# ---------------------------------------------------------------- discriminator
def _design(dfs, tspec, cols):
    allv = pd.concat(dfs, ignore_index=True)
    feats = []
    for c in cols:
        cs = tspec["columns"][c]
        if _is_cat(cs, allv[c]):
            top = allv[c].astype(str).value_counts().index[:20]
            v = allv[c].astype(str)
            feats.append(np.column_stack([(v == t).to_numpy(float) for t in top]))
        else:
            x = _num(allv[c])
            ok = ~np.isnan(x)
            sc = np.zeros(len(x))
            if ok.sum():
                sc[ok] = ndtri(pd.Series(x[ok]).rank().to_numpy() / (ok.sum() + 1))
            feats.append(np.column_stack([sc, sc ** 2, (~ok).astype(float)]))
        feats[-1] = feats[-1] if feats[-1].ndim == 2 else feats[-1][:, None]
    X = np.column_stack(feats + [np.ones(len(allv))])
    return X


def _logreg(X, y, lam=1.0):
    def f(w):
        z = X @ w
        p = expit(z)
        l = np.logaddexp(0, z) - y * z
        return l.sum() + lam * (w ** 2).sum(), X.T @ (p - y) + 2 * lam * w
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def _auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def discriminator_auc(real, syn, tspec, cols, max_rows=5000, seed=0):
    """2-fold cross-validated AUC of a quadratic logistic discriminator
    (0.5 = indistinguishable).  Propensity-score idea of Snoke et al. (pMSE).
    The folds are stratified (half of the real and half of the synthetic rows in
    each) and the AUC is computed within each fold, on one model's scores, then
    averaged: pooling the scores of two models fitted on unbalanced folds biases
    the AUC below 0.5 under the null (Parker et al. 2007; Airola et al. 2011)."""
    if not cols or len(syn) < 20:
        return float("nan")
    n = min(len(real), len(syn), max_rows)
    r = real.sample(n, random_state=seed)
    s = syn.sample(n, random_state=seed)
    X = _design([r[cols], s[cols]], tspec, cols)
    y = np.r_[np.zeros(n), np.ones(n)]
    rng = np.random.default_rng(seed)
    fold = np.r_[rng.permutation(np.arange(n) % 2), rng.permutation(np.arange(n) % 2)]
    aucs = []
    for f in (0, 1):
        w = _logreg(X[fold != f], y[fold != f])
        aucs.append(_auc(X[fold == f] @ w, y[fold == f]))
    return float(np.mean(aucs))


def auc_null_sd(n):
    """Null sd of the fold-averaged AUC with n real and n synthetic rows: that of
    one fold's Mann-Whitney AUC (n/2 + n/2 rows), sqrt((n+1)/(3 n^2)): within a
    fold the scores come from a model fitted on the other fold, so under the null
    they are exchangeable between the classes and the Mann-Whitney null law is
    exact; the sd of a mean of two such AUCs is at most the sd of one (Cauchy-
    Schwarz), so this is an upper bound (simulated 0.072, bound 0.076 at n = 59)."""
    n = max(int(n), 2)
    return float(np.sqrt((n + 1) / (3 * n * n)))


# ---------------------------------------------------------------- relations
def _grand_anchor(spec, f):
    """The entity whose rows a fact's rows hang from (through nested facts)."""
    T = spec["tables"]
    a = T[f]["fact"].get("anchor") or next(x["parent"] for x in spec["relationships"] if x["child"] == f)
    return _grand_anchor(spec, a) if T[a]["role"] == "fact" else a


def _ks_shape(rc, sc, ratio):
    """KS of per-parent counts when the synthetic side should have `ratio` times as
    many rows per parent.  The larger side is binomially thinned (seeded): thinning
    a mixed Poisson count by q gives the mixed Poisson law of rate q * rate, so the
    two sides then have the same law under the null and the integer atoms stay
    aligned (multiplying counts by the ratio misaligns them)."""
    if abs(ratio - 1) < 1e-9:
        return ks(rc, sc)
    g = np.random.default_rng(0)
    if ratio > 1:
        return ks(rc, g.binomial(sc.astype(np.int64), 1 / ratio).astype(float))
    return ks(g.binomial(rc.astype(np.int64), ratio).astype(float), sc)


def cardinality(real_child, syn_child, fk, real_parent, syn_parent, pk, expected_ratio=1.0, real_counts=None):
    """Per-parent row counts, real against synthetic.  real_counts (a frame with
    columns fk and n, from exact statistics) replaces counting real_child, which
    may then be a sample."""
    if real_counts is not None:
        vc = pd.Series(real_counts["n"].to_numpy(), index=keystr(real_counts[fk]).to_numpy())
        vc = vc[vc.index.notna()].groupby(level=0).sum()
    else:
        vc = keystr(real_child[fk]).value_counts()
    rc = vc.reindex(keystr(real_parent[pk]), fill_value=0).to_numpy(float)
    sc = keystr(syn_child[fk]).value_counts().reindex(keystr(syn_parent[pk]), fill_value=0).to_numpy(float)
    mr, ms = rc.mean() if len(rc) else 0, sc.mean() if len(sc) else 0
    if abs(expected_ratio - 1) <= 0.1 and mr < 20:
        # small integer counts: rescaling by a ratio near 1 would only misalign the
        # atoms (8 -> 7.4); compare the raw counts, the mean check covers the ratio
        expected_ratio_shape = 1.0
    else:
        expected_ratio_shape = expected_ratio
    return {"mean_real": float(mr), "mean_syn": float(ms), "parents_real": len(rc), "parents_syn": len(sc),
            "ks_crit": ks_crit(len(rc), len(sc), pp=True), "expected_ratio": float(expected_ratio),
            "mean_rel_err": float(abs(ms / max(mr * expected_ratio, 1e-12) - 1)) if mr > 0 else 0.0,
            # raw counts when per-parent cardinality should be preserved; otherwise
            # the larger side thinned to the other's expected rate (_ks_shape)
            "ks_shape": _ks_shape(rc, sc, expected_ratio_shape) if mr > 0 and ms > 0 else float("nan"),
            "zero_share_real": float((rc == 0).mean()), "zero_share_syn": float((sc == 0).mean()),
            "dispersion_real": float(rc.var() / max(mr, 1e-12)), "dispersion_syn": float(sc.var() / max(ms, 1e-12))}


def keystr(s):
    from .data import canonical_keys
    return canonical_keys(s).astype(str).where(s.notna(), None)


def fk_integrity(child, fk, parent, pk):
    v = child[fk].dropna()
    if not len(v):
        return 1.0
    return float(keystr(v).isin(set(keystr(parent[pk]))).mean())


def joined_view(spec, tables, fact, depth=3):
    """Denormalise a child table with the attributes of its ancestors (up to
    `depth` hops), prefixing columns with the path."""
    T = spec["tables"]
    df = tables[fact].copy()
    frontier = [(fact, "", df)]
    view = df
    for _ in range(depth):
        nxt = []
        for t, prefix, _d in frontier:
            for r in spec["relationships"]:
                if r["child"] != t or r["kind"] != "many_to_one" or r.get("composite"):
                    continue
                p = r["parent"]
                pk = T[p]["primary_key"]
                pcols = [c for c, cs in T[p]["columns"].items() if cs["type"] in MODELLED or c == pk
                         or any(rr["child"] == p and rr["child_key"] == c for rr in spec["relationships"])]
                pdf = tables[p][pcols].copy()
                newp = f"{prefix}{r['child_key']}."
                pdf.columns = [newp + c for c in pdf.columns]
                left = f"{prefix}{r['child_key']}" if prefix else r["child_key"]
                if left not in view.columns:
                    continue
                lk = keystr(view[left]).fillna("__null__")
                rk = keystr(pdf[newp + pk])
                view = view.assign(_k=lk.to_numpy()).merge(pdf.assign(_k=rk.to_numpy()), on="_k", how="left").drop(columns="_k")
                nxt.append((p, newp, None))
        frontier = nxt
        if not frontier:
            break
    return view


def _kish(view, keycol, default):
    """Kish effective sample size of the ancestor rows behind a joined column."""
    if keycol not in view.columns:
        return default
    col = view[keycol]
    if not col.isna().any() and (pd.api.types.is_integer_dtype(col.dtype) and not pd.api.types.is_bool_dtype(col.dtype)
                                 or pd.api.types.infer_dtype(col, skipna=False) == "string"):
        # integers or text without nulls: equal values <=> equal spellings, so the group
        # sizes are those of astype(str).value_counts(), in the same (descending) order
        w = -np.sort(-np.bincount(pd.factorize(col)[0])).astype(float)
    else:
        w = col.astype(str).value_counts().to_numpy().astype(float)
    return float(min(default, w.sum() ** 2 / max((w ** 2).sum(), 1.0))) if len(w) else default


def join_metrics(spec, real, syn, fact):
    rv, sv = joined_view(spec, real, fact), joined_view(spec, syn, fact)
    out, kish = {}, {}
    T = spec["tables"]
    for c in rv.columns:
        if "." not in c:
            continue
        tbl = _col_table(spec, fact, c)
        if tbl is None:
            continue
        cs = T[tbl]["columns"].get(c.split(".")[-1])
        if cs is None or cs["type"] not in MODELLED:
            continue
        # the join repeats each ancestor row once per fact row; the effective
        # sample size is the number of distinct ancestor rows
        # and, as ancestors carry unequal numbers of fact rows, the weighted
        # distribution has Kish's effective size (sum w)^2 / sum w^2 <= #ancestors
        kc = c.rsplit(".", 1)[0] + "." + T[tbl]["primary_key"]
        anc_rows, syn_rows = _kish(rv, kc, len(real[tbl])), _kish(sv, kc, len(syn[tbl]))
        nr_ = int(rv[c].notna().sum())
        if _is_cat(cs, rv[c]):
            pr = rv[c].dropna().astype(str).value_counts(normalize=True).to_numpy()
            mu, sd = tvd_null(pr, anc_rows, syn_rows)
            out[c] = {"tvd": tvd(rv[c].dropna(), sv[c].dropna()),
                      "crit": tvd_crit(pr, anc_rows, syn_rows, pp=True), "n_eff": round(anc_rows, 1),
                      "n_eff_syn": round(syn_rows, 1), "tvd_mu": mu, "tvd_sd": sd, "n_real": nr_}
        else:
            # dates compared as days, whichever side holds them as text (an
            # in-memory source); n_real counts the real values that compare
            num = (lambda x: _num(pd.to_datetime(x, errors="coerce"))) if cs["type"] in ("date", "datetime") else _num
            xr = num(rv[c])
            out[c] = {"ks": ks(xr, num(sv[c])), "crit": ks_crit(anc_rows, syn_rows, pp=True),
                      "n_eff": round(anc_rows, 1), "n_eff_syn": round(syn_rows, 1),
                      "n_real": int(np.isfinite(xr).sum())}
        kish[c] = (anc_rows, syn_rows)
    # association between the fact's own measures and ancestor attributes
    own = [c for c, cs in T[fact]["columns"].items() if cs["type"] in ("integer", "float")]
    anc = list(out)
    pairs = {}
    for a in own[:4]:
        for b in anc[:12]:
            tb = _col_table(spec, fact, b)
            cb = _is_cat(T[tb]["columns"][b.split(".")[-1]], rv[b])
            ra = _assoc(rv[a], rv[b], False, cb)
            sa = _assoc(sv[a], sv[b], False, cb)
            # standardise by the sampling sd, driven by the effective number of
            # ancestor rows behind the joined column on each side (Kish): fact rows
            # concentrated on few ancestors carry few independent ancestor values
            kr, ks_ = kish[b]
            pairs[f"{a}~{b}"] = abs(ra - sa) / np.sqrt(_eff(kr, ks_))
    worst = max(pairs, key=pairs.get) if pairs else None
    return {"columns": out, "assoc_max_z": pairs[worst] if worst else 0.0, "assoc_worst": worst,
            "assoc_k": len(pairs)}


def _col_table(spec, fact, path):
    parts = path.split(".")[:-1]
    t = fact
    for p in parts:
        r = next((r for r in spec["relationships"] if r["child"] == t and r["child_key"] == p), None)
        if r is None:
            return None
        t = r["parent"]
    return t


# ---------------------------------------------------------------- trees
def tree_metrics(real, syn, pk, fk, attr_cols, ppc=None):
    def stats(df):
        ids = keystr(df[pk]).to_numpy()
        par = keystr(df[fk]).to_numpy()
        idx = {k: i for i, k in enumerate(ids)}
        p = np.array([idx.get(x, -1) if x is not None else -1 for x in par])
        depth = np.full(len(p), -1)
        cyc = 0
        for s in range(len(p)):
            path, u, seen = [], s, set()
            while u >= 0 and depth[u] < 0 and u not in seen:
                seen.add(u)
                path.append(u)
                u = p[u]
            if u >= 0 and u in seen:
                cyc += 1
                u = -1
            base = depth[u] + 1 if u >= 0 else 0
            for i, w in enumerate(reversed(path)):
                depth[w] = base + i
        fan = np.bincount(p[p >= 0], minlength=len(p))
        return depth, fan, cyc, (p < 0).mean()
    dr, fr, cr, rr = stats(real)
    ds, fs, cs_, rs = stats(syn)
    out = {"depth_tvd": tvd(dr, ds), "fanout_ks": ks(fr.astype(float), fs.astype(float)),
           "fanout_ks_crit": ks_crit(len(fr), len(fs), pp=True), "nodes_real": len(fr), "nodes_syn": len(fs),
           "fanout_tvd": tvd(np.minimum(fr, 15), np.minimum(fs, 15)),
           "max_depth_real": int(dr.max()), "max_depth_syn": int(ds.max()) if len(ds) else 0,
           "cycles_syn": int(cs_), "root_share_real": float(rr), "root_share_syn": float(rs),
           "size_ratio": float(len(syn) / max(len(real), 1))}
    if ppc:
        # posterior predictive check (Gelman, Meng & Stern 1996): is the real depth
        # histogram as close to the predictive mean as independent model draws are?
        H = max(max(len(h) for h in ppc), dr.max() + 1)
        P = np.array([np.pad(np.array(h, float), (0, H - len(h))) for h in ppc])
        P /= P.sum(1, keepdims=True)
        mean = P.mean(0)
        hr = np.bincount(dr, minlength=H)[:H] / len(dr)
        t_real = 0.5 * np.abs(hr - mean).sum()
        # each draw against the mean of the other draws: a draw is not part of the
        # reference it is compared with, as the real histogram is not
        loo = (P.sum(0) - P) / max(len(P) - 1, 1) if len(P) > 1 else P
        t_sim = 0.5 * np.abs(P - loo).sum(1)
        out["depth_ppc_tvd"] = float(t_real)
        # ties within rounding count as ties (a generator that reproduces the source
        # depth profile exactly gives t_real = t_sim = 0 up to the last bit)
        out["depth_ppc_p"] = float((t_sim >= t_real - 1e-12).mean())
    by_depth, crit = {}, {}
    for c in attr_cols:     # attribute distribution conditional on depth, weighted by real depth share
        dcap_r, dcap_s = np.minimum(dr, 4), np.minimum(ds, 4)
        vals = pd.Series(dcap_r).value_counts(normalize=True)
        rc, sc_ = real[c].astype(str).to_numpy(), syn[c].astype(str).to_numpy()
        by_depth[c] = float(sum(w * tvd(rc[dcap_r == d], sc_[dcap_s == d])
                                for d, w in vals.items() if (dcap_s == d).any()))
        crit[c] = float(sum(w * tvd_crit(pd.Series(rc[dcap_r == d]).value_counts(normalize=True).to_numpy(),
                                         (dcap_r == d).sum(), (dcap_s == d).sum(), pp=True)
                            for d, w in vals.items() if (dcap_s == d).any()))
    out["attr_by_depth_tvd"] = by_depth
    out["attr_by_depth_crit"] = crit
    return out


# ---------------------------------------------------------------- time
def time_metrics(spec, real, syn, fact):
    T = spec["tables"][fact]
    tc = T["fact"]["time_column"]
    b = T["fact"].get("time_bucket", "month")
    fr = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[b]
    rp = pd.to_datetime(real[fact][tc]).dt.to_period(fr).astype(str)
    sp = pd.to_datetime(syn[fact][tc]).dt.to_period(fr).astype(str)
    out = {"bucket_tvd": tvd(rp, sp)}
    # seasonality conditional on the anchor's first categorical attribute
    rel = next((r for r in spec["relationships"] if r["child"] == fact), None)
    anchor = T["fact"].get("anchor") or rel["parent"]
    fk = next(r["child_key"] for r in spec["relationships"] if r["child"] == fact and r["parent"] == anchor)
    # sampling limit: the rows of one anchor share its time profile, so the
    # effective size is Kish's over the anchors' row counts
    kr_all = _kish(real[fact], fk, len(real[fact])) if len(real[fact]) else 1
    ks_all = _kish(syn[fact], fk, len(syn[fact])) if len(syn[fact]) else 1
    pbk = rp.value_counts(normalize=True).to_numpy()
    out["bucket_tvd_crit"] = tvd_crit(pbk, kr_all, ks_all, pp=True)
    out["bucket_tvd_mu"], out["bucket_tvd_sd"] = tvd_null(pbk, kr_all, ks_all)
    out["n_eff"], out["n_eff_syn"] = round(kr_all, 1), round(ks_all, 1)
    A = spec["tables"][anchor]
    cat = next((c for c, cs in A["columns"].items() if cs["type"] in ("categorical", "boolean")), None)
    if cat:
        pk = A["primary_key"]
        def cond(tables, per):
            m = tables[anchor].set_index(keystr(tables[anchor][pk]))[cat]
            return m.reindex(keystr(tables[fact][fk])).astype(str).to_numpy()
        rc, sc = cond(real, rp), cond(syn, sp)
        vals = pd.Series(rc).value_counts(normalize=True)
        out["conditional_on"] = f"{anchor}.{cat}"
        out["bucket_tvd_conditional"] = float(sum(w * tvd(rp[rc == v], sp[sc == v]) for v, w in vals.items()
                                                  if (sc == v).sum() > 0))
        rk, sk = keystr(real[fact][fk]).to_numpy(), keystr(syn[fact][fk]).to_numpy()

        def kish_of(keys):
            w = pd.Series(keys).value_counts().to_numpy(float)
            return float(w.sum() ** 2 / max((w ** 2).sum(), 1.0)) if len(w) else 1.0
        out["bucket_tvd_conditional_crit"] = float(sum(
            w * tvd_crit(rp[rc == v].value_counts(normalize=True).to_numpy(), kish_of(rk[rc == v]), kish_of(sk[sc == v]),
                         pp=True)
            for v, w in vals.items() if (sc == v).sum() > 0))
    return out


# ---------------------------------------------------------------- text
WORD = re.compile(r"\w+|[^\w\s]")


def _nb_fit(texts, labels):
    cls = sorted(set(labels))
    cnt = {c: Counter() for c in cls}
    prior = Counter(labels)
    for t, l in zip(texts, labels):
        cnt[l].update(WORD.findall(str(t).lower()))
    vocab = set().union(*cnt.values()) if cnt else set()
    tot = {c: sum(cnt[c].values()) for c in cls}
    return cls, cnt, prior, len(vocab), tot


def _nb_predict(model, texts):
    """Most probable class of each text under the multinomial naive Bayes model.
    The per-word log probabilities are tabulated once (class x word) with the same
    scalar arithmetic as the per-word formula, and each text's score is
    accumulated word by word in the same order, so the scores (and the first-max
    argmax) are exactly those of the direct loop; only the repeated dictionary
    and log work is removed."""
    cls, cnt, prior, V, tot = model
    n = sum(prior.values())
    C = len(cls)
    if not C:
        return [None] * len(texts)
    vocab = {}
    for c in cls:
        for w in cnt[c]:
            vocab.setdefault(w, len(vocab))
    # column len(vocab): a word no class has seen
    L = np.empty((C, len(vocab) + 1))
    for i, c in enumerate(cls):
        den = tot[c] + V + 1
        L[i, -1] = np.log(1 / den)
        cc = cnt[c]
        for w, j in vocab.items():
            L[i, j] = np.log((cc[w] + 1) / den) if w in cc else L[i, -1]
    lp = np.array([np.log(prior[c] / n) for c in cls])
    oov = len(vocab)
    out = []
    for t in texts:
        acc = np.zeros(C)
        for x in WORD.findall(str(t).lower()):
            acc += L[:, vocab.get(x, oov)]
        out.append(cls[int(np.argmax(lp + acc))])
    return out


def text_metrics(real_df, syn_df, col, cond, k=3, L=8, seed=0):
    r = real_df[[col] + ([cond] if cond else [])].dropna(subset=[col])
    s = syn_df[[col] + ([cond] if cond else [])].dropna(subset=[col])
    out = {"null_diff": float(abs(real_df[col].isna().mean() - syn_df[col].isna().mean()))}
    rt = [WORD.findall(str(t)) for t in r[col]]
    st = [WORD.findall(str(t)) for t in s[col]]
    out["len_mean_real"] = float(np.mean([len(t) for t in rt])) if rt else 0
    out["len_mean_syn"] = float(np.mean([len(t) for t in st])) if st else 0
    rows_with = Counter()
    for t in rt:
        rows_with.update(set(t))
    # the generator's guard counts tokens after scrubbing (an id "abc12345" is the
    # word "abc" and a number placeholder there); a token that is rare in the raw
    # text but held by >= k rows once scrubbed is allowed by the guard and is not a
    # leak.  Tokens absent from the scrubbed text (parts of e-mails, long numbers)
    # stay judged on the raw count.
    from .text import tokenize as _scrub_tokens
    rows_scrubbed = Counter()
    for t in r[col]:
        rows_scrubbed.update(set(_scrub_tokens(str(t))))
    rare = {w for w, c in rows_with.items() if c < k and rows_scrubbed.get(w, 0) < k}
    out["rare_token_leaks"] = int(sum(any(w in rare for w in t) for t in st))
    ngr = Counter()
    for t in rt:
        ngr.update({tuple(t[i:i + L]) for i in range(len(t) - L + 1)})
    rare_ng = {g for g, c in ngr.items() if c < k}
    out["rare_ngram_copies"] = int(sum(any(tuple(t[i:i + L]) in rare_ng for i in range(len(t) - L + 1)) for t in st))
    tf = Counter(" ".join(t) for t in rt)
    out["rare_exact_copies"] = int(sum(tf.get(" ".join(t), k) < k for t in st))
    out["unigram_tvd"] = tvd([w.lower() for t in rt for w in t], [w.lower() for t in st for w in t])
    big_s = [tuple(t[i:i + 2]) for t in st for i in range(len(t) - 1)]
    out["distinct2_syn"] = float(len(set(big_s)) / max(len(big_s), 1))
    if cond and len(r) > 40 and len(s) > 20:
        rng = np.random.default_rng(seed)
        m = rng.random(len(r)) < 0.7
        rtr, rte = r[m], r[~m]
        acc_real = np.mean(np.array(_nb_predict(_nb_fit(rtr[col].tolist(), rtr[cond].astype(str).tolist()),
                                                rte[col].tolist())) == rte[cond].astype(str).to_numpy())
        acc_syn = np.mean(np.array(_nb_predict(_nb_fit(s[col].tolist(), s[cond].astype(str).tolist()),
                                               rte[col].tolist())) == rte[cond].astype(str).to_numpy())
        out.update({"tstr_acc_real": float(acc_real), "tstr_acc_syn": float(acc_syn),
                    "tstr_ratio": float(acc_syn / max(acc_real, 1e-9))})
    return out


# ---------------------------------------------------------------- privacy
def determining_columns(real, tspec, cols, bf=100.0):
    """Drop categoricals that are functions of another categorical in the real
    data (decisive Bayes factor, no more combinations than the determinant has
    levels): a synthetic row that keeps such a dependency adds no information by
    matching a real row on the dependent column."""
    from .table_model import fd_log_bf
    cat = [c for c in cols if tspec["columns"][c]["type"] in ("categorical", "boolean")]
    codes = {c: pd.factorize(real[c].astype(object), use_na_sentinel=True)[0] for c in cat}
    drop = set()
    for b in cat:
        for a in cat:
            if a == b or a in drop:
                continue
            na = len(np.unique(codes[a]))
            if len(set(zip(codes[a].tolist(), codes[b].tolist()))) <= na and na >= len(np.unique(codes[b])) \
                    and fd_log_bf(codes[a], codes[b]) > np.log(bf):
                drop.add(b)
                break
    return [c for c in cols if c not in drop]


def dcr(real, syn, tspec, cols, n=1500, seed=0):
    """Distance to closest record: synthetic->real vs real->real (leave-one-out),
    in rank-normal-score / one-hot space. Ratio of 5th percentiles < 1 means
    synthetic rows sit closer to real rows than real rows sit to each other."""
    if not cols or len(syn) < 10 or len(real) < 10:
        return {}
    rr = real.sample(min(n, len(real)), random_state=seed)
    ss = syn.sample(min(n, len(syn)), random_state=seed)
    X = _design([rr[cols], ss[cols]], tspec, cols)[:, :-1]
    Xr, Xs = X[: len(rr)], X[len(rr):]
    def nn(A, B, loo=False):
        # nearest neighbour by one matrix product (|a|^2 + |b|^2 - 2ab); the distance to
        # the chosen neighbour is then recomputed by subtraction, so exact copies have
        # distance exactly 0 (the duplicate logic below depends on it)
        d = np.empty(len(A))
        bb = (B * B).sum(1)
        for i in range(0, len(A), 2048):
            a = A[i:i + 2048]
            D = (a * a).sum(1)[:, None] + bb[None] - 2 * a @ B.T
            if loo:
                D[np.arange(len(a)), i + np.arange(len(a))] = np.inf
            j = D.argmin(1)
            d[i:i + 2048] = np.sqrt(((a - B[j]) ** 2).sum(1))
        return d
    ds, dr = nn(Xs, Xr), nn(Xr, Xr, loo=True)
    if np.percentile(dr, 5) == 0 and (dr > 0).any() and (ds > 0).any():
        # real rows already duplicate each other (coarse attribute space): the
        # distance distribution is a point mass at 0 plus a positive part.  The
        # mass at 0 is tested by the exact-match rate against the real internal
        # rate; the ratio compares the positive parts of both distributions.
        dr, ds = dr[dr > 0], ds[ds > 0]
    keyr = set(map(tuple, rr[cols].astype(str).to_numpy()))
    exact = float(np.mean([tuple(x) in keyr for x in ss[cols].astype(str).to_numpy()]))
    p5s, p5r = np.percentile(ds, 5), np.percentile(dr, 5)
    # both 5th percentiles 0: every row has an exact twin on both sides (a coarse
    # attribute space); the ratio carries no information, exact matches are judged
    # by the exact-match rate
    ratio = 1.0 if p5s == 0 and p5r == 0 else float(p5s / max(p5r, 1e-9))
    keys_r = rr[cols].astype(str)
    return {"dcr_p5_syn": float(p5s), "dcr_p5_real": float(p5r), "dcr_ratio": ratio,
            "exact_match_rate": exact, "exact_match_rate_real_internal": float(keys_r.duplicated().mean()),
            # share of real rows that equal some other real row: the null value of the
            # synthetic exact-match rate (duplicated() above counts earlier rows only)
            "exact_match_rate_real_any": float(keys_r.duplicated(keep=False).mean()),
            "n": int(len(rr)), "m": int(len(ss))}


# ---------------------------------------------------------------- top level
def evaluate(spec, real, syn, scale=None, privacy=True, sample_log=None, real_stats=None):
    """Compare synthetic tables with the real ones.  real_stats maps a table whose
    real rows are a uniform sample (a pushed-down fact) to its exact statistics
    (relsynth.pushdown.stats.TableStats): sizes, per-parent counts, null keys and
    key-pair multiplicities then come from the statistics, and row-level metrics
    from the sample with the sample's size as their sampling noise."""
    real_stats = real_stats or {}
    T = spec["tables"]
    # the scorecard's tolerances and metric sizes (synthesis.evaluation, relsynth.settings)
    ecfg = (spec.get("synthesis") or {}).get("evaluation") or {}

    def n_real(t):
        return int(real_stats[t].n_rows) if t in real_stats else len(real[t])
    sc = scale or spec["synthesis"]["scale"]
    rep = {"tables": {}, "relations": {}, "trees": {}, "time": {}, "text": {}, "privacy": {}, "rules": {}}
    keycols = {(r["child"], c) for r in spec["relationships"]
               for c in (r["child_key"] if r.get("composite") else [r["child_key"]])}
    kmin = spec["synthesis"].get("privacy", {}).get("suppress_small_counts", 5)
    scfg = sc if isinstance(sc, dict) else {"default": float(sc)}
    scfg = inherit_scale(spec, scfg)          # child entities follow a scaled parent, as sampled
    trees = {r["child"] for r in spec["relationships"] if r["kind"] == "self_reference"}
    fk_of = {}
    for r in spec["relationships"]:
        if not r.get("composite"):
            fk_of.setdefault(r["child"], {})[r["child_key"]] = r
    grains = {t for t, ts in T.items() if any(
        len(u) == 2 and all(c in fk_of.get(t, {}) and fk_of[t][c]["parent"] != t for c in u)
        for u in ts.get("unique", []))}

    def _size_factor(t):
        """Synthetic / real rows of table t as planned: the table's scale (rounded
        as the sampler rounds it), 1 for copied tables, and the realised ratio for
        tables whose size is random (trees, grain tables)."""
        n = n_real(t)
        if T[t].get("mode") == "copy":
            return 1.0
        if t in trees or t in grains or T[t]["role"] != "entity" or not n:
            return len(syn[t]) / max(n, 1)
        s = float((scfg.get("tables") or {}).get(t, scfg.get("default", 1.0)))
        return max(round(s * n), 1) / n
    for t, ts in T.items():
        cols = [c for c, cs in ts["columns"].items() if cs["type"] in MODELLED and (t, c) not in keycols
                and c != ts.get("primary_key")]
        d = {"rows_real": n_real(t), "rows_syn": len(syn[t]),
             "columns": column_metrics(real[t], syn[t], ts, cols)}
        if t in real_stats:
            d["rows_real_sample"] = len(real[t])
        d["association"] = assoc_metrics(real[t], syn[t], ts, cols, min_level_rows=kmin,
                                          max_rows=int(ecfg.get("assoc_max_rows", 20000)),
                                          floor=(ecfg.get("thresholds") or {}).get("assoc_max"))
        if ts.get("mode") == "copy":
            # copied tables equal the source row for row: nothing to discriminate
            d["discriminator_auc"], d["discriminator_skipped"] = float("nan"), "copy"
        else:
            d["discriminator_auc"] = discriminator_auc(real[t], syn[t], ts, cols,
                                                       max_rows=int(ecfg.get("discriminator_rows", 5000)))
            d["discriminator_n"] = int(min(len(real[t]), len(syn[t]), 5000))
            d["discriminator_real_ok"] = bool(cols) and len(real[t]) >= 20
        rep["tables"][t] = d
        # a categorical determined by another one (city -> country) repeats its
        # information, so distances are taken over the determining columns only
        pcols = determining_columns(real[t], ts, cols)
        if privacy and ts["role"] == "entity" and ts.get("privacy", True) and ts.get("mode") != "copy" \
                and len(pcols) >= 3 and len(real[t]) >= 50:
            rep["privacy"][t] = dcr(real[t], syn[t], ts, pcols)
        for c, cs in ts["columns"].items():
            if cs["type"] == "text":
                o = cs.get("text") or {}
                tc = spec["synthesis"]["text"]
                # privacy is judged against a fixed standard (k >= 3 rows, runs of <= 8
                # tokens), never against looser generation settings in the spec
                rep["text"][f"{t}.{c}"] = text_metrics(real[t], syn[t], c, o.get("condition_on"),
                                                       k=max(o.get("min_token_rows", tc["min_token_rows"]), 3),
                                                       L=min(o.get("max_copy_ngram", tc["max_copy_ngram"]), 8))
        if ts["role"] == "fact":
            rep["relations"][f"{t}.joined"] = join_metrics(spec, real, syn, t)
            if ts["fact"].get("time_column"):
                rep["time"][t] = time_metrics(spec, real, syn, t)
    for r in spec["relationships"]:
        c, p = r["child"], r["parent"]
        if r.get("composite"):
            rep["relations"][r["name"]] = composite_metrics(spec, r, real, syn, real_stats)
            continue
        pk = T[p]["primary_key"]
        if r["kind"] == "self_reference":
            attrs = [x for x, cs in T[c]["columns"].items() if cs["type"] in ("categorical", "boolean")]
            ppc = (sample_log or {}).get("tree_ppc", {}).get(r["name"])
            rep["trees"][r["name"]] = tree_metrics(real[c], syn[c], pk, r["child_key"], attrs, ppc)
            continue
        if T[c]["role"] == "fact":
            anchor = _grand_anchor(spec, c)
            # per-anchor counts are preserved; per-partner counts follow the realised
            # size of the anchor table (which is random for trees and grain tables)
            ratio = 1.0 if p == anchor else (len(syn[anchor]) / max(n_real(anchor), 1)) / \
                (len(syn[p]) / max(n_real(p), 1))
        else:
            # entity child: the planned size ratio (scale), so that a child table of the
            # wrong size relative to its parent shows in the per-parent mean; realised
            # sizes only for tables whose size is random (trees, grain tables)
            ratio = _size_factor(c) / _size_factor(p)
        st = real_stats.get(c)
        fk = r["child_key"]
        d = cardinality(real[c], syn[c], fk, real[p], syn[p], pk, ratio,
                        real_counts=st.counts(fk) if st is not None else None)
        d["fk_integrity"] = fk_integrity(syn[c], fk, syn[p], pk)
        nulls_real = (st.n_rows - st.notnull(fk)) if st is not None else int(real[c][fk].isna().sum())
        d["null_fk_real"] = float(nulls_real / max(n_real(c), 1)) if n_real(c) else 0.0
        d["null_fk_syn"] = float(syn[c][fk].isna().mean())
        d["rows_child_real"], d["rows_child_syn"] = n_real(c), int(len(syn[c]))
        d["null_fk_count_real"] = int(nulls_real)
        d["null_fk_count_syn"] = int(syn[c][r["child_key"]].isna().sum())
        rep["relations"][r["name"]] = d
    rep["rules"] = rule_metrics(spec, real, syn, sampled=set(real_stats))
    rep["grain"] = grain_metrics(spec, real, syn, real_stats)
    rep["removed"] = removed_rows(spec, syn, sample_log)
    rep["scorecard"] = scorecard(rep, th=ecfg.get("thresholds") or None)
    return rep


def removed_rows(spec, syn, sample_log):
    """Share of generated rows that the sampler had to delete per table (rules or keys
    that redraws could not satisfy).  Deleting rows conditions the sample on an event
    the model does not know about, so counts and every statistic of the table drift;
    the share is a scorecard item so that losses are never silent."""
    out = {}
    for k, n in ((sample_log or {}).get("rows_removed_by_rules") or {}).items():
        t = str(k).split(" (")[0].split(":")[0]
        if t not in spec["tables"] or not n:
            continue
        d = out.setdefault(t, {"removed": 0, "kept": int(len(syn[t])) if t in syn else 0, "causes": {}})
        d["removed"] += int(n)
        d["causes"][str(k)] = int(n)
    for d in out.values():
        d["share"] = d["removed"] / max(d["removed"] + d["kept"], 1)
    return out


# ---------------------------------------------------------------- rules, keys, grain
def _rule_rate(spec, t, tables, fn):
    from .prepare import make_resolver
    df = tables[t]
    if not len(df):
        return 0.0
    try:
        return float(fn(df, make_resolver(spec, t, df, tables)))
    except Exception:  # noqa: BLE001  (a rule that cannot be evaluated on this side)
        return float("nan")


def rule_metrics(spec, real, syn, sampled=()):
    """Violation rate of every check rule, and the share of rows where a derived,
    lookup or aggregate column differs from its definition (real and synthetic).
    Rules over several rows of a table whose real rows are a sample (running
    balances, aggregates of a sampled fact) cannot be checked on the real side
    (NaN there)."""
    sampled = set(sampled)
    from .expr import check_mask, split_assignment
    from .prepare import _same, compute_aggregate, derive, lookup_values
    out = {}
    for rule in spec["rules"]:
        t = rule["table"]
        if rule["type"] == "check":
            def fn(df, res, e=rule["expr"]):
                return 1 - check_mask(df, e, res).mean()
            name = f"{t}: {rule['expr']}"
        elif rule["type"] == "cumulative":
            name = f"{t}: {rule['column']} = previous + {rule['step']} (by {rule['by']}, order {rule['order']})"
            out[name] = {"kind": "cumulative"}
            for side, tabs in (("real", real), ("syn", syn)):
                def fn5(df, res, r_=rule):
                    from .engine import cumulative_values
                    want = cumulative_values(df, r_, res)
                    by = r_["by"] if isinstance(r_["by"], list) else [r_["by"]]
                    first = ~df.sort_values(r_["order"]).duplicated(by).reindex(df.index).to_numpy()
                    got = pd.to_numeric(df[r_["column"]], errors="coerce").to_numpy(dtype=float)
                    ok = np.isclose(got, want, atol=1e-6) | first
                    return 1 - ok.mean()
                out[name][f"violation_rate_{side}"] = float("nan") if side == "real" and t in sampled \
                    else _rule_rate(spec, t, tabs, fn5)
            continue
        elif rule["type"] in ("derive", "soft"):
            lhs = split_assignment(rule["expr"])[0]
            name = f"{t}: {rule['expr']}"
            out[name] = {"kind": rule["type"], "rows_real": int(len(real[t])), "rows_syn": int(len(syn[t]))}
            for side, tabs in (("real", real), ("syn", syn)):
                def fn2(df, res, r_=rule, t_=t, tabs=tabs, lhs=lhs):
                    return 1 - _same(df[lhs], derive(spec, t_, df, r_, tabs), r_.get("tolerance")).mean()
                out[name][f"violation_rate_{side}"] = _rule_rate(spec, t, tabs, fn2)
            continue
        else:                       # aggregate
            name = f"{t}: {rule['column']} = {rule.get('agg', 'count')}({rule['from']})"
            out[name] = {"kind": "aggregate"}
            for side, tabs in (("real", real), ("syn", syn)):
                def fn3(df, res, r_=rule, tabs=tabs):
                    return 1 - _same(df[r_["column"]], compute_aggregate(spec, r_, tabs)).mean()
                out[name][f"violation_rate_{side}"] = float("nan") if side == "real" and rule["from"] in sampled \
                    else _rule_rate(spec, t, tabs, fn3)
            continue
        out[name] = {"kind": "check", "violation_rate_real": _rule_rate(spec, t, real, fn),
                     "violation_rate_syn": _rule_rate(spec, t, syn, fn)}
    for t, ts in spec["tables"].items():
        for c, cs in ts["columns"].items():
            if cs["type"] != "lookup":
                continue
            name = f"{t}: {c} = {cs['lookup']}"
            out[name] = {"kind": "lookup"}
            for side, tabs in (("real", real), ("syn", syn)):
                def fn4(df, res, c=c, t_=t, tabs=tabs):
                    return 1 - _same(df[c], lookup_values(spec, t_, df, c, tabs)).mean()
                out[name][f"violation_rate_{side}"] = _rule_rate(spec, t, tabs, fn4)
    return out


def _dup_rate(spec, t, tables, cols, where=None):
    from .expr import check_mask
    from .prepare import make_resolver
    df = tables[t]
    if not len(df) or not all(c in df.columns for c in cols):
        return float("nan")
    m = np.ones(len(df), bool)
    if where:
        res = make_resolver(spec, t, df, tables)
        m = check_mask(df, where, res) & ~check_mask(df, f"not ({where})", res)
    sub = df.loc[m, cols].astype(object)
    sub = sub.where(sub.notna(), None)
    return float(sub.duplicated().mean()) if len(sub) else 0.0


def grain_metrics(spec, real, syn, real_stats=None):
    """Generic grain checks.
      * unique keys (full and filtered): duplicate rate
      * pair multiplicity of a fact's (anchor, partner): share of pairs with >1 row
      * support: share of synthetic rows whose combination of two reference keys
        (copied tables, so the keys mean the same thing on both sides) never occurs
        in the real data, measured where the real data covers its combinations densely
    """
    T = spec["tables"]
    real_stats = real_stats or {}
    out = {}
    for t, ts in T.items():
        for u in ts.get("unique", []):
            if t in real_stats and all(c in ts["columns"] for c in u):
                # exact: rows minus distinct tuples (a sample would hide duplicates)
                st = real_stats[t]
                dup = float(1 - len(st.group_counts(list(u))) / st.n_rows) if st.n_rows else 0.0
            else:
                dup = _dup_rate(spec, t, real, u)
            out[f"{t}: unique {u}"] = {"kind": "unique", "real": dup, "syn": _dup_rate(spec, t, syn, u)}
        for pu in ts.get("partial_unique", []):
            out[f"{t}: unique {pu['columns']} where {pu['where']}"] = {
                "kind": "unique", "real": _dup_rate(spec, t, real, pu["columns"], pu["where"]),
                "syn": _dup_rate(spec, t, syn, pu["columns"], pu["where"])}
    copied = {t for t, ts in T.items() if ts.get("mode") == "copy"}
    for f, ts in T.items():
        if ts["role"] != "fact":
            continue
        fks = [r for r in spec["relationships"] if r["child"] == f and not r.get("composite")]
        anchor = ts["fact"].get("anchor")
        partner = ts["fact"].get("partner")
        ra = next((r for r in fks if r["parent"] == anchor), None)
        rp = next((r for r in fks if r["parent"] == partner), None)
        nest = next((r for r in spec["relationships"] if r["child"] == f and r.get("composite")), None)
        acols = [ra["child_key"]] if ra is not None else \
            ([c for c in nest["child_key"] if "." not in c] if nest is not None else [])
        if acols and rp is not None:
            cols = acols + [rp["child_key"]]

            def rep_share(df, cols=cols):
                g = df.groupby([keystr(df[c]) for c in cols]).size()
                return float((g > 1).mean()) if len(g) else 0.0
            if f in real_stats:
                # exact from GROUP BY (a sample thins repeated pairs)
                g = real_stats[f].group_counts(cols)
                g = g[g[cols].notna().all(axis=1)]
                real_share, real_pairs = (float((g["n"] > 1).mean()) if len(g) else 0.0), int(len(g))
            else:
                real_share, real_pairs = rep_share(real[f]), int(real[f].groupby(cols).ngroups)
            out[f"{f}: pairs ({', '.join(cols)})"] = {
                "kind": "multiplicity", "real": real_share, "syn": rep_share(syn[f]),
                "pairs_real": real_pairs, "pairs_syn": int(syn[f].groupby(cols).ngroups)}
        # reference keys reachable from the fact in up to two hops through copied tables
        refs = {}
        for r in spec["relationships"]:
            # a key computed from another key (ship date = order date + gap, a lookup)
            # is new exactly when its base is: pairs with it test nothing of its own
            if r["child"] == f and not r.get("composite") and r["parent"] in copied and not r.get("offset_from") \
                    and spec["tables"][f]["columns"].get(r["child_key"], {}).get("type") != "lookup":
                refs[r["child_key"]] = (r["child_key"],)
        for r in fks:
            p = r["parent"]
            for r2 in spec["relationships"]:
                if r2["child"] == p and not r2.get("composite") and r2["parent"] in copied and p not in copied:
                    refs[f"{r['child_key']}.{r2['child_key']}"] = (r["child_key"], r2["child_key"])
        names = sorted(refs)
        memo = {}

        def refv(tabs, side, x):
            if (side, x) not in memo:
                memo[(side, x)] = _ref_values(spec, f, tabs, x)
            return memo[(side, x)]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = refv(real, "r", a), refv(real, "r", b)
                sa, sb = refv(syn, "s", a), refv(syn, "s", b)
                if va is None or sa is None or vb is None or sb is None:
                    continue
                cnt = Counter(zip(va, vb))
                if not cnt or len(va) / len(cnt) < 5:
                    continue                 # sparse support: novel combinations are expected
                novel = float(np.mean([(x, y) not in cnt for x, y in zip(sa, sb)])) if len(sa) else 0.0
                # null value of the novel share: the real sample's missing mass, estimated
                # by Good-Turing (N1/n), with Esty's (1983) variance plus the synthetic
                # sample's binomial noise
                n = len(va)
                n1 = sum(1 for v in cnt.values() if v == 1)
                n2 = sum(1 for v in cnt.values() if v == 2)
                gt = n1 / n
                sd = float(np.sqrt(max(n1 + 2 * n2 - n1 * n1 / n, 1) / n ** 2 + max(gt, 1 / n) / max(len(sa), 1)))
                out[f"{f}: support ({a}, {b})"] = {"kind": "support", "real": 0.0, "syn": novel,
                                                   "rows_per_combination": round(len(va) / len(cnt), 1),
                                                   "good_turing": gt, "sd": sd, "crit": gt + 2.5 * sd}
    return out


def _ref_values(spec, t, tables, path):
    from .prepare import make_resolver
    df = tables[t]
    try:
        v = df[path] if "." not in path else make_resolver(spec, t, df, tables)(path)
    except Exception:  # noqa: BLE001
        return None
    return keystr(pd.Series(v)).fillna("__null__").to_numpy()


def composite_metrics(spec, r, real, syn, real_stats=None):
    """Integrity and cardinality of a composite (nested) reference.  With
    real_stats for the child (a sampled fact), the real side is computed from the
    rows per key tuple over every row."""
    from .prepare import _index, _keystr, _tuple_frame, make_resolver
    out = {}
    st = (real_stats or {}).get(r["child"])
    for side, tabs in (("real", real), ("syn", syn)):
        child, parent = tabs[r["child"]], tabs[r["parent"]]
        w = None
        if side == "real" and st is not None:
            cols = list(dict.fromkeys(c.split(".", 1)[0] for c in r["child_key"]))
            gc = st.group_counts(cols)
            w = gc["n"].to_numpy(dtype=float)
            child = coerce_like(gc[cols], child)
        res = make_resolver(spec, r["child"], child, tabs)
        ck = _tuple_frame([_keystr(child[c]) if "." not in c else _keystr(res(c)) for c in r["child_key"]])
        pk = _index(_tuple_frame([_keystr(parent[c]) for c in r["parent_key"]]))
        hit = pk.get_indexer(_index(ck)) if len(ck) else np.zeros(0, np.int64)
        if w is None:
            w = np.ones(len(hit))
        out[f"integrity_{side}"] = float(w[hit >= 0].sum() / w.sum()) if len(hit) and w.sum() else 1.0
        cnt = np.bincount(hit[hit >= 0], weights=w[hit >= 0], minlength=len(parent)).astype(float)
        out[f"counts_{side}"] = cnt
    rc, sc = out.pop("counts_real"), out.pop("counts_syn")
    return {"kind": "composite", "fk_integrity": out["integrity_syn"], "fk_integrity_real": out["integrity_real"],
            "mean_real": float(rc.mean()) if len(rc) else 0.0, "mean_syn": float(sc.mean()) if len(sc) else 0.0,
            "zero_share_real": float((rc == 0).mean()) if len(rc) else 0.0,
            "zero_share_syn": float((sc == 0).mean()) if len(sc) else 0.0,
            "ks_shape": ks(rc, sc) if len(rc) and len(sc) else float("nan"),
            "ks_crit": ks_crit(len(rc), len(sc), pp=True), "parents_real": int(len(rc)), "parents_syn": int(len(sc))}


def coerce_like(df, like):
    """df's columns in the dtypes of the same columns of `like` (statistics come
    back in the driver's types; keys must compare equal to the loaded frames)."""
    out = {}
    for c in df.columns:
        s = df[c].reset_index(drop=True)
        if c in like.columns:
            try:
                s = s.astype(like[c].dtype)
            except (TypeError, ValueError):
                from .data import canonical_keys
                s = canonical_keys(s)
        out[c] = s
    return pd.DataFrame(out)


def scorecard(rep, th=None):
    """Pass/fail items.  Each limit is at least its fixed tolerance (THRESHOLDS) and
    at least the sampling limit of its statistic under the null that the synthetic
    data is a posterior predictive (or independent) draw of the real data's law.
    Every statistical item carries `alpha`, its (approximate, upper-bound) level
    under that null; the summary's expected_false_failures is their sum.  An item
    whose real side can be evaluated but whose synthetic side cannot (an all-null
    column, a too-small table, a rule that fails to evaluate) is a failure."""
    th = {**THRESHOLDS, **(th or {})}
    items = []

    def add(area, name, value, limit, ok, alpha=None, real_ok=False):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if real_ok:
                items.append({"area": area, "metric": name, "value": None, "limit": limit, "pass": False,  # nosec B105
                              "alpha": 0.0, "note": "synthetic side could not be evaluated"})
            return
        it = {"area": area, "metric": name, "value": round(float(value), 4), "limit": limit, "pass": bool(ok)}
        if alpha is not None:
            it["alpha"] = round(float(alpha), 6)
        items.append(it)

    def tvd_alpha(lim, mu, sd):
        if mu is None or sd is None or not sd > 0:
            return None
        return float(ndtr(-(lim - mu) / sd))

    nan = float("nan")
    # discriminator: one-sided limits, Sidak-adjusted over the tables that have one
    k_auc = sum(1 for d in rep["tables"].values()
                if d.get("discriminator_auc") == d.get("discriminator_auc") or d.get("discriminator_real_ok"))
    z_auc = _sidak_z(0.01, max(k_auc, 1), two_sided=False)
    for t, d in rep["tables"].items():
        for c, m in d["columns"].items():
            if "tvd" in m:
                lim = max(th["tvd"], m["tvd_crit"])
                add(f"{t}.{c}", "marginal_tvd", m["tvd"], round(lim, 4), m["tvd"] <= lim,
                    tvd_alpha(lim, m.get("tvd_mu"), m.get("tvd_sd")), real_ok=m.get("n", 0) > 0)
            if "ks" in m:
                lim = max(th["ks"], 2 * m["ks_crit_1pct"])
                add(f"{t}.{c}", "marginal_ks", m["ks"], round(lim, 4), m["ks"] <= lim,
                    _ks_alpha(lim, m["n"], m["m"]) if "n" in m else None, real_ok=m.get("n", 0) >= 2)
            lim = round(max(th["null_diff"], m.get("null_crit", 0.0)), 4)
            add(f"{t}.{c}", "null_rate_diff", m["null_diff"], lim, m["null_diff"] <= lim,
                _z_alpha(lim, m.get("null_sd")))
        a = d["association"]
        lim = round(max(th["assoc_max"], a.get("worst_crit", 2.5 * np.sqrt(2 / max(d.get("rows_real_sample", d["rows_real"]), 1)
                                                                + 1 / max(d["rows_syn"], 1)))), 4)
        add(t, "association_max_diff", a["max_abs_diff"], lim, a["max_abs_diff"] <= lim,
            Z25_ALPHA if a.get("k_pairs") else None)
        if d.get("discriminator_skipped"):
            continue
        n_d = d.get("discriminator_n")
        lim = max(th["auc"], 0.5 + z_auc * auc_null_sd(n_d)) if n_d else th["auc"]
        add(t, "discriminator_auc", d["discriminator_auc"], round(lim, 4), d["discriminator_auc"] <= lim,
            float(ndtr(-(lim - 0.5) / auc_null_sd(n_d))) if n_d else None, real_ok=d.get("discriminator_real_ok", False))
    for n, d in rep["relations"].items():
        if d.get("kind") == "composite":
            add(n, "fk_integrity", d["fk_integrity"], th["fk_integrity"], d["fk_integrity"] >= th["fk_integrity"])
            lim = round(max(th["card_ks"], d["ks_crit"]), 4)
            add(n, "cardinality_ks", d["ks_shape"], lim, d["ks_shape"] <= lim,
                _ks_alpha(lim, d.get("parents_real", 1), d.get("parents_syn", 1)) if "parents_real" in d else None,
                real_ok=d.get("parents_real", 0) >= 2)
            continue
        if n.endswith(".joined"):
            for c, m in d["columns"].items():
                lim = round(max(th["join_tvd"], m["crit"]), 4)
                if "tvd" in m:
                    add(n + ":" + c, "join_tvd", m["tvd"], lim, m["tvd"] <= lim,
                        tvd_alpha(lim, m.get("tvd_mu"), m.get("tvd_sd")), real_ok=m.get("n_real", 0) > 0)
                elif "ks" in m:
                    add(n + ":" + c, "join_ks", m["ks"], lim, m["ks"] <= lim,
                        _ks_alpha(lim, m["n_eff"], m["n_eff_syn"]) if "n_eff_syn" in m else None,
                        real_ok=m.get("n_real", 0) >= 2)
            # max over K cells of |difference| / sampling sd: Sidak limit for K cells at
            # 1% (at least the former 3 standard errors)
            k = max(int(d.get("assoc_k", 1)), 1)
            zl = max(3.0, _sidak_z(0.01, k))
            add(n, "join_assoc_max_z", d["assoc_max_z"], round(zl, 3), d["assoc_max_z"] <= zl,
                1 - (1 - 2 * ndtr(-zl)) ** k if d.get("assoc_k") else None)
            continue
        add(n, "fk_integrity", d["fk_integrity"], th["fk_integrity"], d["fk_integrity"] >= th["fk_integrity"])
        lim = round(max(th["card_ks"], d["ks_crit"]), 4)
        add(n, "cardinality_ks", d["ks_shape"], lim, d["ks_shape"] <= lim,
            _ks_alpha(lim, d["parents_real"], d["parents_syn"]),
            real_ok=d["mean_real"] > 0 and d["parents_real"] >= 2)
        # relative error of the mean: both sides' sampling variance
        sd = np.sqrt(max(d["dispersion_real"], 1) / max(d["mean_real"] * d["parents_real"], 1)
                     + max(d.get("dispersion_syn", d["dispersion_real"]), 1)
                     / max(d.get("mean_syn", d["mean_real"]) * d.get("parents_syn", d["parents_real"]), 1))
        lim = round(max(th["card_mean_rel"], 2.5 * sd), 4)
        add(n, "cardinality_mean_rel_err", d["mean_rel_err"], lim, d["mean_rel_err"] <= lim, _z_alpha(lim, sd))
        nd = abs(d["null_fk_real"] - d["null_fk_syn"])
        if "rows_child_real" in d:
            nr_, ns_ = d["rows_child_real"], d["rows_child_syn"]
            pb = (d["null_fk_count_real"] + d["null_fk_count_syn"] + 0.5) / (nr_ + ns_ + 1)
            sd = float(np.sqrt(pb * (1 - pb) * _eff(nr_, ns_)))
        else:
            sd = 0.0
        lim = round(max(th["null_diff"], 2.5 * sd), 4)
        add(n, "null_fk_diff", nd, lim, nd <= lim, _z_alpha(lim, sd))
    for n, d in rep["trees"].items():
        if "depth_ppc_p" in d:
            add(n, "tree_depth_ppc_pvalue", d["depth_ppc_p"], 0.05, d["depth_ppc_p"] >= 0.05, 0.05)
        else:
            add(n, "tree_depth_tvd", d["depth_tvd"], th["tree_depth_tvd"], d["depth_tvd"] <= th["tree_depth_tvd"])
        lim = round(max(th["fanout_ks"], d.get("fanout_ks_crit", 0.0)), 4)
        add(n, "tree_fanout_ks", d["fanout_ks"], lim, d["fanout_ks"] <= lim,
            _ks_alpha(lim, d["nodes_real"], d["nodes_syn"]) if "nodes_real" in d else None)
        add(n, "tree_cycles", d["cycles_syn"], 0, d["cycles_syn"] == 0)
        for c, v in d["attr_by_depth_tvd"].items():
            lim = round(max(th["tree_depth_tvd"], d.get("attr_by_depth_crit", {}).get(c, 0.0)), 4)
            add(f"{n}:{c}", "attr_by_depth_tvd", v, lim, v <= lim)
    for n, d in rep["time"].items():
        lim = round(max(th["time_tvd"], d.get("bucket_tvd_crit", 0.0)), 4)
        add(n, "time_bucket_tvd", d["bucket_tvd"], lim, d["bucket_tvd"] <= lim,
            tvd_alpha(lim, d.get("bucket_tvd_mu"), d.get("bucket_tvd_sd")))
        if "bucket_tvd_conditional" in d:
            v = d["bucket_tvd_conditional"]
            lim = round(max(th["time_tvd"] * 1.5, d.get("bucket_tvd_conditional_crit", 0.0)), 4)
            add(n, "time_bucket_tvd_conditional", v, lim, v <= lim)
    for n, d in rep["text"].items():
        add(n, "text_rare_token_leaks", d["rare_token_leaks"], 0, d["rare_token_leaks"] == 0)
        add(n, "text_rare_ngram_copies", d["rare_ngram_copies"], 0, d["rare_ngram_copies"] == 0)
        add(n, "text_rare_exact_copies", d["rare_exact_copies"], 0, d["rare_exact_copies"] == 0)
        if "tstr_ratio" in d:
            add(n, "text_tstr_ratio", d["tstr_ratio"], th["tstr_ratio"], d["tstr_ratio"] >= th["tstr_ratio"])
    for n, d in rep["privacy"].items():
        if d:
            add(n, "dcr_ratio", d["dcr_ratio"], th["dcr_ratio"], d["dcr_ratio"] >= th["dcr_ratio"])
            if "exact_match_rate_real_any" in d:
                # null value: the share of real rows equal to another real row, plus 2.5 sd
                e = d["exact_match_rate_real_any"]
                sd = float(np.sqrt(max(e * (1 - e), 1 / max(d["n"], 1)) * _eff(d["n"], d["m"])))
                lim = max(th["exact_match"], e + 2.5 * sd)
                al = float(ndtr(-(lim - e) / sd))
            else:
                lim, al = max(th["exact_match"], 2 * d["exact_match_rate_real_internal"]), None
            add(n, "exact_match_rate", d["exact_match_rate"], round(lim, 4), d["exact_match_rate"] <= lim, al)
    for n, d in rep["rules"].items():
        if d.get("kind") == "soft":
            # a soft rule should break as often as in the source, not less
            r0, r1 = d.get("violation_rate_real"), d["violation_rate_syn"]
            r0_ok = r0 is not None and r0 == r0
            r0 = r0 if r0_ok else 0.0
            sd = None
            if "rows_real" in d and r1 == r1:
                rb = (r0 + r1) / 2
                sd = float(np.sqrt(rb * (1 - rb) * _eff(d["rows_real"], d["rows_syn"])))
            lim = max(0.01, 0.25 * r0, 2.5 * sd if sd else 0.0)
            add(n, "soft_rule_rate_diff", round(abs(r1 - r0), 6) if r1 == r1 else nan, round(lim, 6),
                abs(r1 - r0) <= lim, _z_alpha(lim, sd), real_ok=r0_ok)
            continue
        # a rule the real data itself breaks is reported, not held against the synthetic data
        r0 = d.get("violation_rate_real")
        lim = max(th["rule_violation"], r0 or 0.0)
        add(n, "rule_violation_rate", d["violation_rate_syn"], round(lim, 6), d["violation_rate_syn"] <= lim,
            real_ok=r0 is not None and r0 == r0)
    for t, d in rep.get("removed", {}).items():
        add(t, "rows_removed_share", d["share"], th["rows_removed"], d["share"] <= th["rows_removed"])
    for n, d in rep.get("grain", {}).items():
        if d["kind"] == "unique":
            lim = max(0.0, d["real"] if d["real"] == d["real"] else 0.0)
            add(n, "unique_duplicate_rate", d["syn"], round(lim, 6), d["syn"] <= lim, real_ok=d["real"] == d["real"])
        elif d["kind"] == "multiplicity":
            npairs = max(d["pairs_real"], 1)
            ps = max(d.get("pairs_syn", npairs), 1)
            sd = float(np.sqrt(max(d["real"] * (1 - d["real"]), 1 / npairs) * _eff(npairs, ps)))
            lim = round(max(th["pair_repeat"], 3 * sd), 4)
            diff = abs(d["syn"] - d["real"])
            add(n, "pair_repeat_share_diff", diff, lim, diff <= lim, _z_alpha(lim, sd))
        elif d["kind"] == "support":
            lim = max(th["novel_combination"], d.get("crit", 0.0))
            al = float(ndtr(-(lim - d["good_turing"]) / d["sd"])) if d.get("sd") else None
            add(n, "novel_combination_rate", d["syn"], round(lim, 4), d["syn"] <= lim, al)
    npass = sum(i["pass"] for i in items)
    efail = float(sum(i.get("alpha") or 0.0 for i in items))
    return {"passed": npass, "total": len(items), "failed": [i for i in items if not i["pass"]], "items": items,
            "expected_false_failures": round(efail, 3),
            "expected_false_failures_note": "sum of the items' levels under the null that the synthetic data is "
                                            "a draw of the real data's law; items without a level count 0"}

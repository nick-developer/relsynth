# SPDX-License-Identifier: Apache-2.0
"""Metrics for families of rows: marginals, within-family dependence, sequences, C2ST-Agg and DCR.

All metrics compare a synthetic family set with the real training families (same counts and
contexts).  Within-family metrics are the ones a per-row generator cannot match by
construction; the privacy metrics compare distances to training and to held-out families.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def _fam_index(off):
    return np.repeat(np.arange(len(off) - 1), np.diff(off))


def marginals(Xr, Cr, Xs, Cs, widths):
    ks = [float(ks_2samp(Xr[:, j], Xs[:, j]).statistic) for j in range(Xr.shape[1])]
    tvd = []
    for j, k in enumerate(widths):
        p = np.bincount(Cr[:, j], minlength=k) / len(Cr)
        q = np.bincount(Cs[:, j], minlength=k) / len(Cs)
        tvd.append(0.5 * float(np.abs(p - q).sum()))
    return {"ks_mean": float(np.mean(ks)) if ks else None, "tvd_mean": float(np.mean(tvd)) if tvd else None}


def within(X, C, off, widths):
    """Per column: P(two distinct siblings share a level) for categoricals, and the within-family
    share of variance (1 - ICC) for numerics, over families with at least two rows."""
    fi = _fam_index(off)
    n = np.diff(off)
    multi = n >= 2
    out = {}
    for j, k in enumerate(widths):
        cnt = np.zeros((len(n), k))
        np.add.at(cnt, (fi, C[:, j]), 1)
        pairs = (cnt * (cnt - 1)).sum(1)[multi]
        tot = (n * (n - 1))[multi]
        out[f"cat{j}_share"] = float(pairs.sum() / max(tot.sum(), 1))
        out[f"cat{j}_nunique"] = float(((cnt > 0).sum(1)[multi] / n[multi]).mean()) if multi.any() else None
    for j in range(X.shape[1]):
        x = X[:, j]
        s = np.bincount(fi, x, len(n))
        s2 = np.bincount(fi, x * x, len(n))
        within_ss = (s2 - s * s / np.maximum(n, 1))[multi].sum()
        total_ss = ((x - x.mean()) ** 2)[multi[fi]].sum()
        out[f"num{j}_within"] = float(within_ss / max(total_ss, 1e-12))
    return out


def sequence(X, C, off, widths):
    """Lag-1 autocorrelation of each numeric score and the transition matrix of each categorical
    along the family order (ordered datasets)."""
    fi = _fam_index(off)
    same = fi[1:] == fi[:-1]
    out = {}
    for j in range(X.shape[1]):
        a, b = X[:-1, j][same], X[1:, j][same]
        out[f"num{j}_lag1"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None
    for j, k in enumerate(widths):
        T = np.zeros((k, k))
        np.add.at(T, (C[:-1, j][same], C[1:, j][same]), 1)
        out[f"cat{j}_trans"] = T
    return out


def trans_tvd(Tr, Ts):
    """Row-weighted TVD between two transition count matrices."""
    w = Tr.sum(1)
    pr = Tr / np.maximum(w[:, None], 1)
    ps = Ts / np.maximum(Ts.sum(1)[:, None], 1)
    return float((w * 0.5 * np.abs(pr - ps).sum(1)).sum() / max(w.sum(), 1))


def family_features(X, C, off, widths, ctx_C):
    """C2ST-Agg-style features per family: size, mean and std of each numeric, number of distinct
    levels and modal share of each categorical, plus the parent's context levels."""
    fi = _fam_index(off)
    n = np.diff(off).astype(float)
    cols = [np.log(n)]
    for j in range(X.shape[1]):
        s = np.bincount(fi, X[:, j], len(n))
        s2 = np.bincount(fi, X[:, j] ** 2, len(n))
        m = s / n
        cols += [m, np.sqrt(np.maximum(s2 / n - m * m, 0))]
    for j, k in enumerate(widths):
        cnt = np.zeros((len(n), k))
        np.add.at(cnt, (fi, C[:, j]), 1)
        cols += [(cnt > 0).sum(1) / n, cnt.max(1) / n]
    F = np.column_stack(cols + [ctx_C.astype(float)] if ctx_C.size else cols)
    return F


def c2st(Fr, Fs, seed=0, folds=5):
    """5-fold accuracy of a classifier separating real from synthetic family features (0.5 =
    indistinguishable): xgboost with its default settings, as SyntheRela, when installed, else
    relsynth's L2 logistic regression."""
    k = min(len(Fr), len(Fs))
    rng = np.random.default_rng(seed)
    X = np.vstack([Fr[rng.choice(len(Fr), k, replace=False)], Fs[rng.choice(len(Fs), k, replace=False)]])
    y = np.r_[np.zeros(k), np.ones(k)]
    fold = np.r_[rng.permutation(np.arange(k) % folds), rng.permutation(np.arange(k) % folds)]
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    hits = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        if xgb is not None:
            m = xgb.train({"objective": "binary:logistic", "seed": seed + f, "nthread": 1, "verbosity": 0},
                          xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=100)
            pred = m.predict(xgb.DMatrix(X[te])) > 0.5
        else:
            from scipy.special import expit
            from relsynth.quality.multi import _logreg
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            D = np.column_stack([(X - mu) / sd, np.ones(len(X))])
            w = _logreg(D[tr], y[tr])
            pred = expit(D[te] @ w) > 0.5
        hits += int((pred == (y[te] == 1)).sum())
    return float(hits / (2 * k))


def dcr_share_strata(F_syn, s_syn, F_train, s_train, F_hold, s_hold, seed=0):
    """dcr_share within strata (log2 family-size buckets), pooled with weights = synthetic counts:
    family features depend on the family's size, which the engine sets, so items are compared only
    with reference families of a similar size, and each stratum has equal training and held-out sets."""
    num, den = 0.0, 0
    for b in np.unique(s_syn):
        t, h, q = s_train == b, s_hold == b, s_syn == b
        if t.sum() < 2 or h.sum() < 2:
            continue
        num += dcr_share(F_syn[q], F_train[t], F_hold[h], seed) * q.sum()
        den += int(q.sum())
    return float(num / den) if den else float("nan")


def dcr_share(F_syn, F_train, F_hold, seed=0):
    """Share of synthetic items closer to a training item than to a held-out item, with equal-size
    reference sets (0.5 = no more similar to training data than to unseen data; Platzer and
    Reutterer 2021).  Features standardised on the training set."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    k = min(len(F_train), len(F_hold))
    T = F_train[rng.choice(len(F_train), k, replace=False)]
    H = F_hold[rng.choice(len(F_hold), k, replace=False)]
    mu, sd = T.mean(0), T.std(0) + 1e-9
    z = lambda a: (a - mu) / sd
    dt = cKDTree(z(T)).query(z(F_syn))[0]
    dh = cKDTree(z(H)).query(z(F_syn))[0]
    return float(np.mean(dt < dh) + 0.5 * np.mean(dt == dh))


def row_matrix(X, C, widths):
    oh = [np.eye(k)[C[:, j]] for j, k in enumerate(widths)]
    return np.column_stack([X] + oh) if oh else X

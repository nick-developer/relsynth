"""Machine-learning utility: TSTR, TRTS and TRTR (Esteban et al. 2017).

TRTR trains on real rows and tests on held-out real rows; TSTR trains on
synthetic rows and tests on the same held-out rows; TRTS trains on real rows and
tests on synthetic rows.  The learners need no extra dependency: L2 logistic
regression (one-vs-rest for up to 10 classes) scored by prevalence-weighted
AUC, and ridge regression on the target's rank-normal score scored by R^2.
Features are encoded on the real training rows (relsynth.quality.encode).
The multi-table task runs the same learners on parent tables augmented with
child aggregates (relsynth.quality.multi.aggregated).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..evaluate import _auc, _logreg, _num
from .encode import codes, feature_matrix, is_categorical, rank_normal

MAX_CLASSES = 10


def pick_target(spec, t, real, cols):
    """Default target of table t when none is configured: the categorical column
    with 2-10 real levels (each of the two most frequent holding at least 5% of
    the rows) of highest normalised entropy; else the numeric column with the
    most distinct values (at least 20).  A heuristic: say which column matters
    with quality config `targets`."""
    best, score = None, -1.0
    for c in cols:
        if not is_categorical(spec, t, c, real[c]):
            continue
        p = real[c].dropna().astype(str).value_counts(normalize=True)
        if not 2 <= len(p) <= MAX_CLASSES or p.iloc[1] < 0.05:
            continue
        h = float(-(p * np.log(p)).sum() / np.log(len(p)))
        if h > score:
            best, score = c, h
    if best is not None:
        return best
    num = [(real[c].nunique(), c) for c in cols if not is_categorical(spec, t, c, real[c])]
    num = [x for x in num if x[0] >= 20]
    return max(num)[1] if num else None


def _xy(ref, df, spec, t, target, feats, cat):
    X = feature_matrix(ref, df, spec, t, feats)[1] if feats else np.zeros((len(df), 0))
    X = np.column_stack([X, np.ones(len(df))])
    if cat:
        _, y, k = codes(ref[target], df[target], True, max_levels=MAX_CLASSES)
        ok = y != k - 1                                      # code k-1 is null
    else:
        y = rank_normal(_num(ref[target]), _num(df[target]))
        ok = ~np.isnan(y)
    return X[ok], y[ok]


def score(ref, train, test, spec, t, target, feats, seed=0, max_rows=5000):
    """Score of a model trained on `train` and evaluated on `test` (features and
    target encoded on `ref`): prevalence-weighted one-vs-rest AUC for a
    categorical target, R^2 on the rank-normal scale for a numeric one."""
    cat = is_categorical(spec, t, target, ref[target])
    rng = np.random.default_rng(seed)
    train = train.iloc[rng.permutation(len(train))[:max_rows]].reset_index(drop=True)
    test = test.iloc[rng.permutation(len(test))[:max_rows]].reset_index(drop=True)
    Xa, ya = _xy(ref, train, spec, t, target, feats, cat)
    Xb, yb = _xy(ref, test, spec, t, target, feats, cat)
    if len(ya) < 10 or len(yb) < 10:
        return float("nan")
    sd = Xa.std(0)
    sd[sd == 0] = 1
    Xa, Xb = Xa / sd, Xb / sd
    if cat:
        classes = [k for k in np.unique(yb) if (yb == k).any() and (yb != k).any()]
        if not classes:
            return float("nan")
        tot, wsum = 0.0, 0.0
        for k in classes:
            yk = (ya == k).astype(float)
            if yk.min() == yk.max():
                auc = 0.5                                    # never (or always) seen in training
            else:
                w = _logreg(Xa, yk, lam=1.0)
                auc = _auc(expit(Xb @ w), (yb == k).astype(float))
            p = float((yb == k).mean())
            tot, wsum = tot + p * auc, wsum + p
        return float(tot / wsum)
    lam = 1.0
    w = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya)
    res = yb - Xb @ w
    return float(1 - (res ** 2).sum() / max(((yb - yb.mean()) ** 2).sum(), 1e-12))


def utility(ref, other, holdout, spec, t, target, cols, seed=0, max_rows=5000):
    """TRTR, TSTR and TRTS scores and their gaps.  `ref` is the real training
    data, `holdout` real rows the generator did not see (or an internal split)."""
    feats = [c for c in cols if c != target]
    trtr = score(ref, ref, holdout, spec, t, target, feats, seed, max_rows)
    tstr = score(ref, other, holdout, spec, t, target, feats, seed, max_rows)
    trts = score(ref, ref, other, spec, t, target, feats, seed, max_rows)
    cat = is_categorical(spec, t, target, ref[target])
    return {"target": target, "task": "classification" if cat else "regression",
            "score": "weighted one-vs-rest AUC" if cat else "R2 on rank-normal scale",
            "trtr": trtr, "tstr": tstr, "trts": trts,
            "tstr_gap": float(trtr - tstr), "trts_gap": float(abs(trtr - trts)),
            "rows": {"train_real": int(len(ref)), "train_other": int(len(other)), "holdout": int(len(holdout))}}

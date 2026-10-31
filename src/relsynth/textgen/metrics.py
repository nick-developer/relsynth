# SPDX-License-Identifier: Apache-2.0
"""Utility metrics of synthetic text (docs/text.md, Prop. 1507).

* **MAUVE** (Pillutla et al., NeurIPS 2021): both samples are embedded,
  reduced by PCA to 90% of the variance, quantised by k-means into
  k = n / 10 clusters fitted on the union, and the area under the divergence
  frontier (exp(-c KL(Q || R_l)), exp(-c KL(P || R_l))), R_l = l P + (1 - l) Q,
  is reported (c = 5, as in the reference code).  The default embedding is
  the data-independent hashed one of `relsynth.textgen.embed`, so the number
  is not comparable with MAUVE computed on GPT-2 features; with the
  `mauve-text` package and a local featurizer, `mauve(..., official=True)`
  calls the reference implementation instead.
* **Consistency with the row**: a classifier trained on real texts predicts
  each conditioning value from the text; consistency is the share of
  synthetic rows whose predicted value equals the row's own value, against the
  same share on held-out real rows (the ceiling a perfect generator reaches in
  expectation, Prop. 1507).
* **Downstream utility** (TSTR, Esteban et al. 2017): the classifier trained
  on synthetic texts and tested on real held-out rows, against one trained on
  real rows.
* **Memorisation**: exact and near-duplicate copies (`audit.near_duplicates`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .embed import HashEmbedder


# ------------------------------------------------------------------ classifier
def _nb(train_texts, train_y):
    from ..evaluate import _nb_fit
    return _nb_fit([str(t) for t in train_texts], [str(y) for y in train_y])


def _nb_pred(model, texts):
    from ..evaluate import _nb_predict
    return np.array(_nb_predict(model, [str(t) for t in texts]))


def _ok(df, col):
    return df[col].notna() & (df[col].astype(str).str.strip() != "")


def consistency(real: pd.DataFrame, syn: pd.DataFrame, text: str, cols, seed: int = 0, train_share: float = 0.7) -> dict:
    """Per conditioning column: accuracy of a real-trained classifier on held-out
    real rows (`real`), on synthetic rows (`synthetic`), and their ratio."""
    rng = np.random.default_rng(seed)
    r = real[_ok(real, text)]
    s = syn[_ok(syn, text)]
    m = rng.random(len(r)) < train_share
    out = {}
    for c in ([cols] if isinstance(cols, str) else list(cols)):
        tr = r[m & r[c].notna().to_numpy()]
        te = r[~m & r[c].notna().to_numpy()]
        ss = s[s[c].notna()]
        if len(tr) < 20 or len(te) < 10 or len(ss) < 10:
            continue
        clf = _nb(tr[text], tr[c])
        a_real = float(np.mean(_nb_pred(clf, te[text]) == te[c].astype(str).to_numpy()))
        a_syn = float(np.mean(_nb_pred(clf, ss[text]) == ss[c].astype(str).to_numpy()))
        n_s = len(ss)
        se = math.sqrt(max(a_syn * (1 - a_syn), 1e-9) / n_s)
        out[c] = {"real": a_real, "synthetic": a_syn, "ratio": a_syn / max(a_real, 1e-9),
                  "synthetic_ci": (max(0.0, a_syn - 1.96 * se), min(1.0, a_syn + 1.96 * se)), "n_synthetic": n_s,
                  "chance": float(tr[c].astype(str).value_counts(normalize=True).iloc[0])}
    return out


def tstr(real: pd.DataFrame, syn: pd.DataFrame, text: str, label: str, seed: int = 0, train_share: float = 0.7) -> dict:
    """Train on synthetic, test on real (held-out), against train on real."""
    rng = np.random.default_rng(seed)
    r = real[_ok(real, text) & real[label].notna()]
    s = syn[_ok(syn, text) & syn[label].notna()]
    m = rng.random(len(r)) < train_share
    te = r[~m]
    if len(te) < 10 or len(s) < 10:
        return {}
    a_real = float(np.mean(_nb_pred(_nb(r[m][text], r[m][label]), te[text]) == te[label].astype(str).to_numpy()))
    a_syn = float(np.mean(_nb_pred(_nb(s[text], s[label]), te[text]) == te[label].astype(str).to_numpy()))
    return {"label": label, "trtr": a_real, "tstr": a_syn, "ratio": a_syn / max(a_real, 1e-9), "n_test": len(te)}


# ------------------------------------------------------------------ MAUVE
def _kmeans(X, k, rng, iters=50):
    n = len(X)
    C = [X[int(rng.integers(n))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):                                   # k-means++
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        C.append(X[int(rng.choice(n, p=p))])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    lab = np.zeros(n, int)
    for _ in range(iters):
        D = (X * X).sum(1)[:, None] + (C * C).sum(1)[None] - 2 * X @ C.T
        new = D.argmin(1)
        if _ and (new == lab).all():
            break
        lab = new
        for j in range(k):
            sel = lab == j
            if sel.any():
                C[j] = X[sel].mean(0)
    return lab


def mauve(p_texts, q_texts, embed=None, k: int | None = None, c: float = 5.0, seed: int = 0, official: bool = False,
          featurizer: str | None = None) -> float:
    """MAUVE between human (p) and generated (q) texts, in [0, 1]; 1 is identical."""
    p_texts = [str(t) for t in p_texts if t is not None and str(t).strip()]
    q_texts = [str(t) for t in q_texts if t is not None and str(t).strip()]
    if official:
        import mauve as _m
        return float(_m.compute_mauve(p_text=p_texts, q_text=q_texts, featurize_model_name=featurizer or "gpt2-large",
                                      device_id=-1, verbose=False).mauve)
    embed = embed or HashEmbedder()
    P, Q = embed(p_texts), embed(q_texts)
    X = np.vstack([P, Q])
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
    d = int(np.searchsorted(var, 0.9) + 1)
    Z = X @ Vt[:d].T
    rng = np.random.default_rng(seed)
    k = k or max(2, min(500, len(Z) // 10))
    lab = _kmeans(Z, k, rng)
    p = np.bincount(lab[: len(P)], minlength=k) / len(P)
    q = np.bincount(lab[len(P):], minlength=k) / len(Q)
    xs, ys = [1.0], [0.0]
    for lam in np.linspace(1e-6, 1 - 1e-6, 25):
        r = lam * p + (1 - lam) * q
        kl_q = float(np.sum(q[q > 0] * np.log(q[q > 0] / r[q > 0])))
        kl_p = float(np.sum(p[p > 0] * np.log(p[p > 0] / r[p > 0])))
        xs.append(math.exp(-c * kl_q))
        ys.append(math.exp(-c * kl_p))
    xs.append(0.0)
    ys.append(1.0)
    xs, ys = np.array(xs), np.array(ys)
    o = np.argsort(xs)
    return float(np.trapezoid(ys[o], xs[o]) if hasattr(np, "trapezoid") else np.trapz(ys[o], xs[o]))


def report(real: pd.DataFrame, syn: pd.DataFrame, text: str, cols=(), owners=None, label: str | None = None,
           seed: int = 0) -> dict:
    """All text metrics of one column: MAUVE, consistency, TSTR, memorisation."""
    from .audit import near_duplicates
    out = {"column": text, "n_real": int(_ok(real, text).sum()), "n_synthetic": int(_ok(syn, text).sum())}
    rs = real.loc[_ok(real, text), text].astype(str).tolist()
    ss = syn.loc[_ok(syn, text), text].astype(str).tolist()
    if len(rs) >= 20 and len(ss) >= 20:
        rng = np.random.default_rng(seed)
        n = min(len(rs), len(ss), 3000)
        out["mauve"] = mauve([rs[i] for i in rng.choice(len(rs), n, replace=False)],
                             [ss[i] for i in rng.choice(len(ss), n, replace=False)], seed=seed)
    if cols:
        out["consistency"] = consistency(real, syn, text, cols, seed=seed)
    if label or cols:
        out["tstr"] = tstr(real, syn, text, label or list(cols)[0], seed=seed)
    out["memorisation"] = near_duplicates(rs, ss)
    return out

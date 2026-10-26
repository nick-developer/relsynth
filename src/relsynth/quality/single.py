"""Single-table fidelity metrics of the 2.0 report.

Column level: Wasserstein-1 on the raw scale (in real standard deviations) and
on rank-normal scores, Jensen-Shannon distance.  Table level: mean and largest
total variation distance over all 2-way and a seeded set of 3-way marginals of
the discretised columns (the k-marginal score of the NIST CRC / SDNist), and the
alpha-precision and beta-recall curves of Alaa et al. (ICML 2022) with their
integrated scores.  Every function returns plain floats; limits come from
relsynth.quality.calibrate.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from ..evaluate import _num
from .encode import codes, feature_matrix, is_categorical, rank_normal

ALPHAS = np.linspace(0.0, 1.0, 21)[1:]     # 0.05, 0.10, ..., 1.0


# ---------------------------------------------------------------- columns
def w1_raw(real, other):
    """W1 between the non-null values, divided by the real standard deviation
    (0 when the real column is constant and the other equal to it)."""
    a, b = _num(real), _num(other)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 1:
        return float("nan")
    sd = float(np.std(a))
    w = float(wasserstein_distance(a, b))
    return w / sd if sd > 0 else (0.0 if w == 0 else float("inf"))


def w1_rank(real, other):
    """W1 between the rank-normal scores (real mid-ECDF) of the non-null values."""
    a, b = _num(real), _num(other)
    za, zb = rank_normal(a, a), rank_normal(a, b)
    za, zb = za[~np.isnan(za)], zb[~np.isnan(zb)]
    if len(za) < 2 or len(zb) < 1:
        return float("nan")
    return float(wasserstein_distance(za, zb))


def js_distance(real, other, categorical, bins=20):
    """Jensen-Shannon distance (base 2, in [0, 1]) between the distributions of
    the discrete codes (levels, or real quantile bins for a numeric column; a
    null is a level of its own)."""
    a, b, k = codes(real, other, categorical, max_levels=10 ** 6, bins=bins)
    if not len(a) or not len(b):
        return float("nan")
    p = np.bincount(a, minlength=k).astype(float)
    q = np.bincount(b, minlength=k).astype(float)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def column_metrics(real, other, spec, t, cols):
    out = {}
    for c in cols:
        cat = is_categorical(spec, t, c, real[c])
        d = {"categorical": bool(cat), "js_distance": js_distance(real[c], other[c], cat)}
        if not cat:
            d["w1_raw_sd"] = w1_raw(real[c], other[c])
            d["w1_rank_normal"] = w1_rank(real[c], other[c])
        out[c] = d
    return out


# ---------------------------------------------------------------- k-way marginals
def kway_tvd(real, other, spec, t, cols, k3_max=50, seed=0, max_levels=20, bins=5):
    """TVD of every 2-way marginal and of up to k3_max seeded 3-way marginals of
    the discretised columns: {"k2_mean", "k2_max", "k2_worst", "k3_mean", ...}."""
    if len(cols) < 2 or not len(real) or not len(other):
        return {}
    enc = {}
    for c in cols:
        a, b, k = codes(real[c], other[c], is_categorical(spec, t, c, real[c]), max_levels=max_levels, bins=bins)
        enc[c] = (a, b, k)

    def tvd_of(group):
        ka = np.zeros(len(real), dtype=np.int64)
        kb = np.zeros(len(other), dtype=np.int64)
        size = 1
        for c in group:
            a, b, k = enc[c]
            ka, kb, size = ka * k + a, kb * k + b, size * k
        _, inv = np.unique(np.r_[ka, kb], return_inverse=True)
        m = inv.max() + 1
        p = np.bincount(inv[:len(ka)], minlength=m) / len(ka)
        q = np.bincount(inv[len(ka):], minlength=m) / len(kb)
        return float(0.5 * np.abs(p - q).sum())
    out = {}
    pairs = list(itertools.combinations(cols, 2))
    v2 = {f"{a}~{b}": tvd_of((a, b)) for a, b in pairs}
    w = max(v2, key=v2.get)
    out.update(k2_mean=float(np.mean(list(v2.values()))), k2_max=v2[w], k2_worst=w, k2_count=len(v2))
    if len(cols) >= 3:
        triples = list(itertools.combinations(cols, 3))
        if len(triples) > k3_max:
            rng = np.random.default_rng(seed)
            triples = [triples[i] for i in sorted(rng.choice(len(triples), k3_max, replace=False))]
        v3 = {"~".join(g): tvd_of(g) for g in triples}
        w = max(v3, key=v3.get)
        out.update(k3_mean=float(np.mean(list(v3.values()))), k3_max=v3[w], k3_worst=w, k3_count=len(v3))
    return out


# ---------------------------------------------------------------- alpha-precision, beta-recall
def _pit_curve(ref, x, grid, weight=None):
    """mean_i w_i * P(U_i <= a) for each a in grid, where U_i is the randomised
    probability integral transform of x_i under the empirical law of ref
    (uniform on [F(x_i-), F(x_i)]).  For x drawn from the law of ref this is
    exactly a in expectation, also when distances tie (discrete columns); the
    plain indicator 1[x_i <= q_a(ref)] is biased by the ties."""
    r = np.sort(ref)
    lo = np.searchsorted(r, x, side="left") / len(r)
    hi = np.searchsorted(r, x, side="right") / len(r)
    w = np.ones(len(x)) if weight is None else weight
    width = hi - lo
    out = []
    for a in grid:
        p = np.where(width > 0, np.clip((a - lo) / np.where(width > 0, width, 1), 0, 1), (lo <= a).astype(float))
        out.append(float(np.mean(w * p)))
    return np.array(out)


def precision_recall(real, other, spec, t, cols, max_rows=5000, k=5, seed=0):
    """alpha-precision and beta-recall curves (Alaa et al., ICML 2022) in the
    rank-normal feature space, with the hypersphere supports used by their
    reference implementation (synthcity): the alpha-support of a sample is the
    ball around its mean that holds a share alpha of its points.

    precision(a) = share of synthetic points inside the real a-support;
    recall(b)    = share of real points whose nearest synthetic point lies within
                   the real point's k-NN radius among real points and inside the
                   synthetic b-support.
    Points on the boundary of a support (tied distances) count by the
    randomised probability integral transform (_pit_curve).
    Integrated scores 1 - 2 * mean_a |curve(a) - a| (1 = ideal, 0 = worst);
    the report uses their complements (0 = ideal) as distances."""
    if not cols or len(real) < 2 * k + 2 or len(other) < 2:
        return {}
    rng = np.random.default_rng(seed)
    r = real.iloc[rng.permutation(len(real))[:max_rows]].reset_index(drop=True)
    s = other.iloc[rng.permutation(len(other))[:max_rows]].reset_index(drop=True)
    Xr, Xs = feature_matrix(r, s, spec, t, cols, quadratic=False)
    if Xr.shape[1] == 0:
        return {}
    cr, cs = Xr.mean(0), Xs.mean(0)
    dr = np.linalg.norm(Xr - cr, axis=1)
    ds = np.linalg.norm(Xs - cr, axis=1)
    ds_own = np.linalg.norm(Xs - cs, axis=1)
    prec = _pit_curve(dr, ds, ALPHAS)
    tr = cKDTree(Xr)
    rad = tr.query(Xr, k=k + 1)[0][:, -1]                      # k-th real neighbour (self excluded)
    dnn, inn = cKDTree(Xs).query(Xr, k=1)
    near = dnn <= rad
    rec = _pit_curve(ds_own, ds_own[inn], ALPHAS, near.astype(float))
    ip = 1 - 2 * float(np.mean(np.abs(prec - ALPHAS)))
    ir = 1 - 2 * float(np.mean(np.abs(rec - ALPHAS)))
    return {"alpha_precision": ip, "beta_recall": ir,
            "alpha_precision_gap": 1 - ip, "beta_recall_gap": 1 - ir,
            "precision_curve": [round(float(x), 4) for x in prec],
            "recall_curve": [round(float(x), 4) for x in rec],
            "alphas": [round(float(a), 2) for a in ALPHAS], "rows": [int(len(r)), int(len(s))]}

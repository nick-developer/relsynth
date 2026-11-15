# SPDX-License-Identifier: Apache-2.0
"""Data-driven strength of the hierarchical prior of segment marginals.

Every segment marginal of an entity table is a draw from a Dirichlet process
centred on its parent marginal (the coarse group's, else the table's):
``F_g ~ DP(a, F_0)`` (Ferguson 1973).  Its posterior given the n_g rows of the
segment is ``DP(a + n_g, (a F_0 + sum delta_x) / (a + n_g))``, so the
concentration ``a`` is the prior's weight in rows and each row of the segment
carries ``1 / (n_g + a)`` of the posterior predictive (Prop. 1726).

Instead of a fixed number of rows, ``a`` is estimated per column from the
segments themselves (parametric empirical Bayes; Morris 1983, Efron and Morris
1975).  Under ``F_g ~ DP(a, F_0)`` two rows of the same segment fall on the same
side of any threshold ``t`` with intra-class correlation

    rho = corr(1{x_i <= t}, 1{x_j <= t}) = 1 / (a + 1)      (Ferguson 1973, Thm 1)

for every ``t``, and for a categorical column the level indicators are the
Dirichlet-multinomial with the same ``rho`` (Mosimann 1962).  ``rho`` is
estimated with the one-way (nested, when segments sit in coarse groups)
analysis-of-variance moment estimator for unbalanced groups, summed over the
thresholds (or levels) before the ratio is taken, which is the estimator of
Weir and Cockerham (1984) for F_ST and of Fisher's intra-class correlation
(Donner 1986); it is unbiased in numerator and denominator separately and
needs no distributional assumption beyond the hierarchy.  Then
``a = (1 - rho) / rho`` (Prop. 1725).  It is the Bühlmann (1967) credibility
constant: the segment keeps ``n_g / (n_g + a)`` of its own law, the share that
minimises the expected squared error of the segment's distribution function
among linear rules (Bühlmann and Straub 1970).

Nothing here draws a random number: the estimate is a deterministic function
of the data, so turning it on changes the fit only through the prior weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: thresholds (quantile levels of the pooled values) at which a continuous
#: column's segment distribution functions are compared; the moment estimator
#: sums over them, so more thresholds lower its variance only slowly
THRESHOLDS = 15


@dataclass(frozen=True)
class Concentration:
    """The estimate for one column: intra-class correlation, concentration and
    the number of groups and rows it used (``groups`` < 2 means no estimate)."""
    rho: float
    a: float
    groups: int
    rows: int

    def to_dict(self) -> dict:
        return {"icc": round(self.rho, 6), "a": round(self.a, 4), "groups": self.groups, "rows": self.rows}


def icc_indicators(Y: np.ndarray, groups: np.ndarray, ref: np.ndarray | None = None) -> tuple[float, int, int]:
    """Moment estimate of the intra-class correlation of indicator columns.

    Y: (n, m) 0/1 matrix (one column per threshold or level); groups: (n,) group
    of each row; ref: (n,) reference group (coarse group) of each row, or None
    for one reference.  Returns (rho, groups used, rows used); rho is the ratio
    sum_m s2_b,m / sum_m (s2_b,m + s2_w,m) of the nested one-way ANOVA
    components (Sokal and Rohlf 1995, Sect. 10.3), and may be negative or above
    1 by sampling error (the caller truncates)."""
    Y = np.asarray(Y, dtype=float)
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0 or Y.size == 0:
        return float("nan"), 0, 0
    ref = np.zeros(n, dtype=np.int64) if ref is None else np.asarray(ref)
    # dense codes of (reference, group) pairs: a group belongs to one reference
    _, gi = np.unique(np.stack([ref.astype(np.int64), groups.astype(np.int64)]), axis=1, return_inverse=True)
    gi = gi.ravel()
    _, ri = np.unique(ref, return_inverse=True)
    G, C = int(gi.max()) + 1, int(ri.max()) + 1
    if n <= G or G <= C:
        return float("nan"), G, n
    ng = np.bincount(gi, minlength=G).astype(float)
    nc = np.bincount(ri, minlength=C).astype(float)
    g_ref = np.zeros(G, dtype=np.int64)
    g_ref[gi] = ri
    sg = np.zeros((G, Y.shape[1]))
    np.add.at(sg, gi, Y)
    sc = np.zeros((C, Y.shape[1]))
    np.add.at(sc, ri, Y)
    pg = sg / ng[:, None]
    pc = sc / nc[:, None]
    ssb = (ng[:, None] * (pg - pc[g_ref]) ** 2).sum(axis=0)
    ssw = (ng[:, None] * pg * (1 - pg)).sum(axis=0)
    msb = ssb / (G - C)
    msw = ssw / (n - G)
    # effective group size of the nested design (unbalanced groups)
    n0 = (n - float((ng ** 2 / nc[g_ref]).sum())) / (G - C)
    if n0 <= 0:
        return float("nan"), G, n
    s2b = (msb - msw) / n0
    den = float((s2b + msw).sum())
    if den <= 0:
        return float("nan"), G, n
    return float(s2b.sum() / den), G, n


def indicators(codec_kind: str, x: np.ndarray, levels: int = 0, thresholds: int = THRESHOLDS) -> np.ndarray:
    """Indicator matrix of one column: level indicators of a discrete column
    (the null code -1 is one more level), or 1{x <= t} at the pooled quantiles
    of a continuous column's observed values (nulls must be removed first)."""
    x = np.asarray(x)
    if codec_kind == "discrete":
        codes = np.where(x < 0, levels, x).astype(np.int64)
        present = np.unique(codes)
        return (codes[:, None] == present[None, :]).astype(float)
    if len(x) == 0:
        return np.zeros((0, 0))
    t = np.unique(np.quantile(x, np.arange(1, thresholds + 1) / (thresholds + 1)))
    Y = (x[:, None] <= t[None, :]).astype(float)
    keep = (Y.min(axis=0) < 1) & (Y.max(axis=0) > 0)
    return Y[:, keep]


def standardise(x: np.ndarray, groups: np.ndarray, min_rows: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Values centred at their group's median and scaled by its interquartile
    range (groups of fewer than min_rows values, or with no spread, are left
    out): the shape of each group, which is what shape-mode shrinkage pulls
    towards the parent's (Prop. 1724).  Returns (z, kept mask)."""
    x = np.asarray(x, dtype=float)
    keep = np.zeros(len(x), dtype=bool)
    z = np.zeros(len(x))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < min_rows:
            continue
        v = x[idx]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        if q3 - q1 <= 0:
            continue
        z[idx] = (v - med) / (q3 - q1)
        keep[idx] = True
    return z, keep


def concentration(codec_kind: str, x: np.ndarray, groups: np.ndarray, ref: np.ndarray | None = None, *,
                  levels: int = 0, shape: bool = False, a_min: float = 0.0, a_max: float = np.inf) -> Concentration:
    """Empirical-Bayes concentration of one column's segment marginals around
    their parent (Prop. 1725): a = (1 - rho) / rho, truncated to [a_min, a_max].
    shape: compare the groups' shapes (values standardised by their own median
    and interquartile range) instead of their laws, the variant shape-mode
    shrinkage needs.  With fewer than two groups with rows in some reference
    group there is no information: a_max is returned (full pooling is the
    prior's own limit), with groups < 2 marking it."""
    x = np.asarray(x)
    groups = np.asarray(groups)
    ref = np.zeros(len(groups), dtype=np.int64) if ref is None else np.asarray(ref)
    if codec_kind != "discrete":
        obs = ~np.isnan(x.astype(float))
        x, groups, ref = x[obs].astype(float), groups[obs], ref[obs]
        if shape:
            x, keep = standardise(x, groups)
            x, groups, ref = x[keep], groups[keep], ref[keep]
    Y = indicators(codec_kind, x, levels)
    if Y.shape[1] == 0 or len(np.unique(groups)) < 2:
        return Concentration(rho=0.0, a=float(a_max), groups=int(len(np.unique(groups))), rows=int(len(x)))
    rho, G, n = icc_indicators(Y, groups, ref)
    if not np.isfinite(rho):
        return Concentration(rho=0.0, a=float(a_max), groups=G, rows=n)
    a = float(a_max) if rho <= 0 else max((1.0 - min(rho, 1.0)) / min(rho, 1.0), 0.0)
    return Concentration(rho=float(rho), a=float(min(max(a, a_min), a_max)), groups=G, rows=n)


def floor_weight(n_rows: int, a: float, floor_rows: float) -> float:
    """Prior weight of a segment of n_rows rows under the privacy floor
    (Prop. 1726): at least floor_rows - n_rows, so that the segment's posterior
    predictive is a mixture over at least floor_rows rows of mass and one source
    row carries at most 1 / floor_rows of it."""
    return float(max(a, floor_rows - n_rows, 0.0))

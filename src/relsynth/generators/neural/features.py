# SPDX-License-Identifier: Apache-2.0
"""The feature space of the neural generators (numpy only).

A neural generator does not see raw values.  Every modelled column is mapped
through relsynth's own Bayesian marginal of the table (relsynth.marginals,
the copula's marginals), so the network models the *dependence* between
columns and the marginals stay Bayesian (Prop. 1800):

* a **continuous** column (numbers, dates, datetimes) becomes a normal score
  by the randomised probability integral transform: an observation x lies in
  the latent interval (lo, hi] of the marginal (the copula's latent bounds,
  nulls lowest), and its score is Phi^-1(U) with U uniform on (Phi(lo), Phi(hi)].
  Under the marginal the score is exactly N(0, 1) (Ruschendorf 2009), ties
  and point masses included.  A synthetic score z decodes as F^-1(Phi(z)),
  the marginal's quantile function, so the decoded column follows the
  marginal's posterior draw and no observed number is copied: values between
  the marginal's quantile knots are interpolated.
* a **discrete** column (categories, flags, integers coded as levels) becomes
  a one-hot block over its levels, with a level for null when the column has
  nulls and, when it has more than ``max_levels`` levels, one ``other`` level
  for the rare ones.  ``other`` decodes as a draw from the marginal restricted
  to the rare levels, so rare levels keep their marginal frequencies without
  the network memorising them one by one.

Parent attributes (``TableData.aux``) are encoded the same way into the
conditioning vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from ...marginals import Marginal

ZCLIP = 5.0


def marginal_kwargs(cfg: dict, bootstrap: bool) -> dict:
    """The keyword arguments of Marginal.fit used by the copula's table marginal."""
    mcfg, ccfg = cfg["marginals"], cfg["copula"]
    structural = (cfg.get("structure") or {}).get("structural_prior", 0.0)
    return dict(dirichlet_a=mcfg["dirichlet_prior"], grid=ccfg["grid"], bootstrap=bootstrap,
                tail_trim=mcfg["tail_trim"], structural=structural, max_obs=int(mcfg.get("max_obs", 20000)),
                  point_mass_knots=bool(mcfg.get("point_mass_knots", False)))


@dataclass
class _Cat:
    col: int                 # index into data.cols
    keep: np.ndarray         # level codes with a slot of their own
    has_null: bool
    has_other: bool
    lut: np.ndarray          # level code -> slot (or the other slot)
    width: int


class FeatureSpace:
    """Encoding of a table's columns for a network: continuous scores, then one-hot blocks."""

    def __init__(self, data, cols: list[int], max_levels: int, marginals: list):
        self.cols = list(cols)
        self.num: list[int] = []
        self.cats: list[_Cat] = []
        for j in self.cols:
            cd = data.codecs[j]
            if cd.kind == "continuous":
                self.num.append(j)
                continue
            x = np.asarray(data.X[j])
            x = np.where(np.isnan(x), -1, x).astype(np.int64) if x.dtype.kind == "f" else x.astype(np.int64)
            V = len(cd.levels)
            counts = np.bincount(x[x >= 0], minlength=V)[:V]
            seen = np.flatnonzero(counts > 0)
            order = seen[np.argsort(-counts[seen], kind="stable")]
            keep = np.sort(order[:max_levels])
            has_null = bool((x < 0).any())
            has_other = len(order) > max_levels
            off = 1 if has_null else 0
            lut = np.full(max(V, 1), -1, dtype=np.int64)
            lut[keep] = off + np.arange(len(keep))
            width = off + len(keep) + (1 if has_other else 0)
            if has_other:
                lut[lut < 0] = width - 1
            elif len(keep):
                # a level the fit never saw (a rare one dropped by a sample) maps to the most frequent level
                lut[lut < 0] = off + int(np.searchsorted(keep, order[0]))
            self.cats.append(_Cat(j, keep, has_null, has_other, lut, max(width, 1)))
        self.n_num = len(self.num)
        self.cat_widths = [c.width for c in self.cats]
        self.cat_offsets = np.concatenate([[0], np.cumsum(self.cat_widths)]).astype(int) if self.cats \
            else np.zeros(1, int)
        self.n_cat = int(sum(self.cat_widths))
        self.dim = self.n_num + self.n_cat
        self.marg = {j: marginals[j] for j in self.cols}

    # ---- encode
    def scores(self, X: list, j: int, rows, rng) -> np.ndarray:
        """Randomised-PIT normal scores of continuous column j (rows: an index array)."""
        x = np.asarray(X[j])[rows]
        lo, hi = self.marg[j].latent_bounds(x)
        plo, phi = ndtr(lo), ndtr(hi)
        u = plo + (phi - plo) * rng.random(len(x))
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return np.clip(ndtri(u), -ZCLIP, ZCLIP)

    def slots(self, X: list, c: _Cat, rows) -> np.ndarray:
        x = np.asarray(X[c.col])[rows]
        x = np.where(np.isnan(x), -1, x).astype(np.int64) if x.dtype.kind == "f" else x.astype(np.int64)
        if not len(c.lut):
            return np.zeros(len(x), dtype=np.int64)
        s = c.lut[np.clip(x, 0, len(c.lut) - 1)]
        return np.where(x < 0, 0 if c.has_null else s, s)

    def encode(self, X: list, rows, rng) -> tuple[np.ndarray, np.ndarray]:
        """(continuous scores: rows x n_num, category slots: rows x n_cats)."""
        n = len(rows)
        Z = np.column_stack([self.scores(X, j, rows, rng) for j in self.num]) if self.num else np.zeros((n, 0))
        S = np.column_stack([self.slots(X, c, rows) for c in self.cats]) if self.cats \
            else np.zeros((n, 0), dtype=np.int64)
        return Z.astype(np.float32), S.astype(np.int64)

    def one_hot(self, S: np.ndarray) -> np.ndarray:
        out = np.zeros((len(S), self.n_cat), dtype=np.float32)
        for i in range(len(self.cats)):
            out[np.arange(len(S)), self.cat_offsets[i] + S[:, i]] = 1.0
        return out

    def dense(self, X: list, rows, rng) -> np.ndarray:
        """Scores and one-hot blocks side by side (the conditioning vector of parent attributes).

        A missing value in a column that had no nulls at fit time has no slot of its own: it is
        encoded as neutral (score 0, the marginal median; an all-zero one-hot block) rather than
        as the first level or the lowest score."""
        Z, S = self.encode(X, rows, rng)
        H = self.one_hot(S)
        for i, j in enumerate(self.num):
            if float(getattr(self.marg[j], "p_null", 0.0)) <= 0.0:
                miss = np.isnan(np.asarray(X[j], float)[rows])
                Z[miss, i] = 0.0
        for i, c in enumerate(self.cats):
            if not c.has_null:
                x = np.asarray(X[c.col])[rows]
                miss = np.isnan(x) | (x < 0) if x.dtype.kind == "f" else x < 0
                o = int(self.cat_offsets[i])
                H[np.ix_(np.flatnonzero(miss), np.arange(o, o + c.width))] = 0.0
        return np.concatenate([Z, H], axis=1)

    # ---- decode
    def decode_into(self, out: np.ndarray, Z: np.ndarray, S: np.ndarray, marginals: list, rng) -> None:
        """Write encoded engine values of this space's columns into out (rows x len(data.cols)).
        marginals: the marginals to decode with (the current posterior draw)."""
        for i, j in enumerate(self.num):
            u = ndtr(np.asarray(Z[:, i], float))
            out[:, j] = marginals[j].from_uniform(u)
        for i, c in enumerate(self.cats):
            s = np.asarray(S[:, i], np.int64)
            off = 1 if c.has_null else 0
            code = np.full(len(s), -1, dtype=float)
            kept = (s >= off) & (s < off + len(c.keep))
            code[kept] = c.keep[s[kept] - off]
            if c.has_other:
                oth = s == c.width - 1
                if oth.any():
                    p = np.asarray(marginals[c.col].p, float).copy()
                    p[c.keep] = 0.0
                    tot = p.sum()
                    if tot > 0:
                        code[oth] = rng.choice(len(p), size=int(oth.sum()), p=p / tot)
                    else:          # the posterior draw put no mass on the rare levels: fall back to all levels
                        q = np.asarray(marginals[c.col].p, float)
                        code[oth] = rng.choice(len(q), size=int(oth.sum()), p=q / q.sum())
            out[:, c.col] = code


def table_marginals(data, rng, bootstrap: bool) -> list:
    """The table-level marginal of every column (the copula's global marginals)."""
    kw = marginal_kwargs(data.cfg, bootstrap)
    return [Marginal.fit(cd, x, rng, **kw) for cd, x in zip(data.codecs, data.X)]

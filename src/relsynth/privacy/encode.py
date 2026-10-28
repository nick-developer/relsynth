# SPDX-License-Identifier: Apache-2.0
"""Mixed-type encoding and nearest neighbours for the attacks.

Distances are Gower-type (Gower 1971), as in Anonymeter: a numeric column
contributes |x - y| / range, a categorical column contributes 1 when the values
differ.  A missing value against a present one contributes 1; two missing values
are equal.  Ranges are taken over every frame being compared, so the encoding of
a record does not depend on which side it is on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_KINDS = "biufM"


def column_kinds(frames, cols, cat_max_levels: int | None = None) -> dict[str, str]:
    """'num' or 'cat' per column, consistent across frames (numeric only when it
    is numeric in every frame)."""
    out = {}
    for c in cols:
        num = all(f[c].dtype.kind in NUMERIC_KINDS for f in frames)
        if num and cat_max_levels is not None:
            num = pd.concat([f[c] for f in frames]).nunique() > cat_max_levels
        out[c] = "num" if num else "cat"
    return out


class Encoder:
    """Encodes frames into a numeric block (range-scaled) and a code block."""

    def __init__(self, frames, cols, kinds: dict[str, str] | None = None):
        self.cols = list(cols)
        self.kinds = kinds or column_kinds(frames, self.cols)
        self.num = [c for c in self.cols if self.kinds[c] == "num"]
        self.cat = [c for c in self.cols if self.kinds[c] == "cat"]
        self.lo, self.span = {}, {}
        for c in self.num:
            v = pd.concat([_as_float(f[c]) for f in frames])
            lo, hi = np.nanmin(v.to_numpy()) if v.notna().any() else 0.0, np.nanmax(v.to_numpy()) if v.notna().any() else 1.0
            self.lo[c], self.span[c] = float(lo), float(hi - lo) if hi > lo else 1.0
        self.levels = {}
        for c in self.cat:
            v = pd.concat([f[c] for f in frames]).astype(object)
            self.levels[c] = {x: i for i, x in enumerate(pd.unique(v[v.notna()]))}

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = np.column_stack([(_as_float(df[c]).to_numpy() - self.lo[c]) / self.span[c] for c in self.num]) \
            if self.num else np.zeros((len(df), 0))
        C = np.column_stack([df[c].astype(object).map(self.levels[c]).fillna(-1).to_numpy(np.int64)
                             for c in self.cat]) if self.cat else np.zeros((len(df), 0), np.int64)
        return X.astype(float), C


def _as_float(s: pd.Series) -> pd.Series:
    if s.dtype.kind == "M":
        return (s.astype("int64") / 86400e9).where(s.notna())
    return pd.to_numeric(s, errors="coerce").astype(float)


def gower_block(Xq, Cq, Xc, Cc):
    """Distances of each query row to each candidate row (q x n)."""
    D = np.zeros((len(Xq), len(Xc)))
    for j in range(Xq.shape[1]):
        a, b = Xq[:, j][:, None], Xc[:, j][None, :]
        d = np.abs(a - b)
        na, nb = np.isnan(a), np.isnan(b)
        d = np.where(na & nb, 0.0, np.where(na | nb, 1.0, d))
        D += d
    for j in range(Cq.shape[1]):
        a, b = Cq[:, j][:, None], Cc[:, j][None, :]
        D += (a != b)
    return D


def kneighbors(enc: Encoder, queries: pd.DataFrame, candidates: pd.DataFrame, k: int = 1,
               chunk_cells: int = 20_000_000, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Indices (positions in `candidates`) and distances of the k nearest
    candidates of each query.  Ties are broken at random (seeded) so that a
    coarse attribute space does not always favour the first candidate."""
    Xq, Cq = enc.transform(queries)
    Xc, Cc = enc.transform(candidates)
    n, k = len(Xc), min(k, len(Xc))
    rng = rng if rng is not None else np.random.default_rng(0)
    jitter = rng.random(n) * 1e-9
    step = max(1, chunk_cells // max(n * max(len(enc.cols), 1), 1))
    idx = np.empty((len(Xq), k), np.int64)
    dist = np.empty((len(Xq), k))
    for i in range(0, len(Xq), step):
        D = gower_block(Xq[i:i + step], Cq[i:i + step], Xc, Cc) + jitter[None, :]
        part = np.argpartition(D, k - 1, axis=1)[:, :k] if k < n else np.tile(np.arange(n), (len(D), 1))
        dd = np.take_along_axis(D, part, 1)
        o = np.argsort(dd, axis=1)
        idx[i:i + step] = np.take_along_axis(part, o, 1)
        dist[i:i + step] = np.take_along_axis(dd, o, 1) - jitter[idx[i:i + step]]
    return idx, dist

# SPDX-License-Identifier: Apache-2.0
"""Near-copy guard for entity tables (G3; docs/proofs/sec_g3_smalldims.tex, Props. 1720-1723).

A small dimension (a product list of a few hundred rows) is fitted by a few
segments of a few dozen rows each.  Its segment marginals are then close to
the empirical distribution of those rows, and a synthetic row can land closer
to a real row than any other real row is: the case the distance-to-closest-record
check (``dcr_ratio``) flags.

The guard redraws such rows.  A synthetic row ``s`` with nearest real row ``r``
at distance ``d`` is a *near copy* when

* ``0 < d < rho * nn(r)``, where ``nn(r)`` is the distance from ``r`` to its
  nearest real row at a positive distance (``rho`` = ``privacy.near_copy_ratio``), or
* ``d == 0`` and ``r`` has no exact twin among the real rows (a copy of a unique
  record).  A copy of a real row that other real rows share (a coarse attribute
  space, such as components with every optional attribute null) is not a near
  copy: those rows are judged by the exact-match rate.

A near copy is drawn again from its own segment, up to ``privacy.near_copy_rounds``
times; a row that is still a near copy after the last round is kept and counted.

What this does and does not mean (Prop. 1723): after the guard no synthetic row
of the table is closer, in this metric, to a real record than that record's
nearest distinct real neighbour (times rho), unless the rounds ran out, which
the sample log reports.  It is a rule about distances, not a privacy guarantee:
it adds no differential privacy of its own, and when it redraws until it
succeeds an attacker who knows the rule learns that no synthetic row sits inside
a real record's ball (Prop. 1723; Ganev & De Cristofaro 2023 reconstruct
records from such holes).  In DP mode the engine is fitted on the DP seed
database, so the guard reads seed rows and is post-processing of the DP release.
The membership and attribute-inference attacks of ``relsynth privacy`` (T06)
remain the privacy test, and DP mode (``synthesis.privacy.dp``) the guarantee.
The DCR check that reads the same distances is no longer independent evidence
for a guarded table; the scorecard says so.

The metric is the evaluation's (relsynth.evaluate.dcr): rank normal scores
``z, z^2`` and a null flag for numeric columns, one-hot top-20 levels for
categorical ones, over the table's determining columns, except that the scores
are fitted on the real rows only (the synthetic rows do not move the scale).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

MODELLED = ("integer", "float", "categorical", "boolean", "date", "datetime")
TOP_LEVELS = 20


def _is_cat(cs, s):
    """True when the evaluation treats the column as categorical (one-hot)."""
    return cs["type"] in ("categorical", "boolean") or (cs["type"] == "integer" and s.nunique() <= 30)


def _num(s: pd.Series) -> np.ndarray:
    """The column as floats (datetimes as nanoseconds), NaN where missing."""
    if pd.api.types.is_datetime64_any_dtype(s):
        v = s.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        v[s.isna().to_numpy()] = np.nan
        return v
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def nearest(A: np.ndarray, B: np.ndarray, exclude_self: bool = False, positive: bool = False,
            max_cells: int = 4_000_000):
    """(index, distance) of each row of A's nearest row of B.  exclude_self: A is
    B and a row is not its own neighbour (an exact twin of it still is);
    positive: only rows at a distance > 0 count (inf when there is none).
    Distances to the chosen row are recomputed by subtraction, so exact copies
    have distance exactly 0.

    The search runs over the distinct rows of B, in blocks of at most max_cells
    distances (about 32 MB), so memory is bounded whatever the table size; time
    is O(|A| |distinct B| p), which privacy.near_copy_max_rows bounds."""
    n = len(A)
    j_out = np.zeros(n, dtype=np.int64)
    d_out = np.full(n, np.inf)
    if not len(B):
        return j_out, d_out
    U, first, inv, cnt = np.unique(B, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    # a second occurrence of each distinct row (its twin), for exclude_self
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    second = np.where(cnt > 1, order[np.minimum(starts + 1, len(order) - 1)], -1)
    uu = (U * U).sum(1)
    step = max(1, int(max_cells) // max(1, len(U)))
    for i in range(0, n, step):
        a = A[i:i + step]
        rows = np.arange(len(a))
        D = np.maximum((a * a).sum(1)[:, None] + uu[None] - 2 * a @ U.T, 0.0)
        # exact ties computed by subtraction (the expansion above is inexact)
        same = np.zeros_like(D, dtype=bool)
        for q in np.flatnonzero((D < 1e-9).any(1)):
            close = np.flatnonzero(D[q] < 1e-9)
            same[q, close] = ((U[close] - a[q]) ** 2).sum(1) == 0
        if positive:
            D[same] = np.inf
        elif exclude_self:
            own = inv[i:i + len(a)]
            D[rows[cnt[own] == 1], own[cnt[own] == 1]] = np.inf   # a unique row is not its own neighbour
        k = D.argmin(1)
        d = np.sqrt(((a - U[k]) ** 2).sum(1))
        d[~np.isfinite(D[rows, k])] = np.inf
        j = first[k]
        if exclude_self:
            # the chosen row is the row itself: report its twin
            selfpick = j == i + rows
            j[selfpick] = np.where(second[k[selfpick]] >= 0, second[k[selfpick]], j[selfpick])
        j_out[i:i + len(a)], d_out[i:i + len(a)] = j, d
    return j_out, d_out


class NearCopyGuard:
    """The near-copy rule of one table, fitted on its real rows."""

    def __init__(self, real: pd.DataFrame, tspec: dict, cols: list[str], rho: float = 1.0):
        """Fit the encoding on the real rows and store each real row's distance to
        its nearest distinct real row and whether it has an exact twin."""
        self.cols, self.rho = list(cols), float(rho)
        self.enc: list[tuple[str, str, Any]] = []
        for c in self.cols:
            cs, s = tspec["columns"][c], real[c]
            if _is_cat(cs, s):
                self.enc.append(("cat", c, list(s.astype(str).value_counts().index[:TOP_LEVELS])))
            else:
                x = _num(s)
                ok = np.sort(x[~np.isnan(x)])
                self.enc.append(("num", c, ok))
        self.X = self.encode(real)
        _, self.nn = nearest(self.X, self.X, exclude_self=True, positive=True)
        _, d0 = nearest(self.X, self.X, exclude_self=True)
        self.has_twin = d0 == 0
        self.n_real = len(real)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """The rows of df in the distance metric: one-hot top levels for categorical
        columns; normal score, its square and a null flag for numeric ones."""
        feats = []
        for kind, c, ref in self.enc:
            s = df[c]
            if kind == "cat":
                v = s.astype(str).to_numpy()
                feats.append(np.column_stack([(v == t).astype(float) for t in ref]) if ref
                             else np.zeros((len(df), 0)))
            else:
                x = _num(s)
                ok = ~np.isnan(x)
                sc = np.zeros(len(x))
                if ok.any() and len(ref):
                    # mid-rank of x among the real values, as the evaluation's rank / (n + 1)
                    lo = np.searchsorted(ref, x[ok], side="left")
                    hi = np.searchsorted(ref, x[ok], side="right")
                    sc[ok] = ndtri(np.clip(((lo + hi + 1) / 2) / (len(ref) + 1), 1e-6, 1 - 1e-6))
                feats.append(np.column_stack([sc, sc ** 2, (~ok).astype(float)]))
        return np.column_stack(feats) if feats else np.zeros((len(df), 0))

    def near_copies(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean mask of the rows of df that are near copies of a real row."""
        if not len(df) or not self.X.shape[1]:
            return np.zeros(len(df), dtype=bool)
        j, d = nearest(self.encode(df), self.X)
        exact = (d == 0) & ~self.has_twin[j]
        close = (d > 0) & (d < self.rho * self.nn[j])
        return exact | close


def guard_columns(tspec: dict, real: pd.DataFrame, keycols: set, table: str) -> list[str]:
    """The columns the evaluation measures distances on (relsynth.evaluate):
    modelled, not a key, not determined by another categorical."""
    from .evaluate import determining_columns
    cols = [c for c, cs in tspec["columns"].items() if cs["type"] in MODELLED and (table, c) not in keycols
            and c != tspec.get("primary_key") and c in real.columns]
    return determining_columns(real, tspec, cols)

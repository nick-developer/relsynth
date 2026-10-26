# SPDX-License-Identifier: Apache-2.0
"""Exact statistics of a table, computed where the data lives.

The relational part of the model (latent segments, link and block rates, count
laws, participation, inclusion) depends on a fact table only through a few
aggregates of its key columns (Proposition P1 in docs/theory.md):

* the number of rows, and of rows with a non-null key;
* the distinct values of a key (which parent rows are referenced);
* the rows per key tuple: GROUP BY (k1, ..., km) COUNT(*), with nulls kept as a
  group of their own;
* for up to three measures, the sums over each key tuple of the normal scores
  Phi^-1(r / (N + 1)) and of their squares (r the average rank of the value among
  the N non-null values, a null scoring 0): the rank-based features segment
  discovery uses.

`TableStats` is the interface.  `FrameStats` computes it from a pandas frame (the
reference, and the path for files); `relsynth.connectors.sql.SqlStats` computes
the same numbers with GROUP BY and window functions inside the database.  The
test suite checks that both give the same numbers on every backend.
"""
from __future__ import annotations

import abc
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import ndtri

SCORE_PREFIX, SQUARE_PREFIX = "s", "q"


def normal_scores(values: pd.Series) -> np.ndarray:
    """Phi^-1(average rank / (N + 1)) of every non-null value (N non-null values);
    0 for nulls.  Ranks of the raw values: any increasing encoding (dates as day
    numbers, a log scale) gives the same scores."""
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum():
            s = num
    ok = s.notna().to_numpy()
    out = np.zeros(len(s))
    n = int(ok.sum())
    if n:
        r = s[ok].rank(method="average").to_numpy(dtype=float)
        out[ok] = ndtri(r / (n + 1))
    return out


def first_row_differences(sub: pd.DataFrame, keys: list[str], cols: list[str]) -> int:
    """Cells of cols that differ (null-safe) from the same column of the first row
    (in frame order) with the same values of keys: how far the functional
    dependency keys -> cols is from holding."""
    if not cols or not len(sub):
        return 0
    from ..prepare import _keystr
    parts = [_keystr(sub[k]).to_numpy(dtype=object) for k in keys]
    codes = pd.MultiIndex.from_arrays(parts).factorize()[0] if len(parts) > 1 else pd.factorize(parts[0],
                                                                                                use_na_sentinel=False)[0]
    _, first_pos = np.unique(codes, return_index=True)
    lead = first_pos[np.searchsorted(np.unique(codes), codes)]
    diff = 0
    for c in cols:
        a = sub[c].astype(object).to_numpy()
        b = a[lead]
        na, nb = pd.isna(a), pd.isna(b)
        same = (na & nb) | (~na & ~nb & (a == b))
        diff += int((~same).sum())
    return diff


class Require:
    """A row filter of a statistic: col must be non-null and one of the parent's
    keys.  `values` lists the allowed values (used by in-memory statistics, and by
    SQL when the parent is not a stored table); `table`/`column` name the stored
    parent (SQL uses a semi-join), with `exclude` the stored keys the model drops
    (unknown members)."""

    def __init__(self, values=None, table: str | None = None, column: str | None = None, exclude=()):
        self.values = values
        self.table, self.column = table, column
        self.exclude = list(exclude)


class TableStats(abc.ABC):
    """Exact aggregates of one table (see the module docstring)."""

    table: str = ""
    n_rows: int = 0

    @abc.abstractmethod
    def notnull(self, col: str) -> int:
        """Rows where col is not null."""

    @abc.abstractmethod
    def distinct(self, col: str) -> pd.Series:
        """The distinct non-null values of col."""

    @abc.abstractmethod
    def group_counts(self, cols: list[str], scores: tuple[str, ...] | list[str] = (),
                     require: dict[str, Require] | None = None) -> pd.DataFrame:
        """One row per distinct tuple of cols (nulls included as values): the
        columns cols, 'n' (rows), and for the j-th score column 's<j>' and 'q<j>':
        the sums of its normal scores and of their squares over the tuple's rows.
        Scores are ranked over the whole table; `require` keeps only the rows whose
        columns pass their Require filter (applied after ranking)."""

    def counts(self, col: str) -> pd.DataFrame:
        """Rows per value of one column (nulls included)."""
        return self.group_counts([col])

    @abc.abstractmethod
    def first_rows(self, key, cols: list[str], order: list[str] | None = None,
                   differences: bool = True) -> tuple[pd.DataFrame, int | None]:
        """The first row of every value of key (a column or a list of columns):
        columns key and cols, first by `order` (nulls last), then in primary-key
        order when the store has one; and the number of cells of cols that differ
        from their key's first row (None when it cannot be computed or when
        differences is False).  Order
        headers and the opening rows of running balances."""

    @abc.abstractmethod
    def minmax(self, col: str) -> tuple:
        """(smallest, largest) non-null value of col ((None, None) when all null)."""


class FrameStats(TableStats):
    """TableStats of a pandas frame.  `lazy` builds one whose columns are loaded on
    first use (only the columns a statistic needs)."""

    def __init__(self, df: pd.DataFrame | None, table: str = "", n_rows: int | None = None,
                 loader: Callable[[list[str]], pd.DataFrame] | None = None):
        self.table = table
        self._df = df
        self._loader = loader
        self._cache: dict[str, pd.Series] = {}
        self.n_rows = int(len(df) if df is not None else (n_rows or 0))

    @classmethod
    def lazy(cls, loader: Callable[[list[str]], pd.DataFrame], n_rows: int, table: str = "") -> FrameStats:
        return cls(None, table=table, n_rows=n_rows, loader=loader)

    def _cols(self, cols) -> pd.DataFrame:
        if self._df is not None:
            return self._df[list(cols)]
        need = [c for c in cols if c not in self._cache]
        if need:
            got = self._loader(need)
            for c in need:
                self._cache[c] = got[c].reset_index(drop=True)
        return pd.DataFrame({c: self._cache[c] for c in cols})

    def notnull(self, col):
        return int(self._cols([col])[col].notna().sum())

    def distinct(self, col):
        return pd.Series(pd.unique(self._cols([col])[col].dropna()), name=col)

    def group_counts(self, cols, scores=(), require=None):
        cols, scores = list(cols), list(scores)
        require = require or {}
        df = self._cols(list(dict.fromkeys(cols + scores + list(require))))
        out = pd.DataFrame({c: df[c].astype(object).where(df[c].notna(), None) for c in cols})
        extra = {}
        for j, s in enumerate(scores):
            z = normal_scores(df[s])
            extra[f"{SCORE_PREFIX}{j}"] = z
            extra[f"{SQUARE_PREFIX}{j}"] = z ** 2
        out = out.assign(n=1, **extra)
        if require:
            from ..prepare import _keystr
            keep = np.ones(len(out), bool)
            for c, rq in require.items():
                allowed = set(_keystr(pd.Series(list(rq.values))).dropna()) - set(_keystr(pd.Series(rq.exclude)).dropna())
                keep &= _keystr(df[c]).isin(allowed).to_numpy()
            out = out[keep]
        g = out.groupby(cols, dropna=False, sort=False)
        res = g.sum(numeric_only=True).reset_index()
        res["n"] = res["n"].astype(np.int64)
        return res

    def minmax(self, col):
        v = pd.to_numeric(self._cols([col])[col], errors="coerce").dropna()
        return (v.min(), v.max()) if len(v) else (None, None)

    def first_rows(self, key, cols, order=None, differences=True):
        keys = [key] if isinstance(key, str) else list(key)
        cols = [c for c in cols if c not in keys]
        order = [c for c in (order or [])]
        sub = self._cols(list(dict.fromkeys(keys + cols + order)))
        if order:
            from ..profiling import canon
            o = pd.DataFrame({c: pd.to_numeric(pd.Series(canon(sub[c])), errors="coerce").fillna(np.inf)
                              for c in order})
            sub = sub.iloc[np.lexsort([o[c].to_numpy() for c in reversed(order)])].reset_index(drop=True)
        sub = sub[keys + cols]
        H = sub.drop_duplicates(keys).reset_index(drop=True)
        if not differences:
            return H, None
        return H, first_row_differences(sub.reset_index(drop=True), keys, cols)

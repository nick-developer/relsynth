# SPDX-License-Identifier: Apache-2.0
"""Family tables for the context-VAE experiments.

A *family* is the set of child rows that share one parent (the lines of an invoice,
the transactions of an account).  Each experiment dataset names the child view (its
attribute columns, numeric or categorical), the family key, the parent context columns
and, for sequences, the order column.  Every model variant sees the same encoding, so
the comparison isolates the dependence model (see docs in README.md).

Encoding (shared by all variants, close to relsynth's marginal scores):
- numeric: mid-rank normal scores Phi^-1((rank - 1/2) / n), decoded through the
  empirical quantile function of the training rows;
- categorical: the `max_levels` most frequent levels plus one `other` level, decoded
  by drawing a training value of that level (donor draw, as relsynth does).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtri


@dataclass
class DatasetDef:
    name: str
    query: str                    # SQL that returns one row per child row
    family: str                   # family key column
    num: list[str]                # numeric child attributes
    cat: list[str]                # categorical child attributes
    ctx_num: list[str] = field(default_factory=list)   # parent context, numeric
    ctx_cat: list[str] = field(default_factory=list)   # parent context, categorical
    order: str | None = None      # sequence order column (date); None = unordered siblings
    builder: str = ""             # bench.datasets function that builds the SQLite file


DATASETS = {
    # Invoice lines with the attributes of their track: genres and media types co-vary inside an invoice.
    "chinook": DatasetDef(
        "chinook",
        "SELECT il.InvoiceId AS fam, il.Quantity AS Quantity, il.UnitPrice AS UnitPrice, "
        "t.Milliseconds AS Milliseconds, CAST(t.GenreId AS TEXT) AS GenreId, CAST(t.MediaTypeId AS TEXT) AS MediaTypeId, "
        "i.BillingCountry AS BillingCountry "
        "FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId JOIN Invoice i ON i.InvoiceId = il.InvoiceId",
        "fam", ["UnitPrice", "Milliseconds"], ["GenreId", "MediaTypeId", "Quantity"],
        ctx_cat=["BillingCountry"], builder="build_chinook"),
    # Transactions of an account, in date order: the sequential case.
    "berka": DatasetDef(
        "berka",
        "SELECT t.account_id AS fam, t.date AS date, t.trans_id AS tid, t.type AS type, t.operation AS operation, "
        "t.k_symbol AS k_symbol, t.amount AS amount, t.balance AS balance, a.frequency AS frequency "
        "FROM trans t JOIN account a ON a.account_id = t.account_id",
        "fam", ["amount", "balance"], ["type", "operation", "k_symbol"],
        ctx_cat=["frequency"], order="date", builder="build_berka"),
}


def load(d: DatasetDef, path: str) -> pd.DataFrame:
    with sqlite3.connect(path) as con:
        df = pd.read_sql_query(d.query, con)
    sort = ["fam"] + ([d.order, "tid"] if d.order and "tid" in df else [d.order] if d.order else [])
    return df.sort_values(sort, kind="mergesort").reset_index(drop=True)


class Codec:
    """Fitted on training rows; encodes to scores / level indices and decodes back."""

    def __init__(self, df: pd.DataFrame, num: list[str], cat: list[str], max_levels: int = 30):
        self.num, self.cat = list(num), list(cat)
        self.sorted = {c: np.sort(pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)) for c in num}
        self.levels, self.donors = {}, {}
        for c in cat:
            s = df[c].astype(str).where(df[c].notna(), "<null>")
            vc = s.value_counts()
            keep = list(vc.index[:max_levels])
            self.levels[c] = keep + (["<other>"] if len(vc) > max_levels else [])
            self.donors[c] = s[~s.isin(keep)].to_numpy() if len(vc) > max_levels else np.array([])

    @property
    def widths(self) -> list[int]:
        return [len(self.levels[c]) for c in self.cat]

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((len(df), len(self.num)), np.float32)
        for j, c in enumerate(self.num):
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            ref = self.sorted[c]
            lo = np.searchsorted(ref, v, "left")
            hi = np.searchsorted(ref, v, "right")
            p = ((lo + hi) / 2 + 0.5) / (len(ref) + 1)          # mid-rank, strictly inside (0, 1)
            X[:, j] = np.where(np.isnan(v), 0.0, ndtri(p))
        C = np.zeros((len(df), len(self.cat)), np.int64)
        for j, c in enumerate(self.cat):
            s = df[c].astype(str).where(df[c].notna(), "<null>").to_numpy()
            ix = {v: i for i, v in enumerate(self.levels[c])}
            other = ix.get("<other>", 0)
            C[:, j] = [ix.get(v, other) for v in s]
        return X, C

    def decode(self, X: np.ndarray, C: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        from scipy.special import ndtr
        out = {}
        for j, c in enumerate(self.num):
            ref = self.sorted[c]
            q = np.clip(ndtr(X[:, j]), 0, 1)
            out[c] = np.quantile(ref, q, method="inverted_cdf") if len(ref) else np.full(len(X), np.nan)
        for j, c in enumerate(self.cat):
            lv = np.asarray(self.levels[c], dtype=object)
            v = lv[C[:, j]]
            oth = v == "<other>"
            if oth.any() and len(self.donors[c]):
                v[oth] = rng.choice(self.donors[c], oth.sum())
            out[c] = v
        return pd.DataFrame(out)


@dataclass
class Families:
    """Encoded families: row arrays plus offsets (family f = rows off[f]:off[f+1])."""
    keys: np.ndarray
    off: np.ndarray
    X: np.ndarray                 # numeric scores, rows x n_num
    C: np.ndarray                 # level indices, rows x n_cat
    gap: np.ndarray               # log1p(days since the previous row of the family), 0 for the first / unordered
    ctx_X: np.ndarray             # per family
    ctx_C: np.ndarray             # per family

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.off)

    def __len__(self):
        return len(self.keys)


def families(df: pd.DataFrame, d: DatasetDef, codec: Codec, ctx_codec: Codec) -> Families:
    keys, start = np.unique(df["fam"].to_numpy(), return_index=True)
    order = np.argsort(start)
    keys, start = keys[order], start[order]
    off = np.append(start, len(df)).astype(np.int64)
    X, C = codec.encode(df)
    gap = np.zeros(len(df), np.float32)
    if d.order:
        t = pd.to_datetime(df[d.order], errors="coerce").to_numpy("datetime64[D]").astype("int64").astype(float)
        dt = np.diff(t, prepend=t[:1])
        dt[start] = 0.0
        gap = np.log1p(np.clip(dt, 0, None)).astype(np.float32)
    first = df.iloc[start]
    cX, cC = ctx_codec.encode(first)
    return Families(keys, off, X, C, gap, cX, cC)


def split(n: int, rng: np.random.Generator, holdout: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Family-level train / holdout split (privacy checks compare against held-out families)."""
    p = rng.permutation(n)
    k = int(round(n * holdout))
    return np.sort(p[k:]), np.sort(p[:k])


def subset(fam: Families, idx: np.ndarray) -> Families:
    rows = np.concatenate([np.arange(fam.off[i], fam.off[i + 1]) for i in idx]) if len(idx) else np.zeros(0, np.int64)
    sizes = fam.sizes[idx]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    return Families(fam.keys[idx], off, fam.X[rows], fam.C[rows], fam.gap[rows], fam.ctx_X[idx], fam.ctx_C[idx])

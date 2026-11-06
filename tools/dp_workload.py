# SPDX-License-Identifier: Apache-2.0
"""Workload error of a synthetic table: the mean total variation distance over
all one-way and two-way marginals (the error measure of MST and AIM).

Numeric columns with more than `max_levels` distinct values are cut at the
real column's quantiles (`bins` cells, plus null); categorical columns keep
their `max_levels` most frequent real levels and pool the rest.  Only the real
data decides the cells, so every method is measured on the same grid.

    from dp_workload import workload_error
    workload_error(real_df, synth_df)            # {"tvd1": ..., "tvd2": ..., "pairs": n}
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)


def _as_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").astype("int64").where(s.notna()) / 1e9
    return pd.to_numeric(s, errors="coerce").astype(float)


def _canon(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return "<null>"
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x)


class _Coder:
    def __init__(self, real: pd.Series, bins: int, max_levels: int):
        self.num = _is_num(real) and real.nunique(dropna=True) > max_levels
        if self.num:
            x = _as_float(real).dropna().to_numpy()
            e = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1))) if len(x) else np.array([0.0, 1.0])
            self.edges = e[1:-1]
        else:
            vc = real.map(_canon).value_counts()
            self.levels = {v: i for i, v in enumerate(vc.index[:max_levels])}

    def __call__(self, s: pd.Series) -> np.ndarray:
        if self.num:
            x = _as_float(s).to_numpy()
            out = np.searchsorted(self.edges, x, side="right") + 1
            out[np.isnan(x)] = 0
            return out
        k = len(self.levels)
        return s.map(_canon).map(lambda v: self.levels.get(v, k)).to_numpy(dtype=int)


def workload_error(real: pd.DataFrame, synth: pd.DataFrame, columns=None, bins: int = 8, max_levels: int = 20) -> dict:
    cols = [c for c in (columns or real.columns) if c in synth.columns]
    codes_r, codes_s, k = {}, {}, {}
    for c in cols:
        cd = _Coder(real[c], bins, max_levels)
        codes_r[c], codes_s[c] = cd(real[c]), cd(synth[c])
        k[c] = int(max(codes_r[c].max(initial=0), codes_s[c].max(initial=0))) + 1

    def tv(a_r, a_s, size):
        p = np.bincount(a_r, minlength=size) / max(len(a_r), 1)
        q = np.bincount(a_s, minlength=size) / max(len(a_s), 1)
        return 0.5 * float(np.abs(p - q).sum())
    one = [tv(codes_r[c], codes_s[c], k[c]) for c in cols]
    two = [tv(codes_r[a] * k[b] + codes_r[b], codes_s[a] * k[b] + codes_s[b], k[a] * k[b])
           for a, b in itertools.combinations(cols, 2)]
    return {"tvd1": float(np.mean(one)) if one else 0.0, "tvd2": float(np.mean(two)) if two else 0.0,
            "pairs": len(two)}

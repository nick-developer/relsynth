# SPDX-License-Identifier: Apache-2.0
"""Time-series metrics with limits calibrated on real-versus-real splits.

Event sequences (per entity: a customer's orders)
    gap_w1        Wasserstein-1 distance of log(1 + inter-arrival days)
    gap_ks        Kolmogorov-Smirnov distance of the same
    same_day      |difference| of the share of repeat events within a day
    burstiness    W1 of the burstiness B = (sd - mean) / (sd + mean) of each
                  entity's gaps (Goh and Barabasi 2008; n >= 3)
    gap_acf1      |difference| of the lag-1 correlation of consecutive log gaps
    span_w1       W1 of the active span (last - first) / window (n >= 2)
    first_w1      W1 of the first event's position in the window
    weekday_tvd   total variation distance of the day-of-week profile
    month_tvd     TVD of the month-of-year profile (season)
    hour_tvd      TVD of the hour-of-day profile (datetime only)
    trend_tvd     TVD of the share of events per calendar month of the window
    count_gap     |difference| of Spearman(n, mean log gap) over entities

Regular series (a value per series and time step; see series.py)
    acf           mean |difference| of the mean autocorrelation function, lags 1..L
    spectrum      mean |difference| of the log mean periodogram (Welch)
    xcorr_w1      W1 of the pairwise correlations of the series' residuals
    value_w1      W1 of the standardised values
    tstr          forecasting error of a model trained on synthetic series and
                  tested on real ones, relative to the same model trained on real

A metric passes when it is below its limit: the one-sided 99% normal
prediction bound of the metric between two random halves of the real entities
(the null law of "same population"), rescaled from half sizes to the compared
sizes (relsynth.quality.calibrate, Prop. 109).  The splits keep whole entities,
so the dependence of one entity's events is respected (Prop. 1204).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from ..quality.calibrate import kappa, null_summary
from .calendar import to_days

SEQ_METRICS = ("gap_w1", "gap_ks", "same_day", "burstiness", "gap_acf1", "span_w1", "first_w1",
               "weekday_tvd", "month_tvd", "hour_tvd", "trend_tvd", "count_gap")


class Events:
    """Events of a population of entities: times (days) and owner ids, sorted."""

    def __init__(self, t, owner, t0=None, t1=None):
        t = np.asarray(t, dtype=float)
        owner = np.asarray(pd.Series(owner).astype(str).to_numpy(dtype=object))
        ok = np.isfinite(t) & (pd.Series(owner).notna().to_numpy())
        t, owner = t[ok], owner[ok]
        codes, uniq = pd.factorize(owner, sort=False)
        o = np.lexsort((t, codes))
        self.t, self.o = t[o], codes[o]
        self.ids = np.asarray(uniq, dtype=object)
        self.t0 = float(np.floor(self.t.min())) if t0 is None and len(self.t) else t0
        self.t1 = float(np.floor(self.t.max()) + 1) if t1 is None and len(self.t) else t1

    def subset(self, mask_entities):
        keep = mask_entities[self.o]
        e = Events.__new__(Events)
        e.t, e.ids, e.t0, e.t1 = self.t[keep], self.ids[mask_entities], self.t0, self.t1
        remap = np.cumsum(mask_entities) - 1
        e.o = remap[self.o[keep]]
        return e

    @property
    def n_entities(self):
        return len(self.ids)


def _gaps(ev):
    same = ev.o[1:] == ev.o[:-1]
    return np.diff(ev.t)[same], ev.o[1:][same]


def _per_entity(ev):
    n = np.bincount(ev.o, minlength=ev.n_entities)
    first = np.full(ev.n_entities, np.nan)
    last = np.full(ev.n_entities, np.nan)
    if len(ev.t):
        idx = np.flatnonzero(np.r_[True, ev.o[1:] != ev.o[:-1]])
        first[ev.o[idx]] = ev.t[idx]
        idx2 = np.flatnonzero(np.r_[ev.o[1:] != ev.o[:-1], True])
        last[ev.o[idx2]] = ev.t[idx2]
    return n, first, last


def _tvd(a, b):
    a = a / max(a.sum(), 1e-12)
    b = b / max(b.sum(), 1e-12)
    return float(0.5 * np.abs(a - b).sum())


def _profiles(ev, datetime):
    d = pd.to_datetime(np.floor(ev.t).astype("int64"), unit="D")
    out = {"weekday": np.bincount(np.asarray(d.dayofweek), minlength=7).astype(float),
           "month": np.bincount(np.asarray(d.month) - 1, minlength=12).astype(float)}
    ym = np.asarray(d.year) * 12 + np.asarray(d.month) - 1
    out["ym"] = pd.Series(ym).value_counts()
    if datetime:
        h = np.floor((ev.t - np.floor(ev.t)) * 24).astype(np.int64).clip(0, 23)
        out["hour"] = np.bincount(h, minlength=24).astype(float)
    return out


def _entity_stats(ev):
    g, go = _gaps(ev)
    lg = np.log1p(np.maximum(g, 0))
    n, first, last = _per_entity(ev)
    # burstiness per entity with >= 2 gaps
    s1 = np.bincount(go, weights=g, minlength=ev.n_entities)
    s2 = np.bincount(go, weights=g * g, minlength=ev.n_entities)
    c = np.bincount(go, minlength=ev.n_entities)
    m3 = c >= 2
    mu = s1[m3] / c[m3]
    sd = np.sqrt(np.maximum(s2[m3] / c[m3] - mu ** 2, 0))
    burst = (sd - mu) / np.maximum(sd + mu, 1e-12)
    # lag-1 pairs of consecutive gaps of one entity
    pair = go[1:] == go[:-1]
    acf_x, acf_y = lg[:-1][pair], lg[1:][pair]
    mlg = np.bincount(go, weights=lg, minlength=ev.n_entities) / np.maximum(c, 1)
    L = max(ev.t1 - ev.t0, 1e-9)
    return {"lg": lg, "same": g < 1.0, "burst": burst, "acf": (acf_x, acf_y),
            "span": ((last - first) / L)[n >= 2], "first": ((first - ev.t0) / L)[n >= 1],
            "n": n[c >= 1], "mlg": mlg[c >= 1]}


def _corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _w1(a, b):
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else np.nan


def sequence_distances(a: Events, b: Events, datetime=False) -> dict:
    """Every sequence metric between two event populations."""
    sa, sb = _entity_stats(a), _entity_stats(b)
    pa, pb = _profiles(a, datetime), _profiles(b, datetime)
    ym = pa["ym"].index.union(pb["ym"].index)
    out = {
        "gap_w1": _w1(sa["lg"], sb["lg"]),
        "gap_ks": float(ks_2samp(sa["lg"], sb["lg"]).statistic) if len(sa["lg"]) and len(sb["lg"]) else np.nan,
        "same_day": abs(float(sa["same"].mean() if len(sa["same"]) else 0) - float(sb["same"].mean() if len(sb["same"]) else 0)),
        "burstiness": _w1(sa["burst"], sb["burst"]),
        "gap_acf1": abs(_corr(*sa["acf"]) - _corr(*sb["acf"])),
        "span_w1": _w1(sa["span"], sb["span"]),
        "first_w1": _w1(sa["first"], sb["first"]),
        "weekday_tvd": _tvd(pa["weekday"], pb["weekday"]),
        "month_tvd": _tvd(pa["month"], pb["month"]),
        "trend_tvd": _tvd(pa["ym"].reindex(ym, fill_value=0).to_numpy(float),
                          pb["ym"].reindex(ym, fill_value=0).to_numpy(float)),
        "count_gap": abs(_spear(sa) - _spear(sb)),
    }
    if datetime:
        out["hour_tvd"] = _tvd(pa["hour"], pb["hour"])
    return out


def _spear(s):
    if len(s["n"]) < 5 or np.std(s["n"]) == 0:
        return 0.0
    r = spearmanr(s["n"], s["mlg"]).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def calibrate(real: Events, n_other: int, datetime=False, splits=20, confidence=0.99, seed=0) -> dict:
    """Null limits of every metric for comparing `real` with a population of
    n_other entities: splits of the real entities into halves."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {}
    N = real.n_entities
    for _ in range(splits):
        m = np.zeros(N, bool)
        m[rng.permutation(N)[: N // 2]] = True
        d = sequence_distances(real.subset(m), real.subset(~m), datetime)
        for k, v in d.items():
            draws.setdefault(k, []).append(v)
    k = kappa((N, n_other), (N // 2, N - N // 2))
    return {name: null_summary(v, 0.0, confidence, k, log=True) for name, v in draws.items()}


def sequence_checks(real: Events, syn: Events, datetime=False, splits=20, confidence=0.99, seed=0) -> list[dict]:
    """[{metric, value, limit, pass, null}] for one event table."""
    d = sequence_distances(real, syn, datetime)
    lim = calibrate(real, syn.n_entities, datetime, splits, confidence, seed)
    out = []
    for name, v in d.items():
        L = lim[name]["limit"]
        out.append({"metric": name, "value": None if not np.isfinite(v) else round(float(v), 6),
                    "limit": None if L is None else round(float(L), 6),
                    "pass": None if L is None or not np.isfinite(v) else bool(v <= L), "null": lim[name]})
    return out


def table_events(df, entity, time) -> Events:
    return Events(to_days(df[time]), df[entity])


def sequence_report(spec, real: dict, syn: dict, splits=20, confidence=0.99, seed=0) -> dict:
    """Checks of every table with a `sequence` block of a (1.0) spec, on the
    tables as written (header columns on the lines are deduplicated by the
    header key)."""
    rep = {}
    for t, T in spec["tables"].items():
        sq = T.get("sequence")
        if not sq or t not in real or t not in syn:
            continue
        ent, tc = sq["entity"], sq["time"]
        hdr = (T.get("fact") or {}).get("header") or {}
        r, s = real[t], syn[t]
        if hdr.get("key") and tc in (hdr.get("columns") or []):
            r = r.drop_duplicates(hdr["key"])
            s = s.drop_duplicates(hdr["key"])
        is_dt = T["columns"][tc].get("type") == "datetime"
        if T["columns"][tc].get("type") == "lookup":
            v = pd.to_datetime(r[tc], errors="coerce")
            is_dt = bool(((v - v.dt.normalize()).dt.total_seconds() > 0).any())
        er, es = table_events(r, ent, tc), table_events(s, ent, tc)
        es.t0, es.t1 = er.t0, er.t1
        checks = sequence_checks(er, es, is_dt, splits, confidence, seed)
        rep[t] = {"entity": ent, "time": tc, "real_entities": er.n_entities, "synthetic_entities": es.n_entities,
                  "real_events": len(er.t), "synthetic_events": len(es.t),
                  "passed": sum(1 for c in checks if c["pass"]), "total": sum(1 for c in checks if c["pass"] is not None),
                  "checks": checks}
    return rep

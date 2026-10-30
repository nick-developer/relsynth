# SPDX-License-Identifier: Apache-2.0
"""Incremental synthesis: extend a synthetic history day by day.

Change-data-capture tests need a database that keeps living: yesterday's
synthetic orders plus today's new ones, the same customers ordering again at a
realistic pace.  ``extend`` continues every table with a ``sequence`` block
past the end of a synthetic dataset (its *base*) and returns the new rows of
each step (a day by default) as deltas.

Forward model (per entity of segment k, in operational time, see
relsynth.timeseries.calendar extended with the seasonal profile of the last
observed year)

* rate      r_i ~ Gamma(a_k + n_i, b_k + U_i): the posterior of the entity's
            rate given its n_i base events over exposure U_i, under the
            population law Gamma(a_k, b_k) fitted to the real counts;
* rhythm    alpha_i ~ LogNormal(mu_k, sigma_k^2) (the sequence model's law);
* events    a gamma renewal process with shape alpha_i and mean gap 1 / r_i;
            the first new gap is drawn given that it is longer than the time
            already elapsed since the entity's last event (its age), so a
            customer who ordered yesterday is not treated as one who has been
            away for a year (the inspection paradox; Prop. 1206).

Every random number is a function of (seed, entity key, event number,
purpose), computed by a counter-based hash, not drawn from a stream.  So the
events up to a day D do not depend on how far or in how many steps the
history is extended: extending to D in one run, or day by day, gives the same
rows with the same keys (Prop. 1206).

The contents of a new event (its products, store, amounts) are those of a
random event of an entity of the same segment in the base, with new keys, the
entity's key and its dates moved by the same number of days; columns looked
up through the entity are read again, check rules are tested and a template
that breaks one is replaced (up to 10 times, else the event is dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaincc, gammainccinv, gammaincinv, ndtri

from .calendar import from_days, to_days

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(x):
    """splitmix64 finaliser on uint64 arrays (Steele et al. 2014)."""
    with np.errstate(over="ignore"):
        x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hash_uniform(seed, key_hash, j, purpose):
    """Uniform(0, 1) numbers, one per (entity hash, event number), for a purpose."""
    with np.errstate(over="ignore"):
        h = _mix(np.uint64(seed) * np.uint64(1000003) + np.uint64(purpose))
        h = _mix(h ^ np.asarray(key_hash, dtype=np.uint64))
        h = _mix(h ^ (np.asarray(j, dtype=np.uint64) * np.uint64(0x632BE59BD9B4E019)))
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53


def key_hash(keys):
    """Stable 64-bit hashes of key values (the same in every run)."""
    from ..engine import _keystr
    return pd.util.hash_array(_keystr(pd.Series(keys)).fillna("\x00").to_numpy(dtype=object)).astype(np.uint64)


def forward_events(model, seg, n, U, u_last, u_start, u_end, keys_h, seed, B_u=None, max_events=10000):
    """Operational times of the new events of each entity in [u_start, u_end).

    model: a fitted SequenceModel; seg, n, U: segment, base event count and base
    exposure of each entity; u_last: operational time of its last base event
    (NaN without one).  Returns (entity index, event number, u)."""
    a, b = model.rate_ab[seg, 0], model.rate_ab[seg, 1]
    th = model.theta[seg]
    r = gammaincinv(a + n, hash_uniform(seed, keys_h, 0, 1)) / (b + U)
    alpha = np.exp(np.clip(th[:, 4] + np.exp(th[:, 5]) * ndtri(hash_uniform(seed, keys_h, 0, 2)), -12, 8))
    beta = alpha * np.maximum(r, 1e-12)                  # gap ~ Gamma(alpha, rate beta): mean 1 / r
    has = np.isfinite(u_last)
    # first new event: a gap longer than the entity's age, or (no event yet) an
    # exponential wait from the start
    u1 = hash_uniform(seed, keys_h, 0, 3)
    age = np.where(has, np.maximum(u_start - u_last, 0), 0.0)
    q = gammaincc(alpha, beta * age)
    with np.errstate(all="ignore"):
        g = gammainccinv(alpha, np.clip(u1 * q, 1e-300, 1)) / beta
    g = np.where(np.isfinite(g) & (g >= age), g, age + 1.0 / np.maximum(r, 1e-12) * -np.log(u1))
    nxt = np.where(has, u_last + g, u_start - np.log(u1) / np.maximum(r, 1e-12))
    lim = np.full(len(seg), u_end) if B_u is None else np.minimum(u_end, B_u)
    out_e, out_j, out_u = [], [], []
    live = np.flatnonzero(nxt < lim)
    j = 1
    while len(live) and j <= max_events:
        out_e.append(live)
        out_j.append(np.full(len(live), j))
        out_u.append(nxt[live])
        uu = hash_uniform(seed, keys_h[live], j, 4)
        nxt[live] = nxt[live] + gammaincinv(alpha[live], uu) / beta[live]
        live = live[nxt[live] < lim[live]]
        j += 1
    if not out_e:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(out_e), np.concatenate(out_j), np.concatenate(out_u)


def _next_keys(existing, n, like=None):
    """n new key values after the existing ones: integers continue from the
    maximum; strings with a numeric suffix (SO43697) continue that number."""
    s = pd.Series(existing).dropna()
    if not len(s):
        return np.arange(1, n + 1)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return np.arange(int(num.max()) + 1, int(num.max()) + 1 + n)
    m = s.astype(str).str.extract(r"^(.*?)(\d+)$")
    if m[1].notna().all():
        pre = m[0].mode().iloc[0]
        width = int(m[1].str.len().max())
        start = int(m[1].astype(np.int64).max()) + 1
        return np.array([f"{pre}{str(start + i).zfill(width)}" for i in range(n)], dtype=object)
    return np.array([f"X{i:012d}" for i in range(n)], dtype=object)


class Extender:
    """Continues the sequence tables of one synthetic dataset.

    model: relsynth.FittedModel with at least one ``sequence`` table; base: the
    dataset (SyntheticData or dict); segments: {entity table: {key: segment}}
    of the base's entities (None: recovered by sampling the dataset again,
    which is deterministic, when its keys match; else pooled parameters)."""

    def __init__(self, model, base, dataset=0, segments=None):
        self.model = model
        S = model.synthesizer
        if not getattr(S, "seq", None):
            raise ValueError("the spec has no table with a sequence block; nothing to extend")
        self.S = S
        self.user = model.spec
        self.base = {t: df.reset_index(drop=True) for t, df in dict(base).items()}
        self.dataset = int(dataset)
        seed = S.cfg["seed"] if S.cfg["posterior"].get("sample_seed") is None else S.cfg["posterior"]["sample_seed"]
        self.seed = int(seed) * 7919 + 1206 + self.dataset
        self.segments = segments if segments is not None else self._segments()
        self.log = {}

    def _segments(self):
        S = self.S
        zs = getattr(S, "_zs", None)
        out_int = getattr(S, "internal_out", None)
        if zs is None or out_int is None or S._runs != self.dataset + 1:
            self.model.sample(dataset=self.dataset)
            zs, out_int = S._zs, S.internal_out
        from ..engine import _keystr
        res = {}
        for info in S.seq.values():
            P = info["parent"]
            if P in zs and P in out_int:
                pk = S.spec["tables"][P]["primary_key"]
                keys = _keystr(out_int[P][pk]).to_numpy(dtype=object)
                if len(keys) == len(zs[P]):
                    res[P] = dict(zip(keys, np.asarray(zs[P], dtype=np.int64)))
        return res

    # ------------------------------------------------------------ the run
    def _user_table(self, internal_t):
        """(user table, header key or None) for an internal sequence table."""
        for f, h in self.S.plan.headers.items():
            if h["table"] == internal_t:
                return f, h["key"]
        return internal_t, None

    def run(self, until, start=None, step="day"):
        """[(period start, period end, {table: new rows})] for every step from
        start (default: the end of the source's time range) to until."""
        from ..engine import _keystr
        deltas: dict = {}
        T = self.user["tables"]
        tables = dict(self.base)
        for it, info in self.S.seq.items():
            m = info["model"]
            t, hkey = self._user_table(it)
            P, ent, tc = info["parent"], info["entity"], info["time"]
            if t not in tables:
                continue
            d0 = m.t1 if start is None else float(to_days(pd.Series([start]))[0])
            d1 = float(to_days(pd.Series([until]))[0])
            if d1 <= d0:
                continue
            m.intensity.extend_to(d1 + 1)
            ev = tables[t] if hkey is None else tables[t].drop_duplicates(hkey)
            pdf = tables[P]
            ppk = T[P]["primary_key"]
            pkeys = _keystr(pdf[ppk]).to_numpy(dtype=object)
            segmap = self.segments.get(P, {})
            seg = np.array([segmap.get(k, 0) for k in pkeys], dtype=np.int64).clip(0, m.K - 1)
            pos = pd.Index(pkeys).get_indexer(_keystr(ev[ent]).to_numpy(dtype=object))
            tt = to_days(ev[tc])
            ok = (pos >= 0) & np.isfinite(tt)
            n = np.bincount(pos[ok], minlength=len(pdf)).astype(float)
            last = np.full(len(pdf), np.nan)
            np.fmax.at(last, pos[ok], tt[ok] + (0.0 if info["datetime"] else 0.5))
            from .hooks import entity_window
            A, B = entity_window(info["bounds"], pdf, info["datetime"])
            A = np.where(np.isfinite(A), np.maximum(A, m.t0), m.t0)
            Bw = np.where(np.isfinite(B), np.minimum(B, m.t1), m.t1)
            U = np.maximum(m.intensity.forward(Bw, seg) - m.intensity.forward(A, seg), 1e-9)
            u_last = np.where(np.isfinite(last), m.intensity.forward(np.nan_to_num(last), seg), np.nan)
            us, ue = m.intensity.forward(np.full(len(pdf), d0), seg), m.intensity.forward(np.full(len(pdf), d1), seg)
            B_u = np.where(np.isfinite(B), m.intensity.forward(np.nan_to_num(B, nan=d1), seg), np.inf)
            kh = key_hash(pdf[ppk])
            e, j, u = forward_events(m, seg, n, U, u_last, us, ue, kh, self.seed, B_u=B_u)
            new_t = m.intensity.inverse(u, seg[e])
            new_t = new_t if info["datetime"] else np.floor(new_t)
            keep = (new_t >= d0) & (new_t < d1)
            e, j, new_t = e[keep], j[keep], new_t[keep]
            order = np.lexsort((j, kh[e], new_t))
            e, j, new_t = e[order], j[order], new_t[order]
            rows = self._materialise(t, hkey, info, tables, ev, pos, ok, seg, e, j, new_t, kh)
            for c, df in rows.items():
                deltas.setdefault(c, []).append(df)
            self.log[t] = {"entities": int(len(pdf)), "new_events": int(len(e)),
                           "dropped_by_rules": int(self._dropped)}
        new = {c: pd.concat(v, ignore_index=True) for c, v in deltas.items()}
        new = self._views(new, tables)
        return self._split(new, start, until, step)

    def _materialise(self, t, hkey, info, tables, ev, pos, ok, seg, e, j, new_t, kh):
        """Rows of the new events: a template event of the same segment, new keys,
        the entity, dates moved, entity lookups read again, rules tested."""
        from ..engine import _keystr
        from ..expr import check_mask
        from ..prepare import lookup_column, make_resolver
        T = self.user["tables"]
        ent = info["entity"]
        self._dropped = 0
        evk = ev.reset_index(drop=True)
        by_seg = {k: np.flatnonzero(ok & (seg[np.maximum(pos, 0)] == k)) for k in np.unique(seg)}
        allidx = np.flatnonzero(ok)
        pdf = tables[info["parent"]]
        ppk = T[info["parent"]]["primary_key"]
        own_key = hkey or T[t].get("primary_key")
        src = tables[t]
        children = [r for r in self.user["relationships"] if r.get("parent") == t and r.get("kind", "many_to_one")
                    == "many_to_one" and not isinstance(r.get("child_key"), list) and r["child"] in tables
                    and T[r["child"]].get("role") != "view"] if hkey is None else []
        rules = [r for r in self.user["rules"] if r["table"] == t and r["type"] == "check"]
        cal = info["calendar"]
        cal_map = None
        if cal is not None and cal in tables:
            cpk = T[cal]["primary_key"]
            cd = tables[cal]
            cal_days = to_days(cd[info["calendar_col"]])
            cal_map = (pd.Series(cal_days, index=_keystr(cd[cpk]).to_numpy(dtype=object)),
                       pd.Series(cd[cpk].to_numpy(dtype=object), index=np.floor(cal_days)))
            cal_map = (cal_map[0], cal_map[1][~cal_map[1].index.duplicated()])
            date_fks = [r["child_key"] for r in self.user["relationships"] if r["child"] == t and r["parent"] == cal]
        else:
            date_fks = []
        date_cols = [c for c, cs in T[t]["columns"].items() if cs.get("type") in ("date", "datetime")]
        date_look = [c for c, cs in T[t]["columns"].items() if cs.get("type") == "lookup"
                     and isinstance(cs.get("lookup"), str) and cs["lookup"].split(".")[0] in date_fks]
        ent_look = [c for c, cs in T[t]["columns"].items() if cs.get("type") == "lookup"
                    and isinstance(cs.get("lookup"), str) and cs["lookup"].split(".")[0] == ent]
        new_keys = _next_keys(src[own_key], len(e)) if own_key else None
        tmpl = np.zeros(len(e), np.int64)
        good = np.zeros(len(e), bool)
        segs_e = seg[e]
        for attempt in range(11):
            u = hash_uniform(self.seed, kh[e], j, 10 + attempt)
            for k in np.unique(segs_e):
                pool = by_seg.get(k)
                pool = pool if pool is not None and len(pool) else allidx
                mk = ~good & (segs_e == k)
                tmpl[mk] = pool[np.minimum((u[mk] * len(pool)).astype(np.int64), len(pool) - 1)]
            rows = self._build(t, hkey, info, evk, src, tmpl, e, new_t, new_keys, own_key, pdf, ppk,
                               date_cols, date_fks, date_look, cal_map)
            res = make_resolver(self.user, t, rows, {**tables, t: rows})
            for c in ent_look:
                rows[c] = lookup_column(self.user, t, c, rows, {**tables, t: rows}, res)
            okr = np.ones(len(rows), bool)
            for r in rules:
                okr &= check_mask(rows, r["expr"], make_resolver(self.user, t, rows, {**tables, t: rows}))
            bad_ev = np.zeros(len(e), bool)
            bad_ev[np.unique(rows["__event__"].to_numpy()[~okr])] = True
            good = ~bad_ev
            if good.all():
                break
        keep_ev = good
        self._dropped = int((~keep_ev).sum())
        rows = rows[keep_ev[rows["__event__"].to_numpy()]].drop(columns="__event__").reset_index(drop=True)
        out = {t: rows}
        if children and own_key:
            sel = np.flatnonzero(keep_ev)
            tk = _keystr(evk[own_key]).to_numpy(dtype=object)[tmpl[sel]]
            for r in children:
                c = r["child"]
                cdf = tables[c]
                ck = _keystr(cdf[r["child_key"]]).to_numpy(dtype=object)
                grp = pd.Series(np.arange(len(cdf))).groupby(ck).apply(list).to_dict()
                parts, keys = [], []
                for i, k in zip(sel, tk):
                    ix = grp.get(k, [])
                    if ix:
                        parts.extend(ix)
                        keys.extend([new_keys[i]] * len(ix))
                if not parts:
                    continue
                cr = cdf.iloc[parts].reset_index(drop=True).copy()
                cr[r["child_key"]] = np.asarray(keys, dtype=object)
                cpk = T[c].get("primary_key")
                if cpk and cpk != r["child_key"]:
                    cr[cpk] = _next_keys(cdf[cpk], len(cr))
                out[c] = cr
        return out

    def _build(self, t, hkey, info, evk, src, tmpl, e, new_t, new_keys, own_key, pdf, ppk,
               date_cols, date_fks, date_look, cal_map):
        from ..engine import _keystr
        tc = info["time"]
        tdays = to_days(evk[tc]).copy()
        shift = np.floor(new_t) - np.floor(tdays[tmpl]) if not info["datetime"] else new_t - tdays[tmpl]
        if hkey is None:
            rows = evk.iloc[tmpl].reset_index(drop=True).copy()
            ev_of_row = np.arange(len(tmpl))
        else:
            hk = _keystr(src[hkey]).to_numpy(dtype=object)
            grp = pd.Series(np.arange(len(src))).groupby(hk).apply(list).to_dict()
            tk = _keystr(evk[hkey]).to_numpy(dtype=object)[tmpl]
            ix, ev_of_row = [], []
            for i, k in enumerate(tk):
                g = grp.get(k, [])
                ix.extend(g)
                ev_of_row.extend([i] * len(g))
            rows = src.iloc[ix].reset_index(drop=True).copy()
            ev_of_row = np.asarray(ev_of_row, dtype=np.int64)
        rows["__event__"] = ev_of_row
        if own_key:
            rows[own_key] = np.asarray(new_keys, dtype=object)[ev_of_row]
        rows[info["entity"]] = pdf[ppk].to_numpy(dtype=object)[e[ev_of_row]]
        sh = shift[ev_of_row]
        for c in date_cols:
            is_dt = self.user["tables"][t]["columns"][c]["type"] == "datetime"
            v = pd.to_datetime(pd.Series(from_days(to_days(rows[c]) + sh, is_dt)))
            rows[c] = v if is_dt else v.dt.normalize()
        if cal_map is not None:
            k2d, d2k = cal_map
            for c in date_fks:
                dv = k2d.reindex(_keystr(rows[c]).to_numpy(dtype=object)).to_numpy(dtype=float) + sh
                rows[c] = d2k.reindex(np.floor(dv)).to_numpy(dtype=object)
            for c in date_look:
                v = to_days(rows[c]) + sh
                rows[c] = pd.to_datetime(from_days(v, False))
        return rows

    def _views(self, new, tables):
        """View rows of the new base rows (Sales over OrderRows)."""
        from ..prepare import lookup_column
        U = self.user["tables"]
        for v, T in U.items():
            if T.get("role") != "view" or T.get("view_of") not in new:
                continue
            base = new[T["view_of"]].reset_index(drop=True)
            res = {**tables, **{k: pd.concat([tables.get(k, pd.DataFrame()), d], ignore_index=True)
                                for k, d in new.items()}}
            df = pd.DataFrame(index=base.index)
            for c, cs in T["columns"].items():
                bc = cs.get("base_column", c)
                if cs["type"] != "lookup" and bc in base.columns:
                    df[c] = base[bc].to_numpy(dtype=object)
            for c, cs in T["columns"].items():
                if cs["type"] == "lookup":
                    try:
                        df[c] = lookup_column(self.user, v, c, df, res)
                    except KeyError:
                        df[c] = None
            new[v] = df[[c for c in T["columns"] if c in df.columns]]
        return new

    def _split(self, new, start, until, step):
        """The new rows per step: [(start, end, {table: rows})]."""
        freq = {"day": "D", "week": "W-MON", "month": "MS", "hour": "h"}.get(step, step)
        tcol = {}
        for it, info in self.S.seq.items():
            t, _ = self._user_table(it)
            tcol[t] = info["time"]
        times = {t: pd.to_datetime(new[t][c], errors="coerce") for t, c in tcol.items() if t in new}
        if not times:
            return []
        lo = min(s.min() for s in times.values() if len(s)) if any(len(s) for s in times.values()) else None
        if lo is None:
            return []
        lo = pd.Timestamp(start) if start is not None else lo.normalize()
        edges = pd.date_range(lo.floor("D"), pd.Timestamp(until), freq=freq)
        edges = edges.append(pd.DatetimeIndex([pd.Timestamp(until)])).unique().sort_values()
        if edges[0] > lo:
            edges = pd.DatetimeIndex([lo]).append(edges)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            part = {}
            for t, s in times.items():
                m = ((s >= a) & (s < b)).to_numpy()
                part[t] = new[t][m].reset_index(drop=True)
                key = [r for r in self.user["relationships"] if r.get("parent") == t and r["child"] in new]
                for r in key:
                    pk = r["parent_key"]
                    if isinstance(pk, str) and pk in part[t].columns:
                        sel = new[r["child"]][r["child_key"]].isin(set(part[t][pk])).to_numpy()
                        part[r["child"]] = new[r["child"]][sel].reset_index(drop=True)
            for v, d in new.items():
                if v not in part:
                    base_t = self.user["tables"][v].get("view_of")
                    if base_t in part:
                        bk = [c for c in d.columns if c in part[base_t].columns]
                        key = bk[0] if bk else None
                        part[v] = d[d[key].isin(set(part[base_t][key]))].reset_index(drop=True) if key else d
            out.append((a, b, part))
        return out


def extend(model, until, *, base=None, dataset=0, start=None, step="day"):
    """Deltas that continue a synthetic dataset to `until`; see Extender."""
    if base is None:
        base = model.sample(dataset=dataset)
    return Extender(model, base, dataset).run(until, start=start, step=step)

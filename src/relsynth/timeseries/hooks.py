# SPDX-License-Identifier: Apache-2.0
"""The sequence model inside the engine: fitted after the relational model,
applied to a table right after its rows are finished.

A table declares ``sequence: {entity: <foreign key>, time: <column>}``.  The
engine first generates the table as in 1.0 (how many events each entity has,
what each event holds, every key and rule); the sequence model then redraws
the times of each entity's events jointly from the fitted history model and
writes them back in the order of the engine's own times (a rank-preserving
coupling, so an event the engine put late stays late).  Columns computed from
the time (an offset delivery date, a date key, a lookup) are recomputed.  An
entity whose new times break a check rule that held before is redrawn up to
``redraws`` times and otherwise keeps the engine's times, so every rule that
held still holds (Prop. 1203).

Everything here draws from its own random stream, so the other tables and
columns of a dataset are the same as without the sequence model.
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

from .calendar import from_days, to_days
from .events import SequenceModel

DEFAULTS = {"window": "auto", "bucket": "month", "weekly": True, "hourly": "auto", "shrinkage": 1.0,
            "min_entities": 30, "redraws": 10, "calendar": "auto", "calendar_draw": False,
            "calendar_bf": 100.0, "process": "renewal", "states": None}
NULL_LEVEL = "\x00null"


def state_options(sq):
    """{column, prior} of sequence.states (a column name or a mapping), or None."""
    st = (sq or {}).get("states")
    if st is None:
        return None
    if isinstance(st, str):
        return {"column": st, "prior": 1.0}
    return {"prior": 1.0, **st}


def _state_keys(values):
    """The level key of every status value: its text, or NULL_LEVEL for nulls."""
    v = pd.Series(values, dtype=object)
    return np.where(v.isna().to_numpy(), NULL_LEVEL, v.astype(str).to_numpy()).astype(object)


def _state_codes(values, levels):
    """Codes 0..L-1 of status values in the fitted levels (-1: unseen)."""
    return pd.Index(levels).get_indexer(_state_keys(values))
_CMP = re.compile(r"^\s*(?:days\()?\s*([\w.]+)\s*\)?\s*(<=|<)\s*(?:days\()?\s*([\w.]+)\s*\)?\s*$")


def options(sq: dict) -> dict:
    return {**DEFAULTS, **(sq or {})}


def _entity_rel(S, t, col):
    r = next((r for r in S.fk[t] if r["child_key"] == col), None)
    if r is None:
        raise KeyError(f"{t}: sequence.entity {col} is not a foreign key")
    return r


def window_bounds(spec, t, sq, ent_col):
    """[(side, column of the entity, strict)] from sequence.window: explicit
    {start, end} columns, or (auto) the check rules of t that compare the time
    with a column of the entity (days(OrderDate) < days(CustomerKey.EndDT))."""
    o = options(sq)
    w = o["window"]
    tc = o["time"]
    out = []
    if isinstance(w, dict):
        if w.get("start"):
            out.append(("start", w["start"], False))
        if w.get("end"):
            out.append(("end", w["end"], False))
        return out
    if w != "auto":
        return out
    pre = ent_col + "."
    for r in spec["rules"]:
        if r["table"] != t or r["type"] != "check":
            continue
        m = _CMP.match(str(r["expr"]))
        if not m:
            continue
        a, op, b = m.groups()
        if a == tc and b.startswith(pre) and "." not in b[len(pre):]:
            out.append(("end", b[len(pre):], op == "<"))
        elif b == tc and a.startswith(pre) and "." not in a[len(pre):]:
            out.append(("start", a[len(pre):], op == "<"))
    return out


def entity_window(bounds, parent_df, datetime):
    """(A, B) in days per entity row: start bounds are inclusive lower limits,
    end bounds exclusive upper limits (a strict 'date < end' on dates ends at
    the start of that day, a non-strict one at its end)."""
    n = len(parent_df)
    A, B = np.full(n, -np.inf), np.full(n, np.inf)
    eps = 1e-6
    for side, col, strict in bounds:
        if col not in parent_df.columns:
            continue
        v = to_days(parent_df[col])
        if side == "end":
            v = v + (0.0 if strict else (eps if datetime else 1.0))
            B = np.fmin(B, v)
        else:
            v = v + ((eps if datetime else 1.0) if strict else 0.0)
            A = np.fmax(A, v)
    return np.where(np.isfinite(A), A, np.nan), np.where(np.isfinite(B), B, np.nan)


def _time_spec(S, t, tc):
    """(kind, foreign key, parent, parent column) of the time column: a date
    column ('own') or a lookup fk.col into a calendar ('calendar')."""
    cs = S.spec["tables"][t]["columns"][tc]
    if cs["type"] in ("date", "datetime"):
        return "own", None, None, None, cs["type"] == "datetime"
    if cs["type"] == "lookup" and isinstance(cs.get("lookup"), str) and cs["lookup"].count(".") == 1:
        fk, pc = cs["lookup"].split(".")
        r = _entity_rel(S, t, fk)
        pt = S.spec["tables"][r["parent"]]["columns"].get(pc, {}).get("type")
        if pt in ("date", "datetime"):
            return "calendar", fk, r["parent"], pc, pt == "datetime"
    raise ValueError(f"{t}: sequence.time {tc} must be a date or datetime column, or a lookup fk.col of one")


def fit_sequences(S, tables):
    """Fit the sequence model of every table that declares one."""
    S.seq = {}
    for t, T in S.spec["tables"].items():
        sq = T.get("sequence")
        if not sq:
            continue
        o = options(sq)
        ent_col, tc = o["entity"], o["time"]
        r = _entity_rel(S, t, ent_col)
        P = r["parent"]
        kind, fk, cal, cal_col, is_dt = _time_spec(S, t, tc)
        df = tables[t]
        if tc in df.columns:
            tdays = to_days(df[tc])
        else:                                           # a lookup not stored in the source
            idx = S._lookup(cal, df[fk])
            vals = to_days(tables[cal][cal_col])
            tdays = np.where(idx >= 0, vals[np.maximum(idx, 0)], np.nan)
        owner = S._lookup(P, df[ent_col])
        ent = S.E.get(P)
        z = np.asarray(ent.z, dtype=np.int64) if ent is not None else np.zeros(len(tables[P]), np.int64)
        K = int(z.max()) + 1 if len(z) else 1
        bounds = window_bounds(S.spec, t, o, ent_col)
        A, B = entity_window(bounds, tables[P], is_dt)
        hourly = is_dt if o["hourly"] == "auto" else bool(o["hourly"])
        so = state_options(o)
        m = SequenceModel(o["bucket"], bool(o["weekly"]), hourly, float(o["shrinkage"]), int(o["min_entities"]),
                          calendar=o["calendar"], calendar_draw=bool(o["calendar_draw"]),
                          calendar_bf=float(o["calendar_bf"]),
                          process=o["process"], states=None if so is None else {"prior": so["prior"]})
        rng = np.random.default_rng(np.random.SeedSequence([int(S.cfg["seed"]), 1201, len(S.seq)]))
        st_info = None
        if so is None:
            m.fit(tdays, owner, z, K, (A, B), datetime=is_dt, rng=rng)
        else:
            from .states import MAX_LEVELS
            if so["column"] not in df.columns:
                from ..spec import SpecError
                raise SpecError(f"{t}: sequence.states {so['column']} is not a column of the table that carries the "
                                "sequence (with fact.header the sequence moves to the order header; give a header "
                                "column, or leave the time column on the lines)")
            keys = _state_keys(df[so["column"]])
            levels, first = np.unique(keys, return_index=True)
            if len(levels) > MAX_LEVELS:
                raise ValueError(f"{t}: sequence.states {so['column']} has {len(levels)} levels (at most "
                                 f"{MAX_LEVELS}); a status chain needs a categorical status")
            vals = df[so["column"]].to_numpy(dtype=object)[first]
            st_info = {"column": so["column"], "levels": levels.tolist(),
                       "values": [None if k == NULL_LEVEL else v for k, v in zip(levels, vals)]}
            m.fit(tdays, owner, z, K, (A, B), datetime=is_dt, rng=rng,
                  state=_state_codes(df[so["column"]], st_info["levels"]), n_levels=len(levels))
        S.seq[t] = {"model": m, "entity": ent_col, "parent": P, "time": tc, "kind": kind, "fk": fk,
                    "calendar": cal, "calendar_col": cal_col, "datetime": is_dt, "bounds": bounds,
                    "redraws": int(o["redraws"]), "states": st_info}

        k_seq = len(S.seq)

        # a module-level function, not a closure: the synthesizer is pickled to the
        # worker processes of a partitioned run (spawn start on macOS and Windows)
        S._register(functools.partial(_redraw_sequence, m, S, k_seq), now=False)
        S.log["decisions"].setdefault(t, {})["sequence"] = (
            f"event times of each {P} row redrawn jointly (sequence model, {o['process']} process, {K} segment(s), "
            f"window {[b[:2] for b in bounds] or 'global'}"
            + (f", status chain on {st_info['column']} with {len(st_info['levels'])} levels" if st_info else "") + ")")


def _rule_ok(S, t, df, out):
    from ..expr import check_mask
    cons = S.cons.get(t, {})
    ok = np.ones(len(df), bool)
    for r in cons.get("key_rules", []) + cons.get("attr_rules", []):
        ok &= check_mask(df, r["expr"], S._resolver(t, df, out))
    return ok


def _put(col, rr, vals, is_dt):
    """col with the rows rr set to the datetime64 values vals, in col's own type."""
    import datetime as _dt
    if str(col.dtype).startswith("datetime"):
        arr = col.to_numpy(copy=True)
        arr[rr] = vals.astype(arr.dtype)
        return pd.Series(arr, index=col.index)
    arr = col.to_numpy(dtype=object, copy=True)
    first = next((v for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
    ts = pd.to_datetime(vals)
    if isinstance(first, str):
        new = ts.strftime("%Y-%m-%d %H:%M:%S" if is_dt else "%Y-%m-%d").to_numpy(dtype=object)
    elif isinstance(first, _dt.datetime) or isinstance(first, pd.Timestamp):
        new = np.array(list(ts.to_pydatetime()), dtype=object)
    elif isinstance(first, _dt.date):
        new = np.array([x.date() for x in ts], dtype=object)
    else:
        new = np.array(list(ts), dtype=object)
    arr[rr] = new
    return pd.Series(arr, index=col.index)


def apply_sequence(S, t, df, out, index=0):
    """Redraw the event times of table t's synthetic rows (see module docstring),
    with a scenario's intensity windows applied to the calendar (relsynth.scenarios)."""
    info = S.seq.get(t) if hasattr(S, "seq") else None
    if info is None or not len(df):
        return df
    scen = getattr(S, "_scen", None)
    if scen is not None and t in scen.seq_windows:
        with scen.tilt_sequence(t, info["model"]):
            return _apply_sequence(S, t, df, out, index, info)
    return _apply_sequence(S, t, df, out, index, info)


def _apply_sequence(S, t, df, out, index, info):
    from ..engine import _keystr
    m = info["model"]
    P, ent_col, tc = info["parent"], info["entity"], info["time"]
    pdf = out[P]
    pk = S.spec["tables"][P]["primary_key"]
    pos = pd.Index(_keystr(pdf[pk]).to_numpy(dtype=object)).get_indexer(_keystr(df[ent_col]).to_numpy(dtype=object))
    orig_t = to_days(df[tc]) if tc in df.columns else np.full(len(df), np.nan)
    use = (pos >= 0) & np.isfinite(orig_t)
    if not use.any():
        return df
    seed = S.cfg["seed"] if S.cfg["posterior"].get("sample_seed") is None else S.cfg["posterior"]["sample_seed"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1201, int(index), sum(map(ord, t))]))
    zs = getattr(S, "_zs", {}).get(P)
    zP = np.asarray(zs, dtype=np.int64) if zs is not None and len(zs) == len(pdf) else np.zeros(len(pdf), np.int64)
    zP = np.minimum(zP, m.K - 1)
    A, B = entity_window(info["bounds"], pdf, info["datetime"])
    rows = np.flatnonzero(use)
    # rows of each entity in the order of the engine's times (ties at random)
    o = np.lexsort((rng.random(len(rows)), orig_t[rows], pos[rows]))
    rows = rows[o]
    ents, n = np.unique(pos[rows], return_counts=True)
    orig = df.copy()

    def draw(sel):
        """New times (days) for the rows of the entities ents[sel], in row order."""
        tt, _ow = m.sample(n[sel], zP[ents[sel]], A[ents[sel]], B[ents[sel]], rng)
        return tt

    def write(frame, rr, days):
        vals = from_days(days, info["datetime"])
        if info["kind"] == "own":
            frame[tc] = _put(frame[tc], rr, vals, info["datetime"])
        else:
            cal = out[info["calendar"]]
            cpk = S.spec["tables"][info["calendar"]]["primary_key"]
            cd = to_days(cal[info["calendar_col"]])
            okc = np.isfinite(cd)
            order = np.argsort(cd[okc])
            cdays, ckeys = cd[okc][order], cal[cpk].to_numpy(dtype=object)[okc][order]
            target = np.floor(days) if not info["datetime"] else days
            j = np.clip(np.searchsorted(cdays, target), 0, len(cdays) - 1)
            jm = np.clip(j - 1, 0, len(cdays) - 1)
            j = np.where(np.abs(cdays[jm] - target) < np.abs(cdays[j] - target), jm, j)
            col = frame[info["fk"]].to_numpy(dtype=object, copy=True)
            col[rr] = ckeys[j]
            frame[info["fk"]] = col
        return frame

    st_info = info.get("states")
    rewritten = np.zeros(len(df), bool)       # rows whose status the chain set
    if st_info is not None:
        scol = st_info["column"]
        row_code = np.full(len(df), -1, dtype=np.int64)
        row_code[rows] = _state_codes(df[scol].to_numpy(dtype=object)[rows], st_info["levels"])
        ent_rows = np.searchsorted(ents, pos[rows])          # local entity of each row in `rows`

        def place(frame, sel):
            """Times and statuses of the entities ents[sel] (Props. 1822-1823): the
            status chain decides the order of the entity's statuses, its rows are
            put at the positions of their own status (in the engine's order), and a
            row left over takes the status of the position it fills."""
            tt, _ow, ss = m.sample_states(n[sel], zP[ents[sel]], A[ents[sel]], B[ents[sel]], rng)
            idx = np.flatnonzero(sel)
            loc = np.full(len(ents), -1)
            loc[idx] = np.arange(len(idx))
            rsel = rows[sel[ent_rows]]
            er = loc[ent_rows[sel[ent_rows]]]
            ep = np.repeat(np.arange(len(idx)), n[idx])
            Pp = pd.DataFrame({"e": ep, "c": ss, "p": np.arange(len(ss))})
            Rr = pd.DataFrame({"e": er, "c": row_code[rsel], "r": rsel})
            Pp["k"] = Pp.groupby(["e", "c"]).cumcount()
            Rr["k"] = Rr.groupby(["e", "c"]).cumcount()
            M = Pp.merge(Rr, on=["e", "c", "k"], how="inner")
            Pu = Pp[~Pp["p"].isin(M["p"])].sort_values("p", kind="stable").copy()
            Ru = Rr[~Rr["r"].isin(M["r"])].copy()
            Ru["o"] = np.arange(len(Ru))
            Ru = Ru.sort_values(["e", "o"], kind="stable")
            Pu["k2"] = Pu.groupby("e").cumcount()
            Ru["k2"] = Ru.groupby("e").cumcount()
            M2 = Pu.merge(Ru, on=["e", "k2"], how="inner")
            r_all = np.concatenate([M["r"].to_numpy(), M2["r"].to_numpy()]).astype(np.int64)
            p_all = np.concatenate([M["p"].to_numpy(), M2["p"].to_numpy()]).astype(np.int64)
            o_ = np.argsort(r_all, kind="stable")
            r_all, p_all = r_all[o_], p_all[o_]
            frame = write(frame, r_all, tt[p_all])
            vals = frame[scol].to_numpy(dtype=object, copy=True)
            vals[rsel] = orig[scol].to_numpy(dtype=object)[rsel]           # the engine's statuses again
            rr2 = M2["r"].to_numpy(dtype=np.int64)
            if len(rr2):
                vals[rr2] = np.array([st_info["values"][c] for c in ss[M2["p"].to_numpy()]], dtype=object)
            try:
                frame[scol] = pd.Series(vals, index=frame.index).astype(orig[scol].dtype)
            except (TypeError, ValueError):
                frame[scol] = vals
            rewritten[rsel] = False
            rewritten[rr2] = True
            return frame
        df = place(df, np.ones(len(ents), bool))
    else:
        df = write(df, rows, draw(np.ones(len(ents), bool)))
    df = S._computed(t, df, out)
    ok0 = _rule_ok(S, t, orig, out)
    redrawn = reverted = 0
    ent_of_row = np.full(len(df), -1)
    ent_of_row[rows] = np.searchsorted(ents, pos[rows])
    for _ in range(info["redraws"]):
        bad = ok0 & ~_rule_ok(S, t, df, out)
        bad_e = np.unique(ent_of_row[bad & (ent_of_row >= 0)])
        if not len(bad_e):
            break
        redrawn += len(bad_e)
        sel = np.zeros(len(ents), bool)
        sel[bad_e] = True
        rr = rows[np.isin(ent_of_row[rows], bad_e)]
        df = place(df, sel) if st_info is not None else write(df, rr, draw(sel))
        df = S._computed(t, df, out)
    bad = ok0 & ~_rule_ok(S, t, df, out)
    bad_e = np.unique(ent_of_row[bad & (ent_of_row >= 0)])
    if len(bad_e):
        reverted = len(bad_e)
        rr = np.flatnonzero(np.isin(ent_of_row, bad_e))
        for c in df.columns:
            if c in orig.columns:
                v = df[c].to_numpy(dtype=object, copy=True)
                v[rr] = orig[c].to_numpy(dtype=object)[rr]
                try:
                    df[c] = pd.Series(v, index=df.index).astype(orig[c].dtype)
                except (TypeError, ValueError):
                    df[c] = v
        if st_info is not None:
            rewritten[rr] = False
    S.sample_log.setdefault("sequences", {})[t] = {"entities": int(len(ents)), "events": int(len(rows)),
                                                    "redrawn_entities": int(redrawn),
                                                    "kept_engine_times": int(reverted)}
    if st_info is not None:
        S.sample_log["sequences"][t]["status_rewritten"] = int(rewritten.sum())
    return df


# ---------------------------------------------------------------- regular series
SERIES_DEFAULTS = {"ar": 2, "factor_ar": 2, "day": True, "week": True, "year": "auto"}


def split_series(spec):
    """(spec without its series tables, {table: series block}).  A series table
    is generated by relsynth.timeseries.series after the relational model, so
    the engine never sees it (nothing may reference it)."""
    import copy
    ser = {t: T["series"] for t, T in spec["tables"].items() if T.get("series")}
    if not ser:
        return spec, {}
    s = copy.deepcopy(spec)
    for t in ser:
        del s["tables"][t]
    s["relationships"] = [r for r in s.get("relationships", []) if r.get("child") not in ser]
    s["rules"] = [r for r in s.get("rules", []) if r.get("table") not in ser]
    sc = (s.get("synthesis") or {}).get("scale") or {}
    if isinstance(sc.get("tables"), dict):
        sc["tables"] = {k: v for k, v in sc["tables"].items() if k not in ser}
    return s, ser


def fit_series(S, tables):
    """Fit the series model of every series table (S.series_spec)."""
    from .series import SeriesModel
    S.series = {}
    for t, blk in S.series_spec.items():
        o = {**SERIES_DEFAULTS, **blk}
        T = S.user_spec["tables"][t]
        vals = o.get("values") or [c for c, cs in T["columns"].items()
                                   if cs.get("type") in ("float", "integer") and c not in (o["entity"], o["time"])]
        rel = next(r for r in S.user_spec["relationships"] if r["child"] == t and r["child_key"] == o["entity"])
        rng = np.random.default_rng(np.random.SeedSequence([int(S.cfg["seed"]), 1205, len(S.series)]))
        m = SeriesModel(o["ar"], o["factor_ar"], o["day"], o["week"], o["year"]).fit(
            tables[t], o["entity"], o["time"], vals, rng)
        S.series[t] = {"model": m, "entity": o["entity"], "time": o["time"], "parent": rel["parent"],
                       "parent_key": rel.get("parent_key") or S.user_spec["tables"][rel["parent"]]["primary_key"]}
        S.log["decisions"].setdefault(t, {})["series"] = (
            f"regular series per {rel['parent']} row: {m.summary()}")


def _redraw_sequence(m, S, k, rng):
    """The sequence model's posterior draw for a dataset, from its own parameter
    stream: drawing from the engine's stream would change every parameter drawn
    after it (Prop. 1203)."""
    from ..posterior import MeanRNG
    if isinstance(rng, MeanRNG):
        m.redraw(None)
        return
    base = S.cfg["seed"] if S.cfg["posterior"].get("sample_seed") is None else S.cfg["posterior"]["sample_seed"]
    m.redraw(np.random.default_rng(np.random.SeedSequence([int(base), 1202, int(S._runs), k])))


def sample_series(S, ext, prng, index, mode="draw"):
    """Add the series tables to an external dataset ext (dict)."""
    from ..posterior import MeanRNG
    seed = S.cfg["seed"] if S.cfg["posterior"].get("sample_seed") is None else S.cfg["posterior"]["sample_seed"]
    for t, info in getattr(S, "series", {}).items():
        m = info["model"]
        m.redraw(None if (mode == "mean" or isinstance(prng, MeanRNG)) else prng)
        P = info["parent"]
        keys = ext[P][info["parent_key"]].to_numpy(dtype=object) if P in ext else np.arange(m.n)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1205, int(index), sum(map(ord, t))]))
        df = m.sample(len(keys), rng, keys=keys)
        T = S.user_spec["tables"][t]
        out = pd.DataFrame({info["entity"]: df["__entity__"].to_numpy(dtype=object)})
        out[info["time"]] = pd.to_datetime(from_days(df["__time__"].to_numpy(), True))
        for c in m.values:
            out[c] = df[c].to_numpy()
        pk = T.get("primary_key")
        if pk and pk not in out:
            out.insert(0, pk, np.arange(1, len(out) + 1))
        scen = getattr(S, "_scen", None)
        if scen is not None:
            out = scen.apply_series(t, out, info["time"], m.values)
        ext[t] = out[[c for c in T["columns"] if c in out.columns]]
        S.sample_log.setdefault("series", {})[t] = {"series": len(keys), "rows": len(out)}
    return ext

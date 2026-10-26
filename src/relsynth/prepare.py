# SPDX-License-Identifier: Apache-2.0
"""Computed columns: lookups (fk.col), derive rules and cross-table aggregates.

The same functions serve two purposes:
  * on the source, `prepare_source` fills computed columns that the database does
    not store (so a spec can add e.g. a template code derived from a cell code)
    and audits the ones it does store (share of rows where the stored value
    differs from the rule);
  * on the synthetic tables, the engine uses `make_resolver` and
    `compute_aggregate` so that computed columns are exact functions of the
    generated data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import canonical_keys
from .expr import check_mask, derive_values, split_assignment
from .spec import resolve_path, table_edges, topo_order


def _keystr(s):
    """Canonical string keys (1, 1.0 and '1' are the same key; nulls stay None)."""
    s = canonical_keys(pd.Series(s).reset_index(drop=True))
    return s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v))


def _tuple_frame(parts):
    """List of aligned key Series -> object array of keys (values or tuples)."""
    if len(parts) == 1:
        return pd.Series(parts[0]).to_numpy(dtype=object)
    arr = [pd.Series(p).to_numpy(dtype=object) for p in parts]
    out = np.empty(len(arr[0]), dtype=object)
    for i, t in enumerate(zip(*arr)):
        out[i] = t
    return out


def _index(keys):
    return pd.Index(keys, tupleize_cols=False)


def computed_columns(spec) -> dict[str, dict[str, str]]:
    """{table: {column: 'lookup' | 'derive' | 'aggregate'}}"""
    out = {t: {} for t in spec["tables"]}
    for t, ts in spec["tables"].items():
        for c, cs in ts["columns"].items():
            if cs["type"] == "lookup":
                out[t][c] = "lookup"
    for r in spec.get("rules", []):
        if r.get("type") == "derive":
            out[r["table"]][split_assignment(r["expr"])[0]] = "derive"
        elif r.get("type") == "aggregate":
            out[r["table"]][r["column"]] = "aggregate"
    return out


def _parent_index(par, pk, index_cache):
    """Indexes of a parent's keys (raw values, and canonical key strings), reused
    while the same parent frame is unchanged (index_cache is scoped by the caller
    to a stretch of work in which the parents' keys do not change)."""
    if index_cache is not None:
        hit = index_cache.get((id(par), pk))
        if hit is not None and hit[0] is par and hit[1] == len(par):
            return hit[2]
    raw = pd.Index(par[pk].to_numpy(dtype=object))
    ix = (raw if raw.is_unique else None, pd.Index(_keystr(par[pk])))
    if index_cache is not None:
        index_cache[(id(par), pk)] = (par, len(par), ix)
    return ix


def _positions(index, cur):
    """Positions of the values cur in a parent's keys (-1 if absent): equal raw
    values first, canonical key strings ('1' and 1.0 match 1) for the rest."""
    raw, canon_ = index
    vals = pd.Series(cur).to_numpy(dtype=object)
    if raw is not None:
        try:
            ix = raw.get_indexer(vals)
        except (TypeError, ValueError):
            ix = np.full(len(vals), -1)
        miss = (ix < 0) & pd.notna(vals)
        if miss.any():
            ix[miss] = canon_.get_indexer(_keystr(vals[miss]))
        return ix
    return canon_.get_indexer(_keystr(vals))


def make_resolver(spec, t, df, out, index_cache=None):
    """resolve('fk.col' or 'fk.fk2.col') -> Series aligned with df (rows of table t),
    looked up in the tables of `out`."""
    cache = {}

    def resolve(path):
        if path in cache:
            return cache[path]
        hops, (pt, pc) = resolve_path(spec, t, path)
        cur = df[hops[0]["child_key"]]
        for i, h in enumerate(hops):
            par = out[h["parent"]]
            pk = spec["tables"][h["parent"]]["primary_key"]
            nxt = hops[i + 1]["child_key"] if i + 1 < len(hops) else pc
            if nxt not in par.columns:
                raise KeyError(f"{path}: {h['parent']}.{nxt} is not available yet (computed later)")
            ix = _positions(_parent_index(par, pk, index_cache), cur)
            vals = np.append(par[nxt].to_numpy(dtype=object), None)     # ix == -1 reads the None
            cur = pd.Series(vals[ix], index=df.index)
        cache[path] = cur
        return cur
    return resolve


_ISO_DATE = r"^\d{4}-\d{2}-\d{2}(?:[ T]00:00:00(?:\.0+)?)?$"


def _matchkey(v):
    """Canonical match values for keyed lookups: dates (datetimes or ISO strings)
    as 'YYYY-MM-DD' (with the time when it is not midnight), other values as
    canonical key strings (1, 1.0 and '1' are the same); nulls None."""
    s = pd.Series(np.asarray(v, dtype=object) if not isinstance(v, pd.Series) else v).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
    else:
        st = s.astype(str)
        isdate = s.notna() & st.str.match(_ISO_DATE)
        if isdate.any() and isdate.sum() >= 0.95 * s.notna().sum():
            d = pd.to_datetime(s.where(isdate), errors="coerce", format="mixed")
        elif s.map(lambda x: isinstance(x, pd.Timestamp)).any():
            d = pd.to_datetime(s, errors="coerce")
        else:
            return _keystr(s)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    midnight = (d.dropna() == d.dropna().dt.normalize()).all()
    txt = d.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S")
    return txt.astype(object).where(d.notna(), None)


def lookup_deps(lk):
    """Columns of the row a lookup reads first (the first hop of a path, or the
    names in a keyed lookup's match expressions)."""
    from .expr import names_in
    if isinstance(lk, str):
        return {lk.split(".")[0]}
    out = set()
    for v in (lk.get("match") or {}).values():
        out |= {n.split(".")[0] for n in names_in(str(v))}
    return out


def keyed_lookup(lk, df, resolve, out):
    """Value of a reference table's column at the row whose key columns equal
    expressions of this row: lookup {table: R, match: {R.col: expr}, value: R.col}.
    An exchange rate is rate[(OrderDate, 'USD', CurrencyCode)].  Expressions are
    columns, paths (fk.col) or constants ('USD'); dates match as dates."""
    from .expr import Evaluator, is_path
    R = out[lk["table"]]
    match = lk.get("match") or {}
    ev = Evaluator(df, resolve)
    left, right = [], []
    for rc, e in match.items():
        e = str(e).strip()
        if e in df.columns:                       # raw values: dates stay dates
            val = df[e].reset_index(drop=True)
        elif is_path(e):
            val = pd.Series(np.asarray(resolve(e), dtype=object))
        else:
            val = pd.Series(ev.eval(e), dtype=object)
        left.append(_matchkey(val))
        right.append(_matchkey(R[rc]))
    lk_keys = _tuple_frame(left)
    r_keys = _tuple_frame(right)
    vals = R[lk["value"]].to_numpy(dtype=object)
    ix = _index(r_keys)
    if ix.has_duplicates:
        keep = ~ix.duplicated()
        ix, vals = ix[keep], vals[keep]
    pos = ix.get_indexer(_index(lk_keys))
    return pd.Series([vals[j] if j >= 0 else None for j in pos], index=df.index)


def map_values(v, cs):
    """A lookup with `map: {value: mapped}` (country -> currency): a code mapped
    through a fixed table; values outside the map take `default` (null)."""
    m = {str(k): x for k, x in cs["map"].items()}
    keys = _keystr(pd.Series(np.asarray(v, dtype=object)))
    dflt = cs.get("default")
    return np.array([None if k is None else m.get(k, dflt) for k in keys], dtype=object)


def lookup_column(spec, t, c, df, out, resolve=None):
    """Values of lookup column t.c for the rows of df: a path (fk.col), a keyed
    lookup into a reference table, optionally mapped through `map`."""
    cs = spec["tables"][t]["columns"][c]
    res = resolve or make_resolver(spec, t, df, out)
    lk = cs["lookup"]
    v = (keyed_lookup(lk, df, res, out) if isinstance(lk, dict) else res(lk)).to_numpy(dtype=object)
    if cs.get("map") is not None:
        v = map_values(v, cs)
    return v


def lookup_values(spec, t, df, col, out):
    return lookup_column(spec, t, col, df, out)


def derive(spec, t, df, rule, out):
    lhs, _ = split_assignment(rule["expr"])
    ctype = spec["tables"][t]["columns"].get(lhs, {}).get("type")
    return derive_values(df, rule["expr"], make_resolver(spec, t, df, out), ctype)[1]


def compute_aggregate(spec, rule, out):
    """Values of an aggregate rule's column for every row of its table.  `of` is a
    column of the child table or an expression over its columns and paths
    (e.g. "UnitPrice * Quantity")."""
    t, src = rule["table"], rule["from"]
    T = spec["tables"]
    rel = next(x for x in spec["relationships"] if x["child"] == src and x["parent"] == t
               and rule.get("via") in (None, x["name"]))
    child = out[src]
    if rel.get("composite"):
        res = make_resolver(spec, src, child, out)
        ck = _tuple_frame([_keystr(child[c]) if "." not in c else _keystr(res(c)) for c in rel["child_key"]])
        pk = _tuple_frame([_keystr(out[t][c]) for c in rel["parent_key"]])
    else:
        ck = _keystr(child[rel["child_key"]]).to_numpy(dtype=object)
        pk = _keystr(out[t][rel["parent_key"]]).to_numpy(dtype=object)
    agg = rule.get("agg", "count")
    cidx = _index(ck)
    of = rule.get("of")
    ofv = None if of is None else (child[of] if of in child.columns else
                                   pd.Series(derive_values(child, f"_of_ = {of}", make_resolver(spec, src, child, out))[1]))
    dates = None
    if ofv is not None and agg in ("min", "max"):
        # first / last dates (first purchase = min(order date)): compared as dates
        from .profiling import is_dateish
        if is_dateish(ofv):
            dates = pd.to_datetime(pd.Series(np.asarray(ofv, dtype=object)), errors="coerce", format="mixed")
    if agg == "count_distinct":
        s = pd.Series(ofv.to_numpy(dtype=object), index=cidx).groupby(level=0).nunique()
    elif dates is not None:
        s = getattr(pd.Series(dates.to_numpy(), index=cidx).groupby(level=0), agg)()
    else:
        vals = np.ones(len(child)) if of is None else \
            pd.to_numeric(ofv, errors="coerce").to_numpy(dtype=float)
        g = pd.Series(vals, index=cidx).groupby(level=0)
        if agg == "count":
            s = g.count()                      # non-null values of `of` (all rows when no `of`)
        elif agg == "exists":
            s = g.size().gt(0)
        else:
            s = getattr(g, agg)()
    v = s.reindex(_index(pk))
    if agg in ("count", "count_distinct", "sum"):
        v = v.fillna(0)
    elif agg == "exists":
        v = v.fillna(False)
    arr = v.to_numpy()
    if dates is not None:
        ctype = T[t]["columns"][rule["column"]]["type"]
        dv = pd.Series(pd.to_datetime(arr))
        arr = (dv.dt.strftime("%Y-%m-%d") if ctype == "date" else dv).astype(object).where(dv.notna(), None).to_numpy()
    if T[t]["columns"][rule["column"]]["type"] == "integer":
        arr = pd.array(pd.to_numeric(pd.Series(arr), errors="coerce").round(), dtype="Float64").astype("Int64")
    return arr


def _same(a, b, tol=None):
    """Elementwise equality with null == null, numbers compared as numbers (within
    `tol` when a rule declares a tolerance, e.g. 0.0001 for money stored to 4 decimals)."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    na, nb = a.isna(), b.isna()
    an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    num = an.notna() & bn.notna()
    atol = max(1e-9, float(tol or 0) * (1 + 1e-9))
    eq = np.where(num, np.isclose(an.fillna(0).astype(float), bn.fillna(0).astype(float), rtol=1e-9, atol=atol),
                  a.astype(str) == b.astype(str))
    return np.where(na | nb, na & nb, eq)


def prepare_source(spec, tables, max_passes=4, partial=()):
    """Fill computed columns the source lacks; audit the ones it stores.
    `partial` names tables given as samples (pushdown): aggregates over their rows
    are not audited (a sample's aggregates are not the stored ones).

    Returns (tables, report) where report = {'filled': [...], 'mismatch': {"t.c": share},
    'check_violations': {"t: expr": share}}.  Fixed point over the dependency order:
    a column is computed once everything it reads exists."""
    comp = computed_columns(spec)
    T = spec["tables"]
    rules = spec.get("rules", [])
    order = topo_order(list(T), table_edges(spec))
    missing = {(t, c) for t in T for c in comp[t] if c not in tables[t].columns}
    report = {"filled": [], "mismatch": {}, "check_violations": {}}
    for _ in range(max_passes):
        progress = False
        for t in order:
            for c, kind in comp[t].items():
                if (t, c) not in missing:
                    continue
                try:
                    if kind == "lookup":
                        v = lookup_values(spec, t, tables[t], c, tables)
                    elif kind == "derive":
                        r = next(r for r in rules if r["table"] == t and r.get("type") == "derive"
                                 and split_assignment(r["expr"])[0] == c)
                        v = derive(spec, t, tables[t], r, tables)
                    else:
                        r = next(r for r in rules if r["table"] == t and r.get("type") == "aggregate"
                                 and r["column"] == c)
                        v = compute_aggregate(spec, r, tables)
                except (KeyError, ValueError):
                    continue                      # reads a column not computed yet
                tables[t][c] = v
                missing.discard((t, c))
                report["filled"].append(f"{t}.{c}")
                progress = True
        if not missing or not progress:
            break
    if missing:
        raise ValueError("computed columns could not be filled on the source (circular or missing inputs): "
                         + ", ".join(sorted(f"{t}.{c}" for t, c in missing)))
    filled = set(report["filled"])
    # audit stored computed columns against their rule
    agg_from = {(r["table"], r["column"]): r.get("from") for r in rules if r.get("type") == "aggregate"}
    for t in order:
        for c, kind in comp[t].items():
            if f"{t}.{c}" in filled or len(tables[t]) == 0:
                continue
            if kind == "aggregate" and agg_from.get((t, c)) in partial:
                continue
            try:
                if kind == "lookup":
                    v = lookup_values(spec, t, tables[t], c, tables)
                elif kind == "derive":
                    r = next(r for r in rules if r["table"] == t and r.get("type") == "derive"
                             and split_assignment(r["expr"])[0] == c)
                    v = derive(spec, t, tables[t], r, tables)
                else:
                    r = next(r for r in rules if r["table"] == t and r.get("type") == "aggregate"
                             and r["column"] == c)
                    v = compute_aggregate(spec, r, tables)
            except (KeyError, ValueError):
                continue
            tol = next((r.get("tolerance") for r in rules if r["table"] == t and r.get("type") == "derive"
                        and split_assignment(r["expr"])[0] == c), None) if kind == "derive" else None
            report["mismatch"][f"{t}.{c}"] = round(float(1 - _same(tables[t][c], v, tol).mean()), 6)
    for r in rules:
        if r.get("type") != "check" or len(tables[r["table"]]) == 0:
            continue
        df = tables[r["table"]]
        try:
            ok = check_mask(df, r["expr"], make_resolver(spec, r["table"], df, tables))
        except (KeyError, ValueError):
            continue
        report["check_violations"][f"{r['table']}: {r['expr']}"] = round(float(1 - ok.mean()), 6)
    return tables, report

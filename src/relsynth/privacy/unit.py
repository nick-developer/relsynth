"""The privacy unit: the person whose rows are protected together.

A privacy unit is an entity table (for example Customer).  A table is *owned*
by the unit when a chain of single-column foreign keys leads from it to the unit
table (InvoiceLine -> Invoice -> Customer).  Every row of an owned table belongs
to exactly one person, and the attacks treat all of a person's rows as one
record (Ponomareva et al. 2023, sec. 3: the unit of privacy).  Tables that are
not owned (products, calendars, code lists) are shared context: they are kept
whole in every split.

`profiles` turns a database into one row per person:
  * the unit table's own attributes;
  * the attributes of the unit's parents (one or two hops: the person's store,
    region);
  * for each owned table, the number of the person's rows and their mean
    (numbers, dates) or most frequent value (categories), including the
    attributes of the rows' own parents (the genre of the tracks bought).
The profile never contains a key value: synthetic keys are new labels, so an
attack that matched keys would measure the labelling, not the person.  This
aggregate profile is a design choice of relsynth, not a published standard;
it is the attacker's view when the release is joined back to people.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ATTR_TYPES = ("integer", "float", "categorical", "boolean", "date", "datetime")
MAX_LEVELS = 200


@dataclass
class Unit:
    table: str
    key: str
    # owned table -> chain of (child, child_key, parent, parent_key) up to the unit
    paths: dict = field(default_factory=dict)

    @property
    def owned(self) -> list[str]:
        return [t for t in self.paths if t != self.table]

    def to_dict(self) -> dict:
        return {"table": self.table, "key": self.key,
                "owned_tables": {t: " -> ".join([p[0] for p in ch] + [self.table]) for t, ch in self.paths.items()
                                 if t != self.table}}


def _single_fks(spec):
    return [r for r in spec["relationships"] if not r.get("composite") and r.get("kind") != "self_reference"]


def person_tables(spec) -> list[str]:
    """Entity tables that describe people or parties: personal columns (names,
    e-mails, birth dates) or direct identifiers (type pii)."""
    from ..introspect import is_person_table
    T = spec["tables"]
    return [t for t, ts in T.items() if ts.get("role") == "entity" and ts.get("primary_key")
            and ts.get("mode") != "copy" and (is_person_table(T[t]["columns"])
                                              or any(c.get("type") == "pii" for c in T[t]["columns"].values()))]


def default_unit(spec) -> str | None:
    """The person table that owns the most tables (a single entity table when
    no table looks personal)."""
    T = spec["tables"]
    cands = [t for t, ts in T.items() if ts.get("role") == "entity" and ts.get("primary_key")
             and ts.get("mode") != "copy"]
    people = person_tables(spec)
    pool = people or (cands if len(cands) == 1 else [])
    if not pool:
        return None
    return max(pool, key=lambda t: (len(resolve(spec, t).paths), t in people))


def resolve(spec, table: str | None = None) -> Unit:
    """The unit and the foreign-key path of every table it owns (breadth first,
    so each table gets its shortest chain).  A chain never passes through
    another person table: a customer's link to their sales representative does
    not make the customer's rows the representative's."""
    table = table or default_unit(spec)
    if table is None:
        raise ValueError("no privacy unit: name the table that holds the people (privacy.unit)")
    ts = spec["tables"].get(table)
    if ts is None or not ts.get("primary_key"):
        raise ValueError(f"privacy unit {table!r} must be a table with a single-column primary key")
    paths = {table: []}
    q = deque([table])
    fks = _single_fks(spec)
    others = set(person_tables(spec)) - {table}
    while q:
        p = q.popleft()
        for r in fks:
            if r["parent"] == p and r["child"] not in paths and r["child"] not in others:
                paths[r["child"]] = [(r["child"], r["child_key"], r["parent"], r["parent_key"])] + paths[p]
                q.append(r["child"])
    return Unit(table, ts["primary_key"], paths)


def owner_ids(unit: Unit, tables: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """For each owned table, the unit key of each row (NaN when the chain has a
    null key: such rows belong to nobody)."""
    out = {unit.table: tables[unit.table][unit.key]}
    for t, chain in unit.paths.items():
        if t == unit.table:
            continue
        cur = tables[t][chain[0][1]]
        for (_c, _ck, par, pk), nxt in zip(chain, chain[1:] + [None]):
            if nxt is None:
                break
            m = tables[par].drop_duplicates(pk).set_index(pk)[nxt[1]]
            cur = cur.map(m)
        out[t] = pd.Series(cur.to_numpy(), index=tables[t].index)
    return out


def subset(unit: Unit, tables: dict[str, pd.DataFrame], ids) -> dict[str, pd.DataFrame]:
    """The database restricted to the people in ids: owned tables keep the
    rows of those people, shared tables stay whole (Prop. 600)."""
    ids = pd.Index(ids)
    own = owner_ids(unit, tables)
    return {t: (df[own[t].isin(ids).to_numpy()].copy() if t in own else df) for t, df in tables.items()}


def split(unit: Unit, tables, frac_control: float, seed: int = 0, max_people: int | None = None):
    """Split the people (not the rows) into training and control."""
    ids = tables[unit.table][unit.key].dropna().unique()
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    if max_people and len(ids) > max_people:
        ids = ids[:max_people]
    n_c = int(round(frac_control * len(ids)))
    return ids[n_c:], ids[:n_c]


# ------------------------------------------------------------------ profiles
def _attr_cols(spec, t, df):
    ts = spec["tables"][t]
    keys = {ts.get("primary_key")} | {r["child_key"] for r in spec["relationships"]
                                       if r["child"] == t and not r.get("composite")}
    out = []
    for c, cs in ts["columns"].items():
        if c in keys or c not in df.columns or cs.get("type") not in ATTR_TYPES:
            continue
        if cs["type"] in ("categorical",) and df[c].nunique() > MAX_LEVELS:
            continue
        out.append(c)
    return out


def _with_parents(spec, t, df, tables, skip: set, hops: int, prefix=""):
    """df with the attributes of its rows' parents (not on the unit path) joined
    in, up to `hops` levels; parent columns are named parent.column."""
    cols = {f"{prefix}{c}": df[c] for c in _attr_cols(spec, t, df)}
    out = pd.DataFrame(cols, index=df.index)
    if hops <= 0:
        return out
    fks = [r for r in _single_fks(spec) if r["child"] == t and r["parent"] not in skip and r["parent"] in tables]
    many = {p for p in {r["parent"] for r in fks} if sum(r["parent"] == p for r in fks) > 1}
    for r in fks:
        par = tables[r["parent"]].drop_duplicates(r["parent_key"]).set_index(r["parent_key"])
        # a parent reached through several keys (order, due and ship date) is named by its key
        name = f"{r['child_key']}:{r['parent']}" if r["parent"] in many else r["parent"]
        sub = _with_parents(spec, r["parent"], par, tables, skip | {t}, hops - 1, f"{name}.")
        if sub.empty:
            continue
        joined = sub.reindex(df[r["child_key"]].to_numpy())
        joined.index = df.index
        out = pd.concat([out, joined.add_prefix(prefix)], axis=1)
    return out


def _kind(s: pd.Series) -> str:
    if s.dtype.kind == "M":
        return "date"
    if s.dtype.kind in "biuf" and s.nunique() > 2:
        return "num"
    return "cat"


def _mode(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    vc = s.astype(str).value_counts()
    # ties go to the smallest value, so the mode does not depend on row order
    return min(vc.index[vc.to_numpy() == vc.iloc[0]])


def _group_mode(pid: pd.Series, v: pd.Series) -> pd.Series:
    """Most frequent value per person, ties to the smallest value (as _mode),
    computed with one sort instead of a Python call per person."""
    d = pd.DataFrame({"p": pid.to_numpy(), "v": v.astype(str).where(v.notna()).to_numpy()}).dropna()
    if d.empty:
        return pd.Series(dtype=object)
    n = d.groupby(["p", "v"], sort=False).size().rename("n").reset_index()
    n = n.sort_values(["p", "n", "v"], ascending=[True, False, True], kind="mergesort")
    return n.drop_duplicates("p").set_index("p")["v"]


def distinct_columns(prof: pd.DataFrame) -> list[str]:
    """Columns that are not a one-to-one recoding of an earlier column (a
    name in three languages, a code and its label): factorize() gives two such
    columns the same codes, so they repeat information and would weigh it
    double in every distance."""
    seen, keep = set(), []
    for c in prof.columns:
        h = hash(pd.factorize(prof[c], use_na_sentinel=True)[0].tobytes())
        if h not in seen:
            seen.add(h)
            keep.append(c)
    return keep


def profiles(spec, unit: Unit, tables: dict[str, pd.DataFrame], hops: int = 2, dedupe: bool = False,
             kinds: dict | None = None) -> pd.DataFrame:
    """One row per person (index: the unit key), see the module docstring.
    dedupe drops one-to-one recodings of earlier columns (distinct_columns).
    kinds fixes how each owned-table column is summarised ('num', 'date' or
    'cat', by 'table.column'), so a synthetic release whose column looks
    categorical gets the same profile columns as the real data; the kinds used
    are returned in prof.attrs['kinds']."""
    U = tables[unit.table]
    base = _with_parents(spec, unit.table, U, tables, {unit.table}, hops)
    base.index = U[unit.key].to_numpy()
    base = base[~base.index.duplicated()]
    parts = [base]
    used: dict[str, str] = {}
    own = owner_ids(unit, tables)
    on_path = set(unit.paths)
    for t in unit.owned:
        if t not in tables:
            continue
        df = tables[t]
        feats = _with_parents(spec, t, df, tables, on_path, hops)
        pid = own[t]
        feats = feats.assign(__pid=pid.to_numpy())
        feats = feats[feats["__pid"].notna()].copy()
        ks = {}
        for c in feats.columns:
            if c == "__pid":
                continue
            k = ks[c] = used[f"{t}.{c}"] = (kinds or {}).get(f"{t}.{c}") or _kind(feats[c])
            if k == "num" and feats[c].dtype.kind not in "biufc":
                feats[c] = pd.to_numeric(feats[c], errors="coerce")
            elif k == "date" and feats[c].dtype.kind != "M":
                feats[c] = pd.to_datetime(feats[c], errors="coerce")
        g = feats.groupby("__pid", sort=False)
        agg = {f"{t}.n_rows": g.size()}
        for c, k in ks.items():
            if k == "num":
                agg[f"{t}.{c}.mean"] = g[c].mean()
            elif k == "date":
                days = (feats[c].astype("int64") / 86400e9).where(feats[c].notna())
                agg[f"{t}.{c}.mean"] = days.groupby(feats["__pid"], sort=False).mean()
            else:
                agg[f"{t}.{c}.mode"] = _group_mode(feats["__pid"], feats[c])
        a = pd.DataFrame(agg).reindex(base.index)
        a[f"{t}.n_rows"] = a[f"{t}.n_rows"].fillna(0)
        parts.append(a)
    prof = pd.concat(parts, axis=1)
    prof.index.name = unit.key
    if dedupe:
        prof = prof[distinct_columns(prof)]
    # dates as day numbers, booleans and categories as strings: one comparable type per column
    for c in prof.columns:
        s = prof[c]
        if s.dtype.kind == "M":
            prof[c] = (s.astype("int64") / 86400e9).where(s.notna())
        elif s.dtype.kind == "b" or s.dtype == object:
            prof[c] = s.astype(object).where(s.notna(), None).map(lambda v: v if v is None else str(v))
    prof.attrs["kinds"] = used
    return prof

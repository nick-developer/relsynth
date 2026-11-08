# SPDX-License-Identifier: Apache-2.0
"""The privacy unit: the person whose rows are protected together.

A privacy unit is an entity table (for example Customer).  A table is *owned*
by the unit when a chain of foreign keys leads from it to the unit table
(InvoiceLine -> Invoice -> Customer); composite keys count (an order line's
sales reasons, keyed by order number and line number).  The chains come from
the IR's trace (relsynth.ir.privacy).  A row reached along several chains
belongs to every person they reach (union ownership, RFC-T06-01, Prop. 1700),
and the attacks treat all of a person's rows as one record (Ponomareva et al.
2023, sec. 3: the unit of privacy).  Tables that are not owned (products,
calendars, code lists) are shared context: they are kept whole in every split.

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

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ATTR_TYPES = ("integer", "float", "categorical", "boolean", "date", "datetime")
MAX_LEVELS = 200


@dataclass
class Unit:
    table: str
    key: str
    # owned table -> its paths to the unit, shortest first; a path is a chain of
    # (child, child_keys, parent, parent_keys) steps, keys as tuples of columns
    paths: dict = field(default_factory=dict)

    @property
    def owned(self) -> list[str]:
        return [t for t in self.paths if t != self.table]

    def to_dict(self) -> dict:
        return {"table": self.table, "key": self.key,
                "owned_tables": {t: " | ".join(" -> ".join([s[0] for s in ch] + [self.table]) for ch in chains)
                                 for t, chains in self.paths.items() if t != self.table}}


def _single_fks(spec):
    return [r for r in spec["relationships"] if not r.get("composite") and r.get("kind") != "self_reference"]


def _cols(k) -> tuple:
    return tuple(k) if isinstance(k, (list, tuple)) else (k,)


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


def _bfs_order(spec, table, owned: set) -> list[str]:
    """The owned tables breadth first from the unit (relationships in spec order),
    which fixes the order of profile columns."""
    order, seen, q = [table], {table}, deque([table])
    rels = [r for r in spec.get("relationships") or [] if r.get("kind") != "self_reference"]
    while q:
        p = q.popleft()
        for r in rels:
            if r["parent"] == p and r["child"] in owned and r["child"] not in seen:
                seen.add(r["child"])
                order.append(r["child"])
                q.append(r["child"])
    return order + sorted(owned - seen)


def resolve(spec, table: str | None = None, others: set | None = None) -> Unit:
    """The unit and every foreign-key path of each table it owns, from the IR's
    trace (relsynth.ir.privacy, Prop. 107): composite links are followed, and a
    table with several paths belongs to the union of the people they reach
    (RFC-T06-01, Prop. 1700).  A path never passes through another person
    table (or a table in `others`): a customer's link to their sales
    representative does not make the customer's rows the representative's."""
    from ..ir import lift
    from ..ir.privacy import trace
    table = table or default_unit(spec)
    if table is None:
        raise ValueError("no privacy unit: name the table that holds the people (privacy.unit)")
    ts = spec["tables"].get(table)
    if ts is None or not ts.get("primary_key") or isinstance(ts["primary_key"], (list, tuple)):
        raise ValueError(f"privacy unit {table!r} must be a table with a single-column primary key")
    stop = (set(person_tables(spec)) | set(others or ())) - {table}
    est = lift(spec)
    tr = trace(est, table)

    def step(rn):
        r = est.relationship(rn)
        to = r.to.attributes or est.entities[r.to.entity].primary_key.attributes
        return (r.from_.entity, tuple(r.from_.attributes), r.to.entity, tuple(to))
    found: dict[str, list] = {}
    for n, et in tr.entities.items():
        if et.status != "owned" or n in stop:
            continue
        chains = []
        for p in et.paths:
            ch = [step(rn) for rn in p]
            if any(s[2] in stop for s in ch[:-1]):
                continue
            chains.append(ch)
        if chains:
            found[n] = sorted(chains, key=len)
    paths: dict[str, list] = {table: [[]]}
    for t in _bfs_order(spec, table, set(found)):
        if t != table:
            paths[t] = found[t]
    return Unit(table, ts["primary_key"], paths)


def _lookup(parent: pd.DataFrame, pk: tuple, carry: tuple, cur: pd.DataFrame) -> pd.DataFrame:
    """For the rows of cur (columns named like pk), the parent's carry columns."""
    if tuple(carry) == tuple(pk):
        return cur[list(pk)]
    if len(pk) == 1:
        m = parent.drop_duplicates(pk[0]).set_index(pk[0], drop=False)
        return pd.DataFrame({c: cur[pk[0]].map(m[c]).to_numpy() for c in carry}, index=cur.index)
    from ..dp.relational import _canon
    idx = pd.MultiIndex.from_frame(parent[list(pk)].astype(object).map(_canon))
    look = parent.loc[~idx.duplicated(), list(dict.fromkeys(carry))]
    look.index = idx[~idx.duplicated()]
    key = pd.MultiIndex.from_frame(cur[list(pk)].astype(object).map(_canon))
    pos = look.index.get_indexer(key)
    out = {}
    for c in carry:
        v = look[c].to_numpy(dtype=object)
        out[c] = np.where(pos >= 0, v[np.clip(pos, 0, None)] if len(v) else None, None)
    return pd.DataFrame(out, index=cur.index)


def _chain_owner(unit: Unit, chain: list, tables: dict[str, pd.DataFrame]) -> pd.Series:
    """The unit key each row of chain[0]'s table reaches along chain (NaN when a
    key on the way is null or dangling)."""
    first = tables[chain[0][0]]
    cur = pd.DataFrame({p: first[c].to_numpy() for c, p in zip(chain[0][1], chain[0][3])}, index=first.index)
    for i, (_c, _ck, par, pk) in enumerate(chain):
        carry = chain[i + 1][1] if i + 1 < len(chain) else (unit.key,)
        nxt_pk = chain[i + 1][3] if i + 1 < len(chain) else (unit.key,)
        got = _lookup(tables[par], pk, carry, cur)
        cur = pd.DataFrame({p: got[c].to_numpy() for c, p in zip(carry, nxt_pk)}, index=first.index)
        if len(carry) > 1:
            # a composite key with a null part leads nowhere
            bad = cur.isna().any(axis=1).to_numpy()
            if bad.any():
                cur.loc[bad, :] = None
    s = cur[unit.key]
    return s.infer_objects() if s.dtype == object else s


def owners(unit: Unit, tables: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """For each table of the unit, its (row, owner) pairs: a Series of unit keys
    indexed by row position (0..n-1).  A row reached along several paths
    appears once per distinct owner (union ownership, RFC-T06-01); a row
    whose paths all lead nowhere belongs to nobody and is absent."""
    U = tables[unit.table]
    out = {unit.table: _positional(U[unit.key])}
    for t, chains in unit.paths.items():
        if t == unit.table or t not in tables:
            continue
        parts = [_positional(_chain_owner(unit, ch, tables)) for ch in chains]
        if len(parts) == 1:
            out[t] = parts[0]
            continue
        both = pd.concat(parts)
        d = pd.DataFrame({"r": both.index.to_numpy(), "o": both.to_numpy()}).drop_duplicates().sort_values(
            "r", kind="mergesort")
        out[t] = pd.Series(d["o"].to_numpy(), index=d["r"].to_numpy())
    return out


def _positional(s: pd.Series) -> pd.Series:
    s = pd.Series(s.to_numpy(), index=np.arange(len(s)))
    return s[s.notna()]


def owner_ids(unit: Unit, tables: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """For each owned table, the unit key of each row along its shortest path
    (NaN when the chain has a null key).  Kept for callers that need one owner
    per row; `owners` gives all of them."""
    out = {unit.table: tables[unit.table][unit.key]}
    for t, chains in unit.paths.items():
        if t != unit.table and t in tables:
            out[t] = _chain_owner(unit, chains[0], tables)
    return out


def subset(unit: Unit, tables: dict[str, pd.DataFrame], ids) -> dict[str, pd.DataFrame]:
    """The database restricted to the people in ids: owned tables keep the
    rows of which any owner is among those people (union ownership), shared
    tables stay whole (Prop. 600)."""
    ids = pd.Index(ids)
    own = owners(unit, tables)
    out = {}
    for t, df in tables.items():
        if t not in own:
            out[t] = df
            continue
        o = own[t]
        keep = np.zeros(len(df), dtype=bool)
        keep[o.index.to_numpy()[o.isin(ids).to_numpy()]] = True
        out[t] = df[keep].copy()
    return out


def _shared_groups(unit: Unit, tables) -> dict:
    """People linked by a row they share (union ownership, Prop. 1700): person ->
    a representative of its group.  Empty when no row has two owners."""
    parent: dict = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    for t, o in owners(unit, tables).items():
        if t == unit.table or not o.index.has_duplicates:
            continue
        for _, g in o.groupby(level=0):
            vals = pd.unique(g.to_numpy())
            for v in vals[1:]:
                a, b = find(vals[0]), find(v)
                if a != b:
                    parent[b] = a
    return {x: find(x) for x in parent}


def split_people(unit: Unit, tables, ids, sizes) -> list:
    """Cut the drawn people `ids` into consecutive parts of the given sizes and
    a last part with the rest.  People who share a row (a row reached along two
    paths, Prop. 1700) stay in one part, so no control profile holds a row the
    release was fitted on; groups are placed in the order their first member was
    drawn.  Without shared rows this is the plain cut.  If the groups leave a
    part empty (the shared rows join almost everyone), the plain cut is used and
    a warning says that the parts share rows."""
    ids = np.asarray(ids)
    edges = np.cumsum([0, *sizes])
    plain = [ids[a:b] for a, b in zip(edges[:-1], edges[1:])] + [ids[edges[-1]:]]
    grp = _shared_groups(unit, tables) if len(unit.paths) else {}
    if not grp:
        return plain
    members: dict = {}
    for x in ids:
        members.setdefault(grp.get(x, x), []).append(x)
    parts: list = [[] for _ in range(len(sizes) + 1)]
    for g in members.values():
        k = next((i for i, n in enumerate(sizes) if len(parts[i]) < n), len(sizes))
        parts[k].extend(g)
    if any(len(p) == 0 and len(q) > 0 for p, q in zip(parts, plain)):
        log.warning("privacy split: rows shared by people join %d of %d people into one group; "
                    "the parts of the split share rows", max(len(m) for m in members.values()), len(ids))
        return plain
    return [np.array(p, dtype=ids.dtype) for p in parts]


def split(unit: Unit, tables, frac_control: float, seed: int = 0, max_people: int | None = None):
    """Split the people (not the rows) into training and control, keeping
    people who share a row on one side (`split_people`)."""
    ids = tables[unit.table][unit.key].dropna().unique()
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    if max_people and len(ids) > max_people:
        ids = ids[:max_people]
    n_c = int(round(frac_control * len(ids)))
    control, train = split_people(unit, tables, ids, [n_c])
    return train, control


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
    own = owners(unit, tables)
    on_path = set(unit.paths)
    for t in unit.owned:
        if t not in tables:
            continue
        df = tables[t]
        feats = _with_parents(spec, t, df, tables, on_path, hops)
        pid = own[t]
        # one row per (row, owner): a row of several people counts for each of them
        feats = feats.iloc[pid.index.to_numpy()].assign(__pid=pid.to_numpy())
        if len(unit.paths[t]) > 1:
            feats = feats.reset_index(drop=True)
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

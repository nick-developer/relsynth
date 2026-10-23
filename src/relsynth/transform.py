"""Reparametrisations between the user's spec and the engine's internal model.

Two lossless rewrites let the engine generate structures that the user states
on one table:

* **Order headers** (`fact.header`).  A fact whose rows share header values per
  document number (an order: customer, dates, territory, currency) satisfies the
  functional dependency key -> header columns.  By Heath's theorem (1971) the
  table is the lossless join of H(key, header columns) and L(key, line columns),
  because the shared attribute is a key of H.  The engine generates H as an
  entity (one row per order, linked to its own parents) and L as a fact anchored
  on H (lines per order), then joins H back: every order has one header, and
  line numbers are 1..n by construction.

* **Computed keys.**  A foreign key declared as a lookup (the territory of the
  customer's geography) is computed from that path; its link is kept for lookups
  through it but is not modelled as a separate choice of parent.

* **Offset keys** (`offset_from` on a relationship).  Two keys into the same
  ordered parent (order date and ship date into a calendar) are reparametrised as
  the base key and the difference of the parent's order column (ship date minus
  order date, in days).  Given the base key the map offset <-> key is a bijection
  on the parent rows, so modelling the offset as an attribute loses nothing, and
  the gap distribution (always 7 days, never negative) is kept exactly instead of
  drawing two calendar rows independently.

* **Implied dimensions** (`dimensions` on an entity).  A code column k that
  determines other columns of its table (GeoAreaKey -> State, Country,
  Continent) is the same decomposition one level down: the table is the
  lossless join of D(k, determined columns) and the rest, so D (the observed
  combinations, copied like a code list) becomes a parent and the determined
  columns lookups through k.  Exact at any number of levels, where a joint
  categorical would pool rare combinations.  An optional anchor (a coarse
  determined column such as the country) stays a modelled attribute and the
  link is conditioned on it, so the other attributes keep their dependence on
  the dimension.

* **Column offsets** (`offset_from` on a date or number column) and **ratio
  columns** (`ratio_of`): y = x + gap and y = x * factor, with the gap or the
  factor modelled as an attribute (a categorical over the factors the source
  uses, chosen at the coarsest precision that reproduces the stored values) and
  y derived.  The gap distribution (delivery 0 days in store, 1-7 online) and
  the price structure (unit price = list price x one of 16 factors) are kept
  exactly, including y = x on every row where the gap is 0.

* **Reference tables** (`mode: copy` without a primary key) and **views**
  (`role: view`).  A reference table that nothing links to (exchange rates by
  date and currency pair) is copied and read by keyed lookups; a view (a sales
  table that is the join of order lines and orders) is rebuilt from its base
  table's synthetic rows, so it equals the join exactly.

`Plan(spec)` builds the internal spec; `to_internal(tables)` and
`to_external(out)` convert source and synthetic tables.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .expr import names_in, split_assignment
from .prepare import _keystr
from .pushdown.loader import canonical_order
from .spec import SpecError, inherit_scale

EPOCH = pd.Timestamp("1970-01-01")
OFFSET_PREFIX = "__offset__"
RATIO_PREFIX = "__ratio__"
HIDDEN_PREFIXES = (OFFSET_PREFIX, RATIO_PREFIX)


def _order_numbers(vals, unit):
    s = pd.Series(np.asarray(vals, dtype=object))
    if unit == "day":
        d = pd.to_datetime(s, errors="coerce", format="mixed")
        return ((d - EPOCH) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _row_index(keys, pkeys):
    ix = pd.Index(_keystr(pd.Series(np.asarray(pkeys, dtype=object))).to_numpy(dtype=object))
    if ix.has_duplicates:
        ix = ix.drop_duplicates()
    return ix.get_indexer(_keystr(pd.Series(np.asarray(keys, dtype=object))).to_numpy(dtype=object))


def offset_values(base, target, pkeys, porder, unit):
    """order(target row) - order(base row) for each row; NaN when either is missing."""
    num = _order_numbers(porder, unit)
    pk = pd.Series(np.asarray(pkeys, dtype=object)).drop_duplicates()
    num = num[pk.index.to_numpy()]
    bi, ti = _row_index(base, pk), _row_index(target, pk)
    bv = np.where(bi >= 0, num[np.maximum(bi, 0)], np.nan)
    tv = np.where(ti >= 0, num[np.maximum(ti, 0)], np.nan)
    return tv - bv


def offset_keys(base, offsets, pkeys, porder, unit):
    """Key of the parent row whose order value is order(base row) + offset (exact
    match on the parent's order column); None when there is no such row."""
    pkeys = pd.Series(np.asarray(pkeys, dtype=object)).reset_index(drop=True)
    num = _order_numbers(porder, unit)
    bi = _row_index(base, pkeys)
    bv = np.where(bi >= 0, num[np.maximum(bi, 0)], np.nan)
    tv = bv + pd.to_numeric(pd.Series(np.asarray(offsets, dtype=object)), errors="coerce").to_numpy(dtype=float)
    ix = pd.Index(np.round(num, 6))
    keep = ~ix.duplicated() & np.isfinite(num)
    ix2 = pd.Index(np.round(num[keep], 6))
    vals = pkeys.to_numpy(dtype=object)[keep]
    j = ix2.get_indexer(np.where(np.isfinite(tv), np.round(tv, 6), np.inf))
    return np.array([vals[q] if q >= 0 else None for q in j], dtype=object)


def header_table_name(spec, f, key):
    name = f"{f}__{key}"
    while name in spec["tables"]:
        name += "_"
    return name


class Plan:
    """Internal spec plus the conversions of tables in and out of it."""

    def __init__(self, spec):
        self.user = spec
        self.headers = {}
        self.offsets = []
        self.dims = {}
        self.col_offsets = []
        self.ratios = []
        self.views = {t: T["view_of"] for t, T in spec["tables"].items() if T["role"] == "view"}
        self.refs = [t for t, T in spec["tables"].items() if T["role"] != "view" and T.get("mode") == "copy"
                     and not T.get("primary_key")]
        self.ref_frames = {}
        self.log = {}
        # unknown members (a row with key -1 'Unknown' that facts point at when the
        # real member is missing): kept out of the model, references read as null
        self.unknown = {t: [_keystr(pd.Series([k])).iloc[0] for k in T["unknown_members"]]
                        for t, T in spec["tables"].items() if T.get("unknown_members")}
        self.unknown_rows, self.unknown_refs = {}, {}
        ispec = copy.deepcopy(spec)
        gone = set(self.views) | set(self.refs)
        ispec["_refs"] = {t: copy.deepcopy(spec["tables"][t]["columns"]) for t in self.refs}
        for t in gone:
            del ispec["tables"][t]
        ispec["relationships"] = [r for r in ispec["relationships"] if r["child"] not in gone and r["parent"] not in gone]
        ispec["rules"] = [r for r in ispec["rules"] if r["table"] not in gone]
        sc = ispec["synthesis"]["scale"]
        sc["tables"] = {k: v for k, v in (sc.get("tables") or {}).items() if k not in gone}
        for f, T in spec["tables"].items():
            if T["role"] == "fact" and T.get("fact", {}).get("header"):
                self._split_header(ispec, f)
        for e, T in spec["tables"].items():
            if T["role"] == "entity" and T.get("dimensions") and T.get("mode") != "copy":
                self._dimensions(ispec, e)
        self._offsets(ispec)
        self._column_offsets(ispec)
        self._ratios(ispec)
        self._group_effects(ispec)
        self._order_derives(ispec)
        for T in ispec["tables"].values():
            T.pop("key_order", None) if T.get("key_order") not in T["columns"] else None
        self.spec = ispec

    # ------------------------------------------------------------------ spec
    def _split_header(self, s, f):
        T = s["tables"][f]
        h = dict(T["fact"]["header"])
        key, hc, line = h["key"], list(h["columns"]), h.get("line_number")
        H = header_table_name(s, f, key)
        hset = set(hc)
        rels = s["relationships"]
        # header columns that are lookups must start from a header key
        for c in hc:
            cs = T["columns"][c]
            if cs["type"] == "lookup" and cs["lookup"].split(".")[0] not in hset:
                raise SpecError(f"{f}: header column {c} is a lookup through {cs['lookup'].split('.')[0]}, "
                                "which is not a header column")
        hcols = {key: {"type": "id", "nullable": False}}
        hcols.update({c: copy.deepcopy(T["columns"][c]) for c in hc})
        s["tables"][H] = {"role": "entity", "primary_key": key, "columns": hcols, "unique": [],
                          "partial_unique": [], "virtual_header_of": f}
        for r in rels:
            if r["child"] == f and not r.get("composite") and r["child_key"] in hset:
                r["child"] = H
        for c in hc:
            T["columns"][c] = {"type": "lookup", "lookup": f"{key}.{c}", "nullable": True}
        T["columns"][key] = {"type": "id", "nullable": False}
        if line:
            T["columns"][line] = {"type": "sequence", "within": key, "nullable": False}
        rels.append({"child": f, "child_key": key, "parent": H, "parent_key": key, "kind": "many_to_one",
                     "composite": False, "name": f"{f}.{key}->{H}"})
        fact = dict(T["fact"])
        user_anchor = fact.get("anchor")
        fact.pop("header", None)
        fact["anchor"] = H
        line_parents = {r["parent"] for r in rels if r["child"] == f and not r.get("composite") and r["parent"] != H}
        if fact.get("partner") not in line_parents:
            fact.pop("partner", None)
        if fact.get("time_column") in hset:
            fact.pop("time_column", None)
            fact.pop("time_bucket", None)
        if isinstance(fact.get("condition_on_parents"), list):
            fact["condition_on_parents"] = [x for x in fact["condition_on_parents"] if x.split(".")[0] in line_parents]
        T["fact"] = fact
        keep_u = []
        for u in T.get("unique", []):
            if set(u) <= hset | {key} and key not in u:
                s["tables"][H]["unique"].append(list(u))
            elif not (set(u) == {key}):
                keep_u.append(u)
        T["unique"] = keep_u
        # rules: header-only rules move to the header table
        for r in s["rules"]:
            if r["type"] in ("check", "derive") and r["table"] == f:
                used = {n.split(".")[0] for n in names_in(split_assignment(r["expr"])[1]
                                                         if r["type"] == "derive" else r["expr"])}
                lhs = split_assignment(r["expr"])[0] if r["type"] == "derive" else None
                if used <= hset and (lhs is None or lhs in hset):
                    r["table"] = H
                elif lhs in hset:
                    raise SpecError(f"{f}: derive rule '{r['expr']}' defines header column {lhs} from line columns")
            elif r["type"] == "aggregate" and r.get("from") == f:
                via = next((x for x in rels if x["child"] == H and x["parent"] == r["table"]
                            and r.get("via") in (None, x["name"])), None)
                if via is not None:
                    of = r.get("of")
                    if of is not None and not {n.split(".")[0] for n in names_in(of)} <= hset:
                        raise SpecError(f"aggregate {r['table']}.{r['column']}: it reads line columns of {f} "
                                        "through a header key; aggregate a header column instead")
                    if r.get("agg", "count") in ("count", "exists") and of is None:
                        raise SpecError(f"aggregate {r['table']}.{r['column']}: counting rows of {f} through a "
                                        "header key is ambiguous (orders or lines); use count_distinct of the key")
                    r["from"] = H
        # scale: orders follow the parent the fact was anchored on (customers)
        hparents = [r["parent"] for r in rels if r["child"] == H and r.get("kind") == "many_to_one"
                    and not r.get("offset_from")]
        scale_from = user_anchor if user_anchor in hparents else next(
            (p for p in hparents if s["tables"][p].get("mode") != "copy"), None)
        self.headers[f] = {"table": H, "key": key, "columns": hc, "line": line, "scale_from": scale_from}

    def _dimensions(self, s, e):
        T = s["tables"][e]
        for d in T.pop("dimensions"):
            key, cols = d["key"], list(d["columns"])
            D = header_table_name(s, e, key)
            s["tables"][D] = {"role": "entity", "mode": "copy", "primary_key": key,
                              "columns": {key: {"type": "id", "nullable": False},
                                          **{c: copy.deepcopy(T["columns"][c]) for c in cols}},
                              "unique": [], "partial_unique": [], "virtual_dimension_of": e}
            for r in s["rules"]:
                if r["table"] == e and r["type"] == "derive" and split_assignment(r["expr"])[0] in cols:
                    raise SpecError(f"{e}: {split_assignment(r['expr'])[0]} is derived and also determined by "
                                    f"dimension {key}")
            anchor = d.get("anchor")
            if anchor == key:
                # self-anchored (no label to anchor on): the code stays a modelled attribute
                # with its dependence on the other columns, and the columns it determines are
                # read from the dimension by the code's value (keyed lookups)
                for c in cols:
                    T["columns"][c] = {"type": "lookup", "nullable": True,
                                       "lookup": {"table": D, "match": {key: key}, "value": c}}
                T["unique"] = [u for u in T.get("unique", []) if not set(u) & set(cols)]
                # a determined link (genre of a track, fixed by its album) is read, not drawn
                s["relationships"] = [r for r in s["relationships"]
                                      if not (r["child"] == e and r.get("child_key") in cols)]
                self.dims.setdefault(e, []).append({"table": D, "key": key, "columns": cols})
                continue
            for c in cols:
                if c != anchor:
                    T["columns"][c] = {"type": "lookup", "lookup": f"{key}.{c}", "nullable": True}
            if anchor:
                # the anchor (a country) stays a modelled attribute, so the entity's other
                # attributes (latitude) keep their dependence on it, and the link is
                # conditioned on it: k is drawn among the dimension rows of that country
                s["rules"].append({"table": e, "type": "check", "expr": f"{key}.{anchor} == {anchor}",
                                   "anchor_link": key})
            T["columns"][key] = {"type": "id", "nullable": True}
            T["unique"] = [u for u in T.get("unique", []) if not set(u) & set(cols)]
            s["relationships"].append({"child": e, "child_key": key, "parent": D, "parent_key": key,
                                       "kind": "many_to_one", "composite": False, "name": f"{e}.{key}->{D}",
                                       **({"anchor": anchor} if anchor else {})})
            self.dims.setdefault(e, []).append({"table": D, "key": key, "columns": cols})

    def _column_offsets(self, s):
        for t, T in s["tables"].items():
            for c, cs in list(T["columns"].items()):
                base = cs.get("offset_from")
                if base is None or cs["type"] == "lookup":
                    continue
                unit = "day" if cs["type"] in ("date", "datetime") else "number"
                oc = OFFSET_PREFIX + c
                T["columns"][oc] = {"type": "integer" if cs["type"] in ("date", "integer") else "float",
                                    "nullable": True}
                s["rules"].append({"table": t, "type": "derive", "expr": f"{c} = add_days({base}, {oc})"
                                   if unit == "day" else f"{c} = {base} + {oc}"})
                cs.pop("offset_from")
                self.col_offsets.append({"table": t, "column": c, "base": base, "offset": oc, "unit": unit,
                                         "type": T["columns"][oc]["type"]})

    def _ratios(self, s):
        for t, T in s["tables"].items():
            for c, cs in list(T["columns"].items()):
                of = cs.get("ratio_of")
                if of is None or cs["type"] == "lookup":
                    continue
                same = cs.get("ratio_same_as")
                rc = RATIO_PREFIX + (same or c)
                if not same:
                    T["columns"][rc] = {"type": "categorical", "nullable": True}
                s["rules"].append({"table": t, "type": "derive", "expr": f"{c} = {of} * {rc}"})
                cs.pop("ratio_of")
                cs.pop("ratio_same_as", None)
                self.ratios.append({"table": t, "column": c, "of": of, "factor": rc, "own": not same})

    def _group_effects(self, s):
        """A group effect on a ratio column (unit price = list price x factor) is a
        group effect on its factor; on a header column it moves to the header."""
        for t, T in list(s["tables"].items()):
            keep = []
            for g in T.get("group_effects", []):
                g = dict(g)
                rat = next((r for r in self.ratios if r["table"] == t and r["column"] == g["column"]), None)
                if rat is not None:
                    g["column"] = rat["factor"]
                cs = T["columns"].get(g["column"], {})
                if cs.get("type") == "lookup" and t in self.headers and \
                        cs["lookup"] == f"{self.headers[t]['key']}.{g['column']}":
                    hd = self.headers[t]
                    if not all(b in hd["columns"] for b in g["by"]):
                        raise SpecError(f"{t}: group_effects on header column {g['column']} must group by header "
                                        "columns")
                    s["tables"][hd["table"]].setdefault("group_effects", []).append(g)
                    continue
                keep.append(g)
            if "group_effects" in T:
                T["group_effects"] = keep
            if T.get("link_affinity") and t in self.headers:
                hd = self.headers[t]
                stay = []
                for la in T["link_affinity"]:
                    heads = {str(x).split(".")[0] for x in la["match"]}
                    if heads <= set(hd["columns"]):
                        s["tables"][hd["table"]].setdefault("link_affinity", []).append(la)
                    else:
                        stay.append(la)
                T["link_affinity"] = stay

    @staticmethod
    def _order_derives(s):
        """Derive rules of a table ordered so that each comes after the rules that
        define the columns it reads (the engine applies them in list order)."""
        rules = s["rules"]
        der = [i for i, r in enumerate(rules) if r["type"] == "derive"]
        lhs = {i: (rules[i]["table"], split_assignment(rules[i]["expr"])[0]) for i in der}
        defined = {v: i for i, v in lhs.items()}
        deps = {i: {defined[(rules[i]["table"], n)] for n in names_in(split_assignment(rules[i]["expr"])[1])
                    if (rules[i]["table"], n) in defined and defined[(rules[i]["table"], n)] != i} for i in der}
        order, done = [], set()
        while len(order) < len(der):
            ready = [i for i in der if i not in done and deps[i] <= done]
            if not ready:
                cyc = [f"{rules[i]['table']}: {rules[i]['expr']}" for i in der if i not in done]
                raise SpecError("derived columns depend on each other in a cycle (ratio_of, offset_from and derive "
                                "rules included): " + "; ".join(cyc[:4]))
            for i in ready:
                order.append(i)
                done.add(i)
        it = iter(order)
        s["rules"] = [rules[next(it)] if r["type"] == "derive" else r for r in rules]

    def _offsets(self, s):
        for r in s["relationships"]:
            if r.get("kind") == "many_to_one" and not r.get("composite") and r["child"] in s["tables"] and \
                    s["tables"][r["child"]]["columns"].get(r["child_key"], {}).get("type") == "lookup":
                # a key that is a lookup (territory of the customer's geography) is
                # computed, not chosen: its link is kept for paths but not modelled
                r["kind"] = "derived"
                continue
            if r.get("offset_from") is None:
                continue
            t, c, p = r["child"], r["child_key"], r["parent"]
            base = next((x for x in s["relationships"] if x["child"] == t and x["child_key"] == r["offset_from"]
                         and not x.get("composite")), None)
            if base is None:
                raise SpecError(f"{r['name']}: offset_from {r['offset_from']} is not a key of {t} "
                                "(with an order header, both keys must be header columns)")
            otype = s["tables"][p]["columns"][r["order_by"]]["type"]
            unit = "day" if otype in ("date", "datetime") else "number"
            r["kind"] = "derived"
            r["unit"] = unit
            r["offset_column"] = OFFSET_PREFIX + c
            s["tables"][t]["columns"][r["offset_column"]] = {
                "type": "integer" if otype in ("date", "integer") else "float", "nullable": True}
            self.offsets.append({"table": t, "rel": r})

    # ---------------------------------------------------------------- tables
    def to_internal(self, tables, headers=None):
        """Internal tables from source tables.  headers maps a fact given as a sample
        (pushdown) to its header rows and header violations, computed over all rows
        (relsynth.pushdown.stats.TableStats.first_rows)."""
        headers = headers or {}
        out = {t: df for t, df in tables.items() if t not in self.views}
        self.log, self.notes = {}, {}
        for t, ks in self.unknown.items():
            if t not in out:
                continue
            pk = self.user["tables"][t].get("primary_key")
            m = _keystr(out[t][pk]).isin(ks).to_numpy()
            self.unknown_rows[t] = out[t][m].copy()
            out[t] = out[t][~m].reset_index(drop=True)
            for r in self.user["relationships"]:
                if r["parent"] != t or r.get("composite") or r["child"] not in out:
                    continue
                ck = r["child_key"]
                v = _keystr(out[r["child"]][ck])
                hit = v.isin(ks).to_numpy()
                self.unknown_refs[(r["child"], ck, t)] = {
                    "counts": np.array([float((v == k).sum()) for k in ks]), "null": float(v.isna().sum()),
                    "rows": float(len(v))}
                if hit.any():
                    df = out[r["child"]].copy()
                    df[ck] = df[ck].astype(object).where(~hit, None)
                    out[r["child"]] = df
        self.ref_frames = {t: tables[t] for t in self.refs if t in tables}
        for f, h in self.headers.items():
            if f in headers:
                H, diff = headers[f]
                self.log[f] = {"header_rows": len(H), "header_violations": diff}
                out[h["table"]] = canonical_order(H, h["key"])
                continue
            df = out[f]
            cols = [h["key"]] + h["columns"]
            sub = df[cols]
            H = sub.drop_duplicates(h["key"]).reset_index(drop=True)
            # cells that differ from their key's header row (the first row): FD violations
            from .pushdown.stats import first_row_differences
            diff = first_row_differences(sub.reset_index(drop=True), [h["key"]], list(h["columns"]))
            self.log[f] = {"header_rows": len(H), "header_violations": diff}
            out[h["table"]] = canonical_order(H, h["key"])     # in key order, as computed in a database
        for e, ds in self.dims.items():
            df = out[e]
            for d in ds:
                sub = df[[d["key"]] + d["columns"]]
                sub = sub[sub[d["key"]].notna()]
                D = sub.drop_duplicates(d["key"]).reset_index(drop=True)
                first = sub.groupby(_keystr(sub[d["key"]]).to_numpy(dtype=object), sort=False).transform("first")
                diff = sum(int((~((sub[c].astype(object) == first[c].astype(object)) |
                                  (sub[c].isna() & first[c].isna()))).sum()) for c in d["columns"])
                self.notes[f"{e}.{d['key']}"] = {"dimension_rows": len(D), "violations": diff}
                out[d["table"]] = D
        for o in self.offsets:
            r = o["rel"]
            t = o["table"]
            df = out[t].copy()
            par = out[r["parent"]]
            pk = self.spec["tables"][r["parent"]]["primary_key"]
            off = offset_values(df[r["offset_from"]], df[r["child_key"]], par[pk], par[r["order_by"]], r["unit"])
            if self.spec["tables"][t]["columns"][r["offset_column"]]["type"] == "integer":
                off = pd.array(np.where(np.isfinite(off), np.round(off), np.nan), dtype="Float64").astype("Int64")
            df[r["offset_column"]] = off
            out[t] = df
        from .expr import _days
        for o in self.col_offsets:
            df = out[o["table"]].copy()
            if o["unit"] == "day":
                a = _days(np.asarray(df[o["column"]], dtype=object))
                b = _days(np.asarray(df[o["base"]], dtype=object))
            else:
                a = pd.to_numeric(df[o["column"]], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(df[o["base"]], errors="coerce").to_numpy(dtype=float)
            off = a - b
            if o["type"] == "integer":
                off = pd.array(np.where(np.isfinite(off), np.round(off), np.nan), dtype="Float64").astype("Int64")
            df[o["offset"]] = off
            out[o["table"]] = df
        for o in sorted(self.ratios, key=lambda o: not o["own"]):
            out[o["table"]] = self._ratio_factor(o, out[o["table"]], out)
        return out

    def _ratio_factor(self, o, df, out):
        """Factor y / x of each row at the coarsest precision (2..8 decimals) that
        reproduces the stored y on at least 99.9% of rows, round(x * f, d) = y with
        d the stored decimals of y; factors seen in fewer than 5 rows (or beyond
        the 200 most common) are snapped to the nearest common factor."""
        from .prepare import make_resolver
        from .profiling import decimals
        df = df.copy()
        t, c, of = o["table"], o["column"], o["of"]
        x = pd.to_numeric(pd.Series(np.asarray(make_resolver(self.spec, t, df, out)(of) if "." in of else df[of],
                                               dtype=object)), errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
        if not o["own"]:
            f = pd.to_numeric(df[o["factor"]], errors="coerce").to_numpy(dtype=float)
            d = decimals(df[c])
            rep_ = np.isclose(np.round(x * f, d if d is not None else 9), y, atol=0.5 * 10.0 ** -(d or 9) + 1e-12)
            self.notes[f"{t}.{c}"] = {"factor": o["factor"], "reproduced": round(float(rep_[ok].mean()) if ok.any() else 1.0, 6)}
            return df
        r = np.where(ok, y / np.where(ok, x, 1.0), np.nan)
        d = decimals(df[c])
        dd = d if d is not None else 9
        best = None
        for k in range(2, 9):
            rk = np.round(r, k)
            share = float(np.mean(np.isclose(np.round(x[ok] * rk[ok], dd), y[ok], atol=0.5 * 10.0 ** -dd + 1e-12))) \
                if ok.any() else 1.0
            if best is None or share > best[1] + 1e-12:
                best = (k, share, rk)
            if share >= 0.999:
                best = (k, share, rk)
                break
        k, share, rk = best
        vals, cnt = np.unique(rk[ok], return_counts=True)
        keep = cnt >= 5
        if keep.sum() > 200:
            keep = np.zeros(len(cnt), bool)
            keep[np.argsort(-cnt)[:200]] = True
        common = np.sort(vals[keep]) if keep.any() else vals
        snapped = rk.copy()
        if len(common):
            pos = np.clip(np.searchsorted(common, rk[ok]), 1, max(len(common) - 1, 1))
            lo, hi = common[pos - 1], common[np.minimum(pos, len(common) - 1)]
            snapped[ok] = np.where(np.abs(rk[ok] - lo) <= np.abs(hi - rk[ok]), lo, hi) if len(common) > 1 \
                else common[0]
        df[o["factor"]] = pd.Series(np.where(ok, snapped, np.nan), index=df.index).astype(object).where(ok, None)
        self.notes[f"{t}.{c}"] = {"factor": o["factor"], "decimals": k, "levels": int(len(common)),
                                "reproduced": round(share, 6), "snapped_rows": int(np.sum(snapped[ok] != rk[ok]))}
        return df

    def unknown_external(self, res, rng, mode="draw"):
        """Unknown members back in the synthetic data: their rows copied, and each
        null reference turned into a reference to an unknown member with
        probability p ~ Beta(1 + references, 1 + nulls) of the source (its mean in
        mean mode), the member chosen by Dirichlet(counts + 1/2)."""
        for t, rows in self.unknown_rows.items():
            if t in res and len(rows):
                cols = [c for c in res[t].columns if c in rows.columns]
                res[t] = pd.concat([rows[cols], res[t]], ignore_index=True)
        for (child, ck, t), st in self.unknown_refs.items():
            if child not in res or ck not in res[child].columns or st["counts"].sum() == 0:
                continue
            ks = self.unknown_rows[t][self.user["tables"][t]["primary_key"]].to_numpy(dtype=object)
            w = (st["counts"] + 0.5) / (st["counts"] + 0.5).sum() if mode == "mean" else rng.dirichlet(st["counts"] + 0.5)
            df = res[child].copy()
            col = df[ck].astype(object).to_numpy(copy=True)
            nul = np.flatnonzero(pd.isna(col))
            n_unk = st["counts"].sum()
            if st["null"] > 0 or self.user["tables"][child]["role"] != "fact":
                # the link model drew nulls for both: split them with q ~ Beta(1 + unknown, 1 + null)
                a, b = 1 + n_unk, 1 + st["null"]
                q = a / (a + b) if mode == "mean" else float(rng.beta(a, b))
                pick = nul[rng.random(len(nul)) < q]
                extra = 0
            else:
                # a fact leaves rows without a dimension member out of its model: their
                # share p ~ Beta(1 + unknown, 1 + rest) is restored by drawing that many
                # rows from the synthetic rows (the other columns' law given a known
                # member, i.e. the unknown member assumed missing at random)
                a, b = 1 + n_unk, 1 + st["rows"] - n_unk
                p = a / (a + b) if mode == "mean" else float(rng.beta(a, b))
                pick = nul
                extra = int(round(p * len(df) / max(1 - p, 1e-9))) - len(nul)
            if len(pick) and len(ks):
                col[pick] = ks[rng.choice(len(ks), size=len(pick), p=w)] if len(ks) > 1 else ks[0]
                df[ck] = col
            if extra > 0 and len(df) and len(ks):
                df = self._clone_rows(child, df, extra, ck, ks, w, rng)
            res[child] = df
        return res

    def _clone_rows(self, t, df, k, ck, ks, w, rng):
        """k rows drawn from df with foreign key ck set to an unknown member, new
        primary keys, and line numbers continued within their order."""
        T = self.user["tables"][t]
        src = df.iloc[rng.integers(0, len(df), size=k)].copy()
        src[ck] = ks[rng.choice(len(ks), size=k, p=w)] if len(ks) > 1 else ks[0]
        pk = T.get("primary_key")
        if isinstance(pk, str) and pk in df.columns:
            num = pd.to_numeric(df[pk], errors="coerce")
            if num.notna().all():
                src[pk] = np.arange(int(num.max()) + 1, int(num.max()) + 1 + k)
            else:
                src[pk] = [f"{v}-U{i + 1}" for i, v in enumerate(src[pk].astype(str))]
        hd = T.get("fact", {}).get("header") or {}
        ln, hk = hd.get("line_number"), hd.get("key")
        if ln and hk and ln in df.columns:
            top = pd.to_numeric(df[ln], errors="coerce").groupby(_keystr(df[hk]).to_numpy(dtype=object)).max()
            kk = _keystr(src[hk]).to_numpy(dtype=object)
            rank = pd.Series(np.arange(k)).groupby(kk).cumcount().to_numpy() + 1
            src[ln] = top.reindex(kk).fillna(0).to_numpy() + rank
        return pd.concat([df, src], ignore_index=True)

    def to_external(self, out, rng=None, mode="draw"):
        from .prepare import derive, lookup_column, lookup_deps
        res = {}
        hidden = {h["table"] for h in self.headers.values()} | {d["table"] for ds in self.dims.values() for d in ds}
        for t, df in out.items():
            if t in hidden:
                continue
            res[t] = df[[c for c in df.columns if not str(c).startswith(HIDDEN_PREFIXES)]]
        for t in self.refs:
            if t not in res and t in self.ref_frames:
                res[t] = self.ref_frames[t]
        if self.unknown_rows:
            res = self.unknown_external(res, rng if rng is not None else np.random.default_rng(0), mode)
        U = self.user["tables"]
        for v, b in self.views.items():
            if b not in res:
                continue
            base = res[b].reset_index(drop=True)
            cols = U[v]["columns"]
            df = pd.DataFrame(index=base.index)
            for c, cs in cols.items():
                bc = cs.get("base_column", c)
                if cs["type"] != "lookup" and bc in base.columns:
                    df[c] = base[bc].to_numpy(dtype=object) if base[bc].dtype == object else base[bc].to_numpy()
            left = {c: cs["lookup"] for c, cs in cols.items() if cs["type"] == "lookup"}
            while left:
                ready = [c for c, lk in left.items() if not (lookup_deps(lk) & set(left))] or list(left)[:1]
                for c in ready:
                    df[c] = lookup_column(self.user, v, c, df, res)
                    left.pop(c)
            for r in self.user["rules"]:
                if r["table"] == v and r["type"] == "derive":
                    df[split_assignment(r["expr"])[0]] = derive(self.user, v, df, r, res)
            res[v] = df[[c for c in cols if c in df.columns]]
        order = [t for t in self.user["tables"] if t in res]
        return {t: res[t] for t in order + [t for t in res if t not in order]}

    def scale(self, sc):
        """The size factor of every table: its own when set, else the factor of the
        first parent it links to that has one (orders follow their customers, lines
        follow their orders), else the default.  Order headers follow their anchor."""
        sc = {"default": sc.get("default", 1.0), "tables": dict(sc.get("tables", {}) or {})}
        sc = inherit_scale(self.spec, sc, skip={h["table"] for h in self.headers.values()})
        for h in self.headers.values():
            if h["table"] not in sc["tables"]:
                sc["tables"][h["table"]] = sc["tables"].get(h["scale_from"], sc["default"]) \
                    if h["scale_from"] else sc["default"]
        return sc

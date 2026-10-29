# SPDX-License-Identifier: Apache-2.0
"""Model specification: loading, defaults and validation.

The specification ("spec") is one YAML file that carries the technical model
(tables, columns, keys, foreign keys) and the business model (roles, hierarchies,
rules, privacy, scale).  See docs/input-spec.md for the reference.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

COLUMN_TYPES = {"id", "integer", "float", "categorical", "boolean", "date",
                "datetime", "text", "pii", "constant", "lookup", "binary", "sequence"}
TABLE_ROLES = {"entity", "fact", "view"}
REL_KINDS = {"many_to_one", "self_reference", "derived"}   # derived: internal (offset keys)
RULE_TYPES = {"check", "derive", "aggregate", "cumulative", "soft"}   # soft: internal (derive with holds < 1)
SOFT_VIOLATIONS = {"offset", "model"}
AGGREGATES = {"count", "sum", "min", "max", "mean", "exists", "count_distinct"}
PII_GENERATORS = {"email", "person_name", "phone", "uuid", "pattern"}

DEFAULT_SYNTHESIS = {
    "seed": 7,
    # one posterior draw of every parameter per dataset (mode draw), or posterior
    # means (mode mean); datasets > 1 writes that many independent datasets
    "posterior": {"mode": "draw", "datasets": 1, "sample_seed": None},
    "scale": {"default": 1.0, "tables": {}},
    "segments": {"default_cap": 8, "min_rows_per_segment": 20, "tables": {},
                 "stratify": "auto", "stratify_max_levels": 20, "stratify_max_cells": 5e7,
                 "attribute_strata": "auto", "tie_headers": True, "copied_budget_cells": 2e7},
    "gibbs": {"iterations": 120, "final_iterations": 30, "max_final_iterations": 120, "rhat_target": 1.1,
              "chains": 1, "merge": True},
    "copula": {"mode": "rank", "sample_rows": 1500, "iterations": 120, "max_iterations": 3000, "rhat_target": 1.1,
               "burn_in": 60, "lkj_eta": 2.0, "grid": 64, "min_rows": 80, "pooled_cells": 1e6},
    "marginals": {"bayesian_bootstrap": True, "dirichlet_prior": 0.25,
                  "max_levels": 200, "tail_trim": 0.0, "min_dependence_rows": 5, "fd_bf": 100.0,
                  "joint_max_levels": 64, "joint_min_cramers_v": 0.3},
    "facts": {"min_group_rows": 30, "frailty": "auto", "frailty_threshold": 1.5,
              "pairs": "auto", "multiplicity": "auto", "count_model": "auto", "count_model_bf": 100.0,
              "group_fallback": "auto"},
    "structure": {"structural_prior": 0.9, "structural_pooling": 3.0, "beta_prior": 0.5,
                  "conditional_inclusion": "auto", "inclusion_min_expected": 10.0, "participation": "auto"},
    "uniqueness": {"mh_sweeps": 20, "grain": "auto", "rake_rounds": 3},
    "trees": {"max_depth": "auto", "frailty": "auto"},
    "text": {"order": 3, "min_token_rows": 3, "max_copy_ngram": 8,  # nosec B105 - token counts, not passwords
             "backoff_strength": 2.0, "max_retries": 30},
    "rules": {"max_rejection_rounds": 25},
    "privacy": {"suppress_small_counts": 5},
    # fit large fact tables from statistics computed in the database and a bounded
    # uniform sample (relsynth.pushdown); auto: facts with at least min_rows rows
    "pushdown": {"mode": "auto", "min_rows": 1_000_000, "sample_rows": 500_000},
}


class SpecError(ValueError):
    """The specification is invalid (CLI exit code 2)."""
    exit_code = 2


class _SpecLoader(yaml.SafeLoader):
    """yaml.SafeLoader without anchors and aliases: a few lines of aliases expand
    to billions of nodes (an alias bomb), and no spec needs them."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            ev = self.peek_event()
            raise SpecError(f"yaml aliases are not allowed in a spec (*{ev.anchor} at line {ev.start_mark.line + 1})")
        return super().compose_node(parent, index)


def read_yaml(path) -> dict:
    """A spec file (or any YAML the tools read back), aliases refused."""
    # _SpecLoader is a yaml.SafeLoader subclass that also refuses aliases
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_SpecLoader)  # noqa: S506  # nosec B506 - a SafeLoader subclass
    except yaml.YAMLError as e:
        raise SpecError(f"{path} is not valid YAML: {e}") from None


def read_spec(path) -> dict:
    """A spec file as a 1.0 spec mapping: an IR 2.0 document (it has `ir_version`)
    is lowered to the 1.0 spec the engine reads (relsynth.ir.v1)."""
    return from_ir(read_yaml(path))


def from_ir(doc):
    """doc itself when it is a 1.0 spec, else the 1.0 spec of an IR document."""
    from .ir import IRError, is_ir
    if not is_ir(doc):
        return doc
    from .ir.v1 import lower_for_engine
    try:
        return lower_for_engine(doc)
    except IRError as e:
        raise SpecError("Spec has errors:\n  - " + "\n  - ".join(e.errors)) from None


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_spec(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        path_or_dict = from_ir(path_or_dict)
        # private keys of an in-memory spec (a normalised spec passed back) are dropped
        raw = copy.deepcopy({k: v for k, v in path_or_dict.items() if not str(k).startswith("_")})
        base = Path(".")
        errs = []
    else:
        p = Path(path_or_dict)
        raw = read_spec(p)
        base = p.parent
        if not isinstance(raw, dict):
            raise SpecError(f"Spec has errors:\n  - {p}: the spec must be a mapping")
        errs = [f"spec: key '{k}' is internal and cannot be set in a spec file" for k in raw if str(k).startswith("_")]
        raw = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    try:
        spec = normalise(raw)
    except SpecError as e:
        raise SpecError("Spec has errors:\n  - " + "\n  - ".join(errs + [str(e)])) from None
    spec["_base_dir"] = str(base.resolve())
    errs += validate(spec)
    if errs:
        raise SpecError("Spec has errors:\n  - " + "\n  - ".join(errs))
    return spec


def normalise(raw: dict) -> dict:
    spec = copy.deepcopy(from_ir(raw))
    spec.setdefault("version", 1)
    spec.setdefault("relationships", [])
    spec.setdefault("rules", [])
    if not isinstance(spec["rules"], list) or not all(isinstance(r, dict) for r in spec["rules"]):
        raise SpecError("rules must be a list of mappings")
    for r in spec["rules"]:
        # a derive rule that holds on a share of rows only (an age computed to one
        # reference date on 99 % of rows) is a soft rule: a Bayesian mixture of the
        # identity and a violation model, and its column stays modelled
        if r.get("type") == "derive" and r.get("holds") is not None and float(r["holds"]) < 1:
            r["type"] = "soft"
    if not isinstance(spec.get("synthesis") or {}, dict):
        raise SpecError(f"synthesis must be a mapping, not {type(spec['synthesis']).__name__}")
    spec["synthesis"] = deep_merge(DEFAULT_SYNTHESIS, spec.get("synthesis") or {})
    pdm = spec["synthesis"].get("pushdown")
    if isinstance(pdm, dict) and isinstance(pdm.get("mode"), bool):
        # YAML 1.1 reads an unquoted on / off as true / false
        pdm["mode"] = "on" if pdm["mode"] else "off"
    if not isinstance(spec.get("tables") or {}, dict):
        raise SpecError("tables must be a mapping {table name: table}")
    if not isinstance(spec["relationships"], list) or not all(isinstance(r, dict) for r in spec["relationships"]):
        raise SpecError("relationships must be a list of mappings")
    for tname, t in spec.get("tables", {}).items():
        if not isinstance(t, dict):
            raise SpecError(f"tables.{tname} must be a mapping (role, columns, ...), not {type(t).__name__}")
        t.setdefault("role", "entity")
        t.setdefault("columns", {})
        if not isinstance(t["columns"], dict):
            raise SpecError(f"tables.{tname}.columns must be a mapping {{column: type}}")
        for cname, c in list(t["columns"].items()):
            if isinstance(c, str):
                if c.startswith("lookup:"):
                    t["columns"][cname] = c = {"type": "lookup", "lookup": c[len("lookup:"):].strip()}
                else:
                    t["columns"][cname] = c = {"type": c}
            if not isinstance(c, dict):
                raise SpecError(f"tables.{tname}.columns.{cname}: a column is a type name (integer) or a mapping "
                                f"{{type: integer, ...}}, not a {type(c).__name__} ({c!r})")
            c.setdefault("nullable", True)
        u = t.get("unique") or []
        if u and all(isinstance(x, str) for x in u):
            u = [u]
        # full keys: lists of columns; partial keys (a filtered unique index):
        # {columns: [...], where: "<rule expression>"}
        t["partial_unique"] = [{"columns": list(x["columns"]), "where": str(x["where"])}
                               for x in list(t.get("partial_unique") or []) + list(u)
                               if isinstance(x, dict) and x.get("where")]
        t["unique"] = [list(x["columns"] if isinstance(x, dict) else x) for x in u
                       if not (isinstance(x, dict) and x.get("where"))]
        if t["role"] == "fact":
            t.setdefault("fact", {})
        if t.get("group_effects"):
            # one value per group shared by its rows: {column, by: [columns or fk.col paths]}
            t["group_effects"] = [{"column": g.get("column"),
                                   "by": [g["by"]] if isinstance(g.get("by"), str) else list(g.get("by") or [])}
                                  for g in t["group_effects"]]
        if t.get("unknown_members") is not None and not isinstance(t["unknown_members"], list):
            t["unknown_members"] = [t["unknown_members"]]
        if t.get("link_affinity"):
            # two links that point at matching parents more often than by chance
            t["link_affinity"] = [{"match": list(x.get("match") or []),
                                   **({"share": float(x["share"])} if x.get("share") is not None else {})}
                                  for x in t["link_affinity"]]
        if t.get("dimensions"):
            t["dimensions"] = [{"key": d["key"], "columns": list(d.get("columns") or []),
                                **({"anchor": d["anchor"]} if d.get("anchor") else {})} for d in t["dimensions"]]
    for r in spec["relationships"]:
        r.setdefault("kind", "self_reference" if r.get("child") == r.get("parent") else "many_to_one")
        ck = r.get("child_key")
        if isinstance(ck, (list, tuple)):
            r["child_key"] = list(ck)
            r["composite"] = True
            r.setdefault("parent_key", list(ck))
            r["parent_key"] = list(r["parent_key"]) if isinstance(r["parent_key"], (list, tuple)) else [r["parent_key"]]
            r.setdefault("name", f"{r.get('child')}.({'+'.join(ck)})->{r.get('parent')}")
        else:
            r["composite"] = False
            if r.get("parent") in spec.get("tables", {}):
                r.setdefault("parent_key", spec["tables"][r["parent"]].get("primary_key"))
            r.setdefault("name", f"{r.get('child')}.{ck}->{r.get('parent')}")
    # fact.anchor / fact.partner may name a relationship or a foreign-key column
    # instead of a table: needed when one table plays two roles (order date and ship
    # date both into the calendar).  Stored as the table plus the key column.
    for tname, t in spec.get("tables", {}).items():
        f = t.get("fact") if isinstance(t.get("fact"), dict) else None
        if not f:
            continue
        for k in ("anchor", "partner"):
            v = f.get(k)
            if v is None or v in spec.get("tables", {}):
                continue
            r = next((r for r in spec["relationships"] if r.get("child") == tname and not r.get("composite")
                      and v in (r.get("name"), r.get("child_key"))), None)
            if r is not None:
                f[k], f[f"{k}_key"] = r["parent"], r["child_key"]
    return spec


def simple_fks(spec, t):
    """Single-column many-to-one relationships of table t."""
    return [r for r in spec.get("relationships", [])
            if r.get("child") == t and r.get("kind") == "many_to_one" and not r.get("composite")]


def path_fks(spec, t):
    """Single-column references of table t that lookup paths can follow: modelled
    many-to-one links and keys derived from another key (offsets)."""
    return [r for r in spec.get("relationships", [])
            if r.get("child") == t and r.get("kind") in ("many_to_one", "derived") and not r.get("composite")]


def resolve_path(spec, t, path):
    """Follow 'fk.col' or 'fk.fk2.col' from table t.  Returns [(table, fk relationship), ...]
    and the final (table, column), or raises SpecError."""
    parts = path.split(".")
    hops, cur = [], t
    for fk in parts[:-1]:
        r = next((r for r in path_fks(spec, cur) if r["child_key"] == fk), None)
        if r is None:
            raise SpecError(f"{t}: '{path}': {fk} is not a foreign key of {cur}")
        hops.append(r)
        cur = r["parent"]
    col = parts[-1]
    if col not in spec["tables"][cur]["columns"]:
        raise SpecError(f"{t}: '{path}': {cur} has no column {col}")
    return hops, (cur, col)


def table_edges(spec):
    """(child, parent) pairs between different tables, composite keys and facts included."""
    return [(r["child"], r["parent"]) for r in spec.get("relationships", [])
            if r.get("kind") == "many_to_one" and r.get("child") in spec["tables"]
            and r.get("parent") in spec["tables"] and r["child"] != r["parent"]]


def validate(spec: dict, keys: bool = True) -> list[str]:
    from .schema import check_keys
    # unknown keys (typos) are errors, with the closest valid key as a hint; the
    # engine re-validates its internal plan (which adds private keys) with keys=False
    errs = check_keys(spec) if keys else []
    user = keys
    tables = spec.get("tables") or {}
    if not tables:
        errs.append("no tables declared")
    rels = spec.get("relationships", [])
    errs += _validate_io(spec)
    for tname in tables:
        bad = _bad_name(tname)
        if bad:
            errs.append(f"table name {tname!r}: {bad} (table names become file names)")
    for tname, t in tables.items():
        if t.get("mode", "synthesize") not in ("synthesize", "copy"):
            errs.append(f"{tname}: mode must be synthesize or copy")
        if t.get("mode") == "copy":
            if t["role"] != "entity":
                errs.append(f"{tname}: only entity (reference) tables can be copied")
            for r in rels:
                if r.get("child") == tname and r.get("parent") in tables and r["parent"] != tname \
                        and tables[r["parent"]].get("mode") != "copy":
                    errs.append(f"{tname}: a copied table can only reference copied tables ({r['parent']} is synthesised)")
        if t["role"] not in TABLE_ROLES:
            errs.append(f"{tname}: role must be one of {sorted(TABLE_ROLES)}")
        pk = t.get("primary_key")
        referenced = any(r.get("parent") == tname and r.get("child") != tname for r in rels)
        if t["role"] == "entity" and not pk and not (t.get("mode") == "copy" and not referenced):
            errs.append(f"{tname}: entity tables need a primary_key (a copied reference table without one "
                        "may be read by keyed lookups but not referenced by foreign keys)")
        if t["role"] == "view":
            errs += _validate_view(spec, tname, t)
        if t.get("key_order") is not None:
            if t["role"] != "entity" or t["key_order"] not in t["columns"]:
                errs.append(f"{tname}: key_order must name a column of an entity table")
        for g in t.get("group_effects", []):
            c = g.get("column")
            if c not in t["columns"] or t["columns"][c]["type"] in ("id", "lookup", "pii", "text", "binary"):
                errs.append(f"{tname}: group_effects column {c} must be a modelled column (not a key, lookup or text)")
            if not g["by"]:
                errs.append(f"{tname}: group_effects on {c} needs 'by' (the columns that define a group)")
            for b in g["by"]:
                if "." in str(b):
                    try:
                        resolve_path(spec, tname, str(b))
                    except (SpecError, KeyError) as e:
                        errs.append(f"{tname}: group_effects by {b}: {e}")
                elif b not in t["columns"]:
                    errs.append(f"{tname}: group_effects by {b} is not a column of {tname}")
                elif b == c:
                    errs.append(f"{tname}: group_effects column {c} cannot group by itself")
        for la in t.get("link_affinity", []):
            m = la["match"]
            if len(m) != 2 or "." not in str(m[1]):
                errs.append(f"{tname}: link_affinity match needs two paths, the second through the link to move "
                            "(e.g. [CustomerKey.GeoAreaKey, StoreKey.GeoAreaKey])")
                continue
            for x in m:
                if "." in str(x):
                    try:
                        resolve_path(spec, tname, str(x))
                    except (SpecError, KeyError) as e:
                        errs.append(f"{tname}: link_affinity {x}: {e}")
                elif x not in t["columns"]:
                    errs.append(f"{tname}: link_affinity {x} is not a column of {tname}")
            if str(m[1]).count(".") != 1:
                errs.append(f"{tname}: link_affinity {m[1]} must be a column of a direct parent (one link)")
            if la.get("share") is not None and not 0 <= la["share"] <= 1:
                errs.append(f"{tname}: link_affinity share must be in [0, 1]")
        br = t.get("bridge")
        if br is not None:
            if not isinstance(br, dict) or br.get("group") not in t["columns"] or br.get("weight") not in t["columns"]:
                errs.append(f"{tname}: bridge needs group (the column whose rows share the weights) and weight "
                            "(a number column of the table)")
            elif t["columns"][br["weight"]]["type"] not in ("float", "integer"):
                errs.append(f"{tname}: bridge weight {br['weight']} must be a number column")
        if t.get("unknown_members"):
            if t["role"] != "entity" or not pk:
                errs.append(f"{tname}: unknown_members need an entity table with a primary key")
        for cname, c in t["columns"].items():
            d = c.get("distribution")
            if d is None:
                continue
            need = {"fixed": ["value"], "categorical": ["probabilities"], "uniform": ["min", "max"],
                    "normal": ["mean", "sd"], "lognormal": ["mean", "sd"], "poisson": ["mean"],
                    "date_range": ["start", "end"], "sequence": [], "empirical": []}
            if not isinstance(d, dict) or d.get("kind") not in need:
                errs.append(f"{tname}.{cname}: distribution needs kind: one of {sorted(need)}")
                continue
            miss = [k for k in need[d["kind"]] if d.get(k) is None]
            if miss:
                errs.append(f"{tname}.{cname}: distribution {d['kind']} needs {miss}")
            if d["kind"] == "categorical" and isinstance(d.get("probabilities"), dict) and \
                    (any(float(p) < 0 for p in d["probabilities"].values()) or sum(map(float, d["probabilities"].values())) <= 0):
                errs.append(f"{tname}.{cname}: categorical probabilities must be >= 0 with a positive sum")
            if c["type"] in ("id", "lookup", "pii", "text", "binary", "sequence"):
                errs.append(f"{tname}.{cname}: a distribution can be set on modelled columns only, not {c['type']}")
        sc = t.get("scd")
        if sc is not None:
            if t["role"] != "entity" or not isinstance(sc, dict):
                errs.append(f"{tname}: scd is supported on entity (dimension) tables, as a mapping")
            else:
                if sc.get("type", 2) != 2:
                    errs.append(f"{tname}: scd type {sc.get('type')} is not supported (type 2 is; types 0 and 1 need "
                                "no setting: one row per key)")
                for k in ("natural_key", "valid_from"):
                    if sc.get(k) not in t["columns"]:
                        errs.append(f"{tname}: scd {k} must name a column")
                if sc.get("valid_to") is not None and sc["valid_to"] not in t["columns"]:
                    errs.append(f"{tname}: scd valid_to must name a column")
                if isinstance(sc.get("stable"), list):
                    for c in sc["stable"]:
                        if c not in t["columns"]:
                            errs.append(f"{tname}: scd stable column {c} is not a column")
                cur = sc.get("current")
                if cur is not None and (not isinstance(cur, dict) or cur.get("column") not in t["columns"]):
                    errs.append(f"{tname}: scd current must be {{column, value, other}} with a column of {tname}")
                elif isinstance(cur, dict) and set(cur) - {"column", "value", "other"}:
                    errs.append(f"{tname}: unknown scd current keys {sorted(set(cur) - {'column', 'value', 'other'})} "
                                "(allowed: column, value, other)")
                bad = set(sc) - {"type", "natural_key", "valid_from", "valid_to", "stable", "current"}
                if bad:
                    errs.append(f"{tname}: unknown scd keys {sorted(bad)}")
        for d in t.get("dimensions", []):
            if t["role"] != "entity":
                errs.append(f"{tname}: dimensions are supported on entity tables")
            for c in [d.get("key")] + d["columns"]:
                if c not in t["columns"]:
                    errs.append(f"{tname}: dimension column {c} is not a column")
            is_link = any(r.get("child") == tname and r.get("child_key") == d.get("key") for r in rels)
            if d.get("key") == pk or (is_link and d.get("anchor") != d.get("key")):
                # a link may be the key only when self-anchored: the link is drawn as usual and
                # the columns it determines (another link, a code) are read from its observed rows
                errs.append(f"{tname}: dimension key {d.get('key')} must be a plain column, or a link with "
                            f"anchor: {d.get('key')} (not the primary key)")
            if not d["columns"]:
                errs.append(f"{tname}: dimension {d.get('key')} needs the columns it determines")
            if d.get("anchor") is not None and d["anchor"] not in d["columns"] and d["anchor"] != d.get("key"):
                errs.append(f"{tname}: dimension anchor {d['anchor']} must be one of its columns or its key")
        if pk and pk not in t["columns"]:
            errs.append(f"{tname}: primary_key {pk} is not a declared column")
        for pu in t.get("partial_unique", []):
            if t["role"] != "fact":
                errs.append(f"{tname}: partial unique keys (with 'where') are supported on fact tables")
            for c in pu["columns"]:
                if c not in t["columns"]:
                    errs.append(f"{tname}: unique key column {c} is not a declared column")
            try:
                from .expr import names_in
                for nme in names_in(pu["where"]):
                    if "." in nme:
                        resolve_path(spec, tname, nme)
                    elif nme not in t["columns"]:
                        errs.append(f"{tname}: unique-key filter uses {nme}, not a column")
            except (SpecError, ValueError, SyntaxError) as e:
                errs.append(f"{tname}: unique-key filter: {e}")
        for u in t.get("unique", []):
            for c in u:
                if c not in t["columns"]:
                    errs.append(f"{tname}: unique key column {c} is not a declared column")
        for cname, c in t["columns"].items():
            if c.get("type") not in COLUMN_TYPES:
                errs.append(f"{tname}.{cname}: type {c.get('type')!r} not in {sorted(COLUMN_TYPES)}")
            if c.get("type") == "pii":
                g = (c.get("pii") or {}).get("generator")
                if g not in PII_GENERATORS:
                    errs.append(f"{tname}.{cname}: pii.generator must be one of {sorted(PII_GENERATORS)}")
            if c.get("type") == "text":
                cond = (c.get("text") or {}).get("condition_on")
                if cond and cond not in t["columns"]:
                    errs.append(f"{tname}.{cname}: text.condition_on {cond} is not a column")
            if c.get("type") == "lookup" and isinstance(c.get("lookup"), dict):
                errs += _validate_keyed(spec, tname, cname, c["lookup"])
            elif c.get("type") == "lookup":
                if not c.get("lookup") or "." not in str(c.get("lookup")):
                    errs.append(f"{tname}.{cname}: lookup needs a path like 'fk_column.parent_column'")
                else:
                    try:
                        resolve_path(spec, tname, c["lookup"])
                    except SpecError as e:
                        errs.append(f"{tname}.{cname}: {e}")
        for cname, c in t["columns"].items():
            if c.get("map") is not None and (c.get("type") != "lookup" or not isinstance(c["map"], dict)):
                errs.append(f"{tname}.{cname}: map needs a lookup column and a mapping {{value: mapped value}}")
            if c.get("offset_from") is not None:
                b = t["columns"].get(c["offset_from"])
                if c.get("type") not in ("date", "datetime", "integer", "float"):
                    errs.append(f"{tname}.{cname}: offset_from needs a date, datetime or number column")
                elif b is None or b.get("type") not in ("date", "datetime", "integer", "float", "lookup"):
                    errs.append(f"{tname}.{cname}: offset_from {c['offset_from']} must be a date or number column "
                                f"of {tname}")
                elif b.get("offset_from") is not None:
                    errs.append(f"{tname}.{cname}: offset_from {c['offset_from']} is itself an offset; use its base")
            if c.get("ratio_of") is not None:
                if c.get("type") not in ("float", "integer"):
                    errs.append(f"{tname}.{cname}: ratio_of needs a number column")
                ro = str(c["ratio_of"])
                if "." in ro:
                    try:
                        resolve_path(spec, tname, ro)
                    except SpecError as e:
                        errs.append(f"{tname}.{cname}: ratio_of: {e}")
                elif ro not in t["columns"] or ro == cname:
                    errs.append(f"{tname}.{cname}: ratio_of {ro} is not another column of {tname}")
                sa = c.get("ratio_same_as")
                if sa is not None and (sa not in t["columns"] or t["columns"][sa].get("ratio_of") is None
                                       or t["columns"][sa].get("ratio_same_as") is not None):
                    errs.append(f"{tname}.{cname}: ratio_same_as {sa} must be a ratio column of {tname} "
                                "with its own factor")
            if c.get("type") == "sequence" and c.get("start") is not None and not isinstance(c["start"], int):
                errs.append(f"{tname}.{cname}: sequence start must be an integer (0 or 1)")
            if c.get("type") == "sequence":
                if t["role"] != "fact":
                    errs.append(f"{tname}.{cname}: sequence columns (1..n within a key) are supported on facts")
                if c.get("within") not in t["columns"]:
                    errs.append(f"{tname}.{cname}: a sequence needs 'within: <column>' (e.g. the order number)")
        if t["role"] == "fact" and t["fact"].get("header"):
            h = t["fact"]["header"]
            if not isinstance(h, dict) or h.get("key") not in t["columns"]:
                errs.append(f"{tname}: fact.header needs 'key: <column>' (the order or document number)")
            else:
                hc = list(h.get("columns") or [])
                if not hc:
                    errs.append(f"{tname}: fact.header needs 'columns: [...]' (values shared by all rows of a key)")
                for c in hc + ([h["line_number"]] if h.get("line_number") else []):
                    if c not in t["columns"]:
                        errs.append(f"{tname}: fact.header column {c} is not a column")
                if h["key"] in hc or (h.get("line_number") and h["line_number"] in hc):
                    errs.append(f"{tname}: fact.header.columns must not contain the key or the line number")
                if any(r.get("child") == tname and r.get("composite") for r in rels):
                    errs.append(f"{tname}: a fact with a header cannot itself be nested in another fact")
        if t["role"] == "fact":
            f = t["fact"]
            dims = [r["parent"] for r in rels if r.get("child") == tname]
            if not dims:
                errs.append(f"{tname}: fact table has no many_to_one relationships")
            if f.get("anchor") and f["anchor"] not in dims:
                errs.append(f"{tname}: fact.anchor {f['anchor']} is not one of its parents {dims}")
            for k in ("anchor", "partner") if user else ():
                # anchor_key / partner_key: set by normalise from a key column or relationship name
                if f.get(f"{k}_key") is not None and not any(
                        r.get("child") == tname and r.get("parent") == f.get(k) and r.get("child_key") == f[f"{k}_key"]
                        for r in rels):
                    errs.append(f"{tname}: fact.{k}_key {f[f'{k}_key']} is not a key of {tname} that references "
                                f"fact.{k} {f.get(k)} (name the key column in fact.{k} instead)")
            tc = f.get("time_column")
            if tc and tc not in t["columns"]:
                errs.append(f"{tname}: fact.time_column {tc} is not a column")
            if f.get("time_bucket", "month") not in ("day", "week", "month", "quarter", "year"):
                errs.append(f"{tname}: fact.time_bucket must be day/week/month/quarter/year")
            for k, allowed in (("pairs", ("auto", "poisson", "occupancy")),
                               ("multiplicity", ("auto", "none", "partner", "anchor", "block")),
                               ("count_model", ("auto", "poisson", "dp")),
                               ("group_fallback", ("auto", "anchor", "partner"))):
                if k in f and f[k] not in allowed:
                    errs.append(f"{tname}: fact.{k} must be one of {list(allowed)}")
    seen_fk = set()
    for r in rels:
        c, p = r.get("child"), r.get("parent")
        if c not in tables or p not in tables:
            errs.append(f"relationship {r.get('name')}: unknown table")
            continue
        if r["kind"] not in REL_KINDS:
            errs.append(f"relationship {r['name']}: kind must be one of {sorted(REL_KINDS)}")
        if r.get("composite"):
            if r["kind"] != "many_to_one":
                errs.append(f"relationship {r['name']}: composite keys are only supported for many_to_one")
            if len(r["child_key"]) != len(r["parent_key"]):
                errs.append(f"relationship {r['name']}: child_key and parent_key need the same length")
            for ck in r["child_key"]:
                if "." in ck:
                    try:
                        resolve_path(spec, c, ck)
                    except SpecError as e:
                        errs.append(f"relationship {r['name']}: {e}")
                elif ck not in tables[c]["columns"]:
                    errs.append(f"relationship {r['name']}: child_key {ck} is not a column of {c}")
            for pk_ in r["parent_key"]:
                if pk_ not in tables[p]["columns"]:
                    errs.append(f"relationship {r['name']}: parent_key {pk_} is not a column of {p}")
            keys = [[tables[p].get("primary_key")]] + tables[p].get("unique", [])
            if sorted(r["parent_key"]) not in [sorted(k) for k in keys if k and k[0]]:
                errs.append(f"relationship {r['name']}: parent_key {r['parent_key']} must be the primary key "
                            f"or a declared unique key of {p}")
            if tables[c]["role"] != "fact":
                errs.append(f"relationship {r['name']}: composite references are supported from fact tables")
        else:
            if r.get("child_key") not in tables[c]["columns"]:
                errs.append(f"relationship {r['name']}: child_key {r.get('child_key')} is not a column of {c}")
            ppk = tables[p].get("primary_key")
            if r.get("parent_key") not in (None, ppk):
                errs.append(f"relationship {r['name']}: parent_key {r.get('parent_key')} is not the primary key "
                            f"{ppk} of {p}; single-column links join on the parent's primary key (declare a "
                            "composite link with lists for other keys)")
            if r.get("offset_from") is not None:
                base = next((x for x in rels if x.get("child") == c and x.get("child_key") == r["offset_from"]
                             and not x.get("composite")), None)
                if base is None or base.get("parent") != p:
                    errs.append(f"relationship {r['name']}: offset_from {r['offset_from']} must be another key of "
                                f"{c} that references {p}")
                if base is not None and base.get("offset_from") is not None:
                    errs.append(f"relationship {r['name']}: offset_from {r['offset_from']} is itself an offset key; "
                                "use the base key")
                if r.get("order_by") not in tables[p]["columns"]:
                    errs.append(f"relationship {r['name']}: an offset key needs order_by: a column of {p} "
                                "(one value per row, e.g. the calendar date)")
            if tables[p]["role"] == "view":
                errs.append(f"relationship {r['name']}: parent {p} is a view; reference its base table")
            elif tables[p]["role"] != "entity":
                errs.append(f"relationship {r['name']}: parent {p} is a fact; reference it with a composite key "
                            f"(child_key: [..], parent_key: a unique key of {p})")
        if r["kind"] == "self_reference" and c != p:
            errs.append(f"relationship {r['name']}: self_reference needs child == parent")
        if tables[c]["role"] == "fact" and r["kind"] == "self_reference":
            errs.append(f"relationship {r['name']}: facts cannot self-reference")
        key = (c, str(r.get("child_key")))
        if key in seen_fk:
            errs.append(f"relationship {r['name']}: {c}.{r.get('child_key')} is used by two relationships")
        seen_fk.add(key)
    for tname, t in tables.items():
        nested = [r for r in rels if r.get("child") == tname and r.get("composite")]
        if len(nested) > 1:
            errs.append(f"{tname}: at most one composite (nested) parent per fact is supported")
        if nested and t["role"] == "fact" and t["fact"].get("anchor") not in (None, nested[0]["parent"]):
            errs.append(f"{tname}: a fact nested under {nested[0]['parent']} must use it as anchor")
    trees = [r for r in rels if r.get("kind") == "self_reference"]
    if len({r["child"] for r in trees}) < len(trees):
        errs.append("at most one self_reference per table is supported")
    try:
        topo_order(list(tables), table_edges(spec))
    except SpecError as e:
        errs.append(str(e))
    for i, rule in enumerate(spec.get("rules", [])):
        if rule.get("type") not in RULE_TYPES:
            errs.append(f"rule {i}: type must be one of {sorted(RULE_TYPES)}")
        t = rule.get("table")
        if t not in tables:
            errs.append(f"rule {i}: unknown table {t}")
            continue
        tol = rule.get("tolerance")
        if tol is not None and not (isinstance(tol, (int, float)) and tol >= 0):
            errs.append(f"rule {i}: tolerance must be a number >= 0")
        if rule.get("type") == "soft":
            h = rule.get("holds")
            if not (isinstance(h, (int, float)) and 0 < h < 1):
                errs.append(f"rule {i}: holds must be a share between 0 and 1")
            if rule.get("violation", "offset") not in SOFT_VIOLATIONS:
                errs.append(f"rule {i}: violation must be one of {sorted(SOFT_VIOLATIONS)}")
        if rule.get("type") in ("check", "derive", "soft"):
            expr = str(rule.get("expr", ""))
            try:
                from .expr import names_in, split_assignment
                if rule["type"] in ("derive", "soft"):
                    lhs, rhs = split_assignment(expr)
                    if lhs not in tables[t]["columns"]:
                        errs.append(f"rule {i}: derived column {lhs} is not a column of {t}")
                    used = names_in(rhs)
                else:
                    used = names_in(expr)
                for nme in used:
                    if "." in nme:
                        resolve_path(spec, t, nme)
                    elif nme not in tables[t]["columns"]:
                        errs.append(f"rule {i}: {nme} is not a column of {t}")
            except (SpecError, ValueError, SyntaxError) as e:
                errs.append(f"rule {i}: {e}")
        if rule.get("type") == "cumulative":
            for o in ("opening", "floor"):
                if rule.get(o) is not None and not isinstance(rule[o], (int, float)):
                    errs.append(f"rule {i}: cumulative rule {o} must be a number")
            by = rule.get("by")
            by = [by] if isinstance(by, str) else list(by or [])
            for c in [rule.get("column"), rule.get("order")] + by:
                if c not in tables[t]["columns"]:
                    errs.append(f"rule {i}: cumulative rule column {c} is not a column of {t}")
            try:
                from .expr import names_in
                for nme in names_in(str(rule.get("step", ""))):
                    if "." in nme:
                        resolve_path(spec, t, nme)
                    elif nme not in tables[t]["columns"]:
                        errs.append(f"rule {i}: step uses {nme}, not a column of {t}")
            except (SpecError, ValueError, SyntaxError) as e:
                errs.append(f"rule {i}: step: {e}")
        if rule.get("type") == "aggregate":
            if rule.get("column") not in tables[t]["columns"]:
                errs.append(f"rule {i}: aggregate column {rule.get('column')} is not a column of {t}")
            if rule.get("agg", "count") not in AGGREGATES:
                errs.append(f"rule {i}: agg must be one of {sorted(AGGREGATES)}")
            src = rule.get("from")
            if src not in tables:
                errs.append(f"rule {i}: aggregate needs 'from: <child table>'")
            else:
                links = [r for r in rels if r.get("child") == src and r.get("parent") == t
                         and (rule.get("via") in (None, r["name"]))]
                if len(links) != 1:
                    errs.append(f"rule {i}: need exactly one relationship from {src} to {t} (set 'via' to its name)")
                of = rule.get("of")
                if rule.get("agg", "count") not in ("count", "exists") and of is None:
                    errs.append(f"rule {i}: aggregate '{rule.get('agg')}' needs 'of: <column or expression of {src}>'")
                elif of is not None and of not in tables[src]["columns"]:
                    from .expr import names_in
                    try:
                        for nme in names_in(of):
                            if "." in nme:
                                resolve_path(spec, src, nme)
                            elif nme not in tables[src]["columns"]:
                                errs.append(f"rule {i}: 'of' uses {nme}, which is not a column of {src}")
                    except (SpecError, ValueError, SyntaxError) as e:
                        errs.append(f"rule {i}: 'of': {e}")
    if not errs:
        cyc = computed_cycle(spec)
        if cyc:
            errs.append("computed columns depend on each other in a cycle: " + " -> ".join(cyc)
                        + " (keep one direction: a lookup from the parent or an aggregate of the children)")
    sc = spec["synthesis"]["scale"] if isinstance(spec["synthesis"].get("scale"), dict) else {}
    for tn, v in (sc.get("tables") or {}).items():
        if tn not in tables:
            errs.append(f"synthesis.scale.tables: unknown table {tn}")
        elif tables[tn]["role"] == "fact":
            errs.append(f"synthesis.scale.tables.{tn}: facts are not scaled directly; scale their parents")
        if not (isinstance(v, (int, float)) and v > 0):
            errs.append(f"synthesis.scale.tables.{tn}: must be > 0")
    return errs


SOURCE_KINDS = ("csv", "files", "parquet", "sqlite", "postgresql", "mssql", "oracle", "mysql", "mariadb", "databricks",
                "fabric", "snowflake", "sqlalchemy")
TARGET_KINDS = SOURCE_KINDS
WAREHOUSE_KINDS = ("databricks", "fabric", "snowflake")
SERVER_KINDS = ("postgresql", "mssql", "oracle", "mysql", "mariadb", "databricks", "fabric", "snowflake", "sqlalchemy")
TARGET_STRUCTURES = ("auto", "clone", "infer", "existing")
IF_EXISTS = ("replace", "append", "truncate", "fail")


def _bad_name(name):
    """Why a table name cannot be a file name, or None."""
    n = str(name)
    if not n.strip():
        return "empty"
    if any(ch in n for ch in "/\\") or any(ord(ch) < 32 or ord(ch) == 127 for ch in n):
        return "contains '/', '\\', NUL or a control character"
    if n in (".", ".."):
        return "is a '.' or '..' path component"
    return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_io(spec):
    """source, target and the types of the synthesis settings."""
    errs = []
    for key, kinds in (("source", SOURCE_KINDS), ("target", TARGET_KINDS)):
        x = spec.get(key)
        if x is None:
            continue
        if not isinstance(x, dict):
            errs.append(f"{key} must be a mapping {{kind, path or url, ...}}")
            continue
        if x.get("kind", "csv") not in kinds:
            errs.append(f"{key}.kind must be one of {', '.join(kinds)}, not {x.get('kind')!r}")
        for k in ("path", "url", "schema", "url_env", "password_env", "format"):
            if x.get(k) is not None and not isinstance(x[k], str):
                errs.append(f"{key}.{k} must be a string")
        if x.get("engine") is not None and x["engine"] not in ("auto", "duckdb", "pandas"):
            errs.append(f"{key}.engine must be auto, duckdb or pandas")
        if x.get("duckdb") is not None and (not isinstance(x["duckdb"], dict) or
                                            set(x["duckdb"]) - {"memory_limit", "threads", "temp_directory"}):
            errs.append(f"{key}.duckdb must be a mapping of memory_limit, threads and temp_directory")
        for k in ("connect_args", "engine_options", "auth", "guards", "bulk"):
            if x.get(k) is not None and not isinstance(x[k], dict):
                errs.append(f"{key}.{k} must be a mapping")
        if any(x.get(k) is not None for k in ("auth", "guards", "bulk")) and x.get("kind") not in WAREHOUSE_KINDS:
            errs.append(f"{key}: auth, guards and bulk are for the cloud warehouses ({', '.join(WAREHOUSE_KINDS)})")
        if x.get("kind", "csv") in SERVER_KINDS and not (x.get("url") or x.get("url_env")):
            errs.append(f"{key}: kind {x.get('kind')} needs `url` or `url_env`")
        if x.get("format") is not None and x["format"] not in ("csv", "parquet"):
            errs.append(f"{key}.format must be csv or parquet")
        nv = x.get("null_values")
        if nv is not None and not (isinstance(nv, list) and all(isinstance(v, str) for v in nv)):
            errs.append(f"{key}.null_values must be a list of strings (e.g. ['', 'NULL'])")
    tgt = spec.get("target")
    if isinstance(tgt, dict) and tgt.get("if_exists", "replace") not in IF_EXISTS:
        errs.append(f"target.if_exists must be one of {', '.join(IF_EXISTS)}, not {tgt.get('if_exists')!r}")
    if isinstance(tgt, dict) and tgt.get("structure", "auto") not in TARGET_STRUCTURES:
        errs.append(f"target.structure must be one of {', '.join(TARGET_STRUCTURES)}, not {tgt.get('structure')!r}")
    from .schema import FREE
    syn = spec.get("synthesis") or {}

    def blocks(d, defaults, pre):
        for k, dv in defaults.items():
            if k not in d or f"{pre}{k}" in FREE:
                continue
            if isinstance(dv, dict):
                if not isinstance(d[k], dict):
                    errs.append(f"synthesis.{pre}{k} must be a mapping, not {d[k]!r}")
                else:
                    blocks(d[k], dv, f"{pre}{k}.")
    blocks(syn, DEFAULT_SYNTHESIS, "")
    if not _is_int(syn.get("seed")):
        errs.append(f"synthesis.seed must be an integer, not {syn.get('seed')!r}")
    pc = syn.get("posterior") if isinstance(syn.get("posterior"), dict) else {}
    if not (_is_int(pc.get("datasets", 1)) and pc.get("datasets", 1) >= 1):
        errs.append(f"synthesis.posterior.datasets must be a positive integer, not {pc.get('datasets')!r}")
    if pc.get("mode", "draw") not in ("draw", "mean"):
        errs.append(f"synthesis.posterior.mode must be draw or mean, not {pc.get('mode')!r}")
    if pc.get("sample_seed") is not None and not _is_int(pc["sample_seed"]):
        errs.append("synthesis.posterior.sample_seed must be an integer")
    st = syn.get("structure") if isinstance(syn.get("structure"), dict) else {}
    v = st.get("structural_prior", 0.9)
    if not (_is_num(v) and 0 <= v < 1):
        errs.append(f"synthesis.structure.structural_prior must be in [0, 1), not {v!r}")
    v = st.get("structural_pooling", 3.0)
    if not (_is_num(v) and v >= 0):
        errs.append(f"synthesis.structure.structural_pooling must be a number >= 0, not {v!r}")
    cp = syn.get("copula") if isinstance(syn.get("copula"), dict) else {}
    v = cp.get("pooled_cells", 1e6)
    if not (_is_num(v) and v > 0):
        errs.append(f"synthesis.copula.pooled_cells must be a number > 0, not {v!r}")
    pd_ = syn.get("pushdown") if isinstance(syn.get("pushdown"), dict) else {}
    if pd_.get("mode", "auto") not in ("auto", "on", "off"):
        errs.append(f"synthesis.pushdown.mode must be auto, on or off, not {pd_.get('mode')!r}")
    for k, lo in (("min_rows", 0), ("sample_rows", 1)):
        v = pd_.get(k, DEFAULT_SYNTHESIS["pushdown"][k])
        if not (_is_int(v) and v >= lo):
            errs.append(f"synthesis.pushdown.{k} must be an integer >= {lo}, not {v!r}")
    return errs


def _validate_view(spec, tname, t):
    """A view is its base table's rows with the base's columns of the same names,
    columns looked up through its links, and derived columns."""
    errs = []
    tables = spec["tables"]
    b = t.get("view_of")
    if b not in tables or tables[b]["role"] == "view":
        return [f"{tname}: a view needs view_of: <a table that is not a view>"]
    derived = {r["expr"].split("=")[0].strip() for r in spec.get("rules", [])
               if r.get("table") == tname and r.get("type") == "derive"}
    for c, cs in t["columns"].items():
        bc = cs.get("base_column", c)
        if cs.get("type") != "lookup" and c not in derived and bc not in tables[b]["columns"]:
            errs.append(f"{tname}.{c}: a view column must be a column of {b} (or name one with base_column), "
                        "a lookup or a derived column")
    for r in spec.get("rules", []):
        if r.get("table") == tname and r.get("type") not in ("derive", "check"):
            errs.append(f"{tname}: views take derive and check rules only")
        if r.get("type") == "aggregate" and r.get("from") == tname:
            errs.append(f"aggregate {r.get('table')}.{r.get('column')}: aggregate the base table {b}, not the view")
    for r in spec.get("relationships", []):
        if r.get("child") == tname and (r.get("composite") or r.get("kind") != "many_to_one"):
            errs.append(f"relationship {r.get('name')}: a view's links must be single-column many_to_one")
    return errs


def _validate_keyed(spec, tname, cname, lk):
    errs = []
    R = lk.get("table")
    if R not in spec["tables"] and R not in (spec.get("_refs") or {}):
        return [f"{tname}.{cname}: keyed lookup table {R} is not a table"]
    rcols = spec["tables"][R]["columns"] if R in spec["tables"] else spec["_refs"][R]
    if lk.get("value") not in rcols:
        errs.append(f"{tname}.{cname}: keyed lookup value {lk.get('value')} is not a column of {R}")
    match = lk.get("match")
    if not isinstance(match, dict) or not match:
        return errs + [f"{tname}.{cname}: keyed lookup needs match: {{column of {R}: expression of {tname}}}"]
    from .expr import names_in
    for rc, e in match.items():
        if rc not in rcols:
            errs.append(f"{tname}.{cname}: keyed lookup matches {rc}, not a column of {R}")
        try:
            for nme in names_in(str(e)):
                if "." in nme:
                    resolve_path(spec, tname, nme)
                elif nme not in spec["tables"][tname]["columns"]:
                    errs.append(f"{tname}.{cname}: keyed lookup reads {nme}, not a column of {tname}")
        except (SpecError, ValueError, SyntaxError) as ex:
            errs.append(f"{tname}.{cname}: keyed lookup: {ex}")
    return errs


def computed_inputs(spec) -> dict:
    """{'t.c': {'t2.c2', ...}}: the columns each computed column (lookup, derive,
    aggregate) reads, following paths through foreign keys."""
    from .expr import names_in, split_assignment
    T = spec["tables"]
    deps = {}

    def path_cols(t, path):
        out = set()
        try:
            hops, (pt, pc) = resolve_path(spec, t, path)
        except SpecError:
            return out
        cur = t
        for h in hops:
            out.add(f"{cur}.{h['child_key']}")
            cur = h["parent"]
        out.add(f"{pt}.{pc}")
        return out

    def expr_cols(t, expr):
        out = set()
        for nme in names_in(expr):
            out |= path_cols(t, nme) if "." in nme else {f"{t}.{nme}"}
        return out
    for t, ts in T.items():
        for c, cs in ts["columns"].items():
            if cs.get("type") == "lookup":
                lk = cs.get("lookup")
                if isinstance(lk, str):
                    deps[f"{t}.{c}"] = path_cols(t, lk)
                elif isinstance(lk, dict):
                    d = {f"{lk['table']}.{lk['value']}"} | {f"{lk['table']}.{k}" for k in lk.get("match", {})}
                    for v in lk.get("match", {}).values():
                        if isinstance(v, str) and not v.startswith("'"):
                            d |= path_cols(t, v) if "." in v else {f"{t}.{v}"}
                    deps[f"{t}.{c}"] = d
    for r in spec.get("rules", []):
        if r.get("type") == "derive":
            lhs, rhs = split_assignment(r["expr"])
            deps[f"{r['table']}.{lhs}"] = expr_cols(r["table"], rhs)
        elif r.get("type") == "aggregate" and r.get("from") in T:
            deps[f"{r['table']}.{r['column']}"] = expr_cols(r["from"], r["of"]) if r.get("of") else set()
    return deps


def computed_cycle(spec):
    """A cycle among computed columns (each needs the other to be computed first),
    as a list of 't.c' names, or None."""
    deps = computed_inputs(spec)
    state, stack = {}, []

    def visit(n):
        state[n] = 1
        stack.append(n)
        for m in deps.get(n, ()):
            if m not in deps:
                continue
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state:
                cyc = visit(m)
                if cyc:
                    return cyc
        state[n] = 2
        stack.pop()
        return None
    for n in deps:
        if n not in state:
            cyc = visit(n)
            if cyc:
                return cyc
    return None


def inherit_scale(spec, sc, skip=()) -> dict:
    """The scale section with every entity's factor: its own when set, else that of
    the first parent it links to that has one (orders follow their customers, and
    their lines follow them as facts), else the default.  Copied tables never pass
    a factor on; tables in `skip` are left to the caller."""
    sc = {"default": sc.get("default", 1.0), "tables": dict(sc.get("tables", {}) or {})}
    T = spec["tables"]
    for t in topo_order(list(T), table_edges(spec)):
        if t in sc["tables"] or t in skip or T[t].get("role") != "entity" or T[t].get("mode") == "copy":
            continue
        for r in spec["relationships"]:
            p = r["parent"]
            if r["child"] == t and p != t and r.get("kind", "many_to_one") == "many_to_one" \
                    and p in sc["tables"] and T.get(p, {}).get("mode") != "copy":
                sc["tables"][t] = sc["tables"][p]
                break
    return sc


def topo_order(nodes, edges):
    """Parents before children. edges are (child, parent)."""
    parents = {n: set() for n in nodes}
    for c, p in edges:
        if c != p:
            parents[c].add(p)
    order, done = [], set()
    while len(order) < len(nodes):
        ready = [n for n in nodes if n not in done and parents[n] <= done]
        if not ready:
            raise SpecError("relationships contain a cycle between different tables: "
                            + ", ".join(sorted(set(nodes) - done)))
        for n in ready:
            order.append(n)
            done.add(n)
    return order


def dump_spec(spec: dict, path):
    clean = {k: v for k, v in spec.items() if not k.startswith("_")}
    Path(path).write_text(yaml.safe_dump(clean, sort_keys=False, width=100))


def spec_hash(spec: dict) -> str:
    import hashlib
    clean = {k: v for k, v in spec.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]

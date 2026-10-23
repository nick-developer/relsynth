"""Whole-model report: what the specification says about the data model.

`describe(spec)` classifies every table with the dimensional-modelling
vocabulary (Kimball & Ross, *The Data Warehouse Toolkit*, 3rd ed., ch. 2):

* facts: transaction (with an order header when lines share one), periodic
  snapshot, accumulating snapshot, factless, bridge;
* dimensions: calendar, reference (copied), slowly changing type 2,
  role-playing, conformed, outrigger, with unknown members;
* per fact: star (every dimension one join away) or snowflake (a dimension
  that has its own parent dimension);

and lists the join graph, the hierarchies (snowflake chains, parent-child
trees, levels implied inside a table), degenerate dimensions, audit columns,
the rules and computed columns, the structural effects (group effects, link
affinity, soft rules, participation) and the open questions of the draft.
`to_markdown` renders it with a Mermaid ER diagram.  Only the specification is
read; `counts` (rows per table) is optional.
"""
from __future__ import annotations

import re

AUDIT = re.compile(r"(^|_)(created|modified|updated|inserted|changed|load(ed)?|etl|audit|batch|rowversion|"
                   r"deleted|is_?deleted|valid_?flag|last_?update|timestamp)($|_|date|time|at|by|on|id|key)", re.I)
CAL_NAME = re.compile(r"(date|calendar|time|period|day|month)", re.I)


def _cols_of_type(T, *types):
    return [c for c, cs in T["columns"].items() if cs["type"] in types]


def describe(spec, counts=None):
    T = spec["tables"]
    rels = [r for r in spec["relationships"] if not r.get("composite")]
    counts = counts or {}
    facts = [t for t, x in T.items() if x["role"] == "fact"]
    ents = [t for t, x in T.items() if x["role"] == "entity"]
    by_child = {}
    for r in rels:
        by_child.setdefault(r["child"], []).append(r)

    def is_calendar(t):
        x = T[t]
        pk = x.get("primary_key")
        if pk and x["columns"].get(pk, {}).get("type") in ("date", "datetime"):
            return True
        return bool(CAL_NAME.search(t)) and len(_cols_of_type(x, "date", "datetime")) >= 1 and \
            sum(1 for c in x["columns"] if re.search(r"year|month|quarter|week|day", c, re.I)) >= 2
    calendars = {t for t in ents if is_calendar(t)}

    def dims_of(t):
        return [r for r in by_child.get(t, []) if r["parent"] != t]

    views = {t: x.get("view_of") for t, x in T.items() if x["role"] == "view"}
    rels = [r for r in rels if r["child"] not in views and r["parent"] not in views]
    by_child = {}
    for r in rels:
        by_child.setdefault(r["child"], []).append(r)
    # header entities: an entity whose key is part of a fact's grain (an order and its lines)
    headers = {}
    for f in facts:
        for r in dims_of(f):
            if T[r["parent"]]["role"] == "entity" and any(r["child_key"] in u and len(u) >= 2
                                                          for u in T[f].get("unique", [])):
                headers[r["parent"]] = f

    model = {"tables": {}, "facts": {}, "dimensions": {}, "relationships": [], "hierarchies": [], "rules": [],
             "computed": [], "effects": [], "questions": []}
    for r in rels:
        model["relationships"].append({"child": r["child"], "key": r["child_key"], "parent": r["parent"],
                                       "kind": r.get("kind", "many_to_one"),
                                       **({"offset_from": r["offset_from"]} if r.get("offset_from") else {})})
    # ---- facts
    for f in facts:
        x = T[f]
        fk = {r["child_key"] for r in by_child.get(f, [])}
        keys = fk | {x.get("primary_key")}
        meas = [c for c, cs in x["columns"].items() if cs["type"] in ("float", "integer") and c not in keys]
        dates = [c for c, cs in x["columns"].items() if cs["type"] in ("date", "datetime")]
        date_fks = [r["child_key"] for r in by_child.get(f, []) if r["parent"] in calendars]
        hd = x.get("fact", {}).get("header")
        kinds = []
        uniq = [set(u) for u in x.get("unique", [])]
        if x.get("bridge"):
            kinds.append("bridge (weighted, multivalued dimension)")
        elif hd:
            kinds.append("transaction (order header and lines)")
        elif len(dates) + len(date_fks) >= 3 and any(cs.get("offset_from") for cs in x["columns"].values()) or \
                sum(1 for r in by_child.get(f, []) if r.get("offset_from")) >= 2:
            kinds.append("accumulating snapshot (one row per process, several milestone dates)")
        elif any(u & (set(date_fks) | set(dates)) and len(u) >= 2 for u in uniq) and meas:
            kinds.append("periodic snapshot (one row per member and period)")
        elif not meas:
            kinds.append("factless (events or coverage, no measures)")
        else:
            kinds.append("transaction")
        dparents = dims_of(f)
        # the dimensions of an order header are the fact's own (one join through a degenerate key)
        dparents = [r for r in dparents if headers.get(r["parent"]) != f] + \
            [q for r in dparents if headers.get(r["parent"]) == f for q in dims_of(r["parent"])]
        snow = sorted({r["parent"] for r in dparents if any(
            q["parent"] not in calendars and q["parent"] != r["parent"] for q in dims_of(r["parent"]))})
        degenerate = [c for c, cs in x["columns"].items() if cs["type"] == "id" and c not in keys]
        if hd:
            degenerate.append(hd["key"])
        degenerate += [r["child_key"] for r in dims_of(f) if headers.get(r["parent"]) == f]
        roles = {}
        for r in dparents:
            roles.setdefault(r["parent"], []).append(r["child_key"])
        model["facts"][f] = {
            "kind": kinds[0], "grain": x.get("unique") or ([x["primary_key"]] if x.get("primary_key") else []),
            "anchor": x.get("fact", {}).get("anchor"), "partner": x.get("fact", {}).get("partner"),
            "dimensions": sorted({r["parent"] for r in dparents}),
            "shape": "snowflake" if snow else "star", "snowflaked_through": snow,
            "measures": meas, "dates": dates + date_fks, "degenerate_dimensions": degenerate,
            "role_playing": {p: ks for p, ks in roles.items() if len(ks) >= 2},
            "rows": counts.get(f)}
    # ---- dimensions / entities
    for e in ents:
        x = T[e]
        refs = [r for r in rels if r["parent"] == e and r["child"] != e]
        by_tab = {}
        for r in refs:
            by_tab.setdefault(r["child"], []).append(r["child_key"])
        fact_refs = sorted({r["child"] for r in refs if T[r["child"]]["role"] == "fact"})
        kinds = []
        if e in headers:
            kinds.append(f"order header of {headers[e]} (one row per document)")
        if e in calendars:
            kinds.append("calendar")
        if x.get("mode") == "copy":
            kinds.append("reference (copied as-is)")
        if x.get("scd"):
            kinds.append("slowly changing, type 2")
        if any(len(v) >= 2 for v in by_tab.values()):
            kinds.append("role-playing")
        if len(fact_refs) >= 2:
            kinds.append("conformed")
        if refs and not fact_refs and any(T[r["child"]]["role"] == "entity" and r["child"] not in headers
                                          for r in refs):
            kinds.append("outrigger / snowflake level")
        if any(r["parent"] == e and r["child"] == e for r in rels):
            kinds.append("parent-child hierarchy")
        if x.get("unknown_members"):
            kinds.append("has unknown members")
        audit = [c for c in x["columns"] if AUDIT.search(c)]
        model["dimensions"][e] = {
            "kinds": kinds or ["dimension"], "primary_key": x.get("primary_key"), "rows": counts.get(e),
            "referenced_by": {k: v for k, v in by_tab.items()}, "parents": sorted({r["parent"] for r in dims_of(e)}),
            "scd": x.get("scd"), "unknown_members": x.get("unknown_members"),
            "implied_levels": [{"key": d["key"], "levels": d["columns"]} for d in x.get("dimensions", [])],
            "audit_columns": audit}
    for f, x in T.items():
        if x["role"] == "fact":
            model["facts"][f]["audit_columns"] = [c for c in x["columns"] if AUDIT.search(c)]
    # ---- hierarchies
    for e in ents:
        chain, cur, seen = [e], e, {e}
        while True:
            ps = [r["parent"] for r in dims_of(cur) if r["parent"] not in calendars and r["parent"] not in seen
                  and T[r["parent"]]["role"] == "entity"]
            if len(ps) != 1:
                break
            cur = ps[0]
            chain.append(cur)
            seen.add(cur)
        if len(chain) >= 2 and not any(r["parent"] == e and T[r["child"]]["role"] == "entity" and
                                       r["child"] != e for r in rels):
            model["hierarchies"].append({"kind": "snowflake chain", "levels": chain})
        for r in rels:
            if r["child"] == e and r["parent"] == e:
                model["hierarchies"].append({"kind": "parent-child (ragged) tree", "table": e, "key": r["child_key"]})
        for d in T[e].get("dimensions", []):
            model["hierarchies"].append({"kind": "levels inside the table", "table": e, "key": d["key"],
                                         "levels": d["columns"]})
    # ---- rules, computed columns, effects
    for r in spec.get("rules", []):
        model["rules"].append({k: r[k] for k in ("table", "type", "expr", "holds", "column", "agg", "from")
                               if k in r})
    for t, x in T.items():
        for c, cs in x["columns"].items():
            for k in ("lookup", "offset_from", "ratio_of"):
                if cs.get(k):
                    model["computed"].append({"table": t, "column": c, k: cs[k]})
        for g in x.get("group_effects", []):
            model["effects"].append({"table": t, "kind": "group effect", "column": g["column"], "by": g["by"]})
        for la in x.get("link_affinity", []):
            model["effects"].append({"table": t, "kind": "link affinity", "match": la["match"]})
        if x.get("bridge"):
            model["effects"].append({"table": t, "kind": "bridge weights", **x["bridge"]})
    model["views"] = {v: b for v, b in views.items()}
    model["questions"] = [q for q in spec.get("review", []) if str(q).startswith("ASK")]
    model["tables"] = {t: {"role": x["role"], "mode": x.get("mode", "synthesize"), "rows": counts.get(t),
                           "columns": len(x["columns"])} for t, x in T.items()}
    return model


def _txt(v):
    """Names and values from the spec or the data as markdown text: one line, HTML
    escaped: '&' and '<' (model.md is rendered by viewers that run HTML; '>' alone
    opens no tag and stays readable in 'a -> b')."""
    return " ".join(str(v).split()).replace("&", "&amp;").replace("<", "&lt;")


def _row(*cells):
    """A markdown table row; '|' inside a cell is escaped."""
    return "| " + " | ".join(_txt(c).replace("|", "\\|") for c in cells) + " |"


def _code(v):
    """Text in a code span of a table cell (literal there: no HTML, no backtick)."""
    return "`" + " ".join(str(v).split()).replace("`", "'").replace("|", "\\|") + "`"


def _label(v):
    """A Mermaid label: no double quote, one line."""
    return " ".join(str(v).split()).replace('"', "'")


def _mid(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def mermaid(spec):
    T = spec["tables"]
    lines = ["erDiagram"]
    for r in spec["relationships"]:
        if r.get("composite"):
            continue
        lines.append(f'    {_mid(r["parent"])} ||--o{{ {_mid(r["child"])} : "{_label(r["child_key"])}"')
    linked = {r["child"] for r in spec["relationships"]} | {r["parent"] for r in spec["relationships"]}
    for t, x in T.items():
        if t not in linked:
            lines.append(f"    {_mid(t)} {{\n        {_mid(x['role'])} _\n    }}")
    return "\n".join(lines)


def to_markdown(spec, model):
    L = [f"# Data model: {_txt(spec.get('name', ''))}", "",
         "Generated by `relsynth model` from the specification. Facts and dimensions use Kimball's vocabulary.", ""]
    L += ["## Join graph", "", "```mermaid", mermaid(spec), "```", ""]
    L += ["## Facts", "", "| Fact | Kind | Shape | Grain | Dimensions | Measures | Degenerate | Role-playing |",
          "|---|---|---|---|---|---|---|---|"]
    for f, x in model["facts"].items():
        rp = "; ".join(f"{p}: {', '.join(k)}" for p, k in x["role_playing"].items())
        shape = x["shape"] + (f" (through {', '.join(x['snowflaked_through'])})" if x["snowflaked_through"] else "")
        L.append(_row(f, x['kind'], shape, x['grain'], ', '.join(x['dimensions']),
                      f"{', '.join(x['measures'][:8])}{' ...' if len(x['measures']) > 8 else ''}",
                      ', '.join(x['degenerate_dimensions']), rp))
    L += ["", "## Dimensions and other entities", "",
          "| Table | Kinds | Key | Referenced by | Parents | Implied levels | Audit columns |", "|---|---|---|---|---|---|---|"]
    for e, x in model["dimensions"].items():
        rb = "; ".join(f"{t} ({', '.join(k)})" for t, k in x["referenced_by"].items())
        lv = "; ".join(f"{d['key']} -> {', '.join(d['levels'])}" for d in x["implied_levels"])
        kinds = ", ".join(x["kinds"])
        if x["scd"]:
            kinds += f" (natural key {x['scd']['natural_key']}, {x['scd']['valid_from']} to {x['scd'].get('valid_to', '-')})"
        if x["unknown_members"]:
            kinds += f" (unknown: {x['unknown_members']})"
        L.append(_row(e, kinds, x['primary_key'], rb, ', '.join(x['parents']), lv, ', '.join(x['audit_columns'])))
    if model.get("views"):
        L += ["", "## Views", ""] + [f"- {_txt(v)}: rebuilt from {_txt(b)} and its lookups"
                                     for v, b in model["views"].items()]
    if model["hierarchies"]:
        L += ["", "## Hierarchies", ""]
        for h in model["hierarchies"]:
            if h["kind"] == "snowflake chain":
                L.append(f"- Snowflake chain: {_txt(' -> '.join(h['levels']))}")
            elif h["kind"].startswith("parent-child"):
                L.append(f"- Parent-child tree: {_txt(h['table'])}.{_txt(h['key'])}")
            else:
                L.append(f"- Levels inside {_txt(h['table'])}: {_txt(h['key'])} -> {_txt(', '.join(h['levels']))}")
    if model["effects"]:
        L += ["", "## Structural effects", ""]
        for e in model["effects"]:
            if e["kind"] == "group effect":
                L.append(_txt(f"- {e['table']}: one {e['column']} per ({', '.join(map(str, e['by']))})"))
            elif e["kind"] == "link affinity":
                L.append(_txt(f"- {e['table']}: {e['match'][0]} = {e['match'][1]} more often than chance"))
            else:
                L.append(_txt(f"- {e['table']}: {e['weight']} sums to {e.get('total', 1)} per {e['group']}"))
    if model["rules"]:
        L += ["", "## Rules", "", "| Table | Type | Rule |", "|---|---|---|"]
        for r in model["rules"]:
            txt = r.get("expr") or f"{r.get('column')} = {r.get('agg')} of {r.get('from')}"
            if r.get("holds"):
                txt += f" (holds on {r['holds']:.2%})"
            L.append(_row(r['table'], r['type']) + " " + _code(txt) + " |")
    if model["computed"]:
        L += ["", "## Computed columns", "", "| Table | Column | Computed as |", "|---|---|---|"]
        for c in model["computed"]:
            how = (f"lookup {c['lookup']}" if "lookup" in c else f"{c['offset_from']} + gap" if "offset_from" in c
                   else f"{c['ratio_of']} x factor")
            L.append(_row(c['table'], c['column'], how))
    if model["questions"]:
        L += ["", "## Open questions", ""] + [f"- {_txt(q[4:] if q.startswith('ASK ') else q)}" for q in model["questions"]]
    return "\n".join(L) + "\n"

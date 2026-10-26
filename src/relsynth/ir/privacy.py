"""The privacy unit and the foreign-key paths that carry its data.

Differential privacy on a database protects a *unit* (a customer, a patient),
not a row: neighbouring databases differ in all the records of one unit
(Ponomareva et al., JAIR 2026; Cai, Xiao and Cormode, PrivLava, SIGMOD 2023).
A record belongs to a unit when it reaches the unit's record through a chain of
references: an invoice line -> its invoice -> the invoice's customer.

`trace(est)` finds those chains in the schema (Prop. 107): for each entity, every
simple path of many-to-one references from it to the unit entity.  Hierarchies
(an employee's manager, a customer who referred another) and references listed
in `privacy.exclude_paths` link records of *different* units and are not
followed; references into the unit's records that are not followed are reported,
because removing a unit then leaves them dangling (Prop. 108).

`contribution(est, data, unit_keys)` applies the paths to data: which records
of each entity belong to the given units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Estate, IRError

MAX_PATHS = 256


@dataclass
class EntityTrace:
    status: str                    # unit | owned | context | unrelated
    paths: list = field(default_factory=list)       # [[relationship name, ...], ...] to the unit
    optional: list = field(default_factory=list)    # per path: a reference on it is stated optional
    note: str = ""


@dataclass
class Trace:
    unit: str
    entities: dict = field(default_factory=dict)     # name -> EntityTrace
    cross_unit: list = field(default_factory=list)   # references between records of different units
    warnings: list = field(default_factory=list)

    def owned(self):
        return [n for n, t in self.entities.items() if t.status in ("unit", "owned")]

    def to_dict(self):
        return {"unit": self.unit, "cross_unit": self.cross_unit, "warnings": self.warnings,
                "entities": {n: {"status": t.status, "paths": t.paths, "optional": t.optional,
                                 **({"note": t.note} if t.note else {})} for n, t in self.entities.items()}}

    def to_markdown(self, est: Estate | None = None) -> str:
        out = [f"Privacy unit: **{self.unit}**", "", "| Entity | Status | Paths to the unit |", "|---|---|---|"]
        for n, t in self.entities.items():
            ps = "<br>".join(_path_text(p, est, n) + (" (may be null)" if o else "")
                             for p, o in zip(t.paths, t.optional)) or "—"
            out.append(f"| {n} | {t.status} | {ps} |")
        if self.cross_unit:
            out += ["", "Links between units (not followed): " + ", ".join(self.cross_unit)]
        if self.warnings:
            out += ["", "Warnings:", *[f"- {w}" for w in self.warnings]]
        return "\n".join(out) + "\n"


def _path_text(path, est, start):
    if est is None:
        return " → ".join(path)
    parts = [start]
    for rn in path:
        r = est.relationship(rn)
        parts.append(f"{r.to.entity} (via {', '.join(r.from_.attributes or [])})")
    return " → ".join(parts)


def _followed(est: Estate):
    """References that carry ownership: many-to-one or one-to-one references between
    different entities, not excluded."""
    excl = set(est.privacy.exclude_paths) if est.privacy else set()
    return [r for r in est.relationships
            if r.kind == "reference" and r.cardinality in ("many_to_one", "one_to_one")
            and r.from_.entity != r.to.entity and r.name not in excl]


def _nullable(est, r):
    # only a stated optional reference: a 1.0 spec's nullable is informational (true by default)
    return r.optional is True


def trace(est: Estate, unit: str | None = None) -> Trace:
    """Paths from every entity to the privacy unit (privacy.unit.entity, or `unit`)."""
    unit = unit or (est.privacy.unit.entity if est.privacy and est.privacy.unit else None)
    if unit is None:
        raise IRError("no privacy unit: set privacy.unit.entity or pass unit")
    if unit not in est.entities:
        raise IRError(f"privacy unit {unit!r} is not an entity")
    out = Trace(unit=unit)
    rels = _followed(est)
    by_from = {}
    for r in rels:
        by_from.setdefault(r.from_.entity, []).append(r)
    cyclic = _cycle(est.entities, rels)
    if cyclic:
        out.warnings.append("references form a cycle (" + " -> ".join(cyclic) + "): ownership is traced along "
                            "simple paths only; a path around the cycle may add owners")
    for n in est.entities:
        if n == unit:
            out.entities[n] = EntityTrace("unit", [[]], [False])
            continue
        paths, truncated = [], False

        def walk(cur, path, seen):
            nonlocal truncated
            for r in by_from.get(cur, []):
                if r.to.entity in seen:
                    continue
                if len(paths) >= MAX_PATHS:
                    truncated = True
                    return
                if r.to.entity == unit:
                    paths.append(path + [r])
                else:
                    walk(r.to.entity, path + [r], seen | {r.to.entity})
        walk(n, [], {n})
        if paths:
            t = EntityTrace("owned", [[r.name for r in p] for p in paths],
                            [any(_nullable(est, r) for r in p) for p in paths])
            if len(paths) > 1:
                t.note = f"{len(paths)} paths: a record can belong to up to {len(paths)} units"
            if truncated:
                t.note += f" (first {MAX_PATHS} paths only)"
            out.entities[n] = t
        else:
            out.entities[n] = EntityTrace("unrelated")
    # entities the unit (or its records) reference: context, not owned
    owned = set(out.owned())
    frontier = list(owned)
    seen = set(owned)
    while frontier:
        cur = frontier.pop()
        for r in by_from.get(cur, []):
            if r.to.entity not in seen:
                seen.add(r.to.entity)
                frontier.append(r.to.entity)
                if out.entities[r.to.entity].status == "unrelated":
                    out.entities[r.to.entity].status = "context"
    excl = set(est.privacy.exclude_paths) if est.privacy else set()
    for r in est.relationships:
        if r.to.entity in owned and (r.kind == "hierarchy" or r.name in excl):
            out.cross_unit.append(r.name)
            if r.to.entity == unit or r.kind != "hierarchy":
                out.warnings.append(f"{r.name} references records of the unit that it does not follow: removing a "
                                    "unit leaves such references dangling (null them or follow them)")
    return out


def _cycle(ents, rels):
    adj = {}
    for r in rels:
        adj.setdefault(r.from_.entity, []).append(r.to.entity)
    state, stack = {}, []

    def visit(n):
        state[n] = 1
        stack.append(n)
        for m in adj.get(n, []):
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state:
                c = visit(m)
                if c:
                    return c
        state[n] = 2
        stack.pop()
        return None
    for n in ents:
        if n not in state:
            c = visit(n)
            if c:
                return c
    return None


def contribution(est: Estate, data: dict, unit_keys, unit: str | None = None) -> dict:
    """{entity: boolean mask over data[entity]}: the records that belong to any of
    the units whose primary-key values are in unit_keys (pandas frames)."""
    import pandas as pd
    tr = trace(est, unit)
    U = tr.unit
    upk = est.entities[U].primary_key
    if upk is None:
        raise IRError(f"the unit {U} needs a primary key")
    keys = pd.MultiIndex.from_frame(data[U][upk.attributes]) if len(upk.attributes) > 1 else \
        pd.Index(data[U][upk.attributes[0]])
    want = keys.isin(list(unit_keys))
    masks = {U: pd.Series(want, index=data[U].index)}
    for n, t in tr.entities.items():
        if t.status != "owned" or n not in data:
            continue
        m = pd.Series(False, index=data[n].index)
        for path in t.paths:
            m |= _follow(est, data, n, path, U, set(pd.Index(data[U][upk.attributes[0]])[want])
                         if len(upk.attributes) == 1 else set(keys[want]))
        masks[n] = m
    return masks


def _follow(est, data, start, path, unit, unit_keys):
    """Mask of start's records whose path leads to a unit key in unit_keys."""
    import pandas as pd
    cur = data[start]
    vals = None
    for i, rn in enumerate(path):
        r = est.relationship(rn)
        fa = r.from_.attributes
        ta = r.to.attributes or est.entities[r.to.entity].primary_key.attributes
        if any("." in a for a in fa):
            raise IRError(f"{rn}: paths through fk.col composite keys are not followed on data")
        src = cur[fa] if i == 0 else vals
        src = src.copy()
        src.columns = list(ta)
        parent = data[r.to.entity]
        nxt = est.relationship(path[i + 1]) if i + 1 < len(path) else None
        carry = nxt.from_.attributes if nxt is not None else est.entities[unit].primary_key.attributes
        look = parent[list(dict.fromkeys(list(ta) + list(carry)))].drop_duplicates(subset=list(ta))
        joined = src.merge(look, how="left", on=list(ta))
        joined.index = src.index
        vals = joined[list(carry)]
    if len(vals.columns) == 1:
        return vals.iloc[:, 0].isin(unit_keys).fillna(False).astype(bool)
    return pd.Series([tuple(x) in unit_keys for x in vals.itertuples(index=False)], index=vals.index)

# SPDX-License-Identifier: Apache-2.0
"""Check data against an Open Data Contract Standard (ODCS) v3 contract.

`check_contract(contract, read)` runs the checks an ODCS v3 contract states in
its native fields on the tables `read(name, columns)` returns: every schema
object exists, every property is a column, values fit the logical type, the
`required` properties have no nulls, primary keys (composite by
`primaryKeyPosition`) and `unique` properties hold, numbers lie within
`logicalTypeOptions.minimum`/`maximum`, foreign keys (`relationships`) resolve
in their parent object, and the ODCS library quality metrics `rowCount`,
`nullValues` and `duplicateValues` meet their bounds.  Rules written by
relsynth (`engine: relsynth`) and other engines' custom checks are listed as
not run: relsynth checks its own rules when it writes, and Great Expectations
or Soda run theirs.  The contract can be the one `relsynth ir odcs` exports for
the source, so a synthetic lake is checked against the real data's contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

_BOUNDS = {"mustBe": lambda v, b: v == b, "mustNotBe": lambda v, b: v != b,
           "mustBeGreaterThan": lambda v, b: v > b, "mustBeGreaterOrEqualTo": lambda v, b: v >= b,
           "mustBeLessThan": lambda v, b: v < b, "mustBeLessOrEqualTo": lambda v, b: v <= b,
           "mustBeBetween": lambda v, b: b[0] <= v <= b[1], "mustNotBeBetween": lambda v, b: not b[0] <= v <= b[1]}


@dataclass
class ContractReport:
    contract: str
    checks: list[dict] = field(default_factory=list)

    def add(self, obj, prop, check, passed, detail=""):
        self.checks.append({"object": obj, "property": prop, "check": check,
                            "result": "skipped" if passed is None else ("passed" if passed else "failed"),
                            "detail": detail})

    def count(self, result):
        return sum(c["result"] == result for c in self.checks)

    @property
    def ok(self) -> bool:
        return self.count("failed") == 0

    def to_dict(self) -> dict:
        return {"contract": self.contract, "passed": self.count("passed"), "failed": self.count("failed"),
                "skipped": self.count("skipped"), "checks": self.checks}

    def to_markdown(self) -> str:
        lines = [f"# Contract check: {self.contract}", "",
                 f"{self.count('passed')} passed, {self.count('failed')} failed, {self.count('skipped')} not run", ""]
        bad = [c for c in self.checks if c["result"] != "passed"]
        if bad:
            lines += ["| Object | Property | Check | Result | Detail |", "|---|---|---|---|---|"]
            lines += [f"| {c['object']} | {c['property'] or ''} | {c['check']} | {c['result']} | {c['detail']} |"
                      for c in bad]
        return "\n".join(lines) + "\n"


def _col(p: dict) -> str:
    return p.get("physicalName") or p["name"]


def _split(ref) -> list[tuple[str, str]]:
    refs = ref if isinstance(ref, list) else [ref]
    out = []
    for r in refs:
        obj, _, col = str(r).rpartition(".")
        out.append((obj, col))
    return out


def _type_ok(s: pd.Series, logical: str) -> tuple[bool | None, str]:
    v = s.dropna()
    if v.empty or logical in ("string", "object", "array", "map"):
        return True, ""
    if logical in ("integer", "number"):
        if pd.api.types.is_bool_dtype(v):
            return False, "boolean values"
        x = pd.to_numeric(v, errors="coerce")
        bad = int(x.isna().sum())
        if not bad and logical == "integer":
            bad = int((~np.isclose(x.astype(float), np.round(x.astype(float)))).sum())
        return bad == 0, f"{bad} values are not {logical}" if bad else ""
    if logical == "boolean":
        ok = v.isin([True, False, 0, 1, "true", "false", "True", "False"]).all()
        return bool(ok), "" if ok else "values are not boolean"
    if logical in ("date", "timestamp", "time"):
        if pd.api.types.is_datetime64_any_dtype(v):
            return True, ""
        if logical == "time":
            x = pd.to_datetime(v.astype(str), format="mixed", errors="coerce")
        else:
            x = pd.to_datetime(v, errors="coerce", format="mixed")
        bad = int(x.isna().sum())
        return bad == 0, f"{bad} values are not {logical}" if bad else ""
    return None, f"logical type {logical} not checked"


def _bound_ok(value, q: dict) -> tuple[bool | None, str]:
    for k, fn in _BOUNDS.items():
        if k in q:
            return bool(fn(value, q[k])), f"{value} {k} {q[k]}"
    return None, "no bound given"


def check_contract(contract: dict, read: Callable[[str, list[str] | None], pd.DataFrame],
                   tables: list[str] | None = None) -> ContractReport:
    """Check the tables against the contract.  `read(name, columns)` returns a
    table (columns=None: all) or raises KeyError when it does not exist."""
    rep = ContractReport(str(contract.get("id") or contract.get("name") or "contract"))
    objects = {o["name"]: o for o in contract.get("schema") or []}
    frames: dict[str, pd.DataFrame | None] = {}

    def frame(name):
        if name not in frames:
            o = objects.get(name)
            try:
                frames[name] = read((o or {}).get("physicalName") or name, None)
            except KeyError:
                frames[name] = None
        return frames[name]

    for name, o in objects.items():
        if tables is not None and name not in tables:
            continue
        df = frame(name)
        rep.add(name, None, "exists", df is not None, "" if df is not None else "no such table")
        if df is None:
            continue
        pk: list[tuple[int, str]] = []
        for p in o.get("properties") or []:
            c = _col(p)
            if c not in df.columns:
                rep.add(name, c, "column", False, "missing")
                continue
            s = df[c]
            lt = p.get("logicalType")
            if lt:
                ok, why = _type_ok(s, lt)
                rep.add(name, c, f"logicalType {lt}", ok, why)
            if p.get("required"):
                n = int(s.isna().sum())
                rep.add(name, c, "required", n == 0, f"{n} nulls" if n else "")
            if p.get("unique"):
                d = int(s.dropna().duplicated().sum())
                rep.add(name, c, "unique", d == 0, f"{d} duplicates" if d else "")
            if p.get("primaryKey"):
                pk.append((int(p.get("primaryKeyPosition") or len(pk) + 1), c))
            opts = p.get("logicalTypeOptions") or {}
            if ("minimum" in opts or "maximum" in opts) and lt in ("integer", "number"):
                x = pd.to_numeric(s, errors="coerce")
                lo, hi = opts.get("minimum"), opts.get("maximum")
                n = int(((x < lo) if lo is not None else False).sum() + ((x > hi) if hi is not None else False).sum())
                rep.add(name, c, "range", n == 0, f"{n} values outside [{lo}, {hi}]" if n else "")
            for q in p.get("quality") or []:
                _quality(rep, name, c, q, df, [c])
        if pk:
            cols = [c for _, c in sorted(pk)]
            k = df[cols]
            nulls = int(k.isna().any(axis=1).sum())
            dup = int(k.duplicated().sum())
            rep.add(name, ",".join(cols), "primaryKey", nulls == 0 and dup == 0,
                    "; ".join(x for x in (f"{nulls} null keys" if nulls else "", f"{dup} duplicates" if dup else "") if x))
        for r in o.get("relationships") or []:
            if r.get("type", "foreignKey") != "foreignKey":
                continue
            fr, to = _split(r.get("from")), _split(r.get("to"))
            cols = [c for _, c in fr]
            pobj = to[0][0]
            label = f"{','.join(cols)} -> {pobj}.{','.join(c for _, c in to)}"
            parent = frame(pobj)
            if parent is None or any(c not in df.columns for c in cols) or any(c not in parent.columns for _, c in to):
                rep.add(name, label, "foreignKey", False, "column or parent table missing")
                continue
            child = df[cols].dropna()
            keys = set(map(tuple, parent[[c for _, c in to]].astype(str).to_numpy()))
            miss = int(sum(tuple(x) not in keys for x in child.astype(str).to_numpy()))
            rep.add(name, label, "foreignKey", miss == 0, f"{miss} rows without a parent" if miss else "")
        for q in o.get("quality") or []:
            _quality(rep, name, None, q, df, list(df.columns))
    return rep


def _quality(rep, name, col, q: dict, df: pd.DataFrame, cols: list[str]):
    label = q.get("name") or q.get("metric") or q.get("rule") or q.get("type", "quality")
    metric = q.get("metric") or q.get("rule")
    if q.get("type", "library") != "library" or metric is None:
        engine = q.get("engine") or q.get("type")
        rep.add(name, col, label, None, f"{engine} check, not run here")
        return
    unit = q.get("unit", "rows")
    if metric == "rowCount":
        value: Any = len(df)
    elif metric == "nullValues":
        value = int(df[cols].isna().any(axis=1).sum())
    elif metric == "duplicateValues":
        value = int(df[cols].dropna().duplicated().sum())
    else:
        rep.add(name, col, label, None, f"metric {metric} not run here")
        return
    if unit == "percent" and metric != "rowCount":
        value = 100.0 * value / max(len(df), 1)
    ok, why = _bound_ok(value, q)
    rep.add(name, col, label, ok, why)


def check_location(contract: dict, location, tables: list[str] | None = None, **connect_kw) -> ContractReport:
    """Check the tables of a data location (anything `relsynth.connectors.connect` takes)."""
    from ..connectors import connect
    conn = connect(location, **connect_kw)
    try:
        def read(name, columns):
            if not conn.has_table(name):
                raise KeyError(name)
            return conn.read_table(name, columns=columns)
        return check_contract(contract, read, tables)
    finally:
        close = getattr(conn, "close", None)
        if close:
            close()

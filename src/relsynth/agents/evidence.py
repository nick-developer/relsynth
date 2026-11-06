# SPDX-License-Identifier: Apache-2.0
"""Evidence tools: read-only checks of a draft's inferences on every row.

`relsynth introspect` infers keys, links and rules from a sample.  The review
agent checks each inference on the whole table and returns the result as an
aggregate (a count, a share, a distinct count).  The tools never return rows,
and a single value only in the ``values`` scope when it occurs at least k
times (scope.py, Prop. 1600).  They read through `relsynth.source.Source`,
the same reader introspection uses, with the same credentials handling.

Each result is an `Evidence` record with an id (``E1``, ``E2`` ...) that an
answer must cite; an answer that cites no evidence, or evidence that does not
exist, is refused by the guard (guard.py).
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .scope import Scope

log = logging.getLogger("relsynth.agents")


@dataclass
class Evidence:
    id: str
    tool: str
    args: dict
    result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _snake(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return [w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if w]


_PREFIX = {"dim", "fact", "fct", "tbl", "ref", "lkp", "lu", "d", "f", "t"}
_KEYWORDS = {"id", "key", "code", "no", "nr", "num", "number", "fk", "sk", "pk", "alternate"}


def name_affinity(table: str, column: str, parent: str, parent_key: str | None = None) -> float:
    """How strongly a column's name names a parent table, from 0 to 1: the share
    of the parent's name words (without prefixes such as dim_) found among the
    column's words, or among the parent key's words.  Deterministic and
    language-free; semantic matches (support rep -> employee) are left to the
    reviewer or a model."""
    col = [w for w in _snake(column) if w not in _KEYWORDS]
    par = [w for w in _snake(parent) if w not in _PREFIX]
    if not col or not par:
        return 0.0

    def stem(w):
        return w[:-3] + "y" if w.endswith("ies") else w[:-1] if w.endswith("s") and len(w) > 3 else w
    cs, ps = {stem(w) for w in col}, [stem(w) for w in par]
    share = sum(1 for w in ps if w in cs) / len(ps)
    if parent_key:
        pk = {stem(w) for w in _snake(parent_key) if w not in _KEYWORDS}
        if pk and pk <= cs:
            share = max(share, 1.0)
    return round(float(share), 3)


SELF_REF = re.compile(r"(parent|manager|reports_?to|boss|supervisor|superior|lead|head|mentor)", re.I)


class Tools:
    """Evidence tools over one source, within one data scope."""

    def __init__(self, source: dict | None, spec: dict, scope: Scope):
        self.scope = scope
        self.spec = spec
        self.src = None
        if source is not None and scope.reads_data:
            from ..source import Source
            self.src = Source(source)
        self.records: list[Evidence] = []
        self._full: dict[str, pd.DataFrame] = {}
        self._rules: dict | None = None

    # ------------------------------------------------------------- plumbing
    def _add(self, tool: str, args: dict, result: dict) -> Evidence:
        e = Evidence(f"E{len(self.records) + 1}", tool, args, result)
        self.records.append(e)
        return e

    def _blocked(self, tool, args) -> Evidence:
        return self._add(tool, args, {"available": False,
                                      "reason": "the data scope of this task is 'schema': no data is read"})

    def table(self, t: str) -> pd.DataFrame:
        if t not in self._full:
            src = self.src
            if src is None:
                raise RuntimeError("no source")
            cols = src._cols(t)
            df = src._frame(t).copy() if src.kind == "csv" else src.columns(t, cols)
            self._full[t] = src._finish(t, df) if src.kind != "csv" else df
        return self._full[t]

    def has(self, t: str, cols) -> bool:
        try:
            df = self.table(t)
        except Exception:  # noqa: BLE001
            return False
        return all(c in df.columns for c in cols)

    def pk_of(self, t: str):
        ts = (self.spec.get("tables") or {}).get(t) or {}
        pk = ts.get("primary_key")
        return pk if isinstance(pk, str) else (pk[0] if isinstance(pk, list) and len(pk) == 1 else None)

    # ------------------------------------------------------------- tools
    def unique(self, t: str, cols: list[str]) -> Evidence:
        args = {"table": t, "columns": list(cols)}
        if not self.scope.reads_data or self.src is None:
            return self._blocked("unique", args)
        if not self.has(t, cols):
            return self._add("unique", args, {"available": False, "reason": "column not found"})
        df = self.table(t)[list(cols)]
        n = len(df)
        nulls = int(df.isna().any(axis=1).sum())
        dup = int(df.dropna().duplicated().sum())
        return self._add("unique", args, {"available": True, "rows": n, "rows_with_null": nulls,
                                          "duplicate_rows": dup, "unique": bool(n and dup == 0),
                                          "unique_not_null": bool(n and dup == 0 and nulls == 0)})

    def inclusion(self, t: str, c: str, parent: str) -> Evidence:
        pk = self.pk_of(parent)
        args = {"table": t, "column": c, "parent": parent, "parent_key": pk}
        if not self.scope.reads_data or self.src is None:
            return self._blocked("inclusion", args)
        if pk is None or not self.has(t, [c]) or not self.has(parent, [pk]):
            return self._add("inclusion", args, {"available": False, "reason": "no single-column parent key"})
        ch = self.table(t)[c].dropna()
        pv = self.table(parent)[pk].dropna()
        ck, pkv = _keys(ch), set(_keys(pv))
        inside = np.fromiter((v in pkv for v in ck), bool, len(ck)) if len(ck) else np.zeros(0, bool)
        distinct = set(ck)
        res = {"available": True, "child_rows": int(len(ch)), "child_nulls": int(self.table(t)[c].isna().sum()),
               "share_rows_in_parent": float(inside.mean()) if len(ck) else 1.0,
               "orphan_rows": int((~inside).sum()), "child_distinct": len(distinct),
               "parent_rows": int(len(pv)),
               "parent_coverage": float(len(distinct & pkv) / len(pkv)) if pkv else 0.0,
               "name_affinity": name_affinity(t, c, parent, pk),
               "self_reference_name": bool(t == parent and SELF_REF.search(c))}
        return self._add("inclusion", args, res)

    def constant(self, t: str, c: str) -> Evidence:
        args = {"table": t, "column": c}
        if not self.scope.reads_data or self.src is None:
            return self._blocked("distinct", args)
        if not self.has(t, [c]):
            return self._add("distinct", args, {"available": False, "reason": "column not found"})
        s = self.table(t)[c]
        vc = s.value_counts(dropna=True)
        res = {"available": True, "rows": int(len(s)), "nulls": int(s.isna().sum()), "distinct": int(len(vc))}
        if len(vc) == 1:
            v = self.scope.value(_plain(vc.index[0]), int(vc.iloc[0]))
            if v is not None:
                res["value"] = v
        return self._add("distinct", args, res)

    def midnight(self, t: str, c: str) -> Evidence:
        args = {"table": t, "column": c}
        if not self.scope.reads_data or self.src is None:
            return self._blocked("time_of_day", args)
        if not self.has(t, [c]):
            return self._add("time_of_day", args, {"available": False, "reason": "column not found"})
        s = pd.to_datetime(self.table(t)[c], errors="coerce").dropna()
        at0 = (s == s.dt.normalize())
        return self._add("time_of_day", args, {"available": True, "rows": int(len(s)),
                                               "share_at_midnight": float(at0.mean()) if len(s) else 1.0})

    def key_order(self, t: str, c: str) -> Evidence:
        pk = self.pk_of(t)
        args = {"table": t, "order_column": c, "key": pk}
        if not self.scope.reads_data or self.src is None:
            return self._blocked("key_order", args)
        if pk is None or not self.has(t, [pk, c]):
            return self._add("key_order", args, {"available": False, "reason": "no single-column key"})
        df = self.table(t)[[pk, c]].dropna().sort_values(pk, kind="stable")
        v = df[c].to_numpy()
        try:
            ok = (v[1:] >= v[:-1]).mean() if len(v) > 1 else 1.0
        except TypeError:
            ok = float("nan")
        return self._add("key_order", args, {"available": True, "rows": int(len(df)),
                                             "share_non_decreasing": float(ok)})

    def rule(self, name: str, kind: str) -> Evidence:
        """Violation share of a rule of the draft on every row (lookups, derive,
        check and aggregate rules), measured by the scorecard's rule metric."""
        args = {"rule": name, "kind": kind}
        if not self.scope.reads_data or self.src is None:
            return self._blocked("rule", args)
        rates = self._rule_rates()
        hit = rates.get(name)
        if hit is None:
            return self._add("rule", args, {"available": False, "reason": "the rule is not in the draft"})
        v = hit.get("violation_rate_real")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return self._add("rule", args, {"available": False, "reason": "the rule cannot be evaluated on all rows"})
        return self._add("rule", args, {"available": True, "violation_share": float(v), "holds": bool(v == 0)})

    def _rule_rates(self) -> dict:
        if self._rules is None:
            self._rules = {}
            try:
                from ..api import Spec, _load_real
                from ..evaluate import rule_metrics
                raw = dict(self.spec)
                raw.setdefault("source", {"kind": "csv", "path": "."})
                s = Spec(raw).resolved
                frames = {t: self.table(t) for t in s["tables"]}
                real, _ = _load_real(s, frames)
                self._rules = rule_metrics(s, real, real)
            except Exception as e:  # noqa: BLE001  a draft whose rules cannot be checked: say so per rule
                log.warning("rule evidence unavailable: %s", type(e).__name__)
        return self._rules


def _keys(s: pd.Series) -> list:
    """Values comparable across tables whatever their stored type (1, 1.0 and '1' agree)."""
    out = []
    for v in s.tolist():
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            v = int(v)
        out.append(str(v))
    return out


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v) if not isinstance(v, (int, float, bool)) else v


__all__ = ["Tools", "Evidence", "name_affinity", "SELF_REF"]

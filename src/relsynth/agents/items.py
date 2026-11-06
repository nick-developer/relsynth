# SPDX-License-Identifier: Apache-2.0
"""The `review:` lines of an introspected draft, as structured items.

`relsynth introspect` writes every inference it made, and every question the
data could not answer (`ASK` lines), under `review:`.  The agents work on
structured items, so this module parses those lines.  The text is untrusted
(table and column names come from the source catalog), so the parser only
extracts names that the draft spec itself defines; a line it cannot parse
becomes an item of kind ``other`` that the agents explain but never answer.

Kinds and what can verify them:

=================  ==============================================  ===========
kind               the inference                                   verifiable
=================  ==============================================  ===========
pk                 a primary key inferred from the data            all rows
fk                 a foreign key inferred from the data            all rows
fk_ask             ASK: which parent (if any) a column references  all rows +
                                                                   judgement
unique             a two-column unique key (grain) inferred        all rows
lookup             a column copied from a parent's column          all rows
derive             a derive rule (amount = price * quantity)       all rows
check              a check rule (cost < price)                     all rows
aggregate          a parent column = an aggregate of its children  all rows
constant           a column with a single value                    all rows
date               a date-time column that is always midnight      all rows
key_order          keys numbered in the order of a date            all rows
declared           a key or link as declared in the catalog        catalog
privacy            pii, copy-as-is reference data, dropped tables  human only
other              roles, notes, everything else                   explained
=================  ==============================================  ===========
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

VERIFIABLE = ("pk", "fk", "fk_ask", "unique", "lookup", "derive", "aggregate", "check", "constant", "date",
              "key_order")
# decisions that belong to people by the user's rule (no agent makes privacy decisions)
HUMAN_ONLY = ("privacy",)

_N = r"[^\s.:\[\],()][^.:\[\],()]*?"           # a table or column name: no dot, colon, bracket or parenthesis


@dataclass
class ReviewItem:
    id: str                      # stable: sha256 of the line, 12 hex digits
    index: int                   # position in the review list
    kind: str
    text: str                    # the line as written (untrusted)
    table: str = ""
    column: str = ""
    columns: list[str] = field(default_factory=list)
    parent: str = ""
    candidates: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    ask: bool = False

    @property
    def verifiable(self) -> bool:
        return self.kind in VERIFIABLE

    def to_dict(self) -> dict:
        return asdict(self) | {"verifiable": self.verifiable}


def item_id(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()[:12]


def _split_top(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


_PATTERNS: list[tuple[str, re.Pattern]] = [(k, re.compile(p)) for k, p in [
    ("pk", rf"^(?P<t>{_N}): primary key (?P<c>{_N}) inferred \((?P<why>.*)\)$"),
    ("declared", rf"^(?P<t>{_N}): (composite primary key|unique key) .*\(declared\)"),
    ("fk", rf"^(?P<t>{_N})\.(?P<c>{_N}) -> (?P<p>{_N}): foreign key inferred \((?P<why>.*)\)$"),
    ("fk_ask", rf"^ASK (?P<t>{_N})\.(?P<c>{_N}): looks like a foreign key (?P<rest>.*)$"),
    ("ask_other", r"^ASK (?P<rest>.*)$"),
    ("unique", rf"^(?P<t>{_N}): unique key \[(?P<a>{_N}), (?P<b>{_N})\] inferred \((?P<why>.*)$"),
    ("unique", rf"^(?P<t>{_N}): unique key (?P<a>{_N}) inferred \((?P<why>.*)$"),
    ("unique", rf"^(?P<t>{_N}): unique key \[(?P<list>[^\]]*)\] inferred \((?P<why>.*)$"),
    ("check", rf"^(?P<t>{_N}): check (?P<expr>.*) \((holds|equal) on (?P<why>.*)$"),
    ("not_unique", rf"^(?P<t>{_N}): \[(?P<a>{_N}), (?P<b>{_N})\] has no repeat in the sample"),
    ("lookup", rf"^(?P<t>{_N})\.(?P<c>{_N}): lookup (?P<path>\S+) \((?P<why>.*)\)$"),
    ("derive", rf"^(?P<t>{_N}): derive rule (?P<lhs>{_N}) = (?P<expr>.*) \(holds(?P<why>.*)\)$"),
    ("aggregate", rf"^(?P<p>{_N})\.(?P<c>{_N}) = (?P<agg>\w+)\((?P<of>.*)\) of (?P<t>{_N}): aggregate inferred"),
    ("balance", rf"^(?P<t>{_N}): running balance (?P<rest>.*)$"),
    ("offset", rf"^(?P<t>{_N})\.(?P<c>{_N}): offset from (?P<rest>.*)$"),
    ("order_header", rf"^(?P<t>{_N}): order header (?P<rest>.*)$"),
    ("key_order", rf"^(?P<t>{_N}): keys numbered in the order of (?P<c>{_N}) \((?P<why>.*)\)$"),
    ("determines", rf"^(?P<t>{_N}): link (?P<rest>.*) determines (?P<rest2>.*)$"),
    ("role", rf"^(?P<t>{_N}): role (?P<rest>.*)$"),
    ("calendar", rf"^(?P<t>{_N}): calendar \((?P<rest>.*)$"),
    ("copy", rf"^(?P<t>{_N}): (?P<rest>.*copied as-is.*)$"),
    ("dropped", rf"^(?P<t>{_N}): no primary key and no link(?P<rest>.*)$"),
    ("link_dropped", rf"^(?P<t>{_N})\.(?P<rest>.*link dropped)$"),
    ("note", r"^NOTE: (?P<rest>.*)$"),
    ("column", rf"^(?P<t>{_N})\.(?P<c>{_N}): (?P<type>[a-z_]+) \((?P<why>.*)\)$"),
]]


def parse(review: list[str], spec: dict | None = None) -> list[ReviewItem]:
    """Structured items of a draft's review list.  With the draft `spec`, names
    are kept only when the spec defines them (anything else is untrusted text)."""
    tables = (spec or {}).get("tables") or {}
    items = []
    for i, line in enumerate(review or []):
        line = str(line)
        it = ReviewItem(id=item_id(line), index=i, kind="other", text=line)
        for kind, pat in _PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            g = {k: (v or "").strip() for k, v in m.groupdict().items()}
            it.kind, it.table = kind, g.get("t", "")
            if kind == "pk":
                it.column = g["c"]
            elif kind == "fk":
                it.column, it.parent = g["c"], g["p"]
            elif kind == "fk_ask":
                it.column, it.ask = g["c"], True
                it.candidates = _candidates(g["rest"])
            elif kind == "ask_other":
                it.ask = True
            elif kind in ("unique", "not_unique"):
                cols = [x for x in (g.get("a"), g.get("b")) if x] or g.get("list", "").split(",")
                it.columns = [x.strip().strip("'\"") for x in cols if x.strip().strip("'\"")]
            elif kind == "check":
                it.detail = {"expr": g["expr"]}
            elif kind == "lookup":
                it.column = g["c"]
                it.detail = {"path": g["path"]}
            elif kind == "derive":
                it.column = g["lhs"]
                it.detail = {"expr": f"{g['lhs']} = {g['expr']}"}
            elif kind == "aggregate":
                it.table, it.column = g["p"], g["c"]
                it.detail = {"agg": g["agg"], "of": g["of"], "from": g["t"]}
            elif kind in ("offset", "key_order"):
                it.column = g.get("c", "")
            elif kind == "column":
                it.column = g["c"]
                ty = g["type"]
                it.detail = {"type": ty}
                if ty == "constant":
                    it.kind = "constant"
                elif ty == "date" and "midnight" in g["why"]:
                    it.kind = "date"
                elif ty == "pii":
                    it.kind = "privacy"
            elif kind in ("copy", "dropped"):
                it.detail = {"what": kind}
                it.kind = "privacy"
            break
        if tables and it.table and it.table not in tables and it.kind not in ("dropped", "privacy", "link_dropped",
                                                                              "note", "other", "ask_other"):
            # a table the draft does not define: the line is explained, never answered
            it.detail = {"unparsed_kind": it.kind}
            it.kind = "other"
        if tables and it.kind == "fk_ask":
            it.candidates = [c for c in it.candidates if c in tables]
        items.append(it)
    return items


def _candidates(rest: str) -> list[str]:
    """Parent tables named by an ASK foreign-key line."""
    m = re.match(rf"^to (?P<p>{_N}) \(", rest)
    if m:
        return [m.group("p").strip()]
    m = re.match(r"^but the data fits (?P<list>.*) equally well", rest)
    if not m:
        return []
    out = []
    for part in _split_top(m.group("list")):
        name = part.split(" (", 1)[0].strip()
        if name and name not in out:
            out.append(name)
    return out


__all__ = ["ReviewItem", "parse", "item_id", "VERIFIABLE", "HUMAN_ONLY"]

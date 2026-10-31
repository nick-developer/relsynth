# SPDX-License-Identifier: Apache-2.0
"""The guard: what an agent's output may contain, whatever the model says
(Prop. 1601).

The guard does not trust the model.  It assumes the worst case, a model that
follows every instruction an attacker planted in a column name, a data value
or a spec comment, and checks every answer before anyone sees it:

* **Closed answers.**  A verdict is one of a fixed set per item kind; a parent
  table must be one the draft defines and the review line named; privacy
  items (pii, copy-as-is, dropped tables) get an explanation only, never a
  verdict or a change, because those are decisions for people.
* **Typed changes.**  An answer can carry only the change its verdict implies,
  built by the guard itself from the item (add or remove one relationship,
  remove one rule).  There is no free-form patch, so no answer can touch
  privacy, pii, copy modes, DP, epsilon, policies, approvals or releases.
* **Evidence first.**  An answer must cite evidence records that exist.  A
  confirmation that the evidence contradicts (a key with duplicates on all
  rows, a link with orphans, a rule with violations) is turned into a
  rejection and flagged.
* **Clean text.**  Explanations are plain text: links, images, HTML and
  addresses are removed (no channel to send data out through a rendered
  answer, OWASP LLM05), and every number in them must occur in the evidence
  or the review line (no invented figures); otherwise the guard's own
  template replaces the text.
* **No authority.**  Nothing an agent returns is applied.  Every answer is a
  proposal that a person accepts or rejects; the API refuses decisions,
  approvals, releases, epsilon, policy and access changes from an agent
  identity (service/app.py, governance/manifest.py).
"""
from __future__ import annotations

import math
import re

from .items import ReviewItem
from .untrusted import clean, suspicious

VERDICTS = ("confirm", "reject", "answer", "unsure", "explain")
KIND_VERDICTS = {
    "pk": ("confirm", "reject", "unsure"), "fk": ("confirm", "reject", "unsure"),
    "fk_ask": ("answer", "unsure"), "unique": ("confirm", "reject", "unsure"),
    "lookup": ("confirm", "reject", "unsure"), "derive": ("confirm", "reject", "unsure"),
    "aggregate": ("confirm", "reject", "unsure"), "check": ("confirm", "reject", "unsure"),
    "constant": ("confirm", "reject", "unsure"), "date": ("confirm", "reject", "unsure"),
    "key_order": ("confirm", "reject", "unsure"),
}
# actions no agent identity may take, through any interface
FORBIDDEN_ACTIONS = ("approve", "reject_release", "revoke", "release", "download", "set_epsilon", "spend_budget",
                     "change_policy", "policy_exception", "change_access", "manage_service_accounts", "create_run",
                     "write_spec", "decide_proposal", "declassify_pii", "change_privacy")
# spec sections no agent change may touch (the typed operations below cannot reach them; listed for the record)
PROTECTED_PATHS = ("privacy", "dp", "release", "contexts", "synthesis.privacy", "tables.*.columns.*.pii",
                   "tables.*.mode", "tables.*.columns.*.type=pii", "policy", "epsilon")
OPS = ("add_relationship", "remove_relationship", "remove_rule")

ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "parent", "confidence", "evidence", "explanation"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "parent": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
}

_URL = re.compile(r"(https?://|ftp://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b", re.I)
_MD = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<[^>]{0,200}>")
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


def plain_text(s, limit: int = 800) -> str:
    """Model text as safe plain text: no links, images, HTML or addresses."""
    s = clean(s, 4 * limit)
    s = _MD.sub(lambda m: m.group(1), s)
    s = _HTML.sub("", s)
    s = _URL.sub("[link removed]", s)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _numbers(s: str) -> set[str]:
    out = set()
    for m in _NUM.findall(s):
        x = m.replace(",", "").rstrip("%")
        try:
            v = float(x)
        except ValueError:
            continue
        out.add(_canon(v))
        if m.endswith("%"):
            out.add(_canon(v / 100))
    return out


def _canon(v: float) -> str:
    if not math.isfinite(v):
        return str(v)
    return f"{round(v, 4):.4f}".rstrip("0").rstrip(".")


def _grounding(obj) -> set[str]:
    """Numbers an explanation may use: those in the evidence and the item, with
    shares also as percentages (0.62 and 62%), and small counting words."""
    out = {_canon(float(i)) for i in range(0, 11)}

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            v = float(o)
            if math.isfinite(v):
                out.add(_canon(v))
                out.add(_canon(round(v, 2)))
                out.add(_canon(round(v, 1)))
                out.add(_canon(round(v)))
                if abs(v) <= 1:
                    for d in (0, 1, 2):
                        out.add(_canon(round(v * 100, d)))
        elif isinstance(o, str):
            out.update(_numbers(o))
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
    walk(obj)
    return out


def ungrounded_numbers(text: str, *sources) -> list[str]:
    allowed = _grounding(list(sources))
    return sorted(n for n in _numbers(text) if n not in allowed)


def check_answer(raw, item: ReviewItem, evidence: dict, spec: dict) -> tuple[dict, list[str]]:
    """The answer the guard lets through for `item`, and the problems it found
    (empty when the raw answer was used as it is).  `evidence` maps ids to
    Evidence records of this item."""
    probs: list[str] = []
    if not isinstance(raw, dict):
        return _unsure(item, "the answer was not a JSON object"), ["not a JSON object"]
    allowed = KIND_VERDICTS.get(item.kind, ("explain",))
    verdict = raw.get("verdict")
    if verdict not in VERDICTS:
        probs.append(f"unknown verdict {str(verdict)[:30]!r}")
        verdict = "unsure" if "unsure" in allowed else "explain"
    if verdict not in allowed:
        probs.append(f"verdict {verdict!r} is not allowed for a {item.kind} item")
        verdict = "unsure" if "unsure" in allowed else "explain"
    try:
        conf = float(raw.get("confidence", 0.0))
        conf = min(max(conf, 0.0), 1.0) if math.isfinite(conf) else 0.0
    except (TypeError, ValueError):
        conf = 0.0
        probs.append("confidence is not a number")
    cites = [str(e) for e in (raw.get("evidence") or []) if isinstance(e, (str, int))][:20]
    bad = [c for c in cites if c not in evidence]
    if bad:
        probs.append(f"cites evidence that does not exist: {bad[:3]}")
        cites = [c for c in cites if c in evidence]
    if verdict in ("confirm", "reject", "answer") and not cites:
        probs.append("a verdict without evidence")
        verdict, conf = ("unsure" if "unsure" in allowed else "explain"), 0.0
    parent = raw.get("parent")
    if verdict == "answer":
        tables = set(spec.get("tables") or {})
        ok = set(item.candidates) | ({item.table} if item.table in tables else set())
        if parent is not None and (not isinstance(parent, str) or parent not in ok or parent not in tables):
            probs.append("the parent is not one of the tables the question names")
            verdict, parent, conf = "unsure", None, 0.0
    else:
        parent = None
    # evidence first: a confirmation or an answer that the evidence contradicts is refused
    contra = contradiction(item, verdict, parent, [evidence[c] for c in cites] or list(evidence.values()))
    if contra:
        probs.append(contra)
        verdict, conf = ("reject" if verdict == "confirm" else "unsure"), (0.9 if verdict == "confirm" else 0.0)
        if verdict == "unsure":
            parent = None
        cites = cites or list(evidence)
    text = plain_text(raw.get("explanation", ""))
    if not text:
        text = template(item, verdict, parent, [evidence[c] for c in cites])
    else:
        un = ungrounded_numbers(text, item.text, [e.result for e in evidence.values()],
                                [e.args for e in evidence.values()])
        if un:
            probs.append(f"numbers not in the evidence: {un[:5]}")
            text = template(item, verdict, parent, [evidence[c] for c in cites])
        elif suspicious(text):
            probs.append("the explanation reads like an instruction")
            text = template(item, verdict, parent, [evidence[c] for c in cites])
    ans = {"item": item.id, "kind": item.kind, "verdict": verdict, "parent": parent, "confidence": round(conf, 3),
           "evidence": cites, "explanation": text}
    ans["change"] = change_for(item, verdict, parent, spec)
    return ans, probs


def contradiction(item: ReviewItem, verdict: str, parent, evs) -> str | None:
    """Why the evidence contradicts a verdict, or None."""
    res = [e.result for e in evs if e.result.get("available")]
    if verdict == "confirm":
        for e in evs:
            r = e.result
            if not r.get("available"):
                continue
            if e.tool == "unique" and not r.get("unique_not_null" if item.kind == "pk" else "unique"):
                return "the evidence shows duplicates or nulls on all rows; a confirmation is refused"
            if e.tool == "inclusion" and r.get("share_rows_in_parent", 1.0) < 1.0 and item.kind == "fk":
                return "the evidence shows values without a parent; a confirmation is refused"
            if e.tool == "rule" and not r.get("holds"):
                return "the evidence shows the rule fails on some rows; a confirmation is refused"
            if e.tool == "distinct" and r.get("distinct", 1) > 1:
                return "the evidence shows more than one value; a confirmation is refused"
            if e.tool == "time_of_day" and r.get("share_at_midnight", 1.0) < 1.0:
                return "the evidence shows times other than midnight; a confirmation is refused"
    if verdict == "answer" and parent is not None:
        for e in evs:
            r = e.result
            if e.tool == "inclusion" and e.args.get("parent") == parent and r.get("available") \
                    and r.get("share_rows_in_parent", 1.0) < 1.0:
                return f"values of the column are missing from {parent}'s key; that answer is refused"
    del res
    return None


def change_for(item: ReviewItem, verdict: str, parent, spec: dict) -> dict | None:
    """The typed change an accepted answer makes to the spec, built from the
    item (never from model text)."""
    if item.kind == "fk" and verdict == "reject":
        return {"op": "remove_relationship", "child": item.table, "child_key": item.column, "parent": item.parent}
    if item.kind == "fk_ask" and verdict == "answer" and parent:
        return {"op": "add_relationship", "child": item.table, "child_key": item.column, "parent": parent}
    if item.kind in ("derive", "check", "aggregate") and verdict == "reject":
        rule = find_rule(spec, item)
        if rule is not None:
            return {"op": "remove_rule", "rule": rule}
    return None


def find_rule(spec: dict, item: ReviewItem) -> dict | None:
    for r in spec.get("rules") or []:
        if r.get("table") != item.table:
            continue
        if item.kind == "aggregate" and r.get("type") == "aggregate" and r.get("column") == item.column:
            return dict(r)
        if item.kind in ("derive", "check") and r.get("type") in (item.kind, "soft") \
                and _norm(r.get("expr", "")) == _norm(item.detail.get("expr", "")):
            return dict(r)
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def _unsure(item: ReviewItem, why: str) -> dict:
    v = "unsure" if "unsure" in KIND_VERDICTS.get(item.kind, ()) else "explain"
    return {"item": item.id, "kind": item.kind, "verdict": v, "parent": None, "confidence": 0.0, "evidence": [],
            "explanation": why, "change": None}


def template(item: ReviewItem, verdict: str, parent, evs) -> str:
    """The guard's own wording of an answer, from the evidence only."""
    parts = []
    for e in evs:
        r = e.result
        if not r.get("available"):
            parts.append(f"{e.tool}: not available ({r.get('reason', 'no data')})")
        elif e.tool == "unique":
            parts.append(f"{r['duplicate_rows']} duplicate and {r['rows_with_null']} null rows in {r['rows']}")
        elif e.tool == "inclusion":
            parts.append(f"{e.args['parent']}: {r['share_rows_in_parent']:.1%} of {r['child_rows']} rows found in "
                         f"its key, coverage {r['parent_coverage']:.0%}, name affinity {r['name_affinity']:.2f}")
        elif e.tool == "rule":
            v = r["violation_share"]
            parts.append("holds on all rows" if v == 0 else
                         f"violated on {'less than 0.01%' if v < 1e-4 else f'{v:.2%}'} of all rows")
        elif e.tool == "distinct":
            parts.append(f"{r['distinct']} distinct values in {r['rows']} rows")
        elif e.tool == "time_of_day":
            parts.append(f"{r['share_at_midnight']:.1%} of {r['rows']} values at midnight")
        elif e.tool == "key_order":
            parts.append(f"{r['share_non_decreasing']:.1%} of consecutive keys in order")
    head = {"confirm": "Confirmed on all rows", "reject": "Not supported on all rows",
            "answer": f"Proposed parent: {parent}" if parent else "Proposed: not a foreign key",
            "unsure": "Needs a person's judgement", "explain": "For a person to decide"}[verdict]
    return head + (": " + "; ".join(parts) if parts else ".")


__all__ = ["check_answer", "contradiction", "change_for", "plain_text", "ungrounded_numbers", "template",
           "ANSWER_SCHEMA", "VERDICTS", "KIND_VERDICTS", "FORBIDDEN_ACTIONS", "PROTECTED_PATHS", "OPS"]

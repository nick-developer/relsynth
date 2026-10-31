# SPDX-License-Identifier: Apache-2.0
"""Evaluation of the review agent against ground truth.

Ground truth comes from outside the data the agent checks:

* **catalog**: the keys and links the database declares.  The draft is made
  in benchmark mode (``introspect(..., declared=False)``: the catalog is hidden
  and keys and links are identified from the data), the agent answers its
  review items, and each answer about a key or a link is compared with the
  declared catalog (a draft made with ``declared=True``).
* **curated spec**: for sources without a catalog (CSV files), a spec a person
  reviewed by hand.

Items about rules (lookups, derive, check and aggregate rules, constants,
dates, key order) have no independent ground truth: the agent measures them on
every row, so its verdict is that measurement.  They are reported apart, as
the number of sample-based inferences that do or do not hold on all rows.

The human approval step is simulated by two reviewers: an *oracle* that knows
the ground truth (it accepts right answers and rejects wrong ones) and a
*hurried* reviewer that accepts every answer the agent did not flag for
attention.  The report counts the wrong answers each would let into the spec.
"""
from __future__ import annotations

import math
from collections import Counter

from .review import ReviewResult, apply_decisions

JUDGED = ("pk", "fk", "fk_ask", "unique")
VERIFIED = ("lookup", "derive", "check", "aggregate", "constant", "date", "key_order")


def truth_from_spec(spec: dict) -> dict:
    """Keys, links and unique keys of a spec (a catalog draft or a curated spec)."""
    pks, uniques, rels = {}, {}, set()
    for t, ts in (spec.get("tables") or {}).items():
        pk = ts.get("primary_key")
        if pk:
            pks[t] = [pk] if isinstance(pk, str) else list(pk)
        uniques[t] = [frozenset([u] if isinstance(u, str) else u) for u in (ts.get("unique") or [])]
        if pk:
            uniques[t].append(frozenset(pks[t]))
    for r in spec.get("relationships") or []:
        if isinstance(r.get("child_key"), str):
            rels.add((r["child"], r["child_key"], r["parent"]))
    return {"pks": pks, "uniques": uniques, "rels": rels, "tables": set(spec.get("tables") or {})}


def label(item, truth: dict):
    """The right answer for an item: True/False for a claim, a parent name or
    None for a question, or the marker NO_TRUTH when the truth does not say."""
    if item.table not in truth["tables"]:
        return NO_TRUTH
    if item.kind == "pk":
        if item.table not in truth["pks"]:
            return NO_TRUTH                       # no declared key: any unique column is a candidate key
        return truth["pks"][item.table] == [item.column]
    if item.kind == "fk":
        return (item.table, item.column, item.parent) in truth["rels"]
    if item.kind == "fk_ask":
        hits = [p for (c, k, p) in truth["rels"] if c == item.table and k == item.column]
        return hits[0] if hits else None
    if item.kind == "unique":
        if item.table not in truth["pks"] and not truth["uniques"].get(item.table):
            return NO_TRUTH
        return frozenset(item.columns) in truth["uniques"].get(item.table, [])
    return NO_TRUTH


class _NoTruth:
    def __repr__(self):
        return "NO_TRUTH"


NO_TRUTH = _NoTruth()


def outcome(ans: dict, right) -> str:
    """correct | wrong | abstain for one answer."""
    v = ans["verdict"]
    if v in ("unsure", "explain"):
        return "abstain"
    if v == "answer":
        return "correct" if ans["parent"] == right else "wrong"
    return "correct" if (v == "confirm") == bool(right) else "wrong"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(res: ReviewResult, truth: dict, *, minutes_manual: float = 2.0, minutes_read: float = 0.5) -> dict:
    """Accuracy, coverage, the simulated approval step and an estimate of the
    time saved.  The time estimate is an assumption, stated in the result:
    `minutes_manual` per verifiable item for a person to write and run the
    check by hand, `minutes_read` to read an answer that comes with evidence."""
    rows: list[dict] = []
    by_kind: dict[str, Counter] = {}
    for it, a in zip(res.items, res.answers):
        if it.kind in JUDGED:
            right = label(it, truth)
            if right is NO_TRUTH:
                continue
            o = outcome(a, right)
            rows.append({"item": it.id, "kind": it.kind, "line": it.text[:160], "verdict": a["verdict"],
                         "parent": a["parent"], "truth": right if not isinstance(right, frozenset) else sorted(right),
                         "outcome": o, "attention": a["attention"], "confidence": a["confidence"]})
            by_kind.setdefault(it.kind, Counter())[o] += 1
    c = Counter(r["outcome"] for r in rows)
    answered = c["correct"] + c["wrong"]
    lo, hi = wilson(c["correct"], answered)
    # simulated approval step
    oracle = {r["item"]: ("accept" if r["outcome"] == "correct" else "reject") for r in rows}
    hurried = {a["item"]: "accept" for a in res.answers if a["attention"] == "normal"}
    _, log_oracle = apply_decisions(res.spec, res.answers, oracle, decided_by="user:oracle")
    _, log_hurried = apply_decisions(res.spec, res.answers, hurried, decided_by="user:hurried")
    wrong = [r for r in rows if r["outcome"] == "wrong"]
    slipped = [r for r in wrong if r["item"] in hurried]
    wrong_in_oracle_spec = _wrong_changes(log_oracle, truth)
    ver = Counter((it.kind, a["verdict"]) for it, a in zip(res.items, res.answers) if it.kind in VERIFIED)
    ver_items = [(it, a) for it, a in zip(res.items, res.answers) if it.kind in VERIFIED or it.kind in JUDGED]
    n_ver = len(ver_items)
    n_att = sum(1 for _, a in ver_items if a["attention"] == "high")
    manual = n_ver * minutes_manual
    assisted = (n_ver - n_att) * minutes_read + n_att * (minutes_read + minutes_manual)
    asked = {(it.table, it.column) for it in res.items if it.kind == "fk_ask"}
    drafted = {(r["child"], r["child_key"], r["parent"]) for r in res.spec.get("relationships") or []
               if isinstance(r.get("child_key"), str)}
    missed = sorted(r for r in truth["rels"] if r not in drafted and (r[0], r[1]) not in asked
                    and r[0] in truth["tables"] and r[2] in truth["tables"])
    return {
        "judged_items": len(rows), "correct": c["correct"], "wrong": c["wrong"], "abstained": c["abstain"],
        "coverage": answered / len(rows) if rows else 0.0,
        "accuracy": c["correct"] / answered if answered else float("nan"), "accuracy_ci95": [lo, hi],
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "wrong_answers": wrong,
        "approval": {"oracle_wrong_changes_applied": wrong_in_oracle_spec,
                     "wrong_flagged_for_attention": sum(1 for r in wrong if r["attention"] == "high"),
                     "wrong_unflagged": len(slipped),
                     "hurried_reviewer_would_accept_wrong": [r["item"] for r in slipped],
                     "hurried_wrong_changes_applied": _wrong_changes(log_hurried, truth)},
        "links_missed_by_draft": [list(m) for m in missed],
        "verified": {f"{k}:{v}": n for (k, v), n in sorted(ver.items())},
        "time": {"agent_seconds": round(res.seconds, 1), "items_verifiable": n_ver,
                 "attention_items": n_att,
                 "assumed_minutes_manual_per_item": minutes_manual, "assumed_minutes_read_per_item": minutes_read,
                 "estimated_minutes_manual": round(manual, 1), "estimated_minutes_assisted": round(assisted, 1),
                 "note": "an estimate from the stated minutes per item, not a measurement of people"},
    }


def _wrong_changes(log: list[dict], truth: dict) -> int:
    """Applied changes that contradict the ground truth: a link added that the
    catalog does not declare, or a declared link removed."""
    bad = 0
    for e in log:
        ch = e.get("change") or {}
        if not ch.get("applied"):
            continue
        if ch["op"] == "add_relationship" and (ch["child"], ch["child_key"], ch["parent"]) not in truth["rels"]:
            bad += 1
        if ch["op"] == "remove_relationship" and e.get("removed") in truth["rels"]:
            bad += 1
    return bad


__all__ = ["evaluate", "truth_from_spec", "label", "outcome", "wilson", "NO_TRUTH", "JUDGED", "VERIFIED"]

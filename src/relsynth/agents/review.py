# SPDX-License-Identifier: Apache-2.0
"""The review agent: answers the `review:` questions of an introspected draft
with evidence, as proposals a person accepts or rejects.

    from relsynth import agents
    res = agents.review("draft.yaml", source="sqlite:shop.db")       # rules backend, aggregates scope
    for a in res.answers: print(a["verdict"], a["explanation"])
    spec2, log = agents.apply_decisions(res.spec, res.answers, {a["item"]: "accept" for a in res.answers},
                                        decided_by="user:alice")

For each item the agent (1) checks the inference on every row with the
evidence tools, (2) judges it with deterministic rules, (3) asks the model,
if one is configured, only where the rules are unsure, and (4) passes every
answer through the guard.  Nothing is applied: `apply_decisions` makes the
changes of the answers a person accepted, and only those.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field

from .backends import RulesBackend
from .evidence import Evidence, Tools
from .guard import ANSWER_SCHEMA, KIND_VERDICTS, change_for, check_answer, template
from .items import ReviewItem, parse
from .scope import Scope, lower
from .untrusted import SYSTEM_RULE, mark, suspicious

CONFIDENT = 0.8


@dataclass
class ReviewResult:
    spec: dict
    items: list[ReviewItem]
    answers: list[dict]
    evidence: dict[str, list[Evidence]]
    scope: str
    backend: dict
    seconds: float
    flags: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        from collections import Counter
        v = Counter(a["verdict"] for a in self.answers)
        return {"items": len(self.items), "verdicts": dict(v),
                "attention": sum(1 for a in self.answers if a["attention"] == "high"),
                "changes": sum(1 for a in self.answers if a.get("change")),
                "flags": len(self.flags), "seconds": round(self.seconds, 2), "scope": self.scope,
                "backend": self.backend}

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "answers": self.answers, "flags": self.flags,
                "evidence": {k: [e.to_dict() for e in v] for k, v in self.evidence.items()}}

    def to_markdown(self) -> str:
        s = self.summary()
        out = [f"# Review answers ({s['items']} items, {s['attention']} need attention)", "",
               f"Backend {s['backend']['name']} ({s['backend']['model']}), data scope {s['scope']}. "
               "Nothing has been applied: accept or reject each answer.", "",
               "| # | item | verdict | confidence | change | explanation |", "|---|---|---|---|---|---|"]
        for a, it in zip(self.answers, self.items):
            ch = a.get("change") or {}
            cs = f"{ch['op']} {ch.get('child', '')}.{ch.get('child_key', '')}->{ch.get('parent', '')}" if ch else ""
            if ch.get("op") == "remove_rule":
                cs = "remove_rule"
            flag = " ⚠" if a["attention"] == "high" else ""
            out.append(f"| {it.index} | {_md(it.text[:90])} | {a['verdict']}{flag} | {a['confidence']:.2f} | "
                       f"{_md(cs)} | {_md(a['explanation'][:160])} |")
        if self.flags:
            out += ["", "## Flags", ""] + [f"- item {f['item']}: {_md(f['flag'])}" for f in self.flags]
        return "\n".join(out) + "\n"


def _md(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def gather(tools: Tools, item: ReviewItem) -> list[Evidence]:
    """The evidence for one item (none for items that are explained only)."""
    k, t, c = item.kind, item.table, item.column
    if k == "pk":
        return [tools.unique(t, [c])]
    if k == "unique":
        return [tools.unique(t, item.columns)]
    if k == "fk":
        return [tools.inclusion(t, c, item.parent)]
    if k == "fk_ask":
        cands = list(item.candidates)
        if not cands and t:
            cands = [t]
        return [tools.inclusion(t, c, p) for p in cands]
    if k == "lookup":
        cs = ((tools.spec.get("tables") or {}).get(t, {}).get("columns") or {}).get(c)
        path = cs.get("lookup") if isinstance(cs, dict) else item.detail.get("path")
        return [tools.rule(f"{t}: {c} = {path}", "lookup")]
    if k in ("derive", "check"):
        return [tools.rule(f"{t}: {item.detail.get('expr', '')}", k)]
    if k == "aggregate":
        r = next((r for r in tools.spec.get("rules") or [] if r.get("type") == "aggregate" and r.get("table") == t
                  and r.get("column") == c), None)
        if r is None:
            return [tools.rule(f"{t}: {c}", "aggregate")]
        return [tools.rule(f"{t}: {c} = {r.get('agg', 'count')}({r['from']})", "aggregate")]
    if k == "constant":
        return [tools.constant(t, c)]
    if k == "date":
        return [tools.midnight(t, c)]
    if k == "key_order":
        return [tools.key_order(t, c)]
    return []


def judge(item: ReviewItem, evs: list[Evidence]) -> dict:
    """The deterministic judgement of an item from its evidence."""
    ids = [e.id for e in evs]
    av = [e for e in evs if e.result.get("available")]
    k = item.kind
    if k not in KIND_VERDICTS:
        return {"verdict": "explain", "parent": None, "confidence": 0.0, "evidence": [], "explanation": ""}
    if not av:
        return {"verdict": "unsure", "parent": None, "confidence": 0.0, "evidence": ids, "explanation": ""}
    r = av[0].result
    if k in ("pk", "unique"):
        ok = r["unique_not_null"] if k == "pk" else r["unique"]
        conf = 0.95 if ok else 0.95
        if ok and k == "pk" and "best of" in item.text:
            conf = 0.85                    # several unique columns: the name decided
        return _v("confirm" if ok else "reject", conf, ids)
    if k == "fk":
        s = r["share_rows_in_parent"]
        if s >= 1.0:
            return _v("confirm", 0.95 if r["name_affinity"] >= 0.5 else 0.7, ids)
        return _v("reject" if s < 0.99 else "unsure", 0.9 if s < 0.99 else 0.3, ids)
    if k == "fk_ask":
        fit = [e for e in av if e.result["share_rows_in_parent"] >= 1.0]
        if not fit:
            return {"verdict": "answer", "parent": None, "confidence": 0.7, "evidence": ids, "explanation": ""}
        selfs = [e for e in fit if e.result.get("self_reference_name")]
        if selfs:
            return {"verdict": "answer", "parent": selfs[0].args["parent"], "confidence": 0.75,
                    "evidence": [selfs[0].id], "explanation": ""}
        fit.sort(key=lambda e: -e.result["name_affinity"])
        best = fit[0].result["name_affinity"]
        nxt = fit[1].result["name_affinity"] if len(fit) > 1 else 0.0
        if best >= 0.5 and best - nxt >= 0.25:
            return {"verdict": "answer", "parent": fit[0].args["parent"], "confidence": 0.6,
                    "evidence": [fit[0].id], "explanation": ""}
        return _v("unsure", 0.0, ids)
    if k in ("lookup", "derive", "check", "aggregate"):
        return _v("confirm" if r["holds"] else "reject", 0.95, ids)
    if k == "constant":
        return _v("confirm" if r["distinct"] <= 1 else "reject", 0.95, ids)
    if k == "date":
        return _v("confirm" if r["share_at_midnight"] >= 1.0 else "reject", 0.95, ids)
    if k == "key_order":
        s = r["share_non_decreasing"]
        return _v("confirm" if s >= 0.999 else "reject", 0.8, ids)
    return _v("unsure", 0.0, ids)


def _v(verdict, conf, ids):
    return {"verdict": verdict, "parent": None, "confidence": conf, "evidence": ids, "explanation": ""}


def prompt(item: ReviewItem, evs: list[Evidence], allowed) -> str:
    ev = [{"id": e.id, "tool": e.tool, "args": {k: (mark(v, "name") if isinstance(v, str) else v)
                                                for k, v in e.args.items()},
           "result": {k: (mark(v, "value") if isinstance(v, str) else v) for k, v in e.result.items()}}
          for e in evs]
    task = {
        "task": "Judge one inference that relsynth introspect made from a sample of a database, using the evidence "
                "measured on all rows. For a question about a foreign key, choose the parent table the column "
                "references (one of the candidates, or null if it references none).",
        "kind": item.kind, "review_line": mark(item.text, "review_line"),
        "table": mark(item.table, "name"), "column": mark(item.column, "name"),
        "candidates": [mark(c, "name") for c in item.candidates],
        "allowed_verdicts": list(allowed), "evidence": ev,
        "answer_format": "verdict, parent (a candidate's plain name or null), confidence 0-1, evidence (the ids "
                         "you relied on), explanation (one or two plain sentences, numbers only from the evidence)",
    }
    return json.dumps(task, ensure_ascii=False, indent=1)


def review(spec, *, source=None, backend=None, scope: Scope | str = "aggregates", only=None) -> ReviewResult:
    """Answer every review item of a draft.  `spec` is a Spec, a path or a
    mapping; `source` overrides the draft's source section; `only` restricts
    to some item kinds."""
    from ..api import Spec
    t0 = time.perf_counter()
    raw = copy.deepcopy(Spec.coerce(spec).raw if not isinstance(spec, dict) else spec)
    backend = backend or RulesBackend()
    sc = scope if isinstance(scope, Scope) else Scope(scope)
    if getattr(backend, "uses_model", False) and not getattr(backend, "local", True):
        sc = Scope(lower(sc.level, "aggregates"), sc.k)     # a remote model never gets values
    src = source if source is not None else raw.get("source")
    if isinstance(src, str):
        from ..api import _section
        src = _section(src)
    tools = Tools(src, raw, sc)
    items = parse(list(raw.get("review") or []), raw)
    if only:
        items = [i for i in items if i.kind in only]
    answers, evidence, flags = [], {}, []
    for it in items:
        hits = suspicious(it.text)
        if hits:
            flags.append({"item": it.id, "flag": "the review line reads like an instruction to a model "
                                                 f"({len(hits)} pattern(s)); check it by hand"})
        try:
            evs = gather(tools, it)
        except Exception as e:  # noqa: BLE001  one failing check never stops the review
            evs = [tools._add("error", {"item": it.id}, {"available": False, "reason": type(e).__name__})]
        evidence[it.id] = evs
        emap = {e.id: e for e in evs}
        base = judge(it, evs)
        ans, probs = check_answer(base, it, emap, raw)
        source_of = "rules"
        allowed = KIND_VERDICTS.get(it.kind)
        if getattr(backend, "uses_model", False) and allowed and (base["verdict"] == "unsure" or it.kind == "fk_ask"):
            try:
                mraw = backend.complete(SYSTEM_RULE, prompt(it, evs, allowed), ANSWER_SCHEMA)
                mans, mprobs = check_answer(mraw, it, emap, raw)
                if mprobs:
                    flags += [{"item": it.id, "flag": f"model answer refused by the guard: {p}"} for p in mprobs]
                    if mans["verdict"] != "unsure" or base["verdict"] == "unsure":
                        ans = mans if not _worse(mans, ans) else ans
                else:
                    ans, source_of = mans, "model"
            except Exception as e:  # noqa: BLE001  a model failure leaves the rules answer
                flags.append({"item": it.id, "flag": f"model call failed ({type(e).__name__}); rules answer kept"})
        if not ans["explanation"]:
            ans["explanation"] = template(it, ans["verdict"], ans["parent"], [emap[c] for c in ans["evidence"]
                                                                              if c in emap])
        ans["source"] = source_of
        ans["attention"] = "high" if (ans["verdict"] in ("reject", "unsure", "answer")
                                      or ans["confidence"] < CONFIDENT or hits or probs) else "normal"
        if it.kind == "privacy":
            ans["explanation"] = ("A privacy decision (personal data, copying reference data as-is, or leaving a table "
                                  "out): a person decides; the agent does not answer it. " + it.text[:300])
            ans["attention"] = "high"
        answers.append(ans)
    meta = {"name": getattr(backend, "name", "?"), "model": getattr(backend, "model", "?")}
    return ReviewResult(raw, items, answers, evidence, sc.level, meta, time.perf_counter() - t0, flags)


def _worse(a: dict, b: dict) -> bool:
    return a["verdict"] == "unsure" and b["verdict"] != "unsure"


def apply_decisions(spec: dict, answers: list[dict], decisions: dict[str, str], *, decided_by: str):
    """The spec after the changes of accepted answers, and a log of what was
    done.  `decisions` maps item ids to ``accept`` or ``reject``; answers
    without a decision are left open.  Decided items leave the review list.
    `decided_by` must be a person: an agent identity is refused."""
    if not decided_by or decided_by.split(":", 1)[0] in ("agent", "service"):
        raise PermissionError("review answers are accepted or rejected by a person, not by an agent or a service")
    raw = copy.deepcopy(spec)
    review_lines = list(raw.get("review") or [])
    items = {i.id: i for i in parse(review_lines, raw)}
    log, decided = [], set()
    for a in answers:
        d = decisions.get(a["item"])
        if d not in ("accept", "reject"):
            continue
        decided.add(a["item"])
        entry = {"item": a["item"], "decision": d, "verdict": a["verdict"], "by": decided_by, "change": None}
        it = items.get(a["item"])
        if d == "accept" and it is None:
            entry["error"] = "the item is no longer in the review list"
        elif d == "accept" and it is not None:
            # the change is rebuilt from the item and the verdict, never taken from the stored answer
            parent = a.get("parent")
            if a.get("verdict") == "answer" and parent is not None and parent not in set(it.candidates) | {it.table}:
                entry["error"] = "the answer names a table the question did not offer; nothing changed"
            else:
                ch = change_for(it, a.get("verdict", "unsure"), parent, raw)
                if ch:
                    entry["change"] = _apply(raw, ch)
                    if ch["op"] == "remove_relationship":
                        entry["removed"] = (ch["child"], ch["child_key"], ch["parent"])
        log.append(entry)
    raw["review"] = [line for line in review_lines if items.get(_id(line)) is None or _id(line) not in decided]
    return raw, log


def _id(line: str) -> str:
    from .items import item_id
    return item_id(str(line))


def _apply(raw: dict, ch: dict) -> dict:
    op = ch.get("op")
    rels = raw.setdefault("relationships", [])
    if op == "add_relationship":
        r = {"child": ch["child"], "child_key": ch["child_key"], "parent": ch["parent"]}
        if not any(x.get("child") == r["child"] and x.get("child_key") == r["child_key"] for x in rels):
            if r["child"] == r["parent"]:
                r["kind"] = "self_reference"
            rels.append(r)
        return {"op": op, "applied": True, **r}
    if op == "remove_relationship":
        n = len(rels)
        raw["relationships"] = [x for x in rels if not (x.get("child") == ch["child"] and
                                                        x.get("child_key") == ch["child_key"] and
                                                        x.get("parent") == ch["parent"])]
        return {"op": op, "applied": len(raw["relationships"]) < n}
    if op == "remove_rule":
        rules = raw.get("rules") or []
        keep = [r for r in rules if r != ch["rule"]]
        raw["rules"] = keep
        return {"op": op, "applied": len(keep) < len(rules)}
    raise ValueError(f"unknown change {op!r}")


__all__ = ["review", "apply_decisions", "ReviewResult", "judge", "gather", "prompt"]

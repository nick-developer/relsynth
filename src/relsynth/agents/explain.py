# SPDX-License-Identifier: Apache-2.0
"""Plain-language explanations of the quality, privacy and scorecard reports.

The explainer reads a report (never data), collects the facts a reader needs
(verdicts, failed checks, their values and limits) and words them.  With the
rules backend the wording is a fixed template.  With a model, the model words
the same facts, and the guard keeps its text only when every number in it
occurs in the facts and nothing in it reads like an instruction; otherwise
the template is used (Prop. 1602).

An explanation never states a release decision: it says what the report
found and who decides.  For a reader without the privacy role the service
passes the report with its privacy items removed, so the explanation cannot
mention them either.
"""
from __future__ import annotations

import json

from .guard import plain_text, ungrounded_numbers
from .untrusted import SYSTEM_RULE, mark, suspicious

EXPLAIN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary"],
                  "properties": {"summary": {"type": "string"}}}
MAX_ITEMS = 8


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def quality_facts(rep: dict) -> dict:
    vec = rep.get("vector") or {}
    areas = {a: {k: vec[a].get(k) for k in ("checks", "passed", "failed", "allowed_failures", "pass",
                                            "expected_false_failures")}
             for a in ("fidelity", "utility", "integrity") if isinstance(vec.get(a), dict)}
    failed = [c for c in rep.get("checks") or [] if not c.get("pass")]
    failed.sort(key=lambda c: (c.get("p_value") is None, c.get("p_value") or 1.0))
    return {"report": "quality-2.0", "verdict_pass": bool((vec.get("verdict") or {}).get("pass")), "areas": areas,
            "failed": [{"id": c.get("id"), "area": c.get("area"), "metric": c.get("metric"),
                        "value": c.get("value"), "limit": c.get("limit"), "p_value": c.get("p_value")}
                       for c in failed[:MAX_ITEMS]],
            "failed_total": len(failed)}


def quality_text(f: dict) -> str:
    out = [("The synthetic data pass the quality report: " if f["verdict_pass"] else
            "The synthetic data do not pass the quality report: ")
           + "; ".join(f"{a} {x['passed']} of {x['checks']} checks passed"
                       + ("" if x["pass"] else f" (fails: {x['failed']} failed, {x['allowed_failures']} allowed)")
                       for a, x in f["areas"].items()) + "."]
    if f["areas"]:
        out.append("Each area allows the few failures that chance alone would cause (the expected false failures), "
                   "so a single failed check is not by itself a defect.")
    if f["failed"]:
        out.append(f"The {min(len(f['failed']), MAX_ITEMS)} most notable of {f['failed_total']} failed checks:")
        for c in f["failed"]:
            out.append(f"- {c['id']} ({c['area']}): value {_fmt(c['value'])}, limit {_fmt(c['limit'])}"
                       + (f", p-value {_fmt(c['p_value'])}" if c.get("p_value") is not None else ""))
    out.append("Whether these data may be used or released is decided by the people named in the release policy, "
               "not by this explanation.")
    return "\n".join(out)


def privacy_facts(rep: dict) -> dict:
    ctx = {}
    for name, c in (rep.get("contexts") or {}).items():
        bad = [k for k in c.get("checks") or [] if k.get("status") not in ("pass", None) or k.get("pass") is False]
        ctx[name] = {"verdict": c.get("verdict"), "external": c.get("external"), "dp_required": c.get("dp_required"),
                     "sign_off_required": bool((c.get("sign_off") or {}).get("required")),
                     "blocked_reason": c.get("blocked_reason"),
                     "not_passed": [{"criterion": k.get("criterion") or k.get("id"), "attack": k.get("attack"),
                                     "status": k.get("status"), "value": k.get("value"), "limit": k.get("limit")}
                                    for k in bad[:MAX_ITEMS]]}
    return {"report": "privacy", "unit": (rep.get("unit") or {}).get("table") if isinstance(rep.get("unit"), dict)
            else rep.get("unit"), "people": rep.get("people"), "contexts": ctx,
            "warnings": [str(w)[:200] for w in (rep.get("warnings") or [])[:5]]}


def privacy_text(f: dict) -> str:
    out = [f"Privacy was measured per person ({f['people']} people)." if f.get("people") is not None
           else "Privacy was measured per person."]
    for name, c in f["contexts"].items():
        line = f"Context {name}: verdict {c['verdict']}"
        if c.get("external"):
            line += ", an external release"
        if c.get("dp_required"):
            line += ", which needs differential privacy"
        if c.get("blocked_reason"):
            line += f" (blocked: {c['blocked_reason']})"
        out.append(line + ".")
        for k in c["not_passed"]:
            out.append(f"- {k['criterion']}" + (f" by {k['attack']}" if k.get("attack") else "")
                       + f": {k['status']}, value {_fmt(k['value'])}, limit {_fmt(k['limit'])}")
        if c.get("sign_off_required"):
            out.append("  A privacy officer must sign this context off; the agent cannot.")
    for w in f.get("warnings") or []:
        out.append(f"Warning: {w}")
    out.append("The privacy verdicts are measurements; the release decision, any epsilon and any policy exception "
               "belong to the people the policy names.")
    return "\n".join(out)


def scorecard_facts(rep: dict) -> dict:
    sc: dict = rep["scorecard"] if isinstance(rep.get("scorecard"), dict) else rep
    items = sc.get("items") or []
    failed = [i for i in items if not i.get("pass")]
    return {"report": "scorecard", "passed": sc.get("passed"), "total": sc.get("total"),
            "failed": [{"area": i.get("area"), "metric": i.get("metric"), "value": i.get("value"),
                        "limit": i.get("limit")} for i in failed[:MAX_ITEMS]], "failed_total": len(failed)}


def scorecard_text(f: dict) -> str:
    out = [f"{f['passed']} of {f['total']} scorecard checks passed."]
    for i in f["failed"]:
        out.append(f"- {i['area']} / {i['metric']}: value {_fmt(i['value'])}, limit {_fmt(i['limit'])}")
    if f["failed_total"] > len(f["failed"]):
        out.append(f"... and {f['failed_total'] - len(f['failed'])} more.")
    return "\n".join(out)


def _narrate(facts: dict, text: str, backend) -> tuple[str, str, list[str]]:
    """(text, source, problems): the model's wording if the guard accepts it."""
    if not getattr(backend, "uses_model", False):
        return text, "rules", []
    marked = json.dumps({k: (mark(v) if isinstance(v, str) else v) for k, v in facts.items()}, default=str)
    task = ("Explain this report to a data owner in plain language, in at most 8 short sentences. Use only the "
            "facts given; quote numbers exactly as they appear; say who decides a release (people, not you). "
            "Facts: " + marked)
    try:
        out = backend.complete(SYSTEM_RULE, task, EXPLAIN_SCHEMA)
        cand = plain_text(out.get("summary", ""), 2000) if isinstance(out, dict) else ""
    except Exception as e:  # noqa: BLE001
        return text, "rules", [f"model call failed ({type(e).__name__})"]
    probs = []
    if not cand:
        probs.append("empty model text")
    un = ungrounded_numbers(cand, facts)
    if un:
        probs.append(f"numbers not in the report: {un[:5]}")
    if suspicious(cand):
        probs.append("the model text reads like an instruction")
    if any(w in cand.lower() for w in ("approved for release", "is approved", "safe to release", "you may release")):
        probs.append("the model text states a release decision")
    return (text, "rules", probs) if probs else (cand, "model", [])


def explain(report: dict, kind: str | None = None, *, backend=None) -> dict:
    """An explanation of a report: {kind, text, source, facts, problems}."""
    kind = kind or ("quality" if "vector" in report else "privacy" if "contexts" in report else "scorecard")
    facts, text = {"quality": (quality_facts, quality_text), "privacy": (privacy_facts, privacy_text),
                   "scorecard": (scorecard_facts, scorecard_text)}[kind][0](report), None
    text = {"quality": quality_text, "privacy": privacy_text, "scorecard": scorecard_text}[kind](facts)
    out, src, probs = _narrate(facts, text, backend)
    return {"kind": kind, "text": out, "source": src, "facts": facts, "problems": probs}


__all__ = ["explain", "quality_facts", "privacy_facts", "scorecard_facts"]

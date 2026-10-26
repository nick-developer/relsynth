"""Privacy report for a data protection officer: one page per release
context (Markdown and printable HTML) and the full results as JSON.

Each page answers, in order: may this release go to this recipient; what was
tested and against which limits; who is most at risk; what to do next.  The
technical details are in an annex after the context pages.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

CRITERIA = {
    "record_isolation": ("No record isolation", "Can someone use the synthetic data to single out a real person?"),
    "linkage": ("No linkage", "Can someone use it to connect two records about the same real person?"),
    "inference": ("No inference", "Can someone use it to learn a real person's attribute that population data "
                                  "would not reveal?"),
    "membership": ("Membership", "Can someone tell whether a given person's data was used?"),
}
VERDICT = {
    "pass": "PASS: every attack stays below its limit, with the safety margin of the upper confidence bound.",  # nosec B105
    "review": "REVIEW: at least one limit lies inside a confidence interval. More attacks (or more people) are "
              "needed to decide, or the release needs a reviewer's sign-off.",
    "fail": "FAIL: at least one attack is above its limit even at its lower confidence bound. Do not release to "
            "this recipient as it is.",
    "blocked": "BLOCKED: this recipient is outside the organisation and the release has no differential privacy "
               "guarantee, which the organisation requires for external releases.",
}


def _f(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def _ci(ci, pct=False):
    return "–" if not ci else f"{_f(ci[0], pct)} to {_f(ci[1], pct)}"


def _advice(rep, ctx) -> list[str]:
    out = []
    bad = [c for c in ctx["checks"] if c["status"] in ("fail", "review")]
    kinds = {c["criterion"] for c in bad}
    if ctx["verdict"] == "blocked":
        out.append("Generate this release with differential privacy, with the epsilon set by the organisation's "
                   "policy, then assess it again.")
    if "record_isolation" in kinds:
        out.append("Coarsen or drop the attributes used by the isolating queries (see the annex), or raise the "
                   "minimum group sizes.")
    if "linkage" in kinds:
        out.append("Reduce the detail that both halves of the linkage share, or exclude the behavioural tables.")
    if "inference" in kinds:
        out.append("Review the flagged sensitive attributes: generalise them or exclude them from the release.")
    if "membership" in kinds:
        out.append("Some people are recognisable as members. Consider removing or generalising outliers, more "
                   "people in the training data, or differential privacy.")
    if not out:
        out.append("No action needed for this recipient. Re-assess when the data, the settings or the recipient "
                   "change (EDPB para. 35).")
    if rep.get("sensitive_not_found"):
        out.append("Declared sensitive attributes not found in the profile: " + ", ".join(rep["sensitive_not_found"]))
    if not rep.get("sensitive"):
        out.append("No sensitive attributes were declared; inference was tested on behavioural attributes instead. "
                   "Declare the sensitive attributes in privacy.yaml.")
    return out


def context_markdown(rep: dict, name: str) -> str:
    ctx = rep["contexts"][name]
    u = rep["unit"]
    lines = [f"# Privacy report: {name} release of {rep.get('spec_name') or 'the database'}", "",
             f"**{VERDICT[ctx['verdict']]}**", ""]
    if ctx.get("blocked_reason"):
        lines += [f"> {ctx['blocked_reason']}", ""]
    knows = "the quasi-identifiers (" + ", ".join(rep["quasi_identifiers"][:8]) + \
        ("…" if len(rep["quasi_identifiers"]) > 8 else "") + ")" if ctx["attacker_knows"] == "quasi_identifiers" \
        else "every attribute of the people except the one being guessed, and data about the population"
    lines += [f"- **Recipient:** {ctx.get('description') or name}",
              f"- **Assumed attacker:** knows {knows}. Motivation is never assumed to be low (EDPB para. 31).",
              f"- **People protected:** each row of `{u['table']}`"
              + (" together with their rows in " + ", ".join(f"`{t}`" for t in u["owned_tables"])
                 if u["owned_tables"] else "") + ".",
              f"- **Tested on:** {rep['people']['training']} people in the training data, "
              f"{rep['people']['control']} unseen people as a control, {rep['people']['reference']} as the "
              f"attacker's population data. Intervals are {int(100 * rep['confidence'])}% confidence intervals.", ""]
    if ctx.get("dp"):
        dp = ctx["dp"]
        lines += [f"- **Differential privacy:** epsilon {dp.get('epsilon')}, delta {dp.get('delta', 0)}; no "
                  f"membership attack can catch more than {_f(ctx.get('dp_bound_tpr_at_fpr_0.01'), True)} of "
                  "members at a 1% false alarm rate.", ""]
    lines += ["## What was tested", "", "| Criterion | Attack | Result | Interval | Limit | Status |",
              "|---|---|---|---|---|---|"]
    for c in ctx["checks"]:
        crit = CRITERIA[c["criterion"]][0]
        lines.append(f"| {crit} | {c['attack'].replace('_', ' ')} | {_f(c['value'])} | {_ci(c['ci'])} | "
                     f"{_f(c['limit'])} | {c['status'].upper()} |")
    lines += ["", "Risk values are the share of attacks that succeed on people in the training data beyond what "
              "succeeds on unseen people (0 = no more than for strangers, 1 = always). AUC 0.5 means the attacker "
              "cannot tell members from non-members.", ""]
    recs = rep.get("vulnerable_records") or []
    if recs:
        lines += ["## People most at risk", "", "| Rank | Person | Membership AUC | Interval | Singled out | Linked "
                  "| Secret guessed |", "|---|---|---|---|---|---|---|"]
        for r in recs:
            yn = lambda b: "yes" if b else "no"  # noqa: E731
            lines.append(f"| {r['rank']} | {r['person']} | {_f(r['mia_auc'])} | {_ci(r.get('mia_auc_ci'))} | "
                         f"{yn(r['isolated'])} | {yn(r['linked'])} | {yn(r['secret_inferred'])} |")
        sh = ((rep.get("attacks") or {}).get("membership") or {}).get("shadow") or {}
        n_sh = (sh.get("n_shadows_train") or 0) + (sh.get("n_shadows_test") or 0)
        lev = recs[0].get("ci_level")
        lines += ["", "Person labels are salted hashes; the operator can map them back to keys. Each person was "
                  "the target of a membership attack over "
                  + (f"{n_sh} " if n_sh else "") + "shadow releases shared by all targets"
                  + (f"; intervals hold for all listed people at once (each at {100 * lev:.2f}%), because the "
                     "list picks the highest of many results" if lev else "") + ". 'Singled out', 'linked' and "
                  "'secret guessed' are marked only when that attack beats the control as a whole.", ""]
    lines += ["## What to do", ""] + [f"- {a}" for a in _advice(rep, ctx)] + [""]
    return "\n".join(lines)


def annex_markdown(rep: dict) -> str:
    A = rep["attacks"]
    L = ["# Annex: technical results", "", f"Privacy unit: `{rep['unit']['table']}` (key `{rep['unit']['key']}`)."]
    for t, p in rep["unit"]["owned_tables"].items():
        L.append(f"- `{t}`: {p}")
    L += ["", "## Record isolation, linkage and inference (Anonymeter-style)", "",
          "| Attack | Risk | Interval | Attack rate | Control rate | Baseline rate | Beats baseline |",
          "|---|---|---|---|---|---|---|"]
    for r in A["record_isolation"] + A["linkage"] + A["inference"]:
        name = r["attack"] + (f" ({r['secret']}, knows {r.get('attacker_knows')})" if "secret" in r else "")
        L.append(f"| {name} | {_f(r['value'])} | {_ci(r['ci'])} | {_f(r['attack_rate']['value'])} | "
                 f"{_f((r.get('control_rate') or {}).get('value'))} | {_f((r.get('baseline_rate') or {}).get('value'))}"
                 f" | {'yes' if r['attack_beats_baseline'] else 'no'} |")
    ex = [q for r in A["record_isolation"] for q in r.get("example_queries", [])][:5]
    if ex:
        L += ["", "Example isolating queries (on the synthetic data): " + "; ".join(f"`{q}`" for q in ex)]
    if A["attribute_inference"]:
        L += ["", "## Attribute inference against a population baseline", "",
              "| Secret | Knows | Accuracy (synthetic, members) | Accuracy (population, members) | "
              "Advantage members | Advantage non-members | Specific leakage | Interval |", "|---|---|---|---|---|---|---|---|"]
        for r in A["attribute_inference"]:
            L.append(f"| {r['secret']} | {r['attacker_knows']} | {_f(r['accuracy_syn_members'])} | "
                     f"{_f(r['accuracy_population_members'])} | {_f(r['advantage_members']['value'])} | "
                     f"{_f(r['advantage_non_members']['value'])} | {_f(r['specific_leakage']['value'])} | "
                     f"{_ci(r['specific_leakage']['ci'])} |")
    M = A["membership"]
    if M:
        L += ["", "## Membership inference", "", "| Attack | AUC | Interval | TPR at FPR 10% | TPR at 1% | TPR at 0.1% |",
              "|---|---|---|---|---|---|"]
        rows = []
        if "domias" in M and M["domias"].get("identified"):
            rows.append(("DOMIAS (release)", M["domias"]))
        for ro, p in (M.get("shadow", {}).get("pooled") or {}).items():
            rows.append((f"shadow {ro}", p))
        for n, p in rows:
            t = {round(e["fpr_target"], 4): e for e in p.get("tpr_at_fpr", [])}
            cell = lambda f: (f"{_f(t[f]['tpr'])} ({_ci(t[f]['tpr_ci'])})" if f in t and t[f].get("identified")  # noqa: E731
                              else "not identified")
            L.append(f"| {n} | {_f(p['auc'])} | {_ci(p['auc_ci'])} | {cell(0.1)} | {cell(0.01)} | {cell(0.001)} |")
        if "shadow" in M:
            s = M["shadow"]
            L += ["", f"Shadow modelling: {s['n_shadows_train']} training and {s['n_shadows_test']} test fits of "
                  f"about {s['shadow_size']} people ({s['fits_cached']} from the cache), {s['seconds']:.0f} s."]
            for ro, p in s["pooled"].items():
                if "mean_auc_vulnerable" in p:
                    L.append(f"- {ro}: mean AUC on vulnerable targets {_f(p['mean_auc_vulnerable'])}, on random "
                             f"targets {_f(p.get('mean_auc_random'))}.")
    g = rep.get("guards") or {}
    if g.get("dcr"):
        d = g["dcr"]
        L += ["", "## relsynth guards", "", f"- Distance to closest record, ratio of 5th percentiles: {_f(d['dcr_ratio'])}"
              f" (limit {g['dcr_ratio_limit']}).",
              f"- Exact copies of training rows: {_f(d['exact_match_rate'], True)} (real data's own repeat rate "
              f"{_f(d.get('exact_match_rate_real_any'), True)})."]
    tm = rep.get("timings", {})
    L += ["", "## Run time", ""] + [f"- {k.replace('_', ' ')}: {v:.1f}" for k, v in tm.items()]
    return "\n".join(L) + "\n"


def to_markdown(rep: dict) -> str:
    pages = [context_markdown(rep, n) for n in rep["contexts"]] + [annex_markdown(rep)]
    return "\n\n---\n\n".join(pages)


def _md_to_html(md: str) -> str:
    """The small Markdown subset the report uses (headings, bold, quotes,
    bullets, tables, code) to HTML, escaped."""
    out, in_table, in_list = [], False, False
    import re

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-") for c in cells):
                continue
            if not in_table:
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                cls = lambda c: ' class="bad"' if c in ("FAIL",) else ' class="warn"' if c == "REVIEW" else \
                    ' class="ok"' if c == "PASS" else ""  # noqa: E731
                out.append("<tr>" + "".join(f"<td{cls(c)}>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line == "---":
            out.append('<div class="page-break"></div>')
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</tbody></table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;
color:#1b1b1b;background:#fff;line-height:1.45}
h1{font-size:1.5rem;border-bottom:2px solid #444;padding-bottom:.3rem}h2{font-size:1.15rem;margin-top:1.4rem}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.6rem 0}th,td{border:1px solid #bbb;padding:.25rem .4rem;
text-align:left;vertical-align:top}th{background:#eee}td.bad{background:#f6c8c8;font-weight:600}
td.warn{background:#fbe8b5}td.ok{background:#d4ecd4}blockquote{border-left:4px solid #b00;margin:.5rem 0;
padding:.3rem .8rem;background:#fbeaea}code{font-size:.85em}
.page-break{page-break-after:always;break-after:page;height:0;border-top:1px dashed #ccc;margin:2rem 0}
@media print{.page-break{border:0;margin:0}body{margin:0}}
"""


def to_html(rep: dict) -> str:
    body = _md_to_html(to_markdown(rep))
    title = html.escape(f"Privacy report: {rep.get('spec_name') or 'database'}")
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            f"content=\"width=device-width,initial-scale=1\"><title>{title}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>\n")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def to_json(rep: dict) -> str:
    return json.dumps(_jsonable(rep), indent=1, ensure_ascii=False)


def write_report(rep: dict, folder) -> dict:
    """privacy_report.json, privacy_report.md and privacy_report.html in folder."""
    f = Path(folder)
    f.mkdir(parents=True, exist_ok=True)
    paths = {"json": f / "privacy_report.json", "md": f / "privacy_report.md", "html": f / "privacy_report.html"}
    paths["json"].write_text(to_json(rep), encoding="utf-8")
    paths["md"].write_text(to_markdown(rep), encoding="utf-8")
    paths["html"].write_text(to_html(rep), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}

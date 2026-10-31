# SPDX-License-Identifier: Apache-2.0
"""The constraints report of the self-tuning loop: what was met, what was not
and why (docs/research/self-tuning.md, sec. 5).

For every objective that is not met the report gives one of these reasons,
computed from the history:

- insensitive: the settings that act on the family were moved and none
  changed its failures by a whole check per seed; the misfit is in the model
  class or the spec, not the settings (doctor or an engine change);
- conflict: moves that fixed this objective broke another one (or the
  reverse) on the same seeds: a trade-off for the user to choose;
- undecided: a privacy check whose confidence interval straddles its limit;
  no generator setting narrows it (more attack rows, or a larger population);
- budget: settings that act on it were left untried when the budget ran out;
- no setting: no registered setting acts on the family.

For met hard constraints it states the evidence: k violations in n seeds bound
the violation rate at 95% (and the seeds needed for 5% and 1%).
"""
from __future__ import annotations

import math

from . import analysis as A


def effects(hist, obj):
    """One-setting effects learned from the history: every fully evaluated
    candidate against a fully evaluated one that differs from it in exactly one
    setting.  Returns (effects {family: {setting: [change in failures per seed]}},
    pairs [(setting, old, new, family changes, constraint changes)])."""
    by = hist.by_key()
    seeds = set(obj.seeds_search)
    cands = {}
    for key, rs in by.items():
        rs = {r["seed"]: r for r in rs if r["seed"] in seeds}
        if rs:
            first = next(iter(rs.values()))
            cands[key] = (first["delta"], first["patches"], rs)
    full = {k: (d, p, A.judge(list(rs.values()), obj)) for k, (d, p, rs) in cands.items() if set(rs) == seeds}
    eff: dict = {}
    pairs = []
    items = list(cands.values())
    for d1, p1, r1 in items:
        for d0, p0, r0 in items:
            if p1 != p0:
                continue
            diff = [k for k in set(d1) | set(d0) if d1.get(k) != d0.get(k)]
            if len(diff) != 1 or (diff[0] in d0 and diff[0] not in d1):
                continue
            common = sorted(set(r1) & set(r0))
            if not common:
                continue
            # compared on the seeds both were run on (a candidate stopped after one seed counts there)
            j1, j0 = A.judge([r1[s] for s in common], obj), A.judge([r0[s] for s in common], obj)
            k = diff[0]
            f1, f0 = j1.families(), j0.families()
            delta_f = {f: _mean(f1[f]["failed"]) - _mean(f0[f]["failed"]) for f in f1 if f in f0}
            c1, c0 = j1.constraints(obj.hard), j0.constraints(obj.hard)
            delta_c = {c: c1[c]["n_checks"] - c0.get(c, {"n_checks": 0})["n_checks"] for c in c1}
            pairs.append((k, d0.get(k), d1.get(k), delta_f, delta_c))
            for f, dv in delta_f.items():
                eff.setdefault(f, {}).setdefault(k, []).append(round(dv, 2))
    return eff, pairs, full


def analyse(hist, obj, inc, J, base_J, trials, space) -> dict:
    effects_, pairs, full = effects(hist, obj)
    conflicts: dict = {}
    for k, _, _, df, dc in pairs:
        better = [f for f, v in df.items() if v <= -1] + [f"constraint:{c}" for c, v in dc.items() if v < 0]
        worse = [f for f, v in df.items() if v >= 1] + [f"constraint:{c}" for c, v in dc.items() if v > 0]
        for b in better:
            for w in worse:
                key = " vs ".join(sorted((b, w)))
                conflicts.setdefault(key, set()).add(k)
    fam = J.families()
    out_f = {}
    for f, o in sorted(fam.items()):
        P0 = math.exp(-o["mu"]) if o["mu"] else 1.0
        row = {"checks": o["n_checks"], "expected_false_failures": round(o["mu"], 3), "null_quantile": o["q"],
               "failed_per_seed": o["failed"], "met": o["met"], "targeted": o["targeted"],
               "p_all_pass_perfect_model": round(P0, 4)}
        if o["targeted"] and not o["met"]:
            acting = [k for k, p in space.items() if f in p.affects]
            tried = effects_.get(f, {})
            moved = {k: v for k, v in tried.items()}
            best = min((min(v) for v in moved.values()), default=0.0)
            cf = sorted({c for c, ks in conflicts.items() if f in c.split(" vs ")})
            if not acting:
                reason = "no setting"
            elif cf:
                reason = "conflict"
            elif moved and best > -1 and len(moved) >= min(2, len(acting)):
                reason = "insensitive"
            else:
                reason = "budget"
            row.update({"reason": reason, "settings_acting": acting[:12], "effects": moved, "conflicts": cf})
        out_f[f] = row
    cons = J.constraints(obj.hard)
    for c, o in cons.items():
        o["seeds_for_5pct"] = A.seeds_needed(0.05)
        o["seeds_for_1pct"] = A.seeds_needed(0.01)
        if not o["met"]:
            fams = sorted({_fam(i) for i in o["checks"]})
            acting = sorted({k for k, p in space.items() if any(f in p.affects for f in fams)})
            moved = {f: effects_.get(f, {}) for f in fams if effects_.get(f)}
            cf = sorted({x for x in conflicts if f"constraint:{c}" in x.split(" vs ")})
            o["reason"] = "conflict" if cf else ("no setting" if not acting else
                                                 ("insensitive" if moved else "budget"))
            o["settings_acting"] = acting[:12]
            o["conflicts"] = cf
    expected = sum(o["mu"] for o in fam.values() if o["targeted"])
    return {"families": out_f, "constraints": cons, "undecided": J.undecided,
            "conflicts": {k: sorted(v) for k, v in conflicts.items()},
            "expected_false_failures": round(expected, 3),
            "p_all_statistical_pass_perfect_model": round(math.exp(-expected), 6),
            "candidates_evaluated": len(full)}


def _fam(cid):
    from .checks import family
    return family(cid)


def _mean(x):
    return sum(x) / len(x) if x else 0.0


def render(res) -> str:
    """The report as Markdown."""
    an, J, obj = res.analysis, res.judgement, res.objectives
    L = ["# relsynth autotune report", ""]
    verdict = "**All objectives are met on the search seeds.**" if res.met else \
        "**Not every objective is met.** The reasons are below."
    L += [verdict, ""]
    if res.holdout is not None:
        L += [f"Held-out seeds {res.holdout['tuned']['seeds']}: "
              + ("the tuned settings are **confirmed** (no worse than the defaults)." if res.confirmed else
                 "the tuned settings are **not confirmed**: on the held-out seeds they are worse than the defaults. "
                 "Keep the defaults, or tune again with more search seeds."), ""]
    L += [f"Stopped: {res.stopped}. Runs: {res.runs} new, {res.runs_reused} reused from the history; "
          f"{res.seconds:.0f} s.", ""]
    L += ["## Tuned settings", ""]
    if res.delta or res.patches:
        L += ["```yaml", "synthesis:"]
        L += [f"  {k}: {v}" for k, v in sorted(res.delta.items())]
        L += ["```", ""]
        if res.patches:
            L += ["Spec fixes (from `relsynth doctor`):", ""] + [f"- `{k}` = `{v}`" for k, v in sorted(res.patches.items())]
            L += [""]
    else:
        L += ["No change: the spec's own settings are the best found.", ""]
    b = res.extra.get("baseline") or {}
    L += ["## The spec as given and the tuned settings, on the search seeds", "",
          "| | Hard violations | Excess statistical failures per seed | Failures per seed | Lowest scorecard pass rate |",
          "|---|---|---|---|---|",
          f"| spec as given | {b.get('hard_violations')} | {b.get('excess_failures')} | {b.get('failures')} | "
          f"{b.get('min_scorecard_pass_rate')} |",
          f"| tuned | {J['hard_violations']} | {J['excess_failures']} | {J['failures']} | "
          f"{J['min_scorecard_pass_rate']} |", ""]
    L += ["## Hard constraints", "",
          "A constraint is met when none of its checks fails on any seed. The bound is the 95% upper credible "
          "bound of the violation rate given the seeds run (0 of n gives about 3/n).", "",
          "| Constraint | Met | Seeds violated | Rate bound (95%) | Seeds for a 5% / 1% bound | Reason |",
          "|---|---|---|---|---|---|"]
    for c, o in an["constraints"].items():
        if not o.get("evaluated", True):
            L.append(f"| {c} | no checks | | | | this spec has no check of it"
                     + (" (set privacy.attacks or domain_checks in the objectives)" if c in ("privacy", "domain")
                        else "") + " |")
            continue
        L.append(f"| {c} | {'yes' if o['met'] else 'no'} | {o['seeds_violated']} of {o['seeds']} | "
                 f"{o['rate_upper_95']} | {o['seeds_for_5pct']} / {o['seeds_for_1pct']} | {o.get('reason', '')} |")
    L += [""]
    bad = {c: o for c, o in an["constraints"].items() if not o["met"]}
    for c, o in bad.items():
        L += [f"- **{c}**: {o['n_checks']} check(s), e.g. " + ", ".join(f"`{x}`" for x in o["checks"][:5])]
        if o.get("settings_acting"):
            L += [f"  settings that act on it: {', '.join(o['settings_acting'])}"]
        if o.get("conflicts"):
            L += [f"  conflicts: {'; '.join(o['conflicts'])}"]
    if an["undecided"]:
        L += ["", "### Undecided privacy checks", "",
              "These checks' estimates are under their limits but their confidence intervals reach above them. No "
              "generator setting narrows an interval; more attack and control rows do, or the population is too "
              "small to decide.", ""] + [f"- `{x}`" for x in an["undecided"]]
    L += ["", "## Statistical families", "",
          f"Calibrated targets (user decision of 2026-10-31): a family is met when its failures on each seed are at "
          f"most the {obj['null_quantile']} quantile of its null count. For a perfect model the expected number of "
          f"false failures over the targeted families is {an['expected_false_failures']}, so the chance that every "
          f"statistical check passes on one seed is {an['p_all_statistical_pass_perfect_model']}; zero failures is "
          f"not a reachable target.", "",
          "| Family | Checks | Expected false failures | Allowed | Failed per seed | Met | Reason |",
          "|---|---|---|---|---|---|---|"]
    for f, o in sorted(an["families"].items(), key=lambda kv: (kv[1]["met"], kv[0])):
        if not o["targeted"]:
            continue
        L.append(f"| `{f}` | {o['checks']} | {o['expected_false_failures']} | {o['null_quantile']} | "
                 f"{o['failed_per_seed']} | {'yes' if o['met'] else 'no'} | {o.get('reason', '')} |")
    unmet = {f: o for f, o in an["families"].items() if o["targeted"] and not o["met"]}
    if unmet:
        L += ["", "### Why the unmet families are not fixed", ""]
        text = {"insensitive": "none of the moves tried changed its failures by a whole check per seed: the "
                               "misfit is most likely in the model or the spec, not the settings (see `relsynth "
                               "doctor`, or report it as an engine issue); a larger budget tries further moves",
                "conflict": "fixing it breaks another objective on the same seeds (a trade-off for you to choose)",
                "budget": "settings that act on it were not all tried before the budget ran out (raise budget.runs)",
                "no setting": "no registered setting acts on it"}
        for f, o in unmet.items():
            eff = "; ".join(f"{k}: {v}" for k, v in (o.get("effects") or {}).items())
            L += [f"- `{f}`: {text[o['reason']]}." + (f" Effects tried (change in failures per seed): {eff}."
                                                       if eff else "")]
            if o.get("conflicts"):
                L += [f"  Conflicts: {'; '.join(o['conflicts'])}."]
    if an["conflicts"]:
        L += ["", "## Trade-offs found", "", "| Objectives | Settings that moved one against the other |", "|---|---|"]
        L += [f"| {k} | {', '.join(v)} |" for k, v in sorted(an["conflicts"].items())]
    L += ["", "## Trajectory", "", "| Step | Change | From | Accepted | Excess | Hard | Seeds | Why |",
          "|---|---|---|---|---|---|---|---|"]
    for t in res.trajectory:
        L.append(f"| {t['step']} | {t['change']} | {t['origin']} | {'yes' if t['accepted'] else 'no'} | "
                 f"{t['excess']} | {t['hard']} | {t['seeds']} | {t['why']} |")
    L += ["", f"History: `{res.extra.get('history')}` (aggregates only). Run `relsynth autotune` again to continue "
              "from these runs.", ""]
    return "\n".join(L)

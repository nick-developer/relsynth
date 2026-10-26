# SPDX-License-Identifier: Apache-2.0
"""Doctor: read a diagnostics bundle, find model-layer issues, propose fixes as
spec patches, and optionally apply them.

Every rule has an id, looks only at aggregates in the bundle, and returns
(issue, evidence, patches).  A patch is {"path": "a.b.c", "value": v}: the
value is set in the spec (missing keys are created).  Issues whose cause is in
the engine rather than the spec are reported with kind="engine": those need a
code fix (send the bundle to the project).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import yaml

from .introspect import is_party_table


@dataclass
class Issue:
    id: str
    severity: str           # error | warn | info
    kind: str               # spec | data | engine
    where: str
    message: str
    evidence: dict = field(default_factory=dict)
    patches: list = field(default_factory=list)



_REGULAR_MSG = ("Rows per parent are more regular in the source than in the synthetic data, but the source "
                "is too small to show a grain (no pair of keys would repeat often even by chance). This is "
                "reported as a regularity of the source, not changed: declare a unique key only if the "
                "business model says each combination occurs once.")
_GRAIN_MIN_REPEATS = 10.0


def _grain_candidates(prof, child, r):
    """Unique-key candidates with no duplicates, kept only where independence would
    have produced at least 10 repeated pairs (P(0 | Poisson(10)) < 5e-5).  Returns
    None when zero-duplicate candidates exist but none carries that evidence."""
    zero = [kc for kc in (prof.get(child) or {}).get("key_candidates", [])
            if kc["duplicate_rate"] == 0 and not kc["declared"] and r.get("child_key") in kc["columns"]]
    strong = [kc for kc in zero if kc.get("expected_repeats_independent", _GRAIN_MIN_REPEATS) >= _GRAIN_MIN_REPEATS]
    return None if zero and not strong else strong

def _get(d, path, default=None):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _set(d, path, value):
    parts = path.split(".")
    for p in parts[:-1]:
        if isinstance(d.get(p), str):          # column shorthand "x: categorical"
            d[p] = {"type": d[p]}
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _syn(spec, path, default):
    return _get(spec, "synthesis." + path, default)


def diagnose(bundle) -> list[Issue]:
    spec = bundle["spec"]
    issues: list[Issue] = []
    ev = bundle.get("evaluation") or {}
    sc = ev.get("scorecard") or {}
    failed = sc.get("failed", [])
    model = bundle.get("model") or {}
    prof = bundle.get("profile") or {}
    tables = spec.get("tables", {})

    # D01 pipeline errors
    for e in bundle.get("errors", []):
        issues.append(Issue("D01", "error", "engine", e.get("step", "?"),
                            "A pipeline step failed; this needs a code fix. Share this bundle with the project.",
                            {"error": e.get("error"), "traceback": (e.get("traceback") or "")[-1500:]}))

    # D23 rows deleted by the sampler (rules or keys the redraws could not satisfy).
    # Deletion conditions the sample on an event the model does not know, so every
    # count and statistic of the table drifts: segment caps cannot repair that, and
    # patches that only raise caps are withheld for these tables.
    removed = ev.get("removed") or {}
    if not removed:
        for k, n in ((bundle.get("sampling") or {}).get("rows_removed_by_rules") or {}).items():
            t_ = str(k).split(" (")[0].split(":")[0]
            d = removed.setdefault(t_, {"removed": 0, "causes": {}})
            d["removed"] += int(n or 0)
            d["causes"][str(k)] = int(n or 0)
    lossy = set()
    for t_, d in removed.items():
        share = d.get("share")
        if share is None:
            share = d["removed"] / max(d["removed"] + (prof.get(t_) or {}).get("rows", 0), 1)
        if share <= 0.01:
            continue
        lossy.add(t_)
        causes = d.get("causes", {})
        hints = []
        if any("unique" in k for k in causes):
            hints.append("a unique key over columns that are drawn independently (an order number and a line "
                         "number): declare the order structure (fact.header with line_number) so that keys are "
                         "built, not filtered")
        if any("unique" not in k for k in causes):
            hints.append("check rules over keys or columns drawn independently (order <= ship <= due): state "
                         "how the columns are built instead (offset_from for dates, derive or lookup rules)")
        redraw = ((bundle.get("sampling") or {}).get("redrawn_rows") or {}).get(t_) or {}
        patches = []
        if redraw.get("rounds", 0) >= _syn(spec, "rules.max_rejection_rounds", 25) and redraw.get("still_invalid", 0) and \
                redraw.get("still_invalid", 0) < 0.5 * max(redraw.get("invalid_first_draw", 1), 1):
            # redraws were converging when they ran out of rounds
            patches.append({"path": "synthesis.rules.max_rejection_rounds", "value": _syn(spec, "rules.max_rejection_rounds", 25) * 4})
        issues.append(Issue("D23", "error", "spec", t_,
                            f"{share:.0%} of generated rows were deleted because they broke a rule or key; every "
                            "statistic of this table (and of tables nested in it) is affected. Likely cause: "
                            + "; ".join(hints) + ".", {"removed": d["removed"], "share": round(share, 4),
                                                        "causes": causes, "redraws": redraw}, patches))

    referenced = {r["parent"] for r in spec.get("relationships", []) if r.get("parent") != r.get("child")}

    # D17 declared types that disagree with the data profile
    for t, tp in prof.items():
        pk = tables.get(t, {}).get("primary_key")
        fks = {c for r in spec.get("relationships", []) if r["child"] == t
               for c in (r["child_key"] if isinstance(r["child_key"], list) else [r["child_key"]])}
        copied = tables.get(t, {}).get("mode") == "copy"     # reference data: levels are copied, not learned
        for c, p in tp["columns"].items():
            dec = p["declared"]
            if c == pk or c in fks:
                continue
            if dec == "categorical" and p.get("looks_like_email", 0) > 0.5:
                issues.append(Issue("D17", "error", "spec", f"{t}.{c}", "Column holds email addresses but is declared categorical; it would be copied level by level.",
                                    {"looks_like_email": p["looks_like_email"]},
                                    [{"path": f"tables.{t}.columns.{c}", "value": {"type": "pii", "pii": {"generator": "email"}}}]))
            elif dec == "categorical" and p["distinct_ratio"] > 0.5 and p["distinct"] > 50 and not copied \
                    and t in referenced and not is_party_table(t, list(tp["columns"])):
                # labels of a table other tables point to (cities of a geography, lines of a
                # chart of accounts) are codes, not identifiers of people: hiding them as pii
                # would break every join label; the choice is to copy the table or not
                issues.append(Issue("D17", "info", "spec", f"{t}.{c}",
                                    "Near-unique labels of a reference table that other tables point to. If they are "
                                    "public codes, copy the table (mode: copy); if they identify people or firms, "
                                    "declare the column pii.", {"distinct_ratio": p["distinct_ratio"]}))
            elif dec == "categorical" and p["distinct_ratio"] > 0.5 and p["distinct"] > 50 and not copied:
                txt = (p.get("mean_tokens") or 0) >= 5
                issues.append(Issue("D17", "error", "spec", f"{t}.{c}",
                                    "Near-unique categorical column: levels would leak and cannot be learned. "
                                    + ("It reads as free text." if txt else "It looks like an identifier or direct identifier."),
                                    {"distinct_ratio": p["distinct_ratio"], "mean_tokens": p.get("mean_tokens")},
                                    [{"path": f"tables.{t}.columns.{c}",
                                      "value": {"type": "text"} if txt else {"type": "pii", "pii": {"generator": "pattern"}}}]))
            elif dec in ("float", "integer") and p["distinct_ratio"] > 0.99 and p.get("integer_valued") and p.get("monotone") \
                    and not copied and tp.get("rows", 0) >= 100:
                # on a few rows a sorted, distinct measure (thresholds 0, 10, 25, 50) is
                # monotone by construction: evidence of a key needs many rows
                issues.append(Issue("D17", "warn", "spec", f"{t}.{c}", "Unique, monotone integer column looks like a key, not a measure.",
                                    {"distinct_ratio": p["distinct_ratio"]}, [{"path": f"tables.{t}.columns.{c}.type", "value": "id"}]))
            elif dec == "float" and p.get("integer_valued") and p["distinct"] <= 30:
                issues.append(Issue("D17", "info", "spec", f"{t}.{c}", "Float column only holds a few integer values; declare it integer so values stay whole.",
                                    {"distinct": p["distinct"]}, [{"path": f"tables.{t}.columns.{c}.type", "value": "integer"}]))

    # D02 segment caps saturated
    seg = model.get("segments", {})
    for e, cap in (seg.get("caps") or {}).items():
        occ = (seg.get("occupied_before_merge") or {}).get(e, 0)
        rows = (prof.get(e) or {}).get("rows", 0)
        mr = _syn(spec, "segments.min_rows_per_segment", 20)
        involved = any(e in f["area"] for f in failed)
        if cap > 1 and occ >= cap and cap * 2 <= max(rows // max(mr, 1), 1):
            if involved:
                issues.append(Issue("D02", "warn", "spec", e, "Every latent segment was used and checks involving this table fail: raise the cap.",
                                    {"cap": cap, "occupied": occ}, [{"path": f"synthesis.segments.tables.{e}", "value": cap * 2}]))
            else:
                issues.append(Issue("D02", "info", "spec", e, "Every latent segment was used, but all checks on this table pass; no change needed.",
                                    {"cap": cap, "occupied": occ}))
        elif cap == 1 and rows >= 200 and any(r.get("child") == e or r.get("parent") == e for r in spec.get("relationships", [])):
            issues.append(Issue("D02", "warn", "spec", e, "Segment cap is 1 for a related table: its links to other tables cannot depend on its attributes.",
                                {"cap": cap}, [{"path": f"synthesis.segments.tables.{e}", "value": 8}]))

    # D07 copula convergence.  The sampler already extends its own chain up to
    # copula.max_iterations; more iterations are proposed only while the
    # correlations still move (drift) and below a ceiling, never round after round.
    CEIL = 4800
    for t, cops in (model.get("copulas") or {}).items():
        for g, info in cops.items():
            acc = info.get("mh_acceptance")
            rhat = max([x for x in (info.get("rhat_logdet"), info.get("rhat_max_entry")) if x] or [1.0])
            drift = info.get("drift_max")
            bad_acc = info.get("mode") == "bayes" and acc is not None and not 0.08 <= acc <= 0.7
            if not (bad_acc or (rhat > 1.15 and (drift is None or drift > 0.02) and info.get("converged") is not True)):
                continue
            mx = _syn(spec, "copula.max_iterations", 600)
            patches = []
            if info.get("mode") == "bayes":
                patches.append({"path": "synthesis.copula.mode", "value": "rank"})
            elif mx < CEIL:
                patches.append({"path": "synthesis.copula.max_iterations", "value": min(mx * 2, CEIL)})
            k = _syn(spec, "marginals.min_dependence_rows", 5)
            if not patches:
                patches.append({"path": "synthesis.marginals.min_dependence_rows", "value": k * 2})
            issues.append(Issue("D07", "warn", "spec", f"{t}/{g}",
                                "Copula sampler has not converged well (usually columns whose rarer values sit on very "
                                "few rows; those can be kept out of the copula).",
                                {"acceptance": acc, "rhat": rhat, "drift": drift, "independent_columns":
                                 (model.get("independent_columns") or {}).get(t)}, patches))
            break

    rel = ev.get("relations") or {}
    for f in failed:
        area, metric, val = f["area"], f["metric"], f["value"]
        t = area.split(".")[0].split(":")[0].split("/")[0]
        col = area.split(".", 1)[1] if "." in area and ":" not in area else None
        if metric in ("marginal_ks",):
            cs = _get(tables, f"{t}.columns.{col}", {}) if col else {}
            p = _get(prof, f"{t}.columns.{col}", {}) if col else {}
            if p.get("distinct_ratio", 0) > 0.99 and p.get("monotone"):
                continue   # handled by D17
            g = _syn(spec, "copula.grid", 64)
            issues.append(Issue("D04", "warn", "spec", area, "Marginal distribution not reproduced closely enough.",
                                {"ks": val, "limit": f["limit"], "declared": cs.get("type")},
                                [{"path": "synthesis.copula.grid", "value": min(g * 2, 512)},
                                 {"path": "synthesis.marginals.tail_trim", "value": 0.0}]))
        elif metric == "marginal_tvd":
            p = _get(prof, f"{t}.columns.{col}", {}) if col else {}
            ml = _syn(spec, "marginals.max_levels", 200)
            patches = [{"path": "synthesis.marginals.max_levels", "value": ml * 2}] if p.get("distinct", 0) > ml else \
                [{"path": "synthesis.marginals.dirichlet_prior", "value": max(_syn(spec, "marginals.dirichlet_prior", 1.0) / 2, 0.1)}]
            issues.append(Issue("D05", "warn", "spec", area, "Categorical frequencies drift from the source.",
                                {"tvd": val, "limit": f["limit"], "levels": p.get("distinct")}, patches))
        elif metric in ("association_max_diff", "discriminator_auc"):
            tab = (ev.get("tables") or {}).get(t, {})
            a = tab.get("association", {})
            cap = _get(spec, f"synthesis.segments.tables.{t}", _syn(spec, "segments.default_cap", 8))
            patches = [{"path": "synthesis.copula.sample_rows", "value": _syn(spec, "copula.sample_rows", 1500) * 2}]
            if _syn(spec, "copula.mode", "rank") == "fast":
                patches.append({"path": "synthesis.copula.mode", "value": "rank"})
            if tables.get(t, {}).get("role") == "entity":
                patches.append({"path": f"synthesis.segments.tables.{t}", "value": max(cap * 2, 4)})
            issues.append(Issue("D06", "warn", "spec", t, "Dependence between columns is not reproduced (a Gaussian copula per segment is too coarse here).",
                                {"metric": metric, "value": val, "worst_pair": a.get("worst_pair"),
                                 "real": a.get("worst_real"), "syn": a.get("worst_syn")}, patches))
        elif metric == "fk_integrity":
            issues.append(Issue("D09", "error", "engine", area, "Synthetic foreign keys point to missing parents. This is an engine bug.", {"value": val}))
        elif metric in ("cardinality_ks", "cardinality_mean_rel_err"):
            d = rel.get(area, {})
            r = next((r for r in spec.get("relationships", []) if r.get("name") == area), {})
            parent, child = r.get("parent"), r.get("child")
            patches = []
            dr, ds = d.get("dispersion_real", 0) or 0, d.get("dispersion_syn", 0) or 0
            msg = "Rows per parent do not follow the source's shape."
            if dr > 1 and dr > 1.3 * max(ds, 1e-9):
                # over-dispersed (more uneven than Poisson): frailty adds the missing variance
                patches.append({"path": "synthesis.facts.frailty", "value": True})
                if r.get("kind") == "self_reference":
                    patches.append({"path": "synthesis.trees.frailty", "value": True})
            elif dr < 1 and ds > 1.3 * max(dr, 1e-3):
                # under-dispersed (more regular than Poisson): frailty would make it worse
                msg = ("Rows per parent are more regular in the source than in the synthetic data "
                       "(e.g. one row per parent and period, or a fixed set of rows): this is a grain, "
                       "declare it as a unique key so that it is modelled as occupancy.")
                cands = _grain_candidates(prof, child, r)
                if cands is None:
                    msg = _REGULAR_MSG
                elif cands:
                    patches.append({"path": f"tables.{child}.unique",
                                    "value": _get(tables, f"{child}.unique", []) + [cands[0]["columns"]]})
                elif tables.get(child, {}).get("role") == "fact":
                    patches.append({"path": f"tables.{child}.fact.count_model", "value": "dp"})
            if tables.get(parent, {}).get("mode") != "copy":
                # a copied parent has one fixed segment per row: a cap would change nothing
                cap_p = _get(spec, f"synthesis.segments.tables.{parent}", _syn(spec, "segments.default_cap", 8))
                if cap_p * 2 <= max((prof.get(parent) or {}).get("rows", 0) // max(_syn(spec, "segments.min_rows_per_segment", 20), 1), 1):
                    patches.append({"path": f"synthesis.segments.tables.{parent}", "value": cap_p * 2})
            issues.append(Issue("D03", "warn", "spec", area, msg,
                                {"metric": metric, "value": val, "dispersion_real": d.get("dispersion_real"),
                                 "dispersion_syn": d.get("dispersion_syn"), "zero_share_real": d.get("zero_share_real"),
                                 "zero_share_syn": d.get("zero_share_syn")}, patches))
        elif metric in ("join_tvd", "join_ks", "join_assoc_max_z"):
            fact = area.split(".joined")[0]
            anchor = _get(tables, f"{fact}.fact.anchor")
            dims = [r["parent"] for r in spec.get("relationships", []) if r["child"] == fact
                    and tables.get(r["parent"], {}).get("mode") != "copy" and tables.get(r["parent"], {}).get("role") != "fact"]
            patches = [{"path": f"synthesis.segments.tables.{d}",
                        "value": _get(spec, f"synthesis.segments.tables.{d}", _syn(spec, "segments.default_cap", 8)) * 2}
                       for d in dims]
            issues.append(Issue("D12", "warn", "spec", area, "Fact rows do not relate to their ancestors' attributes as in the source.",
                                {"metric": metric, "value": val, "anchor": anchor}, patches))
        elif metric == "null_fk_diff":
            issues.append(Issue("D08", "info", "engine", area, "Share of empty foreign keys differs from the source.", {"value": val}))
        elif metric in ("tree_depth_tvd", "tree_depth_ppc_pvalue", "tree_fanout_ks"):
            d = (ev.get("trees") or {}).get(area, {})
            patches = [{"path": "synthesis.trees.frailty", "value": True}]
            md = _syn(spec, "trees.max_depth", "auto")
            if d.get("max_depth_syn", 0) < d.get("max_depth_real", 0):
                patches.append({"path": "synthesis.trees.max_depth", "value": int(d["max_depth_real"] * 1.5) + 2})
            t0 = area.split(".")[0]
            patches.append({"path": f"synthesis.segments.tables.{t0}",
                            "value": _get(spec, f"synthesis.segments.tables.{t0}", _syn(spec, "segments.default_cap", 8)) + 2})
            issues.append(Issue("D11", "warn", "spec", area, "Hierarchy shape (depth / children per node) differs from the source.",
                                {"metric": metric, "value": val, "max_depth_real": d.get("max_depth_real"),
                                 "max_depth_syn": d.get("max_depth_syn"), "current_max_depth": md}, patches))
        elif metric.startswith("time_bucket"):
            fact = area
            anchor = _get(tables, f"{fact}.fact.anchor") or next((r["parent"] for r in spec.get("relationships", []) if r["child"] == fact), None)
            patches = [{"path": f"synthesis.segments.tables.{anchor}",
                        "value": _get(spec, f"synthesis.segments.tables.{anchor}", _syn(spec, "segments.default_cap", 8)) * 2}]
            issues.append(Issue("D12", "warn", "spec", area, "Seasonality is not reproduced: segment-specific time profiles need more segments.",
                                {"metric": metric, "value": val}, patches))
        elif metric in ("text_rare_token_leaks", "text_rare_ngram_copies", "text_rare_exact_copies"):
            t1, c1 = area.split(".", 1)
            k = _get(tables, f"{t1}.columns.{c1}.text.min_token_rows", _syn(spec, "text.min_token_rows", 3))
            L = _get(tables, f"{t1}.columns.{c1}.text.max_copy_ngram", _syn(spec, "text.max_copy_ngram", 8))
            issues.append(Issue("D13", "error", "spec", area, "Generated text reproduces rare source content.",
                                {"metric": metric, "value": val},
                                [{"path": f"tables.{t1}.columns.{c1}.text.min_token_rows", "value": max(k + 2, 3)},
                                 {"path": f"tables.{t1}.columns.{c1}.text.max_copy_ngram", "value": min(max(L - 2, 4), 8)}]))
        elif metric == "text_tstr_ratio":
            t1, c1 = area.split(".", 1)
            o = _get(tables, f"{t1}.columns.{c1}.text.order", _syn(spec, "text.order", 3))
            issues.append(Issue("D13", "warn", "spec", area, "Generated text carries too little of its class signal.",
                                {"tstr_ratio": val}, [{"path": f"tables.{t1}.columns.{c1}.text.order", "value": min(o + 1, 5)}]))
        elif metric in ("dcr_ratio", "exact_match_rate"):
            issues.append(Issue("D14", "error", "spec", area, "Synthetic rows sit too close to real rows (re-identification risk).",
                                {"metric": metric, "value": val, "limit": f["limit"]},
                                [{"path": "synthesis.marginals.tail_trim", "value": 0.01},
                                 {"path": "synthesis.copula.grid", "value": max(_syn(spec, "copula.grid", 64) // 2, 16)},
                                 {"path": "synthesis.marginals.dirichlet_prior", "value": _syn(spec, "marginals.dirichlet_prior", 1.0) * 4}]))
        elif metric == "unique_duplicate_rate":
            t_ = area.split(":")[0]
            issues.append(Issue("D18", "error", "engine", area,
                                "A declared unique key is broken in the synthetic data. Keys over two foreign keys, a "
                                "fact's (anchor, partner) pair and filtered keys on facts are enforced; check that the "
                                "key's columns are not nullable, or share the bundle with the project.",
                                {"syn": val}))
        elif metric == "pair_repeat_share_diff":
            f_ = area.split(":")[0]
            d = (ev.get("grain") or {}).get(area, {})
            patches = []
            if (d.get("real") or 0) == 0:
                cols = area.split("(")[-1].rstrip(")").split(", ")
                patches.append({"path": f"tables.{f_}.unique", "value": _get(tables, f"{f_}.unique", []) + [cols]})
            else:
                patches.append({"path": f"tables.{f_}.fact.pairs", "value": "occupancy"})
            issues.append(Issue("D19", "warn", "spec", area, "Repeated (anchor, partner) pairs are not reproduced.",
                                {"real": d.get("real"), "syn": d.get("syn")}, patches))
        elif metric == "novel_combination_rate":
            f_ = area.split(":")[0]
            ml = _syn(spec, "segments.stratify_max_levels", 20)
            cur = _syn(spec, "segments.stratify", "auto")
            pd20 = [{"path": "synthesis.segments.stratify", "value": "auto"},
                    {"path": "synthesis.segments.stratify_max_levels", "value": max(ml * 2, 40)}]
            if cur is False or isinstance(cur, dict):
                # the user chose the strata (or none, e.g. for fit cost): do not override it
                pd20 = []
            elif cur == "auto":
                pd20 = pd20[1:]
            # strata are the anchor's copied reference parents: without any, stratifying changes nothing
            anc = (_get(tables, f"{f_}.fact", {}) or {}).get("anchor")
            strata = [r["parent"] for r in spec.get("relationships", [])
                      if r.get("child") == anc and _get(tables, f"{r.get('parent')}.mode", "") == "copy"]
            if not strata:
                pd20 = []
            issues.append(Issue("D20", "warn", "spec", area,
                                "Synthetic rows combine reference keys that never occur together in the source "
                                "(e.g. a code that belongs to another version). "
                                + ("Stratifying the anchor by its small reference parents nests segments in them and "
                                   "makes such links exact; if" if strata else
                                   f"The anchor {anc or '?'} has no reference parents to stratify by, so no setting "
                                   "makes this exact (a limit of the segment model on small links); if")
                                + " the link is a business rule, write it as a check rule on the keys.",
                                {"syn": val}, pd20))
        elif metric == "rule_violation_rate":
            rr = (ev.get("rules") or {}).get(area, {})
            if rr.get("violation_rate_real", 0) > 0:
                issues.append(Issue("D15", "info", "data", area, "The source data itself breaks this rule; check the rule or clean the source.",
                                    {"real": rr.get("violation_rate_real"), "syn": val}))
            else:
                issues.append(Issue("D15", "warn", "spec", area, "Some synthetic rows still break the rule after rejection sampling.",
                                    {"syn": val}, [{"path": "synthesis.rules.max_rejection_rounds",
                                                    "value": _syn(spec, "rules.max_rejection_rounds", 25) * 4}]))
    # rules on the local replication checks (local/replicate_checks.py)
    for c in (bundle.get("replication") or {}).get("checks", []):
        if c["pass"]:
            continue
        if c["id"] == "R2" and isinstance(c.get("value"), (int, float)) and c["value"] <= 0.05:
            issues.append(Issue("R2", "info", "spec", c["what"].split(" ")[0], "Copula chain mixes slowly but its correlations move by "
                                "less than 0.05: no visible effect on the synthetic data.", {"drift": c["value"]}))
        elif c["id"] == "R2":
            mx = _syn(spec, "copula.max_iterations", 600)
            k = _syn(spec, "marginals.min_dependence_rows", 5)
            patches = [{"path": "synthesis.copula.max_iterations", "value": min(mx * 2, 4800)}] if mx < 4800 else \
                [{"path": "synthesis.marginals.min_dependence_rows", "value": k * 2}]
            if _syn(spec, "copula.mode", "rank") == "bayes":
                patches = [{"path": "synthesis.copula.mode", "value": "rank"}]
            issues.append(Issue("R2", "warn", "spec", c["what"].split(" ")[0], "Copula sampler mixes slowly (often two nearly deterministic columns).",
                                {"value": c["value"]}, patches))
        elif c["id"] == "R3" and "dispersion" in c["what"]:
            rel_name = c["what"].split(":")[0]
            r = next((r for r in spec.get("relationships", []) if r.get("name") == rel_name), {})
            p, ch = r.get("parent"), r.get("child")
            prow = (prof.get(p) or {}).get("rows", 0)
            # value = signed z of the real statistic against the synthetic copies
            sign = c.get("sign")
            if sign is None:          # bundles from older kits: read it from the text
                import re as _re
                m = _re.search(r"real ([-+0-9.e]+), synthetic ([-+0-9.e]+)", c["what"])
                sign = (1 if float(m.group(1)) >= float(m.group(2)) else -1) if m else 1
            under = sign < 0
            if under:
                msg = ("Rows per parent are more regular in the source than in every synthetic copy "
                       "(a grain such as one row per parent and period): declare it as a unique key.")
                cands = _grain_candidates(prof, ch, r)
                if cands is None:
                    msg = _REGULAR_MSG
                patches = [{"path": f"tables.{ch}.unique", "value": _get(tables, f"{ch}.unique", []) + [cands[0]["columns"]]}] \
                    if cands else []
            else:
                patches = [{"path": "synthesis.facts.frailty", "value": True}]
                msg = "Some parents get far more rows than others, more than the model reproduces."
                if p and prow and prow <= 200 and tables.get(p, {}).get("mode") != "copy":
                    patches.append({"path": f"tables.{p}.mode", "value": "copy"})
                    msg += f" {p} is a small reference table ({prow} rows): copying it keeps each row's own popularity."
            issues.append(Issue("R3", "warn", "spec", rel_name, msg, {"z": c["value"], "detail": c["what"]}, patches))
        elif c["id"] == "R3":
            issues.append(Issue("R3", "warn", "spec", c["what"].split(":")[0], "A key statistic falls outside the spread of synthetic copies.",
                                {"z": c["value"], "detail": c["what"]}))
        elif c["id"] == "R4":
            issues.append(Issue("R4", "warn", "engine", c["what"].split(",")[0], "Rows per anchor drift when tables are resized.",
                                {"ratio": c["value"], "detail": c["what"]}))
        elif c["id"] == "R6":
            e = c["what"].split(":")[0]
            gi = _syn(spec, "gibbs.iterations", 120)
            issues.append(Issue("R6", "info", "spec", e, "Segments differ between seeds: several partitions explain the data about equally well. "
                                "Synthesis is unaffected, but segment labels should not be interpreted.",
                                {"ari": c["value"]}, [{"path": "synthesis.gibbs.iterations", "value": gi * 2}]))
    # D21 grains visible in the source profile but not declared (proactive, info)
    for t, tp in prof.items():
        if tables.get(t, {}).get("mode") == "copy":
            continue
        for kc in tp.get("key_candidates", []):
            if kc["duplicate_rate"] == 0 and not kc["declared"] and kc["rows"] >= 100 \
                    and kc.get("expected_repeats_independent", _GRAIN_MIN_REPEATS) >= _GRAIN_MIN_REPEATS:
                issues.append(Issue("D21", "info", "spec", f"{t}: {kc['columns']}",
                                    f"No two of the {kc['rows']} source rows share these columns. If that is a rule of "
                                    "the data (a grain), declare it so the synthetic data keeps it.",
                                    {"rows": kc["rows"], "suggested_patch": {"path": f"tables.{t}.unique",
                                                                              "add": kc["columns"]}}))
    # D22 computed columns that the source itself does not follow
    for n, d in (ev.get("rules") or {}).items():
        if d.get("kind") in ("derive", "aggregate", "lookup") and (d.get("violation_rate_real") or 0) > 0:
            issues.append(Issue("D22", "info", "data", n, "The stored column differs from its definition in the source "
                                "for some rows; the synthetic data follows the definition exactly.",
                                {"real": d.get("violation_rate_real")}))
    for w in model.get("warnings", []):
        if "orphan" in w or "duplicate" in w or "null/orphan" in w:
            issues.append(Issue("D10", "info", "data", w.split(":")[0], "Source data quality: " + w))
    if lossy:
        # statistics of a table that lost rows are consequences of D23: raising caps,
        # sweeps or frailty would multiply the fit time and fix nothing
        for i in issues:
            if i.id != "D23" and any(t_ in str(i.where) for t_ in lossy):
                kept = [p for p in i.patches if not str(p["path"]).startswith(("synthesis.segments", "synthesis.gibbs",
                                                                                "synthesis.facts.frailty"))]
                if len(kept) < len(i.patches):
                    i.patches = kept
                    i.message += " (Probably a consequence of the rows deleted in this table, see D23; no tuning proposed.)"
    return _dedupe(issues)


def _dedupe(issues):
    seen, out = set(), []
    for i in issues:
        key = (i.id, i.where, i.message if not i.patches else "", str(i.patches))
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def merge_patches(issues):
    """Combine patches; for numeric values on the same path keep the most
    aggressive (max) except for paths that should shrink."""
    shrink = {"synthesis.copula.grid"} if any(i.id == "D14" for i in issues) else set()
    out = {}
    for i in issues:
        for p in i.patches:
            path, v = p["path"], p["value"]
            if path in out and isinstance(v, (int, float)) and isinstance(out[path], (int, float)) and not isinstance(v, bool):
                out[path] = min(out[path], v) if path in shrink else max(out[path], v)
            elif path in shrink and path in out:
                continue
            else:
                out[path] = v
    return out


def apply_patches(raw_spec, patches: dict):
    s = copy.deepcopy(raw_spec)
    for path, v in patches.items():
        _set(s, path, v)
    return s


def render(issues, patches, bundle):
    sc = (bundle.get("evaluation") or {}).get("scorecard") or {}
    lines = [f"# Doctor report for spec {bundle.get('spec_hash')}", "",
             f"Scorecard: {sc.get('passed', '?')} of {sc.get('total', '?')} checks passed.", ""]
    if not issues:
        lines.append("No issues found.")
    for sev in ("error", "warn", "info"):
        group = [i for i in issues if i.severity == sev]
        if not group:
            continue
        lines += [f"## {sev.upper()} ({len(group)})", ""]
        for i in group:
            lines.append(f"- **{i.id} {i.where}** ({i.kind}): {i.message}")
            if i.evidence:
                lines.append(f"  - evidence: `{ {k: v for k, v in i.evidence.items() if k != 'traceback'} }`")
            for p in i.patches:
                lines.append(f"  - fix: set `{p['path']}` to `{p['value']}`")
            if i.evidence.get("traceback"):
                lines += ["  ```", "  " + i.evidence["traceback"].replace("\n", "\n  "), "  ```"]
    if patches:
        lines += ["", "## Combined spec patch", "", "```yaml", yaml.safe_dump(patches, sort_keys=True).strip(), "```"]
    return "\n".join(lines) + "\n"

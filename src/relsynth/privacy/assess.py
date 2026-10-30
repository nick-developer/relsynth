# SPDX-License-Identifier: Apache-2.0
"""Privacy assessment of a synthetic release: split the people, fit, attack,
report.

    people ──┬─ training (fit the release on them; the members)
             ├─ control   (never seen; the non-members and Anonymeter's control)
             └─ reference (the attacker's population data; never seen)

The split is by person (Prop. 600): every row of a person, in every table, is
on one side.  The release that is attacked is fitted on the training people
only, so the control and reference people are genuinely unseen; the measured
risk is that of a release of the same generator and settings fitted on all
people (the standard proxy in Anonymeter and the shadow-model literature).
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from . import criteria, mia, unit as U
from .attribute import attribute_inference
from .stats import auc_interval


def relsynth_generator(spec: dict):
    """fit-and-sample function for relsynth with the spec's settings, at scale
    1 and with every table in memory (shadow databases are small)."""
    from ..data import coerce
    from ..engine import Synthesizer
    base = copy.deepcopy(spec)
    base["synthesis"]["scale"] = {"default": 1.0}
    base["synthesis"].setdefault("pushdown", {})["mode"] = "off"

    def gen(tables, seed):
        s = copy.deepcopy(base)
        s["synthesis"]["seed"] = int(seed)
        syn = Synthesizer(s).fit({t: df.reset_index(drop=True) for t, df in tables.items()}).sample(index=0)
        return {t: coerce(df, s["tables"][t]) for t, df in syn.items()}
    gen.ident = hashlib.sha256(json.dumps(base, sort_keys=True, default=str).encode()).hexdigest()  # type: ignore[attr-defined]
    return gen


def _quiet(fn, *a, **k):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def default_linkage_columns(spec, unit: U.Unit, prof: pd.DataFrame, qis: list[str]):
    """Two disjoint column sets an attacker could hold in separate sources: the
    person's own attributes and their behaviour in the owned tables (multi-table),
    or the quasi-identifiers and the rest (one table)."""
    own = [c for c in prof.columns if any(c.startswith(f"{t}.") for t in unit.owned)]
    first = [c for c in prof.columns if c not in own]
    if own and first:
        return first, own
    rest = [c for c in prof.columns if c not in qis]
    if qis and rest:
        return list(qis), rest
    cols = list(prof.columns)
    return cols[::2], cols[1::2]


def assess(spec, cfg=None, *, tables: dict | None = None, generator=None, seed: int | None = None,
           salt: bytes | None = None, progress=None) -> dict:
    """Run every configured attack and return the privacy report (a dict that
    validates against relsynth/privacy/report_schema.json).

    People are labelled with an HMAC of their key under salt, a secret that is
    not the split seed (the seed is in the report, so a seed-derived label
    could be reversed by anyone holding the real keys).  Without salt a fresh
    256-bit one is drawn; it is returned in rep["_salt"], which to_json and
    write_report never write.  The report carries only its id."""
    from ..api import _resolved
    t_start = time.time()
    cfg = C.load(cfg)
    s = _resolved(spec)
    if tables is None:
        from ..pushdown import load_for_fit
        tables = load_for_fit(s, mode="off").tables
    tables = {t: df.reset_index(drop=True) for t, df in tables.items()}
    seed = cfg["split"]["seed"] if seed is None else seed
    salt = secrets.token_bytes(32) if salt is None else bytes(salt)
    level = cfg["confidence"]
    say = progress or (lambda *_: None)
    timings = {}

    unit = U.resolve(s, cfg["unit"])
    ids = tables[unit.table][unit.key].dropna().unique()
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    if cfg["split"]["max_people"] and len(ids) > cfg["split"]["max_people"]:
        ids = ids[:cfg["split"]["max_people"]]
    n_c = int(round(cfg["split"]["control"] * len(ids)))
    n_r = int(round(cfg["split"]["reference"] * len(ids)))
    ctl_ids, ref_ids, tr_ids = ids[:n_c], ids[n_c:n_c + n_r], ids[n_c + n_r:]
    if generator is None:
        from ..dp.config import section
        if section(s) is not None:
            from .. import dp
            generator = dp.generator(s)
            if not (cfg.get("release") or {}).get("dp"):
                c = dp.parse(s)
                cfg["release"] = {**(cfg.get("release") or {}),
                                  "dp": {"epsilon": float(dp.epsilon_of_config(c)), "delta": c.delta, "unit": c.unit}}
    gen = generator or relsynth_generator(s)

    t = time.time()
    say("fitting the release on the training people")
    train_db = U.subset(unit, tables, tr_ids)
    syn_db = _quiet(gen, train_db, seed)
    timings["release_fit_seconds"] = time.time() - t
    t = time.time()
    P_all = U.profiles(s, unit, tables, dedupe=True)
    col_kinds = dict(P_all.attrs.get("kinds", {}))
    P_syn = U.profiles(s, unit, syn_db, kinds=col_kinds)
    cols = [c for c in P_all.columns if c in P_syn.columns]
    P_all, P_syn = P_all[cols], P_syn[cols].reset_index(drop=True)
    P_tr, P_c, P_r = P_all.loc[tr_ids], P_all.loc[ctl_ids], P_all.loc[ref_ids]
    timings["profile_seconds"] = time.time() - t

    qis = cfg["quasi_identifiers"]
    if qis is None:
        qis = [c for c in cols if not any(c.startswith(f"{o}.") for o in unit.owned)
               and "." not in c] or cols[: max(1, len(cols) // 2)]
    qis = [c for c in qis if c in cols]
    sensitive = [c for c in cfg["sensitive"] if c in cols]
    missing = [c for c in cfg["sensitive"] if c not in cols]
    A = cfg["attacks"]
    res: dict[str, Any] = {"record_isolation": [], "linkage": [], "inference": [], "attribute_inference": [],
                           "membership": {}}
    flags: dict[str, set] = {k: set() for k in ("isolated", "linked", "inferred", "attribute")}

    t = time.time()
    say("record isolation")
    for mode in A["record_isolation"]["modes"]:
        r = criteria.record_isolation(P_tr, P_syn, P_c, mode=mode, n_attacks=A["record_isolation"]["n_attacks"],
                                      n_cols=A["record_isolation"]["n_cols"], seed=seed, level=level)
        res["record_isolation"].append(r.to_dict())
        # a person counts as singled out only when the attack beats the control
        if r.risk.ci[0] > 0:
            flags["isolated"] |= set(r.isolated_ids)
    say("linkage")
    aux = A["linkage"]["aux_cols"] or default_linkage_columns(s, unit, P_all, qis)
    lk = criteria.linkage(P_tr, P_syn, P_c, aux_cols=aux, n_attacks=A["linkage"]["n_attacks"],
                          n_neighbors=A["linkage"]["n_neighbors"], seed=seed, level=level)
    res["linkage"].append(lk.to_dict())
    if lk.risk.ci[0] > 0:
        flags["linked"] |= set(lk.linked_ids)
    say("inference")
    # without declared sensitive attributes, the person's behaviour in the owned
    # tables (what they bought, how much) stands in, then their other attributes
    behaviour = [c for c in cols if any(c.startswith(f"{o}.") for o in unit.owned) and not c.endswith(".n_rows")]
    guessed = sensitive or ([c for c in behaviour if c not in qis] or [c for c in cols if c not in qis])[:5]
    for sec in guessed:
        for knows, known in (("quasi_identifiers", qis), ("all", [c for c in cols if c != sec])):
            known = [c for c in known if c != sec]
            if not known:
                continue
            inf = criteria.inference(P_tr, P_syn, P_c, attribute=sec, aux_cols=known,
                                     n_attacks=A["inference"]["n_attacks"], seed=seed, level=level)
            d = inf.to_dict()
            d["attacker_knows"] = knows
            d["declared_sensitive"] = sec in sensitive
            res["inference"].append(d)
            if sec in sensitive and inf.risk.ci[0] > 0:
                flags["inferred"] |= set(inf.correct_ids)
            if A["attribute_inference"] is not False and len(P_r) >= 20:
                a = attribute_inference(P_tr, P_c, P_syn, P_r, attribute=sec, known=known,
                                        max_rows=A["attribute_inference"].get("max_rows", 20000), seed=seed,
                                        level=level)
                gained = set(a.pop("_gained_ids"))
                if sec in sensitive and a["specific_leakage"]["ci"][0] > 0:
                    flags["attribute"] |= gained
                a["attacker_knows"] = knows
                a["declared_sensitive"] = sec in sensitive
                res["attribute_inference"].append(a)
    timings["criteria_seconds"] = time.time() - t

    t = time.time()
    M = A["membership"]
    dom_scores = None
    if M.get("domias") and len(P_r) >= 20 and len(P_c) >= 20:
        say("DOMIAS")
        d = mia.domias(P_tr, P_c, P_syn, P_r, level=level, seed=seed)
        dom_scores = d.pop("_member_scores", None)
        res["membership"]["domias"] = d
    timings["domias_seconds"] = time.time() - t

    t = time.time()
    vul = mia.vulnerability(pd.concat([P_tr, P_r]))
    sh = M.get("shadow", {})
    targets_tab = None
    if sh.get("enabled", True) and len(P_tr) >= 10:
        vt = vul.loc[P_tr.index].sort_values(ascending=False)
        top = list(vt.index[: sh["targets_vulnerable"]])
        rest = [i for i in P_tr.index if i not in set(top)]
        rnd = list(rng.choice(rest, size=min(sh["targets_random"], len(rest)), replace=False)) if rest else []
        kinds = pd.Series({**{i: "vulnerable" for i in top}, **{i: "random" for i in rnd}})
        targets = P_all.loc[top + rnd]
        pool = np.r_[ref_ids, ctl_ids]
        size = int(min(sh["shadow_size"], len(tr_ids), max(len(pool) // 2, 1)))
        cache = mia.ShadowCache(cfg["cache"], getattr(gen, "ident", "custom") + json.dumps(cols))

        def synth_profiles(ids_, sd):
            db = _quiet(gen, U.subset(unit, tables, ids_), sd)
            return U.profiles(s, unit, db, kinds=col_kinds)[cols].reset_index(drop=True)
        n_test = _n_test(cfg)
        say(f"shadow modelling: {sh['n_train'] + n_test} fits of {size} people")
        shr = mia.shadow_attack(pool, targets, kinds, synth_profiles, shadow_size=size, n_train=sh["n_train"],
                                n_test=n_test, seed=seed, ref_profiles=P_r if len(P_r) >= 20 else P_all,
                                cache=cache, vuln=vul, level=level, n_jobs=int(sh.get("n_jobs", 1)),
                                progress=lambda i, n: say(f"  shadow {i}/{n}") if i % 8 == 0 else None)
        res["membership"]["shadow"] = shr.to_dict()
        targets_tab = shr.targets
    timings["shadow_seconds"] = time.time() - t

    # the people most at risk: shadow targets ranked by their best attack
    records = []
    if targets_tab is not None:
        auc_cols = [c for c in ("classifier_auc", "lira_auc") if c in targets_tab]
        tab = targets_tab.copy()
        tab["risk_auc"] = tab[auc_cols].max(axis=1) if auc_cols else np.nan
        tab = tab.sort_values("risk_auc", ascending=False)
        # the most exposed person is a maximum over all targets and both attacks:
        # simultaneous (Bonferroni) intervals keep its coverage at the nominal level
        m = max(1, int(tab[auc_cols].notna().sum().sum())) if auc_cols else 1
        sim_level = 1 - (1 - level) / m
        for rank, (pid, row) in enumerate(tab.head(10).iterrows(), 1):
            best = max(auc_cols, key=lambda c: row.get(c, -1)) if auc_cols else None
            se = row.get(f"{best}_se") if best else None
            sim_ci = (auc_interval(float(row[best]), float(se), sim_level)
                      if se is not None and pd.notna(se) else row.get(f"{best}_ci") if best else None)
            records.append({
                "rank": rank, "person": _pseudonym(pid, salt), "kind": row["kind"],
                "vulnerability": float(row.get("vulnerability", np.nan)),
                "mia_auc": float(row["risk_auc"]), "mia_auc_ci": sim_ci, "ci_level": sim_level,
                "best_attack": best.replace("_auc", "") if best else None,
                "isolated": pid in flags["isolated"], "linked": pid in flags["linked"],
                "attribute_inferred": pid in flags["inferred"] or pid in flags["attribute"],
                "domias_percentile": float((dom_scores < dom_scores.get(pid, -np.inf)).mean())
                if dom_scores is not None and pid in dom_scores.index else None})

    guards = _guards(s, unit, train_db, syn_db)
    rep = {
        "schema": "relsynth.privacy-report/1",
        "spec_name": s.get("name"),
        "unit": unit.to_dict(),
        "people": {"total": int(len(ids)), "training": int(len(tr_ids)), "control": int(len(ctl_ids)),
                   "reference": int(len(ref_ids))},
        "profile_columns": cols, "quasi_identifiers": qis, "sensitive": sensitive,
        "sensitive_not_found": missing,
        "confidence": level,
        "release": cfg["release"],
        "attacks": res,
        "vulnerable_records": records,
        "guards": guards,
        "timings": {**timings, "total_seconds": time.time() - t_start},
        "settings": {k: cfg[k] for k in ("split", "attacks")},
        "pseudonyms": {"method": "hmac-sha256", "salt_id": salt_id(salt)},
        "_salt": salt.hex(),
    }
    small = cfg.get("small_population")
    if small and len(ids) < small:
        rep["warnings"] = [f"The privacy unit has {len(ids)} people, fewer than {small}: each person weighs "
                           "heavily in the model, and the attacks have little power. Releases outside the "
                           "organisation fail without differential privacy."]
    rep["contexts"] = {name: judge(rep, name, c, small_population=small) for name, c in cfg["contexts"].items()}
    return rep


def _n_test(cfg) -> int:
    """Test shadows: n_test_gated when a context that can pass gates on the
    single most exposed person (an external context without DP is blocked
    anyway, so the extra fits would decide nothing)."""
    sh = cfg["attacks"]["membership"]["shadow"]
    dp = (cfg.get("release") or {}).get("dp")
    gated = any(c.get("per_person", "gate") == "gate" and (dp or not c.get("external"))
                for c in cfg["contexts"].values())
    return int(max(sh["n_test"], sh.get("n_test_gated") or 0)) if gated else int(sh["n_test"])


def sign_off(rep: dict, by: str | None) -> dict:
    """Record who signed off the "review" verdicts that need it."""
    for c in rep["contexts"].values():
        so = c.get("sign_off")
        if so and so["required"] and by:
            so["by"] = by
    return rep


def gate_verdict(ctx: dict) -> str:
    """The verdict a CI gate sees: blocked counts as fail, and a review that
    needs a sign-off counts as fail until someone signed it off."""
    v = ctx["verdict"]
    if v == "blocked":
        return "fail"
    so = ctx.get("sign_off") or {}
    if v == "review" and so.get("required") and not so.get("by"):
        return "fail"
    return v


def _pseudonym(pid, salt: bytes) -> str:
    """A report never shows a real key: the person is named by a keyed hash
    that only the holder of the salt can recompute from the key list."""
    return "P-" + hmac.new(salt, str(pid).encode(), hashlib.sha256).hexdigest()[:10]


def salt_id(salt: bytes) -> str:
    """Names a salt without revealing it, so a report can say which one labels it."""
    return hashlib.sha256(b"relsynth-privacy-salt:" + salt).hexdigest()[:16]


def _guards(spec, unit, train_db, syn_db) -> dict:
    """relsynth's existing distance-to-closest-record and exact-match guards on
    the unit table, for calibration against the attacks."""
    from ..evaluate import MODELLED, dcr, determining_columns
    ts = spec["tables"][unit.table]
    keycols = {r["child_key"] for r in spec["relationships"] if r["child"] == unit.table and not r.get("composite")}
    cols = [c for c, cs in ts["columns"].items() if cs["type"] in MODELLED and c not in keycols
            and c != ts.get("primary_key")]
    real, syn = train_db[unit.table], syn_db[unit.table]
    pcols = determining_columns(real, ts, cols)
    if len(pcols) < 1 or len(real) < 10:
        return {"dcr": None, "note": "too few columns or rows for the distance guard"}
    d = dcr(real, syn, ts, pcols)
    return {"dcr": d, "dcr_ratio_limit": 0.5, "exact_match_limit": 0.01}


# ------------------------------------------------------------ judgement
def _status(upper, lower, limit):
    if upper is None or limit is None or not np.isfinite(upper):
        return "n/a"
    if upper <= limit:
        return "pass"
    if lower is not None and lower > limit:
        return "fail"
    return "review"


def judge(rep: dict, name: str, ctx: dict, *, small_population: int | None = None) -> dict:
    """Checks of one release context: each attack's upper confidence bound
    against the context's limit (EDPB para. 88: safety margins).  fail when even
    the lower bound is above the limit, review when the limit lies inside the
    interval."""
    th = ctx["thresholds"]
    knows = ctx.get("attacker_knows", "all")
    A = rep["attacks"]
    checks = []

    def add(criterion, attack, value, ci, limit, what):
        checks.append({"criterion": criterion, "attack": attack, "value": value, "ci": ci, "limit": limit,
                       "status": _status(ci[1] if ci else value, ci[0] if ci else value, limit), "what": what})
    for r in A["record_isolation"]:
        add("record_isolation", r["attack"], r["value"], r["ci"], th["risk"],
            "share of people the synthetic data lets an attacker single out, above unseen people")
    for r in A["linkage"]:
        add("linkage", r["attack"], r["value"], r["ci"], th["risk"],
            "share of people whose two partial records the synthetic data links, above unseen people")
    for r in A["inference"]:
        if r.get("attacker_knows") != ("quasi_identifiers" if knows == "quasi_identifiers" else "all"):
            continue
        add("inference", f"{r['attack']}:{r['attribute']}", r["value"], r["ci"], th["risk"],
            f"share of people whose {r['attribute']} is guessed from the synthetic data, above unseen people")
    for r in A["attribute_inference"]:
        if r.get("attacker_knows") != ("quasi_identifiers" if knows == "quasi_identifiers" else "all"):
            continue
        sl = r["specific_leakage"]
        add("inference", f"{r['attack']}:{r['attribute']}", sl["value"], sl["ci"], th["attribute_leakage"],
            f"accuracy on {r['attribute']} gained from the synthetic data beyond population data, members minus "
            "non-members")
    M = A["membership"]
    if "domias" in M and M["domias"].get("identified"):
        d = M["domias"]
        add("membership", "domias_auc", d["auc"], d["auc_ci"], th["mia_auc"],
            "how well the release separates training people from unseen people (0.5 = not at all)")
        add_tpr(checks, "domias", d, th)
    if "shadow" in M:
        for ro, p in M["shadow"]["pooled"].items():
            add("membership", f"shadow_{ro}_auc", p["auc"], p["auc_ci"], th["mia_auc"],
                "how well an attacker who knows the generator tells if a target was in the data")
            add_tpr(checks, f"shadow_{ro}", p, th)
    per_person = ctx.get("per_person", "gate")
    if rep["vulnerable_records"]:
        top = rep["vulnerable_records"][0]
        ci = top.get("mia_auc_ci") or [top["mia_auc"]] * 2
        add("membership", "most_vulnerable_record_auc", top["mia_auc"], ci, th["record_auc"],
            "membership attack on the single most exposed person (EDPB para. 87: one person is enough)")
        if per_person == "info":
            # one person's interval is too wide to decide at the default shadow
            # count: shown, and the pooled attack on the vulnerable targets gates
            checks[-1]["status"] = "info"
    if per_person == "info" and "shadow" in M:
        for ro, p in M["shadow"]["pooled"].items():
            v = p.get("vulnerable")
            if v:
                add("membership", f"shadow_{ro}_vulnerable_auc", v["auc"], v["auc_ci"], th["mia_auc"],
                    "membership attack pooled over the most exposed people")
    dp = (rep.get("release") or {}).get("dp")
    n_people = (rep.get("people") or {}).get("total")
    if small_population and n_people is not None and n_people < small_population and ctx.get("external") \
            and not dp:
        checks.append({"criterion": "population", "attack": "small_population", "value": float(n_people),
                       "ci": None, "limit": float(small_population), "status": "fail",
                       "what": "people in the privacy unit; below the limit an external release needs DP"})
    blocked = bool(ctx.get("external")) and not dp
    st = [c["status"] for c in checks]
    verdict = "blocked" if blocked else "fail" if "fail" in st else "review" if "review" in st else "pass"
    out = {"description": ctx.get("description"), "external": bool(ctx.get("external")), "attacker_knows": knows,
           "thresholds": th, "verdict": verdict, "checks": checks,
           "dp_required": bool(ctx.get("external")), "dp": dp, "per_person": per_person,
           "sign_off": {"required": verdict == "review" and bool(ctx.get("sign_off", True)), "by": None}}
    if blocked:
        out["blocked_reason"] = ("This context is outside the organisation. Differential privacy is required for "
                                 "such releases and this release has none. The empirical results below are "
                                 "shown for information; they cannot replace the DP guarantee.")
    if dp:
        from .stats import dp_tpr_bound
        out["dp_bound_tpr_at_fpr_0.01"] = dp_tpr_bound(0.01, dp["epsilon"], dp.get("delta", 0.0))
        # releases from one domain compose: the bound for everything released so far
        if dp.get("cumulative_epsilon") is not None:
            out["dp_bound_tpr_at_fpr_0.01_cumulative"] = dp_tpr_bound(0.01, dp["cumulative_epsilon"],
                                                                     dp.get("cumulative_delta", dp.get("delta", 0.0)))
    return out


def add_tpr(checks, name, d, th):
    for e in d.get("tpr_at_fpr", []):
        if e.get("identified") and abs(e["fpr_target"] - 0.01) < 1e-12:
            checks.append({"criterion": "membership", "attack": f"{name}_tpr_at_fpr_0.01", "value": e["tpr"],
                           "ci": e["tpr_ci"], "limit": th["mia_tpr_at_fpr_0.01"],
                           "status": _status(e["tpr_ci"][1], e["tpr_ci"][0], th["mia_tpr_at_fpr_0.01"]),
                           "what": "share of training people caught when only 1% of unseen people are wrongly "
                                   "accused (random guessing: 0.01)"})

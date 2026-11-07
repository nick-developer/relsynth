# SPDX-License-Identifier: Apache-2.0
"""The quick privacy check of `relsynth run`: two membership tests on a proxy
release, in seconds (G5; T06 decision 4): DOMIAS and a population-calibrated
closest-distance attack.

    people (at most evaluation.privacy_check_people) ──┬─ members    (fit the proxy release on them)
                                                       ├─ non-members (never seen)
                                                       └─ reference  (the attacker's population data)

The written release is fitted on every person, so it has no unseen people to
compare with.  The check therefore fits the same generator, with the same
settings, on a third of a sample of people and asks whether DOMIAS (van Breugel
et al. 2023) or the distance to the closest synthetic profile tells those
members from people the proxy never saw (Prop. 1740).
It runs after every synthetic file is written and draws only from its own
seeded streams, so the release is the same with or without it (Prop. 1741).

It is a smoke test, not an assessment: two attacks, one internal-context limit,
a proxy fitted on fewer people.  `relsynth privacy` is the assessment (every
criterion, shadow models, every release context).  Only aggregates leave this
module: counts, the AUC with its interval, the verdict.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from . import mia, unit as U
from .config import DEFAULTS

# the internal context's limits (relsynth.privacy.config): the quick check never judges an external release
LIMITS = {"mia_auc": DEFAULTS["contexts"]["internal"]["thresholds"]["mia_auc"],
          "mia_tpr_at_fpr_0.01": DEFAULTS["contexts"]["internal"]["thresholds"]["mia_tpr_at_fpr_0.01"]}
MIN_GROUP = 20          # non-members and reference people each (as in assess)
MIN_MEMBERS = 10        # people the proxy is fitted on
SEED_OFFSET = 7417      # the check's own stream: never the release's seed itself


def _status(ci, limit) -> str:
    lo, hi = ci
    if hi <= limit:
        return "pass"
    if lo > limit:
        return "fail"
    return "review"


def _skipped(reason: str, t0: float, **extra) -> dict:
    return {"status": "skipped", "reason": reason, "seconds": round(time.time() - t0, 2), **extra}


def _pushed(unit: U.Unit, stats: dict, tables: dict) -> list[str]:
    """Tables of the unit that were fitted from in-database statistics and a
    sample: a sample does not hold every row of each person."""
    out = []
    for t in [unit.table, *unit.owned]:
        st = (stats or {}).get(t)
        n = getattr(st, "n_rows", None)
        if n is not None and t in tables and int(n) > len(tables[t]):
            out.append(t)
    return out


def proxy_spec(spec: dict, unit: U.Unit) -> dict:
    """The spec cut down to what a person's profile can depend on: the unit,
    the tables it owns, and every table those reach through foreign keys
    (their parents, recursively).  Tables outside that set (a currency rate
    table, facts of other parties) are not fitted by the proxy, which keeps the
    check in seconds; their effect on the unit's model through shared parents
    is the approximation the proxy makes (docs/privacy.md)."""
    import copy
    names = set(spec["tables"])

    def named(x) -> set:
        """Table names a table's settings mention (lookups, computed columns, scopes)."""
        if isinstance(x, str):
            return {x} & names
        if isinstance(x, dict):
            return set().union(*(named(v) for v in x.values())) if x else set()
        if isinstance(x, (list, tuple)):
            return set().union(*(named(v) for v in x)) if x else set()
        return set()
    keep = {unit.table, *unit.owned}
    rels = spec.get("relationships") or []
    grew = True
    while grew:
        before = len(keep)
        keep |= {r["parent"] for r in rels if r["child"] in keep and r["parent"] in names}
        for t in list(keep):
            keep |= named(spec["tables"][t])
        grew = len(keep) > before
    if keep >= set(spec["tables"]):
        return spec
    out = copy.deepcopy({k: v for k, v in spec.items() if k not in ("tables", "relationships", "rules")})
    out["tables"] = {t: copy.deepcopy(ts) for t, ts in spec["tables"].items() if t in keep}
    out["relationships"] = [copy.deepcopy(r) for r in rels if r["child"] in keep and r["parent"] in keep]
    out["rules"] = [copy.deepcopy(r) for r in spec.get("rules") or [] if r.get("table") in keep]
    return out


def quick_check(spec, tables: dict, *, seed: int | None = None, people: int | None = None,
                unit: str | None = None, stats: dict | None = None, generator=None, level: float = 0.95) -> dict:
    """DOMIAS and the closest-distance attack on a proxy release of `spec`'s generator, fitted on a third of up
    to `people` people of `tables` (the source tables as the run loaded them).

    Returns {"status": pass|review|fail|skipped|error, ...}: status compares the
    AUC's upper confidence bound (and the TPR at FPR 0.01, when identified)
    with the internal context's limits; skipped and error carry a reason and
    never stop a run.  generator: a fit-and-sample function (tables, seed) ->
    tables, default relsynth_generator(spec)."""
    from .assess import _quiet, relsynth_generator
    t0 = time.time()
    s = spec
    ecfg = (s.get("synthesis") or {}).get("evaluation") or {}
    people = int(people if people is not None else ecfg.get("privacy_check_people", 3000))
    seed = int((s.get("synthesis") or {}).get("seed", 0) if seed is None else seed)
    try:
        u = U.resolve(s, unit)
    except ValueError as e:
        return _skipped(str(e), t0)
    if u.table not in tables or not len(tables[u.table]):
        return _skipped(f"the unit table {u.table} was not loaded", t0, unit=u.table)
    pushed = _pushed(u, stats or {}, tables)
    if pushed:
        return _skipped("fitted from in-database statistics and a sample (" + ", ".join(pushed) + "); run "
                        "`relsynth privacy` for this source", t0, unit=u.table)
    ids = pd.Index(tables[u.table][u.key].dropna().unique())
    n_total = len(ids)
    need = 2 * MIN_GROUP + MIN_MEMBERS
    if n_total < need:
        return _skipped(f"{n_total} people in {u.table}: at least {need} are needed", t0,
                        unit=u.table, people={"total": int(n_total)})
    rng = np.random.default_rng([seed, SEED_OFFSET])
    ids = ids[rng.permutation(n_total)[:max(people, need)]]
    n = len(ids)
    n_c = n_r = max(MIN_GROUP, n // 3)          # thirds; small units keep 20 unseen and 20 reference people
    ctl, ref, mem = ids[:n_c], ids[n_c:n_c + n_r], ids[n_c + n_r:]
    out: dict[str, Any] = {"unit": u.table, "context": "internal", "limits": dict(LIMITS),
           "people": {"total": int(n_total), "members": int(len(mem)), "non_members": int(len(ctl)),
                      "reference": int(len(ref))}}
    fit_seed = int(rng.integers(2**31 - 1))
    try:
        sample_db = U.subset(u, tables, ids)
        train_db = U.subset(u, sample_db, mem)
        if generator is not None:
            syn = _quiet(generator, train_db, fit_seed)
        else:
            ps = proxy_spec(s, u)
            try:
                syn = _quiet(relsynth_generator(ps), {t: train_db[t] for t in ps["tables"]}, fit_seed)
                out["proxy_tables"] = sorted(ps["tables"])
            except Exception:  # noqa: BLE001 - a cut-down spec the engine refuses: fit the whole one
                if ps is s:
                    raise
                syn = _quiet(relsynth_generator(s), train_db, fit_seed)
        P = U.profiles(s, u, sample_db, dedupe=True)
        kinds = dict(P.attrs.get("kinds", {}))
        P_syn = U.profiles(s, u, syn, kinds=kinds)
        cols = [c for c in P.columns if c in P_syn.columns]
        if not cols:
            return _skipped("no profile column to compare", t0, **out)
        P, P_syn = P[cols], P_syn[cols].reset_index(drop=True)
        args = (P.loc[mem], P.loc[ctl], P_syn, P.loc[ref])
        attacks = {"domias": mia.domias(*args, level=level, seed=int(rng.integers(2**31 - 1))),
                   "closest_distance": closest_distance(*args, level=level, seed=int(rng.integers(2**31 - 1)))}
    except Exception as e:  # noqa: BLE001 - a quick check that cannot run is reported, never fatal
        return {**out, "status": "error", "reason": repr(e)[:500], "seconds": round(time.time() - t0, 2)}
    checks: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for name, d in attacks.items():
        d.pop("_member_scores", None)
        if not d.get("identified"):
            summary[name] = {"identified": False, "reason": d.get("reason")}
            continue
        ci = [float(x) for x in d["auc_ci"]]
        summary[name] = {"identified": True, "auc": float(d["auc"]), "auc_ci": ci}
        checks.append({"attack": name, "metric": "auc", "value": float(d["auc"]), "ci": ci,
                       "limit": LIMITS["mia_auc"], "status": _status(ci, LIMITS["mia_auc"])})
        tpr = next((e for e in d["tpr_at_fpr"] if abs(e["fpr_target"] - 0.01) < 1e-12 and e.get("identified")),
                   None)
        if tpr is not None:
            summary[name]["tpr_at_fpr_0.01"] = float(tpr["tpr"])
            checks.append({"attack": name, "metric": "tpr_at_fpr_0.01", "value": float(tpr["tpr"]),
                           "ci": [float(x) for x in tpr["tpr_ci"]], "limit": LIMITS["mia_tpr_at_fpr_0.01"],
                           "status": _status(tpr["tpr_ci"], LIMITS["mia_tpr_at_fpr_0.01"])})
    if not checks:
        return _skipped("; ".join(f"{k}: {v['reason']}" for k, v in summary.items()), t0, **out)
    st = [c["status"] for c in checks]
    # the headline is the attack whose AUC interval reaches highest
    worst = max((c for c in checks if c["metric"] == "auc"), key=lambda c: c["ci"][1])
    out.update({"status": "fail" if "fail" in st else "review" if "review" in st else "pass",
                "attack": worst["attack"], "auc": worst["value"], "auc_ci": worst["ci"], "attacks": summary,
                "checks": checks, "confidence": level, "seconds": round(time.time() - t0, 2)})
    small = DEFAULTS["small_population"]
    if n_total < small:
        out["warning"] = (f"{n_total} people, fewer than {small}: the test has little power, and a release "
                          "outside the organisation needs differential privacy")
    return out


def closest_distance(members: pd.DataFrame, non_members: pd.DataFrame, syn: pd.DataFrame, ref: pd.DataFrame, *,
                     level: float = 0.95, seed: int = 0) -> dict:
    """Distance-to-closest-record membership attack, calibrated on the
    population: score = d_ref(x) / d_syn(x), the Gower distance from the person
    to the closest reference person over that to the closest synthetic profile
    (large = the release holds someone unusually like x).  It catches copied
    and near-copied people that a kernel density smooths away (DOMIAS on one
    table of 1000 copied people: AUC 0.62); Prop. 1740 applies to it as to
    DOMIAS, since the score depends on the person, the release and the
    reference only."""
    from .encode import Encoder, column_kinds, kneighbors
    from .stats import auc_delong, auc_interval, tpr_at_fpr
    cols = list(ref.columns)
    enc = Encoder([ref, syn], cols, column_kinds([ref, syn], cols))
    rng = np.random.default_rng(seed)
    q = pd.concat([members, non_members])
    _, ds = kneighbors(enc, q, syn, 1, rng=rng)
    _, dr = kneighbors(enc, q, ref, 2, rng=rng)
    # a reference person is never their own neighbour (members and non-members are not in ref)
    score = np.log(dr[:, 0] + 1e-9) - np.log(ds[:, 0] + 1e-9)
    y = np.r_[np.ones(len(members)), np.zeros(len(non_members))]
    s_m, s_n = score[: len(members)], score[len(members):]
    im, inn = rng.permutation(len(s_m)), rng.permutation(len(s_n))
    hm, hn = len(im) // 2, len(inn) // 2
    sc, yc = np.r_[s_m[im[:hm]], s_n[inn[:hn]]], np.r_[np.ones(hm), np.zeros(hn)]
    st = np.r_[s_m[im[hm:]], s_n[inn[hn:]]]
    yt = np.r_[np.ones(len(im) - hm), np.zeros(len(inn) - hn)]
    a, se = auc_delong(score, y)
    return {"attack": "closest_distance", "identified": True, "auc": a, "auc_ci": auc_interval(a, se, level),
            "tpr_at_fpr": [tpr_at_fpr(sc, yc, st, yt, f, level) for f in mia.FPRS],
            "n_members": int(len(s_m)), "n_non_members": int(len(s_n))}


def line(res: dict) -> str:
    """The one-line summary `relsynth run` prints."""
    if res is None:
        return "privacy check: off"
    st = str(res.get("status"))
    if st in ("skipped", "error"):
        return f"privacy check: {st} ({res.get('reason')})"
    p = res["people"]
    names = {"domias": "DOMIAS", "closest_distance": "distance"}
    parts = [f"{names.get(k, k)} AUC {v['auc']:.3f} [{v['auc_ci'][0]:.3f}, {v['auc_ci'][1]:.3f}]"
             for k, v in (res.get("attacks") or {}).items() if v.get("identified")]
    txt = (f"privacy check: {st.upper()} - membership {', '.join(parts)} vs limit {res['limits']['mia_auc']:.2f} "
           f"(internal), proxy fitted on {p['members']} of {p['total']} people in {res['unit']}, "
           f"{res['seconds']:.1f} s")
    if res.get("warning"):
        txt += f"; only {p['total']} people, so the test has little power"
    if st != "pass":
        txt += "; run `relsynth privacy` for the full assessment"
    return txt

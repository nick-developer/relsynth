# SPDX-License-Identifier: Apache-2.0
"""The self-tuning loop (docs/autotune.md; research: docs/research/self-tuning.md).

    result = autotune("spec.yaml", "objectives.yaml")
    result.met, result.delta, result.patches, result.report_markdown

Each round judges the incumbent on the search seeds, proposes moves (doctor
spec fixes, guided moves, then model-based ones), runs each candidate on one
seed first (successive halving) and on every search seed only when it does
not do worse there, and accepts it when it is better on the same seeds
(analysis.better).  It stops when every objective is met, nothing is left to
propose, or the budget is spent; then it confirms the final settings on the
held-out seeds.  Every run is kept in the history (aggregates only), so a
second call reuses the runs already made and continues from the incumbent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import analysis as A
from . import search as S
from .checks import run_trial
from .history import History
from .objectives import Objectives, load

log = logging.getLogger("relsynth.autotune")


@dataclass
class Candidate:
    delta: dict
    patches: dict
    origin: str
    parent: str | None = None

    @property
    def key(self) -> str:
        return S.key_of(self.delta, self.patches)

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.delta.items())]
        parts += [f"spec {k}={v}" for k, v in sorted(self.patches.items())]
        return ", ".join(parts) or "the spec as given"


@dataclass
class AutotuneResult:
    met: bool
    confirmed: bool | None
    delta: dict
    patches: dict
    spec: dict                         # the tuned raw spec
    judgement: dict                    # incumbent on the search seeds
    holdout: dict | None
    trajectory: list
    analysis: dict
    runs: int
    runs_reused: int
    seconds: float
    objectives: dict
    stopped: str
    report_markdown: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("report_markdown")
        d.pop("spec")
        return d


def fingerprint(raw: dict, obj: Objectives) -> str:
    """Spec (without its seed) and the objectives that judge it: runs of one
    fingerprint are comparable."""
    s = copy.deepcopy(raw)
    (s.get("synthesis") or {}).pop("seed", None)
    blob = json.dumps({"spec": s, "obj": {k: v for k, v in obj.to_dict().items() if k not in ("budget", "seeds")}},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _refuse_dp(raw):
    from ..dp.config import section
    from ..spec import normalise
    if section(normalise(copy.deepcopy(raw))) is not None:
        from ..errors import PrivacyError
        raise PrivacyError("autotune reads the real data many times to choose settings, which spends privacy budget "
                           "outside the DP accountant (Papernot and Steinke 2022): it is refused in DP mode. Tune a "
                           "non-DP copy on public or synthetic data, or tune with DP off and release with DP on.")


def autotune(spec, objectives=None, *, history_dir=None, runs: int | None = None, progress=None,
             resume: bool = True) -> AutotuneResult:
    """Tune the engine settings of a spec toward the objectives."""
    from ..spec import from_ir, read_yaml
    t0 = time.time()
    say = progress or (lambda m: log.info(m))
    if isinstance(spec, (str, Path)):
        raw, base = from_ir(read_yaml(spec)), Path(spec).parent
        stem = Path(spec).with_suffix("")
    else:
        raw = copy.deepcopy(getattr(spec, "raw", spec))
        base = Path(getattr(spec, "base_dir", raw.get("_base_dir", ".")))
        stem = base / str(raw.get("name") or "spec")
    raw = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    _refuse_dp(raw)
    obj = objectives if isinstance(objectives, Objectives) else load(objectives)
    budget = int(runs or obj.runs)
    hist = History(Path(history_dir) if history_dir else Path(f"{stem}.tune"), fingerprint(raw, obj))
    if not resume:
        hist.clear()
    reused0 = 0
    made = 0
    deadline = t0 + obj.hours * 3600 if obj.hours else None

    # runs kept for the confirmation on the held-out seeds (tuned and defaults)
    reserve = 2 * len(obj.seeds_holdout)
    confirming = False

    def budget_left():
        cap = budget if confirming else budget - reserve
        return made < cap and (deadline is None or confirming or time.time() < deadline)

    def run(c: Candidate, seeds) -> list:
        nonlocal made, reused0
        out = []
        for s in seeds:
            rec = hist.get(c.key, s)
            if rec is None:
                if not budget_left():
                    break
                say(f"run {made + 1}/{budget}: seed {s}, {c.describe()}")
                tr = run_trial(raw, str(base), s, c.delta, c.patches, obj)
                made += 1
                rec = hist.add(c.key, c.delta, c.patches, tr.to_dict(), origin=c.origin)
            else:
                reused0 += 1
            out.append(rec)
        return out

    seeds = obj.seeds_search
    inc = Candidate({}, {}, "baseline")
    last = hist.incumbent()
    if resume and last:
        inc = Candidate(last["delta"], last["patches"], "resumed")
    trials = run(inc, seeds)
    if len(trials) < len(seeds):
        from ..spec import SpecError
        raise SpecError(f"the budget ({budget} runs, {reserve} kept for the held-out seeds) does not cover one "
                        f"evaluation of the starting settings on {len(seeds)} search seeds")
    J = A.judge(trials, obj)
    base_J = J if inc.origin == "baseline" else A.judge(run(Candidate({}, {}, "baseline"), seeds), obj)
    trajectory = [{"step": 0, "change": inc.describe(), "origin": inc.origin, "accepted": True,
                   "excess": round(J.excess, 2), "hard": J.hard_violations, "seeds": J.seeds,
                   "why": "starting point"}]
    hist.set_incumbent(inc.key, inc.delta, inc.patches)
    tried = set(hist.keys())
    rng = np.random.default_rng(int(seeds[0]))
    space = S.space_for(obj)
    stopped = "budget"
    stale = 0
    accel: dict = {}            # (setting, direction) -> accepted moves in a row (pattern-search expansion)
    while True:
        if J.all_met(obj):
            stopped = "all objectives met"
            break
        if not budget_left():
            stopped = "budget"
            break
        fam = J.families()
        failing = {f: o["excess"] for f, o in fam.items() if o["targeted"] and not o["met"]}
        sysf = A.systematic(trials)
        for cid in sysf:
            from .checks import family
            f = family(cid)
            if f in failing:
                failing[f] += 0.1          # a systematic failure counts a little more than its excess
        violated: dict = {}
        cons = J.constraints(obj.hard)
        for o in cons.values():
            if not o["met"]:
                for cid in o["checks"]:
                    from .checks import family
                    violated[family(cid)] = violated.get(family(cid), 0) + 10.0
        props: list[Candidate] = []
        if obj.spec_patches:
            doc: dict = {}
            for t in trials:
                doc.update(t.get("doctor") or {})
            new = {k: v for k, v in doc.items() if inc.patches.get(k) != v}
            if new:
                props.append(Candidate(dict(inc.delta), {**inc.patches, **new}, "doctor", inc.key))
        base_syn = raw.get("synthesis") or {}
        from .report import effects as _effects
        learned = _effects(hist, obj)[0]
        for d in S.guided(failing, violated, inc.delta, base_syn, space, learned, accel):
            props.append(Candidate(d, dict(inc.patches), "guided", inc.key))
        if stale >= 1 or not props or all(p.key in tried for p in props):
            hist_scores = [{"delta": r["delta"], "score": r["score"]} for r in hist.scores(obj)]
            for d in S.model_based(hist_scores, inc.delta, base_syn, space, rng):
                props.append(Candidate(d, dict(inc.patches), "model", inc.key))
        props = [p for p in props if p.key not in tried]
        if not props:
            stopped = "nothing left to propose"
            break
        accepted = False
        for c in props:
            if not budget_left():
                break
            tried.add(c.key)
            first = run(c, seeds[:1]) if obj.halving and len(seeds) > 1 else run(c, seeds)
            if not first:
                break
            J1 = A.judge(first, obj)
            Jinc1 = A.judge([t for t in trials if t["seed"] in J1.seeds], obj)
            if obj.halving and len(seeds) > 1:
                ok1, why1 = _not_worse(J1, Jinc1)
                if not ok1:
                    _shrink(accel, c, inc)
                    trajectory.append({"step": len(trajectory), "change": c.describe(), "origin": c.origin,
                                       "accepted": False, "excess": round(J1.excess, 2), "hard": J1.hard_violations,
                                       "seeds": J1.seeds, "why": f"stopped after one seed: {why1}"})
                    continue
                rest = run(c, seeds[1:])
                if len(rest) < len(seeds) - 1:
                    break
                cand_trials = first + rest
            else:
                cand_trials = first
            Jc = A.judge(cand_trials, obj)
            ok, why = A.better(Jc, J)
            _shrink(accel, c, inc)
            trajectory.append({"step": len(trajectory), "change": c.describe(), "origin": c.origin,
                               "accepted": ok, "excess": round(Jc.excess, 2), "hard": Jc.hard_violations,
                               "seeds": Jc.seeds, "why": why})
            if ok:
                moved = [k for k in set(c.delta) | set(inc.delta) if c.delta.get(k) != inc.delta.get(k)]
                if c.origin == "guided" and len(moved) == 1:
                    k = moved[0]
                    syn0: dict = raw.get("synthesis") or {}
                    old, newv = S._current(S.params()[k], inc.delta, syn0), c.delta.get(k)
                    if isinstance(old, (int, float)) and isinstance(newv, (int, float)) and not isinstance(newv, bool):
                        sg = 1 if newv > old else -1
                        prev = accel.get((k, sg), 0)
                        accel = {kk: v for kk, v in accel.items() if kk[0] != k}
                        accel[(k, sg)] = min(prev + 1, 4)
                else:
                    accel = {}
                inc, trials, J = c, cand_trials, Jc
                hist.set_incumbent(c.key, c.delta, c.patches)
                accepted = True
                stale = 0
                say(f"accepted: {c.describe()} ({why})")
                break
        if not accepted:
            stale += 1
            if stale >= 3:
                stopped = "no improvement in three rounds"
                break
    # confirmation on held-out seeds
    hold = None
    confirmed = None
    confirming = True
    if obj.seeds_holdout:
        ht = run(inc, obj.seeds_holdout)
        hb = run(Candidate({}, {}, "baseline"), obj.seeds_holdout) if (inc.delta or inc.patches) else ht
        if len(ht) == len(obj.seeds_holdout) and len(hb) == len(obj.seeds_holdout):
            Jh, Jhb = A.judge(ht, obj), A.judge(hb, obj)
            if inc.delta or inc.patches:
                confirmed = Jh.hard_violations <= Jhb.hard_violations and Jh.excess <= Jhb.excess
            else:
                confirmed = True
            hold = {"tuned": _summary(Jh, obj), "baseline": _summary(Jhb, obj), "confirmed": confirmed}
    from .report import analyse, render
    an = analyse(hist, obj, inc, J, base_J, trials, space)
    from ..spec import deep_merge
    tuned = copy.deepcopy(raw)
    if inc.patches:
        from ..doctor import apply_patches
        tuned = apply_patches(tuned, inc.patches)
    from .checks import _nest
    tuned["synthesis"] = deep_merge(tuned.get("synthesis") or {}, _nest(inc.delta))
    res = AutotuneResult(met=J.all_met(obj), confirmed=confirmed, delta=inc.delta, patches=inc.patches, spec=tuned,
                         judgement=_summary(J, obj), holdout=hold, trajectory=trajectory, analysis=an, runs=made,
                         runs_reused=reused0, seconds=round(time.time() - t0, 1), objectives=obj.to_dict(),
                         stopped=stopped, extra={"baseline": _summary(base_J, obj), "history": str(hist.path)})
    res.report_markdown = render(res)
    return res


def _shrink(accel: dict, c: Candidate, inc: Candidate) -> None:
    """A rejected expansion step falls back to the unit step (the contraction of a pattern search)."""
    if c.origin != "guided":
        return
    for k in set(c.delta) | set(inc.delta):
        if c.delta.get(k) != inc.delta.get(k):
            for key in [kk for kk in accel if kk[0] == k]:
                accel[key] = 0


def _not_worse(new: A.Judgement, old: A.Judgement) -> tuple[bool, str]:
    if new.hard_violations > old.hard_violations:
        return False, f"breaks {new.hard_violations - old.hard_violations} more hard check(s)"
    if new.hard_violations < old.hard_violations:
        return True, "fixes hard checks"
    if new.excess > old.excess:
        return False, f"excess failures {old.excess:.1f} -> {new.excess:.1f}"
    return True, "not worse"


def _summary(J: A.Judgement, obj) -> dict:
    fam = J.families()
    return {"seeds": J.seeds, "hard_violations": J.hard_violations, "excess_failures": round(J.excess, 3),
            "failures": round(J.failures, 3), "min_scorecard_pass_rate": round(J.min_pass_rate, 4),
            "constraints": J.constraints(obj.hard), "families_not_met": sorted(f for f, o in fam.items()
                                                                                if o["targeted"] and not o["met"]),
            "families": fam, "undecided": J.undecided, "all_met": J.all_met(obj)}

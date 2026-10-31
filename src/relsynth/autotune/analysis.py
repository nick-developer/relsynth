# SPDX-License-Identifier: Apache-2.0
"""Judging a candidate across seeds (docs/research/self-tuning.md, sec. 3.3-3.4).

- Hard constraints hold exactly: a candidate meets one when no check of it
  fails on any seed.  Seed evidence is reported as the Beta(1 + k, 1 + n - k)
  posterior of the violation rate: 0 violations in n seeds bound the rate by
  1 - 0.05^(1/(n+1)) (about 3/n) at 95%.
- Statistical checks are calibrated (user decision of 2026-10-31): the failures
  F of a family on one seed are compared with their null count, Poisson with
  mean mu = sum of the checks' levels alpha; the family is met when
  F <= q, the null_quantile quantile.  A check without a level contributes 0
  to mu, so a family of such checks is held exactly.
- A check fails *systematically* when its failures across seeds are unlikely
  under its level: P(Binomial(n, alpha) >= k) < 0.01 (alpha 0.05 when the
  check has none).  Systematic failures are what the tuner moves parameters
  for; sporadic ones are noise and are left alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy import stats

from .checks import constraint_of, family

SYSTEMATIC_P = 0.01
DEFAULT_ALPHA = 0.05


def beta_upper(k: int, n: int, level: float = 0.95) -> float:
    """Upper credible bound of a violation rate after k violations in n seeds (uniform prior)."""
    return float(stats.beta.ppf(level, 1 + k, 1 + n - k)) if n else 1.0


def seeds_needed(delta: float, level: float = 0.95) -> int:
    """Seeds without a violation needed to bound the violation rate by delta."""
    return max(1, math.ceil(math.log(1 - level) / math.log(1 - delta)) - 1)


def null_quantile(mu: float, q: float) -> int:
    return int(stats.poisson.ppf(q, mu)) if mu > 0 else 0


@dataclass
class SeedView:
    """One seed's checks of a candidate, judged against the objectives."""
    seed: int
    hard: dict = field(default_factory=dict)            # constraint -> [failed check ids]
    undecided: list = field(default_factory=list)       # privacy checks in review
    families: dict = field(default_factory=dict)        # family -> {"failed", "mu", "q", "n"}
    pass_rate: float = 1.0
    error: str | None = None
    seen: dict = field(default_factory=dict)            # constraint -> checks evaluated

    @property
    def hard_violations(self) -> int:
        return sum(len(v) for v in self.hard.values())

    def excess(self, targeted=None) -> float:
        return float(sum(max(0, f["failed"] - f["q"]) for k, f in self.families.items()
                         if targeted is None or k in targeted))

    def failures(self, targeted=None) -> int:
        return sum(f["failed"] for k, f in self.families.items() if targeted is None or k in targeted)


def _match(cid, patterns):
    from fnmatch import fnmatchcase
    return any(fnmatchcase(cid, p) for p in patterns)


def view(trial: dict, obj) -> SeedView:
    """Judge one trial record (checks of one seed)."""
    v = SeedView(seed=trial["seed"], error=trial.get("error"))
    if trial.get("error") and not trial.get("checks"):
        v.hard = {"run": [trial["error"][:200]]}
        return v
    hard = set(obj.hard)
    pats = obj.check_patterns
    sc_total = sc_pass = 0
    fam: dict = {}
    for c in trial.get("checks", []):
        cid, ok = c["id"], c.get("pass")
        if cid.startswith("scorecard:"):
            sc_total += 1
            sc_pass += bool(ok)
        if ok is None:
            continue
        con = constraint_of(cid)
        if pats and _match(cid, pats):
            con = con or "checks"
            v.seen["checks"] = v.seen.get("checks", 0) + 1
            if not ok:
                v.hard.setdefault("checks", []).append(cid)
            continue
        if con is not None and con not in hard and cid.startswith(("privacy:", "domain:")):
            continue                            # attack / domain results without a privacy / domain constraint
        if con is not None and con in hard:
            v.seen[con] = v.seen.get(con, 0) + 1
            if not ok and c.get("status") == "review" and obj.review != "fail":
                if obj.review == "undecided":
                    v.undecided.append(cid)
                continue
            if not ok:
                v.hard.setdefault(con, []).append(cid)
            continue
        f = family(cid)
        d = fam.setdefault(f, {"failed": 0, "mu": 0.0, "n": 0})
        d["n"] += 1
        d["mu"] += float(c.get("alpha") or 0.0)
        d["failed"] += 0 if ok else 1
    for d in fam.values():
        d["mu"] = round(d["mu"], 6)
        d["q"] = null_quantile(d["mu"], obj.null_quantile)
    v.families = fam
    v.pass_rate = sc_pass / sc_total if sc_total else 1.0
    if trial.get("error"):
        v.hard.setdefault("run", []).append(trial["error"][:200])
    return v


@dataclass
class Judgement:
    """A candidate over the seeds it was run on."""
    seeds: list
    views: list
    targeted: set | None

    @property
    def n(self) -> int:
        return len(self.views)

    @property
    def hard_violations(self) -> int:
        return sum(v.hard_violations for v in self.views)

    @property
    def excess(self) -> float:
        return sum(v.excess(self.targeted) for v in self.views) / max(self.n, 1)

    @property
    def failures(self) -> float:
        return sum(v.failures(self.targeted) for v in self.views) / max(self.n, 1)

    def per_seed(self, attr="excess"):
        return {v.seed: (v.excess(self.targeted) if attr == "excess" else v.hard_violations) for v in self.views}

    def constraints(self, names) -> dict:
        """constraint -> {violations (seeds with one), n, upper bound, checks}."""
        out = {}
        for c in list(names) + sorted({k for v in self.views for k in v.hard} - set(names)):
            k = sum(1 for v in self.views if v.hard.get(c))
            ids = sorted({i for v in self.views for i in v.hard.get(c, [])})
            seen = max((v.seen.get(c, 0) for v in self.views), default=0)
            out[c] = {"seeds_violated": k, "seeds": self.n, "met": k == 0, "evaluated": seen > 0 or k > 0,
                      "checks_per_seed": seen, "rate_upper_95": round(beta_upper(k, self.n), 4),
                      "checks": ids[:20], "n_checks": len(ids)}
        return out

    def families(self) -> dict:
        """family -> {met (every seed), failed per seed, mu, q}."""
        out: dict = {}
        for v in self.views:
            for f, d in v.families.items():
                o = out.setdefault(f, {"failed": [], "mu": d["mu"], "q": d["q"], "n_checks": d["n"]})
                o["failed"].append(d["failed"])
        for f, o in out.items():
            o["met"] = all(x <= o["q"] for x in o["failed"])
            o["targeted"] = self.targeted is None or f in self.targeted
            o["excess"] = round(sum(max(0, x - o["q"]) for x in o["failed"]) / max(len(o["failed"]), 1), 3)
        return out

    @property
    def undecided(self) -> list:
        return sorted({i for v in self.views for i in v.undecided})

    @property
    def min_pass_rate(self) -> float:
        return min((v.pass_rate for v in self.views), default=1.0)

    def all_met(self, obj) -> bool:
        if self.hard_violations:
            return False
        if obj.min_pass_rate is not None and self.min_pass_rate < obj.min_pass_rate:
            return False
        return all(o["met"] for o in self.families().values() if o["targeted"])


def judge(trials: list, obj) -> Judgement:
    fams = obj.families
    return Judgement(seeds=[t["seed"] for t in trials], views=[view(t, obj) for t in trials],
                     targeted=None if fams is None else set(fams))


def systematic(trials: list) -> dict:
    """check id -> {fails, n, alpha} of the checks that fail systematically across seeds."""
    by: dict = {}
    for t in trials:
        for c in t.get("checks", []):
            if c.get("pass") is None:
                continue
            d = by.setdefault(c["id"], {"fails": 0, "n": 0, "alpha": c.get("alpha")})
            d["n"] += 1
            d["fails"] += 0 if c["pass"] else 1
    out = {}
    for cid, d in by.items():
        if not d["fails"]:
            continue
        a = float(d["alpha"] or DEFAULT_ALPHA)
        p = float(stats.binom.sf(d["fails"] - 1, d["n"], a))
        if p < SYSTEMATIC_P:
            out[cid] = {**d, "p": p}
    return out


def better(new: Judgement, old: Judgement) -> tuple[bool, str]:
    """Is `new` better than `old` on the same seeds?  Fewer hard violations
    first; then lower excess failures by at least one standard error of the
    paired differences (and at least 0.5 failures per seed)."""
    common = [s for s in new.seeds if s in old.seeds]
    if not common:
        return False, "no common seeds"
    hn = {v.seed: v.hard_violations for v in new.views}
    ho = {v.seed: v.hard_violations for v in old.views}
    dh = sum(hn[s] - ho[s] for s in common)
    if dh > 0:
        return False, f"breaks {dh} more hard check(s)"
    en, eo = new.per_seed(), old.per_seed()
    d = [en[s] - eo[s] for s in common]
    mean = sum(d) / len(d)
    if dh < 0:                       # hard constraints first: never traded for a statistical target
        return True, f"fixes {-dh} hard check(s) (excess failures {mean:+.1f} per seed)"
    if len(d) > 1:
        sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        se = sd / len(d) ** 0.5
    else:
        se = 0.0
    if mean <= -0.5 and mean + se < 0:
        return True, f"{-mean:.1f} fewer excess failures per seed (se {se:.2f})"
    return False, f"excess failures change {mean:+.1f} per seed (se {se:.2f})"

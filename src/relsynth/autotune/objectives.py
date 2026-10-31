# SPDX-License-Identifier: Apache-2.0
"""The user's objectives for `relsynth autotune` (objectives.yaml).

    constraints:                     # hard: must hold on every seed
      - referential_integrity        # every foreign key resolves
      - rules                        # no check rule violated beyond the source's own rate
      - unique_keys
      - trees                        # no cycles
      - privacy                      # no privacy check fails (scorecard DCR / exact matches / text
                                     # leaks, and the attack suite when `privacy.attacks` is on)
      - domain                       # every domain identity of `domain_checks` holds
      - {check: "scorecard:sales|discriminator_auc"}   # any check id (fnmatch pattern)
    targets:                         # statistical: calibrated (failures <= the null quantile)
      - {family: association_max_diff}
      - all                          # every statistical family of the scorecard (the default)
      - {scorecard_pass_rate: 0.98}  # share of passed scorecard items, on every seed
    seeds: {search: [1, 2, 3], holdout: [11, 12]}
    budget: {runs: 40, hours: 6}
    search: auto                     # or a list of setting keys, or {key: {low, high}} / {key: [choices]}
    spec_patches: true               # may apply `relsynth doctor` spec fixes (written to the tuned spec only)
    quality: false                   # also run the 2.0 quality report (its calibrated checks become targets)
    privacy: {attacks: false, config: privacy.yaml, review: undecided}
    domain_checks: checks.py:checks  # function(real, synthetic) -> {name: (real value, synthetic value)}
    null_quantile: 0.99              # calibrated targets: a family passes when failures <= this quantile
                                     # of its null count (Poisson with mean sum of levels)

Every key is optional; the defaults hold every hard constraint and every
statistical family of the scorecard.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from ..spec import SpecError, read_yaml

HARD = ("referential_integrity", "rules", "unique_keys", "trees", "privacy", "domain")
DEFAULTS = {
    "constraints": list(HARD),
    "targets": ["all"],
    "seeds": {"search": [1, 2, 3], "holdout": [11, 12]},
    "budget": {"runs": 40, "hours": None},
    "search": "auto",
    "spec_patches": True,
    "quality": False,
    "privacy": {"attacks": False, "config": None, "review": "undecided"},
    "domain_checks": None,
    "null_quantile": 0.99,
    "halving": True,
}


class ObjectivesError(SpecError):
    """Invalid objectives (CLI exit code 2)."""


@dataclass
class Objectives:
    constraints: list = field(default_factory=lambda: list(HARD))
    targets: list = field(default_factory=lambda: ["all"])
    seeds_search: list = field(default_factory=lambda: [1, 2, 3])
    seeds_holdout: list = field(default_factory=lambda: [11, 12])
    runs: int = 40
    hours: float | None = None
    search: object = "auto"
    spec_patches: bool = True
    quality: bool = False
    privacy_attacks: bool = False
    privacy_config: str | None = None
    review: str = "undecided"
    domain_checks: str | None = None
    null_quantile: float = 0.99
    halving: bool = True
    base_dir: str = "."

    def to_dict(self) -> dict:
        return {"constraints": self.constraints, "targets": self.targets,
                "seeds": {"search": self.seeds_search, "holdout": self.seeds_holdout},
                "budget": {"runs": self.runs, "hours": self.hours}, "search": self.search,
                "spec_patches": self.spec_patches, "quality": self.quality,
                "privacy": {"attacks": self.privacy_attacks, "config": self.privacy_config, "review": self.review},
                "domain_checks": self.domain_checks, "null_quantile": self.null_quantile, "halving": self.halving}

    @property
    def hard(self) -> list:
        return [c for c in self.constraints if isinstance(c, str)]

    @property
    def check_patterns(self) -> list:
        return [c["check"] for c in self.constraints if isinstance(c, dict) and "check" in c]

    @property
    def families(self) -> list | None:
        """Statistical families held to their null quantile (None = all)."""
        if any(t == "all" for t in self.targets):
            return None
        return [t["family"] for t in self.targets if isinstance(t, dict) and "family" in t]

    @property
    def min_pass_rate(self) -> float | None:
        r = [float(t["scorecard_pass_rate"]) for t in self.targets if isinstance(t, dict) and "scorecard_pass_rate" in t]
        return max(r) if r else None


def parse(doc: dict | None, base_dir: str | Path = ".") -> Objectives:
    """Objectives from a mapping (the defaults for None), validated."""
    from ..spec import deep_merge
    doc = doc or {}
    if not isinstance(doc, dict):
        raise ObjectivesError("objectives must be a mapping")
    bad = [k for k in doc if k not in DEFAULTS]
    errs = [f"objectives: unknown key '{k}' (allowed: {', '.join(DEFAULTS)})" for k in bad]
    d = deep_merge(copy.deepcopy(DEFAULTS), {k: v for k, v in doc.items() if k in DEFAULTS})
    for c in d["constraints"] or []:
        if isinstance(c, str):
            if c not in HARD:
                errs.append(f"constraints: unknown constraint {c!r} (allowed: {', '.join(HARD)} or {{check: pattern}})")
        elif not (isinstance(c, dict) and set(c) == {"check"} and isinstance(c["check"], str)):
            errs.append(f"constraints: {c!r} is not a name or {{check: pattern}}")
    for t in d["targets"] or []:
        if t == "all":
            continue
        if isinstance(t, dict) and set(t) == {"family"} and isinstance(t["family"], str):
            continue
        if isinstance(t, dict) and set(t) == {"scorecard_pass_rate"}:
            v = t["scorecard_pass_rate"]
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
                continue
        errs.append(f"targets: {t!r} is not 'all', {{family: name}} or {{scorecard_pass_rate: 0..1}}")
    seeds = d["seeds"] or {}
    for k in ("search", "holdout"):
        v = seeds.get(k) or []
        if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in v):
            errs.append(f"seeds.{k} must be a list of non-negative integers")
    if not (seeds.get("search") or []):
        errs.append("seeds.search must name at least one seed")
    if set(seeds.get("search") or []) & set(seeds.get("holdout") or []):
        errs.append("seeds.holdout must not repeat a search seed (a held-out seed is one the search never saw)")
    b = d["budget"] or {}
    runs = b.get("runs")
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        errs.append("budget.runs must be a positive integer (fits; one candidate on one seed is one run)")
    hours = b.get("hours")
    if hours is not None and (not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0):
        errs.append("budget.hours must be a positive number")
    s = d["search"]
    if not (s == "auto" or isinstance(s, (list, dict))):
        errs.append("search must be auto, a list of setting keys or {key: {low, high} | [choices]}")
    else:
        from ..settings import params
        keys = list(s) if isinstance(s, (list, dict)) else []
        for k in keys:
            if k not in params():
                errs.append(f"search: unknown setting {k!r} (relsynth settings reference lists them)")
    pv = d["privacy"] or {}
    if pv.get("review") not in ("undecided", "pass", "fail"):
        errs.append("privacy.review must be undecided, pass or fail")
    q = d["null_quantile"]
    if not isinstance(q, (int, float)) or isinstance(q, bool) or not 0.5 <= q < 1:
        errs.append("null_quantile must be in [0.5, 1)")
    if errs:
        raise ObjectivesError("Objectives have errors:\n  - " + "\n  - ".join(errs))
    return Objectives(constraints=list(d["constraints"] or []), targets=list(d["targets"] or []),
                      seeds_search=list(seeds["search"]), seeds_holdout=list(seeds.get("holdout") or []),
                      runs=int(runs or 0), hours=None if hours is None else float(hours), search=s,
                      spec_patches=bool(d["spec_patches"]), quality=bool(d["quality"]),
                      privacy_attacks=bool(pv.get("attacks")), privacy_config=pv.get("config"),
                      review=pv.get("review", "undecided"), domain_checks=d["domain_checks"],
                      null_quantile=float(q), halving=bool(d["halving"]), base_dir=str(base_dir))


def load(path: str | Path | None) -> Objectives:
    if path is None:
        return parse(None)
    p = Path(path)
    if not p.exists():
        raise ObjectivesError(f"objectives file {p} not found")
    return parse(read_yaml(p), p.parent)

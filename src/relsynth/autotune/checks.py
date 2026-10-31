# SPDX-License-Identifier: Apache-2.0
"""One trial of the self-tuning loop: fit, sample and evaluate a candidate on
one seed, returning its checks as aggregates (the bench history format).

A check is {"id", "pass", "value", "limit", "alpha"?, "status"?}:

- `scorecard:<where>|<metric>`: every scorecard item; `alpha` is its level
  under the null that the synthetic data is a draw of the real data's law
  (statistical items), absent on exact items;
- `quality:<check>:<where>`: the calibrated 2.0 quality checks (objectives `quality`);
- `privacy:<criterion>|<attack>`: the attack suite (objectives `privacy.attacks`),
  `status` pass / fail / review;
- `domain:<name>`: the user's domain checks (objectives `domain_checks`).

Only aggregates are kept: values, limits, verdicts.
"""
from __future__ import annotations

import copy
import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path

# scorecard metrics behind each hard constraint
HARD_METRICS = {
    "referential_integrity": ("fk_integrity",),
    "rules": ("rule_violation_rate",),
    "unique_keys": ("unique_duplicate_rate",),
    "trees": ("tree_cycles",),
    "privacy": ("dcr_ratio", "exact_match_rate", "text_rare_exact_copies", "text_rare_ngram_copies",
                "text_rare_token_leaks"),
}


def family(cid: str) -> str:
    """The family of a check id: its metric (scorecard), check (quality),
    `privacy:<criterion>` or `domain`."""
    kind, _, rest = cid.partition(":")
    if kind == "scorecard":
        return rest.rsplit("|", 1)[-1]
    if kind == "quality":
        return rest.split(":", 1)[0]
    if kind == "privacy":
        return "privacy:" + rest.split("|", 1)[0]
    return kind


def constraint_of(cid: str) -> str | None:
    """The hard constraint a check belongs to (None: a statistical or tracked check)."""
    kind = cid.partition(":")[0]
    if kind == "privacy":
        return "privacy"
    if kind == "domain":
        return "domain"
    if kind == "scorecard":
        f = family(cid)
        for name, metrics in HARD_METRICS.items():
            if f in metrics:
                return name
    return None


@dataclass
class TrialResult:
    seed: int
    checks: list = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None
    doctor: dict = field(default_factory=dict)      # spec patches relsynth doctor proposes for this run

    def to_dict(self) -> dict:
        return {"seed": self.seed, "checks": self.checks, "seconds": self.seconds, "error": self.error,
                "doctor": self.doctor}


def candidate_spec(raw: dict, delta: dict, patches: dict | None, seed: int) -> dict:
    """The raw spec of a candidate: doctor patches, then the settings delta
    over its synthesis block, then the seed."""
    from ..doctor import apply_patches
    from ..spec import deep_merge
    s = apply_patches(raw, patches) if patches else copy.deepcopy(raw)
    syn = deep_merge(s.get("synthesis") or {}, _nest(delta))
    syn["seed"] = int(seed)
    s["synthesis"] = syn
    return s


def _nest(delta: dict) -> dict:
    out: dict = {}
    for k, v in delta.items():
        cur = out
        *head, last = k.split(".")
        for h in head:
            cur = cur.setdefault(h, {})
        cur[last] = copy.deepcopy(v)
    return out


def load_domain(ref: str, base_dir: str | Path):
    """The domain-check function named by 'file.py:function' (the user's own code)."""
    path, _, fn = str(ref).partition(":")
    p = Path(path)
    p = p if p.is_absolute() else Path(base_dir) / p
    spec = importlib.util.spec_from_file_location(f"relsynth_domain_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"domain checks: cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn or "checks")


def _r(x, n=6):
    try:
        return None if x is None else round(float(x), n)
    except (TypeError, ValueError):
        return None


def domain_item(name: str, real, syn) -> dict:
    """An identity (real value 0 or 1: every row obeys a rule, or none breaks it)
    passes when the synthetic value equals it; other values are recorded only."""
    r, s = _r(real), _r(syn)
    identity = r is not None and r in (0.0, 1.0)
    return {"id": f"domain:{name}", "pass": (s == r) if identity else None, "value": s, "real": r}


def run_trial(raw: dict, base_dir: str, seed: int, delta: dict, patches: dict | None, obj) -> TrialResult:
    """Fit, sample and evaluate one candidate on one seed (in this process)."""
    from ..pipeline import execute
    t0 = time.time()
    spec = candidate_spec(raw, delta, patches, seed)
    spec["_base_dir"] = str(Path(base_dir).resolve())
    res = TrialResult(seed=int(seed))
    try:
        r = execute(spec, out=False, write=False, privacy=True)
    except Exception as e:  # noqa: BLE001 - a candidate that cannot run is a result
        res.error, res.seconds = f"{type(e).__name__}: {e}", round(time.time() - t0, 1)
        return res
    if r.errors:
        res.error = "; ".join(f"{e['step']}: {e['error']}" for e in r.errors)[:2000]
    items = ((r.report or {}).get("scorecard") or {}).get("items", [])
    for i in items:
        c = {"id": f"scorecard:{i['area']}|{i['metric']}", "pass": bool(i["pass"]), "value": i.get("value"),
             "limit": i.get("limit")}
        if i.get("alpha"):
            c["alpha"] = float(i["alpha"])
        res.checks.append(c)
    if obj.spec_patches and r.bundle:
        from ..doctor import diagnose, merge_patches
        try:
            res.doctor = merge_patches([i for i in diagnose(r.bundle) if i.kind == "spec" and i.patches])
        except Exception as e:  # noqa: BLE001 - a doctor failure only removes its proposals
            res.doctor = {}
            import logging
            logging.getLogger("relsynth.autotune").warning("doctor failed on the trial bundle: %r", e)
    if obj.quality and r.data is not None and not res.error:
        res.checks += _quality(r.spec, r.data)
    if obj.domain_checks and r.data is not None and r.fit_data is not None and not res.error:
        fn = load_domain(obj.domain_checks, obj.base_dir)
        for k, (rv, sv) in (fn(getattr(r.fit_data, "tables", None), r.data) or {}).items():
            res.checks.append(domain_item(k, rv, sv))
    if obj.privacy_attacks and not res.error:
        res.checks += _privacy(r.spec, seed, obj)
    res.seconds = round(time.time() - t0, 1)
    return res


def _quality(spec, data) -> list:
    from ..api import assess_quality
    rep = assess_quality(spec, data, privacy=False).to_dict()
    out = []
    for c in rep["checks"]:
        if c.get("source") != "quality-2.0":
            continue
        it = {"id": f"quality:{c['id']}", "pass": None if c["value"] is None else bool(c["pass"]),
              "value": c["value"], "limit": c["limit"]}
        if c.get("alpha"):
            it["alpha"] = float(c["alpha"])
        out.append(it)
    return out


def _privacy(spec, seed, obj) -> list:
    from ..privacy import assess
    cfg = None
    if obj.privacy_config:
        p = Path(obj.privacy_config)
        cfg = str(p if p.is_absolute() else Path(obj.base_dir) / p)
    rep = assess(spec, cfg, seed=int(seed))
    out = []
    ctx = (rep.get("contexts") or {}).get("internal") or {}
    for c in ctx.get("checks", []):
        ci = c.get("ci") or [None, None]
        out.append({"id": f"privacy:{c['criterion']}|{c['attack']}",
                    "pass": None if c["status"] == "info" else c["status"] == "pass",
                    "value": _r(c["value"]), "limit": _r(c["limit"]), "upper": _r(ci[1]), "status": c["status"]})
    return out

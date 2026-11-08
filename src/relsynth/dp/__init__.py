# SPDX-License-Identifier: Apache-2.0
"""Differential privacy mode (roadmap T07; guide docs/dp.md, proofs Props. 900-906).

    import relsynth.dp as dp
    res = dp.synthesize("spec.yaml")              # the spec's synthesis.privacy.dp section
    res.data, res.statement, res.charges

A DP run (1) plans which tables hold the unit's data and refuses what it cannot
account for, before reading anything; (2) reads the real tables once and
releases noisy integer statistics through OpenDP, every one charged to an
accountant that cannot exceed the run's budget; (3) decodes them into a seed
database; (4) fits the ordinary engine to the seed database and samples
(post-processing).  The guarantee is user-level, add/remove one unit with
all its rows, stated per NIST SP 800-226 in `statement`.

With a governance registry (`relsynth run --dp --registry ...`) the budget
is reserved in the organisation's ledger before step 2 and settled after it.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

from ..errors import PrivacyError
from .config import DPConfig, parse, section
from .mechanisms import Accountant, library
from .relational import decode, fit, make_plan


__all__ = ["synthesize", "DPResult", "DPConfig", "parse", "section", "plan", "statement", "stage2_spec", "generator",
           "domain_override", "manifest_privacy", "epsilon_of_config"]


@dataclass
class DPResult:
    data: dict                         # the synthetic tables (stage 2's first dataset)
    seed: dict                         # the DP seed database (stage 1)
    statement: dict
    charges: list
    rho: float
    delta: float
    epsilon: float
    config: DPConfig
    run: object = None                 # the stage-2 PipelineRun
    real: dict | None = field(default=None, repr=False)
    report: dict = field(default_factory=dict)
    # runs with several privacy units: unit -> {"rho", "delta", "epsilon", "tables"}; the
    # top-level rho, delta and epsilon are then the largest over the units (Prop. 1703)
    per_unit: dict = field(default_factory=dict)


def plan(spec_or_path, override: dict | None = None):
    """What a DP run of this spec would do, without reading data (raises
    PrivacyError with every reason it cannot run)."""
    from ..spec import load_spec
    spec = load_spec(spec_or_path)
    cfg = parse(spec, override)
    return make_plan(spec, cfg), cfg, spec


def stage2_spec(spec: dict) -> dict:
    """The spec stage 2 fits to the seed database: DP section removed, every table
    loaded (the seed is in memory)."""
    s = copy.deepcopy(spec)
    s["synthesis"]["privacy"] = {k: v for k, v in s["synthesis"].get("privacy", {}).items() if k != "dp"}
    s["synthesis"]["pushdown"] = {**s["synthesis"].get("pushdown", {}), "mode": "off"}
    return s


def synthesize(spec_or_path, *, tables: dict | None = None, override: dict | None = None, before_read=None,
               after_fit=None, out=None, write: bool = True, scale=None, datasets=None, seed=None,
               evaluate: bool = True, privacy: bool = False, ledger_note: dict | None = None,
               after_fit_units=None) -> DPResult:
    """A differentially private synthetic database.

    before_read(): called after planning and before the first read of real data
    (the ledger's touch); after_fit(rho, delta): called when the statistics are
    released (the ledger's settle); with several units, after_fit_units(per_unit)
    is called instead when given ({unit: {rho, delta, ...}}: one settle per unit's
    account).  tables: real tables already in memory (they count as read)."""
    from ..governance.accounting import epsilon_of
    from ..pipeline import execute
    t0 = time.time()
    pl, cfg, spec = plan(spec_or_path, override)
    units = cfg.units or [cfg.unit]
    # each unit's own budget: the run's, or the one set for that unit (RFC-G1-01)
    accts = {u: Accountant(cfg.rho_of(u), cfg.delta * cfg.levels_delta_share) for u in units}
    acct = accts[units[0]]
    if before_read is not None:
        before_read()
    if tables is None:
        from ..pushdown import load_for_fit
        real = load_for_fit(spec, mode="off").tables
    else:
        from ..data import coerce
        real = {t: coerce(df, spec["tables"][t], optional=set()) if t in spec["tables"] else df
                for t, df in tables.items()}
    model = fit(spec, real, cfg, accts if cfg.multi else acct, pl)
    per_unit: dict = {}
    if cfg.multi:
        for u, a in accts.items():
            for c in a.charges:
                c.unit = u
            r_u, d_u = a.spent
            per_unit[u] = {"rho": r_u, "delta": d_u, "epsilon": epsilon_of(r_u, cfg.delta - d_u) if r_u > 0 else 0.0,
                           "tables": pl.owned_by(u), "planned_rho": cfg.rho_of(u)}
        rho, d0 = max(v["rho"] for v in per_unit.values()), max(v["delta"] for v in per_unit.values())
    else:
        rho, d0 = acct.spent
    if cfg.multi and after_fit_units is not None:
        after_fit_units(per_unit)
    elif after_fit is not None:
        after_fit(rho, d0)
    seed_db = decode(spec, model)
    t_fit = time.time() - t0
    s2 = stage2_spec(spec)
    run = execute(s2, tables=seed_db, out=out if write else False, write=write, scale=scale, datasets=datasets,
                  seed=seed, evaluate_data=False, privacy=False)
    if run.exceptions:
        step, e = run.exceptions[0]
        raise PrivacyError(f"stage 2 failed at {step}: {e}") from e
    eps = epsilon_of(rho, cfg.delta - d0) if rho > 0 else 0.0
    every = Accountant(math.inf, 1.0, [c for a in accts.values() for c in a.charges]) if cfg.multi else acct
    st = statement(spec, cfg, pl, every, eps, ledger_note=ledger_note, per_unit=per_unit)
    rep = {"seconds_stage1": round(t_fit, 2), "seconds_total": round(time.time() - t0, 2), **model.report}
    if evaluate:
        from ..evaluate import evaluate as _ev
        rep["scorecard"] = _ev(s2, real, run.data, privacy=privacy,
                               sample_log=run.synthesizer.sample_log if run.synthesizer else None)
    return DPResult(data=run.data or {}, seed=seed_db, statement=st,
                    charges=[c for a in accts.values() for c in a.charges], rho=rho, delta=d0, epsilon=eps,
                    config=cfg, run=run, real=real, report=rep, per_unit=per_unit)


def statement(spec, cfg: DPConfig, pl, acct: Accountant, eps: float, ledger_note: dict | None = None,
              per_unit: dict | None = None) -> dict:
    """The DP statement of a release (schema relsynth.dp-statement/1; design sec. 5.7,
    NIST SP 800-226 secs. 2.3-2.6 and 3.1).  It holds no data and no exact counts.
    A run with several privacy units states one guarantee per unit (schema
    relsynth.dp-statement/2: `unit` is the list, `units` the guarantee of each)."""
    from ..governance.accounting import dec
    if per_unit:
        return _statement_multi(spec, cfg, pl, acct, eps, per_unit, ledger_note)
    rho, d0 = acct.spent
    by_stage: dict = {}
    for c in acct.charges:
        by_stage.setdefault(c.stage, []).append(c.mechanism)
    st: dict = {
        "schema": "relsynth.dp-statement/1",
        "unit": cfg.unit, "adjacency": "add_remove",
        "unit_meaning": f"one {cfg.unit} together with every row of the tables it owns ({', '.join(pl.order)})",
        "guarantee": {"epsilon": dec(eps, 6), "delta": dec(cfg.delta, 30), "rho": dec(rho), "delta_approx": dec(d0, 30)},
        "accounting": {"method": "zero-concentrated DP (Bun and Steinke 2016); approximate zCDP for level selection",
                       "conversion": "Canonne, Kamath and Steinke 2020, Prop. 12, at delta - delta_approx",
                       "composition": "adaptive, rho and delta add"},
        "mechanisms": {k: sorted(set(v)) for k, v in by_stage.items()},
        "noise": {"library": library(), "distribution": "discrete Gaussian on integer statistics",
                  "randomness": "operating-system CSPRNG; not seeded and not recorded (the run is not reproducible)"},
        "post_processing": ["the seed database decoded from the noisy statistics",
                            "the relsynth engine fitted to the seed database (posterior draws, segments, rules)",
                            "primary keys unique and every link valid by construction",
                            "rules declared in the spec hold as they did in non-DP mode"],
        "public_inputs": {"tables_released_as_given": list(pl.public),
                          "columns_declared_public": sorted(cfg.public),
                          "spec_is_public": bool(cfg.spec_is_public)},
        "not_protected": [f"tables without data of {cfg.unit}: {', '.join(pl.public) or 'none'}"],
        "hazards": ["other releases of the same people made without DP are not covered by this guarantee",
                    "an empirical attack audit is an average-case check and does not bound outliers",
                    "rare category levels are suppressed and large contributions are truncated (bias)"],
    }
    if not cfg.spec_is_public:
        st["hazards"].insert(0, "the spec was not declared public: values in it drawn from the data (levels, ranges, "
                                "rules) are not covered by this guarantee")
    if ledger_note:
        st["domain"] = ledger_note
    return st


def _statement_multi(spec, cfg: DPConfig, pl, acct, eps: float, per_unit: dict, ledger_note: dict | None) -> dict:
    """The statement of a run with several privacy units (Prop. 1703): each unit's
    statistics read only the tables of that unit, so each unit has its own
    guarantee, and they do not add up."""
    from ..governance.accounting import dec
    single = statement(spec, cfg, pl, acct, eps, ledger_note)    # acct: every unit's charges
    units = list(per_unit)
    st = {**single, "schema": "relsynth.dp-statement/2", "unit": units,
          "unit_meaning": "; ".join(f"one {u} together with every row of the tables it owns "
                                    f"({', '.join(v['tables'])})" for u, v in per_unit.items()),
          "guarantee": {"epsilon": dec(eps, 6), "delta": dec(cfg.delta, 30),
                        "rho": dec(max(v["rho"] for v in per_unit.values())),
                        "delta_approx": dec(max(v["delta"] for v in per_unit.values()), 30),
                        "per": "each unit separately (the largest of the per-unit guarantees)"},
          "units": {u: {"epsilon": dec(v["epsilon"], 6), "delta": dec(cfg.delta, 30), "rho": dec(v["rho"]),
                        "delta_approx": dec(v["delta"], 30), "tables": list(v["tables"])}
                    for u, v in per_unit.items()},
          "not_protected": [f"tables without data of {' or '.join(units)}: {', '.join(pl.public) or 'none'}"]}
    st["accounting"] = {**single["accounting"],
                        "units": "each unit's charges add up separately; the units' tables are disjoint, so a unit's "
                                 "neighbouring database changes only the statistics of its own tables "
                                 "(parallel composition, Prop. 1703)"}
    st["hazards"] = single["hazards"] + ["a person who is both units at once (a customer who is also an employee) "
                                         "is protected for each role separately: the guarantees then add up"]
    return st


def domain_override(spec: dict, policy, domain: str, override: dict | None = None) -> dict:
    """The DP settings a governed run must use in a privacy domain: the domain's
    unit (or units) and its delta (the delta the domain's epsilon is stated at).
    A run's own delta for level selection is charged to the domain, so by
    default it takes 1 % of the domain's delta, leaving room for many runs
    (Prop. 801).  In a domain with several units a run protects every unit of
    the domain that its source holds; each unit is charged to its own account
    (Prop. 1704)."""
    d = policy.domains.get(domain)
    if d is None:
        raise PrivacyError("source is in no privacy domain of the policy; a governed DP run needs one")
    dom_units = policy.units(domain) if hasattr(policy, "units") else \
        [u for u in [(d.get("unit") or {}).get("entity")] if u]
    raw = {**(section(spec) or {}), **(override or {})}
    want = raw.get("unit")
    if isinstance(want, (list, tuple)):
        want = [str(u) for u in want]
        if len(want) == 1:
            want = want[0]
    asked = want if isinstance(want, list) else [want] if want else []
    if asked and dom_units:
        extra = [u for u in asked if u not in dom_units]
        if extra:
            what = f"privacy unit of domain {domain!r} is {dom_units[0]!r}" if len(dom_units) == 1 else \
                f"privacy units of domain {domain!r} are {', '.join(map(repr, dom_units))}"
            raise PrivacyError(f"the spec protects {', '.join(map(repr, extra))} but the {what}; nothing was read")
    # every unit of the domain that this source holds is protected: a unit left out would have its
    # tables released as public (Prop. 1704)
    held = [u for u in dom_units if u in spec.get("tables", {})]
    missing = [u for u in held if asked and u not in asked]
    if missing:
        raise PrivacyError(f"domain {domain!r} protects {', '.join(map(repr, held))}, and this source holds them all; "
                           f"the spec leaves out {', '.join(map(repr, missing))}: protect every unit of the domain "
                           "in one run (synthesis.privacy.dp.unit: a list); nothing was read")
    out = dict(override or {})
    if asked:
        out["unit"] = want
    else:
        pick = held or dom_units
        out["unit"] = pick if len(pick) > 1 else (pick[0] if pick else None)
    out["delta"] = policy.caps(domain).delta
    out.setdefault("levels_delta_share", raw.get("levels_delta_share", 0.01))
    return out


def manifest_privacy(res: DPResult, run_id: str, ledger: dict | None = None) -> dict:
    """The `privacy` section of a run manifest for a DP run (design sec. 6.2).  A
    run with several units lists them: `unit` is the list, each charge names its
    unit, and `units` holds each unit's total, statement and ledger figures
    (Prop. 1704); `ledger` is then {unit: figures}."""
    from ..governance.accounting import dec
    out: dict = {"mode": "dp", "unit": res.config.unit, "adjacency": "add_remove", "_run_id": run_id,
           "charges": [c.to_manifest() for c in res.charges],
           "run_total": {"rho": dec(res.rho), "delta": dec(res.delta, 30)},
           "stated": {"epsilon": dec(res.epsilon, 6), "delta": dec(res.config.delta, 30)}}
    if res.per_unit:
        out["unit"] = list(res.per_unit)
        out["units"] = {}
        for u, v in res.per_unit.items():
            e = {"run_total": {"rho": dec(v["rho"]), "delta": dec(v["delta"], 30)},
                 "stated": {"epsilon": dec(v["epsilon"], 6), "delta": dec(res.config.delta, 30)},
                 "tables": list(v["tables"])}
            if ledger and u in ledger:
                e["ledger"] = ledger[u]
            out["units"][u] = e
        return out
    if ledger:
        out["ledger"] = ledger
    return out


def generator(spec_or_path, override: dict | None = None):
    """fit-and-sample function of the whole DP pipeline for relsynth.privacy.assess
    (shadow databases at scale 1).  Each call spends a fresh budget on its own
    shadow data: the audit is internal and its report never leaves the
    organisation, so these runs are not charged to a domain."""
    import hashlib
    import json

    from ..spec import load_spec
    base = load_spec(spec_or_path)
    base["synthesis"]["scale"] = {"default": 1.0}
    parse(base, override)                                   # refuse a bad DP section before any shadow fit

    def gen(tables, seed):
        r = synthesize(copy.deepcopy(base), tables={t: df.reset_index(drop=True) for t, df in tables.items()},
                       override=override, write=False, evaluate=False, seed=int(seed))
        return r.data
    gen.ident = hashlib.sha256(json.dumps([base, override], sort_keys=True,  # type: ignore[attr-defined]
                                          default=str).encode()).hexdigest()
    return gen


def epsilon_of_config(cfg: DPConfig) -> float:
    """The epsilon a run with this configuration states at its delta, before it runs."""
    from ..governance.accounting import epsilon_of
    return cfg.epsilon if cfg.epsilon is not None else epsilon_of(cfg.rho, cfg.delta * (1 - cfg.levels_delta_share))

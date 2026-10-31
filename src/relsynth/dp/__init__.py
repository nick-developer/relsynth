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
import logging
import time
from dataclasses import dataclass, field

from ..errors import PrivacyError
from .config import DPConfig, parse, section
from .mechanisms import Accountant, library
from .relational import decode, fit, make_plan

log = logging.getLogger("relsynth")

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
               evaluate: bool = True, privacy: bool = False, ledger_note: dict | None = None) -> DPResult:
    """A differentially private synthetic database.

    before_read(): called after planning and before the first read of real data
    (the ledger's touch); after_fit(rho, delta): called when the statistics are
    released (the ledger's settle).  tables: real tables already in memory (they
    count as read)."""
    from ..governance.accounting import epsilon_of
    from ..pipeline import execute
    t0 = time.time()
    pl, cfg, spec = plan(spec_or_path, override)
    acct = Accountant(cfg.rho, cfg.delta * cfg.levels_delta_share)
    if before_read is not None:
        before_read()
    if tables is None:
        from ..pushdown import load_for_fit
        real = load_for_fit(spec, mode="off").tables
    else:
        from ..data import coerce
        real = {t: coerce(df, spec["tables"][t], optional=set()) if t in spec["tables"] else df
                for t, df in tables.items()}
    model = fit(spec, real, cfg, acct, pl)
    rho, d0 = acct.spent
    if after_fit is not None:
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
    st = statement(spec, cfg, pl, acct, eps, ledger_note=ledger_note)
    rep = {"seconds_stage1": round(t_fit, 2), "seconds_total": round(time.time() - t0, 2), **model.report}
    if evaluate:
        from ..evaluate import evaluate as _ev
        rep["scorecard"] = _ev(s2, real, run.data, privacy=privacy,
                               sample_log=run.synthesizer.sample_log if run.synthesizer else None)
    return DPResult(data=run.data or {}, seed=seed_db, statement=st, charges=list(acct.charges), rho=rho,
                    delta=d0, epsilon=eps, config=cfg, run=run, real=real, report=rep)


def statement(spec, cfg: DPConfig, pl, acct: Accountant, eps: float, ledger_note: dict | None = None) -> dict:
    """The DP statement of a release (schema relsynth.dp-statement/1; design sec. 5.7,
    NIST SP 800-226 secs. 2.3-2.6 and 3.1).  It holds no data and no exact counts."""
    from ..governance.accounting import dec
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


def domain_override(spec: dict, policy, domain: str, override: dict | None = None) -> dict:
    """The DP settings a governed run must use in a privacy domain: the domain's
    unit and its delta (the delta the domain's epsilon is stated at).  A run's
    own delta for level selection is charged to the domain, so by default it
    takes 1 % of the domain's delta, leaving room for many runs (Prop. 801)."""
    d = policy.domains.get(domain)
    if d is None:
        raise PrivacyError("source is in no privacy domain of the policy; a governed DP run needs one")
    unit = (d.get("unit") or {}).get("entity")
    raw = {**(section(spec) or {}), **(override or {})}
    if raw.get("unit") and unit and raw["unit"] != unit:
        raise PrivacyError(f"the spec protects {raw['unit']!r} but the privacy unit of domain {domain!r} is {unit!r}; "
                           "nothing was read")
    out = dict(override or {})
    out["unit"] = raw.get("unit") or unit
    out["delta"] = policy.caps(domain).delta
    out.setdefault("levels_delta_share", raw.get("levels_delta_share", 0.01))
    return out


def manifest_privacy(res: DPResult, run_id: str, ledger: dict | None = None) -> dict:
    """The `privacy` section of a run manifest for a DP run (design sec. 6.2)."""
    from ..governance.accounting import dec
    out = {"mode": "dp", "unit": res.config.unit, "adjacency": "add_remove", "_run_id": run_id,
           "charges": [c.to_manifest() for c in res.charges],
           "run_total": {"rho": dec(res.rho), "delta": dec(res.delta, 30)},
           "stated": {"epsilon": dec(res.epsilon, 6), "delta": dec(res.config.delta, 30)}}
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

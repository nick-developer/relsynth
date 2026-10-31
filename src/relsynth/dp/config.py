# SPDX-License-Identifier: Apache-2.0
"""The `synthesis.privacy.dp` section of a spec (docs/dp.md).

    synthesis:
      privacy:
        dp:
          enabled: true
          unit: Customer                 # the privacy unit (the IR's, or the policy domain's)
          epsilon: 2.0                   # this run's budget, stated at delta ...
          delta: 1.0e-9                  # ... or give rho (zCDP) instead of epsilon
          public_tables: [Genre, MediaType, Track]   # or all_unowned: tables without data of the unit
          public: {Customer.Country: {levels: [GR, DE, FR]}, Invoice.Total: {min: 0, max: 50}}
          max_degree: {Invoice: auto, InvoiceLine: 30}
          spec_is_public: true           # the spec was written from public facts only

relsynth ships no default epsilon: a DP run without `epsilon` or `rho` is
refused.  Inside an organisation with a policy, the ledger may refuse it
too (governance).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import PrivacyError

KEYS = {"enabled", "unit", "epsilon", "delta", "rho", "public_tables", "public", "max_degree", "degree_quantile",
        "split", "bins", "clip", "max_parent_attrs", "drop_columns", "spec_is_public", "levels_delta_share",
        "max_degree_range", "degree_fallback"}
SPLIT = {"discovery": 0.15, "degrees": 0.2, "marginals": 0.35, "correlations": 0.3}


@dataclass
class DPConfig:
    unit: str
    rho: float
    delta: float
    epsilon: float | None = None
    public_tables: list | str = field(default_factory=list)
    public: dict = field(default_factory=dict)
    max_degree: dict = field(default_factory=dict)
    degree_quantile: float = 0.98
    split: dict = field(default_factory=lambda: dict(SPLIT))
    bins: int = 32
    clip: float = 1.5
    max_parent_attrs: int = 6
    drop_columns: list = field(default_factory=list)
    spec_is_public: bool = False
    levels_delta_share: float = 0.5
    max_degree_range: int = 10000
    degree_fallback: int = 8          # cap when no cell of the noisy degree histogram stands above the noise

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def section(spec: dict) -> dict | None:
    dp = ((spec.get("synthesis") or {}).get("privacy") or {}).get("dp")
    return dp if dp and dp.get("enabled", True) else None


def parse(spec: dict, override: dict | None = None) -> DPConfig:
    raw = dict(section(spec) or {})
    raw.update(override or {})
    bad = set(raw) - KEYS
    if bad:
        raise PrivacyError(f"synthesis.privacy.dp: unknown key(s) {sorted(bad)} (known: {sorted(KEYS)})")
    unit = raw.get("unit")
    if not unit:
        raise PrivacyError("synthesis.privacy.dp.unit: name the table of the people DP protects (e.g. Customer)")
    if unit not in spec["tables"]:
        raise PrivacyError(f"synthesis.privacy.dp.unit: {unit!r} is not a table of the spec")
    delta = raw.get("delta")
    if delta is None or not 0 < float(delta) <= 1e-5:
        raise PrivacyError("synthesis.privacy.dp.delta: set delta in (0, 1e-5]; NIST SP 800-226 sec. 2.3 calls larger "
                           "values suspicious, and at most 1/n^2 for n people is common")
    delta = float(delta)
    from ..governance.accounting import rho_of
    if raw.get("rho") is not None and raw.get("epsilon") is not None:
        raise PrivacyError("synthesis.privacy.dp: give epsilon or rho, not both")
    if raw.get("rho") is not None:
        rho, eps = float(raw["rho"]), None
    elif raw.get("epsilon") is not None:
        eps = float(raw["epsilon"])
        share = float(raw.get("levels_delta_share", 0.5))
        rho = rho_of(eps, delta * (1 - share))       # the rest of delta is for releasing category levels
    else:
        raise PrivacyError("synthesis.privacy.dp: set the run's budget (epsilon at delta, or rho); relsynth has no "
                           "default epsilon, see docs/governance/guide.md#choosing-epsilon")
    if not rho > 0:
        raise PrivacyError("synthesis.privacy.dp: the budget must be positive")
    split = {**SPLIT, **(raw.get("split") or {})}
    if set(split) - set(SPLIT) or abs(sum(split.values()) - 1) > 1e-9 or min(split.values()) <= 0:
        raise PrivacyError(f"synthesis.privacy.dp.split: shares of {sorted(SPLIT)}, positive, adding up to 1")
    md = raw.get("max_degree") or {}
    for t, v in md.items():
        if v != "auto" and not (isinstance(v, int) and v >= 1):
            raise PrivacyError(f"synthesis.privacy.dp.max_degree.{t}: 'auto' or a positive integer")
    pub = raw.get("public") or {}
    for k, v in pub.items():
        if "." not in k or not isinstance(v, dict) or not (set(v) <= {"levels", "min", "max"}) or not v:
            raise PrivacyError(f"synthesis.privacy.dp.public.{k}: TABLE.COLUMN: {{levels: [...]}} or {{min, max}}")
    pt = raw.get("public_tables", [])
    if pt != "all_unowned" and not isinstance(pt, list):
        raise PrivacyError("synthesis.privacy.dp.public_tables: a list of tables, or all_unowned")
    return DPConfig(unit=unit, rho=rho, delta=delta, epsilon=eps, public_tables=pt, public=pub, max_degree=md,
                    degree_quantile=float(raw.get("degree_quantile", 0.98)), split=split, bins=int(raw.get("bins", 32)),
                    clip=float(raw.get("clip", 1.5)), max_parent_attrs=int(raw.get("max_parent_attrs", 6)),
                    drop_columns=list(raw.get("drop_columns") or []), spec_is_public=bool(raw.get("spec_is_public")),
                    levels_delta_share=float(raw.get("levels_delta_share", 0.5)),
                    max_degree_range=int(raw.get("max_degree_range", 10000)),
                    degree_fallback=int(raw.get("degree_fallback", 8)))

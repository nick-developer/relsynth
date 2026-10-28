# SPDX-License-Identifier: Apache-2.0
"""Settings of a privacy assessment (privacy.yaml) and the release contexts.

Every key is optional.  The release contexts follow the contextual approach of
the CJEU in EDPS v SRB (C-413/23 P, 4 September 2025) and EDPB Guidelines
02/2026 (paras. 7, 11, 29-35): whether data are anonymous depends on who
receives them and which additional information that recipient can reasonably
use.  Following para. 31, a context never assumes a less *motivated*
attacker; it changes only the additional information assumed and the safety
margin (para. 88).

The thresholds below are relsynth's suggested starting points, not legal
standards.  The organisation that operates relsynth sets its own (T08 policy
will hold them).  Differential privacy is required for every context outside
the organisation (user decision of 2026-10-26); there is no built-in epsilon.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONTEXTS = ("internal", "partner", "public")

DEFAULTS: dict[str, Any] = {
    "unit": None,                     # person table; detected when omitted
    "sensitive": [],                  # profile columns whose values must not be inferable
    "quasi_identifiers": None,        # profile columns a recipient may know; default: the unit table's attributes
    "split": {"control": 0.4, "reference": 0.2, "seed": 0, "max_people": 20000},
    "confidence": 0.95,
    "attacks": {
        "record_isolation": {"modes": ["univariate", "multivariate"], "n_attacks": 500, "n_cols": 3},
        "linkage": {"n_attacks": 500, "n_neighbors": 1, "aux_cols": None},
        "inference": {"n_attacks": 2000},
        "attribute_inference": {"max_rows": 20000},
        "membership": {
            "domias": True,
            # n_test_gated: test shadows when a context gates on the single most exposed
            # person (per_person: gate) and is not blocked; 128 halves the interval width
            "shadow": {"enabled": True, "n_train": 32, "n_test": 32, "n_test_gated": 128, "shadow_size": 1000,
                       "targets_vulnerable": 10, "targets_random": 10, "n_jobs": 1},
        },
    },
    # fewer people than this in the privacy unit: a warning, and a forced fail in
    # every context outside the organisation (each person weighs too much)
    "small_population": 1000,
    # per context: per_person "gate" makes the most exposed person's attack a
    # check; "info" shows it and gates on the pooled attack on the vulnerable
    # targets instead.  sign_off: a "review" verdict needs a named person's
    # sign-off (relsynth privacy --signed-off-by) before a gate passes it.
    "contexts": {
        "internal": {
            "description": "Staff of the organisation, for development and testing.",
            "external": False,
            "attacker_knows": "quasi_identifiers",
            "per_person": "info",
            "sign_off": True,
            "thresholds": {"risk": 0.10, "attribute_leakage": 0.05, "mia_auc": 0.65, "mia_tpr_at_fpr_0.01": 0.10,
                           "record_auc": 0.85},
        },
        "partner": {
            "description": "A named organisation under contract (processor or joint controller).",
            "external": True,
            "attacker_knows": "all",
            "per_person": "gate",
            "sign_off": True,
            "thresholds": {"risk": 0.05, "attribute_leakage": 0.03, "mia_auc": 0.60, "mia_tpr_at_fpr_0.01": 0.05,
                           "record_auc": 0.75},
        },
        "public": {
            "description": "Anyone (open data, publication).",
            "external": True,
            "attacker_knows": "all",
            "per_person": "gate",
            "sign_off": True,
            "thresholds": {"risk": 0.02, "attribute_leakage": 0.02, "mia_auc": 0.55, "mia_tpr_at_fpr_0.01": 0.02,
                           "record_auc": 0.65},
        },
    },
    # DP guarantee of the release (filled by the DP engine, T07): {"epsilon": .., "delta": .., "unit": ..}
    "release": {"dp": None},
    "cache": None,                    # folder for shadow fits (reused across runs)
}

KNOWN = set(DEFAULTS)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(cfg=None) -> dict:
    """Defaults merged with a dict or a YAML file; unknown keys are errors."""
    if isinstance(cfg, (str, Path)):
        cfg = yaml.safe_load(Path(cfg).read_text(encoding="utf-8")) or {}
    cfg = dict(cfg or {})
    if "privacy" in cfg and len(cfg) == 1:
        cfg = cfg["privacy"]
    bad = set(cfg) - KNOWN
    if bad:
        raise ValueError(f"unknown privacy settings: {', '.join(sorted(bad))} (known: {', '.join(sorted(KNOWN))})")
    out = _merge(DEFAULTS, cfg)
    for name, c in out["contexts"].items():
        if name not in DEFAULTS["contexts"]:
            if "thresholds" not in c:
                raise ValueError(f"context {name!r} needs thresholds")
            c.setdefault("external", True)
            c.setdefault("attacker_knows", "all")
            c.setdefault("per_person", "gate")
            c.setdefault("sign_off", True)
        if c.get("per_person", "gate") not in ("gate", "info"):
            raise ValueError(f"context {name!r}: per_person must be 'gate' or 'info'")
    return out

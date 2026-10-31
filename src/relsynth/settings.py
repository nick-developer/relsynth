# SPDX-License-Identifier: Apache-2.0
"""Engine settings: one registry of every adjustable parameter, a settings file,
and editable attributes.

Every key of `synthesis` (relsynth.spec.DEFAULT_SYNTHESIS) is a `Param` here,
with its type, range or choices, whether the self-tuning loop may move it
(`tunable`), on what scale, and which check families it affects (`affects`,
the move directions of relsynth.autotune).  The registry is the single source
of truth for validation, the generated reference (docs/settings-reference.md),
the settings template and the tuner's search space.

Precedence, lowest first:

1. the built-in defaults (this registry);
2. a settings file: `synthesis.settings_file: path` in the spec, `--settings
   path` on the command line, or the RELSYNTH_SETTINGS environment variable;
   `preset:fast` and `preset:thorough` name the built-in files.  A settings
   file is the `synthesis` block on its own (or a mapping with a `synthesis`
   key), plus an optional `per_table` block;
3. the spec's own `synthesis` block;
4. `--set synthesis.KEY=VALUE` and `Spec.set()`.

In Python the settings are editable attributes, checked on assignment::

    s = relsynth.Settings.load("spec.yaml")
    s.copula.sample_rows = 5000
    s.evaluation.thresholds.ks = 0.06
    s.table("FactInternetSales").copula.sample_rows = 20000
    model = relsynth.fit("spec.yaml", settings=s)

With no settings the engine reads exactly the defaults of relsynth 1.0, so the
synthetic output does not change.
"""
from __future__ import annotations

import copy
import difflib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .spec import SpecError

ENV = "RELSYNTH_SETTINGS"
PRESET_DIR = Path(__file__).parent / "data"
PRESETS = ("fast", "thorough")
# the table-level groups a `per_table` block may override (read by a table's model)
TABLE_GROUPS = ("copula", "marginals", "text")


@dataclass(frozen=True)
class Param:
    """One adjustable parameter of the engine."""
    key: str
    default: Any
    type: str                              # bool | int | float | str | choice | map | any
    doc: str = ""
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    nullable: bool = False
    tunable: bool = False
    scale: str = "linear"                  # linear | log (search scale of a tunable number)
    affects: tuple = ()                    # check families the parameter moves (relsynth.autotune)
    direction: dict = field(default_factory=dict, compare=False, hash=False)   # family -> +1 / -1 (raise / lower)

    def check(self, value) -> str | None:
        """None when value is valid, else the reason."""
        if value is None:
            return None if (self.nullable or self.default is None) else "must not be null"
        t = self.type
        if t == "bool":
            return None if isinstance(value, bool) else f"must be true or false, not {value!r}"
        if t in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                if self.choices and value in self.choices:
                    return None
                return f"must be a number, not {value!r}" + (f" (or one of {', '.join(map(str, self.choices))})"
                                                             if self.choices else "")
            if t == "int" and float(value) != int(value):
                return f"must be a whole number, not {value!r}"
            if not math.isfinite(float(value)):
                return f"must be finite, not {value!r}"
            if self.low is not None and value < self.low:
                return f"must be >= {self.low}, not {value!r}"
            if self.high is not None and value > self.high:
                return f"must be <= {self.high}, not {value!r}"
            return None
        if t == "choice":
            if isinstance(value, (dict, list)) and "map" in self.choices:
                return None
            if isinstance(value, bool) and "on" in self.choices:
                return None                 # YAML reads a bare on / off as true / false
            if value in self.choices:
                return None
            return f"must be one of {', '.join(map(str, (c for c in self.choices if c != 'map')))}" + \
                (" or a mapping" if "map" in self.choices else "") + f", not {value!r}"
        if t == "str":
            return None if isinstance(value, str) else f"must be text, not {value!r}"
        if t == "map":
            return None if isinstance(value, dict) else f"must be a mapping, not {type(value).__name__}"
        return None


def _p(key, **kw):
    return key, kw


# Metadata of the keys whose type, range or choices cannot be read from the
# default alone, and of the tunable ones.  `affects` names check families: the
# metric of a scorecard item (`scorecard:<where>|<metric>`), the check of a
# quality item (`quality:<check>:<where>`) or `privacy:<criterion>`.
_FID_ASSOC = ("association_max_diff", "join_assoc_max_z", "discriminator_auc", "kway2_tvd_max", "kway2_tvd_mean",
              "kway3_tvd_mean", "kway3_tvd_max", "beta_recall_gap", "js_distance")
_FID_MARG = ("marginal_ks", "marginal_tvd", "null_rate_diff", "w1_raw_sd", "js_distance")
_CARD = ("cardinality_ks", "cardinality_mean_rel_err", "fanout2_w1_rel", "hop_trend_gap", "hop_trend_worst_gap")
_JOIN = ("join_tvd", "join_ks", "hop_trend_gap", "hop_trend_worst_gap")
_NOVEL = ("novel_combination_rate", "pair_repeat_share_diff")
_PRIV = ("dcr_ratio", "exact_match_rate", "privacy:record_isolation", "privacy:linkage", "privacy:inference",
         "privacy:membership")
_TEXT = ("text_rare_exact_copies", "text_rare_ngram_copies", "text_rare_token_leaks", "text_tstr_ratio")
_RULES = ("rule_violation_rate", "soft_rule_rate_diff", "unique_duplicate_rate")
_TIME = ("time_bucket_tvd", "time_bucket_tvd_conditional")
_TREE = ("tree_depth_tvd", "tree_fanout_ks", "attr_by_depth_tvd", "tree_depth_ppc_pvalue")

META: dict[str, dict] = dict([
    _p("seed", type="int", low=0, high=2 ** 63 - 1),
    _p("posterior.mode", type="choice", choices=("draw", "mean")),
    _p("posterior.datasets", type="int", low=1, high=10_000),
    _p("posterior.sample_seed", type="int", low=0, high=2 ** 63 - 1, nullable=True),
    _p("scale.default", type="float", low=0, high=1e6),
    _p("scale.tables", type="map"),
    _p("segments.default_cap", type="int", low=1, high=512, tunable=True, scale="log",
       affects=_CARD + _JOIN + _FID_ASSOC + _NOVEL, direction={f: +1 for f in _CARD + _JOIN + _FID_ASSOC}
       | {f: -1 for f in _NOVEL}),
    _p("segments.min_rows_per_segment", type="int", low=1, high=1e7, tunable=True, scale="log",
       affects=_CARD + _JOIN + _PRIV, direction={f: -1 for f in _CARD + _JOIN} | {f: +1 for f in _PRIV}),
    _p("segments.tables", type="map"),
    _p("segments.stratify", type="choice", choices=("auto", "none", "map")),
    _p("segments.stratify_max_levels", type="int", low=0, high=1e6),
    _p("segments.stratify_max_cells", type="float", low=0, high=1e12),
    _p("segments.attribute_strata", type="choice", choices=("auto", "none", "map")),
    _p("segments.tie_headers", type="bool"),
    _p("segments.copied_budget_cells", type="float", low=0, high=1e12),
    _p("segments.copied_rows", type="choice", choices=("auto", "none", "map")),
    _p("gibbs.iterations", type="int", low=1, high=1e6, tunable=True, scale="log", affects=_CARD + _JOIN,
       direction={f: +1 for f in _CARD + _JOIN}),
    _p("gibbs.final_iterations", type="int", low=0, high=1e6),
    _p("gibbs.max_final_iterations", type="int", low=0, high=1e7),
    _p("gibbs.rhat_target", type="float", low=1.0, high=10.0),
    _p("gibbs.chains", type="int", low=1, high=1),
    _p("gibbs.merge", type="bool"),
    _p("copula.mode", type="choice", choices=("rank", "bayes", "fast"), tunable=True, affects=_FID_ASSOC),
    _p("copula.sample_rows", type="int", low=10, high=1e8, tunable=True, scale="log", affects=_FID_ASSOC,
       direction={f: +1 for f in _FID_ASSOC}),
    _p("copula.iterations", type="int", low=1, high=1e7),
    _p("copula.max_iterations", type="int", low=0, high=1e8, nullable=True),
    _p("copula.rhat_target", type="float", low=1.0, high=10.0),
    _p("copula.burn_in", type="int", low=0, high=1e7),
    _p("copula.lkj_eta", type="float", low=1e-6, high=1e6, tunable=True, scale="log", affects=_FID_ASSOC,
       direction={f: -1 for f in _FID_ASSOC}),
    _p("copula.grid", type="int", low=4, high=1e5, tunable=True, scale="log", affects=_FID_MARG + _FID_ASSOC,
       direction={f: +1 for f in _FID_MARG + _FID_ASSOC}),
    _p("copula.min_rows", type="int", low=0, high=1e9, tunable=True, scale="log", affects=_FID_ASSOC,
       direction={f: -1 for f in _FID_ASSOC}),
    _p("copula.pooled_cells", type="float", low=0, high=1e12, tunable=True, scale="log", affects=_FID_ASSOC + _JOIN,
       direction={f: +1 for f in _FID_ASSOC + _JOIN}),
    _p("copula.mh_steps", type="int", low=1, high=1e4),
    _p("copula.mcse_z", type="float", low=1e-6, high=1.0),
    _p("marginals.bayesian_bootstrap", type="bool"),
    _p("marginals.dirichlet_prior", type="float", low=0, high=1e6, tunable=True, scale="log",
       affects=_FID_MARG + _NOVEL + _PRIV, direction={f: -1 for f in _FID_MARG} | {f: +1 for f in _NOVEL + _PRIV}),
    _p("marginals.max_levels", type="int", low=1, high=1e7, tunable=True, scale="log", affects=_FID_MARG + _PRIV,
       direction={f: +1 for f in _FID_MARG} | {f: -1 for f in _PRIV}),
    _p("marginals.tail_trim", type="float", low=0, high=0.5, tunable=True, affects=_FID_MARG + _PRIV,
       direction={f: -1 for f in _FID_MARG} | {f: +1 for f in _PRIV}),
    _p("marginals.min_dependence_rows", type="int", low=0, high=1e7, tunable=True, affects=_FID_ASSOC + _PRIV,
       direction={f: -1 for f in _FID_ASSOC} | {f: +1 for f in _PRIV}),
    _p("marginals.fd_bf", type="float", low=1.0, high=1e300),
    _p("marginals.joint_max_levels", type="int", low=1, high=1e7, tunable=True, scale="log",
       affects=_FID_ASSOC + _NOVEL, direction={f: +1 for f in _FID_ASSOC + _NOVEL}),
    _p("marginals.joint_min_cramers_v", type="float", low=0, high=1, tunable=True, affects=_FID_ASSOC + _NOVEL,
       direction={f: -1 for f in _FID_ASSOC + _NOVEL}),
    _p("marginals.discrete_int_max", type="int", low=0, high=1e7),
    _p("marginals.max_obs", type="int", low=10, high=1e9, tunable=True, scale="log", affects=_FID_MARG,
       direction={f: +1 for f in _FID_MARG}),
    _p("marginals.fd_sample_rows", type="int", low=100, high=1e10),
    _p("facts.min_group_rows", type="int", low=0, high=1e9, tunable=True, scale="log", affects=_JOIN + _FID_ASSOC,
       direction={f: -1 for f in _JOIN + _FID_ASSOC}),
    _p("facts.frailty", type="choice", choices=("auto", "on", "off"), tunable=True, affects=_CARD),
    _p("facts.frailty_threshold", type="float", low=1.0, high=1e6, tunable=True, scale="log", affects=_CARD,
       direction={f: -1 for f in _CARD}),
    _p("facts.pairs", type="choice", choices=("auto", "poisson", "occupancy")),
    _p("facts.multiplicity", type="choice", choices=("auto", "none", "partner", "anchor", "block")),
    _p("facts.count_model", type="choice", choices=("auto", "poisson", "dp"), tunable=True, affects=_CARD),
    _p("facts.count_model_bf", type="float", low=1.0, high=1e300, tunable=True, scale="log", affects=_CARD,
       direction={f: -1 for f in _CARD}),
    _p("facts.group_fallback", type="choice", choices=("auto", "anchor", "partner")),
    _p("facts.link_degree", type="choice", choices=("auto", "segment", "dp"), tunable=True, affects=_CARD),
    _p("facts.link_degree_min_rows", type="float", low=0, high=1e9, tunable=True, scale="log", affects=_CARD + _JOIN),
    _p("facts.frailty_min_rows", type="int", low=2, high=1e9),
    _p("facts.occupancy_z", type="float", low=-1e3, high=0),
    _p("facts.occupancy_min_expected", type="float", low=0, high=1e9),
    _p("facts.dp_count_sweeps", type="int", low=1, high=1e5),
    _p("structure.structural_prior", type="float", low=0, high=1, tunable=True, affects=_NOVEL + _FID_ASSOC,
       direction={f: +1 for f in _NOVEL}),
    _p("structure.structural_pooling", type="float", low=0, high=1e6, tunable=True, affects=_NOVEL),
    _p("structure.beta_prior", type="float", low=1e-6, high=1e6, tunable=True, scale="log", affects=_NOVEL + _CARD),
    _p("structure.conditional_inclusion", type="choice", choices=("auto", "none"), tunable=True, affects=_NOVEL),
    _p("structure.inclusion_min_expected", type="float", low=0, high=1e9, tunable=True, scale="log",
       affects=_NOVEL, direction={f: -1 for f in _NOVEL}),
    _p("structure.participation", type="choice", choices=("auto", "none"), tunable=True, affects=_CARD),
    _p("uniqueness.mh_sweeps", type="int", low=0, high=1e6, tunable=True, scale="log", affects=_RULES,
       direction={f: +1 for f in _RULES}),
    _p("uniqueness.grain", type="choice", choices=("auto", "none")),
    _p("uniqueness.rake_rounds", type="int", low=0, high=1e4, tunable=True, affects=_RULES + _FID_MARG,
       direction={f: +1 for f in _RULES}),
    _p("trees.max_depth", type="int", low=1, high=1e6, choices=("auto",)),
    _p("trees.frailty", type="choice", choices=("auto", "on", "off"), tunable=True, affects=_TREE),
    _p("text.order", type="int", low=1, high=12, tunable=True, affects=_TEXT, direction={"text_tstr_ratio": +1}
       | {f: -1 for f in _TEXT if f != "text_tstr_ratio"}),
    _p("text.min_token_rows", type="int", low=1, high=1e6, tunable=True, affects=_TEXT,
       direction={f: +1 for f in _TEXT if f != "text_tstr_ratio"} | {"text_tstr_ratio": -1}),
    _p("text.max_copy_ngram", type="int", low=1, high=1e3, tunable=True, affects=_TEXT,
       direction={f: -1 for f in _TEXT if f != "text_tstr_ratio"}),
    _p("text.backoff_strength", type="float", low=0, high=1e6),
    _p("text.max_retries", type="int", low=0, high=1e6),
    _p("rules.max_rejection_rounds", type="int", low=0, high=1e6, tunable=True, scale="log", affects=_RULES,
       direction={f: +1 for f in _RULES}),
    _p("rules.rake_max_levels", type="int", low=1, high=1e7),
    _p("rules.rake_max_cells", type="int", low=1, high=1e9),
    _p("repair.pair_tries", type="int", low=0, high=1e7, tunable=True, scale="log",
       affects=("pair_repeat_share_diff", "unique_duplicate_rate"), direction={"pair_repeat_share_diff": +1,
                                                                               "unique_duplicate_rate": +1}),
    _p("repair.constraint_rounds", type="int", low=0, high=1e7, tunable=True, scale="log", affects=_RULES,
       direction={f: +1 for f in _RULES}),
    _p("repair.thin_rounds", type="int", low=0, high=1e7),
    _p("repair.tree_redraws", type="int", low=0, high=1e6),
    _p("repair.unique_code_rounds", type="int", low=1, high=1e6),
    _p("repair.participation_rounds", type="int", low=0, high=1e4),
    _p("repair.occupancy_rounds", type="int", low=0, high=1e4),
    _p("detection.group_effect_share", type="float", low=0, high=1),
    _p("detection.scd_stable_share", type="float", low=0, high=1),
    _p("detection.scd_max_versions", type="int", low=1, high=1e6),
    _p("evaluation.thresholds", type="map"),
    _p("evaluation.assoc_max_rows", type="int", low=100, high=1e9),
    _p("evaluation.discriminator_rows", type="int", low=100, high=1e9),
    _p("privacy.suppress_small_counts", type="int", low=0, high=1e6),
    _p("privacy.dp", type="map", nullable=True),
    _p("pushdown.mode", type="choice", choices=("auto", "on", "off")),
    _p("pushdown.min_rows", type="int", low=0, high=1e15),
    _p("pushdown.sample_rows", type="int", low=1, high=1e12, tunable=True, scale="log", affects=_FID_ASSOC + _FID_MARG,
       direction={f: +1 for f in _FID_ASSOC + _FID_MARG}),
    _p("generator", type="str"),
    _p("compute.preset", type="str"),
    _p("compute.file", type="str", nullable=True),
    _p("compute.overrides", type="map"),
    _p("per_table", type="map"),
    _p("settings_file", type="str", nullable=True),
])


def _flat(d: dict, pre: str = "", free=()) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and key not in free and v:
            out.update(_flat(v, key + ".", free))
        else:
            out[key] = v
    return out


def _infer(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "map"
    return "any"


def _free_keys():
    from .schema import FREE
    return set(FREE) | {k for k, m in META.items() if m.get("type") == "map"}


def registry() -> dict[str, Param]:
    """Every engine parameter, keyed by its dotted synthesis key."""
    from .schema import SYNTH_DOC
    from .spec import DEFAULT_SYNTHESIS
    out = {}
    for k, v in _flat(DEFAULT_SYNTHESIS, free=_free_keys()).items():
        m = dict(META.get(k, {}))
        m.setdefault("type", _infer(v))
        out[k] = Param(key=k, default=copy.deepcopy(v), doc=SYNTH_DOC.get(k, ""), **m)
    return out


REGISTRY: dict[str, Param] = {}


def params() -> dict[str, Param]:
    if not REGISTRY:
        REGISTRY.update(registry())
    return REGISTRY


def threshold_keys() -> dict:
    from .evaluate import THRESHOLDS
    return dict(THRESHOLDS)


# ------------------------------------------------------------------ validation
def _get(d, dotted):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return KeyError
        cur = cur[p]
    return cur


def validate(synthesis: dict, where: str = "synthesis") -> list[str]:
    """Type, range and choice errors of a (merged or partial) synthesis block.
    Unknown keys are reported by relsynth.schema.check_keys."""
    errs = []
    if not isinstance(synthesis, dict):
        return [f"{where} must be a mapping"]
    P = params()
    for k, p in P.items():
        v = _get(synthesis, k)
        if v is KeyError:
            continue
        why = p.check(v)
        if why:
            errs.append(f"{where}.{k}: {why}")
    th = synthesis.get("evaluation", {}).get("thresholds") if isinstance(synthesis.get("evaluation"), dict) else None
    if isinstance(th, dict):
        allowed = threshold_keys()
        for k, v in th.items():
            if k not in allowed:
                m = difflib.get_close_matches(str(k), list(allowed), n=1, cutoff=0.6)
                errs.append(f"{where}.evaluation.thresholds: unknown threshold '{k}'"
                            + (f" (did you mean '{m[0]}'?)" if m else f" (allowed: {', '.join(sorted(allowed))})"))
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or v < 0:
                errs.append(f"{where}.evaluation.thresholds.{k}: must be a number >= 0, not {v!r}")
    pt = synthesis.get("per_table")
    if isinstance(pt, dict):
        for t, block in pt.items():
            if not isinstance(block, dict):
                errs.append(f"{where}.per_table.{t}: must be a mapping of groups ({', '.join(TABLE_GROUPS)})")
                continue
            for g, sub in block.items():
                if g not in TABLE_GROUPS:
                    errs.append(f"{where}.per_table.{t}: '{g}' cannot be set per table (allowed: "
                                f"{', '.join(TABLE_GROUPS)})")
                    continue
                if not isinstance(sub, dict):
                    errs.append(f"{where}.per_table.{t}.{g}: must be a mapping")
                    continue
                for kk, vv in sub.items():
                    pk = P.get(f"{g}.{kk}")
                    if pk is None:
                        names = [x.split(".", 1)[1] for x in P if x.startswith(g + ".")]
                        m = difflib.get_close_matches(str(kk), names, n=1, cutoff=0.6)
                        errs.append(f"{where}.per_table.{t}.{g}: unknown key '{kk}'"
                                    + (f" (did you mean '{m[0]}'?)" if m else ""))
                    else:
                        why = pk.check(vv)
                        if why:
                            errs.append(f"{where}.per_table.{t}.{g}.{kk}: {why}")
    return errs


def table_config(cfg: dict, table: str) -> dict:
    """The synthesis block a table's model reads: cfg with the table's own
    per_table groups merged in (cfg itself when the table has none)."""
    over = (cfg.get("per_table") or {}).get(table)
    if not over:
        return cfg
    from .spec import deep_merge
    out = dict(cfg)
    for g, sub in over.items():
        if g in TABLE_GROUPS and isinstance(sub, dict):
            out[g] = deep_merge(cfg.get(g) or {}, sub)
    return out


# ------------------------------------------------------------- settings files
class SettingsError(SpecError):
    """Invalid settings (CLI exit code 2)."""


def resolve_file(ref: str, base: Path | str = ".") -> Path:
    """The path of a settings file reference: preset:NAME or a path (relative to base)."""
    ref = str(ref)
    if ref.startswith("preset:"):
        name = ref.split(":", 1)[1]
        if name not in PRESETS:
            raise SettingsError(f"unknown settings preset {name!r}; choose from {', '.join(PRESETS)}")
        return PRESET_DIR / f"settings_{name}.yaml"
    p = Path(ref).expanduser()
    return p if p.is_absolute() else Path(base) / p


def read_file(ref: str, base: Path | str = ".") -> dict:
    """The synthesis block of a settings file (validated: unknown keys, types, ranges)."""
    from .schema import check_keys
    from .spec import read_yaml
    p = resolve_file(ref, base)
    if not p.exists():
        raise SettingsError(f"settings file {p} not found")
    doc = read_yaml(p) or {}
    if not isinstance(doc, dict):
        raise SettingsError(f"{p}: a settings file must be a mapping")
    syn = doc["synthesis"] if "synthesis" in doc and isinstance(doc["synthesis"], dict) else doc
    extra = [k for k in doc if k != "synthesis"] if "synthesis" in doc else []
    errs = [f"{p}: unknown top-level key '{k}' (a settings file holds only `synthesis`)" for k in extra]
    if "settings_file" in syn:
        errs.append(f"{p}: settings_file cannot be set inside a settings file")
    errs += [e.replace("synthesis.", f"{p.name}: ", 1) for e in check_keys({"tables": {}, "synthesis": syn})
             if e.startswith("synthesis.")]
    errs += validate(syn, where=p.name)
    if errs:
        raise SettingsError("Settings file has errors:\n  - " + "\n  - ".join(errs))
    return syn


def apply_file(raw: dict, base: Path | str = ".") -> dict:
    """raw with its settings file (synthesis.settings_file, else RELSYNTH_SETTINGS)
    merged under its synthesis block.  raw is not changed."""
    from .spec import deep_merge
    syn: dict = raw["synthesis"] if isinstance(raw.get("synthesis"), dict) else {}
    ref = syn.get("settings_file") or os.environ.get(ENV) or None
    if not ref:
        return raw
    file_syn = read_file(ref, base if syn.get("settings_file") else ".")
    out = dict(raw)
    merged = deep_merge(file_syn, syn)
    merged["settings_file"] = str(resolve_file(ref, base if syn.get("settings_file") else "."))
    out["synthesis"] = merged
    return out


# ------------------------------------------------------------------ provenance
def provenance(raw: dict, base: Path | str = ".") -> dict[str, tuple[Any, str]]:
    """{key: (effective value, source)}: source is default, the settings file's
    path, or spec."""
    P = params()
    syn: dict = raw["synthesis"] if isinstance(raw.get("synthesis"), dict) else {}
    ref = syn.get("settings_file") or os.environ.get(ENV) or None
    file_syn = read_file(ref, base if syn.get("settings_file") else ".") if ref else {}
    src_name = str(ref) if ref else "default"
    out = {}
    for k, p in P.items():
        v_spec, v_file = _get(syn, k), _get(file_syn, k)
        if v_spec is not KeyError and k != "settings_file":
            out[k] = (v_spec, "spec")
        elif v_file is not KeyError:
            out[k] = (v_file, src_name)
        else:
            out[k] = (p.default, "default")
    return out


def diff(synthesis: dict) -> dict:
    """{key: value} of the keys of a merged synthesis block that differ from the defaults."""
    out = {}
    for k, p in params().items():
        v = _get(synthesis, k)
        if v is not KeyError and v != p.default:
            out[k] = v
    return out


# --------------------------------------------------------- editable attributes
class _Group:
    """Attribute access to one group of settings, validated on assignment."""

    def __init__(self, owner: Settings, prefix: str, store: dict, table: str | None = None):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_table", table)

    def _keys(self):
        pre = self._prefix + "."
        return sorted({k[len(pre):].split(".")[0] for k in params() if k.startswith(pre)})

    def __dir__(self):
        return self._keys()

    def __getattr__(self, name):
        key = f"{self._prefix}.{name}"
        if key == "evaluation.thresholds":
            return _Thresholds(self._store.setdefault("thresholds", {}))
        if key in params():
            if name in self._store:
                return self._store[name]
            return self._owner._effective(key, self._table)
        if any(k.startswith(key + ".") for k in params()):
            return _Group(self._owner, key, self._store.setdefault(name, {}), self._table)
        raise AttributeError(_unknown(key))

    def __setattr__(self, name, value):
        key = f"{self._prefix}.{name}"
        p = params().get(key)
        if p is None:
            raise AttributeError(_unknown(key))
        if self._table is not None and self._prefix not in TABLE_GROUPS:
            raise AttributeError(f"{key} cannot be set per table (allowed: {', '.join(TABLE_GROUPS)})")
        why = p.check(value)
        if why:
            raise SettingsError(f"{key}: {why}")
        self._store[name] = copy.deepcopy(value)

    def __repr__(self):
        vals = {k: getattr(self, k) for k in self._keys()}
        return f"<settings {self._prefix}: {vals}>"


class _Thresholds:
    def __init__(self, store):
        object.__setattr__(self, "_store", store)

    def __dir__(self):
        return sorted(threshold_keys())

    def __getattr__(self, name):
        th = threshold_keys()
        if name not in th:
            raise AttributeError(f"unknown threshold {name!r} (allowed: {', '.join(sorted(th))})")
        return self._store.get(name, th[name])

    def __setattr__(self, name, value):
        errs = validate({"evaluation": {"thresholds": {name: value}}})
        if errs:
            raise SettingsError(errs[0])
        self._store[name] = value


def _unknown(key):
    m = difflib.get_close_matches(key, list(params()), n=1, cutoff=0.6)
    return f"unknown setting {key!r}" + (f" (did you mean {m[0]!r}?)" if m else "")


class Settings:
    """Engine settings as editable attributes (see the module docstring).

    Only the values set on the object (or loaded from a file or spec) are kept
    as overrides; everything else reads through to the defaults, so
    `to_dict()` is the minimal synthesis block that reproduces the settings."""

    def __init__(self, overrides: dict | None = None):
        object.__setattr__(self, "_over", copy.deepcopy(overrides or {}))
        errs = validate(self._over)
        if errs:
            raise SettingsError("Settings have errors:\n  - " + "\n  - ".join(errs))

    # constructors
    @classmethod
    def defaults(cls) -> Settings:
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> Settings:
        return cls(read_file(str(path)))

    @classmethod
    def load(cls, spec_or_path) -> Settings:
        """The settings in force for a spec: its settings file, then its synthesis block."""
        from .spec import from_ir, read_yaml
        if isinstance(spec_or_path, (str, Path)):
            raw, base = from_ir(read_yaml(spec_or_path)), Path(spec_or_path).parent
        else:
            raw, base = dict(getattr(spec_or_path, "raw", spec_or_path)), Path(".")
        raw = apply_file(raw, base)
        syn = copy.deepcopy(raw.get("synthesis") or {})
        syn.pop("settings_file", None)
        return cls(syn)

    # access
    def __getattr__(self, name):
        if name in params():
            return self._over.get(name, params()[name].default)
        if any(k.startswith(name + ".") for k in params()):
            return _Group(self, name, self._over.setdefault(name, {}))
        raise AttributeError(_unknown(name))

    def __setattr__(self, name, value):
        p = params().get(name)
        if p is None:
            raise AttributeError(_unknown(name))
        why = p.check(value)
        if why:
            raise SettingsError(f"{name}: {why}")
        self._over[name] = copy.deepcopy(value)

    def __dir__(self):
        return sorted({k.split(".")[0] for k in params()})

    def table(self, name: str) -> _TableSettings:
        """Per-table settings of the groups copula, marginals and text."""
        return _TableSettings(self, name)

    def _effective(self, key, table=None):
        v = _get(self._over, key)
        if v is not KeyError:
            return v
        return params()[key].default

    def get(self, key: str):
        if key not in params():
            raise KeyError(_unknown(key))
        return self._effective(key)

    def set(self, key: str, value) -> Settings:
        """Set a dotted key (copula.sample_rows) and return self."""
        p = params().get(key)
        if p is None:
            raise KeyError(_unknown(key))
        why = p.check(value)
        if why:
            raise SettingsError(f"{key}: {why}")
        cur = self._over
        *head, last = key.split(".")
        for h in head:
            cur = cur.setdefault(h, {})
        cur[last] = copy.deepcopy(value)
        return self

    # export
    def to_dict(self) -> dict:
        """The overrides as a synthesis block (empty groups dropped)."""
        def prune(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    v = prune(v)
                    if not v:
                        continue
                out[k] = v
            return out
        return prune(copy.deepcopy(self._over))

    def merged(self) -> dict:
        """The full synthesis block the engine reads (defaults with the overrides)."""
        from .spec import DEFAULT_SYNTHESIS, deep_merge
        return deep_merge(DEFAULT_SYNTHESIS, self.to_dict())

    def to_yaml(self) -> str:
        return yaml.safe_dump({"synthesis": self.to_dict()}, sort_keys=False, width=110)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p

    def apply(self, raw: dict) -> dict:
        """A copy of a raw spec with these settings over its synthesis block."""
        from .spec import deep_merge
        out = copy.deepcopy(raw)
        out["synthesis"] = deep_merge(out.get("synthesis") or {}, self.to_dict())
        return out

    def __repr__(self):
        return f"Settings({self.to_dict()!r})"

    def __eq__(self, other):
        return isinstance(other, Settings) and self.merged() == other.merged()


class _TableSettings:
    def __init__(self, owner: Settings, table: str):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_table", table)

    def __getattr__(self, group):
        if group not in TABLE_GROUPS:
            raise AttributeError(f"{group!r} cannot be set per table (allowed: {', '.join(TABLE_GROUPS)})")
        store = self._owner._over.setdefault("per_table", {}).setdefault(self._table, {}).setdefault(group, {})
        return _Group(self._owner, group, store, table=self._table)


# ------------------------------------------------------------ docs and template
def reference() -> str:
    """Markdown reference of every parameter (docs/settings-reference.md)."""
    P = params()
    out = ["# Settings reference", "",
           "Generated from `relsynth.settings` (`python -m relsynth.settings > docs/settings-reference.md`); "
           "do not edit by hand. See [docs/settings.md](settings.md) for how settings are set.", "",
           "| Key | Type | Default | Range / choices | Tunable | Meaning |", "|---|---|---|---|---|---|"]
    for k, p in P.items():
        rng = ", ".join(str(c) for c in p.choices) if p.type == "choice" else (
            f"{_num(p.low)} … {_num(p.high)}" if p.low is not None or p.high is not None else "")
        if p.choices and p.type != "choice":
            rng += " or " + ", ".join(map(str, p.choices))
        tun = ("yes" + (" (log)" if p.scale == "log" else "")) if p.tunable else ""
        out.append(f"| `{k}` | {p.type} | `{p.default}` | {rng} | {tun} | {p.doc} |")
    out += ["", "## `evaluation.thresholds`", "",
            "Fixed tolerances of the scorecard. Each item's limit is the larger of its tolerance and its sampling "
            "limit, so lowering a tolerance never makes a check stricter than its noise allows.", "",
            "| Key | Default |", "|---|---|"]
    out += [f"| `{k}` | `{v}` |" for k, v in threshold_keys().items()]
    return "\n".join(out) + "\n"


def _num(x):
    if x is None:
        return ""
    return f"{x:g}" if isinstance(x, float) else str(x)


def template() -> str:
    """A settings file with every key at its default, commented."""
    from .spec import DEFAULT_SYNTHESIS
    P = params()
    lines = ["# relsynth settings file: every engine parameter at its default.",
             "# Keep only the keys you change; the rest read through to the defaults.",
             "# Use it with `synthesis.settings_file: <this file>`, `--settings <this file>` or RELSYNTH_SETTINGS.",
             "synthesis:"]

    def walk(d, pre, ind):
        for k, v in d.items():
            key = f"{pre}{k}"
            if key in ("settings_file",):
                continue
            p = P.get(key)
            if isinstance(v, dict) and p is None:
                lines.append(f"{'  ' * ind}{k}:")
                walk(v, key + ".", ind + 1)
                continue
            doc = (p.doc if p else "").replace("\n", " ")
            if len(doc) > 100:
                doc = doc[:97] + "..."
            val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
            lines.append(f"{'  ' * ind}{k}: {val}" + (f"   # {doc}" if doc else ""))
            if key == "evaluation.thresholds":
                for tk, tv in threshold_keys().items():
                    lines.append(f"{'  ' * (ind + 1)}# {tk}: {tv}")
    walk(DEFAULT_SYNTHESIS, "", 1)
    return "\n".join(lines) + "\n"


def search_space(keys=None) -> dict[str, Param]:
    """The tunable parameters (or the named ones): the tuner's search space."""
    P = params()
    if keys is None:
        return {k: p for k, p in P.items() if p.tunable}
    bad = [k for k in keys if k not in P]
    if bad:
        raise SettingsError("; ".join(_unknown(k) for k in bad))
    return {k: P[k] for k in keys}


if __name__ == "__main__":
    print(reference(), end="")

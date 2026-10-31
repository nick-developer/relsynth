# SPDX-License-Identifier: Apache-2.0
"""The Python API of relsynth.

    import relsynth as rs

    spec = rs.introspect("postgresql://analyst@db/sales", schema="dw",
                         password_env="PGPASSWORD")         # draft a specification
    spec.save("sales.yaml")                                # review, edit, commit
    spec = rs.Spec.load("sales.yaml")
    spec.check()                                           # raises SpecError when invalid

    model = rs.fit(spec)                                   # Bayesian model (large facts in-database)
    data = model.sample(scale=2.0)                         # one synthetic database
    report = model.evaluate(data)                          # quality scorecard
    report.require(1.0)                                    # quality gate (raises QualityGateError)
    q = rs.assess_quality(spec, data)                      # calibrated 2.0 report: fidelity, utility, integrity
    data.write()                                           # to the spec's target
    model.export("model/")                                 # the model as JSON / Markdown / PDF / Python

    result = rs.run(spec, scale=2.0)                       # all of the above in one call
    rs.doctor(result).fixed_spec(spec).save("sales.fixed.yaml")

Every function accepts a `Spec`, a path to a YAML spec or a spec dictionary.
"""
from __future__ import annotations

import importlib
import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

from .errors import EngineError, QualityGateError, RelsynthError, SourceError
from .spec import SpecError, from_ir, load_spec, normalise, read_spec, validate

log = logging.getLogger("relsynth")

__all__ = ["Spec", "FittedModel", "SyntheticData", "EvaluationReport", "QualityReport", "RunResult", "Diagnosis",
           "introspect", "validate_spec", "fit", "sample", "evaluate", "assess_quality", "run", "doctor", "tune",
           "export_model", "verify", "clone_schema", "profile", "ping", "connect"]


# ---------------------------------------------------------------------- spec
class Spec:
    """A relsynth specification: the tables, their columns and roles, the
    relationships, the business rules, and the synthesis settings.

    `raw` is the YAML document as written (what `save` writes back); `resolved`
    is the validated form with every default filled in (what the engine reads).
    Relative paths in source and target sections resolve against `base_dir` (the
    spec file's folder)."""

    def __init__(self, raw: dict, base_dir: str | Path | None = None, path: str | Path | None = None):
        if not isinstance(raw, dict):
            raise SpecError("a spec is a mapping (YAML document)")
        # an IR 2.0 document is held as the 1.0 spec the engine runs (its relational
        # part); relsynth.ir reads and writes the whole document
        self.raw = from_ir(raw)
        self.path = Path(path) if path else None
        self.base_dir = Path(base_dir) if base_dir else (self.path.parent if self.path else Path("."))

    # construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Spec:
        """Read a YAML spec (safe loader: no Python objects, no aliases bombs)."""
        p = Path(path)
        if not p.is_file():
            raise SpecError(f"spec file not found: {p}")
        return cls(read_spec(p), path=p)

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | Path | None = None) -> Spec:
        return cls(copy.deepcopy(d), base_dir=base_dir)

    @classmethod
    def coerce(cls, spec: Spec | str | Path | dict) -> Spec:
        """A Spec from a Spec, a YAML path or a dictionary."""
        if isinstance(spec, Spec):
            return spec
        if isinstance(spec, (str, Path)):
            return cls.load(spec)
        if isinstance(spec, dict):
            return cls.from_dict(spec, base_dir=spec.get("_base_dir"))
        raise TypeError(f"expected a Spec, a path or a dict, got {type(spec).__name__}")

    # views ----------------------------------------------------------------
    @property
    def resolved(self) -> dict:
        """The validated spec with defaults (raises SpecError when invalid)."""
        d = copy.deepcopy(self.raw)
        d.pop("_base_dir", None)
        s = load_spec(d)
        s["_base_dir"] = str(self.base_dir.resolve())
        return s

    @property
    def tables(self) -> list[str]:
        return list(self.raw.get("tables") or {})

    @property
    def relationships(self) -> list[dict]:
        return list(self.raw.get("relationships") or [])

    @property
    def review(self) -> list[str]:
        """The inferences `introspect` asks a person to confirm."""
        return list(self.raw.get("review") or [])

    def __repr__(self) -> str:
        where = f" from {self.path}" if self.path else ""
        return f"<Spec {self.raw.get('name', '')!r}: {len(self.tables)} tables, {len(self.relationships)} relationships{where}>"

    # checks ---------------------------------------------------------------
    def errors(self) -> list[str]:
        """Every validation error (empty when the spec is valid)."""
        try:
            d = copy.deepcopy(self.raw)
            d.pop("_base_dir", None)
            return validate(normalise(json.loads(json.dumps(d, default=str))))
        except SpecError as e:
            return [str(e)]

    def check(self) -> Spec:
        """Raise SpecError listing every problem when the spec is invalid."""
        errs = self.errors()
        if errs:
            raise SpecError("invalid spec:\n  - " + "\n  - ".join(errs))
        return self

    # edits ----------------------------------------------------------------
    def copy(self) -> Spec:
        return Spec(copy.deepcopy(self.raw), base_dir=self.base_dir, path=self.path)

    def set(self, key: str, value: Any) -> Spec:
        """Set a dotted key (synthesis.scale.default, target.schema,
        tables.orders.mode) on a copy and return it.  Values given as strings are
        parsed as YAML scalars ("2" -> 2, "true" -> True, "[a, b]" -> list)."""
        if isinstance(value, str) and value.strip().lower() not in _WORDS:
            try:
                value = yaml.safe_load(value) if value.strip() else value
            except yaml.YAMLError:
                pass
        out = self.copy()
        cur = out.raw
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise SpecError("empty key")
        for p in parts[:-1]:
            nxt = cur.get(p)
            if nxt is None:
                nxt = cur[p] = {}
            if not isinstance(nxt, dict):
                raise SpecError(f"{key}: {p} is not a mapping")
            cur = nxt
        cur[parts[-1]] = value
        return out

    def with_source(self, source) -> Spec:
        """A copy reading from another source (a spec section or a location such as
        'postgresql://...', 'sqlite:file.db', 'csv:dir', 'parquet:dir')."""
        return self.set("source", _section(source))

    def with_target(self, target) -> Spec:
        """A copy writing to another target (spec section or location)."""
        return self.set("target", _section(target))

    # output ---------------------------------------------------------------
    def to_yaml(self, mask_secrets: bool = True) -> str:
        from .data import mask_spec_urls
        d = {k: v for k, v in self.raw.items() if k != "_base_dir"}
        if mask_secrets:
            d = mask_spec_urls(d)
        return yaml.safe_dump(json.loads(json.dumps(d, default=str)), sort_keys=False, width=110,
                              allow_unicode=True)

    def save(self, path: str | Path, mask_secrets: bool = True) -> Path:
        """Write the spec as YAML.  Passwords in URLs are masked (the spec then
        reads the URL from source.url_env / target.url_env)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(mask_secrets), encoding="utf-8")
        return p

    def model_report(self, out_stem: str | Path | None = None) -> str:
        """The data model as Markdown (tables, keys, an ER diagram in Mermaid,
        rules); also written to <out_stem>.md and .json when out_stem is given."""
        from .model_report import describe, to_markdown
        d = normalise(json.loads(json.dumps({k: v for k, v in self.raw.items() if k != "_base_dir"}, default=str)))
        d["review"] = self.review
        m = describe(d)
        md = to_markdown(d, m)
        if out_stem is not None:
            Path(f"{out_stem}.md").write_text(md, encoding="utf-8")
            Path(f"{out_stem}.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
        return md


SpecLike = "Spec | str | Path | dict"
# words YAML 1.1 would turn into booleans; as --set values they are settings (pushdown on / off)
_WORDS = {"on", "off", "yes", "no", "y", "n"}


def _section(loc) -> dict:
    if isinstance(loc, dict):
        return dict(loc)
    from .pipeline import parse_source
    return parse_source(str(loc))


def _resolved(spec) -> dict:
    return Spec.coerce(spec).resolved


def _scale_section(scale, spec: dict) -> dict | None:
    """A scale argument as the spec's scale section: a number (every entity), a
    section {"default": x, "tables": {...}}, or {table: factor} (the others keep
    the spec's scale)."""
    if scale is None:
        return None
    base = spec["synthesis"]["scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float, Mapping)):
        raise SpecError(f"scale must be a number or a mapping, not {type(scale).__name__}")
    if isinstance(scale, (int, float)):
        return {**base, "default": float(scale)}
    if set(scale) <= {"default", "tables"}:
        return {"default": float(scale.get("default", base.get("default", 1.0))),
                "tables": {**(base.get("tables") or {}), **(scale.get("tables") or {})}}
    unknown = sorted(set(scale) - set(spec["tables"]))
    if unknown:
        raise SpecError(f"scale names unknown tables: {', '.join(map(str, unknown))}")
    return {**base, "tables": {**(base.get("tables") or {}), **{t: float(v) for t, v in scale.items()}}}


def _raise_first(run, allow=()):
    """Re-raise the first failed step of a pipeline run as a RelsynthError."""
    for step, exc in run.exceptions:
        if step in allow:
            continue
        if isinstance(exc, (RelsynthError, SpecError)):
            raise exc
        cls = SourceError if step in ("load",) or step.startswith("write") else EngineError
        from .errors import exit_code_for
        if exit_code_for(exc) == 3:
            cls = SourceError
        raise cls(f"{step} failed: {exc}") from exc


# ---------------------------------------------------------------------- data
class SyntheticData(Mapping):
    """One synthetic database: a read-only mapping table name -> DataFrame, in the
    spec's table order."""

    def __init__(self, tables: dict[str, pd.DataFrame], spec: dict, dataset: int = 0, sample_log=None):
        self._tables = dict(tables)
        self.spec = spec
        self.dataset = dataset
        self.sample_log = sample_log or {}

    def __getitem__(self, t):
        return self._tables[t]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tables)

    def __len__(self):
        return len(self._tables)

    def __repr__(self):
        return f"<SyntheticData dataset {self.dataset}: " + ", ".join(f"{t} {len(d):,}" for t, d in self.items()) + ">"

    @property
    def rows(self) -> dict[str, int]:
        return {t: len(d) for t, d in self.items()}

    def to_dict(self) -> dict[str, pd.DataFrame]:
        return dict(self._tables)

    def write(self, target=None, *, if_exists: str | None = None, structure: str | None = None) -> dict:
        """Write the tables (parents first) to `target` (a spec target section or a
        location; default the spec's target).  if_exists: replace, append, truncate
        or fail; structure: auto, clone, infer or existing.  Returns the target."""
        from .data import write_tables
        from .pipeline import dataset_target
        tgt = _section(target) if target is not None else dataset_target(self.spec, None, self.dataset)
        if if_exists:
            tgt["if_exists"] = if_exists
        if structure:
            tgt["structure"] = structure
        try:
            write_tables(self.spec, self._tables, tgt)
        except (RelsynthError, SpecError):
            raise
        except Exception as e:  # noqa: BLE001
            raise SourceError(f"writing to {tgt.get('kind')} failed: {e}") from e
        return tgt

    def to_csv(self, folder: str | Path) -> Path:
        self.write({"kind": "csv", "path": str(folder)})
        return Path(folder)

    def to_parquet(self, folder: str | Path) -> Path:
        self.write({"kind": "parquet", "path": str(folder)})
        return Path(folder)


# ---------------------------------------------------------------------- report
@dataclass
class EvaluationReport:
    """The comparison of synthetic with real data: every metric (`raw`) and the
    scorecard of pass/fail checks, each with its limit and significance level."""
    raw: dict

    @property
    def scorecard(self) -> dict:
        return self.raw.get("scorecard") or {}

    @property
    def passed(self) -> int:
        return int(self.scorecard.get("passed", 0))

    @property
    def total(self) -> int:
        return int(self.scorecard.get("total", 0))

    @property
    def score(self) -> float:
        """Share of scorecard checks passed (1.0 when there are none)."""
        return self.passed / self.total if self.total else 1.0

    @property
    def failed(self) -> list[dict]:
        return list(self.scorecard.get("failed", []))

    @property
    def ok(self) -> bool:
        return self.passed == self.total

    def require(self, min_score: float = 1.0) -> EvaluationReport:
        """Quality gate: raise QualityGateError when the share of passed checks is
        below min_score."""
        if self.score + 1e-12 < min_score:
            worst = "; ".join(f"{f['area']}: {f['metric']} = {f['value']} (limit {f['limit']})"
                              for f in self.failed[:5])
            raise QualityGateError(f"scorecard {self.passed}/{self.total} = {self.score:.3f} below "
                                   f"{min_score:.3f}" + (f": {worst}" if worst else ""), report=self)
        return self

    def to_dict(self) -> dict:
        return self.raw

    def to_json(self, path: str | Path | None = None) -> str:
        from .diagnostics import _jsonable
        txt = json.dumps(_jsonable(self.raw), indent=1)
        if path is not None:
            Path(path).write_text(txt, encoding="utf-8")
        return txt

    def to_markdown(self) -> str:
        lines = [f"# Scorecard: {self.passed}/{self.total} checks passed ({self.score:.1%})", ""]
        if self.failed:
            lines += ["| area | metric | value | limit |", "|---|---|---|---|"]
            lines += [f"| {f['area']} | {f['metric']} | {f['value']} | {f['limit']} |" for f in self.failed]
        else:
            lines.append("Every check passed.")
        return "\n".join(lines) + "\n"

    def __repr__(self):
        return f"<EvaluationReport {self.passed}/{self.total} passed>"


# ---------------------------------------------------------------------- model
class FittedModel:
    """A fitted Bayesian relational model: the posterior over latent segments, link
    and count rates, and per-segment attribute models (relsynth.engine.Synthesizer)."""

    def __init__(self, spec: dict, synthesizer, fit_data=None):
        self.spec = spec
        self.synthesizer = synthesizer
        self.fit_data = fit_data
        self._next = 0

    def __repr__(self):
        return f"<FittedModel {len(self.spec['tables'])} tables>"

    @property
    def pushdown(self) -> dict:
        """Which facts were fitted from in-database statistics and samples."""
        return self.fit_data.plan.to_dict() if self.fit_data is not None else {}

    @property
    def log(self) -> dict:
        """Decisions, warnings and timings of the fit."""
        return self.synthesizer.log

    def summary(self) -> dict:
        """Segments, relations and attribute models in brief (JSON-ready)."""
        return self.synthesizer.model_summary()

    def sample(self, scale: float | dict | None = None, dataset: int | None = None) -> SyntheticData:
        """One synthetic database.  dataset numbers independent posterior draws
        (0, 1, ...; default: the next one); scale multiplies the entity sizes (a
        number for every entity, {table: factor} for some, or the spec's section
        {"default": x, "tables": {name: y}})."""
        from .data import coerce
        if dataset is None:
            dataset, self._next = self._next, self._next + 1
        sc = _scale_section(scale, self.spec)
        try:
            out = self.synthesizer.sample(scale=sc, index=dataset)
        except (RelsynthError, SpecError):
            raise
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"sampling failed: {e}") from e
        out = {t: coerce(df, self.spec["tables"][t]) for t, df in out.items()}
        return SyntheticData(out, self.spec, dataset, dict(self.synthesizer.sample_log))

    def evaluate(self, data: SyntheticData | Mapping, privacy: bool = True) -> EvaluationReport:
        """Scorecard of `data` against the source the model was fitted on."""
        from .evaluate import evaluate as _evaluate
        if self.fit_data is None:
            raise RelsynthError("this model has no source data attached; use relsynth.evaluate(spec, data)")
        log_ = getattr(data, "sample_log", None) or self.synthesizer.sample_log
        rep = _evaluate(self.spec, self.fit_data.tables, dict(data), privacy=privacy, sample_log=log_,
                        real_stats=self.fit_data.stats)
        return EvaluationReport(rep)

    def export(self, out_dir: str | Path, formats=("json", "md", "pdf", "py"), draws: int = 5,
               replica_check: bool = True, stan: bool = False) -> dict:
        """Write the model (structure, posterior means and draws, verification
        checks) as JSON, Markdown, PDF and a standalone Python sampler."""
        from .pipeline import export_model_for
        return export_model_for(self.spec, out_dir, draws=draws, formats=formats, replica_check=replica_check,
                                stan=stan, synthesizer=self.synthesizer, fit_data=self.fit_data)


# ---------------------------------------------------------------------- functions
def connect(source, **kw):
    """A connector to a database or folder (relsynth.connectors.connect)."""
    from .connectors import connect as _connect
    return _connect(source, **kw)


def introspect(source, *, schema: str | None = None, password_env: str | None = None, url_env: str | None = None,
               name: str = "draft", sample_rows: int = 20000, include=None, exclude=None,
               evidence: str = "names+data", declared: bool = True) -> Spec:
    """Draft a spec from a database or folder: tables, column types, keys and
    links (declared in the catalog, or identified from the data), roles, rules.
    Review `spec.review` before running.  source: a location ('postgresql://...',
    'sqlite:file.db', 'csv:dir', 'parquet:dir') or a spec source section."""
    from .introspect import introspect as _introspect
    sec = _section(source)
    for k, v in (("schema", schema), ("password_env", password_env), ("url_env", url_env)):
        if v:
            sec[k] = v
    try:
        raw = _introspect(sec, name=name, sample_rows=sample_rows, include=include, exclude=exclude,
                          evidence=evidence, declared=declared)
    except (RelsynthError, SpecError):
        raise
    except Exception as e:  # noqa: BLE001
        from .errors import exit_code_for
        cls = SourceError if exit_code_for(e) == 3 else EngineError
        raise cls(f"introspection failed: {e}") from e
    return Spec(raw)


def validate_spec(spec) -> list[str]:
    """Every validation error of a spec (empty list when valid)."""
    return Spec.coerce(spec).errors()


def fit(spec, *, pushdown: str | None = None, seed: int | None = None) -> FittedModel:
    """Fit the model to the spec's source.  Fact tables of at least
    synthesis.pushdown.min_rows rows are fitted from in-database statistics and a
    uniform sample (pushdown: 'auto' (default), 'on' or 'off')."""
    from .engine import Synthesizer
    from .pushdown import load_for_fit
    s = _resolved(spec)
    if seed is not None:
        s["synthesis"]["seed"] = int(seed)
    try:
        data = load_for_fit(s, mode=pushdown)
    except (RelsynthError, SpecError):
        raise
    except Exception as e:  # noqa: BLE001
        from .errors import exit_code_for
        cls = SourceError if exit_code_for(e) == 3 or isinstance(e, (FileNotFoundError, ValueError)) else EngineError
        raise cls(f"loading the source failed: {e}") from e
    try:
        syn = Synthesizer(s).fit(data.tables, stats=data.stats)
    except (RelsynthError, SpecError):
        raise
    except Exception as e:  # noqa: BLE001
        raise EngineError(f"fitting failed: {e}") from e
    return FittedModel(s, syn, data)


def sample(spec, scale=None, *, dataset: int = 0, pushdown: str | None = None) -> SyntheticData:
    """Fit and draw one synthetic database (see fit and FittedModel.sample)."""
    return fit(spec, pushdown=pushdown).sample(scale=scale, dataset=dataset)


def evaluate(spec, synthetic: SyntheticData | Mapping, *, real: Mapping | None = None,
             privacy: bool = True) -> EvaluationReport:
    """Scorecard of synthetic tables against real ones (default: the spec's
    source, with large facts as samples plus exact statistics)."""
    from .evaluate import evaluate as _evaluate
    s = _resolved(spec)
    real, stats = _load_real(s, real)
    from .data import coerce
    syn = {t: coerce(df, s["tables"][t]) for t, df in dict(synthetic).items()}
    return EvaluationReport(_evaluate(s, real, syn, privacy=privacy,
                                      sample_log=getattr(synthetic, "sample_log", None), real_stats=stats))


def _load_real(s: dict, real: Mapping | None):
    """Real tables as evaluate reads them: the spec's source (large facts as a
    sample plus exact statistics) or given frames, prepared like the source."""
    if real is None:
        from .pushdown import load_for_fit
        fd = load_for_fit(s)
        return fd.tables, fd.stats
    from .data import coerce
    from .prepare import computed_columns, prepare_source
    comp = computed_columns(s)
    return prepare_source(s, {t: coerce(df, s["tables"][t], optional=set(comp[t])) for t, df in real.items()})[0], {}


@dataclass
class QualityReport:
    """The 2.0 quality report (`raw`, schema quality-report-2.0): every check with
    its definition, null limit and confidence level, and the quality vector
    (fidelity, utility, integrity).  `scorecard` is the 1.0 report it includes."""
    raw: dict
    scorecard: EvaluationReport

    @property
    def vector(self) -> dict:
        return self.raw["vector"]

    @property
    def ok(self) -> bool:
        return bool(self.vector["verdict"]["pass"])

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.raw["checks"] if not c["pass"]]

    def require(self, areas=("fidelity", "utility", "integrity")) -> QualityReport:
        """Quality gate: raise QualityGateError when one of `areas` fails its pass rule."""
        bad = [a for a in areas if not self.vector[a]["pass"]]
        if bad:
            msg = "; ".join(f"{a}: {self.vector[a]['failed']} failed, {self.vector[a]['allowed_failures']} allowed"
                            for a in bad)
            raise QualityGateError(f"quality vector fails ({msg})", report=self)
        return self

    def to_dict(self) -> dict:
        return self.raw

    def to_json(self, path: str | Path | None = None) -> str:
        txt = json.dumps(self.raw, indent=1)
        if path is not None:
            Path(path).write_text(txt, encoding="utf-8")
        return txt


def assess_quality(spec, synthetic: SyntheticData | Mapping, *, real: Mapping | None = None,
            holdout: Mapping | None = None, config: dict | None = None, privacy: bool = True) -> QualityReport:
    """The calibrated 2.0 quality report of synthetic tables (relsynth.quality.assess).
    real: the tables the model was fitted on (default: the spec's source);
    holdout: real tables the model did not see, for TSTR/TRTS (default: an internal
    split, reported as optimistic); config: overrides of relsynth.quality.DEFAULTS
    (scale defaults to the spec's synthesis.scale)."""
    from .data import coerce
    from .evaluate import evaluate as _evaluate
    from .quality import assess
    from .quality.report import resolve_config
    try:
        resolve_config(config)
    except ValueError as e:
        raise SpecError(f"quality config: {e}") from e
    s = _resolved(spec)
    real, stats = _load_real(s, real)
    syn = {t: coerce(df, s["tables"][t]) for t, df in dict(synthetic).items()}
    base = _evaluate(s, real, syn, privacy=privacy, sample_log=getattr(synthetic, "sample_log", None),
                     real_stats=stats)
    hold = None if holdout is None else {t: coerce(df, s["tables"][t]) for t, df in dict(holdout).items()}
    config = dict(config or {})
    if config.get("scale") is None:
        config["scale"] = s["synthesis"]["scale"]       # the size the synthetic data was drawn at
    raw = assess(s, real, syn, holdout=hold, base=base, config=config)
    return QualityReport(raw, EvaluationReport(base))


@dataclass
class RunResult:
    """One pipeline run: the synthetic data, the model, the scorecard, the
    diagnostics bundle (for `doctor`) and the errors of failed steps."""
    bundle: dict
    data: SyntheticData | None
    model: FittedModel | None
    report: EvaluationReport | None
    errors: list = field(default_factory=list)
    written: list = field(default_factory=list)
    exceptions: list = field(default_factory=list, repr=False)    # (step, exception)

    @property
    def ok(self) -> bool:
        return not self.errors

    def save_bundle(self, path: str | Path) -> Path:
        from .diagnostics import write_bundle
        write_bundle(self.bundle, path)
        return Path(path)


def run(spec, *, scale=None, out=None, datasets: int | None = None, mode: str | None = None,
        pushdown: str | None = None, seed: int | None = None, privacy: bool = True, write: bool = True,
        bundle: str | Path | None = None, fail_under: float | None = None, raise_errors: bool = True) -> RunResult:
    """The whole pipeline: load (with pushdown) -> fit -> sample -> write ->
    evaluate -> diagnostics bundle.  out: folder or target section (default: the
    spec's target); datasets: independent posterior draws to write; mode: 'draw'
    (default) or 'mean'; fail_under: quality gate on the share of passed checks.
    raise_errors=False returns a RunResult with `errors` instead of raising (the
    bundle explains them; see doctor)."""
    from .pipeline import execute
    r = execute(_resolved(spec), scale=scale, out=out if write else False, bundle_path=bundle, privacy=privacy,
                datasets=datasets, mode=mode, pushdown=pushdown, seed=seed, write=write)
    model = FittedModel(r.spec, r.synthesizer, r.fit_data) if r.synthesizer is not None else None
    data = SyntheticData(r.data, r.spec, 0, dict(r.synthesizer.sample_log)) if r.data is not None else None
    rep = EvaluationReport(r.report) if r.report is not None else None
    res = RunResult(bundle=r.bundle, data=data, model=model, report=rep, errors=r.errors, written=r.written,
                    exceptions=r.exceptions)
    if raise_errors:
        _raise_first(r)
        if fail_under is not None and rep is not None:
            rep.require(fail_under)
    return res


@dataclass
class Diagnosis:
    """What `doctor` found in a diagnostics bundle: issues (each with its cause,
    evidence and, where possible, a spec patch) and the merged patches."""
    issues: list
    patches: dict
    markdown: str

    def fixed_spec(self, spec) -> Spec:
        """A copy of the spec with every patch applied."""
        D = importlib.import_module("relsynth.doctor")  # the module, not the API function
        s = Spec.coerce(spec)
        return Spec(D.apply_patches(copy.deepcopy(s.raw), self.patches), base_dir=s.base_dir)

    def __repr__(self):
        return f"<Diagnosis {len(self.issues)} issue(s), {len(self.patches)} patch(es)>"


def extend(model: FittedModel, until, *, base: SyntheticData | Mapping | None = None, dataset: int = 0,
           start=None, step: str = "day") -> list:
    """Continue the event histories (tables with a `sequence` block) of synthetic
    dataset `dataset` past its end: [(period start, period end, {table: new
    rows})] per step.  The rows up to a day do not depend on how far or in how
    many steps the history is extended (relsynth.timeseries.incremental)."""
    from .timeseries.incremental import extend as _extend
    try:
        return _extend(model, until, base=base, dataset=dataset, start=start, step=step)
    except (RelsynthError, SpecError):
        raise
    except ValueError as e:
        raise SpecError(str(e)) from e


def doctor(bundle: RunResult | dict | str | Path) -> Diagnosis:
    """Diagnose a run from its bundle (a RunResult, a bundle dict or a bundle
    file): failed checks and errors mapped to causes and spec fixes."""
    D = importlib.import_module("relsynth.doctor")  # the module, not the API function
    from .diagnostics import read_bundle
    if isinstance(bundle, RunResult):
        b = bundle.bundle
    elif isinstance(bundle, (str, Path)):
        b = read_bundle(bundle)
    else:
        b = bundle
    issues = D.diagnose(b)
    patches = D.merge_patches(issues)
    return Diagnosis(issues, patches, D.render(issues, patches, b))


def tune(spec, rounds: int = 3) -> tuple[Spec, dict, list]:
    """run -> doctor -> apply until no fix remains: (best spec, its bundle, history)."""
    from .pipeline import tune as _tune
    s = Spec.coerce(spec)
    raw = copy.deepcopy(s.raw)
    raw["_base_dir"] = str(s.base_dir.resolve())
    best, b, hist = _tune(raw, rounds=rounds)
    best = {k: v for k, v in best.items() if k != "_base_dir"}
    return Spec(best, base_dir=s.base_dir), b, hist


def verify(spec, *, draws: int = 5, stability: bool = True, out=None, privacy: bool = True,
           pushdown: str = "off", seed: int | None = None):
    """Model checks on real data, where no ground truth exists (R1-R7: scorecard,
    copula convergence, posterior predictive checks of key statistics, scale rule,
    structural validity, stability across seeds, reproducibility).  Returns a
    relsynth.verify.VerifyReport (passed, total, ok, failed(), to_markdown(),
    to_json(), save_bundle()).  stability=False skips the two extra fits."""
    V = importlib.import_module("relsynth.verify")
    return V.verify(spec, draws=draws, stability=stability, out=out, privacy=privacy, pushdown=pushdown, seed=seed)


def export_model(spec, out_dir, **kw) -> dict:
    """Fit and export the model (see FittedModel.export)."""
    from .pipeline import export_model_for
    return export_model_for(_resolved(spec), out_dir, **kw)


def clone_schema(source, target, *, tables: list[str] | None = None, schema: str | None = None,
                 replace: bool = False) -> list[str]:
    """Create the source's tables (types, primary and unique keys, foreign keys,
    indexes, identity columns) empty in a target database.  Returns the tables
    created.  source/target: locations or spec sections of two databases."""
    from .connectors import from_spec
    from .connectors.ddl import clone_metadata, create_tables, load_order
    src = from_spec(_section(source))
    tgt = from_spec(_section(target), create=True)
    try:
        if not hasattr(src, "engine") or not hasattr(tgt, "engine"):
            raise SourceError("clone_schema needs two databases (files have no DDL)")
        names = tables or src.tables()
        spec_like = {"tables": {t: {} for t in names}, "relationships": [
            {"child": t, "parent": fk.ref_table, "kind": "many_to_one"} for t in names for fk in src.describe(t).foreign_keys
            if fk.ref_table in names and fk.ref_table != t]}
        order = load_order(spec_like, list(names))
        sch = schema or tgt.schema
        if hasattr(tgt, "create_schema"):
            tgt.create_schema(sch)
        md = clone_metadata(src, order, tgt.dialect, sch)
        if hasattr(tgt, "adapt_metadata"):
            md = tgt.adapt_metadata(md)
        create_tables(tgt.engine, md, order, replace=replace, schema=sch)
        return order
    finally:
        src.close()
        tgt.close()


def profile(source, *, tables: list[str] | None = None, top: int = 0) -> dict:
    """Row counts and per-column null and distinct counts computed inside the
    source (no rows leave the database).  top > 0 adds the most frequent values
    of each column - only use it where those values may be shared."""
    from .connectors import from_spec
    conn = from_spec(_section(source))
    try:
        out = {}
        for t in tables or conn.tables():
            info = conn.describe(t)
            st = conn.stats(t)
            cols = {}
            from .connectors.sql import is_binary_type
            for c in info.columns:
                d = {"type": c.sql_type, "nullable": c.nullable, "non_null": st.notnull(c.name)}
                if not is_binary_type(c.sql_type):
                    try:
                        d["distinct"] = int(len(st.distinct(c.name)))
                    except Exception as e:  # noqa: BLE001 - e.g. LOB columns
                        d["distinct_error"] = str(e).splitlines()[0][:200]
                    if top:
                        vc = st.counts(c.name).sort_values("n", ascending=False).head(top)
                        d["top"] = [[None if pd.isna(v) else str(v), int(n)] for v, n in zip(vc[c.name], vc["n"])]
                cols[c.name] = d
            out[t] = {"rows": st.n_rows, "primary_key": info.primary_key,
                      "foreign_keys": [fk.to_dict() for fk in info.foreign_keys], "columns": cols}
        return out
    finally:
        conn.close()


def ping(source) -> dict:
    """Connect to a source or target and report what is there (no data read)."""
    import time

    from .connectors import from_spec
    t0 = time.time()
    try:
        conn = from_spec(_section(source))
    except Exception as e:  # noqa: BLE001 - reported to the caller
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 3)}
    try:
        out = conn.ping()
        out["seconds"] = round(time.time() - t0, 3)
        return out
    finally:
        conn.close()

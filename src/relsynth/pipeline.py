# SPDX-License-Identifier: Apache-2.0
"""The synthesis pipeline shared by the Python API (relsynth.api) and the CLI.

    load (large facts pushed down) -> fit -> sample -> write -> evaluate -> bundle

`execute` runs it and returns a `PipelineRun` with every intermediate result and
the exceptions of failed steps; `run_pipeline` is the bundle-first form (it never
raises: errors go into the diagnostics bundle so that `doctor` can explain them).
`tune` loops run -> doctor -> apply; `export_model_for` fits and exports the model.
"""
from __future__ import annotations

import importlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .data import coerce, write_tables
from .diagnostics import build_bundle, write_bundle
from .engine import Synthesizer
from .evaluate import evaluate
from .spec import load_spec, normalise, read_yaml

D = importlib.import_module("relsynth.doctor")  # the module, not the API function

log = logging.getLogger("relsynth")


def parse_source(s: str) -> dict:
    """A spec source/target section from a command-line location: csv:DIR,
    files:DIR (CSV and Parquet files), parquet:DIR, sqlite:FILE or a database URL."""
    if s.startswith("parquet:"):
        return {"kind": "parquet", "path": s[len("parquet:"):]}
    for pre in ("csv:", "files:"):
        if s.startswith(pre):
            return {"kind": "csv", "path": s[len(pre):]}
    if s.startswith("sqlite:") and not s.startswith("sqlite://"):
        return {"kind": "sqlite", "path": s[len("sqlite:"):]}
    if "://" not in s:
        p = Path(s)
        if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            return {"kind": "sqlite", "path": s}
        return {"kind": "csv", "path": s}
    scheme = s.split(":", 1)[0].split("+", 1)[0].lower()
    kind = {"postgres": "postgresql", "postgresql": "postgresql", "mssql": "mssql", "oracle": "oracle",
            "mysql": "mysql", "mariadb": "mariadb", "databricks": "databricks",
            "sqlite": "sqlite"}.get(scheme, "sqlalchemy")
    return {"kind": kind, "url": s}


def write_model(spec_raw: dict, out_stem: str) -> str:
    """<out_stem>.md (with a Mermaid ER diagram) and <out_stem>.json describing the
    data model of a raw or normalised spec; returns the .md path."""
    from .model_report import describe, to_markdown
    spec = normalise(json.loads(json.dumps(spec_raw, default=str)))
    spec["review"] = spec_raw.get("review", [])
    m = describe(spec)
    Path(f"{out_stem}.md").write_text(to_markdown(spec, m), encoding="utf-8")
    Path(f"{out_stem}.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return f"{out_stem}.md"


def dataset_target(spec: dict, out, i: int) -> dict:
    """Where dataset i (0-based) is written: the target itself for the first one,
    then <path>_<i+1> (folder or SQLite file) or schema <schema>_<i+1>.  `out` (a
    location such as "csv:out/", "sqlite:x.db" or a database URL, a folder path, or
    a target section) replaces the spec's target."""
    if isinstance(out, dict):
        base = dict(out)
    elif out:
        base = parse_source(str(out))
    else:
        base = dict(spec.get("target") or {"kind": "csv", "path": "synthetic"})
    tgt = dict(base)
    if i == 0:
        return tgt
    if tgt.get("kind", "csv") in ("csv", "files", "parquet"):
        tgt["path"] = f"{tgt.get('path', 'synthetic')}_{i + 1}"
    elif tgt.get("kind") == "sqlite" and tgt.get("path"):
        pth = Path(tgt["path"])
        tgt["path"] = str(pth.with_name(f"{pth.stem}_{i + 1}{pth.suffix}"))
    else:
        tgt["schema"] = f"{tgt.get('schema') or 'synthetic'}_{i + 1}"
    return tgt


_dataset_target = dataset_target          # name used by exported model scripts of v0.8-v0.9


@dataclass
class PipelineRun:
    """Everything one pipeline run produced."""
    spec: dict
    bundle: dict
    data: dict | None = None                 # first synthetic dataset (table -> frame)
    synthesizer: Synthesizer | None = None
    fit_data: object | None = None           # relsynth.pushdown.FitData
    report: dict | None = None
    errors: list = field(default_factory=list)          # JSON form (in the bundle)
    exceptions: list = field(default_factory=list)      # (step, exception)
    written: list = field(default_factory=list)         # targets written

    @property
    def ok(self) -> bool:
        return not self.errors


def _prepare_spec(spec_or_path, scale=None, datasets=None, mode=None, pushdown=None, seed=None) -> dict:
    spec = load_spec(spec_or_path)
    if scale is not None:
        from .api import _scale_section
        spec["synthesis"]["scale"] = _scale_section(scale, spec)
    pc = spec["synthesis"]["posterior"]
    if datasets is not None:
        pc["datasets"] = int(datasets)
    if mode is not None:
        pc["mode"] = mode
    if pushdown is not None:
        spec["synthesis"]["pushdown"] = {**spec["synthesis"]["pushdown"],
                                         **({"mode": pushdown} if isinstance(pushdown, str) else dict(pushdown))}
    if seed is not None:
        spec["synthesis"]["seed"] = int(seed)
    return spec


def execute(spec_or_path, *, scale=None, out=None, bundle_path=None, privacy=True, tables=None, datasets=None,
            mode=None, pushdown=None, seed=None, write=True, evaluate_data=True, fit_data=None) -> PipelineRun:
    """Run the pipeline.  Failed steps are recorded (bundle errors and
    PipelineRun.exceptions) and the steps that depend on them are skipped.

    tables: source tables already in memory (instead of reading the spec's source);
    fit_data: a relsynth.pushdown.FitData already loaded; out: output folder or
    target section (default: the spec's target); write=False keeps the data in
    memory only."""
    errors, exceptions, written = [], [], []
    t0 = time.time()
    spec = _prepare_spec(spec_or_path, scale, datasets, mode, pushdown, seed)
    pc = spec["synthesis"]["posterior"]
    syn = report = real = fake = None
    fit = fit_data

    def step(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - recorded for the bundle and re-raised by the API
            errors.append({"step": name, "error": repr(e), "traceback": traceback.format_exc()})
            exceptions.append((name, e))
            log.error("[%s] failed: %r", name, e)
            return None
    if fit is not None:
        real = fit.tables
    elif tables is not None:
        # in-memory tables get the same preparation as loaded ones (computed columns filled)
        from .prepare import computed_columns, prepare_source
        comp = computed_columns(spec)
        real = step("load", lambda: prepare_source(spec, {t: coerce(df, spec["tables"][t], optional=set(comp[t]))
                                                          for t, df in tables.items()})[0])
    else:
        from .pushdown import load_for_fit
        fit = step("load", lambda: load_for_fit(spec))
        real = fit.tables if fit is not None else None
    stats = fit.stats if fit is not None else {}
    if real is not None:
        syn = step("fit", lambda: Synthesizer(spec).fit(real, stats=stats))
    if syn is not None:
        fake = step("sample", lambda: syn.sample(index=0))
    if fake is not None:
        fake = {t: coerce(df, spec["tables"][t]) for t, df in fake.items()}
        if write and out is not False:
            tgt = dataset_target(spec, out, 0)
            if step("write", lambda: write_tables(spec, fake, tgt)) is not False:
                written.append(tgt)
        if evaluate_data:
            report = step("evaluate", lambda: evaluate(spec, real, fake, privacy=privacy,
                                                       sample_log=syn.sample_log, real_stats=stats))
        first_log = syn.sample_log
        # further datasets: independent posterior draws (multiple synthesis, Rubin 1993);
        # the scorecard is computed on the first
        n_sets = int(pc.get("datasets", 1)) if write and out is not False else 1
        for i in range(1, n_sets):
            extra = step(f"sample {i + 1}", lambda: syn.sample(index=i))
            if extra is not None:
                extra = {t: coerce(df, spec["tables"][t]) for t, df in extra.items()}
                tgt = dataset_target(spec, out, i)
                step(f"write {i + 1}", lambda: write_tables(spec, extra, tgt))
                written.append(tgt)
        syn.sample_log = first_log
    extra_b = {"wall_seconds": round(time.time() - t0, 1)}
    if fit is not None and hasattr(fit, "plan"):
        if fit.notes:
            extra_b["load_notes"] = list(fit.notes)
        extra_b["pushdown"] = fit.plan.to_dict()
    bundle = build_bundle(spec, real, syn, report, errors, extra=extra_b,
                          source_rows=fit.source_rows if fit is not None and hasattr(fit, "source_rows") else None)
    if bundle_path:
        write_bundle(bundle, bundle_path)
    return PipelineRun(spec=spec, bundle=bundle, data=fake, synthesizer=syn, fit_data=fit, report=report,
                       errors=errors, exceptions=exceptions, written=written)


def run_pipeline(spec_path_or_dict, scale=None, out=None, bundle_path=None, privacy=True, tables=None, quiet=False,
                 datasets=None, mode=None, pushdown=None):
    """Load -> fit -> sample -> write -> evaluate -> bundle; returns (bundle,
    synthetic tables, fitted Synthesizer).  Never raises: errors go into the bundle
    so that they can be diagnosed.  out=False skips writing."""
    r = execute(spec_path_or_dict, scale=scale, out=out, bundle_path=bundle_path, privacy=privacy, tables=tables,
                datasets=datasets, mode=mode, pushdown=pushdown, write=out is not False)
    return r.bundle, r.data, r.synthesizer


def tune(spec_path, rounds=3, workdir=None, tables=None, quiet=False):
    """run -> doctor -> apply until no spec-level fix remains or rounds run out.
    Returns (best raw spec, its bundle, history); the best spec has the highest
    share of passed scorecard checks."""
    raw = read_yaml(spec_path) if isinstance(spec_path, (str, Path)) else spec_path
    base = Path(spec_path).parent if isinstance(spec_path, (str, Path)) else Path(raw.get("_base_dir", "."))
    history, best = [], None
    for r in range(rounds):
        s_norm = normalise(dict(raw))
        s_norm["_base_dir"] = str(base.resolve())
        bundle, _, _ = run_pipeline(s_norm, out=False, tables=tables, quiet=quiet)
        sc = (bundle.get("evaluation") or {}).get("scorecard") or {"passed": 0, "total": 1}
        issues = [i for i in D.diagnose(bundle) if i.kind == "spec" and i.patches]
        patches = D.merge_patches(issues)
        history.append({"round": r, "passed": sc["passed"], "total": sc["total"],
                        "failed": [f"{f['area']}:{f['metric']}" for f in sc.get("failed", [])],
                        "issues": [f"{i.id} {i.where}" for i in issues], "patches": patches})
        if best is None or sc["passed"] / max(sc["total"], 1) > best[0]:
            best = (sc["passed"] / max(sc["total"], 1), raw, bundle)
        if not patches:
            break
        raw = D.apply_patches(raw, patches)
    return best[1], best[2], history


def export_model_for(spec_or_path, out_dir, *, draws=5, formats=("json", "md", "pdf", "py"), replica_check=True,
                     stan=False, synthesizer=None, fit_data=None) -> dict:
    """Fit (unless `synthesizer` is given) and export the Bayesian model: the
    files listed in formats, the posterior draws and the verification checks
    (relsynth.export.export_model)."""
    from .export import FORMATS, export_model
    bad = [x for x in formats if x not in FORMATS]
    if bad:
        from .spec import SpecError
        raise SpecError(f"unknown export format(s) {bad}; choose from {', '.join(FORMATS)}")
    spec = load_spec(spec_or_path)
    if synthesizer is None or fit_data is None:
        from .pushdown import load_for_fit
        fit_data = fit_data or load_for_fit(spec)
        synthesizer = synthesizer or Synthesizer(spec).fit(fit_data.tables, stats=fit_data.stats)
    spec_path = str(spec_or_path) if isinstance(spec_or_path, (str, Path)) else None
    return export_model(synthesizer, out_dir, spec_path=spec_path, draws=draws, formats=tuple(formats),
                        real=fit_data.tables, real_stats=fit_data.stats, check_replica=replica_check, stan=stan,
                        spec_raw=spec)

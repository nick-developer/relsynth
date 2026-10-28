# SPDX-License-Identifier: Apache-2.0
"""The relsynth command line, for people and for CI/CD pipelines.

    relsynth introspect --source postgresql://analyst@db/sales --schema dw --out spec.yaml
    relsynth check spec.yaml                       # validate (exit 2 when invalid)
    relsynth plan spec.yaml                        # what is fitted in-database (no fitting)
    relsynth run spec.yaml --scale 2 --fail-under 0.98 --json
    relsynth evaluate spec.yaml --synthetic csv:out/
    relsynth doctor diagnostics_bundle.json --spec spec.yaml --apply
    relsynth tune spec.yaml --rounds 3
    relsynth export-model spec.yaml --out model/
    relsynth model spec.yaml                       # data model as Markdown + Mermaid
    relsynth clone-schema --source ... --target ... [--tables a b]
    relsynth profile --source ...                  # in-database counts (no rows leave)
    relsynth ping --source ...                     # connection test
    relsynth version

Exit codes: 0 success, 1 quality gate failed, 2 invalid spec or arguments,
3 connection / read / write failure, 4 fitting, sampling or internal error.
Every command accepts --json (one JSON document on stdout), --log-level and,
where it reads a spec, --set KEY=VALUE overrides (repeatable).  Credentials
come from the URL environment variables named in the spec (url_env,
password_env); they are never printed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from ._version import __version__
from .diagnostics import read_bundle, write_bundle
from .errors import EXIT_GATE, EXIT_OK, EXIT_SPEC, RelsynthError, exit_code_for
from .introspect import introspect
from .pipeline import dataset_target as _dataset_target  # noqa: F401 - used by exported model scripts
from .pipeline import parse_source, run_pipeline, tune, write_model  # noqa: F401 - public re-exports
from .spec import SpecError

D = importlib.import_module("relsynth.doctor")  # the module, not the API function

log = logging.getLogger("relsynth")


# ---------------------------------------------------------------------- helpers
class _Out:
    """Human lines on stdout, or one JSON document with --json."""

    def __init__(self, as_json: bool):
        self.as_json = as_json
        self.data: dict = {}

    def line(self, text: str = "") -> None:
        if not self.as_json:
            print(text)

    def put(self, **kw) -> None:
        self.data.update(kw)

    def flush(self, code: int) -> None:
        if self.as_json:
            from .diagnostics import _jsonable
            self.data.setdefault("exit_code", code)
            print(json.dumps(_jsonable(self.data), indent=1, default=str))


def _spec(a):
    """The Spec of a command, with --set overrides applied."""
    from .api import Spec
    s = Spec.load(a.spec)
    for kv in a.set or []:
        if "=" not in kv:
            raise SpecError(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        s = s.set(k.strip(), v)
    return s


def _location(a, name):
    """The location section of --NAME and its --NAME-schema / -password-env /
    -url-env options; --NAME-url-env alone is enough (the URL stays in the
    environment, only its kind is read from it)."""
    v = getattr(a, name)
    env = getattr(a, f"{name}_url_env", None)
    if not v:
        if not env:
            raise SpecError(f"give --{name} or --{name}-url-env")
        if not os.environ.get(env):
            raise SpecError(f"--{name}-url-env: the environment variable {env} is not set")
        sec = {"kind": parse_source(os.environ[env])["kind"]}
    else:
        sec = parse_source(v)
    for opt in ("schema", "password_env", "url_env"):
        val = getattr(a, f"{name}_{opt}", None)
        if val:
            sec[opt] = val
    return sec


def _scorecard_lines(out, rep, limit=25):
    out.line(f"scorecard: {rep.passed}/{rep.total} checks passed ({rep.score:.1%})")
    for f in rep.failed[:limit]:
        out.line(f"  FAIL {f['area']}: {f['metric']} = {f['value']} (limit {f['limit']})")
    if len(rep.failed) > limit:
        out.line(f"  ... and {len(rep.failed) - limit} more (see the bundle)")


# ---------------------------------------------------------------------- commands
def cmd_version(a, out):
    import numpy
    import pandas
    import sqlalchemy
    out.put(version=__version__, python=sys.version.split()[0], numpy=numpy.__version__,
            pandas=pandas.__version__, sqlalchemy=sqlalchemy.__version__)
    out.line(f"relsynth {__version__} (python {sys.version.split()[0]}, numpy {numpy.__version__}, "
             f"pandas {pandas.__version__}, sqlalchemy {sqlalchemy.__version__})")
    return EXIT_OK


def cmd_introspect(a, out):
    from .data import mask_spec_urls
    from .spec import normalise, validate
    src = _location(a, "source")
    spec = introspect(src, name=a.name, sample_rows=a.sample_rows, include=a.include, exclude=a.exclude,
                      evidence=a.evidence, declared=not a.ignore_declared)
    # the draft is a file to share and review: its URL shows *** for the password, and
    # source.url_env names the environment variable that supplies the real URL
    import yaml
    shown = mask_spec_urls(spec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(json.loads(json.dumps(shown, default=str)), sort_keys=False, width=110),
                           encoding="utf-8")
    if shown.get("source") != spec.get("source"):
        out.line(f"the source URL's password is masked in {a.out}: set the environment variable "
                 f"{shown['source']['url_env']} to the full URL to run the spec")
    errs = validate(normalise(json.loads(json.dumps(spec, default=str))))
    out.line(f"wrote {a.out}: {len(spec['tables'])} tables, {len(spec['relationships'])} relationships")
    try:
        md = write_model(spec, str(Path(a.out).with_suffix("")) + ".model")
        out.line(f"wrote {md} (the data model)")
    except Exception as e:  # noqa: BLE001 - the draft is still useful without its report
        out.line(f"model report not written: {e}")
    if spec.get("review"):
        out.line("review these inferences:\n  - " + "\n  - ".join(spec["review"]))
    if errs:
        out.line("spec needs edits before it validates:\n  - " + "\n  - ".join(errs))
    out.put(spec=a.out, tables=len(spec["tables"]), relationships=len(spec["relationships"]),
            review=spec.get("review", []), errors=errs)
    return EXIT_OK


def cmd_check(a, out):
    s = _spec(a)
    errs = s.errors()
    out.put(spec=str(a.spec), valid=not errs, errors=errs)
    if errs:
        out.line("spec is invalid:\n  - " + "\n  - ".join(errs))
        return EXIT_SPEC
    r = s.resolved
    out.put(tables=len(r["tables"]), relationships=len(r["relationships"]), rules=len(r["rules"]))
    out.line(f"spec OK: {len(r['tables'])} tables, {len(r['relationships'])} relationships, {len(r['rules'])} rules")
    if a.connect:
        from .api import ping
        p = ping(_abs(r, "source"))
        out.put(source=p)
        out.line(f"source: {'reachable' if p.get('ok') else 'NOT reachable'} ({p.get('location', '')})"
                 + (f": {p.get('error')}" if not p.get("ok") else f", {p.get('tables')} tables"))
        if not p.get("ok"):
            return 3
        missing = _missing_tables(r)
        out.put(missing_tables=missing)
        if missing:
            out.line("tables of the spec missing from the source: " + ", ".join(missing))
            return 3
    return EXIT_OK


def _abs(spec, key):
    """A spec section with a relative path resolved against the spec's folder."""
    sec = dict(spec.get(key) or {})
    if sec.get("path") and not Path(sec["path"]).is_absolute():
        sec["path"] = str(Path(spec.get("_base_dir", ".")) / sec["path"])
    return sec


def _missing_tables(spec):
    from .data import source_connector
    with source_connector(spec) as c:
        return [t for t in spec["tables"] if not c.has_table(t)]


def cmd_plan(a, out):
    from .data import source_connector
    from .pushdown import planner
    s = _spec(a).resolved
    if a.pushdown:
        s["synthesis"]["pushdown"]["mode"] = a.pushdown
    with source_connector(s) as conn:
        p = planner.plan(s, conn)
        rows = {t: conn.count(t) for t in s["tables"] if conn.has_table(t)}
        cost = conn.cost_report(list(rows), p.pushed) if hasattr(conn, "cost_report") else None
    out.put(pushdown=p.to_dict(), rows=rows, **({"cost": cost} if cost else {}))
    out.line(f"pushdown mode {p.mode}; samples of {p.sample_rows:,} rows")
    w = max([len(t) for t in s["tables"]] + [10])
    for t in s["tables"]:
        n = rows.get(t)
        how = "statistics in the source + sample" if t in p.pushed else "loaded"
        why = f" ({p.reasons[t]})" if t in p.reasons else ""
        out.line(f"  {t:<{w}} {('-' if n is None else f'{n:,}'):>14} rows  {how}{why}")
    if cost:
        from .connectors.warehouse import human_bytes
        out.line(f"cost: about {human_bytes(cost['scanned'])} scanned in all (query tag {cost['query_tag']})")
        for t, c in cost["tables"].items():
            out.line(f"  {t:<{w}} {human_bytes(c['bytes']):>14} stored  x{c['scans']} scans")
        if cost.get("billing"):
            out.line(f"billing: {cost['billing']}")
    return EXIT_OK


def cmd_run(a, out):
    from .api import run
    s = _spec(a)
    target = _location(a, "target") if (a.target or a.target_url_env) else (a.out or None)
    t0 = time.time()
    res = run(s, scale=a.scale, out=target, datasets=a.datasets, mode=a.mode, pushdown=a.pushdown, seed=a.seed,
              privacy=not a.no_privacy, write=not a.no_write, bundle=a.bundle, raise_errors=False)
    out.put(bundle=a.bundle, seconds=round(time.time() - t0, 1),
            rows={t: len(d) for t, d in (res.data or {}).items()},
            pushdown=res.bundle.get("pushdown"), errors=[{"step": e["step"], "error": e["error"]} for e in res.errors],
            written=[_mask(w) for w in res.written])
    if res.report is not None:
        out.put(scorecard={"passed": res.report.passed, "total": res.report.total, "score": res.report.score,
                           "failed": res.report.failed})
        _scorecard_lines(out, res.report)
        if a.report:
            Path(a.report).write_text(res.report.to_markdown(), encoding="utf-8")
    if a.bundle:
        out.line(f"diagnostics bundle -> {a.bundle}")
    if res.errors:
        out.line(f"{len(res.errors)} step(s) failed: " + "; ".join(f"{e['step']}: {e['error']}" for e in res.errors))
        out.line("run `relsynth doctor` on the bundle for causes and fixes")
        return _run_exit_code(res)
    if a.fail_under is not None and res.report is not None and res.report.score + 1e-12 < a.fail_under:
        out.line(f"quality gate failed: {res.report.score:.3f} < {a.fail_under:.3f}")
        out.put(gate={"fail_under": a.fail_under, "passed": False})
        return EXIT_GATE
    if a.fail_under is not None:
        out.put(gate={"fail_under": a.fail_under, "passed": True})
    return EXIT_OK


def _run_exit_code(res) -> int:
    """Exit code of a run with failed steps: from its first exception; reading the
    source or writing the target count as I/O unless the error is a spec error."""
    from .errors import EXIT_IO
    step, exc = res.exceptions[0]
    code = exit_code_for(exc)
    if code == 4 and (step == "load" or step.startswith("write")) and not isinstance(exc, SpecError):
        return EXIT_IO
    return code


def _mask(sec):
    from .data import mask_spec_urls
    return mask_spec_urls({"target": dict(sec)})["target"]


def cmd_verify(a, out):
    from .api import verify
    s = _spec(a)
    target = _location(a, "target") if (a.target or a.target_url_env) else None
    rep = verify(s, draws=a.draws, stability=not a.no_stability, out=target, privacy=not a.no_privacy,
                 pushdown=a.pushdown or "off", seed=a.seed)
    out.put(verify=rep.to_dict())
    out.line(f"model checks: {rep.passed}/{rep.total} passed ({rep.seconds}s)")
    for c in rep.failed():
        out.line(f"  FAIL {c['id']} {c['what']}: {c['value']} (target {c['target']})")
    for e in rep.errors:
        out.line(f"  could not run {e['step']}: {_scrub(str(e['error']))}")
    if a.report:
        Path(a.report).write_text(rep.to_markdown(), encoding="utf-8")
    if a.bundle:
        rep.save_bundle(a.bundle)
        out.line(f"diagnostics bundle -> {a.bundle}")
    return EXIT_OK if rep.ok else EXIT_GATE


def cmd_evaluate(a, out):
    from .api import evaluate
    from .connectors import from_spec
    s = _spec(a)
    sec = parse_source(a.synthetic)
    r = s.resolved
    with from_spec(sec, Path(".")) as conn:
        syn = {t: conn.read_table(t) for t in r["tables"] if conn.has_table(t)}
    missing = [t for t in r["tables"] if t not in syn]
    if missing:
        raise SpecError("synthetic data lacks tables: " + ", ".join(missing))
    rep = evaluate(s, syn, privacy=not a.no_privacy)
    if a.output:
        rep.to_json(a.output)
    out.put(scorecard={"passed": rep.passed, "total": rep.total, "score": rep.score, "failed": rep.failed})
    _scorecard_lines(out, rep)
    if a.fail_under is not None and rep.score + 1e-12 < a.fail_under:
        out.line(f"quality gate failed: {rep.score:.3f} < {a.fail_under:.3f}")
        return EXIT_GATE
    return EXIT_OK


def cmd_doctor(a, out):
    import yaml
    from .spec import read_yaml
    b = read_bundle(a.bundle)
    issues = D.diagnose(b)
    patches = D.merge_patches(issues)
    md = D.render(issues, patches, b)
    if a.report:
        Path(a.report).write_text(md, encoding="utf-8")
    out.line(md)
    out.put(issues=[{"id": i.id, "severity": i.severity, "kind": i.kind, "where": i.where, "message": i.message}
                    for i in issues], patches=patches)
    if a.apply:
        raw = read_yaml(a.spec) if a.spec else b["spec"]
        new = D.apply_patches(raw, patches)
        dest = a.out or (str(Path(a.spec).with_suffix("")) + ".fixed.yaml" if a.spec else "spec.fixed.yaml")
        Path(dest).write_text(yaml.safe_dump(new, sort_keys=False, width=110), encoding="utf-8")
        out.line(f"applied {len(patches)} change(s) -> {dest}")
        out.put(fixed_spec=dest)
    return EXIT_OK


def cmd_tune(a, out):
    import yaml
    best, bundle, hist = tune(a.spec, a.rounds)
    dest = a.out or str(Path(a.spec).with_suffix("")) + ".tuned.yaml"
    Path(dest).write_text(yaml.safe_dump({k: v for k, v in best.items() if k != "_base_dir"}, sort_keys=False,
                                         width=110), encoding="utf-8")
    if a.bundle:
        write_bundle(bundle, a.bundle)
    for h in hist:
        out.line(f"round {h['round']}: {h['passed']}/{h['total']} passed; fixes: {list(h['patches'])}")
    out.line(f"best spec -> {dest}")
    out.put(best_spec=dest, history=hist)
    return EXIT_OK


def cmd_model(a, out):
    from .spec import read_yaml
    raw = read_yaml(a.spec)
    stem = a.out or str(Path(a.spec).with_suffix("")) + ".model"
    md = write_model(raw, stem)
    out.line(f"wrote {md} and {stem}.json")
    out.put(markdown=md, json=f"{stem}.json")
    return EXIT_OK


def cmd_export_model(a, out):
    from .pipeline import export_model_for
    s = _spec(a)
    fmts = tuple(x.strip() for x in a.formats.split(",") if x.strip())
    dest = a.out or str(Path(a.spec).with_suffix("")) + ".model"
    res = export_model_for(s.resolved, dest, draws=a.draws, formats=fmts, replica_check=not a.no_replica_check,
                           stan=a.stan)
    nok = sum(c["ok"] for c in res["checks"])
    nfail = sum(not c["ok"] and c.get("level") != "warn" for c in res["checks"])
    out.line(f"model verification: {nok}/{len(res['checks'])} checks pass")
    for c in res["checks"]:
        if not c["ok"]:
            out.line(f"  {'WARN' if c.get('level') == 'warn' else 'FAIL'} {c['check']}: {c['detail']}")
    if res.get("replica"):
        out.line(f"replica: {res['replica'].get('replica scorecard')}")
    out.line("wrote " + ", ".join(res["files"]))
    out.put(files=res["files"], checks=res["checks"], replica=res.get("replica"))
    return EXIT_OK if nfail == 0 else EXIT_GATE


def cmd_clone_schema(a, out):
    from .api import clone_schema
    made = clone_schema(_location(a, "source"), _location(a, "target"), tables=a.tables, replace=a.replace)
    out.line(f"created {len(made)} table(s): " + ", ".join(made))
    out.put(tables=made)
    return EXIT_OK


def cmd_profile(a, out):
    from .api import profile
    prof = profile(_location(a, "source"), tables=a.tables, top=a.top)
    if a.output:
        Path(a.output).write_text(json.dumps(prof, indent=1, default=str), encoding="utf-8")
    for t, d in prof.items():
        out.line(f"{t}: {d['rows']:,} rows, {len(d['columns'])} columns, primary key {d['primary_key'] or '-'}")
        for c, cd in d["columns"].items():
            nn = cd["non_null"]
            dist = cd.get("distinct", "?")
            out.line(f"    {c:<32} {cd['type']:<18} non-null {nn:>12,}  distinct {dist}")
    out.put(profile=prof)
    return EXIT_OK


def cmd_ping(a, out):
    from .api import ping
    p = ping(_location(a, "source"))
    out.put(**p)
    if p.get("ok"):
        out.line(f"ok: {p['location']} ({p['tables']} tables, {p['seconds']} s)")
        return EXIT_OK
    out.line(f"cannot connect: {_scrub(str(p.get('error')))}")
    out.put(error=_scrub(str(p.get("error"))))
    return 3


# ---------------------------------------------------------------------- parser
def _common(p):
    p.add_argument("--json", action="store_true", help="print one JSON document on stdout (for CI)")
    p.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error"],
                   help="progress detail on stderr (default info; RELSYNTH_QUIET=1 sets warning)")


def _spec_args(p):
    p.add_argument("spec", help="the YAML specification")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a spec value, e.g. --set synthesis.scale.default=2 (repeatable)")


def _loc_args(p, name, required=True, help_=""):
    # not required by argparse: --NAME-url-env can name the location instead (checked in _location)
    p.add_argument(f"--{name}", help=(help_ or "csv:DIR | parquet:DIR | sqlite:FILE | database URL "
                                      "(postgresql://, mssql://, oracle://, mysql://, mariadb://, databricks://, fabric://, snowflake://)")
                   + ("" if not required else f"; or give --{name}-url-env"))
    p.add_argument(f"--{name}-schema", help="database schema")
    p.add_argument(f"--{name}-password-env", help="environment variable holding the password")
    p.add_argument(f"--{name}-url-env", help="environment variable holding the whole URL")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relsynth", description="Bayesian synthetic data for relational databases.",
                                epilog="exit codes: 0 ok, 1 quality gate failed, 2 invalid spec/arguments, "
                                       "3 connection or I/O failure, 4 engine failure")
    p.add_argument("--version", action="version", version=f"relsynth {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    s = sub.add_parser("introspect", help="draft a spec from a database or folder")
    _loc_args(s, "source")
    s.add_argument("--schema", dest="source_schema_alt", help=argparse.SUPPRESS)
    s.add_argument("--out", default="spec.yaml", help="where the drafted spec is written")
    s.add_argument("--name", default="draft", help="the spec's name")
    s.add_argument("--sample-rows", type=int, default=20000, help="random rows read per table")
    s.add_argument("--include", nargs="*", help="table-name patterns to read (regular expressions)")
    s.add_argument("--exclude", nargs="*", help="table-name patterns to leave out, e.g. sys.* DatabaseLog")
    s.add_argument("--evidence", choices=["names+data", "data"], default="names+data",
                   help="identify keys and links from values and names (default) or from values only")
    s.add_argument("--ignore-declared", action="store_true",
                   help="hide the catalog's keys and links and identify them from the data (for testing)")
    s.set_defaults(fn=cmd_introspect)

    for name, hlp in (("check", "validate a spec"), ("validate", "validate a spec (alias of check)")):
        s = sub.add_parser(name, help=hlp)
        _spec_args(s)
        s.add_argument("--connect", action="store_true", help="also connect to the source and find every table")
        s.set_defaults(fn=cmd_check)

    s = sub.add_parser("plan", help="which tables are fitted from in-database statistics (no fitting)")
    _spec_args(s)
    s.add_argument("--pushdown", choices=["auto", "on", "off"], help="override synthesis.pushdown.mode")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("run", help="fit, sample, write and score")
    _spec_args(s)
    s.add_argument("--scale", type=float, help="size multiplier of the synthetic entities")
    s.add_argument("--out", help="output folder for CSV files (default: the spec's target)")
    _loc_args(s, "target", required=False, help_="write to this location instead of the spec's target")
    s.add_argument("--bundle", default="diagnostics_bundle.json", help="diagnostics bundle path ('' for none)")
    s.add_argument("--report", help="also write the scorecard as Markdown")
    s.add_argument("--no-privacy", action="store_true", help="skip distance-to-closest-record checks")
    s.add_argument("--no-write", action="store_true", help="score only; write no synthetic data")
    s.add_argument("--datasets", type=int, help="number of synthetic datasets (independent posterior draws)")
    s.add_argument("--mode", choices=["draw", "mean"], help="posterior draw (default) or posterior means")
    s.add_argument("--pushdown", choices=["auto", "on", "off"], help="override synthesis.pushdown.mode")
    s.add_argument("--seed", type=int, help="override synthesis.seed")
    s.add_argument("--fail-under", type=float, metavar="SHARE",
                   help="quality gate: exit 1 when the share of passed checks is below SHARE (0-1)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("verify", help="model checks on real data (R1-R7): scorecard, convergence, "
                                      "posterior predictive checks, scale rule, structure, stability")
    _spec_args(s)
    s.add_argument("--draws", type=int, default=5, help="synthetic copies for the posterior predictive checks")
    s.add_argument("--no-stability", action="store_true", help="skip the two extra fits (R6, R7)")
    s.add_argument("--pushdown", choices=["auto", "on", "off"], help="default off: every table is loaded whole")
    s.add_argument("--seed", type=int, help="override synthesis.seed")
    s.add_argument("--no-privacy", action="store_true", help="skip distance-to-closest-record checks")
    _loc_args(s, "target", required=False, help_="also write the first synthetic database here")
    s.add_argument("--report", help="write the checks as Markdown")
    s.add_argument("--bundle", help="write a diagnostics bundle with the checks (for doctor)")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("evaluate", help="score synthetic data already written")
    _spec_args(s)
    s.add_argument("--synthetic", required=True, help="where the synthetic tables are (csv:DIR, sqlite:FILE, URL)")
    s.add_argument("--output", help="write every metric as JSON")
    s.add_argument("--no-privacy", action="store_true", help="skip distance-to-closest-record checks")
    s.add_argument("--fail-under", type=float, metavar="SHARE",
                   help="quality gate: exit 1 when the share of passed checks is below SHARE (0-1)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("doctor", help="causes and spec fixes for a run's failures")
    s.add_argument("bundle", help="diagnostics bundle written by run or verify")
    s.add_argument("--spec", help="the spec to patch (default: the copy inside the bundle)")
    s.add_argument("--apply", action="store_true", help="write the fixed spec")
    s.add_argument("--out", help="where --apply writes the fixed spec (default: <spec>.fixed.yaml)")
    s.add_argument("--report", help="also write the report as Markdown")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("tune", help="run -> doctor -> apply until no fix remains")
    s.add_argument("spec", help="the YAML specification")
    s.add_argument("--rounds", type=int, default=3, help="largest number of run -> doctor -> apply rounds")
    s.add_argument("--out", help="where the best spec is written (default: <spec>.tuned.yaml)")
    s.add_argument("--bundle", help="also write the best round's diagnostics bundle")
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("model", help="the data model as Markdown (Mermaid ER diagram) and JSON")
    s.add_argument("spec", help="the YAML specification")
    s.add_argument("--out", help="output path without extension (default: <spec>.model)")
    s.set_defaults(fn=cmd_model)

    s = sub.add_parser("export-model", help="the fitted Bayesian model as JSON, Markdown, PDF and Python")
    _spec_args(s)
    s.add_argument("--out", help="output folder (default: <spec>.model)")
    s.add_argument("--draws", type=int, default=5, help="posterior draws to export (draw m = dataset m of `run`)")
    s.add_argument("--formats", default="json,md,pdf,py", help="any of json, md, pdf, py")
    s.add_argument("--no-replica-check", action="store_true", help="skip scoring the standalone replica")
    s.add_argument("--stan", action="store_true", help="also check the posterior blocks with Stan (needs pystan)")
    s.set_defaults(fn=cmd_export_model)

    s = sub.add_parser("clone-schema", help="create the source's tables (keys, types, indexes) empty in a target")
    _loc_args(s, "source")
    _loc_args(s, "target")
    s.add_argument("--tables", nargs="*", help="only these tables (default: all)")
    s.add_argument("--replace", action="store_true", help="drop existing tables first")
    s.set_defaults(fn=cmd_clone_schema)

    s = sub.add_parser("profile", help="row, null and distinct counts computed in the source")
    _loc_args(s, "source")
    s.add_argument("--tables", nargs="*", help="only these tables (default: all)")
    s.add_argument("--top", type=int, default=0, help="also list the N most frequent values (shares data values)")
    s.add_argument("--output", help="write the profile as JSON")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("ping", help="test a connection")
    _loc_args(s, "source")
    s.set_defaults(fn=cmd_ping)

    s = sub.add_parser("version", help="versions of relsynth and its main dependencies")
    s.set_defaults(fn=cmd_version)

    for sp in sub.choices.values():
        _common(sp)
    return p


_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+")


def _scrub(msg: str) -> str:
    """An error message with the password of every URL in it masked."""
    from .data import mask_url
    return _URL.sub(lambda m: mask_url(m.group(0)), msg)


def _configure_logging(level: str | None, as_json: bool) -> None:
    if level is None:
        level = "warning" if os.environ.get("RELSYNTH_QUIET") == "1" else "info"
    lg = logging.getLogger("relsynth")
    lg.setLevel(getattr(logging, level.upper()))
    if not any(getattr(h, "_relsynth", False) for h in lg.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("[relsynth %(asctime)s] %(message)s", "%H:%M:%S"))
        h._relsynth = True
        lg.addHandler(h)
        lg.propagate = False


def main(argv=None) -> int:
    p = build_parser()
    try:
        a = p.parse_args(argv)
    except SystemExit as e:          # argparse: --help / --version exit 0, usage errors exit 2
        return int(e.code or 0)
    if getattr(a, "source_schema_alt", None) and not getattr(a, "source_schema", None):
        a.source_schema = a.source_schema_alt
    _configure_logging(a.log_level, a.json)
    out = _Out(a.json)
    out.put(command=a.cmd, version=__version__)
    try:
        code = a.fn(a, out)
    except KeyboardInterrupt:
        out.put(error="interrupted")
        out.flush(130)
        return 130
    except Exception as e:  # noqa: BLE001 - every failure becomes an exit code and one message
        code = exit_code_for(e)
        kind = "invalid spec" if isinstance(e, SpecError) else type(e).__name__
        msg = _scrub(str(e) if isinstance(e, (RelsynthError, SpecError)) else f"{type(e).__name__}: {e}")
        print(f"relsynth {a.cmd}: {kind if isinstance(e, SpecError) else 'error'}: {msg}", file=sys.stderr)
        if logging.getLogger("relsynth").isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        out.put(error=msg, error_type=type(e).__name__)
        out.flush(code)
        return code
    out.flush(code)
    return code


if __name__ == "__main__":
    sys.exit(main())

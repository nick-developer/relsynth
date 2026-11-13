# SPDX-License-Identifier: Apache-2.0
"""relsynth benchmark harness.

    python -m bench all [--tier pr,weekly]   # every suite: lint, unit, e2e, live servers (if configured),
                                             # the tiers' databases on their seeds, fake suite; then compare
    python -m bench fetch [--tier ...]       # download and build the databases
    python -m bench run [--tier pr | --datasets ...] [--seeds 1,2,3] [--suites scorecard,quality,privacy,repro]
                        [--no-fake | --fake-only]
    python -m bench compare [RECORD ...] [--full [--tier ...]]   # the newest record (or the RECORDs of one run)
                                             # against bench/baseline.json
    python -m bench baseline RECORD...       # rebuild bench/baseline.json from history records
    python -m bench list                     # the datasets, their tier and whether each is available
    python -m bench matrix --tier weekly     # the tier's CI jobs as JSON (bench.yml)
    python -m bench coverage [--check]       # the feature x dataset table of bench/README.md

Exit code 1 when `compare` (or `all`) finds a regression; 2 on a usage error.
See docs/development/benchmarks.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import regress
from .datasets import DATASETS, FEATURES, REQUIRED_FEATURES, TIERS, available, is_private, private_store, say, \
    tier_datasets
from .runner import SUITES

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "bench" / "history"
BASELINE = ROOT / "bench" / "baseline.json"
DEFAULT_TIERS = "pr,weekly"


def _seeds(s: str | None) -> list[int] | None:
    """The seeds given on the command line; None: each dataset's own (bench/datasets.py)."""
    return None if not s else [int(x) for x in s.split(",") if x.strip()]


def _tiers(s: str) -> list[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in TIERS]
    if bad:
        raise SystemExit(f"unknown tiers {bad}; choose from {', '.join(TIERS)}")
    return out


def _selected(a) -> list[str]:
    """--datasets, else the datasets of --tier (the PR and weekly tiers by default; private ones too, skipped
    where their data is not configured)."""
    if a.datasets:
        return a.datasets
    return [n for t in _tiers(a.tier) for n in tier_datasets(t)]


def matrix(tier: str) -> list[dict]:
    """The CI jobs of a tier: one per dataset, or one per seed group (`jobs`); datasets that need a bigger
    runner (runner "maintainer") and datasets that cannot run on a hosted runner (local tier) are left out."""
    out = []
    for n in tier_datasets(tier):
        d = DATASETS[n]
        if d.runner != "github" or not d.public:
            continue
        groups = d.jobs or (d.seeds,)
        for i, g in enumerate(groups):
            out.append({"name": n if len(groups) == 1 else f"{n}-{chr(97 + i)}", "datasets": n,
                        "seeds": ",".join(map(str, g)), "fake": "--no-fake"})
    if tier == "weekly":                         # the statistical check suite runs with the weekly bench
        out.append({"name": "fake", "datasets": "", "seeds": "1", "fake": "--fake-only"})
    return out


def coverage_table() -> tuple[str, list[str]]:
    """(markdown table of feature x datasets with their tier, features of REQUIRED_FEATURES under two sets)."""
    rows = ["| Feature | Datasets (tier) | Sets |", "|---|---|---|"]
    short = []
    for f, desc in FEATURES.items():
        names = [n for n, d in DATASETS.items() if f in d.coverage]
        rows.append(f"| {desc} (`{f}`) | {', '.join(f'{n} ({DATASETS[n].tier})' for n in names)} | {len(names)} |")
        if f in REQUIRED_FEATURES and len(names) < 2:
            short.append(f)
    return "\n".join(rows), short


def _suites(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    bad = [x for x in out if x not in SUITES]
    if bad:
        raise SystemExit(f"unknown suites {bad}; choose from {', '.join(SUITES)}")
    return out


def _newest() -> Path:
    files = sorted(HISTORY.glob("*.json"))
    if not files:
        raise SystemExit("no history record yet; run `python -m bench run` first")
    return files[-1]


def _load_baseline(path: Path) -> dict:
    """The public baseline plus the private datasets' baseline, where the private store has one."""
    base = json.loads(path.read_text(encoding="utf-8"))
    store = private_store()
    if store is not None and (store / "baseline.json").exists():
        base["datasets"].update(json.loads((store / "baseline.json").read_text(encoding="utf-8"))["datasets"])
    return base


def cmd_compare(records: list[Path], baseline: Path, full: bool, summary_to: str | None,
                expected: set | None = None) -> int:
    """Records of one run split over several jobs (bench.yml) are compared as one."""
    from .runner import load_record
    base = _load_baseline(baseline)
    recs = [load_record(r) for r in records]
    rec = {**recs[0], "runs": [r for x in recs for r in x["runs"]]}
    reg, notes, summ = regress.compare(base, rec, full=full, expected=expected)
    text = f"Record `{', '.join(r.name for r in records)}` against `{baseline.name}` ({', '.join(base['relsynth'])})\n\n" + \
        regress.report(reg, notes, summ)
    print(text)
    if summary_to:
        with open(summary_to, "a", encoding="utf-8") as f:
            f.write(text)
    return 1 if reg else 0


def cmd_all(a) -> int:
    """Every suite of a clean checkout, then the regression comparison."""
    steps = [["ruff", "check", "src", "tests", "tools", "bench"],
             [sys.executable, "-m", "pytest", "tests/unit", "tests/e2e", "-q"]]
    if any(os.environ.get(v) for v in ("RELSYNTH_TEST_PG_URL", "RELSYNTH_TEST_MSSQL_URL", "RELSYNTH_TEST_ORACLE_URL")):
        steps.append([sys.executable, "-m", "pytest", "tests/integration", "-q"])
    else:
        say("live servers not configured (RELSYNTH_TEST_*_URL): tests/integration skipped; see docs/testing.md")
    failed = []
    for s in steps:
        say(" ".join(s))
        if subprocess.run(s, cwd=ROOT, check=False).returncode:  # noqa: S603
            failed.append(" ".join(s[:4]))
    from .runner import run
    names = _selected(a)
    path = run(names, _seeds(a.seeds), fake=True, pushdown=a.pushdown, label=a.label or "all",
               overrides=tuple(a.overrides), suites=_suites(a.suites), jobs=a.jobs)
    rc = cmd_compare([path], Path(a.baseline), full=True, summary_to=os.environ.get("GITHUB_STEP_SUMMARY"),
                     expected={*names, "fake"})
    if failed:
        print("failed suites: " + "; ".join(failed))
    return 1 if failed or rc else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench", description="relsynth benchmark harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), help="default: the datasets of --tier")
        p.add_argument("--tier", default=DEFAULT_TIERS, help=f"tiers to run, from {', '.join(TIERS)} "
                       f"(default {DEFAULT_TIERS}); bench/README.md lists each tier's datasets")
        p.add_argument("--seeds", default=None, help="default: each dataset's own seeds (5 for Chinook, 3 for "
                       "AdventureWorks, 1 for the large sets)")
        p.add_argument("--pushdown", default="auto", choices=["auto", "on", "off"])
        p.add_argument("--label", default="")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="spec override passed to every `relsynth run` (e.g. synthesis.facts.link_degree=auto)")
        p.add_argument("--suites", default=",".join(SUITES),
                       help="what each real database run includes, from " + ", ".join(SUITES) +
                       " (each dataset declares which apply to it; docs/development/benchmarks.md)")
        p.add_argument("--baseline", default=str(BASELINE))
        p.add_argument("--jobs", type=int, default=1, help="runs in parallel (dataset x seed; each run is its own "
                       "processes, so their timings share the machine)")
        if name == "run":
            p.add_argument("--no-fake", action="store_true", help="skip the statistical check suite")
            p.add_argument("--fake-only", action="store_true", help="only the statistical check suite")
            p.add_argument("--out-dir", default=str(HISTORY))
    p = sub.add_parser("compare")
    p.add_argument("record", nargs="*", help="one record, or the records of one run split over several jobs")
    p.add_argument("--baseline", default=str(BASELINE))
    p.add_argument("--full", action="store_true", help="a dataset of the baseline that was not run is a regression")
    p.add_argument("--tier", default=None, help="with --full: only the datasets of these tiers must have run")
    p.add_argument("--hosted", action="store_true", help="with --tier: only the datasets a GitHub-hosted runner "
                   "runs (bench.yml leaves out the bigger-runner and local sets)")
    p = sub.add_parser("baseline")
    p.add_argument("records", nargs="+")
    p.add_argument("--out", default=str(BASELINE))
    p = sub.add_parser("fetch")
    p.add_argument("--datasets", nargs="*", choices=sorted(DATASETS))
    p.add_argument("--tier", default=DEFAULT_TIERS)
    sub.add_parser("list")
    sub.add_parser("matrix").add_argument("--tier", required=True, choices=TIERS)
    sub.add_parser("coverage").add_argument("--check", action="store_true",
                                            help="exit 1 when a required feature has fewer than two datasets")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for n, d in DATASETS.items():
            ok, why = available(n)
            print(f"{n:16s} {d.tier:8s} {'available' if ok else 'unavailable: ' + why}  {d.note}")
        return 0
    if a.cmd == "matrix":
        print(json.dumps(matrix(a.tier)))
        return 0
    if a.cmd == "coverage":
        table, short = coverage_table()
        print(table)
        if short:
            print(f"features with fewer than two datasets: {', '.join(short)}", file=sys.stderr)
        return 1 if a.check and short else 0
    if a.cmd == "fetch":
        for n in _selected(a):
            ok, why = available(n)
            if ok and DATASETS[n].public and DATASETS[n].kind == "relational":
                say(f"{n}: {DATASETS[n].location()}")
            elif not ok:
                say(f"{n}: skipped ({why})")
        return 0
    if a.cmd == "run":
        from .runner import run
        path = run([] if a.fake_only else _selected(a), _seeds(a.seeds), fake=not a.no_fake, pushdown=a.pushdown,
                   out_dir=Path(a.out_dir), label=a.label, overrides=tuple(a.overrides), suites=_suites(a.suites),
                   jobs=a.jobs)
        print(path)
        return 0
    if a.cmd == "compare":
        expected = None
        if a.tier:
            expected = {p["datasets"] or "fake" for t in _tiers(a.tier) for p in matrix(t)} if a.hosted else \
                {*(n for t in _tiers(a.tier) for n in tier_datasets(t)), *(["fake"] if "weekly" in a.tier else [])}
        return cmd_compare([Path(r) for r in a.record] or [_newest()], Path(a.baseline), a.full,
                           os.environ.get("GITHUB_STEP_SUMMARY"), expected=expected)
    if a.cmd == "baseline":
        recs = regress.load(a.records)
        base = regress.build(recs, sources=[Path(r).name for r in a.records])
        priv = {k: v for k, v in base["datasets"].items() if is_private(k)}
        Path(a.out).write_text(json.dumps({**base, "datasets": {k: v for k, v in base["datasets"].items()
                                                                 if k not in priv}}, indent=1) + "\n", encoding="utf-8")
        if priv:                                 # never into the repository
            store = private_store()
            store.mkdir(parents=True, exist_ok=True)
            (store / "baseline.json").write_text(json.dumps({**base, "datasets": priv}, indent=1) + "\n",
                                                 encoding="utf-8")
            print(f"private datasets' baseline -> {store / 'baseline.json'}")
        for ds, b in base["datasets"].items():
            print(f"{ds:16s} seeds {b['seeds']}  passed {b['passed']} of {b['judged']}  mean {b['mean']:.4f}  "
                  f"sd {b['sd']:.4f}  spread {b['spread']:.4f}  calibrated "
                  f"{sum(c['calibrated'] for c in b['checks'].values())}/{len(b['checks'])}")
        return 0
    if a.cmd == "all":
        return cmd_all(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())

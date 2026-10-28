# SPDX-License-Identifier: Apache-2.0
"""relsynth benchmark harness.

    python -m bench all                      # every suite: lint, unit, e2e, live servers (if configured),
                                             # golden databases x 5 seeds, fake suite; then compare
    python -m bench fetch                    # download and build the golden databases
    python -m bench run [--datasets ...] [--seeds 1,2,3,4,5] [--no-fake]
    python -m bench compare [RECORD]         # the newest record (or RECORD) against bench/baseline.json
    python -m bench baseline RECORD...       # rebuild bench/baseline.json from history records
    python -m bench list                     # the datasets and whether each is available

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
from .datasets import DATASETS, available, is_private, private_store, say

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "bench" / "history"
BASELINE = ROOT / "bench" / "baseline.json"
GOLDEN = ["bi_corep_lr", "chinook", "adventureworks", "contoso_sqlite", "contoso_parquet"]


def _seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


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


def cmd_compare(record: Path, baseline: Path, full: bool, summary_to: str | None) -> int:
    from .runner import load_record
    base = _load_baseline(baseline)
    rec = load_record(record)
    reg, notes, summ = regress.compare(base, rec, full=full)
    text = f"Record `{record.name}` against `{baseline.name}` ({', '.join(base['relsynth'])})\n\n" + \
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
    path = run(a.datasets or GOLDEN, _seeds(a.seeds), fake=True, pushdown=a.pushdown, label=a.label or "all",
               overrides=tuple(a.overrides))
    rc = cmd_compare(path, Path(a.baseline), full=True, summary_to=os.environ.get("GITHUB_STEP_SUMMARY"))
    if failed:
        print("failed suites: " + "; ".join(failed))
    return 1 if failed or rc else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench", description="relsynth benchmark harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), help="default: every golden database")
        p.add_argument("--seeds", default="1,2,3,4,5")
        p.add_argument("--pushdown", default="auto", choices=["auto", "on", "off"])
        p.add_argument("--label", default="")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="spec override passed to every `relsynth run` (e.g. synthesis.facts.link_degree=auto)")
        p.add_argument("--baseline", default=str(BASELINE))
        if name == "run":
            p.add_argument("--no-fake", action="store_true", help="skip the statistical check suite")
            p.add_argument("--out-dir", default=str(HISTORY))
    p = sub.add_parser("compare")
    p.add_argument("record", nargs="?")
    p.add_argument("--baseline", default=str(BASELINE))
    p.add_argument("--full", action="store_true", help="a dataset of the baseline that was not run is a regression")
    p = sub.add_parser("baseline")
    p.add_argument("records", nargs="+")
    p.add_argument("--out", default=str(BASELINE))
    sub.add_parser("fetch").add_argument("--datasets", nargs="*", choices=sorted(DATASETS))
    sub.add_parser("list")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for n, d in DATASETS.items():
            ok, why = available(n)
            print(f"{n:16s} {'available' if ok else 'unavailable: ' + why}  {d.note}")
        return 0
    if a.cmd == "fetch":
        for n in a.datasets or [n for n in GOLDEN if DATASETS[n].public]:
            say(f"{n}: {DATASETS[n].location()}")
        return 0
    if a.cmd == "run":
        from .runner import run
        path = run(a.datasets or GOLDEN, _seeds(a.seeds), fake=not a.no_fake, pushdown=a.pushdown,
                   out_dir=Path(a.out_dir), label=a.label, overrides=tuple(a.overrides))
        print(path)
        return 0
    if a.cmd == "compare":
        return cmd_compare(Path(a.record) if a.record else _newest(), Path(a.baseline), a.full,
                           os.environ.get("GITHUB_STEP_SUMMARY"))
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

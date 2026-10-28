# SPDX-License-Identifier: Apache-2.0
"""Run the golden databases and the fake suite, and store one history record.

Each real database is synthesised with `relsynth run` in a subprocess (so that
one run's memory never carries into the next), once per seed, and scored by
relsynth's scorecard and by the database's own domain checks.  The fake suite
is the statistical check suite C00-C78 (pytest -m science), whose checks are
seeded internally.  The record holds the version, commit, seed and machine of
every result (bench/README.md describes the format).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET  # noqa: S405 (parses the junit file pytest just wrote)
from pathlib import Path

from .datasets import DATASETS, available, is_private, private_store, say

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def machine() -> dict:
    import numpy
    import pandas
    import scipy
    cpu = platform.processor() or ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"node": os.environ.get("RELSYNTH_BENCH_MACHINE") or platform.node(), "platform": platform.platform(),
            "cpu": cpu, "cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "ci": bool(os.environ.get("CI"))}


def git() -> dict:
    def g(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=False)  # noqa: S603, S607
        return r.stdout.strip()
    return {"sha": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def relsynth_version() -> str:
    from relsynth import __version__
    return __version__


def _source_overrides(location: str) -> list[str]:
    kind, path = location.split(":", 1)
    return ["--set", f"source.kind={kind}", "--set", f"source.path={path}"]


def run_real(name: str, seed: int, pushdown: str = "auto") -> dict:
    ds = DATASETS[name]
    location = ds.location()
    rec = {"dataset": name, "seed": seed, "pushdown": pushdown}
    with tempfile.TemporaryDirectory(prefix=f"bench-{name}-") as tmp:
        out = Path(tmp) / "synthetic"
        cmd = [sys.executable, "-m", "relsynth", "run", str(ds.spec()), *_source_overrides(location),
               "--target", f"csv:{out}", "--seed", str(seed), "--pushdown", pushdown, "--bundle", str(Path(tmp) / "bundle.json"),
               "--json"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, check=False,  # noqa: S603
                           env={**os.environ, "RELSYNTH_QUIET": "1"})
        rec["seconds"] = round(time.time() - t0, 1)
        try:
            doc = json.loads(p.stdout)
        except json.JSONDecodeError:
            rec["error"] = (p.stderr or p.stdout)[-2000:]
            return rec
        rec["exit_code"] = doc.get("exit_code")
        rec["rows"] = doc.get("rows")
        rec["pushed"] = sorted((doc.get("pushdown") or {}).get("pushed") or {})
        if doc.get("errors"):
            rec["error"] = json.dumps(doc["errors"])[-2000:]
        elif p.returncode not in (0, 1):             # 1 is a quality gate, which the scorecard reports anyway
            rec["error"] = f"relsynth exited with {p.returncode}: " + (p.stderr or "")[-2000:]
        sc = doc.get("scorecard") or {}
        rec["scorecard"] = {"passed": sc.get("passed"), "total": sc.get("total")}
        # every scorecard item is in the diagnostics bundle (aggregates only)
        bundle = Path(tmp) / "bundle.json"
        items = ((json.loads(bundle.read_text()).get("evaluation") or {}).get("scorecard") or {}).get("items", []) \
            if bundle.exists() else []
        checks = [{"id": f"scorecard:{i['area']}|{i['metric']}", "pass": bool(i["pass"]), "value": i.get("value"),
                   "limit": i.get("limit")} for i in items]
        if ds.checks is not None and out.exists():
            try:
                for k, (real, syn) in ds.checks(location, out).items():
                    checks.append(domain_item(k, real, syn))
            except Exception as e:  # noqa: BLE001 (a failing domain check is a result, not a crash)
                rec["error"] = f"domain checks failed: {e!r}"
        rec["checks"] = checks
    return rec


def domain_item(name: str, real, syn) -> dict:
    """A domain check passes when the synthetic value equals the real one on an
    identity (a share of 0 or 1: every row obeys the rule, or none breaks it);
    other measures (distributions, counts) are recorded without a verdict."""
    def f(v):
        try:
            return None if v is None else round(float(v), 6)
        except (TypeError, ValueError):
            return None
    r, s = f(real), f(syn)
    identity = r is not None and r in (0.0, 1.0) and not name.startswith("T17")
    return {"id": f"domain:{name}", "pass": (s == r) if identity else None, "value": s, "real": r}


def run_fake(shard: str | None = None) -> dict:
    """The statistical check suite C00-C78, one result per check."""
    rec = {"dataset": "fake", "seed": None}
    with tempfile.TemporaryDirectory(prefix="bench-fake-") as tmp:
        junit = Path(tmp) / "science.xml"
        env = {**os.environ}
        if shard:
            env["RELSYNTH_SCIENCE_SHARD"] = shard
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "pytest", "-m", "science", "-q", "-p", "no:cacheprovider",  # noqa: S603
                        f"--junitxml={junit}", str(ROOT / "tests" / "science")], cwd=ROOT, env=env, check=False,
                       stdout=subprocess.DEVNULL)
        rec["seconds"] = round(time.time() - t0, 1)
        checks = []
        if junit.exists():
            for tc in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314
                m = re.search(r"\[(C\d+)\]", tc.get("name", ""))
                if not m:
                    continue
                skipped = tc.find("skipped") is not None
                failed = tc.find("failure") is not None or tc.find("error") is not None
                checks.append({"id": f"science:{m.group(1)}", "pass": None if skipped else not failed})
        else:
            rec["error"] = "pytest wrote no junit report"
        rec["checks"] = checks
    return rec


def private_path(record_path: Path) -> Path:
    store = private_store()
    if store is None:
        raise RuntimeError("a private dataset ran without RELSYNTH_BENCH_BI_DIR")
    return store / "history" / Path(record_path).name


def load_record(path: Path) -> dict:
    """A record with the runs of its private part, where the private store has it."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    store = private_store()
    pp = store / "history" / Path(path).name if store else None
    if pp is not None and pp.exists():
        priv = json.loads(pp.read_text(encoding="utf-8"))["runs"]
        rec["runs"] = [r for r in rec["runs"] if not r.get("private")] + priv
    return rec


def run(datasets: list[str], seeds: list[int], *, fake: bool = True, pushdown: str = "auto",
        out_dir: Path | None = None, label: str = "") -> Path:
    record = {"schema": SCHEMA, "relsynth": relsynth_version(), "git": git(), "machine": machine(),
              "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "label": label,
              "seeds": seeds, "runs": []}
    out_dir = Path(out_dir or ROOT / "bench" / "history")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["started"].replace(":", "").replace("-", "")[:15]
    node = re.sub(r"[^A-Za-z0-9]+", "-", record["machine"]["node"])[:24]
    base = f"{stamp}-{record['git']['sha'][:7] or 'nogit'}-{node}"
    path = out_dir / f"{base}.json"
    k = 1
    while True:                                  # claim a name: runs may start in the same second (in parallel)
        try:
            path.open("x").close()
            break
        except FileExistsError:
            k += 1
            path = out_dir / f"{base}-{k}.json"

    def save():
        # results of a private dataset (even aggregates) stay out of the repository: they go to the same
        # file name in the private store, and the public record keeps only a note that they ran
        pub = {**record, "runs": [r if not is_private(r["dataset"]) or "skipped" in r else
                                  {"dataset": r["dataset"], "seed": r.get("seed"), "private": True}
                                  for r in record["runs"]]}
        path.write_text(json.dumps(pub, indent=1) + "\n", encoding="utf-8")
        priv = [r for r in record["runs"] if is_private(r["dataset"]) and "skipped" not in r]
        if priv:
            pp = private_path(path)
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(json.dumps({**record, "runs": priv}, indent=1) + "\n", encoding="utf-8")

    save()                                       # a record is valid JSON from the start

    for name in datasets:
        ok, why = available(name)
        if not ok:
            say(f"{name}: skipped ({why})")
            record["runs"].append({"dataset": name, "skipped": why})
            continue
        for seed in seeds:
            say(f"{name} seed {seed} ...")
            r = run_real(name, seed, pushdown)
            sc = r.get("scorecard") or {}
            say(f"{name} seed {seed}: scorecard {sc.get('passed')}/{sc.get('total')}, "
                f"{sum(1 for c in r.get('checks', []) if c['id'].startswith('domain') and c['pass'] is False)} domain "
                f"identities broken, {r.get('seconds')} s" + (f", ERROR {r['error'][:200]}" if r.get("error") else ""))
            record["runs"].append(r)
            save()
    if fake:
        say("fake suite (pytest -m science) ...")
        r = run_fake(os.environ.get("RELSYNTH_SCIENCE_SHARD"))
        say(f"fake suite: {sum(1 for c in r['checks'] if c['pass'])}/{len(r['checks'])} checks pass, {r['seconds']} s")
        record["runs"].append(r)
    record["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    save()
    return path

# SPDX-License-Identifier: Apache-2.0
"""The regression rule of the benchmark harness (bench/regress.py)."""
from __future__ import annotations

import pytest

from bench import regress
from bench.runner import domain_item


def _run(ds, seed, fails=(), n=20, error=None):
    r = {"dataset": ds, "seed": seed, "checks": [{"id": f"c{i}", "pass": f"c{i}" not in fails} for i in range(n)]}
    if error:
        r["error"] = error
    return r


def _baseline():
    # c19 failed on seed 3: unstable; every other check calibrated
    rec = {"relsynth": "1.0.0", "runs": [_run("db", s, fails=("c19",) if s == 3 else ()) for s in range(1, 6)]}
    return regress.build([rec])


def test_build_statistics():
    b = _baseline()["datasets"]["db"]
    assert b["passed"] == [20, 20, 19, 20, 20]
    assert b["mean"] == pytest.approx(0.99)
    # spread = max(2 sd, 1 / 20); sd of (1, 1, .95, 1, 1) = .02236
    assert b["spread"] == pytest.approx(0.05, abs=1e-6)
    assert b["checks"]["c0"]["calibrated"] and not b["checks"]["c19"]["calibrated"]
    assert b["checks"]["c19"]["failed_seeds"] == [3]


def test_same_results_no_regression():
    reg, _, _ = regress.compare(_baseline(), {"runs": [_run("db", 1)]})
    assert reg == []


def test_unstable_check_flip_is_not_r1():
    reg, _, _ = regress.compare(_baseline(), {"runs": [_run("db", 1, fails=("c19",))]})
    assert reg == []


def test_calibrated_check_failing_is_r1():
    reg, _, _ = regress.compare(_baseline(), {"runs": [_run("db", 1, fails=("c4",))]})
    assert any(r.startswith("R1 db seed 1") and "c4" in r for r in reg)


def _stat(seed, fails=()):
    # c0 exact (limit 0), c1 statistical (positive limit), as the scorecard records them
    return {"dataset": "db", "seed": seed, "checks": [{"id": "c0", "pass": "c0" not in fails, "value": 0, "limit": 0},
                                                      {"id": "c1", "pass": "c1" not in fails, "value": 0.1, "limit": 0.2}]}


def test_statistical_check_regresses_only_on_most_seeds():
    base = regress.build([{"runs": [_stat(s) for s in range(1, 6)]}])
    # one or two seeds of five: a flip on this machine, reported as a note
    reg, notes, _ = regress.compare(base, {"runs": [_stat(s, ("c1",) if s in (2, 4) else ()) for s in range(1, 6)]})
    assert reg == [] and any("c1 failed on 2 of 5" in n for n in notes)
    # three of five: the check is broken
    reg, _, _ = regress.compare(base, {"runs": [_stat(s, ("c1",) if s <= 3 else ()) for s in range(1, 6)]})
    assert len([r for r in reg if r.startswith("R1") and "c1" in r]) == 3
    # an exact check regresses on a single seed
    reg, _, _ = regress.compare(base, {"runs": [_stat(s, ("c0",) if s == 5 else ()) for s in range(1, 6)]})
    assert [r for r in reg if r.startswith("R1")] == [next(r for r in reg if r.startswith("R1 db seed 5"))]
    # a one-seed run (the pull-request job) keeps the old rule
    reg, _, _ = regress.compare(base, {"runs": [_stat(1, ("c1",))]})
    assert any(r.startswith("R1 db seed 1") for r in reg)


def test_pass_rate_drop_is_r2():
    # three unstable-check failures cannot be R1, but drop the rate by 0.10 > spread
    base = _baseline()
    for k in ("c17", "c18"):
        base["datasets"]["db"]["checks"][k]["calibrated"] = False
    reg, _, _ = regress.compare(base, {"runs": [_run("db", 1, fails=("c17", "c18", "c19"))]})
    assert [r for r in reg if r.startswith("R2")] and not [r for r in reg if r.startswith("R1")]


def test_errors_and_missing_datasets_are_r3():
    reg, _, _ = regress.compare(_baseline(), {"runs": [_run("db", 1, error="boom")]})
    assert any(r.startswith("R3") for r in reg)
    reg, _, _ = regress.compare(_baseline(), {"runs": []}, full=True)
    assert reg == ["R3 db: in the baseline but not run"]
    reg, notes, _ = regress.compare(_baseline(), {"runs": [{"dataset": "db", "skipped": "private"}]}, full=True)
    assert reg == [] and notes


def test_domain_item_judges_identities_only():
    assert domain_item("B1 duplicates", 0, 0)["pass"] is True
    assert domain_item("B2 share", 1.0, 0.98)["pass"] is False
    assert domain_item("T3 mean basket", 2.4, 2.5)["pass"] is None
    assert domain_item("T17 KS", 0.0, 0.1)["pass"] is None


def test_private_results_stay_out_of_the_public_record(tmp_path, monkeypatch):
    from bench import runner
    bi = tmp_path / "bi"
    bi.mkdir()
    (bi / "spec.yaml").write_text("name: x\n")
    monkeypatch.setenv("RELSYNTH_BENCH_BI_DIR", str(bi))
    monkeypatch.setattr(runner, "run_real", lambda name, seed, pushdown: _run(name, seed))
    path = runner.run(["bi_corep_lr", "chinook"], [1], fake=False, out_dir=tmp_path / "history")
    public = path.read_text()
    assert '"c0"' in public and public.count('"c0"') == 1          # chinook's checks only
    assert '"private": true' in public
    private = (bi / "bench" / "history" / path.name).read_text()
    assert "bi_corep_lr" in private and "chinook" not in private
    rec = runner.load_record(path)
    assert sorted(r["dataset"] for r in rec["runs"]) == ["bi_corep_lr", "chinook"]
    assert all("checks" in r for r in rec["runs"])

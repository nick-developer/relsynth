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
    monkeypatch.setattr(runner, "run_real", lambda name, seed, *_a: _run(name, seed))
    path = runner.run(["bi_corep_lr", "chinook"], [1], fake=False, out_dir=tmp_path / "history")
    public = path.read_text()
    assert '"c0"' in public and public.count('"c0"') == 1          # chinook's checks only
    assert '"private": true' in public
    private = (bi / "bench" / "history" / path.name).read_text()
    assert "bi_corep_lr" in private and "chinook" not in private
    rec = runner.load_record(path)
    assert sorted(r["dataset"] for r in rec["runs"]) == ["bi_corep_lr", "chinook"]
    assert all("checks" in r for r in rec["runs"])


def _tracked(seed, value, kind=None, direction="higher"):
    c = {"id": "metric:m", "pass": None, "value": value, "direction": direction, **({"kind": kind} if kind else {})}
    return {"dataset": "db", "seed": seed, "checks": [{"id": "c0", "pass": True}, c]}


def test_r4_drift_of_a_tracked_value():
    base = regress.build([{"runs": [_tracked(s, 0.90 + 0.001 * s) for s in range(1, 6)]}])
    t = base["datasets"]["db"]["tracked"]["metric:m"]
    assert t["n"] == 5 and t["direction"] == "higher"
    # within the floor of 0.02: no regression
    reg, _, _ = regress.compare(base, {"runs": [_tracked(s, 0.89) for s in range(1, 4)]})
    assert reg == []
    # worse by 0.05 on average: R4
    reg, _, _ = regress.compare(base, {"runs": [_tracked(s, 0.85) for s in range(1, 4)]})
    assert any(r.startswith("R4 db metric:m") for r in reg)
    # better is never a regression
    reg, _, _ = regress.compare(base, {"runs": [_tracked(s, 0.99) for s in range(1, 4)]})
    assert reg == []


def test_r4_cost_values_are_notes_only():
    base = regress.build([{"runs": [_tracked(s, 100.0, "cost", "lower") for s in range(1, 6)]}])
    reg, notes, _ = regress.compare(base, {"runs": [_tracked(1, 200.0, "cost", "lower")]})
    assert reg == [] and any("cost metric:m" in n for n in notes)


def test_quality_and_privacy_items():
    from bench.assess import privacy_checks, quality_checks
    rep = {"checks": [
        {"id": "w1:t:a", "metric": "w1_rank_normal", "area": "fidelity", "value": 0.01, "limit": 0.05, "pass": True,
         "alpha": 0.01, "source": "quality-2.0"},
        {"id": "fk_integrity:t", "metric": "fk_integrity", "area": "integrity", "value": 1.0, "limit": 1.0,
         "pass": True, "source": "scorecard-1.0"},
        {"id": "tstr:t", "metric": "tstr_gap", "area": "utility", "value": 0.08, "limit": 0.05, "pass": False,
         "source": "quality-2.0"}],
        "vector": {"fidelity": {"checks": 1, "pass": True, "pass_rate": 1.0},
                   "utility": {"checks": 1, "pass": False, "pass_rate": 0.0}, "integrity": {"checks": 1, "pass": True,
                                                                                        "pass_rate": 1.0}},
        "benchmark": {"syntherela": {"c2st_agg_accuracy_max": 0.7, "cardinality_similarity": {"a": 0.9, "b": 0.8},
                                     "k_hop_trend": {"1": 0.95}}}}
    q = {c["id"]: c for c in quality_checks(rep)}
    assert "quality:fk_integrity:t" not in q                      # 1.0 items stay scorecard items
    assert q["quality:w1:t:a"]["pass"] and not q["quality:tstr:t"]["pass"]
    assert not q["area:utility"]["pass"] and q["area:utility"]["limit"] == 0
    assert q["goal:referential_integrity"]["pass"] and q["goal:distribution_match"]["pass"]
    assert not q["goal:utility_gap"]["pass"]
    assert q["metric:cardinality_similarity_min"]["value"] == 0.8
    prep = {"release": {"dp": None}, "guards": {"dcr": {"ratio": 0.9}}, "contexts": {
        "internal": {"external": False, "verdict": "review", "checks": [
            {"criterion": "linkage", "attack": "nn", "value": 0.0, "ci": [0, 0.01], "limit": 0.1, "status": "pass"},
            {"criterion": "membership", "attack": "rec", "value": 0.7, "ci": [0.4, 0.9], "limit": 0.85,
             "status": "info"}]},
        "public": {"external": True, "verdict": "blocked", "checks": []}}}
    p = {c["id"]: c for c in privacy_checks(prep)}
    assert p["privacy:linkage|nn"]["pass"] and p["privacy:membership|rec"]["pass"] is None
    assert p["goal:privacy_internal"]["pass"] and p["goal:privacy_public_needs_dp"]["pass"]
    assert p["metric:privacy_guard_dcr_ratio"]["direction"] == "higher"


def test_domain_check_modules_hold_on_toy_data():
    import pandas as pd

    from bench.checks import acs, berka
    a = pd.DataFrame({"AGEP": [10, 30, 70], "INDP": [None, 770, 8680], "INDP_CAT": [None, 1, 2],
                      "PINCP": [None, 100.0, 9000.0], "PINCP_DECILE": [None, 1, 9], "EDU": [1, 5, 9],
                      "DREM": [2, 2, 1], "HOUSING_TYPE": [1, 1, 2], "OWN_RENT": [1, 2, 0], "WGTP": [5, 7, 0],
                      "PUMA": ["a", "a", "b"], "DENSITY": [1.0, 1.0, 2.0], "POVPIP": [100, 501, None]})
    r = acs.checks({"acs": a}, {"acs": a})
    assert all(r[k] == 1.0 for k in r if k[:3] in {f"A{i} " for i in range(1, 10)} | {"A10", "A11", "A12"})
    T = {"account": pd.DataFrame({"account_id": [1, 2], "district_id": [1, 1], "date": ["2020-01-01", "2020-02-01"]}),
         "client": pd.DataFrame({"client_id": [1, 2], "district_id": [1, 1]}),
         "disp": pd.DataFrame({"disp_id": [1, 2], "client_id": [1, 2], "account_id": [1, 2], "type": ["OWNER"] * 2}),
         "card": pd.DataFrame({"card_id": [1], "disp_id": [1], "issued": ["2020-03-01"]}),
         "loan": pd.DataFrame({"loan_id": [1], "account_id": [1], "amount": [120], "duration": [12], "payments": [10],
                               "status": ["A"]}),
         "trans": pd.DataFrame({"trans_id": [1, 2, 3], "account_id": [1, 1, 2],
                                "date": ["2020-01-01", "2020-01-05", "2020-02-01"], "type": ["PRIJEM", "PRIJEM", "PRIJEM"],
                                "operation": ["VKLAD", None, "VKLAD"], "k_symbol": [None, "UROK", None],
                                "amount": [10.0, 1.0, 5.0]})}
    r = berka.checks(T, T)
    assert all(r[k] == 1.0 for k in r if k.split()[0] in {f"F{i}" for i in range(1, 12)})
    assert r["F15 loan status, total variation from the bank"] == 0.0


def test_new_kinds_of_check_do_not_move_r2():
    # a new suite adds checks that fail; R2 compares only the checks the baseline has
    run = _run("db", 1)
    run["checks"] += [{"id": f"quality:q{i}", "pass": False} for i in range(10)]
    reg, notes, summ = regress.compare(_baseline(), {"runs": [run]})
    assert reg == [] and summ["db"]["mean"] == 1.0


def test_baseline_skips_runs_without_verdicts():
    from bench import regress
    ok = {"dataset": "d", "seed": 1, "checks": [{"id": "a", "pass": True}]}
    bad = {"dataset": "d", "seed": 2, "error": "boom", "checks": []}
    ok2 = {"dataset": "d", "seed": 3, "checks": [{"id": "a", "pass": True}]}
    b = regress.build([{"runs": [ok, bad, ok2]}])["datasets"]["d"]
    assert b["mean"] == 1.0 and b["sd"] == 0.0 and b["errors"] == 1
    assert b["pass_rates"] == [1.0, None, 1.0]


def test_dataset_that_only_errored_has_no_mean_and_compares():
    import json
    from bench import regress
    bad = {"dataset": "d", "seed": 1, "error": "boom", "checks": []}
    b = regress.build([{"runs": [bad]}])
    assert b["datasets"]["d"]["mean"] is None
    json.dumps(b, allow_nan=False)
    ok = {"dataset": "d", "seed": 1, "checks": [{"id": "a", "pass": True}]}
    regs, _notes, _summary = regress.compare(b, {"runs": [ok]})
    assert not any(x.startswith("R2") for x in regs)

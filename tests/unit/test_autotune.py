# SPDX-License-Identifier: Apache-2.0
"""The self-tuning loop (relsynth.autotune): objectives, cross-seed judging,
proposals, history, the loop on planted response surfaces (a fixable misfit,
an insensitive family, a privacy trade-off, sporadic noise) and the report.
One end-to-end run on a real fit closes the file."""
from __future__ import annotations

import copy
import json
import math

import pytest

from relsynth.autotune import analysis as A
from relsynth.autotune import loop as L
from relsynth.autotune import search as Sx
from relsynth.autotune.checks import TrialResult, constraint_of, family
from relsynth.autotune.history import History
from relsynth.autotune.objectives import ObjectivesError, parse
from relsynth.settings import params


# -------------------------------------------------------------- objectives
def test_objectives_defaults_and_errors():
    o = parse(None)
    assert o.hard == ["referential_integrity", "rules", "unique_keys", "trees", "privacy", "domain"]
    assert o.families is None and o.seeds_search == [1, 2, 3] and o.runs == 40
    o = parse({"targets": [{"family": "join_tvd"}, {"scorecard_pass_rate": 0.97}], "constraints": ["rules"]})
    assert o.families == ["join_tvd"] and o.min_pass_rate == 0.97 and o.hard == ["rules"]
    for bad, needle in [({"nope": 1}, "unknown key"), ({"constraints": ["links"]}, "unknown constraint"),
                        ({"seeds": {"search": [1], "holdout": [1]}}, "must not repeat"),
                        ({"seeds": {"search": []}}, "at least one seed"), ({"budget": {"runs": 0}}, "positive"),
                        ({"search": ["copula.sampl_rows"]}, "unknown setting"),
                        ({"privacy": {"review": "maybe"}}, "undecided, pass or fail"),
                        ({"targets": [{"family": 3}]}, "is not 'all'"), ({"null_quantile": 1.0}, "[0.5, 1)")]:
        with pytest.raises(ObjectivesError, match=needle.replace("[", r"\[").replace(")", r"\)")):
            parse(bad)


def test_check_families_and_constraints():
    assert family("scorecard:sales.CustomerKey->customer|cardinality_ks") == "cardinality_ks"
    assert family("quality:kway2_tvd_max:sales") == "kway2_tvd_max"
    assert family("privacy:inference|nearest_neighbour_inference:DREM") == "privacy:inference"
    assert constraint_of("scorecard:sales.k->c|fk_integrity") == "referential_integrity"
    assert constraint_of("scorecard:sales: x > 0|rule_violation_rate") == "rules"
    assert constraint_of("scorecard:product|dcr_ratio") == "privacy"
    assert constraint_of("scorecard:sales|marginal_ks") is None


# ----------------------------------------------------------------- judging
def _c(cid, ok, alpha=None, status=None):
    d = {"id": cid, "pass": ok, "value": 0.0, "limit": 0.1}
    if alpha:
        d["alpha"] = alpha
    if status:
        d["status"] = status
    return d


def test_calibrated_families_and_hard_constraints():
    obj = parse({"privacy": {"review": "undecided"}})
    checks = [_c(f"scorecard:t.c{i}|marginal_ks", i >= 1, alpha=0.01) for i in range(100)]    # 1 failure, mu 1
    checks += [_c(f"scorecard:t.k{i}|cardinality_ks", i >= 6, alpha=0.01) for i in range(100)]  # 6 failures
    checks += [_c("scorecard:s.k->c|fk_integrity", True), _c("scorecard:t|dcr_ratio", True),
               _c("privacy:inference|nn:DREM", False, status="review")]
    v = A.view({"seed": 1, "checks": checks}, obj)
    assert v.families["marginal_ks"]["q"] == 4 and v.families["marginal_ks"]["failed"] == 1   # Poisson(1) q99
    assert v.families["cardinality_ks"]["failed"] == 6 and v.excess() == 2.0
    assert v.hard_violations == 0 and v.undecided == ["privacy:inference|nn:DREM"]
    assert A.view({"seed": 1, "checks": checks}, parse({"privacy": {"review": "fail"}})).hard_violations == 1
    assert A.view({"seed": 1, "checks": checks}, parse({"privacy": {"review": "pass"}})).undecided == []
    v2 = A.view({"seed": 1, "checks": [_c("scorecard:s.k->c|fk_integrity", False)]}, obj)
    assert v2.hard == {"referential_integrity": ["scorecard:s.k->c|fk_integrity"]}
    # a constraint the objectives leave out is judged as a statistical family
    v3 = A.view({"seed": 1, "checks": [_c("scorecard:s.k->c|fk_integrity", False)]}, parse({"constraints": []}))
    assert v3.hard_violations == 0 and v3.families["fk_integrity"]["failed"] == 1


def test_seed_evidence_numbers():
    assert A.beta_upper(0, 3) == pytest.approx(1 - 0.05 ** 0.25, abs=1e-4)
    assert A.seeds_needed(0.05) == 58 and A.seeds_needed(0.20) == 13
    assert A.null_quantile(0.0, 0.99) == 0 and A.null_quantile(4.41, 0.99) == 10


def test_systematic_versus_sporadic():
    trials = [{"seed": s, "checks": [_c("scorecard:a|marginal_ks", False, alpha=0.01),
                                     _c("scorecard:b|marginal_ks", s != 2, alpha=0.01),
                                     _c("scorecard:c|marginal_ks", s == 2, alpha=0.3)]} for s in (1, 2, 3)]
    sysf = A.systematic(trials)
    assert set(sysf) == {"scorecard:a|marginal_ks"}       # 3/3 at alpha 0.01; 1/3 is noise at either level


def test_better_is_lexicographic_and_paired():
    obj = parse(None)
    mk = lambda fails, hard=False: A.judge([{"seed": s, "checks": [_c(f"scorecard:t{i}|join_tvd", i >= f)  # noqa: E731
                                                                   for i in range(10)]
                                             + [_c("scorecard:k->p|fk_integrity", not hard)]}
                                            for s, f in zip((1, 2, 3), fails)], obj)
    ok, _ = A.better(mk([1, 1, 1]), mk([3, 3, 3]))
    assert ok
    ok, why = A.better(mk([0, 0, 0], hard=True), mk([3, 3, 3]))
    assert not ok and "hard" in why                        # a hard violation is never traded for a target
    ok, _ = A.better(mk([3, 3, 3]), mk([3, 3, 3], hard=True))
    assert ok
    ok, _ = A.better(mk([2, 4, 2]), mk([3, 3, 3]))         # mean 0, no improvement
    assert not ok


# --------------------------------------------------------------- proposals
def test_steps_follow_scale_and_bounds():
    P = params()
    assert Sx.step(P["copula.sample_rows"], 1500, +1) == 3000
    assert Sx.step(P["copula.sample_rows"], 1500, -1) == 750
    assert Sx.step(P["structure.structural_prior"], 0.9, +1) == 1.0
    assert Sx.step(P["structure.structural_prior"], 1.0, +1) is None          # at the bound
    assert Sx.step(P["gibbs.merge"], True, +1) is False
    assert Sx.step(P["text.order"], 3, +1) == 4


def test_guided_moves_use_directions_and_learned_effects():
    space = {k: p for k, p in params().items() if k in ("copula.sample_rows", "copula.lkj_eta", "marginals.max_obs")}
    moves = Sx.guided({"association_max_diff": 2.0}, {}, {}, {}, space)
    assert moves[0] in ({"copula.sample_rows": 3000}, {"copula.lkj_eta": 1.0})
    assert {"copula.sample_rows": 750} not in moves                            # against the registry direction
    learned = {"association_max_diff": {"copula.sample_rows": [0.0, 0.1]}}      # tried: no effect
    moves2 = Sx.guided({"association_max_diff": 2.0}, {}, {}, {}, space, learned)
    assert moves2[0] == {"copula.lkj_eta": 1.0}


# ----------------------------------------------------------------- history
def test_history_persists_and_isolates_fingerprints(tmp_path):
    h = History(tmp_path, "fp1")
    h.add("k1", {"a": 1}, {}, {"seed": 1, "checks": []})
    h.set_incumbent("k1", {"a": 1}, {})
    h2 = History(tmp_path, "fp1")
    assert h2.get("k1", 1) is not None and h2.incumbent()["delta"] == {"a": 1}
    h3 = History(tmp_path, "fp2")
    assert h3.get("k1", 1) is None and h3.incumbent() is None
    with (tmp_path / "history.jsonl").open("a") as f:
        f.write('{"cut line')                                                  # an interrupted write
    assert History(tmp_path, "fp1").get("k1", 1) is not None
    h3.add("k9", {}, {}, {"seed": 1, "checks": []})
    History(tmp_path, "fp1").clear()
    assert History(tmp_path, "fp1").get("k1", 1) is None and History(tmp_path, "fp2").get("k9", 1) is not None


# ------------------------------------------------- the loop on planted surfaces
def _surface(conflict=False, hard_break=False):
    """A fake trial: marginal_tvd fails 3, 2, 1, 0 times for max_levels 2, 4, 8, >=16;
    novel_combination_rate fails 4 times whatever the settings (insensitive); join_tvd
    fails once on seed 2 (noise within its level); with conflict, max_levels >= 16
    breaks the DCR privacy check; with hard_break, copula.sample_rows above 1500
    breaks a foreign key."""
    calls = []

    def run_trial(raw, base, seed, delta, patches, obj):
        calls.append((seed, dict(delta)))
        ml = delta.get("marginals.max_levels", raw["synthesis"]["marginals"]["max_levels"])
        f = 3 if ml <= 2 else 2 if ml < 8 else 1 if ml < 16 else 0
        ch = [_c(f"scorecard:t.c{i}|marginal_tvd", i >= f, alpha=0.0001) for i in range(10)]
        ch += [_c(f"scorecard:t{i}|novel_combination_rate", i >= 4, alpha=0.001) for i in range(6)]
        ch += [_c(f"scorecard:t{i}|join_tvd", not (seed == 2 and i == 0), alpha=0.02) for i in range(10)]
        ch.append(_c("scorecard:t|dcr_ratio", not (conflict and ml >= 16)))
        ch.append(_c("scorecard:s.k->t|fk_integrity",
                     not (hard_break and delta.get("copula.sample_rows", 1500) > 1500)))
        return TrialResult(seed=seed, checks=ch, seconds=0.0)
    return run_trial, calls


RAW = {"version": 1, "name": "planted", "source": {"kind": "csv", "path": "nowhere"},
       "tables": {"t": {"role": "entity", "columns": {"id": "id", "x": "categorical"}, "primary_key": "id"}},
       "synthesis": {"marginals": {"max_levels": 2}}}


def _obj(**kw):
    d = {"seeds": {"search": [1, 2, 3], "holdout": [11]}, "budget": {"runs": 80},
         "search": ["marginals.max_levels", "copula.sample_rows", "structure.beta_prior"], "spec_patches": False}
    d.update(kw)
    return parse(d)


def test_loop_finds_the_planted_fix_and_explains_the_rest(tmp_path, monkeypatch):
    fn, calls = _surface()
    monkeypatch.setattr(L, "run_trial", fn)
    res = L.autotune(copy.deepcopy(RAW), _obj(), history_dir=tmp_path)
    assert res.delta.get("marginals.max_levels", 0) >= 16
    fam = res.analysis["families"]
    assert fam["marginal_tvd"]["met"] and fam["join_tvd"]["met"]          # noise on seed 2 is left alone
    assert not fam["novel_combination_rate"]["met"]
    assert fam["novel_combination_rate"]["reason"] in ("insensitive", "budget")
    assert res.confirmed is True and res.holdout["tuned"]["seeds"] == [11]
    assert not res.met                                                      # novel_combination_rate stays unmet
    md = res.report_markdown
    assert "marginals.max_levels" in md and "novel_combination_rate" in md and "Trajectory" in md
    # resuming reuses every run already made
    n = len(calls)
    res2 = L.autotune(copy.deepcopy(RAW), _obj(), history_dir=tmp_path)
    assert res2.runs_reused >= 3 * 2                                       # incumbent and defaults, every search seed
    assert res2.judgement["excess_failures"] <= res.judgement["excess_failures"]
    assert len(calls) - n <= 80


def test_loop_reports_the_privacy_trade_off(tmp_path, monkeypatch):
    fn, _ = _surface(conflict=True)
    monkeypatch.setattr(L, "run_trial", fn)
    res = L.autotune(copy.deepcopy(RAW), _obj(), history_dir=tmp_path)
    assert 8 <= res.delta["marginals.max_levels"] < 16                     # the best that keeps privacy
    assert res.analysis["constraints"]["privacy"]["met"]
    assert res.analysis["families"]["marginal_tvd"]["reason"] == "conflict"
    assert any("constraint:privacy" in k and "marginal_tvd" in k for k in res.analysis["conflicts"])


def test_loop_never_accepts_a_hard_violation(tmp_path, monkeypatch):
    fn, _ = _surface(hard_break=True)
    monkeypatch.setattr(L, "run_trial", fn)
    res = L.autotune(copy.deepcopy(RAW), _obj(search=["copula.sample_rows", "marginals.max_levels"]),
                     history_dir=tmp_path)
    assert res.delta.get("copula.sample_rows", 1500) <= 1500
    assert res.judgement["hard_violations"] == 0


def test_budget_keeps_runs_for_the_holdout(tmp_path, monkeypatch):
    fn, calls = _surface()
    monkeypatch.setattr(L, "run_trial", fn)
    res = L.autotune(copy.deepcopy(RAW), _obj(budget={"runs": 9}), history_dir=tmp_path)
    assert res.runs <= 9 and res.holdout is not None and res.stopped == "budget"
    with pytest.raises(Exception, match="does not cover"):
        L.autotune(copy.deepcopy(RAW), _obj(budget={"runs": 4}), history_dir=tmp_path / "b", resume=False)


def test_loop_refuses_dp_mode(tmp_path):
    raw = copy.deepcopy(RAW)
    raw["synthesis"]["privacy"] = {"dp": {"epsilon": 1.0, "unit": "t"}}
    from relsynth.errors import PrivacyError
    with pytest.raises(PrivacyError, match="refused in DP mode"):
        L.autotune(raw, _obj(), history_dir=tmp_path)


def test_report_states_the_noise_floor(tmp_path, monkeypatch):
    fn, _ = _surface()
    monkeypatch.setattr(L, "run_trial", fn)
    res = L.autotune(copy.deepcopy(RAW), _obj(budget={"runs": 12}), history_dir=tmp_path)
    an = res.analysis
    mu = 10 * 0.0001 + 6 * 0.001 + 10 * 0.02 + 0.0 + 0.0
    assert an["expected_false_failures"] == pytest.approx(mu, abs=1e-3)
    assert an["p_all_statistical_pass_perfect_model"] == pytest.approx(math.exp(-mu), abs=1e-4)
    json.dumps(res.to_dict(), default=str)                                   # the JSON report serialises


# ---------------------------------------------------------- end to end (real fits)
@pytest.mark.slow
def test_autotune_end_to_end_on_a_planted_misfit(tmp_path, star_csv):
    from relsynth.autotune import autotune
    spec, _ = star_csv
    s = copy.deepcopy(spec)
    s["synthesis"] = {"gibbs": {"iterations": 20, "final_iterations": 5, "max_final_iterations": 10},
                      "copula": {"iterations": 40, "burn_in": 20, "max_iterations": 80},
                      "marginals": {"dirichlet_prior": 1e4}}         # segments shrunk onto the table's law
    obj = parse({"seeds": {"search": [1, 2], "holdout": [3]}, "budget": {"runs": 16},
                 "search": ["marginals.dirichlet_prior", "copula.sample_rows"], "spec_patches": False})
    res = autotune(s, obj, history_dir=tmp_path)
    base = res.extra["baseline"]
    assert base["failures"] > 0
    assert res.delta.get("marginals.dirichlet_prior", 1e4) < 1e4
    assert res.judgement["failures"] < base["failures"]
    assert (tmp_path / "history.jsonl").exists()

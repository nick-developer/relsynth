# SPDX-License-Identifier: Apache-2.0
"""Generator selection (synthesis.generator_selection, relsynth.generators.selection): off by
default, auto rules on table traits, trial ranking (privacy first, then the scorecard share),
the user's per-table choice kept, refusal in differential privacy mode, and the decision record."""
from __future__ import annotations

import copy

import pytest

from relsynth.generators import selection as S
from relsynth.pipeline import execute
from relsynth.spec import SpecError, load_spec

from conftest import write_scenario


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    spec = write_scenario("S9_small_shop", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


def _sel(spec, **kw):
    s = copy.deepcopy(spec)
    s.setdefault("synthesis", {})["generator_selection"] = kw
    return s


def test_off_is_the_default_and_changes_nothing(shop):
    spec = load_spec(shop)
    assert spec["synthesis"]["generator_selection"] == {"mode": "off", "seeds": 1, "candidates": None}
    assert S.select(spec, None) is None
    r = execute(spec, write=False, privacy=False)
    assert "generator_selection" not in r.bundle and not r.errors


@pytest.mark.parametrize("raw, msg", [
    ({"mode": "best"}, "mode must be one of"),
    ({"mode": "trial", "seeds": 0}, "seeds must be an integer"),
    ({"mode": "trial", "seeds": 11}, "seeds must be an integer"),
    ({"mode": "trial", "candidates": []}, "non-empty list"),
    ({"mode": "trial", "candidates": ["nope"]}, "unknown generator"),
    ({"mode": "auto", "extra": 1}, "unknown key"),
])
def test_bad_settings_are_refused(shop, raw, msg):
    with pytest.raises(SpecError, match=msg):
        load_spec(_sel(shop, **raw))


def test_refused_in_differential_privacy_mode(shop):
    s = _sel(shop, mode="auto")
    s["synthesis"]["privacy"] = {"dp": {"epsilon": 1.0}}
    with pytest.raises(SpecError, match="not covered by the privacy budget"):
        load_spec(s)


def _table(cols):
    return {"tables": {"t": {"columns": {f"c{i}": {"type": k} for i, k in enumerate(cols)}}}}


@pytest.mark.parametrize("n, cols, available, gpu, want", [
    (500, ["categorical"] * 4, ["copula", "cart", "tvae", "ddpm"], True, "copula"),     # small: parametric
    (80_000, ["float"] * 4, ["copula", "cart", "tvae", "ddpm"], True, "ddpm"),        # large with a GPU
    (80_000, ["float"] * 4, ["copula", "cart", "tvae", "ddpm"], False, "tvae"),       # no GPU: no diffusion
    (5_000, ["categorical", "boolean", "float"], ["copula", "cart"], False, "cart"),  # categorical-heavy
    (5_000, ["float", "float", "categorical"], ["copula", "cart"], False, "copula"),  # mostly numeric
    (50_000, ["float"] * 3, ["copula", "cart"], False, "copula"),                     # neural not installed
    (500, ["float"], ["cart"], False, "cart"),                                        # rule's choice unusable
])
def test_auto_rules(n, cols, available, gpu, want):
    name, why = S.auto_rule(_table(cols), "t", n, available, lambda g: "cuda:0" if gpu and g == "ddpm" else "cpu")
    assert name == want and why


def test_auto_keeps_the_users_choice_and_skips_copied_tables(shop):
    s = load_spec(_sel(shop, mode="auto"))
    s["tables"]["invoice"]["generator"] = "independent"
    tables = S.selectable(s)
    assert "invoice" not in tables and tables
    assert all(s["tables"][t].get("mode") != "copy" for t in tables)
    r = execute(s, write=False, privacy=False)
    sel = r.bundle["generator_selection"]
    assert sel["mode"] == "auto" and set(sel["chosen"]) == set(tables) and set(sel["reasons"]) == set(tables)
    assert r.spec["tables"]["invoice"]["generator"] == "independent"
    assert all(r.spec["tables"][t]["generator"] == g for t, g in sel["chosen"].items())


def _fake_report(scores):
    """fit_sample_evaluate stub: the scorecard of each generator from `scores` {name: (passed, total, privacy fails)}."""
    def fse(spec, real, stats):
        name = next(iter({T.get("generator") for T in spec["tables"].values()} - {None}))
        passed, total, priv = scores[name]
        items = [{"metric": "dcr_ratio", "pass": False}] * priv + [{"metric": "marginal_ks", "pass": True}]
        return {"scorecard": {"passed": passed, "total": total, "items": items}}
    return fse


def test_trial_ranks_privacy_first_then_fidelity(shop):
    s = load_spec(_sel(shop, mode="trial", candidates=["copula", "independent", "cart"]))
    sel = S.trial(s, None, fit_sample_evaluate=_fake_report({"copula": (80, 100, 0), "independent": (60, 100, 0),
                                                              "cart": (95, 100, 1)}))
    assert set(sel.chosen.values()) == {"copula"}                       # cart scores higher but leaks
    assert [c["generator"] for c in sel.candidates] == ["copula", "independent", "cart"]
    sel = S.trial(s, None, fit_sample_evaluate=_fake_report({"copula": (80, 100, 0), "independent": (60, 100, 0),
                                                              "cart": (95, 100, 0)}))
    assert set(sel.chosen.values()) == {"cart"}


def test_trial_ties_go_to_the_specs_own_generator(shop):
    s = load_spec(_sel(shop, mode="trial", candidates=["cart", "copula"]))
    sel = S.trial(s, None, fit_sample_evaluate=_fake_report({"copula": (90, 100, 0), "cart": (90, 100, 0)}))
    assert set(sel.chosen.values()) == {"copula"}


def test_trial_scores_each_seed_on_its_own_stream(shop):
    s = load_spec(_sel(shop, mode="trial", seeds=3, candidates=["copula"]))
    seen = []

    def fse(spec, real, stats):
        seen.append(spec["synthesis"]["seed"])
        return {"scorecard": {"passed": 1, "total": 1, "items": []}}
    S.trial(s, None, fit_sample_evaluate=fse)
    base = s["synthesis"]["seed"]
    assert seen == [base, base + 1, base + 2]


def test_trial_records_failing_candidates_and_keeps_going(shop):
    s = load_spec(_sel(shop, mode="trial", candidates=["copula", "cart"]))

    def fse(spec, real, stats):
        if any(T.get("generator") == "cart" for T in spec["tables"].values()):
            raise RuntimeError("boom")
        return {"scorecard": {"passed": 1, "total": 1, "items": []}}
    sel = S.trial(s, None, fit_sample_evaluate=fse)
    assert set(sel.chosen.values()) == {"copula"}
    assert any(c["generator"] == "cart" and "boom" in c["error"] for c in sel.candidates)


def test_trial_end_to_end_is_paired_with_the_release(shop):
    """The chosen candidate's trial draw uses the release's seed, so its score is the release's score."""
    s = load_spec(_sel(shop, mode="trial", candidates=["copula", "independent"]))
    r = execute(s, write=False, privacy=True)            # the trial always scores with the privacy items
    sel = r.bundle["generator_selection"]
    best = [c for c in sel["candidates"] if c["generator"] == next(iter(sel["chosen"].values()))][0]
    sc = r.report["scorecard"]
    assert best["score"] == pytest.approx(sc["passed"] / sc["total"], abs=1e-6)
    assert {c["generator"] for c in sel["candidates"]} == {"copula", "independent"}


def test_auto_keeps_the_specs_generator_when_nothing_allowed_fits():
    name, why = S.auto_rule(_table(["float"]), "t", 5_000, [])
    assert name is None and "kept" in why
    spec = _table(["float"])
    spec["synthesis"] = {"generator": "cart"}
    assert S.auto_rule(spec, "t", 500, ["independent", "cart"])[0] == "cart"   # copula not allowed: the spec's


def test_trial_tries_a_candidate_on_the_tables_it_can_model(shop, monkeypatch):
    s = load_spec(_sel(shop, mode="trial", candidates=["copula", "cart"]))
    tables = S.selectable(s)
    assert len(tables) >= 2
    only = tables[0]
    real = S._compatible

    def compatible(spec, t, names, pushed):
        return [n for n in real(spec, t, names, pushed) if n != "cart" or t == only]
    monkeypatch.setattr(S, "_compatible", compatible)
    sel = S.trial(s, None, fit_sample_evaluate=_fake_report({"copula": (80, 100, 0), "cart": (95, 100, 0)}))
    assert sel.chosen == {only: "cart"}                                  # the others keep the spec's generator
    assert all("kept" in sel.reasons[t] for t in tables if t != only)
    assert next(c for c in sel.candidates if c["generator"] == "cart")["tables"] == [only]


def test_trial_gives_an_empty_scorecard_no_credit(shop):
    s = load_spec(_sel(shop, mode="trial", candidates=["copula", "independent"]))

    def fse(spec, real, stats):
        if any(T.get("generator") == "independent" for T in spec["tables"].values()):
            return {"scorecard": {"passed": None, "total": 0, "items": []}}
        return {"scorecard": {"passed": 5, "total": 10, "items": []}}
    sel = S.trial(s, None, fit_sample_evaluate=fse)
    assert set(sel.chosen.values()) == {"copula"}


def test_trial_scores_privacy_even_when_the_run_skips_it(shop, monkeypatch):
    import importlib
    E = importlib.import_module("relsynth.evaluate")       # the module (the package exports a function too)
    seen, real = [], E.evaluate

    def spy(*a, **kw):
        seen.append(kw.get("privacy"))
        return real(*a, **kw)
    monkeypatch.setattr(E, "evaluate", spy)
    s = load_spec(_sel(shop, mode="trial", candidates=["independent"]))
    spec, sel = S.apply(s, *_load(s))
    assert seen and all(p is True for p in seen) and sel.chosen


def _load(spec):
    from relsynth.pushdown import load_for_fit
    fd = load_for_fit(spec)
    return fd.tables, fd.stats


def test_every_fit_path_applies_the_selection(shop):
    import relsynth
    s = load_spec(_sel(shop, mode="auto"))
    m = relsynth.fit(s)
    tables = S.selectable(s)
    assert tables and all(m.spec["tables"][t].get("generator") for t in tables)

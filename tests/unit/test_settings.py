# SPDX-License-Identifier: Apache-2.0
"""Engine settings (relsynth.settings): registry, validation, settings files,
precedence, provenance, editable attributes, per-table overrides, thresholds
and the CLI.  Byte identity of the output with default settings is checked
across trees by tools/identity_check.py (CI job `identity`)."""
from __future__ import annotations

import copy
import json
import pathlib

import pytest
import yaml

import relsynth as rs
from relsynth import settings as S
from relsynth.cli import main
from relsynth.spec import DEFAULT_SYNTHESIS, SpecError, load_spec

SRC = pathlib.Path(rs.__file__).parent


# ---------------------------------------------------------------- registry
def test_registry_covers_every_default_with_its_value_and_a_doc():
    P = S.params()
    flat = S._flat(DEFAULT_SYNTHESIS, free=S._free_keys())
    assert set(P) == set(flat)
    for k, p in P.items():
        assert p.default == flat[k], k
        assert p.doc, f"{k} has no documentation (relsynth.schema.SYNTH_DOC)"
        assert p.check(p.default) is None, f"{k}: default {p.default!r} fails its own check: {p.check(p.default)}"


def test_promoted_constants_keep_the_1_0_values():
    # the constants moved from the code into the registry, with their 1.0 values
    expect = {"copula.mh_steps": 15, "copula.mcse_z": 0.01, "marginals.discrete_int_max": 30,
              "marginals.max_obs": 20000, "marginals.fd_sample_rows": 50000, "facts.frailty_min_rows": 30,
              "facts.occupancy_z": -4.0, "facts.occupancy_min_expected": 10.0, "facts.dp_count_sweeps": 30,
              "rules.rake_max_levels": 50, "rules.rake_max_cells": 2000, "repair.pair_tries": 200,
              "repair.constraint_rounds": 200, "repair.thin_rounds": 200, "repair.tree_redraws": 40,
              "repair.unique_code_rounds": 50, "repair.participation_rounds": 4, "repair.occupancy_rounds": 20,
              "detection.group_effect_share": 0.95, "detection.scd_stable_share": 0.99,
              "detection.scd_max_versions": 50, "evaluation.assoc_max_rows": 20000,
              "evaluation.discriminator_rows": 5000}
    P = S.params()
    for k, v in expect.items():
        assert P[k].default == v, k
    from relsynth.copula import MCSE_Z
    assert P["copula.mcse_z"].default == MCSE_Z


def test_every_setting_is_read_by_the_code():
    """A key that no module reads would be a setting that silently does nothing."""
    code = "".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
                   if p.name not in ("spec.py", "schema.py", "settings.py", "settings_cli.py"))
    reserved = {"gibbs.chains", "per_table", "settings_file"}    # documented as reserved / read by relsynth.settings
    unread = [k for k in S.params() if k not in reserved and f'"{k.split(".")[-1]}"' not in code]
    assert not unread, unread


def test_tunable_settings_have_a_search_range_and_affect_something():
    for k, p in S.search_space().items():
        assert p.affects, k
        if p.type in ("int", "float"):
            assert p.low is not None and p.high is not None, k
        if p.scale == "log":
            assert p.low > 0 or p.default > 0, k
        assert set(p.direction) <= set(p.affects), k


# -------------------------------------------------------------- validation
@pytest.mark.parametrize("syn,needle", [
    ({"copula": {"sample_rows": "many"}}, "must be a number"),
    ({"copula": {"sample_rows": 5}}, "must be >= 10"),
    ({"copula": {"sample_rows": 10.5}}, "whole number"),
    ({"copula": {"mode": "magic"}}, "must be one of rank, bayes, fast"),
    ({"gibbs": {"merge": "yes"}}, "true or false"),
    ({"structure": {"structural_prior": 1.5}}, "must be <= 1"),
    ({"evaluation": {"thresholds": {"kss": 0.1}}}, "did you mean 'ks'"),
    ({"evaluation": {"thresholds": {"ks": -1}}}, "must be a number >= 0"),
    ({"per_table": {"sales": {"gibbs": {"iterations": 3}}}}, "cannot be set per table"),
    ({"per_table": {"sales": {"copula": {"sample_row": 3}}}}, "did you mean 'sample_rows'"),
    ({"per_table": {"sales": {"copula": {"sample_rows": -3}}}}, "must be >= 10"),
    ({"trees": {"max_depth": "deep"}}, "or one of auto"),
])
def test_validation_errors(syn, needle):
    errs = S.validate(syn)
    assert errs and any(needle in e for e in errs), errs


def test_valid_values_pass():
    assert S.validate(copy.deepcopy(DEFAULT_SYNTHESIS)) == []
    assert S.validate({"trees": {"max_depth": 4}, "segments": {"stratify": {"sales": ["k"]}},
                       "copula": {"max_iterations": None}, "evaluation": {"thresholds": {"auc": 0.7}},
                       "per_table": {"sales": {"copula": {"sample_rows": 4000}}}}) == []


def test_spec_with_a_bad_setting_is_refused(star_csv):
    spec, _ = star_csv
    s = copy.deepcopy(spec)
    s["synthesis"] = {"copula": {"sample_rows": "lots"}}
    with pytest.raises(SpecError, match="copula.sample_rows: must be a number"):
        load_spec(s)
    s["synthesis"] = {"copula": {"sampl_rows": 3}}
    with pytest.raises(SpecError, match="did you mean 'sample_rows'"):
        load_spec(s)


# ------------------------------------------------------------ settings files
def _spec_file(tmp_path, spec, synthesis=None):
    s = copy.deepcopy(spec)
    if synthesis is not None:
        s["synthesis"] = synthesis
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(s), encoding="utf-8")
    return p


def test_precedence_defaults_file_spec_set(tmp_path, star_csv, monkeypatch):
    spec, _ = star_csv
    (tmp_path / "s.yaml").write_text(yaml.safe_dump({"synthesis": {"copula": {"sample_rows": 3000, "iterations": 90},
                                                                   "gibbs": {"iterations": 77}}}), encoding="utf-8")
    p = _spec_file(tmp_path, spec, {"settings_file": "s.yaml", "copula": {"sample_rows": 2000}})
    s = load_spec(str(p))["synthesis"]
    assert s["copula"]["sample_rows"] == 2000          # the spec wins over the file
    assert s["copula"]["iterations"] == 90              # the file wins over the default
    assert s["gibbs"]["iterations"] == 77
    assert s["copula"]["burn_in"] == DEFAULT_SYNTHESIS["copula"]["burn_in"]
    assert s["settings_file"] == str(tmp_path / "s.yaml")
    # --set wins over both (through the API's Spec.set, as the CLI applies it)
    sp = rs.Spec.load(p).set("synthesis.copula.sample_rows", "2500")
    assert sp.resolved["synthesis"]["copula"]["sample_rows"] == 2500
    # the environment variable names a file when the spec does not
    p2 = _spec_file(tmp_path, spec, {"gibbs": {"iterations": 5}})
    monkeypatch.setenv(S.ENV, str(tmp_path / "s.yaml"))
    s2 = load_spec(str(p2))["synthesis"]
    assert s2["gibbs"]["iterations"] == 5 and s2["copula"]["iterations"] == 90


def test_settings_file_relative_to_the_spec_through_the_api(tmp_path, star_csv, monkeypatch):
    spec, _ = star_csv
    sub = tmp_path / "proj"
    sub.mkdir()
    (sub / "s.yaml").write_text("copula: {sample_rows: 1234}\n", encoding="utf-8")
    p = _spec_file(sub, spec, {"settings_file": "s.yaml"})
    monkeypatch.chdir(tmp_path)                          # not the spec's folder
    assert rs.Spec.load(p).resolved["synthesis"]["copula"]["sample_rows"] == 1234


def test_settings_file_errors(tmp_path, star_csv):
    spec, _ = star_csv
    (tmp_path / "bad.yaml").write_text("copula: {sample_rows: x}\nnope: 1\n", encoding="utf-8")
    p = _spec_file(tmp_path, spec, {"settings_file": "bad.yaml"})
    with pytest.raises(SpecError) as e:
        load_spec(str(p))
    assert "must be a number" in str(e.value) and "nope" in str(e.value)
    p = _spec_file(tmp_path, spec, {"settings_file": "missing.yaml"})
    with pytest.raises(SpecError, match="not found"):
        load_spec(str(p))
    p = _spec_file(tmp_path, spec, {"settings_file": "preset:turbo"})
    with pytest.raises(SpecError, match="unknown settings preset"):
        load_spec(str(p))


@pytest.mark.parametrize("name", S.PRESETS)
def test_presets_are_valid(name):
    syn = S.read_file(f"preset:{name}")
    assert syn and S.validate(syn) == []


def test_provenance(tmp_path, star_csv):
    spec, _ = star_csv
    (tmp_path / "s.yaml").write_text("gibbs: {iterations: 77}\n", encoding="utf-8")
    p = _spec_file(tmp_path, spec, {"settings_file": "s.yaml", "copula": {"sample_rows": 2000}})
    from relsynth.spec import read_yaml
    prov = S.provenance(read_yaml(p), tmp_path)
    assert prov["copula.sample_rows"] == (2000, "spec")
    assert prov["gibbs.iterations"] == (77, "s.yaml")
    assert prov["copula.burn_in"] == (60, "default")


# ------------------------------------------------------ editable attributes
def test_settings_attributes_validate_and_export(tmp_path):
    s = rs.Settings()
    assert s.copula.sample_rows == 1500 and s.seed == 7
    s.copula.sample_rows = 5000
    s.seed = 3
    s.evaluation.thresholds.ks = 0.06
    s.table("sales").copula.sample_rows = 900
    with pytest.raises(rs.SettingsError, match="must be a number"):
        s.copula.sample_rows = "x"
    with pytest.raises(AttributeError, match="did you mean 'copula.sample_rows'"):
        s.copula.sampl_rows = 1
    with pytest.raises(rs.SettingsError, match="did you mean 'ks'"):
        s.evaluation.thresholds.kss = 0.1
    with pytest.raises(AttributeError, match="cannot be set per table"):
        s.table("sales").gibbs
    assert s.to_dict() == {"copula": {"sample_rows": 5000}, "seed": 3, "evaluation": {"thresholds": {"ks": 0.06}},
                           "per_table": {"sales": {"copula": {"sample_rows": 900}}}}
    m = s.merged()
    assert m["copula"]["sample_rows"] == 5000 and m["copula"]["burn_in"] == 60
    assert S.diff(m) == {"copula.sample_rows": 5000, "seed": 3, "evaluation.thresholds": {"ks": 0.06},
                         "per_table": {"sales": {"copula": {"sample_rows": 900}}}}
    p = s.save(tmp_path / "x.yaml")
    assert rs.Settings.from_file(p) == s
    assert s.set("gibbs.iterations", 50).gibbs.iterations == 50
    assert "copula" in dir(s) and "sample_rows" in dir(s.copula)


def test_settings_load_from_spec_and_apply(tmp_path, star_csv):
    spec, _ = star_csv
    p = _spec_file(tmp_path, spec, {"copula": {"sample_rows": 2000}})
    st = rs.Settings.load(p)
    assert st.copula.sample_rows == 2000
    st.gibbs.iterations = 33
    sp = rs.Spec.load(p).with_settings(st)
    r = sp.resolved["synthesis"]
    assert r["gibbs"]["iterations"] == 33 and r["copula"]["sample_rows"] == 2000
    assert rs.Spec.load(p).settings.copula.sample_rows == 2000


def test_table_config_merges_only_table_groups():
    cfg = copy.deepcopy(DEFAULT_SYNTHESIS)
    cfg["per_table"] = {"sales": {"copula": {"sample_rows": 7}}}
    tc = S.table_config(cfg, "sales")
    assert tc["copula"]["sample_rows"] == 7 and tc["copula"]["burn_in"] == 60
    assert cfg["copula"]["sample_rows"] == 1500          # not changed in place
    assert S.table_config(cfg, "customer") is cfg


def test_per_table_and_thresholds_reach_the_engine_and_the_scorecard(star_csv):
    spec, _ = star_csv
    s = copy.deepcopy(spec)
    s["synthesis"] = {"seed": 3, "gibbs": {"iterations": 10, "final_iterations": 5, "max_final_iterations": 5},
                      "copula": {"iterations": 30, "burn_in": 10, "max_iterations": 30},
                      "per_table": {"sales": {"copula": {"min_rows": 10 ** 8}}},
                      "evaluation": {"thresholds": {"ks": 0.5, "tvd": 0.5}}}
    model = rs.fit(s)
    # min_rows above the table size: no per-segment copula for sales, but the other tables keep theirs
    tm = model.synthesizer.tm
    assert tm["sales"].cfg["copula"]["min_rows"] == 10 ** 8
    assert tm["customer"].cfg["copula"]["min_rows"] == DEFAULT_SYNTHESIS["copula"]["min_rows"]
    rep = model.evaluate(model.sample(), privacy=False)
    lim = [i["limit"] for i in rep.scorecard["items"] if i["metric"] in ("marginal_ks", "marginal_tvd")]
    assert lim and min(lim) >= 0.5


def test_fit_accepts_settings(star_csv):
    spec, _ = star_csv
    st = rs.Settings()
    st.gibbs.iterations = 10
    st.gibbs.final_iterations = 5
    st.gibbs.max_final_iterations = 5
    st.copula.iterations = 30
    st.copula.burn_in = 10
    st.copula.max_iterations = 30
    m = rs.fit(spec, settings=st)
    assert m.spec["synthesis"]["gibbs"]["iterations"] == 10


# ------------------------------------------------------------------- the CLI
def test_cli_settings_commands(tmp_path, star_csv, capsys):
    spec, _ = star_csv
    p = _spec_file(tmp_path, spec, {"copula": {"sample_rows": 2000}})
    assert main(["settings", "show", str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["settings"] == {"copula.sample_rows": {"value": 2000, "source": "spec"}}
    assert main(["settings", "explain", "copula.sample_rows", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)["param"]
    assert d["tunable"] and d["default"] == 1500 and "association_max_diff" in d["affects"]
    assert main(["settings", "explain", "copula.sampl_rows"]) == 2
    capsys.readouterr()
    assert main(["settings", "template", "--out", str(tmp_path / "t.yaml")]) == 0
    syn = S.read_file(str(tmp_path / "t.yaml"))           # the template is itself a valid settings file
    assert S.Settings(syn).merged() == S.Settings().merged()
    assert main(["settings", "validate", str(tmp_path / "t.yaml")]) == 0
    (tmp_path / "bad.yaml").write_text("copula: {sample_rows: -1}\n", encoding="utf-8")
    assert main(["settings", "validate", str(tmp_path / "bad.yaml")]) == 2
    capsys.readouterr()
    assert main(["settings", "reference"]) == 0
    assert "| `copula.sample_rows` |" in capsys.readouterr().out


def test_cli_settings_option_on_run(tmp_path, star_csv, capsys):
    spec, _ = star_csv
    p = _spec_file(tmp_path, spec)
    (tmp_path / "bad.yaml").write_text("copula: {sample_rows: -1}\n", encoding="utf-8")
    assert main(["check", str(p), "--settings", str(tmp_path / "bad.yaml")]) == 2
    capsys.readouterr()
    assert main(["check", str(p), "--settings", "preset:fast"]) == 0


def test_settings_reference_is_up_to_date():
    ref = pathlib.Path(__file__).resolve().parents[2] / "docs" / "settings-reference.md"
    assert ref.read_text(encoding="utf-8") == S.reference(), \
        "docs/settings-reference.md is stale: python -m relsynth.settings > docs/settings-reference.md"

# SPDX-License-Identifier: Apache-2.0
"""Generator plug-ins (relsynth.generators): the contract, the registry, the
spec syntax, the built-in copula / independent / cart generators, a
third-party plug-in installed from its own package, and compute presets."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.cli import main as cli_main
from relsynth.compute import ComputeError, presets, resolve
from relsynth.compute import validate as validate_compute
from relsynth.generators import (Capabilities, Conditioning, GeneratorError, Registry, TableGenerator, check,
                                 default_registry, table_choice, version_matches)
from relsynth.spec import SpecError, load_spec

from conftest import write_scenario

ROOT = Path(__file__).resolve().parents[2]
INTEGRITY = {"fk_integrity", "rule_violation_rate", "unique_duplicate_rate", "tree_cycles"}


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    """S9_small_shop (customers, invoices, lines, products, reps; check rules) as CSV."""
    d = tmp_path_factory.mktemp("shop")
    spec = write_scenario("S9_small_shop", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


def _with(spec, **syn):
    s = copy.deepcopy(spec)
    s.setdefault("synthesis", {}).update(syn)
    return s


# ---------------------------------------------------------------- registry
def test_builtins_are_registered_with_capabilities():
    reg = Registry(entry_points=False)
    assert reg.names()[:3] == ["copula", "independent", "cart"]
    info = {g["name"]: g for g in reg.list()}
    assert info["copula"]["capabilities"]["conditioning"] == ["group", "parent"]
    assert info["independent"]["capabilities"]["conditioning"] == ["group"]
    assert all(g["capabilities"]["gpu"] == "never" for g in info.values())
    with pytest.raises(GeneratorError, match="did you mean 'copula'"):
        reg.entry("copulaa")


@pytest.mark.parametrize("v,spec,ok", [
    ("1.0", None, True), ("1.0", ">=1.0,<2", True), ("2.0", ">=1.0,<2", False), ("1.2.3", "==1.2.3", True),
    ("1.2", "!=1.2", False), ("1.4", "~=1.2", True), ("2.0", "~=1.2", False), ("1.2.7", "==1.2.*", True),
    ("1.3", "==1.2.*", False), ("0.9", "<1", True), ("1.0", "*", True)])
def test_version_specifiers(v, spec, ok):
    assert version_matches(v, spec) is ok


def test_bad_version_specifier():
    with pytest.raises(GeneratorError, match="not understood"):
        version_matches("1.0", "latest")


def test_capabilities_are_validated():
    with pytest.raises(GeneratorError, match="column kinds"):
        Capabilities(column_kinds=("integer", "image"))
    with pytest.raises(GeneratorError, match="group"):
        Capabilities(conditioning=("parent",))
    with pytest.raises(GeneratorError, match="gpu"):
        Capabilities(gpu="sometimes")


def test_registration_checks_the_class():
    reg = Registry(entry_points=False)

    class NoSample(TableGenerator):
        name = "nosample"

        def fit(self, data, context, privacy_budget=None):
            return self
    with pytest.raises(GeneratorError, match="sample"):
        reg.register(NoSample)

    class OtherContract(NoSample):
        name = "other"
        contract = "2.0"

        def sample(self, n, conditioning, rng):
            return pd.DataFrame()
    with pytest.raises(GeneratorError, match="contract"):
        reg.register(OtherContract)


# ---------------------------------------------------------------- spec syntax
def test_spec_generator_syntax(shop):
    s = copy.deepcopy(shop)
    s["tables"]["customer"]["generator"] = "cart"
    s["tables"]["invoice"]["generator"] = {"name": "independent", "version": ">=1.0,<2"}
    s["tables"]["invoice_line"]["generator"] = {"name": "cart", "options": {"min_leaf": 10}}
    spec = load_spec(s)
    assert table_choice(spec, "customer") == {"name": "cart", "version": None, "options": {}}
    assert table_choice(spec, "invoice")["name"] == "independent"
    assert table_choice(spec, "invoice_line")["options"] == {"min_leaf": 10}
    assert table_choice(spec, "track")["name"] == "copula"            # the default
    assert table_choice(load_spec(_with(shop, generator="independent")), "track")["name"] == "independent"


@pytest.mark.parametrize("gen,msg", [
    ("nope", "unknown generator 'nope'"),
    ({"name": "cart", "version": ">=2"}, "pins >=2"),
    ({"name": "cart", "options": [1]}, "options must be a mapping"),
    ({"name": "cart", "bogus": 1}, "unknown key 'bogus'"),
    (3, "generator must be a name or a mapping")])
def test_spec_generator_errors(shop, gen, msg):
    s = copy.deepcopy(shop)
    s["tables"]["customer"]["generator"] = gen
    with pytest.raises(SpecError, match=msg):
        load_spec(s)


def test_copied_tables_take_no_generator(shop):
    s = copy.deepcopy(shop)
    s["tables"]["genre"]["mode"] = "copy"
    s["tables"]["genre"]["generator"] = "cart"
    with pytest.raises(SpecError, match="not generated"):
        load_spec(s)


def test_capability_check_refuses_unsupported_columns(shop):
    class NumbersOnly(TableGenerator):
        name = "numbers"
        capabilities = Capabilities(column_kinds=("integer", "float"))
    spec = load_spec(shop)
    errs, _ = check(NumbersOnly, spec, "customer")
    assert errs and "cannot model columns of type" in errs[0]

    class NoPushdown(TableGenerator):
        name = "nopush"
        capabilities = Capabilities(pushdown=False, posterior=False)
    errs, warns = check(NoPushdown, _with(spec, posterior={"datasets": 3}), "invoice_line", aux=True, pushed=True)
    assert "pushdown.mode: off" in errs[0]
    assert any("parent attributes" in w for w in warns) and any("no posterior" in w for w in warns)
    errs, _ = check(NumbersOnly, spec, "track", ir_version="3.0")
    assert any("supports IR" in e for e in errs)


# ---------------------------------------------------------------- built-in generators
def _run(spec, seed=3, dataset=0):
    model = rs.fit(spec, seed=seed)
    return model, model.sample(dataset=dataset)


def test_copula_is_the_default_and_naming_it_changes_nothing(shop):
    _, a = _run(shop)
    s = _with(shop, generator="copula")
    s["tables"]["invoice"]["generator"] = {"name": "copula", "version": "==1.0"}
    _, b = _run(s)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])


@pytest.mark.parametrize("gen", ["independent", "cart"])
def test_reference_generators_keep_keys_links_and_rules(shop, gen):
    model, d = _run(_with(shop, generator=gen))
    rep = model.evaluate(d, privacy=False)
    items = rep.scorecard["items"]
    bad = [i for i in items if i["metric"] in INTEGRITY and not i["pass"]]
    assert not bad and sum(i["metric"] in INTEGRITY for i in items) >= 8
    for t, T in model.spec["tables"].items():
        pk = T.get("primary_key")
        if pk:
            assert d[t][pk].is_unique and d[t][pk].notna().all()
    for r in model.spec["relationships"]:
        ck = d[r["child"]][r["child_key"]].dropna()
        assert ck.isin(d[r["parent"]][r["parent_key"]]).all(), r["name"]
    assert rep.score > 0.9
    assert model.summary()["generators"]["invoice"]["generator"] == gen
    assert model.log["decisions"]["generators"]["invoice"]["generator"] == gen


@pytest.mark.parametrize("gen", ["independent", "cart"])
def test_reference_generators_are_seeded_and_have_a_posterior(shop, gen):
    s = _with(shop, generator=gen)
    _, a = _run(s, seed=4)
    m, b = _run(s, seed=4)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])
    c = m.sample(dataset=1)
    assert not c["invoice_line"].equals(b["invoice_line"])


def test_generators_mix_per_table(shop):
    s = copy.deepcopy(shop)
    s["tables"]["customer"]["generator"] = "cart"
    s["tables"]["invoice_line"]["generator"] = "independent"
    model, d = _run(s)
    gens = {t: tm.generator.name for t, tm in model.synthesizer.tm.items() if tm.fitted}
    assert gens["customer"] == "cart" and gens["invoice_line"] == "independent" and gens["invoice"] == "copula"
    assert set(model.summary()["generators"]) == {"customer", "invoice_line"}


def test_cart_options(shop):
    s = copy.deepcopy(shop)
    s["tables"]["customer"]["generator"] = {"name": "cart", "options": {"min_leaf": 50, "smoothing": "density"}}
    model, _ = _run(s)
    g = model.synthesizer.tm["customer"].generator
    assert g.opts["min_leaf"] == 50
    assert min(min(t.leaf_rows.values()) for t in g.trees) >= 50
    s["tables"]["customer"]["generator"] = {"name": "cart", "options": {"min_leef": 5}}
    with pytest.raises(rs.EngineError, match="unknown cart options"):
        rs.fit(s)


@pytest.mark.parametrize("gen", ["copula", "independent", "cart"])
def test_serialize_and_load(shop, gen):
    model, _ = _run(_with(shop, generator=gen))
    tm = model.synthesizer.tm["invoice"]
    blob = tm.generator.serialize()
    head = json.loads(blob.split(b"\n", 1)[0])
    assert head["generator"] == gen and head["contract"] == "1.0" and len(head["sha256"]) == 64
    cls = type(tm.generator)
    with pytest.raises(GeneratorError, match="trusted"):
        cls.load(blob)
    g2 = cls.load(blob, trusted=True)
    groups = np.asarray(sorted(tm.generator.fitted_groups()) or [0])[np.arange(50) % max(len(tm.generator.fitted_groups()), 1)]
    a = tm.generator.sample(50, Conditioning(groups), np.random.default_rng(1))
    b = g2.sample(50, Conditioning(groups), np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)
    bad = blob[:-8] + b"AAAAAAAA"
    with pytest.raises(GeneratorError):
        cls.load(bad, trusted=True)
    other = [c for c in default_registry().list() if c["name"] != gen][0]["name"]
    with pytest.raises(GeneratorError, match="not"):
        default_registry().entry(other).cls.load(blob, trusted=True)


def test_posterior_draws_iterates(shop):
    model, _ = _run(_with(shop, generator="cart"))
    g = model.synthesizer.tm["customer"].generator
    rng = np.random.default_rng(0)
    w = [copy.deepcopy(x.weights) for x in g.posterior_draws(2, rng)]
    leaf = next(iter(w[0][0]))
    assert not np.allclose(w[0][0][leaf], w[1][0][leaf])
    assert g.privacy_cost().dp is False


# ---------------------------------------------------------------- third-party plug-ins
def _install_example(target: Path) -> Path:
    import shutil
    # built from a copy: a build writes build/ and *.egg-info next to the sources
    pkg = shutil.copytree(ROOT / "examples" / "plugins" / "relsynth-example-generator", target.parent / "pkg")
    # a normal install (isolated build), as a user would do it; needs the package index for setuptools
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(target),
                        str(pkg)], capture_output=True, text=True)
    if p.returncode:
        pytest.skip(f"cannot build the example plug-in here (no package index?): {p.stderr[-300:]}")
    return target


@pytest.mark.slow
def test_third_party_plugin_from_its_own_package(shop, tmp_path, monkeypatch):
    site = _install_example(tmp_path / "site")
    assert list(site.glob("relsynth_example_generator-*.dist-info"))
    monkeypatch.syspath_prepend(str(site))
    try:
        reg = default_registry(refresh=True)
        e = reg.entry("empirical", ">=0.1")
        assert e.source == "entry point" and e.dist == "relsynth-example-generator" and e.dist_version == "0.1.0"
        s = copy.deepcopy(shop)
        s["tables"]["customer"]["generator"] = {"name": "empirical", "version": ">=0.1"}
        model, d = _run(s)
        assert model.log["decisions"]["generators"]["customer"]["package"] == "relsynth-example-generator"
        assert d["invoice"]["customer_id"].isin(d["customer"]["customer_id"]).all()
        assert model.synthesizer.tm["customer"].generator.name == "empirical"
        # the CLI lists it with its package
        assert cli_main(["generators", "list"]) == 0
    finally:
        monkeypatch.undo()
        default_registry(refresh=True)


def test_broken_plugin_is_reported_not_fatal(tmp_path, monkeypatch):
    di = tmp_path / "relsynth_broken-0.1.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: relsynth-broken\nVersion: 0.1\n")
    (di / "entry_points.txt").write_text("[relsynth.generators]\nbroken = no_such_module_xyz:Gen\n"
                                        "copula = no_such_module_xyz:Gen\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = Registry()
    assert "copula" in reg.names() and reg.entries["copula"].source == "built-in"
    assert any("name is built in" in w for w in reg.warnings)
    with pytest.raises(GeneratorError, match="failed to load"):
        reg.entry("broken")


# ---------------------------------------------------------------- compute presets
def test_compute_presets_and_resolution(monkeypatch):
    assert {"cpu", "gpu-auto", "gpu-single", "gpu-multi"} <= set(presets())
    assert resolve(None).device == "cpu"                     # default: CPU, whatever the generator
    p = resolve({"preset": "gpu-auto"}, "never", gpus={"cuda": [0]})
    assert p.device == "cpu" and "CPU only" in p.notes[0]
    p = resolve({"preset": "gpu-auto"}, "optional", gpus={})
    assert p.device == "cpu" and "fallback: cpu" in p.notes[0]
    with pytest.raises(ComputeError, match="fallback: error"):
        resolve({"preset": "gpu-single"}, "optional", gpus={})
    with pytest.raises(ComputeError, match="needs a GPU"):
        resolve({"preset": "cpu"}, "required")
    with pytest.raises(ComputeError, match="cannot run on a CPU"):
        resolve({"preset": "gpu-auto"}, "required", gpus={})
    p = resolve({"preset": "gpu-multi"}, "optional", gpus={"rocm": [0, 1, 2]})
    assert p.device == "rocm" and p.device_ids == [0, 1, 2]
    p = resolve({"preset": "gpu-single", "overrides": {"devices": [1], "precision": "bf16", "batch_size": 512,
                                                       "memory_limit": "8GiB", "threads": 4}},
                "optional", gpus={"cuda": [0, 1]})
    assert (p.device, p.device_ids, p.precision, p.batch_size, p.memory_limit_bytes, p.threads) == \
        ("cuda", [1], "bf16", 512, 8 * 2**30, 4)
    with pytest.raises(ComputeError):
        resolve({"preset": "gpu-single", "overrides": {"devices": [3]}}, "optional", gpus={"cuda": [0, 1]})
    monkeypatch.setenv("RELSYNTH_GPUS", "cuda:2")
    assert resolve({"preset": "gpu-multi"}, "required").device_ids == [0, 1]


def test_compute_validation_and_user_file(tmp_path):
    assert validate_compute({"preset": "gpu-auto", "overrides": {"precision": "fp8"}})
    assert validate_compute({"preset": "nope"})
    assert validate_compute({"preset": "cpu", "overrides": {"memory_limit": "lots"}})
    assert validate_compute({"preset": "cpu", "overrides": {"gpus": -1}})
    assert validate_compute({"preset": "cpu", "overrides": {"colour": "red"}})
    f = tmp_path / "compute.yaml"
    f.write_text("presets:\n  lab-a100:\n    device: gpu\n    gpus: 2\n    precision: bf16\n")
    assert validate_compute({"preset": "lab-a100", "file": str(f)}) == []
    p = resolve({"preset": "lab-a100", "file": "compute.yaml"}, "optional", base_dir=tmp_path, gpus={"cuda": [0, 1]})
    assert p.device_ids == [0, 1] and p.precision == "bf16" and p.preset == "lab-a100"


def test_spec_compute_block(shop):
    with pytest.raises(SpecError, match="precision"):
        load_spec(_with(shop, compute={"preset": "gpu-auto", "overrides": {"precision": "fp8"}}))
    spec = load_spec(_with(shop, compute={"preset": "gpu-auto", "overrides": {"threads": 2}}))
    assert spec["synthesis"]["compute"]["overrides"] == {"threads": 2}


def test_cli_generators_and_compute(capsys):
    assert cli_main(["generators", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [g["name"] for g in doc["generators"]][:3] == ["copula", "independent", "cart"]
    assert cli_main(["generators", "show", "cart"]) == 0
    assert "conditioning: group, parent" in capsys.readouterr().out
    assert cli_main(["generators", "show", "nope"]) != 0
    assert cli_main(["compute", "--preset", "gpu-auto", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["plans"]["never"]["device"] == "cpu"


# ---------------------------------------------------------------- propositions 704-706
@pytest.fixture(scope="module")
def snow(tmp_path_factory):
    d = tmp_path_factory.mktemp("snow")
    spec = write_scenario("S3_snowflake", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


def _spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank())


def _group_draw(model, t, n=4000, seed=0):
    """Rows of one large fitted group of table t, drawn with the parent attributes of that group's real rows."""
    tm = model.synthesizer.tm[t]
    g = tm.generator
    groups = np.asarray(g._groups)
    vals, cnt = np.unique(groups[groups >= 0], return_counts=True)
    top = vals[cnt.argmax()]
    rows = np.flatnonzero(groups == top)
    rng = np.random.default_rng(seed)
    pick = rows[rng.integers(0, len(rows), n)]
    given = pd.DataFrame({c: tm.codecs[j].decode(tm.X[j][pick]) for j, c in enumerate(tm.cols) if c in tm.aux})
    out = tm.sample(np.full(n, top), rng, given)
    return tm, rows, given, out


def test_parent_conditioning_carries_within_group_dependence(snow):
    """Prop. 704: within one group, a fact column depends on a parent attribute only with parent conditioning."""
    res = {}
    for gen in ("cart", "independent"):
        model, _ = _run(_with(snow, generator=gen))
        tm, rows, given, out = _group_draw(model, "sales")
        aux = "__aux__product_id.price"
        real = _spearman(tm.X[tm.cols.index("amount")][rows], tm.X[tm.cols.index(aux)][rows])
        res[gen] = (real, _spearman(out["amount"], given[aux]))
    real, cart = res["cart"]
    assert abs(real) > 0.2 and abs(cart - real) < 0.1
    assert abs(res["independent"][1]) < 0.05


def test_independent_columns_are_independent_within_a_group(snow):
    """Prop. 705: the independent generator is the copula with C = I."""
    model, _ = _run(_with(snow, generator="independent"))
    tm = model.synthesizer.tm["customer"]
    g = tm.generator
    groups = np.asarray(g._groups)
    top = np.bincount(groups).argmax()
    out = tm.sample(np.full(20000, top), np.random.default_rng(1))
    num = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    assert len(num) >= 2
    assert abs(_spearman(out[num[0]], out[num[1]])) < 0.03


def test_cart_leaf_weights_are_a_bayesian_bootstrap(snow):
    """Prop. 706: Dirichlet(1) leaf weights average to the empirical leaf law; mean mode uses it exactly."""
    model, _ = _run(_with(snow, generator="cart"))
    g = model.synthesizer.tm["customer"].generator
    leaf, don = max(g.donors[0].items(), key=lambda kv: len(kv[1]))
    n = len(don)
    rng = np.random.default_rng(3)
    W = np.stack([g.posterior_draw(rng).weights[0][leaf] for _ in range(3000)])
    assert np.allclose(W.mean(0), 1.0 / n, atol=4 * np.sqrt((n - 1) / (n * n * (n + 1)) / 3000))
    v = don[0]
    c = int((don == v).sum()) if not np.isnan(v) else int(np.isnan(don).sum())
    mask = (don == v) if not np.isnan(v) else np.isnan(don)
    Wv = W[:, mask].sum(1)
    assert Wv.var() == pytest.approx(c * (n - c) / (n * n * (n + 1)), rel=0.15) if c < n else True
    assert np.allclose(g.posterior_draw(rng, mode="mean").weights[0][leaf], 1.0 / n)

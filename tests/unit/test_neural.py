# SPDX-License-Identifier: Apache-2.0
"""Neural generator plug-ins (relsynth.generators.neural): the tvae / ddpm generators
through the engine (torch; CI job `neural`).  Props. 1800-1802, 1804-1805.  The parts
that need no torch are in test_neural_accountant.py."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.errors import EngineError
from relsynth.generators import GeneratorError, default_registry
from relsynth.generators.neural import accountant as acc

from conftest import write_scenario

INTEGRITY = {"fk_integrity", "rule_violation_rate", "unique_duplicate_rate", "tree_cycles"}


# ---------------------------------------------------------------- generators (torch)
torch = pytest.importorskip("torch")

FAST = {"tvae": {"epochs": 15, "batch_size": 256}, "ddpm": {"max_steps": 300, "sample_steps": 20, "batch_size": 256}}


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    spec = write_scenario("S9_small_shop", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


def _with(spec, gen, **opts):
    s = copy.deepcopy(spec)
    s.setdefault("synthesis", {})["generator"] = gen
    o = {**FAST[gen], **opts}
    for T in s["tables"].values():
        if T.get("mode") != "copy" and T.get("role") != "view":
            T["generator"] = {"name": gen, "options": o}
    return s


@pytest.mark.parametrize("gen", ["tvae", "ddpm"])
def test_neural_generators_keep_keys_links_and_rules(shop, gen):
    model = rs.fit(_with(shop, gen), seed=3)
    d = model.sample(dataset=0)
    rep = model.evaluate(d, privacy=False)
    bad = [i for i in rep.scorecard["items"] if i["metric"] in INTEGRITY and not i["pass"]]
    assert not bad
    for r in model.spec["relationships"]:
        ck = d[r["child"]][r["child_key"]].dropna()
        assert ck.isin(d[r["parent"]][r["parent_key"]]).all(), r["name"]
    assert model.summary()["generators"]["invoice"]["generator"] == gen


@pytest.mark.parametrize("gen", ["tvae", "ddpm"])
def test_neural_generators_are_seeded_and_have_a_marginal_posterior(shop, gen):
    """Prop. 701 on the CPU and Prop. 1800: dataset 1 redraws the decoding marginals."""
    s = _with(shop, gen)
    a = rs.fit(s, seed=4).sample(dataset=0)
    m = rs.fit(s, seed=4)
    b = m.sample(dataset=0)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])
    c = m.sample(dataset=1)
    assert not c["invoice_line"].equals(b["invoice_line"])


def test_torch_global_random_state_is_left_alone(shop):
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    rs.fit(_with(shop, "tvae"), seed=1).sample(dataset=0)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("gen", ["tvae", "ddpm"])
def test_serialize_and_load(shop, gen):
    model = rs.fit(_with(shop, gen), seed=2)
    tm = model.synthesizer.tm["invoice"]
    g = tm.generator
    blob = g.serialize()
    cls = default_registry().entry(gen).cls
    with pytest.raises(GeneratorError, match="trusted"):
        cls.load(blob)
    h = cls.load(blob, trusted=True)
    groups = np.asarray(g._groups)[:200]
    from relsynth.generators import Conditioning
    a = g.sample(len(groups), Conditioning(groups), np.random.default_rng(5))
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    b = h.sample(len(groups), Conditioning(groups), np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)
    # rebuilding the network of a loaded model leaves torch's global state alone
    assert torch.equal(before, torch.random.get_rng_state())
    # the model file holds no source rows: no encoded column array and no per-row segment labels
    assert h.data.X is None and "_groups" not in h.__dict__ and getattr(h.data, "generator", None) is None
    assert g.data.X is not None and tm.generator is g          # the live model is unchanged
    # a loaded model cannot redraw its marginals (no source rows): it keeps the saved ones, and the
    # engine's per-dataset redraw then samples it as saved
    dec = h.dec_m
    assert h.posterior_draw(np.random.default_rng(0)) is h
    if h.data.cfg["marginals"]["bayesian_bootstrap"]:
        assert h.dec_m is dec and h.uncertainty()["posterior_frozen"]
    tm.generator = h
    try:
        out = model.sample(dataset=1)
        assert len(out["invoice"]) > 0
    finally:
        tm.generator = g


@pytest.mark.parametrize("gen", ["tvae", "ddpm"])
def test_rare_columns_stay_out_of_the_network(shop, gen):
    """Columns whose non-modal values occur in fewer than min_dependence_rows rows are drawn
    independently from their marginals, as in the copula, and not learned jointly."""
    spec = _with(shop, gen)
    spec.setdefault("synthesis", {}).setdefault("marginals", {})["min_dependence_rows"] = 10 ** 6
    model = rs.fit(spec, seed=1)
    g = model.synthesizer.tm["customer"].generator
    own = [g.data.cols[j] for j in g.own]
    assert g.independent and set(g.independent) <= set(own)
    assert all(j not in g.space.cols for j in g.indep)
    out = model.sample(dataset=0)
    assert len(out["customer"]) > 0


def test_dp_fits_keep_every_column_in_the_network(shop):
    """The rare-column selection depends on the rows and would change the published weight shapes
    between neighbouring tables, so DP-SGD fits learn every column (no data-dependent mask)."""
    spec = _with(shop, "tvae", dp={"epsilon": 4.0}, epochs=2)
    spec.setdefault("synthesis", {}).setdefault("marginals", {})["min_dependence_rows"] = 10 ** 6
    g = rs.fit(spec, seed=1).synthesizer.tm["customer"].generator
    assert g.independent == [] and g.indep == []
    assert g.space.dim > 0


def test_unseen_group_backs_off_through_the_null_token(shop):
    """Prop. 1801: a group the fit never saw is sampled (null group token) without error."""
    model = rs.fit(_with(shop, "ddpm"), seed=2)
    tm = model.synthesizer.tm["customer"]
    out = tm.sample(np.full(50, 10 ** 6), np.random.default_rng(0))
    assert len(out) == 50 and list(out.columns) == tm.out_cols


def test_options_are_validated(shop):
    for bad, msg in [({"epochs": 0}, "positive integer"), ({"nope": 1}, "unknown tvae options"),
                     ({"dp": {"epsilon": -1}}, "epsilon > 0"), ({"dp": {"eps": 1}}, "unknown dp options"),
                     ({"hidden": []}, "hidden"), ({"max_steps": True}, "max_steps"),
                     ({"loss_factor": "2"}, "loss_factor")]:
        with pytest.raises((GeneratorError, EngineError), match=msg):   # wrapped by the API
            rs.fit(_with(shop, "tvae", **bad), seed=1)
    with pytest.raises((GeneratorError, EngineError), match="sample_steps"):
        rs.fit(_with(shop, "ddpm", timesteps=10, sample_steps=20), seed=1)
    # null means the generator's default, not a crash later in training
    model = rs.fit(_with(shop, "tvae", patience=None, lr=None), seed=1)
    assert model.synthesizer.tm["customer"].generator.opts["patience"] == 30


@pytest.mark.parametrize("gen", ["tvae", "ddpm"])
def test_dp_sgd_spends_at_most_the_target(shop, gen):
    """Prop. 1802: the accountant's epsilon after all steps is at most the requested one; the report says
    the guarantee covers the weights only (dp=False end to end)."""
    model = rs.fit(_with(shop, gen, dp={"epsilon": 4.0, "clip": 1.0}, epochs=3, max_steps=40), seed=1)
    g = model.synthesizer.tm["customer"].generator
    cost = g.privacy_cost()
    assert cost.dp is False and cost.mechanism.startswith("DP-SGD")
    assert 0 < cost.epsilon <= 4.0 and 0 < cost.delta <= 1e-5
    r = g.report.dp
    assert r["noise_multiplier"] > 0 and r["steps"] >= 1
    assert acc.epsilon(r["sampling_rate"], r["noise_multiplier"], r["steps"], r["delta"])[0] == \
        pytest.approx(cost.epsilon)
    assert "end-to-end" in cost.note


@pytest.mark.parametrize("gen", ["tvae", "ddpm"])
def test_calibration_matches_the_encoding_marginals(shop, gen):
    """Prop. 1805: under the training contexts the calibrated scores are N(0, 1) and the category slots
    have the encoding marginals' frequencies (up to sampling error)."""
    model = rs.fit(_with(shop, gen), seed=6)
    g = model.synthesizer.tm["invoice_line"].generator
    assert g.num_maps is not None or not g.space.n_num
    groups = np.asarray(g._groups)
    groups = groups[groups >= 0]
    assert len(groups)
    rng = np.random.default_rng(0)
    tm = model.synthesizer.tm["invoice_line"]
    rows = rng.choice(np.flatnonzero(np.asarray(g._groups) >= 0), 6000)
    given = pd.DataFrame({c: tm.codecs[j].decode(tm.X[j][rows]) for j, c in enumerate(tm.cols) if c in tm.aux}) \
        if tm.aux else None
    from relsynth.generators import Conditioning
    torch_ = pytest.importorskip("torch")
    gg, cc, A = g._cond_inputs(len(rows), Conditioning(np.asarray(g._groups)[rows], given), rng)
    with torch_.no_grad():
        gen_ = torch_.Generator().manual_seed(1)
        Z, S = g._generate(len(rows), {"g": torch_.as_tensor(gg), "c": torch_.as_tensor(cc),
                                        "a": torch_.as_tensor(A)}, gen_)
    if g.space.n_num:
        Zc = g._apply_num_maps(Z.numpy())
        assert np.all(np.abs(Zc.mean(0)) < 0.1) and np.all(np.abs(Zc.std(0) - 1) < 0.1)
    if g.space.cats:
        target = g._slot_target(g.enc_m)
        S = S.numpy()
        for i, K in enumerate(g.space.cat_widths):
            o = int(g.space.cat_offsets[i])
            freq = np.bincount(S[:, i], minlength=K)[:K] / len(S)
            assert 0.5 * np.abs(freq - target[o:o + K]).sum() < 0.06

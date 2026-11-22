# SPDX-License-Identifier: Apache-2.0
"""The ctvae family generator (contract 1.1): family grouping and order, engine integration
(keys, links, rules, seeds, persistence), within-family dependence, entity-level DP-SGD and
option checks.  Props. 1860-1863.  Torch tests run in the CI job `neural`."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.generators import CONTRACT_VERSION, Conditioning, GeneratorError, default_registry
from relsynth.generators.neural.ctvae import _families, _gaps

from conftest import write_scenario

INTEGRITY = {"fk_integrity", "rule_violation_rate", "unique_duplicate_rate", "tree_cycles"}


# ---------------------------------------------------------------- no torch needed
def test_contract_declares_family_conditioning():
    assert CONTRACT_VERSION == "1.1"
    c = Conditioning(np.zeros(3, np.int64))
    assert c.family is None and c.order is None
    cls = default_registry().entry("ctvae").cls
    assert "family" in cls.capabilities.conditioning and not cls.capabilities.pushdown


def test_families_group_rows_by_anchor_and_sort_by_order():
    fam = np.array([5, 2, 5, -1, 2, 5, -1])
    order = np.array([30.0, 1.0, 10.0, 0.0, np.nan, 20.0, 0.0])
    fams = _families(fam, order, len(fam))
    as_sets = sorted(tuple(f) for f in fams)
    # anchor 2: row 1 (t=1) before row 4 (unknown time last); anchor 5: rows by time 10, 20, 30;
    # rows without an anchor are families of one row
    assert (1, 4) in as_sets and (2, 5, 0) in as_sets and (3,) in as_sets and (6,) in as_sets
    assert sum(len(f) for f in fams) == len(fam)
    assert [len(f) for f in _families(None, None, 4)] == [1, 1, 1, 1]


def test_gaps_are_log_days_since_the_previous_row():
    order = np.array([0.0, 86400.0, 3 * 86400.0, np.nan])
    g = _gaps(order, np.arange(4))
    assert g[0] == 0 and np.isclose(g[1], np.log1p(1)) and np.isclose(g[2], np.log1p(2)) and g[3] == 0


# ---------------------------------------------------------------- generator (torch)
torch = pytest.importorskip("torch")

FAST = {"vae_steps": 80, "prior_steps": 80, "batch_families": 32, "calib_families": 200}


def _with(spec, tables=None, **opts):
    s = copy.deepcopy(spec)
    for t, T in s["tables"].items():
        if (tables is None and (T.get("fact") or T.get("role") == "fact")) or (tables and t in tables):
            T["generator"] = {"name": "ctvae", "options": {**FAST, **opts}}
    return s


@pytest.fixture(scope="module")
def enterprise(tmp_path_factory):
    d = tmp_path_factory.mktemp("ent")
    spec = write_scenario("S7_enterprise", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec


@pytest.fixture(scope="module")
def families_csv(tmp_path_factory):
    """Accounts and their transactions with strong within-family dependence: an account's
    transactions share one kind 90% of the time and their amounts share an account level."""
    d = tmp_path_factory.mktemp("fam")
    rng = np.random.default_rng(0)
    n = 600
    acc = pd.DataFrame({"account_id": np.arange(1, n + 1), "tier": rng.choice(["a", "b", "c"], n)})
    rows = []
    for a in acc.account_id:
        k = int(rng.integers(3, 9))
        main, level = rng.choice(["w", "x", "y", "z"]), rng.normal(0, 1)
        for i in range(k):
            kind = main if rng.random() < 0.9 else rng.choice(["w", "x", "y", "z"])
            rows.append((len(rows) + 1, a, kind, round(float(np.exp(3 + level + 0.3 * rng.normal())), 2)))
    tx = pd.DataFrame(rows, columns=["tx_id", "account_id", "kind", "amount"])
    (d / "src").mkdir()
    acc.to_csv(d / "src" / "account.csv", index=False)
    tx.to_csv(d / "src" / "tx.csv", index=False)
    spec = {"name": "families", "source": {"kind": "csv", "path": str(d / "src")},
            "target": {"kind": "csv", "path": str(d / "out")},
            "tables": {"account": {"primary_key": "account_id",
                                   "columns": {"account_id": "id", "tier": "categorical"}},
                       "tx": {"role": "fact", "primary_key": "tx_id", "fact": {"anchor": "account"},
                              "columns": {"tx_id": "id", "account_id": "id", "kind": "categorical",
                                          "amount": {"type": "float", "min": 0}}}},
            "relationships": [{"child": "tx", "child_key": "account_id", "parent": "account"}]}
    return spec


def _share(df, fam, col):
    """P(two distinct siblings share a level), over families of 2+ rows."""
    num = den = 0
    for _, g in df.groupby(fam):
        c = g[col].value_counts().to_numpy()
        k = c.sum()
        num += (c * (c - 1)).sum()
        den += k * (k - 1)
    return num / max(den, 1)


def test_keys_links_and_rules_hold(enterprise):
    model = rs.fit(_with(enterprise), seed=3)
    d = model.sample(dataset=0)
    rep = model.evaluate(d, privacy=False)
    assert not [i for i in rep.scorecard["items"] if i["metric"] in INTEGRITY and not i["pass"]]
    for r in model.spec["relationships"]:
        ck = d[r["child"]][r["child_key"]].dropna()
        assert ck.isin(d[r["parent"]][r["parent_key"]]).all(), r["name"]
    u = model.summary()["generators"]["sales"]["uncertainty"]
    assert u["ordered"] and u["families_fitted"] > 100


def test_seeded_and_marginal_posterior(enterprise):
    s = _with(enterprise, tables={"sales"})
    a = rs.fit(s, seed=4).sample(dataset=0)
    m = rs.fit(s, seed=4)
    b = m.sample(dataset=0)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])
    assert not m.sample(dataset=1)["sales"].equals(b["sales"])


def test_family_dependence_is_kept(families_csv):
    """Prop. 1861: the family prior carries sibling dependence that a row-wise generator drops."""
    real = pd.read_csv(families_csv["source"]["path"] + "/tx.csv")
    r = _share(real, "account_id", "kind")
    fam = rs.fit(_with(families_csv, vae_steps=600, prior_steps=600), seed=1).sample(dataset=0)["tx"]
    s = copy.deepcopy(families_csv)
    s["tables"]["tx"]["generator"] = {"name": "tvae", "options": {"epochs": 30, "batch_size": 256}}
    row = rs.fit(s, seed=1).sample(dataset=0)["tx"]
    sf, sr = _share(fam, "account_id", "kind"), _share(row, "account_id", "kind")
    assert r > 0.75
    assert sf > sr + 0.15, (r, sf, sr)


def test_serialize_and_load(enterprise, tmp_path):
    m = rs.fit(_with(enterprise, tables={"sales"}), seed=2)
    gen = m.synthesizer.tm["sales"].generator
    blob = gen.serialize()
    back = type(gen).load(blob, trusted=True)
    assert back.name == "ctvae" and back.weights.keys() == gen.weights.keys()
    assert not hasattr(back, "tensors") and getattr(back.data, "X", None) is None
    groups = np.array(sorted(back.group_ix))[:1].repeat(6)
    out = back.sample(6, Conditioning(groups, family=np.array([0, 0, 0, 1, 1, 2]),
                                      order=np.arange(6) * 86400.0), np.random.default_rng(0))
    assert len(out) == 6


def test_entity_level_dp_sgd(enterprise):
    m = rs.fit(_with(enterprise, tables={"sales"}, vae_steps=10, prior_steps=10, dp={"epsilon": 2.0}), seed=5)
    dp = m.summary()["generators"]["sales"]["dp_sgd"]
    assert dp["unit"] == "family" and dp["epsilon"] <= 2.0 + 1e-9 and dp["steps"] == 20
    pc = m.synthesizer.tm["sales"].generator.privacy_cost()
    assert pc.dp is False and "family" in pc.note


@pytest.mark.parametrize("bad", [{"prior": "lstm"}, {"categorical_head": "x"}, {"window": 0},
                                 {"d_column": 30, "heads": 4}, {"nope": 1}])
def test_options_are_validated(enterprise, bad):
    with pytest.raises((GeneratorError, rs.errors.EngineError, ValueError)):
        rs.fit(_with(enterprise, tables={"sales"}, **bad), seed=1)


def test_non_fact_table_warns(enterprise):
    from relsynth.generators.registry import check
    from relsynth.spec import load_spec
    spec = load_spec(copy.deepcopy(enterprise))
    errs, warns = check(default_registry().entry("ctvae").cls, spec, "customer")
    assert any("families of one row" in w for w in warns)


def test_redraw_is_given_the_kept_siblings(families_csv):
    """Contract 1.1 redraw: rows drawn given kept siblings follow them (teacher forcing)."""
    m = rs.fit(_with(families_csv, vae_steps=400, prior_steps=400), seed=2)
    gen = m.synthesizer.tm["tx"].generator
    rng = np.random.default_rng(0)
    groups = np.array(sorted(gen.group_ix))[:1]
    k, n_fam = 6, 60
    # every family: 6 kept siblings all of one kind, 1 row to redraw
    kinds = np.array(["w", "x", "y", "z"])[np.arange(n_fam) % 4]
    sib = pd.DataFrame({"kind": np.repeat(kinds, k), "amount": 20.0,
                        "__family__": np.repeat(np.arange(n_fam), k), "__order__": np.nan,
                        "__group__": groups[0]})
    cond = Conditioning(np.repeat(groups, n_fam), family=np.arange(n_fam), siblings=sib)
    out = gen.sample(n_fam, cond, rng)
    match = (out["kind"].to_numpy() == kinds).mean()
    free = gen.sample(n_fam, Conditioning(np.repeat(groups, n_fam), family=np.arange(n_fam)), rng)
    assert match > 0.6 and match > (free["kind"].to_numpy() == kinds).mean() + 0.2, match

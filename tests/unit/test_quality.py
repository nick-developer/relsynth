# SPDX-License-Identifier: Apache-2.0
"""Evaluation framework 2.0 (relsynth.quality): metric values on inputs with a
known answer, the calibration helpers, the report schema and the carry-over of
the 1.0 scorecard.  The calibration of the limits themselves (false-failure
rates under the null, power against known defects) is tested by the slower
checks in tests/science/test_quality_calibration.py."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))

import quality_null as Q  # noqa: E402

from relsynth.data import coerce  # noqa: E402
from relsynth.quality import METRICS, assess, quality_vector  # noqa: E402
from relsynth.quality import calibrate as cal  # noqa: E402
from relsynth.quality import multi as mt  # noqa: E402
from relsynth.quality import single as st  # noqa: E402
from relsynth.quality.report import poisson_binomial_quantile  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402

SCHEMA = Path(__file__).resolve().parents[2] / "src/relsynth/quality/schema/quality-report-2.0.schema.json"


@pytest.fixture(scope="module")
def shop():
    spec = load_spec(Q.spec())
    co = (lambda d: {k: coerce(v, spec["tables"][k]) for k, v in d.items()})
    return spec, co(Q.shop(1, 0.5)), co(Q.shop(2, 0.5))


# ---------------------------------------------------------------- single-column metrics
def test_wasserstein_known_values():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 2, 4000))
    assert st.w1_raw(x, x) == 0 and st.w1_rank(x, x) == 0
    # a shift by one real sd is W1 = 1 sd on the raw scale
    assert st.w1_raw(x, x + x.std(ddof=0)) == pytest.approx(1.0, abs=1e-9)
    # monotone transformations leave the rank-normal distance unchanged and change the raw one
    assert st.w1_rank(x, x) == st.w1_rank(np.exp(x / 4), np.exp(x / 4)) == 0
    y = pd.Series(rng.normal(0.5, 2, 4000))
    assert st.w1_rank(x, y) == pytest.approx(st.w1_rank(np.exp(x / 4), np.exp(y / 4)), abs=1e-12)


def test_jensen_shannon_bounds():
    a = pd.Series(["x"] * 50 + ["y"] * 50)
    assert st.js_distance(a, a, True) == pytest.approx(0, abs=1e-12)
    assert st.js_distance(pd.Series(["x"] * 10), pd.Series(["y"] * 10), True) == pytest.approx(1.0)
    # null is a level of its own
    assert st.js_distance(a, pd.Series(["x"] * 50 + [None] * 50), True) > 0.5


def test_pit_curve_is_exact_with_ties():
    """With heavy ties the indicator curve is far from the diagonal; the
    randomised PIT curve of a sample against itself is the diagonal."""
    ref = np.repeat([1.0, 2.0, 3.0], [500, 300, 200])
    grid = np.linspace(0.05, 1, 20)
    assert np.allclose(st._pit_curve(ref, ref, grid), grid, atol=1e-12)
    naive = np.array([np.mean(ref <= np.quantile(ref, a)) for a in grid])
    assert np.abs(naive - grid).max() > 0.2


def test_precision_recall_of_identical_and_collapsed_samples(shop):
    spec, a, _ = shop
    cols = ["age", "income", "member"]
    same = st.precision_recall(a["customer"], a["customer"], spec, "customer", cols)
    assert same["alpha_precision"] > 0.98 and same["beta_recall"] > 0.9
    c = a["customer"]
    collapsed = c.assign(age=c.age.median(), income=c.income.median())      # mode collapse
    bad = st.precision_recall(a["customer"], collapsed, spec, "customer", cols)
    assert bad["beta_recall_gap"] > 0.3


def test_kway_tvd(shop):
    spec, a, _ = shop
    c = a["customer"]
    cols = ["age", "income", "member", "signup"]
    assert st.kway_tvd(c, c, spec, "customer", cols)["k2_max"] == 0
    shuffled = c.assign(member=np.random.default_rng(0).permutation(c.member.to_numpy()))
    out = st.kway_tvd(c, shuffled, spec, "customer", cols)
    assert "member" in out["k2_worst"] and out["k2_max"] > 0.05
    assert out["k3_count"] == 4


# ---------------------------------------------------------------- multi-table metrics
def test_fanout_and_detection_see_relinked_children(shop):
    spec, a, _ = shop
    assert mt.fanout_law(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))["w1_rel_mean"] == 0
    fr, kinds = mt.aggregated(spec, a, "customer")
    assert {"orders.customer_id->customer:count", "orders.customer_id->customer:amount:mean"} <= set(kinds)
    acc_same = mt.detection_accuracy(fr, fr, kinds)
    assert 0.4 < acc_same < 0.6
    hop = mt.hop_trends(spec, a, a)
    assert all(v["mean"] == pytest.approx(1.0) for v in hop.values())
    assert {v["hops"] for v in hop.values()} == {1, 2}


def test_cluster_ids_follow_the_anchor_chain(shop):
    spec, a, _ = shop
    assert [s[2] for s in cal.anchor_chain(spec, a, "lines")] == ["orders", "customer"]
    assert cal.anchor_chain(spec, a, "customer") == []          # region is a copied lookup
    ids = cal.cluster_ids(spec, a, "lines")
    o = a["orders"].set_index("order_id").customer_id
    assert (ids == o.loc[a["lines"].order_id].astype(str).to_numpy()).all()
    m = cal.cluster_halves(ids, np.random.default_rng(0))
    assert not set(ids[m]) & set(ids[~m])                        # no cluster on both sides


# ---------------------------------------------------------------- calibration helpers
def test_prediction_limit_and_level():
    rng = np.random.default_rng(1)
    draws = rng.normal(5, 1, 20)
    lim = cal.prediction_limit(draws, center=0, confidence=0.99, k=1.0)
    assert cal.exceed_prob(draws, lim, center=0, k=1.0) == pytest.approx(0.01, rel=1e-6)
    lim2 = cal.prediction_limit(draws, center=0, confidence=0.99, k=0.5, log=True)
    assert cal.exceed_prob(draws, lim2, center=0, k=0.5, log=True) == pytest.approx(0.01, rel=1e-6)
    assert cal.kappa((100, 100), (50, 50)) == pytest.approx(np.sqrt(0.5))


def test_poisson_binomial_quantile():
    assert poisson_binomial_quantile([], 0.99) == 0
    assert poisson_binomial_quantile([0.0] * 10, 0.99) == 0
    # 100 checks at 1%: P(F <= 3) = 0.981, P(F <= 4) = 0.9966
    assert poisson_binomial_quantile([0.01] * 100, 0.99) == 4


def test_quality_vector_rules():
    checks = [{"id": f"c{i}", "area": "fidelity", "pass": i != 0, "alpha": 0.01} for i in range(100)]
    checks += [{"id": "fk", "area": "integrity", "pass": False, "alpha": None}]
    v = quality_vector(checks)
    assert v["fidelity"]["pass"] and v["fidelity"]["allowed_failures"] == 4
    assert not v["integrity"]["pass"] and not v["verdict"]["pass"]
    # one failure far beyond its null law fails the area on its own
    checks[0]["p_value"] = 1e-6
    v = quality_vector(checks)
    assert v["fidelity"]["severe_failures"] == ["c0"] and not v["fidelity"]["pass"]
    checks[0]["p_value"] = 0.001                                   # above 0.01 / (2 * 100)
    assert quality_vector(checks)["fidelity"]["pass"]


# ---------------------------------------------------------------- the report
def test_report_validates_and_keeps_the_scorecard(shop):
    jsonschema = pytest.importorskip("jsonschema")
    spec, a, b = shop
    from relsynth.evaluate import evaluate
    base = evaluate(spec, a, b)
    before = copy.deepcopy(base["scorecard"])
    rep = assess(spec, a, b, base=base, config={"null_splits": 4, "multi_splits": 3})
    jsonschema.validate(json.loads(json.dumps(rep)), json.loads(SCHEMA.read_text()))
    assert base["scorecard"] == before                           # the 1.0 scorecard is not touched
    old = [c for c in rep["checks"] + rep["privacy_passthrough"] if c["source"] == "scorecard-1.0"]
    assert len(old) == before["total"] and sum(c["pass"] for c in old) == before["passed"]
    new = {c["metric"] for c in rep["checks"] if c["source"] == "quality-2.0"}
    assert new <= set(METRICS) and {"w1_raw_sd", "js_distance", "kway2_tvd_mean", "alpha_precision_gap",
                                    "tstr_gap", "trts_gap", "fanout_w1_rel", "hop_trend_gap", "dda_accuracy",
                                    "mt_tstr_gap"} <= new
    for c in rep["checks"]:
        if c["source"] == "quality-2.0":
            assert c["limit"] >= c["tolerance"]
            assert c["null"]["method"].startswith("real-vs-real") and c["null"]["splits"] >= 3
    sr = rep["benchmark"]["syntherela"]
    assert set(sr["cardinality_similarity"]) == {n for n, *_ in mt.relations(spec)}
    assert 0 <= sr["c2st_agg_accuracy_max"] <= 1


def test_config_validation(shop):
    spec, a, b = shop
    with pytest.raises(ValueError, match="unknown metric"):
        assess(spec, a, b, scorecard=False, config={"tolerances": {"nope": 1}})
    with pytest.raises(ValueError, match="classifier"):
        assess(spec, a, b, scorecard=False, config={"classifier": "svm"})
    with pytest.raises(ValueError, match="scale"):
        assess(spec, a, b, scorecard=False, config={"scale": {"default": -1}})


def test_resized_database_is_compared_per_expected_rate(shop):
    """Prop. 113: a database drawn at twice the size is compared on counts per
    unit of the expected rate; regions are copied, so customers per region double."""
    jsonschema = pytest.importorskip("jsonschema")
    from relsynth.quality.report import fanout_factor, resolve_config
    spec, a, _ = shop
    big = {k: coerce(v, spec["tables"][k]) for k, v in Q.shop(3, 1.0).items()}
    cfg = resolve_config({"scale": 2.0})
    assert fanout_factor(spec, cfg, "region", "customer") == 2.0
    assert fanout_factor(spec, cfg, "customer", "orders") == 1.0
    assert fanout_factor(spec, cfg, "product", "lines") == 1.0     # lines follow their anchor, orders
    ref = {"schema": "relsynth.privacy-report/1", "path": "privacy.json", "sha256": "0" * 64,
           "verdicts": {"internal": "pass"}}
    rep = assess(spec, a, big, scorecard=False, privacy=ref,
                 config={"scale": 2.0, "null_splits": 3, "multi_splits": 3})
    jsonschema.validate(json.loads(json.dumps(rep)), json.loads(SCHEMA.read_text()))
    assert rep["privacy"] == ref
    region = [c for c in rep["checks"] if c["scope"].get("relationship") == "customer.region_id->region"]
    assert [c["metric"] for c in region] == ["fanout_w1_rel"] and "scale factor 2" in region[0]["note"]
    assert region[0]["value"] < 0.2                     # unscaled, the counts would differ by a factor 2
    plain = assess(spec, a, big, scorecard=False, config={"null_splits": 3, "multi_splits": 3})
    raw = [c for c in plain["checks"] if c["scope"].get("relationship") == "customer.region_id->region"]
    assert len(raw) == 3 and raw[0]["value"] > 0.5


def test_uncalibrated_mode_uses_tolerances(shop):
    spec, a, b = shop
    rep = assess(spec, a, b, scorecard=False, config={"calibrate": False, "targets": {"customer": "member"}})
    new = [c for c in rep["checks"] if c["source"] == "quality-2.0"]
    assert new and all(c["limit"] == c["tolerance"] and c["null"]["method"] == "fixed tolerance" for c in new)
    assert any(c["scope"].get("target") == "member" for c in new if c["metric"] == "tstr_gap")


def test_column_kinds_are_decided_on_the_whole_real_table():
    """An integer column with more than 30 values is numeric even when a real
    half has 30 or fewer (AdventureWorks raised a KeyError before)."""
    spec = load_spec({"name": "k", "source": {"kind": "csv", "path": "."},
                      "tables": {"t": {"primary_key": "id", "columns": {
                          "id": "id", "n": {"type": "integer"}, "x": "float"}}},
                      "relationships": []})
    rng = np.random.default_rng(0)
    real = {"t": pd.DataFrame({"id": np.arange(48), "n": np.repeat(np.arange(32), [2] * 16 + [1] * 16),
                               "x": rng.normal(size=48)})}
    syn = {"t": real["t"].sample(frac=1, random_state=1).reset_index(drop=True)}
    real = {"t": coerce(real["t"], spec["tables"]["t"])}
    syn = {"t": coerce(syn["t"], spec["tables"]["t"])}
    rep = assess(spec, real, syn, scorecard=False, config={"null_splits": 6, "multi_splits": 3})
    n_checks = {c["metric"] for c in rep["checks"] if c["scope"].get("column") == "n"}
    assert {"w1_raw_sd", "w1_rank_normal", "js_distance"} <= n_checks


def test_fanout_of_a_sampled_fact_uses_exact_counts():
    """A pushed-down fact is held as a sample; per-parent counts come from its exact statistics."""
    import numpy as np
    import pandas as pd

    from relsynth.pushdown.stats import FrameStats
    from relsynth.quality import multi as mt

    rng = np.random.default_rng(0)
    gp = pd.DataFrame({"g": np.arange(20)})
    par = pd.DataFrame({"k": np.arange(500), "g": rng.integers(0, 20, 500)})
    child = pd.DataFrame({"fk": rng.choice(500, 20_000, p=rng.dirichlet(np.ones(500)))})
    sample = child.sample(frac=0.1, random_state=1)
    full = mt.counts_per_parent(par, "k", child, "fk")
    exact = FrameStats(child, table="c")
    assert np.array_equal(mt.counts_per_parent(par, "k", sample, "fk", exact), full)
    assert mt.counts_per_parent(par, "k", sample, "fk").sum() < 0.2 * full.sum()
    tabs = {"gp": gp, "p": par, "c": sample}
    g_full = mt.grandchild_counts({"gp": gp, "p": par, "c": child}, "gp", "g", "p", "k", "g", "c", "fk")
    assert np.array_equal(mt.grandchild_counts(tabs, "gp", "g", "p", "k", "g", "c", "fk", exact), g_full)


def test_a_one_row_parent_splits_rows_or_falls_back_to_the_fixed_tolerance(shop):
    """Class: relationships whose parent table has a single row, or whose child rows all hang
    from one parent row (one store, warehouse or call centre, as in small TPC-DS scale factors;
    the rates of a single currency). Real-vs-real splits need two clusters; with one, a half was
    empty and the report stopped (KeyError on the empty fan-out law). Now the rows are the split
    units (the hop trends of the lines under the one product), and a one-row table, which
    cannot be split at all, uses the fixed tolerance (the product fan-out)."""
    spec, a, b = shop
    one = []
    for side in (a, b):
        d = dict(side)
        d["product"] = side["product"].iloc[:1].reset_index(drop=True)
        d["lines"] = side["lines"].assign(product_id=d["product"]["product_id"].iloc[0])
        one.append(d)
    assert not cal.splittable(cal.cluster_ids(spec, one[0], "product"))
    rep = assess(spec, one[0], one[1], config={"null_splits": 4, "multi_splits": 3})
    fan = [c for c in rep["checks"] if c["metric"].startswith("fanout")
           and "product" in str(c["scope"].get("relationship", ""))]
    assert fan and all(c["null"]["method"] == "fixed tolerance" for c in fan)
    others = [c for c in rep["checks"] if c["metric"] == "fanout_w1_rel"
              and "product" not in str(c["scope"].get("relationship", ""))]
    assert others and all(c["null"]["method"].startswith("real-vs-real") for c in others)
    assert cal.splittable(np.array(["a", "b"])) and not cal.splittable(np.array(["a", "a"]))
    assert list(cal.split_units(np.array(["a", "a", "a"]))) == ["0", "1", "2"]
    assert list(cal.split_units(np.array(["a", "b", "a"]))) == ["a", "b", "a"]
    hops = [c for c in rep["checks"] if c["metric"].startswith("hop_trend")
            and str(c["scope"].get("path", "")).endswith("->product")]
    assert hops and all(c["null"]["method"].startswith("real-vs-real") for c in hops)
    # a column-pair trend with no rows on a side (a split half under a one-row root) has no score
    e, x = pd.Series([], dtype=object), pd.Series(["u", "v", "u"])
    assert mt._pair_trend(e, e, e, e, True, True) is None
    assert mt._pair_trend(e, x, e, x, True, True) is None
    assert mt._pair_trend(x, x, x, x, True, True) == 1.0

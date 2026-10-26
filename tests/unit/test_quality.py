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


def test_uncalibrated_mode_uses_tolerances(shop):
    spec, a, b = shop
    rep = assess(spec, a, b, scorecard=False, config={"calibrate": False, "targets": {"customer": "member"}})
    new = [c for c in rep["checks"] if c["source"] == "quality-2.0"]
    assert new and all(c["limit"] == c["tolerance"] and c["null"]["method"] == "fixed tolerance" for c in new)
    assert any(c["scope"].get("target") == "member" for c in new if c["metric"] == "tstr_gap")

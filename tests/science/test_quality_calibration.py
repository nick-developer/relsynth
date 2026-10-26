"""Checks C79-C82 of the 2.0 quality report (docs/proofs/sec_v20.tex): the limits
calibrated on real-vs-real splits keep their nominal level when the synthetic
data is an independent draw of the real data's law, also at other sizes, and
the checks see planted defects.  About 15 minutes on 4 cores.

    pytest -m science tests/science/test_quality_calibration.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))

import quality_null as Q  # noqa: E402

from relsynth.data import coerce  # noqa: E402
from relsynth.evaluate import evaluate  # noqa: E402
from relsynth.quality import METRICS, assess  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402

pytestmark = [pytest.mark.science, pytest.mark.slow]

SPEC = load_spec(Q.spec())
NO_TOLERANCE = {m: 0.0 for m, d in METRICS.items() if d["center"] == 0}
LEVEL = 0.01


def _db(seed, scale=1.0):
    return {k: coerce(v, SPEC["tables"][k]) for k, v in Q.shop(seed, scale).items()}


def _null_run(reps, scale):
    """False failures of the new checks (tolerances 0) and of the areas over
    `reps` pairs of independent draws; the second is drawn at `scale`."""
    fails, total, by_metric, area_fail = 0, 0, collections.Counter(), collections.Counter()
    for r in range(reps):
        rep = assess(SPEC, _db(5000 + 2 * r), _db(5001 + 2 * r, scale), scorecard=False,
                     config={"tolerances": NO_TOLERANCE, "seed": r, "scale": scale})
        for c in rep["checks"]:
            total += 1
            fails += not c["pass"]
            if not c["pass"]:
                by_metric[c["metric"]] += 1
        for a in ("fidelity", "utility"):
            area_fail[a] += not rep["vector"][a]["pass"]
    return fails, total, by_metric, area_fail


def _assert_level(fails, total, area_fail, reps):
    # the rate must not exceed the nominal level beyond binomial noise (one-sided 99.9%)
    assert fails <= binom.ppf(0.999, total, LEVEL), (fails, total)
    for a, k in area_fail.items():
        assert k <= binom.ppf(0.999, reps, LEVEL), (a, k, reps)


def test_c79_null_false_failure_rate():
    reps = 8
    fails, total, by_metric, area_fail = _null_run(reps, 1.0)
    print(f"C79: {fails}/{total} new checks failed under the null; by metric {dict(by_metric)}; areas {dict(area_fail)}")
    assert total > 600
    _assert_level(fails, total, area_fail, reps)


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_c80_null_rate_at_other_sizes(scale):
    reps = 6
    fails, total, by_metric, area_fail = _null_run(reps, scale)
    print(f"C80 (synthetic {scale}x): {fails}/{total}; by metric {dict(by_metric)}; areas {dict(area_fail)}")
    _assert_level(fails, total, area_fail, reps)


EXPECTED = {
    "income_shift": {"w1_raw_sd", "w1_rank_normal", "js_distance"},
    "age_shuffled": {"kway2_tvd_mean", "kway2_tvd_max", "kway3_tvd_mean"},
    "orders_relinked": {"hop_trend_gap", "hop_trend_worst_gap"},
    "fanout_concentrated": {"fanout_w1_rel", "fanout2_w1_rel"},
}


def test_c81_power_against_planted_defects():
    real, other = _db(6000), Q.shop(6001)
    for name, bad in Q.defects(other).items():
        rep = assess(SPEC, real, {k: coerce(v, SPEC["tables"][k]) for k, v in bad.items()}, scorecard=False)
        failed = {c["metric"] for c in rep["checks"] if not c["pass"]}
        print(f"C81 {name}: fidelity {rep['vector']['fidelity']['failed']} failed; metrics {sorted(failed)}")
        assert not rep["vector"]["fidelity"]["pass"], name
        assert failed & EXPECTED[name], (name, failed)


def test_c82_scorecard_carried_over():
    a, b = _db(7000), _db(7001)
    base = evaluate(SPEC, a, b)
    rep = assess(SPEC, a, b, base=base, config={"null_splits": 3, "multi_splits": 3})
    old = {c["id"]: c for c in rep["checks"] + rep["privacy_passthrough"] if c["source"] == "scorecard-1.0"}
    assert len(old) == len(base["scorecard"]["items"])
    for i in base["scorecard"]["items"]:
        c = old[f"{i['metric']}:{i['area']}"]
        assert (c["value"], c["limit"], c["pass"]) == (i["value"], i["limit"], i["pass"])

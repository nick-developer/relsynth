# SPDX-License-Identifier: Apache-2.0
"""A fact whose anchor and partner are two roles of one dimension (a trip's pick-up and drop-off
zone, an order's order and ship date), named by their key columns: the evaluation's grain and time
metrics use each role's own key (relsynth.evaluate)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from relsynth.evaluate import grain_metrics, time_metrics
from relsynth.spec import load_spec


def _spec(anchor="pu", partner="do"):
    return load_spec({
        "version": 1, "name": "trips",
        "source": {"kind": "csv", "path": "unused"}, "target": {"kind": "csv", "path": "unused"},
        "tables": {
            "zone": {"role": "entity", "primary_key": "zone_id",
                     "columns": {"zone_id": {"type": "id"}, "borough": {"type": "categorical"}}},
            "trip": {"role": "fact", "primary_key": "trip_id",
                     "columns": {"trip_id": {"type": "id"}, "pu": {"type": "id"}, "do": {"type": "id"},
                                 "at": {"type": "datetime"}, "fare": {"type": "float"}},
                     "fact": {"anchor": anchor, "partner": partner, "time_column": "at", "time_bucket": "day"}}},
        "relationships": [
            {"child": "trip", "child_key": "pu", "parent": "zone", "parent_key": "zone_id"},
            {"child": "trip", "child_key": "do", "parent": "zone", "parent_key": "zone_id"}],
    })


def _data(seed):
    rng = np.random.default_rng(seed)
    n = 400
    zone = pd.DataFrame({"zone_id": np.arange(1, 21), "borough": rng.choice(["A", "B"], 20)})
    # pick-ups crowd into two zones, drop-offs spread over all 20: the two roles have different weights
    trip = pd.DataFrame({"trip_id": np.arange(n), "pu": rng.choice([1, 2], n), "do": rng.integers(1, 21, n),
                         "at": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 30, n), "D"),
                         "fare": rng.gamma(2.0, 5.0, n)})
    return {"zone": zone, "trip": trip}


class _Stats:
    """In-database statistics stand-in: GROUP BY counts, refusing a repeated column as SQL output would."""
    def __init__(self, df):
        self.df, self.n_rows = df, len(df)

    def group_counts(self, cols, scores=(), require=None):
        assert len(set(cols)) == len(cols), cols
        return self.df.groupby(list(cols)).size().rename("n").reset_index()


def test_grain_pairs_use_both_roles_of_one_dimension():
    s = _spec()
    assert s["tables"]["trip"]["fact"].get("anchor_key") == "pu"
    real, syn = _data(1), _data(2)
    out = grain_metrics(s, real, syn, {"trip": _Stats(real["trip"])})
    assert "trip: pairs (pu, do)" in out
    out = grain_metrics(s, real, syn, {})
    assert "trip: pairs (pu, do)" in out


def test_time_metrics_use_the_anchor_role():
    """The anchor is the second link (drop-off): its effective size, not the pick-up's, is used."""
    from relsynth.evaluate import _kish
    s = _spec(anchor="do", partner="pu")
    assert s["tables"]["trip"]["fact"].get("anchor_key") == "do"
    real, syn = _data(1), _data(2)
    want, other = _kish(real["trip"], "do", len(real["trip"])), _kish(real["trip"], "pu", len(real["trip"]))
    assert abs(want - other) > 1
    assert time_metrics(s, real, syn, "trip")["n_eff"] == round(want, 1)

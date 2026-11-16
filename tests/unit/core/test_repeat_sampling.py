# SPDX-License-Identifier: Apache-2.0
"""Dataset i is a function of the fit and its own streams only.

Some parameter draws are one Gibbs scan (occupancy rates given the augmented
activity of anchors without pairs, block means given the measure variances).
Before the fix each redraw started where the previous one stopped, so a second
sample(dataset=0) in the same process differed from the first, and scenario vs
baseline comparisons were not common random numbers.  S8_reporting with 25
reporters has inactive reporters (data-augmented activity) and pair measures,
and drifted on the second draw."""
from __future__ import annotations

import copy

import pytest
import scenarios as S

import relsynth as rs
from relsynth.data import write_tables
from relsynth.spec import load_spec


@pytest.fixture(scope="module")
def reporting_model(tmp_path_factory):
    t, s, _ = S.reporting(n_reporters=25)
    db = tmp_path_factory.mktemp("repeat") / "reporting.db"
    write_tables(load_spec(copy.deepcopy(s)), t, {"kind": "sqlite", "path": str(db), "structure": "infer"})
    s["source"] = {"kind": "sqlite", "path": str(db)}
    return rs.fit(s, seed=3)


def _differing(a, b):
    return [t for t in a if not a[t].equals(b[t])]


def test_repeated_sample_is_identical_and_order_independent(reporting_model):
    m = reporting_model
    first = m.sample(dataset=0)
    assert any("act" in st for _r, st in m.synthesizer._fit_state)       # the case that drifted is exercised
    one = m.sample(dataset=1)
    again = m.sample(dataset=0)
    two = m.sample(dataset=2)
    one_again = m.sample(dataset=1)
    assert _differing(first, again) == []
    assert _differing(one, one_again) == []                        # after 0 and 2 instead of after 0 only
    assert _differing(first, one) != [] and _differing(one, two) != []  # datasets still differ from each other

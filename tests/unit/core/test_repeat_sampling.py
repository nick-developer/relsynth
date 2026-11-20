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

import numpy as np
import pytest
import scenarios as S

import relsynth as rs
from relsynth.data import write_tables
from relsynth.spec import load_spec

pytestmark = pytest.mark.slow             # fits S8_reporting end to end


def _fit(tmp_path_factory):
    t, s, _ = S.reporting(n_reporters=25)
    db = tmp_path_factory.mktemp("repeat") / "reporting.db"
    write_tables(load_spec(copy.deepcopy(s)), t, {"kind": "sqlite", "path": str(db), "structure": "infer"})
    s["source"] = {"kind": "sqlite", "path": str(db)}
    return rs.fit(s, seed=3)


@pytest.fixture(scope="module")
def reporting_model(tmp_path_factory):
    return _fit(tmp_path_factory)


@pytest.fixture(scope="module")
def reporting_model_b(tmp_path_factory):
    return _fit(tmp_path_factory)


def _differing(a, b):
    return [t for t in a if not a[t].equals(b[t])]


def test_repeated_sample_is_identical_and_order_independent(reporting_model, reporting_model_b):
    m = reporting_model
    first = m.sample(dataset=0)
    one = m.sample(dataset=1)
    again = m.sample(dataset=0)
    two = m.sample(dataset=2)
    one_again = m.sample(dataset=1)
    assert _differing(first, again) == []
    assert _differing(one, one_again) == []                        # after 0 and 2 instead of after 0 only
    assert _differing(first, one) != [] and _differing(one, two) != []  # datasets still differ from each other
    # a second fit of the same spec drawn in another order gives the same datasets
    b = reporting_model_b
    assert _differing(b.sample(dataset=2), two) == []
    assert _differing(b.sample(dataset=0), first) == []


def test_carried_state_varies_between_datasets(reporting_model):
    """Each dataset starts from its own state of the carried chain (not one frozen
    state), so the spread between datasets keeps the uncertainty of the augmented
    activity; state 0 is the end of the fit."""
    m = reporting_model
    m.sample(dataset=0)
    m.sample(dataset=3)
    fs = m.synthesizer._fit_state
    occ = [i for i, r in enumerate(fs["rels"]) if "act" in r.CARRIED and (~r.has).any()]
    assert occ                                                  # the case that drifted is exercised
    s0, s3 = fs["states"][0][0], fs["states"][3][0]
    assert any(not np.array_equal(s0[i]["act"], s3[i]["act"]) for i in occ)


def test_far_index_is_rebuilt_from_its_checkpoint(reporting_model):
    """Only every 32nd state (and the latest) is kept: dataset 40 drawn again after
    dataset 35 is rebuilt from state 32 and the chain's saved stream position."""
    m = reporting_model
    forty = m.sample(dataset=40)
    m.sample(dataset=35)
    assert 40 not in m.synthesizer._fit_state["states"]
    assert _differing(m.sample(dataset=40), forty) == []
    assert set(m.synthesizer._fit_state["states"]) <= {0, 32, 40}

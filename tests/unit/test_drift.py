# SPDX-License-Identifier: Apache-2.0
"""Partition-aware statistics (relsynth.runtime.drift, Prop. 904), and the
source's ODCS contract on output generated in partitions."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from conftest import scenario

from relsynth.engine import Synthesizer
from relsynth.ir import lift as from_v1
from relsynth.ir.contract_check import check_contract
from relsynth.ir.odcs import to_odcs
from relsynth.runtime import partition_drift, sample_partitioned
from relsynth.runtime.drift import chi2_homogeneity, holm, welch_anova
from relsynth.spec import load_spec


def _reader(tables):
    def read(name, columns):
        if name not in tables:
            raise KeyError(name)
        return tables[name]
    return read


@pytest.fixture(scope="module")
def dw():
    t, s, _ = scenario("S10_order_dw")
    syn = Synthesizer(load_spec(copy.deepcopy(s))).fit(t)
    return t, s, syn


def test_synthetic_partitioned_output_meets_the_source_contract(dw):
    """The contract exported from the real data's spec holds on the synthetic
    data generated in partitions: keys, links, types, required columns."""
    t, s, syn = dw
    contract = to_odcs(from_v1(copy.deepcopy(s)))
    real = check_contract(contract, _reader(t))
    out, _ = sample_partitioned(syn, index=0, executor="local", partition_rows=200)
    rep = check_contract(contract, _reader(out))
    assert real.ok, real.to_markdown()
    assert rep.ok, rep.to_markdown()
    kinds = {c["check"].split()[0] for c in rep.checks if c["result"] == "passed"}
    assert {"exists", "logicalType", "primaryKey", "foreignKey"} <= kinds


def test_holm_and_tests_against_reference_implementations():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    groups = [rng.normal(m, s, n) for m, s, n in ((0, 1, 40), (0.3, 2, 60), (0, 0.5, 30))]
    f, p, _ = welch_anova([len(g) for g in groups], [g.mean() for g in groups], [g.var(ddof=1) for g in groups])
    # Welch's F computed directly (Welch 1951)
    n = np.array([len(g) for g in groups])
    m = np.array([g.mean() for g in groups])
    w = n / np.array([g.var(ddof=1) for g in groups])
    mw = (w * m).sum() / w.sum()
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f_ref = ((w * (m - mw) ** 2).sum() / 2) / (1 + 2 * 1 / 8 * tmp)
    assert f == pytest.approx(f_ref)
    from scipy import stats as st
    tab = np.array([[30, 20, 10], [25, 25, 12]])
    chi, pc, v = chi2_homogeneity(tab)
    ref = st.chi2_contingency(tab[:, np.argsort(tab.sum(axis=0))], correction=False)
    assert chi == pytest.approx(ref[0]) and pc == pytest.approx(ref[1])


def _frame(rng, n, shift=0.0, flip=0.0):
    d = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    late = d >= pd.Timestamp("2024-07-01")
    x = rng.gamma(2.0, 1.0, n) + shift * late
    p = np.where(late, 0.5 + flip, 0.5)
    return pd.DataFrame({"day": d, "x": x, "y": rng.normal(size=n), "c": np.where(rng.random(n) < p, "a", "b"),
                         "k": rng.choice(list("pqrst"), n)})


@pytest.mark.science
def test_prop_904_family_wise_error_and_power():
    """Prop. 904: under one law across partitions, the chance of flagging any
    column is at most alpha (here: rejection rate of the Holm family below
    alpha plus three Monte Carlo standard errors); a shift is found."""
    rng = np.random.default_rng(904)
    reps, alpha = 300, 0.05
    hits = 0
    for _ in range(reps):
        r = partition_drift(_frame(rng, 1200), "month(day)", alpha=alpha, min_effect={"numeric": 0, "categorical": 0})
        hits += bool(r.drifting)
    rate = hits / reps
    assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps), rate
    r = partition_drift(_frame(rng, 20000, shift=0.6, flip=0.2), "month(day)")
    assert set(r.drifting) == {"x", "c"}, r.to_markdown()


def test_drift_on_a_partitioned_lake_table(tmp_path):
    pytest.importorskip("pyiceberg")
    from relsynth.connectors.lakehouse import IcebergConnector
    rng = np.random.default_rng(5)
    df = _frame(rng, 5000, shift=1.0)
    lake = IcebergConnector(tmp_path / "lake", schema="s", partition_by={"f": ["month(day)", "k"]}, create=True)
    lake.write_tables({"tables": {"f": {"columns": {}}}, "relationships": []}, {"f": df})
    r = partition_drift(lake, table="f")
    assert r.by == ["month(day)", "k"] and r.partitions == 60 and r.rows == 5000
    assert "x" in r.drifting and "y" not in r.drifting

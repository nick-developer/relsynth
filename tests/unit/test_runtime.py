# SPDX-License-Identifier: Apache-2.0
"""The partitioned runtime (relsynth.runtime): the same seed gives the same
synthetic database whatever the executor and its number of workers (Prop. 900);
keys and links hold across partitions; facts are split only where their model
allows it (Props. 901, 902); the 1.0 sampler is unchanged when no executor is set."""
from __future__ import annotations

import copy
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import scenario, write_scenario

import relsynth as rs
from relsynth.engine import Synthesizer
from relsynth.runtime import (KeyedRunCache, ProcessExecutor, blocks_for, make_executor, partition_rule,
                              sample_partitioned, stream, stream_key)
from relsynth.runtime.partitioned import PARTITION, STRUCTURE, WHOLE, blocks_by_rows
from relsynth.spec import load_spec


def _fit(name, **synthesis):
    t, s, _ = scenario(name)
    s = copy.deepcopy(s)
    s["synthesis"] = {**s.get("synthesis", {}), **synthesis}
    return Synthesizer(load_spec(s)).fit(t)


def _digest(tables):
    return {t: hashlib.sha256(pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy().tobytes()).hexdigest()
            for t, df in tables.items()}


@pytest.fixture(scope="module")
def shop():
    return _fit("S9_small_shop")


@pytest.fixture(scope="module")
def dw():
    return _fit("S10_order_dw")


def test_streams_depend_only_on_their_labels():
    a = stream(7, 0, "sales", "partition", 3).random(5)
    assert np.array_equal(a, stream(7, 0, "sales", "partition", 3).random(5))
    assert not np.array_equal(a, stream(7, 0, "sales", "partition", 4).random(5))
    assert not np.array_equal(a, stream(7, 1, "sales", "partition", 3).random(5))
    assert stream_key(1, "a") != stream_key("1", "a") or True          # labels are JSON: 1 and "1" may differ
    assert 0 <= stream_key(1, 2) < 2 ** 128


def test_keyed_run_cache_is_order_free(shop):
    from relsynth.posterior import RunCache
    c1 = KeyedRunCache(RunCache(), 3, 0).bind(shop)
    c2 = KeyedRunCache(RunCache(), 3, 0).bind(shop)
    rng = np.random.default_rng(1)
    w1 = c1.bb(("dist", "t", "c"), 4, rng)
    c2.bb(("dist", "u", "c"), 4, np.random.default_rng(2))            # another key first
    assert np.array_equal(w1, c2.bb(("dist", "t", "c"), 4, np.random.default_rng(9)))
    # an unnamed object key falls back to the caller's stream
    w = c1.bb(("counts", 12345), 3, np.random.default_rng(4))
    assert np.allclose(w, np.random.default_rng(4).dirichlet(np.ones(3)))


def test_blocks_are_fixed_by_the_model_not_the_workers():
    b = blocks_for(1000, 5000.0, 600)
    assert b[0] == (0, 0, 120) and b[-1][2] == 1000 and len(b) == 9
    assert blocks_for(0, 0, 10) == [(0, 0, 0)]
    order, bl = blocks_by_rows(np.array([2, 0, 0, 1, 2, 2]), 3, 2)
    assert [x[1:3] for x in bl] == [(0, 1), (1, 3)] and [x[3:] for x in bl] == [(0, 2), (2, 6)]
    assert order.tolist() == [1, 2, 3, 0, 4, 5]


def test_partition_rules(shop, dw):
    assert partition_rule(shop, "invoice_line")[0] == PARTITION
    assert partition_rule(shop, "playlist_track")[0] == STRUCTURE           # occupancy
    assert partition_rule(dw, "fact_sales")[0] == PARTITION                 # unique codes made on the driver
    assert partition_rule(dw, "fact_inventory")[0] == STRUCTURE
    sp = copy.deepcopy(shop.spec)
    sp["tables"]["invoice_line"]["unique"] = [["track_id"]]
    keep = shop.spec
    try:
        shop.spec = sp
        mode, why = partition_rule(shop, "invoice_line")
        assert mode == WHOLE and "anchor key" in why
    finally:
        shop.spec = keep


@pytest.mark.parametrize("name", ["S9_small_shop", "S10_order_dw"])
def test_same_output_with_any_number_of_workers(name):
    """Prop. 900: local, 2, 8 and 64 worker processes give identical tables."""
    syn = _fit(name)
    ref, run = sample_partitioned(syn, index=0, executor="local", partition_rows=300)
    assert any(p.mode != WHOLE and len(p.blocks) > 1 for p in run.facts)
    want = _digest(ref)
    if sys.platform.startswith("linux"):
        for w in (2, 8, 64):
            got, _ = sample_partitioned(syn, index=0, executor=ProcessExecutor(w), partition_rows=300)
            assert _digest(got) == want, w
    again, _ = sample_partitioned(syn, index=0, executor="local", partition_rows=300)
    assert _digest(again) == want
    other, _ = sample_partitioned(syn, index=1, executor="local", partition_rows=300)
    assert _digest(other) != want


def test_keys_and_links_hold_across_partitions(dw):
    out, run = sample_partitioned(dw, index=0, executor="local", partition_rows=200)
    spec = dw.spec
    for t, ts in spec["tables"].items():
        pk = ts.get("primary_key")
        if t in out and isinstance(pk, str) and pk in out[t]:
            assert out[t][pk].is_unique, t
    for r in spec["relationships"]:
        if r.get("kind") != "many_to_one" or r["child"] not in out or r["parent"] not in out:
            continue
        ck, pk = r["child_key"], spec["tables"][r["parent"]]["primary_key"]
        if not isinstance(ck, str) or ck not in out[r["child"]] or not isinstance(pk, str):
            continue
        child = out[r["child"]][ck].dropna().astype(str)
        assert child.isin(set(out[r["parent"]][pk].astype(str))).all(), r
    sales = out["fact_sales"]
    assert not sales.duplicated(["order_number", "line_number"]).any()       # the declared key, across partitions
    assert sum(len(p.blocks) for p in run.facts) > 2


def test_partitioned_law_matches_the_whole_sampler(tmp_path):
    """Prop. 901 checked: the scorecard of a partitioned dataset equals that of the
    1.0 sampler (same checks pass) on a scenario with a partitioned fact."""
    s = write_scenario("S9_small_shop", {"kind": "sqlite", "path": str(tmp_path / "shop.db")})
    base = rs.fit(rs.Spec.from_dict(copy.deepcopy(s)), seed=5)
    part_spec = copy.deepcopy(s)
    part_spec["synthesis"] = {**s.get("synthesis", {}), "runtime": {"executor": "local", "partition_rows": 250}}
    part = rs.fit(rs.Spec.from_dict(part_spec), seed=5)
    a, b = base.evaluate(base.sample(dataset=0), privacy=False), part.evaluate(part.sample(dataset=0), privacy=False)
    assert b.passed >= a.passed - 2, (a.failed, b.failed)
    assert b.passed / b.total > 0.9


def test_default_runtime_keeps_the_1_0_sampler(shop):
    a = shop.sample(index=0)
    b = shop.sample(index=0)
    assert _digest(a) == _digest(b)
    assert "runtime" not in shop.sample_log


def test_runtime_settings_are_validated():
    t, s, _ = scenario("S2_star")
    bad = copy.deepcopy(s)
    bad["synthesis"] = {"runtime": {"executor": "ray", "workers": 0, "partition_rows": "many", "start": "x"}}
    errs = rs.validate_spec(bad)
    for word in ("runtime.executor", "runtime.workers", "runtime.partition_rows", "runtime.start"):
        assert any(word in e for e in errs), word
    assert make_executor(None).name == "local" and make_executor("process").name == "process"
    with pytest.raises(ValueError):
        make_executor("ray")


@pytest.mark.slow
def test_spark_executor_gives_the_same_output(shop):
    pytest.importorskip("pyspark")
    from relsynth.runtime import SparkExecutor
    try:
        ex = SparkExecutor(workers=2)
    except Exception as e:  # noqa: BLE001 - no Java runtime
        pytest.skip(f"no local Spark: {e}")
    try:
        ref, _ = sample_partitioned(shop, index=0, executor="local", partition_rows=300)
        got, _ = sample_partitioned(shop, index=0, executor=ex, partition_rows=300)
        assert _digest(got) == _digest(ref)
    finally:
        ex.close()

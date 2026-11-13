# SPDX-License-Identifier: Apache-2.0
"""Engine fidelity (G4): count start and segment totals, calendar pooling,
two-hop inclusion levels, stock floors and multi-hot set arrays.
The statistical checks are C1782-C1785 in tests/science/test_g4_checks.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from relsynth.documents import shred as sh
from relsynth.documents.infer import profile_collection
from relsynth.engine import Synthesizer, _balance_totals, _start_counts, cumulative_apply
from relsynth.spec import DEFAULT_SYNTHESIS
from relsynth.timeseries.calendar import Intensity


def test_defaults():
    f = DEFAULT_SYNTHESIS["facts"]
    assert f["segment_totals"] == "auto" and f["dp_count_start"] == "draw"
    assert DEFAULT_SYNTHESIS["structure"]["inclusion_fk_levels"] is True
    assert DEFAULT_SYNTHESIS["rules"]["balance_floor"] == "auto"


def test_start_counts_hits_the_total_and_reaches_large_counts():
    rng = np.random.default_rng(0)
    f = np.zeros(30)
    f[[1, 2, 3, 27]] = [0.5, 0.3, 0.19, 0.01]
    big = 0
    for _ in range(200):
        N = _start_counts(f, 50, 110, rng)
        assert N.sum() == 110 and (N >= 0).all()
        big += int((N >= 20).any())
    assert big > 0                      # a balanced start (2-3 each) never holds 27


def test_balance_totals_moves_segment_totals_towards_their_count_law():
    rng = np.random.default_rng(1)
    K, n = 3, 600
    seg_p = np.full((1, K), 1 / K)
    zc = np.zeros(n, np.int64)
    seg_of = rng.choice(K, size=n)
    M = np.array([100, 100, 100])
    dp = {"counts": [np.full(50, 1), np.full(50, 2), np.full(50, 3)]}   # totals 100, 200, 300
    out = _balance_totals(seg_of, zc, seg_p, M, dp, None, rng, sweeps=6)
    R = np.bincount(out, minlength=K)
    assert R.sum() == n and R[0] < R[1] < R[2]
    assert abs(R[2] - 300) < 40
    # mandatory totals are kept
    need = np.array([250, 0, 0])
    out = _balance_totals(np.zeros(n, np.int64), zc, seg_p, M, dp, need, rng)
    assert np.bincount(out, minlength=K)[0] >= 250


def test_pooled_calendar_has_one_row_for_every_segment():
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 730, 3000)
    seg = rng.integers(0, 4, 3000)
    I = Intensity(pooled=True).fit(t, seg, 4, 0, 730)
    assert I.K == 1 and (I.seg(np.array([0, 3])) == 0).all()
    a, b = I.forward(np.array([100.0]), np.array([0])), I.forward(np.array([100.0]), np.array([3]))
    assert a == b
    J = Intensity().fit(t, seg, 4, 0, 730)
    assert J.K == 4


def _inclusion_syn():
    """product -> subcategory -> category; sales never reach category C."""
    s = Synthesizer.__new__(Synthesizer)
    s.cfg = {"structure": {"inclusion_min_expected": 10.0, "conditional_inclusion": "auto",
                           "inclusion_fk_levels": True}}
    sub = pd.DataFrame({"sk": range(30), "cat": ["A"] * 10 + ["B"] * 10 + ["C"] * 10})
    prod = pd.DataFrame({"pk": range(300), "sk": np.arange(300) % 30, "color": "red"})
    sold = prod.pk[sub.cat.to_numpy()[prod.sk] != "C"].to_numpy()
    sales = pd.DataFrame({"id": range(2000), "pk": np.random.default_rng(3).choice(sold, 2000)})
    s.spec = {"tables": {
        "sub": {"role": "entity", "primary_key": "sk", "columns": {"sk": {"type": "id"}, "cat": {"type": "categorical"}}},
        "prod": {"role": "entity", "primary_key": "pk",
                 "columns": {"pk": {"type": "id"}, "sk": {"type": "id"}, "color": {"type": "categorical"}}},
        "sales": {"role": "fact", "primary_key": "id", "columns": {"id": {"type": "id"}, "pk": {"type": "id"}}}}}
    s.fk = {"sub": [], "prod": [{"parent": "sub", "child_key": "sk", "name": "prod_sub"}],
            "sales": [{"parent": "prod", "child_key": "pk", "name": "sales_prod"}]}
    s._excluded = lambda t: {"sk"} if t == "prod" else set()
    s.lookups = {t: {} for t in s.spec["tables"]}
    s.derived = {t: {} for t in s.spec["tables"]}
    s._ref_values = lambda t, c, tables: tables[t][c]
    s.log = {"decisions": {}}
    return s, {"sub": sub, "prod": prod, "sales": sales}


def test_inclusion_reaches_one_hop_further():
    s, tables = _inclusion_syn()
    s._fit_inclusion(tables)
    rules = s.cind[("sales", "pk")]
    assert ("sk\x1fsub\x1fcat", "C", False) in rules     # no single subcategory carries 10 expected
    assert "sk.cat = C" in s.log["decisions"]["sales"]["sales_prod inclusion"]
    mask = s._inclusion_mask("sales", {"child_key": "pk", "parent": "prod"}, len(tables["prod"]), tables)
    cat = tables["sub"].cat.to_numpy()[tables["prod"].sk]
    assert (mask[cat == "C"] == 0).all() and (mask[cat != "C"] == 1).all()
    s.cfg["structure"]["inclusion_fk_levels"] = False
    s._fit_inclusion(tables)
    assert not any("\x1f" in r[0] for r in s.cind.get(("sales", "pk"), []))


def test_stock_floor_caps_outflow():
    df = pd.DataFrame({"p": [1] * 4, "d": [1, 2, 3, 4], "i": [5, 0, 0, 3], "o": [0, 4, 4, 0], "y": [5, 0, 0, 0]})
    rule = {"column": "y", "by": "p", "order": "d", "step": "i - o", "opening": 0.0, "floor": 0.0}
    out = cumulative_apply(df.copy(), rule)
    assert out.y.tolist() == [5, 1, 0, 3] and out.o.tolist() == [0, 4, 1, 0]


def _tag_docs(n=300, seed=4):
    rng = np.random.default_rng(seed)
    vocab = ["red", "blue", "green", "kids", "office", "school"]
    return [{"_id": i, "tags": [str(x) for x in rng.choice(vocab, size=int(rng.integers(1, 4)), replace=False)]}
            for i in range(n)]


def test_multihot_round_trip_and_length_matching():
    docs = _tag_docs()
    lay = sh.build_layout({"items": profile_collection("items", docs)}, [])
    tl = lay.tables["items"]
    hot = [c for c in tl.columns if c.role == "multihot"]
    assert len(hot) == 6 and "tags__n" in tl.length_bounds and len(lay.tables) == 1
    frames = sh.shred(lay, {"items": docs})
    back = sh.nest(lay, frames, keep_ids=True)["items"]
    assert back == docs                                    # order kept exactly
    # a modelled length the cells disagree with: elements dropped or drawn, never repeated
    f = {k: v.copy() for k, v in frames.items()}
    f["items"]["tags__n"] = np.where(np.arange(len(docs)) % 2 == 0, 1, 3)
    out = sh.nest(lay, f, keep_ids=True)["items"]
    assert [len(d["tags"]) for d in out] == f["items"]["tags__n"].tolist()
    assert all(len(set(d["tags"])) == len(d["tags"]) for d in out)
    assert out == sh.nest(lay, f, keep_ids=True)["items"]     # seeded by holder

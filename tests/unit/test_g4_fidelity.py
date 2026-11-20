# SPDX-License-Identifier: Apache-2.0
"""Engine fidelity (G4): count start and segment totals, calendar pooling,
two-hop inclusion levels, stock floors and multi-hot set arrays.
The statistical checks are C1782-C1785 in tests/science/test_g4_checks.py."""
from __future__ import annotations

import pytest
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
    # a segment with members but no observed counts keeps the pooled law's share
    dp_new = {"counts": [np.full(50, 2), np.full(50, 2), np.zeros(0)]}
    out = _balance_totals(rng.choice(K, size=n), zc, seg_p, M, dp_new, None, rng, sweeps=6)
    assert abs(np.bincount(out, minlength=K)[2] - 200) < 40


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


def _bundle_docs(n=600, seed=4):
    """Sets in a fixed order (like airbnb's host verifications): bundles of
    elements that go together, always listed in the vocabulary's order."""
    rng = np.random.default_rng(seed)
    order = ["email", "phone", "reviews", "jumio", "gov_id", "selfie"]
    bundles = [order[:2], order[:3], order[:5], order, ["email"], ["phone", "gov_id"]]
    return [{"_id": i, "tags": list(bundles[int(rng.integers(len(bundles)))])} for i in range(n)]


def test_multihot_round_trip_and_length_matching():
    docs = _tag_docs()
    lay = sh.build_layout({"items": profile_collection("items", docs)}, [], {"multihot": "on"})
    tl = lay.tables["items"]
    hot = [c for c in tl.columns if c.role == "multihot"]
    assert len(hot) == 6 and "tags__n" in tl.length_bounds and len(lay.tables) == 1
    frames = sh.shred(lay, {"items": docs})
    # orders that disagree: the cells hold positions, and the round trip is exact
    assert lay.shapes["items"]["tags"]["multihot_order"]["kind"] == "positions" and all(c.hot_pos for c in hot)
    assert sh.nest(lay, frames, keep_ids=True)["items"] == docs
    # a modelled length the cells disagree with: elements dropped or drawn, never repeated
    f = {k: v.copy() for k, v in frames.items()}
    f["items"]["tags__n"] = np.where(np.arange(len(docs)) % 2 == 0, 1, 3)
    out = sh.nest(lay, f, keep_ids=True)["items"]
    assert [len(d["tags"]) for d in out] == f["items"]["tags__n"].tolist()
    assert all(len(set(d["tags"])) == len(d["tags"]) for d in out)
    assert out == sh.nest(lay, f, keep_ids=True)["items"]     # seeded by holder


def test_multihot_fixed_order_round_trip_is_exact():
    docs = _bundle_docs()
    lay = sh.build_layout({"items": profile_collection("items", docs)}, [])
    info = lay.shapes["items"]["tags"]
    assert info["multihot_choice"]["multihot"] and info["multihot_choice"]["bic_gain"] > 0
    assert info["multihot_order"]["kind"] == "fixed"
    frames = sh.shred(lay, {"items": docs})
    assert set(np.unique(frames["items"][[c for c in frames["items"] if "__h" in c]].to_numpy())) <= {0, 1}
    assert sh.nest(lay, frames, keep_ids=True)["items"] == docs


def test_multihot_choice_options_and_recorded_dependence():
    """Independent elements are multi-hot too (a child table could not hold a set
    without repairs), with a BIC gain <= 0 recorded; bundles record a positive
    gain; `off` never uses multi-hot; the cost bound applies in auto, not in on;
    unknown options are refused."""
    rng = np.random.default_rng(8)
    vocab = [f"v{i}" for i in range(8)]
    docs = [{"_id": i, "tags": [v for v in vocab if rng.random() < 0.3]} for i in range(3000)]
    prof = {"items": profile_collection("items", docs)}
    lay = sh.build_layout(prof, [])
    assert len([c for c in lay.tables["items"].columns if c.role == "multihot"]) == 8
    assert lay.shapes["items"]["tags"]["multihot_choice"]["bic_gain"] <= 0
    bundled = {"items": profile_collection("items", _bundle_docs())}
    assert sh.build_layout(bundled, []).shapes["items"]["tags"]["multihot_choice"]["bic_gain"] > 0
    assert not [c for c in sh.build_layout(bundled, [], {"multihot": "off"}).tables["items"].columns
                if c.role == "multihot"]
    capped = sh.build_layout(bundled, [], {"multihot_max_columns": 3})
    assert capped.shapes["items"]["tags"]["multihot_choice"]["reason"] == "multihot_max_columns"
    assert [c for c in sh.build_layout(bundled, [], {"multihot": "on", "multihot_max_columns": 3}).tables[
        "items"].columns if c.role == "multihot"]
    with pytest.raises(ValueError):
        sh.build_layout(bundled, [], {"multihot_max": 3})


def test_spec_drafts_no_rule_on_multihot_cells(tmp_path):
    from relsynth.documents.relational import spec_for, write_sqlite
    docs = _tag_docs(400)
    lay = sh.build_layout({"items": profile_collection("items", docs)}, [], {"multihot": "on"})
    db = write_sqlite(lay, sh.shred(lay, {"items": docs}), tmp_path / "t.sqlite")
    raw = spec_for(lay, db)
    assert not [r for r in raw.get("rules", []) if "__h" in repr(r)]
    assert raw["tables"]["items"]["columns"]["tags__h0"] == {"type": "integer", "min": 0}   # positions: orders disagree


def test_multihot_falls_back_on_a_name_clash_or_null_elements():
    docs = _tag_docs(200)
    for d in docs:
        d["tags__h0"] = d["_id"] % 3                     # a source field named like a multi-hot cell
    lay = sh.build_layout({"items": profile_collection("items", docs)}, [])
    assert not [c for c in lay.tables["items"].columns if c.role == "multihot"]
    assert sh.nest(lay, sh.shred(lay, {"items": docs}), keep_ids=True)["items"] == docs
    nul = _tag_docs(200)
    nul[0]["tags"] = ["red", None]
    lay = sh.build_layout({"items": profile_collection("items", nul)}, [])
    assert not [c for c in lay.tables["items"].columns if c.role == "multihot"]


def test_start_counts_fallback_hits_the_total():
    rng = np.random.default_rng(5)
    f = np.array([0.0, 1.0])                             # only counts 0 and 1 have mass
    N = _start_counts(f, 4, 10, rng)                     # 10 rows over 4 parents: unreachable under f
    assert N.sum() == 10 and (N >= 0).all()


def test_inclusion_key_levels_only_for_tables_that_keep_their_keys():
    """A key value is a level only when the referenced table keeps its source rows
    (mode copy): a synthesized table gets fresh keys, so its key 17 is another row."""
    s, tables = _inclusion_syn()
    s.cfg["structure"]["inclusion_min_expected"] = 1.0
    s.copied = set()
    s._fit_inclusion(tables)
    assert not any(r[0] == "sk" for r in s.cind.get(("sales", "pk"), []))
    s.copied = {"sub"}
    s._fit_inclusion(tables)
    assert any(r[0] == "sk" and r[1] is not None for r in s.cind[("sales", "pk")])


def test_constant_column_keeps_its_empty_share_and_suppresses_a_rare_value(tmp_path):
    """A column with one value and empty cells: synthetic rows hold the value with
    the source's share; a value held by fewer than suppress_small_counts rows (one
    record's own value) is never written out."""
    import relsynth as rs
    n = 400
    df = pd.DataFrame({"id": range(n), "x": np.random.default_rng(0).normal(size=n),
                       "half": np.where(np.arange(n) % 2 == 0, "yes", None),
                       "one": [None] * (n - 1) + ["2017-01-01"]})
    (tmp_path / "src").mkdir()
    df.to_csv(tmp_path / "src" / "t.csv", index=False)
    spec = {"version": 1, "source": {"kind": "csv", "path": str(tmp_path / "src")},
            "tables": {"t": {"primary_key": "id", "columns": {
                "id": "id", "x": "float", "half": "constant", "one": "constant"}}},
            "relationships": [], "synthesis": {"seed": 3}}
    syn = rs.fit(spec)
    out = syn.sample()["t"]
    assert abs(out["half"].notna().mean() - 0.5) < 0.1 and set(out["half"].dropna()) == {"yes"}
    assert out["one"].isna().all()
    # the standalone replica reads the exported share too
    from relsynth import replica
    from relsynth.export import posterior_draws, structure
    model = syn.synthesizer
    S, _ = structure(model)
    assert S["constant_shares"]["t"]["half"] == pytest.approx(0.5, abs=0.01)
    mean, D = posterior_draws(model, 1)
    rep = replica.generate({"structure": S, "mean": mean, "draws": D}, "mean", 7)["t"]
    assert abs(rep["half"].notna().mean() - S["constant_shares"]["t"]["half"]) < 0.1
    assert set(rep["half"].dropna()) == {"yes"}


def test_a_date_drawn_as_an_offset_stays_within_its_source_range(tmp_path):
    """Last activity capped at the extract date (right-censored) and drafted as
    creation plus a gap: a late creation leaves little room, and the derived date
    is capped at the latest source date instead of running past it."""
    import sqlite3

    import relsynth as rs
    rng = np.random.default_rng(1)
    n = 3000
    created = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.uniform(0, 1500, n), unit="D")
    end = pd.Timestamp("2018-01-01") + pd.Timedelta(days=1500)
    last = created + pd.to_timedelta(np.where(rng.random(n) < 0.5, rng.exponential(600, n), 0.0), unit="D")
    last = pd.DatetimeIndex(np.minimum(last.values, np.datetime64(end)))
    df = pd.DataFrame({"id": range(1, n + 1), "score": rng.poisson(3, n),
                       "created": created.strftime("%Y-%m-%d %H:%M:%S"),
                       "last_activity": last.strftime("%Y-%m-%d %H:%M:%S")})
    with sqlite3.connect(tmp_path / "c.db") as con:
        df.to_sql("posts", con, index=False)
    d = rs.introspect(f"sqlite:{tmp_path / 'c.db'}", name="p").raw
    assert d["tables"]["posts"]["columns"]["last_activity"].get("offset_from") == "created"
    out = rs.fit(d).sample()["posts"]
    la, c = pd.to_datetime(out["last_activity"]), pd.to_datetime(out["created"])
    assert la.max() <= end + pd.Timedelta(seconds=1)
    assert ((la - c) >= pd.Timedelta(milliseconds=-1)).all()


def test_multihot_never_copies_personal_element_values():
    """An array of e-mail addresses is a set array, but a multi-hot cell would write
    the real addresses back: elements typed as personal data stay a child table,
    whose pii type generates them, under the default multihot auto."""
    import random

    from relsynth.documents.pipeline import DocumentSynthesizer
    random.seed(1)
    staff = [f"person{i}.realname@corp-example.com" for i in range(40)]
    docs = [{"_id": i, "priority": random.randint(1, 5), "cc": random.sample(staff, random.randint(1, 3))}
            for i in range(400)]
    out = DocumentSynthesizer({"tickets": docs}, seed=3).synthesize()
    vals = {v for d in out["tickets"] for v in (d.get("cc") or [])}
    assert vals and not vals & set(staff)

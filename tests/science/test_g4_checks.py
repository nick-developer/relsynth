# SPDX-License-Identifier: Apache-2.0
"""Checks C1780-C1799: engine fidelity and binary columns (gap plan G4;
docs/proofs/sec_g4_fidelity.tex), tested on the shipped code.

    pytest -m science tests/science/test_g4_checks.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytestmark = [pytest.mark.science]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import scenario  # noqa: E402

import relsynth as rs  # noqa: E402
from relsynth.binary import PLACEHOLDERS, BinaryColumn  # noqa: E402
from relsynth.data import write_tables  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


def _photos(n, rng):
    noimg = b"GIF89a" + rng.bytes(1667)
    out = []
    for _ in range(n):
        u = rng.random()
        out.append(None if u < 0.1 else noimg if u < 0.55 else b"\xff\xd8\xff\xe0" + rng.bytes(int(rng.lognormal(9, .7))))
    return out


# --------------------------------------------------------------------- C1780
def test_c1780_binary_stream_changes_nothing_else(tmp_path):
    """Prop. 1780: with the generator on, every other column of every table is
    byte-identical to the run with it off, for posterior draws and means."""
    t, s, _ = scenario("S2_star")
    t, s = copy.deepcopy(t), copy.deepcopy(s)
    t["product"]["photo"] = _photos(len(t["product"]), np.random.default_rng(0))
    s["tables"]["product"]["columns"]["photo"] = "binary"
    db = tmp_path / "s.db"
    write_tables(load_spec(copy.deepcopy(s)), t, {"kind": "sqlite", "path": str(db), "structure": "infer"})
    s["source"] = {"kind": "sqlite", "path": str(db)}
    for mode in ("draw", "mean"):
        runs = {}
        for gen in ("none", "placeholder", "random"):
            sp = copy.deepcopy(s)
            sp["synthesis"]["binary"] = {"generator": gen}
            sp["synthesis"]["posterior"] = {"mode": mode}
            m = rs.fit(sp, seed=2)
            runs[gen] = [dict(m.sample(dataset=i)) for i in range(2)]
        for gen in ("placeholder", "random"):
            for i in range(2):
                for tb, df in runs["none"][i].items():
                    cols = [c for c in df.columns if c != "photo"]
                    pd.testing.assert_frame_equal(df[cols], runs[gen][i][tb][cols])
            p0, p1 = (runs[gen][i]["product"]["photo"] for i in range(2))
            assert p0.notna().any() and not p0.equals(p1)          # datasets get their own values


# --------------------------------------------------------------------- C1781
def test_c1781_binary_release_matches_the_length_law_and_copies_nothing():
    """Prop. 1781: null share, the shared exact length and the bin shares follow
    the source within their posterior predictive spread; no value is a source value."""
    rng = np.random.default_rng(7)
    src = _photos(2000, rng)
    m = BinaryColumn("placeholder", k=5).fit(src)
    n = 20000
    out = m.generate(n, np.random.default_rng(8), "draw")

    def summary(vals):
        L = np.array([-1 if v is None else len(v) for v in vals])
        pos = L[(L > 0) & (L != 1673)]
        bins = pd.Series([int(x).bit_length() for x in pos]).value_counts(normalize=True)
        return (L == -1).mean(), (L == 1673).mean(), bins
    sn, s1673, sb = summary(src)
    gn, g1673, gb = summary(out)
    N = len(src)
    # posterior predictive sd of a share from N source values, plus the sampling sd of n draws
    for a, b in ((sn, gn), (s1673, g1673)):
        sd = np.sqrt(a * (1 - a) * (1 / N + 1 / n))
        assert abs(a - b) < 4 * sd, (a, b, sd)
    idx = sb.index.union(gb.index)
    tv = 0.5 * (sb.reindex(idx, fill_value=0) - gb.reindex(idx, fill_value=0)).abs().sum()
    assert tv < 0.06
    # exact lengths outside the shared one never come back exactly unless shared by k values
    rare = pd.Series([len(v) for v in src if v is not None]).value_counts()
    rare = set(rare[rare < 5].index)
    got = np.array([len(v) for v in out if v is not None and not v.startswith(PLACEHOLDERS["image/gif"])])
    hits = np.isin(got, list(rare)).mean()
    # a rare length comes back only by chance: a draw in bin b hits the rare lengths of that bin with
    # probability (their log-uniform mass); compare with that expectation
    rl = np.array(sorted(rare))
    def p_hit(L):
        b = int(L).bit_length()
        lo = 1 << (b - 1)
        r = rl[(rl >= lo) & (rl < 2 * lo)]
        return float(np.sum(np.log2((r + 1) / r))) if len(r) else 0.0
    exp = np.mean([p_hit(L) for L in got])
    assert abs(hits - exp) < 4 * np.sqrt(exp * (1 - exp) / len(got)) + 1e-3, (hits, exp)
    assert not ({bytes(v) for v in src if v is not None} & {v for v in out if v is not None})
    # the null share's posterior: the generated share sits inside its 99.9% interval
    a0, b0 = sum(v is None for v in src) + .5, sum(v is not None for v in src) + .5
    lo, hi = stats.beta.ppf([0.0005, 0.9995], a0, b0)
    assert lo - 0.01 < gn < hi + 0.01


# --------------------------------------------------------------------- C1782
def _counts_syn(seed):
    from relsynth.engine import Synthesizer
    s = Synthesizer.__new__(Synthesizer)
    s.cfg = {"facts": {"dp_count_sweeps": 30, "dp_count_start": "draw"}}
    s.rng = np.random.default_rng(seed)
    return s


def test_c1782_counts_given_held_rows_follow_the_conditional_law():
    """Prop. 1782: with base counts b, the new counts N follow prod f(b_j + N_j)
    given sum N = R (exact enumeration with a Poisson base, no observed counts)."""
    from itertools import product
    from scipy.special import gammaln
    M, R, b = 3, 4, np.array([0, 3, 1])
    lam = R / M
    exact = {}
    for N in product(range(R + 1), repeat=M):
        if sum(N) == R:
            x = b + np.array(N)
            exact[N] = float(np.exp((x * np.log(lam) - lam - gammaln(x + 1)).sum()))
    z = sum(exact.values())
    p0 = np.zeros(R + 1)
    for N, w in exact.items():
        p0[N[0]] += w / z
    s = _counts_syn(0)
    draws = np.array([s._conditional_counts(R, M, np.zeros(0, np.int64), 1.0, base=b) for _ in range(4000)])
    assert (draws.sum(1) == R).all()
    got = np.bincount(draws[:, 0], minlength=R + 1) / len(draws)
    assert 0.5 * np.abs(got - p0).sum() < 0.03, (got, p0)
    # without a base, the start from i.i.d. draws reaches counts the balanced start cannot
    f_obs = np.array([1] * 40 + [2] * 15 + [27])
    big = sum(int((_counts_syn(i)._conditional_counts(80, 56, f_obs, 0.5, sweeps=1) >= 20).any()) for i in range(60))
    assert big > 0


# --------------------------------------------------------------------- C1783
def test_c1783_segment_totals_follow_the_count_law():
    """Prop. 1783: with phi consistent with the count law (the fitted case), the
    totals keep their mean M_t mu_t and their spread falls from the multinomial
    sd towards the count law's sqrt(M_t sigma_t^2) (precisions add)."""
    from relsynth.engine import _balance_totals
    rng = np.random.default_rng(3)
    K, M = 4, np.array([200, 200, 200, 200])
    obs = [np.minimum(rng.poisson(m, 400), 1 + 2 * int(m)) for m in (1.0, 2.0, 3.0, 5.0)]   # under-dispersed
    dp = {"counts": obs}
    mt = np.array([M[t] * obs[t].mean() for t in range(K)])
    n = int(mt.sum())
    seg_p = (mt / mt.sum())[None, :]
    var_law = M * np.array([o.var() for o in obs])
    before, after = [], []
    for _ in range(60):
        seg_of = rng.choice(K, size=n, p=seg_p[0])
        before.append(np.bincount(seg_of, minlength=K))
        after.append(np.bincount(_balance_totals(seg_of, np.zeros(n, np.int64), seg_p, M, dp, None, rng, sweeps=6),
                                 minlength=K))
    before, after = np.array(before, float), np.array(after, float)
    assert np.all(np.abs(after.mean(0) - mt) < 3 * np.sqrt(var_law / 60) + 2)
    target = 1 / np.sqrt(1 / before.var(0) + 1 / var_law)          # sd when the two precisions add
    assert np.all(after.std(0) < before.std(0)) and np.all(after.std(0) < 1.6 * target), (after.std(0), target)


# --------------------------------------------------------------------- C1784
def _bf_model(K):
    from types import SimpleNamespace

    from relsynth.timeseries.events import SequenceModel
    m = SequenceModel.__new__(SequenceModel)
    m.calendar, m.K, m.datetime, m.fit_log = "auto", K, False, {}
    m.intensity = SimpleNamespace(bucket="month")
    return m


def test_c1784_calendar_is_pooled_unless_segments_differ():
    """Prop. 1784: one calendar for segments drawn from one seasonal profile,
    one per segment when the profiles differ."""
    rng = np.random.default_rng(4)
    n, days = 6000, 730.0
    seg = rng.integers(0, 3, n)

    def times(peak):
        d = rng.uniform(0, days, 4 * n)
        w = 1 + 0.8 * np.cos(2 * np.pi * (d - peak) / 365)
        return rng.choice(d, size=n, p=w / w.sum())
    t_same = times(0.0)
    assert _bf_model(3)._pool_calendar(t_same, seg, 0.0, days) is True
    t_diff = np.where(seg == 0, times(0.0), times(180.0))
    m = _bf_model(3)
    assert m._pool_calendar(t_diff, seg, 0.0, days) is False
    assert m.fit_log["calendar"]["log_bf_segment"] > np.log(100)


# --------------------------------------------------------------------- C1785
def test_c1785_multihot_sets_round_trip_and_never_repeat():
    """Prop. 1785: exact round trip, no repeated element, length as modelled;
    pairwise co-occurrence of values follows the source."""
    from relsynth.documents import shred as sh
    from relsynth.documents.infer import profile_collection
    rng = np.random.default_rng(5)
    vocab = ["a", "b", "c", "d", "e"]
    docs = []
    for i in range(2000):
        k = int(rng.integers(1, 4))
        first = rng.choice(vocab)
        rest = [v for v in vocab if v != first]
        extra = list(rng.choice(rest, size=k - 1, replace=False)) if k > 1 else []
        if first == "a" and "b" not in extra and k > 1:
            extra[0] = "b"                                   # a and b go together
        docs.append({"_id": i, "tags": [str(first)] + [str(x) for x in extra]})
    lay = sh.build_layout({"c": profile_collection("c", docs)}, [])
    frames = sh.shred(lay, {"c": docs})
    assert sh.nest(lay, frames, keep_ids=True)["c"] == docs
    f = {k: v.copy() for k, v in frames.items()}
    f["c"]["tags__n"] = rng.integers(1, 4, len(docs))
    out = sh.nest(lay, f, keep_ids=True)["c"]
    assert all(len(set(d["tags"])) == len(d["tags"]) == n for d, n in zip(out, f["c"]["tags__n"]))


# --------------------------------------------------------------------- C1786
def test_c1786_category_exclusion_through_the_parent_key():
    """Prop. 1786: a never-referenced category spread over subcategories that
    each fall under the threshold is excluded; a referenced one is not."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
    from test_g4_fidelity import _inclusion_syn
    s, tables = _inclusion_syn()
    s._fit_inclusion(tables)
    rules = s.cind[("sales", "pk")]
    assert ("sk\x1fsub\x1fcat", "C", False) in rules
    assert not any(r[0] == "sk\x1fsub\x1fcat" and r[1] in ("A", "B") for r in rules)
    D, n = tables["sales"].pk.nunique(), len(tables["prod"])
    assert D * 10 / n < 10 <= D * 100 / n                 # a subcategory alone is under the threshold


# --------------------------------------------------------------------- C1787
def test_c1787_floor_keeps_a_stock_non_negative_and_exact_otherwise():
    """Prop. 1787: with a floor the balance is never negative and equals the
    running sum where that stays non-negative; without, a walk goes negative."""
    from relsynth.engine import cumulative_apply
    rng = np.random.default_rng(7)
    n = 4000
    df = pd.DataFrame({"p": np.repeat(np.arange(40), n // 40), "d": np.tile(np.arange(n // 40), 40),
                       "i": rng.poisson(5, n), "o": rng.poisson(5, n), "y": 0.0})
    rule = {"column": "y", "by": "p", "order": "d", "step": "i - o", "opening": 20.0}
    free = cumulative_apply(df.copy(), dict(rule))
    capped = cumulative_apply(df.copy(), {**rule, "floor": 0.0})
    assert (free.y < 0).mean() > 0.1 and (capped.y >= 0).all()
    ok = free.groupby("p").y.transform(lambda s: (s >= 0).cummin()).astype(bool)
    assert np.allclose(free.y[ok], capped.y[ok])

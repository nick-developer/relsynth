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

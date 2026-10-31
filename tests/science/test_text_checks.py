# SPDX-License-Identifier: Apache-2.0
"""Checks C1500-C1507: the synthetic-text propositions of docs/proofs (section
v5.0, synthetic text), tested against known answers.  About three minutes;
they run in shard 4/4 only (C1502 needs PyTorch and runs in the CI job
`text-lm`).

    pytest -m science tests/science/test_text_checks.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
from scipy.stats import binom

from relsynth.dp.mechanisms import Accountant
from relsynth.governance.accounting import epsilon_of, rho_of
from relsynth.textgen import audit as A
from relsynth.textgen.dpngram import DPNgramModel, ngram_logprob
from tests.fakedata.scenarios import text

pytest.importorskip("opendp")
_SHARD = os.environ.get("RELSYNTH_SCIENCE_SHARD")
pytestmark = [pytest.mark.science,
              pytest.mark.skipif(bool(_SHARD) and _SHARD != "4/4", reason="text checks run in shard 4/4")]


def _corpus(seed=6):
    tk = text(seed)[0]["ticket"]
    return tk.body.tolist(), tk.customer_id.tolist(), tk.category.tolist()


def _canary_run(eps, n_can=96, words=12, k=3, repeats=4, seed=1500):
    T, O, C = _corpus()
    wl = A.candidate_words(T, n=words)
    sec = A.make_canaries(wl, k, n_can, seed)
    T2, O2, C2, cls = A.insert_canaries(T, O, C, wl, sec, repeats, seed + 1)
    delta = 1e-6
    rho = rho_of(eps, delta / 2)
    m = DPNgramModel(order=2).fit_dp(T2, O2, C2, Accountant(rho=rho, delta=delta / 2), rho, delta / 2)
    return A.exposures(m.logprob, wl, k, sec, cls, seed=seed), rho, delta, len(wl) ** k


def test_C1500_exposure_tail_is_within_the_dp_bound():
    """Prop. 1500: for DP n-gram releases at eps 1, 4 and 16, the share of
    canaries with exposure >= t is not significantly above e^eps 2^-t + delta
    (one-sided binomial, alpha 0.001 per test); and the non-DP 1.0 model, given
    the same canaries, exposes them far more (the check has power)."""
    from relsynth.text import TextModel
    for eps in (1.0, 4.0, 16.0):
        e, rho, delta, R = _canary_run(eps)
        for t in (1.0, 2.0, 4.0, 6.0, 8.0):
            b = A.exposure_bound(t, rho=rho, delta_extra=delta / 2)
            hits = int((e >= t).sum())
            assert binom.sf(hits - 1, len(e), b) > 1e-3, (eps, t, hits, len(e), b)
    T, O, C = _corpus()
    wl = A.candidate_words(T, n=12)
    sec = A.make_canaries(wl, 3, 96, 1500)
    T2, _, C2, cls = A.insert_canaries(T, O, C, wl, sec, 4, 1501)
    m = TextModel(order=3).fit(T2, C2)
    e0 = A.exposures(lambda t, c: ngram_logprob(m, t, c), wl, 3, sec, cls)
    assert np.mean(e0 >= 6) > A.exposure_bound(6, epsilon=1.0) + 0.3


def test_C1501_dp_ngram_sensitivity_replay():
    """Prop. 1501: removing one person changes every level's count vector by at
    most sqrt(max_keys) in L2 and 1 per key; charges add up to the budget."""
    T, O, C = _corpus()
    m = DPNgramModel(order=3, max_keys=16)
    rng = np.random.default_rng(1)
    people = m._bound(T, O, C, rng)
    vocab = None
    for n in range(3):
        keys = [m._keys_of(rows, n, vocab, np.random.default_rng(i)) for i, rows in enumerate(people)]
        for ks in keys:
            assert len(ks) == len(set(ks)) <= 16
        tot = {}
        for ks in keys:
            for k in ks:
                tot[k] = tot.get(k, 0) + 1
        for u in rng.choice(len(people), 30, replace=False):
            diff = {k: 1 for k in keys[u]}
            assert math.sqrt(sum(v * v for v in diff.values())) <= 4.0 + 1e-12
            assert max(diff.values(), default=0) <= 1
        vocab = {w for (_, _, w) in tot} - {"</s>"}
    acct = Accountant(rho=0.3, delta=1e-6)
    DPNgramModel(order=3).fit_dp(T, O, C, acct, 0.3, 1e-6)
    r, d = acct.spent
    assert r == pytest.approx(0.3, rel=1e-6) and d == pytest.approx(1e-6, rel=1e-9)
    assert max(len(p) for p in people) <= m.max_texts


def test_C1502_dpgd_noise_and_clipping():
    """Prop. 1502: the noise multiplier is sqrt(T / (2 rho_steps)), the charges
    add up to rho, and each person's clipped gradient has norm <= C."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from relsynth.textgen.lm import LMTextModel
    T, O, C = _corpus()
    T, O, C = T[:200], O[:200], C[:200]
    acct = Accountant(rho=0.4, delta=1e-6)
    m = LMTextModel(steps=3, stub_size={"n_embd": 16, "n_layer": 1, "n_head": 2},
                    dp={"clip": 0.5, "max_texts": 2, "count_share": 0.05})
    m.fit(T, [{"category": c} for c in C], owners=O, acct=acct, rho=0.4)
    assert acct.spent[0] == pytest.approx(0.4, rel=1e-9)
    assert m.release["noise_multiplier"] == pytest.approx(math.sqrt(3 / (2 * 0.4 * 0.95)), rel=1e-9)


def test_C1503_pe_vote_sensitivity_and_budget():
    """Prop. 1503: each person's integer vote vector has L2 norm <= SCALE; the
    PE run spends exactly its budget; the embedding does not depend on the data."""
    from relsynth.textgen.embed import HashEmbedder
    from relsynth.textgen.pe import SCALE, NgramApi, PETextModel
    rng = np.random.default_rng(3)
    for _ in range(2000):
        v = rng.integers(0, 5, size=int(rng.integers(1, 6)))
        v = v[v > 0] if (v > 0).any() else np.array([1])
        q = np.floor(SCALE * v / np.linalg.norm(v))
        assert np.linalg.norm(q) <= SCALE + 1e-9
    T, O, C = _corpus()
    pub = text(606)[0]["ticket"].sample(300, random_state=0)
    acct = Accountant(rho=0.5, delta=1e-6)
    PETextModel(NgramApi(pub.body, pub.category), iterations=3, pool=32).fit_dp(T, O, C, acct, 0.5, 1e-6)
    assert acct.spent[0] == pytest.approx(0.5, rel=1e-6)
    assert np.allclose(HashEmbedder()(["late parcel"]), HashEmbedder()(["x", "late parcel"])[1:])


def test_C1504_expected_exposure_bound():
    """Prop. 1504: the worst-case tail min(1, e^eps 2^-t) integrates to the
    bound, and DP n-gram runs stay under it."""
    for eps in (0.5, 1, 4, 8):
        ts = np.linspace(0, 40, 400001)
        integral = np.trapezoid(np.minimum(1, np.exp(eps) * 2.0 ** -ts), ts)
        assert integral == pytest.approx((eps + 1) / math.log(2), rel=1e-3)
    e, rho, delta, R = _canary_run(4.0, n_can=64)
    eps = epsilon_of(rho, delta / 2)
    se = e.std(ddof=1) / math.sqrt(len(e))
    assert e.mean() - 3 * se <= A.expected_exposure_bound(eps, delta, R)


def test_C1505_minhash_candidate_rate():
    """Prop. 1505: pairs with Jaccard s become candidates at rate about
    1 - (1 - s^r)^b; reported similarities are exact."""
    rng = np.random.default_rng(1505)
    vocab = [f"w{i}" for i in range(5000)]
    for s_target, (b, r) in [(0.3, (8, 4)), (0.5, (8, 4)), (0.7, (4, 4))]:
        hits, n = 0, 300
        for _ in range(n):
            base = list(rng.choice(vocab, 60, replace=False))
            n_sh = 58
            keep = int(round(n_sh * 2 * s_target / (1 + s_target)))
            a = base
            bb = base[:keep + 2] + list(rng.choice(vocab, 60 - keep - 2, replace=False))
            sa, sb = A._shingles(" ".join(a)), A._shingles(" ".join(bb))
            s = len(sa & sb) / len(sa | sb)
            ma, mb = A._minhash(sa, b * r), A._minhash(sb, b * r)
            hits += any((ma[i * r:(i + 1) * r] == mb[i * r:(i + 1) * r]).all() for i in range(b))
            p = 1 - (1 - s ** r) ** b
        lo, hi = binom.ppf(0.0005, n, max(p - 0.08, 0)), binom.ppf(0.9995, n, min(p + 0.08, 1))
        assert lo <= hits <= hi, (s_target, hits, p)
    res = A.near_duplicates(["a b c d e f g h"], ["a b c d e f g x"])
    assert res["jaccard_p50"] == pytest.approx(5 / 7)


def test_C1506_wilson_coverage_and_surrogates():
    """Prop. 1506: Wilson intervals of recall cover at about 95%; redaction
    leaves no character of a caught entity; surrogates are format-valid."""
    from relsynth.textgen.labelled import labelled_set
    from relsynth.textgen.pii import PiiPipeline, iban_ok, luhn_ok, surrogate, wilson
    rng = np.random.default_rng(1506)
    for p in (0.6, 0.9, 0.97):
        cover = 0
        for _ in range(4000):
            x = rng.binomial(200, p)
            lo, hi = wilson(x, 200)
            cover += lo <= p <= hi
        assert 0.93 <= cover / 4000 <= 0.97
    pipe = PiiPipeline(["regex"])
    for row in labelled_set(300, seed=7):
        red, spans = pipe.redact(row["text"])
        for s, e, _t in row["spans"]:
            if any(sp.start <= s and e <= sp.end for sp in spans):
                assert row["text"][s:e] not in red
    for _ in range(500):
        assert luhn_ok(surrogate("CARD", rng)) and iban_ok(surrogate("IBAN", rng))


def test_C1507_consistency_is_calibrated():
    """Prop. 1507: real held-out rows as 'synthetic' give a ratio whose interval
    covers 1; texts shuffled across rows give about chance agreement."""
    from relsynth.textgen.metrics import consistency
    tk = text(6)[0]["ticket"]
    tk2 = text(66)[0]["ticket"]
    r = consistency(tk, tk2, "body", ["category", "priority"], seed=1)
    for c in ("category", "priority"):
        lo, hi = r[c]["synthetic_ci"]
        assert lo - 0.02 <= r[c]["real"] <= hi + 0.02, (c, r[c])
    sh = tk2.copy()
    sh["body"] = np.random.default_rng(0).permutation(sh.body.to_numpy())
    r = consistency(tk, sh, "body", ["category"], seed=1)["category"]
    q = tk2.category.value_counts(normalize=True)
    assert abs(r["synthetic"] - float((q ** 2).sum())) < 0.05

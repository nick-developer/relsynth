# SPDX-License-Identifier: Apache-2.0
"""Unit tests of relsynth.textgen (roadmap T14): PII layer, DP n-gram, audits,
metrics, spec wiring.  The language-model tests run when PyTorch is installed
(CI job `text-lm`)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from relsynth.dp.mechanisms import Accountant
from relsynth.textgen import TextColumn, is_extended
from relsynth.textgen import audit as A
from relsynth.textgen.dpngram import DPNgramModel, ngram_logprob
from relsynth.textgen.embed import HashEmbedder
from relsynth.textgen.labelled import labelled_set
from relsynth.textgen.metrics import consistency, mauve
from relsynth.textgen.pii import PiiPipeline, evaluate, fill, iban_ok, luhn_ok, surrogate
from tests.fakedata.scenarios import text

pytest.importorskip("opendp")


@pytest.fixture(scope="module")
def s6():
    tabs, spec, truth = text(6)
    return tabs, spec, truth


# ------------------------------------------------------------------ PII
def test_checksums_and_surrogates_are_valid():
    assert luhn_ok("4111 1111 1111 1111") and not luhn_ok("4111 1111 1111 1112")
    assert iban_ok("GB82 WEST 1234 5698 7654 32") and not iban_ok("GB83 WEST 1234 5698 7654 32")
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert luhn_ok(surrogate("CARD", rng))
        assert iban_ok(surrogate("IBAN", rng))
        assert surrogate("EMAIL", rng).endswith("@example.com")
        assert surrogate("IP", rng).split(".")[0] in ("192", "198", "203")


def test_redact_and_fill_round_trip():
    p = PiiPipeline(["regex"])
    red, spans = p.redact("Regards, Marta Lindqvist. Mail marta@corp.example.org, IBAN GB82 WEST 1234 5698 7654 32")
    assert "<PERSON>" in red and "<EMAIL>" in red and "<IBAN>" in red
    assert "Marta" not in red and "marta@" not in red
    out = fill(red, np.random.default_rng(1))
    assert "<" not in out


def test_labelled_offsets_and_regex_recall_floor():
    lab = labelled_set(400, seed=3)
    for row in lab:
        for s, e, t in row["spans"]:
            assert 0 <= s < e <= len(row["text"])
    r = evaluate(PiiPipeline(["regex"]), lab)
    # structured formats are matched by construction; names need context cues (see docs/text.md)
    for t in ("EMAIL", "IBAN", "CARD", "IP", "URL"):
        assert r["types"][t]["recall"] == 1.0
    assert r["types"]["PERSON"]["recall"] > 0.6
    assert r["overall"]["precision"] > 0.95


# ------------------------------------------------------------------ DP n-gram
def test_dp_ngram_charges_stay_in_budget(s6):
    tk = s6[0]["ticket"]
    acct = Accountant(rho=0.2, delta=1e-6)
    m = DPNgramModel(order=2).fit_dp(tk.body.tolist(), tk.customer_id.tolist(), tk.category.tolist(), acct, 0.2, 1e-6)
    r, d = acct.spent
    assert r <= 0.2 * (1 + 1e-9) and d <= 1e-6 * (1 + 1e-9)
    assert len(m.vocab) > 10
    out = m.generate(tk.category.tolist()[:50], np.random.default_rng(0))
    assert sum(bool(isinstance(x, str) and x) for x in out) > 30
    assert m.release["level1"]["released"] <= m.release["level1"]["candidates"]


def test_dp_ngram_drops_one_persons_words():
    # a word used by one person only never passes the selection threshold
    texts = ["the parcel is late"] * 300 + ["zyxwv secret word here"] * 4
    owners = list(range(300)) + [999] * 4
    m = DPNgramModel(order=2).fit_dp(texts, owners, None, Accountant(rho=1.0, delta=1e-6), 1.0, 1e-6)
    assert "zyxwv" not in m.vocab and "parcel" in m.vocab


# ------------------------------------------------------------------ audits
def test_exposure_uniform_without_signal_and_high_when_memorised(s6):
    from relsynth.text import TextModel
    tk = s6[0]["ticket"]
    words = A.candidate_words(tk.body, n=10)
    sec = A.make_canaries(words, 3, 20, seed=1)
    T, O, C, cls = A.insert_canaries(tk.body.tolist(), tk.customer_id.tolist(), tk.category.tolist(), words, sec, 4, 2)
    m = TextModel(order=3).fit(T, C)
    e = A.exposures(lambda t, c: ngram_logprob(m, t, c), words, 3, sec, cls)
    assert e.mean() > 7                      # the 1.0 guard counts rows, not people
    flat = A.exposures(lambda t, c: 0.0, words, 3, sec, cls, seed=5)
    assert flat.mean() < 3                   # ties: rank uniform, E[exposure] ~ 1.44 bits


def test_exposure_bound_monotone_and_valid():
    b = [A.exposure_bound(t, rho=0.5) for t in (1, 2, 4, 8)]
    assert all(x >= y for x, y in zip(b, b[1:]))
    assert A.exposure_bound(3, epsilon=0.0, delta=0.0) == pytest.approx(2 ** -3)
    assert A.expected_exposure_bound(1.0, 0.0, 4096) == pytest.approx(2 / math.log(2))


def test_near_duplicates_finds_copies_and_edits():
    tail = "the courier said it was delivered to the front door but nobody in the building has seen the box since then"
    train = [f"order {i} has not arrived after many days please help me {tail}" for i in range(200)]
    syn = [train[3], train[7].replace("please", "kindly"), "a completely new sentence about billing"]
    r = A.near_duplicates(train, syn)
    assert r["exact_copies"] == 1 and r["near_duplicates"] >= 2


# ------------------------------------------------------------------ metrics
def test_mauve_orders_samples():
    tabs, _, _ = text(66)
    real = text(6)[0]["ticket"].body.tolist()
    same = tabs["ticket"].body.tolist()
    other = [f"lorem ipsum dolor {i}" for i in range(len(same))]
    assert mauve(real, same) > 0.7 > 0.1 > mauve(real, other)


def test_consistency_detects_shuffled_text(s6):
    tk = s6[0]["ticket"]
    shuf = tk.copy()
    shuf["body"] = np.random.default_rng(0).permutation(tk.body.to_numpy())
    good = consistency(tk, tk.sample(frac=1.0, random_state=1), "body", ["category"])["category"]["ratio"]
    bad = consistency(tk, shuf, "body", ["category"])["category"]["ratio"]
    assert good > 0.95 and bad < 0.5


def test_hash_embedding_is_data_independent():
    a = HashEmbedder()(["the parcel is late"])
    b = HashEmbedder()(["something else", "the parcel is late"])[1:]
    assert np.allclose(a, b)


# ------------------------------------------------------------------ spec and engine
def test_extended_detection_keeps_10_path():
    assert not is_extended({"condition_on": "category", "order": 3})
    assert is_extended({"condition_on": ["category", "priority"]})
    assert is_extended({"method": "ngram"})


def test_spec_validation_of_text_options(s6):
    from relsynth.api import Spec
    _, spec, _ = s6
    sp = copy.deepcopy(spec)
    sp["tables"]["ticket"]["columns"]["body"]["text"] = {"condition_on": ["category", "nope"], "method": "pe"}
    errs = "\n".join(Spec.from_dict(sp).errors())
    assert "nope" in errs and "pe needs text.dp" in errs


def test_engine_runs_extended_text(s6):
    from relsynth.pipeline import execute
    tabs, spec, truth = s6
    sp = copy.deepcopy(spec)
    sp["tables"]["ticket"]["columns"]["body"]["text"] = {
        "condition_on": ["category", "priority"], "pii": {"recognizers": ["regex"]},
        "method": "ngram", "dp": {"epsilon": 4.0, "delta": 1e-6, "unit_column": "customer_id"}}
    run = execute(sp, tables=copy.deepcopy(tabs), write=False, privacy=False, seed=1)
    assert not run.exceptions, run.exceptions
    body = run.data["ticket"]["body"]
    assert body.notna().mean() > 0.8
    assert not any(n in str(b) for b in body for n in truth["rare_names"])
    info = run.synthesizer.log["decisions"]["ticket"]["body text"]
    assert info["dp"]["epsilon"] <= 4.0 + 1e-6 and "not DP" in info["dp"]["scope"]


def test_text_column_refuses_pe_without_dp():
    from relsynth.spec import SpecError
    with pytest.raises(SpecError):
        TextColumn("t", "c", {"method": "pe"})


def test_audit_of_dp_configuration_is_consistent(s6):
    tk = s6[0]["ticket"]
    res = A.run_audit(tk, "body", {"condition_on": "category", "method": "ngram",
                                   "dp": {"epsilon": 2.0, "delta": 1e-6, "unit_column": "customer_id"}},
                      canaries=32, repeats=4, words=10, samples=300)
    assert res["exposure"]["consistent"]
    assert res["canaries_generated"] == 0


# ------------------------------------------------------------------ language model (torch)
def test_lm_stub_fits_generates_and_scores(s6):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from relsynth.textgen.lm import LMTextModel
    tk = s6[0]["ticket"].iloc[:300]
    conds = [{"category": c} for c in tk.category]
    m = LMTextModel(steps=30, batch_size=16, lr=3e-3, stub_size={"n_embd": 32, "n_layer": 1, "n_head": 2},
                    max_new_tokens=40).fit(tk.body.tolist(), conds)
    out = m.generate(conds[:5], np.random.default_rng(0))
    assert len(out) == 5
    assert math.isfinite(m.logprob(tk.body.iloc[0], conds[0]))
    acct = Accountant(rho=0.5, delta=1e-6)
    d = LMTextModel(steps=2, stub_size={"n_embd": 32, "n_layer": 1, "n_head": 2},
                    dp={"clip": 1.0, "max_texts": 2}).fit(tk.body.tolist(), conds, owners=tk.customer_id.tolist(),
                                                          acct=acct, rho=0.5)
    assert acct.spent[0] == pytest.approx(0.5)
    assert d.release["noise_multiplier"] == pytest.approx(math.sqrt(2 / (2 * 0.5 * 0.98)))



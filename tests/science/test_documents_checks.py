# SPDX-License-Identifier: Apache-2.0
"""Checks C1400-C1405: the document propositions of docs/proofs (section v4.0,
document databases), on documents drawn from known generating models.  About
two minutes on 4 cores.

    pytest -m science tests/science/test_documents_checks.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from relsynth.documents import DocumentSynthesizer, dynamo
from relsynth.documents.conform import conform
from relsynth.documents.infer import find_references, profile_collection
from relsynth.documents.metrics import shape_checks
from relsynth.documents.shred import build_layout, nest, shred
from relsynth.documents.validate import validate_collection
from relsynth.documents.values import to_extjson_value

pytestmark = [pytest.mark.science]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fakedata"))
sys.path.insert(0, str(HERE.parents[1] / "examples" / "documents"))
import documents as fake  # noqa: E402


def canon(d):
    return json.dumps(to_extjson_value(d), sort_keys=True)


def random_documents(rng: random.Random, n: int) -> list[dict]:
    """Documents with random shapes: optional fields, unions, nested arrays,
    maps, variants and length fields, so that every branch of shredding runs."""
    out = []
    for i in range(n):
        d = {"_id": i, "k": rng.choice(["a", "b", "c"])}
        if rng.random() < 0.7:
            d["opt"] = rng.choice([1, 2.5, "x", None, True])
        arr = [[rng.randint(0, 9) for _ in range(rng.randint(0, 3))] for _ in range(rng.randint(0, 4))]
        d["nested"] = arr
        d["n"] = len(arr)
        if rng.random() < 0.5:
            d["sub"] = {"x": rng.random(), "deep": {"y": [rng.choice(["p", "q"]) for _ in range(rng.randint(1, 3))]}}
        kind = rng.choice(["A", "B"])
        d["v"] = {"kind": kind, **({"a1": rng.randint(0, 5)} if kind == "A" else {"b1": "z", "b2": rng.random()})}
        d["m"] = {f"{rng.getrandbits(40):010x}": {"w": rng.randint(0, 3)} for _ in range(rng.randint(0, 2))}
        out.append(d)
    return out


def _keys_as_positions(d):
    return {**d, "m": [v for v in d["m"].values()]}


def test_c1400_inferred_schema_admits_every_source_document():
    """Prop. 1400 on 20 random document sets and the fake shop."""
    for s in range(20):
        docs = random_documents(random.Random(s), 300)
        prof = profile_collection("c", docs)
        assert validate_collection(prof, docs)["share"] == 1.0, s
        assert prof.root.fields["n"].length_of == "nested"
        assert prof.root.fields["v"].variants["discriminator"] == "kind"
        assert prof.root.fields["m"].is_map
    for c, docs in fake.collections(400, seed=9).items():
        assert validate_collection(profile_collection(c, docs), docs)["share"] == 1.0


def test_c1401_shredding_round_trip_is_exact():
    """Prop. 1401: nest(shred(D)) = D, document by document, on 20 random sets."""
    for s in range(20):
        docs = random_documents(random.Random(100 + s), 300)
        profs = {"c": profile_collection("c", docs)}
        lay = build_layout(profs, [])
        back = nest(lay, shred(lay, {"c": docs}), keep_ids=True)["c"]
        # map keys that are generated identifiers are fresh, like row ids: compare entries in order
        assert [canon(_keys_as_positions(d)) for d in back] == [canon(_keys_as_positions(d)) for d in docs], s


def test_c1402_conform_output_validates_and_repairs_are_rare():
    """Prop. 1402: conform's output validates whatever it is given; on the
    engine's documents the repair rate is small (reported per kind)."""
    rng = random.Random(5)
    docs = random_documents(rng, 400)
    prof = profile_collection("c", docs)
    broken = []
    for d in docs:
        d = json.loads(json.dumps(d))
        r = rng.random()
        if r < 0.2:
            d.pop("k")
        elif r < 0.4:
            d["nested"] = [[1]] * 9
        elif r < 0.5:
            d["v"]["b1"] = "wrong variant field"
        broken.append(d)
    fixed, rep = conform(prof, broken, seed=1)
    assert validate_collection(prof, fixed)["share"] == 1.0 and rep["documents_repaired"] > 100
    ds = DocumentSynthesizer(fake.collections(300, seed=11), seed=2)
    out = ds.synthesize()
    for c, docs in out.items():
        assert validate_collection(ds.profiles[c], docs)["share"] == 1.0
        assert ds.repairs[c]["documents_repaired"] <= 0.5 * len(docs), (c, ds.repairs[c])


def test_c1403_shape_limits_have_their_level():
    """Prop. 1403: two independent draws of the same document law pass a shape
    metric with probability at least 99% (binomial test over 20 pairs x 7 metrics)."""
    fails = total = 0
    for s in range(20):
        a = random_documents(random.Random(1000 + s), 400)
        b = random_documents(random.Random(2000 + s), 400)
        for m in shape_checks(a, b, "c", splits=10, seed=s):
            if m["pass"] is not None:
                total += 1
                fails += not m["pass"]
    # one-sided: more failures than a 1% level allows (p < 0.001) fails the check
    assert binom.sf(fails - 1, total, 0.01) > 1e-3, (fails, total)


def test_c1404_single_table_keys_rebuild_exactly():
    """Prop. 1404: templated keys are rebuilt exactly and stay unique in the
    synthetic table; every reference between item types resolves."""
    import dynamodb_shop as ex
    for seed in (1, 2, 3):
        items = ex.items(150, seed=seed)
        cols, design = dynamo.split(items, indexes=[{"name": "GSI1", "HASH": "GSI1PK", "RANGE": "GSI1SK"}])
        back = dynamo.merge(cols, design)
        assert sorted(map(canon, back)) == sorted(map(canon, items))
        ds = DocumentSynthesizer(cols, seed=seed)
        syn = ds.synthesize()
        rep = ds.report(syn, splits=4)
        assert all(r["share"] == 1.0 for r in rep["references"])
        assert dynamo.check(dynamo.merge(syn, design), design)["pass"]


def test_c1405_references_resolve_and_length_laws_follow_the_source():
    """Prop. 1405: references resolve on every synthetic document; the array
    length law is within its calibrated limit and the length field is exact."""
    cols = fake.collections(400, seed=21)
    ds = DocumentSynthesizer(cols, seed=4)
    out = ds.synthesize()
    rep = ds.report(out, splits=10)
    assert all(r["share"] == 1.0 for r in rep["references"])
    ids = {d["_id"] for d in out["customers"]}
    assert all(o["customer_id"] in ids for o in out["orders"])
    assert all(o["n_items"] == len(o["items"]) for o in out["orders"])
    lens = {c: next(m for m in v["shape"] if m["metric"] == "array_length") for c, v in rep["collections"].items()}
    assert all(m["pass"] for m in lens.values()), lens
    lo, hi = ds.profiles["orders"].root.fields["items"].length_bounds()
    assert all(lo <= len(o["items"]) <= hi for o in out["orders"])
    # the share of documents with an optional sub-document is that of the source
    p_real = np.mean(["shipping" in o for o in cols["orders"]])
    p_syn = np.mean(["shipping" in o for o in out["orders"]])
    n = len(out["orders"])
    assert abs(p_real - p_syn) < 4 * np.sqrt(p_real * (1 - p_real) * (1 / n + 1 / len(cols["orders"]))) + 0.02

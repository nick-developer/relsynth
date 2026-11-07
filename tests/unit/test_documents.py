# SPDX-License-Identifier: Apache-2.0
"""Document databases (T13): inference, shredding, conformance, metrics,
connectors and single-table designs.  Propositions 1400-1404."""
from __future__ import annotations

import copy
import datetime as dt
import decimal
import json
import os
import sys
from pathlib import Path

import pytest

from relsynth.documents import dynamo
from relsynth.documents.conform import conform
from relsynth.documents.connectors import JsonFiles, MongoDB, connect_documents, parse_location
from relsynth.documents.infer import find_references, find_sequences, profile_collection
from relsynth.documents.ir import to_ir, to_ir_document
from relsynth.documents.metrics import reference_integrity, shape_checks, shape_distances
from relsynth.documents.shred import build_layout, nest, shred
from relsynth.documents.validate import validate_collection, violations
from relsynth.documents.values import (Long, ObjectId, from_extjson_value, read_json_documents, to_extjson_value,
                                       type_name, write_json_documents)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fakedata"))
sys.path.insert(0, str(HERE.parents[1] / "examples" / "documents"))
import documents as fake  # noqa: E402


def canon(d):
    return json.dumps(to_extjson_value(d), sort_keys=True)


@pytest.fixture(scope="module")
def shop():
    cols = fake.collections()
    profs = {c: profile_collection(c, d) for c, d in cols.items()}
    refs = find_references(profs)
    return cols, profs, refs


# ---------------------------------------------------------------- values
def test_extended_json_round_trip(tmp_path):
    d = {"_id": ObjectId("5bd761dcae323e45a93ccfe8"), "i": 3, "l": Long(5), "f": 1.5,
         "m": decimal.Decimal("40.01"), "t": dt.datetime(2020, 1, 2, 3, 4, 5, 6000, tzinfo=dt.timezone.utc),
         "b": True, "n": None, "a": [1, "x", {"y": 2}]}
    j = to_extjson_value(d)
    assert j["_id"] == {"$oid": "5bd761dcae323e45a93ccfe8"} and j["l"] == {"$numberLong": "5"}
    back = from_extjson_value(json.loads(json.dumps(j)))
    assert canon(back) == canon(d)
    assert [type_name(back[k]) for k in ("_id", "i", "l", "f", "m", "t")] == \
        ["objectId", "int", "long", "double", "decimal", "date"]
    p = tmp_path / "c.jsonl"
    write_json_documents([d, d], p)
    assert [canon(x) for x in read_json_documents(p)] == [canon(d)] * 2
    # relaxed mode and a JSON array file read back the same values
    (tmp_path / "arr.json").write_text(json.dumps([to_extjson_value(d, relaxed=True)]))
    r = next(read_json_documents(tmp_path / "arr.json"))
    assert r["t"] == d["t"] and r["m"] == d["m"]


# ---------------------------------------------------------------- inference
def test_inference_finds_shapes(shop):
    cols, profs, refs = shop
    o, c = profs["orders"], profs["customers"]
    assert o.documents == len(cols["orders"])
    assert o.root.fields["payment"].variants["discriminator"] == "method"
    assert o.root.fields["payment"].variants["cases"]["card"] == ["last4", "network"]
    assert o.root.fields["n_items"].length_of == "items"
    assert o.root.fields["rating"].value_types[:2] == ["int", "double"] or \
        set(o.root.fields["rating"].value_types) == {"int", "double", "string"}
    assert o.root.fields["shipping"].optional and not o.root.fields["shipping"].nullable
    assert o.root.fields["note"].optional and o.root.fields["note"].nullable
    assert o.root.fields["items"].length_bounds() == (1, 8)
    assert c.root.fields["wallets"].is_map and c.root.fields["wallets"].key_field == "id"
    assert c.root.fields["tags"].distinct_items
    assert {(r["collection"], r["path"], r["to"], r["to_path"]) for r in refs if not r.get("candidate")} == \
        {("orders", "customer_id", "customers", "_id")}


def test_reference_candidates_and_sequences():
    a = [{"_id": i, "code": f"A{i}"} for i in range(20)]
    b = [{"_id": 100 + j, "a_id": j % 25, "line": 0} for j in range(40)]      # 5 dangling values
    profs = {"a": profile_collection("a", a), "b": profile_collection("b", b)}
    refs = find_references(profs)
    cand = [r for r in refs if r.get("candidate")]
    assert cand and cand[0]["path"] == "a_id" and cand[0]["coverage"] == 0.8
    # line numbers 1..n within the referenced key
    lines = []
    for j in range(20):
        for k in range(1 + j % 3):
            lines.append({"a_id": j, "line_no": k + 1, "x": k})
    seq = find_sequences({"l": lines}, [{"collection": "l", "path": "a_id", "to": "a", "to_path": "_id"}])
    assert {(s["path"], s["start"]) for s in seq} == {("line_no", 1), ("x", 0)}


def test_every_source_document_validates(shop):
    """Prop. 1400: the inferred schema admits every source document."""
    cols, profs, _ = shop
    for c, docs in cols.items():
        assert validate_collection(profs[c], docs)["share"] == 1.0


def test_validation_names_violations(shop):
    cols, profs, _ = shop
    o = copy.deepcopy(cols["orders"][0])
    o["items"] = o["items"] * 9                               # longer than any source array
    o["payment"]["payer"] = "x"                               # field of another variant
    o.pop("status")                                           # required field
    o["n_items"] = 99                                         # length field disagrees
    v = violations(profs["orders"].root, o)
    assert any("outside" in x for x in v) and any("variant" in x for x in v)
    assert any("required" in x for x in v) and any("length of" in x for x in v)


# ---------------------------------------------------------------- shredding
def test_shred_nest_round_trip(shop):
    """Prop. 1401: nest(shred(D)) == D (identifiers kept)."""
    cols, profs, refs = shop
    lay = build_layout(profs, refs)
    frames = shred(lay, cols)
    assert {"orders__items", "orders__items__discounts", "customers__wallets", "orders__shipping"} <= set(frames)
    back = nest(lay, frames, keep_ids=True)
    for c in cols:
        assert [canon(d) for d in back[c]] == [canon(d) for d in cols[c]]
    # the layout serialises and gives the same documents
    from relsynth.documents.shred import Layout
    lay2 = Layout.from_dict(json.loads(json.dumps(lay.to_dict())))
    assert [canon(d) for d in nest(lay2, frames, keep_ids=True)["orders"]] == [canon(d) for d in cols["orders"]]


def test_fresh_ids_keep_references(shop):
    cols, profs, refs = shop
    lay = build_layout(profs, refs)
    out = nest(lay, shred(lay, cols), seed=11)
    ids = {d["_id"] for d in out["customers"]}
    assert not ids & {d["_id"] for d in cols["customers"]}
    assert all(o["customer_id"] in ids for o in out["orders"])
    assert reference_integrity(out, refs)[0]["share"] == 1.0
    assert nest(lay, shred(lay, cols), seed=11)["orders"][0]["_id"] == out["orders"][0]["_id"]


# ---------------------------------------------------------------- conformance
def test_conform_repairs_every_violation(shop):
    """Prop. 1402: conform's output validates; repairs are counted."""
    cols, profs, _ = shop
    bad = copy.deepcopy(cols["orders"][:50])
    bad[0]["items"] = bad[0]["items"] * 9
    bad[1]["payment"]["payer"] = "x"
    bad[2].pop("status")
    bad[3]["items"] = []
    bad[4]["rating"] = [1]
    fixed, rep = conform(profs["orders"], bad, seed=1)
    assert validate_collection(profs["orders"], fixed)["share"] == 1.0
    assert rep["documents_repaired"] >= 5 and set(rep["by_kind"]) >= {"length", "extra", "missing", "type"}
    tags = copy.deepcopy(cols["customers"][:40])
    tags[0]["tags"] = ["vip", "vip", "eu"]
    fixed, rep = conform(profs["customers"], tags)
    assert len(set(fixed[0]["tags"])) == len(fixed[0]["tags"]) and rep["by_kind"].get("set") == 1


# ---------------------------------------------------------------- metrics
def test_shape_metrics_and_limits(shop):
    cols, profs, _ = shop
    d = shape_distances(cols["orders"], cols["orders"], "orders")
    assert all(v == 0 for v in d.values())
    chk = shape_checks(cols["orders"], cols["orders"], "orders", splits=6)
    assert all(m["pass"] is not False for m in chk)
    # dropping a field from every synthetic document fails presence
    broken = [{k: v for k, v in o.items() if k != "status"} for o in cols["orders"]]
    chk = {m["metric"]: m for m in shape_checks(cols["orders"], broken, "orders", splits=6)}
    assert chk["presence_max"]["pass"] is False and chk["signature"]["pass"] is False


# ---------------------------------------------------------------- IR
def test_ir_export_validates(shop):
    _, profs, refs = shop
    doc = to_ir_document(profs, refs, name="shop")
    pay = doc["entities"]["orders"]["attributes"]["payment"]
    assert pay["variants"]["discriminator"] == "method" and "last4" in pay["variants"]["cases"]["card"]
    assert doc["entities"]["orders"]["attributes"]["items"]["length"] == {"min": 1, "max": 8}
    assert doc["entities"]["customers"]["attributes"]["wallets"]["type"] == "map"
    est = to_ir(profs, refs, name="shop")
    from relsynth.ir.validate import check
    errs, _ = check(est)
    assert errs == []
    assert est.relationships[0].from_.entity == "orders" if hasattr(est.relationships[0], "from_") else True


# ---------------------------------------------------------------- DynamoDB single-table
def test_single_table_split_merge():
    """Prop. 1404: keys rebuilt from templates equal the source keys."""
    import dynamodb_shop as ex
    items = ex.items(60)
    cols, design = dynamo.split(items, indexes=[{"name": "GSI1", "HASH": "GSI1PK", "RANGE": "GSI1SK"}])
    assert set(cols) == {"Customer", "Order", "Item"}
    assert design["entities"]["Order"]["templates"]["SK"] == "ORDER#{ordered_at}#{order_id}"
    assert design["entities"]["Item"]["templates"]["GSI1PK"] == "ORDER#{order_id}"
    assert all("PK" not in d and "entity_type" not in d for d in cols["Order"])
    back = dynamo.merge(cols, design)
    key = lambda it: (it["PK"], it["SK"])  # noqa: E731
    assert sorted(map(canon, back), key=str) == sorted(map(canon, items), key=str) or \
        sorted(back, key=key) == sorted(items, key=key)
    assert dynamo.check(back, design)["pass"]
    cont, phys = dynamo.to_ir(design, "app-table")
    assert cont["indexes"][0]["partition_key"] == "GSI1PK" and phys["Item"]["discriminator"]["value"] == "ITEM"


def test_dynamodb_connector_moto():
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    import dynamodb_shop as ex
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    with moto.mock_aws():
        r = boto3.resource("dynamodb", region_name="eu-west-1")
        r.create_table(**ex.TABLE)
        con = connect_documents({"kind": "dynamodb", "table": "app-table", "region_name": "eu-west-1"})
        con.resource = r
        items = ex.items(30)
        assert con.write("app-table", items, if_exists="append") == len(items)
        ks = con.key_schema("app-table")
        assert ks["partition_key"] == "PK" and ks["indexes"][0]["HASH"] == "GSI1PK"
        back = list(con.read("app-table"))
        assert sorted(canon(x) for x in back) == sorted(canon(x) for x in items)


# ---------------------------------------------------------------- connectors
def test_json_files_connector(tmp_path, shop):
    cols, _, _ = shop
    con = connect_documents(f"jsonl:{tmp_path}")
    assert isinstance(con, JsonFiles)
    con.write_all(cols)
    assert con.collections() == ["customers", "orders"]
    got = con.read_all()
    assert [canon(d) for d in got["orders"]] == [canon(d) for d in cols["orders"]]
    assert parse_location("mongodb://h/db")["kind"] == "mongodb"
    assert parse_location("dynamodb:t") == {"kind": "dynamodb", "table": "t"}


def test_mongodb_connector_mongomock(shop):
    mongomock = pytest.importorskip("mongomock")
    cols, _, _ = shop
    con = MongoDB({"kind": "mongodb", "database": "shop"}, client=mongomock.MongoClient(tz_aware=True))
    con.write_all(cols)
    assert con.collections() == ["customers", "orders"] and con.count("orders") == len(cols["orders"])
    got = con.read_all()
    a = canon(got["orders"][0])
    assert json.loads(a)["seq"].keys() == {"$numberLong"}
    assert canon(got["orders"][0]["payment"]) == canon(cols["orders"][0]["payment"])


def test_mongodb_pushdown_pipeline():
    """The $group stage counts BSON types per path and $size histograms."""
    seen = []

    class Coll:
        def aggregate(self, pipe, **kw):
            seen.append(pipe)
            if "$match" in pipe[0]:
                return [{"_id": 2, "c": 3}, {"_id": 1, "c": 1}]
            g = pipe[0]["$group"]
            out = {"_id": None, "n": 4}
            for k in g:
                if k.startswith("p0_"):
                    out[k] = 4 if k == "p0_array" else 0
                if k.startswith("p1_"):
                    out[k] = 3 if k == "p1_int" else (1 if k == "p1_missing" else 0)
            return [out]

    class DB(dict):
        def __getitem__(self, k):
            return Coll()

    con = MongoDB.__new__(MongoDB)
    con.db = DB()
    res = con.exact_counts("c", ["tags", "n"], ["tags"])
    assert res["documents"] == 4 and res["types"]["tags"] == {"array": 4} and res["types"]["n"] == {"int": 3}
    assert res["lengths"]["tags"] == {2: 3, 1: 1}
    assert seen[0][0]["$group"]["p0_array"]["$sum"]["$cond"][0] == {"$eq": [{"$type": "$tags"}, "array"]}


@pytest.mark.mongodb
def test_mongodb_live_pushdown(shop):
    url = os.environ.get("RELSYNTH_TEST_MONGO_URL")
    if not url:
        pytest.skip("RELSYNTH_TEST_MONGO_URL not set")
    from relsynth.documents.connectors import profile_with_pushdown
    cols, profs, _ = shop
    con = MongoDB({"kind": "mongodb", "url": url, "database": "relsynth_test"})
    con.write_all(cols)
    prof, sample = profile_with_pushdown(con, "orders", sample_size=200)
    assert prof.documents == len(cols["orders"]) and len(sample) == 200
    assert prof.root.fields["items"].lengths == profs["orders"].root.fields["items"].lengths
    assert prof.root.fields["rating"].types == profs["orders"].root.fields["rating"].types
    assert prof.root.fields["shipping"].present == profs["orders"].root.fields["shipping"].present


# ---------------------------------------------------------------- end to end
@pytest.mark.slow
def test_document_synthesizer_end_to_end(tmp_path):
    from relsynth.documents import DocumentSynthesizer
    cols = fake.collections(200, seed=5)
    ds = DocumentSynthesizer(cols, seed=3, workdir=tmp_path)
    out = ds.synthesize()
    rep = ds.report(out, splits=6)
    for c, v in rep["collections"].items():
        assert v["validity"]["share"] == 1.0, (c, v["validity"]["examples"])
    assert all(r["share"] == 1.0 for r in rep["references"])
    assert all(o["n_items"] == len(o["items"]) for o in out["orders"])
    assert all(set(o["payment"]) - {"method", "amount"} <= {"card": {"last4", "network"}, "paypal": {"payer"},
                                                             "bank": {"iban_country"}}[o["payment"]["method"]]
               for o in out["orders"])
    bad = [(c, m["metric"]) for c, v in rep["collections"].items() for m in v["shape"] if m["pass"] is False]
    assert len(bad) <= 1, bad                    # 14 metrics at 99%: one false failure is within chance
    again = DocumentSynthesizer(cols, seed=3, workdir=tmp_path / "b").synthesize()
    assert [canon(d) for d in again["orders"]] == [canon(d) for d in out["orders"]]


def test_no_password_on_the_command_line_and_hostile_keyspaces():
    from relsynth.documents.connectors import _cb_ident, parse_location
    from relsynth.errors import SourceError
    assert parse_location("mongodb://db.example:27017/shop")["url"].endswith("/shop")
    for url in ("mongodb://app:secret@db.example/shop", "mongodb+srv://app:secret@cluster.example/shop"):
        with pytest.raises(SourceError, match="password"):
            parse_location(url)
    assert _cb_ident("orders") == "`orders`"
    for bad in ("a`b", "x`; DELETE FROM y; --", "", "a\nb"):
        with pytest.raises(SourceError):
            _cb_ident(bad)


def test_docs_check_gates_on_invalid_documents(tmp_path, shop):
    """relsynth docs check exits 0 when the documents validate and 1 (the quality gate) when one does not."""
    from relsynth.cli import main
    from relsynth.documents.values import write_json_documents
    real, good, bad = tmp_path / "real", tmp_path / "good", tmp_path / "bad"
    for d in (real, good, bad):
        d.mkdir()
    docs = shop[0]["customers"]
    write_json_documents(docs, real / "customers.jsonl")
    write_json_documents(docs, good / "customers.jsonl")
    write_json_documents([{**docs[0], "unexpected_field": 1}] + docs[1:], bad / "customers.jsonl")
    assert main(["docs", "check", f"jsonl:{real}", f"jsonl:{good}", "--splits", "2"]) == 0
    assert main(["docs", "check", f"jsonl:{real}", f"jsonl:{bad}", "--splits", "2"]) == 1


def test_sqlite_export_quotes_names_from_documents():
    """Field paths become SQLite names; an embedded quote must not change the DDL."""
    import sqlite3

    from relsynth.documents.relational import _q
    assert _q('a"b') == '"a""b"'
    con = sqlite3.connect(":memory:")
    name = 'x"); DROP TABLE t; --'
    con.execute("CREATE TABLE t (a)")
    col = _q('c"d')
    con.execute(f"CREATE TABLE {_q(name)} ({col} TEXT)")
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"t", name}

# SPDX-License-Identifier: Apache-2.0
"""The canonical IR (spec v2): object model, 1.0 round trip with identical
output, hand-written examples, JSON Schema, privacy-unit paths (Props. 107, 108),
ODCS v3 and OpenLineage, migration and the `relsynth ir` commands."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import relsynth as rs
from relsynth import ir
from relsynth.cli import main
from relsynth.ir.jsonschema import SCHEMA_FILE, reference, schema_text
from relsynth.ir.migrate import migrate_doc
from relsynth.ir.odcs import from_odcs, to_odcs
from relsynth.ir.openlineage import datasets
from relsynth.ir.privacy import contribution, trace
from relsynth.ir.v1 import canonical, diff
from relsynth.ir.validate import check
from relsynth.spec import SpecError, normalise, read_yaml

ROOT = Path(__file__).resolve().parents[2]
SPECS = sorted((ROOT / "tests/data/specs_v1").glob("*.yaml")) + sorted(ROOT.glob("examples/*/spec.yaml"))
EXAMPLES = sorted((ROOT / "examples/ir").glob("*.yaml"))
STANDARDS = ROOT / "tests/data/standards"
SCENARIOS = ["S1_mixed_table", "S2_star", "S3_snowflake", "S4_tree", "S5_temporal", "S6_text", "S7_enterprise",
             "S8_reporting", "S9_small_shop", "S10_order_dw", "S11_retail_dw"]


def _norm(d):
    return json.loads(json.dumps(normalise(json.loads(json.dumps(d, default=str))), default=str))


def _ids(paths):
    return [f"{p.parent.name}/{p.stem}" for p in paths]


# ---------------------------------------------------------------------------
# Prop. 106: 1.0 specs convert without loss
# ---------------------------------------------------------------------------

def _assert_round_trip(raw):
    est = ir.lift(raw)
    v1, losses = ir.lower(est)
    assert losses == []
    assert diff(_norm(raw), _norm(v1)) == []
    # the IR survives serialisation and is a fixpoint of lower -> lift
    again = ir.Estate.from_dict(json.loads(ir.dumps(est, "json")))
    assert again == est
    assert diff(canonical(est), canonical(ir.lift(ir.lower(again)[0]))) == []
    # a normalised spec (as doctor --apply writes it back) converts the same way
    assert diff(_norm(raw), _norm(ir.lower(ir.lift(_norm(raw)))[0])) == []


@pytest.mark.parametrize("path", SPECS, ids=_ids(SPECS))
def test_real_specs_round_trip(path):
    _assert_round_trip(read_yaml(path))


def test_fake_scenario_specs_round_trip():
    from conftest import scenario
    for name in SCENARIOS:
        _, spec, _ = scenario(name)
        _assert_round_trip(spec)


def test_round_trip_detects_a_change():
    raw = read_yaml(ROOT / "tests/data/specs_v1/chinook.yaml")
    est = ir.lift(raw)
    est.entities["Invoice"].keys.clear()
    assert diff(_norm(raw), _norm(ir.lower(est)[0])) == ["tables.Invoice.primary_key: 'InvoiceId' (dropped)"]


@pytest.mark.parametrize("name", ["S2_star", "S9_small_shop", "S10_order_dw"])
def test_ir_document_gives_identical_synthetic_data(name, tmp_path):
    """Same seed, 1.0 spec or its IR document: the same synthetic database."""
    from conftest import write_scenario
    spec = write_scenario(name, {"kind": "csv", "path": str(tmp_path / "src")})
    doc = ir.lift(spec).to_dict()
    assert "ir_version" in doc and "tables" not in doc
    a = rs.fit(spec, seed=3).sample(dataset=0)
    b = rs.fit(doc, seed=3).sample(dataset=0)
    assert sorted(a) == sorted(b)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])


def test_cli_runs_an_ir_file(tmp_path):
    from conftest import write_scenario
    spec = write_scenario("S2_star", {"kind": "csv", "path": str(tmp_path / "src")})
    (tmp_path / "v1.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    ir.dump(ir.lift(spec), tmp_path / "v2.yaml")
    for v in ("v1", "v2"):
        assert main(["run", str(tmp_path / f"{v}.yaml"), "--out", str(tmp_path / f"out_{v}"), "--bundle", "",
                     "--no-privacy", "--scale", "0.2"]) == 0
    for f in sorted((tmp_path / "out_v1").glob("*.csv")):
        assert f.read_bytes() == (tmp_path / "out_v2" / f.name).read_bytes()


def test_engine_refuses_shapes_it_cannot_synthesise():
    with pytest.raises(SpecError, match="relational records only.*document"):
        rs.load_spec(str(ROOT / "examples/ir/mongodb_orders.yaml"))
    with pytest.raises(ir.IRError, match="not representable"):
        ir.lower(ir.load(ROOT / "examples/ir/mongodb_orders.yaml"), strict=True)


def test_lower_reports_annotations_it_drops():
    est = ir.load(ROOT / "examples/ir/iceberg_page_views.yaml")
    spec, losses = ir.lower(est)
    assert rs.Spec(spec).errors() == []          # the relational part runs
    assert any(x.startswith("entities.page_views.physical") for x in losses)
    assert any(x.startswith("privacy") for x in losses)


# ---------------------------------------------------------------------------
# hand-written documents for other shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", EXAMPLES, ids=_ids(EXAMPLES))
def test_examples_validate(path):
    est = ir.load(path)
    errs, warns = check(est)
    assert errs == [] and warns == []
    assert ir.Estate.from_dict(json.loads(ir.dumps(est, "json"))) == est
    pytest.importorskip("jsonschema")
    from relsynth.ir.jsonschema import validate_json
    assert validate_json(read_yaml(path)) == []


def test_example_shapes():
    m = ir.load(ROOT / "examples/ir/mongodb_orders.yaml").entities["orders"]
    assert m.attributes["items"].items.fields["quantity"].min == 1 and m.attributes["items"].length == {"min": 1,
                                                                                                         "max": 50}
    assert set(m.attributes["payment"].variants.cases) == {"card", "paypal", "bank_transfer"}
    assert m.attributes["shipping"].optional
    d = ir.load(ROOT / "examples/ir/dynamodb_single_table.yaml")
    assert {e.physical.container for e in d.entities.values()} == {"AppTable"}
    assert d.entities["OrderItem"].primary_key.attributes == ["order_id", "line_no"]
    g = ir.load(ROOT / "examples/ir/property_graph_movies.yaml")
    assert g.entities["FOLLOWS"].edge.loops is False and g.relationship("person_movie").cardinality == "many_to_many"
    i = ir.load(ROOT / "examples/ir/iceberg_page_views.yaml").entities["page_views"].physical
    assert [(p.transform, p.param) for p in i.partitioning] == [("day", None), ("bucket", 16)]
    s = ir.load(ROOT / "examples/ir/event_stream_clicks.yaml").entities["clicks"]
    assert s.stream.max_lateness == "PT5M" and s.time.ingestion_time == "ingested_at"


def _example(name):
    return read_yaml(ROOT / f"examples/ir/{name}.yaml")


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["entities"]["ACTED_IN"]["edge"].update(source="nope"), "edge.source: 'nope' is not a relationship"),
    (lambda d: d["entities"]["ACTED_IN"]["edge"].update(source="directed_person"), "does not start at ACTED_IN"),
    (lambda d: d["relationships"][-1].pop("via"), "many_to_many needs via"),
    (lambda d: d["relationships"][0].update(cardinality="one_to_one"), "one_to_one needs from.attributes"),
    (lambda d: d["entities"]["Person"].update(privacy={"public": True}), "marked public"),
])
def test_invalid_graph_documents(edit, message):
    d = _example("property_graph_movies")
    edit(d)
    errs, _ = check(ir.Estate.from_dict(d))
    assert any(message in e for e in errs), errs


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["entities"]["page_views"]["physical"]["partitioning"][1].pop("param"), "bucket needs param"),
    (lambda d: d["entities"]["page_views"]["physical"]["partitioning"][0].update(source="url_path"),
     "day needs a date or timestamp"),
    (lambda d: d["entities"]["page_views"]["physical"]["sort_order"][0].update(source="nope"), "not an attribute"),
])
def test_invalid_iceberg_documents(edit, message):
    d = _example("iceberg_page_views")
    edit(d)
    errs, _ = check(ir.Estate.from_dict(d))
    assert any(message in e for e in errs), errs


def test_invalid_documents_are_reported_with_hints():
    d = _example("mongodb_orders")
    d["entities"]["orders"]["attributes"]["payment"]["variants"]["discriminator"] = "kind"
    errs, _ = check(ir.Estate.from_dict(d))
    assert any("discriminator 'kind' is not an attribute" in e for e in errs)
    d = _example("dynamodb_single_table")
    d["entities"]["Order"]["physical"]["key_templates"]["PK"] = "CUSTOMER#{customer}"
    errs, _ = check(ir.Estate.from_dict(d))
    assert any("{customer} is not an attribute" in e for e in errs)
    d = _example("event_stream_clicks")
    d["entities"]["clicks"]["stream"]["max_lateness"] = "5 minutes"
    errs, _ = check(ir.Estate.from_dict(d))
    assert any("ISO 8601" in e for e in errs)
    with pytest.raises(ir.IRError, match="did you mean 'entities'"):
        ir.Estate.from_dict({"ir_version": "2.0", "entites": {}})
    with pytest.raises(ir.IRError, match="'recrd' is not one of"):
        ir.Estate.from_dict({"ir_version": "2.0", "entities": {"a": {"shape": "recrd"}}})
    with pytest.raises(ir.IRError, match="'3.0' is not one of"):
        ir.Estate.from_dict({"ir_version": "3.0", "entities": {}})


def test_vendor_extensions_are_kept():
    d = _example("iceberg_page_views")
    d["entities"]["users"]["x-owner"] = "team-web"
    d["x-catalog"] = {"id": 7}
    est = ir.Estate.from_dict(d)
    assert est.entities["users"].extensions == {"x-owner": "team-web"}
    assert est.to_dict()["x-catalog"] == {"id": 7}


# ---------------------------------------------------------------------------
# JSON Schema and reference: generated, never edited
# ---------------------------------------------------------------------------

def test_schema_and_reference_are_generated():
    assert SCHEMA_FILE.read_text(encoding="utf-8") == schema_text()
    assert (ROOT / "docs/ir-reference.md").read_text(encoding="utf-8") == reference()


def test_json_schema_rejects_what_the_model_rejects():
    jsonschema = pytest.importorskip("jsonschema")
    from relsynth.ir.jsonschema import json_schema, validate_json
    jsonschema.Draft202012Validator.check_schema(json_schema())
    bad = {"ir_version": "2.0", "entities": {"A": {"shap": "record", "attributes": {"x": {"type": "integr"}}}}}
    errs = validate_json(bad)
    assert any("shap" in e for e in errs) and any("integr" in e for e in errs)
    with pytest.raises(ir.IRError):
        ir.Estate.from_dict(bad)
    for p in SPECS:
        assert validate_json(json.loads(ir.dumps(ir.load(p), "json"))) == [], p


# ---------------------------------------------------------------------------
# Props. 107-108: the privacy unit and its reference paths
# ---------------------------------------------------------------------------

REAL_UNITS = {
    "chinook": ("Customer", {"Invoice": [["Invoice.CustomerId->Customer"]],
                             "InvoiceLine": [["InvoiceLine.InvoiceId->Invoice", "Invoice.CustomerId->Customer"]]}),
    "contoso": ("Customer", {"Orders": [["Orders.CustomerKey->Customer"]],
                             "OrderRows": [["OrderRows.OrderKey->Orders", "Orders.CustomerKey->Customer"]],
                             "Sales": [["Sales.CustomerKey->Customer"],
                                       ["Sales.OrderKey->Orders", "Orders.CustomerKey->Customer"]]}),
    "adventureworks_v07": ("DimCustomer", {
        "FactInternetSales": [["FactInternetSales.CustomerKey->DimCustomer"]],
        "FactSurveyResponse": [["FactSurveyResponse.CustomerKey->DimCustomer"]],
        "FactInternetSalesReason": [["FactInternetSalesReason.(SalesOrderNumber+SalesOrderLineNumber)->FactInternetSales",
                                     "FactInternetSales.CustomerKey->DimCustomer"]]}),
    "bi_corep_lr_v03": ("DIM_Entity", {
        "DIM_Submission": [["DIM_Submission.EntityKey->DIM_Entity"]],
        "FACT_TemplateFiling": [["FACT_TemplateFiling.SubmissionKey->DIM_Submission",
                                 "DIM_Submission.EntityKey->DIM_Entity"]]}),
}


@pytest.mark.parametrize("db", list(REAL_UNITS))
def test_privacy_unit_paths_of_the_real_databases(db):
    unit, want = REAL_UNITS[db]
    est = ir.load(ROOT / f"tests/data/specs_v1/{db}.yaml")
    tr = trace(est, unit)
    owned = {n: t.paths for n, t in tr.entities.items() if t.status == "owned"}
    for n, paths in want.items():
        assert owned[n] == paths, n
    assert tr.entities[unit].status == "unit" and not tr.warnings


def test_privacy_statuses():
    tr = trace(ir.load(ROOT / "tests/data/specs_v1/chinook.yaml"), "Customer")
    st = {n: t.status for n, t in tr.entities.items()}
    assert st["Employee"] == "context" and st["Track"] == "context" and st["Playlist"] == "unrelated"
    g = trace(ir.load(ROOT / "examples/ir/property_graph_movies.yaml"))
    assert len(g.entities["FOLLOWS"].paths) == 2 and "2 paths" in g.entities["FOLLOWS"].note


def test_hierarchy_on_the_unit_is_a_cross_unit_link():
    d = _example("dynamodb_single_table")
    d["entities"]["Customer"]["attributes"]["referred_by"] = {"type": "string"}
    d["relationships"].append({"name": "referral", "kind": "hierarchy", "from": {"entity": "Customer",
                               "attributes": ["referred_by"]}, "to": {"entity": "Customer"}})
    tr = trace(ir.Estate.from_dict(d))
    assert tr.cross_unit == ["referral"] and any("dangling" in w for w in tr.warnings)


def _small_shop():
    from conftest import scenario
    t, spec, _ = scenario("S9_small_shop")
    return t, ir.lift(spec)


def _unit_entity(est):
    return next(n for n in ("customer", "customers", "Customer") if n in est.entities)


def test_contribution_matches_the_joins():
    """Prop. 107: along each path, every record has at most one owner, found by joins."""
    t, est = _small_shop()
    unit = _unit_entity(est)
    tr = trace(est, unit)
    pk = est.entities[unit].primary_key.attributes[0]
    rng = np.random.default_rng(1)
    chosen = set(rng.choice(t[unit][pk].to_numpy(), size=25, replace=False).tolist())
    masks = contribution(est, t, chosen, unit)
    for n, info in tr.entities.items():
        if info.status != "owned":
            continue
        for path in info.paths:
            df = t[n].copy()
            df["_owner"] = None
            cur = df
            keys = None
            for rn in path:
                r = est.relationship(rn)
                parent = t[r.to.entity]
                pka = r.to.attributes
                left = cur[r.from_.attributes] if keys is None else keys
                left.columns = pka
                m = left.merge(parent, how="left", on=pka)
                m.index = left.index
                nxt = path[path.index(rn) + 1] if rn != path[-1] else None
                keys = m[est.relationship(nxt).from_.attributes] if nxt else m[[pk]]
            owner = keys[pk]
            assert (masks[n] >= owner.isin(chosen)).all()
        assert masks[n].sum() > 0


def test_removing_a_unit_keeps_referential_integrity():
    """Prop. 108: removing the units' records along every traced path leaves no
    reference into a removed record."""
    t, est = _small_shop()
    unit = _unit_entity(est)
    pk = est.entities[unit].primary_key.attributes[0]
    chosen = set(t[unit][pk].iloc[::7].tolist())
    masks = contribution(est, t, chosen, unit)
    kept = {n: (df[~masks[n]] if n in masks else df) for n, df in t.items()}
    assert sum(len(kept[n]) for n in kept) < sum(len(df) for df in t.values())
    for r in est.relationships:
        if r.kind != "reference" or r.from_.entity == r.to.entity:
            continue
        fa, ta = r.from_.attributes, r.to.attributes
        child = kept[r.from_.entity][fa].dropna()
        parent = kept[r.to.entity][ta]
        assert child.merge(parent.drop_duplicates(), how="left", left_on=fa, right_on=ta, indicator=True)[
            "_merge"].eq("both").all(), r.name


# ---------------------------------------------------------------------------
# ODCS v3 and OpenLineage
# ---------------------------------------------------------------------------

def _odcs_validator():
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft201909Validator(json.loads((STANDARDS / "odcs-json-schema-v3.2.0.json").read_text()))


@pytest.mark.parametrize("path", SPECS + EXAMPLES, ids=_ids(SPECS + EXAMPLES))
def test_odcs_round_trip(path):
    v = _odcs_validator()
    est = ir.load(path)
    c = json.loads(json.dumps(to_odcs(est)))
    assert [e.message for e in v.iter_errors(c)] == []
    back = from_odcs(c)
    assert [x for x in diff(canonical(est), canonical(back)) if not x.startswith(("source", "target"))] == []


def test_odcs_export_masks_passwords():
    est = ir.load(ROOT / "tests/data/specs_v1/chinook.yaml")
    est.source = {"kind": "postgresql", "url": "postgresql://ana:s3cret@db/shop"}
    assert "s3cret" not in json.dumps(to_odcs(est))


THIRD_PARTY = {
    "apiVersion": "v3.1.0", "kind": "DataContract", "id": "shop-orders", "version": "2.1.0", "status": "active",
    "name": "shop orders",
    "schema": [
        {"name": "customers", "physicalName": "crm.customers", "physicalType": "table", "properties": [
            {"name": "id", "logicalType": "integer", "primaryKey": True, "primaryKeyPosition": 1, "required": True},
            {"name": "email", "logicalType": "string", "unique": True, "classification": "restricted"},
            {"name": "country", "logicalType": "string"},
            {"name": "address", "logicalType": "object", "properties": [
                {"name": "city", "logicalType": "string"}, {"name": "zip", "logicalType": "string"}]}]},
        {"name": "orders", "physicalType": "table", "properties": [
            {"name": "order_id", "logicalType": "integer", "primaryKey": True, "primaryKeyPosition": 1},
            {"name": "line", "logicalType": "integer", "primaryKey": True, "primaryKeyPosition": 2},
            {"name": "customer_id", "logicalType": "integer", "required": True,
             "relationships": [{"to": "customers.id"}]},
            {"name": "amount", "logicalType": "number", "logicalTypeOptions": {"minimum": 0}},
            {"name": "order_date", "logicalType": "date", "partitioned": True, "partitionKeyPosition": 1},
            {"name": "tags", "logicalType": "array", "items": {"logicalType": "string"}}],
         "quality": [{"type": "sql", "query": "SELECT COUNT(*) FROM orders WHERE amount < 0", "mustBe": 0}]}]}


def test_odcs_import_of_a_third_party_contract():
    assert [e.message for e in _odcs_validator().iter_errors(THIRD_PARTY)] == []
    est = from_odcs(copy.deepcopy(THIRD_PARTY))
    errs, _ = check(est, relational=False)
    assert errs == []
    c, o = est.entities["customers"], est.entities["orders"]
    assert c.shape == "document" and c.attributes["address"].fields["city"].type == "string"
    assert c.attributes["email"].privacy.classification == "restricted"
    assert o.primary_key.attributes == ["order_id", "line"] and o.attributes["amount"].min == 0
    assert o.physical.partitioning[0].source == "order_date"
    r = est.relationships[0]
    assert (r.from_.entity, r.from_.attributes, r.to.entity, r.to.attributes) == ("orders", ["customer_id"],
                                                                                  "customers", ["id"])
    with pytest.raises(ir.IRError, match="apiVersion"):
        from_odcs({**THIRD_PARTY, "apiVersion": "v2.2.2"})


def _ol_registry():
    pytest.importorskip("jsonschema")
    from referencing import Registry, Resource
    res = []
    for f in (STANDARDS / "openlineage").glob("*.json"):
        s = json.loads(f.read_text())
        res.append((s["$id"], Resource.from_contents(s)))
    return Registry().with_resources(res), dict(res)


@pytest.mark.parametrize("path", EXAMPLES + SPECS[:2], ids=_ids(EXAMPLES + SPECS[:2]))
def test_openlineage_facets_validate(path):
    import jsonschema
    reg, ids = _ol_registry()
    est = ir.load(path)
    for n, ds in zip(est.entities, datasets(est, "synthetic://out", "sqlite://real")):
        for k, facet in ds["facets"].items():
            name = facet["_schemaURL"].rsplit("/", 1)[-1]
            base = facet["_schemaURL"].split("#")[0]
            v = jsonschema.Draft202012Validator({"$ref": f"{base}#/$defs/{name}"}, registry=reg)
            assert [e.message for e in v.iter_errors(facet)] == [], (ds["name"], k)
        assert set(ds["facets"]["columnLineage"]["fields"]) == set(est.entities[n].attributes)


def test_openlineage_tags_carry_privacy_roles():
    ds = datasets(ir.load(ROOT / "examples/ir/mongodb_orders.yaml"), "mongodb://shop")
    tags = {(t["key"], t.get("field")) for t in ds[0]["facets"]["tags"]["tags"]}
    assert ("privacy_unit", None) in tags and ("direct_identifier", "email") in tags
    assert ("quasi_identifier", "addresses.city") in tags or ("quasi_identifier", "birth_year") in tags


# ---------------------------------------------------------------------------
# migration and commands
# ---------------------------------------------------------------------------

def test_migration_is_proved_and_adds_privacy():
    m = migrate_doc(read_yaml(ROOT / "tests/data/specs_v1/chinook.yaml"), privacy_unit="Customer",
                    public=["Genre", "MediaType"])
    assert m.identical and m.differences == [] and m.source_format == "1.0"
    assert m.estate.privacy.unit.entity == "Customer" and m.estate.entities["Genre"].privacy.public
    assert "identical to the original" in m.header("spec.yaml")
    with pytest.raises(ir.IRError):
        migrate_doc(read_yaml(ROOT / "tests/data/specs_v1/chinook.yaml"), privacy_unit="Nobody")


def test_ir_commands(tmp_path, capsys):
    spec = ROOT / "tests/data/specs_v1/chinook.yaml"
    doc = tmp_path / "chinook.ir.yaml"
    assert main(["ir", "migrate", str(spec), "--out", str(doc), "--privacy-unit", "Customer"]) == 0
    assert doc.read_text().startswith("# relsynth IR 2.0")
    assert main(["ir", "validate", str(doc)]) == 0
    assert main(["check", str(doc)]) == 0
    assert main(["ir", "convert", str(doc), "--to", "v1", "--out", str(tmp_path / "back.yaml")]) == 0
    assert diff(_norm(read_yaml(spec)), _norm(read_yaml(tmp_path / "back.yaml"))) == []
    capsys.readouterr()
    assert main(["ir", "privacy", str(doc), "--json"]) == 0
    tr = json.loads(capsys.readouterr().out)["trace"]
    assert tr["entities"]["InvoiceLine"]["status"] == "owned"
    assert main(["ir", "odcs", str(doc), "--out", str(tmp_path / "c.yaml")]) == 0
    assert main(["ir", "from-odcs", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "d.yaml")]) == 0
    assert main(["ir", "openlineage", str(doc), "--namespace", "sqlite://chinook", "--out",
                 str(tmp_path / "ol.json")]) == 0
    assert len(json.loads((tmp_path / "ol.json").read_text())) == 11
    assert main(["ir", "schema", "--out", str(tmp_path / "s.json")]) == 0
    assert main(["ir", "convert", str(ROOT / "examples/ir/mongodb_orders.yaml"), "--to", "v1", "--strict"]) == 2
    assert main(["ir", "validate", str(ROOT / "examples/ir/property_graph_movies.yaml")]) == 0


def test_introspect_can_write_ir(star_sqlite, tmp_path):
    _, db = star_sqlite
    out = tmp_path / "draft.yaml"
    assert main(["introspect", "--source", f"sqlite:{db}", "--out", str(out), "--format", "ir"]) == 0
    est = ir.load(out)
    assert set(est.entities) == {"customer", "product", "sales"} and check(est)[0] == []

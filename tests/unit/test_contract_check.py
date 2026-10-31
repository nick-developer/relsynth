# SPDX-License-Identifier: Apache-2.0
"""Data contract checks on output (ODCS v3, relsynth.ir.contract_check)."""
from __future__ import annotations

import copy

import pandas as pd
import pytest
from conftest import scenario, write_scenario

from relsynth.engine import Synthesizer
from relsynth.ir import lift as from_v1
from relsynth.ir.contract_check import check_contract, check_location
from relsynth.ir.odcs import to_odcs
from relsynth.spec import load_spec


def _reader(tables):
    def read(name, columns):
        if name not in tables:
            raise KeyError(name)
        return tables[name]
    return read


@pytest.fixture(scope="module")
def dw():
    t, s, _ = scenario("S10_order_dw")
    syn = Synthesizer(load_spec(copy.deepcopy(s))).fit(t)
    return t, s, syn


def test_synthetic_output_meets_the_source_contract(dw):
    """The contract exported from the real data's spec holds on the synthetic
    data: keys, links, types, required columns."""
    t, s, syn = dw
    contract = to_odcs(from_v1(copy.deepcopy(s)))
    real = check_contract(contract, _reader(t))
    out = syn.sample(index=0)
    rep = check_contract(contract, _reader(out))
    assert real.ok, real.to_markdown()
    assert rep.ok, rep.to_markdown()
    kinds = {c["check"].split()[0] for c in rep.checks if c["result"] == "passed"}
    assert {"exists", "logicalType", "primaryKey", "foreignKey"} <= kinds


def test_contract_violations_are_found():
    contract = {"id": "c", "schema": [
        {"name": "p", "properties": [{"name": "id", "logicalType": "integer", "primaryKey": True,
                                      "primaryKeyPosition": 1}]},
        {"name": "c", "properties": [
            {"name": "id", "logicalType": "integer", "primaryKey": True},
            {"name": "pid", "logicalType": "integer", "required": True},
            {"name": "code", "logicalType": "string", "unique": True},
            {"name": "amount", "logicalType": "number", "logicalTypeOptions": {"minimum": 0, "maximum": 10},
             "quality": [{"type": "library", "metric": "nullValues", "mustBeLessOrEqualTo": 0}]},
            {"name": "day", "logicalType": "date"},
            {"name": "gone", "logicalType": "string"}],
         "relationships": [{"type": "foreignKey", "from": "c.pid", "to": "p.id"}],
         "quality": [{"type": "library", "metric": "rowCount", "mustBeGreaterThan": 10},
                     {"type": "custom", "engine": "soda", "implementation": "..."}]},
        {"name": "missing", "properties": []}]}
    tables = {"p": pd.DataFrame({"id": [1, 2, 2]}),
              "c": pd.DataFrame({"id": [1, 2, 3], "pid": [1, None, 9], "code": ["a", "a", "b"],
                                 "amount": [1.0, 11.0, None], "day": ["2024-01-01", "x", None]})}
    rep = check_contract(contract, _reader(tables))
    failed = {(c["object"], c["check"]) for c in rep.checks if c["result"] == "failed"}
    assert failed == {("p", "primaryKey"), ("c", "required"), ("c", "unique"), ("c", "range"), ("c", "nullValues"),
                      ("c", "logicalType date"), ("c", "column"), ("c", "foreignKey"), ("c", "rowCount"),
                      ("missing", "exists")}
    assert rep.count("skipped") == 1 and not rep.ok
    assert "foreignKey" in rep.to_markdown()


@pytest.fixture(scope="module")
def dwdb(tmp_path_factory):
    db = tmp_path_factory.mktemp("dw") / "dw.db"
    s = write_scenario("S10_order_dw", {"kind": "sqlite", "path": str(db)})
    return db, to_odcs(from_v1(copy.deepcopy(s)))


def test_contract_check_reads_a_location(dwdb):
    db, contract = dwdb
    rep = check_location(contract, f"sqlite:{db}")
    assert rep.ok and rep.count("passed") > 20, rep.to_markdown()


def test_cli_check_contract(tmp_path, dwdb):
    import yaml

    from relsynth.cli import main
    db, contract = dwdb
    c = tmp_path / "contract.yaml"
    c.write_text(yaml.safe_dump(contract))
    assert main(["ir", "check-contract", str(c), "--data", f"sqlite:{db}", "--out", str(tmp_path / "r.md")]) == 0
    assert "0 failed" in (tmp_path / "r.md").read_text()
    contract["schema"][0]["properties"].append({"name": "not_there", "logicalType": "string"})
    c.write_text(yaml.safe_dump(contract))
    assert main(["ir", "check-contract", str(c), "--data", f"sqlite:{db}"]) != 0

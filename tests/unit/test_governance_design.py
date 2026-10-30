# SPDX-License-Identifier: Apache-2.0
"""T08 design phase: the governance schemas are valid JSON Schema, the examples
validate against them, and the schema enforces the user's decision that any
release outside the organisation requires differential privacy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

jsonschema = pytest.importorskip("jsonschema")

GOV = Path(__file__).resolve().parents[2] / "docs" / "governance"


def _schema(name):
    s = json.loads((GOV / "schemas" / name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(s)
    return jsonschema.Draft202012Validator(s)


def _policy():
    return yaml.safe_load((GOV / "examples" / "org-policy.yaml").read_text(encoding="utf-8"))


def test_example_policy_and_manifest_validate():
    _schema("org-policy-1.schema.json").validate(_policy())
    manifest = json.loads((GOV / "examples" / "run-manifest.example.json").read_text(encoding="utf-8"))
    _schema("run-manifest-1.schema.json").validate(manifest)


def test_external_context_cannot_waive_dp_or_approval():
    v = _schema("org-policy-1.schema.json")
    p = _policy()
    p["contexts"]["partner"]["dp_required"] = False
    assert not v.is_valid(p)
    p = _policy()
    p["contexts"]["public"]["approval"]["on_verdicts"] = ["review"]
    assert not v.is_valid(p)
    p = _policy()
    p["contexts"]["internal"]["dp_required"] = False          # internal may run without DP
    assert v.is_valid(p)


def test_budget_has_no_default_and_bounded_delta():
    v = _schema("org-policy-1.schema.json")
    p = _policy()
    del p["domains"]["customers"]["budget"]["lifetime"]
    assert not v.is_valid(p)
    p = _policy()
    p["domains"]["customers"]["budget"]["delta"] = 1e-4          # NIST SP 800-226: > 1e-5 is suspicious
    assert not v.is_valid(p)


def test_dp_manifest_must_carry_its_ledger_entry():
    v = _schema("run-manifest-1.schema.json")
    m = json.loads((GOV / "examples" / "run-manifest.example.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(m)
    del bad["predicate"]["privacy"]["ledger"]
    assert not v.is_valid(bad)
    nodp = copy.deepcopy(m)
    nodp["predicate"]["privacy"] = {"mode": "none"}
    assert v.is_valid(nodp)

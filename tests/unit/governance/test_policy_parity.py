# SPDX-License-Identifier: Apache-2.0
"""The built-in policy engine is a port of rego/release.rego: both must give the
same decision, reasons included, on every fixture.  The OPA half runs when an
`opa` binary is on PATH or in RELSYNTH_OPA (the CI job `policy` installs it)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import importlib

D = importlib.import_module("relsynth.governance.decide")

FIX = json.loads((Path(__file__).resolve().parents[2] / "data" / "governance" / "decisions.json").read_text())


@pytest.mark.parametrize("case", FIX, ids=[c["name"] for c in FIX])
def test_builtin_engine_matches_fixture(case):
    assert D.builtin(case["input"]) == case["expect"]


needs_opa = pytest.mark.skipif(D.opa_binary() is None, reason="no opa binary (set RELSYNTH_OPA)")


@needs_opa
@pytest.mark.parametrize("case", FIX, ids=[c["name"] for c in FIX])
def test_opa_matches_builtin(case):
    res = D.opa(case["input"])
    assert "engine_error" not in res, res
    assert res == case["expect"]


@needs_opa
def test_opa_unit_tests_pass():
    p = subprocess.run([D.opa_binary(), "test", str(D.rego_dir()), "-v"], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr


def test_engine_failure_denies(monkeypatch):
    monkeypatch.setenv("RELSYNTH_OPA", "/nonexistent/opa")
    res = D.decide(FIX[0]["input"], engine="opa")
    assert not res["allow"] and "failure is a deny" in " ".join(res["reasons"])
    monkeypatch.delenv("RELSYNTH_OPA")
    monkeypatch.setattr(D, "opa_binary", lambda: None)
    res = D.decide(FIX[0]["input"], engine="opa")
    assert not res["allow"] and "no `opa` binary" in " ".join(res["reasons"])

# SPDX-License-Identifier: Apache-2.0
"""Tests of .github/scripts/security_findings.py, the work list of security-patch.yml."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).parents[2] / ".github" / "scripts" / "security_findings.py"
_spec = importlib.util.spec_from_file_location("security_findings", _P)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)


def _cs(number, severity=None, level=None, state="open"):
    return {"number": number, "state": state, "tool": {"name": "CodeQL"},
            "rule": {"id": f"r{number}", "severity": severity, "security_severity_level": level},
            "most_recent_instance": {"location": {"path": "src/x.py", "start_line": 4}, "message": {"text": "m"}}}


def test_clip_removes_control_characters_and_cuts():
    """Free text becomes one printable line no longer than the limit."""
    assert sf.clip("a\x1b[31mb\nc‮d") == "a [31mb c d"
    assert len(sf.clip("x" * 1000, 50)) == 50
    assert sf.clip(None) == ""


def test_code_scanning_keeps_only_above_warning():
    """Project rule 1: error, or security severity high/critical; closed alerts are dropped."""
    doc = [[_cs(1, "error"), _cs(2, "warning"), _cs(3, "warning", "high"), _cs(4, "note", "medium"),
            _cs(5, "error", state="dismissed")]]
    assert [f["alert"] for f in sf.code_scanning(doc)] == [1, 3]


def test_dependabot_fields_and_paging():
    """Alerts from several pages are flattened; only the fields needed for a fix are kept."""
    alert = {"number": 6, "state": "open",
             "dependency": {"package": {"ecosystem": "pip", "name": "cryptography"}, "manifest_path": "pyproject.toml"},
             "security_advisory": {"ghsa_id": "GHSA-1", "severity": "high", "summary": "s", "description": "LONG " * 500},
             "security_vulnerability": {"vulnerable_version_range": "< 50.0.0",
                                        "first_patched_version": {"identifier": "50.0.0"}}}
    out = sf.dependabot([[alert], [dict(alert, number=7, state="fixed")]])
    assert len(out) == 1
    assert out[0]["fixed_in"] == "50.0.0" and out[0]["package"] == "cryptography"
    assert "description" not in out[0]


def test_pip_audit_groups_by_package_and_needs_a_fix():
    """One entry per package; advisories without a fixed version are left out."""
    doc = {"dependencies": [
        {"name": "a", "version": "1.0", "vulns": [{"id": "V1", "fix_versions": ["1.1"]}, {"id": "V2", "fix_versions": []}]},
        {"name": "b", "version": "2.0", "vulns": [{"id": "V3", "fix_versions": []}]},
        {"name": "c", "version": "3.0", "vulns": []},
    ]}
    out = sf.pip_audit("lowest-allowed", doc)
    assert out == [{"source": "pip-audit:lowest-allowed", "package": "a", "version": "1.0", "advisories": {"V1": ["1.1"]}}]


def test_open_prs_finds_pending_patch_and_dependabot():
    """An open security-patch/ pull request blocks a new run; Dependabot pull requests are listed."""
    prs = [[{"number": 1, "head": {"ref": "security-patch/2026-11-24"}, "user": {"login": "claude[bot]"}, "html_url": "u1"},
            {"number": 2, "head": {"ref": "dependabot/pip/x"}, "user": {"login": "dependabot[bot]"}, "title": "t"},
            {"number": 3, "head": {"ref": "feature"}, "user": {"login": "someone"}}]]
    dep, pending = sf.open_prs(prs)
    assert pending == "u1"
    assert [p["number"] for p in dep] == [2]


def test_main_end_to_end(tmp_path):
    """Missing sources are reported, not fatal; outputs and the work list are written."""
    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps([[_cs(9, "error")]]))
    out, gho = tmp_path / "f.json", tmp_path / "gho"
    assert sf.main(["--dependabot", str(tmp_path / "missing.json"), "--code-scanning", str(cs),
                    "--out", str(out), "--github-output", str(gho)]) == 0
    work = json.loads(out.read_text())
    assert len(work["findings"]) == 1 and work["pending_pr"] == ""
    assert work["problems"] and "FileNotFoundError" in work["problems"][0]
    assert gho.read_text() == "count=1\npending_pr=\n"


def test_main_rejects_bad_pip_audit_argument(tmp_path):
    """--pip-audit needs LABEL=FILE."""
    with pytest.raises(SystemExit):
        sf.main(["--pip-audit", "nofile", "--out", str(tmp_path / "f.json")])

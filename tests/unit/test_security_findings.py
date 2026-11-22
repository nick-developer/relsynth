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
    own, fork = {"full_name": "o/r"}, {"full_name": "evil/r"}
    prs = [[{"number": 0, "head": {"ref": "security-patch/x", "repo": fork}, "user": {"login": "evil"}, "html_url": "u0"},
            {"number": 1, "head": {"ref": "security-patch/2026-11-24", "repo": own}, "user": {"login": "claude[bot]"},
             "html_url": "u1"},
            {"number": 2, "head": {"ref": "dependabot/pip/x", "repo": own}, "user": {"login": "dependabot[bot]"}, "title": "t"},
            {"number": 3, "head": {"ref": "feature", "repo": own}, "user": {"login": "someone"}}]]
    dep, pending = sf.open_prs(prs, "o/r")
    assert pending == "u1"
    assert [p["number"] for p in dep] == [2]


def test_fork_branch_named_security_patch_does_not_block():
    """A fork pull request from a security-patch/ branch is not this job's pull request."""
    prs = [{"number": 9, "head": {"ref": "security-patch/x", "repo": {"full_name": "evil/r"}}, "html_url": "u"}]
    assert sf.open_prs(prs, "o/r") == ([], "")


def test_summary_prints_counts_only():
    """Alert details never reach the public job summary."""
    work = {"findings": [{"source": "code-scanning", "path": "src/secret_place.py", "rule": "py/sql-injection"}],
            "manual": [], "dependabot_prs": [], "pending_pr": "", "problems": []}
    out = sf.summary(work)
    assert "| code-scanning | 1 |" in out
    assert "secret_place" not in out and "sql-injection" not in out


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


def test_protected_findings_go_to_the_maintainer(tmp_path):
    """Workflow, test and tool alerts are not counted: the patch job may not change them."""
    def at(n, path, tool="CodeQL"):
        a = _cs(n, "error")
        a["tool"]["name"] = tool
        a["most_recent_instance"]["location"]["path"] = path
        return a
    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps([[at(1, "src/a.py"), at(2, ".github/workflows/ci.yml", "zizmor"), at(3, "tests/x.py"),
                               at(4, "tools/release.py"), at(5, ".github/requirements/semgrep.txt")]]))
    out = tmp_path / "f.json"
    sf.main(["--code-scanning", str(cs), "--out", str(out)])
    work = json.loads(out.read_text())
    assert [f["alert"] for f in work["findings"]] == [1]
    assert [f["alert"] for f in work["manual"]] == [2, 3, 4, 5]


def test_api_error_body_is_a_problem(tmp_path):
    """gh api writes the error body before failing; it must not read as "no alerts"."""
    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps({"message": "no analysis found", "status": "404"}))
    out = tmp_path / "f.json"
    sf.main(["--code-scanning", str(cs), "--out", str(out)])
    assert "API error: no analysis found" in json.loads(out.read_text())["problems"][0]


# --- .github/scripts/patch_gate.py ---------------------------------------------------

def _gate():
    pytest.importorskip("tomllib")   # Python 3.11+, as on the runner
    p = Path(__file__).parents[2] / ".github" / "scripts" / "patch_gate.py"
    spec = importlib.util.spec_from_file_location("patch_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A small git repository with a pyproject.toml, a source file and a test."""
    import subprocess
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["a>=1"]\n[tool.ruff]\nline-length = 100\n')
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text("def test(): pass\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    monkeypatch.chdir(tmp_path)
    return git


def _commit(git, msg="patch"):
    git("add", "-A")
    git("commit", "-qm", msg)


def test_gate_accepts_a_dependency_floor_and_a_code_fix(repo, tmp_path):
    """Dependencies and source code may change."""
    gate = _gate()
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["a>=2"]\n[tool.ruff]\nline-length = 100\n')
    (tmp_path / "src" / "m.py").write_text("x = 2\n")
    _commit(repo)
    assert gate.problems("HEAD~1") == []
    assert gate.main(["HEAD~1"]) == 0


@pytest.mark.parametrize("change", ["tool", "test", "moved_test", "quoted", "workflow", "symlink"])
def test_gate_refuses(repo, tmp_path, change):
    """Tool settings, tests (also moved or oddly named), .github/ and symbolic links are refused."""
    import os
    gate = _gate()
    if change == "tool":
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["a>=1"]\n[tool.ruff]\nline-length = 300\n')
    elif change == "test":
        (tmp_path / "tests" / "test_m.py").write_text("")
    elif change == "moved_test":
        repo("mv", "tests/test_m.py", "src/moved.py")
    elif change == "quoted":
        (tmp_path / "tests" / "t\u00e9 x.py").write_text("")
    elif change == "workflow":
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "ci.yml").write_text("on: push\n")
    elif change == "symlink":
        os.symlink("../tests", tmp_path / "src" / "linked")
    _commit(repo)
    assert gate.problems("HEAD~1")
    assert gate.main(["HEAD~1"]) == 1

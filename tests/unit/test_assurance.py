# SPDX-License-Identifier: Apache-2.0
"""The claim register (docs/assurance/claims.yaml) and the facts the assurance
case relies on that no other test covers.

* Every registered claim still appears in its document and every piece of
  evidence resolves; every claim-like sentence of the documents in scope is
  registered (tools/claims_check.py, also run by the CI job ``lint``).
* The checker itself: proposition numbering as LaTeX prints it, and each kind
  of drift it must catch.
* The package opens network connections only in the modules the security
  guide names (docs/security.md, "Supply chain and code").
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _checker():
    spec = importlib.util.spec_from_file_location("claims_check", ROOT / "tools" / "claims_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["claims_check"] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _checker()


def test_claim_register_traces_every_claim():
    res = cc.run(ROOT)
    assert res.problems == [], "\n".join(res.problems)
    assert len(res.claims) >= 90


def test_proposition_numbers_follow_latex_counters():
    props = cc.proposition_numbers(ROOT)
    assert props["prop:pit"] == 1
    assert props["prop:pdsuff"] == 100 and props["prop:pdeligible"] == 105
    assert props["prop:pusplit"] == 600 and props["prop:puprofile"] == 605       # out-of-order setcounter
    assert props["prop:ledger"] == 800 and props["prop:govchain"] == 803
    assert props["prop:dgauss"] == 900 and props["prop:run"] == 906
    assert props["prop:tschange"] == 1200
    assert props["prop:irround"] == 106                                          # restored after a numbered block
    assert len(set(props.values())) == len(props), "two propositions print the same number"


def _mini_repo(tmp_path: Path, register: str, doc: str) -> Path:
    (tmp_path / "docs/assurance").mkdir(parents=True)
    (tmp_path / "docs/proofs").mkdir()
    (tmp_path / "legacy/checks").mkdir(parents=True)
    (tmp_path / "tests/unit").mkdir(parents=True)
    (tmp_path / "docs/proofs/relsynth-proofs.tex").write_text(
        "\\begin{proposition}\\label{prop:a}x\\end{proposition}\n"
        "\\newcounter{sv}\\setcounter{sv}{\\value{proposition}}\\setcounter{proposition}{899}\n"
        "\\begin{proposition}[T]\\label{prop:b}y\\end{proposition}\n"
        "\\setcounter{proposition}{\\value{sv}}\n"
        "\\begin{proposition}\\label{prop:c}z\\end{proposition} % \\begin{proposition}\n",
        encoding="utf-8")
    (tmp_path / "legacy/checks/run_checks.py").write_text('@check("C01", "x", "Prop. 1")\n', encoding="utf-8")
    (tmp_path / "tests/unit/test_x.py").write_text(
        "def test_c900_b():\n    pass\n\nclass TestK:\n    def test_m(self):\n        pass\n", encoding="utf-8")
    (tmp_path / "docs/guide.md").write_text(doc, encoding="utf-8")
    (tmp_path / "docs/assurance/claims.yaml").write_text(register, encoding="utf-8")
    return tmp_path


GOOD = """
scope: [docs/guide.md]
claims:
  - id: CL-1
    doc: docs/guide.md
    quote: Every key resolves.
    status: supported
    evidence: [{prop: prop:b}, {check: C900}, {check: C01}, {test: "tests/unit/test_x.py::TestK::test_m"}]
not_claims:
  - {doc: docs/guide.md, quote: "Run it every night.", reason: instruction}
"""


def test_checker_accepts_a_traced_document(tmp_path):
    root = _mini_repo(tmp_path, GOOD, "# Guide\n\nEvery key\nresolves. Run it every night.\n\nSee Prop. 900 and Props. 1–2.\n")
    res = cc.run(root)
    assert res.problems == []
    assert cc.proposition_numbers(root) == {"prop:a": 1, "prop:b": 900, "prop:c": 2}
    assert res.claims[0]["evidence"][0] == {"kind": "prop", "ref": "prop:b", "number": 900}


@pytest.mark.parametrize(("doc", "register", "expect"), [
    ("Every key is unique. Run it every night.\n", GOOD, "quote not found"),
    ("Every key resolves. Run it every night. Rows are never copied.\n", GOOD, "unregistered claim"),
    ("Every key resolves. Run it every night. See Prop. 3.\n", GOOD, "Proposition 3, which the proofs do not define"),
    ("Every key resolves. Run it every night.\n", GOOD.replace("prop:b", "prop:zz"), "no proposition labelled prop:zz"),
    ("Every key resolves. Run it every night.\n", GOOD.replace("C900", "C901"), "no check C901"),
    ("Every key resolves. Run it every night.\n", GOOD.replace("TestK::test_m", "TestK::test_n"), "has no TestK::test_n"),
    ("Every key resolves. Run it every night.\n",
     GOOD.replace('[{prop: prop:b}, {check: C900}, {check: C01}, {test: "tests/unit/test_x.py::TestK::test_m"}]',
                  "[{source: 'EDPB 02/2026 para. 55'}]"), "needs a prop, check, test or report"),
    ("Every key resolves. Run it every night.\n", GOOD.replace("reason: instruction", "reason: ''"), "give a reason"),
])
def test_checker_catches_drift(tmp_path, doc, register, expect):
    res = cc.run(_mini_repo(tmp_path, register, doc))
    assert any(expect in p for p in res.problems), res.problems


SUMMARY = """
  - id: CL-2
    doc: docs/guide.md
    quote: Keys resolve in the summary.
    status: supported
    evidence: [{claim: CL-1}]
"""


@pytest.mark.parametrize(("cited", "expect"), [
    ("CL-1", None),
    ("CL-9", "cites claim CL-9, which the register does not have"),
    ("CL-3", "cites claim CL-3, which has no prop, check, test or report of its own"),
])
def test_a_summary_claim_is_as_strong_as_the_claim_it_cites(tmp_path, cited, expect):
    weak = "  - {id: CL-3, doc: docs/guide.md, quote: Every key resolves., status: scope, evidence: [{source: x}]}\n"
    register = GOOD.replace("not_claims:", SUMMARY.replace("CL-1}", cited + "}").lstrip("\n") + weak + "not_claims:")
    doc = "Every key resolves. Run it every night. Keys resolve in the summary.\n"
    problems = cc.run(_mini_repo(tmp_path, register, doc)).problems
    if expect is None:
        assert problems == []
    else:
        assert any(expect in p for p in problems), problems


def test_checker_catches_bad_citations_in_the_assurance_documents(tmp_path):
    root = _mini_repo(tmp_path, GOOD, "Every key resolves. Run it every night.\n")
    (root / "docs/assurance/case.md").write_text(
        "G1 rests on CL-1, PRV-99, `prop:b`, `prop:gone`, C900 and C777.\n", encoding="utf-8")
    problems = cc.run(root).problems
    assert any("cites PRV-99" in p for p in problems)
    assert any("cites prop:gone" in p for p in problems)
    assert any("cites check C777" in p for p in problems)
    assert not any("prop:b," in p or "C900" in p for p in problems)


# Modules allowed to open network connections, and why (docs/security.md).
NETWORK_ALLOWED = {
    "relsynth/service/auth.py": "OIDC discovery and keys of the identity provider the operator configures",
    "relsynth/service/secrets.py": "the Vault server named by VAULT_ADDR",
    "relsynth/service/telemetry.py": "OpenTelemetry export, only when otel is turned on",
}
NETWORK_MODULES = ("urllib.request", "http.client", "requests", "httpx", "aiohttp", "urllib3", "opentelemetry.exporter")


def test_network_access_is_confined_to_named_modules():
    found = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT / "src").as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Attribute) and node.attr in ("create_connection", "connect") and \
                    isinstance(node.value, ast.Name) and node.value.id == "socket":
                names = ["socket." + node.attr]
            hits = [n for n in names if n.startswith(NETWORK_MODULES) or n.startswith("socket.")]
            if hits:
                found.setdefault(rel, set()).update(hits)
    assert set(found) <= set(NETWORK_ALLOWED), {k: v for k, v in found.items() if k not in NETWORK_ALLOWED}

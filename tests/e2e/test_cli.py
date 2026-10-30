# SPDX-License-Identifier: Apache-2.0
"""The command line, as a CI/CD job uses it: every command, its JSON output and
its exit code (0 ok, 1 quality gate, 2 spec/usage, 3 connection/I-O, 4 engine)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from relsynth.cli import main


def cli(capsys, *args):
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def jcli(capsys, *args):
    if args and args[0] == "run" and "--bundle" not in args:
        args = (*args, "--bundle", "")          # no bundle file in the working directory
    code, out, err = cli(capsys, *args, "--json")
    return code, json.loads(out), err


@pytest.fixture
def spec_file(tmp_path, star_sqlite):
    spec, _ = star_sqlite
    s = dict(spec, target={"kind": "csv", "path": str(tmp_path / "out")})
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(s, sort_keys=False))
    return p


def test_console_script_runs_as_a_process():
    r = subprocess.run([sys.executable, "-m", "relsynth", "--version"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, check=False)
    assert r.returncode == 0 and r.stdout.startswith("relsynth ")


def test_version_and_help(capsys):
    code, j, _ = jcli(capsys, "version")
    assert code == 0 and j["version"].count(".") == 2 and "numpy" in j
    assert cli(capsys, "--help")[0] == 0
    assert cli(capsys, "no-such-command")[0] == 2


def test_introspect_check_plan(capsys, tmp_path, star_sqlite):
    _, db = star_sqlite
    out = tmp_path / "draft.yaml"
    code, j, _ = jcli(capsys, "introspect", "--source", f"sqlite:{db}", "--out", out)
    assert code == 0 and out.exists() and j["tables"] == 3 and j["relationships"] == 2
    code, j, _ = jcli(capsys, "check", out)
    assert code == 0 and j["valid"]
    code, j, _ = jcli(capsys, "check", out, "--connect")
    assert code == 0
    code, j, _ = jcli(capsys, "plan", out, "--pushdown", "on")
    assert code == 0 and "sales" in j["pushdown"]["pushed"] and j["rows"]["sales"] > 0


def test_run_writes_data_bundle_and_report(capsys, tmp_path, spec_file):
    rep = tmp_path / "report.md"
    bundle = tmp_path / "b.json"
    code, j, _ = jcli(capsys, "run", spec_file, "--seed", "1", "--bundle", bundle, "--report", rep,
                      "--no-privacy", "--set", "synthesis.scale.default=0.5")
    assert code == 0, j
    assert j["scorecard"]["passed"] == j["scorecard"]["total"]
    assert (tmp_path / "out" / "sales.csv").exists() and bundle.exists() and rep.read_text().startswith("#")
    cust = pd.read_csv(tmp_path / "out" / "customer.csv")
    assert abs(len(cust) - 750) <= 1                     # --set reached the engine
    code, j, _ = jcli(capsys, "evaluate", spec_file, "--synthetic", f"csv:{tmp_path / 'out'}", "--no-privacy")
    assert code == 0 and j["scorecard"]["total"] > 0
    qcfg = tmp_path / "q.yaml"
    qcfg.write_text("null_splits: 3\nmulti_splits: 3\n")
    code, j, _ = jcli(capsys, "evaluate", spec_file, "--synthetic", f"csv:{tmp_path / 'out'}", "--no-privacy",
                      "--quality", str(tmp_path / "q.json"), "--quality-config", str(qcfg))
    assert code == 0, j
    assert j["quality"]["verdict"]["rule"] and (tmp_path / "q.json").exists()
    # the opt-in 2.0 gate: zero tolerances without calibration fail the fidelity area
    strict = tmp_path / "strict.yaml"
    strict.write_text("calibrate: false\ntolerances: {w1_raw_sd: 0, w1_rank_normal: 0, js_distance: 0}\n")
    code, j, _ = jcli(capsys, "evaluate", spec_file, "--synthetic", f"csv:{tmp_path / 'out'}", "--no-privacy",
                      "--quality", str(tmp_path / "q2.json"), "--quality-config", str(strict), "--quality-gate")
    assert code == 1 and "fidelity" in j["gate"]["failed_areas"], j.get("gate")
    code, _, _ = jcli(capsys, "evaluate", spec_file, "--synthetic", f"csv:{tmp_path / 'out'}", "--quality-gate")
    assert code == 2
    code, j, _ = jcli(capsys, "doctor", bundle)
    assert code == 0 and "issues" in j


def test_quality_gate_exit_code(capsys, spec_file):
    # an impossible gate: more than every check
    code, j, err = jcli(capsys, "run", spec_file, "--no-write", "--no-privacy", "--fail-under", "1.01")
    assert code == 1 and j["gate"] == {"fail_under": 1.01, "passed": False}


def test_spec_errors_exit_2(capsys, tmp_path, spec_file):
    code, j, err = jcli(capsys, "run", spec_file, "--set", "synthesis.pushdown.mode=sometimes")
    assert code == 2 and "pushdown.mode" in j["error"] and "invalid spec" in err
    bad = tmp_path / "bad.yaml"
    bad.write_text("tables: [unclosed\n")
    assert jcli(capsys, "check", bad)[0] == 2
    assert jcli(capsys, "check", tmp_path / "missing.yaml")[0] == 2
    assert cli(capsys, "run")[0] == 2                    # usage error


def test_connection_errors_exit_3_and_mask_passwords(capsys, tmp_path, spec_file):
    code, j, err = jcli(capsys, "ping", "--source", "postgresql://scott:tiger@127.0.0.1:9/nodb?connect_timeout=2")
    assert code in (0, 3)                                # ping reports instead of raising
    assert "tiger" not in json.dumps(j) and "tiger" not in err
    code, j, err = jcli(capsys, "profile", "--source", "postgresql://scott:tiger@127.0.0.1:9/nodb?connect_timeout=2")
    assert code == 3 and "tiger" not in j["error"] and "tiger" not in err
    code, j, _ = jcli(capsys, "run", spec_file, "--set", f"source.path={tmp_path / 'nope.db'}")
    assert code == 3


def test_model_export_profile_ping_clone(capsys, tmp_path, spec_file, star_sqlite):
    _, db = star_sqlite
    code, j, _ = jcli(capsys, "model", spec_file, "--out", tmp_path / "m")
    assert code == 0 and (tmp_path / "m.md").exists()
    code, j, _ = jcli(capsys, "export-model", spec_file, "--out", tmp_path / "x", "--formats", "json,md,py",
                      "--draws", "2", "--no-replica-check")
    assert code == 0 and (tmp_path / "x").is_dir() and any((tmp_path / "x").glob("*.py"))
    code, j, _ = jcli(capsys, "profile", "--source", f"sqlite:{db}", "--output", tmp_path / "p.json")
    assert code == 0 and json.loads((tmp_path / "p.json").read_text())["sales"]["rows"] > 0
    code, j, _ = jcli(capsys, "ping", "--source", f"sqlite:{db}")
    assert code == 0 and j["ok"]
    code, j, _ = jcli(capsys, "clone-schema", "--source", f"sqlite:{db}", "--target", f"sqlite:{tmp_path / 'e.db'}")
    assert code == 0 and set(j["tables"]) == {"customer", "product", "sales"}


def test_tune(capsys, tmp_path, spec_file):
    code, j, _ = jcli(capsys, "tune", spec_file, "--rounds", "1", "--out", tmp_path / "tuned.yaml")
    assert code == 0 and (tmp_path / "tuned.yaml").exists()


def test_doctor_reports_and_applies_patches(capsys, tmp_path, spec_file, monkeypatch):
    """doctor prints each issue and --apply writes the spec with the merged patches."""
    import importlib
    D = importlib.import_module("relsynth.doctor")
    bundle = tmp_path / "b.json"
    assert jcli(capsys, "run", spec_file, "--no-write", "--no-privacy", "--bundle", bundle)[0] == 0
    issue = D.Issue(id="D05", severity="warn", kind="spec", where="customer.segment", message="drift",
                    patches=[{"path": "synthesis.marginals.dirichlet_prior", "value": 0.125}])
    monkeypatch.setattr(D, "diagnose", lambda b: [issue])
    fixed = tmp_path / "fixed.yaml"
    code, j, _ = jcli(capsys, "doctor", bundle, "--spec", spec_file, "--apply", "--out", fixed)
    assert code == 0, j
    assert j["issues"][0]["message"] == "drift" and j["fixed_spec"] == str(fixed)
    assert yaml.safe_load(fixed.read_text())["synthesis"]["marginals"]["dirichlet_prior"] == 0.125
    assert jcli(capsys, "check", fixed)[0] == 0


@pytest.fixture
def ir_spec_file(tmp_path, spec_file):
    """spec_file as an IR document with annotations a 1.0 spec cannot carry."""
    from relsynth import ir
    from relsynth.ir.migrate import migrate_doc
    est = migrate_doc(yaml.safe_load(spec_file.read_text()), privacy_unit="customer").estate
    est.entities["product"].extensions["x-owner"] = "catalogue team"
    p = tmp_path / "spec.ir.yaml"
    ir.dump(est, p)
    return p


def test_doctor_and_tune_write_ir_when_given_ir(capsys, tmp_path, ir_spec_file, monkeypatch):
    """doctor --apply and tune keep an IR document an IR document, with its annotations."""
    import importlib

    from relsynth import ir
    D = importlib.import_module("relsynth.doctor")
    bundle = tmp_path / "b.json"
    assert jcli(capsys, "run", ir_spec_file, "--no-write", "--no-privacy", "--bundle", bundle)[0] == 0
    issue = D.Issue(id="D05", severity="warn", kind="spec", where="customer.segment", message="drift",
                    patches=[{"path": "synthesis.marginals.dirichlet_prior", "value": 0.125}])
    monkeypatch.setattr(D, "diagnose", lambda b: [issue])
    fixed = tmp_path / "fixed.yaml"
    assert jcli(capsys, "doctor", bundle, "--spec", ir_spec_file, "--apply", "--out", fixed)[0] == 0
    doc = yaml.safe_load(fixed.read_text())
    assert ir.is_ir(doc) and doc["synthesis"]["marginals"]["dirichlet_prior"] == 0.125
    assert doc["privacy"]["unit"]["entity"] == "customer"
    assert doc["entities"]["product"]["x-owner"] == "catalogue team"
    assert jcli(capsys, "check", fixed)[0] == 0
    tuned = tmp_path / "tuned.yaml"
    assert jcli(capsys, "tune", ir_spec_file, "--rounds", "1", "--out", tuned)[0] == 0
    doc = yaml.safe_load(tuned.read_text())
    assert ir.is_ir(doc) and doc["privacy"]["unit"]["entity"] == "customer"
    assert jcli(capsys, "check", tuned)[0] == 0


def test_verify_command(capsys, tmp_path, spec_file):
    rep = tmp_path / "verify.md"
    code, j, _ = jcli(capsys, "verify", spec_file, "--draws", "2", "--no-stability", "--no-privacy", "--report", rep)
    v = j["verify"]
    assert code == (0 if v["passed"] == v["total"] else 1) and v["total"] >= 5
    assert {c["id"] for c in v["checks"]} >= {"R1", "R3", "R4", "R5"}
    assert rep.read_text().startswith("# Model checks")


def test_locations_from_environment(capsys, tmp_path, spec_file, star_sqlite, monkeypatch):
    """--source-url-env / --target-url-env alone name the location; the URL is never printed."""
    _, db = star_sqlite
    monkeypatch.setenv("RS_SRC", f"sqlite:///{db}")
    code, j, _ = jcli(capsys, "ping", "--source-url-env", "RS_SRC")
    assert code == 0 and j["ok"]
    monkeypatch.setenv("RS_TGT", f"sqlite:///{tmp_path / 'env.db'}")
    code, j, _ = jcli(capsys, "run", spec_file, "--target-url-env", "RS_TGT", "--no-privacy", "--bundle", "")
    assert code == 0 and (tmp_path / "env.db").exists()
    assert jcli(capsys, "ping", "--source-url-env", "RS_UNSET_VARIABLE")[0] == 2
    assert jcli(capsys, "ping")[0] == 2


def test_if_exists_fail_is_an_io_error(capsys, tmp_path, spec_file):
    out = tmp_path / "o"
    assert jcli(capsys, "run", spec_file, "--out", out, "--no-privacy", "--bundle", "")[0] == 0
    spec = yaml.safe_load(open(spec_file))
    spec["target"] = {"kind": "csv", "path": str(out), "if_exists": "fail"}
    p = tmp_path / "fail.yaml"
    p.write_text(yaml.safe_dump(spec))
    code, j, _ = jcli(capsys, "run", p, "--no-privacy", "--bundle", "")
    assert code == 3 and "if_exists: fail" in j["errors"][0]["error"]

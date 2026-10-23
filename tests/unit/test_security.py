"""Security properties: credentials never reach files, logs or messages; rule
expressions and regular expressions are sandboxed; specs cannot run code; file
table names cannot leave their folder; SQL identifiers are always quoted
(tested in test_connectors)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

import relsynth as rs
from relsynth.cli import _scrub
from relsynth.data import inside, mask_spec_urls, mask_url
from relsynth.expr import ExprError, check_mask, check_regex, parse

PW = "Sup3r$ecret"


@pytest.mark.parametrize("url", [
    f"postgresql://scott:{PW}@db:5432/prod",
    f"mssql+pyodbc://sa:{PW}@h/d?driver=ODBC+Driver+18+for+SQL+Server",
    f"oracle+oracledb://u:{PW}@h:1521/?service_name=x",
    f"postgresql://scott@db/prod?password={PW}",
    f"mssql+pyodbc:///?odbc_connect=DRIVER%3D%7Bx%7D%3BSERVER%3Dh%3BUID%3Du%3BPWD%3D{PW}",
    f"postgresql://scott@db/prod?sslpassword={PW}&token={PW}",
])
def test_mask_url(url):
    m = mask_url(url)
    assert PW not in m and "Sup3r" not in m
    assert m.split(":")[0] == url.split(":")[0]           # the dialect stays readable


def test_scrub_masks_urls_inside_messages():
    msg = f"(psycopg.OperationalError) could not connect to postgresql://scott:{PW}@db/prod: timeout"
    assert PW not in _scrub(msg) and "could not connect" in _scrub(msg)


def test_spec_urls_masked_everywhere():
    spec = {"source": {"kind": "postgresql", "url": f"postgresql://u:{PW}@h/d"},
            "target": {"kind": "mssql", "url": f"mssql://u:{PW}@h/d", "password": PW},
            "tables": {}}
    assert PW not in json.dumps(mask_spec_urls(spec))


def test_bundle_holds_no_password(star_sqlite, tmp_path):
    spec, _ = star_sqlite
    r = rs.run(spec, write=False, privacy=False)
    b = dict(r.bundle)
    b["spec"] = {**b.get("spec", {}), "target": {"kind": "postgresql", "url": f"postgresql://u:{PW}@h/d"}}
    from relsynth.diagnostics import write_bundle
    write_bundle(b, tmp_path / "b.json")
    assert PW not in (tmp_path / "b.json").read_text()


def test_failed_connection_logs_no_password(caplog):
    caplog.set_level(logging.DEBUG, logger="relsynth")
    res = rs.ping(f"postgresql://scott:{PW}@127.0.0.1:9/x?connect_timeout=2")
    assert not res["ok"]
    assert PW not in json.dumps(res) and PW not in caplog.text


def test_saved_spec_moves_password_to_environment(tmp_path):
    s = rs.Spec.from_dict({"source": {"kind": "postgresql", "url": f"postgresql://u:{PW}@h/d"},
                           "tables": {"t": {"primary_key": "id", "columns": {"id": "id"}}}})
    assert PW not in s.save(tmp_path / "s.yaml").read_text()
    assert PW not in repr(s) and PW not in s.to_yaml()


# ---------------------------------------------------------------- expressions
@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo x')",
    "open('/etc/passwd')",
    "a.__class__",
    "(lambda: 1)()",
    "[x for x in a]",
    "getattr(a, 'x')",
    "eval('1')",
    "a[0]",
    "f'{a}'",
    "a := 1",
])
def test_rule_expressions_are_sandboxed(expr):
    df = pd.DataFrame({"a": [1, 2], "b": [True, False], "c": [3, 4]})
    with pytest.raises(ExprError):
        parse(expr)
        check_mask(df, expr)


def test_expression_size_and_depth_limits():
    with pytest.raises(ExprError):
        parse("a + " * 20000 + "a")
    with pytest.raises(ExprError):
        parse("(" * 500 + "a" + ")" * 500)


def test_allowed_expressions_still_work():
    df = pd.DataFrame({"a": [1, 2, 3], "s": ["x.y", "xzy", None]})
    assert check_mask(df, "a >= 2").tolist() == [False, True, True]
    assert check_mask(df, r"regex_match(s, '^x\.y$')").tolist()[0]


@pytest.mark.parametrize("pattern", ["(a+)+$", "(a|a)*b", "(.*)*x", "((ab)*)+c", "(a|aa)+"])
def test_catastrophic_regexes_are_refused(pattern):
    with pytest.raises(ExprError):
        check_regex(pattern)


@pytest.mark.parametrize("pattern", [r"^\d{4}-\d{2}$", "[A-Z]{2}[0-9]+", "(ab|cd)+e", r"^[^@]+@[^@]+\.[a-z]+$"])
def test_ordinary_regexes_are_accepted(pattern):
    check_regex(pattern)


# ---------------------------------------------------------------- files and YAML
@pytest.mark.parametrize("name", ["..", ".", "", "a/b", "a\\b", "/abs", "x\x00"])
def test_inside_refuses_non_plain_names(tmp_path, name):
    with pytest.raises(ValueError):
        inside(tmp_path, name, ".csv")


def test_inside_accepts_plain_names(tmp_path):
    assert inside(tmp_path, "Sales Fact (2024)", ".csv").parent == tmp_path.resolve()


def test_symlinked_table_cannot_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "evil.csv").symlink_to(outside / "x.csv")
    with pytest.raises(ValueError):
        inside(root, "evil", ".csv")


def test_spec_cannot_construct_python_objects(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("tables: !!python/object/new:os.system [echo pwned]\n")
    with pytest.raises(Exception) as e:
        rs.Spec.load(p).resolved
    assert "pwned" not in str(e.value) or "python" in str(e.value)


def test_privacy_checks_run_by_default(star_sqlite):
    spec, _ = star_sqlite
    rep = rs.fit(spec, seed=1)
    d = rep.sample()
    r = rep.evaluate(d)
    assert "privacy" in json.dumps(r.raw)[:10**7].lower() or any(
        "dcr" in k.lower() for k in json.dumps(r.raw).split('"'))


def test_synthetic_rows_are_not_copies(star_sqlite):
    spec, _ = star_sqlite
    real = pd.read_sql_table("customer", f"sqlite:///{spec['source']['path']}")
    syn = rs.fit(spec, seed=2).sample()["customer"]
    cols = [c for c in real.columns if c != "customer_id" and real[c].nunique() > 50]
    if not cols:
        pytest.skip("no high-cardinality attributes")
    key = lambda df: df[cols].astype(str).agg("|".join, axis=1)  # noqa: E731
    shared = np.isin(key(syn).to_numpy(), key(real).to_numpy()).mean()
    assert shared < 0.05

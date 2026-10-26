# SPDX-License-Identifier: Apache-2.0
"""v0.9 input/output, security and spec-validation fixes: one test per finding of the
review (S1-S9 security, C1-C7 data loading, E1-E6 and P1 expressions, V1-V3 spec
validation, R readability).  Each test builds its case in a temporary folder (no
real database needed) and returns [(what, value, target, ok), ...].

    PYTHONPATH=. python3 checks/v09_io.py            # every test
    PYTHONPATH=. python3 checks/v09_io.py t_s3_sql   # selected tests
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from relsynth.data import (canonical_keys, coerce, load_tables, mask_spec_urls, mask_url,  # noqa: E402
                           read_table_file, write_tables)
from relsynth.expr import Evaluator, ExprError, check_mask, check_regex, derive_values, names_in  # noqa: E402
from relsynth.spec import SpecError, load_spec  # noqa: E402

os.environ.setdefault("RELSYNTH_QUIET", "1")


def _tmp():
    return Path(tempfile.mkdtemp(prefix="v09io-"))


def _raises(fn, exc=Exception, text=None):
    """(raised, message): whether fn() raised exc (and the message holds text)."""
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return (text is None or text in str(e)), str(e)[:160]
    return False, "no error"


def _spec_err(spec):
    try:
        load_spec(spec)
    except SpecError as e:
        return str(e)
    return ""


BASE = {"source": {"kind": "csv", "path": "."},
        "tables": {"c": {"primary_key": "id", "columns": {"id": "id", "v": "integer"}},
                   "f": {"role": "fact", "columns": {"cid": "id", "amt": "float"}}},
        "relationships": [{"child": "f", "child_key": "cid", "parent": "c"}]}


def _base(**over):
    s = json.loads(json.dumps(BASE))
    s.update(over)
    return s


# ====================================================================== security
def t_s1_latex():
    """A data value with a newline and \\end{verbatim} cannot leave the verbatim
    block; pdflatex runs without shell escape in a private folder."""
    from relsynth.export import _col_summary, to_latex, write_pdf
    w = _tmp()
    level = ('x\n    \\end{verbatim}\\newwrite\\f\\immediate\\openout\\f=numpy.py'
             '\\immediate\\write\\f{print(1)}\\immediate\\closeout\\f\\begin{verbatim}')
    line = _col_summary({"kind": "discrete", "levels": ["plain", level]}, {"p": [0.5, 0.5], "p_null": 0})
    md = "# t\n\n- seg: " + line + "\n\n    code block\n    \\end{verbatim}\\input{/etc/hostname}\n"
    tex = to_latex(md, "t")
    one_line = "\n" not in line
    bal = tex.count("\\begin{verbatim}") == tex.count("\\end{verbatim}") == 1
    raw_cmd = any(ln.startswith("\\newwrite") or ln.startswith("    \\end{verbatim}") for ln in tex.splitlines())
    out = [("column summary is one line", one_line, True, one_line),
           ("verbatim blocks balanced (inner \\end{verbatim} neutralised)", bal, True, bal),
           ("no LaTeX command from data at a line start", raw_cmd, False, not raw_cmd)]
    if shutil.which("pdflatex"):
        p = w / "model.tex"
        p.write_text(tex)
        # a crafted .tex (as if the escaping failed) must still not write next to model.tex
        evil = w / "evil.tex"
        evil.write_text("\\documentclass{article}\\begin{document}\\newwrite\\f\\immediate\\openout\\f=numpy.py"
                        "\\immediate\\write\\f{print(1)}\\immediate\\closeout\\f x\\end{document}\n")
        pdf = write_pdf(p)
        write_pdf(evil)
        files = sorted(x.name for x in w.iterdir())
        ok = pdf is not None and "numpy.py" not in files and not any(f.endswith((".aux", ".log")) for f in files)
        out.append(("files in the export folder after pdflatex", files, "model.tex/pdf, evil.tex/pdf only", ok))
    shutil.rmtree(w, ignore_errors=True)
    return out


def t_s2_run_model():
    """Nothing from the spec is formatted into run_model.py's code."""
    from relsynth.export import RUN_MODEL, _code_text
    evil = 'x"""\nprint("INJECTED")\n"""\\'
    src = RUN_MODEL.format(spec=_code_text(evil), datasets=int("3"))
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    top = [type(n).__name__ for n in tree.body]
    injected = "print(\"INJECTED\")" in src.replace(doc, "") or top.count("Expr") > 2
    err = _spec_err(_base(synthesis={"posterior": {"datasets": '1\n"""\nprint(1)\n"""'}}))
    return [("generated script parses, name only in the docstring", injected, False, not injected),
            ("name sanitised (no quotes, backslashes, newlines)", _code_text(evil), "x___ print_INJECTED_ ____",
             not any(ch in _code_text(evil) for ch in "\"'\\\n")),
            ("posterior.datasets with code refused by the spec", "datasets must be a positive integer" in err, True,
             "datasets must be a positive integer" in err)]


def t_s3_sql():
    """Table names are reflected, never pasted into SQL: injection fails, odd names work."""
    w = _tmp()
    db = w / "src.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE cust(a INTEGER PRIMARY KEY, b TEXT); INSERT INTO cust VALUES (1,'ok'),(2,'ok');
        CREATE TABLE salaries(a INTEGER, b TEXT); INSERT INTO salaries VALUES (99,'secret');
        CREATE TABLE "Order Details"(a INTEGER PRIMARY KEY, b TEXT); INSERT INTO "Order Details" VALUES (1,'x');
        CREATE TABLE y(a INTEGER PRIMARY KEY, b TEXT); INSERT INTO y VALUES (1,'y');""")
    con.commit()
    con.close()

    def run(t):
        spec = load_spec({"source": {"kind": "sqlite", "path": str(db)},
                          "tables": {t: {"primary_key": "a", "columns": {"a": "id", "b": "categorical"}}}})
        return load_tables(spec)[t]
    union, _ = _raises(lambda: run("cust UNION SELECT a, b FROM salaries"))
    stacked, _ = _raises(lambda: run("cust; DROP TABLE y"))
    legit = run("Order Details").to_dict("records")
    alive = sqlite3.connect(db).execute("select count(*) from y").fetchone()[0]
    shutil.rmtree(w, ignore_errors=True)
    return [("UNION through a table name refused", union, True, union),
            ("stacked statement refused", stacked, True, stacked),
            ("table y still there", alive, 1, alive == 1),
            ("table 'Order Details' reads", legit, "[{a: 1, b: x}]", len(legit) == 1 and legit[0]["b"] == "x")]


def t_s4_credentials():
    """Passwords are masked in every URL written for sharing; url_env supplies the real one."""
    from relsynth import cli
    from relsynth.diagnostics import clean_spec
    cases = {"postgresql://scott:tiger@db/prod": "postgresql://scott:***@db/prod",
             "postgresql://scott:p@ss@db/prod": "postgresql://scott:***@db/prod",
             "postgresql://:tiger@db/prod": "postgresql://:***@db/prod",
             "oracle+oracledb://scott@db/?password=tiger": "oracle+oracledb://scott@db/?password=***",
             "mssql+pyodbc://db/x?pwd=tiger&driver=ODBC": "mssql+pyodbc://db/x?pwd=***&driver=ODBC",
             "postgresql://scott@db/prod?sslmode=require": "postgresql://scott@db/prod?sslmode=require",
             "sqlite:////tmp/a.db": "sqlite:////tmp/a.db"}
    got = {u: mask_url(u) for u in cases}
    bad = [u for u in cases if got[u] != cases[u]]
    odbc = mask_url("mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BODBC%7D%3BSERVER%3Ddb%3BUID%3Dsa%3BPWD%3Dtiger")
    out = [("URL forms masked as expected", bad, [], not bad),
           ("odbc_connect PWD masked", "tiger" not in odbc and "PWD%3D%2A%2A%2A" in odbc.upper().replace("***", "%2A%2A%2A"),
            True, "tiger" not in odbc)]
    spec = {"source": {"kind": "sqlalchemy", "url": "postgresql://scott:tiger@db/prod"},
            "target": {"kind": "sqlalchemy", "url": "postgresql://u:pw2@db/out"}, "tables": {}}
    cs = json.dumps(clean_spec(spec))
    ms = mask_spec_urls(spec)
    out.append(("bundle spec without passwords", "tiger" in cs or "pw2" in cs, False, "tiger" not in cs and "pw2" not in cs))
    out.append(("shared spec names url_env", [ms["source"].get("url_env"), ms["target"].get("url_env")],
                ["RELSYNTH_SOURCE_URL", "RELSYNTH_TARGET_URL"],
                ms["source"].get("url_env") == "RELSYNTH_SOURCE_URL" and ms["target"].get("url_env") == "RELSYNTH_TARGET_URL"))
    # introspect: the draft spec is written masked
    w = _tmp()
    orig = cli.introspect
    cli.introspect = lambda source, **kw: {"name": "d", "source": dict(source), "tables": {
        "c": {"primary_key": "id", "columns": {"id": "id"}}}, "relationships": [], "review": []}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["introspect", "--source", "postgresql://scott:Pa55w0rd@db/prod", "--out", str(w / "spec.yaml")])
    finally:
        cli.introspect = orig
    txt = (w / "spec.yaml").read_text()
    out.append(("introspect draft without the password", "Pa55w0rd" in txt, False,
                "Pa55w0rd" not in txt and "url_env: RELSYNTH_SOURCE_URL" in txt))
    # url_env: the real URL from the environment
    db = w / "s.db"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE c(id INTEGER PRIMARY KEY, v INT); INSERT INTO c VALUES (1, 2), (2, 3);")
    con.commit()
    con.close()
    os.environ["V09_TEST_URL"] = f"sqlite:///{db}"
    try:
        sp = load_spec({"source": {"kind": "sqlalchemy", "url": "postgresql://u:***@x/y", "url_env": "V09_TEST_URL"},
                        "tables": {"c": {"primary_key": "id", "columns": {"id": "id", "v": "integer"}}}})
        n = len(load_tables(sp)["c"])
    finally:
        del os.environ["V09_TEST_URL"]
    missing, msg = _raises(lambda: load_tables(sp), ValueError, "V09_TEST_URL")
    shutil.rmtree(w, ignore_errors=True)
    return out + [("source.url_env read from the environment", n, 2, n == 2),
                  ("missing url_env variable named in the error", missing, True, missing)]


def t_s5_paths():
    """Table names cannot lead outside the output folder."""
    from relsynth import replica
    errs = {n: _spec_err({"tables": {n: {"primary_key": "id", "columns": {"id": "id"}}}})
            for n in ("../x", "a/b", "a\\b", "a\x00b", "a\nb", "..")}
    refused = [n for n, e in errs.items() if "table name" in e]
    ok_names = not _spec_err({"tables": {"Order Details": {"primary_key": "id", "columns": {"id": "id"}},
                                         "x.y": {"primary_key": "id", "columns": {"id": "id"}}}})
    w = _tmp()
    df = pd.DataFrame({"a": [1]})
    spec = {"_base_dir": str(w), "tables": {}, "relationships": []}
    esc, _ = _raises(lambda: write_tables(spec, {"../../escaped": df}, {"kind": "csv", "path": "out"}), ValueError)
    absl, _ = _raises(lambda: write_tables(spec, {str(w / "abs"): df}, {"kind": "csv", "path": "out"}), ValueError)
    write_tables(spec, {"good": df}, {"kind": "csv", "path": "out"})
    rep, _ = _raises(lambda: replica.table_path(w / "out", "../evil"), SystemExit)
    written = sorted(str(p.relative_to(w)) for p in w.rglob("*.csv"))
    shutil.rmtree(w, ignore_errors=True)
    return [("bad table names refused by the spec", refused, list(errs), len(refused) == len(errs)),
            ("names with spaces and dots still valid", ok_names, True, ok_names),
            ("write_tables refuses ../ and absolute names", [esc, absl], [True, True], esc and absl),
            ("replicate.py refuses ../ names", rep, True, rep),
            ("files written", written, ["out/good.csv"], written == ["out/good.csv"])]


def t_s6_redos():
    """Regular expressions with nested repetitions are refused, before any data is read."""
    bad = ["^(a+)+$", "(a*)*", "(a|a)*", "(a|ab)+$", "(\\w+\\s?)+$", "x" * 501]
    good = ["\\.[a-z]$", "^F\\d+", "(ab|cd)+", "^[A-Z]{2}\\d{4}$", "(a+)?b", "(?:\\d{3}-)?\\d{4}"]
    refused = [p for p in bad if _raises(lambda p=p: check_regex(p), ExprError)[0]]
    passed = [p for p in good if not _raises(lambda p=p: check_regex(p))[0]]
    s = _base(rules=[{"table": "c", "type": "check", "expr": "regex_match(v, '^(a+)+$')"}])
    err = _spec_err(s)
    t = time.perf_counter()
    r, _ = _raises(lambda: Evaluator(pd.DataFrame({"s": ["a" * 30 + "!"]})).eval("regex_match(s, '^(a+)+$')"), ExprError)
    sec = time.perf_counter() - t
    return [("dangerous patterns refused", len(refused), len(bad), len(refused) == len(bad)),
            ("ordinary patterns accepted", passed, good, passed == good),
            ("refused at spec validation", "repeats a repetition" in err, True, "repeats a repetition" in err),
            ("evaluation refuses at once (seconds)", round(sec, 3), "< 0.5", r and sec < 0.5)]


def t_s7_yaml_alias():
    """YAML aliases are refused (an alias bomb expands to billions of nodes)."""
    w = _tmp()
    doc = "a: &a [x,x,x,x,x,x,x,x,x,x]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]\n"
    p = w / "spec.yaml"
    p.write_text(yaml.safe_dump(BASE, sort_keys=False) + "description: &d hello\nname: *d\n")
    err = ""
    try:
        load_spec(p)
    except SpecError as e:
        err = str(e)
    (w / "bomb.yaml").write_text(doc)
    from relsynth.spec import read_yaml
    bomb, _ = _raises(lambda: read_yaml(w / "bomb.yaml"), SpecError, "aliases are not allowed")
    p.write_text(yaml.safe_dump(BASE, sort_keys=False))
    plain = load_spec(p)["tables"]["c"]["primary_key"] == "id"
    shutil.rmtree(w, ignore_errors=True)
    return [("spec with an alias refused", "yaml aliases are not allowed in a spec" in err, True,
             "yaml aliases are not allowed in a spec" in err),
            ("alias bomb refused while reading", bomb, True, bomb),
            ("plain spec file loads", plain, True, plain)]


def t_s8_markdown():
    """HTML and '|' are escaped in model.md cells; Mermaid labels carry no '\"'."""
    from relsynth.model_report import describe, to_markdown
    from relsynth.spec import normalise
    spec = normalise({"tables": {'<img src=x onerror=alert(1)>': {"primary_key": "id", "columns": {"id": "id"}},
                                 "f": {"role": "fact", "columns": {'k"} ; x|y': "id"}}},
                      "relationships": [{"child": "f", "child_key": 'k"} ; x|y', "parent": '<img src=x onerror=alert(1)>'}]})
    md = to_markdown(spec, describe(spec))
    merm = md.split("```mermaid")[1].split("```")[0]
    label_ok = all(ln.count('"') in (0, 2) for ln in merm.splitlines())
    cells_ok = all(ln.count("|") - ln.count("\\|") == 9 for ln in md.splitlines() if ln.startswith("| f |"))
    return [("no raw HTML tag", "<img" in md, False, "<img" not in md),
            ("Mermaid labels closed (inner quotes replaced)", label_ok, True, label_ok),
            ("'|' in a name escaped (cell count kept)", cells_ok, True, cells_ok)]


def _tiny_model():
    from relsynth import Synthesizer
    from relsynth.export import posterior_draws, structure
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"pid": range(1, 41), "g": rng.choice(["a", "b"], 40), "n": 0})
    c = pd.DataFrame({"cid": range(1, 121), "pid": rng.integers(1, 41, 120), "v": rng.normal(10, 2, 120).round(2)})
    p["n"] = c.groupby("pid").size().reindex(p.pid).fillna(0).astype(int).to_numpy()
    spec = {"tables": {"p": {"primary_key": "pid", "columns": {"pid": "id", "g": "categorical", "n": "integer"}},
                       "c": {"primary_key": "cid", "columns": {"cid": "id", "pid": "id", "v": "float"}}},
            "relationships": [{"child": "c", "child_key": "pid", "parent": "p"}],
            "rules": [{"table": "p", "type": "aggregate", "column": "n", "agg": "count", "from": "c"}],
            "synthesis": {"gibbs": {"iterations": 4, "final_iterations": 2, "max_final_iterations": 2},
                          "copula": {"iterations": 10, "max_iterations": 10, "burn_in": 2}}}
    syn = Synthesizer(load_spec(spec)).fit({"p": p, "c": c})
    S, _ = structure(syn)
    mean, D = posterior_draws(syn, 1)
    return json.loads(json.dumps({"structure": S, "mean": mean, "draws": D}))


def t_s9_aggregates():
    """replicate.py only calls the known aggregate functions."""
    from relsynth import replica
    m = _tiny_model()
    g = replica.generate(m, "mean", 7, 1.0)
    n_ok = int((g["p"]["n"].to_numpy() == g["c"].groupby("pid").size().reindex(g["p"]["pid"]).fillna(0).to_numpy()).all())
    m["structure"]["aggregates"][0]["agg"] = "__class__"
    bad, msg = _raises(lambda: replica.generate(m, "mean", 7, 1.0), SystemExit, "unknown aggregate")
    return [("count aggregate still computed", n_ok, 1, n_ok == 1),
            ("unknown aggregate name refused", bad, True, bad)]


# ====================================================================== data loading
def t_c1_long_keys():
    """18-digit keys in a CSV column with nulls keep every digit."""
    w = _tmp()
    (w / "a.csv").write_text("id,ref\n1,123456789012345678\n2,123456789012345679\n3,\n")
    df = read_table_file(w / "a.csv")
    got = coerce(df, {"columns": {"id": {"type": "id"}, "ref": {"type": "id"}}})["ref"].tolist()
    txt = canonical_keys(pd.Series(["123456789012345678", None], dtype=object)).tolist()
    shutil.rmtree(w, ignore_errors=True)
    want = [123456789012345678, 123456789012345679]
    return [("CSV keys", got[:2], want, got[:2] == want and pd.isna(got[2])),
            ("text keys", txt[0], want[0], txt[0] == want[0])]


def t_c2_exact_integer_test():
    """A key 1000000.4 is not the integer 1000000."""
    a = canonical_keys(pd.Series([1000000.4, 2.0]))
    b = canonical_keys(pd.Series([1.0, 2.0, np.nan]))
    return [("non-integer floats kept", a.tolist(), [1000000.4, 2.0], a.tolist() == [1000000.4, 2.0]),
            ("integer floats become Int64", str(b.dtype), "Int64", str(b.dtype) == "Int64" and b.tolist()[:2] == [1, 2])]


def t_c3_far_dates():
    """Sentinel dates (9999-12-31, 0001-01-01) load as null and are counted."""
    w = _tmp()
    (w / "t.csv").write_text("id,d\n1,2020-01-31\n2,9999-12-31\n3,0001-01-01\n")
    spec = load_spec({"source": {"kind": "csv", "path": str(w)},
                      "tables": {"t": {"primary_key": "id", "columns": {"id": "id", "d": "date"}}}})
    d = load_tables(spec)["t"]["d"]
    notes = list(load_tables.notes)
    shutil.rmtree(w, ignore_errors=True)
    return [("dates", [str(x) for x in d], ["2020-01-31", "NaT", "NaT"],
             str(d.dtype) == "datetime64[ns]" and d.iloc[0] == pd.Timestamp("2020-01-31") and d.iloc[1:].isna().all()),
            ("load note counts them", notes, "t.d: 2 date(s) outside 1677-2262",
             any("t.d: 2 date(s) outside" in n for n in notes))]


def t_c4_mixed_dates():
    """Several date formats in one column are each parsed; failures are counted."""
    notes = []
    df = coerce(pd.DataFrame({"d": ["2020-01-31", "31/01/2020", "2020-02-01 10:00", "soon"]}),
                {"columns": {"d": {"type": "date"}}}, notes=notes, table="t")
    got = [str(x) for x in df["d"]]
    want = ["2020-01-31 00:00:00", "2020-01-31 00:00:00", "2020-02-01 10:00:00", "NaT"]
    return [("parsed values", got, want, got == want),
            ("note counts the value that is no date", notes, "t.d: 1 value(s) that are not dates",
             any("t.d: 1 value(s) that are not dates" in n for n in notes))]


def t_c5_na_text():
    """'NA' (Namibia), 'None' and 'null' are values in a CSV; null_values configures it."""
    w = _tmp()
    (w / "t.csv").write_text("id,country,word\n1,NA,None\n2,DE,null\n3,,x\n")
    df = read_table_file(w / "t.csv")
    df2 = read_table_file(w / "t.csv", null_values=["", "NA", "None", "null"])
    spec = load_spec({"source": {"kind": "csv", "path": str(w), "null_values": ["", "null"]},
                      "tables": {"t": {"primary_key": "id", "columns": {"id": "id", "country": "categorical",
                                                                        "word": "categorical"}}}})
    df3 = load_tables(spec)["t"]
    shutil.rmtree(w, ignore_errors=True)
    return [("default: text kept", df["country"].tolist()[:2] + df["word"].tolist()[:2], ["NA", "DE", "None", "null"],
             df["country"].tolist()[:2] == ["NA", "DE"] and df["word"].tolist()[:2] == ["None", "null"]
             and pd.isna(df["country"].iloc[2])),
            ("null_values as given", int(df2.isna().sum().sum()), 4, int(df2.isna().sum().sum()) == 4),
            ("source.null_values used by load_tables", df3["word"].isna().tolist(), [False, True, False],
             df3["word"].isna().tolist() == [False, True, False])]


def t_c6_category_codes():
    """Integer category codes with a null come out as '1', not '1.0'."""
    w = _tmp()
    (w / "d.csv").write_text("id,grade,score\n1,1,1.5\n2,2,\n3,,2.0\n")
    df = coerce(read_table_file(w / "d.csv"), {"columns": {"id": {"type": "id"}, "grade": {"type": "categorical"},
                                                           "score": {"type": "categorical"}}})
    shutil.rmtree(w, ignore_errors=True)
    g = df["grade"].tolist()
    return [("integer codes", g[:2], ["1", "2"], g[:2] == ["1", "2"] and pd.isna(g[2])),
            ("non-integer codes unchanged", df["score"].tolist()[0], "1.5", df["score"].tolist()[0] == "1.5")]


def t_c7_db_write():
    """A subset of tables writes; dates are DATE; truncate keeps the target's DDL."""
    w = _tmp()
    db = w / "t.db"
    spec = {"_base_dir": str(w), "tables": {"c": {"columns": {"id": {"type": "id"}, "d": {"type": "date"}}},
                                             "f": {"columns": {"fid": {"type": "id"}, "cid": {"type": "id"},
                                                               "amt": {"type": "float"}}}},
            "relationships": [{"child": "f", "parent": "c", "kind": "many_to_one", "child_key": "cid"}]}
    tables = {"f": pd.DataFrame({"fid": pd.array([1, 2], dtype="Int64"), "cid": pd.array([1, 1], dtype="Int64"),
                                 "amt": [1.5, 2.25]}),
              "c": pd.DataFrame({"id": pd.array([1], dtype="Int64"), "d": pd.to_datetime(["2020-01-01"])})}
    write_tables(spec, tables, {"kind": "sqlite", "path": "t.db"})
    ddl = {n: " ".join(s.split()) for n, s in sqlite3.connect(db).execute("select name, sql from sqlite_master")}
    sub, msg = _raises(lambda: write_tables(spec, {"f": tables["f"]}, {"kind": "sqlite", "path": "t.db",
                                                                       "if_exists": "append"}))
    # truncate into a target with its own DDL (keys, NOT NULL, a foreign key)
    db2 = w / "u.db"
    con = sqlite3.connect(db2)
    con.executescript("""CREATE TABLE c(id INTEGER PRIMARY KEY, d DATE NOT NULL);
        CREATE TABLE f(fid INTEGER PRIMARY KEY, cid INTEGER NOT NULL REFERENCES c(id), amt DECIMAL(10,2));
        INSERT INTO c VALUES (7, '2019-01-01'); INSERT INTO f VALUES (9, 7, 1.0);""")
    con.commit()
    before = sorted(s for (s,) in con.execute("select sql from sqlite_master"))
    con.close()
    write_tables(spec, dict(tables), {"kind": "sqlite", "path": "u.db", "if_exists": "truncate"})
    con = sqlite3.connect(db2)
    after = sorted(s for (s,) in con.execute("select sql from sqlite_master"))
    rows = (con.execute("select id from c").fetchall(), con.execute("select fid from f order by fid").fetchall())
    con.close()
    bad_mode, _ = _raises(lambda: load_spec(_base(target={"kind": "sqlite", "path": "o.db", "if_exists": "replac"})),
                          SpecError)
    shutil.rmtree(w, ignore_errors=True)
    return [("subset of tables writes (no false cycle)", sub, False, not sub),
            ("date column created as DATE", "d DATE" in ddl["c"], True, "d DATE" in ddl["c"]),
            ("truncate keeps the DDL", before == after, True, before == after),
            ("truncate replaces the rows", rows, "([(1,)], [(1,), (2,)])", rows == ([(1,)], [(1,), (2,)])),
            ("if_exists typo refused by the spec", bad_mode, True, bad_mode)]


# ====================================================================== expressions
def _df():
    return pd.DataFrame({"x": [5.0, -1.0], "y": [5.0, 5.0], "code": ["10", "3"], "flag": ["False", "TRUE"],
                         "d": pd.to_datetime(["2021-06-01", "2020-01-01"]), "q": [2.5, -2.5], "s": ["abc", "x"]})


def t_e1_precedence():
    """x > 0 & y > 0 is refused with an explanation; parenthesised forms work."""
    r, msg = _raises(lambda: Evaluator(_df()).eval("x > 0 & y > 0"), ExprError, "bind tighter")
    ok = Evaluator(_df()).eval("(x > 0) & (y > 0)").tolist() == [1.0, 0.0]
    err = _spec_err(_base(rules=[{"table": "c", "type": "check", "expr": "v > 0 | v < 10"}]))
    return [("refused with the precedence hint", r, True, r),
            ("(x > 0) & (y > 0)", ok, True, ok),
            ("refused at spec validation", "bind tighter" in err, True, "bind tighter" in err)]


def t_e2_e3_text_vs_number():
    """Text compared with numbers or dates is converted when it is numeric or ISO; else an error."""
    E = lambda e: Evaluator(_df()).eval(e).tolist()  # noqa: E731
    out = [("d >= '2021-01-01'", E("d >= '2021-01-01'"), [1.0, 0.0], E("d >= '2021-01-01'") == [1.0, 0.0]),
           ("code > 9 (text code)", E("code > 9"), [1.0, 0.0], E("code > 9") == [1.0, 0.0]),
           ("code == 10", E("code == 10"), [1.0, 0.0], E("code == 10") == [1.0, 0.0]),
           ("code in [10]", E("code in [10]"), [1.0, 0.0], E("code in [10]") == [1.0, 0.0]),
           ("x in ['5']", E("x in ['5']"), [1.0, 0.0], E("x in ['5']") == [1.0, 0.0]),
           ("check d >= '2021-01-01' fails on the 2020 row", check_mask(_df(), "d >= '2021-01-01'").tolist(),
            [True, False], check_mask(_df(), "d >= '2021-01-01'").tolist() == [True, False])]
    r, _ = _raises(lambda: E("s > 3"), ExprError, "must be a number")
    r2, _ = _raises(lambda: E("s in [1, 2]"), ExprError)
    return out + [("s > 3 (text that is no number)", r, True, r), ("s in [1, 2]", r2, True, r2)]


def t_e4_text_booleans():
    """'true' / 'false' text is boolean; other text has no truth value."""
    got = Evaluator(_df()).eval("not flag").tolist()
    r, _ = _raises(lambda: Evaluator(_df()).eval("not s"), ExprError, "only 'true' and 'false'")
    w = Evaluator(_df()).eval("where(flag, 1, 0)").tolist()
    return [("not flag ('False', 'TRUE')", got, [1.0, 0.0], got == [1.0, 0.0]),
            ("where(flag, 1, 0)", w, [0.0, 1.0], w == [0.0, 1.0]),
            ("other text as a condition refused", r, True, r)]


def t_e5_round_half_away():
    """round() and integer derives round half away from zero (SQL)."""
    r = Evaluator(_df()).eval("round(q)").tolist()
    r2 = Evaluator(pd.DataFrame({"v": [0.125, 2.675, 1.005, -0.125, 3.14159]})).eval("round(v, 2)").tolist()
    i = list(derive_values(_df(), "n = q", ctype="integer")[1])
    rng = np.random.default_rng(5)
    v = rng.normal(0, 100, 2000).round(4)
    plain = Evaluator(pd.DataFrame({"v": v})).eval("round(v, 1)")
    same = bool(np.array_equal(plain[np.abs(v * 10 % 1 - 0.5) > 1e-6], np.round(v, 1)[np.abs(v * 10 % 1 - 0.5) > 1e-6]))
    return [("round(2.5), round(-2.5)", r, [3.0, -3.0], r == [3.0, -3.0]),
            ("round(., 2) on decimal halves", r2, [0.13, 2.68, 1.01, -0.13, 3.14], r2 == [0.13, 2.68, 1.01, -0.13, 3.14]),
            ("integer derive of 2.5 / -2.5", i, [3, -3], i == [3, -3]),
            ("values that are no half: as before", same, True, same)]


def t_e6_limits():
    """Long or deeply nested expressions are an ExprError, also in names_in."""
    out = []
    for label, e in (("not x 5000 times", "not " * 5000 + "x"), ("- x 50000 times", "-" * 50000 + "x"),
                     ("300 parentheses", "(" * 300 + "x" + ")" * 300), ("150 nested calls", "abs(" * 150 + "x" + ")" * 150)):
        a, _ = _raises(lambda e=e: Evaluator(_df()).eval(e), ExprError)
        b, _ = _raises(lambda e=e: names_in(e), ExprError)
        out.append((label, [a, b], [True, True], a and b))
    ok = Evaluator(_df()).eval("abs(" * 20 + "x" + ")" * 20).tolist() == [5.0, 1.0]
    return out + [("20 nested calls still evaluate", ok, True, ok)]


def t_p1_vectorised():
    """Text comparisons, in and where are vectorised; results equal the element loop."""
    import ast as _ast
    from relsynth import expr as X
    rng = np.random.default_rng(3)
    n = 20000
    s = rng.choice(["abc", "xyz", "", "Q"], n).astype(object)
    s[rng.random(n) < 0.1] = None
    t = rng.choice(["abc", "def"], n).astype(object)
    df = pd.DataFrame({"s": s, "t": t})
    ev = Evaluator(df)
    a, b = X.to_array(df["s"]), X.to_array(df["t"])
    out = []
    for op, sym in ((_ast.Eq(), "=="), (_ast.Lt(), "<"), (_ast.GtE(), ">=")):
        loop = np.array([np.nan if x is None else float({"==": x == y, "<": x < y, ">=": x >= y}[sym])
                         for x, y in zip(a, b, strict=True)])
        got = X._compare(op, a, b)
        same = bool(np.array_equal(loop, got, equal_nan=True))
        out.append((f"s {sym} t equals the loop", same, True, same))
    isin = ev.eval("s in ['abc', 'Q']")
    loop = np.array([np.nan if x is None else float(x in ("abc", "Q")) for x in s])
    out.append(("s in [...] equals the loop", bool(np.array_equal(isin, loop, equal_nan=True)), True,
                bool(np.array_equal(isin, loop, equal_nan=True))))
    w = ev.eval("where(s == 'abc', s, t)")
    loop = [x if x == "abc" else y for x, y in zip(s, t, strict=True)]
    out.append(("where equals the loop", list(w) == loop, True, list(w) == loop))
    big = pd.DataFrame({"s": np.where(np.arange(200000) % 3, "abc", "xyz").astype(object)})
    t0 = time.perf_counter()
    Evaluator(big).eval("s == 'abc'")
    sec = time.perf_counter() - t0
    return out + [("s == 'abc' on 200,000 rows (seconds)", round(sec, 3), "< 0.35", sec < 0.35)]


# ====================================================================== spec validation
def t_v1_internal_keys():
    """Keys of the engine's internal plan are not accepted from a spec."""
    w = _tmp()
    cases = {"tables.f.anchor": lambda s: s["tables"]["f"].update(anchor="c"),
             "columns.amt.unit / virtual": lambda s: s["tables"]["f"]["columns"].update(
                 amt={"type": "float", "unit": "EUR", "virtual": True}),
             "relationships[0].anchor_link": lambda s: s["relationships"][0].update(anchor_link="x"),
             "relationships[0].anchor": lambda s: s["relationships"][0].update(anchor="x"),
             "fact.anchor_key that is no key": lambda s: s["tables"]["f"].update(fact={"anchor": "c", "anchor_key": "amt"})}
    hit = []
    for k, mut in cases.items():
        s = _base()
        mut(s)
        if _spec_err(s):
            hit.append(k)
    p = w / "spec.yaml"
    p.write_text(yaml.safe_dump(_base(_refs={"x": {}}, _base_dir="/"), sort_keys=False))
    e = ""
    try:
        load_spec(p)
    except SpecError as ex:
        e = str(ex)
    # a normalised spec written back (doctor --apply) still validates
    from relsynth.spec import normalise
    n = {k: v for k, v in normalise(_base(tables={**BASE["tables"], "f": {"role": "fact", "fact": {"anchor": "cid"},
                                                                          "columns": {"cid": "id", "amt": "float"}}}
                                          )).items() if not k.startswith("_")}
    p.write_text(yaml.safe_dump(n, sort_keys=False))
    back = not _raises(lambda: load_spec(p))[0]
    shutil.rmtree(w, ignore_errors=True)
    return [("internal keys refused", hit, list(cases), hit == list(cases)),
            ("'_' keys refused in a spec file", "_refs" in e and "_base_dir" in e, True, "_refs" in e and "_base_dir" in e),
            ("normalised spec written back validates", back, True, back)]


def t_v2_enums_and_types():
    """Enumerations and types of source, target and synthesis are checked."""
    cases = {"datasets 'three'": {"synthesis": {"posterior": {"datasets": "three"}}},
             "datasets 0": {"synthesis": {"posterior": {"datasets": 0}}},
             "mode 'avg'": {"synthesis": {"posterior": {"mode": "avg"}}},
             "gibbs: 5": {"synthesis": {"gibbs": 5}},
             "seed {x: 1}": {"synthesis": {"seed": {"x": 1}}},
             "seed 1.5": {"synthesis": {"seed": 1.5}},
             "structural_prior 1": {"synthesis": {"structure": {"structural_prior": 1.0}}},
             "structural_pooling -1": {"synthesis": {"structure": {"structural_pooling": -1}}},
             "pooled_cells 0": {"synthesis": {"copula": {"pooled_cells": 0}}},
             "if_exists replac": {"target": {"kind": "sqlite", "path": "o.db", "if_exists": "replac"}},
             "source.kind postgres": {"source": {"kind": "postgres", "url": "x"}},
             # v1.0: parquet is a target kind; an unknown kind and bad pushdown settings are refused
             "target.kind excel": {"target": {"kind": "excel", "path": "o"}},
             "pushdown.mode sometimes": {"synthesis": {"pushdown": {"mode": "sometimes"}}},
             "pushdown.sample_rows 0": {"synthesis": {"pushdown": {"sample_rows": 0}}},
             "null_values 'NA'": {"source": {"kind": "csv", "path": ".", "null_values": "NA"}}}
    hit = [k for k, over in cases.items() if _spec_err(_base(**over))]
    s = _base()
    s["tables"]["c"]["columns"].update(nk="text", vf="date", cur="categorical")
    s["tables"]["c"]["scd"] = {"natural_key": "nk", "valid_from": "vf", "current": {"column": "cur", "vaule": "Y"}}
    scd = "unknown scd current keys" in _spec_err(s)
    good = not _spec_err(_base(synthesis={"seed": 3, "posterior": {"datasets": 2, "mode": "mean"},
                                          "structure": {"structural_prior": 0.5, "structural_pooling": 0},
                                          "copula": {"pooled_cells": 2e5}},
                               target={"kind": "sqlite", "path": "o.db", "if_exists": "truncate"}))
    return [("bad values refused", hit, list(cases), hit == list(cases)),
            ("scd.current typo refused", scd, True, scd),
            ("valid values accepted", good, True, good)]


def t_v3_column_as_list():
    """A column given as a list is a SpecError naming the column."""
    s = _base()
    s["tables"]["c"]["columns"]["v"] = ["integer"]
    e = _spec_err(s)
    return [("error names tables.c.columns.v", "tables.c.columns.v" in e, True, "tables.c.columns.v" in e)]


def t_new_synthesis_keys():
    """structure.structural_pooling and copula.pooled_cells: defaults and reference docs."""
    from relsynth.schema import SYNTH_DOC, reference_tables
    from relsynth.spec import DEFAULT_SYNTHESIS
    d = (DEFAULT_SYNTHESIS["structure"]["structural_pooling"], DEFAULT_SYNTHESIS["copula"]["pooled_cells"])
    ref = reference_tables()
    docs = "`structure.structural_pooling`" in ref and "`copula.pooled_cells`" in ref
    return [("defaults", d, (3.0, 1e6), d == (3.0, 1e6)),
            ("in the generated reference", docs, True, docs and "structure.structural_pooling" in SYNTH_DOC)]


def t_readability():
    """Unused imports and the unused table_files parameter are gone."""
    import inspect
    from relsynth import data
    cli_src = (ROOT / "relsynth" / "cli.py").read_text()
    st_src = (ROOT / "relsynth" / "stancheck.py").read_text()
    params = list(inspect.signature(data.table_files).parameters)
    return [("cli imports dump_spec", "dump_spec" in cli_src, False, "dump_spec" not in cli_src),
            ("stancheck imports norm", "import norm" in st_src, False, "import norm" not in st_src),
            ("table_files parameters", params, ["root"], params == ["root"])]


def t_specs_validate():
    """Every spec of the repository and the fake scenarios still validates."""
    import glob
    import re
    sys.path.insert(0, str(ROOT / "fakedata"))
    import scenarios as S
    files = sorted(glob.glob(str(ROOT / "realdata" / "**" / "spec*.yaml"), recursive=True)
                   + glob.glob(str(ROOT / "examples" / "**" / "spec.yaml"), recursive=True)
                   + [str(ROOT / "docs" / "spec-template.yaml")])
    known_bad = {"spec.introspected.yaml"}        # an introspect draft that needs edits (also before v0.9)
    bad = [Path(f).name for f in files if Path(f).name not in known_bad and _raises(lambda f=f: load_spec(f))[0]]
    src = (ROOT / "docs" / "spec-reference.src.md").read_text()
    ex = yaml.safe_load(re.search(r"## 2\. A complete annotated example\n\n```yaml\n(.*?)```", src, re.S).group(1))
    ex_ok = not _raises(lambda: load_spec(ex))[0]
    sc_bad = [n for n, fn in S.SCENARIOS.items() if _raises(lambda fn=fn: load_spec(fn()[1]))[0]]
    return [(f"repository specs that fail (of {len(files)})", bad, [], not bad),
            ("annotated example of the reference", ex_ok, True, ex_ok),
            ("fake scenarios that fail", sc_bad, [], not sc_bad)]


TESTS = [t_s1_latex, t_s2_run_model, t_s3_sql, t_s4_credentials, t_s5_paths, t_s6_redos, t_s7_yaml_alias,
         t_s8_markdown, t_s9_aggregates, t_c1_long_keys, t_c2_exact_integer_test, t_c3_far_dates, t_c4_mixed_dates,
         t_c5_na_text, t_c6_category_codes, t_c7_db_write, t_e1_precedence, t_e2_e3_text_vs_number,
         t_e4_text_booleans, t_e5_round_half_away, t_e6_limits, t_p1_vectorised, t_v1_internal_keys,
         t_v2_enums_and_types, t_v3_column_as_list, t_new_synthesis_keys, t_readability, t_specs_validate]


def main(names):
    npass = 0
    run = [t for t in TESTS if not names or t.__name__ in names]
    for t in run:
        t0 = time.time()
        try:
            items = t()
            err = None
        except Exception:  # noqa: BLE001
            items, err = [], traceback.format_exc()
        ok = err is None and all(i[3] for i in items)
        npass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {t.__name__} ({time.time() - t0:.1f}s)")
        for what, value, target, good in items:
            print(f"       {'ok ' if good else 'BAD'} {what}: {value} (target {target})")
        if err:
            print("       ERROR", err)
    print(f"\n{npass}/{len(run)} tests passed")
    return 0 if npass == len(run) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

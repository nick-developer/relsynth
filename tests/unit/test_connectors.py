"""Connectors on files (CSV, Parquet) and SQLite: catalog, reads, samples,
writes in every mode, structure cloning, and identifier safety."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from relsynth.connectors import FileConnector, connect, from_spec
from relsynth.data import write_tables
from relsynth.spec import load_spec


# ---------------------------------------------------------------- locations
@pytest.mark.parametrize("loc, cls, kind", [
    ("csv:{d}", "FileConnector", "csv"), ("files:{d}", "FileConnector", "csv"),
    ("parquet:{d}", "FileConnector", "parquet"), ("sqlite:{d}/x.db", "SQLiteConnector", "sqlite"),
    ("sqlite:///{d}/x.db", "SQLiteConnector", "sqlite"), ("{d}/x.db", "SQLiteConnector", "sqlite"),
])
def test_connect_dispatch(tmp_path, loc, cls, kind):
    c = connect(loc.format(d=tmp_path), create=True)
    assert type(c).__name__ == cls and c.kind == kind
    c.close()


def test_server_urls_pick_recommended_drivers(monkeypatch):
    from relsynth.connectors.sql import resolve_url
    assert resolve_url("postgresql://u:p@h/d").startswith("postgresql+psycopg://")
    assert resolve_url("oracle://u:p@h:1521/?service_name=x").startswith("oracle+oracledb://")
    assert resolve_url("mssql://u:p@h/d").startswith(("mssql+pyodbc://", "mssql+pymssql://"))
    assert resolve_url("postgresql+psycopg2://u:p@h/d").startswith("postgresql+psycopg2://")   # explicit kept


def test_password_env_completes_the_url(monkeypatch):
    from relsynth.connectors.sql import resolve_url
    monkeypatch.setenv("RS_PW", "s3cr3t:@/x")
    u = resolve_url("postgresql://scott@db/prod", password_env="RS_PW")
    assert sa.engine.make_url(u).password == "s3cr3t:@/x"
    monkeypatch.setenv("RS_URL", "sqlite:///:memory:")
    assert sa.engine.make_url(resolve_url(None, url_env="RS_URL")).database == ":memory:"
    with pytest.raises(ValueError):
        resolve_url(None, url_env="RS_MISSING")


# ---------------------------------------------------------------- files
@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_files_roundtrip_and_catalog(tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", None, "c"], "x": [1.5, np.nan, 2.0],
                       "d": pd.to_datetime(["2024-01-01", "2024-02-01", None])})
    c = FileConnector(tmp_path, fmt=fmt, create=True)
    c.write_table("t", df)
    assert c.tables() == ["t"] and c.has_table("t") and c.count("t") == 3
    back = c.read_table("t")
    assert back["id"].astype(int).tolist() == [1, 2, 3]
    assert back["name"].isna().tolist() == [False, True, False]
    assert set(c.describe("t").column_names) == {"id", "name", "x", "d"}
    assert c.read_table("t", columns=["x"]).columns.tolist() == ["x"]
    chunks = list(c.iter_table("t", chunksize=2))
    assert sum(len(ch) for ch in chunks) == 3


def test_csv_null_values(tmp_path):
    (tmp_path / "t.csv").write_text("id,v\n1,n/a\n2,5\n3,\n")
    c = FileConnector(tmp_path, null_values=["", "n/a"])
    assert c.read_table("t")["v"].isna().tolist() == [True, False, True]


def test_file_sample_is_seeded_and_streams(tmp_path):
    pd.DataFrame({"id": np.arange(30000), "v": np.arange(30000) % 5}).to_csv(tmp_path / "t.csv", index=False)
    c = FileConnector(tmp_path)
    a, whole = c.sample("t", 3000, seed=5)
    b, _ = c.sample("t", 3000, seed=5)
    assert not whole and a.equals(b) and abs(len(a) - 3000) < 300


@pytest.mark.parametrize("name", ["../secret", "a/b", "..", "/etc/passwd", "t\x00"])
def test_file_table_names_cannot_escape_the_folder(tmp_path, name):
    c = FileConnector(tmp_path, create=True)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        c.write_table(name, pd.DataFrame({"a": [1]}))
    assert not (tmp_path.parent / "secret.csv").exists()


def test_csv_binary_columns_are_hex(tmp_path):
    c = FileConnector(tmp_path, create=True)
    c.write_table("b", pd.DataFrame({"id": [1], "blob": [b"\x00\xffAB"]}))
    assert "0x00ff4142" in (tmp_path / "b.csv").read_text().lower()


# ---------------------------------------------------------------- sqlite
def _spec2():
    return load_spec({"tables": {
        "p": {"primary_key": "id", "columns": {"id": "id", "name": "categorical"}},
        "c": {"primary_key": "id", "columns": {"id": "id", "pid": "id", "v": "float", "when": "date"},
              "role": "fact", "fact": {"anchor": "p"}}},
        "relationships": [{"child": "c", "child_key": "pid", "parent": "p", "parent_key": "id"}]})


def _tables2(n=5):
    p = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    c = pd.DataFrame({"id": np.arange(1, n + 1), "pid": [1, 2, 3, 1, 2][:n], "v": np.linspace(0, 1, n),
                      "when": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n), unit="D")})
    return {"p": p, "c": c}


def test_sqlite_write_modes(tmp_path):
    db = tmp_path / "w.db"
    spec, t = _spec2(), _tables2()
    tgt = {"kind": "sqlite", "path": str(db)}
    write_tables(spec, t, tgt)
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from c").fetchone()[0] == 5
    # foreign key declared by the inferred structure
    fks = con.execute("pragma foreign_key_list(c)").fetchall()
    assert fks and fks[0][2] == "p"
    con.close()
    more = {"p": t["p"].assign(id=t["p"]["id"] + 10), "c": t["c"].assign(id=t["c"]["id"] + 10)}
    write_tables(spec, more, {**tgt, "if_exists": "append", "structure": "existing"})
    assert sqlite3.connect(db).execute("select count(*) from p").fetchone()[0] == 6
    write_tables(spec, t, {**tgt, "if_exists": "truncate"})
    assert sqlite3.connect(db).execute("select count(*) from c").fetchone()[0] == 5
    with pytest.raises(FileExistsError):
        write_tables(spec, t, {**tgt, "if_exists": "fail"})


def test_sqlite_catalog_and_reads(tmp_path):
    db = tmp_path / "r.db"
    write_tables(_spec2(), _tables2(), {"kind": "sqlite", "path": str(db)})
    c = connect(f"sqlite:{db}")
    info = c.describe("c")
    assert info.primary_key == ["id"]
    assert [(f.columns, f.ref_table, f.ref_columns) for f in info.foreign_keys] == [(["pid"], "p", ["id"])]
    assert c.count("c") == 5 and set(c.tables()) == {"p", "c"}
    assert c.read_table("c", columns=["v"]).shape == (5, 1)
    assert c.ping()["ok"]


def test_clone_structure_between_sqlite_files(tmp_path):
    src = tmp_path / "src.db"
    con = sqlite3.connect(src)
    con.executescript("""
        CREATE TABLE p (id INTEGER PRIMARY KEY, code VARCHAR(10) NOT NULL UNIQUE);
        CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER NOT NULL REFERENCES p(id), amount NUMERIC(10,2));
        CREATE INDEX ix_c_pid ON c(pid);
        INSERT INTO p VALUES (1, 'a'), (2, 'b');
        INSERT INTO c VALUES (1, 1, 2.5), (2, 2, 3.5);""")
    con.commit()
    con.close()
    import relsynth as rs
    made = rs.clone_schema(f"sqlite:{src}", f"sqlite:{tmp_path / 'dst.db'}")
    assert made == ["p", "c"]
    dst = sqlite3.connect(tmp_path / "dst.db")
    ddl = " ".join(r[0] for r in dst.execute("select sql from sqlite_master where sql is not null"))
    assert "REFERENCES" in ddl.upper() and "UNIQUE" in ddl.upper() and "ix_c_pid" in ddl
    assert dst.execute("select count(*) from c").fetchone()[0] == 0


def test_hostile_table_and_column_names_are_quoted(tmp_path):
    """Identifiers are never pasted into SQL: names with quotes and semicolons
    are read and written as names."""
    db = tmp_path / "h.db"
    evil_t, evil_c = 'x"; DROP TABLE keep; --', "v'); DELETE FROM keep; --"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE keep (a INTEGER)")
    con.execute("INSERT INTO keep VALUES (1)")
    con.execute(f'CREATE TABLE "{evil_t.replace(chr(34), chr(34) * 2)}" (id INTEGER PRIMARY KEY, '
                f'"{evil_c}" INTEGER)')
    con.execute(f'INSERT INTO "{evil_t.replace(chr(34), chr(34) * 2)}" VALUES (1, 2), (2, 3)')
    con.commit()
    con.close()
    c = connect(f"sqlite:{db}")
    df = c.read_table(evil_t)
    assert df[evil_c].tolist() == [2, 3]
    st = c.stats(evil_t)
    assert st.n_rows == 2 and st.notnull(evil_c) == 2 and len(st.group_counts([evil_c], [evil_c])) == 2
    s, _ = c.sample(evil_t, 1, seed=1)
    c.write_table(evil_t, df.assign(id=df["id"] + 10), mode="append")
    assert c.count(evil_t) == 4
    assert sqlite3.connect(db).execute("select count(*) from keep").fetchone()[0] == 1


def test_from_spec_resolves_relative_paths(tmp_path):
    (tmp_path / "d").mkdir()
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "d" / "t.csv", index=False)
    c = from_spec({"kind": "csv", "path": "d"}, base_dir=tmp_path)
    assert c.tables() == ["t"]

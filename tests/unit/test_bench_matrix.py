# SPDX-License-Identifier: Apache-2.0
"""The G7 benchmark matrix: tiers, coverage list, the schema stress generator,
the schema-level checks X1-X6 and the engine findings fixed with it."""
from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.__main__ import coverage_table, matrix
from bench.datasets import DATASETS, FEATURES, REQUIRED_FEATURES, TIERS, tier_datasets

ROOT = Path(__file__).resolve().parents[2]
STRESS = ROOT / "bench" / "builders" / "stress_schema.py"


def test_every_dataset_has_a_known_tier_and_known_features():
    for n, d in DATASETS.items():
        assert d.tier in TIERS, n
        assert d.runner in ("github", "maintainer"), n
        assert set(d.coverage) <= set(FEATURES), (n, set(d.coverage) - set(FEATURES))
        assert d.seeds, n
        assert all(set(g) <= set(d.seeds) for g in d.jobs), n


def test_every_required_feature_is_exercised_by_two_datasets():
    counts = Counter(f for d in DATASETS.values() for f in d.coverage)
    assert {f: counts[f] for f in REQUIRED_FEATURES if counts[f] < 2} == {}
    table, short = coverage_table()
    assert short == []
    assert all(f"`{f}`" in table for f in FEATURES)


def test_tiers_follow_the_g7_matrix():
    assert set(tier_datasets("pr")) == {"chinook", "adult", "stress_s", "mongodb_samples"}
    assert "chinook_x2" not in DATASETS
    assert DATASETS["adventureworks"].seeds == (1, 2, 3)
    assert DATASETS["contoso_parquet"].tier == "weekly" and DATASETS["contoso_sqlite"].tier == "monthly"
    assert {"tpch_sf10", "tpch_sf100", "tpcds_sf10", "synthea_1m", "nyc_tlc_12m", "stress_l"} <= \
        set(tier_datasets("monthly"))
    assert set(tier_datasets("local")) == {"bi_corep_lr", "imdb_movielens"}


def test_ci_matrix_splits_seeds_and_leaves_out_bigger_runner_sets():
    weekly = matrix("weekly")
    names = [p["name"] for p in weekly]
    assert {"adventureworks-a", "adventureworks-b", "fake", "stress_m", "tpch_sf1"} <= set(names)
    aw = {p["name"]: p["seeds"] for p in weekly if p["datasets"] == "adventureworks"}
    assert aw == {"adventureworks-a": "1,2", "adventureworks-b": "3"}
    monthly = {p["datasets"] for p in matrix("monthly")}
    assert "tpch_sf10" in monthly and "tpch_sf100" not in monthly and "synthea_1m" not in monthly
    assert all(p["datasets"] not in tier_datasets("local") for t in ("weekly", "monthly") for p in matrix(t))


def _stress(path: Path, *args: str) -> Path:
    subprocess.run([sys.executable, str(STRESS), str(path), *args], check=True, stdout=subprocess.DEVNULL)
    return path


SMALL = ("--tables", "24", "--depth", "4", "--wide", "30", "--rows", "4000")


@pytest.fixture(scope="module")
def stress_db(tmp_path_factory) -> Path:
    return _stress(tmp_path_factory.mktemp("stress") / "s.db", *SMALL)


def _catalog(db: Path):
    con = sqlite3.connect(db)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        fks = {t: con.execute(f'PRAGMA foreign_key_list("{t}")').fetchall() for t in tables}
        cols = {t: con.execute(f'PRAGMA table_info("{t}")').fetchall() for t in tables}
    finally:
        con.close()
    return tables, fks, cols


def test_stress_schema_holds_every_hard_part(stress_db, tmp_path):
    tables, fks, cols = _catalog(stress_db)
    assert len(tables) >= 24
    assert {"b0_l1", "b0_l4", "ck_account", "ck_txn", "employee", "department", "bridge_emp_project",
            "document", "wide_profile", "sensor_reading", "fact_0"} <= set(tables)
    # snowflake depth 4: b0_l1 -> b0_l2 -> b0_l3 -> b0_l4
    assert [(r[2]) for r in fks["b0_l1"]] == ["b0_l2"] and [r[2] for r in fks["b0_l3"]] == ["b0_l4"]
    # composite link: both columns in one constraint
    ck = [r for r in fks["ck_txn"] if r[2] == "ck_account"]
    assert len(ck) == 2 and len({r[0] for r in ck}) == 1
    # a self-reference and a two-table cycle
    assert ("employee", "manager_id") in {(r[2], r[3]) for r in fks["employee"]}
    assert "employee" in {r[2] for r in fks["department"]} and "department" in {r[2] for r in fks["employee"]}
    assert {c[2] for c in cols["document"] if c[1] in ("payload", "thumbnail")} == {"BLOB"}
    assert len(cols["wide_profile"]) == 30
    # the same arguments build the same contents
    again = _stress(tmp_path / "again.db", *SMALL)
    dump = lambda p: hashlib.sha256("\n".join(sqlite3.connect(p).iterdump()).encode()).hexdigest()  # noqa: E731
    assert dump(again) == dump(stress_db)


def test_stress_schema_reaches_the_table_count(tmp_path):
    tables, _, cols = _catalog(_stress(tmp_path / "m.db", "--tables", "120", "--depth", "8", "--wide", "500",
                                       "--rows", "2000"))
    assert len(tables) == 120
    assert len(cols["wide_profile"]) == 500
    assert "b0_l8" in tables


def test_generic_checks_hold_on_a_copy_of_the_real_tables(stress_db, tmp_path):
    import relsynth as rs
    from bench.checks import generic
    from relsynth.spec import load_spec
    spec_path = tmp_path / "spec.yaml"
    rs.introspect(f"sqlite:{stress_db}", name="s").save(spec_path)
    spec = load_spec(str(spec_path))
    syn = tmp_path / "syn"
    syn.mkdir()
    con = sqlite3.connect(stress_db)
    for t in spec["tables"]:
        pd.read_sql_query(f'SELECT * FROM "{t}"', con).to_csv(syn / f"{t}.csv", index=False)
    con.close()
    out = generic.checks(spec_path, f"sqlite:{stress_db}", syn)
    assert out["X5 rows_ratio"] == (None, 1.0)
    ids = [k for k in out if k.startswith(("X1", "X2", "X3"))]
    assert ids and all(out[k][0] == out[k][1] == 0.0 for k in ids), {k: out[k] for k in ids if out[k] != (0.0, 0.0)}
    assert "X3 selfref_cycle|employee.manager_id" in out
    assert any(k.startswith("X4 fanout_top1pct|fact_0.") for k in out)
    # an orphan and a duplicate key in the synthetic copy are caught
    f = pd.read_csv(syn / "fact_0.csv")
    key = next(c for c in f.columns if c.startswith("b") and c.endswith("_id"))
    f.loc[0, key] = 10 ** 9
    f.loc[1, "id"] = f.loc[2, "id"]
    f.to_csv(syn / "fact_0.csv", index=False)
    out = generic.checks(spec_path, f"sqlite:{stress_db}", syn)
    assert out[f"X2 fk_orphans|fact_0.{key}"][1] > 0
    assert out["X1 pk_duplicates|fact_0"][1] > 0


def test_cycle_share_finds_rows_on_a_cycle():
    import duckdb

    from bench.checks.generic import _cycle_share
    con = duckdb.connect()
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.t AS SELECT * FROM (VALUES ('1', NULL), ('2', '1'), ('3', '4'), ('4', '3'), "
                "('5', '3')) v(id, parent)")
    assert _cycle_share(con, "s", "t", "id", "parent") == pytest.approx(2 / 5)


# ---------------------------------------------------------------- engine findings fixed with the matrix (G7)
def _tiny_cycle_db(path: Path) -> Path:
    """department.head_id -> employee and employee.department_id -> department (declared), plus an
    account keyed by a composite (region, no) that only a composite link reaches."""
    from bench.builders.sqlite_ddl import create_table, insert_rows
    con = sqlite3.connect(path)
    create_table(con, "department", [("id", "INTEGER", True), ("head_id", "INTEGER", False)], ["id"],
                 foreign_keys=[(["head_id"], "employee", ["id"])])
    create_table(con, "employee", [("id", "INTEGER", True), ("department_id", "INTEGER", True),
                                   ("salary", "REAL", True)], ["id"],
                 foreign_keys=[(["department_id"], "department", ["id"])])
    create_table(con, "account", [("region", "TEXT", True), ("no", "INTEGER", True), ("tier", "TEXT", True)],
                 ["region", "no"])
    create_table(con, "txn", [("id", "INTEGER", True), ("region", "TEXT", True), ("no", "INTEGER", True),
                              ("amount", "REAL", True)], ["id"],
                 foreign_keys=[(["region", "no"], "account", ["region", "no"])])
    insert_rows(con, "department", ["id", "head_id"], [[d, d * 7 if d % 3 else None] for d in range(1, 11)])
    insert_rows(con, "employee", ["id", "department_id", "salary"],
                [[e, 1 + e % 10, 1000.0 + 13 * e] for e in range(1, 201)])
    insert_rows(con, "account", ["region", "no", "tier"], [[r, n, "gold" if n % 2 else "basic"]
                                                          for r in ("N", "S") for n in range(1, 31)])
    insert_rows(con, "txn", ["id", "region", "no", "amount"],
                [[i, "N" if i % 2 else "S", 1 + i % 30, float(i % 17)] for i in range(1, 401)])
    con.commit()
    con.close()
    return path


def test_introspect_breaks_a_declared_cycle_and_keeps_a_composite_child(tmp_path):
    import relsynth as rs
    spec = rs.introspect(f"sqlite:{_tiny_cycle_db(tmp_path / 'c.db')}", name="c")
    review = "\n".join(spec.raw.get("review", []))
    rels = {(r["child"], r["parent"]) for r in spec.resolved["relationships"]}
    assert ("employee", "department") in rels and ("department", "employee") not in rels
    assert "department.head_id -> employee: declared link closes a cycle" in review
    assert "account" not in spec.tables and "txn" in spec.tables
    assert spec.resolved["tables"]["txn"]["role"] == "entity"
    assert "account: no single-column primary key (only composite links reach it)" in review


def test_discriminator_design_skips_an_all_missing_column():
    import numpy as np

    from relsynth.evaluate import _design
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0], "blob": pd.Series([None, None, None], dtype="string")})
    tspec = {"columns": {"x": {"type": "float"}, "blob": {"type": "categorical"}}}
    X = _design([a, a], tspec, ["x", "blob"])
    # x gives 3 features plus the intercept; pandas < 3 turns the missing values into one '<NA>' level
    assert X.shape[0] == 6 and X.shape[1] in (4, 5) and np.isfinite(X).all()


def test_bench_readme_lists_the_current_coverage():
    table, _ = coverage_table()
    assert table in (ROOT / "bench" / "README.md").read_text(encoding="utf-8"), \
        "bench/README.md: regenerate the coverage list from `python -m bench coverage`"


def test_introspect_reads_a_large_parquet_file_on_demand(tmp_path, monkeypatch):
    import decimal

    import relsynth as rs
    import relsynth.source as src
    d = tmp_path / "pq"
    d.mkdir()
    pd.DataFrame({"id": range(1, 301), "name": [f"c{i % 40}" for i in range(300)]}).to_parquet(d / "customer.parquet")
    pd.DataFrame({"id": range(1, 3001), "customer_id": [1 + i % 300 for i in range(3000)],
                  "amount": [float(i % 97) for i in range(3000)],
                  "store": [None if i % 7 == 0 else float(i % 13) for i in range(3000)],
                  "qty": [decimal.Decimal(i % 50) for i in range(3000)]}).to_parquet(d / "orders.parquet")
    eager = src.Source({"kind": "parquet", "path": str(d)})
    monkeypatch.setattr(src, "LAZY_PARQUET_ROWS", 0)
    lazy = src.Source({"kind": "parquet", "path": str(d)})
    assert set(lazy._lazy) == {"customer", "orders"} and not eager._lazy
    assert lazy.count("orders") == eager.count("orders") == 3000
    assert lazy.column("orders", "customer_id").equals(eager.column("orders", "customer_id"))
    assert lazy.columns("orders", ["id", "amount"]).equals(eager.columns("orders", ["id", "amount"]))
    pd.testing.assert_frame_equal(lazy.rows_in("orders", "customer_id", [3, 7]),
                                  eager.rows_in("orders", "customer_id", [3, 7]))
    u, n = lazy.int_distinct("orders", "customer_id")
    assert n == 3000 and list(u) == list(range(1, 301)) and eager.int_distinct("orders", "customer_id") is None
    from relsynth.keyfind import _as_int, _keyvals
    for c in ("customer_id", "store", "amount", "qty"):          # the same key set as the whole-column path
        k = _keyvals(eager.column("orders", c))
        old = np.sort(_as_int(np.array(pd.unique(k.astype(object)), dtype=object)))
        assert list(lazy.int_distinct("orders", c)[0]) == list(old), c
    part, whole = lazy.sample("orders", 500)
    assert not whole and 350 < len(part) < 650 and list(part.columns) == ["id", "customer_id", "amount", "store", "qty"]
    spec = rs.introspect(f"parquet:{d}", name="lazy", sample_rows=500)
    assert {(r["child"], r["parent"]) for r in spec.resolved["relationships"]} == {("orders", "customer")}


def test_rows_in_matches_float_keys_of_a_column_with_nulls(tmp_path, monkeypatch):
    """A child key with NULLs comes back from Parquet as floats (2450815.0); the parent's rows are still
    found (TPC-DS SF10: no parent row was found and the lookup search indexed an empty table)."""
    import relsynth as rs
    import relsynth.source as src
    d = tmp_path / "pq"
    d.mkdir()
    pd.DataFrame({"item_sk": range(1, 2001), "category": [f"c{i % 10}" for i in range(2000)]}).to_parquet(
        d / "item.parquet")
    pd.DataFrame({"ticket": range(1, 6001), "item_sk": [None if i % 9 == 0 else float(1 + i * 31 % 2000)
                                                        for i in range(6000)],
                  "price": [float(i % 97) for i in range(6000)]}).to_parquet(d / "sales.parquet")
    for lazy_rows in (10**9, 0):
        monkeypatch.setattr(src, "LAZY_PARQUET_ROWS", lazy_rows)
        s = src.Source({"kind": "parquet", "path": str(d)})
        assert len(s.rows_in("item", "item_sk", [32.0, 63.0, 5.0])) == 3
        assert len(s.rows_in("sales", "item_sk", [32, 63])) == 6
        spec = rs.introspect(f"parquet:{d}", name="nullkeys", sample_rows=500)
        assert {(r["child"], r["parent"]) for r in spec.resolved["relationships"]} == {("sales", "item")}


def test_nulls_and_repeats_of_a_large_parquet_table_match_pandas(tmp_path, monkeypatch):
    """The whole-table uniqueness test of a large Parquet table runs in DuckDB (bounded memory) with the
    answer of the pandas test it replaces."""
    import relsynth.source as src
    d = tmp_path / "pq"
    d.mkdir()
    df = pd.DataFrame({"a": [1, 2, 3, 1, None, 4], "b": [1, 1, 1, 2, 5, 1], "c d": [1, 1, 1, 1, 1, 1],
                       "x": [0.5, 1.5, 2.5, 3.5, 4.5, 0.5]})
    df.to_parquet(d / "t.parquet")
    assert src.Source({"kind": "parquet", "path": str(d)}).nulls_and_repeats("t", ["a"]) is None
    monkeypatch.setattr(src, "LAZY_PARQUET_ROWS", 0)
    s = src.Source({"kind": "parquet", "path": str(d)})
    for cols in (["a"], ["a", "b"], ["b", "c d"], ["x"], ["x", "b"], ["c d"]):
        full = df[cols]
        assert s.nulls_and_repeats("t", cols) == (bool(full.isna().any().any()),
                                                  bool(full.dropna().duplicated().any())), cols


def test_introspect_keeps_one_self_reference_per_table(tmp_path):
    """posts.parent_id and posts.accepted_answer_id both point at posts (Stack Exchange): the run
    supports one self-reference, so the draft keeps the one with fewer NULLs."""
    import relsynth as rs
    from bench.builders.sqlite_ddl import create_table, insert_rows
    db = tmp_path / "p.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys = OFF")
    create_table(con, "posts", [("id", "INTEGER", True), ("parent_id", "INTEGER", False),
                                ("accepted_answer_id", "INTEGER", False), ("score", "INTEGER", True)], ["id"],
                 foreign_keys=[(["parent_id"], "posts", ["id"]), (["accepted_answer_id"], "posts", ["id"])])
    rows = []
    for q in range(1, 201):                      # questions 1..200, answers 201..800
        rows.append([q, None, 200 + q if q % 4 == 0 else None, q % 9])
    for a in range(201, 801):
        rows.append([a, 1 + a % 200, None, a % 5])
    insert_rows(con, "posts", ["id", "parent_id", "accepted_answer_id", "score"], rows)
    con.commit()
    con.close()
    spec = rs.introspect(f"sqlite:{db}", name="p")
    selfs = [r for r in spec.resolved["relationships"] if r["child"] == r["parent"] == "posts"]
    assert [r["child_key"] for r in selfs] == ["parent_id"]
    from relsynth.spec import load_spec, validate
    spec.save(tmp_path / "p.yaml")
    assert validate(load_spec(str(tmp_path / "p.yaml"))) == []
    assert any("posts.accepted_answer_id -> posts: a second self-reference" in x for x in spec.raw.get("review", []))

# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures.

Fake databases come from tests/fakedata/scenarios.py (known generating models).
Live database servers are used when their URL is in the environment:

    RELSYNTH_TEST_PG_URL      postgresql://user:pw@host/db
    RELSYNTH_TEST_MSSQL_URL   mssql://sa:pw@host:1433/master?TrustServerCertificate=yes&driver=ODBC+Driver+18+for+SQL+Server
    RELSYNTH_TEST_ORACLE_URL  oracle://user:pw@host:1521/?service_name=FREEPDB1
    RELSYNTH_TEST_MYSQL_URL   mysql://root:pw@host:3306/relsynth   (a user that may create databases)
    RELSYNTH_TEST_MARIADB_URL mariadb://root:pw@host:3307/relsynth
    RELSYNTH_TEST_DATABRICKS_URL databricks://host?http_path=/sql/1.0/warehouses/id&catalog=main
                              (credentials as in production: DATABRICKS_TOKEN or the SDK's chain)
    RELSYNTH_TEST_FABRIC_URL  fabric://<id>.datawarehouse.fabric.microsoft.com/<warehouse> (Azure
                              credential chain), or the SQL Server stand-in
                              fabric://sa:pw@host:1433/rs_fabric?TrustServerCertificate=yes
                              (a database in the UTF-8 collation: fabric.stand_in_database)

Extra driver arguments for a server come from RELSYNTH_TEST_<KIND>_CONNECT_ARGS
(JSON); the local Databricks stand-in (tests/emulators/databricks) needs them.
docker-compose.test.yml starts the five database servers; tests for a server
that is not configured are skipped.
"""
from __future__ import annotations

import copy
import logging
import os
import sys
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fakedata"))

import scenarios as S  # noqa: E402

from relsynth.data import write_tables  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402

SERVER_ENVS = {"postgresql": "RELSYNTH_TEST_PG_URL", "mssql": "RELSYNTH_TEST_MSSQL_URL",
               "oracle": "RELSYNTH_TEST_ORACLE_URL", "mysql": "RELSYNTH_TEST_MYSQL_URL",
               "mariadb": "RELSYNTH_TEST_MARIADB_URL", "databricks": "RELSYNTH_TEST_DATABRICKS_URL",
               "fabric": "RELSYNTH_TEST_FABRIC_URL"}


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    caplog.set_level(logging.WARNING, logger="relsynth")


_CACHE: dict = {}


def scenario(name: str):
    """(tables, spec, truth) of a fake scenario, built once per session."""
    if name not in _CACHE:
        _CACHE[name] = S.SCENARIOS[name]()
    t, s, truth = _CACHE[name]
    return {k: v.copy() for k, v in t.items()}, copy.deepcopy(s), truth


def write_scenario(name: str, target: dict) -> dict:
    """Write a scenario's tables to a target section; returns the spec reading them back."""
    t, s, _ = scenario(name)
    write_tables(load_spec(copy.deepcopy(s)), t, dict(target, structure=target.get("structure", "infer")))
    out = copy.deepcopy(s)
    out["source"] = {k: v for k, v in target.items() if k not in ("structure", "if_exists")}
    return out


@pytest.fixture(scope="session")
def star_sqlite(tmp_path_factory):
    """S2_star (customer, product, sales) in a SQLite file: (spec dict, db path)."""
    d = tmp_path_factory.mktemp("star")
    db = d / "star.db"
    spec = write_scenario("S2_star", {"kind": "sqlite", "path": str(db)})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec, db


@pytest.fixture(scope="session")
def star_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("starcsv")
    spec = write_scenario("S2_star", {"kind": "csv", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec, d / "src"


@pytest.fixture(scope="session")
def star_parquet(tmp_path_factory):
    pytest.importorskip("pyarrow")
    d = tmp_path_factory.mktemp("starpq")
    spec = write_scenario("S2_star", {"kind": "parquet", "path": str(d / "src")})
    spec["target"] = {"kind": "csv", "path": str(d / "out")}
    return spec, d / "src"


def server_url(kind: str) -> str:
    url = os.environ.get(SERVER_ENVS[kind])
    if not url:
        pytest.skip(f"{SERVER_ENVS[kind]} not set: no live {kind} server")
    return url


@pytest.fixture
def unique_schema():
    """A fresh schema / table-name prefix per test (servers are shared)."""
    return "rs_" + uuid.uuid4().hex[:10]


def server_target(kind: str, schema: str) -> dict:
    """A target section for a live server in its own schema (Oracle: the user's schema)."""
    url = server_url(kind)
    sec = {"kind": kind, "url": url}
    if kind != "oracle":
        sec["schema"] = schema
    args = server_connect_args(kind)
    if args:
        sec["connect_args"] = args
    return sec


def server_connect_args(kind: str) -> dict:
    """Extra driver arguments of a test server (RELSYNTH_TEST_<KIND>_CONNECT_ARGS, JSON)."""
    import json
    v = os.environ.get(SERVER_ENVS[kind].replace("_URL", "_CONNECT_ARGS"))
    return json.loads(v) if v else {}


def drop_schema(kind: str, schema: str, tables=()):
    """Remove what a server test created."""
    import sqlalchemy as sa

    from relsynth.connectors import connect
    url = os.environ.get(SERVER_ENVS[kind])
    if not url:
        return
    conn = connect(url, connect_args=server_connect_args(kind) or None)
    try:
        with conn.engine.begin() as con:
            if kind == "postgresql":
                con.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            elif kind in ("mssql", "fabric"):
                insp = sa.inspect(con)
                if schema in insp.get_schema_names():
                    md = sa.MetaData(schema=schema)
                    md.reflect(con)
                    md.drop_all(con)
                    con.execute(sa.text(f"DROP SCHEMA [{schema}]"))
            elif kind == "databricks":
                con.exec_driver_sql(f"DROP SCHEMA IF EXISTS `{schema}` CASCADE")
            elif kind in ("mysql", "mariadb"):     # a schema is a database
                con.execute(sa.text(f"DROP DATABASE IF EXISTS `{schema}`"))
            elif kind == "oracle":          # the user's schema: drop the named tables only
                insp = sa.inspect(con)
                for t in tables:
                    if insp.has_table(t):
                        con.execute(sa.text(f'DROP TABLE "{t.upper()}" CASCADE CONSTRAINTS PURGE'))
    finally:
        conn.close()

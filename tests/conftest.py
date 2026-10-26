# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures.

Fake databases come from tests/fakedata/scenarios.py (known generating models).
Live database servers are used when their URL is in the environment:

    RELSYNTH_TEST_PG_URL      postgresql://user:pw@host/db
    RELSYNTH_TEST_MSSQL_URL   mssql://sa:pw@host:1433/master?TrustServerCertificate=yes&driver=ODBC+Driver+18+for+SQL+Server
    RELSYNTH_TEST_ORACLE_URL  oracle://user:pw@host:1521/?service_name=FREEPDB1

docker-compose.test.yml starts all three; tests for a server that is not
configured are skipped.
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
               "oracle": "RELSYNTH_TEST_ORACLE_URL"}


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
    return sec


def drop_schema(kind: str, schema: str, tables=()):
    """Remove what a server test created."""
    import sqlalchemy as sa

    from relsynth.connectors import connect
    url = os.environ.get(SERVER_ENVS[kind])
    if not url:
        return
    conn = connect(url)
    try:
        with conn.engine.begin() as con:
            if kind == "postgresql":
                con.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            elif kind == "mssql":
                insp = sa.inspect(con)
                if schema in insp.get_schema_names():
                    md = sa.MetaData(schema=schema)
                    md.reflect(con)
                    md.drop_all(con)
                    con.execute(sa.text(f"DROP SCHEMA [{schema}]"))
            elif kind == "oracle":          # the user's schema: drop the named tables only
                insp = sa.inspect(con)
                for t in tables:
                    if insp.has_table(t):
                        con.execute(sa.text(f'DROP TABLE "{t.upper()}" CASCADE CONSTRAINTS PURGE'))
    finally:
        conn.close()

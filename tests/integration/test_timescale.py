# SPDX-License-Identifier: Apache-2.0
"""TimescaleDB as source and target (skipped unless RELSYNTH_TEST_TIMESCALE_URL
is set): hypertable detection, the introspect note, a series table synthesised
from a hypertable and written back as a hypertable."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

import relsynth as rs
from relsynth.connectors import connect

URL = os.environ.get("RELSYNTH_TEST_TIMESCALE_URL")
pytestmark = [pytest.mark.timescale,
              pytest.mark.skipif(not URL, reason="RELSYNTH_TEST_TIMESCALE_URL is not set")]


@pytest.fixture
def source():
    e = sa.create_engine(URL.replace("postgresql://", "postgresql+psycopg://", 1))
    with e.begin() as c:
        c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        c.execute(sa.text("DROP SCHEMA IF EXISTS rs_ts_src CASCADE; DROP SCHEMA IF EXISTS rs_ts_tgt CASCADE; "
                          "CREATE SCHEMA rs_ts_src"))
        c.execute(sa.text("CREATE TABLE rs_ts_src.meter (meter_id int primary key, kind text)"))
        c.execute(sa.text("CREATE TABLE rs_ts_src.reading (meter_id int references rs_ts_src.meter, "
                          "ts timestamptz not null, kw double precision)"))
        c.execute(sa.text("SELECT create_hypertable('rs_ts_src.reading', by_range('ts', INTERVAL '7 days'))"))
    rng = np.random.default_rng(0)
    n, T = 30, 24 * 40
    ts = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    Y = np.exp(2 + rng.normal(0, .5, (n, 1)) + 0.5 * np.sin(2 * np.pi * np.arange(T) / 24)[None, :]
               + 0.1 * rng.standard_normal((n, T)))
    pd.DataFrame({"meter_id": np.arange(1, n + 1), "kind": rng.choice(["home", "shop"], n)}).to_sql(
        "meter", e, schema="rs_ts_src", if_exists="append", index=False)
    pd.DataFrame({"meter_id": np.repeat(np.arange(1, n + 1), T), "ts": np.tile(ts, n), "kw": Y.reshape(-1)}).to_sql(
        "reading", e, schema="rs_ts_src", if_exists="append", index=False, chunksize=20000)
    yield n, T
    with e.begin() as c:
        c.execute(sa.text("DROP SCHEMA IF EXISTS rs_ts_src CASCADE; DROP SCHEMA IF EXISTS rs_ts_tgt CASCADE"))


def test_hypertables_in_and_out(source):
    n, T = source
    c = connect(URL, schema="rs_ts_src")
    assert c.timescale_version()
    assert c.hypertables()["reading"]["time_column"] == "ts"
    spec = rs.introspect(URL, schema="rs_ts_src")
    assert any("hypertable" in x.lower() for x in spec.raw.get("review", []))
    d = spec.raw
    d["tables"]["reading"]["series"] = {"entity": "meter_id", "time": "ts", "values": ["kw"]}
    d["tables"]["reading"]["columns"]["ts"]["type"] = "datetime"
    d["target"] = {"kind": "postgresql", "url": URL, "schema": "rs_ts_tgt",
                   "timescale": {"hypertables": "auto", "chunk_interval": "1 day"}}
    out = rs.fit(d).sample()
    assert len(out["reading"]) == n * T
    out.write()
    hyp = connect(URL, schema="rs_ts_tgt").hypertables()
    assert hyp["reading"]["time_column"] == "ts"

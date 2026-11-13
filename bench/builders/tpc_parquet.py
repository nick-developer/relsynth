# SPDX-License-Identifier: Apache-2.0
"""Generate a TPC-H or TPC-DS database at a scale factor with DuckDB's tpch /
tpcds extensions (dbgen, dsdgen) and write one Parquet file per table (G7).

    python tpc_parquet.py tpch|tpcds SF OUTDIR

The data is generated, never downloaded, so it is checked by row counts rather
than a checksum: TPC-H's fixed-size tables scale exactly with SF (TPC-H 2.x
clause 4.2.5) and the builder fails when they do not.  The tables carry no
declared keys (Parquet has none): introspect finds the keys and links from the
data, as it would on a lakehouse.

TPC terms (docs/datasets.md): results from this data are scale and integrity
tests of relsynth, "not comparable to TPC Benchmark Results", and are never
called TPC-H or TPC-DS results.
"""
import sys
from pathlib import Path

import duckdb

kind, sf, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
if kind not in ("tpch", "tpcds"):
    raise SystemExit("kind must be tpch or tpcds")
out.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(str(out.with_suffix(".duckdb")))     # on disk: SF10 does not fit in a runner's memory
con.execute(f"INSTALL {kind}")
con.execute(f"LOAD {kind}")
con.execute(f"CALL {'dbgen' if kind == 'tpch' else 'dsdgen'}(sf = {sf})")
tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")
          .fetchall()]
counts = {}
for t in sorted(tables):
    counts[t] = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
    path = str(out / f"{t}.parquet").replace("'", "''")
    con.execute(f"COPY \"{t}\" TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    print(f"{t}: {counts[t]} rows")
con.close()
out.with_suffix(".duckdb").unlink(missing_ok=True)
if kind == "tpch":
    expect = {"region": 5, "nation": 25, "supplier": 10_000 * sf, "customer": 150_000 * sf, "part": 200_000 * sf,
              "partsupp": 800_000 * sf, "orders": 1_500_000 * sf}
    bad = {t: (counts.get(t), n) for t, n in expect.items() if counts.get(t) != int(n)}
    if bad:
        raise SystemExit(f"TPC-H SF{sf}: row counts differ from the specification: {bad}")
elif len(tables) != 24:
    raise SystemExit(f"TPC-DS: {len(tables)} tables generated, 24 expected")

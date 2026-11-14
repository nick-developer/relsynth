# SPDX-License-Identifier: Apache-2.0
"""NYC TLC yellow taxi trips (NYC Open Data terms) as a two-table Parquet source (G7).

    python nyc_tlc_parquet.py ZONES.csv OUTDIR TRIPS.parquet [TRIPS.parquet ...]

`zone.parquet` is the taxi zone lookup (LocationID, Borough, Zone,
service_zone); `trip.parquet` is the monthly files in one file, with a trip_id
added (file order, then row order) and every other column as published:
pick-up and drop-off zones link to `zone`, and the messy values (negative
fares, zero distances, unknown zones 264/265, passenger counts of 0 or NULL)
stay.  Written with DuckDB, so twelve months (about 40M rows) never pass
through pandas.
"""
import sys
from pathlib import Path

import duckdb

zones, out, months = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)


def lit(p) -> str:
    return "'" + str(p).replace("'", "''") + "'"


con = duckdb.connect()
con.execute(f"COPY (SELECT * FROM read_csv({lit(zones)}, header = true) ORDER BY LocationID) "
            f"TO {lit(out / 'zone.parquet')} (FORMAT parquet)")
files = "[" + ", ".join(lit(p) for p in months) + "]"
# union_by_name: a column added in a later month (e.g. a new fee) is NULL in the earlier ones
con.execute(f"COPY (SELECT row_number() OVER (ORDER BY filename, file_row_number) AS trip_id, "
            f"* EXCLUDE (filename, file_row_number) FROM read_parquet({files}, union_by_name = true, "
            f"filename = true, file_row_number = true)) TO {lit(out / 'trip.parquet')} (FORMAT parquet, "
            "COMPRESSION zstd)")
for t in ("zone", "trip"):
    print(t, con.execute(f"SELECT count(*) FROM read_parquet({lit(out / (t + '.parquet'))})").fetchone()[0], "rows")

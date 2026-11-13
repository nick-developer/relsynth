# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite database from a dataset in SDV's public demo format, with the
keys of its metadata (G7: Adult, CoverType, Census-Income KDD, Synthea 1k).

    python sdv_sqlite.py DATA.zip|DIR METADATA.json OUT.db

DATA.zip holds one CSV per table (data/<table>.csv), or DIR holds <table>.csv
(Synthea's own CSV export, read with the 1K sample's metadata: columns that the
metadata does not list are kept with the type pandas reads, listed ones that a
newer export lacks are skipped); METADATA.json is SDV's
metadata (primary keys where the table has one, column types, relationships).
Every relationship becomes a declared FOREIGN KEY, so introspect reads the
schema the way it reads a production catalog; a relationship whose parent has
no declared key is skipped (SQLite cannot reference it).  Datetime columns are
written as ISO dates, or ISO timestamps where any value has a time of day; the
numbers and categories are written as they are.  A single table without a key
gets a surrogate `row_id` primary key (row number from 1), as the ACS builder
gives `pid`: introspect leaves a lone table without a key out of the draft.  The Berka builder
(berka_sqlite.py) is the same idea, kept as it was written for the 2.0 baseline.
"""
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sqlite_ddl import create_table

data_zip, meta_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads(meta_path.read_text(encoding="utf-8"))
keyed = {t for t, tm in meta["tables"].items() if tm.get("primary_key")}
fks: dict = {}
for r in meta.get("relationships", []):
    if r["parent_table_name"] in keyed:
        fks.setdefault(r["child_table_name"], []).append((r["child_foreign_key"], r["parent_table_name"],
                                                          r["parent_primary_key"]))

SQLTYPE = {"id": "INTEGER", "numerical": "NUMERIC", "datetime": "DATE", "categorical": "TEXT", "boolean": "INTEGER"}


def depth(t, seen=()):
    return 1 + max((depth(p, (*seen, t)) for _, p, _ in fks.get(t, []) if p not in seen), default=0)


out.unlink(missing_ok=True)
con = sqlite3.connect(out)
if data_zip.is_dir():
    files = {p.stem: p for p in data_zip.glob("*.csv")}
    read = lambda t: pd.read_csv(files[t], low_memory=False)  # noqa: E731
else:
    z = zipfile.ZipFile(data_zip)
    files = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}
    read = lambda t: pd.read_csv(io.BytesIO(z.read(files[t])), low_memory=False)  # noqa: E731
for t in sorted((t for t in meta["tables"] if t in files), key=lambda t: (depth(t), t)):
    tm = meta["tables"][t]
    df = read(t)
    listed = {c: cm for c, cm in tm["columns"].items() if c in df.columns}
    for c in df.columns:
        if c not in listed:
            listed[c] = {"sdtype": "numerical" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"}
    cols = []
    for c, cm in listed.items():
        st = cm["sdtype"]
        ty = SQLTYPE.get(st, "TEXT")
        if st == "id" and not pd.api.types.is_numeric_dtype(df[c]):
            ty = "TEXT"
        if st == "numerical" and pd.api.types.is_integer_dtype(df[c]):
            ty = "INTEGER"
        if st == "datetime":
            v = pd.to_datetime(df[c], errors="coerce", format=cm.get("datetime_format"), utc=True)
            timed = bool(((v.dt.hour != 0) | (v.dt.minute != 0) | (v.dt.second != 0)).any())
            ty = "DATETIME" if timed else "DATE"
            df[c] = v.dt.strftime("%Y-%m-%d %H:%M:%S" if timed else "%Y-%m-%d").where(v.notna(), None)
        cols.append((c, ty, c == tm.get("primary_key")))
    pk = [tm["primary_key"]] if tm.get("primary_key") else []
    if not pk and len(meta["tables"]) == 1:
        df.insert(0, "row_id", range(1, len(df) + 1))
        cols.insert(0, ("row_id", "INTEGER", True))
        pk = ["row_id"]
    create_table(con, t, cols, primary_key=pk,
                 foreign_keys=[([c], p, [k]) for c, p, k in fks.get(t, []) if c in listed and p in files])
    df[[c for c, _, _ in cols]].to_sql(t, con, if_exists="append", index=False, chunksize=50_000)
    print(f"{t}: {len(df)} rows")
    del df
con.commit()
con.close()

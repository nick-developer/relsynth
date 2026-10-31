# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite copy of the Berka (PKDD'99 Czech bank, "financial") database
from the SDV public demo copy, with the keys of its metadata.

    python berka_sqlite.py DATA.zip METADATA.json OUT.db

DATA.zip holds one CSV per table (data/<table>.csv); METADATA.json is SDV's
multi-table metadata (primary keys, column types, relationships).  Every
relationship becomes a declared FOREIGN KEY, so introspect reads the schema the
way it reads a production catalog.  Datetime columns are written as ISO dates,
the numbers and categories as they are.  `order` is an SQL keyword and stays the
table's name: quoting it is part of the test.
"""
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd

data_zip, meta_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads(meta_path.read_text(encoding="utf-8"))
fks: dict = {}
for r in meta["relationships"]:
    fks.setdefault(r["child_table_name"], []).append((r["child_foreign_key"], r["parent_table_name"],
                                                      r["parent_primary_key"]))

with zipfile.ZipFile(data_zip) as z:
    files = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv")}
    frames = {t: pd.read_csv(io.BytesIO(z.read(files[t])), low_memory=False) for t in meta["tables"]}

SQLTYPE = {"id": "INTEGER", "numerical": "NUMERIC", "datetime": "DATE", "categorical": "TEXT", "boolean": "INTEGER"}


def depth(t, seen=()):
    return 1 + max((depth(p, (*seen, t)) for _, p, _ in fks.get(t, []) if p not in seen), default=0)


out.unlink(missing_ok=True)
con = sqlite3.connect(out)
for t in sorted(meta["tables"], key=depth):
    tm, df = meta["tables"][t], frames[t]
    parts = []
    for c, cm in tm["columns"].items():
        st = cm["sdtype"]
        ty = SQLTYPE.get(st, "TEXT")
        if st == "id" and df[c].dtype == object:
            ty = "TEXT"
        if st == "numerical" and pd.api.types.is_integer_dtype(df[c]):
            ty = "INTEGER"
        parts.append(f'"{c}" {ty}{" NOT NULL" if c == tm.get("primary_key") else ""}')
        if st == "datetime":
            v = pd.to_datetime(df[c], errors="coerce", format=cm.get("datetime_format"))
            df[c] = v.dt.strftime("%Y-%m-%d").where(v.notna(), None)
    parts.append(f'PRIMARY KEY ("{tm["primary_key"]}")')
    for c, p, pk in fks.get(t, []):
        parts.append(f'FOREIGN KEY ("{c}") REFERENCES "{p}" ("{pk}")')
    con.execute(f'CREATE TABLE "{t}" ({", ".join(parts)})')
    df[list(tm["columns"])].to_sql(t, con, if_exists="append", index=False, chunksize=50_000)
    print(f"{t}: {len(df)} rows")
con.commit()
con.close()

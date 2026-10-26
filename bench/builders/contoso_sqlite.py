# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite copy of a Contoso Data Generator V2 release (CSV or Parquet files)
with the keys of its SQL Server scripts.

    python build_sqlite.py SQL_DIR DATA_DIR OUT.db

SQL_DIR holds CreateTablesCommon.sql, CreateTablesSales.sql and CreateTablesOrders.sql
(scripts/sql in sql-bi/Contoso-Data-Generator-V2).  The T-SQL DDL is parsed
generically: columns and types, inline PRIMARY KEY constraints (composite keys
kept), and ALTER TABLE ... FOREIGN KEY.  Files are matched to tables without
case; columns are matched by name, or by position when the file has the table's
column count but other names (the Parquet release calls OrderDate "DT" and
LineNumber "RowNumber").  Dates are written as ISO dates, decimals as numbers.
"""
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sql_dir, csv_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ddl = "\n".join((sql_dir / f).read_text(encoding="utf-8-sig", errors="replace")
                for f in ("CreateTablesCommon.sql", "CreateTablesSales.sql", "CreateTablesOrders.sql"))

TYPE = [(r"^(tinyint|smallint|int|bigint)$", "INTEGER"), (r"^bit$", "INTEGER"), (r"^(money|decimal|numeric)", "NUMERIC"),
        (r"^(float|real)$", "FLOAT"), (r"^date$", "DATE"), (r"^datetime", "DATETIME"), (r"", "TEXT")]


def sqltype(t):
    for pat, s in TYPE:
        if re.search(pat, t.lower()):
            return s


tables = {}
for m in re.finditer(r"CREATE TABLE \[\w+\]\.\[(\w+)\]\s*\((.*?)\n\s*\)\s*ON", ddl, re.S):
    cols, pk = [], None
    for line in m.group(2).splitlines():
        c = re.match(r"\s*\[(\w+)\]\s*\[(\w+)\]\s*(\([^)]*\))?\s*(NOT NULL|NULL)?", line)
        if c and not line.strip().upper().startswith("CONSTRAINT"):
            cols.append((c.group(1), sqltype(c.group(2)), (c.group(4) or "").upper() == "NOT NULL"))
        k = re.search(r"PRIMARY KEY\s*\w*\s*\(([^)]*)\)", line)
        if k:
            pk = [re.sub(r"[\[\]]|\s+(ASC|DESC)", "", x).strip() for x in k.group(1).split(",")]
    tables[m.group(1)] = {"cols": cols, "pk": pk, "fk": []}

for m in re.finditer(r"ALTER TABLE \[\w+\]\.\[(\w+)\] WITH CHECK ADD\s+CONSTRAINT \[\w+\]\s+FOREIGN KEY\s*\(([^)]*)\)\s*"
                     r"REFERENCES \[\w+\]\.\[(\w+)\]\s*\(([^)]*)\)", ddl):
    strip = lambda s: [x.strip(" []") for x in s.split(",")]
    tables[m.group(1)]["fk"].append((strip(m.group(2)), m.group(3), strip(m.group(4))))

files = {p.stem.lower(): p for p in sorted(csv_dir.glob("*.csv")) + sorted(csv_dir.glob("*.parquet"))}
out.unlink(missing_ok=True)
con = sqlite3.connect(out)
order = sorted(tables, key=lambda t: len(tables[t]["fk"]))
for t in order:
    T = tables[t]
    parts = [f'"{c}" {ty}{" NOT NULL" if nn else ""}' for c, ty, nn in T["cols"]]
    if T["pk"]:
        parts.append("PRIMARY KEY (" + ", ".join(f'"{c}"' for c in T["pk"]) + ")")
    for cs, rt, rc in T["fk"]:
        parts.append(f'FOREIGN KEY ({", ".join(cs)}) REFERENCES "{rt}" ({", ".join(rc)})')
    con.execute(f'CREATE TABLE "{t}" ({", ".join(parts)})')
    f = files.get(t.lower())
    if f is None:
        print(f"{t}: no CSV file")
        continue
    df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)
    names = [c for c, _, _ in T["cols"]]
    if len(df.columns) == len(names) and {c.lower() for c in df.columns} != {c.lower() for c in names}:
        print(f"{t}: columns matched by position: " + ", ".join(f"{a}->{b}" for a, b in zip(df.columns, names) if a != b))
        df.columns = names
    types = {c: ty for c, ty, _ in T["cols"]}
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(df[c]):
            v = df[c].dt.tz_localize(None) if isinstance(df[c].dtype, pd.DatetimeTZDtype) else df[c]
            df[c] = v.dt.strftime("%Y-%m-%d").where(v.notna(), None)
        elif df[c].dtype == object and types.get(c) in ("NUMERIC", "FLOAT"):
            df[c] = pd.to_numeric(df[c].astype(str).where(df[c].notna(), None), errors="coerce")
    lower = {c.lower(): c for c in df.columns}
    missing = [c for c in names if c.lower() not in lower]
    extra = [c for c in df.columns if c.lower() not in {n.lower() for n in names}]
    df = df[[lower[c.lower()] for c in names if c.lower() in lower]]
    df.columns = [c for c in names if c.lower() in lower]
    con.executemany(f'INSERT INTO "{t}" ({", ".join(chr(34) + c + chr(34) for c in df.columns)}) VALUES '
                    f'({", ".join("?" * df.shape[1])})',
                    df.astype(object).where(df.notna(), None).itertuples(index=False, name=None))
    print(f"{t}: {len(df):,} rows, pk {T['pk']}, {len(T['fk'])} fks" +
          (f", missing columns {missing}" if missing else "") + (f", extra columns {extra}" if extra else ""))
con.commit()
con.close()

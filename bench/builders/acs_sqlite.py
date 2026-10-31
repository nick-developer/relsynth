# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite copy of the NIST CRC "diverse communities" ACS excerpt
(American Community Survey 2019 PUMS, national sample, 27,253 people).

    python acs_sqlite.py EXCERPTS.zip OUT.db

EXCERPTS.zip is diverse_communities_data_excerpts.zip of the SDNist release.
The excerpt codes "not applicable" as the letter N in otherwise numeric columns
(INDP, POVPIP, NOC, ...); its data dictionary defines N as a missing value, so it
is written as NULL and the column keeps its numeric type.  Each person gets a
surrogate key `pid` (the excerpt has none) so that the person can be the privacy
unit.  PUMA ("01-01301") stays a code.
"""
import io
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(src) as z:
    df = pd.read_csv(io.BytesIO(z.read("diverse_communities_data_excerpts/national/national2019.csv")),
                     dtype=str, keep_default_na=False)
df = df.replace({"N": None, "": None})
for c in df.columns:
    if c != "PUMA":
        df[c] = pd.to_numeric(df[c])
        if (df[c].dropna() % 1 == 0).all():
            df[c] = df[c].astype("Int64")
df.insert(0, "pid", range(1, len(df) + 1))

out.unlink(missing_ok=True)
con = sqlite3.connect(out)
cols = ", ".join(f'"{c}" {"TEXT" if c == "PUMA" else "INTEGER" if str(df[c].dtype) == "Int64" else "REAL"}'
                 + (" NOT NULL" if c == "pid" else "") for c in df.columns)
con.execute(f'CREATE TABLE acs ({cols}, PRIMARY KEY ("pid"))')
df.to_sql("acs", con, if_exists="append", index=False)
con.commit()
con.close()
print(f"acs: {len(df)} rows")

# SPDX-License-Identifier: Apache-2.0
"""Build a SQLite copy of the MIMIC-IV Clinical Database Demo 2.2 (ODbL 1.0) (G7).

    python mimic_sqlite.py DEMO.zip OUT.db

The zip holds hosp/*.csv.gz and icu/*.csv.gz (100 patients); each file becomes
one table named after the file (module prefixed where two modules share a name).
No keys are declared, as in the published CSV release: introspect finds them
from the data.  Derived databases are ODbL: the harness keeps them in the
ignored cache and publishes aggregates only.
"""
import io
import sqlite3
import sys
import zipfile
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.unlink(missing_ok=True)
con = sqlite3.connect(out)
with zipfile.ZipFile(src) as z:
    files = sorted(n for n in z.namelist() if n.endswith(".csv.gz") and ("/hosp/" in n or "/icu/" in n))
    stems = [Path(n).name[:-len(".csv.gz")] for n in files]
    for n, stem in zip(files, stems):
        name = stem if stems.count(stem) == 1 else f"{Path(n).parent.name}_{stem}"
        df = pd.read_csv(io.BytesIO(z.read(n)), compression="gzip", low_memory=False)
        df.to_sql(name, con, index=False, chunksize=50_000)
        print(f"{name}: {len(df)} rows")
con.commit()
con.close()
if not files:
    raise SystemExit("no hosp/ or icu/ CSV files in the zip")

# SPDX-License-Identifier: Apache-2.0
"""An extract of the Open Food Facts product database (ODbL; contents DbCL) (G7).

    python openfoodfacts_extract.py food.parquet OUTDIR [ROWS]

The Hugging Face Parquet export (8 GB, one row per product) is read with DuckDB
and ROWS products (default 200,000; one row per barcode, the latest when the
export repeats one; a fixed hash order, so the same file gives the same extract) are written to OUTDIR/product.parquet with the scalar and
long-text columns, and OUTDIR/product_tag.parquet with one row per product and
category tag (the nested list as a child table).  Derived databases are ODbL:
the harness keeps them in the ignored cache and publishes aggregates only.
"""
import sys
from pathlib import Path

import duckdb

src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
out.mkdir(parents=True, exist_ok=True)


def lit(p) -> str:
    return "'" + str(p).replace("'", "''") + "'"


con = duckdb.connect()
cols = {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({lit(src)})").fetchall()}
scalar = [c for c, t in cols.items() if not t.endswith("[]") and not t.startswith(("STRUCT", "MAP"))
          and c != "code"]
# the export repeats some barcodes: one row per code, or the extract has no key (introspect then leaves
# both tables out of the draft)
latest = "last_modified_t DESC NULLS LAST, " if "last_modified_t" in cols else ""
con.execute(f"CREATE TABLE pick AS SELECT * FROM read_parquet({lit(src)}) AS f WHERE code IS NOT NULL "
            f"QUALIFY row_number() OVER (PARTITION BY code ORDER BY {latest}md5(CAST(f AS VARCHAR))) = 1 "
            f"ORDER BY hash(code), code LIMIT {rows}")
sel = ", ".join(f'"{c}"' for c in scalar)
con.execute(f"COPY (SELECT code, {sel} FROM pick ORDER BY code) TO {lit(out / 'product.parquet')} (FORMAT parquet)")
if "categories_tags" in cols:
    con.execute("COPY (SELECT code, unnest(categories_tags) AS tag FROM pick ORDER BY code, tag) "
                f"TO {lit(out / 'product_tag.parquet')} (FORMAT parquet)")
for f in sorted(out.glob("*.parquet")):
    print(f.stem, con.execute(f"SELECT count(*) FROM read_parquet({lit(f)})").fetchone()[0], "rows")

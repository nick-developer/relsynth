# SPDX-License-Identifier: Apache-2.0
"""Facts whose partner link is optional: rows with a NULL partner key are modelled
as rows without a partner (facts.null_partner, Prop. 1843), not left out."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest


def _header_detail_db(path, n=600, seed=0):
    """emar (anchor) -> detail <- pharmacy (partner): each emar row has one header row
    (no pharmacy, a status) and one to three detail rows (a pharmacy, a dose)."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ward (ward_id INTEGER PRIMARY KEY, kind TEXT)")
    con.execute("CREATE TABLE nurse (nurse_id INTEGER PRIMARY KEY, grade TEXT)")
    con.execute("CREATE TABLE pharmacy (pharmacy_id INTEGER PRIMARY KEY, drug TEXT, strength REAL)")
    con.execute("CREATE TABLE emar (emar_id INTEGER PRIMARY KEY, ward_id INTEGER REFERENCES ward(ward_id), route TEXT)")
    con.execute("CREATE TABLE detail (detail_id INTEGER PRIMARY KEY, emar_id INTEGER REFERENCES emar(emar_id), "
                "pharmacy_id INTEGER REFERENCES pharmacy(pharmacy_id), nurse_id INTEGER REFERENCES nurse(nurse_id), "
                "status TEXT, dose REAL)")
    con.executemany("INSERT INTO ward VALUES (?, ?)", [(i, str(rng.choice(["icu", "gen"]))) for i in range(20)])
    con.executemany("INSERT INTO nurse VALUES (?, ?)", [(i, str(rng.choice(["a", "b"]))) for i in range(10)])
    con.executemany("INSERT INTO pharmacy VALUES (?, ?, ?)",
                    [(i, f"d{i % 15}", float(rng.gamma(2, 5))) for i in range(200)])
    con.executemany("INSERT INTO emar VALUES (?, ?, ?)",
                    [(i, int(rng.integers(0, 20)), str(rng.choice(["iv", "po"]))) for i in range(n)])
    rows, j = [], 0
    for i in range(n):
        # the header row is signed by a senior nurse (0-2), detail rows by any nurse
        rows.append((j, i, None, int(rng.integers(0, 3)), str(rng.choice(["given", "held", "refused"], p=[.8, .15, .05])),
                     None))
        j += 1
        for _ in range(int(rng.integers(1, 4))):
            rows.append((j, i, int(rng.integers(0, 200)), int(rng.integers(0, 10)), None,
                         float(np.round(rng.gamma(2, 3), 1))))
            j += 1
    con.executemany("INSERT INTO detail VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return f"sqlite:{path}"


def _shares(df):
    return {"null_partner": float(df["pharmacy_id"].isna().mean()),
            "header_per_anchor": float(df["pharmacy_id"].isna().groupby(df["emar_id"]).sum().mean()),
            "status_with_partner": int((df["status"].notna() & df["pharmacy_id"].notna()).sum()),
            "dose_without_partner": int((df["dose"].notna() & df["pharmacy_id"].isna()).sum()),
            "junior_on_header": float((df.loc[df["pharmacy_id"].isna(), "nurse_id"].astype(float) >= 3).mean())
            if df["pharmacy_id"].isna().any() else 0.0}


@pytest.mark.parametrize("mode,pushdown", [("model", "off"), ("model", "on"), ("drop", "off")])
def test_null_partner_rows(tmp_path, mode, pushdown):
    import relsynth as rs
    src = _header_detail_db(tmp_path / "hd.db")
    spec = rs.introspect(src, name="hd").set("synthesis.facts.null_partner", mode)
    spec = spec.set("synthesis.pushdown.mode", pushdown).set("synthesis.pushdown.sample_rows", 500)
    assert spec.raw["tables"]["detail"]["fact"]["partner"] == "pharmacy"
    model = rs.fit(spec)
    syn = model.sample()["detail"]
    real = pd.read_sql("SELECT * FROM detail", sqlite3.connect(tmp_path / "hd.db"))
    r, s = _shares(real), _shares(syn)
    if mode == "drop":                                     # 2.2: the header rows are lost
        assert s["null_partner"] == 0
        return
    assert abs(s["null_partner"] - r["null_partner"]) < 0.03
    assert s["header_per_anchor"] == pytest.approx(1.0, abs=0.02)   # exactly one in the source
    assert s["status_with_partner"] == 0 and s["dose_without_partner"] == 0
    assert s["junior_on_header"] < 0.05                    # the extra link's law given a missing partner
    assert "null_partner" in model.log["decisions"]["detail"]

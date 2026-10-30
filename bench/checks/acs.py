# SPDX-License-Identifier: Apache-2.0
"""ACS PUMS (NIST CRC excerpt) ground-truth checks on one person table.

Independent of relsynth.  A1-A11 are the survey's structural rules, identities
on the real data (share 1.0): which items are not asked of whom (the NULLs of
the excerpt), group-quarters coding, and the code lists that are functions of
another column.  A12-A14 are measures recorded without a verdict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load(folder) -> dict:
    p = Path(folder) / "acs.csv"
    return {"acs": pd.read_csv(p, low_memory=False)} if p.exists() else {}


def _share(mask) -> float | None:
    mask = pd.Series(mask).dropna()
    return float(mask.mean()) if len(mask) else None


def _pairs_seen(d, ref, a, b) -> float | None:
    seen = set(zip(ref[a], ref[b]))
    sub = d.dropna(subset=[a, b])
    return _share([p in seen for p in zip(sub[a], sub[b])])


def checks(T: dict, ref: dict) -> dict:
    d, r = T["acs"], ref["acs"]
    n = d.isna()
    out = {
        "A1 industry and industry group missing together": _share(n["INDP"] == n["INDP_CAT"]),
        "A2 industry group is the industry's group": _pairs_seen(d, r, "INDP", "INDP_CAT"),
        "A3 income and income decile missing together": _share(n["PINCP"] == n["PINCP_DECILE"]),
        "A4 income asked from age 15": _share(n["PINCP"] == (d["AGEP"] < 15)),
        "A5 education asked from age 3": _share(n["EDU"] == (d["AGEP"] < 3)),
        "A6 cognitive difficulty asked from age 5": _share(n["DREM"] == (d["AGEP"] < 5)),
        "A7 no industry under age 16": _share(n["INDP"][d["AGEP"] < 16]),
        "A8 households own or rent, group quarters do not": _share((d["HOUSING_TYPE"] == 1) == (d["OWN_RENT"] > 0)),
        "A9 housing weight zero exactly in group quarters": _share((d["WGTP"] == 0) == (d["HOUSING_TYPE"] > 1)),
        "A10 density is the PUMA's density": _pairs_seen(d, r, "PUMA", "DENSITY"),
        "A11 poverty ratio capped at 501": _share(d["POVPIP"].dropna() <= 501),
    }
    med = d.dropna(subset=["PINCP", "PINCP_DECILE"]).groupby("PINCP_DECILE")["PINCP"].median().sort_index()
    out["A12 income median rises with the decile (share of steps)"] = _share(np.diff(med.values) > 0) \
        if len(med) > 1 else None
    out["A13 mean age"] = float(d["AGEP"].mean())
    out["A14 median income of earners"] = float(d["PINCP"].dropna().median()) if n["PINCP"].sum() < len(d) else None
    return out

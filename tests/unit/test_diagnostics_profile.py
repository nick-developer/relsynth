# SPDX-License-Identifier: Apache-2.0
"""The column profile of the diagnostics bundle on nullable pandas dtypes."""
from __future__ import annotations

import json

import pandas as pd

from relsynth.diagnostics import profile


def test_profile_constant_nullable_integer_column():
    # AdventureWorks DW: a constant Int64 column made `relsynth run --bundle` fail (skew is pd.NA)
    df = pd.DataFrame({"k": pd.array([1, 2, 3, 4], dtype="Int64"), "c": pd.array([7, 7, 7, None], dtype="Int64"),
                       "x": pd.array([0.5, 1.5, 9.0, 2.0], dtype="Float64")})
    spec = {"tables": {"t": {"columns": {"k": {"type": "id"}, "c": {"type": "integer"}, "x": {"type": "float"}}}},
            "relationships": []}
    p = profile(spec, {"t": df})["t"]["columns"]
    assert p["c"]["skew"] is None
    assert isinstance(p["x"]["skew"], float) and p["x"]["skew"] > 0
    json.dumps(p, allow_nan=False)

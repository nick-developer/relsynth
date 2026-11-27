# SPDX-License-Identifier: Apache-2.0
"""Code columns read with a nullable (masked) dtype, as a pushed-down sample reads a Parquet code
column with nulls: they become text codes, nulls stay missing, on every pandas version."""
from __future__ import annotations

import pandas as pd
import pytest

from relsynth.data import coerce


@pytest.mark.parametrize("values,dtype,want", [
    ([1, None, 2], "Int64", ["1", None, "2"]),
    ([1.0, None, 2.0], "Float64", ["1", None, "2"]),
    ([1.5, None, 2.0], "Float64", ["1.5", None, "2.0"]),
    ([True, None, False], "boolean", ["True", None, "False"]),
    ([1.0, None, 2.0], "float64", ["1", None, "2"]),
])
def test_masked_codes_become_text(values, dtype, want):
    df = pd.DataFrame({"c": pd.array(values, dtype=dtype)})
    out = coerce(df, {"columns": {"c": {"type": "categorical"}}})["c"]
    assert [None if pd.isna(v) else v for v in out] == want


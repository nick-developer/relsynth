# SPDX-License-Identifier: Apache-2.0
"""Date parsing in profiling when a column mixes UTC offsets (pandas 3 raises on those)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from relsynth.profiling import canon, is_dateish, parse_dates

MIXED = pd.Series(["2020-01-01 10:00:00+01:00", "2020-01-02 10:00:00-05:00", "2020-01-03", None], dtype=object)


def test_parse_dates_converts_mixed_offsets_to_utc():
    d = parse_dates(MIXED)
    assert d.dt.tz is None
    assert list(d[:3]) == [pd.Timestamp("2020-01-01 09:00"), pd.Timestamp("2020-01-02 15:00"),
                           pd.Timestamp("2020-01-03")]
    assert pd.isna(d[3])


def test_parse_dates_leaves_naive_values_unchanged():
    s = pd.Series(["2020-01-01", "2020-01-02 06:00:00", "not a date"], dtype=object)
    pd.testing.assert_series_equal(parse_dates(s), pd.to_datetime(s, errors="coerce", format="mixed"))


def test_canon_and_is_dateish_accept_mixed_offsets():
    assert is_dateish(MIXED)
    out = canon(MIXED, as_date=True)
    assert np.allclose(out[:3].astype(float), [18262 + 9 / 24, 18263 + 15 / 24, 18264])
    assert out[3] is None

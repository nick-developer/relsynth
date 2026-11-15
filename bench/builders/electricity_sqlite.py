# SPDX-License-Identifier: Apache-2.0
"""UCI ElectricityLoadDiagrams20112014 (CC BY 4.0) as a regular hourly series (G7).

    python electricity_sqlite.py electricity.txt.gz OUT.db [HOURS]

The T12 build (benchmarks/timeseries/run_timeseries.py): the last HOURS
(default 8760, one year) of the 321 clients' hourly load, as Meter(MeterKey)
and Load(MeterKey, Hour, kW), one row per meter and hour.
"""
import gzip
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
hours = int(sys.argv[3]) if len(sys.argv) > 3 else 8760
Y = np.loadtxt(gzip.open(src), delimiter=",")[-hours:, :].T
n, T = Y.shape
ts = pd.date_range("2014-01-01", periods=T, freq="h").strftime("%Y-%m-%d %H:%M:%S")
out.unlink(missing_ok=True)
con = sqlite3.connect(out)
pd.DataFrame({"MeterKey": np.arange(1, n + 1)}).to_sql("Meter", con, index=False)
pd.DataFrame({"MeterKey": np.repeat(np.arange(1, n + 1), T), "Hour": np.tile(ts, n),
              "kW": Y.reshape(-1)}).to_sql("Load", con, index=False, chunksize=200000)
con.close()
print(f"Meter {n}, Load {n * T}")

# SPDX-License-Identifier: Apache-2.0
"""The Snowflake stand-in: fakesnow's HTTP server (Snowflake's REST protocol over
DuckDB), with one fix for the tests.

fakesnow 0.12 sends a NULL TIMESTAMP_NTZ as the struct (epoch 0, fraction 0),
which the Snowflake connector reads as 1970-01-01; the fix keeps the NULL by
giving the struct array the column's null mask.

    python tests/emulators/snowflake/serve.py [PORT]      (default 8085)
"""
from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.compute as pc

import fakesnow.arrow
import fakesnow.cli

_struct = fakesnow.arrow.timestamp_to_sf_struct


def timestamp_to_sf_struct(ts):
    if isinstance(ts, pa.ChunkedArray):
        ts = ts.combine_chunks()
    out = _struct(ts.fill_null(0) if ts.null_count else ts)
    if not ts.null_count:
        return out
    return pa.StructArray.from_arrays([out.field(i) for i in range(out.type.num_fields)],
                                      fields=list(out.type), mask=pc.is_null(ts))


fakesnow.arrow.timestamp_to_sf_struct = timestamp_to_sf_struct

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8085"
    sys.exit(fakesnow.cli.main(["-s", "-p", port]))

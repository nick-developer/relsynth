"""Values converted to the target column's type before a load, so that every
engine receives what its column accepts (an integral float 3.0 for an INTEGER
column, a timestamp for a DATE column, text for a VARCHAR column)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import sqlalchemy as sa


def _kind(ty) -> str:
    try:
        py = ty.python_type
    except (NotImplementedError, AttributeError):
        return "other"
    import datetime as dt
    import decimal
    if isinstance(ty, sa.Boolean) or py is bool:
        return "bool"
    if py is int:
        return "int"
    if py in (float, decimal.Decimal):
        return "float"
    if py is dt.datetime:
        return "datetime"
    if py is dt.date:
        return "date"
    if py is str:
        return "str"
    return "other"


def prepare_frame(df: pd.DataFrame, tbl: sa.Table) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        s = df[c]
        if c not in tbl.c:
            out[c] = s
            continue
        k = _kind(tbl.c[c].type)
        if k == "int" and not pd.api.types.is_integer_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            if pd.api.types.is_bool_dtype(num):
                num = num.astype("Int64")
            out[c] = num.round().astype("Int64") if len(num.dropna()) == 0 or \
                np.allclose(num.dropna(), np.round(num.dropna())) else num
        elif k == "float" and not pd.api.types.is_float_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce").astype(float)
        elif k == "date":
            d = pd.to_datetime(s, errors="coerce")
            out[c] = pd.Series([None if pd.isna(x) else x.date() for x in d], index=s.index, dtype=object)
        elif k == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
            out[c] = pd.to_datetime(s, errors="coerce")
        elif k == "bool" and not pd.api.types.is_bool_dtype(s):
            out[c] = s.map(lambda v: None if v is None or (isinstance(v, float) and v != v) or v is pd.NA
                           else bool(v) if not isinstance(v, str) else v.strip().lower() in ("1", "true", "t", "yes", "y"))
        elif k == "str" and not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            if pd.api.types.is_float_dtype(s):
                v = s.to_numpy(dtype=float)
                txt = [None if x != x else (str(int(x)) if float(x).is_integer() else repr(float(x))) for x in v]
                out[c] = pd.Series(txt, index=s.index, dtype=object)
            elif pd.api.types.is_datetime64_any_dtype(s):
                out[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S").where(s.notna(), None)
            else:
                out[c] = s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v))
        else:
            out[c] = s
    return pd.DataFrame(out, index=df.index)

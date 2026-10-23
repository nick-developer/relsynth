"""Reading and writing tables: CSV folders, SQLite files, or any SQLAlchemy URL
(PostgreSQL, Oracle, MS SQL Server, ...)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve(spec, p):
    p = Path(p)
    return p if p.is_absolute() else Path(spec.get("_base_dir", ".")) / p


def inside(root, name, suffix=""):
    """root / (name + suffix), refused when it resolves outside root (a table name
    such as '../x' or an absolute path must not write or read elsewhere)."""
    name = str(name)
    if (not name or name in (".", "..") or "\x00" in name or "/" in name or "\\" in name
            or Path(name).is_absolute()):
        raise ValueError(f"table name {name!r} is not a plain file name")
    root = Path(root).resolve()
    p = (root / f"{name}{suffix}").resolve()
    if p.parent != root and root not in p.parents:
        raise ValueError(f"table name {name!r} points outside the folder {root}")
    return p


_PW_KEYS = r"password|passwd|pwd|secret|token|api_?key|credential"


_ISO_TZ = r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:?\d{2}|Z)$"


def mask_url(url):
    """The URL with its password replaced by ***: user:pw@ and :pw@ forms, query
    parameters (password, pwd, secret, token, ...) and PWD= inside odbc_connect."""
    if not url:
        return url
    from urllib.parse import quote, quote_plus, unquote_plus
    s = str(url)
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError
    try:                              # the password as SQLAlchemy reads it
        pw = make_url(s).password
    except (ArgumentError, ValueError, TypeError):
        pw = None
    if pw:
        for v in {str(pw), quote(str(pw), safe=""), quote_plus(str(pw))}:
            s = s.replace(f":{v}@", ":***@")
    head, sep, query = s.partition("?")
    # user:pw@host (the password up to the last '@' before the path) and :pw@host
    head = re.sub(r"^([\w+.\-]+://[^:/@]*):(.*)@", r"\1:***@", head)
    if sep:
        def odbc(m):
            inner = re.sub(r"(?i)\b(pwd|password)\s*=\s*(\{[^}]*\}|[^;]*)", r"\1=***", unquote_plus(m.group(2)))
            return m.group(1) + quote_plus(inner, safe="*")
        query = re.sub(r"(?i)(odbc_connect=)([^&]*)", odbc, query)
        # any parameter whose name contains a secret word (password, sslpassword, auth_token, ...)
        query = re.sub(rf"(?i)((?:^|[&;])[\w.\-]*(?:{_PW_KEYS})[\w.\-]*=)[^&;]*", r"\1***", query)
    return head + sep + query


def mask_spec_urls(raw):
    """A copy of a spec (raw or normalised) whose source and target URLs have their
    password masked, for every file written for sharing (exported spec.yaml,
    diagnostics bundles, introspected drafts).  A masked URL gets `url_env`, the
    environment variable that supplies the real URL when the spec is run."""
    out = dict(raw)
    for key, env in (("source", "RELSYNTH_SOURCE_URL"), ("target", "RELSYNTH_TARGET_URL")):
        s = out.get(key)
        if not isinstance(s, dict):
            continue
        s = {k: ("***" if is_secret_key(k) and v not in (None, "") else v) for k, v in s.items()}
        if s.get("url"):
            m = mask_url(s["url"])
            if m != str(s["url"]):
                s.setdefault("url_env", env)
            s["url"] = m
        out[key] = s
    return out


def is_secret_key(name) -> bool:
    """A settings key that holds a secret (password, token, ...), not one that
    names where a secret is (password_env, url_env)."""
    n = str(name).lower()
    return bool(re.search(_PW_KEYS, n)) and not n.endswith(("_env", "_file"))


FILE_KINDS = ("csv", "files", "parquet")
FILE_SUFFIXES = (".csv", ".csv.gz", ".parquet")


def table_files(root) -> dict[str, Path]:
    """{table name: file} for the CSV (.csv, .csv.gz) and Parquet files of a folder."""
    out = {}
    for p in sorted(Path(root).iterdir()):
        for suf in FILE_SUFFIXES:
            if p.is_file() and p.name.lower().endswith(suf):
                out.setdefault(p.name[: -len(suf)], p)
    return out


_LEADING_ZERO = r"^[+-]?0\d"


def _parse_text_column(s: pd.Series) -> pd.Series:
    """A CSV column read as text: numbers become numbers unless a value has a leading
    zero ('0101001' is a code, and 101001 would be another code); everything else
    stays text.  Dates stay text here and are parsed by normalise_frame."""
    v = s.dropna()
    if not len(v):
        return s
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().all() and not v.astype(str).str.strip().str.match(_LEADING_ZERO).any():
        if len(v) < len(s) and pd.api.types.is_integer_dtype(num) and (num.abs() > 2**53).any():
            # integers beyond float precision (18-digit keys) with nulls: nullable
            # integers, not a float64 round trip that changes their last digits
            return pd.to_numeric(s, errors="coerce", dtype_backend="numpy_nullable")
        return pd.to_numeric(s, errors="coerce")
    return s


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One representation for every source format.  Time-zone aware date-times
    (Parquet timestamps with a zone, strings ending in +00:00) are converted to UTC
    and the zone dropped, so dates from different files and databases compare;
    Decimal values (SQL money and decimal, Parquet decimal) become floats; text
    columns of pandas' string dtype become plain objects."""
    import datetime as _dt
    import decimal
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            df[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
            continue
        if pd.api.types.is_string_dtype(s) and not pd.api.types.is_object_dtype(s):
            s = s.astype(object).where(s.notna(), None)
            df[c] = s
        if not pd.api.types.is_object_dtype(s):
            continue
        v = s.dropna()
        if not len(v):
            continue
        head = v.head(200)
        if head.map(lambda x: isinstance(x, decimal.Decimal)).all():
            df[c] = pd.to_numeric(s.map(lambda x: None if x is None else float(x)), errors="coerce")
        elif head.map(lambda x: isinstance(x, (_dt.datetime, pd.Timestamp)) and x.tzinfo is not None).all():
            df[c] = pd.to_datetime(s, utc=True).dt.tz_localize(None)
        elif head.map(lambda x: isinstance(x, str)).all() and \
                head.astype(str).str.match(_ISO_TZ).mean() > 0.95:
            d = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
            if d.notna().sum() >= 0.95 * s.notna().sum():
                d = d.dt.tz_localize(None)
                midnight = (d.dropna() == d.dropna().dt.normalize()).all()
                df[c] = d.dt.strftime("%Y-%m-%d").where(d.notna(), None) if midnight else d
    return df


def read_table_file(path, null_values=None) -> pd.DataFrame:
    """A CSV (codes with leading zeros kept as text) or Parquet file, normalised.
    In a CSV only an empty field is null by default: 'NA' (Namibia), 'None' and
    'null' are values; `null_values` (source.null_values) lists the null tokens."""
    path = Path(path)
    if path.name.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        nv = [""] if null_values is None else [str(x) for x in null_values]
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=nv, low_memory=False)
        for c in df.columns:
            df[c] = _parse_text_column(df[c])
    return normalise_frame(df)


def source_connector(spec, source=None):
    """The connector of the spec's source (or of `source`, a spec section)."""
    from .connectors import from_spec
    src = dict(source or spec.get("source") or {})
    return from_spec(src, spec.get("_base_dir", "."))


def read_source_tables(spec, names, source=None, conn=None) -> dict[str, pd.DataFrame]:
    """The named tables of the source, read as stored (normalised, not coerced)."""
    conn = conn or source_connector(spec, source)
    out = {}
    try:
        for t in names:
            if not conn.has_table(t):
                if conn.kind in ("csv", "parquet"):
                    raise FileNotFoundError(f"{t}: no .csv, .csv.gz or .parquet file in {conn.location()}")
                raise ValueError(f"{t}: no such table in {conn.location()}")
            out[t] = conn.read_table(t)
    finally:
        if conn is not None and source is None:
            conn.close()
    return out


def load_tables(spec, source=None, tables=None, partial=()) -> dict[str, pd.DataFrame]:
    """Every table of the spec from its source, coerced to the declared types and
    prepared (computed columns filled).  `tables` gives frames already read (the
    pushdown loader passes samples of large fact tables)."""
    src = dict(source or spec.get("source") or {})
    kind = src.get("kind", "csv")
    from .spec import SOURCE_KINDS
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {kind}")
    out = dict(tables) if tables is not None else read_source_tables(spec, list(spec["tables"]), src)
    from .prepare import computed_columns, prepare_source
    comp = computed_columns(spec)
    notes = []
    out = {t: coerce(df, spec["tables"][t], optional=set(comp[t]), notes=notes, table=t) for t, df in out.items()}
    load_tables.notes = notes
    for n in notes:
        logging.getLogger("relsynth").warning("[load] %s", n)
    return prepare_source(spec, out, partial=partial)[0]


load_tables.notes = []


_TS_MIN, _TS_MAX = pd.Timestamp.min, pd.Timestamp.max


def to_dates(s: pd.Series, notes=None, where="") -> pd.Series:
    """datetime64[ns] values of a date column.  Text in several formats is parsed
    value by value (month first); values that are no date, and dates outside
    1677-2262 (sentinels such as 9999-12-31), become null and are counted in `notes`."""
    text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    d = pd.to_datetime(s, errors="coerce")
    if text:
        miss = s.notna() & d.isna()
        if miss.any():
            d2 = pd.to_datetime(s[miss], errors="coerce", format="mixed", dayfirst=False)
            if getattr(d2.dt, "tz", None) is not None:
                d2 = d2.dt.tz_convert("UTC").dt.tz_localize(None)
            if d2.notna().any():
                d = d.astype("datetime64[us]")
                d[miss] = d2.astype("datetime64[us]")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    n_oob = 0
    if d.dtype != "datetime64[ns]":
        oob = d.notna() & ((d < _TS_MIN) | (d > _TS_MAX))
        n_oob = int(oob.sum())
        if n_oob:
            d = d.where(~oob)
        d = d.astype("datetime64[ns]")
    if notes is not None:
        if n_oob:
            notes.append(f"{where}: {n_oob} date(s) outside 1677-2262 (e.g. 9999-12-31) read as null")
        bad = int((s.notna() & d.isna()).sum()) - n_oob if text else 0
        if bad > 0:
            notes.append(f"{where}: {bad} value(s) that are not dates read as null")
    return d


def coerce(df: pd.DataFrame, tspec: dict, optional=(), notes=None, table="") -> pd.DataFrame:
    """Keep the spec's columns and cast them to their declared types.  Columns in
    `optional` (lookups, derived and aggregate columns) may be absent: they are
    computed from the rest of the database (relsynth.prepare).  What the casts
    turn into nulls is listed in `notes`."""
    df = df.copy()
    cols = [c for c in tspec["columns"] if c in df.columns]
    missing = set(tspec["columns"]) - set(df.columns) - set(optional)
    if missing:
        raise ValueError(f"columns missing from data: {sorted(missing)}")
    df = df[cols]
    for c, cs in tspec["columns"].items():
        if c not in df.columns:
            continue
        t = cs["type"]
        if t in ("date", "datetime"):
            df[c] = to_dates(df[c], notes, f"{table}.{c}" if table else c)
        elif t in ("integer", "float"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t == "boolean":
            df[c] = df[c].map(_to_bool)
        elif t == "id":
            df[c] = canonical_keys(df[c])
        elif t in ("categorical", "text", "pii") and not is_binary(df[c]):
            s = df[c]
            if pd.api.types.is_float_dtype(s) and s.isna().any() and _exact_integers(s.dropna()):
                # integer codes read as floats because of a null: '1', not '1.0'
                df[c] = s.where(s.isna(), s.map(lambda x: str(int(x)), na_action="ignore"))
            else:
                df[c] = s.where(s.isna(), s.astype(str))
    return df


def _exact_integers(v) -> bool:
    v = np.asarray(v, dtype=float)
    return bool(np.isfinite(v).all() and np.array_equal(v, np.round(v)))


def is_stringlike(s: pd.Series) -> bool:
    """Text column under pandas 2 (object dtype) and pandas 3 (the default 'str' dtype)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def is_binary(s: pd.Series) -> bool:
    """Values are raw bytes (varbinary, image, bytea, BLOB)."""
    v = s.dropna()
    return bool(len(v)) and v.head(50).map(lambda x: isinstance(x, (bytes, bytearray, memoryview))).all()


def to_hex(v):
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "0x" + bytes(v).hex()
    return v


def canonical_keys(s: pd.Series) -> pd.Series:
    """Keys read from CSV come back as floats when the column has nulls; make
    integer-valued keys Int64 so 1.0 and 1 are the same key everywhere.  The test
    is exact (1000000.4 is not a key 1000000), and integer text is read as
    integers, never through floats (18-digit keys keep their last digits)."""
    if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
        if pd.api.types.is_integer_dtype(s):
            return s.astype("Int64")
        v = s.dropna()
        if len(v) == 0 or _exact_integers(v):
            return s.round().astype("Int64")
        return s
    txt = s.dropna().astype(str).str.strip()
    if len(txt) and not txt.str.match(_LEADING_ZERO).any():
        if txt.str.fullmatch(r"[+-]?\d{1,18}").all():
            mask = s.notna().to_numpy()
            vals = np.zeros(len(s), dtype=np.int64)
            vals[mask] = pd.to_numeric(txt).to_numpy(dtype=np.int64)
            return pd.Series(pd.arrays.IntegerArray(vals, ~mask), index=s.index, name=s.name)
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == len(txt) and _exact_integers(num.dropna()):
            return num.round().astype("Int64")
    return s.where(s.isna(), s.astype(str))


def _to_bool(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return None


def target_connector(spec, target=None):
    from .connectors import from_spec
    tgt = dict(target or spec.get("target") or {"kind": "csv", "path": "synthetic"})
    return from_spec(tgt, spec.get("_base_dir", "."), create=True)


def write_tables(spec, tables: dict[str, pd.DataFrame], target=None, source_conn=None):
    """Write synthetic tables to the target (spec section or `target`), parents
    first.  target.if_exists: replace (drop and create the tables), append (insert
    into existing tables), truncate (delete the rows of existing tables, children
    first, then insert: the target's DDL and constraints stay) or fail (stop when a
    table exists).  target.structure: clone (the source's DDL: types, keys,
    constraints; when source and target are databases), infer (from the
    specification and the data) or existing (tables created beforehand); auto
    (default) clones when it can."""
    tgt = dict(target or spec.get("target") or {"kind": "csv", "path": "synthetic"})
    mode = tgt.get("if_exists", "replace")
    if mode not in IF_EXISTS:
        raise ValueError(f"target.if_exists must be one of {', '.join(IF_EXISTS)}, not {mode}")
    structure = tgt.get("structure", "auto")
    conn = target_connector(spec, tgt)
    src = None
    try:
        from .connectors.sql import SqlConnector
        if isinstance(conn, SqlConnector) and structure in ("auto", "clone"):
            try:
                src = source_conn or source_connector(spec)
            except (ValueError, FileNotFoundError, KeyError):
                src = None
            if not isinstance(src, SqlConnector):
                if structure == "clone":
                    import logging
                    logging.getLogger("relsynth").warning(
                        "target.structure clone needs a database source; creating the tables from the specification")
                src = None
            elif not all(src.has_table(t) for t in tables):
                src = None
        struct = "clone" if src is not None else ("existing" if structure == "existing" else "infer")
        out = conn.write_tables(spec, tables, mode=mode, structure=struct, source=src)
    finally:
        if src is not None and source_conn is None:
            src.close()
        conn.close()
    return mask_url(out) if isinstance(out, str) and "://" in out else out


IF_EXISTS = ("replace", "append", "truncate", "fail")

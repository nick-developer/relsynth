# SPDX-License-Identifier: Apache-2.0
"""SQL expressions that differ between database engines, built with SQLAlchemy
(identifiers are quoted by the dialect, never pasted into text).

* `ln(x)`: natural logarithm (LN in PostgreSQL, Oracle, MySQL and SQLite, LOG in SQL Server);
* `to_float(x)`: x as a double (CAST AS FLOAT; x + 0E0 in MySQL, whose integer
  division is DECIMAL);
* `ndtri(p)`: the standard normal quantile Phi^-1(p).  SQLite and DuckDB use scipy's ndtri
  through a registered function (exact); the server databases use Acklam's
  rational approximation (relative error below 1.15e-9 on (0, 1), Acklam 2003),
  written as one CASE expression;
* `row_hash(cols, seed)`: a seeded pseudo-random integer in [0, 2^32) per row,
  computed from the row's key values, for reproducible Bernoulli samples
  (hashtextextended in PostgreSQL, HASHBYTES in SQL Server, ORA_HASH in Oracle,
  SHA2 in MySQL, MariaDB and Databricks, sha256 in DuckDB, a registered BLAKE2 function in SQLite).
"""
from __future__ import annotations

import hashlib
import math

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement

HASH_RANGE = 2**32

# Acklam's coefficients
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
P_LOW = 0.02425


class ln(FunctionElement):           # noqa: N801 - SQL function name
    """Natural logarithm."""
    type = sa.Float()
    inherit_cache = True


@compiles(ln)
def _ln_default(element, compiler, **kw):
    return f"LN({compiler.process(element.clauses, **kw)})"


@compiles(ln, "mssql")
def _ln_mssql(element, compiler, **kw):
    return f"LOG({compiler.process(element.clauses, **kw)})"


class to_float(FunctionElement):     # noqa: N801 - SQL function name
    """x as a double-precision number."""
    type = sa.Float()
    inherit_cache = True


@compiles(to_float)
def _to_float_default(element, compiler, **kw):
    return compiler.process(sa.cast(list(element.clauses)[0], sa.Float), **kw)


@compiles(to_float, "duckdb")
@compiles(to_float, "databricks")
def _to_float_duckdb(element, compiler, **kw):
    return f"CAST({compiler.process(element.clauses, **kw)} AS DOUBLE)"     # FLOAT is 4 bytes in DuckDB and Databricks


@compiles(to_float, "mysql")
@compiles(to_float, "mariadb")
def _to_float_mysql(element, compiler, **kw):
    # MySQL has no CAST AS FLOAT before 8.0.17 and integer division there is
    # DECIMAL with 4 digits: adding a DOUBLE zero makes the expression DOUBLE
    return f"({compiler.process(element.clauses, **kw)} + 0E0)"


def _horner(coefs, x):
    expr = sa.literal(coefs[0], sa.Float)
    for c in coefs[1:]:
        expr = expr * x + sa.literal(c, sa.Float)
    return expr


def acklam(p):
    """Phi^-1(p) for 0 < p < 1 as a SQLAlchemy expression (Acklam 2003)."""
    q_lo = sa.func.sqrt(-2.0 * ln(p))
    q_hi = sa.func.sqrt(-2.0 * ln(1.0 - p))
    q_c = p - 0.5
    r = q_c * q_c
    low = _horner(_C, q_lo) / (_horner(_D, q_lo) * q_lo + 1.0)
    high = -(_horner(_C, q_hi) / (_horner(_D, q_hi) * q_hi + 1.0))
    central = _horner(_A, r) * q_c / (_horner(_B, r) * r + 1.0)
    return sa.case((p < P_LOW, low), (p > 1 - P_LOW, high), else_=central)


def acklam_py(p: float) -> float:
    """The same approximation in Python (used by the tests to bound its error)."""
    if p < P_LOW:
        q = math.sqrt(-2 * math.log(p))
        return _poly(_C, q) / (_poly(_D, q) * q + 1)
    if p > 1 - P_LOW:
        q = math.sqrt(-2 * math.log(1 - p))
        return -_poly(_C, q) / (_poly(_D, q) * q + 1)
    q = p - 0.5
    r = q * q
    return _poly(_A, r) * q / (_poly(_B, r) * r + 1)


def _poly(coefs, x):
    v = coefs[0]
    for c in coefs[1:]:
        v = v * x + c
    return v


def ndtri(p, dialect_name: str):
    """Phi^-1(p): exact in SQLite and DuckDB (registered functions), Acklam elsewhere."""
    if dialect_name in ("sqlite", "duckdb"):
        return sa.func.rs_ndtri(p, type_=sa.Float)
    return acklam(p)


def _text(col):
    return sa.func.coalesce(sa.cast(col, sa.String(4000)), sa.literal(""))


def _joined(cols):
    expr = _text(cols[0])
    for c in cols[1:]:
        expr = expr + sa.literal("|") + _text(c)
    return expr


def row_hash(cols, seed: int, dialect_name: str):
    """A seeded hash of the row's values of cols in [0, 2^32), or None when the
    dialect has no hash function (the caller falls back to random order)."""
    seed = int(seed) % HASH_RANGE
    if dialect_name == "postgresql":
        h = sa.func.hashtextextended(_joined(cols), sa.literal(seed, sa.BigInteger))
        return ((h % HASH_RANGE) + HASH_RANGE) % HASH_RANGE
    if dialect_name == "mssql":
        # constants inline: pyodbc's typed parameter binding makes HASHBYTES return
        # NULL for a bound algorithm name (seed is an int, so the literal is safe)
        txt = sa.func.concat(sa.literal_column(f"'{seed}|'"), _joined(cols))
        digest = sa.func.hashbytes(sa.literal_column("'SHA2_256'"), sa.cast(txt, sa.NVARCHAR(None)))
        return sa.cast(sa.cast(digest, sa.BINARY(4)), sa.BigInteger)
    if dialect_name == "fabric":
        # SQL Server functions over VARCHAR, whose bytes are UTF-8 in a UTF-8
        # collation: the same 32 bits as py_row_hash_sha256 (Fabric has no NVARCHAR)
        txt = sa.func.concat(sa.literal_column(f"'{seed}|'"), _joined(cols))
        digest = sa.func.hashbytes(sa.literal_column("'SHA2_256'"), sa.cast(txt, sa.VARCHAR(None)))
        return sa.cast(sa.cast(digest, sa.BINARY(4)), sa.BigInteger)
    if dialect_name == "oracle":
        return sa.func.ora_hash(_joined(cols), sa.literal(HASH_RANGE - 1), sa.literal(seed))
    if dialect_name == "sqlite":
        return sa.func.rs_hash(sa.literal(seed), _joined(cols), type_=sa.BigInteger)
    if dialect_name == "duckdb":
        # the same 32 bits as MySQL: SHA-256 of 'seed|text', first 8 hex digits
        txt = sa.func.concat(sa.literal(f"{seed}|"), _joined(cols))
        hex32 = sa.func.concat(sa.literal_column("'0x'"), sa.func.left(sa.func.sha256(txt), sa.literal_column("8")))
        return sa.cast(hex32, sa.BigInteger)
    if dialect_name == "databricks":
        # the same 32 bits again: sha2 gives lower-case hex, conv reads base 16
        txt = sa.func.concat(sa.literal(f"{seed}|"), _joined(cols))
        hex8 = sa.func.left(sa.func.sha2(txt, sa.literal_column("256")), sa.literal_column("8"))
        return sa.cast(sa.func.conv(hex8, sa.literal_column("16"), sa.literal_column("10")), sa.BigInteger)
    if dialect_name in ("mysql", "mariadb"):
        # the first 32 bits of SHA-256('seed|text'), as py_row_hash_sha256 computes them
        txt = sa.func.concat(sa.literal(f"{seed}|"), _joined(cols))
        hexd = sa.func.left(sa.func.sha2(txt, sa.literal_column("256")), sa.literal_column("8"))
        return sa.cast(sa.func.conv(hexd, sa.literal_column("16"), sa.literal_column("10")), sa.BigInteger)
    return None


def py_row_hash(seed: int, text: str) -> int:
    """The SQLite hash function: BLAKE2b of 'seed|text', first 4 bytes."""
    return int.from_bytes(hashlib.blake2b(f"{seed}|{text}".encode(), digest_size=4).digest(), "big")


def py_row_hash_sha256(seed: int, text: str) -> int:
    """The MySQL, MariaDB, DuckDB, Databricks and Fabric hash: SHA-256 of 'seed|text' (UTF-8), first 4 bytes."""
    return int(hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()[:8], 16)


def register_sqlite_functions(dbapi_con) -> None:
    """rs_ndtri and rs_hash on a sqlite3 connection (deterministic functions)."""
    from scipy.special import ndtri as _nd

    def _ndtri(p):
        return None if p is None else float(_nd(p))

    def _hash(seed, text):
        return None if text is None else py_row_hash(int(seed), str(text))
    try:
        dbapi_con.create_function("rs_ndtri", 1, _ndtri, deterministic=True)
        dbapi_con.create_function("rs_hash", 2, _hash, deterministic=True)
    except TypeError:            # Python built against an old SQLite without the flag
        dbapi_con.create_function("rs_ndtri", 1, _ndtri)
        dbapi_con.create_function("rs_hash", 2, _hash)

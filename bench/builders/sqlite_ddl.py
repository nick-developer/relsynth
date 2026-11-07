# SPDX-License-Identifier: Apache-2.0
"""CREATE TABLE statements for the SQLite builders, with quoted identifiers.

Table and column names come from the downloaded files, so they are quoted with
SQLite's own identifier quoting (embedded double quotes are doubled) and column
types must be one of a fixed list of type names (the declared type, such as DATE,
is what relsynth's introspection reads).  No value is ever written into SQL
text: rows are inserted with bound parameters (``insert_rows``).
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence

from sqlalchemy.dialects import sqlite

_PREPARER = sqlite.dialect().identifier_preparer
TYPES = frozenset({"TEXT", "INTEGER", "REAL", "FLOAT", "NUMERIC", "DATE", "DATETIME", "TIME", "BLOB"})
_VARCHAR = re.compile(r"VARCHAR\(\d{1,5}\)")


def quote(name: object) -> str:
    """name as a quoted SQLite identifier (always quoted, inner quotes doubled)."""
    return _PREPARER.quote_identifier(str(name))


def create_table(con: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str, bool]],
                 primary_key: Sequence[str] = (),
                 foreign_keys: Iterable[tuple[Sequence[str], str, Sequence[str]]] = (),
                 unique: Iterable[Sequence[str]] = ()) -> None:
    """Create `table` with `columns` as (name, type, not null), a primary key,
    unique keys and foreign keys as (columns, referenced table, referenced columns)."""
    parts = []
    for name, typ, not_null in columns:
        if typ not in TYPES and not _VARCHAR.fullmatch(typ):
            raise ValueError(f"{table}.{name}: column type {typ!r} is not one of {sorted(TYPES)}")
        parts.append(f"{quote(name)} {typ}{' NOT NULL' if not_null else ''}")
    if primary_key:
        parts.append(f"PRIMARY KEY ({', '.join(map(quote, primary_key))})")
    for cols in unique:
        parts.append(f"UNIQUE ({', '.join(map(quote, cols))})")
    for cols, ref_table, ref_cols in foreign_keys:
        parts.append(f"FOREIGN KEY ({', '.join(map(quote, cols))}) "
                     f"REFERENCES {quote(ref_table)} ({', '.join(map(quote, ref_cols))})")
    statement = f"CREATE TABLE {quote(table)} ({', '.join(parts)})"
    con.execute(statement)


def insert_rows(con: sqlite3.Connection, table: str, columns: Sequence[str], rows: Iterable[Sequence]) -> None:
    """Insert `rows` into `table`'s `columns` with bound values and quoted names."""
    cols = ", ".join(map(quote, columns))
    marks = ", ".join("?" * len(columns))
    con.executemany(f"INSERT INTO {quote(table)} ({cols}) VALUES ({marks})", rows)  # nosemgrep  # names quoted, values bound

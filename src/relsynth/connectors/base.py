"""The connector interface: one object per data location (a database schema or a
folder of files) that the rest of relsynth reads and writes through.

A connector offers

* the catalog: tables, columns with their declared types, primary keys,
  foreign keys and unique constraints (`describe`);
* reads: whole tables (`read_table`, streamed in chunks for large tables) and
  seeded uniform samples (`sample`);
* writes: bulk loads in the vendor's fastest safe path, parents before children,
  with the target table structure cloned from the source or created from the
  specification (`write_tables`);
* exact statistics computed where the data lives (`stats`), which the pushdown
  fit uses instead of loading a large fact table (see relsynth.pushdown).

Identifiers are never pasted into SQL text: tables are reflected by name and
quoted by the dialect, and values are bound parameters.
"""
from __future__ import annotations

import abc
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator

import pandas as pd

log = logging.getLogger("relsynth.connectors")

WRITE_MODES = ("replace", "append", "truncate", "fail")


@dataclass
class ColumnInfo:
    """One column of a table as the catalog declares it."""
    name: str
    sql_type: str = ""
    nullable: bool = True
    identity: bool = False
    length: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForeignKey:
    """A declared foreign key: `columns` of this table reference `ref_columns` of `ref_table`."""
    columns: list[str]
    ref_table: str
    ref_columns: list[str]
    name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableInfo:
    """Catalog entry of one table."""
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    unique: list[list[str]] = field(default_factory=list)
    row_count: int | None = None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> dict:
        return {"name": self.name, "columns": [c.to_dict() for c in self.columns],
                "primary_key": list(self.primary_key), "foreign_keys": [f.to_dict() for f in self.foreign_keys],
                "unique": [list(u) for u in self.unique], "row_count": self.row_count}


def match_table(name: str, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> bool:
    """Whether a table name passes the include / exclude patterns (regular expressions,
    case-insensitive, matched against the whole name)."""
    if include and not any(re.fullmatch(p, name, flags=re.I) for p in include):
        return False
    if exclude and any(re.fullmatch(p, name, flags=re.I) for p in exclude):
        return False
    return True


class Connector(abc.ABC):
    """Base class of every data location.  Subclasses implement the abstract
    methods; the defaults below build on them."""

    #: short name of the backend: csv, parquet, sqlite, postgresql, mssql, oracle, sql
    kind: str = "base"
    #: True when `stats` computes in the data store (no full table is loaded)
    pushdown: bool = False

    # ------------------------------------------------------------ catalog
    @abc.abstractmethod
    def tables(self, include=None, exclude=None) -> list[str]:
        """Names of the tables, optionally filtered by include / exclude patterns."""

    @abc.abstractmethod
    def describe(self, table: str) -> TableInfo:
        """Catalog entry of one table."""

    def catalog(self, include=None, exclude=None) -> dict[str, TableInfo]:
        return {t: self.describe(t) for t in self.tables(include, exclude)}

    @abc.abstractmethod
    def count(self, table: str) -> int:
        """Number of rows."""

    def has_table(self, table: str) -> bool:
        return table in self.tables()

    # ------------------------------------------------------------ reads
    @abc.abstractmethod
    def read_table(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """The whole table (or the given columns), normalised (relsynth.data.normalise_frame)."""

    def iter_table(self, table: str, columns: list[str] | None = None, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
        """The table in chunks of at most `chunksize` rows."""
        df = self.read_table(table, columns)
        for i in range(0, max(len(df), 1), chunksize):
            yield df.iloc[i:i + chunksize].reset_index(drop=True)

    @abc.abstractmethod
    def sample(self, table: str, n: int, seed: int = 7, key: list[str] | None = None) -> tuple[pd.DataFrame, bool]:
        """A uniform random sample of about n rows: (rows, is_whole_table).  The same
        data and seed give the same sample."""

    def read_tables(self, names: Iterable[str]) -> dict[str, pd.DataFrame]:
        return {t: self.read_table(t) for t in names}

    # ------------------------------------------------------------ statistics
    def stats(self, table: str):
        """Exact statistics of one table computed where the data lives (see
        relsynth.pushdown.stats).  The default loads the needed columns."""
        from ..pushdown.stats import FrameStats
        return FrameStats.lazy(lambda cols: self.read_table(table, cols), self.count(table), table=table)

    # ------------------------------------------------------------ writes
    @abc.abstractmethod
    def write_table(self, table: str, df: pd.DataFrame, mode: str = "replace", tspec: dict | None = None) -> int:
        """Write one table; returns the number of rows written."""

    def write_tables(self, spec: dict, tables: dict[str, pd.DataFrame], mode: str = "replace",
                     structure: str = "infer", source: Connector | None = None) -> str:
        """Write the synthetic tables of a spec, parents first."""
        from ..spec import topo_order
        edges = [(r["child"], r["parent"]) for r in spec.get("relationships", [])
                 if r.get("kind") == "many_to_one" and r["child"] in tables and r["parent"] in tables]
        for t in topo_order(list(tables), edges):
            self.write_table(t, tables[t], mode=mode, tspec=spec["tables"].get(t))
        return self.location()

    # ------------------------------------------------------------ misc
    def location(self) -> str:
        """Where the data is (credentials masked)."""
        return self.kind

    def ping(self) -> dict[str, Any]:
        """Check the connection: {ok, kind, location, tables, server}."""
        try:
            names = self.tables()
            return {"ok": True, "kind": self.kind, "location": self.location(), "tables": len(names)}
        except Exception as e:  # noqa: BLE001 - reported to the caller
            return {"ok": False, "kind": self.kind, "location": self.location(), "error": f"{type(e).__name__}: {e}"}

    def close(self) -> None:  # noqa: B027 - optional hook: nothing to release by default
        """Release connections."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.location()}>"

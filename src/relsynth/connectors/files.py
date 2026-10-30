# SPDX-License-Identifier: Apache-2.0
"""Folders of CSV and Parquet files: one file per table, named <table>.csv,
<table>.csv.gz or <table>.parquet.

Files declare no keys: the specification (drafted by `introspect` from the
values) carries them.  Reads are column-pruned (the statistics of a large fact
table read only its key and measure columns; Parquet reads only those column
chunks) and samples are drawn while streaming the file, so a sample never needs
the whole table in memory.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import FILE_SUFFIXES, inside, normalise_frame, read_table_file, table_files, to_hex
from .base import WRITE_MODES, ColumnInfo, Connector, TableInfo, match_table

log = logging.getLogger("relsynth.connectors")

CSV_CHUNK = 200_000
#: statistics and samples of pushed-down facts: duckdb (in SQL over the files), pandas
#: (in memory, column by column, as in 1.0), auto (duckdb when installed)
ENGINES = ("auto", "duckdb", "pandas")


def binary_columns(df: pd.DataFrame, tspec: dict | None = None) -> list:
    """Columns typed binary in the table's spec, or whose values are all bytes."""
    bincols = [c for c, cs in (tspec or {}).get("columns", {}).items()
               if cs.get("type") == "binary" and c in df.columns]
    for c in df.columns:     # bytes values mark a binary column even without a spec
        if c not in bincols and pd.api.types.is_object_dtype(df[c]):
            v = df[c].dropna()
            if len(v) and v.map(lambda x: isinstance(x, (bytes, bytearray, memoryview))).all():
                bincols.append(c)
    return bincols


def parquet_frame(df: pd.DataFrame, tspec: dict | None = None) -> pd.DataFrame:
    """The frame as Parquet stores it: an object column that is neither text nor
    binary (mixed values) is written as text.  Shared by the folder and the
    lakehouse targets, so the same synthetic table reads back the same."""
    bincols = binary_columns(df, tspec)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) and c not in bincols:
            v = out[c].dropna()
            if len(v) and not v.map(lambda x: isinstance(x, str)).all():
                out[c] = out[c].map(lambda x: None if x is None or (isinstance(x, float) and x != x) else str(x))
    return out


class FileConnector(Connector):
    """A folder of table files.  fmt (csv or parquet) is the format written; any
    of the supported formats is read."""

    def __init__(self, path, fmt: str = "csv", null_values=None, create: bool = False, engine: str = "auto",
                 duckdb: dict | None = None):
        self.root = Path(path)
        if engine not in ENGINES:
            raise ValueError(f"file engine must be one of {', '.join(ENGINES)}")
        self.engine_name = engine
        self.duckdb_settings = dict(duckdb or {})
        self._duck = None
        if fmt not in ("csv", "parquet"):
            raise ValueError("file format must be csv or parquet")
        self.fmt = fmt
        self.kind = fmt
        self.null_values = null_values
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        self._counts: dict[str, int] = {}

    def location(self):
        return str(self.root)

    @property
    def duck(self):
        """The DuckDB engine over this folder (None with engine pandas, or auto
        without DuckDB installed)."""
        if self._duck is None and self.engine_name != "pandas":
            from .duckfiles import DuckDBFiles, available
            if available():
                self._duck = DuckDBFiles(self, self.duckdb_settings)
            elif self.engine_name == "duckdb":
                raise ImportError("engine: duckdb needs the duckdb and duckdb-engine packages")
        return self._duck

    @property
    def pushdown(self) -> bool:  # type: ignore[override]  # a class attribute on the other connectors
        return self.duck is not None

    def stats(self, table):
        if self.duck is not None:
            return self.duck.stats(table)
        return super().stats(table)

    def close(self):
        if self._duck is not None:
            self._duck.close()
            self._duck = None

    # ------------------------------------------------------------ catalog
    def files(self) -> dict[str, Path]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"no folder {self.root}")
        return table_files(self.root)

    def file(self, table) -> Path:
        f = self.files().get(table)
        if f is None:
            raise FileNotFoundError(f"{table}: no .csv, .csv.gz or .parquet file in {self.root}")
        return f

    @staticmethod
    def is_parquet(p: Path) -> bool:
        return p.name.lower().endswith(".parquet")

    def tables(self, include=None, exclude=None):
        return [t for t in self.files() if match_table(t, include, exclude)]

    def has_table(self, table):
        return self.root.is_dir() and table in self.files()

    def describe(self, table):
        f = self.file(table)
        if self.is_parquet(f):
            import pyarrow.parquet as pq
            sch = pq.read_schema(f)
            cols = [ColumnInfo(name=n, sql_type=str(sch.field(n).type)) for n in sch.names]
        else:
            head = pd.read_csv(f, nrows=0)
            cols = [ColumnInfo(name=str(n)) for n in head.columns]
        return TableInfo(name=table, columns=cols)

    def count(self, table):
        if table not in self._counts:
            f = self.file(table)
            if self.is_parquet(f):
                import pyarrow.parquet as pq
                self._counts[table] = int(pq.ParquetFile(f).metadata.num_rows)
            else:
                n = 0
                for ch in pd.read_csv(f, dtype=str, usecols=[0], chunksize=CSV_CHUNK, keep_default_na=False):
                    n += len(ch)
                self._counts[table] = n
        return self._counts[table]

    # ------------------------------------------------------------ reads
    def read_table(self, table, columns=None):
        f = self.file(table)
        if columns is None:
            return read_table_file(f, self.null_values)
        if self.is_parquet(f):
            return normalise_frame(pd.read_parquet(f, columns=list(columns)))
        nv = [""] if self.null_values is None else [str(x) for x in self.null_values]
        from ..data import _parse_text_column
        df = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=nv, low_memory=False, usecols=list(columns))
        for c in df.columns:
            df[c] = _parse_text_column(df[c])
        return normalise_frame(df[list(columns)])

    def iter_table(self, table, columns=None, chunksize=CSV_CHUNK):
        f = self.file(table)
        if self.is_parquet(f):
            import pyarrow.parquet as pq
            for b in pq.ParquetFile(f).iter_batches(batch_size=chunksize, columns=columns):
                yield normalise_frame(b.to_pandas())
            return
        nv = [""] if self.null_values is None else [str(x) for x in self.null_values]
        yield from pd.read_csv(f, dtype=str, keep_default_na=False, na_values=nv, chunksize=chunksize,
                               usecols=columns)

    def sample(self, table, n, seed=7, key=None):
        """A Bernoulli(n / rows) sample: with DuckDB, rows whose seeded SHA-256 hash
        falls below n / rows (as on the database servers); with pandas, drawn while
        streaming the file (seeded)."""
        if self.duck is not None:
            return self.duck.sample(table, n, seed=seed, key=key)
        total = self.count(table)
        if total <= n:
            return self.read_table(table), True
        rng = np.random.default_rng([int(seed), sum(map(ord, table))])
        q = n / total
        parts = [ch[rng.random(len(ch)) < q] for ch in self.iter_table(table)]
        df = pd.concat(parts, ignore_index=True)
        if not self.is_parquet(self.file(table)):
            from ..data import _parse_text_column
            for c in df.columns:
                df[c] = _parse_text_column(df[c])
        return normalise_frame(df), False

    # ------------------------------------------------------------ writes
    def write_table(self, table, df, mode="replace", tspec=None):
        if mode not in WRITE_MODES:
            raise ValueError(f"mode must be one of {', '.join(WRITE_MODES)}")
        self.root.mkdir(parents=True, exist_ok=True)
        path = inside(self.root, table, ".parquet" if self.fmt == "parquet" else ".csv")
        if path.exists() and mode == "fail":
            raise FileExistsError(f"{path} exists (if_exists: fail)")
        df = df[[c for c in df.columns if not str(c).startswith("__")]]
        bincols = binary_columns(df, tspec)
        if self.fmt == "parquet":
            out = parquet_frame(df, tspec)
            if mode == "append" and path.exists():
                out = pd.concat([pd.read_parquet(path), out], ignore_index=True)
            out.to_parquet(path, index=False)
        else:
            if bincols:          # bytes are written as 0x-prefixed hex, as SQL tools export them
                df = df.copy()
                for c in bincols:
                    df[c] = df[c].map(to_hex)
            app = mode == "append" and path.exists()
            df.to_csv(path, index=False, mode="a" if app else "w", header=not app)
        self._counts.pop(table, None)
        if self._duck is not None:
            self._duck.refresh()
        return len(df)

    def write_tables(self, spec, tables, mode="replace", structure="infer", source=None):
        if mode == "fail":
            clash = [t for t in tables if inside(self.root, t, ".parquet" if self.fmt == "parquet" else ".csv").exists()]
            if clash:
                raise FileExistsError(f"files exist for {', '.join(clash)} (if_exists: fail)")
        if mode in ("replace", "truncate") and self.root.is_dir():
            # an old file of the other format would be read instead of the new one
            for t in tables:
                for suf in FILE_SUFFIXES:
                    p = inside(self.root, t, suf)
                    if p.exists() and not p.name.endswith(".parquet" if self.fmt == "parquet" else ".csv"):
                        p.unlink()
        for t, df in tables.items():
            self.write_table(t, df, mode="append" if mode == "append" else "replace", tspec=spec["tables"].get(t))
        return str(self.root)

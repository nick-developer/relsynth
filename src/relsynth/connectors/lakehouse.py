# SPDX-License-Identifier: Apache-2.0
"""Open table formats: Apache Iceberg and Delta Lake, on a local disk or on
object storage (Amazon S3 and S3-compatible stores, Azure Data Lake Storage
Gen2 / Blob Storage).

A lakehouse table is a set of Parquet data files plus a transaction log that
says which files form each version (Iceberg: snapshots and manifests, Apache
Iceberg table spec v2; Delta: the `_delta_log` commits, Armbrust et al., PVLDB
2020).  relsynth reads one version of every table and writes each synthetic
table as one new version (one atomic commit per table).

    src = connect({"kind": "iceberg", "path": "./lake", "schema": "sales"})
    src = connect({"kind": "iceberg", "schema": "sales",
                   "catalog": {"type": "rest", "uri": "https://catalog.example.com"},
                   "storage": {"s3.endpoint": "https://s3.eu-west-1.amazonaws.com"}})
    src = connect({"kind": "delta", "path": "s3://bucket/lake/sales",
                   "storage": {"AWS_REGION": "eu-west-1"}})
    src = connect({"kind": "delta", "path": "abfss://lake@acct.dfs.core.windows.net/sales"})

Layout.  Delta: each table is the folder <path>/<table>.  Iceberg: tables live in
a catalog namespace (`schema`, default `default`); `catalog` holds the
properties of any PyIceberg catalog (REST, Glue, Hive, SQL, ...); with only a
local `path`, relsynth keeps a SQL catalog in <path>/catalog.db whose
warehouse is that folder.

Versions.  A source reads, per table, the version current when the connector
first touches it, or a pinned one: `snapshot` (Iceberg snapshot id or Delta
version, one value or a mapping table -> value) or `as_of` (an ISO time: the
version current at that time, the same instant for every table).  `versions()`
reports what was read, so a run can be repeated on the same data.

Statistics of pushed-down facts run in DuckDB over the table's Arrow dataset
(the files of the version read, with partition columns), the same SQL as on
the database servers (relsynth.connectors.duckfiles).  Versions with Iceberg
delete files or Delta deletion vectors, and Iceberg tables whose schema has
evolved since files were written, are read through the format library
instead (merge-on-read), held in memory.

Credentials never go into a specification: the format libraries read the
platform's standard variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_PROFILE, AZURE_STORAGE_ACCOUNT_KEY, AZURE_CLIENT_ID, ...), and
`storage_env` maps a storage option to the environment variable holding it.

Keys.  Neither format enforces keys.  relsynth records the primary key,
unique keys and foreign keys of the tables it writes (Iceberg: table
properties relsynth.*; Delta: the table description), so a synthetic lake
introspects with its keys; other tables describe no keys and `introspect`
finds them from the values.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..data import normalise_frame
from .base import WRITE_MODES, ColumnInfo, Connector, ForeignKey, TableInfo, match_table
from .files import ENGINES, parquet_frame

log = logging.getLogger("relsynth.connectors")

LAKE_KINDS = ("iceberg", "delta")
_META_PREFIX = "relsynth:"            # Delta: the description that carries the keys
_TRANSFORMS = ("identity", "year", "month", "day", "hour", "bucket", "truncate")


def available(kind: str) -> bool:
    import importlib.util
    mod = {"iceberg": "pyiceberg", "delta": "deltalake"}[kind]
    return importlib.util.find_spec(mod) is not None and importlib.util.find_spec("pyarrow") is not None


def _require(kind: str) -> None:
    if not available(kind):
        extra = {"iceberg": "iceberg", "delta": "delta"}[kind]
        raise ImportError(f"kind {kind} needs relsynth[{extra}] (pip install 'relsynth[{extra}]')")


def _is_uri(path: str) -> bool:
    return "://" in str(path)


def storage_options(storage: dict | None, storage_env: dict | None) -> dict[str, str]:
    """Storage options from the spec (no secrets) and from environment variables
    named in storage_env (option -> variable)."""
    out = {str(k): str(v) for k, v in (storage or {}).items()}
    for k, var in (storage_env or {}).items():
        val = os.environ.get(str(var))
        if val is None:
            raise ValueError(f"storage_env: environment variable {var} (for {k}) is not set")
        out[str(k)] = val
    return out


def parse_time(value) -> datetime:
    """An ISO date or time as an aware UTC datetime (a naive value is UTC)."""
    if isinstance(value, datetime):
        t = value
    else:
        t = pd.Timestamp(str(value)).to_pydatetime()
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def parse_partition(expr: str) -> tuple[str, str, int | None]:
    """'Region', 'month(OrderDate)', 'bucket(16, CustomerKey)' or
    'truncate(4, Code)' -> (transform, column, width)."""
    s = str(expr).strip()
    if "(" not in s:
        return "identity", s, None
    fn, rest = s.split("(", 1)
    fn = fn.strip().lower()
    if fn not in _TRANSFORMS or not rest.endswith(")"):
        raise ValueError(f"partition_by: {expr!r} is not one of col, {', '.join(f'{t}(col)' for t in _TRANSFORMS[1:4])}, "
                         "hour(col), bucket(n, col), truncate(n, col)")
    args = [a.strip() for a in rest[:-1].split(",")]
    if fn in ("bucket", "truncate"):
        if len(args) != 2 or not args[0].isdigit():
            raise ValueError(f"partition_by: {expr!r} needs ({fn}(n, column))")
        return fn, args[1], int(args[0])
    if len(args) != 1:
        raise ValueError(f"partition_by: {expr!r} takes one column")
    return fn, args[0], None


def arrow_table(df: pd.DataFrame, tspec: dict | None = None):
    """The synthetic frame as an Arrow table both formats store: hidden columns
    dropped, mixed object columns as text (as the Parquet folder target), integer
    columns with nulls as nullable int64 and time stamps in microseconds (the
    finest unit Iceberg v2 and Delta store)."""
    import pyarrow as pa
    df = df[[c for c in df.columns if not str(c).startswith("__")]]
    out = parquet_frame(df, tspec)
    tbl = pa.Table.from_pandas(out, preserve_index=False)
    fields = []
    for f in tbl.schema:
        t = f.type
        if pa.types.is_timestamp(t):
            t = pa.timestamp("us", tz=t.tz)
        elif pa.types.is_null(t):
            t = pa.string()                                    # a column with no value at all
        elif pa.types.is_large_string(t):
            t = pa.string()
        fields.append(pa.field(f.name, t, nullable=True))
    return tbl.cast(pa.schema(fields), safe=False).replace_schema_metadata(None)


class LakeConnector(Connector):
    """Common part of the Iceberg and Delta connectors: reads through an Arrow
    dataset per table, statistics in DuckDB over it, writes from Arrow."""

    kind = "lake"

    def __init__(self, *, schema: str | None = None, storage: dict | None = None, storage_env: dict | None = None,
                 snapshot=None, as_of=None, partition_by: dict | None = None, engine: str = "auto",
                 duckdb: dict | None = None, create: bool = False):
        _require(self.kind)
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {', '.join(ENGINES)}")
        if snapshot is not None and as_of is not None:
            raise ValueError("give either snapshot or as_of, not both")
        self.schema = schema
        self.storage = storage_options(storage, storage_env)
        self.snapshot = snapshot
        self.as_of = parse_time(as_of) if as_of is not None else None
        self.partition_by = {str(t): [str(e) for e in (v if isinstance(v, list) else [v])]
                             for t, v in (partition_by or {}).items()}
        for exprs in self.partition_by.values():
            for e in exprs:
                parse_partition(e)
        self.engine_name = engine
        self.duckdb_settings = dict(duckdb or {})
        self.null_values = None
        self._duck = None
        self._data: dict[str, Any] = {}           # table -> Arrow dataset or table of the version read
        self._read_version: dict[str, Any] = {}   # table -> snapshot id / version read
        self._counts: dict[str, int] = {}
        self.committed: dict[str, Any] = {}       # table -> snapshot id / version written
        self.create = create

    # ------------------------------------------------------------ format specifics
    def _names(self) -> list[str]:
        raise NotImplementedError

    def _open(self, table: str):
        """(Arrow dataset or table of the version to read, version id)."""
        raise NotImplementedError

    def _keys(self, table: str) -> dict:
        """{primary_key, unique, foreign_keys} recorded by relsynth (empty without)."""
        raise NotImplementedError

    def _write(self, table: str, data, mode: str, meta: dict, props: dict):
        """Commit `data` (an Arrow table) as a new version; returns its id."""
        raise NotImplementedError

    def history(self, table: str) -> list[dict]:
        """The table's versions, newest first: [{version, time, operation}]."""
        raise NotImplementedError

    # ------------------------------------------------------------ versions
    def _pinned(self, table: str):
        s = self.snapshot
        if isinstance(s, dict):
            return s.get(table)
        return s

    def versions(self) -> dict[str, Any]:
        """The version of each table read so far (pin them with `snapshot` to read
        the same data again)."""
        return dict(self._read_version)

    def dataset(self, table: str):
        if table not in self._data:
            if not self.has_table(table):
                raise ValueError(f"{table}: no such table in {self.location()}")
            self._data[table], self._read_version[table] = self._open(table)
        return self._data[table]

    # ------------------------------------------------------------ DuckDB
    @property
    def duck(self):
        if self._duck is None and self.engine_name != "pandas":
            from .duckfiles import available as duck_available
            if duck_available():
                self._duck = DuckDBLake(self, self.duckdb_settings)
            elif self.engine_name == "duckdb":
                raise ImportError("engine: duckdb needs the duckdb and duckdb-engine packages")
        return self._duck

    @property
    def pushdown(self) -> bool:  # type: ignore[override]
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
    def tables(self, include=None, exclude=None):
        return [t for t in self._names() if match_table(t, include, exclude)]

    def has_table(self, table):
        return table in self._names()

    def describe(self, table):
        ds = self.dataset(table)
        keys = self._keys(table)
        cols = [ColumnInfo(name=f.name, sql_type=str(f.type), nullable=f.nullable) for f in ds.schema]
        fks = [ForeignKey(columns=list(f["columns"]), ref_table=f["ref_table"], ref_columns=list(f["ref_columns"]))
               for f in keys.get("foreign_keys", [])]
        return TableInfo(name=table, columns=cols, primary_key=list(keys.get("primary_key") or []),
                         foreign_keys=fks, unique=[list(u) for u in keys.get("unique", [])])

    def count(self, table):
        if table not in self._counts:
            ds = self.dataset(table)
            self._counts[table] = int(ds.count_rows() if hasattr(ds, "count_rows") else ds.num_rows)
        return self._counts[table]

    # ------------------------------------------------------------ reads
    def _arrow(self, table, columns=None):
        ds = self.dataset(table)
        if hasattr(ds, "to_table"):
            return ds.to_table(columns=list(columns) if columns is not None else None)
        return ds.select(list(columns)) if columns is not None else ds

    def read_table(self, table, columns=None):
        df = self._arrow(table, columns).to_pandas()
        return normalise_frame(df[list(columns)] if columns is not None else df)

    def iter_table(self, table, columns=None, chunksize=200_000):
        ds = self.dataset(table)
        batches = ds.to_batches(columns=list(columns) if columns is not None else None, batch_size=chunksize) \
            if hasattr(ds, "to_batches") else ds.to_batches(max_chunksize=chunksize)
        for b in batches:
            yield normalise_frame(b.to_pandas())

    def sample(self, table, n, seed=7, key=None):
        """A Bernoulli(n / rows) sample: with DuckDB the rows whose seeded SHA-256
        hash falls below n / rows, as on the database servers; with pandas drawn
        while streaming the version's batches (seeded)."""
        if self.duck is not None:
            return self.duck.sample(table, n, seed=seed, key=key)
        import numpy as np
        total = self.count(table)
        if total <= n:
            return self.read_table(table), True
        rng = np.random.default_rng([int(seed), sum(map(ord, table))])
        q = n / total
        parts = [ch[rng.random(len(ch)) < q] for ch in self.iter_table(table)]
        return normalise_frame(pd.concat(parts, ignore_index=True)), False

    # ------------------------------------------------------------ writes
    def _props(self, spec: dict | None) -> dict[str, str]:
        from .._version import __version__
        p = {"relsynth.version": __version__}
        seed = ((spec or {}).get("synthesis") or {}).get("seed")
        if seed is not None:
            p["relsynth.seed"] = str(seed)
        return p

    @staticmethod
    def _table_keys(spec: dict, table: str) -> dict:
        ts = (spec.get("tables") or {}).get(table) or {}
        pk = ts.get("primary_key")
        pk = [pk] if isinstance(pk, str) else list(pk or [])
        uniq = []
        for u in ts.get("unique") or []:
            cols = u.get("columns") if isinstance(u, dict) else u
            if isinstance(u, dict) and u.get("where"):
                continue                                  # filtered keys are not table keys
            uniq.append([cols] if isinstance(cols, str) else list(cols or []))
        fks = []
        for r in spec.get("relationships") or []:
            if r.get("child") != table or r.get("kind", "many_to_one") not in ("many_to_one", "self_reference"):
                continue
            ck = r.get("child_key")
            pk_ = r.get("parent_key") or (spec.get("tables") or {}).get(r.get("parent"), {}).get("primary_key")
            ck = [ck] if isinstance(ck, str) else list(ck or [])
            pk_ = [pk_] if isinstance(pk_, str) else list(pk_ or [])
            if ck and len(ck) == len(pk_):
                fks.append({"columns": ck, "ref_table": r["parent"], "ref_columns": pk_})
        return {"primary_key": pk, "unique": uniq, "foreign_keys": fks}

    def write_table(self, table, df, mode="replace", tspec=None):
        spec = {"tables": {table: tspec or {}}, "relationships": []}
        self.write_tables(spec, {table: df}, mode=mode)
        return len(df)

    def write_tables(self, spec, tables, mode="replace", structure="infer", source=None):
        """One atomic commit per table, parents first.  replace and truncate write a
        new version holding only the new rows (earlier versions stay readable until
        the table's snapshots are expired); append adds a version with the rows
        added; fail stops before writing anything when a table exists."""
        if mode not in WRITE_MODES:
            raise ValueError(f"mode must be one of {', '.join(WRITE_MODES)}")
        if mode == "fail":
            clash = [t for t in tables if self.has_table(t)]
            if clash:
                raise FileExistsError(f"tables exist in {self.location()}: {', '.join(clash)} (if_exists: fail)")
        from ..spec import topo_order
        edges = [(r["child"], r["parent"]) for r in spec.get("relationships", [])
                 if r.get("kind") == "many_to_one" and r["child"] in tables and r["parent"] in tables]
        props = self._props(spec)
        for t in topo_order(list(tables), edges):
            data = arrow_table(tables[t], (spec.get("tables") or {}).get(t))
            self.committed[t] = self._write(t, data, mode, self._table_keys(spec, t), props)
            self._data.pop(t, None)
            self._counts.pop(t, None)
            log.info("%s: %s: %s rows committed as version %s", self.kind, t, f"{data.num_rows:,}", self.committed[t])
        if self._duck is not None:
            self._duck.refresh()
        return self.location()


# ---------------------------------------------------------------- Iceberg
class IcebergConnector(LakeConnector):
    """Apache Iceberg tables of one catalog namespace (PyIceberg)."""

    kind = "iceberg"

    def __init__(self, path: str | Path | None = None, *, catalog: dict | None = None, schema: str | None = None,
                 base_dir: str | Path | None = None, **kw):
        super().__init__(schema=schema or "default", **kw)
        self.path = str(path) if path is not None else None
        if self.path and not _is_uri(self.path):
            p = Path(self.path)
            if not p.is_absolute() and base_dir is not None:
                p = Path(base_dir) / p
            self.path = str(p.resolve())
        props = {str(k): v for k, v in (catalog or {}).items()}
        self.catalog_name = str(props.pop("name", "relsynth"))
        props.update(self.storage)
        if not catalog:
            if not self.path or _is_uri(self.path):
                raise ValueError("iceberg: give `catalog` (the properties of a PyIceberg catalog), or a local `path` "
                                 "for a SQL catalog kept in <path>/catalog.db")
            Path(self.path).mkdir(parents=True, exist_ok=True)
            props.update({"type": "sql", "uri": f"sqlite:///{Path(self.path) / 'catalog.db'}",
                          "warehouse": Path(self.path).as_uri()})
        elif self.path and "warehouse" not in props:
            props["warehouse"] = Path(self.path).as_uri() if not _is_uri(self.path) else self.path
        self._props_catalog = props
        self._cat = None

    @property
    def catalog(self):
        if self._cat is None:
            from pyiceberg.catalog import load_catalog
            self._cat = load_catalog(self.catalog_name, **self._props_catalog)
        return self._cat

    def location(self):
        where = self.path or self._props_catalog.get("uri") or self._props_catalog.get("type", "catalog")
        return f"iceberg {where} namespace {self.schema}"

    def _ident(self, table):
        return (self.schema, table)

    def _names(self):
        try:
            return [i[-1] for i in self.catalog.list_tables(self.schema)]
        except Exception as e:  # noqa: BLE001 - a missing namespace has no tables
            if type(e).__name__ == "NoSuchNamespaceError":
                return []
            raise

    def _table(self, table):
        return self.catalog.load_table(self._ident(table))

    def _snapshot_id(self, tbl, table):
        pin = self._pinned(table)
        if pin is not None:
            if tbl.snapshot_by_id(int(pin)) is None:
                raise ValueError(f"{table}: no snapshot {pin}")
            return int(pin)
        if self.as_of is not None:
            snap = tbl.snapshot_as_of_timestamp(int(self.as_of.timestamp() * 1000))
            if snap is None:
                raise ValueError(f"{table}: no snapshot at or before {self.as_of.isoformat()}")
            return snap.snapshot_id
        snap = tbl.current_snapshot()
        return snap.snapshot_id if snap is not None else None

    def _open(self, table):
        import pyarrow as pa
        import pyarrow.dataset as pds
        from pyiceberg.io.pyarrow import PyArrowFileIO, schema_to_pyarrow
        tbl = self._table(table)
        sid = self._snapshot_id(tbl, table)
        if sid is None:                                  # created, never written
            return pa.table({f.name: pa.array([], type=schema_to_pyarrow(f.field_type)) for f in tbl.schema().fields}), None
        scan = tbl.scan(snapshot_id=sid)
        tasks = list(scan.plan_files())
        schema = scan.projection()
        target = schema_to_pyarrow(schema, include_field_ids=False)
        # files written under the current schema hold every column under its current
        # name (partition source columns included), so they can be scanned directly
        plain = not any(t.delete_files for t in tasks) and len(tbl.schemas()) == 1
        if plain and tasks:
            io = PyArrowFileIO({**tbl.io.properties})
            scheme, netloc, _ = io.parse_location(tasks[0].file.file_path)
            paths = [io.parse_location(t.file.file_path)[2] for t in tasks]
            if scheme in ("file", "s3", "s3a", "s3n", "hdfs", "viewfs"):
                fs = io.fs_by_scheme(scheme, netloc)
            else:
                # Azure and Google storage through fsspec (adlfs, gcsfs), which honours
                # the catalog's connection string and SAS tokens (pyarrow's own Azure
                # file system signs in only through the default credential chain)
                from pyarrow.fs import FSSpecHandler, PyFileSystem
                from pyiceberg.io.fsspec import FsspecFileIO
                ffs = FsspecFileIO({**tbl.io.properties}).get_fs(scheme)
                fs = PyFileSystem(FSSpecHandler(ffs))
                from urllib.parse import urlparse
                paths = [f"{urlparse(t.file.file_path).netloc.split('@')[0]}/{urlparse(t.file.file_path).path.lstrip('/')}"
                         for t in tasks]
            ds = pds.dataset(paths, format="parquet", filesystem=fs)
            if [f.name for f in ds.schema] == target.names:
                return ds.replace_schema(pa.schema([pa.field(f.name, f.type) for f in ds.schema])), sid
        if not tasks:
            return target.empty_table(), sid
        # delete files, schema evolution or partition values outside the files:
        # merge-on-read through PyIceberg (in memory)
        log.info("iceberg: %s: snapshot %s read through PyIceberg (deletes, evolved schema or partitions)", table, sid)
        return scan.to_arrow(), sid

    def _keys(self, table):
        props = self._table(table).properties
        out = {}
        for k in ("primary_key", "unique", "foreign_keys"):
            v = props.get(f"relsynth.{k}")
            if v:
                try:
                    out[k] = json.loads(v)
                except ValueError:
                    log.warning("iceberg: %s: table property relsynth.%s is not JSON; ignored", table, k)
        return out

    def _partition_spec(self, tbl, table):
        exprs = self.partition_by.get(table)
        if not exprs:
            return
        from pyiceberg import transforms as T
        cur = {(f.source_id, str(f.transform)) for f in tbl.spec().fields}
        with tbl.update_spec() as us:
            for e in exprs:
                fn, col, width = parse_partition(e)
                tr = {"identity": T.IdentityTransform, "year": T.YearTransform, "month": T.MonthTransform,
                      "day": T.DayTransform, "hour": T.HourTransform}.get(fn)
                tr = tr() if tr is not None else (T.BucketTransform(width) if fn == "bucket" else T.TruncateTransform(width))
                fid = tbl.schema().find_field(col).field_id
                if (fid, str(tr)) not in cur:
                    us.add_field(col, tr, f"{col}_{fn}" if fn != "identity" else col)

    def _write(self, table, data, mode, meta, props):
        cat = self.catalog
        cat.create_namespace_if_not_exists(self.schema)
        tprops = {f"relsynth.{k}": json.dumps(v) for k, v in meta.items() if v}
        exists = self.has_table(table)
        if exists and mode in ("replace", "truncate"):
            tbl = self._table(table)
            same = [f.name for f in tbl.schema().fields] == data.schema.names
            if mode == "replace" and not same:
                cat.drop_table(self._ident(table))
                exists = False
        if not exists:
            tbl = cat.create_table(self._ident(table), schema=data.schema, properties=tprops)
            self._partition_spec(tbl, table)
            tbl = self._table(table)
        else:
            tbl = self._table(table)
            if tprops and any(tbl.properties.get(k) != v for k, v in tprops.items()):
                with tbl.transaction() as tx:
                    tx.set_properties(tprops)
                tbl = self._table(table)
        data = data.cast(_iceberg_arrow(tbl, data))
        if mode == "append" or tbl.current_snapshot() is None:
            tbl.append(data, snapshot_properties=props)
        else:
            tbl.overwrite(data, snapshot_properties=props)
        tbl = self._table(table)
        return tbl.current_snapshot().snapshot_id

    def history(self, table):
        tbl = self._table(table)
        out = []
        for s in reversed(tbl.snapshots()):
            out.append({"version": s.snapshot_id, "time": datetime.fromtimestamp(s.timestamp_ms / 1000, timezone.utc).isoformat(),
                        "operation": str(s.summary.operation.value) if s.summary else None,
                        "relsynth": {k: v for k, v in (s.summary.additional_properties if s.summary else {}).items()
                                     if k.startswith("relsynth.")}})
        return out


def _iceberg_arrow(tbl, data):
    """The Arrow schema of an existing table for `data` (names of the table's
    schema, nullability of its fields)."""
    import pyarrow as pa
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    target = schema_to_pyarrow(tbl.schema(), include_field_ids=False)
    have = {f.name: f for f in target}
    missing = [n for n in data.schema.names if n not in have]
    if missing:
        raise ValueError(f"iceberg: {tbl.name()}: columns {missing} are not in the table (if_exists: replace "
                         "recreates it)")
    return pa.schema([pa.field(n, have[n].type, nullable=have[n].nullable) for n in data.schema.names])


# ---------------------------------------------------------------- Delta
class DeltaConnector(LakeConnector):
    """Delta Lake tables, one per folder <path>/<table> (delta-rs)."""

    kind = "delta"

    def __init__(self, path: str | Path, *, base_dir: str | Path | None = None, **kw):
        super().__init__(**kw)
        p = str(path)
        if not _is_uri(p):
            q = Path(p)
            if not q.is_absolute() and base_dir is not None:
                q = Path(base_dir) / q
            p = str(q.resolve())
        self.path = p.rstrip("/")

    def location(self):
        return f"delta {self.path}"

    def uri(self, table):
        from ..data import inside
        if not _is_uri(self.path):
            return str(inside(Path(self.path), table))
        inside(Path("/x"), table)                       # refuses '/', '..' and control characters
        return f"{self.path}/{table}"

    def _names(self):
        from deltalake import DeltaTable
        if not _is_uri(self.path):
            root = Path(self.path)
            if not root.is_dir():
                return []
            return sorted(p.name for p in root.iterdir() if (p / "_delta_log").is_dir())
        import pyarrow.fs as pafs
        if self.storage:
            raise ValueError("delta on object storage with storage options: name the tables with `include` "
                             "(or in the spec); listing a bucket needs the store's own client")
        fs, root = pafs.FileSystem.from_uri(self.path)
        names = []
        for info in fs.get_file_info(pafs.FileSelector(root, allow_not_found=True)):
            if info.type == pafs.FileType.Directory and DeltaTable.is_deltatable(f"{self.path}/{info.base_name}"):
                names.append(info.base_name)
        return sorted(names)

    def has_table(self, table):
        from deltalake import DeltaTable
        if not _is_uri(self.path):
            return table in self._names()
        return DeltaTable.is_deltatable(self.uri(table), storage_options=self.storage or None)

    def tables(self, include=None, exclude=None):
        try:
            names = self._names()
        except ValueError:
            if include:
                names = [p for p in include if self.has_table(p)]
            else:
                raise
        return [t for t in names if match_table(t, include, exclude)]

    def _dt(self, table, version=None):
        from deltalake import DeltaTable
        dt = DeltaTable(self.uri(table), storage_options=self.storage or None)
        if version is not None:
            dt.load_as_version(version)
        return dt

    def _open(self, table):
        dt = self._dt(table)
        pin = self._pinned(table)
        if pin is not None:
            dt.load_as_version(int(pin))
        elif self.as_of is not None:
            dt.load_as_version(self.as_of)
        try:
            return dt.to_pyarrow_dataset(), dt.version()
        except Exception as e:  # noqa: BLE001 - deletion vectors and other reader features
            log.info("delta: %s: version %s read through DataFusion (%s)", table, dt.version(), e)
            from deltalake import QueryBuilder
            import pyarrow as pa
            reader = QueryBuilder().register("t", dt).execute("SELECT * FROM t")
            return pa.Table.from_batches(list(reader), schema=None), dt.version()

    def _keys(self, table):
        desc = self._dt(table).metadata().description or ""
        if desc.startswith(_META_PREFIX):
            try:
                return json.loads(desc[len(_META_PREFIX):])
            except ValueError:
                log.warning("delta: %s: description starting %r is not JSON; ignored", table, _META_PREFIX)
        return {}

    def _write(self, table, data, mode, meta, props):
        from deltalake import write_deltalake
        exprs = self.partition_by.get(table) or []
        parts = []
        for e in exprs:
            fn, col, _ = parse_partition(e)
            if fn != "identity":
                raise ValueError(f"delta: partition_by {e!r}: Delta partitions by column values only; derive a "
                                 "column (a month) in the spec and partition by it")
            parts.append(col)
        exists = self.has_table(table)
        dmode = "append" if mode == "append" else "overwrite"
        kw: dict[str, Any] = {}
        if mode == "replace":
            kw["schema_mode"] = "overwrite"
        elif mode == "truncate" and exists:
            have = self._dt(table).schema().to_arrow()
            data = data.select([n for n in have.names if n in data.schema.names])
        desc = _META_PREFIX + json.dumps(meta) if any(meta.values()) else None
        write_deltalake(self.uri(table), data, mode=dmode, partition_by=parts or None,
                        storage_options=self.storage or None, description=desc if not exists or mode == "replace" else None,
                        name=table if not exists or mode == "replace" else None,
                        commit_properties=_commit(props), **kw)
        return self._dt(table).version()

    def history(self, table):
        out = []
        for h in self._dt(table).history():
            out.append({"version": h.get("version"), "time": pd.Timestamp(h.get("timestamp"), unit="ms", tz="UTC").isoformat()
                        if h.get("timestamp") is not None else None, "operation": h.get("operation"),
                        "relsynth": {k: v for k, v in h.items() if str(k).startswith("relsynth.")}})
        return out


def _commit(props: dict):
    from deltalake import CommitProperties
    return CommitProperties(custom_metadata=dict(props))


# ---------------------------------------------------------------- DuckDB over the datasets
def _duck_class():
    from .duckfiles import ROW_COLUMN, ROWS_SCHEMA, DuckDBFiles

    class _DuckDBLake(DuckDBFiles):
        """The tables of a LakeConnector as views of an in-memory DuckDB database
        over their Arrow datasets (DuckDB scans them with projection and filter
        pushdown, streaming; Raasveldt and Mühleisen 2019).  First rows follow the
        dataset's scan order (the version's file order), which DuckDB keeps under
        preserve_insertion_order."""

        kind = "duckdb"

        def __init__(self, lake, settings=None):
            self._arrow: dict[str, Any] = {}
            self._names: dict[str, str] = {}
            super().__init__(lake, settings)

        def _setup_engine(self, engine):
            import sqlalchemy as sa
            arrow = self._arrow

            # the datasets first: the views replayed on a new connection read them
            @sa.event.listens_for(engine, "connect")
            def _register(dbapi_con, _rec):
                for name, obj in arrow.items():
                    dbapi_con.register(name, obj)
            super()._setup_engine(engine)

        def location(self):
            return f"duckdb over {self.files_conn.location()}"

        def _ensure(self, table):
            if table in self._views:
                return
            ds = self.files_conn.dataset(table)
            # one name per table, kept across refreshes (the views that read it are replayed)
            name = self._names.setdefault(table, f"rs_arrow_{len(self._names)}")
            self._arrow[name] = ds
            q = self._q(table)
            with self.engine.begin() as con:
                con.connection.driver_connection.register(name, ds)
                cols = [f.name for f in ds.schema]
                body = ", ".join(self._q(c) for c in cols)
                self._run(con, f"CREATE OR REPLACE VIEW {q} AS SELECT {body} FROM {name}")  # noqa: S608  # nosec B608
                self._run(con, f"CREATE OR REPLACE VIEW {ROWS_SCHEMA}.{q} AS SELECT {body}, "  # noqa: S608  # nosec B608
                               f"row_number() OVER () - 1 AS {ROW_COLUMN} FROM {name}")
            self._views[table] = cols

        def count(self, table):
            return self.files_conn.count(table)

    return _DuckDBLake


def DuckDBLake(lake, settings=None):  # noqa: N802 - a class built on first use (duckdb is imported lazily)
    return _duck_class()(lake, settings)


def from_section(sec: dict, base_dir=None, create: bool = False) -> LakeConnector:
    """The connector of a spec section of kind iceberg or delta."""
    kind = sec["kind"]
    common = dict(storage=sec.get("storage"), storage_env=sec.get("storage_env"), snapshot=sec.get("snapshot"),
                  as_of=sec.get("as_of"), partition_by=sec.get("partition_by"), engine=sec.get("engine", "auto"),
                  duckdb=sec.get("duckdb"), create=create)
    if kind == "iceberg":
        return IcebergConnector(sec.get("path"), catalog=sec.get("catalog"), schema=sec.get("schema"),
                                base_dir=base_dir, **common)
    if kind == "delta":
        if not sec.get("path"):
            raise ValueError("delta needs `path` (a folder or object-store URI holding one folder per table)")
        return DeltaConnector(sec["path"], base_dir=base_dir, schema=sec.get("schema"), **common)
    raise ValueError(f"not a lakehouse kind: {kind}")

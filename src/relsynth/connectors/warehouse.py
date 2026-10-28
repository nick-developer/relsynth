"""What the cloud warehouses (Databricks SQL, Microsoft Fabric, Snowflake,
BigQuery) share: credentials from the environment or the platform's credential
chain, cost guards, and loads through staged Parquet files.

Credentials (`auth` in a source or target section).  A warehouse connector
never takes a secret from the specification or the command line: `auth` names
the method and the environment variables (or files) that hold the secrets,

    auth: {method: oauth-m2m, client_id_env: DATABRICKS_CLIENT_ID,
           client_secret_env: DATABRICKS_CLIENT_SECRET}

and each connector maps the method to its driver's arguments (`AUTH_METHODS`).
A key `<name>_env` is replaced by the variable's value under `<name>`, a key
`<name>_file` by the file's path; other keys are plain settings (a role, a
tenant id).  A secret written directly into `auth` is refused.  Resolved
secrets go to the driver only: they are not kept on the connector, and
`location()`, `ping()`, logs, bundles and error messages never show them.

Cost guards (`guards`).  Every query a warehouse runs is billed by bytes scanned
or by compute time, so the connector

* tags its queries (`query_tag`, default `relsynth`) so that they can be found
  and charged back in the warehouse's query history;
* refuses to read a table of more than `max_rows` rows into memory, and to
  run a statement whose estimated scan exceeds `max_bytes` (the estimate is the
  warehouse's own dry run where it has one, BigQuery, and otherwise the stored
  size of the tables the statement reads, an upper bound for columnar storage);
* reports, for `relsynth plan`, the bytes each loaded and pushed-down table
  costs to scan (`cost_report`).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import sqlalchemy as sa

from ..errors import SourceError
from .sql import SqlConnector

log = logging.getLogger("relsynth.connectors")

GUARD_KEYS = ("query_tag", "max_rows", "max_bytes")
_UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15,
          "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40, "PIB": 2**50}
#: statements one pushed-down fact costs, at most (SqlStats: counts, distinct values,
#: grouped statistics per parent and key, first rows, the hash sample)
PUSHDOWN_SCANS = 12


class CostGuardError(SourceError):
    """A read or a statement would exceed a cost guard (max_rows, max_bytes)."""


def parse_bytes(v) -> int | None:
    """'100GB', '1.5 TiB', 10**9 -> bytes (None for None)."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*", str(v))
    if not m or m.group(2).upper() not in _UNITS:
        raise ValueError(f"not a size: {v!r} (use e.g. 500MB, 100GB, 2TiB)")
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()])


def human_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} PB"


def resolve_auth(auth: dict | None, methods: dict[str, tuple[str, ...]], default: str | None = None) -> dict:
    """The auth section with every `<name>_env` replaced by the variable's value
    (as `<name>`) and every `<name>_file` by an existing path.

    methods: method -> the names it needs (after resolution).  Raises ValueError
    naming the missing variable, never its value."""
    from ..data import is_secret_key
    a = dict(auth or {})
    method = a.pop("method", None) or default
    if method not in methods:
        raise ValueError(f"auth.method must be one of {', '.join(methods)}, not {method!r}")
    out: dict[str, Any] = {"method": method}
    for k, v in a.items():
        if k.endswith("_env"):
            val = os.environ.get(str(v))
            if val is None or val == "":
                raise ValueError(f"auth: set the environment variable {v} ({k[:-4]} for {method})")
            out[k[:-4]] = val
        elif k.endswith("_file"):
            p = os.path.expanduser(str(v))
            if not os.path.isfile(p):
                raise ValueError(f"auth: {k} names {v}, which is not a file")
            out[k] = p
        elif is_secret_key(k):
            raise ValueError(f"auth.{k}: secrets are never written in the specification; "
                             f"use {k}_env with the name of an environment variable")
        else:
            out[k] = v
    missing = [n for n in methods[method] if n not in out]
    if missing:
        raise ValueError(f"auth method {method} needs " + ", ".join(f"{n}_env" if not n.endswith("_file") else n
                                                                      for n in missing))
    return out


class WarehouseConnector(SqlConnector):
    """A schema of a cloud warehouse (see the module docstring)."""

    #: auth method -> required names (resolved from <name>_env / <name>_file)
    AUTH_METHODS: dict[str, tuple[str, ...]] = {}
    DEFAULT_TAG = "relsynth"

    def __init__(self, url=None, schema=None, *, auth: dict | None = None, guards: dict | None = None,
                 bulk: dict | None = None, **kw):
        g = dict(guards or {})
        unknown = set(g) - set(GUARD_KEYS)
        if unknown:
            raise ValueError(f"guards: unknown keys {sorted(unknown)} (allowed: {', '.join(GUARD_KEYS)})")
        self.query_tag = str(g.get("query_tag") or self.DEFAULT_TAG)
        self.max_rows = int(g["max_rows"]) if g.get("max_rows") is not None else None
        self.max_bytes = parse_bytes(g.get("max_bytes"))
        self.bulk = dict(bulk or {})
        self._auth = auth
        self._bytes: dict[str, int | None] = {}
        super().__init__(url, schema, **kw)

    # ------------------------------------------------------------ credentials
    def _connect_args(self, url: str) -> dict:
        return self.auth_connect_args(sa.engine.make_url(url))

    def auth_connect_args(self, url: sa.URL) -> dict:
        """Driver connection arguments for the configured auth method (subclasses)."""
        return {}

    def _setup_engine(self, engine):
        self.tag_statements(engine)

    def tag_statements(self, engine) -> None:
        """Put the query tag in front of every statement as a comment, which the
        warehouse keeps in its query history (Snowflake and BigQuery use their
        own tag and label settings instead)."""
        tag = self.query_tag.replace("*/", "")

        @sa.event.listens_for(engine, "before_cursor_execute", retval=True)
        def _tag(conn, cursor, statement, parameters, context, executemany):
            return f"/* {tag} */ {statement}", parameters

    # ------------------------------------------------------------ cost guards
    def table_bytes(self, table: str) -> int | None:
        """Stored size of a table in bytes, from the warehouse's catalog (None when unknown)."""
        if table not in self._bytes:
            try:
                self._bytes[table] = self._table_bytes(table)
            except sa.exc.DBAPIError as e:
                log.debug("%s: size unknown (%s)", table, type(e).__name__)
                self._bytes[table] = None
        return self._bytes[table]

    def _table_bytes(self, table: str) -> int | None:
        return None

    def estimate_bytes(self, q, tables: list[str]) -> int | None:
        """Bytes a statement scans at most: the stored size of the tables it reads
        (the BigQuery connector asks for a dry run instead)."""
        sizes = [self.table_bytes(t) for t in tables]
        return None if any(s is None for s in sizes) else sum(sizes)  # type: ignore[misc]

    def guard_scan(self, table: str, q=None, what: str = "statement") -> None:
        if self.max_bytes is None:
            return
        n = self.estimate_bytes(q, [table])
        if n is not None and n > self.max_bytes:
            raise CostGuardError(f"{table}: the {what} would scan about {human_bytes(n)}, more than "
                                 f"guards.max_bytes ({human_bytes(self.max_bytes)}); raise the guard or push "
                                 "the table down (synthesis.pushdown)")

    def guard_rows(self, table: str, rows: int) -> None:
        if self.max_rows is not None and rows > self.max_rows:
            raise CostGuardError(f"{table}: reading {rows:,} rows into memory exceeds guards.max_rows "
                                 f"({self.max_rows:,}); push the table down (synthesis.pushdown) or raise the guard")

    def read_table(self, table, columns=None):
        self.guard_rows(table, self.count(table))
        self.guard_scan(table, what="read")
        return super().read_table(table, columns)

    def sample(self, table, n, seed=7, key=None):
        self.guard_scan(table, what="sample")
        return super().sample(table, n, seed, key)

    def stats(self, table):
        self.guard_scan(table, what="in-database statistics")
        return super().stats(table)

    def cost_report(self, tables: list[str], pushed: dict[str, int] | None = None) -> dict:
        """What reading these tables costs: stored bytes per table, and for pushed
        facts the bytes of all their statistics statements (at most PUSHDOWN_SCANS
        scans).  Nothing is executed beyond catalog look-ups."""
        rows = {}
        total = 0
        known = True
        for t in tables:
            b = self.table_bytes(t)
            scans = PUSHDOWN_SCANS if t in (pushed or {}) else 1
            rows[t] = {"bytes": b, "scans": scans, "scanned": None if b is None else b * scans}
            if b is None:
                known = False
            else:
                total += b * scans
        return {"tables": rows, "scanned": total if known else None, "query_tag": self.query_tag,
                "max_bytes": self.max_bytes, "max_rows": self.max_rows, **self.cost_extra()}

    def cost_extra(self) -> dict:
        """Warehouse-specific cost facts (warehouse size, billing model)."""
        return {}

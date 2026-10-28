# Writing a connector

Every data location relsynth reads or writes goes through one *connector*, which is an object built from `relsynth.connectors.base.Connector`. This guide covers what a new connector must provide, which hooks exist for engine quirks, where to register it, and which tests and documents a pull request must include. MySQL/MariaDB (`connectors/mysql.py`) and DuckDB over files (`connectors/duckfiles.py`) were added this way and serve as worked examples. The cloud warehouses (Databricks SQL, Microsoft Fabric, Snowflake, BigQuery; roadmap thread T18) are meant to follow the same pattern; [Warehouses](#warehouses) lists what is known to differ for them.

## 1. Choose the base class

| The store | Base class | Example |
|---|---|---|
| Has a SQLAlchemy dialect with window functions (`RANK() OVER`, `ROW_NUMBER() OVER`) | `SqlConnector` (`connectors/sql.py`) | PostgreSQL, SQL Server, Oracle, MySQL, SQLite |
| Is a cloud warehouse with such a dialect (billed per query, signed in with tokens or a credential chain) | `WarehouseConnector` (`connectors/warehouse.py`), a `SqlConnector` with `auth`, cost guards and `bulk` | Databricks SQL |
| Is a folder or object store of files | `FileConnector`, with DuckDB for statistics | CSV, Parquet |
| Has neither (an API, a document store) | `Connector` (`connectors/base.py`) | none yet; T13 (documents) |

A `SqlConnector` subclass is usually under 150 lines, because the base class already does reflection, reads, streamed reads, hash samples, the in-database statistics (`SqlStats`), DDL cloning and parents-first loads. The work is in the engine's differences.

## 2. The contract

What every connector must do, and which test checks it:

| Method | Must | Checked by |
|---|---|---|
| `tables(include, exclude)`, `describe(t)` | List tables and return columns with declared types, the primary key, foreign keys and unique keys (empty lists when the store has none) | `test_write_read_and_catalog` |
| `count(t)` | Return the exact number of rows | the same |
| `read_table(t, columns)` | Return the table, normalised with `data.normalise_frame`: decimals become floats, time-zone-aware timestamps become UTC without a zone, text is plain `object` | `test_full_sample_pushdown_matches_sqlite_in_memory` (cell by cell against SQLite) |
| `sample(t, n, seed)` | Return a Bernoulli sample in which a row's inclusion depends only on its key and the seed (Prop. 103), and `(whole table, True)` when `n ≥ rows` | `test_statistics_match_the_in_memory_reference`, `test_introspect_fit_and_pushdown_run` |
| `stats(t)` | Return a `TableStats` whose counts equal `FrameStats` exactly and whose normal-score sums agree within the Acklam bound (Props. 100, 102) | `test_statistics_match_the_in_memory_reference` |
| `write_tables(spec, tables, mode, structure)` | Write parents first, in every mode (`replace`, `append`, `truncate`, `fail`), and create tables whose types hold the values exactly | `test_write_synthetic_to_server_in_every_mode` |
| `location()`, `ping()` | Mask the password and never raise from `ping` | `test_profile_and_masked_errors` |
| Identifiers | Quote every identifier through the dialect and never paste one into SQL text unquoted | `test_hostile_identifiers_on_server` |

The tests in `tests/integration/test_servers.py` are this contract. A new server connector joins them by adding its kind to `KINDS` there, and its URL variable to `SERVER_ENVS` in `tests/conftest.py`.

The decisive check is that **the same data gives the same synthetic database**. When an engine changes a value on the way in or out (precision, case, time zone, trailing spaces), the fit sees different data and the output differs from SQLite. Most of the work on a new engine is finding those changes.

## 3. Hooks for engine differences

### On the connector (`SqlConnector`)

| Hook | Use it for | Example |
|---|---|---|
| `_connect_args(url)` | Driver arguments the connector adds, such as credentials from the environment (`WarehouseConnector` calls `auth_connect_args`) | Databricks access token or OAuth credentials provider |
| `_engine_url(url)` | Driver options the URL must carry | Fabric: ODBC Driver 18, `Encrypt=yes`, `Authentication=ActiveDirectoryInteractive` |
| `adapt_metadata(md)` | Constraints, types and table options the engine needs, applied to the tables `write_tables` and `clone-schema` create | Databricks: keys only in Unity Catalog, no `UNIQUE` or indexes, Delta column mapping. Fabric: `VARCHAR` and `DATETIME2(6)`, keys `NOT ENFORCED` |
| `_row_hash(cols, seed)` | The sample hash, when the engine shares a dialect but not its functions | Fabric: SQL Server's dialect, but no `NVARCHAR` |
| `_engine_options(url)` | Engine and pool options | `pool_recycle` (MySQL), `fast_executemany` (SQL Server), `arraysize` (Oracle) |
| `_setup_engine(engine)` | Per-connection session settings and SQL functions | MySQL `time_zone = '+00:00'`; SQLite registers `rs_ndtri` and `rs_hash`; DuckDB registers `rs_ndtri` |
| `table(name)` | Fix reflected types | MySQL `DOUBLE` reflects as `asdecimal=True` (rounds to 10 decimals) |
| `create_schema(schema)` | Engines where a schema is something else | MySQL: a database with a binary collation |
| `_insert_rows(con, tbl, df)` | The vendor's bulk path | `COPY` (PostgreSQL), array binding (Oracle) |
| `before_load` / `after_load` | Identity columns and sequences | `IDENTITY_INSERT` (SQL Server), `setval` (PostgreSQL) |
| `_sample_key(t, key)` | Which columns the sample hash reads | DuckDB views have no key: all columns |
| `frame(q)` | How a query result becomes a DataFrame (every statistic and sample goes through it) | DuckDB fetches columnar results (`.df()`), because row tuples of a large grouped result cost several GB |

### On the statistics (`SqlStats`)

| Hook | Use it for | Example |
|---|---|---|
| `col(name)` | How a column is compared, grouped and filtered | MySQL compares text in the binary no-pad collation, because the default collations ignore case, accents and trailing spaces |
| `_row_order(tbl)` | What orders a key's rows after `order` in `first_rows` | the primary key; DuckDB uses the file row number |

### SQL functions (`connectors/sqlfuncs.py`)

Each function compiles per dialect with `@compiles(..., "<dialect>")`:

| Function | Default | Add a dialect branch when |
|---|---|---|
| `to_float(x)` | `CAST(x AS FLOAT)` | `FLOAT` is not 8 bytes (DuckDB: `DOUBLE`) or cannot be cast (MySQL: `x + 0E0`) |
| `ln(x)` | `LN(x)` | the name differs (SQL Server: `LOG`) |
| `ndtri(p, dialect)` | Acklam's approximation in SQL (Prop. 102) | the engine can register an exact function (SQLite, DuckDB) |
| `row_hash(cols, seed, dialect)` | none: random order, not reproducible | always. Give a seeded hash in [0, 2³²) of `'seed\|key'`. Prefer SHA-256's first 32 bits (`py_row_hash_sha256` checks it from Python) |

### Created types (`connectors/ddl.py`)

`_generic` maps a source type to the target engine's type, and `adapt_types(md, dialect)` adjusts the tables relsynth creates (MySQL: `DOUBLE`, `DATETIME(6)`, binary collation, `VARCHAR` within the index and row limits). Add a branch there when a generic type loses values on the new engine. Every new branch needs a unit test that runs without a server, such as `test_mysql_types_keep_values_and_fit_mysql_limits`.

## 4. Register the kind

| Where | What |
|---|---|
| `connectors/__init__.py` | `_sql_class(dialect)` returns the class; `KINDS`; the module docstring table |
| `connectors/sql.py` `_driver()` | The default driver for a bare `kind://` URL, plus any URL options it needs (MySQL adds `charset=utf8mb4`) |
| `spec.py` | `SOURCE_KINDS`, `TARGET_KINDS`, `SERVER_KINDS`, and `WAREHOUSE_KINDS` for a warehouse (it allows `auth`, `guards`, `bulk`) |
| `schema.py` | The `kind` choices and help text; regenerate `docs/spec-reference.md` (`python docs/make_spec_reference.py`) |
| `pipeline.py` | The URL scheme → kind map |
| `cli.py` | The `--source` help; regenerate `docs/cli.md` (`python docs/make_cli_reference.py`) |
| `pyproject.toml` | An extra with the driver (`relsynth[mysql]`), added to `all`; a pytest marker |

## 5. Tests and evidence for the pull request

1. **Contract.** Add the kind to `KINDS` in `tests/integration/test_servers.py` and to `SERVER_ENVS`. Add a service to `docker-compose.test.yml` and to the `integration` job in `.github/workflows/ci.yml`.
2. **Engine-specific tests.** Add `tests/integration/test_<kind>.py` for each difference you handled. Write each test so it fails without the fix. For example, `test_text_is_compared_exactly_under_a_case_insensitive_collation` creates a table in the server's default collation.
3. **Unit tests without a server.** Cover compiled SQL, type mapping and URL handling.
4. **Real-database equivalence.** Copy Chinook and Contoso (both allowed in the dataset register) from SQLite to the new engine with relsynth itself, then fit with the same seed. The synthetic database must be identical cell by cell, in memory and with full-sample pushdown. The script is `roadmap/reports/T04/scripts/mysql_equivalence.py` in the project files; it moves to the benchmark harness when T01 lands.
5. **Documentation.** Update the connector table, schemas, bulk path, sample hash, type-mapping column and troubleshooting in `docs/connectors.md`, the per-engine row in `docs/pushdown.md`, the live-server list in `docs/testing.md`, the README table and the changelog.

## Warehouses

Notes for T18. Databricks and Fabric are done: see [warehouses.md](warehouses.md) for what was found. The rest still need a live check before they are relied on.

| Topic | What to expect | Consequence for the connector |
|---|---|---|
| Constraints | Snowflake, BigQuery, Databricks and Fabric warehouses store primary and foreign keys as informational only, or not at all | `describe` may return keys that are not enforced. `introspect` already identifies keys from values; write targets must not rely on the database rejecting bad rows |
| Bulk loads | Row inserts are slow and costly | Stage Parquet files (Arrow → Parquet) and run `COPY INTO` / load jobs from `_insert_rows`. This is the place where Arrow saves copies |
| Sample hash | `SHA2` (Snowflake, Databricks), `SHA256` (BigQuery), `HASHBYTES` (Fabric) | The same `'seed\|key'` SHA-256 scheme as MySQL and DuckDB, so samples can be checked from Python |
| Window functions and `GROUP BY` | Available on all four | `SqlStats` should run unchanged. Check `to_float` on each |
| Cost | Every query is billed by bytes scanned or by compute time | `SqlStats` runs a handful of scans per pushed fact. Report the queries in `relsynth plan` so users can estimate the cost |
| Authentication | OAuth, key pairs, workload identity | Use `connect_args` and `url_env`; never put tokens in specs |

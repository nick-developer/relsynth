# Connectors: databases and files

relsynth reads from and writes to one *location* at a time: a schema of a SQL database or a folder of files. A location is given either as a short string on the command line or in Python, or as a section of the specification (`source:` and `target:`).

| Kind | Location string | Spec section | Driver (pip extra) | Tested live |
|---|---|---|---|---|
| PostgreSQL | `postgresql://user@host:5432/db` | `{kind: postgresql, url: ..., schema: sales}` | psycopg 3 (`relsynth[postgres]`); psycopg2 also works | PostgreSQL 16.15 |
| SQL Server, Azure SQL | `mssql://user@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server` | `{kind: mssql, url: ..., schema: dbo}` | pyodbc and Microsoft ODBC Driver 17/18 (`relsynth[mssql]`), or pymssql (`relsynth[mssql-pymssql]`) | SQL Server 2022 (16.0) |
| Oracle | `oracle://user@host:1521/?service_name=ORCLPDB1` | `{kind: oracle, url: ..., schema: SALES}` | python-oracledb, thin mode (`relsynth[oracle]`) | Oracle Database 26ai Free (23.26) |
| MySQL, MariaDB | `mysql://user@host:3306/sales`, `mariadb://user@host:3306/sales` | `{kind: mysql, url: ..., schema: sales}` | PyMySQL (`relsynth[mysql]`); mysqlclient is used when installed | MySQL 8.4.13, MariaDB 11.4.13 |
| Databricks SQL | `databricks://<host>?http_path=/sql/1.0/warehouses/<id>&catalog=main` | `{kind: databricks, url: ..., schema: sales, auth: {...}}` | databricks-sqlalchemy (`relsynth[databricks]`) | Spark 3.5.3 + Delta 3.2.1 stand-in; see [warehouses.md](warehouses.md) |
| SQLite | `sqlite:shop.db`, `sqlite:///shop.db`, or a path ending in `.db` | `{kind: sqlite, path: shop.db}` | standard library | SQLite 3.45 |
| CSV folder | `csv:data/` (also `files:data/`) | `{kind: csv, path: data, null_values: ["", "n/a"]}` | pandas | yes |
| Parquet folder | `parquet:lake/sales/` | `{kind: parquet, path: lake/sales}` | pyarrow (`relsynth[parquet]`) | yes |

The cloud warehouses take three more settings, `auth`, `guards` and `bulk`, described in [warehouses.md](warehouses.md).

Relative paths in a spec are relative to the spec file's folder. Any other SQLAlchemy dialect can be tried with its full URL (`dialect+driver://...`), but only the kinds above are tested and supported.

## Credentials

Passwords never belong in a spec file or on a command line, where they end up in shell history, CI logs and process lists. Give them through the environment instead:

| Setting | Command line | Meaning |
|---|---|---|
| `password_env: PGPASSWORD` | `--source-password-env PGPASSWORD` | The variable holds the password; the URL has none |
| `url_env: SOURCE_URL` | `--source-url-env SOURCE_URL` | The variable holds the whole URL |

The target takes the same settings (`--target-password-env`, `--target-url-env`). When a spec is saved by relsynth (introspect, doctor, tune, `Spec.save`), passwords are replaced by `***` and a `url_env` is added. The masked file can be committed, and the run then reads the real URL from that variable. Integrated authentication works as the driver supports it:
- SQL Server: `?trusted_connection=yes` with ODBC, or `Authentication=ActiveDirectoryMsi` for Azure.
- Oracle: wallets through `connect_args: {config_dir: ..., wallet_location: ...}`.
- PostgreSQL: `.pgpass` or `PGPASSWORD`.

Driver options go in `connect_args` and engine options in `engine_options`. For example, `engine_options: {pool_pre_ping: true}` for connections that pass through firewalls with idle timeouts.

## Schemas

- **PostgreSQL and SQL Server.** `schema` names the schema to read or write; the default is the user's default schema (`public`, `dbo`). A target schema is created when it does not exist (`CREATE SCHEMA`).
- **Oracle.** A schema is a user: `schema: SALES` reads that user's tables, which needs `SELECT` grants on them. Without `schema`, the connecting user's own tables are used. Oracle stores an empty string as NULL, so text columns whose source has empty strings read back as nulls.
- **MySQL and MariaDB.** A schema is a database: `schema: sales` (or the database in the URL) is read or written, and a missing target database is created with `utf8mb4` and a binary collation. The user needs `CREATE` on it for writes. `mysql://` on a MariaDB server works; the connector asks the server which it is.
- **SQLite.** There is one schema per file.
- **Files.** There is one table per file: `<table>.csv`, `<table>.csv.gz` or `<table>.parquet`.

Names are case-sensitive as the database stores them. Oracle stores unquoted names in upper case, and relsynth reflects them in lower case, as SQLAlchemy does. A name that needs quotes, such as one with spaces, quotes or `;`, works on every database: identifiers are always quoted by the dialect, never pasted into SQL text.

## Reading

- **Catalog.** relsynth reads tables, columns and their declared types, primary keys, foreign keys, unique constraints and indexes. `introspect` uses declared keys when they exist, and otherwise identifies keys and links from the values.
- **Whole tables.** Large tables are streamed in chunks.
- **Samples.** A sample keeps a row when a seeded hash of its key is below a threshold, so it is a uniform Bernoulli sample. The same seed gives the same sample, and the database does the work:

  | Engine | Hash |
  |---|---|
  | PostgreSQL | `hashtextextended` |
  | SQL Server | `HASHBYTES('SHA2_256')` |
  | Oracle | `ORA_HASH` |
  | MySQL, MariaDB | the first 32 bits of `SHA2('seed\|key', 256)`, equal to Python's `hashlib.sha256` |
  | SQLite | a registered BLAKE2 function |
  | Files | SHA-256, computed by DuckDB over the file (`engine: pandas`: a seeded Bernoulli draw while streaming) |

- **Statistics** for the in-database fit are computed with `GROUP BY` and window functions. See [pushdown.md](pushdown.md).
- **Binary columns** (BLOB, VARBINARY, BYTEA, RAW) are reported as binary and left out of the model.
- **Canonical row order.** Every loaded table is put in a canonical order: by primary key, or by a hash of the row's values when there is none. The fitted model therefore does not depend on the order in which a database returns rows. The same data on PostgreSQL, SQL Server, Oracle, MySQL, MariaDB, SQLite or in files gives the same synthetic database for the same seed, and the tests check this cell by cell.

## Writing

`target:` (or `--target` on the command line, or `SyntheticData.write(target)` in Python) says where synthetic tables go:

```yaml
target:
  kind: postgresql
  url: postgresql://loader@test-db/sales_synth
  password_env: TEST_DB_PASSWORD
  schema: synth
  if_exists: replace        # replace | append | truncate | fail
  structure: auto           # auto | clone | infer | existing
```

| `if_exists` | Behaviour |
|---|---|
| `replace` (default) | Drop the target tables (children first) and create them again |
| `append` | Insert into the existing tables (created when missing) |
| `truncate` | Delete all rows (children first), then insert |
| `fail` | Stop with exit code 3 when any target table exists |

| `structure` | Target tables are created from |
|---|---|
| `clone` | The source's DDL: column types, nullability, primary keys, unique constraints, foreign keys, indexes and identity columns. Types are mapped between engines; see below |
| `infer` | The spec: key and link constraints, with column types from the data |
| `existing` | Nothing: the tables exist (a DBA created them); used with `append` or `truncate` |
| `auto` (default) | `clone` when the source is a SQL database, `infer` otherwise |

Tables are loaded parents first, so foreign keys hold after every table. A self-referencing table, such as an employee's manager, is loaded by depth. Each table is one transaction.

Bulk paths:

| Engine | How rows are loaded |
|---|---|
| PostgreSQL | `COPY ... FROM STDIN` (psycopg 3); identity sequences restarted after the largest loaded key |
| SQL Server | ODBC array binding (`fast_executemany`); explicit keys into identity columns with `IDENTITY_INSERT`; pymssql uses multi-row INSERT batches (5 to 10 times slower) |
| Oracle | Array binding (`executemany`); identity columns created `GENERATED BY DEFAULT` so explicit keys load, and restarted after the largest key |
| MySQL, MariaDB | The driver's `executemany`, sent as multi-row `INSERT` statements; `AUTO_INCREMENT` continues after the largest loaded key by itself. `LOAD DATA LOCAL INFILE` is not used: servers disable it by default, because it lets the server read a client file |
| SQLite | `executemany` in one transaction; foreign keys enforced |
| CSV | `pandas.to_csv`; bytes as `0x`-prefixed hex |
| Parquet | pyarrow; types kept |

### Type mapping when cloning between engines

On the same engine the source type is kept exactly. Between engines each column gets the generic equivalent, compiled for the target server version. The table shows what the live test servers received:

| Source type | PostgreSQL 16 | SQL Server 2022 | Oracle 23 | MySQL 8.4, MariaDB 11.4 | SQLite |
|---|---|---|---|---|---|
| `INTEGER` | `INTEGER` | `INTEGER` | `INTEGER` | `INTEGER` | `INTEGER` |
| `BIGINT` | `BIGINT` | `BIGINT` | `NUMBER(19)` | `BIGINT` | `BIGINT` |
| `NUMERIC(10,2)` | `NUMERIC(10, 2)` | `NUMERIC(10, 2)` | `NUMERIC(10, 2)` | `NUMERIC(10, 2)` | `NUMERIC(10, 2)` |
| `NUMERIC(p,0)`, p ≤ 9 / p ≤ 18 | `INTEGER` / `BIGINT` | `INTEGER` / `BIGINT` | `INTEGER` / `NUMBER(19)` | `INTEGER` / `BIGINT` | `INTEGER` / `BIGINT` |
| Oracle `NUMBER` without precision | `FLOAT` | `FLOAT` | `FLOAT` | `DOUBLE` | `FLOAT` |
| `FLOAT`, `DOUBLE`, `REAL` | `FLOAT` | `FLOAT` | `FLOAT` | `DOUBLE` | `FLOAT` |
| `VARCHAR(n)`, `NVARCHAR(n)` | `VARCHAR(n)` | `NVARCHAR(n)` | `VARCHAR2(n CHAR)` | `VARCHAR(n)` | `VARCHAR(n)` |
| `VARCHAR` without length, or longer than the target allows | `VARCHAR` | `NVARCHAR(max)` | `CLOB` | `LONGTEXT` | `VARCHAR` |
| `TEXT`, `CLOB` | `TEXT` | `NVARCHAR(max)` | `CLOB` | `LONGTEXT` | `TEXT` |
| `DATE` | `DATE` | `DATE` | `DATE` | `DATE` | `DATE` |
| `TIMESTAMP`, `DATETIME` | `TIMESTAMP` | `DATETIME` | `DATE` | `DATETIME(6)` | `DATETIME` |
| `BOOLEAN`, `BIT` | `BOOLEAN` | `BIT` | `BOOLEAN` | `BOOL` (`TINYINT(1)`) | `BOOLEAN` |
| `BLOB`, `VARBINARY`, `BYTEA` | `BYTEA` | `VARBINARY(max)` | `BLOB` | `BLOB` | `BLOB` |

On MySQL and MariaDB, text columns get the character set `utf8mb4` and a binary collation without trailing-space padding (`utf8mb4_0900_bin` on MySQL, `utf8mb4_nopad_bin` on MariaDB), so a unique key holds on the exact values, as on the other engines. Two MySQL limits change the types of some columns:
- A text column in a key, unique constraint, foreign key or index is created as `VARCHAR`, sized so the index fits MySQL's 3,072-byte limit (768 characters, shared among the index's columns).
- The `VARCHAR` columns of a row share 65,535 bytes. When a wide table would exceed that, its longest non-key `VARCHAR` columns become `LONGTEXT`, which is stored off the row.

Between MySQL databases (same engine) the source types, including their collations, are kept as they are.

Three conversions lose precision:
- SQL Server `DATETIME` keeps time to 3.33 ms.
- Oracle `DATE` keeps whole seconds.
- MySQL `BOOL` reads back as 0 and 1.

When fractional seconds matter, create the target tables yourself with `DATETIME2` or `TIMESTAMP` and write with `structure: existing`.

Use `relsynth clone-schema --source ... --target ...` to create the empty structure alone, for example for a DBA to review.

## Files

```yaml
source: {kind: csv, path: ../extract, null_values: ["", "n/a", "NULL"]}
```

- **CSV.** CSV files are read as text and then typed. Numbers become numbers unless a value has a leading zero, so `0101` stays a code. ISO dates and timestamps with time zones are parsed, and a `.csv.gz` file is read directly. Binary values are written as `0x`-prefixed hex, as SQL tools export them.
- **Parquet.** Parquet keeps its types. Statistics read only the needed column chunks.
- **Keys.** Files declare no keys, so the spec (drafted by `introspect` from the values) carries them.
- **Large files.** The statistics and samples of a pushed-down fact are computed by DuckDB, an in-process analytical database, running SQL over the files without loading them. Entities are still loaded with pandas. Settings:

  ```yaml
  source:
    kind: parquet
    path: lake/sales
    engine: duckdb            # auto (default: duckdb when installed) | duckdb | pandas (the 1.0 path)
    duckdb: {memory_limit: 4GB, threads: 8, temp_directory: /scratch/duckdb}
  ```

  A CSV column read by DuckDB gets the type the pandas reader gives it. All fields are read as text, only the null tokens become nulls, and a column is a number when every value parses as one and none has a leading zero.
- **Table names.** A table name must be a plain file name. Names with `/`, `\`, `..`, NUL or an absolute path are refused, and so is a name whose file resolves outside the folder (a symbolic link).

## Testing a connection

```bash
relsynth ping --source "oracle://scott@db:1521/?service_name=ORCLPDB1" --source-password-env ORA_PW
relsynth profile --source "mssql://reader@sql01/sales?driver=ODBC+Driver+18+for+SQL+Server" \
                 --source-password-env MSSQL_PW --output profile.json
```

`ping` reports the server version and the number of tables it can see; it never raises, so a script can read its JSON (`ok: false` with the error, password masked). `profile` reports row, null and distinct counts per column, computed in the database. No rows are read, and `--top N` must be asked for to list frequent values.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Data source name not found` (SQL Server) | The ODBC driver is not installed or not named in the URL; add `?driver=ODBC+Driver+18+for+SQL+Server`, or install `relsynth[mssql-pymssql]` |
| `SSL Provider: certificate verify failed` (SQL Server) | Driver 18 encrypts by default; use a trusted certificate, or for test servers `&TrustServerCertificate=yes` |
| `DPY-6005` / `ORA-12514` (Oracle) | Wrong service name; use `?service_name=...` (not a SID) or an Easy Connect string |
| `set the environment variable X` | `password_env` or `url_env` names a variable that is not set in this process |
| `Access denied for user ... to database` (MySQL) | A target database is created when missing, which needs `CREATE`; create it beforehand or grant the right |
| Statistics that merge `abc` and `ABC` | Not with relsynth: the connector compares text in a binary collation whatever the column's collation. Other tools reading the same server with its default collation do merge them |
| Exit code 3 | Connection, permission or file error; the message names the location with its password masked |

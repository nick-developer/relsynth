# Cloud warehouses

relsynth reads from and writes to four cloud warehouses, added in the order the
roadmap set: **Databricks SQL**, **Microsoft Fabric**, **Snowflake** and
**BigQuery**. Each is a connector built on the [connector guide](connector-sdk.md),
so everything in [connectors.md](connectors.md) and [pushdown.md](pushdown.md)
applies: reading, writing in every mode, `clone-schema`, and fitting large facts
inside the warehouse.

What the warehouses add is in three settings of a `source` or `target` section:

| Key | What it sets |
|---|---|
| `auth` | How to sign in. The secret itself is never in the file: `auth` names the method and the environment variables (or key files) that hold the secrets |
| `guards` | Cost guards: a query tag, the largest table read into memory, the largest scan of one statement |
| `bulk` | Target only: how rows are loaded (`INSERT` statements, or Parquet files staged and loaded with the warehouse's bulk command) |

## Credentials

```yaml
source:
  kind: databricks
  url: databricks://adb-1234.5.azuredatabricks.net?http_path=/sql/1.0/warehouses/abc123&catalog=main
  schema: sales
  auth:
    method: oauth-m2m
    client_id_env: DATABRICKS_CLIENT_ID
    client_secret_env: DATABRICKS_CLIENT_SECRET
```

- A key ending in `_env` names an environment variable. relsynth reads it when it connects and passes the value to the driver only.
- A key ending in `_file` names a file, such as a private key.
- Any other key is a plain setting, such as a profile name.
- A secret written directly into `auth` (`token: dapi...`) is refused with a message that says which `_env` key to use.
- An error about a missing variable names the variable, never a value.
- The connector does not keep resolved secrets as attributes. `location()`, `ping()`, logs, saved specs, bundles and error messages never show them. `tests/unit/test_warehouses.py` and the shared `test_profile_and_masked_errors` check this.

The methods of each warehouse are listed in its section below.

## Cost guards

Every query a warehouse runs costs money, by bytes scanned (BigQuery on demand) or by compute time (Databricks, Fabric, Snowflake). The guards stop expensive mistakes before they run:

```yaml
source:
  kind: databricks
  ...
  guards:
    query_tag: relsynth-sales-test   # default: relsynth
    max_rows: 20000000               # the largest table read into memory
    max_bytes: 200GB                 # the largest estimated scan of one statement
```

| Guard | What happens |
|---|---|
| `query_tag` | Every statement is tagged, so its cost can be found and charged back in the warehouse's query history. Databricks and Fabric: a leading comment `/* <tag> */`. Snowflake: the session's `QUERY_TAG`. BigQuery: the job label `relsynth` |
| `max_rows` | Reading a table with more rows into memory stops with a `CostGuardError`, which tells you to push the table down (`synthesis.pushdown`) or raise the guard. Counts come from the warehouse's metadata, so the check itself costs almost nothing |
| `max_bytes` | A read, a sample or the in-warehouse statistics of a table whose estimated scan exceeds the guard stop before they run. The estimate is BigQuery's dry run where it exists, and otherwise the stored size of the table (Delta `sizeInBytes`, Snowflake `BYTES`). For columnar storage that is an upper bound, because a query reads only the columns it needs |

`relsynth plan` (and `connector.cost_report(tables, pushed)` in Python) lists the stored bytes of every table, and how many scans a pushed-down fact costs: at most 12, one per statistics statement. It runs only catalog look-ups.

## Databricks SQL

```
databricks://<workspace host>?http_path=/sql/1.0/warehouses/<id>&catalog=<catalog>[&schema=<schema>]
```

Install with `pip install "relsynth[databricks]"` (databricks-sqlalchemy, the Databricks SQL connector and the Databricks SDK). `schema` in the section overrides the URL's. The session starts in the URL's catalog.

### Sign-in

| `auth.method` | Needs | What it is |
|---|---|---|
| `token` | `token_env` | A personal access token |
| `oauth-m2m` | `client_id_env`, `client_secret_env` | A Databricks service principal with an OAuth secret (machine to machine) |
| `azure-sp` | `client_id_env`, `client_secret_env`, `tenant_id_env` | A Microsoft Entra ID service principal (Azure Databricks) |
| `oauth-u2m` | | Browser sign-in to Databricks (user to machine) |
| `azure-u2m` | | Browser sign-in with Entra ID |
| `default` | | The Databricks SDK's credential chain: `DATABRICKS_*` variables, a `~/.databrickscfg` profile (`profile: <name>`), the Azure CLI, managed identities |

Without `auth`, a token in `DATABRICKS_TOKEN` is used when that variable is set, and the SDK's chain otherwise.

### What differs from the other databases

| Topic | Databricks | What relsynth does |
|---|---|---|
| Floating point | `FLOAT` is 4 bytes | Created float columns are `DOUBLE`. Statistics cast with `CAST(x AS DOUBLE)` |
| Text | `STRING`, compared byte by byte (`UTF8_BINARY`) | Text columns are `STRING` whatever their declared length. No collation is needed for exact statistics |
| Date-times | `TIMESTAMP` is in the session's time zone | Created date-time columns are `TIMESTAMP_NTZ` (the dialect's mapping), so values are stored as given |
| Keys | Primary and foreign keys are informational (never enforced) and exist only in Unity Catalog. `UNIQUE` and indexes do not exist | On a Unity Catalog schema, primary keys and foreign keys to a primary key are declared, and identity columns are created `GENERATED BY DEFAULT AS IDENTITY`, so keys can be loaded. In `hive_metastore` (and on open-source Spark) no keys are declared, and key columns stay `NOT NULL`. `introspect` finds keys from the values either way |
| Column names | Spaces, commas, parentheses, `;`, `{}`, `=` and tabs need Delta column mapping | Column mapping (`delta.columnMapping.mode = 'name'`) is switched on for the tables that have such names |
| Sample hash | `sha2` | The first 32 bits of `sha2('seed\|key', 256)`, the same as MySQL and DuckDB. Python's `hashlib.sha256` reproduces it |
| Transactions | None across statements | Each table is loaded parents first. A failed load leaves the tables written so far; `if_exists: replace` starts over |

### Loading rows

| `bulk.method` | How | When |
|---|---|---|
| `insert` (default) | Multi-row `INSERT` statements with typed literals (doubles in exponent form, strings escaped by the Databricks rule), at most about 4 MB of SQL each. Each statement is one Delta commit | Up to a few million rows |
| `volume` | Each table is written as one Parquet file, uploaded with the connector's `PUT` to `bulk.volume` (a Unity Catalog volume, `/Volumes/<catalog>/<schema>/<volume>`), loaded with `COPY INTO`, and removed (`keep_files: true` keeps it) | Large tables; needs `WRITE VOLUME` on the volume |
| `path` | As `volume`, but the file is written to `bulk.path`, a location the warehouse reads (an external location such as `abfss://`, `s3://` or `gs://`, written through pyarrow's file systems) | When a volume is not available |

### Troubleshooting

| Message | Cause and fix |
|---|---|
| `Setting initial namespace not supported` | The cluster runs a Databricks Runtime older than 9.0. Use a SQL warehouse, or a cluster with DBR 9.0 or later |
| `[DELTA_INVALID_CHARACTERS_IN_COLUMN_NAMES]` on a table you created yourself | Switch on column mapping for it, or let relsynth create it (`if_exists: replace`) |
| `ping` fails after 30 seconds while queries work | A stopped warehouse takes minutes to start. `ping` gives up after 30 seconds; reads and writes wait as long as the driver does (15 minutes by default) |
| `CostGuardError: ... more than guards.max_bytes` | The table is larger than the guard. Push it down (`synthesis.pushdown`) or raise the guard |

### How it was tested

No Databricks emulator exists. The tests run the real Databricks SQL connector and dialect against Apache Spark 3.5.3 with Delta Lake 3.2.1, through a small proxy that adapts the protocol: `tests/emulators/databricks` (its README lists what it can and cannot check). Unity Catalog keys, identity columns, `COPY INTO` itself, volume uploads and OAuth sign-in exist only on Databricks and have unit tests of the SQL they send, but no live test yet. Run `pytest -m databricks tests/integration` against a real workspace (set `RELSYNTH_TEST_DATABRICKS_URL` and the usual credential variables) before relying on them.

### Sources

- Databricks SQL language reference: [FLOAT type](https://docs.databricks.com/en/sql/language-manual/data-types/float-type.html), [`sha2`](https://docs.databricks.com/en/sql/language-manual/functions/sha2.html), [`conv`](https://docs.databricks.com/en/sql/language-manual/functions/conv.html), [`COPY INTO`](https://docs.databricks.com/en/sql/language-manual/delta-copy-into.html), [`DESCRIBE DETAIL`](https://docs.databricks.com/en/sql/language-manual/delta-describe-detail.html).
- [Constraints on Databricks](https://docs.databricks.com/en/tables/constraints.html) (informational primary and foreign keys, Unity Catalog only) and [identity columns](https://docs.databricks.com/en/delta/generated-columns.html).
- [Delta column mapping](https://docs.databricks.com/en/delta/column-mapping.html).
- [Databricks SQL Connector for Python](https://docs.databricks.com/en/dev-tools/python-sql-connector.html) (authentication and `PUT`/`REMOVE` on volumes) and [databricks-sqlalchemy](https://github.com/databricks/databricks-sqlalchemy).

## Microsoft Fabric

```
fabric://<id>.datawarehouse.fabric.microsoft.com/<warehouse>
```

Install with `pip install "relsynth[fabric]"` (pyodbc, azure-identity and the ADLS client), and the [Microsoft ODBC Driver 18 for SQL Server](https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server). The host and warehouse are on the warehouse's settings page in Fabric ("SQL connection string"). The URL becomes `mssql+pyodbc` with the ODBC Driver 18 and `Encrypt=yes`; options already in the URL are kept. `schema` defaults to `dbo`.

A Fabric Warehouse is read and written. A Lakehouse's SQL analytics endpoint is read only, so it can be a source but not a target.

### Sign-in

Fabric accepts Microsoft Entra ID only.

| `auth.method` | Needs | What it is |
|---|---|---|
| `default` | | azure-identity's `DefaultAzureCredential`: `AZURE_*` variables, workload identity, managed identity, the Azure CLI |
| `azure-sp` | `client_id_env`, `client_secret_env`, `tenant_id_env` | A service principal with a client secret. The workspace must allow service principals |
| `azure-cli` | | The account of `az login` on this machine |
| `managed-identity` | optional `client_id_env` | The managed identity of the Azure resource relsynth runs on |
| `interactive` | | Browser sign-in through the ODBC driver |
| `sql-login` | user in the URL, `password_env` | SQL Server stand-ins only. Fabric has no SQL logins |

Without `auth`, the credential chain (`default`) is used, or `sql-login` when the URL has a password. A token is requested for each new connection and handed to the driver directly (`SQL_COPT_SS_ACCESS_TOKEN`), so it is renewed when it expires and is never part of a connection string.

### What differs from SQL Server

| Topic | Fabric | What relsynth does |
|---|---|---|
| Text | No `NVARCHAR`, `NCHAR` or `TEXT`. `VARCHAR(n)` counts bytes, in a UTF-8 collation | Text columns are `VARCHAR(4 × characters)`, or `VARCHAR(MAX)` beyond 8,000 bytes, so every UTF-8 value fits |
| Date-times | No `DATETIME` or `SMALLDATETIME` | `DATETIME2(6)` and `TIME(6)`: microseconds, as pandas has them |
| Other types | No `TINYINT` or `MONEY` | `SMALLINT` and `DECIMAL(19, 4)`. Floats are `FLOAT(53)` |
| Collation | `Latin1_General_100_BIN2_UTF8` by default; a warehouse can be created case-insensitive (`Latin1_General_100_CI_AS_KS_WS_SC_UTF8`) | In a case-insensitive warehouse, text is grouped and filtered with `COLLATE Latin1_General_100_BIN2_UTF8`, so `'a'` and `'A'` stay different values |
| Keys | Primary, unique and foreign keys exist only as `NOT ENFORCED` (and `NONCLUSTERED`). No indexes. No identity columns by default | Keys are added after each table as `NOT ENFORCED` constraints; no indexes, identity columns or defaults are created. Fabric never rejects a row, so write targets do not rely on it |
| Sample hash | `HASHBYTES` | The first 32 bits of `HASHBYTES('SHA2_256', 'seed\|key')` over `VARCHAR`, whose bytes are UTF-8: the same value as Python's `hashlib.sha256`, MySQL, DuckDB and Databricks |
| Cost | Capacity units of the Fabric capacity; no stored table size in the catalog | `query_tag` and `max_rows` work; `max_bytes` cannot be checked before a query, and `relsynth plan` says so |

### Loading rows

| `bulk.method` | How | When |
|---|---|---|
| `insert` (default) | ODBC parameter arrays (`fast_executemany`), 20,000 rows per batch | Up to a few hundred thousand rows. Fabric recommends `COPY INTO` for more |
| `copy` | Each table is written as one Parquet file to `bulk.stage` (an `https://<account>.dfs.core.windows.net/<container>/<folder>` or OneLake `https://onelake.dfs.fabric.microsoft.com/<workspace>/<item>/Files/<folder>` URL), uploaded with the same Entra ID credential, loaded with `COPY INTO ... WITH (FILE_TYPE = 'PARQUET')`, and removed (`keep_files: true` keeps it) | Large tables. The identity needs write access to the folder, and the warehouse reads it with the caller's identity |

### Troubleshooting

| Message | Cause and fix |
|---|---|
| `Data source name not found and no default driver specified` | The ODBC Driver 18 is not installed |
| `Login failed for user '<token-identified principal>'` | The identity has no access to the workspace, or the workspace does not allow service principals (tenant setting "Service principals can use Fabric APIs") |
| `the database collation ... is not UTF-8` | Only on a SQL Server stand-in: create its database with `fabric.stand_in_database` |
| `The data type 'nvarchar' is not supported` on a table you created yourself | Fabric has no `NVARCHAR`. Let relsynth create the table (`if_exists: replace`) or use `VARCHAR` |

### How it was tested

No Fabric emulator exists. Fabric's Warehouse speaks T-SQL over TDS, so the tests run the connector against SQL Server 2022 in a database with Fabric's default collation (`tests/emulators/fabric`, also in the integration CI job). The connector's types, loads, statistics and hash are the same T-SQL on both. `NOT ENFORCED` keys, Entra ID sign-in and `COPY INTO` exist only on Fabric; the first two have unit tests of what is sent, and none has a live test yet. Run `pytest -m fabric tests/integration` against a real warehouse (`RELSYNTH_TEST_FABRIC_URL` and `az login`) before relying on them.

### Sources

- [Data types in Fabric Data Warehouse](https://learn.microsoft.com/fabric/data-warehouse/data-types) and [T-SQL surface area](https://learn.microsoft.com/fabric/data-warehouse/tsql-surface-area).
- [Primary, foreign and unique keys in Fabric Data Warehouse](https://learn.microsoft.com/fabric/data-warehouse/table-constraints) (`NOT ENFORCED` only).
- [Collation in Fabric Data Warehouse](https://learn.microsoft.com/fabric/data-warehouse/collation).
- [Connectivity to Fabric Data Warehouse](https://learn.microsoft.com/fabric/data-warehouse/connectivity) and [Microsoft Entra authentication](https://learn.microsoft.com/fabric/data-warehouse/entra-id-authentication).
- [`COPY INTO` (Fabric)](https://learn.microsoft.com/sql/t-sql/statements/copy-into-transact-sql?view=fabric).

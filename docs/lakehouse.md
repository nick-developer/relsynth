# Lakehouse tables: Apache Iceberg and Delta Lake

relsynth reads and writes [Apache Iceberg](https://iceberg.apache.org/spec/) and [Delta Lake](https://github.com/delta-io/delta/blob/master/PROTOCOL.md) tables on a local disk, on Amazon S3 (and S3-compatible stores such as MinIO) and on Azure Data Lake Storage Gen2 / Blob Storage. A lakehouse table is a set of Parquet data files plus a transaction log that says which files make up each version (Armbrust et al., *Delta Lake*, PVLDB 2020; Armbrust et al., *Lakehouse*, CIDR 2021). relsynth reads one version of every table and writes each synthetic table as one new version.

```bash
pip install "relsynth[iceberg]"     # PyIceberg (with its SQL catalog and pyiceberg-core)
pip install "relsynth[delta]"       # delta-rs (the deltalake package)
```

## Sources and targets

| Kind | Location string | Spec section |
|---|---|---|
| Iceberg, local | `iceberg:./lake` | `{kind: iceberg, path: ./lake, schema: sales}`: a SQL catalog kept in `./lake/catalog.db`, tables in namespace `sales` (default `default`) |
| Iceberg, any catalog | – | `{kind: iceberg, schema: sales, catalog: {type: rest, uri: https://..., warehouse: ...}}`: the properties of any [PyIceberg catalog](https://py.iceberg.apache.org/configuration/) (REST, AWS Glue, Hive, SQL, ...) |
| Delta, local | `delta:./lake/sales` | `{kind: delta, path: ./lake/sales}`: table `t` is the folder `./lake/sales/t` |
| Delta on S3 | `delta:s3://bucket/sales` | `{kind: delta, path: s3://bucket/sales, storage: {AWS_REGION: eu-west-1}}` |
| Delta on ADLS Gen2 | `delta:abfss://container@account.dfs.core.windows.net/sales` | `{kind: delta, path: ..., storage: {AZURE_STORAGE_ACCOUNT_NAME: account}}` |

`storage` holds storage options without secrets: PyIceberg [FileIO properties](https://py.iceberg.apache.org/configuration/#fileio) for Iceberg (`s3.endpoint`, `s3.region`, `adls.account-name`, ...) and [object_store options](https://delta-io.github.io/delta-rs/usage/loading-table/) for Delta (`AWS_REGION`, `AWS_ENDPOINT_URL`, `AZURE_STORAGE_ACCOUNT_NAME`, ...).

## Credentials

Secrets never go into a spec. Either

* let the libraries find them the usual way: AWS (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, `AWS_PROFILE`, instance and pod roles) and Azure (`AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_CLIENT_SECRET`, managed identity), or
* name the variable that holds an option in `storage_env`:

```yaml
source:
  kind: delta
  path: s3://lake/sales
  storage: {AWS_REGION: eu-west-1}
  storage_env: {AWS_SECRET_ACCESS_KEY: SALES_LAKE_SECRET, AWS_ACCESS_KEY_ID: SALES_LAKE_KEY}
```

A variable named in `storage_env` that is not set is an error, not an empty value.

## Versions: snapshots and time travel

A source reads, per table, the version current when relsynth first opens the table. To read the same data again:

| Setting | Meaning |
|---|---|
| `snapshot: 123` or `snapshot: {Sales: 123, Customer: 7}` | Iceberg snapshot id or Delta version, for every table or per table |
| `as_of: "2026-10-01T00:00:00Z"` | The version current at that time, the same instant for every table (a consistent cut of the lake) |

`connector.versions()` (and the run bundle's load notes) report what was read. Writes add a version; earlier versions stay readable until the table's snapshots are expired or the Delta table is vacuumed. Every version relsynth writes carries `relsynth.version` and `relsynth.seed` in its commit (Iceberg snapshot summary, Delta commit info), and `connector.history(table)` lists them.

## Writing

| `if_exists` | Iceberg | Delta |
|---|---|---|
| `replace` (default) | Overwrite (new snapshot) when the columns are the same; drop and create when they differ | Overwrite with `schema_mode: overwrite` |
| `append` | Append (new snapshot) | Append |
| `truncate` | Overwrite, keeping the table's schema | Overwrite, keeping the table's schema |
| `fail` | Stop before writing when any table exists | The same |

Each table is one atomic commit; tables are written parents first. Neither format has multi-table transactions, so a reader that starts during a write can see new parents with old children; read with `as_of` set to a time after the write to get a consistent set.

**Partitioning** (`target.partition_by`, per table):

```yaml
target:
  kind: iceberg
  path: ./lake
  partition_by:
    Sales: ["month(OrderDate)", "bucket(16, CustomerKey)"]
    Orders: [StoreKey]
```

Iceberg takes the spec's transforms: a column (identity), `year`, `month`, `day`, `hour`, `bucket(n, c)`, `truncate(n, c)`. Delta partitions by column values only; derive a month column in the spec to partition by month.

**Types.** Integers are `long`, numbers `double`, text `string`, dates as stored in the synthetic frame (text dates stay text, as in the Parquet folder target), date-times `timestamp` in microseconds (the finest unit both formats keep; nanoseconds are truncated). Mixed object columns are written as text.

**Keys.** Neither format enforces keys. relsynth records the primary key, unique keys and foreign keys of the tables it writes: Iceberg table properties `relsynth.primary_key`, `relsynth.unique`, `relsynth.foreign_keys` (JSON); Delta the table description `relsynth:{...}`. `introspect` reads them back, so a synthetic lake introspects with its keys; tables written by other tools describe no keys and `introspect` identifies them from the values.

## Reading and fitting inside the lake

A table is read as the Arrow dataset of its version (its Parquet files, with partition columns), so only the columns a step needs are read. The statistics of pushed-down facts (`synthesis.pushdown`) run in an in-process DuckDB database over that dataset: the same SQL as on the database servers ([pushdown.md](pushdown.md)), streaming and out of core. First rows follow the version's file order; the sample hash is the SHA-256 one of MySQL, DuckDB files and the warehouses. `source.engine: pandas` and `source.duckdb` work as for folders.

Versions with Iceberg delete files (merge-on-read) or Delta deletion vectors, and Iceberg tables whose schema has changed since files were written, are read through PyIceberg or delta-rs (DataFusion) instead, which apply the deletes and the column mapping. That path holds the table in memory; compact the table first to keep large facts streaming.

## Checking the output against a data contract

An [ODCS v3](https://bitol-io.github.io/open-data-contract-standard/) contract states what consumers expect of the tables: columns and their types, required columns, keys, links, ranges and quality metrics. Export the contract of the real data, then check the synthetic lake against it:

```bash
relsynth ir odcs spec.yaml --out contract.yaml
relsynth ir check-contract contract.yaml --data iceberg:./synthetic-lake --out contract-check.md
```

The command exits with code 1 when a check fails. It checks tables, columns, logical types, `required`, primary keys (composite), `unique`, `logicalTypeOptions.minimum` / `maximum`, foreign keys and the library metrics `rowCount`, `nullValues` and `duplicateValues`; checks written for other engines (Great Expectations, Soda, SQL) are listed as not run. Any location works, not only lakes. From Python: `relsynth.ir.contract_check.check_location(contract, location)`.

## Checks

| What | Where | Result |
|---|---|---|
| Contract (catalog, counts, reads, samples, statistics equal to the in-memory reference, first rows, modes, versions, keys, partitioning, hostile names) | `tests/unit/test_lakehouse.py` | local disk, both formats |
| Object storage | `tests/integration/test_lakehouse_stores.py` | S3 (moto server; MinIO in `docker-compose.test.yml`) and Azure Blob / ADLS Gen2 (Azurite), both formats |
| Same data, same synthetic database | `tests/unit/test_lakehouse.py`, `roadmap/reports/T10/scripts/lake_equivalence.py` | Chinook (in memory, full-sample and sampled pushdown) and Contoso: identical cell by cell to SQLite on Iceberg and Delta |

Not yet run against a cloud account (AWS S3 with IAM roles, ADLS Gen2 with hierarchical namespace and Entra ID, AWS Glue or a REST catalog such as Polaris or Unity Catalog); those checks need the user's sandboxes. Apache Hudi is not supported.

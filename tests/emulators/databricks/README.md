# Databricks SQL stand-in (tests only)

No Databricks emulator exists, so the integration tests run the real Databricks
SQL connector and SQLAlchemy dialect against Apache Spark 3.5 with Delta Lake
3.2, through `thrift_proxy.py`. `start.sh` starts both; its header lists the
environment variables the tests read.

What the stand-in checks: the dialect's SQL (window functions, `sha2`, `conv`,
casts, `INSERT` literals, `DESCRIBE DETAIL`), Delta's types and column mapping,
and the connector's reflection, reads, writes, statistics and samples.

What it cannot check, and what therefore needs a real workspace before a
release (see `docs/warehouses.md`): Unity Catalog primary and foreign keys
(open-source Spark has none, so the connector declares none there), identity
columns, `COPY INTO` itself (the proxy turns the connector's `COPY INTO` of a staged file into `INSERT ... SELECT`) and volume uploads (`PUT`/`REMOVE`), OAuth and Entra ID
sign-in, and `current_version()`.

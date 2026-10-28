# Microsoft Fabric stand-in

No Fabric emulator exists. Fabric's Warehouse speaks T-SQL over TDS, so the
tests run the Fabric connector against SQL Server 2022 (the `mssql` service of
`docker-compose.test.yml`), in a database with Fabric's default collation,
`Latin1_General_100_BIN2_UTF8`:

```bash
docker compose -f docker-compose.test.yml up -d mssql
python -c "from relsynth.connectors.fabric import stand_in_database; \
stand_in_database('mssql+pyodbc://sa:RelSynth_Test1!@localhost:1433/master?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes', 'rs_fabric')"
export RELSYNTH_TEST_FABRIC_URL='fabric://sa:RelSynth_Test1!@localhost:1433/rs_fabric?TrustServerCertificate=yes'
pytest -m fabric tests/integration
```

The connector finds out which engine it is on from `SERVERPROPERTY('EngineEdition')`.

## What the stand-in checks

- The connector's own code: URL and driver options, the Fabric column types
  (`VARCHAR` in bytes, `DATETIME2(6)`, `FLOAT(53)`, no `NVARCHAR`), loads,
  reads, in-database statistics and the UTF-8 SHA-256 sample hash, which are
  the same T-SQL on both engines.
- Exact text statistics in a case-insensitive UTF-8 collation (a second
  database, `rs_fabric_ci`, created by the test).
- The shared contract tests (`tests/integration/test_servers.py`) and the
  Chinook and Contoso equivalence with SQLite.

## What it cannot check

- `NOT ENFORCED` keys. SQL Server has no such form, so the stand-in creates
  ordinary constraints; the Fabric DDL is checked as text in
  `tests/unit/test_fabric_connector.py`.
- Microsoft Entra ID sign-in. The token reaching the driver is checked in the
  unit tests with a fake credential.
- `COPY INTO` from ADLS Gen2 or OneLake, and Fabric's capacity billing.
- Fabric-only limits on T-SQL. The connector uses only statements documented
  for the Fabric Warehouse (see docs/warehouses.md), but a run against a real
  warehouse (`RELSYNTH_TEST_FABRIC_URL=fabric://<id>.datawarehouse.fabric.microsoft.com/<warehouse>`
  with `az login`) is the final check.

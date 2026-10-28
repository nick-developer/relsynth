# Snowflake stand-in

The tests run the real Snowflake connector and dialect against
[fakesnow](https://github.com/tekumara/fakesnow) in server mode: Snowflake's
REST protocol answered by DuckDB. `serve.py` starts it with one fix (NULL
`TIMESTAMP_NTZ` values reached the connector as 1970-01-01).

```bash
pip install "snowflake-connector-python[pandas]<5" "fakesnow==0.12.1"   # fakesnow does not speak connector 5's protocol yet
python tests/emulators/snowflake/serve.py 8085 &
export RELSYNTH_TEST_SNOWFLAKE_URL='snowflake://rs:rs-stand-in@fake/RSDB/PUBLIC?warehouse=WH'
export RELSYNTH_TEST_SNOWFLAKE_CONNECT_ARGS='{"host": "localhost", "port": 8085, "protocol": "http"}'
python -c "import json, os, sqlalchemy as sa; sa.create_engine('snowflake://rs:x@fake/RSDB', connect_args=json.loads(os.environ['RELSYNTH_TEST_SNOWFLAKE_CONNECT_ARGS'])).connect().exec_driver_sql('CREATE DATABASE IF NOT EXISTS RSDB')"
pytest -m snowflake tests/integration
```

The connector tells the stand-in from Snowflake by `CURRENT_ACCOUNT()`, which
fakesnow does not have (`SnowflakeConnector.platform`).

## What the stand-in checks

- The connector's SQL: created types, multi-row inserts, reads, the
  in-database statistics and the SHA-256 sample hash.
- Upper-case and mixed-case names through the dialect.
- The shared contract tests and the Chinook and Contoso equivalence with SQLite.

## What it cannot check

- Foreign keys. DuckDB enforces them (Snowflake never does) and fakesnow reads
  them back wrongly, so no foreign keys are declared on the stand-in.
- `PUT` to a stage and `COPY INTO` (`bulk: stage`).
- Sign-in methods, the session `QUERY_TAG` and `INFORMATION_SCHEMA.TABLES.BYTES`
  (always 0 on the stand-in).

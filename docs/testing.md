# Testing

The test suite has four layers, from fast checks of single functions to an hour-long statistical acceptance suite.

```bash
pip install -e ".[dev]"
pytest                                   # unit, end-to-end and security tests: SQLite, CSV, Parquet (a few minutes)
pytest -m "not slow"                     # skip the longer end-to-end runs
pytest tests/integration                 # live PostgreSQL, SQL Server, Oracle, MySQL and MariaDB (skipped unless configured)
pytest -m science                        # the statistical check suite C00–C78 (about an hour on 4 cores)
RELSYNTH_SCIENCE_SHARD=2/4 pytest -m science   # one of four shards of about equal run time (as CI runs it)
pytest tests/unit/core                   # the fast core-engine tests (about 15 s; mutation testing uses them)
pytest tests/unit --cov=relsynth --cov-branch --cov-report=term
```

CI runs every layer on every pull request; [development/ci.md](development/ci.md) lists the jobs.

## Layout

| Folder | What it tests | Needs |
|---|---|---|
| `tests/unit/core/` | The core engine against closed forms, brute force and known answers: the count-model choice (`counts.py`), posterior draws and streams (`posterior.py`), the copula's vine parametrisation, truncated normals, convergence diagnostics and recovery of a known correlation (`copula.py`), spike-and-slab draws against their exact posteriors, collapsed marginals, the grain test and a star schema end to end (`relational.py`), metric primitives and the scorecard on identical and deliberately damaged data (`evaluate.py`) | |
| `tests/unit/test_spec.py` | Loading, validation, overrides (`--set`), YAML safety, saving with masked passwords | |
| `tests/unit/test_connectors.py` | Location strings, file and SQLite catalogs, reads, streamed samples, every write mode, cloning DDL, hostile identifiers, path confinement | |
| `tests/unit/test_sqlfuncs_stats.py` | Acklam's bound (Prop. 102); hash uniformity (Prop. 103); SQL statistics equal the in-memory ones (Prop. 100); key filters on every path; first rows; seeded samples | |
| `tests/unit/test_duckfiles.py` | DuckDB over CSV and Parquet folders: statistics equal the in-memory ones, first rows in file order, CSV typing as the pandas reader, seeded samples that survive a reconnect, full-sample pushdown equals in-memory fitting, engine settings, hostile file and folder names | |
| `tests/unit/test_pushdown.py` | Planner decisions and reasons (Prop. 105); a full-sample pushdown equals in-memory fitting for every fact shape (Prop. 101); sampled fits keep quality and use exact counts (Prop. 104) | |
| `tests/unit/test_api.py` | The Python API end to end: introspect, fit, sample (datasets, scale), evaluate, write, run with quality gates, doctor, export, clone, profile, ping, typed errors | |
| `tests/unit/test_security.py` | URL masking, scrubbed messages, masked bundles and specs, the expression sandbox, regex vetting, path confinement, privacy defaults, no copied rows | |
| `tests/unit/test_ir.py` | The IR: 1.0 round trips of the golden and fake specs (Prop. 106), identical synthetic data from an IR document, the five examples against the model and the JSON Schema, refusals, schema and reference drift, privacy paths of the four real databases, ownership and unit removal on S9 (Props. 107–108), ODCS and OpenLineage against their official schemas, migration and the `ir` commands | `relsynth[ir]` for the schema checks |
| `tests/e2e/test_cli.py` | Every CLI command as a CI job runs it: JSON output, exit codes 0–4, `--set`, reports, bundles | |
| `tests/integration/test_servers.py` | On each live server: write and read, catalog, introspect, pushdown fit, identical output to SQLite, every write mode, cloning both ways, statistics parity, seeded samples, hostile identifiers, masked errors | `RELSYNTH_TEST_*_URL` |
| `tests/integration/test_mysql.py` | MySQL and MariaDB specifics: exact text statistics under case-insensitive collations, the SHA-256 sample hash against Python, DOUBLE / DATETIME(6) / 4-byte UTF-8 round trips, TEXT keys and wide tables within MySQL's index and row limits, a MariaDB server behind a `mysql://` URL | `RELSYNTH_TEST_MYSQL_URL`, `RELSYNTH_TEST_MARIADB_URL` |
| `tests/unit/test_warehouses.py` | Cloud warehouses without a warehouse: credentials from the environment (never the spec), each sign-in method's driver arguments, compiled SQL (casts, the SHA-256 sample hash), created types, declared keys per platform, literals, cost guards and query tags | |
| `tests/unit/test_quality.py` | The 2.0 quality report: metric values with known answers, the tie-corrected precision/recall curve (Prop. 110), cluster splits, prediction bounds, the area rule (Prop. 111), schema validity, the 1.0 scorecard carried over unchanged | |
| `tests/science/test_quality_calibration.py` | Checks C79–C82: false-failure rate of the calibrated limits under the null at equal and unequal sizes, power against four planted defects, the scorecard carried over (Props. 109–113) | about 30 minutes on 4 cores |
| `tests/science/test_legacy_checks.py` | The check suite C00–C78. Each check tests a numbered proposition on fake data with known ground truth: posterior recovery, calibration, identification, rules, privacy, reproducibility | about an hour |

Fake databases with known generating models are in `tests/fakedata/scenarios.py` (S1–S11), and `tests/conftest.py` caches them per session. `tests/fakedata/quality_null.py` draws a shop database (customers, orders, order lines, products) from a law whose parameters do not depend on the seed, so two seeds give two draws of one law, and derives databases with one planted defect each.

| Scenario | Shape |
|---|---|
| S1 | One mixed table |
| S2 | A star schema |
| S3 | A snowflake |
| S4 | A tree |
| S5 | Temporal facts |
| S6 | Text |
| S7 | An enterprise schema |
| S8 | A reporting model with nested facts |
| S9 | A small shop |
| S10 | An order warehouse with headers and running balances |
| S11 | A retail warehouse with views |

## Live database servers

```bash
docker compose -f docker-compose.test.yml up -d --wait
export RELSYNTH_TEST_PG_URL=postgresql://relsynth:RelSynthTest1@localhost:5432/relsynth
export RELSYNTH_TEST_MSSQL_URL="mssql://sa:RelSynth_Test1!@localhost:1433/master?TrustServerCertificate=yes&driver=ODBC+Driver+18+for+SQL+Server"
export RELSYNTH_TEST_ORACLE_URL="oracle://relsynth:RelSynthTest1@localhost:1521/?service_name=FREEPDB1"
export RELSYNTH_TEST_MYSQL_URL=mysql://root:RelSynthTest1@localhost:3306/relsynth
export RELSYNTH_TEST_MARIADB_URL=mariadb://root:RelSynthTest1@localhost:3307/relsynth
pytest tests/integration -v
```

- Each test works in a fresh schema (`rs_<random>`) and drops it afterwards.
- On Oracle, a schema is a user, so tests use the connecting user's schema and drop the tables they created.
- On MySQL and MariaDB, a schema is a database, so the test user must be allowed to create and drop databases (the compose file uses `root`).
- Databricks runs against a local stand-in (Spark 3.5 with Delta Lake behind a protocol proxy): `tests/emulators/databricks/start.sh`, then the two variables its header lists, then `pytest -m databricks tests/integration`. A real workspace works the same way with its own URL and credentials.
- Microsoft Fabric runs against SQL Server 2022 in a UTF-8 database (`tests/emulators/fabric/README.md`): set `RELSYNTH_TEST_FABRIC_URL`, then `pytest -m fabric tests/integration`. The integration CI job does this on its SQL Server service.
- SQL Server tests need the Microsoft ODBC Driver 18, or `relsynth[mssql-pymssql]` with a `mssql+pymssql://` URL.

## The statistical check suite

`legacy/checks/run_checks.py` holds 79 checks (C00–C78) with more than 500 assertions. They were written with the engine (versions 0.2–0.9), and each states the proposition it tests. `pytest -m science` runs them once, in a mirror of the layout they expect, and reports each as one test. To run single checks from a checkout laid out as in layer 2:

```bash
python legacy/checks/run_checks.py C05 C11
```

C76 also reads the layer 2 Chinook results (`RELSYNTH_LAYER2_ROOT`) and the Chinook database (`RELSYNTH_CHINOOK_DB`, default `bench/.cache/downloads/Chinook_Sqlite.sqlite`, which `python -m bench fetch --datasets chinook` downloads), and is skipped without them. C00 re-runs the layer 1 algorithm checks, and two of its 14 need the `datasketches` package.

## Mutation testing

Mutation testing changes the core engine in small ways (an operator, a constant, a boundary) and checks that a test fails. It measures whether the tests check behaviour, which coverage cannot (a line can be executed and its result never checked).

- Tool: [mutmut](https://github.com/boxed/mutmut) 3, configured under `[tool.mutmut]` in `pyproject.toml`: the five core modules `counts.py`, `posterior.py`, `copula.py`, `relational.py` and `evaluate.py` are mutated, only on lines that `tests/unit/core` executes, and each mutant runs only the tests that reach it.
- The modules have about 7,800 mutants, eight hours on four cores. `tools/mutation_sample.py` tests a seeded random sample per module and reports the score killed / (killed + survived) with a Wilson 95% interval. Mutants on lines that no fast test reaches count as survived.
- The weekly workflow `mutation.yml` runs the sample of 150 per module and publishes the table and the surviving mutants. The baseline is in [development/mutation-baseline.md](development/mutation-baseline.md).
- The score is a trend to watch, not a gate. As at Google (Petrović and Ivanković, ICSE-SEIP 2018), the useful output is the list of surviving mutants, which a reviewer reads as tests to write; many survivors are equivalent mutants (a change that cannot alter behaviour).

```bash
mutmut run "relsynth.counts*"        # every mutant of one module
mutmut results; mutmut show <name>   # the survivors, and the diff of one
python tools/mutation_sample.py --per-module 150 --seed 1
```

## The quality report on benchmark databases

`benchmarks/syntherela/run_syntherela.py` runs relsynth on the SyntheRela benchmark databases (downloaded outside the repository; never commit them or synthetic data derived from them) and writes the SyntheRela-compatible numbers of the 2.0 report. Results and the comparison with published methods are in [benchmarks/syntherela/README.md](../benchmarks/syntherela/README.md). Needs `pip install relsynth[benchmark]` for the XGBoost detector.

## Quality checks in CI

```bash
ruff check src tests              # style, bugs and the S (security) rules
bandit -q -c pyproject.toml -r src
pip-audit                         # known vulnerabilities in installed dependencies
python -m build && twine check dist/*
```

The project's CI ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs lint and security scans, the type ratchet, unit tests on Python 3.10–3.13 (Linux) and 3.12 on Windows and macOS, the end-to-end tests, the five live servers, the statistical suite in four shards, and a wheel build with SBOM and signature, on every pull request. See [development/ci.md](development/ci.md).

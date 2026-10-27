# Testing

The test suite has four layers, from fast checks of single functions to an hour-long statistical acceptance suite.

```bash
pip install -e ".[dev]"
pytest                                   # unit, end-to-end and security tests: SQLite, CSV, Parquet (a few minutes)
pytest -m "not slow"                     # skip the longer end-to-end runs
pytest tests/integration                 # live PostgreSQL, SQL Server and Oracle (skipped unless configured)
pytest -m science                        # the statistical check suite C00–C78 (about an hour on 4 cores)
pytest --cov=relsynth --cov-report=term-missing
```

## Layout

| Folder | What it tests | Needs |
|---|---|---|
| `tests/unit/test_spec.py` | Loading, validation, overrides (`--set`), YAML safety, saving with masked passwords | |
| `tests/unit/test_connectors.py` | Location strings, file and SQLite catalogs, reads, streamed samples, every write mode, cloning DDL, hostile identifiers, path confinement | |
| `tests/unit/test_sqlfuncs_stats.py` | Acklam's bound (Prop. 102); hash uniformity (Prop. 103); SQL statistics equal the in-memory ones (Prop. 100); key filters on every path; first rows; seeded samples | |
| `tests/unit/test_pushdown.py` | Planner decisions and reasons (Prop. 105); a full-sample pushdown equals in-memory fitting for every fact shape (Prop. 101); sampled fits keep quality and use exact counts (Prop. 104) | |
| `tests/unit/test_api.py` | The Python API end to end: introspect, fit, sample (datasets, scale), evaluate, write, run with quality gates, doctor, export, clone, profile, ping, typed errors | |
| `tests/unit/test_security.py` | URL masking, scrubbed messages, masked bundles and specs, the expression sandbox, regex vetting, path confinement, privacy defaults, no copied rows | |
| `tests/e2e/test_cli.py` | Every CLI command as a CI job runs it: JSON output, exit codes 0–4, `--set`, reports, bundles | |
| `tests/integration/test_servers.py` | On each live server: write and read, catalog, introspect, pushdown fit, identical output to SQLite, every write mode, cloning both ways, statistics parity, seeded samples, hostile identifiers, masked errors | `RELSYNTH_TEST_*_URL` |
| `tests/unit/test_quality.py` | The 2.0 quality report: metric values with known answers, the tie-corrected precision/recall curve (Prop. 107), cluster splits, prediction bounds, the area rule (Prop. 108), schema validity, the 1.0 scorecard carried over unchanged | |
| `tests/science/test_quality_calibration.py` | Checks C79–C82: false-failure rate of the calibrated limits under the null at equal and unequal sizes, power against four planted defects, the scorecard carried over (Props. 106–110) | about 30 minutes on 4 cores |
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
pytest tests/integration -v
```

- Each test works in a fresh schema (`rs_<random>`) and drops it afterwards.
- On Oracle, a schema is a user, so tests use the connecting user's schema and drop the tables they created.
- SQL Server tests need the Microsoft ODBC Driver 18, or `relsynth[mssql-pymssql]` with a `mssql+pymssql://` URL.

## The statistical check suite

`legacy/checks/run_checks.py` holds 79 checks (C00–C78) with more than 500 assertions. They were written with the engine (versions 0.2–0.9), and each states the proposition it tests. `pytest -m science` runs them once, in a mirror of the layout they expect, and reports each as one test. To run single checks from a checkout laid out as in layer 2:

```bash
python legacy/checks/run_checks.py C05 C11
```

C76 also reads the real Chinook database from the layer 2 data folder (`RELSYNTH_LAYER2_ROOT`) and is skipped without it. C00 re-runs the layer 1 algorithm checks, and two of its 14 need the `datasketches` package.

## The quality report on benchmark databases

`benchmarks/syntherela/run_syntherela.py` runs relsynth on the SyntheRela benchmark databases (downloaded outside the repository; never commit them or synthetic data derived from them) and writes the SyntheRela-compatible numbers of the 2.0 report. Results and the comparison with published methods are in [benchmarks/syntherela/README.md](../benchmarks/syntherela/README.md). Needs `pip install relsynth[benchmark]` for the XGBoost detector.

## Quality checks in CI

```bash
ruff check src tests              # style, bugs and the S (security) rules
bandit -q -c pyproject.toml -r src
pip-audit                         # known vulnerabilities in installed dependencies
python -m build && twine check dist/*
```

The project's CI ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs the following jobs:
- lint;
- unit tests on Python 3.10–3.13 (Linux), plus Python 3.12 on Windows and macOS;
- integration tests against the three server images;
- a wheel build with a clean-environment smoke test;
- the statistical suite weekly, or on demand.

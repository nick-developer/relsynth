# Changelog

All notable user-visible changes are recorded here, following [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/). relsynth uses [Semantic Versioning](https://semver.org/); see [CONTRIBUTING.md](CONTRIBUTING.md) for what counts as the public API.

## Unreleased

### Added

- **MySQL 8 and MariaDB connector** (`mysql://`, `mariadb://`, `relsynth[mysql]`): read, write in every mode, `clone-schema`, and fitting inside the database (GROUP BY statistics, window-function rank sums, a seeded SHA-256 hash sample). Text is grouped and compared in a binary collation, so statistics stay exact on case- and accent-insensitive columns. Created columns keep the data: `DOUBLE`, `DATETIME(6)`, `utf8mb4` text with a binary no-pad collation, key columns within MySQL's index limit and wide tables within its row limit. Tested live on MySQL 8.4 and MariaDB 11.4.
- **Databricks SQL connector** (`databricks://`, `relsynth[databricks]`): read, write in every mode, `clone-schema` and fitting inside the warehouse. Sign-in without secrets in the spec (`auth`: token, OAuth M2M and U2M, Entra ID service principal, or the Databricks SDK's credential chain). Created columns keep the data (`DOUBLE`, `STRING`, `TIMESTAMP_NTZ`, Delta column mapping for unusual names); keys are declared where Unity Catalog allows them. Loads by multi-row `INSERT` or staged Parquet and `COPY INTO`. See `docs/warehouses.md`.
- **Microsoft Fabric connector** (`fabric://`, `relsynth[fabric]`): the Fabric Warehouse and Lakehouse SQL endpoints over the ODBC Driver 18. Sign-in with Microsoft Entra ID (`auth`: the Azure credential chain, service principal, Azure CLI, managed identity or browser). Created columns use the types Fabric has (`VARCHAR` sized in UTF-8 bytes, `DATETIME2(6)`, `FLOAT(53)`), keys are declared `NOT ENFORCED`, text statistics stay exact under case-insensitive collations, and the sample hash is the UTF-8 SHA-256 one shared with MySQL, DuckDB and Databricks. Loads by ODBC parameter arrays or Parquet staged in ADLS Gen2 / OneLake and `COPY INTO`.
- **Snowflake connector** (`snowflake://`, `relsynth[snowflake]`): read, write in every mode, `clone-schema` and fitting inside the warehouse. Sign-in with key pairs, programmatic access tokens, OAuth, SSO, workload identity or a `connections.toml` entry. Every session carries the `QUERY_TAG`. Keys are declared (informational), date-times are `TIMESTAMP_NTZ`, and the sample hash is the same SHA-256 one. Loads by multi-row `INSERT` or Parquet `PUT` to a temporary stage and `COPY INTO`.
- **Google BigQuery connector** (`bigquery://`, `relsynth[bigquery]`): read, write and fitting inside the warehouse, signed in with Application Default Credentials, a service-account key, workload identity federation or impersonation. Every job is labelled `relsynth=<tag>` and limited by `maximum_bytes_billed`; reads, samples and statistics are dry-run first. Loads use free load jobs from Parquet. Not yet run against BigQuery itself.
- **Cost guards for cloud warehouses** (`guards`): a query tag on every statement, `max_rows` and `max_bytes` limits checked before a read, and the bytes each table costs in `relsynth plan`.
- **DuckDB for CSV and Parquet folders.** The statistics, first rows and samples of pushed-down facts in a folder source are computed by an in-process DuckDB database running the servers' SQL over views of the files, instead of in pandas. Statistics are exact (normal scores use scipy's quantile inside DuckDB) and CSV typing follows the pandas reader. `source.engine: pandas` keeps the 1.0 path, and `source.duckdb` sets the memory limit, threads and spill folder. `duckdb` and `duckdb-engine` are now dependencies.
- **Connector guide** (`docs/connector-sdk.md`): the contract, hooks and tests a new connector needs, with notes for the cloud warehouses.
- **Profile of 1.0** (`docs/profiling.md`): time, memory and I/O by stage on Contoso and AdventureWorks.
- **IR 2.0** (`relsynth.ir`): one versioned, modality-neutral document of a data estate: entities with a shape (record, document, node, edge, event) and role; nested objects, arrays with length laws and polymorphic variants with a discriminator; composite, surrogate and natural keys; references, hierarchies, one-to-one and derived links, including from nested paths; graph edges with properties; valid time, event time and SCD2 versioning; containers, partitions with Iceberg-style transforms and sort orders; hard and soft rules; privacy roles, public entities and the privacy unit. Python object model, generated JSON Schema (draft 2020-12) and [reference](docs/ir-reference.md); [guide](docs/ir-guide.md); [ADR 0001](docs/adr/0001-canonical-ir.md); five examples in `examples/ir/` (MongoDB, DynamoDB single table, property graph, Iceberg, event stream).
- **Loss-free 1.0 conversion.** Every command and `Spec` read IR documents as well as 1.0 specs; an IR document lowered for the engine gives the same synthetic data as the 1.0 spec it came from (Proposition 106; all eleven golden specs of the four real databases give byte-identical output).
- **`relsynth ir`** commands: `migrate` (with proof of identity), `validate`, `convert --to ir|v1 [--strict]`, `privacy`, `odcs`, `from-odcs`, `openlineage`, `schema`; `introspect --format ir`.
- **Privacy unit tracing**: reference paths from every entity to the unit, owned/context/unrelated status, cross-unit links, and per-record owners on data (Propositions 107–108).
- **Standards**: ODCS v3.2.0 export and v3.0–3.2 import (exact round trip through `customProperties`); OpenLineage Schema, DatasetType, Documentation, Tags and ColumnLineage facets.
- Extra `relsynth[ir]` (jsonschema) for the JSON Schema check.
- **Generator plug-ins** (`relsynth.generators`). A table's attribute columns now come from a generator chosen in the spec. You can set it per table (`tables.<t>.generator`: a name or `{name, version, options}`) or as a default (`synthesis.generator`). The engine keeps keys, links, latent segments, counts and rules, whatever the generator (Props. 700–706).
  - Built-in generators: `copula` (the default; byte-identical to 1.0), `independent` (a latent class model) and `cart` (sequential CART, synthpop style).
  - Third-party generators install as packages through the `relsynth.generators` entry-point group. `relsynth generators list|show` lists them with their capabilities.
  - See `docs/generators.md` and ADR 0002.
- **Generator choice in the IR.** `entities.<name>.synthesis.generator` holds a table's generator (`{name, version, options}`), so 1.0 specs that choose generators still convert without loss. A spec read from an IR document checks generators against its IR version.
- `relsynth doctor --apply` and `relsynth tune` write an IR document when they are given one, merging their changes into it so its annotations are kept.
- `relsynth introspect --format` takes `auto`, the new default: it writes an IR document from release 2.0 and a 1.0 spec before.
- **Compute presets** (`compute.yaml`: `cpu`, `gpu-auto`, `gpu-single`, `gpu-multi`). Set them in `synthesis.compute` with expert overrides (devices, backend, precision, batch size, memory limit, threads, fallback).
  - GPU use is opt-in, the default engine is CPU-only, and every fallback to the CPU is explicit and logged.
  - `relsynth compute` shows the GPUs found and the device each kind of generator would get.
  - See `docs/compute.md`.
- **Quality report 2.0** (`relsynth.assess_quality`, `relsynth evaluate --quality FILE`): a versioned, machine-readable report (JSON Schema `quality-report-2.0`) with every check's metric, scope, value, limit, tolerance, null calibration, level and confidence, grouped in a quality vector (fidelity, utility, integrity) with a calibrated pass rule per area and no opaque overall score. The 1.0 scorecard is carried over unchanged.
- **Opt-in 2.0 quality gate**: `relsynth evaluate --quality FILE --quality-gate` exits 1 when an area of the quality vector fails. The default gate stays `--fail-under` until 3.0. Utility targets of the golden databases are declared in `benchmarks/targets.yaml`.
- **New metrics**: Wasserstein-1 on the raw and rank-normal scales, Jensen–Shannon distance, 2- and 3-way marginal TVD, alpha-precision and beta-recall, TSTR/TRTS/TRTR, fan-out laws (W1, childless share, dispersion) and two-hop fan-out, k-hop column-pair trends, aggregation detection C2ST-Agg (logistic, or XGBoost with `relsynth[benchmark]`), and a multi-table utility task. Definitions in `docs/metrics.md`, generated from `quality/registry.py`.
- **Calibrated limits**: each new check's limit is a prediction bound from real-versus-real splits that keep families together, rescaled to the compared sizes (Propositions 109–112, checks C79–C82). A synthetic database drawn at another size is compared on per-parent counts divided by the expected change of rows per parent under the engine's scale rule (`scale` setting, defaulting to the spec's `synthesis.scale` in `assess_quality`; Proposition 113).
- **Privacy link** (RFC T06-02): an optional `privacy` member references the privacy report of roadmap thread T06 by path, sha256 and verdicts; privacy is never folded into the quality vector.
- **`relsynth privacy`** (`relsynth.privacy`): measures privacy risk per person (the privacy unit and the rows it owns) with record isolation, linkage and inference attacks, attribute inference against population data, DOMIAS and shadow-model membership inference, and writes a report for a data protection officer per release context (internal, partner, public; Propositions 600–606, checks C600–C606). Contexts outside the organisation are blocked without differential privacy. Defaults: a *review* needs a named sign-off (`--signed-off-by`) before a gate passes it; internal releases gate on the pooled attack on the most exposed people, partner and public on the single most exposed person with 128 test shadows; a privacy unit of fewer than 1000 people warns and fails every external context without DP. The scorecard's distance-to-closest-record guard stays for continuity but is no longer treated as a privacy check.
- **SyntheRela benchmark** script and results (`benchmarks/syntherela/`), and precise definitions of the roadmap's acceptance metrics (`docs/acceptance-metrics.md`).

### Deprecation

- 2.x reads 1.0 specs and IR 2.x documents alike. Reading a 1.0 spec warns from 3.0 at the earliest and stops no earlier than 4.0 and 24 months after 2.0. `relsynth ir migrate` stays in every release.

## 1.0.1 (2026-10-28)

Synthetic draws for the same seed are unchanged from 1.0.0.

### Changed

- relsynth is open source under the Apache License 2.0 (previously all rights reserved). The repository has LICENSE, NOTICE, a contributor guide with DCO sign-off, a code of conduct, a security policy with private vulnerability reporting, and a dataset register (`docs/datasets.md`).

### Fixed

- `relsynth run --bundle` (and `api.run(..., bundle=...)`) no longer fails after writing and scoring when a numeric column of a nullable pandas dtype is constant: its skew in the bundle's profile is recorded as null (seen on AdventureWorks DW).

## 1.0.0 

The first packaged release: a pip-installable library and command-line tool built on the v0.2–v0.9 engine.

### Added

- **Package and CLI.** `pip install relsynth` with extras per database. Commands: `introspect`, `check`/`validate`, `plan`, `run`, `evaluate`, `verify`, `doctor`, `tune`, `model`, `export-model`, `clone-schema`, `profile`, `ping`, `version`. Every command has `--json` output and documented exit codes (0 ok, 1 quality gate, 2 spec, 3 I/O, 4 engine, 130 interrupted).
- **Python API.** `introspect`, `fit`, `sample`, `evaluate`, `run`, `verify`, `doctor`, `tune`, `export_model`, `clone_schema`, `profile`, `ping`, `connect`, with the classes `Spec`, `FittedModel`, `SyntheticData`, `EvaluationReport`, `RunResult`, `VerifyReport` and `Diagnosis`, and typed errors.
- **Connectors.** Oracle, SQL Server, PostgreSQL and SQLite, read and write, plus CSV and Parquet folders. Each has four write modes (replace, append, truncate, fail) and three target structures (clone, infer, existing). Bulk paths: COPY on PostgreSQL, fast executemany on SQL Server, array binds on Oracle. `clone-schema` copies types, keys, unique constraints, indexes and identity columns.
- **Fitting inside the database (pushdown).** Large facts are fitted from exact `GROUP BY` statistics, rank-score sums computed with window functions, first rows per document key, and a seeded Bernoulli hash sample. Propositions 100–105 prove sufficiency, agreement with in-memory fitting, Acklam's quantile bound, the hash sample, the exact sample posterior, and eligibility. `relsynth plan` explains the decision for each table.
- **Canonical row order.** Rows are put in key order (or hash order) on load, so the same seed gives the same synthetic database on every backend.
- **`verify`.** Model checks on real data without ground truth (R1–R7), from the CLI and from Python.
- **Credentials from the environment.** `--source-url-env` and `--target-url-env` alone name a location; `password_env`/`url_env` in specs. Passwords are masked in every message, log, saved spec, bundle and exported model.
- **Links that determine links.** `introspect` finds a link that fixes another link of the same row (a track's album fixes its genre). The second is then read from the observed pairs instead of being drawn independently.
- **Documentation.** README, user guide with a worked example, CLI and API references generated from the code, connectors, pushdown, CI/CD, security, theory and testing guides, and the proofs (105 propositions) as PDF and LaTeX.
- **Tests.** Unit, end-to-end CLI, security and live-server tests (PostgreSQL 16, SQL Server 2022, Oracle Database 26ai Free), and the statistical check suite C00–C78 as `pytest -m science`. CI files for GitHub Actions, GitLab CI, Azure DevOps and Jenkins; a Dockerfile with every driver.

### Changed

- The fit visits rows in canonical order. In-memory results differ from v0.9 by a permutation of the input rows; the quality figures on the four real databases are unchanged within noise.
- Order headers split from a fact are kept in key order on every path.
- An entity without its own scale factor follows the first scaled parent it links to: `scale: {tables: {customer: 10}}` now also gives ten times the invoices when invoices are an entity below customers. The scorecard expects the same sizes. The v0.9 check of a parent scaled to zero rows (C75, `t_empty_parent`) now gives the child its own factor to keep testing that case.
- `if_exists: fail` raises `FileExistsError` (exit code 3).
- SQL Server text columns are created as `NVARCHAR(max)`.

### Fixed

- A dimension anchored on a column that is null for part of the rows (cities of countries without states) no longer leaves those rows without a city.
- `export-model` writes the bundle's `spec.yaml` also when the spec was given as a dict, and `run_model.py --out` is relative to the current folder.
- `doctor --json` reported a missing field; `doctor` no longer offers a stratification patch that cannot change the result.
- Specs that are not valid YAML are spec errors (exit 2), not engine errors.

## 0.9 (2026-10-19)

Whole-engine review: pooling and multiplicity control for many small units across the engine, scorecard and identification; learned structural weights at the group-column level.

## 0.8

Keys and links identified from the data (names optional), exact kernels (1.9× faster on Contoso), `export-model` with a verified Bayesian model as PDF, Markdown, JSON and Python.

## 0.7

Posterior predictive synthesis; type 2 slowly changing dimensions, unknown members, bridges, group effects, link affinity and participation; the model report; the strict YAML specification and its reference.

## 0.6

Contoso (SQLite and Parquet) from unedited drafts; scale 0.5 and 2.

## 0.5

AdventureWorks DW (27 tables, 1.05 million rows) from an unedited draft.

## 0.4 

Chinook from an unedited draft.

## 0.3 

First real database (EBA COREP reporting model); parameter-expanded copula.

## 0.2 

Prototype: one YAML spec drafted by `introspect`, Bayesian Gaussian copula per table, latent segments across tables.

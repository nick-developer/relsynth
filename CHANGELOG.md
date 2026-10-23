# Changelog

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

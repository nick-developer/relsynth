# relsynth

**Bayesian synthetic data for whole relational databases.** relsynth reads a database (Oracle, SQL Server, PostgreSQL, MySQL, MariaDB, SQLite), a cloud warehouse (Databricks SQL, Microsoft Fabric, Snowflake, BigQuery), a lakehouse table (Apache Iceberg, Delta Lake) or a folder of CSV or Parquet files. It identifies the tables, keys, links and business rules, fits one Bayesian model of the whole schema, and writes a synthetic database with the same table structure. Every primary and foreign key resolves, every rule holds, and the statistics match the source. Each statement the engine relies on is a numbered proposition with a proof ([docs/proofs/relsynth-proofs.pdf](docs/proofs/relsynth-proofs.pdf)) and a test that checks it.

Release 2.0 adds an optional **differential privacy mode** with a per-person guarantee, **privacy attacks** and a report per release context, **governance** (signed manifests, a privacy budget ledger, policy and approvals), **time series and event sequences**, **partitioned generation** on many processes or Spark, **generator plug-ins**, a canonical **IR 2.0** document for the data estate, adjustable **engine settings** with **autotune**, and a **service** with an API, a web UI and single sign-on for running relsynth for many teams. See the [changelog](CHANGELOG.md).

```bash
pip install "relsynth[postgres]"
relsynth introspect --source "postgresql://analyst@db/sales" --source-password-env PGPASSWORD --out spec.yaml
relsynth run spec.yaml --target "sqlite:synthetic.db" --fail-under 0.98
```

| You are | Start with |
|---|---|
| A business user who wants test data or a shareable copy of a database | [Quick start (command line)](#quick-start-command-line), then the [user guide](docs/user-guide.md) |
| A data scientist or engineer who wants it in Python | [Quick start (Python)](#quick-start-python), then the [API reference](docs/api.md) |
| Setting up a CI/CD job | [CI/CD](#cicd) and [docs/cicd.md](docs/cicd.md) |
| Running relsynth as a service for many teams (SSO, approvals, Kubernetes, air-gapped) | [docs/service/](docs/service/README.md) |
| A privacy officer or data protection officer | [Privacy and differential privacy](#privacy-differential-privacy-and-governance), [docs/privacy.md](docs/privacy.md), [docs/dp.md](docs/dp.md), [governance](docs/governance/guide.md) |
| A database administrator | [Connectors](docs/connectors.md), [security](docs/security.md), [large tables](docs/pushdown.md) |
| Reviewing the method | [Theory](#theory-propositions-and-proofs), [docs/theory.md](docs/theory.md) and the [proofs](docs/proofs/relsynth-proofs.pdf) |

---

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start (command line)](#quick-start-command-line)
- [Quick start (Python)](#quick-start-python)
- [How it works](#how-it-works)
- [Sources and targets](#sources-and-targets)
- [Command line](#command-line)
- [Python API](#python-api)
- [CI/CD](#cicd)
- [Tables larger than memory](#tables-larger-than-memory)
- [Quality: the scorecard, the doctor and model checks](#quality-the-scorecard-the-doctor-and-model-checks)
- [Privacy, differential privacy and governance](#privacy-differential-privacy-and-governance)
- [Time series, generators, settings and scale](#time-series-generators-settings-and-scale)
- [Security](#security)
- [Performance](#performance)
- [Validation](#validation)
- [Theory: propositions and proofs](#theory-propositions-and-proofs)
- [Testing the package](#testing-the-package)
- [Limitations](#limitations)
- [Documentation map](#documentation-map)

---

## What it does

- **Whole databases.** relsynth handles star and snowflake schemas, operational schemas, self-referencing hierarchies (an employee's manager, a category tree), nested facts, order headers with lines, bridges, type 2 slowly changing dimensions, unknown members, views and reference tables, time-stamped facts and free text.
- **Structure first.** Synthetic tables have the source's columns and types. Primary keys are unique, foreign keys resolve, hierarchies have no cycles, and composite and unique keys hold.
- **Business rules.** Rules such as `amount = unit_price * quantity`, `ship_date >= order_date`, lookups, running balances and aggregates (an order total equal to its lines) hold on every row. You can declare rules, or relsynth can find them in the data.
- **Statistics.** Each table's marginals and dependence come from a Bayesian Gaussian copula over mixed types. Links between tables come from latent segments, so customers who buy more of one product kind keep doing so. Counts, participation and multiplicities follow fitted laws.
- **Bayesian.** Each synthetic dataset is one draw from the posterior predictive distribution. Several datasets show the model's own uncertainty, and `posterior.mode: mean` gives stable test data.
- **Scale.** You set the size per table: `--scale 0.1` for a small test database, or `--scale 10` for load tests. Facts follow the tables they hang from by the fitted rates.
- **Privacy.** No row is copied. Distance-to-closest-record checks compare synthetic rows with real ones, against the real data's own spacing. Text cannot reproduce rare source content, and memorisation guards are on by default. `relsynth privacy` attacks a release per person (membership, attribute inference, singling out, linkability) and writes a report per release context.
- **Differential privacy (optional).** `relsynth run --epsilon 1` gives a user-level (ε, δ) guarantee for each person of the privacy unit and all their rows. The organisation sets ε; there is no default. Releases outside the organisation are refused without it.
- **Governance.** Signed, hash-chained manifests of every run, a privacy budget ledger per domain, an organisation policy and human approvals before release (`relsynth gov`).
- **Time.** Event sequences (a customer's orders) keep their gaps, life times and calendar; meter-style series keep their seasonal profiles and autocorrelation; `relsynth extend` adds new days to a synthetic dataset.
- **Large tables.** Facts larger than memory are fitted from statistics computed inside the database (`GROUP BY`, window functions) plus a uniform sample. Rows never leave the database as a whole. Generation can run in partitions on many processes or Spark, with the same output for any number of workers.
- **Pluggable and adjustable.** Each table's columns come from a generator plug-in (`copula` by default, `independent`, `cart`, or third-party packages). 101 engine settings can be changed in a file, a preset or per table, and `relsynth autotune` tunes them toward your objectives.
- **Quality gate.** A scorecard compares synthetic with real data, check by check, with calibrated limits. `--fail-under 0.98` makes a CI job fail when quality drops.
- **Explainable.** `relsynth model` draws the identified data model, and `relsynth export-model` writes the fitted Bayesian model as PDF, Markdown, JSON and a standalone Python sampler. `relsynth doctor` turns failed checks into spec fixes.

## Install

Python 3.10 or newer. The core needs numpy, pandas, scipy, SQLAlchemy and PyYAML. Add the extra for each database you use:

```bash
pip install relsynth                       # CSV and SQLite
pip install "relsynth[postgres]"           # + PostgreSQL (psycopg 3)
pip install "relsynth[mssql]"              # + SQL Server (pyodbc; needs Microsoft ODBC Driver 17 or 18)
pip install "relsynth[mssql-pymssql]"      # + SQL Server without ODBC (pymssql)
pip install "relsynth[oracle]"             # + Oracle (python-oracledb, thin mode: no Oracle client needed)
pip install "relsynth[mysql]"              # + MySQL 8 and MariaDB (PyMySQL)
pip install "relsynth[databricks]"         # + Databricks SQL
pip install "relsynth[fabric]"             # + Microsoft Fabric Warehouse and Lakehouse SQL endpoints
pip install "relsynth[snowflake]"          # + Snowflake
pip install "relsynth[bigquery]"           # + Google BigQuery
pip install "relsynth[parquet]"            # + Parquet files (pyarrow)
pip install "relsynth[iceberg]"            # + Apache Iceberg tables (PyIceberg)
pip install "relsynth[delta]"              # + Delta Lake tables (delta-rs)
pip install "relsynth[all]"                # all of the above
pip install "relsynth[dp]"                 # differential privacy mode (OpenDP)
pip install "relsynth[gov]"                # signed manifests, registry, policy (relsynth gov)
pip install "relsynth[ir]"                 # JSON Schema check of IR documents
pip install "relsynth[service]"            # the REST API, workers and web UI (relsynth-service)
pip install "relsynth[speed]"              # optional numba kernels (same output, faster on large schemas)
```

From a checkout or a wheel file:

```bash
pip install .                   # or: pip install relsynth-2.0.0-py3-none-any.whl
relsynth version                # versions of relsynth and its main dependencies
```

A container image is described in the [Dockerfile](Dockerfile) (all drivers and ODBC Driver 18 included):

```bash
docker build -t relsynth:2.0.0 .
docker run --rm -v "$PWD:/work" -e PGPASSWORD relsynth:2.0.0 run /work/spec.yaml --json
```

## Quick start (command line)

**1. Draft a specification from your database.** relsynth reads the catalog and a sample of rows. It identifies keys and links (from declared constraints and from the data), then writes `spec.yaml`. The file lists every table and column with its role, every link and every rule found, and a `review:` list of questions only you can answer.

```bash
relsynth introspect --source "postgresql://analyst@db.internal/sales" \
                    --source-password-env PGPASSWORD --out spec.yaml
relsynth check spec.yaml --connect           # validate the spec and that every table is there
relsynth model spec.yaml --out model         # model.md: the data model with a join diagram
```

Sources are written as `csv:folder`, `parquet:folder`, `sqlite:file.db`, `iceberg:...`, `delta:...`, or a database URL (`postgresql://`, `mssql://`, `oracle://`, `mysql://`, `mariadb://`, `databricks://`, `fabric://`, `snowflake://`, `bigquery://`). From release 2.0, `introspect` writes an IR 2.0 document by default (`--format v1` keeps the 1.0 spec); every command reads both. Passwords come from an environment variable (`--source-password-env`), or the whole URL does (`--source-url-env`). Never put a password on the command line.

**2. Generate, write and score.**

```bash
relsynth run spec.yaml --target "sqlite:synthetic.db" --scale 1 --report report.md
```

The run fits the model, draws a synthetic database, writes it, and compares it with the source. It prints the scorecard and writes `diagnostics_bundle.json` (aggregates only, no rows).

**3. If checks fail, ask the doctor.**

```bash
relsynth doctor diagnostics_bundle.json --spec spec.yaml --apply --out spec.fixed.yaml --report doctor.md
relsynth tune spec.yaml --rounds 3 --out spec.tuned.yaml      # run, doctor and apply, until no fix remains
```

**4. Share the model, not the data (optional).**

```bash
relsynth export-model spec.yaml --out model/        # model.pdf, model.md, model.json, replicate.py
```

Every command takes `--json` for machine-readable output and returns an [exit code](#exit-codes) for scripts.

## Quick start (Python)

```python
import relsynth as rs

spec = rs.introspect("sqlite:shop.db")            # a Spec; review spec.review
spec = spec.set("synthesis.scale.default", 0.5)   # any setting, by dotted key
model = rs.fit(spec)                               # the fitted Bayesian model
data = model.sample()                              # dataset 0: a dict-like of DataFrames
report = model.evaluate(data)                      # the scorecard
print(report.passed, "/", report.total)
report.require(0.98)                               # raises QualityGateError below 98%
data.write("postgresql://loader@test-db/sales_synth")   # or data.to_csv("out/"), data.to_parquet("out/")

more = [model.sample(dataset=i) for i in range(1, 4)]   # three more posterior draws
big = model.sample(scale={"customer": 10})               # 10x customers, their facts follow
```

One call runs the whole pipeline, as the CLI does:

```python
result = rs.run("spec.yaml", out="csv:synthetic/", fail_under=0.98, bundle="bundle.json")
diagnosis = rs.doctor(result)                 # causes and spec patches for any failed check
fixed = diagnosis.fixed_spec("spec.yaml")
```

## How it works

```
 source DB / files ──introspect──▶ spec.yaml ──fit──▶ Bayesian model ──sample──▶ synthetic DB
        │                            ▲  (you review)        │                        │
        │                            └────── doctor ◀── scorecard ◀──── evaluate ◀───┘
        └── large facts: GROUP BY statistics + uniform sample, computed in the database
```

1. **Specification.** A YAML file says what each table is (entity, fact, view, reference), how tables link, which rules hold, and how to synthesise each column. `introspect` drafts it, you review it, and it is validated strictly: every option is documented in [docs/spec-reference.md](docs/spec-reference.md), and misspelt keys are refused with a suggestion.
2. **Model.**
   - Every entity row (a customer, a product) gets a latent segment, fitted by Gibbs sampling over the whole schema. The segments explain who links to whom and how often.
   - Each table's columns follow a Bayesian Gaussian copula with exact marginals, per segment or pooled.
   - Facts are counts of links between segments, with fitted rates, multiplicities, participation and seasonality.
   - Hierarchies follow a Galton–Watson process, and text follows a proper predictive law with memorisation guards.
3. **Synthesis.** A dataset draws every parameter once from its posterior, then the rows, entities first and facts after. Rules are imposed by exact conditioning: rejection, a redraw instead of a deletion, or calibrated link draws.
4. **Evaluation.** The scorecard checks marginals, dependence, joins, cardinalities, trees, time, text, rules and privacy, each against a limit calibrated to the sampling noise of a posterior predictive draw.

The [user guide](docs/user-guide.md) walks through each step with a worked example.

## Sources and targets

| | Read (source) | Write (target) | Fit in the database | Tested live |
|---|---|---|---|---|
| PostgreSQL | yes | yes (COPY) | yes | PostgreSQL 16.15 |
| SQL Server | yes | yes (fast executemany) | yes | SQL Server 2022 (16.0), ODBC Driver 18 |
| Oracle | yes | yes (array insert) | yes | Oracle Database 26ai Free (23.26) |
| MySQL, MariaDB | yes | yes (multi-row insert) | yes | MySQL 8.4.13, MariaDB 11.4.13 |
| Databricks SQL | yes | yes (multi-row insert, or Parquet and `COPY INTO`) | yes | Spark 3.5.3 + Delta 3.2.1 stand-in ([warehouses](docs/warehouses.md)) |
| Microsoft Fabric | yes | yes (ODBC parameter arrays, or Parquet and `COPY INTO`) | yes | SQL Server 2022 stand-in ([warehouses](docs/warehouses.md)) |
| Snowflake | yes | yes (multi-row insert, or Parquet staged and `COPY INTO`) | yes | fakesnow 0.12 stand-in ([warehouses](docs/warehouses.md)) |
| Google BigQuery | yes | yes (load jobs from Parquet, or DML `INSERT`) | yes | not yet: unit tests only ([warehouses](docs/warehouses.md)) |
| SQLite | yes | yes | yes | SQLite 3.45 |
| CSV folder | yes (also `.csv.gz`) | yes | yes, with DuckDB over the files | yes |
| Parquet folder | yes | yes | yes, with DuckDB over the files | yes (pyarrow 26, DuckDB 2.0) |
| Apache Iceberg (local, S3, ADLS; any PyIceberg catalog) | yes, one snapshot per table (`snapshot`, `as_of`) | yes, one commit per table, partitioned by Iceberg transforms | yes, with DuckDB over the snapshot | local disk; S3 (moto) and Azurite stand-ins ([lakehouse](docs/lakehouse.md)) |
| Delta Lake (local, S3, ADLS) | yes, one version per table | yes, one commit per table | yes, with DuckDB over the version | local disk; S3 (moto) and Azurite stand-ins ([lakehouse](docs/lakehouse.md)) |

Writes come in four modes:
- `replace`: drop and create.
- `append`.
- `truncate`: delete rows, children first.
- `fail`: stop when a target table exists.

The target's structure comes in three forms:
- `clone`: the source's DDL, with types, keys, unique constraints, indexes and identity columns.
- `infer`: from the spec and the data.
- `existing`: tables created beforehand.

`relsynth clone-schema` creates an empty copy of a schema. URL forms, drivers, schemas, authentication and type mappings per database are in [docs/connectors.md](docs/connectors.md).

## Command line

| Command | What it does |
|---|---|
| `relsynth introspect --source LOC --out spec.yaml` | Draft a spec from a database or folder |
| `relsynth check spec.yaml [--connect]` | Validate a spec (alias `validate`); `--connect` also checks the source |
| `relsynth plan spec.yaml` | Which tables would be fitted inside the database, and why the others are not |
| `relsynth run spec.yaml [--target LOC] [--scale X] [--fail-under S]` | Fit, sample, write and score; writes the diagnostics bundle |
| `relsynth evaluate spec.yaml --synthetic LOC` | Score synthetic data that is already written |
| `relsynth verify spec.yaml [--draws 5]` | Model checks without ground truth (R1–R7): posterior predictive, scale rule, stability, reproducibility |
| `relsynth doctor bundle.json [--spec spec.yaml --apply]` | Causes of failed checks and the spec fixes |
| `relsynth tune spec.yaml [--rounds 3]` | Run, doctor and apply, repeated |
| `relsynth autotune spec.yaml` | Tune engine settings toward your objectives over several seeds; constraints report for what cannot be met |
| `relsynth settings show\|explain\|validate\|template\|reference` | The engine settings, their ranges and the checks each affects |
| `relsynth run spec.yaml --epsilon E [--dp-unit TABLE]` | Differentially private run (needs `relsynth[dp]`); writes a DP statement |
| `relsynth privacy spec.yaml` | Attack a release per person; privacy report per release context |
| `relsynth extend spec.yaml --until DATE [--step day]` | Continue the event histories of a synthetic dataset step by step |
| `relsynth generators list\|show`, `relsynth compute` | Installed generator plug-ins; GPUs found and the device each generator would get |
| `relsynth ir migrate\|validate\|convert\|privacy\|odcs\|from-odcs\|openlineage\|check-contract` | IR 2.0 documents, data contracts (ODCS) and lineage (OpenLineage) |
| `relsynth gov init\|keys\|policy\|budget\|approve\|release\|verify ...` | Governance: registry, keys, policy, privacy budget, approvals and releases |
| `relsynth-service serve\|worker\|migrate\|tenant\|audit\|openapi` | The service: REST API, web UI and job workers (needs `relsynth[service]`) |
| `relsynth model spec.yaml` | The identified data model (Markdown with a Mermaid ER diagram, and JSON) |
| `relsynth export-model spec.yaml --out DIR` | The fitted Bayesian model as JSON, Markdown, PDF and runnable Python |
| `relsynth clone-schema --source LOC --target LOC` | Create the source's tables empty in a target database |
| `relsynth profile --source LOC` | Row, null and distinct counts computed in the source (no rows read) |
| `relsynth ping --source LOC` | Test a connection |
| `relsynth version` | Versions of relsynth and its dependencies |

Shared options:
- `--json`: one JSON document on stdout.
- `--log-level debug|info|warning|error`.
- `--set KEY=VALUE`: override any spec value, for example `--set synthesis.seed=7`.

The full reference with every option is in [docs/cli.md](docs/cli.md).

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Quality gate failed (`--fail-under`) |
| 2 | Invalid spec or invalid arguments |
| 3 | Connection, permission or file error |
| 4 | Engine failure (fitting or sampling) |
| 130 | Interrupted |

## Python API

| Call | Returns |
|---|---|
| `rs.introspect(source, ...)` | `Spec` drafted from a source |
| `rs.Spec.load(path)`, `Spec.from_dict(d)`, `.set(key, value)`, `.with_source(loc)`, `.with_target(loc)`, `.errors()`, `.check()`, `.save(path)`, `.model_report(stem)` | Load, edit, validate and save specs (saved specs never contain passwords) |
| `rs.fit(spec, pushdown=None, seed=None)` | `FittedModel` |
| `FittedModel.sample(scale=None, dataset=None)` | `SyntheticData` (a mapping of table name to DataFrame) |
| `FittedModel.evaluate(data)` / `rs.evaluate(spec, data, real=None)` | `EvaluationReport` (`.passed`, `.total`, `.score`, `.failed`, `.require(min)`, `.to_markdown()`, `.to_json()`) |
| `FittedModel.export(out_dir)` / `rs.export_model(spec, out_dir)` | Paths of the exported model files |
| `SyntheticData.write(target)`, `.to_csv(dir)`, `.to_parquet(dir)` | Written locations |
| `rs.run(spec, ...)` | `RunResult` (`.data`, `.model`, `.report`, `.bundle`, `.errors`) |
| `rs.verify(spec, draws=5)` | `VerifyReport` (R1–R7) |
| `rs.doctor(bundle)` | `Diagnosis` (`.issues`, `.patches`, `.markdown`, `.fixed_spec(spec)`) |
| `rs.tune(spec, rounds=3)` | `(best spec, bundle, history)` |
| `rs.clone_schema(source, target)`, `rs.profile(source)`, `rs.ping(source)`, `rs.connect(source)` | Database utilities |
| `rs.Settings`, `Spec.with_settings(...)`, `fit(spec, settings=...)` | Engine settings with editable, validated attributes |
| `rs.privacy(spec, ...)`, `rs.assess_quality(...)` | Privacy attacks per person; the 2.0 quality report |
| `rs.extend(...)` | Continue a synthetic dataset's event histories |
| `rs.Synthesizer` | The engine class, for full control |

Errors are typed:
- `SpecError` (exit code 2).
- `SourceError` (3).
- `EngineError` (4).
- `QualityGateError` (1).

All derive from `RelsynthError`, except `SpecError`, which is a `ValueError`. The full reference, with every parameter, is in [docs/api.md](docs/api.md).

## CI/CD

A job that refreshes a test database every night and fails when quality drops:

```yaml
# .github/workflows/synthetic-data.yml (excerpt)
- run: pip install "relsynth[postgres]==2.0.0"
- run: relsynth check spec.yaml --connect --json
- run: relsynth run spec.yaml --target-url-env TEST_DB_URL --fail-under 0.98 --json > run.json
  env:
    PGPASSWORD: ${{ secrets.PROD_READONLY_PASSWORD }}
    TEST_DB_URL: ${{ secrets.TEST_DB_URL }}
- uses: actions/upload-artifact@v4
  with: { name: diagnostics, path: diagnostics_bundle.json }
```

Ready-made pipelines for GitHub Actions, GitLab CI, Azure DevOps and Jenkins are in [examples/ci/](examples/ci/), explained in [docs/cicd.md](docs/cicd.md). The package's own CI is in [.github/workflows/ci.yml](.github/workflows/ci.yml): lint, security and licence scans, types, unit tests on Python 3.10–3.13 and three operating systems, integration tests against live PostgreSQL, SQL Server, Oracle, MySQL, MariaDB, TimescaleDB, S3 and Azure stand-ins, warehouse stand-ins, the statistical check suite in four shards, the benchmark regression gate, the proofs build, and a signed wheel with an SBOM ([docs/development/ci.md](docs/development/ci.md)).

## Tables larger than memory

A fact table is fitted inside the database (*pushed down*) when `synthesis.pushdown.mode` is `on`, or `auto` with at least `min_rows` rows (default 1,000,000). The engine then reads:
- exact `GROUP BY` counts over the key tuples the model uses;
- sums of rank normal scores, computed with window functions;
- first rows per document key, for order headers and running balances;
- a uniform, seeded sample of `sample_rows` rows (default 500,000) for the column models.

Entities (dimensions) are always loaded, because each of their rows carries its own latent segment.

```yaml
synthesis:
  pushdown: {mode: auto, min_rows: 1000000, sample_rows: 500000}
```

```bash
relsynth plan spec.yaml          # what will be pushed down, and why other tables are not
```

The relational part of the model is unchanged: Proposition 100 proves it, and a run with a sample of every row gives exactly the in-memory result on all four databases. The column models are an exact posterior given the sample (Proposition 104). Memory grows with the number of distinct key tuples, not with rows times columns. Details: [docs/pushdown.md](docs/pushdown.md).

## Quality: the scorecard, the doctor and model checks

- **Scorecard** (`run`, `evaluate`). It holds hundreds of pass/fail checks: marginals, dependence, join distributions, cardinalities, duplicate rates, trees, time, text, rules, privacy and a discriminator AUC. Each limit is the larger of a fixed tolerance and the sampling limit of its statistic under the null that the synthetic data is a posterior predictive draw. Families of checks are Šidák-corrected, and each check reports its significance level, so the number of false failures to expect is known.
- **Doctor**. It maps each failed check and each error in the diagnostics bundle to a cause and a spec patch. `tune` applies the patches in a loop.
- **Model checks** (`verify`). These are checks that need no ground truth: posterior predictive checks of key statistics (R3), the scale rule (R4), structural validity (R5), segment stability across seeds (R6), and reproducibility (R7).

## Privacy, differential privacy and governance

- **Measured risk** (`relsynth privacy`, [docs/privacy.md](docs/privacy.md)). Attacks count per person of the privacy unit, not per row: membership inference (shadow models, DOMIAS, LiRA), attribute inference against the population baseline, singling out and linkability. The report gives a verdict per release context (internal, partner, public) for a privacy officer to sign.
- **Differential privacy** (`relsynth run --epsilon E`, `relsynth[dp]`, [docs/dp.md](docs/dp.md)). Stage 1 bounds each person's contribution and releases noisy integer statistics with OpenDP's discrete Gaussian mechanism; stage 2 fits the ordinary engine to them. The guarantee is user-level (ε, δ) for one person and all their rows; δ is at most 10⁻⁵ and ε has no default. Every run writes a DP statement in the NIST SP 800-226 style. On Contoso at ε = 1 the measured membership-inference rate stayed below the DP bound.
- **Governance** (`relsynth gov`, `relsynth[gov]`, [docs/governance/guide.md](docs/governance/guide.md)). A registry of signed (Ed25519, DSSE / in-toto) manifests in a hash-chained log; an organisation policy with ε caps; a privacy budget ledger per domain that reserves before any data is read; approvals with quorum and separation of duties; signed release packages. A run with `--context partner` or `public` is refused before reading data unless it is differentially private.
- **Service** ([docs/service/](docs/service/README.md)). The same engine behind a REST API and web UI with OIDC single sign-on, roles per tenant, encrypted artefacts, approval before download, a durable job queue, Helm and docker compose deployments and an air-gapped bundle.

## Time series, generators, settings and scale

- **Time series** ([docs/timeseries.md](docs/timeseries.md)). `tables.<t>.sequence` redraws the times of an entity's events from a point-process model (calendar in operational time, life-time copula, gamma renewal gaps), keeping keys, counts and rules. `tables.<t>.series` synthesises regular readings with daily, weekly and yearly profiles. `relsynth extend` adds the events after a dataset day by day; a stepwise run equals one long run. TimescaleDB hypertables are read and created.
- **Generator plug-ins** ([docs/generators.md](docs/generators.md), [docs/compute.md](docs/compute.md)). `tables.<t>.generator` picks the model of a table's columns; the engine keeps keys, links, segments, counts and rules whatever the generator. Third-party generators install through the `relsynth.generators` entry point; `compute.yaml` presets control GPUs.
- **Engine settings and autotune** ([docs/settings.md](docs/settings.md), [docs/autotune.md](docs/autotune.md)). 101 settings with type, range and the checks they affect, from defaults, a settings file, presets (`fast`, `thorough`), the spec or `--set`. With default settings the output is unchanged byte for byte. `relsynth autotune` separates systematic failures from noise over several seeds and changes settings until your objectives hold, or says why they cannot.
- **Partitioned generation** ([docs/runtime.md](docs/runtime.md)). `synthesis.runtime` runs fact rows in partitions on the `local`, `process` or `spark` executor. Each partition draws from a Philox stream keyed by seed, dataset, fact and partition, so the output is the same for 1, 8 or 64 workers and for Spark (Prop. 950).
- **IR 2.0** ([docs/ir-guide.md](docs/ir-guide.md)). One versioned document describes a data estate: entities with shapes (record, document, node, edge, event), nested arrays, keys, references, hierarchies, time and versioning, partitions, rules and the privacy unit. 1.0 specs convert without loss (same synthetic data); ODCS v3 data contracts and OpenLineage are exported and checked.

## Security

- **Credentials.** Passwords are never accepted on the command line; they come from `password_env` or `url_env`. Passwords are masked in every message, log, saved spec, diagnostics bundle and exported model. The masking covers URL passwords, `password=`/`token=` query parameters and ODBC `PWD=`.
- **SQL.** Every SQL statement is built with SQLAlchemy Core, with quoted identifiers and bound values. Table and column names cannot inject SQL; tests use hostile names on every database.
- **Specs.** A spec is YAML read with a safe loader that also refuses aliases, so an alias bomb fails. Rule expressions run in a sandboxed evaluator: no attribute access, calls only to a fixed list of functions, and size and depth limits. Regular expressions with catastrophic backtracking are refused.
- **Files.** Table names cannot leave the target folder (`..`, separators, absolute paths and symlinks are refused).
- **Data.** Rows never leave the source database for pushed-down facts. Bundles and profiles hold aggregates only, and `profile --top` must be asked for explicitly. Privacy checks run by default.
- **Supply chain.** The code is scanned with bandit and ruff's security rules, and dependencies with pip-audit, in CI.

Details and the threat model: [docs/security.md](docs/security.md).

## Performance

- Fits use exact vectorised kernels, with numba kernels as an option (`relsynth[speed]`). Sampling is linear in the output size.
- Writes use each database's bulk path: COPY on PostgreSQL, fast executemany on SQL Server, array binds on Oracle.
- Large facts are pushed down (see above). On AdventureWorks DW (27 tables, 1.05 million rows), pushing the ten eligible facts down cut peak memory from 2.1 GiB to 1.6 GiB and run time from 803 s to 668 s on a 4-core machine.

Tuning knobs and measurements: [docs/pushdown.md](docs/pushdown.md#performance).

## Validation

**Release 2.0.** The statistical check suite now has C00–C82 plus the 2.0 checks C700–C706 (generators), C800–C803 (governance), C900–C906 (differential privacy) and C1200–C1206 (time series): 113 science tests, all run in CI on every pull request in four shards, next to 775 unit, end-to-end, security and live-server integration tests. A benchmark harness (`bench/`) reruns the real databases (bi/COREP, Chinook, AdventureWorks DW, Contoso, plus Berka, ACS PUMS and a doubled Chinook) on every change and flags drift against a five-seed baseline. Draws for the same seed change on databases with link tables, because degree laws are now the default (`synthesis.facts.link_degree: segment` restores the 1.0 draws); the [changelog](CHANGELOG.md) lists every change.

relsynth 1.0.0 was checked at three levels. Every number below comes from the final code.

**Fake data with known ground truth.** The statistical check suite (`pytest -m science`) generates eleven databases from known models, from one mixed-type table to an enterprise schema with nested facts, views, trees and order headers. It then checks that each numbered proposition holds: posterior recovery and calibration, identification of keys and rules, rules on every row, privacy guards, reproducibility. All 79 checks (C00–C78, more than 500 assertions) pass.

**Four real databases**, each from an unedited `introspect` draft, fitted in memory and with every eligible fact pushed down:

| Database | Rows | Scorecard in memory | Scorecard pushdown | Domain checks (rules and invariants) |
|---|---|---|---|---|
| EBA COREP reporting | 68 k | 177/178 | 178/178 | B1–B8 equal in both modes; B6 per-cell KS 0.10 and 0.13 |
| Chinook | 15.6 k | 144/146 | 144/146 | |
| Contoso V2 100k (Parquet) | 753 k | 253/253 | 253/253 | T1–T17 within 0.015 between modes |
| AdventureWorks DW | 1.05 M | 1229/1286 | 1237/1285 | G1–G13 equal in both modes, except two counts within 2% |

In both modes, every key resolves, every unique key but one holds (an AdventureWorks alternate key with a start date), and every declared or found rule holds on every row. The remaining failures, database by database, are described in [docs/pushdown.md](docs/pushdown.md#real-databases-in-memory-against-pushdown).

**Software tests** (`pytest`): 136 unit, end-to-end CLI and security tests, and 24 integration tests run live against PostgreSQL 16.15, SQL Server 2022 (16.0) and Oracle Database 26ai Free (23.26). On every server, the same seed gives the same synthetic database as SQLite, value for value. The wheel was installed and tested in clean environments on Python 3.11, 3.12 and 3.13. Python 3.10 is supported by the code and the CI matrix, but it was not available on the build machine.

## Theory: propositions and proofs

The method is described in two papers shipped with the package:

- [docs/proofs/algorithms.pdf](docs/proofs/algorithms.pdf): the layer 1 algorithms, with complexity and proofs. It covers the Bayesian Gaussian copula, latent segments from in-database sufficient statistics, the merge step and differential privacy on summaries.
- [docs/proofs/relsynth-proofs.pdf](docs/proofs/relsynth-proofs.pdf): 152 propositions with proofs, each tied to a numbered check. They are grouped as follows:

| Area | Propositions (examples) |
|---|---|
| One table, mixed types | Marginal preservation; the copula sampler targets the posterior; rank likelihood with parameter expansion; rare columns kept independent |
| Links between tables | Identifiability of segments; block-rate posterior; splitting and degree correction; the scale rule; chained segments compose; nested refinement |
| Counts and structure | Spike-and-slab multiplicities; structural zeros; sparse Dirichlet inclusion; mandatory participation; learned structural weights; Pólya-urn marginal |
| Hierarchies, time and text | Galton–Watson generation sizes; hierarchies of a given size; seasonality by multinomial thinning; proper predictive text with memorisation guards |
| Rules and keys | Rules by rejection; three-valued rules agree with the database; computed columns are exact; unique keys by exact conditioning; order headers (lossless split); running balances |
| Bayesian synthesis | One parameter draw per dataset; posterior predictive draws; soft rules; group effects; SCD type 2; exported draw *m* is dataset *m* |
| Identification | Significance of discovered dependencies; inclusion by chance; permutation test of a forest |
| Evaluation | Calibrated limits under the posterior predictive null; Šidák over families; distances with an atom at zero |
| In-database fitting (v1.0) | **100** sufficiency of the in-database statistics; **101** agreement with in-memory fitting; **102** Acklam's quantile bound; **103** hash sampling is Bernoulli; **104** the sample posterior is exact; **105** eligibility |
| IR, generators and quality (v2.0) | **106** loss-free 1.0 conversion; **107–108** privacy-unit tracing; **109–113** calibrated quality report; **700–706** the skeleton is kept whatever the generator |
| Privacy and governance (v2.0) | Per-person attack metrics; **800–803** the budget ledger bounds the domain's privacy loss, reserve before reading, tamper evidence; **900–906** discrete Gaussian, stable contribution bounding, exact partition selection, two-stage DP |
| Time and scale (v2.0) | **1200–1206** time change, gamma renewal bridge, sequence posterior, calibrated limits, incremental runs; degree laws of link tables; **950–954** worker independence, partitions keep the law, keys across partitions, drift tests |

[docs/theory.md](docs/theory.md) gives a readable summary of the model and of the six v1.0 propositions, with the proofs in full.

## Testing the package

```bash
pip install -e ".[dev]"
pytest                                    # unit, end-to-end and security tests (SQLite, CSV, Parquet)
docker compose -f docker-compose.test.yml up -d       # PostgreSQL, SQL Server, Oracle, MySQL, MariaDB
export RELSYNTH_TEST_PG_URL=postgresql://relsynth:RelSynthTest1@localhost/relsynth
export RELSYNTH_TEST_MSSQL_URL="mssql://sa:RelSynth_Test1!@localhost:1433/master?TrustServerCertificate=yes&driver=ODBC+Driver+18+for+SQL+Server"
export RELSYNTH_TEST_ORACLE_URL="oracle://relsynth:RelSynthTest1@localhost:1521/?service_name=FREEPDB1"
export RELSYNTH_TEST_MYSQL_URL=mysql://root:RelSynthTest1@localhost:3306/relsynth
export RELSYNTH_TEST_MARIADB_URL=mariadb://root:RelSynthTest1@localhost:3307/relsynth
pytest tests/integration                  # the same checks against live servers
pytest -m science                         # the statistical check suite: C00–C82 and the 2.0 checks (about an hour and a half; CI runs it in four shards)
pytest --cov=relsynth --cov-report=term   # coverage
ruff check src tests && bandit -c pyproject.toml -r src && pip-audit
```

What each test module covers is in [docs/testing.md](docs/testing.md).

## Limitations

- The scorecard on real data compares with the source, which is itself only one draw. A check can fail by chance, and each check's level is reported (`alpha`).
- Rules that hold on every source row but are not declared or found are reproduced statistically, not exactly. Declare them in the spec.
- Differential privacy mode releases one-way statistics, degrees and one correlation matrix per table, so its utility is nearly flat in ε (0.80–0.91 of the checks pass from ε = 0.5 to 8 on Chinook, Contoso and AdventureWorks). It has one privacy unit, no in-database pushdown, and placeholders for pii, text and binary columns ([docs/dp.md](docs/dp.md)). Selected two-way marginals are the next step.
- Document databases (MongoDB, Cosmos DB, DynamoDB) and graph stores are planned. The IR already describes documents and graphs; data lakes on S3 and ADLS work through Iceberg and Delta.
- Free text is synthesised by a word-level predictive model. It keeps vocabulary and length laws, not meaning.
- BigQuery is tested with unit tests only; the other warehouses against stand-ins, not the live services.

## Documentation map

| Document | Contents |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | The workflow step by step, with a worked example |
| [docs/spec-reference.md](docs/spec-reference.md) | Every specification option, with recipes ([template](docs/spec-template.yaml)) |
| [docs/settings.md](docs/settings.md) | Engine settings: settings files, presets, per-table overrides, editable attributes ([reference](docs/settings-reference.md)) |
| [docs/autotune.md](docs/autotune.md) | `relsynth autotune`: objectives, the search, the history and the constraints report |
| [docs/ir-guide.md](docs/ir-guide.md) | The IR 2.0 document: migration from 1.0 specs, documents, graphs, streams, the privacy unit, ODCS and OpenLineage, deprecation policy |
| [docs/ir-reference.md](docs/ir-reference.md) | Every IR field (generated from the model; JSON Schema in `src/relsynth/ir/schema/`) |
| [docs/adr/](docs/adr/) | Architecture decision records |
| [docs/cli.md](docs/cli.md) | Every command and option |
| [docs/api.md](docs/api.md) | Every public class and function |
| [docs/connectors.md](docs/connectors.md) | Databases and files: URLs, drivers, schemas, write modes, types |
| [docs/pushdown.md](docs/pushdown.md) | Fitting inside the database; performance |
| [docs/connector-sdk.md](docs/connector-sdk.md) | Writing a connector for a new database or warehouse |
| [docs/warehouses.md](docs/warehouses.md) | Databricks SQL, Microsoft Fabric, Snowflake and BigQuery: sign-in, cost guards, loading |
| [docs/lakehouse.md](docs/lakehouse.md) | Apache Iceberg and Delta Lake on disk, S3 and ADLS; time travel; data contracts |
| [docs/runtime.md](docs/runtime.md) | Partitioned generation: executors, streams, partition modes, drift statistics |
| [docs/timeseries.md](docs/timeseries.md) | Event sequences, regular series, incremental runs, TimescaleDB |
| [docs/generators.md](docs/generators.md) | Generator plug-ins and how to write one ([compute presets](docs/compute.md)) |
| [docs/privacy.md](docs/privacy.md) | Privacy attacks per person and the report per release context |
| [docs/dp.md](docs/dp.md) | Differential privacy mode: guarantee, parameters, DP statement, limits |
| [docs/governance/guide.md](docs/governance/guide.md) | Registry, signed manifests, policy, budget ledger, approvals ([design](docs/governance/design.md)) |
| [docs/service/](docs/service/README.md) | The service: API, UI, SSO, tenancy, deployment, operations, security |
| [docs/metrics.md](docs/metrics.md) | The quality report 2.0 and its metrics |
| [docs/profiling.md](docs/profiling.md) | Where 1.0 spends time and memory, and which changes that justifies |
| [docs/cicd.md](docs/cicd.md) | Pipelines, quality gates, secrets, containers |
| [docs/security.md](docs/security.md) | Threat model and protections |
| [docs/theory.md](docs/theory.md) | The model in brief and the v1.0 propositions with proofs |
| [docs/proofs/](docs/proofs/) | The full proofs (PDF and LaTeX source) |
| [docs/testing.md](docs/testing.md) | The test suite |
| [CHANGELOG.md](CHANGELOG.md) | Versions |
| [docs/datasets.md](docs/datasets.md) | Which datasets may be used, committed and where |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute: workflow, commits, sign-off, tests |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |
| [docs/assurance/](docs/assurance/README.md) | For privacy, legal and audit teams: the assurance case, the regulation mapping (GDPR and EDPB 02/2026, AI Act, HIPAA, NIST, ISO), a DPIA template, release guidance and the claim register |

## Licence

relsynth is licensed under the [Apache License 2.0](LICENSE). Third-party material and its licences are listed in [NOTICE](NOTICE). Contributions are accepted under the same licence with a [Developer Certificate of Origin](CONTRIBUTING.md#developer-certificate-of-origin) sign-off.

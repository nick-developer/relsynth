# Changelog

All notable user-visible changes are recorded here, following [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/). relsynth uses [Semantic Versioning](https://semver.org/); see [CONTRIBUTING.md](CONTRIBUTING.md) for what counts as the public API.

## Unreleased
### Added

- **Several privacy units in one DP run** (gap plan G1). `synthesis.privacy.dp.unit` takes a list (`[DimCustomer, DimEmployee]`), and `--dp-unit` a comma-separated list. Each unit owns the tables whose path to it passes through no other unit, has its own accountant with the run's budget, and gets its own (ε, δ) guarantee; the budgets do not add up because the units' tables are disjoint (Prop. 1703). The plan refuses a table with data of two units and links between the units' data before any read. The statement of such a run has schema `relsynth.dp-statement/2` (`unit` is the list, `units` each unit's ε, ρ and tables), each charge names its unit, and `DPResult.per_unit` holds the same figures. A unit's budget can be set by hand (`epsilon: {DimCustomer: 2, DimEmployee: 1}`).
- **Governed runs with several units** (RFC-G1-01, Prop. 1704). A policy domain can name several units (`unit: {entities: [...]}`); the ledger keeps one account per unit, the domain's budget applies to each unit, and `unit_budgets` sets a unit's budget by hand. A run reserves every unit's budget before it reads data (all or nothing), touches and settles each account, and its manifest lists the units (`privacy.unit` as a list, `charges[].unit`, `privacy.units`). The release rules run once per unit with that unit's figures and must all allow; no Rego change. `gov budget` and `gov policy check` show each unit's account. Ledger entries of such a domain carry `unit`; one-unit domains are unchanged.
- **Real composite keys in the DP seed database.** When an id groups an owned table's rows (an order number with a sequence column `within` it, or a fact's header key), stage 1 releases a histogram of rows per group as one more block of the table's marginals (Prop. 1702). The seed has orders with several lines numbered within each order, and children point to them by the full composite key; other free key columns number the rows instead of the old `(i, 1)` placeholder, attribute parts of a composite key (a date) are drawn like attributes, and seed rows that repeat a declared composite key are dropped.
- Checks C1700–C1706 (`tests/science/test_privacy_units_checks.py`), Props. 1700–1704 (`docs/proofs/sec_g1_units.tex`), docs/dp.md section 8; C1705 replays the two-unit sensitivity check on AdventureWorks DW when the bench cache has it; C1706 replays 120 governed two-unit releases against the per-unit ledger accounts.

### Changed

- **Privacy-unit ownership comes from the IR's trace** (`relsynth.ir.privacy`, RFC-T06-01). The attack suite (`relsynth privacy`) now follows composite foreign keys, so AdventureWorks `FactInternetSalesReason` is owned by `DimCustomer` and appears in its profiles and report. A row that reaches the unit along several paths belongs to every person they reach (union ownership, Prop. 1700); on the bench specs whose paths all agree (Contoso, bi/COREP) the subsets and profiles are unchanged. New `relsynth.privacy.unit.owners`; `owner_ids` keeps one owner per row along the shortest path.
- The attack suite keeps people who share a row (union ownership) on the same side of its training/control/reference split, so no control profile holds a row the release was fitted on, and control and reference never grow past their sizes. If the shared rows join almost everyone, the split falls back to the plain cut: the report then warns (`split_shared_rows`) and every context is at least `review`. Without shared rows the split is unchanged.
- DP mode: a parent table measures and truncates only the children on their ownership links, so a link dropped with `drop_columns` (to another unit's table, or off the path) is never read; a declared unique key with a null part no longer counts as a repeat in the seed; `--dp-unit` with no table name (including `--dp-unit ""`) is an argument error. When every parent draws degree 0 from a child's noisy degree histogram (rare, small tables), the seed keeps one child row, as a root table keeps at least one row; stage 2 failed on such an empty fact with children (post-processing, no budget).
- DP mode output changes where documented above (composite keys, groups); the non-DP output is byte-identical (`tools/identity_check.py`, 24 runs).

## 2.1.0 (2026-11-07)
### Security

- **The service no longer writes a job's description to disk.** The worker passed the job (source section, spec, parameters) to the job process in a `job.json` file in the work directory; it now goes over the process's stdin, and credential keys are removed from the source section before it leaves the worker (CodeQL `py/clear-text-storage-sensitive-data`). `python -m relsynth.service.runner -` reads stdin; a file argument still works for development.
- **`synthesis.compute.file` is confined to the spec's folder.** A spec could name any file on the machine as its compute presets (`../`, absolute paths, symlinks); the file must now be a `.yaml`/`.yml` file inside the folder of the spec (CodeQL `py/path-injection`). `relsynth compute --file` from the command line is unchanged apart from the suffix check.
- **Generator version specifiers are parsed without a backtracking regular expression** and limited to 200 characters (CodeQL `py/polynomial-redos`).
- The release workflow passes `github.workflow_ref` to the Sigstore step through an environment variable instead of expanding it in the script (zizmor `template-injection`).
- **Document export quotes table and column names.** `relsynth docs` wrote the shredded tables to SQLite with names taken from document field paths inside unescaped double quotes, so a field name containing `"` could change the DDL; names are now quoted with embedded quotes doubled. Found by the new project Semgrep rules.
- **More code scanning in CI**: CodeQL advanced setup with the `security-extended` and `security-and-quality` suites for Python, Actions and JavaScript (`codeql.yml`); a `security` workflow with Semgrep, gitleaks, zizmor and dependency review (Semgrep and zizmor report to code scanning); project Semgrep rules (`.semgrep/relsynth.yml`) for relsynth's own security rules; a CodeRabbit configuration (`.coderabbit.yaml`) with five security pre-merge checks, per-area security instructions, ten scanners and this repository's check results. The scanners install from hash-pinned requirement files, gitleaks runs from a digest-pinned image without privileges or network, and no source text matched by Semgrep reaches its uploaded SARIF or its log. See [docs/development/ci.md](docs/development/ci.md#security-scanning).
- **Tenant specs can no longer name files on the service.** A spec saved through the service API could set `_base_dir`, a file path in `synthesis.settings_file` or `synthesis.compute.file`, and so make a job read YAML files on the service host. The service now refuses these keys when a spec is saved and again when a job runs (only built-in `preset:NAME` settings are accepted); see [docs/service/tenancy.md](docs/service/tenancy.md).
- **A spec without a folder confines `synthesis.compute.file` to the working directory.** Specs built in code (no `_base_dir`) could still name any YAML file; `relsynth compute --file` names the folder of the typed file itself.
- **Service log lines cannot be forged.** Audit values written to the service log (targets, actions, tenant names) have control characters replaced (CodeQL `py/log-injection`).
- **HTTP calls accept only `http(s)` URLs.** A shared opener (`relsynth._http`) without `file:`, `ftp:` or `data:` handlers serves OIDC discovery, Vault, the agent model backends and the benchmark downloaders (Semgrep `dynamic-urllib-use-detected`). Redirects never go from https to http, and a request that carries credentials (Authorization, Vault token, API key, cookie) is not redirected to another origin; its own error messages (a refused scheme, a refused redirect) name only scheme, host and port, and a refused redirect's error keeps no `Location`, cookie or body (including urllib's own refusals of a non-http(s) target, a redirected POST or a redirect loop). The end-to-end script (`deploy/e2e/e2e_flow.py`), whose requests all carry a token or password, refuses every redirect.
- XML in the bench and documents benchmark is parsed with `defusedxml`; the bench SQLite builders create tables through a quoting, type-checked helper; the governance secrets folder is owner-only; legacy and connector SQL is literal or parameterised.
- **Dependabot alerts**: the Semgrep CI requirements force `mcp>=1.30.0` and OpenTelemetry `>=1.45.1` past Semgrep's own pins (5 moderate CVEs). Minimum versions of optional extras with known CVEs below them are raised (pyarrow, azure-identity, snowflake, PyJWT, cryptography, OpenTelemetry, pymongo, couchbase, torch, transformers, py7zr, pytest). `filelock` and `dnspython` fixes need Python 3.11+, so they are left to the resolver on 3.10.
- Docker base images are pinned by digest and watched by Dependabot; the bench baseline job no longer persists the checkout token; the example GitHub Actions workflow pins actions by commit (zizmor, Semgrep).

### Added

- **Agents** (`relsynth.agents`, `relsynth agent review|apply|explain|doctor|draft|eval`, guide `docs/agents.md`; roadmap T16). Agents draft and explain; people decide. `review` answers the `review:` items of an introspected draft with evidence measured on every row and proposes the change each answer implies (add or remove a relationship, remove a rule); `apply --by` applies the answers a person accepts. `explain` words quality, privacy and scorecard reports; `doctor` turns findings into proposals (privacy settings only as advice to the privacy officer); `draft contract` and `draft policy` write an ODCS contract and an organisation policy with every budget empty. Backends: deterministic rules (default) or an OpenAI-compatible model server on the same machine (Ollama, vLLM); a remote server needs `allow_remote`, https, and gets aggregates only.
- **Agent safeguards**: data scopes by role (`schema`, `aggregates`, `values` with k = 10) under a ceiling, untrusted text marked as data, closed verdicts and typed changes, evidence before any confirmation, grounded plain-text explanations, and no authority: only a person's decision applies a proposal. Props. 1600–1604 with checks C1600–C1604, and a prompt-injection suite (OWASP LLM01/02/05/06).
- **Agent accounts in the service**: the `agent` role (service accounts only; OIDC tokens cannot claim it) can read and propose and nothing else; `POST /specs/{id}/agent/review`, `GET /specs/{id}/proposals`, `POST /specs/{id}/proposals/decisions` (people only; an accepted proposal makes a new spec version), `POST /runs/{id}/agent/explain`; settings `RELSYNTH_SVC_AGENT_*`; every agent action and refusal audited.
- **Governance**: an agent action can be recorded as a signed `agent-action/1` statement in the registry; run manifests may list agent actions under `agents`; `approve` refuses keys trusted with the role `agent` and `gov verify` reports any such approval.
- **Evaluation**: `python -m bench.agents` and the CI job `agents` measure the review agent against the catalogs of Chinook, AdventureWorks DW and Contoso (and the curated bi/COREP spec) with the draft made in benchmark mode: 122 of 124 answered items correct with the rules backend.
- **Compliance evidence and assurance case** (`docs/assurance/`; roadmap T17): an assurance case for a release (claims, evidence, assumptions, defeaters), a mapping to EDPB Guidelines 02/2026 and *EDPS v SRB*, the AI Act (Art. 10 and the new Art. 4a), HIPAA expert determination, NIST SP 800-226 and SP 800-188 and ISO/IEC 27559 and 20889, a DPIA template for a synthetic release, and release guidance for internal, partner and public recipients.
- **Claim register** (`docs/assurance/claims.yaml`, `tools/claims_check.py`, CI job `lint`): every claim of the README and of the privacy, DP, security and governance guides (96 claims) traced to a proposition, check, test or measured report; a README sentence that sums up a guide cites that guide's claim. With the documents and agents guides added, the register holds 147 claims in 7 documents at release. CI fails when a claim changes, its evidence disappears, or a new claim is not registered.
- **Two-way marginals in DP mode** (`synthesis.privacy.dp.measure: mst | aim`, `relsynth run --dp-measure`; roadmap T07b, guide `docs/dp.md` sec. 7, Props. 1300-1306, checks C1300-C1306). Stage 1 can release privately chosen two-way histograms in place of the copula's correlation matrix: MST picks a maximum spanning tree with the exponential mechanism, AIM picks pairs over adaptive rounds with annealing (kept to a forest). The nodes of a table include its parent row's cells and its rows per parent row, so the chosen pairs carry the links between tables. Domains are compressed to `pair_bins` cells (default 8) from released marginals; the seed is drawn from the fitted forest. Same accountant, same per-person guarantee and contribution bounding. The default (`copula`) is unchanged. New tools `tools/dp_compare.py` (against smartnoise-synth MST and AIM on ACS), `tools/dp_curves.py` and `tools/dp_workload.py`, the CI workflow `dp-eval` and `tools/dp_audit.py` (the attack audit against the ε bound, run in `dp-eval`).

#### Synthetic text (roadmap T14)

- **Row-conditioned text with a method per column** (`text.method: ngram | lm | pe`, guide `docs/text.md`, ADR 0010, Props. 1500–1507, checks C1500–C1507). `condition_on` takes one or several columns, so a ticket's text fits its category and priority. `ngram` is the 1.0 word-level model and stays the default; with one conditioning column (or none) its output is byte-identical to 2.0.
- **Differentially private text**: `text.dp` gives user-level DP for the column by a DP n-gram release (CPU, no weights, air-gapped), DP-GD fine-tuning of a local language model on full batches (zCDP cost exactly T/(2σ²)), or Private Evolution with a public generator (refused without `dp`). Charges go to the DP mode's accountant. The guarantee covers that column only; text columns are still placeholders inside a whole-database DP run.
- **Local language models** (`relsynth[text-lm]`: PyTorch, transformers, peft) from a model folder only (`local_files_only`), with optional LoRA; relsynth ships no weights. `relsynth text models` lists the allowed models with their licences (`src/relsynth/data/text_models.yaml`, checked by `tools/check_text_models.py`).
- **PII layer**: recognised personal data (regex by default; Presidio, spaCy and GLiNER with `relsynth[text-pii]`) is replaced by typed placeholders before training and by valid surrogates after generation; `relsynth text pii-eval` measures recognizer recall with intervals.
- **Leakage audit and text metrics**: `relsynth text audit` inserts canaries and measures their exposure against the DP bound and searches the output for near-duplicates of training texts (MinHash); `relsynth text metrics` reports consistency with the row, MAUVE-style distance and the rest of the text metrics. CI job `text-lm` runs a small local model end to end.

#### Document databases (roadmap T13)

- **`relsynth docs discover|synth|check`** and **`relsynth.documents.DocumentSynthesizer`**: synthesise document collections from MongoDB (also Cosmos DB for MongoDB and Amazon DocumentDB), Azure Cosmos DB for NoSQL, Amazon DynamoDB, Couchbase and Extended JSON files, keeping their shape. Extras `relsynth[documents]` (pymongo, azure-cosmos, boto3) and `relsynth[couchbase]`. Guide `docs/documents.md`, ADR 0011, Props. 1400–1405, checks C1400–C1405.
- **Schema inference with counts**: types, presence, array lengths, distinct values and field sets per path; maps keyed by identifiers (with the key rebuilt from a field of the value), variants with a discriminator, length fields, set arrays, references by `_id` or a natural key (partial matches reported as candidates with their coverage) and sequences 1..n within a parent. Exported as an IR 2.0 document with the counts in `x-relsynth-counts` (maps in `x-relsynth-map` until RFC-T13-01 is decided). Every source document validates against the inferred schema (Prop. 1400).
- **Shred, synthesise, nest**: one table per collection, array, map and optional sub-document, with tag columns for absent, null and the type of union fields and length columns modelled with the parent; the relational engine synthesises them and the documents are nested again, with fresh deterministic ObjectIds and references to synthetic documents. `nest(shred(D)) = D` exactly (Prop. 1401).
- **Conformance**: every synthetic document validates against the inferred schema; values that break it are replaced from other synthetic documents (never from the source) and every repair is counted by kind and path (Prop. 1402).
- **Shape metrics** (presence, presence_max, type_mix, array_length, signature, depth, size) with limits calibrated on real-versus-real splits (Prop. 1403), validity and reference integrity.
- **DynamoDB single-table designs**: item types by discriminator or key prefix, key templates (`CUSTOMER#{customer_id}`) learned and rendered back exactly, GSI keys included (Prop. 1404); `examples/documents/dynamodb_shop.py`.
- **MongoDB pushdown**: exact type and length counts from one `$group` pipeline and a `$sample` for the rest of the profile.
- **Acceptance runner** `benchmarks/documents/run_documents.py` (MongoDB sample datasets and a StackExchange dump, downloaded with SHA-256 pins, never committed) and CI job `documents` with a MongoDB 7 service.
- Research spike on a direct document generator (`docs/research/document-generator-spike.md`): no-go for now; parent-conditioned element columns proposed as the follow-up.
#### Autotune, service and CI

- `relsynth autotune --workers N`: the seeds of a candidate run in parallel processes, with the same records and search path as one worker.
- `tools/autotune_bench.py` and the `autotune` workflow: autotune on a bench database (AdventureWorks with its G1-G14 identities as hard constraints) on GitHub runners; the report and a per-candidate summary go to the job log, and a later run resumes the search. First results: [docs/research/autotune-adventureworks.md](docs/research/autotune-adventureworks.md).
- The `autotune` workflow searches 4.5 hours by default, so the held-out confirmation fits the 6-hour job limit.

- **Service end to end on Kubernetes and docker compose** (`service-e2e.yml`): the Helm chart is installed on a kind cluster and the compose stack is started, and the same API flow (sign-in, spec, run, approval, download) runs against both, and a worker killed mid-run must leave its job to the other worker.

### Changed

- **Draws for the same seed are unchanged** with default settings: the new options (`text.method`, `dp.measure`, document collections) are opt-in, and the default text path is byte-identical to 2.0 on every fake scenario.
- README and documentation describe release 2.1: what each new feature does, install extras (`documents`, `couchbase`, `service-otel`), the `docs` and `agent` commands, the new checks and propositions (178 in the proofs, whose PDF is rebuilt), and the security scanning. Proof sections numbered by roadmap version (v2.6, v5.0, T16) are titled v2.1.
- README: "No row is copied" now reads "Rows are drawn from a fitted model; only tables you declare `mode: copy` are copied"; "Privacy checks run by default" now says the copy guards run by default and are not a privacy check; the 2.0 science checks are C600–C606 (privacy attacks), not C700–C706, which do not exist (the generator propositions are tested in `tests/unit/test_generators.py`).
- `docs/security.md` now describes 2.0: differential privacy is optional rather than absent, the distance guards are copy guards rather than privacy checks, seven core dependencies, four `subprocess` call sites, and the service's connections to the OIDC provider and Vault.

### Fixed

- **Helm chart: a fresh `helm install` no longer times out.** The pre-install migrate job ran as the release's ServiceAccount, which Helm only creates after the pre-install hooks; the job now runs as its own hook ServiceAccount (`<release>-migrate`, same annotations), created just before it. Found by the new kind-cluster end-to-end job.
- **Helm chart: workers can store run outputs.** The data PVC was mounted at `/var/lib/relsynth`, so the container runtime created `/var/lib/relsynth/data` (the parent of the work `emptyDir`) as root and the non-root workers failed with `Permission denied: '/var/lib/relsynth/data/tenants'`, losing every synthesis run. The PVC is now mounted at `<data_dir>/tenants`, the only directory the service writes. An install of 2.0.0 that got this far keeps its artefacts under `data/tenants/` on the volume: move that folder's contents to the volume root before upgrading. Found by the kind-cluster end-to-end job.
- Code-quality findings of CodeQL: NaN tests use `math.isnan`; intentional string wraps in lists are parenthesised; files are closed with `with`; uninitialised-variable, redundant-comparison, unused-variable/import and lazy-export findings are resolved. Output is unchanged.
- The text TSTR metric of the scorecard no longer builds a dense class x word table for a condition with very many levels (it needed 32 GB on a Contoso DP run); above 50 M cells it uses an inverted index with the same model. Smaller tables, and so every earlier result, are unchanged.
- The scorecard's naive Bayes accepts labels that mix strings and nulls, and an empty seed table keeps its columns in check enforcement.
- Scorecard: a check rule the source itself breaks (rate above 0) passes within three binomial standard errors of the source rate instead of at the source rate exactly; an exact rule stays exact.
- `structure.structural_prior` is bounded by 0.999 in the settings registry (the spec refuses 1), so autotune no longer proposes a value every run fails on.
- Autotune history: the fingerprint includes an engine signature (relsynth source hash and numpy/pandas/scipy/numba/duckdb/SQLAlchemy/pyarrow versions), so runs made under another engine or library version are not reused. On AdventureWorks a pandas 2.3 incumbent made every pandas 3.1 candidate look like a DimProduct privacy regression.
- Autotune report: a trade-off is reported only when the change in failures is beyond seed noise (one check per seed and twice its paired standard error).
- Autotune tries a rejected set of `relsynth doctor` fixes block by block, so one fix that breaks a constraint no longer blocks the others.

## 2.0.0 (2026-10-31)
Release 2.0 is the first release of the enterprise roadmap: IR 2.0, generator plug-ins, MySQL/MariaDB, cloud warehouses, Iceberg and Delta Lake, privacy attacks, differential privacy, governance, time series, partitioned generation, engine settings with autotune, and the service platform. **Draws for the same seed change** on databases with link tables (see Changed). The proofs PDF is rebuilt with all 2.0 propositions; the partitioned-runtime propositions are renumbered 950–954 (they clashed with DP's 900–904).

### Added

#### Data model: IR 2.0

- **IR 2.0** (`relsynth.ir`): one versioned, modality-neutral document of a data estate: entities with a shape (record, document, node, edge, event) and role; nested objects, arrays with length laws and polymorphic variants with a discriminator; composite, surrogate and natural keys; references, hierarchies, one-to-one and derived links, including from nested paths; graph edges with properties; valid time, event time and SCD2 versioning; containers, partitions with Iceberg-style transforms and sort orders; hard and soft rules; privacy roles, public entities and the privacy unit. Python object model, generated JSON Schema (draft 2020-12) and [reference](docs/ir-reference.md); [guide](docs/ir-guide.md); [ADR 0001](docs/adr/0001-canonical-ir.md); five examples in `examples/ir/` (MongoDB, DynamoDB single table, property graph, Iceberg, event stream).
- **Loss-free 1.0 conversion.** Every command and `Spec` read IR documents as well as 1.0 specs; an IR document lowered for the engine gives the same synthetic data as the 1.0 spec it came from (Proposition 106; all eleven golden specs of the four real databases give byte-identical output).
- **`relsynth ir`** commands: `migrate` (with proof of identity), `validate`, `convert --to ir|v1 [--strict]`, `privacy`, `odcs`, `from-odcs`, `openlineage`, `schema`; `introspect --format ir`.
- **Privacy unit tracing**: reference paths from every entity to the unit, owned/context/unrelated status, cross-unit links, and per-record owners on data (Propositions 107–108).
- **Standards**: ODCS v3.2.0 export and v3.0–3.2 import (exact round trip through `customProperties`); OpenLineage Schema, DatasetType, Documentation, Tags and ColumnLineage facets.
- Extra `relsynth[ir]` (jsonschema) for the JSON Schema check.
- **Generator choice in the IR.** `entities.<name>.synthesis.generator` holds a table's generator (`{name, version, options}`), so 1.0 specs that choose generators still convert without loss. A spec read from an IR document checks generators against its IR version.

#### Connectors, warehouses and lakehouse

- **MySQL 8 and MariaDB connector** (`mysql://`, `mariadb://`, `relsynth[mysql]`): read, write in every mode, `clone-schema`, and fitting inside the database (GROUP BY statistics, window-function rank sums, a seeded SHA-256 hash sample). Text is grouped and compared in a binary collation, so statistics stay exact on case- and accent-insensitive columns. Created columns keep the data: `DOUBLE`, `DATETIME(6)`, `utf8mb4` text with a binary no-pad collation, key columns within MySQL's index limit and wide tables within its row limit. Tested live on MySQL 8.4 and MariaDB 11.4.
- **Databricks SQL connector** (`databricks://`, `relsynth[databricks]`): read, write in every mode, `clone-schema` and fitting inside the warehouse. Sign-in without secrets in the spec (`auth`: token, OAuth M2M and U2M, Entra ID service principal, or the Databricks SDK's credential chain). Created columns keep the data (`DOUBLE`, `STRING`, `TIMESTAMP_NTZ`, Delta column mapping for unusual names); keys are declared where Unity Catalog allows them. Loads by multi-row `INSERT` or staged Parquet and `COPY INTO`. See `docs/warehouses.md`.
- **Microsoft Fabric connector** (`fabric://`, `relsynth[fabric]`): the Fabric Warehouse and Lakehouse SQL endpoints over the ODBC Driver 18. Sign-in with Microsoft Entra ID (`auth`: the Azure credential chain, service principal, Azure CLI, managed identity or browser). Created columns use the types Fabric has (`VARCHAR` sized in UTF-8 bytes, `DATETIME2(6)`, `FLOAT(53)`), keys are declared `NOT ENFORCED`, text statistics stay exact under case-insensitive collations, and the sample hash is the UTF-8 SHA-256 one shared with MySQL, DuckDB and Databricks. Loads by ODBC parameter arrays or Parquet staged in ADLS Gen2 / OneLake and `COPY INTO`.
- **Snowflake connector** (`snowflake://`, `relsynth[snowflake]`): read, write in every mode, `clone-schema` and fitting inside the warehouse. Sign-in with key pairs, programmatic access tokens, OAuth, SSO, workload identity or a `connections.toml` entry. Every session carries the `QUERY_TAG`. Keys are declared (informational), date-times are `TIMESTAMP_NTZ`, and the sample hash is the same SHA-256 one. Loads by multi-row `INSERT` or Parquet `PUT` to a temporary stage and `COPY INTO`.
- **Google BigQuery connector** (`bigquery://`, `relsynth[bigquery]`): read, write and fitting inside the warehouse, signed in with Application Default Credentials, a service-account key, workload identity federation or impersonation. Every job is labelled `relsynth=<tag>` and limited by `maximum_bytes_billed`; reads, samples and statistics are dry-run first. Loads use free load jobs from Parquet. Not yet run against BigQuery itself.
- **Cost guards for cloud warehouses** (`guards`): a query tag on every statement, `max_rows` and `max_bytes` limits checked before a read, and the bytes each table costs in `relsynth plan`.
- **DuckDB for CSV and Parquet folders.** The statistics, first rows and samples of pushed-down facts in a folder source are computed by an in-process DuckDB database running the servers' SQL over views of the files, instead of in pandas. Statistics are exact (normal scores use scipy's quantile inside DuckDB) and CSV typing follows the pandas reader. `source.engine: pandas` keeps the 1.0 path, and `source.duckdb` sets the memory limit, threads and spill folder. `duckdb` and `duckdb-engine` are now dependencies.
- **Apache Iceberg and Delta Lake** (`iceberg:`, `delta:`, `relsynth[iceberg]`, `relsynth[delta]`): read and write lakehouse tables on a local disk, Amazon S3 (and S3-compatible stores) and Azure Data Lake Storage Gen2 / Blob Storage, through any PyIceberg catalog (REST, Glue, Hive, SQL) or delta-rs. A source reads one version per table, pinned with `snapshot` or `as_of` (time travel), and reports the versions it read. Fitting inside the lake runs the pushdown SQL in DuckDB over the Arrow dataset of each version. Writes commit one version per table, parents first, with `relsynth.version` and `relsynth.seed` in the commit and the keys recorded for `introspect`; `target.partition_by` takes Iceberg transforms (`month(c)`, `bucket(n, c)`, ...). Credentials come from the libraries' chains or from variables named in `storage_env`, never from the spec. On Chinook and Contoso a lake source gives the same synthetic database, cell by cell, as SQLite. See `docs/lakehouse.md`.
- **Data contract checks** (`relsynth ir check-contract`, `relsynth.ir.contract_check`): check any data location against an ODCS v3 contract (tables, columns, logical types, required columns, primary and unique keys, ranges, foreign keys, the library metrics `rowCount`, `nullValues` and `duplicateValues`), for example a synthetic lake against the contract exported from the real data.
- **Connector guide** (`docs/connector-sdk.md`): the contract, hooks and tests a new connector needs, with notes for the cloud warehouses.

#### Engine

- **Generator plug-ins** (`relsynth.generators`). A table's attribute columns now come from a generator chosen in the spec. You can set it per table (`tables.<t>.generator`: a name or `{name, version, options}`) or as a default (`synthesis.generator`). The engine keeps keys, links, latent segments, counts and rules, whatever the generator (Props. 700–706).
  - Built-in generators: `copula` (the default; byte-identical to 1.0), `independent` (a latent class model) and `cart` (sequential CART, synthpop style).
  - Third-party generators install as packages through the `relsynth.generators` entry-point group. `relsynth generators list|show` lists them with their capabilities.
  - See `docs/generators.md` and ADR 0002.
- **Compute presets** (`compute.yaml`: `cpu`, `gpu-auto`, `gpu-single`, `gpu-multi`). Set them in `synthesis.compute` with expert overrides (devices, backend, precision, batch size, memory limit, threads, fallback).
  - GPU use is opt-in, the default engine is CPU-only, and every fallback to the CPU is explicit and logged.
  - `relsynth compute` shows the GPUs found and the device each kind of generator would get.
  - See `docs/compute.md`.
- **Degree laws for the keys of link tables** (`synthesis.facts.link_degree`, per fact `fact.link_degree`). With `auto`, each key of a many-to-many table (bonds per second atom, actors per movie) whose degrees prefer a Dirichlet-process or negative-binomial law by the count-model Bayes factor gets it; degrees are drawn given the block totals and matched uniformly within blocks, and repeated pairs or self-links absent from the source are removed by degree-preserving swaps (proofs in `docs/proofs/sec_linkdeg.tex`). Pairs of complete blocks (a product × date grid) and keys whose links sit on fewer than `facts.link_degree_min_rows` (50) effective rows keep the 1.0 law. `auto` is the default; `segment` restores the 1.0 sampler.
- **Partitioned generation** (`synthesis.runtime`, `relsynth.runtime`): entities on the driver, fact rows in partitions run by the `local`, `process` or `spark` executor. Every partition draws from a Philox stream keyed by (seed, dataset, fact, partition), so the output is the same for 1, 8 or 64 workers and for Spark (Prop. 950); partitions keep the model's law (Props. 951, 952) and keys and links hold across partitions (Prop. 953). Facts that cannot be split are run whole, with the reason in the run report. The default (`executor: none`) is the 1.0 sampler, byte for byte. See `docs/runtime.md` and ADR 0003.
- **Partition-aware statistics** (`relsynth.runtime.partition_drift`): tests whether a table's columns change their law across its partitions (Welch ANOVA and chi-square from per-partition sufficient statistics in DuckDB, Holm-adjusted, with an effect-size threshold; Prop. 954), defaulting to the partition fields of an Iceberg or Delta table.
- **Engine settings** (`relsynth.settings`, `relsynth settings show|explain|validate|template|reference`, guide `docs/settings.md`, reference `docs/settings-reference.md`): one registry of 101 engine settings with type, range, meaning and the checks each affects. 24 constants that were fixed in the code are now settings (copula MH steps and MCSE target, marginal and FD sample sizes, occupancy and frailty thresholds, rake limits, the new `repair` and `detection` groups, scorecard thresholds and sample sizes under `evaluation`). Settings come from defaults < a settings file (`synthesis.settings_file`, `--settings`, `RELSYNTH_SETTINGS`, `preset:fast`, `preset:thorough`) < the spec < `--set`; `per_table` overrides the copula, marginal and text groups for one table. In Python, `relsynth.Settings` has editable attributes validated on assignment, `Spec.with_settings()`, and `fit`/`run(settings=)`. With default settings the output is byte for byte unchanged (`tools/identity_check.py`, CI job `settings-autotune`).
- **Self-tuning** (`relsynth autotune`, guide `docs/autotune.md`, research `docs/research/self-tuning.md`): runs the engine on several seeds, separates systematic failures from noise, and changes settings and (optionally) applies doctor spec fixes until the user's objectives hold. Hard constraints (keys, links, rules, trees, privacy, domain identities, any check id) must hold on every seed; statistical families are calibrated (failures at most the 0.99 quantile of their null count). Guided moves from the registry, learned effects from the history, successive halving, paired-seed acceptance and confirmation on held-out seeds. The history (`<spec>.tune/history.jsonl`, aggregates only) persists and is resumed. A constraints report gives, for every objective not met, the reason (no setting, insensitive, conflict, undecided, budget), the seed evidence and the noise floor. Refused in DP mode.

#### Time series

- **Event sequences** (`tables.<t>.sequence`): the times of an entity's events (a customer's orders) are redrawn jointly from a point-process model given the engine's counts: a calendar per segment in operational time, a life-time copula (start and active span given the count) and gamma renewal gaps with an entity shape. Keys, counts, links, other tables and rules are unchanged. See `docs/timeseries.md`, ADR 0004 and Props. 1200-1206.
- **Regular time series** (`tables.<t>.series`): meter-style readings synthesised per entity with daily, weekly and yearly profiles, AR and seasonal-AR noise and a common factor, from a Bayesian mixture over entities.
- **Incremental runs**: `relsynth extend` and `relsynth.extend` add the events after a dataset day by day (or hour, week, month); a stepwise run equals one long run.
- **TimescaleDB**: hypertables noted by `introspect`, and created on load in a PostgreSQL target with the extension (`target.timescale`).
- **Time-series metrics** (`relsynth.timeseries.metrics`): inter-arrival, life-time and calendar checks for sequences; ACF, spectrum, cross-series correlation and forecasting TSTR for series; limits calibrated on real splits, with posterior-predictive and shared-shock corrections. Science checks C1200-C1206; benchmark `benchmarks/timeseries/run_timeseries.py` (UCI electricity load).

#### Privacy, differential privacy and governance

- **`relsynth privacy`** (`relsynth.privacy`): measures privacy risk per person (the privacy unit and the rows it owns) with record isolation, linkage and inference attacks, attribute inference against population data, DOMIAS and shadow-model membership inference, and writes a report for a data protection officer per release context (internal, partner, public; Propositions 600–606, checks C600–C606). Contexts outside the organisation are blocked without differential privacy. Defaults: a *review* needs a named sign-off (`--signed-off-by`) before a gate passes it; internal releases gate on the pooled attack on the most exposed people, partner and public on the single most exposed person with 128 test shadows; a privacy unit of fewer than 1000 people warns and fails every external context without DP. The scorecard's distance-to-closest-record guard stays for continuity but is no longer treated as a privacy check.
- **Privacy link** (RFC T06-02): an optional `privacy` member references the privacy report of roadmap thread T06 by path, sha256 and verdicts; privacy is never folded into the quality vector.
- **Governance core** (`relsynth.governance`, `relsynth gov`, `relsynth[gov]`; roadmap T08, design `docs/governance/design.md`, guide `docs/governance/guide.md`): a local registry with content-addressed objects and a SHA-256 hash-chained log with signed checkpoints; Ed25519 keys and DSSE-signed in-toto statements over RFC 8785 canonical JSON; the organisation policy (`relsynth.org-policy/1`, no built-in ε) with ε caps converted to zCDP ρ; the privacy budget ledger per domain with reserve, touch, settle and refund (Props. 800–802); release rules in `release.rego` with a built-in Python engine and an optional OPA engine, plus four invariants in code; approvals with roles, quorum, separation of duties and expiry; signed release tokens and release packages; `gov init, keys, policy, ls, show, budget, approve, release, revoke, verify, checkpoint`.
- `relsynth run --registry --context --release-target --source-id --key [--assess | --privacy-report]` records a signed manifest of the run; a context outside the organisation is refused before any data is read unless the run is differentially private.
- Checks C800–C803 now test the shipped code; a CI job runs `opa test` and checks that both policy engines agree.
- **Differential privacy mode** (`relsynth.dp`, `relsynth run --dp | --epsilon`, `relsynth[dp]`; roadmap T07, guide `docs/dp.md`). It gives a user-level (ε, δ) guarantee for one person of the privacy unit and all their rows.
  - Stage 1 releases integer statistics with discrete Gaussian noise from OpenDP: ranges on a fixed grid, category levels by partition selection, histograms, rows per parent row and correlations of normal scores. It bounds each person's contribution first, with private per-parent caps.
  - Stage 2 fits the ordinary engine to the decoded seed database.
  - Every run writes a DP statement (`relsynth.dp-statement/1`, NIST SP 800-226). There is no default ε, and δ is at most 10⁻⁵.
  - Governed runs reserve, touch and settle the domain's budget. The manifest carries the charges, and external releases work end to end.
  - Props. 900–906 with checks C900–C906.
- `relsynth privacy` audits a DP spec through the DP pipeline and compares the attacks with the bound that ε implies.

#### Quality and benchmarks

- **Quality report 2.0** (`relsynth.assess_quality`, `relsynth evaluate --quality FILE`): a versioned, machine-readable report (JSON Schema `quality-report-2.0`) with every check's metric, scope, value, limit, tolerance, null calibration, level and confidence, grouped in a quality vector (fidelity, utility, integrity) with a calibrated pass rule per area and no opaque overall score. The 1.0 scorecard is carried over unchanged.
- **Opt-in 2.0 quality gate**: `relsynth evaluate --quality FILE --quality-gate` exits 1 when an area of the quality vector fails. The default gate stays `--fail-under` until 3.0. Utility targets of the golden databases are declared in `benchmarks/targets.yaml`.
- **New metrics**: Wasserstein-1 on the raw and rank-normal scales, Jensen–Shannon distance, 2- and 3-way marginal TVD, alpha-precision and beta-recall, TSTR/TRTS/TRTR, fan-out laws (W1, childless share, dispersion) and two-hop fan-out, k-hop column-pair trends, aggregation detection C2ST-Agg (logistic, or XGBoost with `relsynth[benchmark]`), and a multi-table utility task. Definitions in `docs/metrics.md`, generated from `quality/registry.py`.
- **Calibrated limits**: each new check's limit is a prediction bound from real-versus-real splits that keep families together, rescaled to the compared sizes (Propositions 109–112, checks C79–C82). A synthetic database drawn at another size is compared on per-parent counts divided by the expected change of rows per parent under the engine's scale rule (`scale` setting, defaulting to the spec's `synthesis.scale` in `assess_quality`; Proposition 113).
- **SyntheRela benchmark** script and results (`benchmarks/syntherela/`), and precise definitions of the roadmap's acceptance metrics (`docs/acceptance-metrics.md`).
- **Benchmark harness: two more databases and the privacy and 2.0 quality suites.** `python -m bench` now runs Berka (the PKDD'99 Czech bank, 8 tables, 1.08M rows) and the NIST CRC ACS PUMS excerpt (27,253 people), a Chinook variant at scale 2, and for every database the calibrated 2.0 quality report, the T06 privacy attacks (where a database has people), a determinism check and the generation cost. New drift rule R4 flags tracked values (SyntheRela summaries, area pass rates, utility gap, attack values) that move the wrong way beyond the seed spread. `--suites` chooses what runs; `compare` accepts the records of one run split over several jobs, and `bench.yml` runs one job per database. See `docs/development/benchmarks.md`.
- **Profile of 1.0** (`docs/profiling.md`): time, memory and I/O by stage on Contoso and AdventureWorks.

#### Service platform

- **relsynth service** (`relsynth[service]`, `relsynth-service serve|worker`): a REST API with OpenAPI (projects, connections, specs, runs, reports, approvals, audit, service accounts) and a web UI, so an organisation can run relsynth for many teams. Single sign-on with OIDC (SAML through an OIDC broker), roles per tenant (data owner, privacy officer, engineer, consumer, tenant admin), optional ABAC on connection classification, and service accounts. A durable job queue in the metadata store with leases, retries, cancellation and per-job limits; runs survive the loss of a worker. Tenants are isolated in the API, the store, the files and the keys; every artefact is encrypted (AES-256-GCM, per-tenant key-encryption keys, Vault or cloud KMS); source credentials are only secret references; no source row is persisted. Approval by a privacy officer or data owner before download, with separation of duties; external releases wait for differential privacy. Hash-chained audit log per tenant, JSON logs, Prometheus metrics and OpenTelemetry traces. See `docs/service/`.
- **Deployment**: one container image for API and workers (non-root, any UID, read-only root filesystem), docker compose with Keycloak, a Helm chart for Kubernetes and OpenShift, and an air-gapped bundle (`tools/make_airgap_bundle.py`). Guides for AWS, Azure and GCP accounts, operations, a STRIDE threat model, an OWASP ASVS level 2 checklist, a penetration test plan and a SOC 2 / ISO 27001 / NIST 800-53 control mapping.

### Changed

- The ordinary pipeline, `tune`, `verify`, `export-model` and `doctor --apply` refuse a spec in DP mode instead of reading it without DP.
- `relsynth introspect --format` takes `auto`, the new default: it writes an IR document from release 2.0 and a 1.0 spec before.
- `relsynth doctor --apply` and `relsynth tune` write an IR document when they are given one, merging their changes into it so its annotations are kept.
- **Synthetic draws for the same seed change** on databases with link tables, because `synthesis.facts.link_degree` defaults to `auto` (see Added). On the golden databases: Chinook's `InvoiceLine.TrackId` and `PlaylistTrack.TrackId`, and AdventureWorks' internet-sales order key. Set `link_degree: segment` for the 1.0 draws. `python -m bench run|all --set KEY=VALUE` passes spec overrides to every run.

### Deprecated

- 2.x reads 1.0 specs and IR 2.x documents alike. Reading a 1.0 spec warns from 3.0 at the earliest and stops no earlier than 4.0 and 24 months after 2.0. `relsynth ir migrate` stays in every release.

## 1.0.1 (2026-10-28)

Synthetic draws for the same seed are unchanged from 1.0.0.

### Changed

- relsynth is open source under the Apache License 2.0 (previously all rights reserved). The repository has LICENSE, NOTICE, a contributor guide with DCO sign-off, a code of conduct, a security policy with private vulnerability reporting, and a dataset register (`docs/datasets.md`).

### Fixed

- **Sampling no longer fails when a rule reads a parent's aggregated column through a link** (an introspect draft of the Berka bank: `account.date` = min(date) of the transactions, with checks `days(account_id.date) < days(date)` on loans and transactions). The check is evaluated after the aggregates are computed, like rules that read an aggregate of their own table. Other specs draw the same data.

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

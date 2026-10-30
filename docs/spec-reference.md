# The model specification: reference and recipes (relsynth v0.9)

The specification ("spec") is one YAML file. It describes the source database (tables, columns, keys, links), the business model on top of it (roles, hierarchies, rules), and how to synthesise it (what to copy, what to draw, how many rows, which statistical settings). It is the only input the engine needs besides the data.

This page has five parts:

1. how a spec is made and used;
2. a complete annotated example;
3. recipes for common needs;
4. what only you can tell the engine (the guidance a spec should carry);
5. the full key reference, generated from the code's schema, so it always matches what the engine reads.

## 1. How a spec is made and used

```
python -m relsynth introspect --source sqlite:shop.db --out spec.yaml   # draft spec + spec.model.md
python -m relsynth model spec.yaml                                      # data-model report (again, after edits)
python -m relsynth validate spec.yaml                                   # strict check, with "did you mean" hints
python -m relsynth run spec.yaml --datasets 3 --bundle diag.json        # synthesise and evaluate
python -m relsynth doctor diag.json --spec spec.yaml                    # proposed fixes
python -m relsynth export-model spec.yaml --out model/                  # the fitted model, verified, as json/md/pdf/py
```

- **`introspect`** reads the catalog (or files) and drafts everything it can prove from the data. Each inference holds on at least 99.9 % of at least 30 rows and passes a test against chance. Its `review:` list says what it found. Lines starting with `ASK` need your answer.
- **`spec.model.md`** is the data-model report. It shows:
  - the join graph as a Mermaid ER diagram;
  - each fact's kind (transaction, header and lines, periodic or accumulating snapshot, factless, bridge), and whether it is a star or a snowflake;
  - dimensions and their kinds (calendar, reference, SCD type 2, role-playing, conformed, outrigger, parent-child, unknown members);
  - hierarchies, degenerate dimensions, audit columns, rules, computed columns, structural effects and open questions.
- **`validate`** rejects unknown keys anywhere in the file, so a typo is never ignored silently. For example, `colums` gives *did you mean 'columns'?*.
- **`introspect --evidence data`** identifies keys and links from the values alone; the default `names+data` also weighs column and table names by how rare each word is in this database. `--ignore-declared` hides the catalog's keys and links (to test identification against them). Inclusion counts as evidence only when it is unlikely by chance (an exact hypergeometric probability below 1e-6); a self-reference only when a random column would form a forest with probability at most 0.01. Everything weaker becomes an `ASK`.
- **`run`** fits the Bayesian model and writes the synthetic data, with the same tables and columns as the source.
- **`export-model`** fits the model and writes it as a mathematical object you can inspect and re-run on your own:
  - `model.json`: the structure, the posterior mean and `--draws` posterior draws of every parameter block (draw *m* is exactly the parameter set of dataset *m* of `run`);
  - `model.md` / `model.pdf`: the generative model in full, this database's numbers and a verification table. Each parameter set is checked against the conditions of the propositions it relies on (simplex and support, positive semi-definite correlations, the scale rule, stationary chains) before anything is written; sampler warnings name the columns concerned and the setting that fixes them;
  - `replicate.py` (with `relsynth_expr.py`): a standalone sampler of the core generative process needing only numpy, pandas and scipy; `model.json` lists under `replica_scope` what it leaves to the engine;
  - `run_model.py`: the engine's own calls, reproducing `run` byte for byte.

  Options: `--formats json,md,pdf,py`, `--draws 5`, `--no-replica-check` (skip scoring the replica against the source), `--stan` (re-check the posterior blocks with Stan when it is installed).

The draft runs as it is. You edit only what the data cannot reveal (part 4).

## 2. A complete annotated example

```yaml
version: 1
name: retail
source: {kind: sqlite, path: shop.db}            # or {kind: csv, path: ./data} or {kind: sqlalchemy, url: ..., schema: ...}
target: {kind: sqlalchemy, url: "postgresql+psycopg2://u@h/db", schema: synth, if_exists: append}

tables:
  calendar:                                       # a calendar: copied as it is
    mode: copy
    primary_key: date_key
    columns: {date_key: id, date: date, month: integer, year: integer}

  region:                                         # small public reference table: copied
    mode: copy
    primary_key: region_id
    columns: {region_id: id, country: categorical, area: categorical}

  customer:
    primary_key: customer_id
    unknown_members: [-1]                         # row -1 'Unknown' that facts point at: kept, never modelled
    dimensions:                                   # levels implied inside the table
    - {key: geo_code, columns: [state, country], anchor: country}
    columns:
      customer_id: id
      region_id: id
      geo_code: integer
      state: categorical
      country: categorical
      birthday: date
      age: integer                                # an age on the extract date (soft rule below)
      email: {type: pii, pii: {generator: email}}
      segment: {type: categorical, distribution: {kind: categorical, probabilities: {retail: 0.8, pro: 0.2}}}

  product:                                        # SCD type 2: rows are versions of a product code
    primary_key: product_key
    scd: {type: 2, natural_key: product_code, valid_from: start_date, valid_to: end_date,
          current: {column: status, value: Current, other: null}}
    columns:
      product_key: id
      product_code: id
      category: categorical
      list_price: float
      start_date: date
      end_date: date
      status: categorical

  store: {mode: copy, primary_key: store_id, columns: {store_id: id, region_id: id, country: categorical}}

  orders:                                         # order header: one row per order
    primary_key: order_id
    key_order: order_date                         # order numbers follow order dates
    link_affinity:                                # customers shop in their own area more often than chance
    - match: [customer_id.region_id, store_id.region_id]
    columns: {order_id: id, customer_id: id, store_id: id, order_date: date, n_lines: integer,
              delivery_date: {type: date, offset_from: order_date}}

  order_line:
    role: fact
    primary_key: line_id
    unique: [[order_id, line_no]]
    fact: {anchor: order_id, partner: product_key}  # anchor by key column (or a relationship name or table)
    group_effects:                                # one price factor per category and day
    - {column: unit_price, by: [product_key.category, order_id.order_date]}
    columns:
      line_id: id
      order_id: id
      line_no: {type: sequence, within: order_id}
      product_key: id
      quantity: {type: integer, min: 1}
      unit_price: {type: float, ratio_of: product_key.list_price}
      discount: integer
      net_price: float

  customer_account:                               # bridge: allocation weights sum to 1 per customer
    role: fact
    fact: {anchor: customer, partner: account}
    bridge: {group: customer_id, weight: weight, decimals: 4}
    columns: {customer_id: id, account_id: id, weight: float}

  account: {primary_key: account_id, columns: {account_id: id, kind: categorical}}

  sales_view: {role: view, view_of: order_line, columns: {line_id: id, order_id: id, quantity: integer}}

relationships:
- {child: customer, child_key: region_id, parent: region}
- {child: orders, child_key: customer_id, parent: customer}
- {child: orders, child_key: store_id, parent: store}
- {child: store, child_key: region_id, parent: region}
- {child: order_line, child_key: order_id, parent: orders}
- {child: order_line, child_key: product_key, parent: product}
- {child: customer_account, child_key: customer_id, parent: customer}
- {child: customer_account, child_key: account_id, parent: account}

rules:
- {table: order_line, type: derive, expr: "net_price = unit_price * (1 - discount / 100)", tolerance: 0.01}
- {table: orders, type: check, expr: "customer_id.country == store_id.country"}
- {table: customer, type: derive, expr: "age = years_between(birthday, '2021-02-02')", holds: 0.994, violation: offset}
- {table: orders, type: aggregate, column: n_lines, agg: count, from: order_line}

synthesis:
  seed: 7
  posterior: {mode: draw, datasets: 3}           # three datasets, each one posterior draw
  scale: {default: 1.0, tables: {customer: 10}}   # 10x customers; their orders and lines follow
  segments: {default_cap: 8}
```

## 3. Recipes

### Copy every dimension, synthesise only the facts

Set `mode: copy` on every entity and leave the facts as `role: fact`. The dimensions are written as they are. The facts are drawn from the model fitted on the real facts, so their counts per dimension row, their measures and their rules follow the source, while every fact row is new. Large copied tables keep one latent segment per row when the budget allows (`segments.copied_budget_cells`). Otherwise their rows are clustered, and each row keeps its own share of links. Check C66 tests this recipe.

```yaml
tables:
  customer: {mode: copy, primary_key: customer_id, columns: {...}}
  product:  {mode: copy, primary_key: product_id, columns: {...}}
  sales:    {role: fact, fact: {anchor: customer, partner: product}, columns: {...}}
```

Copied rows are real rows, so do this only when the dimensions may be shared.

### Simulate one dimension in a specific way

- **Keep it as it is:** `mode: copy`.
- **Draw some of its columns from a law you state:** `distribution` on the column. The kinds are:
  - `fixed`;
  - `categorical` with `probabilities`;
  - `uniform`, `normal`, `lognormal`, `poisson`;
  - `date_range`;
  - `sequence`;
  - `empirical`, which keeps the source's marginal and drops its dependence on the other columns.

  A `distribution` column is drawn independently of the other columns. Derived columns, rules and lookups that read it are computed from the drawn value.
- **Give it a different size:** `synthesis.scale.tables.<table>: factor`. Facts anchored on it follow, and so do entities below it that have no factor of their own (an order table that links to customers). Give such a table its own factor to keep it at another size.
- **Stop it mixing groups:** `synthesis.segments.attribute_strata: {table: column}`. Its segments are nested in the levels of that column (only sales staff have quotas).
- **Model it more coarsely or more finely:** `synthesis.segments.tables.<table>: cap`.

```yaml
customer:
  columns:
    segment: {type: categorical, distribution: {kind: categorical, probabilities: {retail: 0.7, pro: 0.3}}}
    signup:  {type: date, distribution: {kind: date_range, start: 2020-01-01, end: 2024-12-31}}
    credit:  {type: float, distribution: {kind: lognormal, mean: 8, sd: 0.5, decimals: 2}}
```

### Test data at another scale, or several datasets

- **More rows:** `synthesis.scale.default: 10` multiplies entities. Rows per parent stay the source's, so a 10× customer table gets about 10× the orders.
- **Several independent datasets:** set `synthesis.posterior.datasets: m`, or pass `--datasets m`. Each dataset is drawn with its own posterior draw of every parameter. This is the posterior predictive synthesis that inference from synthetic data needs (Raghunathan, Reiter & Rubin 2003).
- **The same data on every run:** set `synthesis.posterior.mode: mean` (or pass `--mode mean`). Every parameter is set to its posterior mean, which suits regression tests (Reiter & Kinney 2012).

### Slowly changing dimension (type 2)

```yaml
product:
  scd: {type: 2, natural_key: product_code, valid_from: start_date, valid_to: end_date}
```

Introspect proposes it when a code repeats on a few rows with different start dates and at most one open version (a null end) per code, the latest one. The engine draws:

- the number of versions per natural key;
- one new code per key;
- columns that never change across versions, repeated from the first version (`stable`, detected or listed);
- strictly increasing start dates;
- each end date as the next start plus the source's gap (−1 day in a clean history);
- the open (current) version: the row the model drew with no end date becomes the latest version of its key, so the facts that point at current versions keep doing so; then the current flag.

Facts keep pointing at version rows, and date-window rules (`days(order_date) >= days(product_key.start_date)`) are enforced when declared. Types 0 and 1 need no setting: they have one row per key.

### Role-playing dimension

A calendar referenced as order date and ship date is two relationships to one table. Each is its own role in the model report. Two things help:

- `fact.anchor: OrderDateKey` picks the role that drives row counts.
- `offset_from` on the second relationship models the gap (ship date = order date + 7 days) instead of two independent dates.

### Order header and lines

An order header carried on its lines (a degenerate dimension) is `fact.header: {key, columns, line_number}`. A separate orders table that the lines reference is an entity; the model report calls it the order header. Every order then has one customer and one set of dates, and its lines are numbered 1..n.

### Unknown members

`unknown_members: [-1]` on the dimension:

- the −1 row is copied as it is;
- the model learns only the real members;
- references to −1 come back at the source's share, drawn from its Beta posterior.

On a fact, the rows that point at the unknown member are restored by drawing that share of rows from the synthetic fact. This assumes the member is missing at random given the other columns. If the unknown member means something specific in your data, say so in the spec by modelling it as a real row (remove `unknown_members`).

### Bridge table (multivalued dimension)

`bridge: {group: customer_id, weight: weight}`. The weights of each group are rescaled to sum to 1 (or `total`). The rounding remainder goes on the largest weight, so groups sum exactly at the stored precision.

### Hierarchies

- **Snowflake chain** (product → subcategory → category): each level is its own table with a many-to-one relationship. Nothing else is needed.
- **Levels inside one table** (`geo_code` determines state and country): `dimensions: [{key, columns, anchor}]`. The combinations are copied and the levels always agree.
- **Parent-child tree** (employee → manager): `{child: employee, child_key: manager_id, parent: employee, kind: self_reference}`. Depth and branching follow the source.

### Periodic and accumulating snapshots, factless facts

- **Periodic snapshot** (one row per account and month): declare the grain as `unique: [[account_id, month_key]]`. The fact is then generated as the occupancy of the pairs.
- **Accumulating snapshot** (order, ship and delivery dates on one row): chain the dates with `offset_from` (on relationships for date keys, on columns for dates).
- **Factless fact** (attendance, coverage): it needs nothing special. Rows per parent and the pairs are modelled as for any fact.

### A value shared by a group, co-location, rules that mostly hold

- `group_effects: [{column, by}]`: one value per group (a daily price factor per category). Introspect finds it when the value is constant in at least 99.9 % of rows in groups of 2+ rows, beyond chance.
- `link_affinity: [{match: [a.col, b.col]}]`: two links point at matching parents more often than chance (a customer's store in the customer's region). The share is the source's, drawn from its posterior. You can set it with `share`.
- `holds: 0.99` on a derive rule: the rule holds on that share of rows, and the other rows take the source's offsets from it (`violation: offset`) or their modelled value (`violation: model`).

### Writing into a database with the original structure

Create the target schema from the source DDL, then use `target: {kind: sqlalchemy, url: ..., schema: synth, if_exists: append}`. Tables are loaded parents first, so foreign-key constraints are satisfied. `if_exists: replace` (the default) creates the tables from the data instead, without constraints.

## 4. What only you can tell the engine

The data cannot settle these points. Introspect marks them `ASK` or proposes a default that you should confirm:

| Question | Where it goes | Default when unanswered |
|---|---|---|
| Which tables may be copied (public reference data) and which are confidential | `mode` | copy for code lists, calendars and small tables whose parents are copied; never for people |
| Which parent drives the volume of a fact (rows per customer or per product) | `fact.anchor` | the parent with the most rows |
| What a key with no clear parent references | `relationships` | left as a plain column |
| Whether an unknown member is "missing at random" or a meaningful category | `unknown_members` | missing at random |
| The reference date of an age or tenure when the source does not pin it down | the soft rule's expression | the date covered by most rows |
| Whether a share that only *mostly* holds is a rule or a coincidence | `holds`, `group_effects`, `link_affinity` | kept when it is beyond chance at alpha 1e-9 |
| Which columns of an SCD2 dimension change across versions | `scd.stable` | columns equal on at least 99 % of keys are stable |
| Target size, number of datasets, stable or varying runs | `synthesis.scale`, `synthesis.posterior` | 1×, one dataset, a posterior draw |
| Columns that must follow a law you know rather than the data | `distribution` | learned from the data |

## 5. Key reference

Every key below is checked by `validate`. Any other key is an error, with the closest valid key as a hint. Defaults apply when a key is absent.

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `version` | int | `1` | Format version of the file. |
| `name` | str | `draft` | Label used in reports and file names. |
| `description` | str |  | Free text about the model (kept, not read). |
| `source` | source |  | Where the real data is read from (required for `run`). |
| `target` | target | `csv into ./synthetic` | Where the synthetic data is written. |
| `tables` | map of table |  | Every table: role, columns, keys and modelling options. |
| `relationships` | list of relationship | `[]` | Foreign keys (declared or inferred). |
| `rules` | list of rule | `[]` | Business rules: checks, derived columns, aggregates, running balances. |
| `synthesis` | synthesis | `defaults` | Method and statistical parameters (all optional). |
| `review` | list of str | `[]` | What introspect inferred, and its questions (lines starting ASK). |

### `source`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | csv | parquet | files | iceberg | delta | sqlite | postgresql | mssql | oracle | mysql | mariadb | databricks | fabric | snowflake | bigquery | sqlalchemy | `csv` | csv / parquet / files: one <table>.csv, .csv.gz or .parquet file per table in `path`; sqlite: `path` to a file; iceberg: `catalog` (PyIceberg catalog properties) or a local `path` (a SQL catalog in <path>/catalog.db), tables in namespace `schema`; delta: `path`, a folder or object-store URI (s3://, abfss://) with one folder per table; postgresql, mssql, oracle, mysql, mariadb, databricks, fabric, snowflake, bigquery (or sqlalchemy for any other engine): `url`. |
| `path` | str |  | Folder (files) or file (sqlite); relative to the spec's folder. |
| `url` | str |  | Database URL, e.g. postgresql://user@host/db, mssql+pyodbc://user@host/db?driver=ODBC+Driver+18+for+SQL+Server, oracle://user@host:1521/?service_name=ORCLPDB1 (driver chosen when omitted). |
| `url_env` | str |  | Environment variable that holds the URL (with its password), used instead of `url`: keeps the password out of the file. Specs written for sharing (introspect drafts, exported spec.yaml, bundles) show the URL with *** for the password and set url_env (RELSYNTH_SOURCE_URL). |
| `password_env` | str |  | Environment variable that holds only the password; the URL then carries user, host and database. |
| `schema` | str |  | Database schema to read or write (Oracle: the owning user). |
| `include` | list of str |  | Table-name patterns (regular expressions) to read. |
| `exclude` | list of str |  | Table-name patterns to leave out. |
| `null_values` | list of str | `['']` | CSV fields read as null. Only an empty field by default, so 'NA' (Namibia), 'None' and 'null' stay values; e.g. ['', 'NULL', 'n/a']. |
| `connect_args` | map |  | Driver connection arguments (TLS settings, timeouts), passed to the driver as they are. |
| `engine_options` | map |  | SQLAlchemy engine options (pool_size, pool_recycle, ...). |
| `auth` | map |  | Cloud warehouses: how to sign in, without secrets in the file. `method` (databricks: token, oauth-m2m, azure-sp, oauth-u2m, azure-u2m, default; fabric: default, azure-sp, azure-cli, managed-identity, interactive, sql-login; snowflake: key-pair, pat, oauth, password, sso, workload-identity, default; bigquery: default, service-account, external-account, impersonate) and the environment variables that hold the secrets as <name>_env (token_env, client_id_env, client_secret_env, tenant_id_env, password_env, private_key_passphrase_env) or key files as <name>_file. Without it: databricks uses DATABRICKS_TOKEN when set, else the platform's credential chain; fabric uses the Azure credential chain; snowflake uses the URL's password; bigquery uses Application Default Credentials. |
| `guards` | map |  | Cloud warehouses: cost guards. query_tag (default relsynth: marks every query in the warehouse's history), max_rows (largest table read into memory), max_bytes (largest estimated scan of one statement, e.g. 100GB). `relsynth plan` reports the bytes each table costs. |
| `bulk` | map |  | Cloud warehouse targets: the load path. databricks: {method: insert} (multi-row INSERT, the default), {method: volume, volume: /Volumes/cat/schema/vol} (Parquet staged in a Unity Catalog volume, then COPY INTO) or {method: path, path: <location the warehouse reads>}. fabric: {method: insert} (ODBC parameter arrays, the default) or {method: copy, stage: https://<account>.dfs.core.windows.net/<container>/<dir>} (Parquet staged in ADLS Gen2 or OneLake, then COPY INTO). snowflake: {method: insert} (multi-row INSERT, the default) or {method: stage} (Parquet PUT in a temporary stage, then COPY INTO). bigquery: {method: load} (load jobs from Parquet, the default) or {method: insert} (DML INSERT). |
| `catalog` | map |  | Iceberg: the properties of a PyIceberg catalog (type: rest | glue | hive | sql | ..., uri, warehouse, name). Secrets come from the environment (storage_env, or the platform's credential chain). |
| `storage` | map |  | Iceberg and Delta on object storage: storage options without secrets, e.g. {s3.endpoint: ..., s3.region: ...} (Iceberg) or {AWS_REGION: ..., AZURE_STORAGE_ACCOUNT_NAME: ...} (Delta). |
| `storage_env` | map |  | Iceberg and Delta: storage option -> environment variable holding its value (keys and tokens stay out of the file). |
| `snapshot` | int or map |  | Iceberg and Delta sources: the version to read (Iceberg snapshot id, Delta version), one value or table -> value. Default: the current version; the bundle records what was read. |
| `as_of` | str |  | Iceberg and Delta sources: read every table as it was at this ISO time (one consistent instant for all tables). |
| `format` | csv | parquet | `csv` | File format written (target of kind files). |
| `engine` | auto | duckdb | pandas | `auto` | Folders and lakehouse tables: what computes the statistics and samples of pushed-down facts. duckdb runs SQL over the files without loading them; pandas reads the needed columns into memory (the 1.0 path); auto is duckdb when it is installed. |
| `duckdb` | map |  | Folders, engine duckdb: memory_limit (e.g. '4GB'), threads, temp_directory (where larger-than-memory sorts and groupings spill). |

### `target`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | csv | parquet | files | iceberg | delta | sqlite | postgresql | mssql | oracle | mysql | mariadb | databricks | fabric | snowflake | bigquery | sqlalchemy | `csv` | As for source; csv and parquet write one file per table. |
| `path` | str | `synthetic` | Output folder or SQLite file; dataset i > 1 goes to <path>_i. |
| `url` | str |  | URL of the target database. |
| `url_env` | str |  | As for source (RELSYNTH_TARGET_URL in shared specs). |
| `password_env` | str |  | As for source. |
| `schema` | str |  | Target schema (created when missing on PostgreSQL and SQL Server); dataset i > 1 goes to <schema>_i. |
| `if_exists` | replace | append | truncate | fail | `replace` | replace: drop and create the tables; append: insert into existing tables; truncate: delete the rows of the existing tables (children first) and insert, keeping the target's DDL and constraints; fail: stop when a table exists. |
| `structure` | auto | clone | infer | existing | `auto` | How target tables are created. clone: the source's DDL (column types, nullability, primary, unique and foreign keys, indexes, identity columns) when source and target are databases; infer: from the specification (keys and links as constraints) and the data's types; existing: tables created beforehand. auto: clone when possible, else infer. |
| `connect_args` | map |  | As for source. |
| `engine_options` | map |  | As for source. |
| `auth` | map |  | As for source. |
| `guards` | map |  | As for source. |
| `bulk` | map |  | As for source. |
| `catalog` | map |  | As for source. |
| `storage` | map |  | As for source. |
| `storage_env` | map |  | As for source. |
| `partition_by` | map |  | Iceberg and Delta targets: table -> list of partition expressions. Iceberg: column, year(c), month(c), day(c), hour(c), bucket(n, c), truncate(n, c); Delta: columns. |
| `format` | csv | parquet | `csv` | File format of a files target. |

### `tables.<name>`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `role` | entity | fact | view | `entity` | entity: a thing (customer, product, order header); fact: an event or link whose rows exist because of its parents (sales lines, visits); view: a table that is the join of others (rebuilt, never modelled). |
| `mode` | synthesize | copy | `synthesize` | copy keeps the table exactly (reference data, calendars, code lists, or any dimension you want to keep: see the recipe 'copy dimensions, synthesise facts'). |
| `primary_key` | column |  | Key column; synthetic keys are new values (1..n for integers). |
| `columns` | map of column |  | Column name -> type, or -> {type, options}. |
| `unique` | list of keys | `[]` | Keys no two rows share: [a, b], or on facts {columns, where} (a filtered unique index). A key of two foreign keys makes the table a grain (occupancy of pairs). |
| `partial_unique` | list of {columns, where} | `[]` | Same as a filtered `unique` entry. |
| `fact` | fact | `{}` | How fact rows are generated (anchor, partner, header, time). |
| `dimensions` | list of dimension | `[]` | Levels implied inside an entity: a code column that determines others (GeoAreaKey -> State, Country, Continent), generated through a copied level table. |
| `key_order` | column |  | Keys numbered in the order of this column (order numbers follow order dates). |
| `view_of` | table |  | For role view: the base table the view is rebuilt from. |
| `group_effects` | list of group_effect | `[]` | Columns shared by all rows of a group (one price factor per category and day): one value drawn per group. |
| `link_affinity` | list of link_affinity | `[]` | Two links that point at matching parents more often than chance (customer and store in one area). |
| `scd` | scd |  | Slowly changing dimension type 2: rows are versions of a natural key. |
| `unknown_members` | list of key values |  | Rows that stand for 'no member' (key -1 'Unknown'): copied, kept out of the model, referenced with the source's share. |
| `bridge` | bridge |  | Weights that sum to 1 (or `total`) over the rows of a group (multivalued dimension / allocation). |
| `generator` | name | generator | `synthesis.generator` | Generator plug-in of the table's attribute columns: a name (copula, independent, cart, or an installed plug-in) or {name, version, options}. See docs/generators.md. |
| `description` | str |  | Free text (kept, not read). |

### generator

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `copula` | Registered generator name (`relsynth generators list`). |
| `version` | version specifier |  | Pin of the generator's model version, e.g. '>=1.0,<2' or '==1.0'. |
| `options` | map | `{}` | Options passed to the generator (its own documentation lists them). |

### `tables.<name>.fact`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `anchor` | table | relationship name | key column | `first parent` | Parent whose rows generate the fact rows (rows per customer); scaling it scales the fact. A key column (OrderDateKey) or a relationship name picks one role of a role-playing dimension. |
| `partner` | table | relationship name | key column | `next parent` | Second dimension of the block model (customer segment x product segment rates). |
| `pairs` | auto | poisson | occupancy | `auto` | Rows per (anchor, partner) pair as counts, or each pair present or absent. |
| `multiplicity` | auto | none | partner | anchor | block | `auto` | For occupancy facts: rows per present pair beyond the first, grouped by. |
| `count_model` | auto | poisson | dp | `auto` | Rows per anchor: Poisson/negative binomial or a Dirichlet-process count law. |
| `group_fallback` | auto | anchor | partner | `auto` | Which dimension's marginals a sparse block borrows. |
| `link_degree` | auto | segment | dp | `auto` | Degree law of the two keys of a link table (bonds per atom as the second atom, actors per movie). auto: a key whose degrees prefer a Dirichlet-process or negative-binomial law by the count-model Bayes factor gets it, with the block totals kept (unless its links sit on fewer than facts.link_degree_min_rows effective rows). segment: rows spread within segments as in 1.0. dp: the DP law on every modelled key. |
| `time_column` | column |  | Date column with a seasonal profile per segment. |
| `time_bucket` | day | week | month | quarter | year | `month` | Bucket of the seasonal profile. |
| `condition_on_parents` | bool | list of fk.col | `True` | Condition measures on parents' continuous attributes (unit price follows list price). |
| `header` | header |  | Order header carried on the lines: {key, columns, line_number}. |

### `fact.header`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `key` | column |  | Document number (SalesOrderNumber). |
| `columns` | list of column |  | Columns the document number determines (customer, dates, currency). |
| `line_number` | column |  | Line number column, generated 1..n per document. |

### `tables.<name>.dimensions[]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `key` | column |  | The code (GeoAreaKey). |
| `columns` | list of column |  | Columns it determines (State, Country, Continent). |
| `anchor` | column |  | A coarse determined column kept as a modelled attribute (Country). |

### `tables.<name>.columns.<name>`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | id | integer | float | categorical | boolean | date | datetime | text | pii | lookup | sequence | constant | binary |  | What the column holds (see the column types table). |
| `nullable` | bool | `True` | Informational; null shares are always learned from the data. |
| `min` | number |  | Hard lower bound of a number column. |
| `max` | number |  | Hard upper bound. |
| `decimals` | int | `from the source` | Stored precision of a number column. |
| `lookup` | fk.col path | keyed lookup |  | For type lookup: the value of a parent's column, or {table, match, value} into a reference table. |
| `map` | map |  | Lookup values mapped through a fixed table (country -> currency). |
| `default` | any |  | Value for codes outside `map`. |
| `offset_from` | column |  | This date/number = that column + a modelled gap (delivery = order + 0..7 days). |
| `ratio_of` | column | fk.col |  | This number = that number x a modelled factor (unit price = list price x one of 16 factors). |
| `ratio_same_as` | column |  | Use the factor of another ratio column (unit cost moves with unit price). |
| `text` | text |  | Options of a text column. |
| `pii` | pii |  | Options of a pii column. |
| `within` | column |  | For type sequence: numbered 1..n within each value of this column. |
| `start` | int | `1` | For type sequence: first number. |
| `base_column` | column |  | For views: the base-table column this view column copies. |
| `distribution` | distribution |  | Draw the column from a stated law instead of the learned one. |
| `description` | str |  | Free text (kept, not read). |

### `columns.<name>.text`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `condition_on` | column |  | Column the text depends on (category of a product title). |
| `order` | int | `3` | n-gram order. |
| `min_token_rows` | int | `3` | A token must appear in at least this many rows. |
| `max_copy_ngram` | int | `8` | No rare source n-gram of this length is copied. |
| `whole_values` | bool | `False` | Generate whole labels seen in the source (nested labels). |

### `columns.<name>.pii`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `generator` | email | person_name | phone | uuid | pattern | `pattern` | How identifiers are generated. |
| `pattern` | str |  | Shape for the pattern generator (learned when absent). |

### `columns.<name>.lookup` (keyed)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `table` | table |  | Reference table (mode copy, no primary key). |
| `match` | map |  | Reference column -> expression of this row (OrderDate, 'USD', CurrencyCode). |
| `value` | column |  | Reference column returned. |

### `columns.<name>.distribution`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | fixed | categorical | uniform | normal | lognormal | poisson | date_range | sequence | empirical |  | The stated law (drawn independently of the other columns). |
| `value` | any |  | fixed: the value. |
| `probabilities` | map level -> p |  | categorical: shares (normalised). |
| `min` | number |  | uniform: lower end; others: clip below. |
| `max` | number |  | uniform: upper end; others: clip above. |
| `mean` | number |  | normal, poisson; lognormal: mean of the log. |
| `sd` | number |  | normal; lognormal: sd of the log. |
| `start` | date | number |  | date_range: first date; sequence: first value. |
| `end` | date |  | date_range: last date. |
| `step` | number | `1` | sequence: increment. |
| `decimals` | int |  | Rounding of a float. |
| `nulls` | number | `0` | Share of nulls. |

### `tables.<name>.scd`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | 2 | `2` | Only type 2 needs a setting (types 0 and 1 have one row per key). |
| `natural_key` | column |  | The business key shared by the versions (ProductAlternateKey). |
| `valid_from` | column |  | Start date of a version. |
| `valid_to` | column |  | End date (null on the open, current version). |
| `stable` | auto | list of column | `auto` | Columns repeated across versions (auto: equal on 99% of the source's keys). |
| `current` | {column, value, other} | `auto` | Current flag: `value` on the open version, `other` on the rest. |

### `tables.<name>.bridge`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `group` | column |  | Column whose rows share the weights (customer). |
| `weight` | column |  | Weight column. |
| `total` | number | `1.0` | What the weights of a group sum to (1 or 100). |
| `decimals` | int |  | Stored precision; the rounding remainder goes to the largest weight. |

### `tables.<name>.group_effects[]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `column` | column |  | Column shared by the rows of a group (for a ratio column, its factor). |
| `by` | list of column | fk.col |  | Columns that define a group (ProductKey.CategoryKey, OrderKey.OrderDate). |

### `tables.<name>.link_affinity[]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `match` | [path, path] |  | Two paths whose values match more often than chance; the link of the second is moved (CustomerKey.GeoAreaKey, StoreKey.GeoAreaKey). |
| `share` | number | `learned` | Matching share to reach (default: the source's, drawn from its posterior). |

### `unique[]` (filtered key)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `columns` | list of column |  | Key columns. |
| `where` | expression |  | Rows the key applies to. |

### `relationships[]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `child` | table |  | Table holding the foreign key. |
| `child_key` | column | list |  | Foreign key column(s); a list for a composite key (may use fk.col). |
| `parent` | table |  | Referenced table. |
| `parent_key` | column | list | `parent's primary key` | Referenced key (a declared unique key for composites). |
| `kind` | many_to_one | self_reference | `many_to_one` | self_reference: a parent-child hierarchy in one table (manager of an employee). |
| `name` | str | `child.key->parent` | Name used in messages, `via` and `fact.anchor`. |
| `offset_from` | column |  | Another key into the same ordered parent (ship date from order date): the gap is modelled. |
| `order_by` | column |  | Parent column that orders it (the calendar's date). |

### `rules[]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `table` | table |  | Table the rule is on. |
| `type` | check | derive | aggregate | cumulative |  | check: must hold (rows redrawn); derive: column = expression; aggregate: parent column from its children; cumulative: running balance. |
| `expr` | expression |  | check/derive expression (columns, fk.col paths, functions). round() and integer derives round half away from zero (SQL: round(2.5) = 3, round(-2.5) = -3). `&` and `|` bind tighter than comparisons: write (a > 0) & (b > 0), or a > 0 and b > 0. Text compared with a number or a date must be a number or an ISO date ('2021-01-01'); 'true' / 'false' are booleans. |
| `tolerance` | number | `0` | derive: accepted absolute difference. |
| `holds` | number in (0, 1) | `1` | derive: share of rows on which the rule holds (a soft rule: a Bayesian mixture of the rule and its violations). |
| `violation` | offset | model | `offset` | soft rule: violating rows take one of the source's offsets from the rule, or the modelled value. |
| `column` | column |  | aggregate/cumulative: the computed column. |
| `agg` | count | sum | min | max | mean | exists | count_distinct | `count` | aggregate function. |
| `from` | table |  | aggregate: the child table. |
| `of` | column | expression |  | aggregate: what is aggregated. |
| `via` | relationship name |  | aggregate: which link when there are several. |
| `by` | column |  | cumulative: the group (product). |
| `order` | column |  | cumulative: the order (date). |
| `step` | expression |  | cumulative: change per row (UnitsIn - UnitsOut). |
| `opening` | number | `fitted` | cumulative: first value of each group. |
| `floor` | number | `fitted` | cumulative: lowest allowed balance. |

### `synthesis`

| Key | Default | Meaning |
|---|---|---|
| `seed` | `7` | Same seed and spec give the same datasets. |
| `posterior.mode` | `draw` | draw: one posterior draw of every parameter per dataset (posterior predictive synthesis); mean: posterior means (stable test data). |
| `posterior.datasets` | `1` | Number of independent synthetic datasets (each its own posterior draw). |
| `posterior.sample_seed` | `None` | Seed of the parameter and data streams (default: seed). |
| `scale.default` | `1.0` | Row multiplier for entities; facts follow their anchors. |
| `scale.tables` | `{}` | Per-table multipliers {table: factor}. An entity without its own factor takes the factor of the first parent it links to that has one (orders follow their customers); facts follow their anchors. |
| `segments.default_cap` | `8` | Upper bound on latent segments per table. |
| `segments.min_rows_per_segment` | `20` | Rows per segment at least (also caps the segment count). |
| `segments.tables` | `{}` | Per-table segment caps {table: cap}. |
| `segments.stratify` | `auto` | auto | none | {table: [key, ...]}: segments nested in the rows of small copied parents. |
| `segments.stratify_max_levels` | `20` | Largest code list used as strata without a Bayes-factor test. |
| `segments.stratify_max_cells` | `50000000.0` | Budget of rows x segments per sweep for strata. |
| `segments.attribute_strata` | `auto` | auto | none | {table: column}: segments nested in an own column (only sales staff have quotas). |
| `segments.tie_headers` | `True` | Order headers take the segment of their customer. |
| `segments.copied_budget_cells` | `20000000.0` | Budget for one segment per row of copied tables. |
| `gibbs.iterations` | `120` | Sweeps of the relational segment sampler. |
| `gibbs.final_iterations` | `30` | Sweeps after merging segments. |
| `gibbs.max_final_iterations` | `120` | Upper bound when extending the final sweeps until the split R-hat of their second half <= rhat_target. |
| `gibbs.rhat_target` | `1.1` | Convergence target of the final sweeps (split R-hat of the log-likelihood trace, second half; the first half is burn-in after the merge). |
| `gibbs.chains` | `1` | Reserved (one chain). |
| `gibbs.merge` | `True` | Merge segments the data does not support. |
| `copula.mode` | `rank` | rank | bayes | fast: sampler of the correlation matrix. |
| `copula.sample_rows` | `1500` | Rows used to fit a copula. |
| `copula.iterations` | `120` | Sweeps of the copula sampler. |
| `copula.max_iterations` | `3000` | Upper bound when extending the chain until it converges (every correlation: split R-hat <= rhat_target, or Monte Carlo standard error <= 0.01 on the Fisher-z scale). |
| `copula.rhat_target` | `1.1` | Convergence target (split R-hat); a correlation whose standard error is already below 0.01 on the Fisher-z scale passes too. |
| `copula.burn_in` | `60` | Sweeps discarded. |
| `copula.lkj_eta` | `2.0` | LKJ prior concentration on the correlation. |
| `copula.grid` | `64` | Grid of the rank likelihood. |
| `copula.min_rows` | `80` | Fewer rows: independence copula. |
| `copula.pooled_cells` | `1000000.0` | Budget of rows x columns^2 latent updates per sweep for the pooled copula (the one fact tables use); it is fitted on max(2 * sample_rows, pooled_cells / p^2) rows, because a dataset drawn from it shows the posterior spread, about 1/sqrt(rows fitted), in every correlation. |
| `marginals.bayesian_bootstrap` | `True` | Draw continuous marginals by the Bayesian bootstrap. |
| `marginals.dirichlet_prior` | `0.25` | Shrinkage of a segment's categorical law to the table's. |
| `marginals.max_levels` | `200` | Levels kept per categorical (rarer ones pooled). |
| `marginals.tail_trim` | `0.0` | Share trimmed from continuous tails. |
| `marginals.min_dependence_rows` | `5` | Rows away from the mode needed to model a column's dependence, in the whole table and in each segment's copula sample. |
| `marginals.fd_bf` | `100.0` | Bayes factor to treat a categorical dependency as functional. |
| `marginals.joint_max_levels` | `64` | Largest joint of dependent categoricals. |
| `marginals.joint_min_cramers_v` | `0.3` | Association that joins two categoricals. |
| `facts.min_group_rows` | `30` | Rows below which a block borrows the coarser dimension's marginals. |
| `facts.frailty` | `auto` | auto | on | off: heavy-tailed rows per parent (gamma frailty). |
| `facts.frailty_threshold` | `1.5` | Dispersion index above which frailty is used. |
| `facts.pairs` | `auto` | Default of fact.pairs. |
| `facts.multiplicity` | `auto` | Default of fact.multiplicity. |
| `facts.count_model` | `auto` | Default of fact.count_model. |
| `facts.count_model_bf` | `100.0` | Bayes factor for the Dirichlet-process count law. |
| `facts.group_fallback` | `auto` | Default of fact.group_fallback. |
| `facts.link_degree` | `auto` | Default of fact.link_degree. |
| `facts.link_degree_min_rows` | `50.0` | auto gives a key a degree law only when each segment's links are spread over at least this many rows (Kish effective number, link-weighted); below it a few rows carry the links and attributes joined through the key would follow them. |
| `structure.structural_prior` | `0.9` | Prior probability that an unseen combination is forbidden. |
| `structure.structural_pooling` | `3.0` | Prior concentration of the structural weights learned from a relation's group-columns (zero / one / free exchangeable across them, w ~ Dirichlet(kappa * (pi/2, pi/2, 1 - pi)) centred on structural_prior); blocks inside a free group-column keep the fixed weights. Learned weights keep fully complete relations complete. 0 keeps the fixed weights of structural_prior (v0.8). |
| `structure.beta_prior` | `0.5` | Beta prior of link and occupancy rates. |
| `structure.conditional_inclusion` | `auto` | auto | none: links never point at excluded parent levels. |
| `structure.inclusion_min_expected` | `10.0` | Expected references needed to call a level excluded (Bayes factor e^n). |
| `structure.participation` | `auto` | auto | none: hurdle on the share of parents without children. |
| `uniqueness.mh_sweeps` | `20` | Sweeps of the unique-key sampler. |
| `uniqueness.grain` | `auto` | auto | none: unique keys of two foreign keys become grains. |
| `uniqueness.rake_rounds` | `3` | Rounds of raking for rule cells. |
| `trees.max_depth` | `auto` | auto | int: deepest hierarchy level generated. |
| `trees.frailty` | `auto` | auto | on | off: branching heterogeneity in trees. |
| `text.order` | `3` | n-gram order of text columns. |
| `text.min_token_rows` | `3` | Rows a token must appear in. |
| `text.max_copy_ngram` | `8` | Longest rare source n-gram allowed in output. |
| `text.backoff_strength` | `2.0` | Back-off weight of the n-gram model. |
| `text.max_retries` | `30` | Retries before a text value is replaced. |
| `rules.max_rejection_rounds` | `25` | Redraw rounds for rows that fail a check rule. |
| `privacy.suppress_small_counts` | `5` | Counts below this are suppressed in the diagnostics bundle. |
| `pushdown.mode` | `auto` | auto | on | off: fit eligible fact tables from exact statistics computed where the data lives (row counts, rows per key and key pair, distinct keys, rank-score sums) and a uniform sample for their attribute models, instead of loading them. auto: facts with at least min_rows rows; on: every eligible fact; off: load every table. |
| `pushdown.min_rows` | `1000000` | Rows from which auto mode pushes a fact table down. |
| `pushdown.sample_rows` | `500000` | Rows of a pushed-down fact sampled (seeded, uniform) for its attribute models; the relational model always uses every row through the statistics. |
| `generator` | `copula` | Default generator plug-in of every table's attribute columns (a table's own `generator` wins): copula (Bayesian Gaussian copula, relsynth 1.0), independent, cart, or an installed plug-in. |
| `compute.preset` | `cpu` | cpu | gpu-auto | gpu-single | gpu-multi, or a preset of compute.file: the devices generator plug-ins may use. The default engine is CPU-only; see docs/compute.md. |
| `compute.file` | `None` | A compute.yaml with more presets (relative to the spec). |
| `compute.overrides` | `{}` | Expert settings that replace the preset's fields: device, gpus, devices, backend, precision, batch_size, memory_limit, threads, fallback. |


### Column types

| Type | Use | Main options |
|---|---|---|
| `id` | primary keys, foreign keys, natural keys and codes | new values in the source's shapes |
| `integer`, `float` | measures | `min`, `max`, `decimals`, `ratio_of`, `offset_from`, `distribution` |
| `categorical`, `boolean` | codes, labels, flags | `distribution` |
| `date`, `datetime` | dates | `offset_from`, `distribution` |
| `text` | free text, titles | `text: {condition_on, order, min_token_rows, max_copy_ngram, whole_values}` |
| `pii` | direct identifiers | `pii: {generator: email \| person_name \| phone \| uuid \| pattern}` |
| `lookup` | a value read from a parent or a reference table | `lookup: fk.col` or `{table, match, value}`, `map`, `default` |
| `sequence` | line numbers | `within`, `start` |
| `constant`, `binary` | left out of modelling; binary columns are written empty | |

### Expressions

Rules and keyed lookups use a small, safe expression language over the columns of one table and dotted paths to parents (`customer_id.country`). It supports:

- arithmetic, comparisons, `and` / `or` / `not`, and `between`;
- null handling: `isnull`, `notnull`, `coalesce`, `where(cond, a, b)`;
- numbers: `abs`, `floor`, `ceil`, `sqrt`, `log`, `exp`, `round`, `minimum`, `maximum`;
- dates: `days(date)`, `add_days(date, n)`, `years_between(start, end)`, `months_between(start, end)`;
- strings: `regex_replace`, `regex_match`, `substr`, `concat`, `upper`, `lower`, `strip`, `length`.

Nulls follow SQL: a `check` fails only when it is false.

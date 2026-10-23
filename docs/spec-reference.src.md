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

<!-- KEYS -->

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

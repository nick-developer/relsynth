"""The specification schema: every key the YAML file accepts, with its type,
default and meaning, in one table.

`check_keys(raw)` reports keys that are not in the schema (a typo is an error,
with the closest valid key as a hint), so a misspelt option is never ignored in
silence.  `docs/spec-reference.md` is generated from this table
(`python -m relsynth.schema > docs/spec-keys.md`), so the reference cannot drift
from what the code reads.
"""
from __future__ import annotations

import difflib

# (type, default, meaning); "map" = free keys (table or column names)
S = {
    "spec": {
        "version": ("int", 1, "Format version of the file."),
        "name": ("str", "draft", "Label used in reports and file names."),
        "description": ("str", None, "Free text about the model (kept, not read)."),
        "source": ("source", None, "Where the real data is read from (required for `run`)."),
        "target": ("target", "csv into ./synthetic", "Where the synthetic data is written."),
        "tables": ("map of table", None, "Every table: role, columns, keys and modelling options."),
        "relationships": ("list of relationship", [], "Foreign keys (declared or inferred)."),
        "rules": ("list of rule", [], "Business rules: checks, derived columns, aggregates, running balances."),
        "synthesis": ("synthesis", "defaults", "Method and statistical parameters (all optional)."),
        "review": ("list of str", [], "What introspect inferred, and its questions (lines starting ASK)."),
    },
    "source": {
        "kind": ("csv | parquet | files | sqlite | postgresql | mssql | oracle | sqlalchemy", "csv",
                 "csv / parquet / files: one <table>.csv, .csv.gz or .parquet file per table in `path`; sqlite: "
                 "`path` to a file; postgresql, mssql, oracle (or sqlalchemy for any other engine): `url`."),
        "path": ("str", None, "Folder (files) or file (sqlite); relative to the spec's folder."),
        "url": ("str", None, "Database URL, e.g. postgresql://user@host/db, mssql+pyodbc://user@host/db?driver=ODBC"
                "+Driver+18+for+SQL+Server, oracle://user@host:1521/?service_name=ORCLPDB1 (driver chosen "
                "when omitted)."),
        "url_env": ("str", None, "Environment variable that holds the URL (with its password), used instead of "
                    "`url`: keeps the password out of the file. Specs written for sharing (introspect drafts, "
                    "exported spec.yaml, bundles) show the URL with *** for the password and set url_env "
                    "(RELSYNTH_SOURCE_URL)."),
        "password_env": ("str", None, "Environment variable that holds only the password; the URL then carries "
                         "user, host and database."),
        "schema": ("str", None, "Database schema to read or write (Oracle: the owning user)."),
        "include": ("list of str", None, "Table-name patterns (regular expressions) to read."),
        "exclude": ("list of str", None, "Table-name patterns to leave out."),
        "null_values": ("list of str", [""], "CSV fields read as null. Only an empty field by default, so 'NA' "
                        "(Namibia), 'None' and 'null' stay values; e.g. ['', 'NULL', 'n/a']."),
        "connect_args": ("map", None, "Driver connection arguments (TLS settings, timeouts), passed to the driver "
                         "as they are."),
        "engine_options": ("map", None, "SQLAlchemy engine options (pool_size, pool_recycle, ...)."),
        "format": ("csv | parquet", "csv", "File format written (target of kind files)."),
    },
    "target": {
        "kind": ("csv | parquet | files | sqlite | postgresql | mssql | oracle | sqlalchemy", "csv",
                 "As for source; csv and parquet write one file per table."),
        "path": ("str", "synthetic", "Output folder or SQLite file; dataset i > 1 goes to <path>_i."),
        "url": ("str", None, "URL of the target database."),
        "url_env": ("str", None, "As for source (RELSYNTH_TARGET_URL in shared specs)."),
        "password_env": ("str", None, "As for source."),
        "schema": ("str", None, "Target schema (created when missing on PostgreSQL and SQL Server); dataset i > 1 "
                   "goes to <schema>_i."),
        "if_exists": ("replace | append | truncate | fail", "replace", "replace: drop and create the tables; "
                      "append: insert into existing tables; truncate: delete the rows of the existing tables "
                      "(children first) and insert, keeping the target's DDL and constraints; fail: stop when a "
                      "table exists."),
        "structure": ("auto | clone | infer | existing", "auto", "How target tables are created. clone: the "
                      "source's DDL (column types, nullability, primary, unique and foreign keys, indexes, identity "
                      "columns) when source and target are databases; infer: from the specification (keys and "
                      "links as constraints) and the data's types; existing: tables created beforehand. auto: clone "
                      "when possible, else infer."),
        "connect_args": ("map", None, "As for source."),
        "engine_options": ("map", None, "As for source."),
        "format": ("csv | parquet", "csv", "File format of a files target."),
    },
    "table": {
        "role": ("entity | fact | view", "entity", "entity: a thing (customer, product, order header); fact: an event "
                 "or link whose rows exist because of its parents (sales lines, visits); view: a table that is the "
                 "join of others (rebuilt, never modelled)."),
        "mode": ("synthesize | copy", "synthesize", "copy keeps the table exactly (reference data, calendars, code "
                 "lists, or any dimension you want to keep: see the recipe 'copy dimensions, synthesise facts')."),
        "primary_key": ("column", None, "Key column; synthetic keys are new values (1..n for integers)."),
        "columns": ("map of column", None, "Column name -> type, or -> {type, options}."),
        "unique": ("list of keys", [], "Keys no two rows share: [a, b], or on facts {columns, where} (a filtered "
                   "unique index). A key of two foreign keys makes the table a grain (occupancy of pairs)."),
        "partial_unique": ("list of {columns, where}", [], "Same as a filtered `unique` entry."),
        "fact": ("fact", {}, "How fact rows are generated (anchor, partner, header, time)."),
        "dimensions": ("list of dimension", [], "Levels implied inside an entity: a code column that determines "
                       "others (GeoAreaKey -> State, Country, Continent), generated through a copied level table."),
        "key_order": ("column", None, "Keys numbered in the order of this column (order numbers follow order dates)."),
        "view_of": ("table", None, "For role view: the base table the view is rebuilt from."),
        "group_effects": ("list of group_effect", [], "Columns shared by all rows of a group (one price factor per "
                          "category and day): one value drawn per group."),
        "link_affinity": ("list of link_affinity", [], "Two links that point at matching parents more often than "
                          "chance (customer and store in one area)."),
        "scd": ("scd", None, "Slowly changing dimension type 2: rows are versions of a natural key."),
        "unknown_members": ("list of key values", None, "Rows that stand for 'no member' (key -1 'Unknown'): copied, "
                            "kept out of the model, referenced with the source's share."),
        "bridge": ("bridge", None, "Weights that sum to 1 (or `total`) over the rows of a group (multivalued "
                   "dimension / allocation)."),
        "description": ("str", None, "Free text (kept, not read)."),
    },
    "fact": {
        "anchor": ("table | relationship name | key column", "first parent", "Parent whose rows generate the fact "
                   "rows (rows per customer); scaling it scales the fact. A key column (OrderDateKey) or a "
                   "relationship name picks one role of a role-playing dimension."),
        "partner": ("table | relationship name | key column", "next parent", "Second dimension of the block model "
                    "(customer segment x product segment rates)."),
        "pairs": ("auto | poisson | occupancy", "auto", "Rows per (anchor, partner) pair as counts, or each pair "
                  "present or absent."),
        "multiplicity": ("auto | none | partner | anchor | block", "auto", "For occupancy facts: rows per present "
                         "pair beyond the first, grouped by."),
        "count_model": ("auto | poisson | dp", "auto", "Rows per anchor: Poisson/negative binomial or a "
                        "Dirichlet-process count law."),
        "group_fallback": ("auto | anchor | partner", "auto", "Which dimension's marginals a sparse block borrows."),
        "time_column": ("column", None, "Date column with a seasonal profile per segment."),
        "time_bucket": ("day | week | month | quarter | year", "month", "Bucket of the seasonal profile."),
        "condition_on_parents": ("bool | list of fk.col", True, "Condition measures on parents' continuous "
                                 "attributes (unit price follows list price)."),
        "header": ("header", None, "Order header carried on the lines: {key, columns, line_number}."),
    },
    "header": {
        "key": ("column", None, "Document number (SalesOrderNumber)."),
        "columns": ("list of column", None, "Columns the document number determines (customer, dates, currency)."),
        "line_number": ("column", None, "Line number column, generated 1..n per document."),
    },
    "dimension": {
        "key": ("column", None, "The code (GeoAreaKey)."),
        "columns": ("list of column", None, "Columns it determines (State, Country, Continent)."),
        "anchor": ("column", None, "A coarse determined column kept as a modelled attribute (Country)."),
    },
    "column": {
        "type": ("id | integer | float | categorical | boolean | date | datetime | text | pii | lookup | sequence | "
                 "constant | binary", None, "What the column holds (see the column types table)."),
        "nullable": ("bool", True, "Informational; null shares are always learned from the data."),
        "min": ("number", None, "Hard lower bound of a number column."),
        "max": ("number", None, "Hard upper bound."),
        "decimals": ("int", "from the source", "Stored precision of a number column."),
        "lookup": ("fk.col path | keyed lookup", None, "For type lookup: the value of a parent's column, or "
                   "{table, match, value} into a reference table."),
        "map": ("map", None, "Lookup values mapped through a fixed table (country -> currency)."),
        "default": ("any", None, "Value for codes outside `map`."),
        "offset_from": ("column", None, "This date/number = that column + a modelled gap (delivery = order + 0..7 "
                        "days)."),
        "ratio_of": ("column | fk.col", None, "This number = that number x a modelled factor (unit price = list "
                     "price x one of 16 factors)."),
        "ratio_same_as": ("column", None, "Use the factor of another ratio column (unit cost moves with unit "
                          "price)."),
        "text": ("text", None, "Options of a text column."),
        "pii": ("pii", None, "Options of a pii column."),
        "within": ("column", None, "For type sequence: numbered 1..n within each value of this column."),
        "start": ("int", 1, "For type sequence: first number."),
        "base_column": ("column", None, "For views: the base-table column this view column copies."),
        "distribution": ("distribution", None, "Draw the column from a stated law instead of the learned one."),
        "description": ("str", None, "Free text (kept, not read)."),
    },
    "text": {
        "condition_on": ("column", None, "Column the text depends on (category of a product title)."),
        "order": ("int", 3, "n-gram order."),
        "min_token_rows": ("int", 3, "A token must appear in at least this many rows."),
        "max_copy_ngram": ("int", 8, "No rare source n-gram of this length is copied."),
        "whole_values": ("bool", False, "Generate whole labels seen in the source (nested labels)."),
    },
    "pii": {
        "generator": ("email | person_name | phone | uuid | pattern", "pattern", "How identifiers are generated."),
        "pattern": ("str", None, "Shape for the pattern generator (learned when absent)."),
    },
    "keyed_lookup": {
        "table": ("table", None, "Reference table (mode copy, no primary key)."),
        "match": ("map", None, "Reference column -> expression of this row (OrderDate, 'USD', CurrencyCode)."),
        "value": ("column", None, "Reference column returned."),
    },
    "distribution": {
        "kind": ("fixed | categorical | uniform | normal | lognormal | poisson | date_range | sequence | empirical",
                 None, "The stated law (drawn independently of the other columns)."),
        "value": ("any", None, "fixed: the value."),
        "probabilities": ("map level -> p", None, "categorical: shares (normalised)."),
        "min": ("number", None, "uniform: lower end; others: clip below."),
        "max": ("number", None, "uniform: upper end; others: clip above."),
        "mean": ("number", None, "normal, poisson; lognormal: mean of the log."),
        "sd": ("number", None, "normal; lognormal: sd of the log."),
        "start": ("date | number", None, "date_range: first date; sequence: first value."),
        "end": ("date", None, "date_range: last date."),
        "step": ("number", 1, "sequence: increment."),
        "decimals": ("int", None, "Rounding of a float."),
        "nulls": ("number", 0, "Share of nulls."),
    },
    "scd": {
        "type": ("2", 2, "Only type 2 needs a setting (types 0 and 1 have one row per key)."),
        "natural_key": ("column", None, "The business key shared by the versions (ProductAlternateKey)."),
        "valid_from": ("column", None, "Start date of a version."),
        "valid_to": ("column", None, "End date (null on the open, current version)."),
        "stable": ("auto | list of column", "auto", "Columns repeated across versions (auto: equal on 99% of the "
                   "source's keys)."),
        "current": ("{column, value, other}", "auto", "Current flag: `value` on the open version, `other` on the rest."),
    },
    "bridge": {
        "group": ("column", None, "Column whose rows share the weights (customer)."),
        "weight": ("column", None, "Weight column."),
        "total": ("number", 1.0, "What the weights of a group sum to (1 or 100)."),
        "decimals": ("int", None, "Stored precision; the rounding remainder goes to the largest weight."),
    },
    "group_effect": {
        "column": ("column", None, "Column shared by the rows of a group (for a ratio column, its factor)."),
        "by": ("list of column | fk.col", None, "Columns that define a group (ProductKey.CategoryKey, "
               "OrderKey.OrderDate)."),
    },
    "link_affinity": {
        "match": ("[path, path]", None, "Two paths whose values match more often than chance; the link of the "
                  "second is moved (CustomerKey.GeoAreaKey, StoreKey.GeoAreaKey)."),
        "share": ("number", "learned", "Matching share to reach (default: the source's, drawn from its posterior)."),
    },
    "unique_where": {
        "columns": ("list of column", None, "Key columns."),
        "where": ("expression", None, "Rows the key applies to."),
    },
    "relationship": {
        "child": ("table", None, "Table holding the foreign key."),
        "child_key": ("column | list", None, "Foreign key column(s); a list for a composite key (may use fk.col)."),
        "parent": ("table", None, "Referenced table."),
        "parent_key": ("column | list", "parent's primary key", "Referenced key (a declared unique key for "
                       "composites)."),
        "kind": ("many_to_one | self_reference", "many_to_one", "self_reference: a parent-child hierarchy in one "
                 "table (manager of an employee)."),
        "name": ("str", "child.key->parent", "Name used in messages, `via` and `fact.anchor`."),
        "offset_from": ("column", None, "Another key into the same ordered parent (ship date from order date): "
                        "the gap is modelled."),
        "order_by": ("column", None, "Parent column that orders it (the calendar's date)."),
    },
    "rule": {
        "table": ("table", None, "Table the rule is on."),
        "type": ("check | derive | aggregate | cumulative", None, "check: must hold (rows redrawn); derive: "
                 "column = expression; aggregate: parent column from its children; cumulative: running balance."),
        "expr": ("expression", None, "check/derive expression (columns, fk.col paths, functions). round() and "
                 "integer derives round half away from zero (SQL: round(2.5) = 3, round(-2.5) = -3). `&` and `|` "
                 "bind tighter than comparisons: write (a > 0) & (b > 0), or a > 0 and b > 0. Text compared with "
                 "a number or a date must be a number or an ISO date ('2021-01-01'); 'true' / 'false' are "
                 "booleans."),
        "tolerance": ("number", 0, "derive: accepted absolute difference."),
        "holds": ("number in (0, 1)", 1, "derive: share of rows on which the rule holds (a soft rule: a Bayesian "
                  "mixture of the rule and its violations)."),
        "violation": ("offset | model", "offset", "soft rule: violating rows take one of the source's offsets from "
                      "the rule, or the modelled value."),
        "column": ("column", None, "aggregate/cumulative: the computed column."),
        "agg": ("count | sum | min | max | mean | exists | count_distinct", "count", "aggregate function."),
        "from": ("table", None, "aggregate: the child table."),
        "of": ("column | expression", None, "aggregate: what is aggregated."),
        "via": ("relationship name", None, "aggregate: which link when there are several."),
        "by": ("column", None, "cumulative: the group (product)."),
        "order": ("column", None, "cumulative: the order (date)."),
        "step": ("expression", None, "cumulative: change per row (UnitsIn - UnitsOut)."),
        "opening": ("number", "fitted", "cumulative: first value of each group."),
        "floor": ("number", "fitted", "cumulative: lowest allowed balance."),
    },
}

SYNTH_DOC = {
    "seed": "Same seed and spec give the same datasets.",
    "posterior.mode": "draw: one posterior draw of every parameter per dataset (posterior predictive synthesis); "
                      "mean: posterior means (stable test data).",
    "posterior.datasets": "Number of independent synthetic datasets (each its own posterior draw).",
    "posterior.sample_seed": "Seed of the parameter and data streams (default: seed).",
    "scale.default": "Row multiplier for entities; facts follow their anchors.",
    "scale.tables": "Per-table multipliers {table: factor}. An entity without its own factor takes the factor of the first parent it links to that has one (orders follow their customers); facts follow their anchors.",
    "segments.default_cap": "Upper bound on latent segments per table.",
    "segments.min_rows_per_segment": "Rows per segment at least (also caps the segment count).",
    "segments.tables": "Per-table segment caps {table: cap}.",
    "segments.stratify": "auto | none | {table: [key, ...]}: segments nested in the rows of small copied parents.",
    "segments.stratify_max_levels": "Largest code list used as strata without a Bayes-factor test.",
    "segments.stratify_max_cells": "Budget of rows x segments per sweep for strata.",
    "segments.attribute_strata": "auto | none | {table: column}: segments nested in an own column (only sales staff "
                                 "have quotas).",
    "segments.tie_headers": "Order headers take the segment of their customer.",
    "segments.copied_budget_cells": "Budget for one segment per row of copied tables.",
    "segments.copied_rows": "auto | {table: true}: one segment per row of a copied table.",
    "gibbs.iterations": "Sweeps of the relational segment sampler.",
    "gibbs.final_iterations": "Sweeps after merging segments.",
    "gibbs.max_final_iterations": "Upper bound when extending the final sweeps until the split R-hat of their second half <= rhat_target.",
    "gibbs.rhat_target": "Convergence target of the final sweeps (split R-hat of the log-likelihood trace, second half; the first half is burn-in after the merge).",
    "gibbs.chains": "Reserved (one chain).",
    "gibbs.merge": "Merge segments the data does not support.",
    "copula.mode": "rank | bayes | fast: sampler of the correlation matrix.",
    "copula.sample_rows": "Rows used to fit a copula.",
    "copula.iterations": "Sweeps of the copula sampler.",
    "copula.max_iterations": "Upper bound when extending the chain until it converges (every correlation: split R-hat <= rhat_target, or Monte Carlo standard error <= 0.01 on the Fisher-z scale).",
    "copula.rhat_target": "Convergence target (split R-hat); a correlation whose standard error is already below 0.01 on the Fisher-z scale passes too.",
    "copula.burn_in": "Sweeps discarded.",
    "copula.lkj_eta": "LKJ prior concentration on the correlation.",
    "copula.grid": "Grid of the rank likelihood.",
    "copula.min_rows": "Fewer rows: independence copula.",
    "copula.pooled_cells": "Budget of rows x columns^2 latent updates per sweep for the pooled copula (the one fact "
                           "tables use); it is fitted on max(2 * sample_rows, pooled_cells / p^2) rows, because a "
                           "dataset drawn from it shows the posterior spread, about 1/sqrt(rows fitted), in every "
                           "correlation.",
    "marginals.bayesian_bootstrap": "Draw continuous marginals by the Bayesian bootstrap.",
    "marginals.dirichlet_prior": "Shrinkage of a segment's categorical law to the table's.",
    "marginals.max_levels": "Levels kept per categorical (rarer ones pooled).",
    "marginals.tail_trim": "Share trimmed from continuous tails.",
    "marginals.min_dependence_rows": "Rows away from the mode needed to model a column's dependence, in the whole table and in each segment's copula sample.",
    "marginals.fd_bf": "Bayes factor to treat a categorical dependency as functional.",
    "marginals.joint_max_levels": "Largest joint of dependent categoricals.",
    "marginals.joint_min_cramers_v": "Association that joins two categoricals.",
    "facts.min_group_rows": "Rows below which a block borrows the coarser dimension's marginals.",
    "facts.frailty": "auto | on | off: heavy-tailed rows per parent (gamma frailty).",
    "facts.frailty_threshold": "Dispersion index above which frailty is used.",
    "facts.pairs": "Default of fact.pairs.",
    "facts.multiplicity": "Default of fact.multiplicity.",
    "facts.count_model": "Default of fact.count_model.",
    "facts.count_model_bf": "Bayes factor for the Dirichlet-process count law.",
    "facts.group_fallback": "Default of fact.group_fallback.",
    "structure.structural_prior": "Prior probability that an unseen combination is forbidden.",
    "structure.structural_pooling": "Prior concentration of the structural weights learned from a relation's "
                                    "group-columns (zero / one / free exchangeable across them, w ~ Dirichlet(kappa * "
                                    "(pi/2, pi/2, 1 - pi)) centred on structural_prior); blocks inside a free "
                                    "group-column keep the fixed weights. Learned weights keep fully complete "
                                    "relations complete. 0 keeps the fixed weights of structural_prior (v0.8).",
    "structure.beta_prior": "Beta prior of link and occupancy rates.",
    "structure.conditional_inclusion": "auto | none: links never point at excluded parent levels.",
    "structure.inclusion_min_expected": "Expected references needed to call a level excluded (Bayes factor e^n).",
    "structure.participation": "auto | none: hurdle on the share of parents without children.",
    "uniqueness.mh_sweeps": "Sweeps of the unique-key sampler.",
    "uniqueness.grain": "auto | none: unique keys of two foreign keys become grains.",
    "uniqueness.rake_rounds": "Rounds of raking for rule cells.",
    "trees.max_depth": "auto | int: deepest hierarchy level generated.",
    "trees.frailty": "auto | on | off: branching heterogeneity in trees.",
    "text.order": "n-gram order of text columns.",
    "text.min_token_rows": "Rows a token must appear in.",  # nosec B105
    "text.max_copy_ngram": "Longest rare source n-gram allowed in output.",
    "text.backoff_strength": "Back-off weight of the n-gram model.",
    "text.max_retries": "Retries before a text value is replaced.",
    "rules.max_rejection_rounds": "Redraw rounds for rows that fail a check rule.",
    "privacy.suppress_small_counts": "Counts below this are suppressed in the diagnostics bundle.",
    "pushdown.mode": "auto | on | off: fit eligible fact tables from exact statistics computed where the data lives "
                     "(row counts, rows per key and key pair, distinct keys, rank-score sums) and a uniform sample "
                     "for their attribute models, instead of loading them. auto: facts with at least min_rows rows; "
                     "on: every eligible fact; off: load every table.",
    "pushdown.min_rows": "Rows from which auto mode pushes a fact table down.",
    "pushdown.sample_rows": "Rows of a pushed-down fact sampled (seeded, uniform) for its attribute models; the "
                            "relational model always uses every row through the statistics.",
}

# synthesis keys whose value is a free map (table names) or may be a map
FREE = {"scale.tables", "segments.tables", "segments.stratify", "segments.attribute_strata", "segments.copied_rows"}
# keys normalise() adds (a normalised spec written back, e.g. by `doctor --apply`, still
# validates); the keys of the engine's internal plan (anchor_link, virtual, unit, _refs,
# ...) are not accepted from a spec
INTERNAL = {"relationship": {"composite"}, "fact": {"anchor_key", "partner_key"}, "spec": {"_base_dir"}}

NESTED = {("spec", "source"): "source", ("spec", "target"): "target", ("table", "fact"): "fact",
          ("fact", "header"): "header", ("column", "text"): "text", ("column", "pii"): "pii",
          ("column", "distribution"): "distribution", ("table", "scd"): "scd", ("table", "bridge"): "bridge"}
LISTS = {("table", "dimensions"): "dimension", ("table", "group_effects"): "group_effect",
         ("table", "link_affinity"): "link_affinity", ("table", "partial_unique"): "unique_where",
         ("spec", "relationships"): "relationship", ("spec", "rules"): "rule"}


def _hint(k, allowed):
    m = difflib.get_close_matches(str(k), list(allowed), n=1, cutoff=0.6)
    return f" (did you mean '{m[0]}'?)" if m else f" (allowed: {', '.join(sorted(allowed))})"


def _check(obj, kind, where, errs):
    if not isinstance(obj, dict):
        return
    allowed = S[kind]
    for k, v in obj.items():
        if k in INTERNAL.get(kind, ()):
            continue
        if k not in allowed:
            errs.append(f"{where}: unknown key '{k}'{_hint(k, allowed)}")
            continue
        if (kind, k) in NESTED and isinstance(v, dict):
            _check(v, NESTED[(kind, k)], f"{where}.{k}", errs)
        elif (kind, k) in LISTS and isinstance(v, list):
            for i, x in enumerate(v):
                _check(x, LISTS[(kind, k)], f"{where}.{k}[{i}]", errs)
        elif kind == "column" and k == "lookup" and isinstance(v, dict):
            _check(v, "keyed_lookup", f"{where}.lookup", errs)


def _check_synth(obj, defaults, pre, errs):
    for k, v in (obj or {}).items():
        key = f"{pre}{k}"
        if key in FREE:
            continue
        if k not in defaults:
            errs.append(f"synthesis.{key}: unknown key{_hint(k, defaults)}")
            continue
        if isinstance(defaults[k], dict) and isinstance(v, dict):
            _check_synth(v, defaults[k], key + ".", errs)


def check_keys(raw):
    """Unknown keys anywhere in a raw (or normalised) spec, with hints."""
    from .spec import DEFAULT_SYNTHESIS
    errs = []
    if not isinstance(raw, dict):
        return ["the spec must be a mapping"]
    _check({k: v for k, v in raw.items() if k != "synthesis"}, "spec", "spec", errs)
    for t, T in (raw.get("tables") or {}).items():
        if not isinstance(T, dict):
            continue
        _check(T, "table", f"tables.{t}", errs)
        for c, cs in (T.get("columns") or {}).items():
            if isinstance(cs, dict):
                _check(cs, "column", f"tables.{t}.columns.{c}", errs)
        for i, u in enumerate(T.get("unique") or []):
            if isinstance(u, dict):
                _check(u, "unique_where", f"tables.{t}.unique[{i}]", errs)
    for i, r in enumerate(raw.get("relationships") or []):
        _check(r, "relationship", f"relationships[{i}]", errs)
    for i, r in enumerate(raw.get("rules") or []):
        _check(r, "rule", f"rules[{i}]", errs)
    _check_synth(raw.get("synthesis"), DEFAULT_SYNTHESIS, "", errs)
    return errs


def reference_tables():
    """Markdown tables of every key, generated from the schema."""
    from .spec import DEFAULT_SYNTHESIS
    titles = {"spec": "Top level", "source": "`source`", "target": "`target`", "table": "`tables.<name>`",
              "fact": "`tables.<name>.fact`", "header": "`fact.header`", "dimension": "`tables.<name>.dimensions[]`",
              "column": "`tables.<name>.columns.<name>`", "text": "`columns.<name>.text`",
              "pii": "`columns.<name>.pii`", "keyed_lookup": "`columns.<name>.lookup` (keyed)",
              "distribution": "`columns.<name>.distribution`", "scd": "`tables.<name>.scd`",
              "bridge": "`tables.<name>.bridge`", "group_effect": "`tables.<name>.group_effects[]`",
              "link_affinity": "`tables.<name>.link_affinity[]`", "unique_where": "`unique[]` (filtered key)",
              "relationship": "`relationships[]`", "rule": "`rules[]`"}
    out = []
    for kind, keys in S.items():
        out += [f"### {titles.get(kind, kind)}", "", "| Key | Type | Default | Meaning |", "|---|---|---|---|"]
        for k, (ty, df, doc) in keys.items():
            d = "" if df is None else f"`{df}`"
            out.append(f"| `{k}` | {ty} | {d} | {doc} |")
        out.append("")
    out += ["### `synthesis`", "", "| Key | Default | Meaning |", "|---|---|---|"]

    def walk(d, pre):
        for k, v in d.items():
            if isinstance(v, dict) and f"{pre}{k}" not in FREE:
                walk(v, f"{pre}{k}.")
            else:
                out.append(f"| `{pre}{k}` | `{v}` | {SYNTH_DOC.get(pre + k, '')} |")
    walk(DEFAULT_SYNTHESIS, "")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(reference_tables())

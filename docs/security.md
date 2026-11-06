# Security

relsynth reads production databases and writes data that is meant to be shared. This page states what it protects, against what, and how each protection is tested.

## Threat model

| Asset | Threat | Protection |
|---|---|---|
| Database credentials | Leaked through spec files, logs, error messages, CI output, diagnostics bundles, process lists | Credentials come from the environment, and every output masks them (see below) |
| Source data | Synthetic rows that copy real ones; rare values that identify a person; bundles that carry rows | Bayesian synthesis (no row is copied), memorisation guards for text, privacy attacks per person ([privacy.md](privacy.md)), optional differential privacy ([dp.md](dp.md)), and aggregate-only bundles |
| The machine running relsynth | A malicious spec (YAML tricks, rule expressions, regexes) or a malicious file name | Safe YAML loading, a sandboxed expression language, regex vetting, and confined paths |
| The target database | SQL injection through table or column names or values | SQLAlchemy Core with quoted identifiers and bound values everywhere |
| The source database | Accidental writes, or heavy load | Only `SELECT` is issued against a source; pushdown statistics are plain aggregate queries |

## Credentials

- **No passwords on the command line.** Locations take `--source-password-env VAR` or `--source-url-env VAR` (and the `--target-...` equivalents). In a spec, `password_env` and `url_env` name environment variables.
- **Masked everywhere.** `mask_url` replaces:
  - the password in `user:password@host`;
  - query parameters whose name contains password, pwd, secret, token, api key or credential (`sslpassword`, `auth_token`, ...);
  - `PWD=` inside `odbc_connect`.

  It is applied to every error message the CLI prints (`_scrub`), the JSON output, saved specs, diagnostics bundles (again when they are written, whoever built them), exported models and `ping` results. Keys named like secrets (`password`, `token`, ...) in a source or target section are written as `***`.
- **Saved specs are safe to commit.** `introspect`, `doctor --apply`, `tune` and `Spec.save` write the URL with `***` and add `url_env: RELSYNTH_SOURCE_URL`, so the run reads the real URL from the environment.
- **CI.** Pass secrets as masked CI variables (see [cicd.md](cicd.md)), and use read-only database accounts for sources.

Tests: `test_mask_url` (six URL forms), `test_scrub_masks_urls_inside_messages`, `test_spec_urls_masked_everywhere`, `test_bundle_holds_no_password`, `test_failed_connection_logs_no_password`, `test_saved_spec_moves_password_to_environment`, `test_connection_errors_exit_3_and_mask_passwords`, and `test_save_masks_passwords`.

## SQL

- Tables are reflected by name and every statement is built with SQLAlchemy Core. Identifiers are quoted by the dialect, and values are bound parameters.
- The only literals inlined into SQL are integers relsynth computes itself (the sample seed and hash range). On SQL Server the hash algorithm name is inlined too, because `HASHBYTES` does not accept it as a parameter.
- Rule expressions are never translated into SQL text; they are evaluated in Python on loaded rows.

Tests: `test_hostile_table_and_column_names_are_quoted` (SQLite) and `test_hostile_identifiers_on_server` (PostgreSQL, SQL Server, Oracle, MySQL, MariaDB). Both use names such as `x"; DROP TABLE keep; --` and `v'); DELETE FROM keep; --`, read, write, sample and compute statistics on them, and check that the other tables are untouched.

## Specifications

A spec is configuration that may come from someone else, so relsynth treats it as untrusted input.

- **YAML** is read with a safe loader that also refuses anchors and aliases. Python tags (`!!python/object`) cannot construct objects, and an alias bomb (a few lines that expand to billions of nodes) is refused before it expands.
- **Rule expressions** are parsed with Python's `ast` module and evaluated by relsynth's own interpreter, never by `eval`:
  - the allowed node types are constants, column names, `fk.column` paths, arithmetic, comparisons, `and`/`or`/`not`, `in` with constant lists, and calls to a fixed list of functions;
  - attribute access, subscripts, lambdas, comprehensions, f-strings, assignment expressions and keyword arguments are refused;
  - expressions longer than 10,000 characters or nested more than 100 levels are refused.
- **Regular expressions** in rules (`regex_match`, `regex_replace`) are vetted before use. Patterns that can backtrack catastrophically are refused, such as nested repetition `(a+)+` or repeated alternatives that can match the same text `(a|aa)+`, and so are patterns longer than 500 characters.
- **Strict validation.** Unknown keys are errors (with a "did you mean" suggestion), so a misspelt privacy setting cannot silently turn a protection off.

Tests: `test_yaml_aliases_are_refused`, `test_yaml_python_tags_are_refused`, `test_spec_cannot_construct_python_objects`, `test_rule_expressions_are_sandboxed` (10 attack forms), `test_expression_size_and_depth_limits`, `test_catastrophic_regexes_are_refused` (5 patterns) and `test_ordinary_regexes_are_accepted`.

## Files

A table name becomes a file name in folder sources and targets, so it must be a plain name:
- `..`, `.`, empty names, `/`, `\`, NUL and absolute paths are refused;
- so is any name whose file resolves outside the folder, through a symbolic link for example.

Tests: `test_inside_refuses_non_plain_names`, `test_symlinked_table_cannot_escape` and `test_file_table_names_cannot_escape_the_folder`.

## Data protection

- **Nothing is copied.** Synthetic rows are drawn from a fitted model; only tables declared `mode: copy` (a calendar, a code list) are copied. Key values are generated in the source's format, so a synthetic key that equals a real key does not carry that real row's data. `test_synthetic_rows_are_not_copies` checks that fewer than 5% of synthetic rows match a real row on their high-cardinality attributes.
- **Copy guards run by default.** They catch gross copying; they are not a privacy measurement (see [privacy.md](privacy.md#relation-to-the-existing-guards)).
  - For each table, the distance from synthetic rows to their closest real row (in rank-normal-score and one-hot space) is compared with the distance from real rows to their closest other real row (leave-one-out). The check fails when the 5th percentile of the first is below half that of the second, or when synthetic rows equal real rows more often than real rows equal each other.
  - Text outputs that reproduce rare source content are rejected (below).
  - `--no-privacy` turns the checks off, and the bundle records that.
- **Memorisation guards.**
  - Text uses only tokens that occur in at least `min_token_rows` distinct source rows.
  - A text output is rejected when it contains a run of tokens, or equals a whole text, that occurs in fewer than `min_token_rows` source rows.
  - E-mail addresses and long digit runs are replaced by typed placeholders before training.
  - Rare columns are kept independent.
- **Aggregate-only outputs.**
  - Diagnostics bundles, `verify` reports and `profile` output hold counts and statistics, not rows.
  - Frequent values appear only with an explicit `profile --top N`.
  - Counts below 5 are suppressed in what a bundle reports about category levels.
- **Pushdown.** Rows of pushed facts never leave the source as a whole; only aggregates and a sample do.
- **Differential privacy is optional.** Without it, protection is measured, not guaranteed: `relsynth privacy` attacks the release per person ([privacy.md](privacy.md)). With `--epsilon`, a run has a user-level (ε, δ) guarantee ([dp.md](dp.md)). Under the governance policy, a release outside the organisation is refused without DP ([governance/guide.md](governance/guide.md)). Use `mode: copy` only for tables that are public anyway.

## Supply chain and code

- The core package depends on seven well-known libraries (numpy, pandas, scipy, SQLAlchemy, PyYAML, DuckDB and its SQLAlchemy dialect) and the database driver you choose; every other dependency comes with an extra you install. Versions have lower bounds only, so you can patch them yourself.
- CI runs on every change:
  - `ruff` with the `S` rule set (the flake8-bandit security rules);
  - `bandit` over `src/`;
  - `pip-audit` over the installed dependencies.

  Every suppressed finding is annotated in the code with its test ID and a reason, for example `# nosec B608` where a query is built with SQLAlchemy and not by string formatting.
- The package and the CLI connect only to the databases, warehouses and storage you name, and send no telemetry. The service also contacts the OIDC identity provider and the Vault server you configure, and exports traces only when you turn on OpenTelemetry (`relsynth[service-otel]`) and give it an endpoint.
- `subprocess` runs fixed argument lists, never a shell, in four places: `pdflatex` (when present) for the exported model's PDF, the OPA binary when the policy engine is `opa`, the GPU listing commands of `relsynth compute`, and the service worker's job runner.

## Reporting a vulnerability

Report security issues privately to the maintainers of your installation rather than in a public issue tracker. Include the version (`relsynth version`) and a minimal spec that reproduces the issue, with credentials removed.

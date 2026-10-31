# Command-line reference

Generated from the parser by `python docs/make_cli_reference.py`; every option below is what `relsynth COMMAND --help` prints.

```
relsynth COMMAND [options]
```

Locations (`--source`, `--target`, `--synthetic`) are written as `csv:DIR`, `parquet:DIR`, `sqlite:FILE` or a database URL (`postgresql://`, `mssql://`, `oracle://`). A plain folder path means CSV. Instead of a location, `--source-url-env NAME` or `--target-url-env NAME` names an environment variable that holds the whole URL; `--source-password-env NAME` supplies only the password. Passwords are never accepted on the command line, and they are masked in all output.

Every command also takes:

| Option | Meaning |
|---|---|
| `--json` | Print exactly one JSON document on stdout (`command`, `version`, `exit_code`, and the command's results; `error` and `error_type` on failure). Progress goes to stderr. |
| `--log-level debug\|info\|warning\|error` | Progress detail on stderr (default `info`; `RELSYNTH_QUIET=1` sets `warning`). |

Commands that read a spec also take `--set KEY=VALUE` (repeatable), which overrides any spec value by its dotted path, for example `--set synthesis.seed=7` or `--set 'synthesis.scale.tables={customer: 10}'` (the value is read as YAML).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Quality gate failed (`--fail-under`), or `verify` found a failing check |
| 2 | Invalid spec or arguments (including a spec that is not valid YAML) |
| 3 | Connection, permission or file error (also a target that exists with `if_exists: fail`) |
| 4 | Engine failure while fitting or sampling |
| 130 | Interrupted |

## Commands

| Command | What it does |
|---|---|
| [`introspect`](#introspect) | draft a spec from a database or folder |
| [`check`](#check) | validate a spec |
| [`validate`](#validate) | validate a spec (alias of check) |
| [`plan`](#plan) | which tables are fitted from in-database statistics (no fitting) |
| [`run`](#run) | fit, sample, write and score |
| [`extend`](#extend) | continue the event histories of a synthetic dataset step by step (new rows per day as CSV deltas) |
| [`verify`](#verify) | model checks on real data (R1-R7): scorecard, convergence, posterior predictive checks, scale rule, structure, stability |
| [`privacy`](#privacy) | attack a release per person and write a privacy report per release context |
| [`evaluate`](#evaluate) | score synthetic data already written |
| [`doctor`](#doctor) | causes and spec fixes for a run's failures |
| [`tune`](#tune) | run -> doctor -> apply until no fix remains |
| [`model`](#model) | the data model as Markdown (Mermaid ER diagram) and JSON |
| [`export-model`](#export-model) | the fitted Bayesian model as JSON, Markdown, PDF and Python |
| [`clone-schema`](#clone-schema) | create the source's tables (keys, types, indexes) empty in a target |
| [`profile`](#profile) | row, null and distinct counts computed in the source |
| [`ping`](#ping) | test a connection |
| [`generators`](#generators) | installed generator plug-ins and their capabilities |
| [`compute`](#compute) | GPUs found and the device each kind of generator would get |
| [`version`](#version) | versions of relsynth and its main dependencies |
| [`text`](#text) | synthetic text: model licences, PII recall, leakage audit, metrics |
| [`settings`](#settings) | engine settings: show, explain, validate, template, reference |
| [`autotune`](#autotune) | tune engine settings toward your objectives over several seeds; write the tuned spec and a constraints report |
| [`ir`](#ir) | IR 2.0 documents: migrate, validate, convert, privacy paths, ODCS, OpenLineage |
| [`gov`](#gov) | governance: registry, keys, policy, privacy budget, approvals and releases |

## introspect

Draft a spec from a database or folder.

```
usage: relsynth introspect [-h] [--source SOURCE] [--source-schema SOURCE_SCHEMA]
                           [--source-password-env SOURCE_PASSWORD_ENV]
                           [--source-url-env SOURCE_URL_ENV] [--out OUT] [--name NAME]
                           [--sample-rows SAMPLE_ROWS] [--include [INCLUDE ...]]
                           [--exclude [EXCLUDE ...]] [--evidence {names+data,data}]
                           [--ignore-declared] [--format {auto,v1,ir}] [--json]
                           [--log-level {debug,info,warning,error}]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--source` SOURCE |  |  | csv:DIR \| parquet:DIR \| sqlite:FILE \| database URL (postgresql://, mssql://, oracle://, mysql://, mariadb://, databricks://, fabric://, snowflake://, bigquery://); or give --source-url-env |
| `--source-schema` SOURCE_SCHEMA |  |  | database schema |
| `--source-password-env` SOURCE_PASSWORD_ENV |  |  | environment variable holding the password |
| `--source-url-env` SOURCE_URL_ENV |  |  | environment variable holding the whole URL |
| `--out` OUT |  | `spec.yaml` | where the drafted spec is written |
| `--name` NAME |  | `draft` | the spec's name |
| `--sample-rows` SAMPLE_ROWS |  | `20000` | random rows read per table |
| `--include` INCLUDE ... |  |  | table-name patterns to read (regular expressions) |
| `--exclude` EXCLUDE ... |  |  | table-name patterns to leave out, e.g. sys.* DatabaseLog |
| `--evidence` names+data\|data |  | `names+data` | identify keys and links from values and names (default) or from values only |
| `--ignore-declared` |  |  | hide the catalog's keys and links and identify them from the data (for testing) |
| `--format` auto\|v1\|ir |  | `auto` | write a relsynth 1.0 spec or an IR 2.0 document; auto (default) writes IR from release 2.0 and 1.0 before |

## check

Validate a spec.

```
usage: relsynth check [-h] [--set KEY=VALUE] [--settings FILE] [--connect] [--json]
                      [--log-level {debug,info,warning,error}]
                      spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--connect` |  |  | also connect to the source and find every table |

## validate

Validate a spec (alias of check).

```
usage: relsynth validate [-h] [--set KEY=VALUE] [--settings FILE] [--connect] [--json]
                         [--log-level {debug,info,warning,error}]
                         spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--connect` |  |  | also connect to the source and find every table |

## plan

Which tables are fitted from in-database statistics (no fitting).

```
usage: relsynth plan [-h] [--set KEY=VALUE] [--settings FILE] [--pushdown {auto,on,off}] [--json]
                     [--log-level {debug,info,warning,error}]
                     spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--pushdown` auto\|on\|off |  |  | override synthesis.pushdown.mode |

## run

Fit, sample, write and score.

```
usage: relsynth run [-h] [--set KEY=VALUE] [--settings FILE] [--scale SCALE] [--out OUT]
                    [--target TARGET] [--target-schema TARGET_SCHEMA]
                    [--target-password-env TARGET_PASSWORD_ENV] [--target-url-env TARGET_URL_ENV]
                    [--bundle BUNDLE] [--report REPORT] [--no-privacy] [--no-write]
                    [--datasets DATASETS] [--mode {draw,mean}] [--pushdown {auto,on,off}]
                    [--seed SEED] [--fail-under SHARE] [--registry REGISTRY] [--context CONTEXT]
                    [--release-target RELEASE_TARGET] [--source-id SOURCE_ID] [--key KEY]
                    [--privacy-report PRIVACY_REPORT] [--assess] [--privacy-config PRIVACY_CONFIG]
                    [--dp] [--epsilon EPSILON] [--dp-unit DP_UNIT] [--dp-statement DP_STATEMENT]
                    [--json] [--log-level {debug,info,warning,error}]
                    spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--scale` SCALE |  |  | size multiplier of the synthetic entities |
| `--out` OUT |  |  | output folder for CSV files (default: the spec's target) |
| `--target` TARGET |  |  | write to this location instead of the spec's target |
| `--target-schema` TARGET_SCHEMA |  |  | database schema |
| `--target-password-env` TARGET_PASSWORD_ENV |  |  | environment variable holding the password |
| `--target-url-env` TARGET_URL_ENV |  |  | environment variable holding the whole URL |
| `--bundle` BUNDLE |  | `diagnostics_bundle.json` | diagnostics bundle path ('' for none) |
| `--report` REPORT |  |  | also write the scorecard as Markdown |
| `--no-privacy` |  |  | skip distance-to-closest-record checks |
| `--no-write` |  |  | score only; write no synthetic data |
| `--datasets` DATASETS |  |  | number of synthetic datasets (independent posterior draws) |
| `--mode` draw\|mean |  |  | posterior draw (default) or posterior means |
| `--pushdown` auto\|on\|off |  |  | override synthesis.pushdown.mode |
| `--seed` SEED |  |  | override synthesis.seed |
| `--fail-under` SHARE |  |  | quality gate: exit 1 when the share of passed checks is below SHARE (0-1) |
| `--registry` REGISTRY |  |  | record a signed manifest of this run in this registry |
| `--context` CONTEXT |  |  | intended release context from the organisation policy |
| `--release-target` RELEASE_TARGET |  |  | intended target id from the context's allowed_targets |
| `--source-id` SOURCE_ID |  |  | the source's id in the organisation policy (selects its privacy domain) |
| `--key` KEY |  |  | the operator's private signing key |
| `--privacy-report` PRIVACY_REPORT |  |  | privacy_report.json of `relsynth privacy` for this spec |
| `--assess` |  |  | run the privacy assessment against the policy's contexts |
| `--privacy-config` PRIVACY_CONFIG |  |  | privacy.yaml for --assess (its contexts are replaced by the policy's) |
| `--dp` |  |  | differentially private run (the spec's synthesis.privacy.dp) |
| `--epsilon` EPSILON |  |  | this run's epsilon (implies --dp; overrides the spec) |
| `--dp-unit` DP_UNIT |  |  | the privacy unit table (default: the spec's, or the policy domain's) |
| `--dp-statement` DP_STATEMENT |  |  | also write the DP statement (JSON) here |

## extend

Continue the event histories of a synthetic dataset step by step (new rows per day as CSV deltas).

```
usage: relsynth extend [-h] [--set KEY=VALUE] [--settings FILE] --until UNTIL [--start START]
                       [--step {hour,day,week,month}] [--dataset DATASET] [--out OUT]
                       [--with-base] [--seed SEED] [--json]
                       [--log-level {debug,info,warning,error}]
                       spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--until` UNTIL | yes |  | end of the extension (date or datetime, exclusive) |
| `--start` START |  |  | start of the extension (default: the end of the source's time range) |
| `--step` hour\|day\|week\|month |  | `day` | delta period |
| `--dataset` DATASET |  |  | the base dataset (posterior draw number) |
| `--out` OUT |  | `deltas` | folder for delta-<period>/<table>.csv |
| `--with-base` |  |  | also write the base dataset to <out>/base |
| `--seed` SEED |  |  | override synthesis.seed |

## verify

Model checks on real data (R1-R7): scorecard, convergence, posterior predictive checks, scale rule, structure, stability.

```
usage: relsynth verify [-h] [--set KEY=VALUE] [--settings FILE] [--draws DRAWS] [--no-stability]
                       [--pushdown {auto,on,off}] [--seed SEED] [--no-privacy] [--target TARGET]
                       [--target-schema TARGET_SCHEMA] [--target-password-env TARGET_PASSWORD_ENV]
                       [--target-url-env TARGET_URL_ENV] [--report REPORT] [--bundle BUNDLE]
                       [--json] [--log-level {debug,info,warning,error}]
                       spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--draws` DRAWS |  | `5` | synthetic copies for the posterior predictive checks |
| `--no-stability` |  |  | skip the two extra fits (R6, R7) |
| `--pushdown` auto\|on\|off |  |  | default off: every table is loaded whole |
| `--seed` SEED |  |  | override synthesis.seed |
| `--no-privacy` |  |  | skip distance-to-closest-record checks |
| `--target` TARGET |  |  | also write the first synthetic database here |
| `--target-schema` TARGET_SCHEMA |  |  | database schema |
| `--target-password-env` TARGET_PASSWORD_ENV |  |  | environment variable holding the password |
| `--target-url-env` TARGET_URL_ENV |  |  | environment variable holding the whole URL |
| `--report` REPORT |  |  | write the checks as Markdown |
| `--bundle` BUNDLE |  |  | write a diagnostics bundle with the checks (for doctor) |

## privacy

Attack a release per person and write a privacy report per release context.

```
usage: relsynth privacy [-h] [--set KEY=VALUE] [--settings FILE] [--config CONFIG] [--out OUT]
                        [--cache CACHE] [--jobs JOBS] [--no-shadow] [--salt-file SALT_FILE]
                        [--context CONTEXT] [--fail-on {review,fail}]
                        [--signed-off-by SIGNED_OFF_BY] [--json]
                        [--log-level {debug,info,warning,error}]
                        spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--config` CONFIG |  |  | privacy.yaml: privacy unit, sensitive attributes, quasi-identifiers, contexts and limits (docs/privacy.md) |
| `--out` OUT |  | `privacy_report` | folder for privacy_report.json, .md and .html |
| `--cache` CACHE |  |  | folder that keeps shadow fits between runs |
| `--jobs` JOBS |  |  | parallel shadow fits |
| `--no-shadow` |  |  | skip shadow-model membership inference (fast) |
| `--salt-file` SALT_FILE |  |  | secret salt for person labels (hex); reused when it exists, else a new one is written there (default: OUT/privacy_salt.key) |
| `--context` CONTEXT |  |  | gate on this release context (internal, partner, public): exit 1 when its verdict reaches --fail-on |
| `--fail-on` review\|fail |  | `fail` |  |
| `--signed-off-by` SIGNED_OFF_BY |  |  | the person (the DPO) who signs off a 'review' verdict; without it a review that needs a sign-off counts as fail at the gate |

## evaluate

Score synthetic data already written.

```
usage: relsynth evaluate [-h] [--set KEY=VALUE] [--settings FILE] --synthetic SYNTHETIC
                         [--output OUTPUT] [--no-privacy] [--fail-under SHARE] [--quality FILE]
                         [--quality-config YAML] [--holdout LOC] [--quality-gate] [--json]
                         [--log-level {debug,info,warning,error}]
                         spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--synthetic` SYNTHETIC | yes |  | where the synthetic tables are (csv:DIR, sqlite:FILE, URL) |
| `--output` OUTPUT |  |  | write every metric as JSON |
| `--no-privacy` |  |  | skip distance-to-closest-record checks |
| `--fail-under` SHARE |  |  | quality gate: exit 1 when the share of passed checks is below SHARE (0-1) |
| `--quality` FILE |  |  | also write the calibrated 2.0 quality report (fidelity, utility, integrity) as JSON |
| `--quality-config` YAML |  |  | settings of the 2.0 report (see docs/metrics.md) |
| `--holdout` LOC |  |  | real tables the model did not see, for TSTR/TRTS |
| `--quality-gate` |  |  | with --quality: exit 1 when an area of the 2.0 quality vector fails (the default gate stays --fail-under until 3.0) |

## doctor

Causes and spec fixes for a run's failures.

```
usage: relsynth doctor [-h] [--spec SPEC] [--apply] [--out OUT] [--report REPORT] [--json]
                       [--log-level {debug,info,warning,error}]
                       bundle
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `bundle` | yes |  | diagnostics bundle written by run or verify |
| `--spec` SPEC |  |  | the spec to patch (default: the copy inside the bundle) |
| `--apply` |  |  | write the fixed spec |
| `--out` OUT |  |  | where --apply writes the fixed spec (default: <spec>.fixed.yaml) |
| `--report` REPORT |  |  | also write the report as Markdown |

## tune

Run -> doctor -> apply until no fix remains.

```
usage: relsynth tune [-h] [--rounds ROUNDS] [--out OUT] [--bundle BUNDLE] [--json]
                     [--log-level {debug,info,warning,error}]
                     spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--rounds` ROUNDS |  | `3` | largest number of run -> doctor -> apply rounds |
| `--out` OUT |  |  | where the best spec is written (default: <spec>.tuned.yaml) |
| `--bundle` BUNDLE |  |  | also write the best round's diagnostics bundle |

## model

The data model as Markdown (Mermaid ER diagram) and JSON.

```
usage: relsynth model [-h] [--out OUT] [--json] [--log-level {debug,info,warning,error}] spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--out` OUT |  |  | output path without extension (default: <spec>.model) |

## export-model

The fitted Bayesian model as JSON, Markdown, PDF and Python.

```
usage: relsynth export-model [-h] [--set KEY=VALUE] [--settings FILE] [--out OUT] [--draws DRAWS]
                             [--formats FORMATS] [--no-replica-check] [--stan] [--json]
                             [--log-level {debug,info,warning,error}]
                             spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value, e.g. --set synthesis.scale.default=2 (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) merged under the spec's synthesis block; see docs/settings.md |
| `--out` OUT |  |  | output folder (default: <spec>.model) |
| `--draws` DRAWS |  | `5` | posterior draws to export (draw m = dataset m of `run`) |
| `--formats` FORMATS |  | `json,md,pdf,py` | any of json, md, pdf, py |
| `--no-replica-check` |  |  | skip scoring the standalone replica |
| `--stan` |  |  | also check the posterior blocks with Stan (needs pystan) |

## clone-schema

Create the source's tables (keys, types, indexes) empty in a target.

```
usage: relsynth clone-schema [-h] [--source SOURCE] [--source-schema SOURCE_SCHEMA]
                             [--source-password-env SOURCE_PASSWORD_ENV]
                             [--source-url-env SOURCE_URL_ENV] [--target TARGET]
                             [--target-schema TARGET_SCHEMA]
                             [--target-password-env TARGET_PASSWORD_ENV]
                             [--target-url-env TARGET_URL_ENV] [--tables [TABLES ...]] [--replace]
                             [--json] [--log-level {debug,info,warning,error}]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--source` SOURCE |  |  | csv:DIR \| parquet:DIR \| sqlite:FILE \| database URL (postgresql://, mssql://, oracle://, mysql://, mariadb://, databricks://, fabric://, snowflake://, bigquery://); or give --source-url-env |
| `--source-schema` SOURCE_SCHEMA |  |  | database schema |
| `--source-password-env` SOURCE_PASSWORD_ENV |  |  | environment variable holding the password |
| `--source-url-env` SOURCE_URL_ENV |  |  | environment variable holding the whole URL |
| `--target` TARGET |  |  | csv:DIR \| parquet:DIR \| sqlite:FILE \| database URL (postgresql://, mssql://, oracle://, mysql://, mariadb://, databricks://, fabric://, snowflake://, bigquery://); or give --target-url-env |
| `--target-schema` TARGET_SCHEMA |  |  | database schema |
| `--target-password-env` TARGET_PASSWORD_ENV |  |  | environment variable holding the password |
| `--target-url-env` TARGET_URL_ENV |  |  | environment variable holding the whole URL |
| `--tables` TABLES ... |  |  | only these tables (default: all) |
| `--replace` |  |  | drop existing tables first |

## profile

Row, null and distinct counts computed in the source.

```
usage: relsynth profile [-h] [--source SOURCE] [--source-schema SOURCE_SCHEMA]
                        [--source-password-env SOURCE_PASSWORD_ENV]
                        [--source-url-env SOURCE_URL_ENV] [--tables [TABLES ...]] [--top TOP]
                        [--output OUTPUT] [--json] [--log-level {debug,info,warning,error}]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--source` SOURCE |  |  | csv:DIR \| parquet:DIR \| sqlite:FILE \| database URL (postgresql://, mssql://, oracle://, mysql://, mariadb://, databricks://, fabric://, snowflake://, bigquery://); or give --source-url-env |
| `--source-schema` SOURCE_SCHEMA |  |  | database schema |
| `--source-password-env` SOURCE_PASSWORD_ENV |  |  | environment variable holding the password |
| `--source-url-env` SOURCE_URL_ENV |  |  | environment variable holding the whole URL |
| `--tables` TABLES ... |  |  | only these tables (default: all) |
| `--top` TOP |  |  | also list the N most frequent values (shares data values) |
| `--output` OUTPUT |  |  | write the profile as JSON |

## ping

Test a connection.

```
usage: relsynth ping [-h] [--source SOURCE] [--source-schema SOURCE_SCHEMA]
                     [--source-password-env SOURCE_PASSWORD_ENV] [--source-url-env SOURCE_URL_ENV]
                     [--json] [--log-level {debug,info,warning,error}]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--source` SOURCE |  |  | csv:DIR \| parquet:DIR \| sqlite:FILE \| database URL (postgresql://, mssql://, oracle://, mysql://, mariadb://, databricks://, fabric://, snowflake://, bigquery://); or give --source-url-env |
| `--source-schema` SOURCE_SCHEMA |  |  | database schema |
| `--source-password-env` SOURCE_PASSWORD_ENV |  |  | environment variable holding the password |
| `--source-url-env` SOURCE_URL_ENV |  |  | environment variable holding the whole URL |

## generators

Installed generator plug-ins and their capabilities.

```
usage: relsynth generators [-h] [--json] [--log-level {debug,info,warning,error}]
                           [{list,show}] [name]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `action` | yes | `list` |  |
| `name` | yes |  | generator name (for show) |

## compute

GPUs found and the device each kind of generator would get.

```
usage: relsynth compute [-h] [--preset PRESET] [--file FILE] [--json]
                        [--log-level {debug,info,warning,error}]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `--preset` PRESET |  | `cpu` | cpu \| gpu-auto \| gpu-single \| gpu-multi \| a preset of --file |
| `--file` FILE |  |  | a compute.yaml with more presets |

## version

Versions of relsynth and its main dependencies.

```
usage: relsynth version [-h] [--json] [--log-level {debug,info,warning,error}]
```


## text

Synthetic text: model licences, PII recall, leakage audit, metrics.

```
usage: relsynth text [-h] [--use USE] [--labelled LABELLED] [--recognizer RECOGNIZER] [--n N]
                     [--seed SEED] [--table TABLE] [--column COLUMN] [--owner OWNER]
                     [--canaries CANARIES] [--repeats REPEATS] [--words WORDS] [--samples SAMPLES]
                     [--condition CONDITION] [--json] [--log-level {debug,info,warning,error}]
                     {models,pii-eval,audit,metrics} [target] [synthetic]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `action` | yes |  |  |
| `target` | yes |  | audit: spec; metrics: real CSV |
| `synthetic` | yes |  | metrics: synthetic CSV |
| `--use` USE |  |  | models: lm, pii, embedding or mauve-features |
| `--labelled` LABELLED |  |  | pii-eval: JSON lines {text, spans: [[start, end, type]]} |
| `--recognizer` RECOGNIZER |  |  | pii-eval: regex, presidio, spacy=<model>, gliner=<folder> |
| `--n` N |  | `1000` | pii-eval: size of the generated labelled set |
| `--seed` SEED |  | `1506` |  |
| `--table` TABLE |  |  |  |
| `--column` COLUMN |  |  |  |
| `--owner` OWNER |  |  | audit: column of the person (default: text.dp.unit_column, else the row) |
| `--canaries` CANARIES |  | `64` |  |
| `--repeats` REPEATS |  | `4` |  |
| `--words` WORDS |  | `16` |  |
| `--samples` SAMPLES |  | `2000` |  |
| `--condition` CONDITION |  |  | metrics: conditioning column (repeatable) |

## settings

Engine settings: show, explain, validate, template, reference.

```
usage: relsynth settings [-h] [--all] [--out OUT] [--json]
                         [--log-level {debug,info,warning,error}]
                         {show,explain,validate,template,reference} [target]
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `action` | yes |  |  |
| `target` | yes |  | spec (show), key (explain) or settings file (validate) |
| `--all` |  |  | show: every key, not only the ones set |
| `--out` OUT |  |  | template / reference: write to this file |

## autotune

Tune engine settings toward your objectives over several seeds; write the tuned spec and a constraints report.

```
usage: relsynth autotune [-h] [--set KEY=VALUE] [--settings FILE] [--objectives OBJECTIVES]
                         [--runs RUNS] [--seeds SEEDS] [--holdout HOLDOUT] [--history HISTORY]
                         [--fresh] [--out OUT] [--report REPORT] [--json]
                         [--log-level {debug,info,warning,error}]
                         spec
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `spec` | yes |  | the YAML specification |
| `--set` KEY=VALUE |  |  | override a spec value before tuning, e.g. --set source.path=db.sqlite (repeatable) |
| `--settings` FILE |  |  | an engine settings file (or preset:fast \| preset:thorough) as the starting point |
| `--objectives` OBJECTIVES |  |  | objectives.yaml (default: every hard constraint and every statistical family) |
| `--runs` RUNS |  |  | budget: fits (one candidate on one seed is one run) |
| `--seeds` SEEDS |  |  | search seeds, e.g. 1,2,3 (overrides the objectives) |
| `--holdout` HOLDOUT |  |  | held-out seeds, e.g. 11,12 ('' for none) |
| `--history` HISTORY |  |  | history folder (default: <spec>.tune) |
| `--fresh` |  |  | forget this spec's earlier runs |
| `--out` OUT |  |  | tuned spec (default: <spec>.tuned.yaml) |
| `--report` REPORT |  |  | report file (default: <history>/report.md, plus report.json) |

## ir

IR 2.0 documents: migrate, validate, convert, privacy paths, ODCS, OpenLineage.

```
usage: relsynth ir [-h] [--json] [--log-level {debug,info,warning,error}] IR_COMMAND ...
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `ir_cmd` | yes |  |  |

## gov

Governance: registry, keys, policy, privacy budget, approvals and releases.

```
usage: relsynth gov [-h] [--json] [--log-level {debug,info,warning,error}] ACTION ...
```

| Option | Required | Default | Meaning |
|---|---|---|---|
| `gov_cmd` | yes |  |  |

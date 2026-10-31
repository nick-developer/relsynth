# Python API reference

Generated from the code by `python docs/make_api_reference.py`. Everything listed here is importable from `relsynth` (`import relsynth as rs`) and covered by `tests/unit/test_api.py`.

## A typical session

```python
import relsynth as rs

spec = rs.introspect("postgresql://analyst@db/sales", password_env="PGPASSWORD")
spec.errors()                                   # [] when the spec is valid
model = rs.fit(spec)                            # FittedModel
data = model.sample(scale=0.5)                  # SyntheticData: {table: DataFrame}
report = model.evaluate(data)                   # EvaluationReport
report.require(0.98)                            # raises QualityGateError below 98%
data.write("sqlite:synthetic.db")
```

Specs can be given to every function as a path, a dict or a `Spec`. Locations are strings (`csv:DIR`, `parquet:DIR`, `sqlite:FILE`, a database URL) or sections (`{"kind": "postgresql", "url_env": "DB_URL", "schema": "dw"}`).

## Errors

| Exception | Raised when | CLI exit code |
|---|---|---|
| `SpecError` (a `ValueError`) | The spec or an argument is invalid | 2 |
| `SourceError` (a `RelsynthError` and an `OSError`) | A connection, permission, file or write fails | 3 |
| `EngineError` | Fitting or sampling fails | 4 |
| `QualityGateError` | `require()` or `fail_under` finds the score too low | 1 |

Messages never contain passwords.

## Functions

### `rs.introspect`

```python
introspect(source, *, schema: str | None = None, password_env: str | None = None, url_env: str | None = None, name: str = 'draft', sample_rows: int = 20000, include=None, exclude=None, evidence: str = 'names+data', declared: bool = True) -> Spec
```

Draft a spec from a database or folder: tables, column types, keys and
links (declared in the catalog, or identified from the data), roles, rules.
Review `spec.review` before running.  source: a location ('postgresql://...',
'sqlite:file.db', 'csv:dir', 'parquet:dir') or a spec source section.

### `rs.validate_spec`

```python
validate_spec(spec) -> list[str]
```

Every validation error of a spec (empty list when valid).

### `rs.fit`

```python
fit(spec, *, pushdown: str | None = None, seed: int | None = None) -> FittedModel
```

Fit the model to the spec's source.  Fact tables of at least
synthesis.pushdown.min_rows rows are fitted from in-database statistics and a
uniform sample (pushdown: 'auto' (default), 'on' or 'off').

### `rs.sample`

```python
sample(spec, scale=None, *, dataset: int = 0, pushdown: str | None = None) -> SyntheticData
```

Fit and draw one synthetic database (see fit and FittedModel.sample).

### `rs.evaluate`

```python
evaluate(spec, synthetic: SyntheticData | Mapping, *, real: Mapping | None = None, privacy: bool = True) -> EvaluationReport
```

Scorecard of synthetic tables against real ones (default: the spec's
source, with large facts as samples plus exact statistics).

### `rs.assess_quality`

```python
assess_quality(spec, synthetic: SyntheticData | Mapping, *, real: Mapping | None = None, holdout: Mapping | None = None, config: dict | None = None, privacy: bool = True) -> QualityReport
```

The calibrated 2.0 quality report of synthetic tables (relsynth.quality.assess).
real: the tables the model was fitted on (default: the spec's source);
holdout: real tables the model did not see, for TSTR/TRTS (default: an internal
split, reported as optimistic); config: overrides of relsynth.quality.DEFAULTS
(scale defaults to the spec's synthesis.scale).

### `rs.run`

```python
run(spec, *, scale=None, out=None, datasets: int | None = None, mode: str | None = None, pushdown: str | None = None, seed: int | None = None, privacy: bool = True, write: bool = True, bundle: str | Path | None = None, fail_under: float | None = None, raise_errors: bool = True) -> RunResult
```

The whole pipeline: load (with pushdown) -> fit -> sample -> write ->
evaluate -> diagnostics bundle.  out: folder or target section (default: the
spec's target); datasets: independent posterior draws to write; mode: 'draw'
(default) or 'mean'; fail_under: quality gate on the share of passed checks.
raise_errors=False returns a RunResult with `errors` instead of raising (the
bundle explains them; see doctor).

### `rs.verify`

```python
verify(spec, *, draws: int = 5, stability: bool = True, out=None, privacy: bool = True, pushdown: str = 'off', seed: int | None = None)
```

Model checks on real data, where no ground truth exists (R1-R7: scorecard,
copula convergence, posterior predictive checks of key statistics, scale rule,
structural validity, stability across seeds, reproducibility).  Returns a
relsynth.verify.VerifyReport (passed, total, ok, failed(), to_markdown(),
to_json(), save_bundle()).  stability=False skips the two extra fits.

### `rs.doctor`

```python
doctor(bundle: RunResult | dict | str | Path) -> Diagnosis
```

Diagnose a run from its bundle (a RunResult, a bundle dict or a bundle
file): failed checks and errors mapped to causes and spec fixes.

### `rs.tune`

```python
tune(spec, rounds: int = 3) -> tuple[Spec, dict, list]
```

run -> doctor -> apply until no fix remains: (best spec, its bundle, history).

### `rs.export_model`

```python
export_model(spec, out_dir, **kw) -> dict
```

Fit and export the model (see FittedModel.export).

### `rs.extend`

```python
extend(model: FittedModel, until, *, base: SyntheticData | Mapping | None = None, dataset: int = 0, start=None, step: str = 'day') -> list
```

Continue the event histories (tables with a `sequence` block) of synthetic
dataset `dataset` past its end: [(period start, period end, {table: new
rows})] per step.  The rows up to a day do not depend on how far or in how
many steps the history is extended (relsynth.timeseries.incremental).

### `rs.clone_schema`

```python
clone_schema(source, target, *, tables: list[str] | None = None, schema: str | None = None, replace: bool = False) -> list[str]
```

Create the source's tables (types, primary and unique keys, foreign keys,
indexes, identity columns) empty in a target database.  Returns the tables
created.  source/target: locations or spec sections of two databases.

### `rs.profile`

```python
profile(source, *, tables: list[str] | None = None, top: int = 0) -> dict
```

Row counts and per-column null and distinct counts computed inside the
source (no rows leave the database).  top > 0 adds the most frequent values
of each column - only use it where those values may be shared.

### `rs.ping`

```python
ping(source) -> dict
```

Connect to a source or target and report what is there (no data read).

### `rs.connect`

```python
connect(source, **kw)
```

A connector to a database or folder (relsynth.connectors.connect).

### `rs.load_spec`

```python
load_spec(path_or_dict) -> dict
```



### `rs.dump_spec`

```python
dump_spec(spec: dict, path)
```



## Classes

### `rs.Spec`

A relsynth specification: the tables, their columns and roles, the
relationships, the business rules, and the synthesis settings.

`raw` is the YAML document as written (what `save` writes back); `resolved`
is the validated form with every default filled in (what the engine reads).
Relative paths in source and target sections resolve against `base_dir` (the
spec file's folder).

Properties: `relationships`, `resolved`, `review`, `tables`.

#### `Spec.check`

```python
check(self) -> Spec
```

Raise SpecError listing every problem when the spec is invalid.

#### `Spec.coerce`

```python
coerce(spec: Spec | str | Path | dict) -> Spec
```

A Spec from a Spec, a YAML path or a dictionary.

#### `Spec.copy`

```python
copy(self) -> Spec
```

#### `Spec.errors`

```python
errors(self) -> list[str]
```

Every validation error (empty when the spec is valid).

#### `Spec.from_dict`

```python
from_dict(d: dict, base_dir: str | Path | None = None) -> Spec
```

#### `Spec.load`

```python
load(path: str | Path) -> Spec
```

Read a YAML spec (safe loader: no Python objects, no aliases bombs).

#### `Spec.model_report`

```python
model_report(self, out_stem: str | Path | None = None) -> str
```

The data model as Markdown (tables, keys, an ER diagram in Mermaid,
rules); also written to <out_stem>.md and .json when out_stem is given.

#### `Spec.save`

```python
save(self, path: str | Path, mask_secrets: bool = True) -> Path
```

Write the spec as YAML.  Passwords in URLs are masked (the spec then
reads the URL from source.url_env / target.url_env).

#### `Spec.set`

```python
set(self, key: str, value: Any) -> Spec
```

Set a dotted key (synthesis.scale.default, target.schema,
tables.orders.mode) on a copy and return it.  Values given as strings are
parsed as YAML scalars ("2" -> 2, "true" -> True, "[a, b]" -> list).

#### `Spec.to_yaml`

```python
to_yaml(self, mask_secrets: bool = True) -> str
```

#### `Spec.with_source`

```python
with_source(self, source) -> Spec
```

A copy reading from another source (a spec section or a location such as
'postgresql://...', 'sqlite:file.db', 'csv:dir', 'parquet:dir').

#### `Spec.with_target`

```python
with_target(self, target) -> Spec
```

A copy writing to another target (spec section or location).

### `rs.FittedModel`

A fitted Bayesian relational model: the posterior over latent segments, link
and count rates, and per-segment attribute models (relsynth.engine.Synthesizer).

Properties: `log`, `pushdown`.

#### `FittedModel.evaluate`

```python
evaluate(self, data: SyntheticData | Mapping, privacy: bool = True) -> EvaluationReport
```

Scorecard of `data` against the source the model was fitted on.

#### `FittedModel.export`

```python
export(self, out_dir: str | Path, formats=('json', 'md', 'pdf', 'py'), draws: int = 5, replica_check: bool = True, stan: bool = False) -> dict
```

Write the model (structure, posterior means and draws, verification
checks) as JSON, Markdown, PDF and a standalone Python sampler.

#### `FittedModel.sample`

```python
sample(self, scale: float | dict | None = None, dataset: int | None = None) -> SyntheticData
```

One synthetic database.  dataset numbers independent posterior draws
(0, 1, ...; default: the next one); scale multiplies the entity sizes (a
number for every entity, {table: factor} for some, or the spec's section
{"default": x, "tables": {name: y}}).

#### `FittedModel.summary`

```python
summary(self) -> dict
```

Segments, relations and attribute models in brief (JSON-ready).

### `rs.SyntheticData`

One synthetic database: a read-only mapping table name -> DataFrame, in the
spec's table order.

Properties: `rows`.

#### `SyntheticData.to_csv`

```python
to_csv(self, folder: str | Path) -> Path
```

#### `SyntheticData.to_dict`

```python
to_dict(self) -> dict[str, pandas.DataFrame]
```

#### `SyntheticData.to_parquet`

```python
to_parquet(self, folder: str | Path) -> Path
```

#### `SyntheticData.write`

```python
write(self, target=None, *, if_exists: str | None = None, structure: str | None = None) -> dict
```

Write the tables (parents first) to `target` (a spec target section or a
location; default the spec's target).  if_exists: replace, append, truncate
or fail; structure: auto, clone, infer or existing.  Returns the target.

### `rs.EvaluationReport`

The comparison of synthetic with real data: every metric (`raw`) and the
scorecard of pass/fail checks, each with its limit and significance level.

Properties: `failed`, `ok`, `passed`, `score`, `scorecard`, `total`.

#### `EvaluationReport.require`

```python
require(self, min_score: float = 1.0) -> EvaluationReport
```

Quality gate: raise QualityGateError when the share of passed checks is
below min_score.

#### `EvaluationReport.to_dict`

```python
to_dict(self) -> dict
```

#### `EvaluationReport.to_json`

```python
to_json(self, path: str | Path | None = None) -> str
```

#### `EvaluationReport.to_markdown`

```python
to_markdown(self) -> str
```

### `rs.QualityReport`

The 2.0 quality report (`raw`, schema quality-report-2.0): every check with
its definition, null limit and confidence level, and the quality vector
(fidelity, utility, integrity).  `scorecard` is the 1.0 report it includes.

Properties: `failed`, `ok`, `vector`.

#### `QualityReport.require`

```python
require(self, areas=('fidelity', 'utility', 'integrity')) -> QualityReport
```

Quality gate: raise QualityGateError when one of `areas` fails its pass rule.

#### `QualityReport.to_dict`

```python
to_dict(self) -> dict
```

#### `QualityReport.to_json`

```python
to_json(self, path: str | Path | None = None) -> str
```

### `rs.RunResult`

One pipeline run: the synthetic data, the model, the scorecard, the
diagnostics bundle (for `doctor`) and the errors of failed steps.

Properties: `ok`.

#### `RunResult.save_bundle`

```python
save_bundle(self, path: str | Path) -> Path
```

### `rs.VerifyReport`

The R1-R7 checks of one spec: `checks` (id, what, value, target, pass),
`errors` of checks that could not run, and the run's diagnostics bundle.

Properties: `ok`, `passed`, `total`.

#### `VerifyReport.failed`

```python
failed(self) -> list[dict]
```

#### `VerifyReport.save_bundle`

```python
save_bundle(self, path: str | Path) -> Path
```

The diagnostics bundle with the checks added (aggregates only).

#### `VerifyReport.to_dict`

```python
to_dict(self) -> dict
```

#### `VerifyReport.to_json`

```python
to_json(self, path: str | Path | None = None) -> str
```

#### `VerifyReport.to_markdown`

```python
to_markdown(self) -> str
```

### `rs.Diagnosis`

What `doctor` found in a diagnostics bundle: issues (each with its cause,
evidence and, where possible, a spec patch) and the merged patches.

#### `Diagnosis.fixed_spec`

```python
fixed_spec(self, spec) -> Spec
```

A copy of the spec with every patch applied.

### `rs.Synthesizer`



#### `Synthesizer.fit`

```python
fit(self, tables: dict[str, pandas.DataFrame], stats=None)
```

Fit the model.  `stats` (pushdown) maps fact tables given as uniform
samples to their exact statistics (relsynth.pushdown.stats.TableStats):
the relational model reads those facts only through the statistics.

#### `Synthesizer.model_summary`

```python
model_summary(self)
```

#### `Synthesizer.redraw`

```python
redraw(self, mode='draw', rng=None)
```

One posterior draw (posterior means in mean mode) of every parameter
given the fitted partitions: segment weights, link and block rates,
occupancy and activity probabilities, marginals, copula matrices, time
profiles, extra-dimension laws, refined segment weights and rake targets.
A dataset sampled after it is a draw from the posterior predictive
distribution (Rubin 1993); the per-call draws of the samplers (count-law
and tail weights) are fixed for the run as well (see posterior.py).

#### `Synthesizer.sample`

```python
sample(self, scale=None, index=None)
```

One synthetic dataset: every parameter drawn once from its posterior
(or set to its posterior mean) from the dataset's parameter stream, then
the rows from its data stream.  index numbers the datasets (0, 1, ...);
without it each call gives the next dataset.  relsynth.runtime runs the
same steps with the facts split into partitions (sample_partitioned).

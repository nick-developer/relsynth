# Engine settings

Every setting of the engine can be changed by the user: chain lengths, sample sizes, priors, thresholds of the
detection steps, repair rounds and the scorecard limits. This page explains where settings come from and how to
set them. The full list of keys, with type, default, range and meaning, is in
[settings-reference.md](settings-reference.md) (generated from the code, `relsynth settings reference`).

## Where a setting comes from

From lowest to highest precedence:

1. **Built-in defaults.** With no settings at all, the output is byte for byte what it was before settings
   existed (the CI job `settings-autotune` checks this against `main` on every change).
2. **A settings file**, named explicitly in one of these ways (no file is ever picked up implicitly):
   - `synthesis.settings_file: tuning/fast.yaml` in the spec (a relative path is relative to the spec);
   - `--settings FILE` on `run`, `fit`, `sample`, `verify`, `autotune` and the other spec commands;
   - the environment variable `RELSYNTH_SETTINGS=FILE`;
   - a preset: `preset:fast` (about 2-3x faster fits for drafts; the convergence rules still apply) or
     `preset:thorough` (longer chains and larger samples).
3. **The spec's own `synthesis:` block.**
4. **`--set KEY=VALUE`** on the command line, or `Spec.set()` in Python.

`relsynth settings show spec.yaml` prints the effective value of every key that is set and where it came from
(`default`, the settings file, or `spec`); `--all` lists every key.

## A settings file

A settings file holds the `synthesis` groups, either at the top level or under `synthesis:`. It is checked as
strictly as a spec: unknown keys are refused with a suggestion, and every value is checked against its type and
range.

```yaml
# tuning.yaml
gibbs: {iterations: 200}
copula: {sample_rows: 5000, lkj_eta: 4}
marginals: {dirichlet_prior: 0.1}
repair: {constraint_rounds: 400}          # rounds of the constraint repair before rows are dropped
detection: {scd_stable_share: 0.98}       # thresholds of the structure detection steps
per_table:                                # overrides for one table (groups copula, marginals, text)
  InvoiceLine: {copula: {sample_rows: 20000}}
evaluation:
  thresholds: {ks: 0.06, auc: 0.60}       # scorecard limits (the keys of relsynth.evaluate.THRESHOLDS)
  assoc_max_rows: 50000                   # rows used by the association checks
```

`relsynth settings template --out tuning.yaml` writes a commented file with every key at its default.
`relsynth settings validate tuning.yaml` checks a file. `relsynth settings explain copula.sample_rows` prints one
key with its range, whether autotune may move it, and the checks it affects.

## In Python

The same settings are editable attributes, validated when you assign them:

```python
import relsynth as rs

s = rs.Settings.load("spec.yaml")            # defaults < settings file < spec
s.copula.sample_rows = 5000                   # checked against type and range
s.evaluation.thresholds.ks = 0.06
s.table("InvoiceLine").copula.sample_rows = 20000
s.copula.sampl_rows = 1                       # AttributeError: did you mean 'copula.sample_rows'?

spec = rs.Spec.load("spec.yaml").with_settings(s)
model = rs.fit(spec)                          # or rs.run(spec, settings=s)
s.save("tuning.yaml")                         # only the keys that differ from the defaults
```

`Spec.settings` returns the effective settings of a spec; `Settings.to_dict()` returns the changed keys only, and
`Settings.merged()` the full synthesis block.

## Which settings to change

- **Speed**: `gibbs.iterations`, `copula.sample_rows`, `copula.iterations`, or `preset:fast`.
- **Associations between columns** (association, discriminator and k-way checks): `copula.sample_rows`,
  `copula.lkj_eta`, `copula.mode`, `marginals.joint_min_cramers_v`.
- **Rows per parent** (cardinality checks): `facts.count_model`, `facts.frailty`, `facts.link_degree`,
  `segments.default_cap`.
- **Categories and rare values**: `marginals.dirichlet_prior`, `marginals.max_levels`.
- **Rules and keys**: `rules.max_rejection_rounds`, `repair.*`.

When you do not know which setting to move, let [`relsynth autotune`](autotune.md) search for it: it uses the
same registry (each key records which checks it affects and in which direction).

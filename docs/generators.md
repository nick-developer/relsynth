# Generator plug-ins

relsynth splits a synthetic database into two parts.

- **The relational skeleton** is always relsynth's: which rows exist, their keys, the links between tables, each
  row's latent segment, row counts and scaling, hard rules, uniqueness and the random streams.
- **The attribute columns of each table** (numbers, categories, dates, flags) come from a *generator*.

A generator is fitted once per table and then asked, for every synthetic row, for attribute values given the row's
context. The default generator, `copula`, is relsynth 1.0's Bayesian Gaussian copula; with it, relsynth gives
byte-identical output to 1.0 for the same seed (Prop. 702). Other generators can be chosen per table, and anyone
can publish a generator as a separate Python package.

This page covers choosing a generator in a spec, the built-in generators, what the engine still guarantees, and how
to write a plug-in. The design record is [adr/0002-generator-plugins.md](adr/0002-generator-plugins.md). The
propositions are in [theory.md](theory.md#generator-plug-ins-v20) and `docs/proofs/sec_v20_generators.tex`.

## Choosing a generator

Set one default for the whole spec, and override it per table:

```yaml
synthesis:
  generator: copula            # the default for every table (this is also relsynth's default)

tables:
  customer:
    generator: cart            # a name...
  invoice_line:
    role: fact
    generator:                 # ...or a mapping
      name: cart
      version: ">=1.0,<2"      # optional pin of the generator's model version
      options: {min_leaf: 10}  # passed to the generator
```

Copied tables (`mode: copy`) and views are not generated, so they take no `generator`.

`relsynth check` validates the choice before anything is fitted. The generator must be installed, its version must
satisfy the pin, and it must be able to model the table: its column types, its IR version, and pushdown when the
table is fitted from in-database statistics. Run `relsynth generators list` to see the installed generators and
`relsynth generators show NAME` for one generator's capabilities.

The fit log (`model.log["decisions"]["generators"]`) and the model summary record every non-default choice: the
generator, its model version, its package and package version, its options and its device.

## Built-in generators

| Name | Model | Conditions on | Posterior | Cost |
|---|---|---|---|---|
| `copula` (default) | Dirichlet / Bayesian-bootstrap marginals per latent segment, shrunk table → coarse group → segment; a rank-likelihood Gaussian copula (LKJ prior) per segment for entities, pooled for facts (Hoff 2007). | segment, parent attributes | yes | MCMC per table |
| `independent` | The copula's marginals with an identity correlation: a latent class model (Goodman 1974; Dunson and Xing 2009). Columns depend on each other only through the row's segment. | segment | yes | no MCMC, fastest |
| `cart` | Sequential CART in the style of synthpop (Reiter 2005; Nowok, Raab and Dibben 2016). One tree per column on the segment, the coarse group, parent attributes and the earlier columns; leaf values drawn from the leaf's observed rows with Bayesian-bootstrap weights (Rubin 1981). | segment, parent attributes | approximate (Prop. 706) | one tree per column |

**Options of `cart`.**

| Option | Default | Meaning |
|---|---|---|
| `min_leaf` | 5 | Fewest rows in a leaf (synthpop `minbucket`). |
| `max_depth` | 12 | Deepest tree. |
| `bins` | 32 | Candidate thresholds per numeric predictor (quantiles). |
| `max_rows` | 50000 | Rows a tree is fitted on (a seeded uniform sample beyond that). |
| `min_gain` | 1e-8 | Smallest relative impurity decrease for a split (rpart `cp`). |
| `smoothing` | `none` | `density` adds Gaussian kernel noise to numbers drawn from a leaf (synthpop `smoothing`), so that no observed number is released exactly. |

**Privacy.** None of the built-in generators gives a formal privacy guarantee (`privacy_cost().dp` is false).
`cart` without smoothing releases observed values of each column, as synthpop's CART does by default. The copula's
continuous marginals are a Bayesian bootstrap of the observed values too. Measure every release with the privacy
attacks (`relsynth.privacy`). Differential privacy comes with T07: a generator then declares `dp: input` or
`dp: output` and is given a budget.

## What the engine guarantees, and what the generator owns

| Property | Who | Statement |
|---|---|---|
| Primary keys unique and non-null; foreign keys point at existing rows; trees acyclic; row counts, scale and link structure | engine | Prop. 700. These do not depend on the generator. |
| Derive, aggregate and cumulative rules; computed columns | engine | Prop. 700. Recomputed from the generated values, so exact. |
| Check rules and unique attribute keys | engine, by rejection | Prop. 700. Rows that break a rule are redrawn from the generator's own conditional, so the accepted law is the generator's law given the rules. Violations left after `synthesis.rules.max_rejection_rounds` are reported, as in 1.0. |
| Same seed, same data → same dataset | both | Prop. 701. The engine passes its seeded generator; the plug-in must draw every random number from it and be deterministic on its device. |
| The joint law of a row's attributes given its context | generator | Prop. 704. With `group` conditioning only, a fact's attributes depend on its parents' attributes only through the segments. |
| Posterior predictive datasets | generator | Prop. 703. True when `posterior_draw` draws the generator's parameters from their posterior. |
| Marginal fidelity, column dependence, rare values, privacy | generator | Measured by the scorecard and the attack suite. |

## Writing a plug-in

A plug-in is a subclass of `relsynth.generators.TableGenerator` plus one entry point. The complete example in
[`examples/plugins/relsynth-example-generator`](../examples/plugins/relsynth-example-generator) has about 60 lines.

```python
import numpy as np
from relsynth.generators import Capabilities, TableGenerator

class MyGenerator(TableGenerator):
    name = "mygen"                     # the name used in specs
    version = "1.0"                    # model format and algorithm version (pinned by specs, checked on load)
    description = "One line for `relsynth generators list`."
    capabilities = Capabilities(
        column_kinds=("integer", "float", "categorical", "boolean", "date", "datetime"),
        conditioning=("group", "parent"),   # group is required; parent = parent attributes of fact rows
        dp="none",                          # none | input | output
        gpu="optional",                     # never | optional | required (resolved by compute.yaml)
        pushdown=True,                      # can fit on the bounded sample of a pushed-down fact
        posterior=True,                     # posterior_draw draws new parameters
    )

    def fit(self, data, context, privacy_budget=None):
        ...                                 # read data.X / data.codecs, context.groups / context.rng
        return self

    def sample(self, n, conditioning, rng):
        out = np.zeros((n, len(self.data.cols)))  # encoded; columns in data.aux are ignored
        ...
        return self.data.decode(out, rng)    # a DataFrame with exactly data.out_cols

    def posterior_draw(self, rng, mode="draw"):
        ...                                 # one posterior draw, or posterior means when mode == "mean"
        return self
```

Then, in the plug-in's `pyproject.toml`:

```toml
[project.entry-points."relsynth.generators"]
mygen = "mypackage.module:MyGenerator"
```

Once the package is installed (`pip install mypackage`), `generator: mygen` works in any spec, with no change to
relsynth.

### What `fit` receives

`data` is the engine's encoding of the table (`relsynth.table_model.TableModel`):

- `codecs`: one per modelled column, with `encode` and `decode`.
  - Discrete columns are integer codes, with `-1` for null.
  - Continuous columns are floats, with NaN for null. Dates are days and datetimes are seconds.
  - Categoricals that determine each other are joined into one `JointCodec`.
- `X`: the encoded columns, one array per codec.
- `cols`: all modelled columns. `out_cols` are the table's own columns. `aux` are the parent attributes joined onto
  each fact row.
- `n_rows`, `cfg` (the spec's synthesis settings) and `decode(values, rng)`, which turns encoded values into the
  output frame.

`context` (`FitContext`) holds:

- `groups`: the latent group of each row. For entities this is the row's segment. For facts it is the segment block,
  and `-1` marks rows the engine left out.
- `parent`: the coarse group each group backs off to.
- `per_group`, `rng`, `config` and `options` (from the spec).
- `compute`: the resolved device plan (see [compute.md](compute.md)).
- `privacy_budget`: `None` until T07.

### What `sample` receives

`conditioning.groups` holds one group per row to draw. When a fact block was too sparse to fit, its group arrives as
its coarse group. `conditioning.given` is a frame of the parent attributes (`data.aux`); it is given only to
generators that declare `parent` conditioning.

Return exactly `data.out_cols` and `n` rows. The engine checks both.

### Optional methods

- `posterior_draws(m, rng)` iterates over `m` posterior draws.
- `log_prob(frame, conditioning)` returns the log density; set `capabilities.log_prob`.
- `uncertainty()` returns convergence and uncertainty figures for the model summary.
- `privacy_cost()` reports the privacy spent.
- `fitted_groups()` returns the groups with their own parameters.

### Saving a model

`serialize()` writes a model file: a JSON header followed by the pickled state. The header holds the generator,
its model version, the contract version and a SHA-256 of the payload. `Class.load(blob, trusted=True)` reads it back
and refuses a file from another generator, another model version, another contract major version, or a corrupt
file. Pickles can run code when loaded, so only load files you wrote yourself.

### Rules for plug-in authors

1. Draw every random number from the `rng` you are given, or from generators seeded from it. Never use global state
   or wall-clock seeds. On a GPU, use deterministic kernels, or document that results are only reproducible up to
   floating point.
2. Never write files or open connections during `fit` or `sample`. The engine owns all I/O.
3. Bump `version` whenever the model or its saved state changes meaning.
4. Declare only the capabilities you have. The registry checks them before fitting.
5. A plug-in that fails to import is listed as broken (`relsynth generators list`). It does not break relsynth, and a
   spec that names it gets the import error.
6. A plug-in cannot replace a built-in name. A name declared by two packages is refused until one of them is
   uninstalled.

## References

- J. P. Reiter (2005). Using CART to generate partially synthetic public use microdata. *Journal of Official
  Statistics* 21(3).
- B. Nowok, G. M. Raab, C. Dibben (2016). synthpop: Bespoke creation of synthetic data in R. *Journal of
  Statistical Software* 74(11).
- N. Patki, R. Wedge, K. Veeramachaneni (2016). The Synthetic Data Vault. *IEEE DSAA*.
- J. Pang et al. (2024). ClavaDDPM: Multi-relational data synthesis with cluster-guided diffusion models. *NeurIPS*.
  Its parent-context conditioning is the pattern behind `parent` conditioning.
- D. B. Rubin (1981). The Bayesian bootstrap. *Annals of Statistics* 9(1).
- L. A. Goodman (1974). Exploratory latent structure analysis using both identifiable and unidentifiable models.
  *Biometrika* 61(2). D. B. Dunson, C. Xing (2009). Nonparametric Bayes modeling of multivariate categorical data.
  *JASA* 104(487).
- P. D. Hoff (2007). Extending the rank likelihood for semiparametric copula estimation. *Annals of Applied
  Statistics* 1(1).
- L. Breiman, J. Friedman, R. Olshen, C. Stone (1984). *Classification and Regression Trees*. Wadsworth.
- PyPA, *Entry points specification*, packaging.python.org/specifications/entry-points.

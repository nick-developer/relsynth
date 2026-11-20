# Scenarios

A fitted relsynth model reproduces the source database. A **scenario** changes that model while a dataset is drawn,
so the synthetic database answers a what-if question: more customers, a different customer mix, higher prices in the
top decile, a December twice as busy, sales concentrated on fewer products. The fitted parameters are not touched:
the same model gives the baseline and every scenario, and each dataset's sample log records what the scenario did.

Scenarios are declared in the spec, so they are reviewed, versioned and reproduced like the rest of it. Nothing in
the engine is tuned to a dataset; every change is a key of the scenario. The proofs are Propositions 1824–1828 in
`docs/proofs` (section "Simulation: event processes and scenarios").

```yaml
synthesis:
  scenario: holiday_peak            # applied to every dataset; leave out (null) for the fitted model as it is
  scenarios:
    holiday_peak:
      description: December demand doubles, top-decile prices up 5%
      interventions:
        - {type: scale, table: Customer, factor: 1.2}
        - {type: shares, table: Customer, column: Segment, factors: {SMB: 0.7}}
        - {type: marginal, table: Orders, column: Amount, quantiles: [[0.9, 1.0], [1.0, 1.05]]}
        - {type: intensity, table: Orders, column: OrderDate, start: 2024-12-01, end: 2025-01-01, factor: 2.0}
        - {type: degree, table: Sales, key: ProductKey, concentration: 1.5}
    recession:
      interventions:
        - {type: degree, table: Sales, key: OrderKey, rate: 0.8}
```

Choose a scenario per run without editing the spec:

```bash
relsynth run spec.yaml --scenario recession --out out/recession
```

```python
model = relsynth.fit("spec.yaml")
base = model.sample(dataset=0)
peak = model.sample(dataset=0, scenario="holiday_peak")     # same posterior draw, scenario applied
peak.sample_log["scenario"]                                 # every intervention with its effect
```

Interventions apply in order. On a fact split into order headers and lines (`fact.header`), an intervention on a
header column (the order date, the customer key, the sequence) acts on the header, and the lines follow their order. A dataset sampled under a scenario is compared with the real data by the scorecard as
usual, so checks that the scenario is meant to move (a share, a count, a date profile) are expected to fail.

## Interventions

| Type | Keys | What it does |
|---|---|---|
| `scale` | `table`, `factor` | An entity table gets `factor` times its rows, and its child entities follow (as `synthesis.scale`). On a fact, the rows per anchor are multiplied. |
| `shares` | `table`, `column`, `factors` or `shares` | Tilts an entity table's rows by a categorical column: `factors: {level: f}` multiplies each level's weight (other levels 1) and keeps the row count; `shares: {level: s}` sets target shares exactly (levels not named share the rest in proportion to the model). |
| `marginal` | `table`, `column`, `multiply`, `add`, `quantiles` | Changes a number column's values: `x * multiply + add`, after `quantiles: [[level, factor], ...]` multiplies each value by a factor interpolated at its quantile level. A date column takes `add` (days). |
| `intensity` | `table`, `start`, `end`, `factor`, `column` | Multiplies the event rate between `start` (inclusive) and `end` (exclusive). `column` is the time column (default: the table's sequence or series time). |
| `degree` | `table`, `key`, `concentration`, `rate` | Link-degree laws of one foreign key. `concentration` c reshapes how links spread over the parent rows (c > 1: a few rows take most links; c < 1: more even; 0: uniform). `rate` multiplies a fact's rows per anchor. |

Every intervention has an optional `note` (free text, kept in the logs).

## How each change is made

**scale** resizes entities exactly as `synthesis.scale` does. On a fact the rows are drawn as `ceil(factor)`
independent copies of the fitted count process and thinned with probability `factor / copies`: for the engine's
Poisson count laws this is exactly the process with the rate multiplied (superposition and thinning). For negative
binomial or DP count laws the mean is multiplied and the spread changes; an occupancy fact keeps unique pairs, so
repeated pairs are dropped and its rate rises a little less when pairs saturate. The record gives the rows.

**shares** draws rows from the fitted model and accepts each with probability w / max w (rejection sampling). The
accepted rows follow the model tilted by w, which is the change closest to the model (in Kullback–Leibler divergence)
with those shares: every other column keeps its law given the tilted column, and each row keeps its segment, so its
children (a Gov customer's orders) follow it. Target `shares` fill a quota per level with the first rows of that level
drawn from the model. At most 50 rows are drawn per target row; a shortfall is recorded (it only happens for a level
the model almost never draws, which the scenario should then not ask for).

**marginal** maps the sampled values and then puts them back in the original rank order (monotone rearrangement).
Ranks, and so the dependence with every other column (the copula), are kept exactly, and the other columns use the same
random numbers as the baseline, so the baseline and the scenario differ only in this column. Integer columns are
rounded. The map is applied to the finished rows of the table, after the near-copy guard and rule redraws, so no row
escapes it; with the partitioned runtime it is applied once to the joined table, so quantile levels are the whole
table's. An entity's mapped column is in place before its children are drawn, so children that depend on it follow. On
a fact, the `derive` rules that read the mapped column are computed again, and the rows that break a check rule reading
it are counted in the record (`rule_violations`): a scenario may push values past a bound the source respected, and the
record says how often. A column the engine derives once the rows are complete (`ratio_of`, `offset_from`) is mapped on
the finished table the same way. A column that a `derive` rule computes is rejected (change the rule's inputs), and so is
a sequence's time column, which the sequence model draws (use `intensity`).

**intensity** depends on the table:

- a fact: rows are superposed `ceil(max factor)` times and each kept with probability `w(t) / copies`, with `w =
  factor` inside the window and 1 outside. The rows' times come from the model, so this is exactly the event rate
  multiplied on the window, and the total rises or falls with it;
- an entity table with a date column: its rows are tilted by `w(time)` as for `shares`, and the row count is multiplied
  by the mean of `w` (the integrated change of the rate);
- a sequence table (`sequence:`): the sequence model's calendar intensity is multiplied on the window and the event
  count by the integrated change (recorded as `count_factor`); each entity's history keeps its rhythm in operational
  time, so more of its events fall in the window;
- a series table (`series:`): the value columns are multiplied in the window.

**degree** `concentration` raises each parent row's degree weight to the power c within its segment and rescales the
segment to its old total (Prop. 1828). The expected links of each segment do not change; within it, c > 1 moves links
to the rows that already had most (more concentrated in the Lorenz order) and c < 1 spreads them. It acts on the
heterogeneity the fit found among rows of a segment (gamma frailties, or a copied table's own degree posterior); when
the fit found none (`synthesis.facts.frailty` off, or one row per segment), the record says that there was nothing to
temper. The record gives the Gini coefficient of the weights before and after.

## What is recorded

- The model log (`FittedModel.log["scenarios"]`, the diagnostics bundle) lists every scenario with its interventions.
- Each dataset's sample log has `scenario: {name, description, interventions}`, where each intervention carries an
  `effect`: rows before and after, rows drawn, shortfall, realised shares, the scale applied, the count factor of a
  window, the mean of a mapped column, the Gini coefficients of tempered weights.

## Privacy

A scenario reads only the fitted model and the rows it is sampling, never the source. It is post-processing, so a
model fitted in DP mode keeps its guarantee: `relsynth run --dp --scenario NAME` applies the scenario in stage 2
(the ordinary engine fitted to the DP seed). Outside DP mode the privacy checks of the run apply to the scenario's
data as to any other.

## Not covered yet

Shares on fact tables (use `intensity` or `degree`), shares or an entity window on a column the table model does not
draw with the row (a `dimensions:` group's columns, lookups and expressions; the run stops with a spec error),
degree tempering of anchored links (links chosen through another table's key), continuing Hawkes or status-chain
histories with `relsynth extend` (it refuses them), tilting tree (self-referencing) and grain tables, and scenarios
on tables generated by the DP decoder in stage 1. Agent-style lifecycles built on the event processes are planned.

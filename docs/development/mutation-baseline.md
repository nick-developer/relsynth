# Mutation-testing baseline

Measured 2026-10-26 on commit `e770586` (branch roadmap/T01-ci-supply-chain), mutmut 3.8.0, Python 3.13, 4 cores.
Command: `python tools/mutation_sample.py --per-module 150 --seed 1` (about 3 hours). Raw result: [mutation-baseline.json](mutation-baseline.json).

Each module's mutants (lines reached by `tests/unit/core` only) were sampled at random, 150 per module, with seed 1. The score is killed / (killed + survived); the interval is the Wilson 95% interval of that sample proportion. A survivor is either a gap in the tests or an equivalent mutant (one that cannot change behaviour); the sample has not been triaged yet, so the scores are lower bounds on test strength.

| Module | Mutants | Sampled | Killed | Survived | Score | 95% CI |
|---|---|---|---|---|---|---|
| copula.py | 1305 | 150 | 96 | 54 | 0.64 | 0.56–0.71 |
| counts.py | 228 | 150 | 129 | 21 | 0.86 | 0.80–0.91 |
| evaluate.py | 3682 | 150 | 102 | 48 | 0.68 | 0.60–0.75 |
| posterior.py | 200 | 150 | 116 | 34 | 0.77 | 0.70–0.83 |
| relational.py | 2387 | 150 | 88 | 62 | 0.59 | 0.51–0.66 |

## How to use it

- The weekly `mutation.yml` run reports the same table. A module whose score falls below the lower end of its interval here has lost test strength and needs a look.
- There is no blanket target (see CONTRIBUTING.md, coverage policy). Work on the survivors below, function by function, and add a test where a survivor is a real gap.
- After a deliberate improvement, re-measure with the same seed and replace this file in the same pull request.

## Surviving mutants by function

Show one with `mutmut show <name>`.

- **copula**: `GaussianCopula.fit` 38, `GaussianCopula._fit_rank` 8, `chain_converged` 5, `rtruncnorm` 1, `GaussianCopula.sample` 1, `GaussianCopula.sample_conditional` 1
- **counts**: `nb_loglik` 9, `count_model_evidence` 5, `choose_count_model` 3, `dp_logmarginal` 3, `poisson_loglik` 1
- **evaluate**: `scorecard` 12, `column_metrics` 6, `join_metrics` 5, `_canon_counts` 4, `assoc_metrics` 4, `evaluate` 3, `joined_view` 3, `_assoc` 2, `_assoc_fast` 2, `_collapse_rare` 2, `_col_table` 1, `_grand_anchor` 1, `_kish` 1, `determining_columns` 1, `discriminator_auc` 1
- **posterior**: `MeanRNG.gamma` 5, `MeanRNG.standard_gamma` 5, `MeanRNG.beta` 4, `MeanRNG.choice` 4, `MeanRNG._out` 3, `MeanRNG.chisquare` 3, `MeanRNG.normal` 3, `MeanRNG.dirichlet` 2, `RunCache.__init__` 2, `param_rng` 1, `run_streams` 1, `MeanRNG._shape` 1
- **relational**: `sparse_dirichlet_draw` 7, `Entity.attr_marginal_stats` 6, `RelationalModel.merge` 5, `RelationalModel._merge_terms` 4, `eb_beta_rows` 3, `PairRel.draw_params` 3, `PairRel.loglik_rows` 3, `PairRel.marginal_stats` 3, `RelationalModel.__init__` 3, `RelationalModel.fit` 3, `grouped_zoib_draw` 2, `kmeanspp` 2, `RelationalModel.init` 2, `RelationalModel.update_z` 2, `_merge_axis` 1, `dispersion_alpha` 1, `multiplicity_marginal` 1, `slab_rate` 1, `zi_gamma_draw` 1, `Entity.__init__` 1, `Entity.attr_stats` 1, `Entity.draw_attr_params` 1, `OccRel.draw_params` 1, `PairRel.block_stats` 1, `RelationalModel._merge_terms_ab` 1, `RelationalModel._sync` 1, `RelationalModel.draw_params` 1, `RelationalModel.loglik` 1

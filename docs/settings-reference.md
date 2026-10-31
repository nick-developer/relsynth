# Settings reference

Generated from `relsynth.settings` (`python -m relsynth.settings > docs/settings-reference.md`); do not edit by hand. See [docs/settings.md](settings.md) for how settings are set.

| Key | Type | Default | Range / choices | Tunable | Meaning |
|---|---|---|---|---|---|
| `seed` | int | `7` | 0 … 9223372036854775807 |  | Same seed and spec give the same datasets. |
| `posterior.mode` | choice | `draw` | draw, mean |  | draw: one posterior draw of every parameter per dataset (posterior predictive synthesis); mean: posterior means (stable test data). |
| `posterior.datasets` | int | `1` | 1 … 10000 |  | Number of independent synthetic datasets (each its own posterior draw). |
| `posterior.sample_seed` | int | `None` | 0 … 9223372036854775807 |  | Seed of the parameter and data streams (default: seed). |
| `scale.default` | float | `1.0` | 0 … 1e+06 |  | Row multiplier for entities; facts follow their anchors. |
| `scale.tables` | map | `{}` |  |  | Per-table multipliers {table: factor}. An entity without its own factor takes the factor of the first parent it links to that has one (orders follow their customers); facts follow their anchors. |
| `segments.default_cap` | int | `8` | 1 … 512 | yes (log) | Upper bound on latent segments per table. |
| `segments.min_rows_per_segment` | int | `20` | 1 … 1e+07 | yes (log) | Rows per segment at least (also caps the segment count). |
| `segments.tables` | map | `{}` |  |  | Per-table segment caps {table: cap}. |
| `segments.stratify` | choice | `auto` | auto, none, map |  | auto | none | {table: [key, ...]}: segments nested in the rows of small copied parents. |
| `segments.stratify_max_levels` | int | `20` | 0 … 1e+06 |  | Largest code list used as strata without a Bayes-factor test. |
| `segments.stratify_max_cells` | float | `50000000.0` | 0 … 1e+12 |  | Budget of rows x segments per sweep for strata. |
| `segments.attribute_strata` | choice | `auto` | auto, none, map |  | auto | none | {table: column}: segments nested in an own column (only sales staff have quotas). |
| `segments.tie_headers` | bool | `True` |  |  | Order headers take the segment of their customer. |
| `segments.copied_budget_cells` | float | `20000000.0` | 0 … 1e+12 |  | Budget for one segment per row of copied tables. |
| `gibbs.iterations` | int | `120` | 1 … 1e+06 | yes (log) | Sweeps of the relational segment sampler. |
| `gibbs.final_iterations` | int | `30` | 0 … 1e+06 |  | Sweeps after merging segments. |
| `gibbs.max_final_iterations` | int | `120` | 0 … 1e+07 |  | Upper bound when extending the final sweeps until the split R-hat of their second half <= rhat_target. |
| `gibbs.rhat_target` | float | `1.1` | 1 … 10 |  | Convergence target of the final sweeps (split R-hat of the log-likelihood trace, second half; the first half is burn-in after the merge). |
| `gibbs.chains` | int | `1` | 1 … 1 |  | Reserved (one chain). |
| `gibbs.merge` | bool | `True` |  |  | Merge segments the data does not support. |
| `copula.mode` | choice | `rank` | rank, bayes, fast | yes | rank | bayes | fast: sampler of the correlation matrix. |
| `copula.sample_rows` | int | `1500` | 10 … 1e+08 | yes (log) | Rows used to fit a copula. |
| `copula.iterations` | int | `120` | 1 … 1e+07 |  | Sweeps of the copula sampler. |
| `copula.max_iterations` | int | `3000` | 0 … 1e+08 |  | Upper bound when extending the chain until it converges (every correlation: split R-hat <= rhat_target, or Monte Carlo standard error <= 0.01 on the Fisher-z scale). |
| `copula.rhat_target` | float | `1.1` | 1 … 10 |  | Convergence target (split R-hat); a correlation whose standard error is already below 0.01 on the Fisher-z scale passes too. |
| `copula.burn_in` | int | `60` | 0 … 1e+07 |  | Sweeps discarded. |
| `copula.lkj_eta` | float | `2.0` | 1e-06 … 1e+06 | yes (log) | LKJ prior concentration on the correlation. |
| `copula.grid` | int | `64` | 4 … 100000 | yes (log) | Grid of the rank likelihood. |
| `copula.min_rows` | int | `80` | 0 … 1e+09 | yes (log) | Fewer rows: independence copula. |
| `copula.pooled_cells` | float | `1000000.0` | 0 … 1e+12 | yes (log) | Budget of rows x columns^2 latent updates per sweep for the pooled copula (the one fact tables use); it is fitted on max(2 * sample_rows, pooled_cells / p^2) rows, because a dataset drawn from it shows the posterior spread, about 1/sqrt(rows fitted), in every correlation. |
| `copula.mh_steps` | int | `15` | 1 … 10000 |  | Metropolis steps per sweep of the correlation sampler (copula.mode bayes). |
| `copula.mcse_z` | float | `0.01` | 1e-06 … 1 |  | Fixed-width rule of the copula chain: a correlation whose Monte Carlo standard error on the Fisher-z scale is at most this passes the convergence test (Jones et al. 2006). |
| `marginals.bayesian_bootstrap` | bool | `True` |  |  | Draw continuous marginals by the Bayesian bootstrap. |
| `marginals.dirichlet_prior` | float | `0.25` | 0 … 1e+06 | yes (log) | Shrinkage of a segment's categorical law to the table's. |
| `marginals.max_levels` | int | `200` | 1 … 1e+07 | yes (log) | Levels kept per categorical (rarer ones pooled). |
| `marginals.tail_trim` | float | `0.0` | 0 … 0.5 | yes | Share trimmed from continuous tails. |
| `marginals.min_dependence_rows` | int | `5` | 0 … 1e+07 | yes | Rows away from the mode needed to model a column's dependence, in the whole table and in each segment's copula sample. |
| `marginals.fd_bf` | float | `100.0` | 1 … 1e+300 |  | Bayes factor to treat a categorical dependency as functional. |
| `marginals.joint_max_levels` | int | `64` | 1 … 1e+07 | yes (log) | Largest joint of dependent categoricals. |
| `marginals.joint_min_cramers_v` | float | `0.3` | 0 … 1 | yes | Association that joins two categoricals. |
| `marginals.discrete_int_max` | int | `30` | 0 … 1e+07 |  | Integer columns with at most this many distinct values are modelled as discrete. |
| `marginals.max_obs` | int | `20000` | 10 … 1e+09 | yes (log) | Observations kept per continuous marginal (a seeded subsample above it). |
| `marginals.fd_sample_rows` | int | `50000` | 100 … 1e+10 |  | Rows of the seeded subsample on which functional dependencies are tested. |
| `facts.min_group_rows` | int | `30` | 0 … 1e+09 | yes (log) | Rows below which a block borrows the coarser dimension's marginals. |
| `facts.frailty` | choice | `auto` | auto, on, off | yes | auto | on | off: heavy-tailed rows per parent (gamma frailty). |
| `facts.frailty_threshold` | float | `1.5` | 1 … 1e+06 | yes (log) | Dispersion index above which frailty is used. |
| `facts.pairs` | choice | `auto` | auto, poisson, occupancy |  | Default of fact.pairs. |
| `facts.multiplicity` | choice | `auto` | auto, none, partner, anchor, block |  | Default of fact.multiplicity. |
| `facts.count_model` | choice | `auto` | auto, poisson, dp | yes | Default of fact.count_model. |
| `facts.count_model_bf` | float | `100.0` | 1 … 1e+300 | yes (log) | Bayes factor for the Dirichlet-process count law. |
| `facts.group_fallback` | choice | `auto` | auto, anchor, partner |  | Default of fact.group_fallback. |
| `facts.link_degree` | choice | `auto` | auto, segment, dp | yes | Default of fact.link_degree. |
| `facts.link_degree_min_rows` | float | `50.0` | 0 … 1e+09 | yes (log) | auto gives a key a degree law only when each segment's links are spread over at least this many rows (Kish effective number, link-weighted); below it a few rows carry the links and attributes joined through the key would follow them. |
| `facts.frailty_min_rows` | int | `30` | 2 … 1e+09 |  | Rows of a segment needed to estimate its dispersion (frailty). |
| `facts.occupancy_z` | float | `-4.0` | -1000 … 0 |  | facts.pairs auto: a pair fact is an occupancy (each pair present or absent) when the z score of its pair repeats is below this ... |
| `facts.occupancy_min_expected` | float | `10.0` | 0 … 1e+09 |  | ... and at least this many repeats are expected under Poisson. |
| `facts.dp_count_sweeps` | int | `30` | 1 … 100000 |  | Gibbs sweeps of the Dirichlet-process count law's conditional draw. |
| `structure.structural_prior` | float | `0.9` | 0 … 1 | yes | Prior probability that an unseen combination is forbidden. |
| `structure.structural_pooling` | float | `3.0` | 0 … 1e+06 | yes | Prior concentration of the structural weights learned from a relation's group-columns (zero / one / free exchangeable across them, w ~ Dirichlet(kappa * (pi/2, pi/2, 1 - pi)) centred on structural_prior); blocks inside a free group-column keep the fixed weights. Learned weights keep fully complete relations complete. 0 keeps the fixed weights of structural_prior (v0.8). |
| `structure.beta_prior` | float | `0.5` | 1e-06 … 1e+06 | yes (log) | Beta prior of link and occupancy rates. |
| `structure.conditional_inclusion` | choice | `auto` | auto, none | yes | auto | none: links never point at excluded parent levels. |
| `structure.inclusion_min_expected` | float | `10.0` | 0 … 1e+09 | yes (log) | Expected references needed to call a level excluded (Bayes factor e^n). |
| `structure.participation` | choice | `auto` | auto, none | yes | auto | none: hurdle on the share of parents without children. |
| `uniqueness.mh_sweeps` | int | `20` | 0 … 1e+06 | yes (log) | Sweeps of the unique-key sampler. |
| `uniqueness.grain` | choice | `auto` | auto, none |  | auto | none: unique keys of two foreign keys become grains. |
| `uniqueness.rake_rounds` | int | `3` | 0 … 10000 | yes | Rounds of raking for rule cells. |
| `trees.max_depth` | int | `auto` | 1 … 1e+06 or auto |  | auto | int: deepest hierarchy level generated. |
| `trees.frailty` | choice | `auto` | auto, on, off | yes | auto | on | off: branching heterogeneity in trees. |
| `text.order` | int | `3` | 1 … 12 | yes | n-gram order of text columns. |
| `text.min_token_rows` | int | `3` | 1 … 1e+06 | yes | Rows a token must appear in. |
| `text.max_copy_ngram` | int | `8` | 1 … 1000 | yes | Longest rare source n-gram allowed in output. |
| `text.backoff_strength` | float | `2.0` | 0 … 1e+06 |  | Back-off weight of the n-gram model. |
| `text.max_retries` | int | `30` | 0 … 1e+06 |  | Retries before a text value is replaced. |
| `rules.max_rejection_rounds` | int | `25` | 0 … 1e+06 | yes (log) | Redraw rounds for rows that fail a check rule. |
| `rules.rake_max_levels` | int | `50` | 1 … 1e+07 |  | Rule cells: a column with more levels is not raked. |
| `rules.rake_max_cells` | int | `2000` | 1 … 1e+09 |  | Rule cells: largest joint table raked. |
| `repair.pair_tries` | int | `200` | 0 … 1e+07 | yes (log) | Degree-preserving swaps tried per repeated pair or self-link before the pair is dropped. |
| `repair.constraint_rounds` | int | `200` | 0 … 1e+07 | yes (log) | Redraw rounds of entity rows that break a check rule or a unique key. |
| `repair.thin_rounds` | int | `200` | 0 … 1e+07 |  | Rounds of anchor thinning (participation of facts). |
| `repair.tree_redraws` | int | `40` | 0 … 1e+06 |  | Redraws of a hierarchy that breaks a depth or cycle rule. |
| `repair.unique_code_rounds` | int | `50` | 1 … 1e+06 |  | Redraw rounds of identifier codes until they are distinct. |
| `repair.participation_rounds` | int | `4` | 0 … 10000 |  | Rounds that move rows away from parents the participation hurdle leaves empty. |
| `repair.occupancy_rounds` | int | `20` | 0 … 10000 |  | Redraws that give every parent its mandatory first row in occupancy facts. |
| `detection.group_effect_share` | float | `0.95` | 0 … 1 |  | A group effect holding on fewer source groups than this share is reported. |
| `detection.scd_stable_share` | float | `0.99` | 0 … 1 |  | SCD2 stable auto: columns equal across versions on at least this share of keys. |
| `detection.scd_max_versions` | int | `50` | 1 … 1e+06 |  | Largest number of versions per key modelled exactly (more are pooled). |
| `evaluation.thresholds` | map | `{}` |  |  | Fixed tolerances of the scorecard by name (ks, tvd, auc, ...; see the table below). Each limit is the larger of its tolerance and its sampling limit. |
| `evaluation.assoc_max_rows` | int | `20000` | 100 … 1e+09 |  | Rows sampled for the association metrics. |
| `evaluation.discriminator_rows` | int | `5000` | 100 … 1e+09 |  | Rows sampled per side for the discriminator AUC. |
| `per_table` | map | `{}` |  |  | Per-table settings {table: {copula: {...}, marginals: {...}, text: {...}}}. |
| `settings_file` | str | `None` |  |  | A settings file merged under this block (a path relative to the spec, or preset:fast / preset:thorough). The spec's own keys win. See docs/settings.md. |
| `privacy.suppress_small_counts` | int | `5` | 0 … 1e+06 |  | Counts below this are suppressed in the diagnostics bundle. |
| `privacy.dp` | map | `None` |  |  | Differential privacy mode: unit, epsilon or rho, delta, public_tables, public, max_degree, spec_is_public and expert keys; validated by relsynth.dp (docs/dp.md). Off when absent. |
| `pushdown.mode` | choice | `auto` | auto, on, off |  | auto | on | off: fit eligible fact tables from exact statistics computed where the data lives (row counts, rows per key and key pair, distinct keys, rank-score sums) and a uniform sample for their attribute models, instead of loading them. auto: facts with at least min_rows rows; on: every eligible fact; off: load every table. |
| `pushdown.min_rows` | int | `1000000` | 0 … 1e+15 |  | Rows from which auto mode pushes a fact table down. |
| `pushdown.sample_rows` | int | `500000` | 1 … 1e+12 | yes (log) | Rows of a pushed-down fact sampled (seeded, uniform) for its attribute models; the relational model always uses every row through the statistics. |
| `generator` | str | `copula` |  |  | Default generator plug-in of every table's attribute columns (a table's own `generator` wins): copula (Bayesian Gaussian copula, relsynth 1.0), independent, cart, or an installed plug-in. |
| `compute.preset` | str | `cpu` |  |  | cpu | gpu-auto | gpu-single | gpu-multi, or a preset of compute.file: the devices generator plug-ins may use. The default engine is CPU-only; see docs/compute.md. |
| `compute.file` | str | `None` |  |  | A compute.yaml with more presets (relative to the spec). |
| `compute.overrides` | map | `{}` |  |  | Expert settings that replace the preset's fields: device, gpus, devices, backend, precision, batch_size, memory_limit, threads, fallback. |

## `evaluation.thresholds`

Fixed tolerances of the scorecard. Each item's limit is the larger of its tolerance and its sampling limit, so lowering a tolerance never makes a check stricter than its noise allows.

| Key | Default |
|---|---|
| `ks` | `0.08` |
| `tvd` | `0.08` |
| `null_diff` | `0.03` |
| `assoc_max` | `0.15` |
| `auc` | `0.65` |
| `card_ks` | `0.12` |
| `card_mean_rel` | `0.1` |
| `join_tvd` | `0.1` |
| `fk_integrity` | `1.0` |
| `tree_depth_tvd` | `0.12` |
| `fanout_ks` | `0.12` |
| `time_tvd` | `0.1` |
| `tstr_ratio` | `0.85` |
| `text_rare_leaks` | `0` |
| `dcr_ratio` | `0.5` |
| `exact_match` | `0.01` |
| `rule_violation` | `0.0` |
| `pair_repeat` | `0.01` |
| `novel_combination` | `0.005` |
| `rows_removed` | `0.01` |

# Small dimensions: privacy margin of DimProduct (gap plan G3)

Gap plan G3 (release 2.2.0). The AdventureWorks autotune runs ([autotune-adventureworks.md](autotune-adventureworks.md)) found the distance-to-closest-record ratio (`dcr_ratio`) of `DimProduct` close to its 0.5 limit at the 2.1 defaults (0.40 on one seed in five). This page records the study of the remedies, the choice, and the validation. Propositions 1720-1723 are in `docs/proofs/sec_g3_smalldims.tex`, with checks C1720-C1723 in `tests/science/test_small_dims_checks.py`. The user guide is [docs/privacy.md](../privacy.md#small-dimensions).

## Cause

`DimProduct` has 606 rows. Its strata give 24 latent segments of 6 to 49 rows. Every segment is below `copula.min_rows` (80), so every segment already uses the table's pooled copula, and only the marginals are the segment's own. With the Dirichlet weight 0.25 (`marginals.dirichlet_prior`), a segment marginal is almost the empirical law of its few rows. A third of the real rows are exact duplicates (components with every optional attribute null), so the check compares the positive parts of the distance distributions. Synthetic rows that match a real product's categories and land between two of its family's prices sit closer to that product than any other real product does.

## Options studied

All numbers are `DimProduct` `dcr_ratio` (limit 0.5) and the AdventureWorks scorecard (1,257 items in this evaluation, without the privacy items). They come from two fitted models (seeds 1 and 12) and two sampling streams each. Every variant re-samples the same fitted model with the same seeds, so the comparisons are paired.

| Variant | DCR (4 paired draws) | Scorecard change vs defaults (paired) |
|---|---|---|
| 2.1 defaults | 0.473, 0.567, 0.421, 0.575 | — |
| segment cap 4 for DimProduct (`segments.tables`) | 0.498 (one full fit) | −3 |
| copula pooled toward the table | no effect: no segment of DimProduct has its own copula | — |
| marginal shrinkage, 10 rows | 0.902, 1.079, 0.932, 0.808 | −6, −3, +1, +2 |
| **marginal shrinkage, 20 rows** | **1.026, 0.985, 1.037, 0.943** | −2, −10, −1, +1 |
| near-copy guard alone (ratio 1) | 1.105, 0.518, 0.774, 0.529 | −6, −1, −3, +3 |
| shrinkage 10 rows + guard | 1.135, 1.137 | −3, −4 |

- **Segment cap.** The cap applies per stratum, and the strata make the segments, so a lower cap barely changes them. It did not help, and it cost joins.
- **Copula pooling.** Small segments already share the pooled copula, so pooling has nothing to act on here. It was dropped from the change.
- **Marginal shrinkage** (`marginals.segment_prior_rows`) raises the weight of the table marginal in each segment's Dirichlet / Dirichlet-process posterior to that many rows. It is the hierarchical prior relsynth already has, made stronger for small segments. The posterior predictive of a segment of thousands of rows moves by at most `rows / n` in total variation (Prop. 1721; the bound is for the predictive mixture, not for each posterior draw or for the fitted marginal on its grid). The variates drawn do not change with `rows`, but the generator's state after a Beta draw can, so only `rows: 0` reproduces the 2.1 stream (Prop. 1720). It applies to every draw, in that each one is drawn from the shrunk posterior.
- **Near-copy guard** (`privacy.near_copy`). It redraws rows that sit closer to a real row than that row's nearest distinct neighbour. Alone it is unreliable (0.52 on two draws), because the evaluation's metric ranks real and synthetic rows together and looks only at positive distances. It also rewrote rows in six AdventureWorks tables (24% of `ProspectiveBuyer`, 5% of `DimCustomer`), and on Chinook it broke `Track` associations on 4 of 5 seeds. Redrawing until it succeeds, it is not differentially private for any ε (Prop. 1723); with finite rounds it gives no DP guarantee of its own, and in DP mode it is post-processing of the seed database. It makes the DCR item of a guarded table circular. It ships as an opt-in safety net, off by default.

**Choice:** marginal shrinkage with 20 rows was the first choice for the default, on entity tables only. The full validation below and the science suite showed that it weakens the dimension-fact links (see "Science checks"), so release 2.2 uses **shape-mode shrinkage with 40 rows** by default (see "Shape mode"), approved by Cod3r on 2026-11-08. The guard is opt-in.

**Cost.** Over the four paired draws the AdventureWorks scorecard drops by 3 items on average out of 1,257. The draw-to-draw spread at the defaults is about ±5. The items that fail more often are systematic: `FactInternetSales` joined product prices (`ListPrice`, `DealerPrice`, `StandardCost` `join_ks`, 3 of 4 draws), `StartDate` `join_ks`, and the `FactInternetSales` discriminator. Internet sales are concentrated on a few products. When those products' prices are shrunk toward the catalogue, the price distribution weighted by sales moves. This is the price of not reproducing small product families. A deployment that needs these joins exactly can set `marginals.segment_prior_rows: 0` (the 2.1 behaviour), or set it per table under `per_table`.

## Other golden databases

| Database | Defaults 2.1 | Table-mode shrinkage 20 | Guard on (with shrinkage) |
|---|---|---|---|
| Chinook, 5 seeds, scorecard | 142/142 on every seed | 142/142 on every seed | 142/142 on seed 1, 141/142 on seeds 2-5 (`Track` association) |
| Chinook `Track` DCR | 1.63-1.79 | 1.83-2.10 | 2.83-3.35 |

Contoso and Berka run in the CI bench job of the pull request.

## Validation of table-mode shrinkage 20 (AdventureWorks, 5 seeds)

Five full fits per setting (seeds 1, 2, 3, 11, 12), each fitted and sampled once with the same seed, scored on the 1,267-item AdventureWorks scorecard. "2.1" is `marginals.segment_prior_rows: 0`.

| Seed | DimProduct DCR, 2.1 | DimProduct DCR, shrinkage 20 | Scorecard, 2.1 | Scorecard, shrinkage 20 | Change |
|---|---|---|---|---|---|
| 1 | 0.501 | 0.969 | 1215 | 1202 | −13 |
| 2 | 0.648 | 0.963 | 1202 | 1202 | 0 |
| 3 | 0.576 | 1.010 | 1214 | 1201 | −13 |
| 11 | 0.549 | 0.963 | 1207 | 1198 | −9 |
| 12 | 0.581 | 1.101 | 1202 | 1198 | −4 |
| Mean | 0.571 | 1.001 | 1208.0 | 1200.2 | −7.8 |

The DCR ratio of `DimProduct` is at least 0.96 on every seed (limit 0.5), against 0.50-0.65 at the 2.1 defaults. The scorecard cost on full fits is larger than on the paired re-samples above: 7.8 items on average (0.6% of 1,267), at most 13. The other dimension tables keep DCR ratios of at least 0.97 (`DimReseller`, seed 12).

With shrinkage on, `tools/identity_check.py` reports every scenario as different, as expected. With `marginals.segment_prior_rows: 0` it reproduces 2.1 byte for byte on all 24 runs.

## Science checks

With shrinkage 20 on by default, three fake-data checks of the science suite fail, and pass again with 0 (same code, local run):

| Check | With 0 (2.1) | With 20 |
|---|---|---|
| C09 snowflake: Spearman(sale amount, product price) across 3 tables, real 0.909 | 0.910 ± 0.002 | 0.799 ± 0.026 (z = 4.2) |
| C19 doctor: planted problems left after tuning | none | `D02 customer` |
| C43 sales warehouse: bike share in segment A / C orders, target 0.60 / 0.03 ± 0.08 | pass | 0.51 / 0.04 |

The Berka bench also failed one calibrated item (`order` association, 0.155 against a limit of 0.15). The cause is the same as the AdventureWorks price joins: the segments of a dimension carry what the facts depend on (which products sell, at what price), and shrinking each segment toward the table weakens exactly that link. Shape mode, below, keeps each segment's location and shrinks only its shape.

## Shape mode (the default mode)

`marginals.segment_prior_mode: shape` shrinks only continuous columns, and its prior is the parent distribution (the table's, or a coarse group's when the segment has one) moved to the segment's median and scaled to its interquartile range (at least `a / (4 (n + a))` of the parent's), clipped to the parent's range (Prop. 1724). Categorical columns keep the weight `dirichlet_prior`. When the segment's median lies inside the parent's range (the table's or a coarse group's), it does not move in the posterior mean, up to the grid interpolation (posterior draws vary around it, as without shrinkage; a median beyond the range is pulled toward it), so the fact tables' view of a segment (which products sell, at what price) stays, and only the exact values of its few rows are smoothed.

The first version did not clip to the table's range and failed C48: shifted dates left the range of a keyed lookup (an exchange rate per order date, 0.990 of rows matched instead of 1). Clipping fixed it.

**Science checks** with shape mode: C09, C19, C43 and C48 pass at 20 and at 40 rows. The full suite passes apart from C75 `t_unique_moves_fast` and C76. C76 needs the layer 2 data and fails the same way at 2.1 in this container (it is skipped in CI). C75 found one duplicate of a link key after the moves; the same check failed on 4 of 8 synthesis seeds at 2.1 (0, 0, 1, 2, 0, 1, 0, 1 duplicates for seeds 1-8), an existing engine issue: the repair of unique keys over links could run out of proposals when nearly every parent needs exactly one child. This pull request fixes it (rows the link model cannot place get a uniformly drawn allowed parent), and C75 passes.

**AdventureWorks**, the same five seeds as above (DimProduct DCR / scorecard of 1,267):

| Seed | 2.1 | Table 20 | Shape 20 | Shape 40 | Shape 20, 2.2 code |
|---|---|---|---|---|---|
| 1 | 0.501 / 1215 | 0.969 / 1202 | 0.822 / 1216 | 0.888 / 1212 | 1.124 / 1209 |
| 2 | 0.648 / 1202 | 0.963 / 1202 | 0.853 / 1201 | 1.101 / 1200 | 0.883 / 1203 |
| 3 | 0.576 / 1214 | 1.010 / 1201 | 1.000 / 1211 | 1.035 / 1201 | 1.158 / 1204 |
| 11 | 0.549 / 1207 | 0.963 / 1198 | 0.737 / 1199 | 0.977 / 1201 | 0.820 / 1208 |
| 12 | 0.581 / 1202 | 1.101 / 1198 | 0.586 / 1203 | 1.274 / 1196 | 0.586 / 1206 |
| Mean | 0.571 / 1208.0 | 1.001 / 1200.2 | 0.800 / 1206.0 | 1.055 / 1202.0 | 0.914 / 1206.0 |

Shape 40 (fact tables not shrunk) keeps `DimProduct` at 0.88 or more on every seed, for 6 scorecard items on average (0.5%). Shape 20 costs 2 items but comes within 0.09 of the limit on seed 12. The other dimension tables stay at 1.0 or more with shape 40. The last column is the 2.2.0 default (shape 20) refitted after the C75 fix and with fact tables never shrunk; it is the code that ships.

## Contoso and the default

The CI bench of the pull request ran Chinook, Contoso (Parquet) and Berka with shape 40 as the default. Chinook and Berka did not regress. Contoso did: 18 quality items failed that pass at 2.1, all of them prices of the fact tables (`sales` and `orderrows` UnitPrice, NetPrice and UnitCost), and the fidelity pass rate fell from 0.93 to 0.885. Local runs, Wasserstein distance on normal scores for `sales.UnitPrice` (limit 0.05) and the fidelity pass rate:

| Contoso | Seed 1 | Seed 2 |
|---|---|---|
| 0 rows (2.1) | 0.025 / 0.933 | 0.043 / 0.915 |
| Shape 20 | 0.092 / 0.907 | |
| Shape 40 | 0.107 / 0.885 | 0.099 / 0.877 |
| Shape 40, columns that fact tables read left out | 0.095 / 0.907 | |

The fact tables copy the product's price (`UnitPrice = ProductKey.Price` times a factor), but leaving `Price` and `Cost` out of the shrinkage does not fix it: the joined `Brand` and `Manufacturer`, which are never shrunk, move too (TVD 0.021 to 0.033-0.051), so the facts buy a different mix of products. That variant also lowered the AdventureWorks `DimProduct` ratio (0.74 on seed 2) and was dropped.

**Default (2.2.0), the owner's choice:** shape mode at 20 rows, with the bench baseline rebuilt to accept the Contoso cost. With the current code (fact tables not shrunk, C75 fix), 20 rows give AdventureWorks `DimProduct` ratios of 1.124, 0.883, 1.158, 0.820 and 0.586 on seeds 1, 2, 3, 11 and 12 (mean 0.91; 2.1: 0.50-0.65, mean 0.57), with scorecards of 1209, 1203, 1204, 1208 and 1206 (mean 1206.0, 2.0 items below 2.1's 1208.0). Seed 12 stays at its 2.1 level (0.586 against 0.581), so 20 rows lift the margin on most seeds but do not guarantee it; 40 rows gave 0.89 or more on every seed for 6 items. `segment_prior_rows: 0` reproduces 2.1 byte for byte.

## Open items

- Why shrinking a dimension's numbers shifts which products the Contoso facts buy is not explained yet (a candidate: the facts' link or attribute models read the product's segment-level values). A fix would remove the Contoso cost.
- The autotune rerun the gap plan asks for (`segments.default_cap`, the doctor's parts, `facts.*`) needs the CI autotune workflow (4.5 hours). It is best started after this pull request merges. `marginals.segment_prior_rows` and `marginals.segment_prior_mode` are registered as tunable.
- `dcr_ratio` remains a guard, not a privacy test (T06). The privacy attacks (`relsynth privacy`) on AdventureWorks have not been rerun for this change.

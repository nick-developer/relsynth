# Small dimensions: privacy margin of DimProduct (gap plan G3)

Gap plan G3 (release 2.2.0). The AdventureWorks autotune runs ([autotune-adventureworks.md](autotune-adventureworks.md)) found the distance-to-closest-record ratio (`dcr_ratio`) of `DimProduct` close to its 0.5 limit at the 2.1 defaults (0.40 on one seed in five). This page records the study of the remedies, the choice, and the validation. Propositions 1720-1723 are in `docs/proofs/sec_g3_smalldims.tex`, with checks C1720-C1723 in `tests/science/test_small_dims_checks.py`. The user guide is [docs/privacy.md](../privacy.md#small-dimensions).

## Cause

`DimProduct` has 606 rows. Its strata give 24 latent segments of 6 to 49 rows. Every segment is below `copula.min_rows` (80), so every segment already uses the table's pooled copula, and only the marginals are the segment's own. With the Dirichlet weight 0.25 (`marginals.dirichlet_prior`), a segment marginal is almost the empirical law of its few rows. A third of the real rows are exact duplicates (components with every optional attribute null), so the check compares the positive parts of the distance distributions. Synthetic rows that match a real product's categories and land between two of its family's prices sit closer to that product than any other real product does.

## Options studied

All numbers are `DimProduct` `dcr_ratio` (limit 0.5) and the AdventureWorks scorecard (1,257 items in this evaluation, without the privacy items). They come from two fitted models (seeds 1 and 12) and two sampling streams each. Every variant re-samples the same fitted model with the same random streams, so the comparisons are paired.

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
- **Marginal shrinkage** (`marginals.segment_prior_rows`) raises the weight of the table marginal in each segment's Dirichlet / Dirichlet-process posterior to that many rows. It is the hierarchical prior relsynth already has, made stronger for small segments. Segments of thousands of rows move by at most `rows / n` in total variation (Prop. 1721), and the random stream is unchanged (Prop. 1720). It works on every draw.
- **Near-copy guard** (`privacy.near_copy`). It redraws rows that sit closer to a real row than that row's nearest distinct neighbour. Alone it is unreliable (0.52 on two draws), because the evaluation's metric ranks real and synthetic rows together and looks only at positive distances. It also rewrote rows in six AdventureWorks tables (24% of `ProspectiveBuyer`, 5% of `DimCustomer`), and on Chinook it broke `Track` associations on 4 of 5 seeds. Redrawing until it succeeds, it is not differentially private for any ε (Prop. 1723); with finite rounds it gives no DP guarantee of its own, and in DP mode it is post-processing of the seed database. It makes the DCR item of a guarded table circular. It ships as an opt-in safety net, off by default.

**Choice:** marginal shrinkage with 20 rows by default, on entity tables only. The guard is opt-in.

**Cost.** Over the four paired draws the AdventureWorks scorecard drops by 3 items on average out of 1,257. The draw-to-draw spread at the defaults is about ±5. The items that fail more often are systematic: `FactInternetSales` joined product prices (`ListPrice`, `DealerPrice`, `StandardCost` `join_ks`, 3 of 4 draws), `StartDate` `join_ks`, and the `FactInternetSales` discriminator. Internet sales are concentrated on a few products. When those products' prices are shrunk toward the catalogue, the price distribution weighted by sales moves. This is the price of not reproducing small product families. A deployment that needs these joins exactly can set `marginals.segment_prior_rows: 0` (the 2.1 behaviour), or set it per table under `per_table`.

## Other golden databases

| Database | Defaults 2.1 | Shrinkage 20 | Guard on (with shrinkage) |
|---|---|---|---|
| Chinook, 5 seeds, scorecard | 142/142 on every seed | 142/142 on every seed | 142/142 on seed 1, 141/142 on seeds 2-5 (`Track` association) |
| Chinook `Track` DCR | 1.63-1.79 | 1.83-2.10 | 2.83-3.35 |

Contoso and Berka run in the CI bench job of the pull request.

## Validation at the new defaults (AdventureWorks, 5 seeds)

VALIDATION_TABLE

## Open items and decisions for Cod3r

- **Default change (decision).** Shrinkage on by default changes the output of every database with small entity segments, so `tools/identity_check.py` fails by design. The alternative is to ship both settings opt-in and keep 2.1's output byte-identical.
- The autotune rerun the gap plan asks for (`segments.default_cap`, the doctor's parts, `facts.*`) needs the CI autotune workflow (4.5 hours). It is best started after this pull request merges, so that it tunes against the new default. `marginals.segment_prior_rows` is registered as tunable.
- `dcr_ratio` remains a guard, not a privacy test (T06). The privacy attacks (`relsynth privacy`) on AdventureWorks have not been rerun for this change.

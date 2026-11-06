# `relsynth autotune` on AdventureWorks (2026-11-01)

The first run of the self-tuning loop ([autotune.md](../autotune.md)) on the AdventureWorks DW bench database
(27 tables, 1.05M rows). The goal was to show which systematic failures the engine settings fix, which need an engine or
spec change, and what in autotune itself needed fixing.

## Setup

- Spec: `bench/specs/adventureworks.yaml` (the unedited introspect draft), as given.
- Objectives (`tools/autotune_bench.py`): every hard constraint (`referential_integrity`, `rules`, `unique_keys`,
  `trees`, `privacy`) plus `domain`, the AdventureWorks identities G1-G14 of `bench/checks/adventureworks.py`, and every
  statistical family of the scorecard, calibrated (a family is met when its failures per seed are at most the 0.99
  quantile of its null count). The search used seeds 1, 2 and 3, the held-out seeds were 11 and 12, and doctor spec
  fixes were allowed.
- Where it ran: the GitHub workflow `autotune.yml`, two 5-hour jobs (runs 41244918993 and 41265066430; the second
  resumed from the first one's history). Each candidate ran its three seeds in parallel (`--workers 3`). That made
  64 runs at 20 to 70 minutes each, about 22 runner-hours.

## Result

| | Hard violations | Excess statistical failures per seed | Failures per seed |
|---|---|---|---|
| spec as given, search seeds 1-3 | 0 | 43.7 | 55.3 |
| tuned (`segments.default_cap: 16`), search seeds | 0 | 42.0 | 53.3 |
| spec as given, held-out seeds 11-12 | 0 | 48.0 | — |
| tuned, held-out seeds 11-12 | 0 | 43.0 | 54.5 |

- Every hard constraint and every G1-G14 identity holds on all five seeds (0 of 5: rate below 0.39 at 95%).
- The only setting accepted was `segments.default_cap: 8 → 16`. The held-out seeds confirm it, but the gain is small:
  1.7 fewer excess failures per seed on the search seeds and 5 fewer on the held-out seeds.
- The statistical families stay unmet, and settings will not close them. 22 of the 43 excess failures per seed are
  one family, `novel_combination_rate`, and no setting moved it.

## What settings fixed, and what needs an engine or spec change

| Family (failures per seed) | Systematic checks (values; limit) | Settings tried | Verdict |
|---|---|---|---|
| `novel_combination_rate` (22 of 35) | FactResellerSales (EmployeeKey.SalesTerritoryKey, ResellerKey.GeographyKey) 0.87 (limit 0.005); (OrderDateKey, ResellerKey.GeographyKey) 0.77; FactInternetSales (·, ProductKey.ProductSubcategoryKey) 0.47-0.69; FactFinance (·, OrganizationKey.CurrencyKey) 0.07-0.76 | structure.* (prior, pooling, beta prior, inclusion), marginals.joint_*, dirichlet prior, segments | **Engine.** Synthesised dimension rows (reseller, employee, product, organization) draw their own copied-ancestor keys apart from the fact rows that link to them. Pairs two hops away then fall outside the real support. This is the open v0.5 gap "cross-parent consistency". A spec `link_affinity` on the fact's territory and the employee's territory left it at 0.87. |
| `pair_repeat_share_diff` (1-2) | FactResellerSales (ResellerKey, ProductKey) 0.53 (limit 0.011) | segments, structure.* | **Engine.** The doctor proposes `fact.pairs: occupancy` for FactResellerSales and FactFinance (see the doctor set below). |
| `association_max_diff` (3-5), `discriminator_auc` (3-4) | FactResellerSales assoc 0.61 (0.15); DimProduct assoc 0.60 (0.34), AUC 0.83 (0.65); DimCustomer assoc 0.21 (0.15) | copula.*, joint_*, segments | **Spec, maybe settings.** The doctor set (copula sample rows 3000, grid 128, more segments) lowered the excess by 7 per seed but broke privacy (below). Its parts have not been tried alone yet (new in this PR). |
| `marginal_tvd` (2-4) | FactResellerSales CarrierTrackingNumber and CustomerPONumber 0.966 (limit 0.30) | marginals.* | **Spec.** These are order-level codes (one value per order): the synthetic codes never equal real ones, which is wanted. They should be typed as identifiers or text, not categoricals. |
| `marginal_ks` (3-4) | FactInternetSales Freight and TaxAmt 0.22-0.26 (0.08); FactProductInventory UnitsBalance 0.17-0.26 | marginals.*, copula.grid | **Engine.** Freight and TaxAmt are the derived 2.5% and 8% of SalesAmount, and G1 holds exactly, so the misfit lies in how the derived amounts are rounded or kept as values. UnitsBalance is a cumulative rule (`prev + In - Out`): its opening level drifts. |
| `cardinality_ks` (4-5) | FIS ProductKey→DimProduct 0.29; FRS ResellerKey→DimReseller 0.25; FRS ProductKey 0.17; FactSurveyResponse subcategory 0.53 | segments, structure.beta_prior | **Not settled.** `facts.count_model`, `facts.frailty` and `facts.link_degree` were not reached before the budget ran out. |
| `join_tvd`, `join_ks` (3-17) | FRS joined product DaysToManufacture, ReorderPoint; reseller BusinessType; FactSurveyResponse Title 0.35 (0.10) | segments, pooling | Mostly follows the cardinality and support misfits above. It is very noisy across seeds (4, 2 and 17 on one candidate). |

### The doctor's spec fixes

`relsynth doctor` proposed 13 patches at once: `copula.sample_rows 3000`, `copula.grid 128`,
`copula.max_iterations 4800`, `marginals.max_levels 400`, `marginals.tail_trim 0`, `marginals.dirichlet_prior 0.125`,
`facts.frailty: true`, `segments.stratify_max_levels 40`, segments 16-32 for DimCustomer, DimEmployee, DimProduct and
DimReseller, and `fact.pairs: occupancy` for FactResellerSales and FactFinance. Together they cut the excess failures
from 45.4 to 38.7 (and to 36.7 with `default_cap: 16`), the largest gain of the search. They were rejected because
they broke a scorecard privacy check on 3 of 3 seeds, and once a G1-G14 identity. The likely cause is the larger
segment counts (more segments copy smaller groups). Autotune tried only the whole set; this PR now tries its parts
when the whole set is rejected.

**Third run (2026-11-06, run 42142808423, block by block).** Each block was tried alone over
`segments.default_cap: 16`, on seeds 1-3:

| Block | Excess per seed (incumbent 42.0) | Hard checks broken |
|---|---|---|
| segment counts 32 for DimCustomer, DimEmployee, DimProduct and DimReseller | 41.3 | privacy 1 |
| marginals (`dirichlet_prior 0.125`, `max_levels 400`, `tail_trim 0`) | 43.3 | privacy 2 |
| copula (`sample_rows 3000`, `grid 128`) | 40.7 | privacy 1 |
| `facts.frailty: true` | 47.0 | privacy 1 |
| `segments.stratify_max_levels 40` | 38.0 | privacy 3 |
| FactFinance `fact.pairs: occupancy` | 45.3 | privacy 1, rules 1 |
| FactResellerSales `fact.pairs: occupancy` | 47.0 | privacy 1 |

Every block, even ones that cannot plausibly change disclosure (`facts.frailty`), breaks a privacy check on at least
one of three seeds, and so do `structure.beta_prior: 1` and `segments.default_cap: 32`. The incumbent passes privacy
narrowly, and any change to the draws can flip it. So the privacy constraint acts as a noisy check held literally,
not as a real trade-off. A fourth run names the failing check and its values per seed to settle this.

## Autotune and engine issues found, and fixed in this PR

1. **A rule the source data itself breaks was treated as exact.** The FactResellerSales weekday rule is broken by 9
   of 60,855 source rows. A synthetic run with 12 such rows was a hard `rules` violation on 4 of 5 bench seeds. The
   limit now adds three binomial standard errors when the source rate is above 0, and an exact rule stays exact.
2. **A setting bound the spec refuses.** `structure.structural_prior` had registry bound 1, but the spec accepts only
   [0, 1). The guided search stepped to 1.0 and all six of those runs failed (about 2 runner-hours). The bound is now
   0.999, and a test checks every tunable bound against spec validation.
3. **Trade-offs read from seed noise.** The report called `novel_combination_rate` (22, 22, 22 failures per seed) a
   "conflict", because a change of one check per seed counted as an effect. A change now counts only when it is at
   least one check per seed and twice its paired standard error.
4. **The doctor's fixes were all-or-nothing.** When the whole set is rejected, autotune now tries the patches block
   by block (all per-table segment counts together, the copula patches together, and so on).
5. **Cost.** A candidate's seeds now run in parallel (`--workers`). The bench runner and workflow
   (`tools/autotune_bench.py`, `autotune.yml`) print the report and a per-candidate summary to the job log, and a
   later run can resume a search past the 6-hour job limit.

## Not explained yet

- One link-affinity run broke `G6 FIS Due = Order + 12d` on 1 of about 30,000 FactInternetSales rows (seed 2). The
  spec change touched only FactResellerSales, and the identity held on all other runs. It needs a reproduction.
- Runs in the second job took 2 to 3 times longer (2,200-4,200 s against 1,200-1,500 s). The likely cause is memory
  pressure from three parallel fits with larger doctor segment counts. It was not measured.

## Next

- Resume the search with fixes 3 and 4 (the doctor's parts alone, then `facts.count_model`, `facts.frailty` and
  `facts.link_degree` for `cardinality_ks`).
- Engine work, which the settings cannot replace: cross-parent support for hop pairs (`novel_combination_rate`), the
  rounding of derived amounts, and the opening level of cumulative rules.
- Spec or introspect: type order-level codes such as CarrierTrackingNumber as identifiers.

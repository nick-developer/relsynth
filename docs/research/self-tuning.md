# Research: a fully adjustable engine and a self-tuning loop

Status: research report, written before implementation (2026-10-31).
Scope: the user request of 2026-10-31. (1) Make every relevant engine function adjustable by the user, through a YAML
settings file or editable class attributes. (2) Study whether the engine can learn from a history of runs over seeds,
find error patterns, and adjust its parameters until the user's targets hold, for example "no domain links broken" or
"no privacy rule violated". Where that cannot be done, the engine should report the practical constraints instead.

The report has five parts:

1. What can be adjusted today, and what cannot (an audit of the code).
2. What the bench history already shows about errors across seeds.
3. The academic and technical background.
4. The design that follows from it.
5. Feasibility limits, and the constraints report that replaces a fix when no fix exists.

---

## 1. Audit: what the user can adjust today

### 1.1 Already adjustable

| Layer | Where | Count |
|---|---|---|
| Model structure | `tables`, `relationships`, `rules`, column options (spec) | about 150 keys, `relsynth.schema.S` |
| Engine parameters | `synthesis.*` (spec), defaults in `relsynth.spec.DEFAULT_SYNTHESIS` | 78 keys in 16 groups |
| Per-table overrides | `segments.tables`, `scale.tables`, `tables.<t>.generator`, `fact.*` | partial |
| Devices | `synthesis.compute` and a `compute.yaml` preset file | 9 fields |
| Privacy assessment | `privacy.yaml` (`relsynth.privacy.config`) | about 30 keys |
| DP | `synthesis.privacy.dp` (`relsynth.dp.config`) | about 15 keys |
| Command line | `--set KEY=VALUE` on `bench run` only | any synthesis key |

`relsynth.schema.check_keys` already rejects unknown keys and suggests the nearest one. That property must be kept.

### 1.2 Not adjustable: constants in the code

An AST scan of `src/relsynth` (leaving out connectors and the service) finds 1,342 numeric literals other than 0, 1,
2, 3, 10, 100, 0.5 and -1. Most of them are numerical guards, such as `1e-300` before a `log` or `1e-9` in a
comparison, or fixed formats such as nanoseconds per day. They are not modelling choices, and exposing them would only
let a user break the arithmetic. Sorting the rest by what they control gives about 60 real modelling or effort choices.
None of them can be set today:

| Module | Constant (current value) | What it decides |
|---|---|---|
| engine | pair repair tries (200), entity constraint rounds (200), anchor thinning rounds (200), sample redraw rounds (40), unique-code rounds (50) | effort spent repairing a draw before it is accepted as is |
| engine | occupancy calibration rounds (21, 80), participation rounds (4), conditional-count sweeps (30) | inner fixed-point and Gibbs effort |
| engine | rake levels (50) and cells (2000) | which rule cells are raked |
| engine | group-effect share (0.95), SCD stable share (0.99), SCD versions kept (50) | detection thresholds of structures |
| engine | occupancy z threshold (-4) and expected count (10) | Poisson or occupancy for a pair fact |
| engine | auxiliary parent columns (more than 30 levels, 2 kept) | which parent attributes condition a fact |
| engine | rule-fitting samples (5,000 and 20,000 rows) | rows used to learn a rule's floor and decimals |
| copula | Metropolis steps per sweep (15), MCSE tolerance (0.01), rare-column share (0.01) | copula sampler effort and its convergence rule |
| relational | ZOIB weight sweeps (50), k-means++ iterations (25), NIG prior (k0 0.1, a0 2, b0 1), dispersion minimum rows (30) | the segment model's priors and initialisation |
| marginals | integer-as-discrete limit (30), tail rows (5), marginal grid (48), observations kept (20,000) | how a column's law is represented |
| table model | functional-dependency sample (50,000 rows) | rows used to find a functional dependency |
| evaluate | the 20 `THRESHOLDS` (KS 0.08, TVD 0.08, AUC 0.65, ...), the 2.5 sd level, the association rows (20,000), the discriminator rows (5,000) | what counts as a failed check |

The last row matters most for this request. **The thresholds that decide "pass" cannot be changed**, so a user
cannot say what "good enough" means for their own data. A target-driven loop needs that first.

### 1.3 How the engine reads its settings

The engine reads one merged dict, `spec["synthesis"]`. `normalise` builds it as `deep_merge(DEFAULT_SYNTHESIS,
spec.synthesis)`, and the engine indexes it directly (`self.cfg["copula"]["sample_rows"]`). This has three
consequences:

- **No types or ranges.** `copula.sample_rows: "many"` or `gibbs.rhat_target: 0.5` is accepted, and the error comes
  later, deep in the fit, or never.
- **No provenance.** Nothing records whether a value came from the default, a file or an override. A run's record
  cannot say what the user changed.
- **No search space.** A tuner needs bounds, a scale (linear or log) and a list of the keys it may move. None of that
  exists.

## 2. What the bench history shows about errors across seeds

The bench records every check of every seed (`bench/history/*-ccdabff-*.json`: five seeds, six databases, GitHub
runners, 2026-10-31). Two counts from those records shape the design.

### 2.1 Most failures are systematic, not noise

For each quality or scorecard check, count the seeds (out of 5) on which it fails:

| Database | fails on 1 seed | 2 | 3 | 4 | **5 of 5** |
|---|---|---|---|---|---|
| AdventureWorks | 28 | 23 | 12 | 9 | **106** |
| Berka | 5 | 4 | 0 | 0 | **41** |
| Contoso (SQLite) | 9 | 3 | 3 | 2 | **16** |
| Chinook | 3 | 1 | 1 | 0 | **4** |
| ACS | 0 | 1 | 0 | 0 | **7** |

A check that fails on every seed is a misfit of the model, and it repeats. That is the signal a learning loop can use.
A check that fails on one seed out of five behaves like sampling noise. Changing parameters to make it pass would be
fitting the seed, not the data. The families that fail systematically are few. On AdventureWorks they are:

- `novel_combination_rate` (22 checks);
- `hop_trend_worst_gap` (15);
- `js_distance` (8);
- `beta_recall_gap` (6);
- cardinality KS and k-way TVD (4 each).

So the loop should learn per family, not per check.

### 2.2 "Zero failed checks" is not reachable even for a perfect model

Each statistical check has a level α_i, its false-failure rate when the synthetic data really is a draw from the real
data's law. The quality report records α_i. If all K checks were independent, the chance that a perfect generator
passes all of them is ∏(1 − α_i):

| Database | quality checks with a level | expected false failures Σα | P(all pass, perfect model) per seed |
|---|---|---|---|
| AdventureWorks | 484 | 4.41 | **0.012** |
| Contoso | 140 | 1.02 | 0.33 |
| Chinook | 87 | 0.81 | 0.44 |
| Berka | 144 | 0.90 | 0.41 |
| ACS | 38 | 0.05 | 0.95 |

On AdventureWorks, a perfect model passes every quality check on 1.2% of seeds. It passes on five seeds in a row with
probability about 2·10⁻¹⁰. A loop told to reach "zero failures" on statistical checks would never stop. Worse, it
would select settings that happen to pass on the seeds it saw, which is the selection bias described by Cawley and
Talbot (2010). The target for statistical checks must therefore be calibrated: observed failures no more than an
upper quantile of their null count. Literal zero belongs only to checks that hold by construction, such as foreign
keys resolving, check rules, domain identities and unique keys.

### 2.3 Privacy "violations" are often an undecided test, not a leak

The privacy checks that fail in the history are mostly `status: review`. The point estimate is below the limit, but
the upper confidence bound is above it:

- Contoso, singling out: estimate 0.029, upper bound 0.110, limit 0.10.
- Berka, singling out: estimate 0.046, upper bound 0.113.
- ACS, DPHY inference: estimate 0.020, upper bound 0.225.

Changing the generator cannot narrow a confidence interval. Only more attack and control rows can, and a small
population caps those. A loop that treated these as violations would push the generator toward noise, and fidelity
would drop, for no privacy gain. Only ACS `DREM` inference (0.188 against 0.10) and the LiRA AUC (0.665 against 0.65)
have point estimates over the limit. They are real targets for privacy measures.

## 3. Background

### 3.1 The problem in optimisation terms

Let θ be the engine settings, s the seed and c_j(θ, s) the value of check j. The request is a **noisy, constrained,
black-box optimisation**:

- minimise the systematic misfit E_s[f(θ, s)];
- subject to P_s[c_j(θ, s) violates its limit] ≤ δ_j for every user constraint j.

Each evaluation is a full fit, sample and evaluate cycle. It takes minutes on AdventureWorks, so the budget is tens of
evaluations, not thousands. The literature for this setting:

- **Bayesian optimisation** with a Gaussian-process surrogate and expected improvement (Snoek, Larochelle and Adams,
  NeurIPS 2012). It is efficient for fewer than about 20 continuous parameters and small budgets.
- **Constrained Bayesian optimisation.** Gardner et al. (ICML 2014) and Gelbart, Snoek and Adams (UAI 2014) model each
  constraint with its own surrogate. A point's improvement is weighted by the probability that it is feasible.
  Letham, Karrer, Ottoni and Bakshy (Bayesian Analysis 2019) extend this to **noisy** objectives and constraints with
  noisy expected improvement, which is exactly our seed noise.
- **Tree-structured Parzen estimators** (TPE; Bergstra, Bardenet, Bengio and Kégl, NeurIPS 2011) and **SMAC**
  (Hutter, Hoos and Leyton-Brown, LION 2011). Both handle categorical and conditional parameters, which most of ours
  are (`auto | on | off`, `rank | bayes | fast`). Optuna (Akiba et al., KDD 2019) implements TPE with a constraint
  function.
- **Successive halving and Hyperband** (Jamieson and Talwalkar, AISTATS 2016; Li et al., JMLR 2018). They spend little
  on bad candidates: here that means one seed first, and more seeds only for candidates that look good.
- **Variance in benchmarks.** Bouthillier et al. (MLSys 2021) show that seed variance often exceeds the differences
  between methods. A comparison must use several seeds and paired seeds, with the same seeds for the incumbent and the
  candidate.

### 3.2 Learning which parameter to move

Black-box search ignores what we know about the engine. That knowledge is large: `relsynth doctor` already maps 40
bundle patterns to spec patches. The Bayesian workflow (Gelman et al., 2020, arXiv:2011.01808) frames model
improvement as a loop: posterior predictive checks find the misfit, then the modeller expands the model where the
check failed. Our checks are posterior predictive checks (`verify` R3), so a check family points to the model part to
change. For example:

- `association_max_diff` points to the copula sample and the pooled cells;
- `cardinality_ks` points to the count model, frailty and link degree;
- `novel_combination_rate` points to the structural prior and conditional inclusion.

A **guided search** first moves the parameters linked to the failing families, in the known direction. It then falls
back to model-based search (TPE) in the reduced space. Lindauer et al. (JMLR 2022, SMAC3) and Feurer et al. (NeurIPS
2015, auto-sklearn's meta-learned warm start) show that a good prior over where to search matters more than the
surrogate.

### 3.3 Deciding that a constraint holds

Seeds are Bernoulli trials for each constraint. After n seeds with k violations, Beta(1 + k, 1 + n − k) is the
posterior of the violation probability p_j:

- with **k = 0**, the 95% upper bound is 1 − 0.05^{1/(n+1)}, about 3/n by the "rule of three" (Hanley and
  Lippman-Hand, JAMA 1983);
- a claim "p_j ≤ 5%" with 95% confidence and zero failures needs **n ≥ 58 seeds**;
- a claim "p_j ≤ 20%" needs 13 seeds.

This is the main practical limit. On AdventureWorks one seed takes about 20 minutes, so a strong guarantee costs
hours. The loop must therefore say what its evidence supports: "0 violations in 5 seeds: p ≤ 0.45 at 95%". It must not
say "no violations".

For calibrated statistical families, the count of failures F under the null is approximately Poisson-binomial with
mean Σα_i. A family passes when F ≤ q_{0.99}(F_null). The rule then holds a perfect model to its own noise and nothing
more.

### 3.4 Seed overfitting and held-out seeds

Settings chosen because they pass on seeds 1–3 are biased toward those seeds (Cawley and Talbot, JMLR 2010). The fix
is the usual one: tune on search seeds and **confirm on held-out seeds** the search never saw. A run that does not
confirm is reported as not confirmed, and the earlier settings stay.

### 3.5 Privacy: tuning is itself a query on the real data

Every tuning trial evaluates the synthetic data against the real data, and the chosen settings depend on the real
data:

- **Without DP.** Choosing among many candidates by fidelity can drift toward memorisation. Privacy constraints
  (DCR, singling out, linkage, inference, membership) must therefore be constraints of the search, not reports after
  it. Stadler, Oprisanu and Troncoso (USENIX Security 2022) show that utility and privacy trade off for every
  generator. A loop that maximises fidelity alone will find the leaky end of that trade-off.
- **With DP.** Selection leaks privacy. Liu and Talwar (STOC 2019) and Papernot and Steinke (ICLR 2022) give private
  selection with a bounded extra cost: random stopping and an (ε, δ) or Rényi DP overhead about three times one run's
  budget. Each trial also spends budget. Under DP mode the loop must either refuse, as `relsynth tune` does today, or
  reserve the selection budget in the governance ledger before it reads (T08 rule D2, reserve before read).

### 3.6 Existing tools

None of these tools does what is asked:

- SDV's `HyperTransformer` and Synthcity (Qian et al., NeurIPS 2023 Datasets track) expose per-plugin hyperparameter
  spaces and let a user run Optuna against one utility score. They have no constraints and no seed-aware acceptance.
- Gretel and Mostly AI tune a neural generator's training. They do not tune relational integrity or business rules.

The closest is relsynth's own `tune`: run, doctor, apply, repeat. It has four gaps:

- it uses one seed, so it cannot tell systematic failures from noise;
- it has no user targets;
- it has no memory between runs;
- it changes only the spec structure, not the engine parameters.

## 4. Design

### 4.1 One parameter registry (`relsynth.settings`)

Every adjustable parameter gets one entry, built from `DEFAULT_SYNTHESIS` and `SYNTH_DOC` plus the constants promoted
from §1.2:

```
Param(key="copula.sample_rows", default=1500, type=int, low=100, high=200_000, scale="log",
      tunable=True, affects=("association", "discriminator", "kway"), doc="Rows used to fit a copula.")
```

The registry is the single source of truth for:

- validation: types, ranges and choices are checked when the settings are loaded;
- the generated reference (`docs/settings-reference.md`);
- the templates (`relsynth settings template`);
- the tuner's search space (`tunable`, `low`, `high`, `scale`, `choices`) and its move directions (`affects`).

### 4.2 Three ways to set a parameter, one precedence

From lowest to highest precedence:

1. Built-in defaults (the registry).
2. A **settings file**, named explicitly by `synthesis.settings_file: path` (relative to the spec),
   `--settings path`, `RELSYNTH_SETTINGS` or a preset (`preset:fast`, `preset:thorough`). No file is picked up
   implicitly, so a run never depends on a file that happens to sit next to the spec. It is a plain YAML of the
   `synthesis` groups. It can also hold a `per_table:` block of overrides for the table-level groups (copula,
   marginals, text) and `evaluation.thresholds` for the scorecard.
3. The spec's own `synthesis:` block.
4. `--set KEY=VALUE` on `run`, `fit`, `sample`, `verify` and `autotune`, and `Spec.set()` in the API.

The effective value of every key records its source, so `relsynth settings show spec.yaml` prints value and source.
History records store only the keys that differ from the defaults.

In Python the same settings are **editable attributes** with validation on assignment:

```python
s = relsynth.Settings.load("spec.yaml")       # defaults < file < spec
s.copula.sample_rows = 5000                    # checked against type and range
s.evaluation.thresholds.ks = 0.06
s.tables["FactInternetSales"].copula.sample_rows = 20000
model = relsynth.fit("spec.yaml", settings=s)
```

The engine keeps reading the same merged dict (`Settings.to_dict()`). Defaults stay identical, so with no settings the
synthetic output stays byte-identical to today's. That is checked on every suite.

### 4.3 The learning loop (`relsynth autotune`)

> The sketch below is the design as researched. The implemented syntax differs in small ways (for example
> `budget.runs` instead of `trials`, privacy as a plain constraint name with `privacy.review` deciding the
> undecided tests, TPE replaced by a built-in Parzen-style proposal with no Optuna dependency, and no
> report-only mode yet); [docs/autotune.md](../autotune.md) is the reference for the shipped version.

```
relsynth autotune spec.yaml --objectives objectives.yaml [--budget 20] [--history DIR]
```

The objectives file holds the user's specification:

```yaml
constraints:                      # must hold on every seed (hard)
  - referential_integrity          # every foreign key resolves
  - rules                          # no check rule violated
  - unique_keys
  - privacy: {status: pass}        # no privacy criterion fails; 'review' counts as pass or fail (user's choice)
targets:                           # statistical: calibrated (observed failures <= null quantile)
  - family: association_max_diff
  - family: cardinality_ks
  - scorecard: {min_pass_rate: 0.98}
seeds: {search: [1, 2, 3], holdout: [11, 12]}
budget: {trials: 20, hours: 4}
search: auto                       # parameters linked to the failing families; or an explicit list with bounds
```

Each round:

1. **Evaluate.** Fit, sample and evaluate each candidate on the search seeds, with paired seeds, the same for every
   candidate. Store an aggregate-only record in the history (`<spec>.tune/history.jsonl`): settings delta, seed, every
   check's value, limit, pass and level, cost. No rows and no category values are stored, the same rule as bundles.
2. **Classify.** For each check, compute its posterior failure rate across seeds. Label it *systematic* (lower bound
   above its α), *sporadic* (consistent with α) or *passing*. Aggregate the systematic ones into families.
3. **Propose.**
   - First the doctor's spec patches, when the user allows spec changes.
   - Then guided moves: for each systematic family, the registry parameters whose `affects` names it, stepped in the
     known direction on the parameter's scale.
   - When the guided moves are spent, a TPE proposal over those parameters, with the history as its observations.
     Optuna is used when installed; otherwise a built-in TPE on the same history.
4. **Successive halving.** A candidate runs on one seed first. It goes on to all search seeds only if it does not
   break a constraint the incumbent holds and does not raise its systematic misfit.
5. **Accept** a candidate when it holds every constraint on every search seed and has lower misfit than the
   incumbent. Ties go to the smaller change.
6. **Confirm** the final incumbent on the held-out seeds, then write `<spec>.tuned.yaml` and the report.

The history persists. A second `autotune` on the same spec and data fingerprint starts from the last incumbent and
reuses every trial already run. `relsynth autotune --report` rebuilds the report from the history without running
anything.

### 4.4 What the loop never does

- It never weakens a user threshold to make a check pass. Thresholds are the user's objectives, not parameters.
- It never trades a hard constraint for a target.
- Under DP mode it refuses unless a selection budget is reserved (§3.5).
- It never claims more than the seed evidence supports (§3.3).

## 5. Feasibility limits and the constraints report

The loop always produces a report. When some objective cannot be met, the report says why, using five tests that can
be computed from the history:

| Test | Evidence | Conclusion |
|---|---|---|
| Noise floor | Expected false failures Σα and its 99% quantile for each family (§2.2) | A target of fewer failures than the null quantile is unreachable for any generator. The report gives the reachable target. |
| Seed evidence | Beta posterior of each constraint's violation rate (§3.3) | "0 of n" supports p ≤ 3/n. The report gives the seeds needed for the user's δ. |
| Insensitive | Every parameter linked to the family was moved over its range with no significant change (paired seeds) | The misfit is in the model class, not the settings. It needs a spec change (doctor) or an engine change. |
| Conflict | Over the trials, two objectives' pass indicators are negatively associated (for example novel combinations against singling out) | A Pareto trade-off. The report lists the non-dominated settings and lets the user choose. |
| Undecided test | Privacy `review`: the confidence interval straddles the limit (§2.3) | Not fixable by the generator. Needs more attack rows, or the population is too small. |

The report also states the cost: trials, CPU hours, and the seeds per trial the user would need for the confidence
they asked for.

## 6. Plan and tests

1. **Settings** (`relsynth.settings`, generated reference, `relsynth settings` CLI, API attributes). Tests:
   - every registry default equals the 1.0 constant it replaces;
   - unknown keys and out-of-range values are refused with hints;
   - precedence across the four layers;
   - provenance;
   - **byte-identical output with no settings on every bench suite** (GitHub CI bench run).
2. **Promoted constants** (§1.2), threaded through the engine with unchanged defaults. Tests: identity as above, plus
   one test per promoted key showing that the key is read (changing it changes the output or the effort).
3. **Thresholds** (`evaluation.thresholds`). Tests: the scorecard uses them, and the defaults reproduce today's
   scorecard.
4. **Autotune** (`relsynth.autotune`, history store, classifier, guided moves, TPE, successive halving, confirm,
   report). Tests:
   - fake scenarios with a planted misfit that a known parameter fixes, where the loop must find it;
   - a planted infeasible target (below the noise floor), where the loop must report it infeasible and not
     "optimise" it;
   - a planted conflict;
   - DP refusal;
   - reproducibility of the whole loop for a fixed seed list;
   - a GitHub CI job that runs autotune on Chinook and Contoso.

---

### References

- Akiba, Sano, Yanase, Ohta, Koyama. Optuna: a next-generation hyperparameter optimization framework. KDD 2019.
- Bergstra, Bardenet, Bengio, Kégl. Algorithms for hyper-parameter optimization. NeurIPS 2011.
- Bouthillier et al. Accounting for variance in machine learning benchmarks. MLSys 2021.
- Cawley, Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. JMLR 11, 2010.
- Feurer et al. Efficient and robust automated machine learning. NeurIPS 2015.
- Gardner, Kusner, Xu, Weinberger, Cunningham. Bayesian optimization with inequality constraints. ICML 2014.
- Gelbart, Snoek, Adams. Bayesian optimization with unknown constraints. UAI 2014.
- Gelman et al. Bayesian workflow. arXiv:2011.01808, 2020.
- Hanley, Lippman-Hand. If nothing goes wrong, is everything all right? JAMA 249, 1983.
- Hutter, Hoos, Leyton-Brown. Sequential model-based optimization for general algorithm configuration. LION 2011.
- Jamieson, Talwalkar. Non-stochastic best arm identification and hyperparameter optimization. AISTATS 2016.
- Letham, Karrer, Ottoni, Bakshy. Constrained Bayesian optimization with noisy experiments. Bayesian Analysis 14, 2019.
- Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. Hyperband. JMLR 18, 2018.
- Lindauer et al. SMAC3: a versatile Bayesian optimization package for hyperparameter optimization. JMLR 23, 2022.
- Liu, Talwar. Private selection from private candidates. STOC 2019.
- Papernot, Steinke. Hyperparameter tuning with Rényi differential privacy. ICLR 2022.
- Qian, Cebere, van der Schaar. Synthcity: facilitating innovative use cases of synthetic data in different data modalities. NeurIPS 2023 (Datasets and Benchmarks).
- Snoek, Larochelle, Adams. Practical Bayesian optimization of machine learning algorithms. NeurIPS 2012.
- Stadler, Oprisanu, Troncoso. Synthetic data – anonymisation groundhog day. USENIX Security 2022.
- Giomi, Boenisch, Wehmeyer, Tasnádi. A unified framework for quantifying privacy risk in synthetic data (Anonymeter). PoPETs 2023.

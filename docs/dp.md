# Differential privacy mode

relsynth can produce a synthetic database with a formal differential privacy (DP) guarantee. The guarantee is user level: anyone who sees the synthetic data learns about the same about one person (one row of the privacy unit table and every row that belongs to it) whether or not that person was in the real database. Install the extra first:

```
pip install 'relsynth[dp]'          # OpenDP, MIT licence
```

An external release (a context with `external: true` in the organisation policy) must be a DP run. relsynth enforces this before it reads any data (see the [governance guide](governance/guide.md)).

## 1. Turn it on

Add a `dp` section under `synthesis.privacy`, or pass `--epsilon` on the command line:

```yaml
synthesis:
  privacy:
    dp:
      unit: Customer                 # the table of the people DP protects; one row per person
      epsilon: 2.0                   # this run's budget ...
      delta: 1.0e-9                  # ... stated at this delta (at most 1e-5; at most 1/n^2 is common)
      public_tables: all_unowned     # or a list; tables without data of the unit, released as given
      public:                        # optional: facts known without the data, which save budget
        Customer.Country: {levels: [GR, DE, FR, US]}
        Invoice.Total: {min: 0, max: 50}
      max_degree: {Invoice: auto, InvoiceLine: 30}   # rows kept per parent row; auto = chosen privately
      spec_is_public: true           # the spec was written from public facts only (see section 5)
```

```
relsynth run spec.yaml                       # the dp section makes it a DP run
relsynth run spec.yaml --epsilon 1 --dp-unit Customer --dp-statement dp.json
```

relsynth has no default ε. A DP run without `epsilon` or `rho` is refused, and so is a δ above 10⁻⁵. Instead of `epsilon` you can give the budget in zero-concentrated DP as `rho`.

## 2. What happens

1. **Plan, before any read.** relsynth finds the tables that hold data of the unit: those with a chain of foreign keys to it. Every other table must be listed in `public_tables`, or `all_unowned` must be set. The plan refuses three cases, before reading anything or charging the budget: a table with two chains to the unit, a self-reference on an owned table, and a foreign key between owned tables off the chain.
2. **Stage 1: noisy statistics.** relsynth reads the real tables once. For each owned table it releases the following, all as integers with discrete Gaussian noise from OpenDP:
   - ranges of numeric columns, found on a fixed logarithmic grid;
   - category levels, by partition selection (rare levels are dropped);
   - one histogram per column and the row count;
   - a histogram of rows per parent row, which is used to choose how many child rows to keep per parent;
   - a noisy correlation matrix of normal scores, which includes the parent row's attributes.

   Each statistic is charged to an accountant that refuses anything beyond the run's budget.
3. **Seed database.** The statistics are decoded into a small database with the same tables, keys and links.
4. **Stage 2.** The ordinary relsynth engine is fitted to the seed database and samples the synthetic database at the requested scale. This is post-processing, so it keeps the guarantee.

The proofs are Propositions 900 to 906 in [docs/proofs](proofs/sec_v25_dp.tex). Checks C900 to C906 (`pytest -m science tests/science/test_dp_checks.py`) test them on this code. C902, for example, removes one customer at a time and confirms that no released statistic moves by more than the sensitivity it was charged for.

## 3. The DP statement

Every DP run produces a statement (schema `relsynth.dp-statement/1`; write it with `--dp-statement`). It follows NIST SP 800-226 and contains:

- the unit and the adjacency;
- the guarantee in ε, δ and ρ, and how it was computed (zCDP, converted with Canonne, Kamath and Steinke 2020);
- the mechanisms by stage, and the noise library and randomness;
- what is post-processing, what was public, and what is not protected.

It holds no data and no exact counts. In a governed run it is signed into the manifest and goes into the release package.

## 4. Governed runs and the budget

```
relsynth run spec.yaml --epsilon 1 --registry registry/ --context partner --release-target partner-sftp \
    --source-id dwh-prod --key ana.key --assess
```

- The unit must be the privacy unit of the source's domain in the policy. δ is the domain's δ, and the run's δ for level selection is 1% of it by default (`levels_delta_share`), so many runs fit.
- The budget is **reserved** in the domain's ledger before the read. It is marked **touched** just before the read, and **settled** at the amount actually spent. A run refused by the per-release, period or lifetime cap reads nothing.
- `--assess` runs the attack suite against the DP pipeline itself, and compares each attack's true-positive rate with the bound that ε implies.

## 5. Choosing ε, and what DP does not cover

- NIST SP 800-226 (sec. 2.2) gives no rule. ε ≤ 1 is strong, and deployments use 1 < ε ≤ 20. Above 1, measure the risk (`relsynth privacy` audits DP specs through the DP pipeline). Section 6 shows what each ε costs in quality on real databases.
- **The spec.** `introspect` drafts a spec from the real data: levels, ranges and rules. Those values are outside the guarantee unless you write the spec from public facts only and set `spec_is_public: true`. For an external release the policy requires this (`spec.public_facts_only`). Declare known facts under `public`; anything not declared is found privately in stage 1.
- **Public tables** are released as given. List only tables without personal data.
- **Other releases** of the same people made without DP are not covered. The ledger counts them separately.
- **Bias.** Rare category levels are suppressed, and rows beyond the per-parent cap are dropped before the statistics. Both pull the synthetic data towards typical people.
- **Not reproducible.** The noise comes from the operating system's random source and is not seeded. Two runs give different data, and each one spends budget.

What DP mode refuses: `tune`, `verify`, `export-model` and `doctor --apply` refuse a DP spec, because they would turn real data into models or spec values outside the accounting. The ordinary pipeline (`relsynth.run`, `execute`) also refuses a DP spec rather than run it without DP. Run those on a copy of the spec without the `dp` section, for internal use only.

Not supported yet: in-database pushdown in DP mode (every owned table is loaded), more than one privacy unit, composite parent keys (generated as `(i, 1)`), and values of `pii`, `text` and `binary` columns (placeholders). Views are derived in stage 2.

## 6. Utility against ε

See the [T07 report](../roadmap/reports/T07.md) for the scorecards at ε ∈ {0.5, 1, 2, 4, 8} on Chinook, Contoso and AdventureWorks. It also compares DP mode with MST and AIM and checks the attack audit against the bound.

## API

```python
import relsynth.dp as dp
res = dp.synthesize("spec.yaml", write=False)
res.data, res.statement, res.epsilon, res.charges
pl, cfg, spec = dp.plan("spec.yaml")      # what a run would do, without reading data
```

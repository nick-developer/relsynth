# Privacy risk measurement

`relsynth privacy` attacks a synthetic release the way a recipient could, person by person, and writes a privacy report with one page per release context (internal, partner, public). A data protection officer can read the pages without the annex; the annex and the JSON file hold the technical results.

For the people who approve a release, the assurance case, the regulation mapping and a DPIA template are in [docs/assurance](assurance/README.md).

```bash
relsynth privacy spec.yaml --config privacy.yaml --out privacy_report --cache .relsynth-cache --jobs 4
relsynth privacy spec.yaml --config privacy.yaml --context internal --fail-on review   # CI gate: exit 1
```

```python
import relsynth as rs
rep = rs.privacy.assess("spec.yaml", "privacy.yaml")
rs.privacy.write_report(rep, "privacy_report")        # privacy_report.json, .md, .html
rep["contexts"]["internal"]["verdict"]                # 'pass' | 'review' | 'fail' | 'blocked'
rs.privacy.reference("privacy_report/privacy_report.json")   # embedded by the quality report: schema, sha256, verdicts
```

The report never shows a real key. People are named by keyed hashes (`P-3f2a…`, HMAC-SHA256 of the key) under a secret 256-bit salt that is not the split seed: the seed is in the report, so it must not unlock the labels. `relsynth privacy` writes the salt to `privacy_report/privacy_salt.key` (owner-only), or reuses `--salt-file` so labels stay stable between runs; the report carries only the salt's id. In the API, `assess(..., salt=...)` takes it and `rep["_salt"]` returns it; neither the JSON nor the pages ever contain it. Keep the salt apart from the report.

## What is measured

| EDPB 02/2026 criterion | Question | Attack | Result |
|---|---|---|---|
| No record isolation (sec. 3.4.1) | Can someone single out a real person? | Singling-out queries built from the synthetic data, univariate and multivariate (Giomi et al. 2023) | Risk: share of successful queries on training people above unseen people |
| No linkage (sec. 3.4.2) | Can someone connect two records about one person? | Nearest-neighbour linkage of two partial records through the synthetic data (Giomi et al. 2023) | Risk, as above |
| No inference (sec. 3.4.3) | Can someone learn a person's attribute? | Nearest-neighbour inference (Giomi et al. 2023), and attribute inference against an attacker that sees only population data (para. 67: an inference counts only when population data would not give it) | Risk; specific leakage = advantage on members minus advantage on non-members |
| Membership | Can someone tell that a person's data was used? | Shadow modelling with TAPAS-style features (Stadler et al. 2022; Houssiau et al. 2022), a LiRA-style likelihood ratio (Carlini et al. 2022), DOMIAS (van Breugel et al. 2023) | AUC and TPR at FPR 10%, 1%, 0.1%, with exact intervals |

Every number has a confidence interval (95% by default). Membership attacks are run on the most exposed people, chosen by the Meeus et al. (2023) outlier score, and on as many random people. The ten people most at risk are listed with their attack AUC and whether they were singled out, linked or had a secret guessed (EDPB para. 87: an attack that works on one person is enough).

### People, not rows

The **privacy unit** is the table that holds the people (`unit`, detected when omitted: the person table that owns the most tables). Every table that reaches it through foreign keys belongs to it: a customer's invoices and invoice lines are attacked together with the customer. The attacks run on one profile row per person: the person's attributes, the attributes of their parents (store, region), and per owned table the number of rows and the mean or most frequent value of each attribute, including those of the rows' parents (the genre of the tracks bought). Profiles hold no key values, because synthetic keys are new labels. Chains stop at other person tables: a customer's sales representative does not own the customer.

### How the release is tested

The people are split at random into **training** (40%), **control** (40%) and **reference** (20%). The release is fitted on the training people only. Control people are never seen: the attacks' success on them is what the synthetic data teaches about anyone, and only the excess on training people is a risk to them. Reference people are the attacker's population data. Training and control have the same size so the record-isolation rates are exactly comparable (Prop. 601); Anonymeter's fitted size correction showed risks of 0.10–0.14 under the null in our checks.

## Release contexts

A context describes the recipient (CJEU, EDPS v SRB, C-413/23 P; EDPB 02/2026 paras. 7, 11, 29–35). It sets what the attacker is assumed to know and the limits. It never assumes a less motivated attacker (para. 31).

| Context | Recipient | Attacker knows | DP |
|---|---|---|---|
| internal | Staff, for development and testing | The quasi-identifiers | Not required |
| partner | A named organisation under contract | Everything except the guessed attribute, and population data | **Required** |
| public | Anyone | Same as partner | **Required** |

Each check compares the **upper** confidence bound with the limit (safety margin, para. 88): *pass* when it is below, *fail* when even the lower bound is above, *review* otherwise. Contexts outside the organisation are **blocked** unless the release carries a differential-privacy guarantee: the organisation decided that DP is required for every external release, with ε set by the organisation (there is no built-in ε). The empirical results are still shown for those contexts, for information.

The default limits are relsynth's suggestions, not legal standards. Set your own:

| Limit | internal | partner | public |
|---|---|---|---|
| `risk` (record isolation, linkage, inference) | 0.10 | 0.05 | 0.02 |
| `attribute_leakage` | 0.05 | 0.03 | 0.02 |
| `mia_auc` | 0.65 | 0.60 | 0.55 |
| `mia_tpr_at_fpr_0.01` | 0.10 | 0.05 | 0.02 |
| `record_auc` (most exposed person) | 0.85 | 0.75 | 0.65 |

Defaults decided on 2026-10-27:

- **A *review* needs a sign-off.** A named person (the DPO) signs off a *review* with `relsynth privacy ... --signed-off-by "Name, DPO"`; the report records the name. Until then `--context X --fail-on fail` counts the review as a fail. Set `sign_off: false` on a context to turn this off.
- **The most exposed person** (`per_person`). One person's membership interval is about ±0.3 wide at 32 test shadows. In *internal* (`per_person: info`) that check is shown as INFO and the AUC pooled over the ten most exposed people gates instead. In *partner* and *public* (`per_person: gate`) it gates, and relsynth runs `n_test_gated` (128) test shadows, about ±0.15, whenever such a context can pass (with DP); without DP those contexts are blocked anyway and the default 32 are used.
- **Small person tables.** With fewer than `small_population` (1000) people in the unit, the report warns, and every context outside the organisation fails without DP (check `small_population`).

When the release carries a DP guarantee (`release.dp: {epsilon, delta}`), the page also shows the largest TPR any attack can reach at FPR 1% (Kairouz et al. 2015; Prop. 603). With `cumulative_epsilon` (and `cumulative_delta`), the budget spent by all releases from the same domain so far, it also shows the bound for all of them together, since releases about the same people compose.

## privacy.yaml

Every key is optional.

```yaml
unit: Customer                       # the people; detected when omitted
sensitive: [Invoice.Total.mean, InvoiceLine.Track.Genre.Name.mode]   # profile columns
quasi_identifiers: [City, State, Country]   # default: the unit table's own attributes
split: {control: 0.4, reference: 0.2, seed: 0, max_people: 20000}
confidence: 0.95
attacks:
  record_isolation: {modes: [univariate, multivariate], n_attacks: 500, n_cols: 3}
  linkage: {n_attacks: 500, n_neighbors: 1, aux_cols: null}    # default: own attributes vs behaviour
  inference: {n_attacks: 2000}
  attribute_inference: {max_rows: 20000}                      # false to skip
  membership:
    domias: true
    shadow: {enabled: true, n_train: 32, n_test: 32, n_test_gated: 128, shadow_size: 1000,
             targets_vulnerable: 10, targets_random: 10, n_jobs: 1}
small_population: 1000                # below: warning, and external contexts fail without DP
contexts:
  internal: {per_person: info, sign_off: true, thresholds: {risk: 0.10, attribute_leakage: 0.05, mia_auc: 0.65, mia_tpr_at_fpr_0.01: 0.10, record_auc: 0.85}}
  regulator:                          # your own context
    description: Supervisory authority
    external: true
    attacker_knows: all
    per_person: gate                  # default for your own contexts
    thresholds: {risk: 0.02, attribute_leakage: 0.02, mia_auc: 0.55, mia_tpr_at_fpr_0.01: 0.02, record_auc: 0.65}
release: {dp: null}                   # {epsilon: 1.0, delta: 1.0e-6, cumulative_epsilon: 3.0, unit: Customer}, filled by the DP engine
cache: .relsynth-cache/privacy        # shadow fits are kept here and reused
```

Profile column names are `Column` for the unit table, `Parent.Column` for its parents, and `Table.Column.mean`, `Table.Column.mode`, `Table.n_rows` for owned tables (`Table.Parent.Column.mode` for their parents; `Key:Parent` when a table reaches the same parent through several keys). `relsynth privacy` lists the columns in the JSON report (`profile_columns`).

## Cost

Shadow modelling fits the generator `n_train + n_test` times (64 by default) on about `shadow_size` people each. One set of fits serves every target (Prop. 602). Fits run in parallel with `--jobs` and are cached with `--cache`, so a second run with the same settings costs only the attacks. Measured on 4 cores (`benchmarks/privacy`, see `roadmap/reports/T06.md`): Adult (48,790 people) 5 minutes in total, of which shadow fits 3 minutes. Use `--no-shadow` for a quick run without generator-level membership inference.

## Relation to the existing guards

The scorecard's distance-to-closest-record ratio and exact-match rate stay in every run (`relsynth run`) for continuity, but they are **not a privacy check**: calibration showed the distance guard misses even 20% copied records, which DOMIAS catches at 5%. The privacy report lists them in the annex and no verdict depends on them. `benchmarks/privacy/calibrate_guards.py` measures which leaks each one detects: see the T06 report for the numbers. The text model's memorisation guards (vocabulary k-anonymity, no rare n-gram copies) are unchanged.

## Limits

- Results hold for the attacks run, not for every possible attack. Only differential privacy gives a guarantee against all attacks; this module measures, it does not protect.
- The release attacked is fitted on 40% of the people. It stands for a release of the same generator fitted on everyone, as in Anonymeter and the shadow-model literature.
- Composite foreign keys are not followed when finding owned tables.
- Small databases (tens of people) give wide intervals; the report says *review* rather than guess.

## Sources

- EDPB, Guidelines 02/2026 on anonymisation, version 1.0 for public consultation, adopted 7 July 2026.
- CJEU, C-413/23 P, EDPS v SRB, judgment of 4 September 2025.
- M. Giomi et al., A unified framework for quantifying privacy risk in synthetic data (Anonymeter), PoPETs 2023.
- T. Stadler, B. Oprisanu, C. Troncoso, Synthetic data: anonymisation groundhog day, USENIX Security 2022.
- F. Houssiau et al., TAPAS: a toolbox for adversarial privacy auditing of synthetic data, 2022.
- B. van Breugel et al., Membership inference attacks against synthetic data through overfitting detection (DOMIAS), AISTATS 2023.
- M. Meeus et al., Achilles' heels: vulnerable record identification in synthetic data publishing, ESORICS 2023.
- N. Carlini et al., Membership inference attacks from first principles (LiRA), IEEE S&P 2022.
- P. Kairouz, S. Oh, P. Viswanath, The composition theorem for differential privacy, ICML 2015.
- NIST SP 800-188, De-Identifying Government Datasets: Techniques and Governance, September 2023.

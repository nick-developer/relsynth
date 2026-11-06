# DPIA template: a synthetic data release

A data protection impact assessment for generating a synthetic dataset with relsynth from personal data and giving it to a recipient. It follows the content GDPR Article 35(7) requires: (a) a description of the processing and its purposes, (b) necessity and proportionality, (c) the risks to the rights and freedoms of data subjects, and (d) the measures to address them. Copy this file, fill in the parts in *italics*, and keep it with the release's registry entry.

Whether a DPIA is legally required (Art. 35(1) and (3), and the supervisory authority's list under Art. 35(4)) is for the controller's DPO to decide. The template is useful either way: EDPB Guidelines 02/2026 para. 41 asks for "adequate documentation of the anonymisation processing", including testing. **It is not legal advice.**

The text in plain type is pre-filled with what relsynth does and where the evidence is. Check it against your installation: settings can change it.

---

## 0. Summary

| | |
|---|---|
| Release name and run id | *e.g. `crm-test-2026q4`, registry run `r-...`* |
| Controller | *organisation* |
| DPO consulted | *name, date* |
| Source system and privacy domain | *e.g. `dwh-prod`, domain `customers`* |
| Privacy unit (the people protected) | *table, approximate number of people* |
| Recipient and release context | *internal / partner / public / own context; recipient organisation* |
| Differential privacy | *ε, δ at the domain's δ, or "none (internal only)"* |
| Decision | *approve / approve with conditions / reject* — signed in the registry with `relsynth gov approve` |

## 1. Description of the processing (Art. 35(7)(a))

**1.1 Purpose.** *Why synthetic data are needed (software testing, analytics, model development, sharing with a partner) and why real data or aggregates do not suffice.*

**1.2 Data.** *Tables read, special categories (Art. 9) or criminal-offence data (Art. 10), the privacy unit and the owned tables (`relsynth privacy` lists them), approximate rows and people.*

**1.3 The processing, step by step.**

1. relsynth reads the source catalog and a sample of rows to draft a spec (`introspect`), unless the spec is written from public facts.
2. It fits a Bayesian model of the whole schema inside the organisation. Large facts are fitted from aggregates computed in the database plus a sample (docs/pushdown.md). In DP mode, it reads each owned table once and releases noisy statistics before fitting (docs/dp.md, section 2).
3. It draws a synthetic database. No source row is copied, except tables declared `mode: copy` (*list them, and confirm each holds no personal data*).
4. It attacks the synthetic database per person (`relsynth privacy`) and scores its quality.
5. It records a signed manifest, and releases only after the policy allows it and the required people approve.

**1.4 Where it runs and who has access.** *Hosts, accounts, the registry location, who holds the signing keys.*

**1.5 Retention.** *How long the source extract, the fitted model, the privacy report and the synthetic data are kept, and where. The privacy report and the model are computed on real data and stay inside the organisation (claim GOV-07).*

**1.6 Legal basis.** *Article 6 basis for fitting the model, and Article 9(2) condition for special categories. EDPB 02/2026 para. 38: anonymisation needs a legal basis, presumed to be that of the processing it follows when it is part of the same activity and purpose.* **Lawyer.**

## 2. Necessity and proportionality (Art. 35(7)(b))

| Question | Answer |
|---|---|
| Could the purpose be met with less data (fewer tables, columns, people)? | *Which tables and columns were left out; `relsynth` can be pointed at a subset* |
| Could it be met with public or already anonymous data? | *...* |
| Is a formal guarantee used? | DP is required for every release outside the organisation (claim DP-02). *For internal releases, say why DP is or is not used.* |
| Minimisation of what relsynth reads | Pushed-down facts are read as aggregates and a sample (SEC-10); DP mode reads each owned table once |
| Transparency to data subjects | *Do the privacy notices of the source mention the use of their data to build synthetic data? CJEU C-413/23 P para. 111: assessed from the controller's point of view at collection.* **Lawyer** |

## 3. The recipient and the context (EDPB 02/2026 paras. 11–15, 26–35)

| Question | Answer |
|---|---|
| Who will receive the data, and who could obtain it after them (onward transfers, leaks)? | *...* (para. 21; CJEU C-413/23 P para. 85) |
| What other data could they combine it with? | *Set the context's `attacker_knows` accordingly* |
| Which relevant entities must be considered? | *Rogue employees, journalists, unethical companies, cybercriminals, authorities (para. 30). Motivation is not a factor (para. 31).* |
| Are there legal prohibitions on re-identification, and are they enforced? | *A contract is not a legal prohibition (para. 34).* |
| Will the data be public? | *If yes, anyone may access it (para. 89): use the `public` context.* |

## 4. Risks to data subjects (Art. 35(7)(c))

Fill in from the privacy report of this run (`privacy_report.md`, page of the release context). Each line has a value, its 95% interval, the limit and the status.

| Risk (EDPB criterion) | Attack | Value [95% CI] | Limit | Status |
|---|---|---|---|---|
| Singling out (No record isolation, sec. 3.4.1) | Singling-out queries, univariate and multivariate | | | |
| Linkage (No linkage, sec. 3.4.2) | Nearest-neighbour linkage | | | |
| Inference of attributes (No inference, sec. 3.4.3) | Inference per sensitive attribute, against a population baseline | | | |
| Membership (No inference, para. 79) | Shadow models, LiRA-style ratio, DOMIAS: AUC and TPR at FPR 1% | | | |
| Most exposed person (para. 87) | Membership AUC of the person most at risk | | | |
| DP bound (if DP) | Largest TPR at FPR 1% any attack can reach, this release and cumulative for the domain | | | |

**4.1 Verdict for the context:** *pass / review (signed off by: ...) / fail / blocked.*

**4.2 What the measurement does not cover** (assurance case G2.8):

- Results hold for the attacks run; only DP bounds every attack (PRV-19).
- The spec drafted by `introspect` holds levels, ranges and rules read from the data, outside any DP guarantee, unless written from public facts (DP-11).
- Releases of the same people without DP are not covered by the DP accounting (GOV-11).
- *Other risks: harm from inaccurate synthetic data used to make decisions about people; bias from suppressed rare groups in DP mode (docs/dp.md section 5); risks specific to this release.*

**4.3 Severity and likelihood.** *For each risk, the impact on people if it materialised and how likely it is given section 3; use the organisation's risk matrix.*

## 5. Measures (Art. 35(7)(d))

| Measure | In place? | Evidence |
|---|---|---|
| Synthetic rows drawn from a model; no row copied | built in | SEC-02 |
| Per-person attacks with a safety margin, verdict per context | built in, run with `--assess` | PRV-01, PRV-12 |
| Differential privacy for external contexts | built in, enforced before any read | DP-01, DP-02 |
| Privacy budget per domain with a lifetime cap, reserved before reading | built in | GOV-08, GOV-10 |
| Named sign-off of a *review*, approvals with separation of duties | built in, configured in the policy | PRV-14, GOV-04 |
| Signed, tamper-evident record of the run and the release | built in | GOV-01, GOV-09 |
| Privacy report and model stay inside the organisation | built in | GOV-07 |
| Credentials from the environment, masked in every output | built in | SEC-01, SEC-04 |
| *Contract with the recipient (no re-identification, no onward transfer)* | *yes / no* | *reference* |
| *Access control on the registry and keys* | *yes / no* | *reference* |
| *Other organisational measures* | | |

## 6. Residual risk and decision

*State the risk that remains after the measures, whether it is acceptable, and why. If a high risk remains, Art. 36 prior consultation of the supervisory authority applies.* **Lawyer / DPO.**

## 7. Advice of the DPO (Art. 35(2))

*DPO's advice, and whether it was followed.*

## 8. Views of data subjects or their representatives (Art. 35(9)), where appropriate

*...*

## 9. Review

EDPB 02/2026 para. 35: re-identification risk can rise over time; reassess periodically. Para. 42: a security incident may trigger reassessment.

| Trigger | Action |
|---|---|
| *Every 12 months, or the organisation's period* | Re-run `relsynth privacy` with the current release of relsynth on the same registry entry |
| The final EDPB guidelines are adopted | Check the release against the [regulation mapping](regulation-mapping.md#7-tracking-the-final-edpb-guidelines) |
| A new attack is published that the suite does not cover | Reassess non-DP releases |
| A security incident affects the source, the registry or the recipient | Reassess, and revoke the release if needed (`relsynth gov revoke`) |

## Annex: files to attach

- The privacy report (internal only): `privacy_report.json`, `.md`, `.html`.
- The DP statement (`dp.json`), for a DP run.
- The quality report or scorecard.
- `relsynth gov show RUN` output and `relsynth gov verify registry/` result.
- The [assurance case](assurance-case.md) at the version of relsynth used (`relsynth version`).

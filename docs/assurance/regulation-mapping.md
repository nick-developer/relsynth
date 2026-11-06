# Regulation mapping

This page maps what relsynth does to the texts a privacy, legal or audit team will test a synthetic release against. Each row cites the provision (article, paragraph or section), says what relsynth contributes, names the evidence by its [claim register](claims.yaml) id, and says what remains for the organisation. **It is not legal advice.** Rows marked **Lawyer** are decisions the controller's counsel or DPO must take; relsynth can only supply evidence for them.

Texts read for this mapping (2026-10-31): EDPB Guidelines 02/2026 v1.0; CJEU judgment C-413/23 P; Regulation (EU) 2024/1689 as amended by Regulation (EU) 2026/1744; 45 CFR 164.514 and the HHS guidance of 26 November 2012; NIST SP 800-226 and SP 800-188. ISO/IEC 27559:2022 and ISO/IEC 20889:2018 are sold by ISO and were not available; their rows map by topic and say so. Where a paragraph was not read in full, the row says so too.

Contents: [GDPR anonymisation](#1-gdpr-anonymisation-edpb-guidelines-022026-and-edps-v-srb) · [AI Act](#2-eu-ai-act-article-10-data-and-data-governance) · [HIPAA](#3-hipaa-de-identification-expert-determination) · [NIST SP 800-226](#4-nist-sp-800-226-evaluating-differential-privacy-guarantees) · [NIST SP 800-188](#5-nist-sp-800-188-de-identifying-government-datasets) · [ISO/IEC 27559 and 20889](#6-isoiec-275592022-and-isoiec-208892018) · [Tracking the final EDPB text](#7-tracking-the-final-edpb-guidelines) · [Sources](#sources)

## 1. GDPR anonymisation: EDPB Guidelines 02/2026 and EDPS v SRB

Status of the guidelines: version 1.0, "adopted on 07 July 2026", for public consultation until 30 October 2026. They are not final; section 7 says how this mapping tracks the final text.

### 1.1 The legal test

| Provision | What it requires | What relsynth contributes | Evidence | Left to the organisation |
|---|---|---|---|---|
| GDPR Art. 4(1), Recital 26; EDPB paras. 6–8 | Data are anonymous only if no one can identify a person with the means reasonably likely to be used; the question "needs to be viewed as one of likelihood" (para. 8) | Synthetic rows are drawn from a fitted model, not copied; the risk that remains is measured per person against the three criteria below | SEC-02, PRV-01 | **Lawyer**: whether the measured likelihood is "insignificant in reality" for this release |
| EDPB para. 22; CJEU C-413/23 P para. 82 | The likelihood "does not need to be zero" but "should be insignificant in reality"; a means is not reasonably likely when identification is prohibited by law or impossible in practice | Every risk is reported with a confidence interval and judged on its upper bound | PRV-08, PRV-12 | Choosing the limits (relsynth's defaults are suggestions, docs/privacy.md) |
| EDPB paras. 11–15 (perspective), para. 12 ("for whom is the data intended to be anonymous?"); CJEU C-413/23 P paras. 75–77, 86 | Anonymity is assessed from the perspective of those who will have access; pseudonymised data may be non-personal for a recipient only if measures "actually put in place" prevent attribution, "including by recourse to other means of identification such as cross-checking" (para. 77) | A report page per release context (internal, partner, public) with the attacker's knowledge set per context | PRV-01, PRV-12 | Defining each recipient and what they hold (the context's `attacker_knows`) |
| CJEU C-413/23 P paras. 84–85; EDPB para. 21 | Data become personal when put at the disposal of someone with means to identify, both for that recipient and, indirectly, for the provider; an onward transfer must be assessed for the third party | External contexts assume the attacker knows every attribute but the one guessed, plus population data; they are blocked without DP | PRV-13, DP-02 | Onward-transfer terms; **Lawyer** for transfers outside the EU |
| CJEU C-413/23 P para. 111 | For the information duty (Art. 15(1)(d) Reg. 2018/1725, Art. 13–14 GDPR), identifiability is assessed at collection and from the controller's point of view | Nothing: this concerns the source data | | Privacy notices must name synthetic-data recipients where the source data remain personal for the controller |
| EDPB paras. 26–34 (means reasonably likely); para. 30 (relevant entities include rogue employees, journalists, intelligence agencies, unethical companies, cybercriminals); para. 31 (no "lack of motivation") | Means are read broadly; motivation is not a factor; a contractual ban is not a legal prohibition (para. 34) | Contexts never assume a less motivated attacker; the partner context assumes everything but the target attribute | PRV-12 | Whether legal prohibitions apply and are enforced (para. 33) |
| EDPB para. 35 | Risk can rise over time; "good practice ... to periodically reassess the likelihood of re-identification" | Reports and manifests carry dates, versions and the policy digest, so a release can be re-assessed with the same code | GOV-06 | A reassessment schedule (DPIA section 9) |
| EDPB para. 36 | A dataset is anonymous only if "the anonymisation is effective for all of the included individuals" | The ten most exposed people are attacked individually; in partner and public contexts the most exposed person gates the verdict | PRV-08 | |
| EDPB para. 37 | A controller that keeps the source still processes personal data | Nothing: the source and the privacy report stay with the controller | GOV-07 | GDPR compliance for the source and the generation |
| EDPB para. 38 | "anonymisation must have a legal basis under Article 6 GDPR" (Art. 9(2) for special categories), presumed to be that of the preceding processing when it is part of the same activity and purposes | | | **Lawyer**: the legal basis for fitting the model |
| EDPB para. 40 | Do not call data "anonymous" if people are still identifiable | The report says *pass*, *review*, *fail* or *blocked* per context, never "anonymous" | PRV-12 | Wording of any release notice |
| EDPB para. 41 | "adequate documentation of the anonymisation processing", including testing, retained after completion | Signed manifest per run, privacy report, DP statement, quality report, this assurance case | GOV-06, DP-08, PRV-01 | Retention of the registry |
| EDPB para. 42; GDPR Art. 33–34 | A security incident may trigger reassessment | Tamper-evident registry; release revocation | GOV-09, GOV-08 | Incident response |

### 1.2 The three technical criteria (EDPB section 3.4)

| Criterion | EDPB text | relsynth attack | Evidence | Notes |
|---|---|---|---|---|
| No record isolation (sec. 3.4.1, paras. 54–57) | "met if the data does not contain a unique combination of attribute values that relate to a single individual" (para. 55) | Singling-out queries built from the synthetic data, univariate and multivariate; success on training people above unseen control people | PRV-04, PRV-11 | Section 3.5.3, which explains singling out "in detail" (fn. 37), was not available in the text read |
| No linkage (sec. 3.4.2, paras. 58–65) | "likely to be met if the information has not been recorded elsewhere and is not correlated to similar information" (para. 65); "noisy data may leak information in unexpected ways" (para. 64) | Nearest-neighbour linkage of two partial records through the synthetic data | PRV-05 | Paras. 58–63 were not read verbatim |
| No inference (sec. 3.4.3, paras. 66–80) | "met if no specific and meaningful inference can be drawn" (para. 67); test by comparing outputs with and without the person (para. 71(b)); membership inference (para. 79) | Attribute inference against a population baseline (only the excess over population data counts); shadow-model, LiRA-style and DOMIAS membership inference | PRV-06, PRV-07 | Para. 82: inference from "AI models or synthetic data" is the case this criterion is written for |
| Applying the criteria (sec. 3.5, paras. 83–94) | Factors: dimensionality, resolution, number of individuals, additional information (para. 85); success "against a single individual" suffices (para. 87); "adequate safety margins" (para. 88); public data is open to anyone (para. 89); access restrictions "are not by themselves sufficient" (para. 90) | Per-person results, upper confidence bounds, stricter limits for public than partner, small-population rule | PRV-08, PRV-12, PRV-15 | Para. 92: agentic AI lowers attack cost; relsynth's attacks are fixed and will not cover every future method (PRV-19) |
| Compiling the results (sec. 3.5.3) and Annex 1 flowchart | Not available in the text read | The verdict per context is the worst of its checks | PRV-12 | Re-check against the final text (section 7) |

The guidelines do not mention differential privacy in the parts read. A DP guarantee bounds every membership test (Prop. 603) and so supports the No Inference criterion (para. 71(b) compares outputs with and without a person, which is the DP definition), but the EDPB does not say that DP alone makes data anonymous. **Lawyer**: whether a DP release with the organisation's ε meets the guidelines' test.

### 1.3 Other GDPR duties around a synthetic release

| Provision | Relevance | What relsynth contributes |
|---|---|---|
| Art. 5(1)(b), 5(1)(c) (purpose limitation, minimisation) | The source is read for a new purpose | Pushdown reads aggregates and a sample, not whole facts (SEC-10); DP mode reads each owned table once |
| Art. 25 (data protection by design and by default) | Choice of a privacy-enhancing technique | External releases need DP by default (DP-02); privacy checks and copy guards on by default (SEC-02) |
| Art. 32 (security of processing) | The generation reads production data | docs/security.md (SEC-01 to SEC-15) |
| Art. 35 (DPIA) | Likely high risk when special categories or large-scale data are processed; Art. 35(7) sets the content | [DPIA template](dpia-template.md), built on Art. 35(7)(a)–(d) |
| Art. 89(1) | Safeguards for research and statistics, including pseudonymisation | DP mode, per-person attacks |

## 2. EU AI Act, Article 10 (data and data governance)

Regulation (EU) 2024/1689, as amended by the Digital Omnibus on AI, Regulation (EU) 2026/1744 of 8 July 2026 (OJ L, 24 July 2026; in force 27 July 2026). The amendment **deleted Article 10(5)** and moved the bias-detection basis for special categories of personal data to a **new Article 4a**. Article 10's obligations apply from 2 December 2027 for Annex III high-risk systems and 2 August 2028 for Annex I systems (Art. 113(c) as amended).

| Provision | What it requires | What relsynth contributes | Evidence | Left to the provider |
|---|---|---|---|---|
| Art. 10(2)(a)–(c) | Governance of design choices, data origin and original purpose, preparation operations | The spec, the IR document and the signed manifest record source, spec, versions, seeds and outputs of each synthetic dataset | GOV-06 | The provider's data governance documentation (Annex IV) |
| Art. 10(2)(d)–(e) | Assumptions about what the data represent; availability, quantity, suitability | The quality report states what is matched (marginals, dependence, joins, rules) and with what limits | | Whether the synthetic data are suitable for the intended purpose |
| Art. 10(2)(f)–(g) | Examine possible biases; measures to detect, prevent and mitigate them | Synthetic data preserve the source's group statistics, so bias found in synthetic data reflects the source; DP mode suppresses rare levels and caps rows per parent, which pulls data towards typical people | DP-15 | **Bias**: DP mode is known to under-represent small groups (NIST SP 800-226 sec. 3.3); do not use DP synthetic data alone for bias detection on small groups |
| Art. 10(2)(h) | Identify data gaps and shortcomings | Failed checks and the doctor's causes | | |
| Art. 10(3) | Relevant, sufficiently representative, free of errors, complete; appropriate statistical properties, including for the groups the system is used on | Calibrated scorecard per table and join | | **Statistical**: representativeness for the target population is the provider's to show |
| Art. 10(4) | Characteristics of the geographical, contextual, behavioural or functional setting | `--scale` and per-table sizes can generate test data for settings not in the source; their realism is not guaranteed | | |
| Art. 4a(1)(a) (former Art. 10(5)(a)) | Special categories may be processed for bias detection only if this "cannot be effectively fulfilled by processing other data, including synthetic or anonymised data" | Synthetic data are the "other data" this condition asks the provider to try first; the quality report is the evidence of whether they suffice | | **Lawyer**: the record of why synthetic data were or were not sufficient (Art. 4a(1)(f)) |
| Art. 59(1)(b) | Sandbox processing of personal data only where requirements "cannot effectively be fulfilled by processing anonymised, synthetic or other non-personal data" | Same as above | | |

## 3. HIPAA de-identification (expert determination)

| Provision | What it requires | What relsynth contributes | Left to the covered entity |
|---|---|---|---|
| 45 CFR 164.514(a) | De-identified information: "no reasonable basis to believe that the information can be used to identify an individual" | | |
| 164.514(b)(1)(i) | An expert "determines that the risk is very small that the information could be used, alone or in combination with other reasonably available information, by an anticipated recipient to identify an individual" | The release context is the anticipated recipient; the partner/public contexts assume the attacker holds every other attribute and population data | **Expert**: relsynth is not the expert; a qualified person must make the determination (HHS guidance sec. 2.2) |
| 164.514(b)(1)(ii) | The expert "documents the methods and results of the analysis" | Privacy report (JSON, Markdown, HTML) with every attack, interval and setting; DP statement; manifest (PRV-01, DP-08, GOV-06) | The expert's own determination letter |
| HHS guidance sec. 2.3 | "There is no explicit numerical level of identification risk" | Limits are configurable per context; defaults are suggestions | The threshold |
| HHS guidance sec. 2.4 | Time-limited determinations are common practice | Each report is dated and versioned | The validity period |
| HHS guidance sec. 2.6, Table 1 | Principles: replicability, data source availability, distinguishability | Distinguishability ≈ record isolation (PRV-04); data source availability ≈ the context's `attacker_knows` | Replicability of each attribute in the real world |
| 164.514(b)(2) (Safe Harbor) | Removal of 18 identifiers (names, geographic units below a state, all date elements except year, ...) and no actual knowledge | Not applicable as such: synthetic dates and places are generated, not removed. Direct identifiers declared `type: pii` are generated from a format (`email`, `person_name`, `phone`, `uuid`, or a pattern shape), not copied from source values | Safe Harbor is a different route; use expert determination for synthetic data |
| 164.514(c) | Re-identification codes not derived from the person | Synthetic keys are new labels; the privacy report's person labels are HMAC under a secret salt and never leave the organisation (PRV-03) | |

## 4. NIST SP 800-226: evaluating differential privacy guarantees

NIST SP 800-226 (March 2025) organises a DP guarantee as a pyramid (Fig. 1): unit of privacy and ε at the top, algorithms in the middle, deployment at the base. relsynth's DP statement follows it (DP-08).

| Section | Hazard or requirement | relsynth | Evidence |
|---|---|---|---|
| 2.2 The privacy parameter ε | No rule for choosing ε; large ε may not protect | No default ε; the organisation sets it in a signed policy; above 1, measure with `relsynth privacy` | DP-03, GOV-10 |
| 2.3 Variants | Variants convert to (ε, δ) | zCDP accounting converted with Canonne, Kamath and Steinke (2020); δ ≤ 10⁻⁵ | DP-08, DP-03 |
| 2.4 Unit of privacy | User-level is stronger than event-level | User level: one person of the privacy unit and every row they own | DP-01, PRV-09 |
| 2.5 Comparing guarantees | A large δ misleads | δ capped at 10⁻⁵ and stated with ε | DP-03 |
| 2.6 Mixing with other releases | DP does not cover non-DP releases | The ledger counts non-DP releases of a domain separately and the guide says they are not covered | GOV-11 |
| 2.7 Auditing and empirical measures | Audits complement proofs | `--assess` compares attack TPR with the bound ε implies; Contoso ε = 1: 0.014 against 0.027 | DP-10, DP-14 |
| 3.2 Utility and accuracy | Report utility | Utility against ε on three real databases | DP-15 |
| 3.3 Bias | DP can magnify systemic bias | Rare levels suppressed and rows capped; stated in docs/dp.md section 5 | DP-15 |
| 3.6 Synthetic data | Two-stage measure-then-generate designs | Stage 1 noisy statistics, stage 2 post-processing | DP-06 |
| 4.2 Trust models | Central model trusts the curator | Central model: relsynth runs inside the organisation, which holds the data | |
| 4.3 Implementation challenges | Floating point, side channels; use validated libraries | Discrete Gaussian on integers from OpenDP; noise from the OS CSPRNG, not seeded | DP-05 |
| 4.4 Data security and access control | A breach of raw data voids the guarantee | docs/security.md; registry and keys | SEC-01, GOV-02 |
| 4.5 Data collection exposure | DP does not protect collection | Out of scope; stated | |

## 5. NIST SP 800-188: de-identifying government datasets

| Section | Topic | relsynth | Evidence |
|---|---|---|---|
| 3.2 | Risks and benefits of a release; probability and impact of re-identification | The DPIA template asks for both; the privacy report measures probability | PRV-01 |
| 3.4 | Data-sharing models: publish de-identified data, publish synthetic data, query interface, enclave | The three release contexts map to these models ([release guidance](release-guidance.md)) | |
| 3.5 | The Five Safes | Release guidance section 2 maps projects, people, settings, data and outputs | |
| 3.6 | Disclosure Review Boards | Approvals with quorum and separation of duties play the board's signing role | GOV-04 |
| 4.3.11 | Composition | DP budget per privacy domain with a lifetime cap | GOV-08, GOV-10 |
| 4.3.13 | Post-release monitoring | Manifests and revocation; reassessment schedule in the DPIA | GOV-06 |
| 4.4.4–4.4.5 | Fully synthetic data; synthetic data with validation | relsynth produces fully synthetic data (no row copied); validation against the source is the quality report, kept inside the organisation | SEC-02 |
| 4.4.7 | Synthetic data with DP | DP mode | DP-01 |
| 4.6.2–4.6.3 | Validating privacy; re-identification studies | The attack suite is a re-identification study run on every release | PRV-01 |
| 5 | Evaluate the software | This assurance case and the claim register | |

## 6. ISO/IEC 27559:2022 and ISO/IEC 20889:2018

The texts are sold by ISO and were not available for this mapping; ISO's catalogue gives only their scope. The rows below map by topic. **Clause numbers must be added from the purchased standards before this table is used in an audit.**

| Standard | Topic (from the published scope) | relsynth |
|---|---|---|
| ISO/IEC 27559 | A framework for identifying and mitigating re-identification risk across the lifecycle of de-identified data, for PII controllers and processors | Context assessment = release contexts and attacker knowledge; data assessment = the privacy unit and owned tables (PRV-09); identifiability assessment = the attack suite (PRV-04 to PRV-07); mitigation = DP mode (DP-01); governance and monitoring = registry, policy, approvals, revocation (GOV-01 to GOV-09) |
| ISO/IEC 20889 | Terminology and classification of de-identification techniques, including synthetic data and formal privacy models | relsynth's techniques in its terms: fully synthetic data from a statistical model; differential privacy as the formal model; the attacks measure singling out, linking and inference |

## 7. Tracking the final EDPB guidelines

The consultation on Guidelines 02/2026 closed on 30 October 2026. When the EDPB adopts the final version:

1. Compare the final section 3.4 and 3.5 with the rows of section 1.2, paragraph by paragraph (the numbering may change).
2. Read section 3.5.3 and Annex 1, which were not available, and decide whether relsynth's per-context verdict compiles results the same way.
3. Check whether the final text mentions differential privacy or synthetic data beyond para. 82.
4. Update the paragraph numbers quoted in docs/privacy.md; the claim register then flags every quote that changed.
5. Record the change and its date in this section.

| Date | Version checked | Change |
|---|---|---|
| 2026-10-31 | v1.0 (consultation draft) | First mapping |

## Sources

- EDPB, *Guidelines 02/2026 on Anonymisation*, version 1.0, adopted 7 July 2026 for public consultation. https://www.edpb.europa.eu/system/files/2026-07/edpb_guidelines_202602_anonymisation_v1_en_0.pdf
- Court of Justice of the EU, judgment of 4 September 2025, *EDPS v SRB*, C-413/23 P, ECLI:EU:C:2025:645. https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62023CJ0413
- Regulation (EU) 2016/679 (GDPR), Art. 4, 5, 6, 9, 25, 32, 35, 89 and Recital 26.
- Regulation (EU) 2024/1689 (AI Act), Art. 10, 59, 113; consolidated text as amended by Regulation (EU) 2026/1744 (Digital Omnibus on AI), Art. 4a. https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-10 and /article-4a
- 45 CFR 164.514; HHS Office for Civil Rights, *Guidance Regarding Methods for De-identification of Protected Health Information in Accordance with the HIPAA Privacy Rule*, 26 November 2012. https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- J. Near, D. Darais, N. Lefkovitz, G. Howarth, *Guidelines for Evaluating Differential Privacy Guarantees*, NIST SP 800-226, March 2025. https://csrc.nist.gov/pubs/sp/800/226/final
- S. Garfinkel et al., *De-Identifying Government Datasets: Techniques and Governance*, NIST SP 800-188, September 2023. https://csrc.nist.gov/pubs/sp/800/188/final
- ISO/IEC 27559:2022, *Privacy enhancing data de-identification framework*. https://www.iso.org/standard/71677.html
- ISO/IEC 20889:2018, *Privacy enhancing data de-identification terminology and classification of techniques*. https://www.iso.org/standard/69373.html

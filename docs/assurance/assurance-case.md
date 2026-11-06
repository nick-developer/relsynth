# Assurance case for a relsynth release

An assurance case is a structured argument that a system has a property, supported by evidence (claims, arguments and evidence, as in ISO/IEC/IEEE 15026-2 and the Goal Structuring Notation). This one argues that a synthetic dataset produced and released with relsynth is fit to give to its intended recipient. It is written for the organisation's privacy officer, legal counsel and auditors, who decide whether to approve a release; relsynth supplies the evidence, they supply the judgement.

Every leaf points at the [claim register](claims.yaml) by id (PRV-, DP-, SEC-, GOV-) or at a proposition of the [proofs](../proofs/relsynth-proofs.tex) by its label. `tools/claims_check.py` checks in CI that every id cited here exists and that its evidence still resolves. The [regulation mapping](regulation-mapping.md) says which provision each part of the argument answers.

## Top claim

**G0.** A synthetic dataset produced by relsynth from a source, and released to a recipient through the governance workflow, (a) has the structure and statistics the recipient needs, (b) carries a re-identification risk for that recipient that is measured, below the organisation's limits and, for any recipient outside the organisation, bounded by differential privacy, and (c) was produced and released under recorded, tamper-evident and approved governance.

Context of G0:

- **C1. Release context.** One of the contexts of the organisation's policy: `internal`, `partner`, `public` or the organisation's own ([release guidance](release-guidance.md)). The context fixes what the attacker is assumed to know and the limits.
- **C2. Privacy unit.** The table of the people protected; every row reachable from it through foreign keys belongs to one person (PRV-09; Props. `prop:pusplit`, `prop:irowner`).
- **C3. Organisation policy.** Signed by the privacy owner; holds ε caps per privacy domain, the contexts and the approvals they need (GOV-03).

Assumptions (if one is false, the argument does not hold):

- **A1.** The spec describes the source correctly: keys, links and rules the spec does not declare are reproduced statistically, not exactly.
- **A2.** The machine and accounts that run relsynth, the registry and the signing keys are protected as the organisation's other production systems are (NIST SP 800-226 sec. 4.4: a breach of the raw data voids any DP guarantee).
- **A3.** OpenDP's discrete Gaussian sampler and the operating system's random source are correct (DP mode relies on them; NIST SP 800-226 sec. 4.3 recommends validated libraries).
- **A4.** The ε in the policy was chosen by the organisation for the context and the domain. relsynth ships no ε (DP-03).
- **A5.** For a DP release, the spec was written from public facts only (`spec_is_public: true`), as the policy requires for external contexts (DP-11).

Strategy **S0**: argue each of (a), (b) and (c) separately, then argue that the claims in the documentation used for this argument are themselves traced (G4).

## G1. Structure and statistics (part a)

| Claim | Argument | Evidence |
|---|---|---|
| **G1.1** Every primary and foreign key resolves, every declared or found rule holds on every row | Keys are generated, never sampled; rules are imposed by exact conditioning (rejection, redraw instead of delete) | Props. `prop:rules`, `prop:k3`, `prop:computed`, `prop:unique`, `prop:refill`; checks C00–C78 (`pytest -m science`); with any generator plug-in: `prop:genskeleton`; with partitions: `prop:rtkeys` |
| **G1.2** Statistics match the source within limits calibrated to sampling noise | Each scorecard limit is the larger of a fixed tolerance and the sampling limit under the posterior predictive null; families are Šidák-corrected | Props. `prop:splitnull`, `prop:pitcurve`, `prop:arearule`, `prop:carry`; checks C79–C82 |
| **G1.3** This holds on real databases, not only on fake data with known ground truth | The bench re-runs four real databases on every change against a five-seed baseline | `bench/` and its baseline; the T18/T10 equivalence tests on warehouses and lakehouses |
| **G1.4** In DP mode, the loss of utility is known | Utility measured against ε on three real databases | DP-15 |

Defeater **D1.1**: a check can fail by chance on real data, because the source is one draw. *Rebuttal*: each check reports its level (`alpha`), so the expected number of false failures is known.

## G2. Re-identification risk (part b)

Strategy **S2**: show that the risk is measured per person against the EDPB's three criteria and membership inference, with intervals and a safety margin (G2.1–G2.4); that releases outside the organisation are additionally bounded by DP (G2.5–G2.7); and state what neither covers (G2.8).

| Claim | Argument | Evidence |
|---|---|---|
| **G2.1** Risk is measured per person, not per row | Attacks run on one profile per person of the privacy unit, built from every owned table; keys are not in profiles | PRV-09, PRV-10 |
| **G2.2** The measurement answers the EDPB criteria | Singling out (No record isolation), linkage (No linkage), attribute and membership inference (No inference) | PRV-04, PRV-05, PRV-06, PRV-07 |
| **G2.3** A measured risk is a risk to the training people, not an artefact | Training and control people are split at random with equal sizes; only the excess on training people counts; under the null the rates are centred | PRV-11; Props. `prop:puctrl`, `prop:puattr`; checks C601, C606 |
| **G2.4** The decision is conservative | Verdicts use the upper confidence bound (safety margin); the most exposed people are attacked one by one; small populations fail external contexts; a *review* needs a named sign-off | PRV-08, PRV-12, PRV-14, PRV-15 |
| **G2.5** No release leaves the organisation without DP | Enforced before any data is read, and as an invariant that the policy cannot waive | DP-02, PRV-13, GOV-04, GOV-05 |
| **G2.6** A DP run satisfies user-level (ε, δ)-DP for the people of the unit | Contribution bounding is stable, every released statistic is charged its sensitivity, partition selection bounds δ, stage 2 is post-processing, the accountant refuses anything beyond the budget | DP-01, DP-04, DP-05, DP-06, DP-07; Props. `prop:dgauss` to `prop:run`; checks C900–C906 |
| **G2.7** The DP bound and the measurements agree | The largest TPR any attack can reach at a given FPR follows from ε (Kairouz et al. 2015); the audit on Contoso at ε = 1 stayed below it | PRV-16, DP-10, DP-14; Prop. `prop:pudp`; check C603 |
| **G2.8** What is not covered is stated | Empirical attacks cover the attacks run; the spec drafted by `introspect` is outside the DP guarantee; non-DP releases of the same people are not covered; DP mode has known gaps | PRV-19, DP-11, DP-13, GOV-11 |

Defeaters and their status:

- **D2.1** *The distance-to-closest-record guard misses copying.* True: it missed 20% copied records. *Rebuttal*: no verdict depends on it; DOMIAS and the shadow models detect copying at 5% (PRV-18).
- **D2.2** *Small databases cannot be judged.* True: with tens of people the intervals are wide. *Rebuttal*: the report says *review*, and external contexts fail below 1,000 people without DP (PRV-15). Chinook (59 customers) fails internally on membership inference (AUC 0.84, roadmap report T06).
- **D2.3** *The spec leaks.* Values that `introspect` reads (levels, ranges, rules) are in the spec. *Rebuttal*: for external contexts the policy requires a spec written from public facts (A5, DP-11); for internal contexts the spec stays inside the organisation.
- **D2.4** *Repeated releases about the same people compose.* *Rebuttal*: the budget is kept per privacy domain with a lifetime cap that never resets (GOV-08, GOV-10); the privacy report shows the cumulative bound (PRV-16).
- **D2.5** *A future attack may beat today's suite* (EDPB para. 92: AI lowers attack cost). Open for non-DP releases; for DP releases the bound holds against any attack (G2.7). Reassess on a schedule (EDPB para. 35; [DPIA](dpia-template.md) section 9).
- **D2.6** *Floating-point and timing side channels in DP mechanisms* (NIST SP 800-226 sec. 4.3). *Rebuttal*: noise is integer discrete Gaussian from OpenDP; timing is not addressed and is out of scope in the central model, where only the synthetic output leaves.

## G3. Governance (part c)

| Claim | Argument | Evidence |
|---|---|---|
| **G3.1** Every run is recorded and later changes are detected | Signed manifests in a hash chain with signed checkpoints; verification catches edits, deletions, insertions, reordering and truncation | GOV-01, GOV-09; Prop. `prop:govchain`; check C803 |
| **G3.2** Only accountable people act | Ed25519 keys per person and service, roles from the privacy owner's root key; policy signed by the privacy owner | GOV-02, GOV-03 |
| **G3.3** Privacy budget cannot be overspent | Reserve before reading, never refund after a read, lifetime cap per domain, period caps only throttle | GOV-08, GOV-10, DP-09; Props. `prop:ledger`, `prop:period`, `prop:reserve`; checks C800–C802 |
| **G3.4** A release needs a verdict and approvals | No verdict, a `fail` or `blocked` verdict, or a missing approval denies the release; external releases need someone other than the operator | GOV-04, GOV-05 |
| **G3.5** What leaves is what was approved, and nothing computed on real data | The package holds the tables byte for byte as signed, the manifest, approvals and token; the privacy report never leaves; manifests hold no data or credentials | GOV-06, GOV-07 |

Defeater **D3.1**: *the registry is a folder; an administrator could delete it.* Deletion of the log is detected against checkpoints held elsewhere (the recipient verifies a package against the root key id); keeping a copy of the checkpoints outside the registry is the organisation's control (A2).

## G4. The documentation the argument relies on is traced

**G4.1** Every claim that this argument cites appears in the product documentation and traces to a proposition, a check, a test or a measured report. *Argument*: `tools/claims_check.py` runs in CI on every pull request (job `lint`): it fails when a quote leaves its document, a test or proposition is renamed or removed, a claim-like sentence in a document in scope is not registered, or a `Prop. N` reference names an undefined proposition. *Evidence*: the register, `tests/unit/test_assurance.py`.

Defeater **D4.1**: *a claim without a trigger word escapes the scan.* True: the scan finds sentences with words such as "never", "every", "exact", "guarantee", "refused". *Rebuttal*: documents are reviewed by pull request; the register is extended document by document (the README is next).

## How to use this case for one release

1. Run the release through the governance workflow with `--assess` (or `--privacy-report`) and keep the registry.
2. Fill in the [DPIA template](dpia-template.md); its sections cite the parts of this case they rely on.
3. Check the assumptions A1–A5 for this release and record any that do not hold.
4. Approve or reject in the registry (`relsynth gov approve`). The approval records who decided.

## Sources

- ISO/IEC/IEEE 15026-2:2022, *Systems and software engineering — Systems and software assurance — Part 2: Assurance case*.
- Assurance Case Working Group, *Goal Structuring Notation Community Standard*, version 3, 2021.
- P. Kairouz, S. Oh, P. Viswanath, *The composition theorem for differential privacy*, ICML 2015.
- The texts in the [regulation mapping](regulation-mapping.md#sources).
